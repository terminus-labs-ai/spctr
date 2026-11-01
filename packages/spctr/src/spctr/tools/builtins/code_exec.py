"""Code execution tool — run a Python snippet in a contained child process.

Snippets arrive from whoever is talking to the agent, so they are treated as
hostile input. Earlier revisions ran them in the agent's own interpreter with
``{"__builtins__": __builtins__}``, which meant a snippet could read the
process environment (Discord bot token, API keys), spawn a shell without the
terminal tool being enabled, and ignore the timeout entirely — a thread started
by ``asyncio.to_thread`` cannot be cancelled. Containment now comes from the
process boundary instead (obsidian-nr73):

- a separate interpreter, started in isolated mode (``-I``), so nothing the
  snippet defines or mutates can reach the agent;
- an environment built from a small allowlist rather than inherited, so
  secrets in the agent's environment are never handed over;
- its own process group, killed with SIGKILL on timeout, so busy loops and
  anything the snippet spawned die with it;
- address-space and file-size rlimits, so a runaway allocation cannot take
  the host down.

None of this replaces an OS-level sandbox for genuinely untrusted code. It
removes the failure modes that made the tool unsafe to expose at all.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys

#: Environment variables the child is allowed to inherit. Everything else is
#: dropped: the agent's environment carries bot tokens and API keys, and a
#: snippet that prints ``os.environ`` would otherwise publish them.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")

_FALLBACK_PATH = "/usr/local/bin:/usr/bin:/bin"

# Runs inside the child. Reads the snippet from stdin so nothing user-supplied
# ever lands in argv, and writes a single JSON result to the real stdout after
# the redirect has been torn down.
_RUNNER = r'''
import io, json, sys
from contextlib import redirect_stderr, redirect_stdout

max_bytes = int(sys.argv[1])
if max_bytes > 0:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))
    except Exception:
        pass

source = sys.stdin.read()
real_stdout = sys.stdout
out, err = io.StringIO(), io.StringIO()
result = {"status": "success", "return_value": None, "error": None}

try:
    with redirect_stdout(out), redirect_stderr(err):
        namespace = {"__name__": "__code_exec__"}
        try:
            compiled = compile(source, "<code_exec>", "eval")
        except SyntaxError:
            compiled = None
        if compiled is not None:
            value = eval(compiled, namespace)
            if value is not None:
                result["return_value"] = repr(value)
        else:
            exec(compile(source, "<code_exec>", "exec"), namespace)
except BaseException as exc:
    result["status"] = "error"
    result["error"] = "%s: %s" % (type(exc).__name__, exc)

result["stdout"] = out.getvalue()
result["stderr"] = err.getvalue()

try:
    real_stdout.write(json.dumps(result))
    real_stdout.flush()
except Exception:
    pass
'''


class CodeExecTool:
    """Execute a Python snippet in a contained child process.

    Captures the snippet's stdout, stderr and the value of its final
    expression. The timeout is enforced by killing the child's process group,
    so it holds for CPU-bound code and for anything the snippet spawned.
    """

    name = "code_exec"
    description = (
        "Execute a Python code snippet in a separate, short-lived interpreter "
        "with a scrubbed environment. Returns stdout, stderr, return_value, "
        "and execution status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code snippet to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Maximum execution time in seconds (default: 10).",
            },
        },
        "required": ["code"],
    }

    def __init__(
        self,
        timeout: int = 10,
        workspace_root: str | None = None,
        max_memory_mb: int = 1024,
    ) -> None:
        """Initialize the code execution tool.

        Args:
            timeout: Maximum execution time in seconds.
            workspace_root: When set, the snippet runs with cwd set to this
                directory. Injected by the runtime from SR2_WORKSPACE.
            max_memory_mb: Address-space and file-size cap for the child, in
                megabytes. Set to 0 to disable the rlimits.
        """
        self.timeout = timeout
        if workspace_root is not None:
            self.workspace_root: str | None = str(
                os.path.realpath(os.path.expanduser(workspace_root))
            )
        else:
            self.workspace_root = None
        self.max_memory_mb = max_memory_mb

    def _child_env(self) -> dict[str, str]:
        """Build the child's environment from the allowlist.

        Allowlisted rather than filtered: a denylist would need updating every
        time a new secret is added to the agent's environment.
        """
        env = {
            key: os.environ[key]
            for key in _ENV_ALLOWLIST
            if key in os.environ
        }
        env.setdefault("PATH", _FALLBACK_PATH)
        env.setdefault("HOME", self.workspace_root or "/tmp")
        env.setdefault("LANG", "C.UTF-8")
        # Keep the child from writing .pyc files into the workspace.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    async def __call__(self, code: str, timeout: int | None = None) -> str:
        effective_timeout = timeout if timeout is not None else self.timeout
        max_bytes = max(0, int(self.max_memory_mb)) * 1024 * 1024

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                _RUNNER,
                str(max_bytes),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._child_env(),
                cwd=self.workspace_root,
                # Its own process group, so the timeout can take down
                # grandchildren the snippet spawned, not just the snippet.
                start_new_session=True,
            )
        except OSError as exc:
            return self._format_result(
                status="error",
                stdout="",
                stderr="",
                return_value=None,
                error=f"Could not start the execution subprocess: {exc}",
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(code.encode()), timeout=effective_timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            await self._kill_process_group(proc)
            return self._format_result(
                status="timeout",
                stdout="",
                stderr="",
                return_value=None,
                error=f"Execution timed out after {effective_timeout}s",
            )

        return self._parse_child_result(stdout, stderr, proc.returncode)

    @staticmethod
    async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the child's whole process group and reap it.

        The group, not the process: a snippet that spawned helpers would
        otherwise leave them running after the timeout fired.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            await proc.wait()
        except ProcessLookupError:
            pass

    def _parse_child_result(
        self, stdout: bytes, stderr: bytes, returncode: int | None
    ) -> str:
        """Turn the child's JSON payload back into the tool's output format.

        A missing or unparseable payload means the child died before it could
        report — ``os._exit``, a segfault, or an rlimit kill — which is an
        error for the caller, not a crash for the agent.
        """
        raw = stdout.decode(errors="replace").strip()
        child_stderr = stderr.decode(errors="replace").strip()

        if not raw:
            return self._format_result(
                status="error",
                stdout="",
                stderr=child_stderr,
                return_value=None,
                error=(
                    "The snippet exited without returning a result "
                    f"(exit code {returncode})."
                ),
            )

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self._format_result(
                status="error",
                stdout=raw,
                stderr=child_stderr,
                return_value=None,
                error=(
                    "The execution subprocess returned malformed output "
                    f"(exit code {returncode})."
                ),
            )

        merged_stderr = payload.get("stderr", "")
        if child_stderr:
            merged_stderr = f"{merged_stderr}{child_stderr}".strip()

        return self._format_result(
            status=payload.get("status", "error"),
            stdout=payload.get("stdout", ""),
            stderr=merged_stderr,
            return_value=payload.get("return_value"),
            error=payload.get("error"),
        )

    @staticmethod
    def _format_result(
        status: str,
        stdout: str,
        stderr: str,
        return_value: str | None,
        error: str | None = None,
    ) -> str:
        """Render the result block.

        ``return_value`` arrives already repr'd by the child: the value itself
        cannot cross the process boundary, and re-repr'ing the string here
        would double-quote it.
        """
        lines = [f"Status: {status}"]
        if stdout:
            lines.append(f"Stdout:\n{stdout}")
        if stderr:
            lines.append(f"Stderr:\n{stderr}")
        if status == "success" and return_value is not None:
            lines.append(f"Return: {return_value}")
        if error:
            lines.append(f"Error: {error}")
        return "\n".join(lines)
