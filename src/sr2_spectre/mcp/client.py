"""MCPClient — connects to an MCP server (stdio or http) and returns MCPToolBridges."""
from __future__ import annotations

import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from sr2_spectre.mcp.tool_bridge import MCPToolBridge


class MCPConnectionError(Exception):
    """Raised when the MCP transport or session handshake fails."""


class MCPClient:
    """Connects to an MCP server and exposes its tools as MCPToolBridge instances.

    Args:
        server_type: "stdio", "http" (SSE), or "streamable-http"
        **transport_kwargs:
            stdio: command (list[str]), args (list[str]=[]), env (dict={})
            http / streamable-http:  url (str)
    """

    def __init__(self, server_type: str, **transport_kwargs: object) -> None:
        self._server_type = server_type
        self._transport_kwargs = transport_kwargs
        # Held open between connect() and close()
        self._transport_ctx = None
        self._session_ctx = None

    async def connect(self) -> list[MCPToolBridge]:
        """Open the transport, initialise the session, and return one bridge per tool."""
        try:
            if self._server_type == "stdio":
                command = self._transport_kwargs.get("command", "")
                args = self._transport_kwargs.get("args", [])
                env = self._transport_kwargs.get("env", {})
                # command may be a str (real usage) or a list (test stubs).
                # StdioServerParameters requires a str; when mocked the
                # validation never runs, so we normalise here.
                if isinstance(command, list):
                    cmd_str = command[0] if command else ""
                    extra_args = list(command[1:]) + list(args)
                else:
                    cmd_str = command
                    extra_args = list(args)
                params = StdioServerParameters(command=cmd_str, args=extra_args, env=env or None)
                transport_ctx = stdio_client(params)
            elif self._server_type in ("streamable-http", "streamable_http"):
                url = self._transport_kwargs["url"]
                transport_ctx = streamablehttp_client(url=url)
            else:
                url = self._transport_kwargs["url"]
                transport_ctx = sse_client(url=url)

            self._transport_ctx = transport_ctx
            # streamablehttp_client yields a 3-tuple (read, write, get_session_id);
            # sse_client/stdio_client yield 2. Positional slice tolerates both.
            entered = await transport_ctx.__aenter__()
            read, write = entered[0], entered[1]

            session_ctx = ClientSession(read, write)
            self._session_ctx = session_ctx
            session = await session_ctx.__aenter__()

            await session.initialize()
            result = await session.list_tools()

        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(str(exc)) from exc

        return [
            MCPToolBridge(mcp_tool=tool, call_tool=session.call_tool)
            for tool in result.tools
        ]

    async def close(self) -> None:
        """Exit session and transport context managers. Safe to call before connect()."""
        if self._session_ctx is not None:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except (RuntimeError, GeneratorExit):
                pass  # Cancel scope mismatch on shutdown — ignore
            except asyncio.CancelledError:
                self._suppress_spurious_cancel()
            self._session_ctx = None

        if self._transport_ctx is not None:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except (RuntimeError, GeneratorExit):
                pass  # stdio_client cancel scope crash on shutdown — ignore
            except asyncio.CancelledError:
                self._suppress_spurious_cancel()
            self._transport_ctx = None

    @staticmethod
    def _suppress_spurious_cancel() -> None:
        """Swallow a CancelledError leaked by an MCP transport's internal anyio
        cancel scope on shutdown — but re-raise if THIS task is genuinely being
        cancelled, so real cancellation/timeouts still propagate.

        The SSE / streamable-http transports wrap a background reader/writer in
        an anyio task group. Tearing that group down inside ``__aexit__`` cancels
        its scope; an in-flight HTTP request gets a CancelledError that anyio can
        leak out of ``__aexit__`` even when nobody cancelled us. Left unhandled
        it turns a successful run into a nonzero exit. ``task.cancelling()`` (>0)
        distinguishes a genuine external cancel from this spurious internal one.
        """
        task = asyncio.current_task()
        if task is not None and task.cancelling() > 0:
            raise  # genuine external cancellation — must propagate
