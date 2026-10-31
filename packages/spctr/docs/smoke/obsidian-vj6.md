# Smoke Test Runbook — obsidian-vj6: edit tool relative path resolution

**Purpose:** Verify that `edit` resolves relative paths against `workspace_root`
before file operations, matching `file_write` behavior. Before the fix, the
confinement check validated against workspace_root but `os.path.exists()` and
`_edit_file()` used the raw path relative to process cwd.

## Setup

```bash
export SMOKE_ROOT=$(mktemp -d)
export WORKSPACE=$SMOKE_ROOT/workspace
mkdir -p "$WORKSPACE/src"
echo "hello world" > "$WORKSPACE/src/file.txt"
export OTHER_DIR=$SMOKE_ROOT/other
mkdir -p "$OTHER_DIR"
```

## Scenarios

### 1. edit resolves relative path against workspace_root (not cwd)

```bash
cd "$OTHER_DIR" && python3 -c "
import asyncio, sys, os
from sr2_spectre.tools.builtins.edit import EditTool

# cwd is OTHER_DIR — NOT the workspace
assert os.getcwd() != '$WORKSPACE', 'cwd should NOT be workspace'

tool = EditTool(workspace_root='$WORKSPACE')

async def run():
    result = await tool(path='src/file.txt', old_string='world', new_string='spectre')
    print(result)

asyncio.run(run())
print(open('$WORKSPACE/src/file.txt').read().strip())
"
```

**Expect:**
- Prints `Made 1 replacement(s) in .../src/file.txt`
- File content is `hello spectre`
- No `FileNotFoundError` (the key regression: before the fix this would fail
  because `os.path.exists("src/file.txt")` resolves relative to OTHER_DIR)

### 2. edit with no workspace_root passes through raw path (unchanged behavior)

```bash
cd "$WORKSPACE" && python3 -c "
import asyncio
from sr2_spectre.tools.builtins.edit import EditTool

tool = EditTool()  # no workspace_root
asyncio.run(tool(path='src/file.txt', old_string='spectre', new_string='world'))
print(open('src/file.txt').read().strip())
"
```

**Expect:** Prints `hello world` (restores original content)

### 3. edit with absolute path works unchanged

```bash
cd "$OTHER_DIR" && python3 -c "
import asyncio
from sr2_spectre.tools.builtins.edit import EditTool

tool = EditTool(workspace_root='$WORKSPACE')
asyncio.run(tool(path='$WORKSPACE/src/file.txt', old_string='world', new_string='spectre'))
print(open('$WORKSPACE/src/file.txt').read().strip())
"
```

**Expect:** Prints `hello spectre`

## Teardown

```bash
rm -rf "$SMOKE_ROOT"
```

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | edit resolves relative path against workspace_root (cwd divergence) | [ ] |
| 2 | edit with no workspace_root passes through raw path | [ ] |
| 3 | edit with absolute path works unchanged | [ ] |
