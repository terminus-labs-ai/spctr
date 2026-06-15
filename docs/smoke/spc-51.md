# Smoke Test Runbook — spc-51: Workspace Confinement for Agent File Tools

**Purpose:** Verify that `file_write`, `edit`, and `terminal` tools enforce workspace boundaries when `SR2_WORKSPACE` is set, and remain permissive when unset.

## Setup

```bash
export SMOKE_ROOT=$(mktemp -d)
export WORKSPACE=$SMOKE_ROOT/workspace
mkdir -p $WORKSPACE
export OUTSIDE=$SMOKE_ROOT/outside
mkdir -p $OUTSIDE
```

## Scenarios

### 1. file_write accepts paths inside workspace

```bash
SR2_WORKSPACE=$WORKSPACE python3 -c "
import asyncio
from sr2_spectre.tools.builtins.file_write import FileWriteTool
tool = FileWriteTool(workspace_root='$WORKSPACE')
asyncio.run(tool(path='test.txt', content='hello'))
print(open('$WORKSPACE/test.txt').read())
"
```

**Expect:** Prints `hello`

### 2. file_write rejects absolute path outside workspace

```bash
SR2_WORKSPACE=$WORKSPACE python3 -c "
import asyncio, sys
from sr2_spectre.tools.builtins.file_write import FileWriteTool
tool = FileWriteTool(workspace_root='$WORKSPACE')
async def try_write():
    try:
        await tool(path='$OUTSIDE/leak.txt', content='leaked')
        print('ERROR: should have raised')
        sys.exit(1)
    except ValueError as e:
        print(f'OK: {e}')
asyncio.run(try_write())
"
```

**Expect:** Prints `OK: Path ... is outside workspace ...`

### 3. file_write rejects .. traversal

```bash
SR2_WORKSPACE=$WORKSPACE python3 -c "
import asyncio, sys
from sr2_spectre.tools.builtins.file_write import FileWriteTool
tool = FileWriteTool(workspace_root='$WORKSPACE')
async def try_write():
    try:
        await tool(path='$WORKSPACE/../outside/leak.txt', content='escaped')
        print('ERROR: should have raised')
        sys.exit(1)
    except ValueError as e:
        print(f'OK: rejected traversal')
asyncio.run(try_write())
"
```

**Expect:** Prints `OK: rejected traversal`

### 4. file_write is permissive without workspace_root

```bash
python3 -c "
import asyncio
from sr2_spectre.tools.builtins.file_write import FileWriteTool
tool = FileWriteTool()  # no workspace_root
asyncio.run(tool(path='$OUTSIDE/free.txt', content='free'))
print(open('$OUTSIDE/free.txt').read())
"
```

**Expect:** Prints `free`

### 5. terminal runs with workspace cwd

```bash
SR2_WORKSPACE=$WORKSPACE python3 -c "
import asyncio
from sr2_spectre.tools.builtins.terminal import TerminalTool
tool = TerminalTool(workspace_root='$WORKSPACE')
result = asyncio.run(tool(command='pwd'))
print(result.strip())
"
```

**Expect:** Prints `$WORKSPACE` (the resolved path)

### 6. edit rejects path outside workspace

```bash
echo "original" > "$OUTSIDE/target.txt"
SR2_WORKSPACE=$WORKSPACE python3 -c "
import asyncio, sys
from sr2_spectre.tools.builtins.edit import EditTool
tool = EditTool(workspace_root='$WORKSPACE')
async def try_edit():
    try:
        await tool(path='$OUTSIDE/target.txt', old_string='original', new_string='hacked')
        print('ERROR: should have raised')
        sys.exit(1)
    except ValueError as e:
        print(f'OK: rejected edit outside workspace')
asyncio.run(try_edit())
"
```

**Expect:** Prints `OK: rejected edit outside workspace`

### 7. Resolver returns correct project with SR2_PROJECT env

```bash
python3 -c "
import os, sys
# Simulate worktree named 'spc-47'
os.environ['SR2_PROJECT'] = 'sr2-spectre'
from sr2_spectre.planning.resolver import PlanResolver
from sr2.config.models import ResolverConfig
r = PlanResolver(ResolverConfig(type='plan', config={'project': '__auto__'}))
project = r._resolve_project()
assert project == 'sr2-spectre', f'Expected sr2-spectre, got {project}'
print(f'OK: project resolved to {project}')
"
```

**Expect:** Prints `OK: project resolved to sr2-spectre`

## Teardown

```bash
rm -rf $SMOKE_ROOT
```

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | file_write inside workspace | [ ] |
| 2 | file_write rejects outside path | [ ] |
| 3 | file_write rejects .. traversal | [ ] |
| 4 | file_write permissive without root | [ ] |
| 5 | terminal cwd = workspace root | [ ] |
| 6 | edit rejects outside path | [ ] |
| 7 | SR2_PROJECT overrides worktree name | [ ] |

## Next Action

After smoke pass: close spc-51, proceed to obsidian-p93 (harbinger env passing).
