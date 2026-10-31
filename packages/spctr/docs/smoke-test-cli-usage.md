# Smoke Test Runbook: CLI Usage (obsidian-nid)

**Purpose:** Verify the sr2-spectre CLI argument parsing and usage documentation
are consistent and correct after the obsidian-nid fix.

**Background:** argparse requires positional arguments to appear contiguously.
The `config` and `prompt` positionals must come before options like
`--interface`. The broken form `config.yaml --interface single_shot "prompt"`
exits with code 2.

---

## Prerequisites

```bash
cd /home/shepard/git/sr2-spectre
uv sync
```

## Smoke Tests

### 1. Help output shows correct examples

```bash
uv run sr2-spectre --help
```

**Expected:** The epilog section shows:
- `sr2-spectre config.yaml 'What is the weather?' --interface single_shot`
- `sr2-spectre config.yaml --interface tui`
- `sr2-spectre config.yaml 'Hello' --trace`

**Check:** Prompt appears BEFORE `--interface`, not after.

### 2. Correct single-shot form parses

```bash
uv run python -c "
from sr2_spectre.cli import _parse_args
args = _parse_args(['config.yaml', 'Hello world', '--interface', 'single_shot'])
assert args.config == 'config.yaml'
assert args.prompt == ['Hello world']
assert args.interface == 'single_shot'
print('PASS: single-shot prompt before --interface')
"
```

### 3. TUI form (no prompt) parses

```bash
uv run python -c "
from sr2_spectre.cli import _parse_args
args = _parse_args(['config.yaml', '--interface', 'tui'])
assert args.config == 'config.yaml'
assert args.prompt == []
assert args.interface == 'tui'
print('PASS: tui without prompt')
"
```

### 4. Broken form exits 2

```bash
uv run python -c "
import sys
from sr2_spectre.cli import _parse_args
try:
    _parse_args(['config.yaml', '--interface', 'single_shot', 'Hello'])
    sys.exit(1)  # should not reach here
except SystemExit as e:
    assert e.code == 2, f'Expected exit 2, got {e.code}'
    print('PASS: broken form (option between positionals) exits 2')
" 2>&1
```

### 5. Default interface is single_shot

```bash
uv run python -c "
from sr2_spectre.cli import _parse_args
args = _parse_args(['config.yaml', 'test prompt'])
assert args.interface == 'single_shot'
print('PASS: default interface is single_shot')
"
```

### 6. Full test suite (relevant subset)

```bash
uv run pytest tests/test_cli_argparse.py tests/test_trace_flag.py tests/test_interface.py tests/test_single_shot.py -v
```

**Expected:** All tests pass green.

---

## Files Changed

| File | Change |
|------|--------|
| `src/sr2_spectre/cli.py` | Fixed module docstring usage; added argparse epilog with correct examples |
| `src/sr2_spectre/interfaces/single_shot.py` | Fixed usage docstring (added config.yaml, correct order) |
| `src/sr2_spectre/interfaces/tui.py` | Added note about positional ordering |
| `tests/test_cli_argparse.py` | New: 12 tests for CLI argument parsing |
