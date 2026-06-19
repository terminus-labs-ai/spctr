# Smoke Runbook: spc-71 — Per-character content cap enforcement

## What This Tests

The `GenerateImageTool` accepts a `max_content` cap (sfw|nsfw) from character
config. When the agent requests a scenario whose content level exceeds the cap,
the tool refuses — generating nothing and returning a refusal string.

This is a hard floor, independent of the model's scenario choice.

## Prerequisites

- Python 3.12+ with project deps installed (`uv sync` or `uv run`)
- Working directory: project root (`sr2-spectre`)

## Smoke Steps

### 1. Run the test suite

```bash
PYTHONPATH=src pytest tests/test_content_cap.py -v
```

**Expected:** 15 tests pass, 0 failures.

### 2. Verify cap enforcement with bundled config

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.image_scenarios import load_image_scenarios
from sr2_spectre.tools.builtins.generate_image import GenerateImageTool
from pathlib import Path

registry = load_image_scenarios(
    Path('src/sr2_spectre/resources/scenarios/image_scenarios.yaml')
)

# SFW-capped character
sfw_tool = GenerateImageTool(max_content='sfw', scenario_registry=registry)

# Check refusal on nsfw scenario
refusal = sfw_tool._check_content_cap('boudoir')
print(f'boudoir (nsfw) with sfw cap: {refusal}')
assert refusal is not None
assert 'boudoir' in refusal.lower()
assert 'nsfw' in refusal.lower()
assert 'sfw' in refusal.lower()

# Check allowance on sfw scenario
ok = sfw_tool._check_content_cap('selfie')
print(f'selfie (sfw) with sfw cap: {ok}')
assert ok is None

# NSFW-capped character allows everything
nsfw_tool = GenerateImageTool(max_content='nsfw', scenario_registry=registry)
assert nsfw_tool._check_content_cap('boudoir') is None
assert nsfw_tool._check_content_cap('selfie') is None
print('nsfw cap allows all scenarios: PASS')

print()
print('All content cap checks passed.')
"
```

**Expected output:**
```
boudoir (nsfw) with sfw cap: Image generation refused: scenario 'boudoir' has content level 'nsfw' which exceeds the character's max_content cap of 'sfw'.
selfie (sfw) with sfw cap: None
nsfw cap allows all scenarios: PASS

All content cap checks passed.
```

### 3. Verify invalid cap raises ValueError

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.builtins.generate_image import GenerateImageTool

try:
    GenerateImageTool(max_content='ecchi')
    print('FAIL: should have raised ValueError')
except ValueError as e:
    print(f'PASS: {e}')
"
```

**Expected output:**
```
PASS: Invalid max_content 'ecchi'. Must be one of: nsfw, sfw
```

### 4. Verify content level ranking

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.builtins.generate_image import _content_level_rank

# sfw < nsfw
assert _content_level_rank('sfw') < _content_level_rank('nsfw')
print(f'sfw rank: {_content_level_rank(\"sfw\")}')
print(f'nsfw rank: {_content_level_rank(\"nsfw\")}')

# Unknown levels default to 0 (most restrictive)
assert _content_level_rank('unknown') == 0
print(f'unknown rank: {_content_level_rank(\"unknown\")} (defaults to 0)')

print('PASS: content level ranking correct')
"
```

**Expected output:**
```
sfw rank: 0
nsfw rank: 1
unknown rank: 0 (defaults to 0)
PASS: content level ranking correct
```

## Verification Matrix

| Test | Expected |
|------|----------|
| 15 unit tests | All pass |
| SFW cap blocks NSFW scenario | Refusal string with scenario + cap names |
| SFW cap allows SFW scenario | None (allowed) |
| NSFW cap allows all | None for both sfw and nsfw scenarios |
| Invalid cap | ValueError with valid options |
| Content level ranking | sfw=0 < nsfw=1; unknown=0 |

## Files Changed

- `src/sr2_spectre/tools/builtins/generate_image.py` — added `max_content` param, `scenario_registry` param, `_check_content_cap()`, `_content_level_rank()`
- `tests/test_content_cap.py` — 15 tests
