# Smoke Runbook: spc-63 — Scenario Config Loader + Validation

## What This Tests

The `image_scenarios.py` module loads `image_scenarios.yaml`, validates all
fragment references at load time, and returns resolved scenarios. Bad config
fails fast with clear errors — never at call time.

## Prerequisites

- Python 3.12+ with project deps installed (`uv sync` or `uv run`)
- Working directory: project root (`sr2-spectre`)

## Smoke Steps

### 1. Run the test suite

```bash
uv run pytest tests/test_image_scenarios.py -v
```

**Expected:** 37 tests pass, 0 failures.

### 2. Load the bundled sample config

```bash
uv run python -c "
from sr2_spectre.tools.image_scenarios import load_image_scenarios
from pathlib import Path

config_path = Path('src/sr2_spectre/resources/scenarios/image_scenarios.yaml')
registry = load_image_scenarios(config_path)

print('Scenarios:', registry.scenario_names())
for name in registry.scenario_names():
    s = registry.get(name)
    print(f'  {name}: model={s.model.file}, frame={s.frame.tags[:30]}..., content={s.content.level}')
"
```

**Expected output:**
```
Scenarios: ['boudoir', 'rooftop_scene', 'selfie']
  boudoir: model=Pony/ponyDiffusionV6XL.safetensors, frame=full body, standing, full figure..., content=nsfw
  rooftop_scene: model=SDXL/dreamshaperXL_alpha2Xl10.safetensors, frame=wide shot, environment, cinematic..., content=sfw
  selfie: model=SDXL/dreamshaperXL_alpha2Xl10.safetensors, frame=close-up, face focus, looking at camera..., content=sfw
```

### 3. Verify fail-fast on bad config

```bash
uv run python -c "
from sr2_spectre.tools.image_scenarios import ImageScenarioRegistry, ScenarioConfigError
import tempfile, os

# Bad model reference
bad_yaml = '''
models:
  dreamshaper:
    file: test.safetensors
    modalities: [txt2img]
frames:
  portrait:
    tags: close-up
contents:
  sfw:
    tags: sfw
    level: sfw
modalities:
  txt2img:
    template: txt2img.json
scenarios:
  bad:
    modality: txt2img
    model: nonexistent
    frame: portrait
    content: sfw
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(bad_yaml)
    path = f.name

try:
    ImageScenarioRegistry(path)
    print('FAIL: should have raised')
except ScenarioConfigError as e:
    print('PASS: caught error:', e)
finally:
    os.unlink(path)
"
```

**Expected output:**
```
PASS: caught error: Scenario validation failed:
  bad: model 'nonexistent' not found in models. Available: ['dreamshaper']
```

### 4. Verify model-modality mismatch detection

```bash
uv run python -c "
from sr2_spectre.tools.image_scenarios import ImageScenarioRegistry, ScenarioConfigError
import tempfile, os

# Model doesn't support img2img
bad_yaml = '''
models:
  dreamshaper:
    file: test.safetensors
    modalities: [txt2img]
frames:
  portrait:
    tags: close-up
contents:
  sfw:
    tags: sfw
    level: sfw
modalities:
  txt2img:
    template: txt2img.json
  img2img:
    template: img2img.json
scenarios:
  bad:
    modality: img2img
    model: dreamshaper
    frame: portrait
    content: sfw
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(bad_yaml)
    path = f.name

try:
    ImageScenarioRegistry(path)
    print('FAIL: should have raised')
except ScenarioConfigError as e:
    print('PASS: caught error:', e)
finally:
    os.unlink(path)
"
```

**Expected output:**
```
PASS: caught error: Scenario validation failed:
  bad: model 'dreamshaper' does not support modality 'img2img'. Model supports: ['txt2img']
```

## Verification Matrix

| Test | Expected |
|------|----------|
| 37 unit tests | All pass |
| Load bundled config | 3 scenarios resolved |
| Bad model ref | ScenarioConfigError with available list |
| Model-modality mismatch | ScenarioConfigError naming model + modality |

## Files Changed

- `src/sr2_spectre/tools/image_scenarios.py` — loader + validation module
- `src/sr2_spectre/resources/scenarios/image_scenarios.yaml` — sample config
- `tests/test_image_scenarios.py` — 37 tests
