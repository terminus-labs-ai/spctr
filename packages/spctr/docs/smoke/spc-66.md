# Smoke Runbook: spc-66 — Zone-1 Deterministic Scaffold Compiler + Logging

## What This Tests

The `scaffold_compiler.py` module assembles a deterministic positive prompt
from resolved scenario fragments — no LLM involved. Order is fixed:

    quality → frame.tags → content.tags → scenario.extra → LoRA triggers → intent

Negative prompt comes from the checkpoint's `negative` field.

FR10: Every call logs intent → compiled positive + negative + chosen
model/scenario at INFO level.

## Prerequisites

- Python 3.12+ with project deps installed (`uv sync` or `uv run`)
- Working directory: project root (`sr2-spectre`)

## Smoke Steps

### 1. Run the test suite

```bash
PYTHONPATH=src pytest tests/test_scaffold_compiler.py -v
```

**Expected:** 24 tests pass, 0 failures.

### 2. Compile scaffold from bundled config

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.image_scenarios import load_image_scenarios
from sr2_spectre.tools.scaffold_compiler import compile_scaffold
from pathlib import Path

registry = load_image_scenarios(
    Path('src/sr2_spectre/resources/scenarios/image_scenarios.yaml')
)

for name in registry.scenario_names():
    scenario = registry.get(name)
    compiled = compile_scaffold(scenario, 'character doing something')
    print(f'{name}:')
    print(f'  positive: {compiled.positive}')
    print(f'  negative: {compiled.negative}')
    print()
"
```

**Expected output:**
```
boudoir:
  positive: score_9, score_8_up, source_anime, full body, standing, full figure, nsfw, dramatic lighting, vexatoken, character doing something
  negative: score_4, score_5, worst quality, blurry

rooftop_scene:
  positive: masterpiece, best quality, detailed, wide shot, environment, cinematic, sfw, rainy night, neon signs, cyberpunk aesthetic, character doing something
  negative: low quality, blurry, deformed, bad anatomy, watermark, text

selfie:
  positive: masterpiece, best quality, detailed, close-up, face focus, looking at camera, sfw, selfie angle, casual, natural lighting, vexatoken, character doing something
  negative: low quality, blurry, deformed, bad anatomy, watermark, text
```

### 3. Verify FR10 logging

```bash
PYTHONPATH=src python -c "
import logging
from sr2_spectre.tools.image_scenarios import load_image_scenarios
from sr2_spectre.tools.scaffold_compiler import compile_scaffold
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

registry = load_image_scenarios(
    Path('src/sr2_spectre/resources/scenarios/image_scenarios.yaml')
)
scenario = registry.get('selfie')
compile_scaffold(scenario, 'vexa grinning on a rooftop at night')
"
```

**Expected output (INFO log):**
```
Scaffold compiled — scenario=selfie, model=SDXL/dreamshaperXL_alpha2Xl10.safetensors (natural):
  intent: vexa grinning on a rooftop at night
  positive: masterpiece, best quality, detailed, close-up, face focus, looking at camera, sfw, selfie angle, casual, natural lighting, vexatoken, vexa grinning on a rooftop at night
  negative: low quality, blurry, deformed, bad anatomy, watermark, text
```

### 4. Verify empty-field handling

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.image_scenarios import (
    ResolvedScenario, ModelFragment, FrameFragment,
    ContentFragment, ModalityFragment,
)
from sr2_spectre.tools.scaffold_compiler import compile_scaffold

# Minimal scenario — all optional fields empty
bare = ResolvedScenario(
    name='bare',
    modality=ModalityFragment(),
    model=ModelFragment(file='test.safetensors', quality='', negative=''),
    frame=FrameFragment(tags=''),
    content=ContentFragment(tags=''),
    loras=[],
    extra='',
)
result = compile_scaffold(bare, 'just the intent')
print(f'positive: [{result.positive}]')
print(f'negative: [{result.negative}]')
assert result.positive == 'just the intent', f'Expected only intent, got: {result.positive}'
assert result.negative == '', f'Expected empty negative, got: {result.negative}'
print('PASS: minimal scenario compiles to intent-only')
"
```

**Expected output:**
```
positive: [just the intent]
negative: []
PASS: minimal scenario compiles to intent-only
```

### 5. Verify LoRA trigger injection

```bash
PYTHONPATH=src python -c "
from sr2_spectre.tools.image_scenarios import (
    ResolvedScenario, ModelFragment, FrameFragment,
    ContentFragment, ModalityFragment, LoraFragment,
)
from sr2_spectre.tools.scaffold_compiler import compile_scaffold

scenario = ResolvedScenario(
    name='multi_lora',
    modality=ModalityFragment(),
    model=ModelFragment(
        file='pony.safetensors',
        quality='score_9',
        negative='score_4',
        modalities=['txt2img'],
    ),
    frame=FrameFragment(tags='full body'),
    content=ContentFragment(tags='nsfw'),
    loras=[
        LoraFragment(file='face.safetensors', trigger='facetoken'),
        LoraFragment(file='outfit.safetensors', trigger='outfittoken'),
        LoraFragment(file='style.safetensors', trigger=''),  # no trigger
    ],
    extra='',
)
result = compile_scaffold(scenario, 'character')
print(f'positive: {result.positive}')
assert 'facetoken, outfittoken' in result.positive
assert 'style' not in result.positive.lower()
print('PASS: triggers injected in order, empty triggers skipped')
"
```

**Expected output:**
```
positive: score_9, full body, nsfw, facetoken, outfittoken, character
PASS: triggers injected in order, empty triggers skipped
```

## Verification Matrix

| Test | Expected |
|------|----------|
| 24 unit tests | All pass |
| Compile bundled config | 3 scenarios, correct scaffold order |
| FR10 logging | INFO log with intent, positive, negative, model, scenario |
| Empty fields | Intent-only positive, empty negative |
| LoRA triggers | In-order injection, empty triggers skipped |

## Files Changed

- `src/sr2_spectre/tools/scaffold_compiler.py` — Zone-1 compiler + FR10 logging
- `tests/test_scaffold_compiler.py` — 24 tests
