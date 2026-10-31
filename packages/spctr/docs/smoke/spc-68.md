# Smoke Test Runbook — spc-68: Template loader + structured patch-map

**Proves:** ComfyUI workflow templates load by modality and accept structured patches keyed by `(node_id, input_key)`. The seeded `txt2img.json` template is equivalent to the legacy `_build_text2img_workflow()` builder.

**Does NOT cover:** img2img or other modalities (Phase 1 is txt2img only).

## Setup

```bash
cd ~/git/sr2-spectre
```

## Scenarios

### 1. Unit: template loader + patch-map tests

```bash
uv run pytest tests/test_workflow_loader.py -v
```

**Expect:** `15 passed` — covers load, patch, deep-copy isolation, error paths, and round-trip equivalence with `_build_text2img_workflow`.

### 2. Unit: generate_image tests still pass (no regression)

```bash
uv run pytest tests/test_generate_image.py -q
```

**Expect:** `21 passed, 1 skipped` (live smoke skipped unless `COMFYUI_URL` set).

### 3. Interactive: build_workflow round-trip

```bash
uv run python -c "
from sr2_spectre.tools.builtins.workflow_loader import build_workflow
wf = build_workflow('txt2img', {('3', 'seed'): 42, ('6', 'text'): 'a cat', ('7', 'text'): 'blurry'})
print('seed:', wf['3']['inputs']['seed'])
print('positive:', wf['6']['inputs']['text'])
print('negative:', wf['7']['inputs']['text'])
print('nodes:', sorted(wf.keys()))
"
```

**Expect:**
```
seed: 42
positive: a cat
negative: blurry
nodes: ['3', '4', '5', '6', '7', '8', '9']
```

### 4. Interactive: template equivalence with legacy builder

```bash
uv run python -c "
from sr2_spectre.tools.builtins.generate_image import GenerateImageTool
from sr2_spectre.tools.builtins.workflow_loader import build_workflow

tool = GenerateImageTool()
built = tool._build_text2img_workflow('hello', 'world', 123)
patched = build_workflow('txt2img', {('3','seed'):123, ('6','text'):'hello', ('7','text'):'world'})
assert built == patched, 'MISMATCH'
print('OK: template+patches == _build_text2img_workflow')
"
```

**Expect:** `OK: template+patches == _build_text2img_workflow`

## Teardown

None — no state written.

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | template loader unit tests | [ ] |
| 2 | generate_image regression | [ ] |
| 3 | interactive build_workflow | [ ] |
| 4 | template equivalence | [ ] |

## Next Action

All green → spc-69 (node-stack injection) can use `build_workflow()` + `apply_patches()` as its foundation for injecting LoRA nodes between checkpoint and sampler.
