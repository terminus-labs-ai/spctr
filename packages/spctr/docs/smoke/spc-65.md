# Smoke Test Runbook — spc-65: Tool contract flip (intent + tool-owned negative)

**Proves:** The generate_image tool contract has flipped from literal SDXL prompt
to intent-based input, negative_prompt is tool-owned per checkpoint, and
`style` has been replaced by `scenario`. **Does NOT cover:** live ComfyUI
generation (requires running ComfyUI instance; covered by existing live smoke
test gated on `COMFYUI_URL`).

## Setup

```bash
cd ~/git/sr2-spectre
```

## Scenarios

### 1. Unit: input schema reflects new contract

```bash
uv run pytest tests/test_generate_image.py -k "input_schema or class_attributes" -v
```

**Expect:** `2 passed` — schema has `intent` and `scenario`, no `prompt`, no `negative_prompt`, no `style`.

### 2. Unit: tool description states contract

```bash
uv run pytest tests/test_generate_image.py -k "description_mentions" -v
```

**Expect:** `2 passed` — description mentions intent is not literal SDXL prompt, and negative prompt is tool-owned.

### 3. Unit: negative prompt is tool-owned

```bash
uv run pytest tests/test_generate_image.py -k "negative" -v
```

**Expect:** `4 passed` — default negative used when none configured, custom negative honored, tool-owned negative used in workflow, negative not in input schema.

### 4. Unit: scenario presets replace style presets

```bash
uv run pytest tests/test_generate_image.py -k "scenario" -v
```

**Expect:** `3 passed` — scenario presets defined, scenario applied in assembly, unknown scenario ignored.

### 5. Unit: legacy prompt kwarg rejected

```bash
uv run pytest tests/test_generate_image.py -k "legacy_prompt" -v
```

**Expect:** `1 passed` — calling with `prompt=` raises TypeError.

### 6. Unit: full generate_image + comfyui_client suite

```bash
uv run pytest tests/test_generate_image.py tests/test_comfyui_client.py -q
```

**Expect:** `38 passed, 1 skipped` (live smoke skipped without COMFYUI_URL).

### 7. Live: generate with intent (optional, requires ComfyUI)

```bash
COMFYUI_URL=http://192.168.50.233:8188 uv run pytest tests/test_generate_image.py -k "live_smoke" -v
```

**Expect:** `1 passed` — image generated from intent, PNG file exists and is >1KB.

## Teardown

None — no state written.

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | input schema reflects new contract | [ ] |
| 2 | tool description states contract | [ ] |
| 3 | negative prompt is tool-owned | [ ] |
| 4 | scenario presets replace style presets | [ ] |
| 5 | legacy prompt kwarg rejected | [ ] |
| 6 | full test suite green | [ ] |
| 7 | live smoke (optional) | [ ] |

## Next Action

All green → contract flip is complete. Agent sends `intent` and optional `scenario`; tool compiles the full SDXL prompt with checkpoint-owned negative.
