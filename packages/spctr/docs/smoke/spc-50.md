# Smoke Test Runbook — spc-50: Wire persistent SQLiteProvenanceStore

**Proves:** Runtime constructs a shared SQLiteProvenanceStore from config, threads it through Session → SR2, and provenance entries persist across process restarts. **Does NOT cover:** live agent conversation with provenance tracking (that's the natural usage path; the unit/integration tests cover the wiring).

## Setup

```bash
cd ~/git/sr2-spectre
```

## Scenarios

### 1. Unit: Config field defaults and accepts overrides

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestConfigProvenanceStorePath" -q
```

**Expect:** `4 passed` — defaults to None, accepts custom path, empty string disables, tilde paths work.

### 2. Unit: Path resolution (default, custom, disabled, tilde)

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestResolveProvenancePath" -q
```

**Expect:** `4 passed` — default resolves to `~/.sr2-spectre/provenance.db`, custom paths pass through, empty string → None, tilde expands.

### 3. Unit: Runtime.initialize() connects store

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestRuntimeInitializeProvenance" -q
```

**Expect:** `2 passed` — connects when enabled, skips when disabled.

### 4. Unit: Runtime.aclose() closes store

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestRuntimeACloseProvenance" -q
```

**Expect:** `2 passed` — closes store, no-op when absent.

### 5. Unit: Threading Runtime → Session → SR2

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestProvenanceStoreThreading" -q
```

**Expect:** `3 passed` — store passed to Session, Session passes to SR2, None when not initialized.

### 6. Integration: Persistence across reconnect (simulated restart)

```bash
uv run pytest tests/test_provenance_wiring.py -k "TestPersistenceAcrossRestart" -q
```

**Expect:** `3 passed` — entries survive close/reopen, multiple sessions share store, full Runtime lifecycle works.

### 7. Regression: Existing Runtime/Session/Agent tests unaffected

```bash
uv run pytest tests/test_runtime.py tests/test_agent.py -q
```

**Expect:** `54 passed` (31 runtime + 23 agent), exit 0.

### 8. Full wiring suite

```bash
uv run pytest tests/test_provenance_wiring.py -v
```

**Expect:** `18 passed`, exit 0.

## Teardown

None — temp files are cleaned up by test fixtures.

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | Config field tests | [ ] |
| 2 | Path resolution | [ ] |
| 3 | Initialize connects | [ ] |
| 4 | AClose closes | [ ] |
| 5 | Threading | [ ] |
| 6 | Persistence | [ ] |
| 7 | Regression (runtime + agent) | [ ] |
| 8 | Full wiring suite | [ ] |

## Config Usage

Add to any agent YAML or spectre.yaml:

```yaml
# Enable persistent provenance (default behavior — path auto-resolved)
# provenance_store_path: ~/.sr2-spectre/provenance.db

# Custom path
provenance_store_path: /var/lib/spectre/provenance.db

# Disable (fall back to in-memory)
provenance_store_path: ""
```
