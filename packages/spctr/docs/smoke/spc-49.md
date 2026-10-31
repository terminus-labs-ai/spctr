# Smoke: spc-49 — in-memory memory subsystem wiring

**Proves:** Spectre's Runtime constructs one shared `InMemoryMemoryStore` and threads it through `new_session()` → `Session` → `SR2`, so the memory resolver/transformer share a store across frames, and a fact extracted from one turn is retrievable on a later turn of the same process.

**Single caveat:** in-memory only. The store is dict-backed and lost on process restart — persistence across restart is OUT of scope (follow-on `obsidian-cor`). This runbook does NOT exercise a real LLM turn; the extract→inject roundtrip uses the rule-based extractor + real store/resolver directly.

---

## Setup

Move to the repo.

```bash
cd ~/git/sr2-spectre
```

Confirm you are on the branch with the wiring.

```bash
git log --oneline -1
```

---

## Scenario 1 — Runtime builds a shared in-memory store

Run the wiring test class.

```bash
uv run python -m pytest tests/test_memory_wiring.py::TestRuntimeMemoryStoreConstruction -q
```

**Expect:** `2 passed`. Runtime has a non-None `_memory_store` of type `InMemoryMemoryStore`, available at construction (no `initialize()` needed).

---

## Scenario 2 — Store threads Runtime → Session → SR2

```bash
uv run python -m pytest tests/test_memory_wiring.py::TestMemoryStoreThreading tests/test_memory_wiring.py::TestSessionMemoryStoreParam -q
```

**Expect:** `4 passed`. `new_session()` passes the Runtime's store to SR2; `Session(memory_store=...)` forwards it; absent param defaults to None.

---

## Scenario 3 — One store shared across frames

```bash
uv run python -m pytest tests/test_memory_wiring.py::TestSharedStoreAcrossSessions -q
```

**Expect:** `1 passed`. Two sessions receive the same store object (`is` identity).

---

## Scenario 4 — Extract → inject roundtrip in one process

```bash
uv run python -m pytest tests/test_memory_wiring.py::TestExtractInjectRoundtrip -q
```

**Expect:** `2 passed`. A fact stated in an assistant response is saved by the transformer and injected by the resolver on a later overlapping user turn; an unrelated query injects nothing.

---

## Scenario 5 — Resolver-only-injection invariant

```bash
uv run python -m pytest tests/test_memory_wiring.py::TestResolverOnlyInjection -q
```

**Expect:** `1 passed`. `resolve()` reads only — patching `store.save` to raise proves the resolver never writes.

---

## Scenario 6 — Live import check (no test harness)

Construct a Runtime and inspect the store directly. Single line, copy-paste.

```bash
uv run python -c "from unittest.mock import patch; from sr2_spectre.runtime import Runtime; from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig; cfg=SpectreConfig(agent=AgentConfig(name='t'), models={'default': ModelConfig(model='m', base_url='http://x')}, pipeline={'layers':[{'name':'system','target':'system','resolvers':[{'type':'static','config':{'text':'hi'}}]}]}); p=patch('sr2_spectre.runtime.LiteLLMCallable'); p.start(); rt=Runtime(config=cfg); print(type(rt._memory_store).__name__)"
```

**Expect:** prints `InMemoryMemoryStore`.

---

## Scenario 7 — No regression in the threading path

```bash
uv run python -m pytest tests/test_provenance_wiring.py tests/test_memory_wiring.py -q
```

**Expect:** all pass (provenance threading untouched by the memory addition).

---

## Teardown

Nothing to clean — no files, no DB, no processes. The in-memory store dies with each test process.

---

## Pass criteria

| Scenario | Command target | Expect |
|---|---|---|
| 1 | `TestRuntimeMemoryStoreConstruction` | 2 passed |
| 2 | `TestMemoryStoreThreading` + `TestSessionMemoryStoreParam` | 4 passed |
| 3 | `TestSharedStoreAcrossSessions` | 1 passed |
| 4 | `TestExtractInjectRoundtrip` | 2 passed |
| 5 | `TestResolverOnlyInjection` | 1 passed |
| 6 | live `-c` import | prints `InMemoryMemoryStore` |
| 7 | provenance + memory suites | all pass |

**Next real action unlocked when all green:** declare a `memory` resolver + `memory_extraction` transformer in a real pipeline config and run a live Discord/CLI turn to confirm cross-turn recall end-to-end with an LLM; then start `obsidian-cor` (persistent MemoryStore) to make memory survive restart.
