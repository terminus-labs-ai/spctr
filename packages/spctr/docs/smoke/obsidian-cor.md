# Smoke Runbook — obsidian-cor: Persistent Postgres MemoryStore

**Proves:** A `PostgresMemoryStore` persists agent memory to Postgres so a fact written by one process is readable by a separate, freshly-started process (restart / multi-process), the frequency counter increments across processes without loss, and the Runtime selects the Postgres backend from config/env (falling back to in-memory when disabled).
**Does NOT cover:** a live Discord/agent conversation actually accruing memory through the resolver+transformer at runtime (that is the natural usage path; wiring is covered by the automated suites). Also does not test concurrent simultaneous writers from threads sharing one connection.

> Every command is on a single line. Copy one line at a time. No line continuations.

---

## 0. One-time setup

```bash
export SPECTRE_MEMORY_DSN=postgresql://postgres:postgres@192.168.50.117:5432/spectre_memory
```

```bash
PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -tAc "SELECT version();"
```

**Expect:** prints a `PostgreSQL 17.x ...` line (DB reachable). If this fails, stop — the DB is down and nothing below will work.

Reset helper — gives a clean `memories` table per scenario (constructs a store so the table exists, then truncates):

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.close()" && PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -c "TRUNCATE memories;"
```

**Why the reset helper:** rows written by an earlier scenario (or an earlier run) would pollute `get_all()`/`search()` assertions. Run it at the top of each scenario for a clean fixture.

---

## 1. Cross-process persistence (headline)

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.close()" && PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -c "TRUNCATE memories;"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore, Memory; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.save(Memory(id='smoke1', content='diego prefers dark mode', tags=['pref'])); s.close(); print('PROC-A saved')"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); print('PROC-B reads:', [m.content for m in s.get_all()]); s.close()"
```

**Expect:** the second command prints `PROC-A saved`; the third (a brand-new process) prints `PROC-B reads: ['diego prefers dark mode']`. The fact survived the death of the writing process.

---

## 2. Frequency increments across processes (no lost update)

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.close()" && PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -c "TRUNCATE memories;"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore, Memory; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); print('first freq:', s.save(Memory(id='smoke2', content='repeated fact')).frequency); s.close()"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore, Memory; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); print('second freq:', s.save(Memory(id='smoke2', content='repeated fact')).frequency); s.close()"
```

**Expect:** first process prints `first freq: 0` (new id), second process prints `second freq: 1` — the increment is computed from the persisted row, not a private copy. Run the third command again → `second freq: 2`.

---

## 3. Search + tag lookup from a fresh process

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.close()" && PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -c "TRUNCATE memories;"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore, Memory; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); s.save(Memory(id='smoke3', content='the capital of France is Paris', tags=['geography'])); s.close(); print('saved')"
```

```bash
cd ~/git/sr2-spectre && .venv/bin/python -c "from sr2.memory import PostgresMemoryStore; import os; s=PostgresMemoryStore(os.environ['SPECTRE_MEMORY_DSN']); print('search:', [r.content for r in s.search('paris')]); print('tag:', [r.content for r in s.get_by_tag('geography')]); s.close()"
```

**Expect:** last command prints `search: ['the capital of France is Paris']` and `tag: ['the capital of France is Paris']` — case-insensitive content match and exact tag match work across processes.

---

## 4. Store unit + persistence suite (automated, real DB)

```bash
cd ~/git/sr2 && .venv/bin/python -m pytest tests/test_pg_memory_store.py -q
```

**Expect:** `25 passed` (or `skipped` only if the DB is unreachable). Covers roundtrip, ranking, delete, scope filters, and the two headline tests (cross-connection persistence + cross-instance frequency increment).

---

## 5. Runtime backend selection (automated)

```bash
cd ~/git/sr2-spectre && .venv/bin/python -m pytest tests/test_memory_store_selection.py -q
```

**Expect:** `10 passed`. Covers: default → in-memory; config DSN → Postgres; env `SPECTRE_MEMORY_DSN` → Postgres; config beats env; `""` disables even with env set; `aclose()` closes Postgres store; selected store threaded to sessions; real-DB Runtime construct+aclose.

---

## 6. Regression — existing memory + provenance wiring unaffected

```bash
cd ~/git/sr2-spectre && .venv/bin/python -m pytest tests/test_memory_wiring.py tests/test_provenance_wiring.py -q
```

**Expect:** `28 passed`, exit 0.

---

## 7. Teardown

```bash
PGPASSWORD=postgres psql -h 192.168.50.117 -U postgres -d spectre_memory -c "TRUNCATE memories;"
```

```bash
unset SPECTRE_MEMORY_DSN
```

---

## Pass criteria

| # | Expect |
|---|--------|
| 1 | PROC-B (fresh process) reads the fact PROC-A wrote |
| 2 | freq 0 on first save, 1 on second (separate process), 2 on third |
| 3 | search + get_by_tag return the fact from a fresh process |
| 4 | `25 passed` store suite |
| 5 | `10 passed` selection suite |
| 6 | `28 passed` regression |

All green → **persistent shared memory is live; next real action is enabling it in the deployed spectre config (`memory_store_dsn: postgresql://...`) and validating it through an actual agent conversation.**

**Caveat:** This proves the store and its wiring against real Postgres, but not a live agent turn accruing memory through the resolver/transformer, and not concurrent simultaneous writers on a shared connection. The DSN here carries plaintext `postgres:postgres` creds — fine for the lab PG, not for anything exposed.
