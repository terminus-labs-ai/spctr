# Smoke Runbook — obsidian-165h (turn-status indicator: TUI + Discord)

**Proves:** the agent-working/done indicator is actually visible — TUI shows `⏳ Working...` on its own screen row (not hidden behind the Footer), and the Discord adapter holds the typing indicator for the whole turn via a real async context manager.
**Does NOT cover:** a live Discord connection. The Discord scenario asserts the adapter contract with a fake bot/channel; it does not log into Discord and watch the real "Bot is typing…" dots. For that, restart the live bot and eyeball it (step 3).

> Every command is on a single line. Copy one line at a time. No line continuations.

---

## 0. One-time setup

```bash
cd /home/shepard/git/sr2-spectre
```

```bash
git log --oneline -3
```

**Expect:** top two commits are `fix(discord): make channel_typing a real async context manager` and `fix(tui): un-dock status row so it is not occluded by Footer`.

---

## 1. TUI — working indicator renders on its own row (not occluded)

```bash
.venv/bin/python -m pytest tests/test_tui_streaming.py::test_status_row_not_occluded_by_footer -q
```

**Expect:** `1 passed`. The test asserts `#status` and `Footer` occupy different screen regions and that `"Working"` appears in `app.export_screenshot()`.

---

## 2. TUI — visual region check (manual, no eyeballing a terminal)

```bash
.venv/bin/python -c "import asyncio; from unittest.mock import MagicMock; from sr2_spectre.interfaces.tui import SpectreTUI; from textual.widgets import Static, Footer
async def m():
 app=SpectreTUI(MagicMock())
 async with app.run_test(size=(80,24)) as p:
  app.set_working_status(); await p.pause()
  print('status', app.query_one('#status',Static).region.y, 'footer', app.query_one(Footer).region.y, 'visible', 'Working' in app.export_screenshot())
asyncio.run(m())"
```

**Expect:** `status 20 footer 23 visible True` — status row (y=20) is above the prompt and distinct from Footer (y=23), and the text renders.

---

## 3. Discord — channel_typing holds typing for the whole block

```bash
.venv/bin/python -m pytest tests/test_discord_adapter.py::test_channel_typing_is_usable_as_async_context_manager -q
```

**Expect:** `1 passed`. The test enters `async with adapter.channel_typing(id)`, asserts the underlying `channel.typing()` was entered during the block and exited after.

---

## 4. Regression — full TUI + Discord suites

```bash
.venv/bin/python -m pytest tests/test_tui_streaming.py -k discord tests/test_discord_adapter.py tests/test_discord_interface.py -q
```

```bash
.venv/bin/python -m pytest tests/ -k discord -q
```

**Expect:** all pass (TUI 17, Discord 118). Note: `test_tool_test_guard.py::TestRealCollection` (2 tests) fail independently of this work — pre-existing, tracked separately.

---

## 5. Live restart sanity (optional, real path)

```bash
ps -o pid,lstart,cmd -C python --sort=start_time | grep -E "interface (tui|discord)"
```

**Expect:** every running `--interface tui|discord` process has an `lstart` AFTER the merge commit time. Any process older than the merge is stale and still runs the occluded/broken code — restart it.

---

## N. Teardown

Nothing to tear down — tests use temp fixtures and fake bots; no external state created.

---

## Pass criteria

| # | Expect |
|---|--------|
| 1 | `test_status_row_not_occluded_by_footer` passes |
| 2 | `status 20 footer 23 visible True` |
| 3 | `test_channel_typing_is_usable_as_async_context_manager` passes |
| 4 | TUI + Discord suites green |
| 5 | no `--interface tui|discord` process older than the merge commit |

All green → **obsidian-165h closeable; the turn-status indicator is live on both surfaces after daemon restart.**

**Caveat:** step 3 proves the adapter *contract*, not a real Discord session. Final confirmation that Discord users see "Bot is typing…" requires a restarted live bot and a human watching a real channel.
