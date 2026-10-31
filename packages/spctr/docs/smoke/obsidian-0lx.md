# Smoke runbook — obsidian-0lx: Discord /hb Harbinger probe

**Proves:** typing `/hb` in a bot channel runs `harbinger status` and posts the dashboard as a code block, with no LLM call. Covers the probe function (offline) and the live bot path (manual Discord step).

**Does NOT cover:** Discord API delivery itself (rate limits, permissions) — only that the bot produces and sends the probe text. Masked `sr2-discord@*-dc` units are unrelated dead ends; the live bots are `sr2-discord@{edi,tali,liara,miranda}`.

---

## Setup

```
cd /home/shepard/git/sr2-spectre
```

```
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
```

---

## Scenario 1 — Probe unit tests green

```
$PY -m pytest tests/test_discord_handler.py::TestProbeHarbingerStatus -q
```

**Expect:** `6 passed`, exit 0.

---

## Scenario 2 — Command wiring tests green

```
$PY -m pytest tests/test_discord_handler.py tests/test_discord_interface.py -q
```

**Expect:** `67 passed`, exit 0.

---

## Scenario 3 — Probe runs the real CLI (offline of Discord)

```
$PY -c "import asyncio; from sr2_spectre.interfaces.discord.handler import probe_harbinger_status; print(asyncio.run(probe_harbinger_status()))"
```

**Expect:** A fenced code block starting with ```` ``` ```` then `Harbinger status —`, a `Live slots:` line, run outcomes, and recent runs. Total length ≤ 2000 chars.

---

## Scenario 4 — `hb` is in the loaded command set

```
$PY -c "from sr2_spectre.interfaces.discord.handler import SLASH_COMMANDS; print('hb' in SLASH_COMMANDS)"
```

**Expect:** `True`.

---

## Scenario 5 — Live bots restarted with this build

```
for s in edi tali liara miranda; do printf "%-10s " "$s"; systemctl --user is-active "sr2-discord@$s.service"; done
```

**Expect:** all four print `active`.

---

## Scenario 6 — End-to-end in Discord (manual)

In any channel one of the bots watches, send:

```
/hb
```

**Expect:** within ~2s the bot replies with a code block showing live slots + run outcomes + recent runs. No "⏳ Thinking…" message appears (proves the LLM was bypassed).

---

## Pass criteria

| Scenario | Pass when |
|----------|-----------|
| 1 | TestProbeHarbingerStatus → 6 passed |
| 2 | handler + interface → 67 passed |
| 3 | Code block with live `Harbinger status` text, ≤ 2000 chars |
| 4 | `hb` present in SLASH_COMMANDS |
| 5 | All four bot services `active` |
| 6 | `/hb` in Discord returns the dashboard, no "Thinking…" |

## Next real action unlocked

`/hb` is the on-demand fleet probe. If the truncated done/blocked feed matters, follow-up: trim the done feed for Discord or split into a second message with its own fence.
