# Smoke runbook — obsidian-9fs: slash commands in mention_only mode

**Proves:** with `mention_only: true` (the live config), an `@bot /hb` message reaches the slash fast-path because the leading bot mention is stripped before parsing. Before the fix it fell through to the LLM and produced nothing.

**Does NOT cover:** bare `/hb` with no mention — that is still (correctly) ignored in mention_only mode. You must @mention a bot.

---

## Setup

```
cd /home/shepard/git/sr2-spectre
```

```
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
```

---

## Scenario 1 — strip_bot_mention unit tests

```
$PY -m pytest tests/test_discord_handler.py::TestStripBotMention -q
```

**Expect:** `6 passed`.

---

## Scenario 2 — mention_only interface repro

```
$PY -m pytest "tests/test_discord_interface.py::test_slash_hb_works_with_mention_prefix_in_mention_only" -q
```

**Expect:** `1 passed`.

---

## Scenario 3 — strip enables parse (one-liner)

```
$PY -c "from sr2_spectre.interfaces.discord.handler import strip_bot_mention, parse_slash_command; print(parse_slash_command(strip_bot_mention('<@111> /hb', ['<@111>'], 111)))"
```

**Expect:** `('hb', '')`.

---

## Scenario 4 — live bots restarted with the fix

```
for s in edi tali liara miranda; do printf "%-10s " "$s"; systemctl --user is-active "sr2-discord@$s.service"; done
```

**Expect:** all four `active`.

---

## Scenario 5 — end-to-end in Discord (manual)

In a channel a bot watches, send (replace `@edi` with a real bot):

```
@edi /hb
```

**Expect:** edi (only edi) replies within ~2s with the Harbinger dashboard code block. Bare `/hb` with no mention is ignored — that is expected.

---

## Pass criteria

| Scenario | Pass when |
|----------|-----------|
| 1 | TestStripBotMention → 6 passed |
| 2 | mention_only interface test → 1 passed |
| 3 | prints `('hb', '')` |
| 4 | all four bot services active |
| 5 | `@edi /hb` returns the dashboard; bare `/hb` ignored |

## Next real action unlocked

All slash commands (`/hb`, `/reset`, `/help`, `/status`) now work via `@bot /cmd` in mention_only mode, not just bare `/cmd` in open channels.
