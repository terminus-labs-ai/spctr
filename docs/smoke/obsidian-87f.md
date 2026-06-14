# Smoke — obsidian-87f: 4 spectre agent Discord bots as systemd services

**Proves:** edi/tali/liara/miranda each run as an always-on systemd user service
(`sr2-discord@<agent>`), auto-start on boot, log into Discord on their own bot
token, and reply to an @mention end-to-end.

**Does NOT cover:** GPU model-swap thrash under concurrent load (4 bots, 2 router
lanes — messaging two same-lane agents at once will serialize/swap); the
first reply after an idle period is slow (cold qwen3.6:27b load). Also does not
cover the pre-existing `MCP server failed to connect` warning (a tool is absent;
unrelated to the Discord seam).

**Cutover note:** these replace the old `hermes-gateway-<agent>` services on the
SAME bot tokens. The two MUST NOT run together (one token, two gateway clients =
Discord thrashes the session). This runbook assumes hermes is stopped+disabled.

---

## Artifacts (already installed by the task — listed for reference)

Templated unit:
```
/home/shepard/.config/systemd/user/sr2-discord@.service
```

Per-agent token files (mode 600, token only, sourced from each hermes profile):
```
/home/shepard/.sr2/discord/edi.env
```
```
/home/shepard/.sr2/discord/tali.env
```
```
/home/shepard/.sr2/discord/liara.env
```
```
/home/shepard/.sr2/discord/miranda.env
```

Global discord block (token via `${DISCORD_BOT_TOKEN}`, `mention_only: true`):
```
/home/shepard/.sr2/spectre.yaml
```

---

## Setup

Reload systemd in case units changed:
```
systemctl --user daemon-reload
```

---

## Scenario 1 — all 4 hermes gateways are down (no token collision)

```
for a in edi tali liara miranda; do printf '%-8s ' "$a"; systemctl --user is-active hermes-gateway-$a; done
```
**Expect:** every line reads `inactive` or `failed` (not `active`).

## Scenario 2 — all 4 spectre bots are active and enabled

```
for a in edi tali liara miranda; do printf '%-8s active=' "$a"; systemctl --user is-active sr2-discord@$a; done
```
**Expect:** four lines, each `active=active`.

```
for a in edi tali liara miranda; do printf '%-8s enabled=' "$a"; systemctl --user is-enabled sr2-discord@$a; done
```
**Expect:** four lines, each `enabled=enabled`.

## Scenario 3 — each bot logged into Discord on its own identity

```
grep "logged in as" /home/shepard/.sr2-spectre/spectre.log | tail -4
```
**Expect:** four distinct lines naming EDI, Tali, Liara, Miranda, each with a
different bot ID.

## Scenario 4 — live reply end-to-end (MANUAL, in Discord)

In a channel the edi bot can see, send (literally @-mention the bot):
```
@EDI say hello in five words
```
**Expect:** EDI replies in-channel. First reply may take 10–60s (cold model
load); subsequent replies are fast. Because `mention_only: true`, the bot
answers ONLY when mentioned.

Confirm the inbound message was handled (no traceback):
```
tail -30 /home/shepard/.sr2-spectre/spectre.log | grep -iE "error|traceback|exception" | tail
```
**Expect:** no output (no errors).

## Scenario 5 — auto-restart on crash (proves Restart=always)

Get the edi bot PID:
```
systemctl --user show sr2-discord@edi -p MainPID --value
```

Kill it (substitute the PID printed above):
```
kill -9 <PID>
```

Wait a few seconds, then re-check:
```
systemctl --user is-active sr2-discord@edi
```
**Expect:** `active` again (systemd restarted it within ~5s).

```
grep "logged in as EDI" /home/shepard/.sr2-spectre/spectre.log | tail -1
```
**Expect:** a fresh login line with a newer timestamp.

---

## Teardown / rollback (only if reverting to hermes)

Stop + disable the spectre bots:
```
systemctl --user disable --now sr2-discord@edi sr2-discord@tali sr2-discord@liara sr2-discord@miranda
```

Restore the hermes gateways (frees nothing — same tokens, now single client):
```
systemctl --user enable --now hermes-gateway-edi hermes-gateway-tali hermes-gateway-liara hermes-gateway-miranda
```

---

## Pass criteria

| # | Scenario | Pass when |
|---|----------|-----------|
| 1 | hermes down | all 4 hermes-gateway not `active` |
| 2 | spectre up | all 4 sr2-discord@ `active` AND `enabled` |
| 3 | logged in | 4 distinct "logged in as" lines (EDI/Tali/Liara/Miranda) |
| 4 | live reply | @mention → in-channel reply, no traceback |
| 5 | auto-restart | killed bot returns to `active` + fresh login |

**Next real action when all green:** decide whether the 2-lane GPU thrash under
concurrent multi-bot chat is acceptable, or whether to add per-lane batching
(beeline Phase 3). If concurrent chat is painful, that's the trigger.
