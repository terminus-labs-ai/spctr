# Discord Smoke Test Runbook

**Purpose:** Verify all 4 spectre Discord bots (edi, liara, miranda, tali) respond to @mention after the systemd cutover.

## Prerequisites

- Discord desktop app or web client open
- Access to the shared Discord channel where all bots are present
- Terminal access to the machine running systemd user services

## Pre-flight: Verify Services Running

```bash
systemctl --user list-units 'sr2-discord@*'
```

Expected: 4 units listed, all `active running`.

| Service | Agent | Expected |
|---------|-------|----------|
| `sr2-discord@edi.service` | EDI | active running |
| `sr2-discord@liara.service` | Liara | active running |
| `sr2-discord@miranda.service` | Miranda | active running |
| `sr2-discord@tali.service` | Tali | active running |

If any service is failed, check logs:
```bash
journalctl --user -u 'sr2-discord@<agent>' --no-pager -n 30
```

## Smoke Test: @Mention Each Bot

In the Discord channel, send one message per bot:

1. `@EDI-Bot ping` → EDI should reply (⏳ Thinking... → response)
2. `@Liara-Bot ping` → Liara should reply
3. `@Miranda-Bot ping` → Miranda should reply
4. `@Tali-Bot ping` → Tali should reply

**Pass criteria:** Each bot responds with a non-error message within ~30 seconds.

## Troubleshooting

### Bot not responding
```bash
# Check service status
systemctl --user status sr2-discord@<agent>

# Check logs for startup errors
journalctl --user -u 'sr2-discord@<agent>' --no-pager -n 50

# Restart service
systemctl --user restart sr2-discord@<agent>
```

### Token issues
Verify the token file exists and is readable:
```bash
cat ~/.sr2/discord/<agent>.env
```

### Conflicting hermes-gateway service
Ensure old hermes-gateway services are stopped:
```bash
systemctl --user list-units 'hermes-gateway-<agent>*'
```
If running, stop:
```bash
systemctl --user stop hermes-gateway-<agent>
```

## Architecture Notes

- **4 bots, 2 GPU router lanes:** Concurrent conversations may cause model-swap thrash. The router alternates models between active conversations. Not a bug — a known constraint.
- **Token storage:** `~/.sr2/discord/<agent>.env` (mode 600). Contains only `DISCORD_BOT_TOKEN`.
- **Systemd template:** `~/.config/systemd/user/sr2-discord@.service`. Uses `%i` for agent name, `%h/.sr2/discord/%i.env` for env file.
- **Global discord config:** `~/.sr2/spectre.yaml` `discord:` block (reused by all bots). Token is injected via env var `${DISCORD_BOT_TOKEN}`.
- **Per-agent config:** `~/.sr2/agents/<agent>.yaml` → extends `base.yaml` → extends `~/.sr2/config.yaml`.

## Auto-start on Boot

All services are enabled via `WantedBy=default.target`:
```bash
systemctl --user is-enabled sr2-discord@edi
# Expected: enabled
```
