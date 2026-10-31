# Smoke Test Runbook: spc-46 — Tool-Log Merge + /ask Thread Routing

**Purpose:** Verify that tool events render inline within the thinking message
(no separate embed messages), and that `/ask` routes to the thread channel
instead of the parent channel.

**Commits:**
- `1fe329f` feat(discord): collapse tool events into single log message (spc-46)
- `09baa28` spc-46: merge tool-log into thinking message + fix /ask thread routing

---

## Prerequisites

```bash
cd /home/shepard/git/sr2-spectre
git pull origin main
```

## Unit Test Gate

```bash
python3 -m pytest tests/test_discord_interface.py -v
```

**Expected:** 24 tests pass, including these new tests:
- `test_slash_ask_resolves_thread_channel` — `/ask` routes through `_resolve_target_channel`
- `test_tool_events_merged_into_thinking_message` — tool lines go into thinking message, not embeds
- `test_tool_log_suppressed_when_disabled` — no tool log when `tool_embed_enabled=False`
- `test_mention_bypassed_in_thread_with_active_session` — mention not required in active thread
- `test_mention_still_required_in_thread_without_session` — orphan threads still need mention
- `test_mention_still_required_in_parent_channel` — parent channels always need mention
- `test_mention_bypass_not_applied_when_mention_only_false` — bypass irrelevant when mention_only=False

```bash
python3 -m pytest tests/test_discord_handler.py tests/test_discord_session_map.py tests/test_discord_auto_thread.py tests/test_discord_adapter.py tests/test_discord_config.py -v
```

**Expected:** All 73 additional Discord tests pass (97 total across all Discord test files).

## Manual Smoke Tests (Discord)

These require a live Discord bot running with the updated code.

### 1. Tool events appear inline in thinking message

1. Start the bot or restart the service:
   ```bash
   systemctl --user restart sr2-discord@<agent>
   ```
2. In Discord, mention the bot and ask it to do something that triggers tool use (e.g., "search for Python documentation" or "grep for 'TODO' in the codebase").
3. Observe the thinking message:
   - **PASS:** Tool activity appears as inline text within the thinking message:
     ```
     ▶ `web_search`
     ✓ `web_search`
     ⏳ Thinking...
     ```
   - **FAIL:** Separate embed messages appear for each tool event, or a separate tool-log message is sent.

### 2. Tool embeds disabled

1. Set `tool_embed_enabled: false` in your Discord config.
2. Trigger a tool use as above.
3. **PASS:** No tool activity appears in the thinking message.
4. **FAIL:** Tool lines still appear, or separate messages are sent.

### 3. /ask routes to thread

1. With `auto_thread: true` enabled, send a regular message to trigger thread creation.
2. Then send `/ask follow-up question` from the **parent channel**.
3. **PASS:** The response appears in the **thread**, not the parent channel.
4. **FAIL:** The response appears in the parent channel, splitting the conversation.

### 4. Mention bypass in thread

1. Enter the auto-created thread from step 3.
2. Send a message **without** mentioning the bot.
3. **PASS:** The bot responds (mention check bypassed because there's an active session in the thread).
4. **FAIL:** The bot ignores the message (mention still required).

### 5. Mention still required in parent

1. In the parent channel (not a thread), send a message without a mention.
2. **PASS:** The bot ignores it.
3. **FAIL:** The bot responds.

## Files Changed

| File | Change |
|------|--------|
| `src/sr2_spectre/interfaces/discord/interface.py` | Merged tool-log into thinking message; fixed /ask thread routing |
| `tests/test_discord_interface.py` | 7 new tests for tool-log merge + mention bypass + /ask routing |

## Rollback

If issues arise, revert both commits:
```bash
git revert --no-edit 09baa28 1fe329f
```
