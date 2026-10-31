# Smoke Test — obsidian-a3q: Discord auto_thread wrong-topic fix

**Proves:** with `auto_thread: true`, each new `@bot` mention in a parent channel spawns its OWN fresh thread, so a second topic does NOT land in the first topic's thread. Continuation still works by talking inside a thread.

**Does NOT cover:** the real LLM reply content/quality, GPU model-load timing, or DM behavior. This is routing only.

---

## Setup

Confirm the fix is in the running tree and services pick it up (editable install).

```
cd /home/shepard/git/sr2-spectre
```

```
git log --oneline -1
```
Expect: HEAD is the obsidian-a3q fix commit.

```
grep -n "always starts a NEW thread" src/sr2_spectre/interfaces/discord/interface.py
```
Expect: one match (the fix comment present).

```
grep -rn "get_thread_for_parent\|link_parent_thread" src/sr2_spectre/
```
Expect: NO matches (parent->thread reuse fully removed).

```
systemctl --user restart sr2-discord@edi sr2-discord@tali sr2-discord@liara sr2-discord@miranda
```
Expect: returns with no error.

```
systemctl --user is-active sr2-discord@edi sr2-discord@tali sr2-discord@liara sr2-discord@miranda
```
Expect: four lines, all `active`.

---

## Scenario 1 — Two topics in a parent channel get two threads

In the Normandy server, in a normal text channel (NOT inside a thread), post:

```
@edi topic one: what is 2+2
```
Expect: EDI creates a thread named "topic one: what is 2+2" and replies INSIDE that thread.

Then, back in the SAME parent channel, post:

```
@edi topic two: what is the capital of France
```
Expect: EDI creates a SECOND, distinct thread named "topic two: ..." and replies inside it. The reply does NOT appear in the "topic one" thread. **(This is the bug being fixed.)**

---

## Scenario 2 — Continuation inside a thread stays in that thread

Inside the "topic one" thread from Scenario 1, post (no mention needed):

```
and times ten?
```
Expect: EDI replies in the SAME "topic one" thread, with context from that thread. No new thread is created.

---

## Scenario 3 — No mention in parent = silence

In the parent channel (not a thread), post WITHOUT mentioning the bot:

```
just chatting, no bot here
```
Expect: no reply, no thread created.

---

## Teardown

```
journalctl --user -u sr2-discord@edi -n 20 --no-pager
```
Expect: no `Agent stream error` lines from the run. Optionally archive/delete the test threads in Discord.

---

## Pass criteria

| Scenario | Pass condition |
|----------|----------------|
| Setup | grep shows fix present, no reuse API; 4 services `active` |
| 1 | Two parent mentions → two distinct threads; topic-two reply NOT in topic-one thread |
| 2 | In-thread follow-up stays in same thread, no new thread |
| 3 | Unmentioned parent message → no reply, no thread |

**Next real action when all green:** close obsidian-a3q; spot-check tali/liara/miranda with the same Scenario 1 (each runs identical code, own token).
