'''Tests that AGENTS.md reconciles bd dolt push/pull guidance with vault policy.

Covers the bead: AGENTS.md instructs bd dolt push, which the vault CLAUDE.md
forbids.

These are documentation-content assertions in the style of
test_claude_md_docs.py (bead obsidian-n1r3): they read the text of AGENTS.md
and check for required / forbidden substrings, not prose quality.

Acceptance criteria (from the bead body):
  AC1. AGENTS.md hand-written (non-auto-generated) sections that mention
       bd dolt push / bd dolt pull must include an explicit caveat that a
       vault-synced checkout (e.g. /data/obsidian) must NOT run bd dolt push,
       bd dolt pull, or bd import, and that such a checkout own CLAUDE.md is
       authoritative there.
  AC2. The auto-generated BEGIN BEADS INTEGRATION ... END BEADS INTEGRATION
       block must remain byte-for-byte unchanged (it is tool-managed).
  AC3. AGENTS.md must still be valid, non-empty markdown containing the
       Agent Instructions H1 title.
'''

import hashlib
from pathlib import Path

AGENTS_MD = Path(__file__).parent.parent / 'AGENTS.md'

BEGIN_MARKER = '<!-- BEGIN BEADS INTEGRATION'
END_MARKER = '<!-- END BEADS INTEGRATION -->'

# Snapshot of the tool-managed BEADS INTEGRATION block (markers inclusive),
# captured from the current AGENTS.md. This chore only edits hand-written text
# OUTSIDE the markers, so the block must stay byte-for-byte identical.
BEADS_BLOCK_SHA256 = '42095e304e103f4d506977cef645c19a2acd368eff6d11bbf5cdc18008f65fbd'
BEADS_BLOCK_LEN = 1879


def _read_agents_md():
    return AGENTS_MD.read_text(encoding='utf-8')


def _beads_block(text):
    '''Return the auto-generated block, markers inclusive.'''
    i = text.index(BEGIN_MARKER)
    j = text.index(END_MARKER) + len(END_MARKER)
    return text[i:j]


def _handwritten(text):
    '''Return AGENTS.md with the auto-generated block removed.

    What remains is exactly the hand-written material this chore is allowed
    to edit.
    '''
    return text.replace(_beads_block(text), '')


# =========================================================================
# AC1 - hand-written sections carry the vault-sync caveat
# =========================================================================


def test_handwritten_has_vault_sync_caveat():
    '''AC1: hand-written text calls out the vault-synced checkout case.'''
    hand = _handwritten(_read_agents_md()).lower()
    assert 'vault' in hand
    assert ('/data/obsidian' in hand) or ('vault-synced' in hand)


def test_handwritten_caveat_forbids_dolt_and_import_commands():
    '''AC1: the caveat names the three forbidden commands and prohibits them.'''
    hand = _handwritten(_read_agents_md())
    assert 'bd dolt push' in hand
    assert 'bd dolt pull' in hand
    assert 'bd import' in hand
    lower = hand.lower()
    assert any(p in lower for p in ('must not', 'do not', 'never', 'not be run'))


def test_handwritten_caveat_defers_to_vault_claude_md():
    '''AC1: the caveat points at the vault checkout CLAUDE.md as authoritative.'''
    hand = _handwritten(_read_agents_md()).lower()
    assert 'claude.md' in hand
    assert 'authoritative' in hand


# =========================================================================
# AC2 - auto-generated block is untouched
# =========================================================================


def test_beads_integration_block_present_exactly_once():
    '''AC2: exactly one BEGIN and one END marker remain.'''
    text = _read_agents_md()
    assert text.count(BEGIN_MARKER) == 1
    assert text.count(END_MARKER) == 1


def test_beads_integration_block_unchanged():
    '''AC2: the tool-managed block matches its captured snapshot exactly.'''
    block = _beads_block(_read_agents_md())
    assert len(block) == BEADS_BLOCK_LEN
    assert hashlib.sha256(block.encode('utf-8')).hexdigest() == BEADS_BLOCK_SHA256


# =========================================================================
# AC3 - sanity: valid, non-empty markdown with the expected H1
# =========================================================================


def test_agents_md_non_empty():
    '''AC3: AGENTS.md is not empty.'''
    assert _read_agents_md().strip() != ''


def test_agents_md_has_agent_instructions_h1():
    '''AC3: the Agent Instructions H1 title survives.'''
    lines = [line.strip() for line in _read_agents_md().splitlines()]
    assert '# Agent Instructions' in lines
