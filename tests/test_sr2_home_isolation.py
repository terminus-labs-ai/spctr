"""Guard: the suite must never write session logs under the real ~/.sr2."""

from pathlib import Path

from sr2_spectre.run_log import SessionLogManager


def test_session_log_directory_is_not_real_sr2_home():
    real_home = Path("~/.sr2").expanduser().resolve()
    assert not SessionLogManager().directory.is_relative_to(real_home)
