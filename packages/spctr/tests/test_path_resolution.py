"""Tests for path_resolution module (FR10).

Tests cover:
- ${VAR} interpolation with known env var
- ${SR2_HOME} interpolation
- Unset ${VAR} raises ConfigPathError
- Absolute path (no vars) returned as-is
- Absolute path with ${VAR} interpolated then used as-is
- Relative path resolved against declaring file's directory
- Relative path with ${VAR} prefix combined correctly
- Multiple ${VAR} tokens in one string
- declaring_file in nested dir resolves against that dir, not project root
- env={} with ${VAR} in path raises ConfigPathError
- Result is always an absolute Path object
"""


from pathlib import Path

import pytest

from sr2_spectre.path_resolution import ConfigPathError, resolve_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_declaring_file(tmp_path: Path, *parts: str) -> Path:
    """Return a path to a (notional) config file in tmp_path/parts."""
    config_file = tmp_path.joinpath(*parts)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    return config_file


# ---------------------------------------------------------------------------
# ${VAR} interpolation — happy path
# ---------------------------------------------------------------------------


def test_interpolation_known_var(tmp_path: Path) -> None:
    """${VAR} is replaced with its value from env."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {"MY_DIR": "/some/abs/dir"}
    result = resolve_path("${MY_DIR}/sub", declaring, env=env)
    assert result == Path("/some/abs/dir/sub")
    assert result.is_absolute()


def test_sr2_home_interpolation(tmp_path: Path) -> None:
    """${SR2_HOME} resolves to the value passed in env."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {"SR2_HOME": "/opt/sr2"}
    result = resolve_path("${SR2_HOME}/resolvers/chat.yaml", declaring, env=env)
    assert result == Path("/opt/sr2/resolvers/chat.yaml")
    assert result.is_absolute()


def test_multiple_var_tokens(tmp_path: Path) -> None:
    """All ${VAR} tokens in a single string are interpolated."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {"BASE": "/mnt", "SUBDIR": "data", "SR2_HOME": "/opt/sr2"}
    result = resolve_path("${BASE}/${SUBDIR}/file.yaml", declaring, env=env)
    assert result == Path("/mnt/data/file.yaml")
    assert result.is_absolute()


def test_interpolation_yields_relative_then_resolved_against_declaring(tmp_path: Path) -> None:
    """${VAR} that yields a relative fragment is anchored to declaring file's dir."""
    declaring = make_declaring_file(tmp_path, "nested", "config.yaml")
    env = {"REL": "sub/path"}
    result = resolve_path("${REL}/file.yaml", declaring, env=env)
    expected = (tmp_path / "nested" / "sub" / "path" / "file.yaml").resolve()
    assert result == expected
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# Unset var — error cases
# ---------------------------------------------------------------------------


def test_unset_var_raises_config_path_error(tmp_path: Path) -> None:
    """${UNSET_VAR} raises ConfigPathError with the var name in the message."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {}
    with pytest.raises(ConfigPathError) as exc_info:
        resolve_path("${UNSET_VAR}/path", declaring, env=env)
    assert "UNSET_VAR" in str(exc_info.value)


def test_unset_var_in_nonempty_env_raises(tmp_path: Path) -> None:
    """Only the referenced var being missing (not all vars) raises ConfigPathError."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {"OTHER_VAR": "/something"}
    with pytest.raises(ConfigPathError) as exc_info:
        resolve_path("${MISSING}/path", declaring, env=env)
    assert "MISSING" in str(exc_info.value)


def test_sr2_home_absent_raises(tmp_path: Path) -> None:
    """If ${SR2_HOME} referenced but not in env, raises ConfigPathError."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {}
    with pytest.raises(ConfigPathError) as exc_info:
        resolve_path("${SR2_HOME}/thing", declaring, env=env)
    assert "SR2_HOME" in str(exc_info.value)


def test_never_silently_substitutes_empty(tmp_path: Path) -> None:
    """A truly absent var raises ConfigPathError (not silently substituted)."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    env = {}
    with pytest.raises(ConfigPathError):
        resolve_path("${ABSENT}", declaring, env=env)


# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------


def test_absolute_path_no_vars(tmp_path: Path) -> None:
    """Absolute path with no vars is returned as-is (resolved)."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    result = resolve_path("/usr/local/bin/thing", declaring, env={})
    assert result == Path("/usr/local/bin/thing")
    assert result.is_absolute()


def test_absolute_path_with_var(tmp_path: Path) -> None:
    """${VAR} in an absolute path string: var interpolated, result used as-is."""
    declaring = make_declaring_file(tmp_path, "subdir", "config.yaml")
    env = {"PREFIX": "/opt/myapp"}
    result = resolve_path("${PREFIX}/resolvers/main.yaml", declaring, env=env)
    assert result == Path("/opt/myapp/resolvers/main.yaml")
    assert result.is_absolute()


def test_absolute_path_ignores_declaring_file_dir(tmp_path: Path) -> None:
    """Absolute path is NOT resolved relative to declaring file's directory."""
    declaring = make_declaring_file(tmp_path, "deep", "nested", "config.yaml")
    result = resolve_path("/absolute/path/file.yaml", declaring, env={})
    assert result == Path("/absolute/path/file.yaml")


# ---------------------------------------------------------------------------
# Relative paths
# ---------------------------------------------------------------------------


def test_relative_path_resolved_against_declaring_dir(tmp_path: Path) -> None:
    """Relative path is resolved against the declaring file's parent directory."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    result = resolve_path("resolvers/chat.yaml", declaring, env={})
    expected = (tmp_path / "resolvers" / "chat.yaml").resolve()
    assert result == expected
    assert result.is_absolute()


def test_relative_path_nested_declaring_file(tmp_path: Path) -> None:
    """Relative path resolves against the NESTED declaring file's dir, not project root."""
    declaring = make_declaring_file(tmp_path, "envs", "prod", "config.yaml")
    result = resolve_path("../shared/base.yaml", declaring, env={})
    expected = (tmp_path / "envs" / "prod" / ".." / "shared" / "base.yaml").resolve()
    assert result == expected
    assert result.is_absolute()


def test_relative_path_not_resolved_against_cwd(tmp_path: Path) -> None:
    """Relative path must NOT fall back to cwd."""
    declaring = make_declaring_file(tmp_path, "sub", "config.yaml")
    result = resolve_path("somefile.yaml", declaring, env={})
    assert result == (tmp_path / "sub" / "somefile.yaml").resolve()


def test_relative_path_with_var_prefix(tmp_path: Path) -> None:
    """${VAR} that expands to a relative fragment combines with declaring file's dir."""
    declaring = make_declaring_file(tmp_path, "configs", "config.yaml")
    env = {"SUBPATH": "resolvers/main.yaml"}
    result = resolve_path("${SUBPATH}", declaring, env=env)
    expected = (tmp_path / "configs" / "resolvers" / "main.yaml").resolve()
    assert result == expected
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# env defaults
# ---------------------------------------------------------------------------


def test_env_defaults_to_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When env=None (default), os.environ is used for lookups."""
    monkeypatch.setenv("TEST_SPECTRE_VAR", "/from/environ")
    declaring = make_declaring_file(tmp_path, "config.yaml")
    result = resolve_path("${TEST_SPECTRE_VAR}/file.yaml", declaring)
    assert result == Path("/from/environ/file.yaml")


def test_explicit_env_overrides_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit env dict is used instead of os.environ."""
    monkeypatch.setenv("MY_OVERRIDE_VAR", "/from/os_environ")
    declaring = make_declaring_file(tmp_path, "config.yaml")
    explicit_env = {"MY_OVERRIDE_VAR": "/from/explicit"}
    result = resolve_path("${MY_OVERRIDE_VAR}/x", declaring, env=explicit_env)
    assert result == Path("/from/explicit/x")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_result_is_always_path_object(tmp_path: Path) -> None:
    """resolve_path always returns a Path, never a string."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    result = resolve_path("relative/file.yaml", declaring, env={})
    assert isinstance(result, Path)


def test_result_is_always_absolute(tmp_path: Path) -> None:
    """resolve_path always returns an absolute Path."""
    declaring = make_declaring_file(tmp_path, "config.yaml")
    result = resolve_path("relative/file.yaml", declaring, env={})
    assert result.is_absolute()
