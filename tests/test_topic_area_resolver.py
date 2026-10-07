"""Identity-based topic context resolution through the public resolver API."""
import logging

import pytest

from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.resolvers.markdown_file import MarkdownTokenBudgetError
from sr2.pipeline.token_counting import CHARS_PER_TOKEN


def _topic(path, area="sr2-spectre"):
    path.mkdir(parents=True)
    (path / "README.md").write_text(f"---\nkind: topic\nid: topic:{area}\n---\nTopic\n")
    (path / "NOW.md").write_text(
        f"---\nkind: continuity\nfor: topic:{area}\nmode: source\n---\nSource NOW\n"
    )
    (path / "AGENTS.md").write_text("Selected AGENTS\n")
    return path


def _doc_text(filename, area, body):
    """Full document text for one filename, keeping identity frontmatter intact."""
    if filename == "README.md":
        return f"---\nkind: topic\nid: topic:{area}\n---\n{body}"
    if filename == "NOW.md":
        return f"---\nkind: continuity\nfor: topic:{area}\nmode: source\n---\n{body}"
    return body


def _build(root, filename="AGENTS.md", provider=lambda: {"area": "sr2-spectre"}, **options):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    config = ResolverConfig(
        type="topic_area", config={"topics_root": str(root), "filename": filename, **options}
    )
    return TopicAreaResolver.build(config, Dependencies(run_context_provider=provider))


async def _text(resolver):
    result = await resolver.resolve([])
    assert all(isinstance(block, TextBlock) for block in result.content)
    return "".join(block.text for block in result.content)


@pytest.mark.parametrize("filename", ["AGENTS.md", "NOW.md", "README.md"])
@pytest.mark.parametrize("folder,area", [
    ("sr2/sr2-spectre", "sr2-spectre"),
    ("family/henrique (pai)", "henrique"),
    ("unrelated/deep/directory", "sr2-spectre"),
])
async def test_selects_identity_at_any_depth(tmp_path, filename, folder, area):
    topic = _topic(tmp_path / folder, area)
    (topic / "extra.md").write_text("Excluded extra")
    resolver = _build(tmp_path, filename, provider=lambda: {"area": area})
    assert await _text(resolver) == (topic / filename).read_text()


@pytest.mark.parametrize("filename", ["AGENTS.md", "README.md"])
@pytest.mark.parametrize("document,metadata", [
    ("README.md", "kind: family\nid: topic:sr2-spectre"),
    ("README.md", "kind: topic\nid: topic:other"),
    ("README.md", "kind: topic"),
    ("README.md", "- topic:sr2-spectre"),
    ("README.md", "null"),
    ("README.md", "kind: [unterminated"),
    ("NOW.md", "kind: continuity\nfor: topic:sr2-spectre\nmode: view"),
    ("NOW.md", "kind: topic\nfor: topic:sr2-spectre\nmode: source"),
    ("NOW.md", "kind: continuity\nfor: topic:other\nmode: source"),
    ("NOW.md", "kind: continuity\nfor: topic:sr2-spectre"),
    ("NOW.md", "- continuity"),
    ("NOW.md", "null"),
    ("NOW.md", "kind: [unterminated"),
])
async def test_ineligible_metadata_never_injects(tmp_path, caplog, filename, document, metadata):
    topic = _topic(tmp_path / "sr2-spectre")
    (topic / document).write_text(f"---\n{metadata}\n---\nExcluded content")
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(tmp_path, filename)) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("document", ["README.md", "NOW.md"])
async def test_metadata_without_frontmatter_is_ineligible(tmp_path, document):
    topic = _topic(tmp_path / "sr2-spectre")
    (topic / document).write_text("kind: topic\nid: topic:sr2-spectre")
    assert await _text(_build(tmp_path)) == ""


@pytest.mark.parametrize("filename", ["AGENTS.md", "NOW.md", "README.md"])
async def test_nested_decoy_and_family_view_do_not_override_identity(tmp_path, filename):
    source = _topic(tmp_path / "sr2" / "actual")
    decoy = _topic(tmp_path / "characters" / "sr2-spectre", "vexa")
    view = _topic(tmp_path / "family" / "sr2-spectre")
    for topic, marker in [(source, "Intended source"), (decoy, "Excluded decoy"),
                          (view, "Excluded view")]:
        (topic / "AGENTS.md").write_text(marker + "\n")
        now = topic / "NOW.md"
        now.write_text(now.read_text().replace("Source NOW", marker))
        readme = topic / "README.md"
        readme.write_text(readme.read_text() + marker + "\n")
    now = view / "NOW.md"
    now.write_text(now.read_text().replace("mode: source", "mode: view"))
    text = await _text(_build(tmp_path, filename))
    assert text == (source / filename).read_text()
    assert "Intended source" in text
    assert "Excluded decoy" not in text and "Excluded view" not in text


@pytest.mark.parametrize("folder", [".hidden/topic", "parent/.hidden/topic"])
async def test_hidden_directories_are_excluded(tmp_path, folder):
    _topic(tmp_path / folder)
    assert await _text(_build(tmp_path)) == ""


@pytest.mark.parametrize("filename", ["AGENTS.md", "README.md"])
async def test_duplicate_eligible_identities_skip_with_warning(tmp_path, caplog, filename):
    _topic(tmp_path / "first")
    _topic(tmp_path / "nested/second")
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(tmp_path, filename)) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("filename,missing", [
    ("AGENTS.md", "README.md"), ("AGENTS.md", "NOW.md"), ("AGENTS.md", "AGENTS.md"),
    ("README.md", "README.md"), ("README.md", "NOW.md"),
])
async def test_missing_documents_skip_with_warning(tmp_path, caplog, filename, missing):
    topic = _topic(tmp_path / "topic")
    (topic / missing).unlink()
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(tmp_path, filename)) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_absent_root_is_valid_configuration_and_skips(tmp_path, caplog):
    resolver = _build(tmp_path / "absent")
    with caplog.at_level(logging.DEBUG):
        assert await _text(resolver) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("context", [None, {}, {"area": ""}, {"area": None}])
async def test_absent_area_skips_at_debug(tmp_path, caplog, context):
    _topic(tmp_path / "topic")
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(tmp_path, provider=lambda: context)) == ""
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


async def test_no_provider_skips_at_debug(tmp_path, caplog):
    _topic(tmp_path / "topic")
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(tmp_path, provider=None)) == ""
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("area", ["*", "../outside", "/absolute", "sr2-*"])
async def test_area_is_not_a_path_or_glob(tmp_path, area):
    _topic(tmp_path / "sr2-spectre")
    _topic(tmp_path / "outside", "outside")
    assert await _text(_build(tmp_path, provider=lambda: {"area": area})) == ""


@pytest.mark.parametrize("escaping", ["README.md", "NOW.md", "AGENTS.md", "directory"])
async def test_escaping_symlinks_never_inject(tmp_path, caplog, escaping):
    root = tmp_path / "topics"
    outside = _topic(tmp_path / "outside")
    root.mkdir()
    if escaping == "directory":
        (root / "linked").symlink_to(outside, target_is_directory=True)
    else:
        topic = _topic(root / "topic")
        (topic / escaping).unlink()
        (topic / escaping).symlink_to(outside / escaping)
    with caplog.at_level(logging.DEBUG):
        assert await _text(_build(root)) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize("filename", ["AGENTS.md", "NOW.md", "README.md"])
async def test_area_content_and_topic_moves_are_live(tmp_path, filename):
    topic = _topic(tmp_path / "first")
    other = _topic(tmp_path / "second", "henrique")
    (topic / "AGENTS.md").write_text("Spectre AGENTS\n")
    (other / "AGENTS.md").write_text("Henrique AGENTS\n")
    area = "sr2-spectre"
    resolver = _build(tmp_path, filename, provider=lambda: {"area": area})
    original, other_text = (topic / filename).read_text(), (other / filename).read_text()
    assert original != other_text
    assert await _text(resolver) == original
    changed = original + "Changed content"
    (topic / filename).write_text(changed)
    assert await _text(resolver) == changed
    area = "henrique"
    assert await _text(resolver) == other_text
    area = None
    assert await _text(resolver) == ""
    area = "henrique"
    assert await _text(resolver) == other_text
    area = "sr2-spectre"
    topic.rename(tmp_path / "moved")
    assert await _text(resolver) == changed


async def test_revalidates_metadata_and_duplicates_each_turn(tmp_path):
    topic = _topic(tmp_path / "topic")
    resolver = _build(tmp_path)
    assert await _text(resolver) == "Selected AGENTS\n"
    readme = topic / "README.md"
    identity = readme.read_text()
    readme.write_text(identity.replace("topic:sr2-spectre", "topic:other"))
    assert await _text(resolver) == ""
    readme.write_text(identity)
    assert await _text(resolver) == "Selected AGENTS\n"
    now = topic / "NOW.md"
    original = now.read_text()
    now.write_text(original.replace("mode: source", "mode: view"))
    assert await _text(resolver) == ""
    now.write_text(original)
    assert await _text(resolver) == "Selected AGENTS\n"
    duplicate = _topic(tmp_path / "duplicate")
    assert await _text(resolver) == ""
    (duplicate / "README.md").unlink()
    assert await _text(resolver) == "Selected AGENTS\n"
    (topic / "AGENTS.md").unlink()
    assert await _text(resolver) == ""


async def test_selected_path_containment_is_revalidated_each_turn(tmp_path, caplog):
    root = tmp_path / "topics"
    topic = _topic(root / "topic")
    outside = tmp_path / "outside.md"
    outside.write_text("Excluded outside")
    agents = topic / "AGENTS.md"
    resolver = _build(root)
    assert await _text(resolver) == "Selected AGENTS\n"
    agents.unlink()
    agents.symlink_to(outside)
    with caplog.at_level(logging.DEBUG):
        assert await _text(resolver) == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    agents.unlink()
    agents.write_text("Restored local content")
    assert await _text(resolver) == "Restored local content"


@pytest.mark.parametrize("filename", ["AGENTS.md", "README.md"])
async def test_budget_uses_sr2_approximation_and_rechecks_edits(tmp_path, filename):
    topic = _topic(tmp_path / "topic")
    target = topic / filename
    envelope = len(_doc_text(filename, "sr2-spectre", ""))
    base_tokens = -(-envelope // CHARS_PER_TOKEN)
    max_tokens = base_tokens + 2
    body = "x" * (max_tokens * CHARS_PER_TOKEN + CHARS_PER_TOKEN - 1 - envelope)
    target.write_text(_doc_text(filename, "sr2-spectre", body))
    resolver = _build(tmp_path, filename, max_tokens=max_tokens)
    assert await _text(resolver) == target.read_text()
    target.write_text(
        _doc_text(filename, "sr2-spectre", "x" * ((max_tokens + 1) * CHARS_PER_TOKEN - envelope))
    )
    with pytest.raises(MarkdownTokenBudgetError):
        await resolver.resolve([])


@pytest.mark.parametrize("filename", ["AGENTS.md", "README.md"])
@pytest.mark.parametrize("options", [{}, {"max_tokens": None}])
async def test_omitted_or_null_budget_is_unlimited(tmp_path, options, filename):
    topic = _topic(tmp_path / "topic")
    (topic / filename).write_text(
        _doc_text(filename, "sr2-spectre", "x" * (5000 * CHARS_PER_TOKEN))
    )
    assert await _text(_build(tmp_path, filename, **options)) == (topic / filename).read_text()


@pytest.mark.parametrize("override", [
    {"topics_root": None}, {"topics_root": ""}, {"topics_root": "relative"},
    {"topics_root": "/tmp/topics/*"},
    {"filename": "../AGENTS.md"}, {"filename": None},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 1.5},
    {"max_tokens": "4"}, {"max_tokens": True},
])
def test_invalid_configuration_is_rejected(tmp_path, override):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    config = {"topics_root": str(tmp_path), "filename": "AGENTS.md", **override}
    with pytest.raises((ValueError, TypeError)):
        TopicAreaResolver(ResolverConfig(type="topic_area", config=config))


def test_readme_filename_configuration_is_accepted(tmp_path):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    config = ResolverConfig(
        type="topic_area", config={"topics_root": str(tmp_path), "filename": "README.md"}
    )
    resolver = TopicAreaResolver.build(config, Dependencies(run_context_provider=None))
    assert resolver.name == "topic_area"


@pytest.mark.parametrize("filename", [
    "readme.md", "Readme.md", "AGENTS.MD", "now.md", "NOTES.md", "README.markdown",
    " README.md", "README.md ", "directory/README.md", "../README.md", "", "md",
])
def test_arbitrary_filenames_are_not_accepted(tmp_path, filename):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    config = ResolverConfig(
        type="topic_area", config={"topics_root": str(tmp_path), "filename": filename}
    )
    with pytest.raises((ValueError, TypeError)):
        TopicAreaResolver(config)


def test_missing_required_configuration_is_rejected(tmp_path):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    for config in [{"filename": "NOW.md"}, {"topics_root": str(tmp_path)}]:
        with pytest.raises((ValueError, TypeError)):
            TopicAreaResolver(ResolverConfig(type="topic_area", config=config))


def test_resolver_subscription_and_execution_contract(tmp_path):
    from sr2_spectre.pipeline.topic_area_resolver import TopicAreaResolver

    config = ResolverConfig(
        type="topic_area", config={"topics_root": str(tmp_path), "filename": "NOW.md"},
        max_executions=7,
        subscriptions=[EventSubscriptionConfig(event="custom_turn", phase="completed")],
    )
    resolver = TopicAreaResolver(config)
    assert resolver.max_executions == 7
    assert [(s.event_name, s.phase.value) for s in resolver.subscriptions] == [
        ("custom_turn", "completed")
    ]
