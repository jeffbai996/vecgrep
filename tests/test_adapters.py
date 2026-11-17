from __future__ import annotations

import json
from datetime import datetime, timezone

from vecgrep.backend.api import routes
from vecgrep.backend.api.schemas import IndexRequest
from vecgrep.backend.ingestion.adapters import detect_adapter
from vecgrep.backend.ingestion.adapters.chatgpt_export import ChatGPTExportAdapter
from vecgrep.backend.ingestion.adapters.claude_export import ClaudeExportAdapter
from vecgrep.backend.ingestion.adapters.discord_jsonl import DiscordJSONLAdapter
from vecgrep.backend.ingestion.adapters.markdown import MarkdownAdapter
from vecgrep.backend.ingestion.adapters.plaintext import PlaintextAdapter
from vecgrep.backend.ingestion.dates import extract_timestamp
from vecgrep.backend.ingestion.enrich import chunk_enrichment


def test_detects_markdown(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# hi")
    assert detect_adapter(str(p)) is MarkdownAdapter


def test_detects_plaintext(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    assert detect_adapter(str(p)) is PlaintextAdapter


def test_discord_jsonl_detected_by_content(tmp_path):
    p = tmp_path / "ch.jsonl"
    p.write_text(json.dumps({"author": "alice", "content": "hi"}) + "\n")
    assert detect_adapter(str(p)) is DiscordJSONLAdapter


def test_discord_jsonl_rejects_non_discord_jsonl(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"metric": "cpu", "value": 0.5}) + "\n")
    # No author/content → not discord. Either falls back to plaintext (yes,
    # .jsonl isn't in plaintext exts) or raises AdapterError. Either is fine
    # — the key invariant is we don't misclassify it as discord.
    assert not DiscordJSONLAdapter.matches(str(p))


def test_claude_export_detected(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text(
        json.dumps(
            [
                {
                    "uuid": "x",
                    "name": "test",
                    "created_at": "2026",
                    "chat_messages": [
                        {"sender": "human", "content": "hi"},
                        {"sender": "assistant", "content": "yes"},
                    ],
                }
            ]
        )
    )
    assert detect_adapter(str(p)) is ClaudeExportAdapter


def test_chatgpt_export_detected(tmp_path):
    p = tmp_path / "chatgpt.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "title": "test",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                        "n1": {
                            "id": "n1",
                            "parent": "n0",
                            "children": [],
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["hi"]},
                            },
                        },
                    },
                }
            ]
        )
    )
    assert detect_adapter(str(p)) is ChatGPTExportAdapter


def test_claude_and_chatgpt_dont_collide(tmp_path):
    """Both use .json — make sure schemas keep them separate."""
    claude = tmp_path / "claude.json"
    claude.write_text(
        json.dumps(
            [{"uuid": "x", "chat_messages": [{"sender": "u", "content": "hi"}]}]
        )
    )
    chatgpt = tmp_path / "chatgpt.json"
    chatgpt.write_text(
        json.dumps(
            [
                {
                    "title": "x",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": [], "message": None}
                    },
                }
            ]
        )
    )
    assert ClaudeExportAdapter.matches(str(claude))
    assert not ClaudeExportAdapter.matches(str(chatgpt))
    assert ChatGPTExportAdapter.matches(str(chatgpt))
    assert not ChatGPTExportAdapter.matches(str(claude))


def test_chatgpt_linearizes_main_thread(tmp_path):
    """When the message graph branches (edits / regenerations), we follow
    the first child. Branch siblings are intentionally dropped — they
    represent alternate threads the user didn't pick.
    """
    p = tmp_path / "chatgpt.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "c",
                    "title": "branchy",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                        "n1": {
                            "id": "n1",
                            "parent": "n0",
                            # Two children = a branch. We take the first.
                            "children": ["n2", "n3"],
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["q?"]},
                            },
                        },
                        "n2": {
                            "id": "n2",
                            "parent": "n1",
                            "children": [],
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["main answer"]},
                            },
                        },
                        "n3": {
                            "id": "n3",
                            "parent": "n1",
                            "children": [],
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["alt regen"]},
                            },
                        },
                    },
                }
            ]
        )
    )
    docs = list(ChatGPTExportAdapter().load(str(p)))
    assert len(docs) == 1
    assert "main answer" in docs[0].text
    assert "alt regen" not in docs[0].text


def test_discord_load_preserves_message_order(tmp_path):
    p = tmp_path / "ch.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"author": "alice", "content": f"msg{i}"})
            for i in range(3)
        )
    )
    docs = list(DiscordJSONLAdapter().load(str(p)))
    assert len(docs) == 1
    text = docs[0].text
    assert text.index("msg0") < text.index("msg1") < text.index("msg2")


def test_json_file_cache_avoids_repeated_loads(tmp_path, monkeypatch):
    from unittest.mock import patch

    from vecgrep.backend.ingestion.adapters.base import (
        load_json_file,
        try_load_json_file,
    )
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"key": "val"}))

    # First read parses and caches
    obj1 = try_load_json_file(p)
    assert obj1 == {"key": "val"}

    # Subsequent reads with unmodified file should not call json.loads
    with patch("json.loads") as loads_mock:
        obj2 = try_load_json_file(p)
        assert obj2 == {"key": "val"}
        obj3 = load_json_file(p, str(p))
        assert obj3 == {"key": "val"}
        loads_mock.assert_not_called()


def test_chatgpt_and_claude_export_adapters_reject_non_list_json_root(tmp_path):
    import pytest

    from vecgrep.backend.ingestion.adapters.base import AdapterError

    null_json = tmp_path / "null.json"
    null_json.write_text("null", encoding="utf-8")
    dict_json = tmp_path / "dict.json"
    dict_json.write_text('{"mapping": {}}', encoding="utf-8")

    for adapter in (ChatGPTExportAdapter(), ClaudeExportAdapter()):
        with pytest.raises(AdapterError):
            list(adapter.load(str(null_json)))
        with pytest.raises(AdapterError):
            list(adapter.load(str(dict_json)))

# ── test_index_include_glob ──

def test_include_glob_restricts_directory_walk(svc, tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    # Same conversation, two formats side by side.
    (d / "2026-05-01.md").write_text("The committee held rates steady through Q2.")
    (d / "2026-05-01.jsonl").write_text(
        '{"author": "fed", "content": "The committee held rates steady through Q2."}'
    )
    (d / "2026-05-02.md").write_text("Labor data softened in the spring.")
    (d / "2026-05-02.jsonl").write_text(
        '{"author": "fed", "content": "Labor data softened in the spring."}'
    )

    docs, _unused_chunks, _ = svc.index(str(d), "md_only", include="*.md")
    assert docs == 2, f"expected only the 2 .md files, indexed {docs}"

    # Every indexed source must be a .md — no .jsonl leaked in.
    sources = svc.registry.get("md_only").sources
    assert all(s.endswith(".md") for s in sources), sources
    assert not any(s.endswith(".jsonl") for s in sources)


def test_no_include_indexes_everything(svc, tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "a.md").write_text("markdown content here.")
    (d / "b.jsonl").write_text('{"author": "x", "content": "jsonl content here."}')

    docs, _, _ = svc.index(str(d), "all", include=None)
    assert docs == 2  # both formats indexed when no filter

# ── test_index_chunker_failure ──

class _RaisesOn:
    def __init__(self, inner, suffix: str) -> None:
        self.inner, self.suffix = inner, suffix

    def chunk_doc(self, doc):
        if doc.source_id.endswith(self.suffix):
            raise RecursionError("maximum recursion depth exceeded")
        return self.inner.chunk_doc(doc)


def _repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "a.py").write_text("def alpha():\n    return 1\n")
    (d / "b.py").write_text("def beta():\n    return 2\n")
    (d / "c.py").write_text("def gamma():\n    return 3\n")
    return d


def test_a_chunker_failure_skips_that_source_and_reports_it(svc, tmp_path, monkeypatch):
    d = _repo(tmp_path)
    real = svc.chunker("code_symbol")
    monkeypatch.setattr(svc, "chunker", lambda name: _RaisesOn(real, "b.py"))
    monkeypatch.setattr(routes, "_service", lambda: svc)

    out = routes.index(IndexRequest(source=str(d), corpus="code", chunker="code_symbol"))

    assert out.docs == 2
    assert len(out.failed) == 1
    assert out.failed[0].startswith(str((d / "b.py").resolve())) and "RecursionError" in out.failed[0]
    corpus = svc.registry.get("code")
    assert sorted(s.rsplit("/", 1)[-1] for s in corpus.sources) == ["a.py", "c.py"]


def test_a_skipped_source_is_tried_again_on_the_next_run(svc, tmp_path, monkeypatch):
    d = _repo(tmp_path)
    real = svc.chunker("code_symbol")
    monkeypatch.setattr(svc, "chunker", lambda name: _RaisesOn(real, "b.py"))
    svc.index(str(d), "code", "code_symbol")

    monkeypatch.setattr(svc, "chunker", lambda name: real)
    failed: list[str] = []
    docs, _, skipped = svc.index(str(d), "code", "code_symbol", failed=failed)

    assert (docs, skipped, failed) == (1, 2, [])

# ── test_enrich ──

ARCHIVER_CHUNK = """**alice** · 21:27
> ok can you queue that
**helper_bot [bot]** · 21:28
> Queued. Also here is a table:
> | col | val |
> |-----|-----|
> | a   | 1   |
**carol** · 21:30
> see https://example.com/thing
"""

CODE_CHUNK = """**bot_a [bot]** · 10:00
> here's the fix:
> ```python
> print("hi")
> ```
"""

PLAIN_CHUNK = "just some prose with no structure at all"


def test_speakers_extracted_in_order_no_dupes():
    meta = chunk_enrichment(ARCHIVER_CHUNK)
    assert meta["speakers"] == ["alice", "helper_bot [bot]",
                                "carol"]


def test_bot_flag_from_bot_marker_not_name_list():
    assert chunk_enrichment(ARCHIVER_CHUNK)["has_bot_speaker"] is True
    assert chunk_enrichment(CODE_CHUNK)["has_bot_speaker"] is True
    solo = chunk_enrichment("**alice** · 09:00\n> hi\n")
    assert solo["has_bot_speaker"] is False


def test_content_shapes():
    meta = chunk_enrichment(ARCHIVER_CHUNK)
    assert meta["has_table"] is True
    assert meta["has_link"] is True
    assert meta["has_code"] is False

    code = chunk_enrichment(CODE_CHUNK)
    assert code["has_code"] is True
    assert code["has_table"] is False
    assert code["has_link"] is False


def test_plain_text_all_negative_no_speakers():
    meta = chunk_enrichment(PLAIN_CHUNK)
    assert meta["speakers"] == []
    assert meta["has_bot_speaker"] is False
    assert meta["has_code"] is False and meta["has_table"] is False
    assert meta["has_link"] is False


def test_pipe_in_prose_is_not_a_table():
    """A stray | in prose (or a single pipe row with no separator) must not
    count — a table needs the |---| separator row."""
    meta = chunk_enrichment("a | b in prose\n| lone row |\n")
    assert meta["has_table"] is False

# ── test_dates ──

def _epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def test_saved_line_iso_with_microseconds_and_tz() -> None:
    """Shared-memory format: 'Saved: 2026-05-18T06:45:21.178768+00:00'."""
    text = "# Memory\n\nName: foo\nSaved: 2026-05-18T06:45:21.178768+00:00\n\nbody"
    ts = extract_timestamp(text)
    assert ts == _epoch("2026-05-18T06:45:21.178768+00:00")


def test_frontmatter_date_only() -> None:
    """transcript frontmatter: 'date: 2026-05-27'."""
    text = "---\nchannel: cl-2\ndate: 2026-05-27\n---\n\nalice: hi"
    ts = extract_timestamp(text)
    assert ts == _epoch("2026-05-27")


def test_labeled_line_wins_over_filename() -> None:
    text = "Saved: 2026-01-02T00:00:00+00:00\nblah"
    ts = extract_timestamp(text, path="/x/2099-12-31.md")
    assert ts == _epoch("2026-01-02T00:00:00+00:00")


def test_filename_date_fallback_when_no_labeled_line() -> None:
    ts = extract_timestamp("just prose, no dates here", path="/logs/2026-05-27.md")
    assert ts == _epoch("2026-05-27")


def test_mtime_fallback(tmp_path) -> None:
    p = tmp_path / "undated.md"
    p.write_text("no date anywhere")
    ts = extract_timestamp("no date anywhere", path=str(p))
    assert ts == p.stat().st_mtime


def test_none_when_no_text_date_and_no_path() -> None:
    assert extract_timestamp("no date, no path") is None


def test_date_deep_in_body_does_not_win() -> None:
    """A date mentioned far down in prose must not be treated as the doc date."""
    body = "\n".join([f"line {i}" for i in range(40)] + ["Saved: 2020-01-01"])
    assert extract_timestamp(body) is None  # past the 15-line head window


def test_never_raises_on_garbage() -> None:
    assert extract_timestamp("Saved: not-a-date\ndate: also-bad") is None


def test_markdown_adapter_populates_timestamp(tmp_path) -> None:
    p = tmp_path / "memory-1.md"
    p.write_text("Name: x\nSaved: 2026-05-18T06:45:21+00:00\n\nbody text here")
    docs = list(MarkdownAdapter().load(str(p)))
    assert len(docs) == 1
    assert docs[0].timestamp == _epoch("2026-05-18T06:45:21+00:00")
