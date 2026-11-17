from __future__ import annotations

import json
import multiprocessing
from dataclasses import asdict
from pathlib import Path

import pytest

from vecgrep.backend import config as config_mod
from vecgrep.backend.store.corpora import Corpus, CorpusError, CorpusRegistry

# ── test_diagnose_embed_split ──

def _reg(svc, name, model, backend="ollama"):
    svc.registry.upsert(Corpus(name=name, embed_backend=backend,
                               embed_model=model, dim=8, chunk_count=0))


def test_one_model_everywhere_is_clean(svc):
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    kinds = [i["kind"] for i in svc.diagnose()]
    assert "embed_model_split" not in kinds


def test_the_minority_corpus_is_named(svc):
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    _reg(svc, "c", "bge-m3")
    split = [i for i in svc.diagnose() if i["kind"] == "embed_model_split"]
    assert [i["corpus"] for i in split] == ["c"]
    assert "bge-m3:batch8k" in split[0]["detail"]
    assert split[0]["fixable"] is False


def test_a_different_backend_is_not_a_split(svc):
    """openai and ollama corpora obviously use different model names; only
    disagreement WITHIN one backend costs anything."""
    _reg(svc, "a", "bge-m3:batch8k", backend="ollama")
    _reg(svc, "b", "text-embedding-3-small", backend="openai")
    assert [i for i in svc.diagnose() if i["kind"] == "embed_model_split"] == []


def test_filtering_still_compares_against_the_whole_install(svc):
    """`--corpora c` must not make c its own majority and report clean."""
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    _reg(svc, "c", "bge-m3")
    split = [i for i in svc.diagnose(corpora={"c"}) if i["kind"] == "embed_model_split"]
    assert [i["corpus"] for i in split] == ["c"]

# ── test_corpus_context ──

def test_old_registry_loads_empty_routing_context(tmp_path) -> None:
    corpus = Corpus("old", "stub", "stub-1", 16)
    payload = asdict(corpus)
    payload.pop("description")
    payload.pop("use_for")
    payload.pop("avoid_for")
    path = tmp_path / "corpora.json"
    path.write_text(json.dumps({"old": payload}))

    loaded = CorpusRegistry(path).get("old")

    assert loaded.description == ""
    assert loaded.use_for == []
    assert loaded.avoid_for == []


def test_context_round_trips_clears_and_survives_export_import(
    svc, make_doc, tmp_path
) -> None:
    svc.index(str(make_doc("doc.md", "alpha beta gamma")), "notes")
    updated = svc.set_corpus_context(
        "notes",
        "  Durable notes.  ",
        [" facts ", " project state"],
        [" live prices "],
    )
    assert updated.description == "Durable notes."
    assert updated.use_for == ["facts", "project state"]
    assert updated.avoid_for == ["live prices"]
    assert svc.list_corpora()[0].description == "Durable notes."

    archive = svc.export_corpus("notes", tmp_path / "notes.tar.gz")
    svc.delete_corpus("notes")
    restored = svc.import_corpus(archive, rename="restored")
    assert restored.description == "Durable notes."
    assert restored.use_for == ["facts", "project state"]
    assert restored.avoid_for == ["live prices"]

    cleared = svc.set_corpus_context("restored")
    assert (cleared.description, cleared.use_for, cleared.avoid_for) == ("", [], [])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"description": "x" * 501}, "500"),
        ({"use_for": [str(i) for i in range(9)]}, "at most 8"),
        ({"avoid_for": ["x" * 241]}, "240"),
        ({"use_for": ["  "]}, "cannot be empty"),
    ],
)
def test_context_limits(svc, make_doc, kwargs, message) -> None:
    svc.index(str(make_doc("doc.md", "alpha")), "notes")
    with pytest.raises(CorpusError, match=message):
        svc.set_corpus_context("notes", **kwargs)

# ── test_cross_corpus_scope ──

def _index(svc, make_doc, name, corpus, text="alpha beta gamma delta. epsilon zeta eta theta."):
    p = make_doc(name, text)
    svc.index(str(p), corpus)
    return str(p)


def test_unscoped_search_skips_eval_corpora(svc, make_doc) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "copy.md", "eval-chats-base")

    hits = svc.search("alpha beta gamma", top_k=20)
    assert hits, "sanity: the live corpus must still match"
    assert {h.corpus for h in hits} == {"chats"}


def test_explicit_corpus_still_reaches_an_eval_corpus(svc, make_doc) -> None:
    _index(svc, make_doc, "copy.md", "eval-chats-base")
    hits = svc.search("alpha beta gamma", corpus_name="eval-chats-base", top_k=20)
    assert hits, "naming an eval corpus explicitly must still search it"
    assert {h.corpus for h in hits} == {"eval-chats-base"}


def test_exclusion_patterns_come_from_config(svc, make_doc, monkeypatch) -> None:
    """The list is config, not a hardcoded prefix."""
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "scratch.md", "scratch-pad")

    # default config does not exclude scratch-*
    assert "scratch-pad" in {h.corpus for h in svc.search("alpha beta", top_k=20)}

    monkeypatch.setattr(svc.settings, "cross_corpus_exclude", ["scratch-*"])
    corpora = {h.corpus for h in svc.search("alpha beta", top_k=20)}
    assert "scratch-pad" not in corpora
    assert "chats" in corpora


def test_empty_exclusion_list_searches_everything(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "copy.md", "eval-chats-base")
    monkeypatch.setattr(svc.settings, "cross_corpus_exclude", [])
    assert {h.corpus for h in svc.search("alpha beta", top_k=20)} == {"chats", "eval-chats-base"}


def test_exclusion_never_empties_an_explicit_single_corpus_search(svc, make_doc) -> None:
    """Guard against the exclusion accidentally applying to the scoped path."""
    _index(svc, make_doc, "copy.md", "eval-run-2")
    assert svc.search("alpha", corpus_name="eval-run-2", top_k=5)


def test_parallel_fanout_matches_serial_results(svc, make_doc, monkeypatch) -> None:
    """Concurrency must not change what comes back, only how fast."""
    for i, c in enumerate(("notes", "code", "docs", "wiki")):
        _index(svc, make_doc, f"d{i}.md", c,
               f"alpha beta gamma corpus {c}. delta epsilon zeta {c} tail.")

    parallel = svc.search("alpha beta gamma", top_k=20)

    monkeypatch.setattr(svc.settings, "search_fanout_workers", 1)
    serial = svc.search("alpha beta gamma", top_k=20)

    key = lambda rs: sorted((r.corpus, r.source_id, r.chunk_start) for r in rs)
    assert key(parallel) == key(serial)
    assert len(parallel) == len(serial)

# ── test_corpora_registry_churn ──

def _corpus(name: str = "c", **kw) -> Corpus:
    return Corpus(name=name, embed_backend="ollama", embed_model="m", dim=4, **kw)


def test_identical_upsert_does_not_rewrite_the_file(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())

    # _save() writes a temp file and os.replace()s it in, so the inode -- not
    # the mtime, which the replacement carries over -- is what proves whether
    # 2.2 MB and two fsyncs actually hit the disk.
    before_ino = path.stat().st_ino
    before_bytes = path.read_bytes()

    reg.upsert(_corpus())

    assert path.stat().st_ino == before_ino
    assert path.read_bytes() == before_bytes


def test_a_real_change_still_writes(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    before = path.read_bytes()

    reg.upsert(_corpus(doc_count=7))

    assert path.read_bytes() != before
    assert reg.get("c").doc_count == 7


def test_delete_still_writes(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    reg.upsert(_corpus("d"))
    before = path.read_bytes()

    reg.delete("d")

    assert path.read_bytes() != before
    assert not reg.has("d")


def test_a_new_corpus_is_written_even_on_a_fresh_registry(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    assert path.exists()
    assert CorpusRegistry(path, in_memory=False).has("c")


def test_reload_if_needed_skips_lock_when_stat_matches(tmp_path):
    from unittest.mock import patch
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus("test_corp"))

    # When querying get / list / has without disk modification, registry_read lock is NOT acquired
    with patch.object(reg.locks, "registry_read") as lock_mock:
        assert reg.has("test_corp")
        c = reg.get("test_corp")
        assert c.name == "test_corp"
        all_c = reg.list()
        assert len(all_c) == 1
        lock_mock.assert_not_called()

# ── test_unscoped_exclude ──

def _index_unscoped_exclude(svc, make_doc, name, corpus, text="alpha beta gamma delta. epsilon zeta eta theta."):
    p = make_doc(name, text)
    svc.index(str(p), corpus)
    return str(p)


def test_unscoped_search_skips_a_corpus_in_unscoped_exclude(svc, make_doc, monkeypatch) -> None:
    _index_unscoped_exclude(svc, make_doc, "live.md", "chats")
    _index_unscoped_exclude(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    corpora = {h.corpus for h in svc.search("alpha beta gamma", top_k=20)}
    assert corpora == {"chats"}


def test_naming_an_unscoped_excluded_corpus_still_searches_it(svc, make_doc, monkeypatch) -> None:
    _index_unscoped_exclude(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    hits = svc.search("alpha beta gamma", corpus_name="repos", top_k=20)
    assert {h.corpus for h in hits} == {"repos"}


def test_an_unscoped_excluded_corpus_is_not_hidden_from_lists(svc, monkeypatch) -> None:
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos", "cli"])
    assert svc.is_hidden_corpus("repos") is False
    assert svc.is_hidden_corpus("cli") is False


def test_build_artifacts_stay_excluded_alongside_it(svc, make_doc, monkeypatch) -> None:
    _index_unscoped_exclude(svc, make_doc, "live.md", "chats")
    _index_unscoped_exclude(svc, make_doc, "copy.md", "eval-chats-base")
    _index_unscoped_exclude(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    corpora = {h.corpus for h in svc.search("alpha beta gamma", top_k=20)}
    assert corpora == {"chats"}


def test_the_default_excludes_nothing_extra() -> None:
    assert config_mod.Settings().unscoped_exclude == []


def test_unscoped_exclude_loads_from_the_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"unscoped_exclude": ["repos", "cli"]}))
    assert config_mod.load_settings().unscoped_exclude == ["repos", "cli"]


def test_unscoped_exclude_is_an_editable_setting() -> None:
    assert "unscoped_exclude" in config_mod.EDITABLE_FIELDS


# ─── telling callers ────────────────────────────────────────────────────────
# A bot that omits `corpus` silently misses an unscoped-excluded corpus, so
# the tool surface has to say which corpora a bare query covers.

def test_in_default_search_reflects_both_exclusion_settings(svc, monkeypatch) -> None:
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])
    assert svc.in_default_search("chats") is True
    assert svc.in_default_search("repos") is False
    assert svc.in_default_search("eval-chats-base") is False


def test_list_corpora_marks_corpora_outside_the_default_search(svc, make_doc, monkeypatch) -> None:
    from vecgrep.mcp import server as mcp_server
    _index_unscoped_exclude(svc, make_doc, "live.md", "chats")
    _index_unscoped_exclude(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])
    monkeypatch.setattr(mcp_server, "_svc", lambda: svc)

    listed = {c["name"]: c["in_default_search"] for c in json.loads(mcp_server._run_list_corpora())}
    assert listed == {"chats": True, "repos": False}

# ── test_corpus_listing_hides_eval ──

class TestTheHiddenPredicate:
    def test_eval_prefixed_corpora_are_hidden(self, svc):
        assert svc.is_hidden_corpus("eval-docs-base") is True
        assert svc.is_hidden_corpus("eval-code-base") is True

    def test_real_corpora_are_not(self, svc):
        # "evaluation" is here on purpose: eval-* must not match a corpus that
        # merely starts with those letters.
        for name in ("docs", "code", "notes", "evaluation"):
            assert svc.is_hidden_corpus(name) is False, name

    def test_it_reads_the_same_setting_search_uses(self, svc):
        svc.settings.cross_corpus_exclude = ["scratch-*"]
        assert svc.is_hidden_corpus("scratch-1") is True
        assert svc.is_hidden_corpus("eval-docs-base") is False, (
            "one setting drives both, so it must not hard-code the eval- prefix"
        )

    def test_no_patterns_hides_nothing(self, svc):
        svc.settings.cross_corpus_exclude = []
        assert svc.is_hidden_corpus("eval-docs-base") is False


class TestTheServiceItselfStaysHonest:
    """The guard on the design: filtering belongs at the edges, not in here."""

    def test_list_corpora_still_returns_everything(self, svc, make_doc):
        svc.index(str(make_doc("a.md", "alpha beta gamma.")), "docs")
        svc.index(str(make_doc("b.md", "delta epsilon zeta.")), "eval-docs-base")
        names = {c.name for c in svc.list_corpora()}
        assert names == {"docs", "eval-docs-base"}, (
            "backup and the eval harness call this; hiding corpora here would "
            "silently drop them from a backup"
        )


def _corpus_corpus_listing_hides_eval(name):
    """A real Corpus, not a stand-in — the HTTP edge runs asdict() over it."""
    from vecgrep.backend.store.corpora import Corpus
    return Corpus(
        name=name,
        embed_backend="ollama",
        embed_model="bge-m3",
        dim=8,
        description=f"context for {name}",
        use_for=["routing"],
        avoid_for=["guessing"],
    )


class _FakeSvc:
    def __init__(self, names, patterns=("eval-*",)):
        self._names = list(names)
        self._patterns = list(patterns)

    def list_corpora(self):
        return [_corpus_corpus_listing_hides_eval(n) for n in self._names]

    def is_hidden_corpus(self, name):
        import fnmatch
        return any(fnmatch.fnmatch(name, p) for p in self._patterns)

    def filterable_fields(self, name):
        return {"filters": []}


NAMES = ["docs", "eval-docs-base", "code", "eval-code-base"]


class TestTheMcpTool:
    def test_eval_corpora_are_omitted_by_default(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(NAMES))
        got = [c["name"] for c in json.loads(mcp_server._run_list_corpora())]
        assert got == ["docs", "code"]

    def test_they_can_still_be_asked_for(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(NAMES))
        got = [c["name"] for c in
               json.loads(mcp_server._run_list_corpora(include_hidden=True))]
        assert got == NAMES, "the eval harness has to be able to see its own builds"

    def test_routing_context_is_exposed(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(["docs"]))
        item = json.loads(mcp_server._run_list_corpora())[0]
        assert item["description"] == "context for docs"
        assert item["use_for"] == ["routing"]
        assert item["avoid_for"] == ["guessing"]
        detail = json.loads(mcp_server._run_get_corpus("docs"))
        assert detail["description"] == "context for docs"
        assert detail["use_for"] == ["routing"]
        assert detail["avoid_for"] == ["guessing"]


class TestTheHttpRoute:
    def test_eval_corpora_are_omitted_by_default(self, monkeypatch):
        from vecgrep.backend.api import routes
        monkeypatch.setattr(routes, "_service", lambda: _FakeSvc(NAMES))
        assert [c.name for c in routes.list_corpora()] == ["docs", "code"]

    def test_they_can_still_be_asked_for(self, monkeypatch):
        from vecgrep.backend.api import routes
        monkeypatch.setattr(routes, "_service", lambda: _FakeSvc(NAMES))
        assert [c.name for c in routes.list_corpora(include_hidden=True)] == NAMES

# ── test_registry_race ──

def _corpus_registry_race(name: str, model: str = "nomic-embed-text", dim: int = 768) -> Corpus:
    return Corpus(name=name, embed_backend="ollama", embed_model=model, dim=dim)


def _barrier_upsert(path: str, corpus: str, ready, start) -> None:
    registry = CorpusRegistry(path=Path(path))
    ready.put(corpus)
    start.wait(5)
    registry.upsert(_corpus_registry_race(corpus))


def test_upsert_reloads_so_concurrent_writes_survive(tmp_path):
    """Writer A and writer B each hold their own registry on the same file.
    B adds a corpus; A (with stale in-memory state) then adds a different one.
    A's save must NOT erase B's corpus."""
    path = tmp_path / "corpora.json"
    reg_a = CorpusRegistry(path)
    reg_b = CorpusRegistry(path)

    reg_b.upsert(_corpus_registry_race("from_b"))  # B writes first
    reg_a.upsert(_corpus_registry_race("from_a"))  # A writes with stale in-memory (never saw from_b)

    # Fresh read from disk must contain BOTH.
    final = CorpusRegistry(path)
    names = {c.name for c in final.list()}
    assert names == {"from_a", "from_b"}, f"a writer clobbered the other: {names}"


def test_simultaneous_registry_writers_do_not_lose_an_update(tmp_path):
    path = tmp_path / "corpora.json"
    ctx = multiprocessing.get_context("spawn")
    ready, start = ctx.Queue(), ctx.Event()
    workers = [
        ctx.Process(target=_barrier_upsert, args=(str(path), name, ready, start))
        for name in ("alpha", "beta")
    ]
    for worker in workers:
        worker.start()
    assert {ready.get(timeout=5), ready.get(timeout=5)} == {"alpha", "beta"}
    start.set()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0

    assert {c.name for c in CorpusRegistry(path).list()} == {"alpha", "beta"}


def test_delete_reloads_and_preserves_others(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus_registry_race("keep"))
    reg.upsert(_corpus_registry_race("drop"))

    # A second handle deletes 'drop'; 'keep' must remain.
    reg2 = CorpusRegistry(path)
    reg2.delete("drop")

    final = CorpusRegistry(path)
    assert {c.name for c in final.list()} == {"keep"}


def test_mixed_model_corpora_coexist_in_registry(tmp_path):
    """Two corpora with different embed models persist side by side."""
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus_registry_race("legacy", model="nomic-embed-text", dim=768))
    reg.upsert(_corpus_registry_race("upgraded", model="bge-m3", dim=1024))

    final = CorpusRegistry(path)
    by_name = {c.name: c for c in final.list()}
    assert by_name["legacy"].embed_model == "nomic-embed-text"
    assert by_name["upgraded"].embed_model == "bge-m3"
    assert by_name["upgraded"].dim == 1024


def test_corrupt_registry_raises_instead_of_wiping(tmp_path):
    """A torn/corrupt corpora.json must NOT load as empty — because the next
    upsert would then save that empty registry over the good data, silently
    dropping every corpus. The recurring "corpora vanished" bug. Loading a
    corrupt file raises so the mutation aborts and disk data survives."""
    import pytest

    from vecgrep.backend.store.corpora import CorpusError

    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus_registry_race("keep_me"))
    assert path.exists()

    # Simulate a torn write: valid-JSON-prefix, truncated mid-object.
    path.write_text('{"keep_me": {"name": "keep_me", "embed_backend": "oll')

    # A fresh handle must refuse to load an empty registry over the file.
    with pytest.raises(CorpusError):
        CorpusRegistry(path)


def test_save_is_atomic_no_partial_file(tmp_path, monkeypatch):
    """_save writes via a temp file + os.replace, so an interrupted serialize
    never leaves a truncated corpora.json. If the replace step blows up, the
    original file is untouched (not half-written)."""
    import os
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus_registry_race("first"))
    good = path.read_text()

    # Force os.replace to fail during the next save; the original must survive.
    real_replace = os.replace
    def boom(src, dst):
        raise OSError("simulated crash during replace")
    monkeypatch.setattr(os, "replace", boom)
    try:
        reg.upsert(_corpus_registry_race("second"))
    except OSError:
        pass
    monkeypatch.setattr(os, "replace", real_replace)

    # File is still the intact pre-crash version (never truncated), and no
    # stray .tmp litter remains.
    assert path.read_text() == good
    leftovers = [q.name for q in tmp_path.iterdir() if q.name.endswith(".tmp")]
    assert not leftovers, f"temp files left behind: {leftovers}"
