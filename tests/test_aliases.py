"""Phase 6 — alias / entity expansion.

One entity, many surface forms (a nickname ⇄ a handle ⇄ a name in another
language). The map is USER-SUPPLIED and lives outside the repo (it's
personal data) — default `$VECGREP_HOME/aliases.json`, overridable via
`VECGREP_ALIASES_FILE`. vecgrep ships only a generic example. Query mentions
one form → the BM25/vector query expands with the entity's other forms.
Config-driven, not learned; no map = exact no-op.

Map format (flat): {"alice": ["ali", "小艾"], ...} — key is the canonical
form, values are alternates; matching is symmetric across all forms.
"""
from __future__ import annotations

import json
from pathlib import Path

from vecgrep.backend.aliases import expand_query, load_alias_map

MAP = {"alice": ["ali", "小艾"], "relay-service": ["the relay", "rs-1"]}


# ── loading ──────────────────────────────────────────────────────────────────

def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_alias_map(tmp_path / "nope.json") == {}


def test_load_valid_map(tmp_path: Path) -> None:
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(MAP), encoding="utf-8")
    m = load_alias_map(p)
    assert m["alice"] == ["ali", "小艾"]


def test_load_corrupt_file_is_empty_not_crash(tmp_path: Path) -> None:
    p = tmp_path / "aliases.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_alias_map(p) == {}


def test_load_rejects_non_string_forms(tmp_path: Path) -> None:
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps({"a": ["ok", 7], "b": "not-a-list"}), encoding="utf-8")
    m = load_alias_map(p)
    assert m == {"a": ["ok"]}  # bad entries dropped, good ones kept


# ── expansion ────────────────────────────────────────────────────────────────

def test_expand_canonical_form_adds_alternates() -> None:
    q, matched = expand_query("what did alice say about the cert", MAP)
    assert "ali" in q.split() and "小艾" in q
    assert matched == ["alice"]
    assert q.startswith("what did alice say")  # original query preserved


def test_expand_alternate_form_adds_whole_set() -> None:
    q, matched = expand_query("小艾 root cause", MAP)
    assert "alice" in q.split() and "ali" in q.split()
    assert matched == ["alice"]


def test_expand_ascii_needs_word_boundary() -> None:
    # 'ali' must not fire inside 'alignment' — ASCII forms match whole words.
    q, matched = expand_query("alignment of the planets", MAP)
    assert matched == []
    assert q == "alignment of the planets"


def test_expand_cjk_matches_by_containment() -> None:
    # CJK has no word boundaries — containment is the right test.
    q, matched = expand_query("今天小艾说了什么", MAP)
    assert matched == ["alice"]


def test_expand_no_map_is_identity() -> None:
    q, matched = expand_query("anything at all", {})
    assert q == "anything at all" and matched == []


def test_expand_does_not_duplicate_forms_already_present() -> None:
    q, _ = expand_query("alice and ali argued", MAP)
    # Both present already — only the missing form gets appended.
    assert q.split().count("alice") == 1
    assert q.split().count("ali") == 1
    assert "小艾" in q


def test_expand_multiple_entities() -> None:
    q, matched = expand_query("alice restarted rs-1", MAP)
    assert set(matched) == {"alice", "relay-service"}
    assert "relay-service" in q and "小艾" in q


# ── service wiring: the review's golden alias case ───────────────────────────

def test_alias_query_surfaces_evidence_under_other_form(svc, tmp_path, monkeypatch) -> None:
    """Query uses 小艾; the substantive evidence (the postmortem root cause)
    was written under 'ali'. Without the map the root-cause chunk is
    invisible to the entity term; with it, it surfaces."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(
        json.dumps({"alice": ["ali", "小艾"]}), encoding="utf-8"
    )
    monkeypatch.setenv("VECGREP_ALIASES_FILE", str(alias_file))

    hits = svc.search("小艾 root cause TLS certificate", "evalchat", top_k=8)
    # Evidence AUTHORED under 'ali' must surface for a query that only named
    # the 小艾 form. (The discriminating hit is ali's 09:05 message, which
    # shares no term with the unexpanded query — only the expanded 'ali'
    # token can reach it.)
    assert any("**ali**" in h.chunk for h in hits), (
        "no ali-authored evidence surfaced for a 小艾 query"
    )


def test_alias_expansion_can_be_disabled(svc, tmp_path, monkeypatch) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"alice": ["ali", "小艾"]}), encoding="utf-8")
    monkeypatch.setenv("VECGREP_ALIASES_FILE", str(alias_file))

    with_exp = svc.search("小艾 certificate", "evalchat", top_k=8)
    without = svc.search("小艾 certificate", "evalchat", top_k=8,
                         expand_aliases=False)
    # Expansion ON must surface at least as much ali-authored evidence.
    def ali_hits(hits):
        return sum(1 for h in hits if "**ali**" in h.chunk)
    assert ali_hits(with_exp) >= ali_hits(without)
    assert ali_hits(with_exp) > 0
