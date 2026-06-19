"""Write-tool phase 1 — metadata schema on the READ path (zero write risk).

The write tool's docs carry a metadata schema (status / version / origin / tier /
source_kind / supersedes…). Phase 1 only touches retrieval: search defaults to
status:active so a superseded version never surfaces as current truth, but the
superseded ones stay retrievable on request (audit trail). Legacy chunks with no
status field must keep behaving as active (back-compat).

These exercise the service search() filtering directly with the stub backend.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.service import _passes_filters, SearchResult


def _result(text="t", metadata=None, corpus="c", source_id="s"):
    r = SearchResult.__new__(SearchResult)
    r.corpus = corpus
    r.source_id = source_id
    r.text = text
    r.metadata = metadata or {}
    r.score = 1.0
    r.matched_by = ["vector"]
    r.explain = {}
    return r


# --- default status:active filtering (the one real behavior change) ---

class TestDefaultActiveFilter:
    def test_active_chunk_passes_default(self):
        r = _result(metadata={"status": "active"})
        assert _passes_filters(r, ["meta.status=active"]) is True

    def test_superseded_chunk_blocked_by_active_filter(self):
        r = _result(metadata={"status": "superseded"})
        assert _passes_filters(r, ["meta.status=active"]) is False

    def test_legacy_chunk_without_status_treated_as_active(self):
        # Back-compat: a chunk indexed before the schema has no status key.
        # It must NOT vanish under a default status:active filter.
        r = _result(metadata={})
        assert _passes_filters(r, ["meta.status=active"], default_active=True) is True

    def test_legacy_chunk_blocked_only_under_strict_match(self):
        # Without the back-compat default, a literal meta.status=active match
        # excludes a chunk that has no status (the pre-existing strict behavior).
        r = _result(metadata={})
        assert _passes_filters(r, ["meta.status=active"]) is False


# --- source_kind / tier filters (already meta.* keys; pin they work) ---

class TestSchemaFieldFilters:
    def test_filter_by_source_kind(self):
        decision = _result(metadata={"source_kind": "decision"})
        journal = _result(metadata={"source_kind": "journal"})
        assert _passes_filters(decision, ["meta.source_kind=decision"]) is True
        assert _passes_filters(journal, ["meta.source_kind=decision"]) is False

    def test_filter_by_tier(self):
        protected = _result(metadata={"tier": "protected"})
        normal = _result(metadata={"tier": "normal"})
        assert _passes_filters(protected, ["meta.tier=protected"]) is True
        assert _passes_filters(normal, ["meta.tier=protected"]) is False

    def test_filters_and_together(self):
        r = _result(metadata={"status": "active", "tier": "protected"})
        assert _passes_filters(r, ["meta.status=active", "meta.tier=protected"]) is True
        assert _passes_filters(r, ["meta.status=active", "meta.tier=normal"]) is False
