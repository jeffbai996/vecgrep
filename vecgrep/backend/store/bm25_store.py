"""BM25 keyword index, per corpus.

Persisted as a pickle next to the Qdrant store. Lives in lockstep with the
vector index — same chunk IDs, same lifecycle (create/upsert/delete-by-source).
Lowercased word-token split, no stemming. Predictable across languages
without dragging in nltk.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

# Match runs of letters/digits, treating underscore and CamelCase as
# token boundaries so identifiers like `sharpe_ratio` and `getUserName`
# are searchable as their constituent words. Pure prose is unaffected.
_TOKEN = re.compile(r"[A-Za-z]+|\d+", re.UNICODE)
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Minimum query-term coverage required for a doc to be a BM25 candidate.
# Without this, a 2-token query like "glucose monitoring" against a repo
# corpus where only "monitoring" appears (e.g. an architecture diagram in
# some unrelated README) returns that README at top rank — its single-term
# IDF score survives the BM25 sort, gets fused via RRF with the BM25
# weight, and outranks genuine vector hits. We require that short queries
# match every token, longer queries match at least half. Override per-call
# via env var; the safety hatch fully disables the filter.
BM25_SHORT_QUERY_THRESHOLD = 3
BM25_SHORT_QUERY_COVERAGE = 1.0
BM25_LONG_QUERY_COVERAGE = 0.5

# Coverage mode controls how partial-coverage docs are handled.
#  - "filter" (default): hard rejection below the threshold. Safe but loses
#    legitimate signal when a topic is split across chunks (e.g. "glucose"
#    appears in one chunk, "monitoring" in the neighbour, neither alone).
#  - "penalty": keep partial-coverage docs but multiply their BM25 score by
#    (K/N) ** PENALTY_EXPONENT, where K is the number of distinct query
#    tokens the doc matches and N is the total. Zero-overlap docs are still
#    excluded — no point keeping pure noise.
# The exponent defaults to 2.0: linear (exp=1) is too gentle — a
# single-token match at 0.5 coverage only loses half its score, which often
# isn't enough to demote a high-IDF partial match below a genuine
# full-coverage hit. Quadratic gives 0.25 at 0.5 coverage, which empirically
# is firm enough to flip the order on the cases that motivated this fix.
BM25_COVERAGE_MODE_DEFAULT = "filter"
BM25_COVERAGE_PENALTY_EXPONENT = 2.0


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for chunk in _TOKEN.findall(text):
        for piece in _CAMEL_SPLIT.split(chunk):
            if piece:
                out.append(piece.lower())
    return out


def _required_coverage(n_query_tokens: int) -> int:
    """Number of distinct query tokens a doc must contain to be a candidate.

    Env vars are read at call time so tests can monkeypatch them.
    """
    if n_query_tokens <= 0:
        return 0
    short_frac = float(
        os.environ.get("VECGREP_BM25_SHORT_QUERY_COVERAGE", BM25_SHORT_QUERY_COVERAGE)
    )
    long_frac = float(
        os.environ.get("VECGREP_BM25_LONG_QUERY_COVERAGE", BM25_LONG_QUERY_COVERAGE)
    )
    frac = short_frac if n_query_tokens <= BM25_SHORT_QUERY_THRESHOLD else long_frac
    # ceil so 50% of 5 -> 3, not 2; one-token queries always require 1.
    needed = math.ceil(n_query_tokens * frac)
    return max(1, min(needed, n_query_tokens))


def _meets_coverage(q_tokens: list[str], doc_tokens: list[str]) -> bool:
    """True if `doc_tokens` covers enough of the query's distinct tokens.

    Reads env vars on every call (cheap, and lets the safety-hatch
    `VECGREP_BM25_DISABLE_COVERAGE_FILTER` flip mid-process).
    """
    if os.environ.get("VECGREP_BM25_DISABLE_COVERAGE_FILTER") == "1":
        return True
    q_set = set(q_tokens)
    if not q_set:
        return True
    needed = _required_coverage(len(q_set))
    doc_set = set(doc_tokens)
    return sum(1 for t in q_set if t in doc_set) >= needed


def _coverage_factor(q_tokens: list[str], doc_tokens: list[str]) -> float | None:
    """Return the score multiplier this doc's coverage earns, or None to drop.

    Returns:
      None  -> doc is excluded entirely (zero overlap, or filter mode below
              threshold).
      1.0   -> no penalty (filter mode passing, or penalty mode at full
              coverage).
      0..1  -> penalty mode at partial coverage; multiply BM25 score by this.

    Env vars are read at call time so tests can monkeypatch and the safety
    hatch can flip mid-process.
    """
    # Safety hatch overrides everything — no coverage logic at all.
    if os.environ.get("VECGREP_BM25_DISABLE_COVERAGE_FILTER") == "1":
        return 1.0
    q_set = set(q_tokens)
    if not q_set:
        return 1.0
    doc_set = set(doc_tokens)
    matched = sum(1 for t in q_set if t in doc_set)
    # Zero overlap is dropped in every mode — keeping pure-noise docs adds
    # nothing and pollutes RRF fusion downstream.
    if matched == 0:
        return None
    n = len(q_set)
    mode = os.environ.get("VECGREP_BM25_COVERAGE_MODE", BM25_COVERAGE_MODE_DEFAULT).lower()
    if mode == "penalty":
        if matched == n:
            return 1.0
        exp = float(
            os.environ.get(
                "VECGREP_BM25_COVERAGE_PENALTY_EXPONENT",
                BM25_COVERAGE_PENALTY_EXPONENT,
            )
        )
        return (matched / n) ** exp
    # Default: filter mode. Below threshold -> drop; otherwise no penalty.
    needed = _required_coverage(n)
    if matched < needed:
        return None
    return 1.0


@dataclass
class _CorpusIndex:
    ids: list[str] = field(default_factory=list)
    docs: list[list[str]] = field(default_factory=list)
    payloads: list[dict] = field(default_factory=list)
    # source_id -> list of array indices, so delete-by-source is O(n) once.
    by_source: dict[str, list[int]] = field(default_factory=dict)


class BM25Store:
    def __init__(self, root: Path | None) -> None:
        # root=None -> ephemeral (in-memory only).
        self.root = root
        self._cache: dict[str, _CorpusIndex] = {}
        self._bm25_instances: dict[str, BM25Okapi] = {}
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, corpus: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{corpus}.pkl"

    def _load(self, corpus: str) -> _CorpusIndex:
        if corpus in self._cache:
            return self._cache[corpus]
        p = self._path(corpus)
        if p and p.exists():
            try:
                idx = pickle.loads(p.read_bytes())
            except Exception:
                idx = _CorpusIndex()
        else:
            idx = _CorpusIndex()
        self._cache[corpus] = idx
        return idx

    def _persist(self, corpus: str) -> None:
        p = self._path(corpus)
        if p is None:
            return
        idx = self._cache[corpus]
        p.write_bytes(pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL))

    def upsert(self, corpus: str, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        if not ids:
            return
        idx = self._load(corpus)
        for i, (cid, text, payload) in enumerate(zip(ids, texts, payloads)):
            arr_pos = len(idx.ids)
            idx.ids.append(cid)
            idx.docs.append(tokenize(text))
            idx.payloads.append(payload)
            sid = payload.get("source_id", "")
            idx.by_source.setdefault(sid, []).append(arr_pos)
        self._bm25_instances.pop(corpus, None)
        self._persist(corpus)

    def delete_by_source(self, corpus: str, source_id: str) -> None:
        idx = self._load(corpus)
        positions = set(idx.by_source.pop(source_id, []))
        if not positions:
            return
        # Rebuild parallel arrays without those positions, then re-derive
        # the by_source map. Simpler than splice math; BM25 needs full
        # rebuild on every change anyway.
        new = _CorpusIndex()
        for i, cid in enumerate(idx.ids):
            if i in positions:
                continue
            new_pos = len(new.ids)
            new.ids.append(cid)
            new.docs.append(idx.docs[i])
            new.payloads.append(idx.payloads[i])
            sid = idx.payloads[i].get("source_id", "")
            new.by_source.setdefault(sid, []).append(new_pos)
        self._cache[corpus] = new
        self._bm25_instances.pop(corpus, None)
        self._persist(corpus)

    def drop(self, corpus: str) -> None:
        self._cache.pop(corpus, None)
        self._bm25_instances.pop(corpus, None)
        p = self._path(corpus)
        if p and p.exists():
            p.unlink()

    def search(self, corpus: str, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        idx = self._load(corpus)
        if not idx.docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        
        bm25 = self._bm25_instances.get(corpus)
        if bm25 is None:
            bm25 = BM25Okapi(idx.docs)
            self._bm25_instances[corpus] = bm25

        scores = bm25.get_scores(q_tokens)
        # BM25Okapi can score 0 for valid matches when IDF is degenerate
        # (single-doc corpus, or every doc contains the term). Fall back to
        # token-overlap counting in that case so the retriever still surfaces
        # something rather than nothing.
        #
        # Coverage handling lives in `_coverage_factor`: factor=None drops the
        # doc, factor=1.0 keeps it at full BM25 score, factor<1 demotes it
        # (penalty mode). We multiply factor into the sort key so partial-
        # coverage docs survive but rank below full-coverage hits.
        candidates: list[tuple[float, int]] = []
        for i, s in enumerate(scores):
            if s <= 0:
                continue
            factor = _coverage_factor(q_tokens, idx.docs[i])
            if factor is None:
                continue
            candidates.append((float(s) * factor, i))
        ranked = sorted(candidates, reverse=True)[:top_k]
        if not ranked:
            q_set = set(q_tokens)
            fallback: list[tuple[float, int]] = []
            for i in range(len(idx.docs)):
                factor = _coverage_factor(q_tokens, idx.docs[i])
                if factor is None:
                    continue
                # Token-occurrence count (not distinct) preserves the prior
                # fallback ordering behavior; the coverage factor still
                # demotes partial matches in penalty mode.
                o = sum(1 for t in idx.docs[i] if t in q_set)
                if o <= 0:
                    continue
                fallback.append((float(o) * factor, i))
            ranked = sorted(fallback, reverse=True)[:top_k]
        return [(idx.ids[i], float(s), idx.payloads[i]) for s, i in ranked]
