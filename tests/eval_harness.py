"""Memory-retrieval eval harness (Phase 0 of the v1 release).

Synthetic Discord-style transcript fixtures + golden benchmark queries +
assembly-quality metrics. Every retrieval improvement in the v1 release is
measured against this harness, with the v0.7.0 numbers captured in
tests/fixtures/baselines/v0_7_0.json so deltas are visible.

Hermetic by construction: StubEmbed vectors + BM25, embedded qdrant, no live
embedder. That scopes what it can measure — RESULT ASSEMBLY quality (dedup,
filters, chronology, budget), not embedding quality. Semantic-vague queries
are included as probes, but their absolute numbers mean little under a stub
embedder; their value is regression tracking, not benchmarking.

Run standalone to (re)generate a baseline:

    python -m tests.eval_harness > tests/fixtures/baselines/v0_7_0.json
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from vecgrep import __version__
from vecgrep.backend.service import SearchResult, VecgrepService

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcripts"
EVAL_CORPUS = "evalchat"
DEFAULT_TOP_K = 10

# Transcript speaker headers (`**name** · HH:MM`). The sentence-window chunker
# joins lines with spaces, so message bodies can't be extracted line-wise —
# instead we split chunk text at speaker headers and compare the message
# bodies between them. Repeated verbatim messages (bot alert spam, quoted
# replies) are the transcript-native duplicate class span overlap can't see.
_SPEAKER_HEADER = re.compile(r"\*\*[^*\n]+\*\*\s*·\s*\d{1,2}:\d{2}")
_MIN_MESSAGE_CHARS = 20


# ── golden queries ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    # "graded": expected/forbidden sources are meaningful pass/fail signals.
    # "probe": tracked for regressions; absolute numbers not meaningful yet
    #          (semantic-vague under a stub embedder, pre-Phase-6 alias case).
    kind: str = "graded"
    expected_sources: tuple[str, ...] = ()
    forbidden_sources: tuple[str, ...] = ()
    top_k: int = DEFAULT_TOP_K
    search_kwargs: dict = field(default_factory=dict)


GOLDEN_QUERIES: list[GoldenQuery] = [
    GoldenQuery(
        id="incident_today",
        query="why did relay-service fail today?",
        expected_sources=("channel-a/2026-01-15.md", "channel-b/2026-01-15.md"),
        # The week-old outage uses the same vocabulary; an unfiltered query
        # drags it in (the review's over-eagerness complaint). Phase 3's date
        # filters are what eventually excludes it.
        forbidden_sources=("channel-a/2026-01-08.md",),
    ),
    GoldenQuery(
        id="argument",
        query="what was alice arguing with bob about this morning?",
        expected_sources=("channel-a/2026-01-15.md",),
        forbidden_sources=("channel-a/2026-01-08.md",),
    ),
    GoldenQuery(
        id="greatest_hits",
        query="show the greatest hits of charlie this morning",
        expected_sources=("channel-c/2026-01-15.md",),
    ),
    GoldenQuery(
        id="exact_keyword",
        query="ERR-0451",
        # All three files legitimately contain the error code — keyword
        # precision means finding all of them, not date-filtering.
        expected_sources=(
            "channel-a/2026-01-15.md",
            "channel-a/2026-01-08.md",
            "channel-b/2026-01-15.md",
        ),
    ),
    GoldenQuery(
        id="semantic_vague",
        query="that thing where the bot went quiet",
        kind="probe",
        expected_sources=("channel-c/2026-01-15.md",),
    ),
    GoldenQuery(
        id="alias_cn",
        query="小艾 certificate renewal root cause",
        kind="probe",
        # ali's postmortem holds the substance; only alias expansion
        # (Phase 6) can connect the 小艾 form to chunks authored as "ali".
        expected_sources=("channel-b/2026-01-15.md",),
    ),
    GoldenQuery(
        id="dup_stress",
        query="relay-service ERR-0451 down",
        kind="probe",
        expected_sources=("channel-a/2026-01-15.md",),
    ),
]


# ── corpus build ─────────────────────────────────────────────────────────────

def build_eval_corpus(svc: VecgrepService, corpus: str = EVAL_CORPUS) -> dict:
    """Index every fixture transcript into `corpus`. Returns {files, chunks}."""
    files = sorted(FIXTURES_DIR.rglob("*.md"))
    chunks = 0
    for f in files:
        _docs, n_chunks, _skipped = svc.index(str(f), corpus)
        chunks += n_chunks
    return {"files": len(files), "chunks": chunks}


# ── metrics ──────────────────────────────────────────────────────────────────

def _message_bodies(text: str) -> set[str]:
    """Distinct message bodies in a chunk: segments between speaker headers,
    normalized (quote markers + whitespace collapsed), short fragments dropped
    (a partial message at a chunk edge shouldn't count as 'the same message')."""
    out: set[str] = set()
    for seg in _SPEAKER_HEADER.split(text):
        body = " ".join(seg.replace(">", " ").split())
        if len(body) >= _MIN_MESSAGE_CHARS:
            out.add(body)
    return out


def _pair_is_redundant(a: SearchResult, b: SearchResult) -> bool:
    """Would an assistant reading both results learn anything from the second?

    Redundant when both come from the same source file AND any of:
      - char spans overlap >= 50% of the shorter span (chunker stride dups),
      - they share a verbatim message body (bot alert spam / quoted replies —
        the class span-overlap can't see),
      - normalized text similarity >= 0.85 (same content, drifted spans).
    """
    if a.corpus != b.corpus or a.source_id != b.source_id:
        return False
    lo = max(a.chunk_start, b.chunk_start)
    hi = min(a.chunk_end, b.chunk_end)
    shorter = min(a.chunk_end - a.chunk_start, b.chunk_end - b.chunk_start)
    if shorter > 0 and max(0, hi - lo) / shorter >= 0.5:
        return True
    if _message_bodies(a.chunk) & _message_bodies(b.chunk):
        return True
    ratio = difflib.SequenceMatcher(
        None, " ".join(a.chunk.split()), " ".join(b.chunk.split())
    ).ratio()
    return ratio >= 0.85


def redundancy(results: list[SearchResult]) -> float:
    """Fraction of result pairs that are near-duplicates (0.0 = all distinct)."""
    n = len(results)
    if n < 2:
        return 0.0
    pairs = redundant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if _pair_is_redundant(results[i], results[j]):
                redundant += 1
    return redundant / pairs


def _matches_source(result: SearchResult, suffix: str) -> bool:
    return result.source_id.replace("\\", "/").endswith(suffix)


def query_metrics(q: GoldenQuery, results: list[SearchResult]) -> dict:
    hit_expected = [
        s for s in q.expected_sources if any(_matches_source(r, s) for r in results)
    ]
    forbidden_hits = sum(
        1 for r in results if any(_matches_source(r, s) for s in q.forbidden_sources)
    )
    return {
        "n_results": len(results),
        "distinct_sources": len({(r.corpus, r.source_id) for r in results}),
        "redundancy": round(redundancy(results), 4),
        "expected_recall": (
            round(len(hit_expected) / len(q.expected_sources), 4)
            if q.expected_sources
            else 1.0
        ),
        "forbidden_hits": forbidden_hits,
    }


def run_eval(svc: VecgrepService, corpus: str = EVAL_CORPUS) -> dict:
    """Run every golden query; return {version, queries: {id: metrics}}."""
    out: dict = {"version": __version__, "corpus": corpus, "queries": {}}
    for q in GOLDEN_QUERIES:
        results = svc.search(q.query, corpus, top_k=q.top_k, **q.search_kwargs)
        out["queries"][q.id] = query_metrics(q, results)
    return out


# ── standalone baseline generation ───────────────────────────────────────────

def _standalone_service(tmp_home: str) -> VecgrepService:
    """A hermetic service outside pytest: temp VECGREP_HOME + stub embedder."""
    import os

    from tests.conftest import StubEmbed
    from vecgrep.backend import config as cfg_mod

    os.environ["VECGREP_HOME"] = tmp_home
    cfg_mod._settings = None
    svc = VecgrepService(ephemeral=False)
    svc._backend_cache["auto"] = StubEmbed()
    return svc


def main() -> int:
    import json
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vecgrep-eval-") as td:
        svc = _standalone_service(td)
        build_eval_corpus(svc)
        report = run_eval(svc)
        try:
            svc.store.client.close()
        except Exception:
            pass
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
