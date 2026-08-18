"""Build side-by-side eval corpora, run gold queries against them, report.

In-process against the same VECGREP_HOME the daemon uses: the corpus locks,
the WAL'd embed cache and server-mode qdrant are all multi-process safe, and
running in-process is what lets a run test ranking constants (BM25 weight,
RRF k) and store options that the deployed daemon does not expose yet.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vecgrep.backend import service as service_mod
from vecgrep.backend.service import VecgrepService, _collection_for
from vecgrep.backend.store.corpora import Corpus

from .gold import GoldCase, load_gold
from .metrics import CaseResult, per_case_rows, score_case, summarize

EVAL_PREFIX = "eval-"


def eval_home(svc: VecgrepService) -> Path:
    p = svc.settings.home / "eval"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── build ────────────────────────────────────────────────────────────────────

def _common_root(paths: list[str]) -> Path:
    if not paths:
        raise ValueError("source corpus has no sources")
    common = os.path.commonpath([str(Path(p).parent) for p in paths])
    return Path(common)


@dataclass
class BuildReport:
    src: str
    dst: str
    chunker: str
    embed_model: str
    root: str
    docs: int
    chunks: int
    skipped: int
    seconds: float
    points: int
    qdrant_bytes: int
    bm25_bytes: int
    options: dict = field(default_factory=dict)


def build_side_by_side(
    svc: VecgrepService,
    src: str,
    dst: str,
    *,
    chunker: str | None = None,
    include: str | None = None,
    root: str | None = None,
    decay_half_life_days: float | None | str = "inherit",
    quantization: str | None = None,
    datatype: str = "float32",
    force: bool = False,
) -> BuildReport:
    """Index the LIVE corpus `src`'s sources into a fresh corpus `dst`.

    `dst` must start with `eval-` — that prefix is the whole safety story: no
    bot's recall is scoped to an eval corpus, and `python -m vecgrep.eval
    drop` only ever deletes corpora carrying it.

    Pins `dst` to `src`'s embedding backend/model so the embed cache is warm
    (same chunker => identical chunk texts => zero re-embedding). A different
    chunker re-embeds only the chunks whose text differs.
    """
    if not dst.startswith(EVAL_PREFIX):
        raise ValueError(f"eval corpus names must start with {EVAL_PREFIX!r}: {dst}")
    live = svc.registry.get(src)
    chunker = chunker or live.chunker
    sources = list(live.source_hashes) or list(live.sources)
    src_root = Path(root) if root else _common_root(sources)
    if not src_root.exists():
        raise FileNotFoundError(f"source root {src_root} for {src} does not exist")

    if svc.registry.has(dst):
        if force:
            svc.delete_corpus(dst)
        else:
            raise ValueError(f"{dst} already exists (pass --force to rebuild)")

    # Pre-create with the live corpus's pinned model so _index_locked resolves
    # the same backend identity the cache was filled under.
    now = time.time()
    c = Corpus(
        name=dst,
        embed_backend=live.embed_backend,
        embed_model=live.embed_model,
        dim=live.dim,
        chunker=chunker,
        created_at=now,
        updated_at=now,
        decay_half_life_days=(
            live.decay_half_life_days if decay_half_life_days == "inherit"
            else decay_half_life_days
        ),
        rank_weight=getattr(live, "rank_weight", 1.0),
    )
    svc.registry.upsert(c)
    collection = _collection_for(dst)
    svc.store.ensure_collection(collection, live.dim, datatype=datatype, quantization=quantization)

    t0 = time.time()
    docs, chunks, skipped = svc.index(str(src_root), dst, chunker_name=chunker, include=include)
    secs = time.time() - t0
    q_bytes, b_bytes = corpus_bytes(svc, dst)
    return BuildReport(
        src=src, dst=dst, chunker=chunker, embed_model=live.embed_model,
        root=str(src_root), docs=docs, chunks=chunks, skipped=skipped,
        seconds=round(secs, 1), points=svc.store.count(collection),
        qdrant_bytes=q_bytes, bm25_bytes=b_bytes,
        options={"quantization": quantization, "datatype": datatype, "include": include},
    )


def _du(path: Path) -> int:
    """Allocated bytes (st_blocks), not apparent size: qdrant pre-sizes its
    mmap files, so a 20-point collection has ~136 MB of apparent size and a
    few hundred KB actually on disk. `du` semantics are the honest ones."""
    if not path.exists():
        return 0

    def alloc(p: Path) -> int:
        st = p.stat()
        blocks = getattr(st, "st_blocks", None)
        return blocks * 512 if blocks is not None else st.st_size

    if path.is_file():
        return alloc(path)
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += alloc(p)
        except OSError:
            pass
    return total


def corpus_bytes(svc: VecgrepService, corpus: str) -> tuple[int, int]:
    """(qdrant bytes, bm25 bytes) for one corpus, from the filesystem.

    Server-mode qdrant keeps its storage under `qdrant-server/collections/`
    beside VECGREP_HOME; embedded mode under `qdrant/collection/`. Sizes settle
    only after the optimizer has run, so a build measures after a short wait
    (see `settle_collection`)."""
    home = svc.settings.home
    coll = _collection_for(corpus)
    candidates = [
        home / "qdrant-server" / "collections" / coll,
        home / "qdrant-server" / "storage" / "collections" / coll,
        home / "qdrant" / "collection" / coll,
    ]
    q = next((_du(p) for p in candidates if p.exists()), 0)
    b = _du(home / "bm25" / f"{corpus}.pkl")
    return q, b


def settle_collection(svc: VecgrepService, corpus: str, timeout_s: float = 600.0) -> str:
    """Wait for qdrant to finish optimizing (status green) so on-disk bytes are
    the steady-state number, not a half-merged segment set."""
    coll = _collection_for(corpus)
    t0 = time.time()
    status = "unknown"
    while time.time() - t0 < timeout_s:
        try:
            info = svc.store.client.get_collection(coll)
            status = str(getattr(info, "status", "unknown")).split(".")[-1].lower()
        except Exception:
            status = "error"
        if status == "green":
            break
        time.sleep(5)
    return status


def drop_eval_corpora(svc: VecgrepService, names: list[str] | None = None) -> list[str]:
    dropped = []
    for c in svc.registry.list():
        if not c.name.startswith(EVAL_PREFIX):
            continue
        if names and c.name not in names:
            continue
        svc.delete_corpus(c.name)
        dropped.append(c.name)
    return dropped


# ── run ──────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """One row of the table. `corpora` maps logical gold corpus -> actual
    corpus name (default: identity, i.e. the live corpus — only useful for
    reading, never for a build). Ranking knobs override module constants for
    the duration of the run."""
    name: str
    corpora: dict[str, str] = field(default_factory=dict)
    mode: str = "hybrid"
    rerank: bool = False
    rerank_model: str | None = None
    top_k: int = 10
    bm25_weight: float | None = None
    rrf_k: int | None = None
    cosine_floor_margin: float | None = None
    decay_half_life_days: float | None | str = "keep"    # per actual corpus, restored after
    notes: str = ""


class _Patched:
    """Temporarily override ranking constants on the service module."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.saved: dict[str, object] = {}

    def __enter__(self):
        pairs = {
            "BM25_WEIGHT": self.cfg.bm25_weight,
            "RRF_K": self.cfg.rrf_k,
            "COSINE_FLOOR_MARGIN": self.cfg.cosine_floor_margin,
        }
        for k, v in pairs.items():
            if v is not None:
                self.saved[k] = getattr(service_mod, k)
                setattr(service_mod, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(service_mod, k, v)


def _hits_as_dicts(results) -> list[dict]:
    return [{"source_id": r.source_id, "similarity_pct": r.similarity_pct,
             "corpus": r.corpus} for r in results]


def run_config(
    svc: VecgrepService,
    cfg: RunConfig,
    cases: list[GoldCase],
    *,
    warmup: bool = True,
    repeat: int = 1,
) -> dict:
    """Run every case under `cfg`. Returns {config, summary, cases, storage}."""
    results: list[CaseResult] = []
    actual_corpora = {c.corpus: cfg.corpora.get(c.corpus, c.corpus) for c in cases}
    missing = [a for a in set(actual_corpora.values()) if not svc.registry.has(a)]
    if missing:
        raise ValueError(f"config {cfg.name}: corpora not found: {sorted(missing)}")

    saved_decay: dict[str, float | None] = {}
    if cfg.decay_half_life_days != "keep":
        for a in set(actual_corpora.values()):
            saved_decay[a] = svc.registry.get(a).decay_half_life_days
            svc.set_decay(a, cfg.decay_half_life_days)  # type: ignore[arg-type]
    try:
        with _Patched(cfg):
            if warmup and cases:
                # first query pays model/connection warmup; keep it out of p50/p95
                c0 = cases[0]
                try:
                    svc.search(c0.query, actual_corpora[c0.corpus], top_k=cfg.top_k,
                               mode=cfg.mode, rerank=cfg.rerank, rerank_model=cfg.rerank_model)
                except Exception:
                    pass
            for case in cases:
                actual = actual_corpora[case.corpus]
                lats = []
                hits: list[dict] = []
                for _ in range(max(1, repeat)):
                    t0 = time.perf_counter()
                    res = svc.search(case.query, actual, top_k=cfg.top_k, mode=cfg.mode,
                                     rerank=cfg.rerank, rerank_model=cfg.rerank_model)
                    lats.append(time.perf_counter() - t0)
                    hits = _hits_as_dicts(res)
                results.append(score_case(case, hits, statistics.median(lats)))
    finally:
        for a, d in saved_decay.items():
            svc.set_decay(a, d)

    storage = {}
    for logical, actual in sorted(set(actual_corpora.items())):
        q, b = corpus_bytes(svc, actual)
        storage[actual] = {
            "logical": logical,
            "points": svc.store.count(_collection_for(actual)),
            "qdrant_mb": round(q / 1e6, 1),
            "bm25_mb": round(b / 1e6, 1),
        }
    return {
        "config": asdict(cfg),
        "summary": summarize(results),
        "storage": storage,
        "cases": per_case_rows(results),
        "ts": time.time(),
    }


def load_configs(path: str | Path) -> list[RunConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["configs"] if isinstance(data, dict) else data
    return [RunConfig(**c) for c in raw]


# ── report ───────────────────────────────────────────────────────────────────

_COLS = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "p@3", "spread_1v3",
         "neg_fp_rate", "neg_top_pct_mean", "lat_p50_ms", "lat_p95_ms")


def table(runs: list[dict]) -> str:
    """Markdown table: one row per config, storage columns summed over the
    corpora that config touched."""
    head = ["config", *_COLS, "points", "qdrant_MB", "bm25_MB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        s = r["summary"]
        st = r.get("storage", {})
        pts = sum(v.get("points", 0) for v in st.values())
        qmb = round(sum(v.get("qdrant_mb", 0) for v in st.values()), 1)
        bmb = round(sum(v.get("bm25_mb", 0) for v in st.values()), 1)
        cells = [r["config"]["name"]] + [
            ("" if s.get(c) is None else str(s.get(c))) for c in _COLS
        ] + [str(pts), str(qmb), str(bmb)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_runs(paths: list[str | Path]) -> list[dict]:
    runs = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.extend(data if isinstance(data, list) else [data])
    return runs
