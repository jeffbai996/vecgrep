"""CLI: python -m vecgrep.eval {build,run,report,drop,list}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from vecgrep.backend.service import VecgrepService

from . import harness
from .gold import load_gold


def _svc() -> VecgrepService:
    return VecgrepService()


def cmd_build(a) -> int:
    svc = _svc()
    rep = harness.build_side_by_side(
        svc, a.src, a.dst, chunker=a.chunker, include=a.include, root=a.root,
        quantization=a.quantization, datatype=a.datatype, force=a.force,
    )
    if a.settle:
        status = harness.settle_collection(svc, a.dst)
        q, b = harness.corpus_bytes(svc, a.dst)
        rep.qdrant_bytes, rep.bm25_bytes = q, b
        rep.options["settle_status"] = status
    d = asdict(rep)
    d["qdrant_mb"] = round(rep.qdrant_bytes / 1e6, 1)
    d["bm25_mb"] = round(rep.bm25_bytes / 1e6, 1)
    print(json.dumps(d, indent=2))
    out = harness.eval_home(svc) / "builds.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**d, "ts": time.time()}) + "\n")
    return 0


def cmd_run(a) -> int:
    svc = _svc()
    cases = load_gold(a.gold, default_corpus=a.default_corpus)
    if a.corpus_filter:
        keep = set(a.corpus_filter.split(","))
        cases = [c for c in cases if c.corpus in keep]
    if a.tag:
        cases = [c for c in cases if a.tag in c.tags]
    if a.config:
        configs = harness.load_configs(a.config)
    else:
        corpora = dict(kv.split("=", 1) for kv in (a.map or []))
        configs = [harness.RunConfig(
            name=a.name or "adhoc", corpora=corpora, mode=a.mode, rerank=a.rerank,
            top_k=a.top_k, bm25_weight=a.bm25_weight, rrf_k=a.rrf_k,
        )]
    if a.only:
        keep = set(a.only.split(","))
        configs = [c for c in configs if c.name in keep]
    runs = []
    for cfg in configs:
        print(f"== {cfg.name}: {len(cases)} cases", file=sys.stderr, flush=True)
        r = harness.run_config(svc, cfg, cases, repeat=a.repeat)
        runs.append(r)
        s = r["summary"]
        print(f"   hit@1 {s['hit@1']}  hit@3 {s['hit@3']}  hit@5 {s['hit@5']}  mrr {s['mrr']}"
              f"  neg_fp {s['neg_fp_rate']}  p50 {s['lat_p50_ms']}ms  misses={len(s['misses'])}",
              file=sys.stderr, flush=True)
    out = Path(a.out) if a.out else (
        harness.eval_home(svc) / "reports" / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
    print(harness.table(runs))
    print(f"\nreport: {out}", file=sys.stderr)
    return 0


def cmd_report(a) -> int:
    runs = harness.load_runs(a.paths)
    print(harness.table(runs))
    if a.cases:
        for r in runs:
            print(f"\n### {r['config']['name']}")
            for c in r["cases"]:
                flag = "NEG" if c["negative"] else ("MISS" if c["rank"] is None else f"r{c['rank']}")
                print(f"  {flag:>5}  {c['top_pct']:5.1f}%  {c['latency_ms']:7.1f}ms  {c['id']}")
    return 0


def cmd_drop(a) -> int:
    svc = _svc()
    dropped = harness.drop_eval_corpora(svc, a.names or None)
    print(json.dumps({"dropped": dropped}))
    return 0


def cmd_list(a) -> int:
    svc = _svc()
    for c in svc.registry.list():
        tag = "eval" if c.name.startswith(harness.EVAL_PREFIX) else "live"
        q, b = harness.corpus_bytes(svc, c.name)
        print(f"{tag:4} {c.name:28} {c.chunker:16} {c.embed_model:16} docs={c.doc_count:6} "
              f"chunks={c.chunk_count:7} decay={c.decay_half_life_days} w={getattr(c,'rank_weight',1.0)} "
              f"qdrant={q/1e6:8.1f}MB bm25={b/1e6:7.1f}MB")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vecgrep.eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="index a live corpus's sources into an eval-* corpus")
    b.add_argument("--from", dest="src", required=True)
    b.add_argument("--to", dest="dst", required=True)
    b.add_argument("--chunker")
    b.add_argument("--include")
    b.add_argument("--root", help="override the source root (default: common parent of sources)")
    b.add_argument("--quantization", choices=["int8", "binary"])
    b.add_argument("--datatype", choices=["float32", "float16"], default="float32")
    b.add_argument("--force", action="store_true")
    b.add_argument("--no-settle", dest="settle", action="store_false")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="run gold queries under one or more configs")
    r.add_argument("--gold", required=True)
    r.add_argument("--default-corpus")
    r.add_argument("--corpus-filter", help="comma list of logical corpora to keep")
    r.add_argument("--tag")
    r.add_argument("--config", help="json with a list of RunConfig dicts")
    r.add_argument("--only", help="comma list of config names to run")
    r.add_argument("--name")
    r.add_argument("--map", action="append", help="logical=actual (repeatable)")
    r.add_argument("--mode", default="hybrid")
    r.add_argument("--rerank", action="store_true")
    r.add_argument("--top-k", type=int, default=10)
    r.add_argument("--bm25-weight", type=float)
    r.add_argument("--rrf-k", type=int)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--cases", action="store_true")
    rp.set_defaults(fn=cmd_report)

    d = sub.add_parser("drop", help="delete eval-* corpora (never anything else)")
    d.add_argument("names", nargs="*")
    d.set_defaults(fn=cmd_drop)

    ls = sub.add_parser("list")
    ls.set_defaults(fn=cmd_list)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
