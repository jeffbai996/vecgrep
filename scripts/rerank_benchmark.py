#!/usr/bin/env python3
"""Offline, fixed-candidate reranker comparison; defaults to a NOT_RUN plan.

Run only with a local model directory and an approved CPU work budget. Inputs
and output may contain private evaluation material: keep them outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def identity(root):
    root = Path(root).resolve()
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("benchmark roots must be clean committed artifacts")
    return {"commit": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"), "clean": True,
            "rerank_blob": git(root, "rev-parse", "HEAD:vecgrep/backend/rerank.py")}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("fixture exceeds 8 MiB budget")
    cases = json.loads(raw)["cases"]
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    if not 1 <= len(cases) <= 50:
        raise ValueError("use 1..50 frozen cases")
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError("duplicate case ID")
        ids.add(case["id"])
        if not isinstance(case.get("negative", False), bool):
            raise ValueError("negative must be boolean")
        if any(not isinstance(w, str) for w in case.get("want", [])):
            raise ValueError("want must contain source strings")
        if not isinstance(case["query"], str) or not case["query"]:
            raise ValueError("query must be nonempty text")
        candidates = case["candidates"]
        if not 1 <= len(candidates) <= 50:
            raise ValueError("use 1..50 candidates per case")
        if not case.get("negative") and not case.get("want"):
            raise ValueError("positive cases require existing gold expectations")
        for c in candidates:
            if not isinstance(c["text"], str) or len(c["text"]) > 40000:
                raise ValueError("candidate must be text of at most 40000 characters")
            if not isinstance(c["source_id"], str) or not c["source_id"]:
                raise ValueError("candidate requires source_id")
        if not case.get("negative") and not any(
            w in c["source_id"] for w in case["want"] for c in candidates
        ):
            raise ValueError("positive gold is absent from the frozen pool")
    return cases, hashlib.sha256(raw).hexdigest()


def model_identity(path):
    """Fingerprint local weights/config without storing paths or model content."""
    digest = hashlib.sha256()
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    if not files or len(files) > 500:
        raise ValueError("model file count exceeds budget")
    total = 0
    for file in files:
        total += file.stat().st_size
        if total > 16 * 1024**3:
            raise ValueError("model exceeds 16 GiB fingerprint budget")
        digest.update(str(file.relative_to(path)).encode())
        digest.update(b"\0")
        with file.open("rb") as stream:
            file_hash = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(block)
        digest.update(file_hash.digest())
    return {"sha256": digest.hexdigest(), "files": len(files), "bytes": total}


def distribution(values):
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"batch_size": kwargs["batch_size"],
                           "pairs": list(pairs)})
        return self.model.predict(pairs, **kwargs)


def measure(module, model, case, cache_hit=False):
    candidates = [(c["text"], {"index": i, "source_id": c["source_id"]})
                  for i, c in enumerate(case["candidates"])]
    module.clear_score_cache()  # Only the isolated imported module's cache.
    if cache_hit:
        module.rerank(case["query"], candidates, "offline-benchmark")
    model.calls.clear()
    started = time.perf_counter()
    rows = module.rerank(case["query"], candidates, "offline-benchmark")
    elapsed = time.perf_counter() - started
    if cache_hit and model.calls:
        raise AssertionError("cache-hit trial unexpectedly predicted")
    if not cache_hit and not model.calls:
        raise AssertionError("uncached trial did not predict")
    if len(rows) != len(candidates) or any(not math.isfinite(s) for s, _ in rows):
        raise AssertionError("invalid or incomplete scores")
    return {"latency_s": elapsed, "order": [p["index"] for _, p in rows],
            "scores": {str(p["index"]): s for s, p in rows},
            "predict_calls": [{"batch_size": call["batch_size"],
                "pair_char_lengths": [len(q) + len(t) for q, t in call["pairs"]]}
                for call in model.calls]}, rows


def worker(config):
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(config["threads"])
    roots = config["roots"]
    # Recheck identities after dispatch before any candidate code is imported.
    for name, root in roots.items():
        if identity(root) != config["identities"][name]:
            raise ValueError("artifact identity changed before worker execution")
    sys.path.insert(0, roots["candidate"])
    from vecgrep.eval.gold import GoldCase
    from vecgrep.eval.metrics import score_case, summarize
    cases, fingerprint = load_cases(config["cases"])
    if fingerprint != config["fixture_sha256"]:
        raise ValueError("fixture changed before worker execution")
    model_fingerprint = model_identity(config["model"])
    started = time.perf_counter()
    kwargs = {"device": "cpu"}
    if config["max_length"]:
        kwargs["max_length"] = config["max_length"]
    real = CrossEncoder(config["model"], **kwargs)
    initialization = time.perf_counter() - started
    model = RecordingModel(real)
    modules = {}
    for name, root in roots.items():
        rr = load_module("benchmark_" + name, Path(root)/"vecgrep/backend/rerank.py")
        rr.RERANK_WORKER_ENABLED = False
        rr.RERANK_BATCH = config["batch_size"]
        rr.RERANK_SCORE_CACHE_SIZE = 100000
        rr._cache["offline-benchmark"] = model
        rr._release_cuda_cache = lambda: None  # CPU-only worker.
        modules[name] = rr
    token_lengths = []
    effective_lengths = []
    for case in cases:
        for c in case["candidates"]:
            tok = real.tokenizer(case["query"], c["text"], truncation=False)["input_ids"]
            token_lengths.append(len(tok))
            effective_lengths.append(min(len(tok), real.max_length) if real.max_length else len(tok))
    # Warm both code paths. This time is never an uncached latency sample.
    for rr in modules.values():
        measure(rr, model, cases[0])
    trials = {name: {"warm_uncached": [], "score_cache_hit": []} for name in roots}
    quality = {name: [] for name in roots}
    comparison = []
    for rep in range(config["repetitions"]):
        order = list(roots) if rep % 2 == 0 else list(reversed(roots))
        for case in cases:
            observed = {}
            for name in order:
                for mode in trials[name]:
                    sample, rows = measure(modules[name], model, case, mode == "score_cache_hit")
                    sample.update(case_id=case["id"], repetition=rep)
                    trials[name][mode].append(sample)
                    if mode == "warm_uncached":
                        observed[name] = sample
                        if rep == 0:
                            gold = GoldCase(id=case["id"], corpus=case.get("corpus", "frozen"),
                                query=case["query"], want=tuple(case.get("want", [])),
                                forbid=tuple(case.get("forbid", [])), negative=case.get("negative", False))
                            quality[name].append(score_case(gold, [
                                {"source_id": p["source_id"], "similarity_pct": s * 100}
                                for s, p in rows], sample["latency_s"]))
            a, b = observed.values()
            comparison.append({"case_id": case["id"], "repetition": rep,
                "same_order": a["order"] == b["order"],
                "max_sigmoid_score_difference": max(abs(a["scores"][i] - b["scores"][i])
                                                     for i in a["scores"])})
    versions = {}
    for package in ["torch", "sentence-transformers", "transformers", "numpy"]:
        versions[package] = importlib.metadata.version(package)
    return {"result": "COMPLETED", "model_identity": model_fingerprint,
        "model_initialization_s": initialization,
        "initialization_note": "local files; OS page cache uncontrolled, not a cold-load comparison",
        "environment": {"platform": platform.system(), "architecture": platform.machine(),
            "python": platform.python_version(), "cpu": platform.processor(), "device": "cpu",
            "threads": config["threads"], "dependencies": versions},
        "model_max_length": real.max_length,
        "pair_token_lengths_untruncated": distribution(token_lengths),
        "pair_token_lengths_effective": distribution(effective_lengths),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if sys.platform == "darwin" else 1024),
        "memory_note": "whole-worker peak RSS, shared model; not attributed to a variant",
        "latency": {name: {mode: distribution([s["latency_s"] for s in samples])
                      for mode, samples in modes.items()} for name, modes in trials.items()},
        "fixed_pool_metrics": {name: summarize(rows) for name, rows in quality.items()},
        "metric_note": "rerank-only source-level gold metrics, sigmoid 60% negative floor; not end-to-end retrieval or production display confidence",
        "comparison": comparison, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, help="local immutable model directory; no downloads")
    parser.add_argument("--run", action="store_true", help="execute after CPU work budget is approved")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        config = json.loads(args.worker_config.read_text())
        result = worker(config)
        Path(config["worker_output"]).write_text(json.dumps(result))
        return
    if any(x is None for x in [args.baseline_root, args.candidate_root, args.cases, args.output]):
        parser.error("roots, cases and output are required")
    if not (1 <= args.repetitions <= 20 and 1 <= args.batch_size <= 32 and
            1 <= args.threads <= 4 and 1 <= args.timeout <= 1800 and args.max_length >= 0):
        parser.error("work budget exceeds bounds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "result": "NOT_RUN",
        "reason": "artifact/input validation has not completed"}) + "\n")
    roots = {"baseline": str(args.baseline_root.resolve()), "candidate": str(args.candidate_root.resolve())}
    cases, fingerprint = load_cases(args.cases)
    config = {"roots": roots, "identities": {n: identity(p) for n, p in roots.items()},
        "cases": str(args.cases.resolve()), "fixture_sha256": fingerprint,
        "repetitions": args.repetitions, "batch_size": args.batch_size,
        "max_length": args.max_length, "threads": args.threads}
    report = {"schema_version": 1, "result": "NOT_RUN", "started_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": config["identities"], "fixture_sha256": fingerprint, "cases": len(cases),
        "repetitions": args.repetitions, "batch_size": args.batch_size,
        "pair_character_lengths": distribution([len(c["query"]) + len(d["text"])
                                                 for c in cases for d in c["candidates"]]),
        "reason": "plan only; no model was loaded"}
    # Invalidate a previous green report before initialization or execution.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.run:
        try:
            if args.model is None or not args.model.is_dir():
                raise ValueError("--run requires an existing local model directory")
            with tempfile.TemporaryDirectory(prefix="vecgrep-benchmark-") as tmp:
                config.update(model=str(args.model.resolve()), worker_output=str(Path(tmp)/"result.json"))
                path = Path(tmp)/"config.json"
                path.write_text(json.dumps(config))
                env = {k:v for k,v in os.environ.items() if not k.startswith("VECGREP_")}
                env.pop("OPENAI_API_KEY", None)
                env.update(VECGREP_HOME=str(Path(tmp)/"home"), HF_HOME=str(Path(tmp)/"hf"),
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-config", str(path)],
                               env=env, check=True, timeout=args.timeout)
                report.update(json.loads(Path(config["worker_output"]).read_text()))
                report.pop("reason", None)
        except BaseException as exc:
            report.update(result="FAILED", reason=type(exc).__name__)
            raise
        finally:
            report["ended_at"] = datetime.now(timezone.utc).isoformat()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "cases": report["cases"]}))


if __name__ == "__main__":
    main()
