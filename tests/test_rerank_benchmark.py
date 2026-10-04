"""Benchmark results must distinguish predictions, cache hits and artifact identity."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from vecgrep.backend import rerank

spec = importlib.util.spec_from_file_location(
    "rerank_benchmark", Path(__file__).parents[1]/"scripts/rerank_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_uncached_and_cache_hit_trials_exercise_distinct_paths(monkeypatch):
    class Model:
        def predict(self, pairs, **kwargs):
            return [len(text) / 100 for _, text in pairs]
    model = bench.RecordingModel(Model())
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", False)
    monkeypatch.setattr(rerank, "RERANK_SCORE_CACHE_SIZE", 100)
    monkeypatch.setattr(rerank, "_load", lambda _: model)
    monkeypatch.setattr(rerank, "_release_cuda_cache", lambda: None)
    case = {"id": "mixed", "query": "q", "candidates": [
        {"text": "short", "source_id": "a"}, {"text": "long "*500, "source_id": "b"}]}
    uncached, _ = bench.measure(rerank, model, case)
    cached, _ = bench.measure(rerank, model, case, cache_hit=True)
    assert len(uncached["predict_calls"]) == 2
    assert cached["predict_calls"] == []
    assert uncached["scores"] == cached["scores"]
    assert uncached["order"] == cached["order"]


def test_cache_hit_cannot_be_claimed_when_cache_is_disabled(monkeypatch):
    class Model:
        def predict(self, pairs, **kwargs):
            return [0.0 for _ in pairs]
    model = bench.RecordingModel(Model())
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", False)
    monkeypatch.setattr(rerank, "RERANK_SCORE_CACHE_SIZE", 0)
    monkeypatch.setattr(rerank, "_load", lambda _: model)
    monkeypatch.setattr(rerank, "_release_cuda_cache", lambda: None)
    case = {"query": "q", "candidates": [{"text": "a", "source_id": "a"}]}
    with pytest.raises(AssertionError, match="unexpectedly predicted"):
        bench.measure(rerank, model, case, cache_hit=True)


def test_stale_gold_is_rejected_before_prediction(tmp_path):
    path = tmp_path/"cases.json"
    path.write_text(json.dumps({"cases": [{"id": "missing", "query": "q", "want": ["absent"],
        "candidates": [{"text": "document", "source_id": "present"}]}]}))
    with pytest.raises(ValueError, match="gold is absent"):
        bench.load_cases(path)


def test_dirty_artifact_cannot_be_benchmarked_as_committed(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path/"dirty.txt").write_text("uncommitted")
    with pytest.raises(ValueError, match="clean committed"):
        bench.identity(tmp_path)


def test_failed_model_start_replaces_previous_green_report(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path/"report.json"
    output.write_text('{"result":"COMPLETED"}')
    cases = tmp_path/"cases.json"
    cases.write_text(json.dumps({"cases": [{"id": "neg", "query": "q", "negative": True,
        "candidates": [{"text": "a", "source_id": "a"}]}]}))
    # Exercise the executable without a model download. Roots are intentionally
    # invalid; an earlier successful report must not remain current.
    import sys
    result = subprocess.run([sys.executable, str(root/"scripts/rerank_benchmark.py"),
        "--baseline-root", str(tmp_path), "--candidate-root", str(tmp_path),
        "--cases", str(cases), "--output", str(output), "--run"], capture_output=True)
    assert result.returncode != 0
    assert json.loads(output.read_text())["result"] != "COMPLETED"


def test_model_identity_changes_with_weights_without_exposing_paths(tmp_path):
    (tmp_path/"config.json").write_text('{"model_type":"test"}')
    weights = tmp_path/"model.safetensors"
    weights.write_bytes(b"weights-a")
    before = bench.model_identity(tmp_path)
    weights.write_bytes(b"weights-b")
    after = bench.model_identity(tmp_path)
    assert before["sha256"] != after["sha256"]
    assert after["files"] == 2
    assert set(after) == {"sha256", "files", "bytes"}


def test_plan_records_artifact_and_input_identity_without_loading_model(tmp_path):
    import sys
    root = tmp_path/"root"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    code = root/"vecgrep/backend/rerank.py"
    code.parent.mkdir(parents=True)
    code.write_text("raise AssertionError('planning must not import this')\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    cases = tmp_path/"cases.json"
    cases.write_text(json.dumps({"cases": [{"id": "neg", "query": "q", "negative": True,
        "candidates": [{"text": "a", "source_id": "a"}]}]}))
    output = tmp_path/"report.json"
    command = [sys.executable, str(Path(__file__).parents[1]/"scripts/rerank_benchmark.py"),
               "--baseline-root", str(root), "--candidate-root", str(root),
               "--cases", str(cases), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["result"] == "NOT_RUN"
    assert report["artifacts"]["candidate"]["clean"] is True
    assert report["artifacts"]["candidate"]["commit"] == bench.git(root, "rev-parse", "HEAD")
    assert report["fixture_sha256"] == bench.load_cases(cases)[1]
    assert "latency" not in report
