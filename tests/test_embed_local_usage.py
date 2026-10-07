import sqlite3

import httpx
import pytest

from vecgrep.backend.embed.ollama import OllamaBackend
from vecgrep.backend.embed.local_usage import record


@pytest.fixture
def receipt_db(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite3"
    monkeypatch.setenv("LOCAL_LLM_USAGE_DB", str(path))
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vecgrep"))
    return path


def rows(path):
    with sqlite3.connect(path) as db:
        return db.execute("SELECT request_id,at,host,model,input_tokens,output_tokens FROM usage").fetchall()


@pytest.mark.parametrize("lane", ["batch", "single", "overflow"])
def test_each_embedding_lane_records_reported_input_tokens(receipt_db, lane):
    backend = OllamaBackend("http://primary.invalid", "bge-m3", overflow_url="http://overflow.invalid")
    def answer(request):
        return httpx.Response(200, json={"model": "bge-test", "embeddings": [[0.1] * 1024],
                                        "prompt_eval_count": 17})
    backend._client.close()
    backend._overflow_client.close()
    backend._client = httpx.Client(transport=httpx.MockTransport(answer))
    backend._overflow_client = httpx.Client(transport=httpx.MockTransport(answer))
    try:
        if lane == "batch": backend._embed_batch(["private text"])
        elif lane == "single": backend._embed_one_resilient("private text")
        else: backend._embed_overflow(["private text"])
    finally:
        backend._client.close()
        backend._overflow_client.close()
    row, = rows(receipt_db)
    assert row[0].startswith("vecgrep:") and row[1] > 0
    assert row[2:] == ("vecgrep", "bge-test", 17, 0)
    assert b"private text" not in receipt_db.read_bytes()
    assert receipt_db.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("data", [{}, {"prompt_eval_count": True},
    {"prompt_eval_count": -1}, {"prompt_eval_count": 2**63}])
def test_unknown_or_invalid_counts_are_not_estimated(receipt_db, data):
    record(data, model="bge-test")
    assert not receipt_db.exists()


def test_unavailable_receipt_storage_does_not_fail_embedding(receipt_db):
    receipt_db.mkdir()
    record({"prompt_eval_count": 12}, model="bge-test")
