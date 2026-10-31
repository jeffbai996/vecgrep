"""The CLI's JSON-reading handlers catch exactly the errors bad input produces."""
from __future__ import annotations

import json

import click
import httpx
import pytest
from click.testing import CliRunner

from vecgrep.cli import main as cli_main


def _detail_of(response: httpx.Response) -> str:
    with pytest.raises(click.ClickException) as raised:
        cli_main._raise_for_status(response)
    return raised.value.message


def test_error_detail_uses_the_json_detail_field() -> None:
    assert _detail_of(httpx.Response(400, json={"detail": "unknown corpus"})) == "unknown corpus"


def test_error_detail_falls_back_to_body_text_when_body_is_not_json() -> None:
    assert _detail_of(httpx.Response(502, text="bad gateway")) == "bad gateway"


@pytest.mark.parametrize("body", [b'["a list"]', b'"a string"', b"7", b"null"])
def test_error_detail_falls_back_to_body_text_when_json_is_not_an_object(body: bytes) -> None:
    assert _detail_of(httpx.Response(500, content=body)) == body.decode()


def test_error_detail_does_not_mask_an_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_json(self, **kwargs):
        raise RuntimeError("bug in the response handling")

    monkeypatch.setattr(httpx.Response, "json", broken_json)
    with pytest.raises(RuntimeError, match="bug in the response handling"):
        cli_main._raise_for_status(httpx.Response(500, text="x"))


def test_pending_lists_readable_proposals_and_skips_unreadable_ones(tmp_path, monkeypatch) -> None:
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    from vecgrep.backend import config

    monkeypatch.setattr(config, "_settings", None)
    pending = home / "write" / "_pending"
    pending.mkdir(parents=True)
    (pending / "a-good.json").write_text(json.dumps({
        "proposal_id": "p-good", "doc_id": "notes-001", "corpus": "notes",
        "meta": {"origin": "test"},
    }))
    (pending / "b-truncated.json").write_text('{"proposal_id": ')
    (pending / "c-not-utf8.json").write_bytes(b"\xff\xfe\x00")
    (pending / "d-directory.json").mkdir()

    result = CliRunner().invoke(cli_main.cli, ["pending"])

    assert result.exit_code == 0, result.output
    assert "p-good  [new] notes-001 (notes)  origin=test" in result.output
    assert result.output.count("\n") == 1
