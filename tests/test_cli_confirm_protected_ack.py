"""End-to-end CLI coverage for protected pending-proposal acknowledgement."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import vecgrep.backend.write.confirm as confirm_module
from vecgrep.backend.write import proposal as proposal_module
from vecgrep.cli import main as cli_main


class FakeService:
    def __init__(self, *args, **kwargs):
        self.indexed = ""

    def index(self, source, _corpus, **_kwargs):
        self.indexed = str(source)
        return (1, 1, 0)

    def search(self, *_args, **_kwargs):
        return [SimpleNamespace(source_id=self.indexed)] if self.indexed else []

    def delete_source(self, *_args, **_kwargs):
        return None


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    from vecgrep.backend import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(cli_main, "VecgrepService", FakeService)
    return home, CliRunner()


def _seed(corpus_dir: Path, doc_id: str, *, tier: str = "normal") -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntier: {tier}\n---\n\nbody {doc_id}\n"
    )


def _pending_protected(home: Path, operation: str):
    corpus_dir = home / "write" / "notes"
    if operation == "write":
        proposal = proposal_module.propose(
            "notes",
            "protected body",
            corpus_dir,
            meta={"origin": "bot-suggested", "tier": "protected"},
        )
    elif operation == "edit":
        _seed(corpus_dir, "notes-1", tier="protected")
        proposal = proposal_module.propose(
            "notes", "changed", corpus_dir, edit_id="notes-1"
        )
    elif operation == "delete":
        _seed(corpus_dir, "notes-1", tier="protected")
        proposal = proposal_module.propose_delete("notes", "notes-1", corpus_dir)
    elif operation == "merge":
        _seed(corpus_dir, "notes-1")
        _seed(corpus_dir, "notes-2", tier="protected")
        proposal = proposal_module.propose_merge(
            "notes", ["notes-1", "notes-2"], "merged", corpus_dir
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(operation)

    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)
    return proposal, store, corpus_dir


@pytest.mark.parametrize("operation", ["write", "edit", "delete", "merge"])
def test_cli_requires_explicit_ack_for_every_protected_operation(
    cli_home, operation
):
    home, runner = cli_home
    proposal, store, corpus_dir = _pending_protected(home, operation)
    before = {path.name: path.read_text() for path in corpus_dir.glob("*.md")}

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code != 0
    assert "protected" in result.output.lower()
    assert store.get(proposal.proposal_id) is not None
    after = {path.name: path.read_text() for path in corpus_dir.glob("*.md")}
    assert after == before


def test_cli_exact_ack_allows_protected_confirmation(cli_home):
    home, runner = cli_home
    proposal, store, _corpus_dir = _pending_protected(home, "write")

    result = runner.invoke(
        cli_main.cli,
        ["confirm", proposal.proposal_id, "--ack", proposal.doc_id],
    )

    assert result.exit_code == 0, result.output
    assert Path(proposal.target_path).exists()
    assert store.get(proposal.proposal_id) is None


def test_cli_wrong_ack_keeps_protected_proposal_pending(cli_home):
    home, runner = cli_home
    proposal, store, _corpus_dir = _pending_protected(home, "write")

    result = runner.invoke(
        cli_main.cli,
        ["confirm", proposal.proposal_id, "--ack", "wrong-id"],
    )

    assert result.exit_code != 0
    assert store.get(proposal.proposal_id) is not None
    assert not Path(proposal.target_path).exists()


@pytest.mark.parametrize("tier_line", ["tier:\tprotected", "tier:  protected"])
def test_cli_requires_ack_for_whitespace_variant_protected_tier(
    cli_home, tier_line
):
    home, runner = cli_home
    corpus_dir = home / "write" / "notes"
    _seed(corpus_dir, "notes-1")
    target = corpus_dir / "notes-1.md"
    target.write_text(target.read_text().replace("tier: normal", tier_line))
    proposal = proposal_module.propose(
        "notes", "changed", corpus_dir, edit_id="notes-1"
    )
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code != 0
    assert "protected" in result.output.lower()
    assert store.get(proposal.proposal_id) is not None
    assert "changed" not in target.read_text()


def test_cli_ordinary_proposal_still_needs_no_ack(cli_home):
    home, runner = cli_home
    corpus_dir = home / "write" / "notes"
    proposal = proposal_module.propose("notes", "ordinary", corpus_dir)
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code == 0, result.output
    assert Path(proposal.target_path).exists()


@pytest.mark.parametrize("operation", ["write", "merge"])
def test_core_requires_ack_before_protected_writethrough(
    cli_home, monkeypatch, operation
):
    home, _runner = cli_home
    proposal, store, corpus_dir = _pending_protected(home, operation)
    monkeypatch.setenv("VECGREP_WRITETHROUGH_NOTES", "/bin/true")
    called = False

    def fake_writethrough(*_args, **_kwargs):
        nonlocal called
        called = True
        return confirm_module.ConfirmResult(
            ok=True, doc_id=proposal.doc_id, path="", message="routed"
        )

    monkeypatch.setattr(confirm_module, "_run_writethrough", fake_writethrough)

    with pytest.raises(confirm_module.ConfirmError, match="protected"):
        confirm_module.confirm(
            proposal.proposal_id,
            store,
            FakeService(),
            "notes",
            corpus_dir,
            confirmed_by="owner",
        )

    assert called is False
    assert store.get(proposal.proposal_id) is not None


def test_writethrough_merge_checks_absorbed_docs_in_real_mirror_directory(
    cli_home, monkeypatch
):
    home, _runner = cli_home
    mirror_dir = home / "mirror" / "notes"
    _seed(mirror_dir, "notes-1")
    _seed(mirror_dir, "notes-2", tier="protected")
    proposal = proposal_module.propose_merge(
        "notes", ["notes-1", "notes-2"], "merged", mirror_dir
    )
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)
    monkeypatch.setenv("VECGREP_WRITETHROUGH_NOTES", "/bin/true")
    called = False

    def fake_writethrough(*_args, **_kwargs):
        nonlocal called
        called = True
        return confirm_module.ConfirmResult(
            ok=True, doc_id=proposal.doc_id, path="", message="routed"
        )

    monkeypatch.setattr(confirm_module, "_run_writethrough", fake_writethrough)

    with pytest.raises(confirm_module.ConfirmError, match="protected"):
        confirm_module.confirm(
            proposal.proposal_id,
            store,
            FakeService(),
            "notes",
            home / "write" / "notes",
            confirmed_by="owner",
        )

    assert called is False
    assert store.get(proposal.proposal_id) is not None
