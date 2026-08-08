"""CLI contracts for bounded corpus repair."""
from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from vecgrep.cli.main import cli


class _DoctorService:
    def __init__(self) -> None:
        self.diagnose_scope = None
        self.reconcile_scope = None
        self.reindex = None

    def list_corpora(self):
        return [SimpleNamespace(name="cli")]

    def diagnose(self, *, corpora=None):
        self.diagnose_scope = corpora
        return [{"corpus": "cli", "kind": "count_drift", "detail": "short", "fixable": True}]

    def reconcile(self, *, reindex=False, corpora=None):
        self.reindex = reindex
        self.reconcile_scope = corpora
        return [{"corpus": "cli", "kind": "count_drift", "action": "reindexed"}]


def test_doctor_fix_scopes_recovery_and_uses_read_only_cache(monkeypatch):
    service = _DoctorService()
    made = {}

    def build_service(*, embed_cache_read_only=False):
        made["read_only"] = embed_cache_read_only
        return service

    monkeypatch.setattr("vecgrep.cli.main.VecgrepService", build_service)
    result = CliRunner().invoke(cli, ["doctor", "--fix", "--corpus", "cli"])

    assert result.exit_code == 0, result.output
    assert made["read_only"] is True
    assert service.diagnose_scope == {"cli"}
    assert service.reconcile_scope == {"cli"}
    assert service.reindex is True
