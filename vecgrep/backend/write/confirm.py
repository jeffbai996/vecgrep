"""Confirm gate + write + re-embed + verify.

A proposal (from propose()) is persisted in a ProposalStore. confirm() looks it
up by id — an unknown/stale id is rejected (the gate) — then:

  1. WRITE the file. A NEW entry won't clobber an existing id; an EDIT overwrites
     its target in place (simple, no versioning).
  2. RE-EMBED just the written file (incremental index), not a full rebuild.
  3. VERIFY the doc is retrievable; if not, return ok=False with a flag rather
     than silently leaving a written-but-unsearchable file.

Identity-of-confirmer enforcement (human-only authorization; bots may propose
but never confirm) + the no-read->write-loop bar is the wall (confirm_gate.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .proposal import Proposal


class ConfirmError(RuntimeError):
    """A confirm that can't proceed: unknown proposal, overwrite attempt, etc."""


@dataclass
class ConfirmResult:
    ok: bool
    doc_id: str
    path: str
    message: str = ""


class ProposalStore:
    """Disk-persisted pending proposals, keyed by proposal_id. Survives across
    the propose→confirm gap (separate calls / processes)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, proposal_id: str) -> Path:
        safe = "".join(c for c in proposal_id if c.isalnum() or c in "-_")
        return self.root / f"{safe}.json"

    def put(self, proposal: Proposal) -> None:
        self._path(proposal.proposal_id).write_text(
            json.dumps(proposal.__dict__, default=str, indent=2)
        )

    def get(self, proposal_id: str) -> Proposal | None:
        p = self._path(proposal_id)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        return Proposal(**d)

    def delete(self, proposal_id: str) -> None:
        self._path(proposal_id).unlink(missing_ok=True)


def _verify_searchable(svc, corpus: str, doc_id: str, content: str) -> bool:
    """Confirm the just-written doc is retrievable. Best-effort: query with a
    slice of the content and check the new doc_id shows up in the results."""
    probe = (content or "").strip().split("\n", 1)[0][:120] or doc_id
    try:
        results = svc.search(probe, corpus_name=corpus, top_k=10)
    except Exception:
        return False
    for r in results:
        sid = getattr(r, "source_id", "") or ""
        if doc_id in sid:
            return True
    return False


def confirm(
    proposal_id: str,
    store: ProposalStore,
    svc,
    corpus: str,
    corpus_dir: Path,
) -> ConfirmResult:
    """Run the gated write pipeline for a pending proposal. Raises ConfirmError
    if the proposal is unknown (the gate) or the write would overwrite."""
    proposal = store.get(proposal_id)
    if proposal is None:
        raise ConfirmError(
            f"No pending proposal {proposal_id!r} — confirm must cite a live "
            "proposal id (a stale or replayed confirm is rejected)."
        )

    target = Path(proposal.target_path)
    # A NEW entry must not silently clobber an existing doc id (that'd be an
    # accidental overwrite, not an edit). An EDIT targets an existing id on
    # purpose and overwrites it in place — simple, no versioning.
    if target.exists() and not proposal.is_edit:
        raise ConfirmError(
            f"{proposal.doc_id} already exists — use an edit (edit_id=...) to "
            f"change it, not a new write."
        )
    if not target.exists() and proposal.is_edit:
        raise ConfirmError(
            f"Edit target {proposal.doc_id} does not exist."
        )

    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(proposal.rendered)

    # Re-embed just this file (incremental), then verify retrievability.
    try:
        svc.index(str(target), corpus)
    except Exception as e:
        # The file is on disk (truth); flag the embed failure rather than
        # pretending success. A re-index later recovers it.
        store.delete(proposal_id)
        return ConfirmResult(
            ok=False, doc_id=proposal.doc_id, path=str(target),
            message=f"written but re-embed failed: {e}",
        )

    searchable = _verify_searchable(svc, corpus, proposal.doc_id,
                                    proposal.rendered)
    store.delete(proposal_id)
    if not searchable:
        return ConfirmResult(
            ok=False, doc_id=proposal.doc_id, path=str(target),
            message="written + embedded but post-write verify did not retrieve "
                    "it — flag for reindex (not silently dropped).",
        )
    return ConfirmResult(
        ok=True, doc_id=proposal.doc_id, path=str(target),
        message="written, embedded, verified retrievable.",
    )
