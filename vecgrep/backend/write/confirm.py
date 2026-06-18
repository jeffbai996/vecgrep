"""Confirm gate + append-only write + re-embed + verify (write-tool phase 3).

A proposal (from propose()) is persisted in a ProposalStore. confirm() looks it
up by id — an unknown/stale id is rejected (the gate) — then runs the write
pipeline:

  1. WRITE, append-only. Refuses to overwrite an existing file path; a real edit
     goes through versioning (a later phase) which writes a NEW file + supersede.
     "Destructive overwrite" is structurally not a code path here.
  2. RE-EMBED just the written file (incremental index), not a full rebuild.
  3. VERIFY the new doc is retrievable; if not, return ok=False with a flag
     rather than silently leaving a written-but-unsearchable file.

Identity-of-confirmer enforcement (human-only authorization; bots may propose
but never confirm) is the wall, built + adversarially tested in a later phase.
This phase delivers the pipeline + the no-overwrite + verify guarantees.
"""
from __future__ import annotations

import json
import re
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


def _version_of(text: str) -> int:
    """Parse `version: N` from a rendered doc's frontmatter; default 1."""
    m = re.search(r"^version:\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else 1


def _mark_superseded(text: str, superseded_by: str, new_version) -> str:
    """Return the doc text with status flipped to superseded + a forward
    pointer added (so the frozen archive records what replaced it and when)."""
    # status: active -> superseded (insert if absent).
    if re.search(r"^status:\s*", text, re.MULTILINE):
        text = re.sub(r"^status:\s*.*$", "status: superseded", text, count=1,
                      flags=re.MULTILINE)
    else:
        text = re.sub(r"^---\s*$", "---\nstatus: superseded", text, count=1,
                      flags=re.MULTILINE)
    # Add superseded_by pointer just after the status line (idempotent-ish).
    pointer = f"superseded_by: {superseded_by}-v{new_version}" if new_version else \
              f"superseded_by: {superseded_by}"
    if "superseded_by:" not in text:
        text = re.sub(r"^status:\s*superseded$",
                      f"status: superseded\n{pointer}", text, count=1,
                      flags=re.MULTILINE)
    return text


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
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Append-only: never clobber an existing file on a NEW write.
    if target.exists() and not proposal.is_update:
        raise ConfirmError(
            f"Refusing to overwrite existing file {target} — writes are "
            "append-only; an edit must go through versioning."
        )

    archived_paths: list[Path] = []
    if proposal.is_update and target.exists():
        # Supersede, don't destroy: freeze the CURRENT live content to a
        # versioned archive ({doc_id}-vN.md) with status flipped to superseded
        # + a forward pointer, THEN advance the live file to the new version.
        # The original is preserved on disk (audit trail); the live path always
        # holds the active/current version.
        old_text = target.read_text()
        prev_ver = _version_of(old_text)
        archive = corpus_dir / f"{proposal.doc_id}-v{prev_ver}.md"
        archived = _mark_superseded(old_text, superseded_by=proposal.doc_id,
                                    new_version=proposal.meta.get("version"))
        archive.write_text(archived)
        archived_paths.append(archive)

    target.write_text(proposal.rendered)

    # Re-embed the new live file AND any archived version (so the index reflects
    # the status flip — the superseded archive must drop out of active search).
    try:
        svc.index(str(target), corpus)
        for ap in archived_paths:
            svc.index(str(ap), corpus)
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
