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
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .proposal import Proposal, render_doc


def _body_of(rendered: str) -> str:
    """Extract the body below the YAML frontmatter of a rendered doc."""
    parts = rendered.split("---", 2)
    return parts[2].strip() if len(parts) >= 3 else rendered.strip()


def _writethrough_cmd(corpus: str) -> str | None:
    """A corpus may be a downstream MIRROR of some upstream store (e.g. a
    transcript dump). For such a corpus, writing the local mirror file is wrong —
    the next dump clobbers it. Instead, an operator points the corpus at its
    upstream via env `VECGREP_WRITETHROUGH_<corpus>=<command>`. On confirm, that
    command receives the operation as JSON on stdin and performs the REAL write;
    vecgrep skips its mirror write/delete + embed (the upstream's own dump
    re-renders the mirror). Keeps vecgrep generic — the upstream-specific routing
    lives entirely in the operator's command, never in vecgrep source.

    Corpus names aren't valid env-var suffixes verbatim (dashes), so we map any
    non-alphanumeric char to '_' and uppercase: 'squad-store' → SQUAD_STORE."""
    key = "VECGREP_WRITETHROUGH_" + "".join(
        c.upper() if c.isalnum() else "_" for c in corpus)
    cmd = os.environ.get(key, "").strip()
    return cmd or None


def _run_writethrough(cmd: str, *, op: str, corpus: str, doc_id: str,
                      body: str, confirmed_by: str) -> ConfirmResult:
    """Run the operator's write-through command with the op as JSON on stdin.
    The command is responsible for the REAL upstream write; a non-zero exit (or
    spawn failure) is surfaced as a failed confirm so the proposal is NOT
    consumed — the human can retry once the upstream is reachable again."""
    payload = json.dumps({
        "op": op, "corpus": corpus, "doc_id": doc_id,
        "body": body, "confirmed_by": confirmed_by,
    })
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd], input=payload, text=True,
            capture_output=True, timeout=60,
        )
    except Exception as e:
        return ConfirmResult(ok=False, doc_id=doc_id, path="",
                             message=f"write-through spawn failed: {e}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        return ConfirmResult(ok=False, doc_id=doc_id, path="",
                             message=f"write-through failed (exit {proc.returncode}): {err}")
    out = (proc.stdout or "").strip()
    return ConfirmResult(ok=True, doc_id=doc_id, path="",
                         message=f"{op} routed upstream ({corpus}): {out}" if out
                                 else f"{op} routed upstream ({corpus})")


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
    confirmed_by: str | None = None,
    protected_ack: str | None = None,
) -> ConfirmResult:
    """Run the gated write pipeline for a pending proposal.

    THE WALL: a write requires a human confirmer identity (`confirmed_by`).
    A bot may PROPOSE (origin=bot-suggested) but can never authorize its own
    write — only a human confirm turns a proposal into a write. No human
    identity → no write, regardless of the proposal's origin. (The read->write
    bar is enforced one step earlier: propose() only accepts origin in
    {human, bot-suggested}; ingested/retrieved content can't even become a
    proposal.) The confirmer is recorded on the written doc for audit.
    """
    confirmer = (confirmed_by or "").strip()
    if not confirmer:
        raise ConfirmError(
            "A write requires a human confirmer identity (confirmed_by). Bots "
            "may propose but never authorize — no confirmer, no write."
        )

    proposal = store.get(proposal_id)
    if proposal is None:
        raise ConfirmError(
            f"No pending proposal {proposal_id!r} — confirm must cite a live "
            "proposal id (a stale or replayed confirm is rejected)."
        )

    target = Path(proposal.target_path)

    # ── WRITE-THROUGH: if this corpus is a mirror of an upstream store, route the
    #    confirmed op there instead of touching the local mirror file. The human-
    #    confirm wall already passed (confirmer required, above). Protected-tier
    #    acks still apply on an edit/delete of an on-disk protected doc — check
    #    before handing off so a protected entry can't be mutated upstream without
    #    the exact-id ack. The upstream's own dump re-renders the mirror, so we do
    #    NOT write/delete/embed locally here. Consumes the proposal on success.
    wt_cmd = _writethrough_cmd(corpus)
    if wt_cmd is not None:
        op = "delete" if getattr(proposal, "is_delete", False) else (
            "edit" if proposal.is_edit else "write")
        if op in ("edit", "delete") and target.exists():
            if "tier: protected" in target.read_text() \
                    and (protected_ack or "").strip() != proposal.doc_id:
                raise ConfirmError(
                    f"{proposal.doc_id} is protected — re-state its exact id as "
                    f"protected_ack to {op} it (got {protected_ack!r}).")
        body = "" if op == "delete" else _body_of(proposal.rendered)
        res = _run_writethrough(wt_cmd, op=op, corpus=corpus,
                                doc_id=proposal.doc_id, body=body,
                                confirmed_by=confirmer)
        if res.ok:
            store.delete(proposal_id)  # consume only on a successful upstream write
        return res

    # ── DELETE: remove the target doc + its embeddings. Same human-confirm wall
    #    (confirmer required, checked above). The doc must still exist; a
    #    protected doc still needs the exact-id ack so a delete can't be slipped
    #    in by a bot suggestion. Returns early — no write/embed pipeline.
    if getattr(proposal, "is_delete", False):
        if not target.exists():
            # Idempotent-ish: already gone. Clear the pending proposal, report it.
            store.delete(proposal_id)
            return ConfirmResult(
                ok=True, doc_id=proposal.doc_id, path=str(target),
                message=f"{proposal.doc_id} was already absent — nothing to delete.",
            )
        on_disk = target.read_text()
        if "tier: protected" in on_disk and (protected_ack or "").strip() != proposal.doc_id:
            raise ConfirmError(
                f"{proposal.doc_id} is protected — re-state its exact id as "
                f"protected_ack to confirm the DELETE (got {protected_ack!r})."
            )
        # Remove embeddings first (best-effort), then the file. Doing the index
        # side first means a crash leaves an orphaned file (recoverable by a
        # reindex), not orphaned vectors pointing at a missing file.
        index_note = ""
        try:
            svc.delete_source(corpus, str(target))
        except Exception as e:
            index_note = f" (de-index warning: {e}; a reindex will reconcile)"
        try:
            target.unlink()
        except OSError as e:
            raise ConfirmError(f"could not remove {target}: {e}")
        store.delete(proposal_id)
        return ConfirmResult(
            ok=True, doc_id=proposal.doc_id, path=str(target),
            message=f"deleted {proposal.doc_id} from {corpus}{index_note}",
        )

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

    # Protected tier: stronger confirm. Fires if the proposal is protected OR
    # the doc being edited is protected ON DISK (so a bot can't strip protection
    # by omitting the tier in an edit). The confirmer must re-state the exact
    # doc id — deliberate intent, not a fat-finger or a slipped bot suggestion.
    is_protected = proposal.meta.get("tier") == "protected" or (
        proposal.is_edit and target.exists()
        and "tier: protected" in target.read_text()
    )
    if is_protected and (protected_ack or "").strip() != proposal.doc_id:
        raise ConfirmError(
            f"{proposal.doc_id} is protected — re-state its exact id as "
            f"protected_ack to confirm (got {protected_ack!r})."
        )

    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Stamp the confirmer into the doc at WRITE time (it isn't known at propose
    # time) so the on-disk record carries who authorized it — audit provenance.
    final_meta = {**proposal.meta, "confirmed_by": confirmer}
    rendered = render_doc(proposal.doc_id, _body_of(proposal.rendered), final_meta)
    target.write_text(rendered)

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
