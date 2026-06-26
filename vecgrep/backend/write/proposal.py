"""Propose a write — a new entry or an edit. Writes NOTHING.

A Proposal is the rendered file the write tool WOULD create (new) or replace
(edit), plus the assigned/target id and path. Confirming it (confirm.py,
human-gated) is what writes. Pure + read-only here: reads the corpus dir to
pick the next id or to render an edit-preview, never writes.

Deliberately simple: write entries, edit entries (overwrite), behind a gate.
No versioning / supersede / archive — an edit overwrites in place.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Schema field vocabularies (see spec). Validated at propose time so a bad
# value is caught before it ever reaches a write.
ORIGINS = ("human", "bot-suggested")
TIERS = ("normal", "protected")
SOURCE_KINDS = ("insight", "fact", "correction", "journal", "decision",
                "memory", "todo")

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*-(\d+)$")


class ProposalError(ValueError):
    """A proposal that can't be constructed (bad metadata, bad target)."""


@dataclass
class Proposal:
    """A pending, un-written proposal. The confirm step references proposal_id."""
    proposal_id: str          # stable handle a human confirm must cite
    doc_id: str               # e.g. note-001 (new) or note-007 (edit/delete target)
    corpus: str
    target_path: str          # where the file WOULD be written (or removed)
    rendered: str             # full file content (write/edit) OR a removal preview
    is_edit: bool             # True = overwrite an existing doc; False = new
    meta: dict = field(default_factory=dict)
    is_delete: bool = False   # True = REMOVE the target doc on confirm (no write).
    # Default False keeps existing on-disk pending proposals (no is_delete key)
    # deserializing cleanly via Proposal(**dict) — back-compat for the store.


def _slug_prefix(corpus: str) -> str:
    """Per-corpus file prefix: 'note', 'memory', etc. Sanitized, never empty."""
    s = re.sub(r"[^a-z0-9_]", "", corpus.lower()) or "doc"
    if not s[0].isalpha():
        s = "d" + s
    return s


def next_doc_id(corpus_dir: Path, corpus: str) -> str:
    """A fresh doc id for a corpus dir: `<prefix>-<epoch-nanos>`.

    Timestamp-based, not a running serial: agent-proposed entries are throwaway
    (many route upstream on confirm and never persist locally), so a serial
    counter just adds noise and a scan. NANOSECOND precision (time.time_ns), not
    seconds — propose() writes NOTHING to disk, so a same-second disk-existence
    check can't catch a collision; back-to-back proposals would otherwise share a
    seconds-granularity id and overwrite each other on confirm. Nanos make
    distinct ids for rapid-fire calls. Still all-digits → `_ID_RE`-valid, so
    edit/delete id-validation and the corpus-boundary check are unaffected. The
    disk-existence bump stays as a paranoia backstop."""
    prefix = _slug_prefix(corpus)
    ts = time.time_ns()
    while (corpus_dir / f"{prefix}-{ts}.md").exists():
        ts += 1
    return f"{prefix}-{ts}"


def render_doc(doc_id: str, content: str, meta: dict) -> str:
    """Render a doc as YAML-frontmatter + body, deterministic key order."""
    order = [
        "id", "title", "status", "created_at", "origin", "confirmed_by",
        "confirmed_at", "tier", "corpus", "source_kind", "tags",
    ]
    fm = {**meta, "id": doc_id}
    lines = ["---"]
    for k in order:
        if k not in fm or fm[k] is None:
            continue
        v = fm[k]
        rendered = ("[" + ", ".join(str(x) for x in v) + "]"
                    if isinstance(v, (list, tuple)) else str(v))
        lines.append(f"{k}: {rendered}")
    lines.append("---")
    lines.append("")
    lines.append(content.strip())
    lines.append("")
    return "\n".join(lines)


def _validate_meta(meta: dict) -> None:
    if meta.get("origin", "bot-suggested") not in ORIGINS:
        raise ProposalError(f"origin must be one of {ORIGINS}")
    if meta.get("tier", "normal") not in TIERS:
        raise ProposalError(f"tier must be one of {TIERS}")
    sk = meta.get("source_kind")
    if sk is not None and sk not in SOURCE_KINDS:
        raise ProposalError(f"source_kind must be one of {SOURCE_KINDS}")


def propose(
    corpus: str,
    content: str,
    corpus_dir: Path,
    meta: dict | None = None,
    edit_id: str | None = None,
    proposal_id: str | None = None,
) -> Proposal:
    """Build a Proposal. WRITES NOTHING.

    - New entry: assigns the next sequential id under corpus_dir.
    - Edit: targets edit_id and overwrites it on confirm (no versioning).
    """
    if not content or not content.strip():
        raise ProposalError("content is empty")
    meta = dict(meta or {})
    meta.setdefault("origin", "bot-suggested")
    meta.setdefault("tier", "normal")
    meta.setdefault("status", "active")
    meta.setdefault("corpus", corpus)
    # Auto-derive a short human-readable title from the first content line when
    # the caller didn't supply one — a generated, scannable handle for the entry
    # (vs the opaque timestamp id). First non-empty line, leading markdown '#'
    # stripped, capped. Callers can still pass an explicit meta["title"].
    if not (meta.get("title") or "").strip():
        derived = ""
        for line in content.splitlines():
            s = line.strip().lstrip("#").strip()
            if s:
                derived = s[:80]
                break
        if derived:
            meta["title"] = derived
    _validate_meta(meta)

    is_edit = edit_id is not None
    if is_edit:
        # An edit_id becomes a filename and a path component, so it must be a
        # plain doc-id (prefix-NNN) and nothing else. Without this check an
        # edit_id like "../outside" would resolve target_path outside
        # corpus_dir, letting confirm() read+overwrite an arbitrary file —
        # the corpus boundary is the whole point of the gate.
        if not _ID_RE.match(edit_id):
            raise ProposalError(
                f"edit_id {edit_id!r} is not a valid doc id (expected "
                f"prefix-NNN, e.g. note-007) — path separators and '..' are "
                f"rejected so an edit can't escape the corpus directory."
            )
        doc_id = edit_id
    else:
        doc_id = next_doc_id(corpus_dir, corpus)

    # Belt-and-suspenders: the resolved target must stay under corpus_dir even
    # if doc_id ever slips past _ID_RE. A new entry's id is generated, but we
    # confine both paths the same way so the invariant holds regardless of how
    # doc_id was produced.
    base = corpus_dir.resolve()
    target = (corpus_dir / f"{doc_id}.md").resolve()
    if base not in target.parents:
        raise ProposalError(
            f"refusing target {target} outside corpus dir {base}"
        )

    rendered = render_doc(doc_id, content, meta)
    # proposal_id MUST be unique per proposal. Deriving it only from doc_id
    # collided: two un-confirmed new entries on the same corpus both pick the
    # same next doc_id, so the second proposal's file overwrote the first in
    # the store — a human could then confirm id X and write content Y they
    # never reviewed. A uuid nonce makes every proposal its own handle.
    pid = proposal_id or f"prop-{doc_id}-{uuid.uuid4().hex[:8]}"
    return Proposal(
        proposal_id=pid,
        doc_id=doc_id,
        corpus=corpus,
        target_path=str(target),
        rendered=rendered,
        is_edit=is_edit,
        meta=meta,
    )


def propose_delete(
    corpus: str,
    delete_id: str,
    corpus_dir: Path,
    meta: dict | None = None,
    proposal_id: str | None = None,
) -> Proposal:
    """Build a DELETE proposal — WRITES/REMOVES NOTHING. Confirming it (human-
    gated) removes the doc file + its embeddings.

    Mirrors propose()'s corpus-boundary safety: delete_id must be a plain
    doc-id (prefix-NNN) so it can't escape corpus_dir via path separators/'..'.
    The target must exist (you can't delete what isn't there). `rendered` carries
    a human-readable preview of what WILL be removed, so the confirm card shows
    the doomed entry rather than a blank — a delete you can't see is a delete you
    can't review."""
    meta = dict(meta or {})
    meta.setdefault("origin", "bot-suggested")
    if meta.get("origin") not in ORIGINS:
        raise ProposalError(f"origin must be one of {ORIGINS}")
    if not _ID_RE.match(delete_id):
        raise ProposalError(
            f"delete_id {delete_id!r} is not a valid doc id (expected "
            f"prefix-NNN, e.g. note-007) — path separators and '..' are "
            f"rejected so a delete can't escape the corpus directory."
        )
    base = corpus_dir.resolve()
    target = (corpus_dir / f"{delete_id}.md").resolve()
    if base not in target.parents:
        raise ProposalError(
            f"refusing target {target} outside corpus dir {base}"
        )
    if not target.exists():
        raise ProposalError(
            f"delete target {delete_id} does not exist in corpus {corpus!r}."
        )
    # Preview the doomed doc so the confirm step shows what's being removed.
    try:
        current = target.read_text()
    except OSError:
        current = ""
    rendered = (f"# DELETE {delete_id} from {corpus}\n"
                f"# The entry below will be REMOVED on confirm:\n\n{current}")
    pid = proposal_id or f"prop-del-{delete_id}-{uuid.uuid4().hex[:8]}"
    return Proposal(
        proposal_id=pid,
        doc_id=delete_id,
        corpus=corpus,
        target_path=str(target),
        rendered=rendered,
        is_edit=False,
        meta=meta,
        is_delete=True,
    )
