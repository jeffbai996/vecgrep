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
from dataclasses import dataclass, field
from pathlib import Path

# Schema field vocabularies (see spec). Validated at propose time so a bad
# value is caught before it ever reaches a write.
ORIGINS = ("human", "bot-suggested")
TIERS = ("normal", "protected")
SOURCE_KINDS = ("insight", "fact", "correction", "journal", "decision")

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*-(\d+)$")


class ProposalError(ValueError):
    """A proposal that can't be constructed (bad metadata, bad target)."""


@dataclass
class Proposal:
    """A pending, un-written proposal. The confirm step references proposal_id."""
    proposal_id: str          # stable handle a human confirm must cite
    doc_id: str               # e.g. note-001 (new) or note-007 (edit target)
    corpus: str
    target_path: str          # where the file WOULD be written
    rendered: str             # the full file content (frontmatter + body)
    is_edit: bool             # True = overwrite an existing doc; False = new
    meta: dict = field(default_factory=dict)


def _slug_prefix(corpus: str) -> str:
    """Per-corpus file prefix: 'note', 'memory', etc. Sanitized, never empty."""
    s = re.sub(r"[^a-z0-9_]", "", corpus.lower()) or "doc"
    if not s[0].isalpha():
        s = "d" + s
    return s


def next_doc_id(corpus_dir: Path, corpus: str) -> str:
    """Next sequential id for a corpus dir: <prefix>-NNN (max existing + 1)."""
    prefix = _slug_prefix(corpus)
    highest = 0
    if corpus_dir.exists():
        for p in corpus_dir.glob(f"{prefix}-*.md"):
            m = _ID_RE.match(p.stem)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"{prefix}-{highest + 1:03d}"


def render_doc(doc_id: str, content: str, meta: dict) -> str:
    """Render a doc as YAML-frontmatter + body, deterministic key order."""
    order = [
        "id", "status", "created_at", "origin", "confirmed_by", "confirmed_at",
        "tier", "corpus", "source_kind", "tags",
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
    _validate_meta(meta)

    is_edit = edit_id is not None
    doc_id = edit_id if is_edit else next_doc_id(corpus_dir, corpus)
    rendered = render_doc(doc_id, content, meta)
    return Proposal(
        proposal_id=proposal_id or f"prop-{doc_id}",
        doc_id=doc_id,
        corpus=corpus,
        target_path=str(corpus_dir / f"{doc_id}.md"),
        rendered=rendered,
        is_edit=is_edit,
        meta=meta,
    )
