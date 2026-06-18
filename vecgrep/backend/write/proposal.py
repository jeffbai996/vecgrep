"""Propose + preview (write-tool phase 2). Writes NOTHING.

A Proposal is the rendered file the write tool WOULD create, plus the assigned
id, target path, and a diff if it updates an existing doc. Confirming a proposal
(a later phase, human-gated) is what actually writes. This module is pure +
read-only: it reads the corpus dir to pick the next id / diff against an existing
file, but never writes.
"""
from __future__ import annotations

import difflib
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
    doc_id: str               # e.g. note-001 (assigned) or note-007 (update target)
    corpus: str
    target_path: str          # where the file WOULD be written
    rendered: str             # the full file content (frontmatter + body)
    is_update: bool
    diff: str                 # unified diff vs the existing file ("" for new)
    meta: dict = field(default_factory=dict)


def _slug_prefix(corpus: str) -> str:
    """Per-corpus file prefix: 'note', 'memory', etc. Sanitized, never empty."""
    s = re.sub(r"[^a-z0-9_]", "", corpus.lower()) or "doc"
    if not s[0].isalpha():
        s = "d" + s
    return s


def next_doc_id(corpus_dir: Path, corpus: str) -> str:
    """Next sequential id for a corpus dir: <prefix>-NNN, scanning existing files.

    Append-only id allocation: max existing numeric suffix + 1, zero-padded to 3.
    """
    prefix = _slug_prefix(corpus)
    highest = 0
    if corpus_dir.exists():
        for p in corpus_dir.glob(f"{prefix}-*.md"):
            m = _ID_RE.match(p.stem)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"{prefix}-{highest + 1:03d}"


def render_doc(doc_id: str, content: str, meta: dict) -> str:
    """Render a doc as YAML-frontmatter + body. Deterministic key order so a
    diff between versions is stable and readable."""
    order = [
        "id", "version", "status", "created_at", "updated_at",
        "origin", "confirmed_by", "confirmed_at", "tier",
        "corpus", "supersedes", "superseded_by", "source_kind", "tags",
    ]
    fm = {**meta, "id": doc_id}
    lines = ["---"]
    for k in order:
        if k not in fm or fm[k] is None:
            continue
        v = fm[k]
        if isinstance(v, (list, tuple)):
            rendered = "[" + ", ".join(str(x) for x in v) + "]"
        else:
            rendered = str(v)
        lines.append(f"{k}: {rendered}")
    lines.append("---")
    lines.append("")
    lines.append(content.strip())
    lines.append("")
    return "\n".join(lines)


def _validate_meta(meta: dict) -> None:
    origin = meta.get("origin", "bot-suggested")
    if origin not in ORIGINS:
        raise ProposalError(f"origin must be one of {ORIGINS}, got {origin!r}")
    tier = meta.get("tier", "normal")
    if tier not in TIERS:
        raise ProposalError(f"tier must be one of {TIERS}, got {tier!r}")
    sk = meta.get("source_kind")
    if sk is not None and sk not in SOURCE_KINDS:
        raise ProposalError(f"source_kind must be one of {SOURCE_KINDS}, got {sk!r}")


def propose(
    corpus: str,
    content: str,
    corpus_dir: Path,
    meta: dict | None = None,
    update_id: str | None = None,
    proposal_id: str | None = None,
) -> Proposal:
    """Build a Proposal. WRITES NOTHING.

    - New doc: assigns the next sequential id under corpus_dir.
    - Update: target update_id; renders as the next version and diffs against
      the existing file so the human can see what changes before confirming.
    """
    if not content or not content.strip():
        raise ProposalError("content is empty")
    meta = dict(meta or {})
    meta.setdefault("origin", "bot-suggested")
    meta.setdefault("tier", "normal")
    meta.setdefault("status", "active")
    meta.setdefault("corpus", corpus)
    _validate_meta(meta)

    is_update = update_id is not None
    if is_update:
        doc_id = update_id
        existing_path = corpus_dir / f"{doc_id}.md"
        old = existing_path.read_text() if existing_path.exists() else ""
        prev_ver = 0
        if old:
            m = re.search(r"^version:\s*(\d+)", old, re.MULTILINE)
            prev_ver = int(m.group(1)) if m else 1
        meta["version"] = prev_ver + 1
        meta.setdefault("supersedes", f"{doc_id}-v{prev_ver}" if prev_ver else None)
    else:
        doc_id = next_doc_id(corpus_dir, corpus)
        meta["version"] = 1
        old = ""

    rendered = render_doc(doc_id, content, meta)
    target_path = str(corpus_dir / f"{doc_id}.md")
    diff = ""
    if is_update and old:
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{doc_id} (current)",
            tofile=f"{doc_id} (proposed v{meta['version']})",
        ))

    return Proposal(
        proposal_id=proposal_id or f"prop-{doc_id}",
        doc_id=doc_id,
        corpus=corpus,
        target_path=target_path,
        rendered=rendered,
        is_update=is_update,
        diff=diff,
        meta=meta,
    )
