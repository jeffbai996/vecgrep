from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..dates import extract_timestamp
from .base import Adapter, AdapterError, Document
from .registry import register_adapter

_MD_EXTS = {".md", ".markdown", ".mdx"}

_FM_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*)$")


def parse_frontmatter(text: str) -> dict:
    """Parse a leading YAML-ish frontmatter block into a flat dict.

    Recognises only the shape vecgrep itself renders (write/proposal.py): a
    `---` fence, then `key: value` lines, with `[a, b, c]` for lists. No nested
    maps, no anchors — deliberately tiny so it needs no YAML dependency and
    can't be surprised by arbitrary markdown that merely contains `---` rules.
    Returns {} when there's no frontmatter block at the very top.

    Why this matters: without it, a markdown doc's frontmatter (status, tier,
    source_kind on confirmed write-tool docs) stays as opaque body text and
    never reaches Document.metadata — so status/tier filters silently match
    nothing and superseded docs leak into default search.
    """
    if not text.startswith("---"):
        return {}
    # Split off the block between the first two `---` fences.
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return {}
    block = rest[:end]
    meta: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FM_LINE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            meta[key] = [x.strip() for x in inner.split(",") if x.strip()] if inner else []
        else:
            meta[key] = raw
    return meta


@register_adapter
class MarkdownAdapter(Adapter):
    name = "markdown"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        return p.is_file() and p.suffix.lower() in _MD_EXTS

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise AdapterError(f"Could not read {source}: {e}") from e
        # Markdown is mostly plaintext for embedding purposes — preserving
        # headers and prose verbatim gives the model maximum signal. We
        # deliberately don't strip markup.
        #
        # Frontmatter, if present, is lifted into metadata so search filters
        # (meta.status, meta.tier, meta.source_kind) work on write-tool docs.
        # adapter/path stay authoritative — frontmatter can't override them.
        meta = {"adapter": self.name, "path": str(p.resolve())}
        meta = {**parse_frontmatter(text), **meta}
        yield Document(
            source_id=str(p.resolve()),
            text=text,
            metadata=meta,
            timestamp=extract_timestamp(text, str(p)),
        )
