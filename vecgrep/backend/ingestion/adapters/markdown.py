from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

_MD_EXTS = {".md", ".markdown", ".mdx"}


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
        yield Document(
            source_id=str(p.resolve()),
            text=text,
            metadata={"adapter": self.name, "path": str(p.resolve())},
        )
