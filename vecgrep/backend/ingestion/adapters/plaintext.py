from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

_TEXT_EXTS = {".txt", ".log", ".csv", ".tsv", ".rst", ".org", ".tex", ".srt", ""}


@register_adapter
class PlaintextAdapter(Adapter):
    name = "plaintext"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        if not p.is_file():
            return False
        return p.suffix.lower() in _TEXT_EXTS

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise AdapterError(f"Could not read {source}: {e}") from e
        yield Document(
            source_id=str(p.resolve()),
            text=text,
            metadata={"adapter": self.name, "path": str(p.resolve())},
        )
