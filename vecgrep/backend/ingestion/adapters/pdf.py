from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .base import Adapter, AdapterError, Document
from .registry import register_adapter
from ..visual_queue import VisualQueue, enabled


@register_adapter
class PDFAdapter(Adapter):
    name = "pdf"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        return p.is_file() and p.suffix.lower() == ".pdf"

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        try:
            reader = PdfReader(str(p))
        except (OSError, PdfReadError) as e:
            raise AdapterError(f"Could not open PDF {source}: {e}") from e

        # Concatenate page text with page markers. Stays as one Document so
        # the chunker decides chunk boundaries — page-per-doc fragmented too
        # aggressively in testing.
        parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                parts.append(f"[page {i}]\n{txt.strip()}")

        full = "\n\n".join(parts)
        visual_metadata = {}
        if enabled():
            try:
                queue = VisualQueue.default()
                enrichment = queue.result(p)
                visual_metadata = {
                    "visual_state": queue.state(p),
                    "visual_original": str(p.resolve()),
                    "visual_generated": bool(enrichment),
                }
                if enrichment:
                    full += "\n\n[visual enrichment]\n" + enrichment
            except (OSError, ValueError) as exc:
                raise AdapterError(f"Could not queue PDF {source}: {exc}") from exc
        if not full.strip() and not visual_metadata:
            raise AdapterError(
                f"PDF {source} produced no extractable text. "
                "It may be scanned/image-only — vecgrep does not OCR."
            )

        yield Document(
            source_id=str(p.resolve()),
            text=full,
            metadata={
                "adapter": self.name,
                "path": str(p.resolve()),
                "pages": len(reader.pages),
                **visual_metadata,
            },
        )
