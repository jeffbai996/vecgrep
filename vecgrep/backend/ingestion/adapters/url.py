from __future__ import annotations

from typing import Iterator

import httpx
from bs4 import BeautifulSoup

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

# Tags whose content we always want to drop — they never contain user-readable prose.
_STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form")


@register_adapter
class URLAdapter(Adapter):
    name = "url"

    @classmethod
    def matches(cls, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    def load(self, source: str) -> Iterator[Document]:
        try:
            r = httpx.get(
                source,
                follow_redirects=True,
                timeout=20.0,
                headers={"User-Agent": "vecgrep/0.1 (+https://github.com/jeffbai996/vecgrep)"},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise AdapterError(f"Failed to fetch {source}: {e}") from e

        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "xml" not in ctype:
            # Treat anything non-HTML as text-ish. If it's binary, .text will
            # be lossy but we don't try to be clever here.
            text = r.text
        else:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(_STRIP_TAGS):
                tag.decompose()
            title = (soup.title.string or "").strip() if soup.title else ""
            body = soup.get_text(separator="\n")
            # Collapse runs of blank lines without disturbing single newlines.
            lines = [ln.strip() for ln in body.splitlines()]
            cleaned = "\n".join(ln for ln in lines if ln)
            text = (f"# {title}\n\n" if title else "") + cleaned

        if not text.strip():
            raise AdapterError(f"URL {source} returned no readable text.")

        yield Document(
            source_id=source,
            text=text,
            metadata={"adapter": self.name, "url": source, "status": r.status_code},
        )
