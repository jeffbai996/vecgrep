from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

# Tags whose content we always want to drop — they never contain user-readable prose.
_STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form")
_MAX_REDIRECTS = 5


def _assert_public_url(url: str, *, resolver=socket.getaddrinfo) -> None:
    """Reject non-http(s) URLs or hosts resolving to loopback/private/tailnet IPs."""
    try:
        parsed = urlsplit((url or "").strip())
    except ValueError as exc:
        raise AdapterError(f"Invalid URL: {url}") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AdapterError(f"Unsupported URL scheme: {url}")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".ts.net")):
        raise AdapterError(f"Refusing non-public URL host: {host}")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise AdapterError(f"Refusing non-public URL IP: {host}")
        return
    except ValueError:
        pass
    try:
        answers = resolver(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise AdapterError(f"Could not resolve URL host {host}: {exc}") from exc
    addrs = {a[4][0] for a in answers}
    if not addrs:
        raise AdapterError(f"No addresses resolved for {host}")
    for addr in addrs:
        try:
            if not ipaddress.ip_address(addr).is_global:
                raise AdapterError(f"Refusing URL resolving to non-public IP: {host}")
        except ValueError as exc:
            raise AdapterError(f"Invalid resolved IP for {host}") from exc


@register_adapter
class URLAdapter(Adapter):
    name = "url"

    @classmethod
    def matches(cls, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    def load(self, source: str) -> Iterator[Document]:
        current = source
        _assert_public_url(current)
        headers = {"User-Agent": "vecgrep/0.1 (+https://github.com/jeffbai996/vecgrep)"}
        try:
            r = None
            for _ in range(_MAX_REDIRECTS + 1):
                r = httpx.get(
                    current,
                    follow_redirects=False,
                    timeout=20.0,
                    headers=headers,
                )
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    current = urljoin(current, r.headers["location"])
                    _assert_public_url(current)
                    continue
                break
            else:
                raise AdapterError(f"Too many redirects fetching {source}")
            assert r is not None
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
