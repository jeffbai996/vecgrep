from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

# Tags whose content we always want to drop — they never contain user-readable prose.
_STRIP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form")
_MAX_REDIRECTS = 5


def _public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return _public_address(str(ip.ipv4_mapped))
        if (ip.scope_id or ip.sixtofour or ip.teredo or ip.is_site_local
                or ip in ipaddress.ip_network("64:ff9b::/96")):
            return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved)


def _public_destination(url: str, *, resolver=None) -> tuple[httpx.URL, list[str]]:
    try:
        parsed = httpx.URL(url.strip())
    except (httpx.InvalidURL, ValueError) as exc:
        raise AdapterError("Invalid URL") from exc
    if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
        raise AdapterError("Expected an HTTP or HTTPS URL without credentials")
    host = parsed.raw_host.decode("ascii").lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".ts.net")):
        raise AdapterError(f"Refusing non-public URL host: {host}")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            answers = (resolver or socket.getaddrinfo)(
                host, parsed.port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise AdapterError(f"Could not resolve URL host {host}") from exc
        addresses = list(dict.fromkeys(row[4][0] for row in answers))
    else:
        addresses = [str(literal)]
    if not addresses or not all(_public_address(addr) for addr in addresses):
        raise AdapterError(f"Refusing URL with no public DNS answer or a non-public IP: {host}")
    return parsed, addresses


def _assert_public_url(url: str, *, resolver=None) -> None:
    _public_destination(url, resolver=resolver)


def _get_public(url: str, headers: dict[str, str]) -> httpx.Response:
    original, addresses = _public_destination(url)
    # One pool per hop. Use the vetted numeric address for the dial while
    # retaining the URL's authority for Host, TLS SNI and certificate checks.
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=20.0) as client:
        for index, address in enumerate(addresses):
            try:
                return client.get(
                    original.copy_with(host=address),
                    headers={**headers, "Host": original.netloc.decode("ascii")},
                    extensions={"sni_hostname": original.raw_host.decode("ascii").rstrip(".")},
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if index == len(addresses) - 1:
                    raise
    raise AssertionError("destination has no addresses")


@register_adapter
class URLAdapter(Adapter):
    name = "url"

    @classmethod
    def matches(cls, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    def load(self, source: str) -> Iterator[Document]:
        current = source
        headers = {"User-Agent": "vecgrep/0.1 (+https://github.com/jeffbai996/vecgrep)"}
        try:
            r = None
            for _ in range(_MAX_REDIRECTS + 1):
                r = _get_public(current, headers)
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    current = urljoin(current, r.headers["location"])
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
