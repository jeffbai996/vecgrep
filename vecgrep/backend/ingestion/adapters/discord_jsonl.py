"""Discord JSONL adapter.

Reads exported Discord chat logs (DiscordChatExporter, discord-logger,
or any one-message-per-JSON-line file). Each .jsonl file becomes one
Document; the channel name is taken from the filename. Messages are
flattened into a transcript with `author: content` lines so prose-style
chunkers do something reasonable with it.

Detection: file extension is .jsonl AND the first line parses as JSON
with at least an `author` (or `username`) and a content-like field.
This lets the same .jsonl extension still be picked up by other adapters
or skipped if it's some other JSONL data.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import Adapter, AdapterError, Document
from .registry import register_adapter

_AUTHOR_KEYS = ("author", "username", "user", "from")
_CONTENT_KEYS = ("content", "text", "message", "body")
_TIMESTAMP_KEYS = ("timestamp", "ts", "time", "created_at", "date")


def _peek_first_line(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    except OSError:
        return None
    return None


def _looks_like_discord(obj: dict) -> bool:
    has_author = any(k in obj for k in _AUTHOR_KEYS)
    has_content = any(k in obj for k in _CONTENT_KEYS)
    return has_author and has_content


def _author_name(obj: dict) -> str:
    for k in _AUTHOR_KEYS:
        v = obj.get(k)
        if isinstance(v, dict):
            # Discord exports often nest: {"author": {"name": "...", "id": "..."}}
            for sub in ("name", "username", "global_name"):
                if sub in v:
                    return str(v[sub])
        elif v:
            return str(v)
    return "unknown"


def _content_text(obj: dict) -> str:
    for k in _CONTENT_KEYS:
        v = obj.get(k)
        if v:
            return str(v)
    return ""


def _timestamp(obj: dict) -> str | None:
    for k in _TIMESTAMP_KEYS:
        v = obj.get(k)
        if v:
            return str(v)
    return None


@register_adapter
class DiscordJSONLAdapter(Adapter):
    name = "discord_jsonl"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        if not p.is_file() or p.suffix.lower() != ".jsonl":
            return False
        first = _peek_first_line(p)
        return bool(first and _looks_like_discord(first))

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise AdapterError(f"Could not read {source}: {e}") from e

        lines: list[str] = []
        first_ts: str | None = None
        last_ts: str | None = None
        message_count = 0

        for raw_line in raw.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            content = _content_text(obj)
            if not content:
                continue
            author = _author_name(obj)
            ts = _timestamp(obj)
            if ts and first_ts is None:
                first_ts = ts
            if ts:
                last_ts = ts
            # One line per message keeps message boundaries visible to the
            # chunker — sentence-window then groups consecutive messages.
            lines.append(f"{author}: {content}")
            message_count += 1

        if not lines:
            raise AdapterError(
                f"No usable messages in {source}. "
                "Discord adapter expects one JSON object per line with author + content."
            )

        text = "\n".join(lines)
        channel = p.stem
        metadata = {
            "adapter": self.name,
            "path": str(p.resolve()),
            "channel": channel,
            "messages": message_count,
        }
        if first_ts:
            metadata["first_timestamp"] = first_ts
        if last_ts:
            metadata["last_timestamp"] = last_ts

        yield Document(
            source_id=str(p.resolve()),
            text=text,
            metadata=metadata,
        )
