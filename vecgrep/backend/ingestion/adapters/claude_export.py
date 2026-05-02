"""Claude export adapter.

Anthropic's data export ships as `conversations.json`: a top-level array
of conversation objects. Each conversation has `name`, `created_at`,
`chat_messages` (or `messages`), each with `sender` ("human" or "assistant")
and `content` (string or list of content blocks).

We emit ONE Document per conversation so search returns whole-conversation
hits, and the chunker slices within. Conversation title goes in metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Adapter, AdapterError, Document
from .registry import register_adapter


def _looks_like_claude_export(obj) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    if not isinstance(first, dict):
        return False
    has_messages = any(k in first for k in ("chat_messages", "messages"))
    has_meta = any(k in first for k in ("uuid", "name", "created_at"))
    return has_messages and has_meta


def _block_text(block) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # Claude content blocks: {"type": "text", "text": "..."}
        if "text" in block:
            return str(block["text"])
        if "content" in block:
            return _content_text(block["content"])
    return ""


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p for p in (_block_text(b) for b in content) if p)
    return ""


@register_adapter
class ClaudeExportAdapter(Adapter):
    name = "claude_export"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        if not p.is_file() or p.suffix.lower() != ".json":
            return False
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return False
        return _looks_like_claude_export(obj)

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as e:
            raise AdapterError(f"Could not parse {source}: {e}") from e

        for conv in data:
            if not isinstance(conv, dict):
                continue
            messages = conv.get("chat_messages") or conv.get("messages") or []
            lines: list[str] = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                sender = msg.get("sender") or msg.get("role") or "?"
                content = (
                    _content_text(msg.get("content"))
                    or msg.get("text", "")
                )
                if content:
                    lines.append(f"{sender}: {content}")
            if not lines:
                continue

            title = conv.get("name") or conv.get("title") or conv.get("uuid") or "untitled"
            cid = conv.get("uuid") or conv.get("id") or title
            yield Document(
                source_id=f"{p.resolve()}#{cid}",
                text="\n\n".join(lines),
                metadata={
                    "adapter": self.name,
                    "path": str(p.resolve()),
                    "title": title,
                    "created_at": conv.get("created_at") or "",
                    "message_count": len(lines),
                },
            )
