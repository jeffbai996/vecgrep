"""ChatGPT export adapter.

OpenAI's data export ships `conversations.json`: a top-level array of
conversations. Each conversation has `title`, `create_time`, and a `mapping`
field — a dict keyed by node UUID forming a tree of messages. We linearize
the tree following each node's `parent` chain to the root, then traverse
top-down to produce a flat transcript.

One Document per conversation, like the Claude adapter. Title in metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Adapter, Document, load_json_file, try_load_json_file
from .registry import register_adapter


def _looks_like_chatgpt_export(obj) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    if not isinstance(first, dict):
        return False
    return "mapping" in first and ("title" in first or "create_time" in first)


def _linearize(mapping: dict) -> list[dict]:
    """Walk the mapping tree top-down, returning messages in conversation order.

    ChatGPT exports represent conversations as a node graph: each node has
    `parent` and `children`. The root is the node with no parent; we traverse
    breadth-first along children, taking the first child at each branch (the
    'main' thread — branches happen on edits/regenerations).
    """
    if not mapping:
        return []
    # Find root
    root_id = None
    for nid, node in mapping.items():
        if not node.get("parent"):
            root_id = nid
            break
    if root_id is None:
        return []
    out: list[dict] = []
    cursor = root_id
    while cursor:
        node = mapping.get(cursor) or {}
        msg = node.get("message")
        if msg:
            out.append(msg)
        children = node.get("children") or []
        cursor = children[0] if children else None
    return out


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # ChatGPT content blocks: {"content_type": "text", "parts": ["..."]}
        parts = content.get("parts") or []
        return "\n".join(str(p) for p in parts if isinstance(p, (str, int, float)))
    return ""


@register_adapter
class ChatGPTExportAdapter(Adapter):
    name = "chatgpt_export"

    @classmethod
    def matches(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return False
        p = Path(source)
        if not p.is_file() or p.suffix.lower() != ".json":
            return False
        obj = try_load_json_file(p)
        if obj is None:
            return False
        return _looks_like_chatgpt_export(obj)

    def load(self, source: str) -> Iterator[Document]:
        p = Path(source)
        data = load_json_file(p, source)

        for conv in data:
            if not isinstance(conv, dict):
                continue
            messages = _linearize(conv.get("mapping") or {})
            lines: list[str] = []
            for msg in messages:
                role = (msg.get("author") or {}).get("role") or "?"
                content = _content_to_text(msg.get("content"))
                if content.strip():
                    lines.append(f"{role}: {content}")
            if not lines:
                continue

            title = conv.get("title") or conv.get("id") or "untitled"
            cid = conv.get("id") or title
            yield Document(
                source_id=f"{p.resolve()}#{cid}",
                text="\n\n".join(lines),
                metadata={
                    "adapter": self.name,
                    "path": str(p.resolve()),
                    "title": title,
                    "create_time": conv.get("create_time") or "",
                    "message_count": len(lines),
                },
            )
