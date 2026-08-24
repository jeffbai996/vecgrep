"""Document-level catalog for the web corpus explorer.

Qdrant and BM25 index chunks.  Navigation should not: this module collapses
their payloads to one entry per source and gives each source one predictable
home based on metadata already written by ingestion.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


_MONTHS = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_KIND_PLURALS = {
    "analysis": "Analyses",
    "correction": "Corrections",
    "decision": "Decisions",
    "fact": "Facts",
    "insight": "Insights",
    "journal": "Journals",
    "memory": "Memories",
    "reference": "References",
    "scratchpad": "Scratchpads",
    "todo": "To-dos",
}

_CONVERSATION_ADAPTERS = {"chatgpt_export", "claude_export"}

_RECORD_PREFIXES = {
    "deep": ("References", "reference"),
    "journal": ("Journals", "journal"),
    "memory": ("Memories", "memory"),
    "reference": ("References", "reference"),
    "todo": ("To-dos", "todo"),
}


@dataclass(frozen=True)
class ExplorerEntry:
    source_id: str
    name: str
    parent_path: tuple[str, ...]
    kind: str
    doc_timestamp: float | None
    chunk_count: int
    tags: tuple[str, ...]
    metadata: dict

    def summary(self) -> dict:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "parent_path": list(self.parent_path),
            "display_path": [*self.parent_path, self.name],
            "kind": self.kind,
            "doc_timestamp": self.doc_timestamp,
            "chunk_count": self.chunk_count,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ExplorerCatalog:
    scheme: str
    entries: tuple[ExplorerEntry, ...]

    def find(self, source_id: str) -> ExplorerEntry | None:
        return next((entry for entry in self.entries if entry.source_id == source_id), None)


def build_catalog(records: list[dict]) -> ExplorerCatalog:
    """Build one deterministic document catalog from source-level payloads."""
    scheme = _choose_scheme(records)
    common_root = _common_file_root(records) if scheme == "files" else None
    entries = tuple(
        _entry_from_record(record, scheme=scheme, common_root=common_root)
        for record in records
    )
    return ExplorerCatalog(scheme=scheme, entries=entries)


def list_catalog(
    catalog: ExplorerCatalog,
    *,
    corpus: str,
    path: list[str] | None = None,
    query: str = "",
    sort: str = "name",
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """List immediate folders and paginated documents under a logical path."""
    current = tuple(_clean_path(path or []))
    if sort not in {"name", "newest", "oldest"}:
        raise ValueError("sort must be one of: name, newest, oldest")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")

    descendants = [
        entry for entry in catalog.entries if _starts_with(entry.parent_path, current)
    ]
    folder_stats: dict[str, dict] = {}
    for entry in descendants:
        if len(entry.parent_path) <= len(current):
            continue
        child = entry.parent_path[len(current)]
        stats = folder_stats.setdefault(
            child,
            {"document_count": 0, "latest_timestamp": None},
        )
        stats["document_count"] += 1
        if entry.doc_timestamp is not None:
            latest = stats["latest_timestamp"]
            stats["latest_timestamp"] = max(latest, entry.doc_timestamp) if latest else entry.doc_timestamp

    folders = [
        {
            "name": name,
            "path": [*current, name],
            "document_count": stats["document_count"],
            "latest_timestamp": stats["latest_timestamp"],
        }
        for name, stats in sorted(folder_stats.items(), key=lambda item: item[0].casefold())
    ]

    needle = query.strip().casefold()
    if needle:
        documents = [entry for entry in descendants if _matches(entry, needle)]
    else:
        documents = [entry for entry in descendants if entry.parent_path == current]
    documents = _sort_entries(documents, sort)
    total_documents = len(documents)
    page = documents[offset:offset + limit]

    recent: list[ExplorerEntry] = []
    if not current and not needle:
        recent = _sort_entries(list(catalog.entries), "newest")[:12]

    kind_counts = Counter(entry.kind for entry in descendants)
    tag_counts = Counter(tag for entry in descendants for tag in entry.tags)
    return {
        "corpus": corpus,
        "scheme": catalog.scheme,
        "path": list(current),
        "folders": folders,
        "documents": [entry.summary() for entry in page],
        "recent_documents": [entry.summary() for entry in recent],
        "total_documents": total_documents,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total_documents,
        "facets": {
            "kinds": dict(sorted(kind_counts.items())),
            "tags": dict(sorted(tag_counts.items())),
        },
    }


def _choose_scheme(records: list[dict]) -> str:
    if not records:
        return "files"
    total = len(records)

    def dominant(predicate) -> bool:
        return sum(1 for record in records if predicate(record)) / total >= 0.6

    if dominant(lambda record: bool(_metadata(record).get("source_kind"))):
        return "kinds"
    if dominant(lambda record: bool(_metadata(record).get("channel"))):
        return "channels"
    if dominant(
        lambda record: bool(
            _metadata(record).get("session_id") or _metadata(record).get("client")
        )
    ):
        return "sessions"
    if dominant(
        lambda record: str(_metadata(record).get("adapter") or "")
        in _CONVERSATION_ADAPTERS
    ):
        return "conversations"
    if dominant(lambda record: _is_url(str(record.get("source_id") or ""))):
        return "web"
    if dominant(lambda record: _record_prefix(record) in _RECORD_PREFIXES):
        return "records"
    return "files"


def _entry_from_record(
    record: dict,
    *,
    scheme: str,
    common_root: Path | None,
) -> ExplorerEntry:
    source_id = str(record.get("source_id") or "")
    metadata = dict(_metadata(record))
    timestamp = _timestamp(record.get("doc_timestamp"))
    parent_path, name, kind = _display_home(
        source_id,
        metadata,
        timestamp,
        scheme=scheme,
        common_root=common_root,
    )
    return ExplorerEntry(
        source_id=source_id,
        name=name,
        parent_path=tuple(parent_path),
        kind=kind,
        doc_timestamp=timestamp,
        chunk_count=max(1, int(record.get("chunk_count") or 1)),
        tags=tuple(_tags(metadata.get("tags"))),
        metadata=metadata,
    )


def _display_home(
    source_id: str,
    metadata: dict,
    timestamp: float | None,
    *,
    scheme: str,
    common_root: Path | None,
) -> tuple[list[str], str, str]:
    raw_path, fragment = _split_fragment(source_id)
    file_name = Path(raw_path).name or raw_path or "Untitled"
    date_parts = _date_parts(timestamp)

    if scheme == "kinds":
        kind = str(metadata.get("source_kind") or "record").strip().casefold()
        parent = [_plural_kind(kind)]
        return parent, _preferred_title(metadata, file_name), kind

    if scheme == "channels":
        channel = _label(metadata.get("channel"), "Unknown channel")
        return [channel, *date_parts], file_name, "conversation"

    if scheme == "sessions":
        client = _label(metadata.get("client") or metadata.get("model"), "Sessions")
        name = _preferred_title(metadata, file_name)
        return [client, *date_parts], name, "session"

    if scheme == "conversations":
        return date_parts, _preferred_title(metadata, fragment or file_name), "conversation"

    if scheme == "web":
        parsed = urlparse(source_id)
        pieces = [piece for piece in parsed.path.split("/") if piece]
        name = _preferred_title(metadata, pieces[-1] if pieces else parsed.netloc)
        return [parsed.netloc, *pieces[:-1]], name, "web"

    if scheme == "records":
        prefix = _record_prefix({"source_id": source_id})
        folder, kind = _RECORD_PREFIXES.get(prefix, ("Records", "record"))
        return [folder], file_name, kind

    path = Path(raw_path)
    relative = _relative_file_path(path, common_root)
    parts = [part for part in relative.parts if part not in {"", "."}]
    if fragment:
        return [*parts[:-1], Path(file_name).stem], _preferred_title(metadata, fragment), "conversation"
    if not parts:
        return [], _preferred_title(metadata, file_name), "file"
    return parts[:-1], parts[-1], "file"


def _common_file_root(records: list[dict]) -> Path | None:
    parents: list[str] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        if not source_id or _is_url(source_id):
            continue
        raw_path, _ = _split_fragment(source_id)
        path = Path(raw_path)
        if path.is_absolute():
            parents.append(str(path.parent))
    if not parents:
        return None
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        return None


def _relative_file_path(path: Path, common_root: Path | None) -> Path:
    if common_root is not None:
        try:
            return path.relative_to(common_root)
        except ValueError:
            pass
    if path.is_absolute():
        try:
            return path.relative_to(Path.home())
        except ValueError:
            return Path(*path.parts[-3:])
    return path


def _date_parts(timestamp: float | None) -> list[str]:
    if timestamp is None:
        return ["Undated"]
    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return [str(value.year), f"{value.month:02d} {_MONTHS[value.month]}"]


def _plural_kind(kind: str) -> str:
    if kind in _KIND_PLURALS:
        return _KIND_PLURALS[kind]
    label = kind.replace("_", " ").strip().title() or "Records"
    if label.endswith("s"):
        return label
    if label.endswith("y"):
        return f"{label[:-1]}ies"
    return f"{label}s"


def _preferred_title(metadata: dict, fallback: str) -> str:
    for key in ("title", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback or "Untitled"


def _tags(value: object) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if isinstance(value, str):
        return sorted({item.strip() for item in value.split(",") if item.strip()})
    return []


def _matches(entry: ExplorerEntry, needle: str) -> bool:
    haystack = "\n".join(
        [
            entry.name,
            entry.source_id,
            "/".join(entry.parent_path),
            entry.kind,
            *entry.tags,
        ]
    ).casefold()
    return needle in haystack


def _sort_entries(entries: list[ExplorerEntry], sort: str) -> list[ExplorerEntry]:
    if sort == "newest":
        return sorted(
            entries,
            key=lambda entry: (
                entry.doc_timestamp is None,
                -(entry.doc_timestamp or 0.0),
                entry.name.casefold(),
            ),
        )
    if sort == "oldest":
        return sorted(
            entries,
            key=lambda entry: (
                entry.doc_timestamp is None,
                entry.doc_timestamp or 0.0,
                entry.name.casefold(),
            ),
        )
    return sorted(entries, key=lambda entry: entry.name.casefold())


def _clean_path(path: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in path:
        value = str(raw).strip()
        if not value:
            continue
        if value in {".", ".."}:
            raise ValueError("explorer paths cannot contain '.' or '..'")
        cleaned.append(value)
    return cleaned


def _starts_with(value: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return value[:len(prefix)] == prefix


def _metadata(record: dict) -> dict:
    value = record.get("metadata") or {}
    return value if isinstance(value, dict) else {}


def _timestamp(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label(value: object, fallback: str) -> str:
    text = str(value or "").strip().strip("\"'")
    return text or fallback


def _split_fragment(source_id: str) -> tuple[str, str]:
    if "#" not in source_id:
        return source_id, ""
    return tuple(source_id.split("#", 1))  # type: ignore[return-value]


def _record_prefix(record: dict) -> str:
    source_id = str(record.get("source_id") or "")
    raw_path, _ = _split_fragment(source_id)
    stem = Path(raw_path).stem.casefold()
    for separator in ("-", "_", " "):
        if separator in stem:
            stem = stem.split(separator, 1)[0]
    return stem


def _is_url(source_id: str) -> bool:
    return source_id.startswith(("http://", "https://"))
