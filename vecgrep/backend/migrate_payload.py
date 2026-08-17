"""Migrate stored payloads to the carrier layout (v1.1).

Before v1.1 every chunk carried a byte-identical copy of its entire source
document. This walks a collection and:

  1. backfills the derived per-chunk fields (source_length, context_before,
     context_after, line_start, line_end) that the search path now reads,
  2. removes `source_text` from every chunk except the carrier (chunk_index
     0), which keeps the one copy the whole-document readers resolve.

Payload-only: vectors are never read or rewritten, so nothing re-embeds and
search quality cannot move. Re-running is a no-op.

Memory is bounded to one document at a time. The obvious implementation --
group every payload by source_id, then process -- would hold the entire
duplicated corpus resident (~8.3 GB on a 186k-point install), which is the
exact problem being fixed. Instead we take two cheap passes that never
carry source_text, then pull one document at a time.

Take a Qdrant snapshot first. The carrier retains every source's text, so a
correct run loses nothing -- but an interrupted or mistargeted one is not
undoable from inside this process.
"""
from __future__ import annotations

from qdrant_client.http import models as qm

from .service import _chunk_context_fields

# Fields this migration derives and writes.
DERIVED_KEYS = (
    "source_length",
    "context_before",
    "context_after",
    "line_start",
    "line_end",
)

# Points per payload-write request.
_BATCH = 256
# Points per scroll page.
_SCROLL = 1000


def _needs_backfill(payload: dict) -> bool:
    return any(k not in payload for k in DERIVED_KEYS)


def _ids_holding_text(store, collection: str) -> set[str]:
    """Point ids whose payload still has a non-empty source_text.

    Ids only -- requesting the payload here would stream the very bytes we
    are trying to stop storing.
    """
    out: set[str] = set()
    flt = qm.Filter(must_not=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key="source_text"))])
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            offset=offset,
            limit=_SCROLL,
            with_payload=False,
            with_vectors=False,
        )
        out.update(str(p.id) for p in points)
        if offset is None:
            return out


def _chunk_index_by_source(store, collection: str) -> dict[str, list[dict]]:
    """Per-source chunk metadata, excluding source_text.

    ~60 bytes per point, so this stays in the tens of MB even on a corpus
    whose duplicated text runs to gigabytes.
    """
    by_source: dict[str, list[dict]] = {}
    for point_id, payload in store.iter_payloads(
        collection, exclude_payload_fields={"source_text"}
    ):
        sid = payload.get("source_id")
        if not isinstance(sid, str):
            continue
        by_source.setdefault(sid, []).append(
            {
                "id": point_id,
                "index": int(payload.get("chunk_index", 0) or 0),
                "start": int(payload.get("chunk_start", 0) or 0),
                "end": int(payload.get("chunk_end", 0) or 0),
                "backfill": _needs_backfill(payload),
            }
        )
    return by_source


def _fetch_text(store, collection: str, point_id: str) -> str:
    payload = store.get_by_id(collection, point_id) or {}
    return payload.get("source_text", "") or ""


def migrate_collection(store, collection: str, *, dry_run: bool = False) -> dict:
    """Bring one collection to the carrier layout.

    Returns counts: points seen, payloads backfilled, source_text copies
    stripped, and sources skipped because no copy of their text survived.
    """
    stats = {
        "points": 0,
        "backfilled": 0,
        "stripped": 0,
        "sources": 0,
        "orphan_sources": 0,
    }
    holding = _ids_holding_text(store, collection)
    by_source = _chunk_index_by_source(store, collection)

    # Pending payload edits, flushed in batches. One request per point would
    # burn one ephemeral port per point; on a host with a narrow
    # ip_local_port_range that exhausts the range and the run dies mid-way.
    pending_set: list[tuple[str, dict]] = []
    pending_del: list[tuple[str, list[str]]] = []

    def flush(force: bool = False) -> None:
        nonlocal pending_set, pending_del
        if dry_run:
            pending_set, pending_del = [], []
            return
        if not force and len(pending_set) + len(pending_del) < _BATCH:
            return
        store.apply_payload_ops(collection, pending_set, pending_del)
        pending_set, pending_del = [], []

    for source_id, chunks in by_source.items():
        stats["sources"] += 1
        stats["points"] += len(chunks)

        strip = [c["id"] for c in chunks if c["index"] != 0 and c["id"] in holding]
        backfill = [c for c in chunks if c["backfill"]]
        if not strip and not backfill:
            continue

        # Prefer the real carrier; fall back to any surviving copy so a
        # collection whose chunk 0 was pruned still migrates.
        carrier = next((c["id"] for c in chunks if c["index"] == 0 and c["id"] in holding), None)
        if carrier is None:
            carrier = next((c["id"] for c in chunks if c["id"] in holding), None)
        source_text = _fetch_text(store, collection, carrier) if carrier else ""
        if not source_text:
            # Nothing left to derive from. Leave the source exactly as found
            # rather than writing offsets against an empty document.
            stats["orphan_sources"] += 1
            continue

        for c in backfill:
            pending_set.append(
                (c["id"], _chunk_context_fields(source_text, c["start"], c["end"]))
            )
            stats["backfilled"] += 1
            flush()

        for point_id in strip:
            pending_del.append((point_id, ["source_text"]))
            stats["stripped"] += 1
            flush()

    flush(force=True)
    return stats


def migrate_all(store, *, dry_run: bool = False) -> dict[str, dict]:
    """migrate_collection across every vecgrep collection."""
    return {
        name: migrate_collection(store, name, dry_run=dry_run)
        for name in sorted(store.list_collections())
    }
