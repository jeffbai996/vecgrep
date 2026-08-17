"""Timeline reconstruction: contiguous, chronological slices around an incident.

Ranked chunks answer "which passages match?" — a "what happened?" question
needs the EVENT SEQUENCE: who said what, in order, with timestamps. This
module turns the source files behind a search's anchor hits into exactly
that:

    1. the caller's search picks anchor chunks (relevance, dedup, filters
       all apply — the retrieval core is untouched);
    2. anchors are grouped by source file, the best few files are kept;
    3. ONE contiguous slice per file is cut from the source text, spanning
       all of that file's anchors plus padding on each side;
    4. transcript slices are parsed into (speaker, time, text) events using
       the archiver format (`**speaker** · HH:MM` headers + `> ` bodies);
       non-transcript sources keep the raw slice and no events — degrading
       honestly instead of inventing structure.

Within a transcript file, position order IS chronological order, so events
come out time-sorted without any re-sorting heuristics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Archiver speaker header: **name** · HH:MM  (name may contain spaces,
# CJK, or a "[bot]" suffix — anything but '*' and newline).
_EVENT_HEADER = re.compile(
    r"\*\*(?P<speaker>[^*\n]+)\*\*\s*·\s*(?P<time>\d{1,2}:\d{2})"
)

# How many chars of padding to keep on each side of a file's anchor span.
SLICE_PADDING = 1200
# How many source files a timeline may span (the best-scoring ones win).
MAX_GROUPS = 4
# How many anchor hits the underlying search pulls.
ANCHOR_TOP_K = 10


@dataclass(frozen=True)
class Event:
    speaker: str
    time: str
    text: str


def parse_events(slice_text: str) -> list[Event]:
    """Parse a transcript slice into ordered events.

    Text before the first speaker header has no attributable speaker (the
    slice may start mid-message) and is dropped rather than misattributed.
    Returns [] for text with no speaker headers at all — callers treat that
    as "not a transcript" and fall back to the raw slice.
    """
    matches = list(_EVENT_HEADER.finditer(slice_text))
    events: list[Event] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(slice_text)
        body = slice_text[m.end() : end]
        # Strip quote markers + collapse whitespace; the marker is rendering,
        # not content.
        text = " ".join(body.replace(">", " ").split())
        if not text:
            continue
        events.append(
            Event(speaker=m.group("speaker").strip(), time=m.group("time"), text=text)
        )
    return events


def build_timeline(
    anchors: list,
    payload_for: "callable",
    max_groups: int = MAX_GROUPS,
    padding: int = SLICE_PADDING,
    source_text_for: "callable | None" = None,
) -> list[dict]:
    """Assemble timeline groups from anchor SearchResults.

    `payload_for(result)` must return the stored payload dict for a result's
    chunk, or None. `source_text_for(corpus, source_id)` returns that
    source's full document; slices span a file's anchors plus `padding` on
    each side, which routinely reaches past the bounded context stored with
    any single chunk. When it is not supplied, the payload's own
    `source_text` is used — the pre-v1.1 layout, where every chunk carried
    the document. Groups come back ordered by document date (undated last),
    events chronological within each group.
    """
    by_file: dict[tuple[str, str], list] = {}
    for r in anchors:
        by_file.setdefault((r.corpus, r.source_id), []).append(r)

    # Keep the best few files, ranked by their strongest anchor.
    ranked = sorted(
        by_file.items(), key=lambda kv: max(a.score for a in kv[1]), reverse=True
    )[:max_groups]

    groups: list[dict] = []
    for (corpus, source_id), hits in ranked:
        payload = None
        for h in hits:
            payload = payload_for(h)
            if payload:
                break
        if not payload:
            continue
        source_text = ""
        if source_text_for is not None:
            source_text = source_text_for(corpus, source_id) or ""
        if not source_text:
            source_text = payload.get("source_text", "") or ""
        if not source_text:
            continue
        lo = max(0, min(h.chunk_start for h in hits) - padding)
        hi = min(len(source_text), max(h.chunk_end for h in hits) + padding)
        slice_text = source_text[lo:hi]
        events = parse_events(slice_text)
        groups.append(
            {
                "corpus": corpus,
                "source_id": source_id,
                "doc_timestamp": hits[0].doc_timestamp,
                "slice_start": lo,
                "slice_end": hi,
                "events": [
                    {"speaker": e.speaker, "time": e.time, "text": e.text}
                    for e in events
                ],
                # Raw slice only when event parsing found nothing — the
                # non-transcript degrade path. Transcript groups don't repeat
                # the text they already carry as events.
                "slice_text": "" if events else slice_text,
            }
        )

    # Oldest → newest reads as a narrative; undated files go last.
    groups.sort(
        key=lambda g: (g["doc_timestamp"] is None, g["doc_timestamp"] or 0.0)
    )
    return groups
