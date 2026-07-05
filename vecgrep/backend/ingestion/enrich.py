"""Chunk enrichment — per-chunk metadata derived from the chunk's own text.

Runs at payload-build time (index path in service.py), so it needs no adapter
or chunker cooperation and a plain re-index upgrades an existing corpus.
Powers the `speaker:`, `bot:` and `has:` filters.

Speaker extraction targets the archiver transcript format
(`**name** · HH:MM` headers — see timeline._EVENT_HEADER, reused here so the
two parsers can't drift). Non-transcript sources simply yield no speakers,
which is correct: the filters are hard constraints, so a `speaker:` query
against a code corpus returns nothing rather than everything.

Bot detection rides the archiver's own `[bot]` suffix in speaker names — no
hardcoded bot-name list to rot.
"""
from __future__ import annotations

from ..timeline import _EVENT_HEADER

_CODE_FENCE = "```"


def _has_table(text: str) -> bool:
    """A markdown table = a pipe row followed (within the chunk) by the
    |---|-style separator row. A stray `|` in prose doesn't count."""
    saw_pipe_row = False
    for line in text.splitlines():
        stripped = line.strip().lstrip("> ").strip()
        if not stripped.startswith("|"):
            continue
        body = stripped.strip("|").replace(" ", "")
        if body and set(body) <= set("-:|"):
            if saw_pipe_row:
                return True
        else:
            saw_pipe_row = True
    return False


def chunk_enrichment(text: str) -> dict:
    """Metadata for one chunk: speakers (document order, deduped), bot flag,
    and cheap content-shape booleans."""
    speakers: list[str] = []
    seen: set[str] = set()
    for m in _EVENT_HEADER.finditer(text):
        name = m.group("speaker").strip()
        if name and name not in seen:
            seen.add(name)
            speakers.append(name)
    return {
        "speakers": speakers,
        "has_bot_speaker": any("[bot]" in s for s in speakers),
        "has_code": _CODE_FENCE in text,
        "has_table": _has_table(text),
        "has_link": ("http://" in text) or ("https://" in text),
    }
