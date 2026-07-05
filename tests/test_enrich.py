"""Chunk enrichment — index-time metadata that powers the speaker:/bot:/has:
filters. Pure function over chunk text; derived at payload-build time so no
adapter or chunker changes are needed (and a re-index picks it up corpus-wide).
"""
from __future__ import annotations

from vecgrep.backend.ingestion.enrich import chunk_enrichment

ARCHIVER_CHUNK = """**alice** · 21:27
> ok can you queue that
**helper_bot [bot]** · 21:28
> Queued. Also here is a table:
> | col | val |
> |-----|-----|
> | a   | 1   |
**carol** · 21:30
> see https://example.com/thing
"""

CODE_CHUNK = """**bot_a [bot]** · 10:00
> here's the fix:
> ```python
> print("hi")
> ```
"""

PLAIN_CHUNK = "just some prose with no structure at all"


def test_speakers_extracted_in_order_no_dupes():
    meta = chunk_enrichment(ARCHIVER_CHUNK)
    assert meta["speakers"] == ["alice", "helper_bot [bot]",
                                "carol"]


def test_bot_flag_from_bot_marker_not_name_list():
    assert chunk_enrichment(ARCHIVER_CHUNK)["has_bot_speaker"] is True
    assert chunk_enrichment(CODE_CHUNK)["has_bot_speaker"] is True
    solo = chunk_enrichment("**alice** · 09:00\n> hi\n")
    assert solo["has_bot_speaker"] is False


def test_content_shapes():
    meta = chunk_enrichment(ARCHIVER_CHUNK)
    assert meta["has_table"] is True
    assert meta["has_link"] is True
    assert meta["has_code"] is False

    code = chunk_enrichment(CODE_CHUNK)
    assert code["has_code"] is True
    assert code["has_table"] is False
    assert code["has_link"] is False


def test_plain_text_all_negative_no_speakers():
    meta = chunk_enrichment(PLAIN_CHUNK)
    assert meta["speakers"] == []
    assert meta["has_bot_speaker"] is False
    assert meta["has_code"] is False and meta["has_table"] is False
    assert meta["has_link"] is False


def test_pipe_in_prose_is_not_a_table():
    """A stray | in prose (or a single pipe row with no separator) must not
    count — a table needs the |---| separator row."""
    meta = chunk_enrichment("a | b in prose\n| lone row |\n")
    assert meta["has_table"] is False
