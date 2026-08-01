"""Fixed-token chunker: char offsets must address the chunk's own text.

Chunk.start/end are used to slice the ORIGINAL document — for context
previews and source highlighting — so text[start:end] has to round-trip back
to Chunk.text. A cursor that advanced past a chunk's end broke that on
overlapping windows, silently yielding offsets pointing at the wrong span.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.ingestion.chunkers.fixed_token import FixedTokenChunker


DOCS = {
    "prose": "The quick brown fox jumps over the lazy dog. " * 20,
    "code_like": "def f(x):\n    return x+1\n\n" * 25,
    "repeated_phrase": (
        ("alpha beta gamma delta " * 12) + "UNIQUE " + ("alpha beta gamma delta " * 12)
    ),
}


@pytest.mark.parametrize("name", sorted(DOCS))
def test_offsets_round_trip_to_chunk_text(name: str) -> None:
    text = DOCS[name]
    chunks = FixedTokenChunker(tokens=32, overlap=8).chunk(text)

    assert chunks, "chunker produced nothing"
    mismatched = [c.index for c in chunks if text[c.start : c.end] != c.text]
    assert not mismatched, (
        f"{name}: chunks {mismatched} have offsets that don't slice back "
        f"to their own text"
    )


@pytest.mark.parametrize("name", sorted(DOCS))
def test_offsets_are_ordered_and_in_bounds(name: str) -> None:
    text = DOCS[name]
    chunks = FixedTokenChunker(tokens=32, overlap=8).chunk(text)

    for c in chunks:
        assert 0 <= c.start <= c.end <= len(text), f"chunk {c.index} out of bounds"
    # Overlapping windows may share text, but they must still march forward.
    starts = [c.start for c in chunks]
    assert starts == sorted(starts), f"{name}: chunk starts not monotonic: {starts}"


def test_indices_are_sequential() -> None:
    chunks = FixedTokenChunker(tokens=32, overlap=8).chunk(DOCS["prose"])
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_overlap_still_round_trips() -> None:
    text = DOCS["prose"]
    chunks = FixedTokenChunker(tokens=24, overlap=0).chunk(text)
    assert chunks
    assert all(text[c.start : c.end] == c.text for c in chunks)
