"""Markdown-section chunker: windows must not blend across a heading."""
from __future__ import annotations

from vecgrep.backend.ingestion.chunkers import (
    MarkdownSectionChunker,
    SentenceWindowChunker,
)


def test_no_chunk_spans_a_heading():
    doc = (
        "# Alpha\n\nalphaword one. alphaword two. alphaword three.\n\n"
        "# Beta\n\nbetaword one. betaword two. betaword three.\n"
    )
    chunks = MarkdownSectionChunker().chunk(doc)
    for c in chunks:
        blends = "alphaword" in c.text and "betaword" in c.text
        assert not blends, f"chunk blends two sections: {c.text!r}"
    # both sections still produced chunks
    assert any("alphaword" in c.text for c in chunks)
    assert any("betaword" in c.text for c in chunks)


def test_char_offsets_stay_in_bounds():
    doc = "# H\n\nfirst sentence here. second sentence here. third one here.\n"
    chunks = MarkdownSectionChunker().chunk(doc)
    assert chunks
    for c in chunks:
        assert 0 <= c.start < c.end <= len(doc)


def test_no_headings_falls_back_to_plain_sentence_window():
    # With no markdown structure, it must behave exactly like the default chunker.
    doc = "one here. two here. three here. four here. five here."
    md = MarkdownSectionChunker().chunk(doc)
    sw = SentenceWindowChunker().chunk(doc)
    assert [c.text for c in md] == [c.text for c in sw]
    assert [(c.start, c.end) for c in md] == [(c.start, c.end) for c in sw]


def test_heading_travels_with_its_section():
    doc = "# Title One\n\nbody sentence alpha. body sentence beta.\n"
    chunks = MarkdownSectionChunker().chunk(doc)
    # the first chunk of the section carries the heading text for topical context
    assert any("Title One" in c.text for c in chunks)


def test_registered_in_service_chunkers():
    from vecgrep.backend.service import CHUNKERS
    assert "markdown_section" in CHUNKERS
    assert CHUNKERS["markdown_section"] is MarkdownSectionChunker
