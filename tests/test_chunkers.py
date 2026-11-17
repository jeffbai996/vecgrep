from __future__ import annotations

import pytest

from vecgrep.backend.ingestion.adapters.base import Document
from vecgrep.backend.ingestion.chunkers import (
    MarkdownSectionChunker,
    SentenceWindowChunker,
)
from vecgrep.backend.ingestion.chunkers.code_symbol import MAX_CHARS, CodeSymbolChunker
from vecgrep.backend.ingestion.chunkers.fixed_token import FixedTokenChunker
from vecgrep.backend.ingestion.chunkers.markdown_section import (
    MarkdownSectionChunker as MarkdownSectionChunker_area1_chunking,
)
from vecgrep.backend.ingestion.chunkers.sentence_window import (
    SentenceWindowChunker as SentenceWindowChunker_area1_chunking,
)
from vecgrep.backend.ingestion.chunkers.turn_window import TurnWindowChunker

# ── test_turn_window_chunker ──

T = "**{who}** · 10:0{i}\n{body}\n\n"


def _transcript(bodies):
    return "".join(T.format(who="a" if i % 2 == 0 else "b", i=i % 10, body=b) for i, b in enumerate(bodies))


def test_groups_whole_turns_with_overlap():
    text = _transcript([f"message number {i} says something." for i in range(7)])
    chunks = TurnWindowChunker(turns=4, stride=3).chunk(text)
    assert [(c.meta or {})["turns"] for c in chunks] == [4, 4]
    for c in chunks:
        assert c.text.startswith("**"), "every chunk begins on a speaker line"
        assert text[c.start:c.end] == c.text, "offsets must address the original text"


def test_oversized_window_is_split_never_emitted_whole():
    big = "word " * 3000   # ~15k chars in one turn
    text = _transcript(["short one.", big, "short two.", "short three."])
    chunks = TurnWindowChunker(turns=4, stride=3, max_chars=4000).chunk(text)
    assert chunks, "must not drop the content"
    assert all(len(c.text) <= 4000 for c in chunks), max(len(c.text) for c in chunks)
    for c in chunks:
        # split pieces come from the sentence windower (whitespace-normalized
        # text), so offsets are approximate there; they must at least be
        # in-range and monotone so context reconstruction stays sane
        assert 0 <= c.start < c.end <= len(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    joined = " ".join(c.text for c in chunks)
    assert "short one." in joined and "short three." in joined


def test_falls_back_to_sentence_windows_without_turn_markers():
    text = "Plain prose. No speakers here. Just sentences. Four of them."
    chunks = TurnWindowChunker().chunk(text)
    assert chunks and all(text[c.start:c.end] == c.text for c in chunks)

# ── test_markdown_section_chunker ──

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

# ── test_fixed_token_chunker ──

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

# ── test_area1_chunking ──

def test_f_chunk_01_sentence_window_drops_short_non_tail_windows():
    """F-CHUNK-01a: SentenceWindowChunker silently drops non-tail windows < min_chars (30)."""
    chunker = SentenceWindowChunker_area1_chunking(window=3, stride=2, min_chars=30)
    # First 3 sentences join to "Fix bug. Ship it. Done now." (27 chars < 30).
    # Fourth sentence makes len(sents) == 4, so at i=0, i + window (3) < len(sents) (4).
    # The i=0 window is skipped and i advances by stride=2 to i=2, permanently dropping
    # sentences 0 ("Fix bug.") and 1 ("Ship it.") from all emitted chunks.
    text = (
        "Fix bug. Ship it. Done now. "
        "Later we wrote a much longer fourth sentence that easily exceeds thirty characters."
    )
    chunks = chunker.chunk(text)
    combined = " ".join(c.text for c in chunks)
    # BUG: "Fix bug." and "Ship it." should be preserved in chunks, but are completely lost.
    assert "Fix bug." in combined
    assert "Ship it." in combined


def test_f_chunk_01_sentence_window_fails_to_split_chinese_prose():
    """F-CHUNK-01b: _SENT_SPLIT requires trailing whitespace after '。！？', failing on standard Chinese."""
    chunker = SentenceWindowChunker_area1_chunking(window=3, stride=2, min_chars=10)
    # 6 distinct Chinese sentences with standard fullwidth punctuation and no spaces.
    text = (
        "这是第一句话，包含重要信息。"
        "这是第二句话，记录了系统状态！"
        "这是第三句话，提出了关键问题？"
        "这是第四句话，给出了修复方案。"
        "这是第五句话，验证了测试结果。"
        "这是第六句话，完成了部署流程。"
    )
    chunks = chunker.chunk(text)
    # With 6 sentences and window=3, stride=2, there should be 3 sliding-window chunks.
    # BUG: _SENT_SPLIT does not split at all without spaces after '。！？', returning 1 chunk.
    assert len(chunks) >= 2


def test_sentence_window_splits_oversized_unpunctuated_text():
    chunker = SentenceWindowChunker_area1_chunking(max_chars=500)
    text = "word " * 1000  # 5000 chars, no sentence-ending punctuation
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_sentence_window_splits_oversized_unbroken_character_stream():
    chunker = SentenceWindowChunker_area1_chunking(max_chars=200)
    text = "x" * 2000  # completely unbroken characters
    chunks = chunker.chunk(text)
    assert len(chunks) == 10
    assert all(len(c.text) <= 200 for c in chunks)
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_f_chunk_02_turn_window_drops_preamble_and_short_tail_turn():
    """F-CHUNK-02: TurnWindowChunker drops preamble before first turn and short tail turns."""
    chunker = TurnWindowChunker(turns=2, stride=2, min_chars=30)
    text = (
        "# Incident 42 Postmortem\n"
        "Critical database failover summary before transcript begins.\n\n"
        "**alice** · 10:00\n"
        "> The primary database node went down at 09:58 UTC due to disk full.\n\n"
        "**bob** · 10:01\n"
        "> I promoted the replica and verified traffic is healthy again.\n\n"
        "**alice** · 10:02\n"
        "> Fixed!\n"
    )
    chunks = chunker.chunk(text)
    combined = "\n".join(c.text for c in chunks)
    # BUG 1: Preamble before first turn marker is silently dropped.
    # BUG 2: Final window (i=2, single turn "**alice** · 10:02\n> Fixed!" = 24 chars < 30) is dropped.
    assert "Incident 42 Postmortem" in combined
    assert "Fixed!" in combined


def test_f_chunk_03_markdown_section_oversized_offset_corruption_and_dedup_loss():
    """F-CHUNK-03: MarkdownSectionChunker prepends header to 2nd+ sub-chunks of oversized sections,
    causing text.find() to return -1, freezing chunk_start at cursor, and causing dedup_near_duplicates
    to drop disjoint sub-chunks as 'overlapping'."""
    from vecgrep.backend.assembly import dedup_near_duplicates
    from vecgrep.backend.service import SearchResult

    chunker = MarkdownSectionChunker_area1_chunking(window=2, stride=2, min_chars=20)
    para1 = "Alpha paragraph with first distinct fact about mercury. " * 2
    para2 = "Bravo paragraph with second distinct fact about venus. " * 2
    para3 = "Charlie paragraph with third distinct fact about mars. " * 2
    text = f"# Planetary Report\n\n{para1.strip()}\n\n{para2.strip()}\n\n{para3.strip()}\n"

    chunks = chunker.chunk(text)
    assert len(chunks) >= 3
    # Sub-chunk 1 and Sub-chunk 2 represent disjoint paragraphs (para2 and para3),
    # so their character spans in the source document must not start at the exact same offset.
    # BUG: text.find("# Planetary Report\n\nBravo...", cursor) returns -1, so chunks[1].start == chunks[2].start!
    assert chunks[1].start != chunks[2].start

    # Furthermore, when both chunks[1] and chunks[2] match a query, dedup_near_duplicates
    # sees identical chunk_start and overlapping chunk_end and drops chunks[2]!
    results = [
        SearchResult(
            score=0.9,
            similarity_pct=85.0,
            chunk=chunks[1].text,
            chunk_start=chunks[1].start,
            chunk_end=chunks[1].end,
            context_before="",
            context_after="",
            source_id="planets.md",
            corpus="docs",
            metadata={},
            chunk_id="c1",
            matched_by=["bm25"],
        ),
        SearchResult(
            score=0.85,
            similarity_pct=80.0,
            chunk=chunks[2].text,
            chunk_start=chunks[2].start,
            chunk_end=chunks[2].end,
            context_before="",
            context_after="",
            source_id="planets.md",
            corpus="docs",
            metadata={},
            chunk_id="c2",
            matched_by=["bm25"],
        ),
    ]
    deduped = dedup_near_duplicates(results)
    assert len(deduped) == 2


def test_f_chunk_04_fixed_token_splits_multibyte_utf8_and_corrupts_offsets():
    """F-CHUNK-04: FixedTokenChunker slices mid-character in cl100k_base, emitting U+FFFD and wrong offsets."""
    # '🎉' is 3 tokens in cl100k_base: [9468, 239, 235].
    # Slicing at tokens=4 cuts right through the 2nd emoji's UTF-8 byte tokens.
    chunker = FixedTokenChunker(tokens=4, overlap=0)
    text = "🎉" * 10
    chunks = chunker.chunk(text)
    for c in chunks:
        # BUG: decoded chunk text contains U+FFFD replacement characters ('\ufffd').
        assert "\ufffd" not in c.text
    # Offsets should advance monotonically across non-overlapping chunks.
    assert chunks[1].start > chunks[0].start


def test_f_chunk_01_sentence_window_preserves_short_multi_sentence_doc():
    """F-CHUNK-01c: When a document has > window sentences whose combined length is still
    < min_chars, SentenceWindowChunker must preserve the leading sentences instead of
    skipping i=0 because i + window < len(sents)."""
    chunker = SentenceWindowChunker_area1_chunking(window=3, stride=2, min_chars=30)
    text = "One. Two. Six. Ten."
    chunks = chunker.chunk(text)
    combined = " ".join(c.text for c in chunks)
    assert "One." in combined
    assert "Two." in combined
    assert "Six." in combined
    assert "Ten." in combined


def test_f_chunk_05_code_symbol_unbounded_when_no_blank_lines():
    """F-CHUNK-05: CodeSymbolChunker._bounded never splits a symbol > MAX_CHARS if it has no blank lines,
    and when splitting oversized blocks it must split on line boundaries rather than mid-line."""
    chunker = CodeSymbolChunker()
    body_lines = [f"    x_{i} = {i} + 1\n" for i in range(600)]
    code = "def huge_function():\n" + "".join(body_lines)
    assert len(code) > MAX_CHARS

    doc = Document(source_id="huge.py", text=code, metadata={})
    chunks = chunker.chunk_doc(doc)
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    # Every non-final chunk must end at a newline boundary (no mid-identifier cuts).
    for c in chunks[:-1]:
        assert c.text.endswith("\n")
    for c in chunks:
        assert code[c.start : c.end] == c.text
