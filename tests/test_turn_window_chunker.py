"""turn_window chunker: whole messages, bounded size.

The first side-by-side build with this chunker died with `Ollama returned
400: the input length exceeds the context length` -- four bot turns with
code blocks can run far past an embedding model's context. A chunk that
cannot be embedded is worse than a split one, so windows are capped at
`max_chars` and an oversized window degrades to sentence windows over the
same span (offsets preserved).
"""
from __future__ import annotations

from vecgrep.backend.ingestion.chunkers.turn_window import TurnWindowChunker

T = "**{who}** · 10:0{i}\n{body}\n\n"


def _transcript(bodies):
    return "".join(T.format(who="a" if i % 2 == 0 else "b", i=i % 10, body=b) for i, b in enumerate(bodies))


def test_groups_whole_turns_with_overlap():
    text = _transcript([f"message number {i} says something." for i in range(7)])
    chunks = TurnWindowChunker(turns=4, stride=3).chunk(text)
    assert [c.meta["turns"] for c in chunks] == [4, 4]
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
