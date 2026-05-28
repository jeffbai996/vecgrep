"""Document-date extraction tests.

Covers the real formats in production: squad-store memory files (`Saved: <iso>`),
transcript files (frontmatter `date:` + `YYYY-MM-DD` filename), and the graceful
fallbacks (filename date, mtime, None).
"""
from __future__ import annotations

from datetime import datetime, timezone

from vecgrep.backend.ingestion.adapters.markdown import MarkdownAdapter
from vecgrep.backend.ingestion.dates import extract_timestamp


def _epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def test_saved_line_iso_with_microseconds_and_tz() -> None:
    """squad-store memory format: 'Saved: 2026-05-18T06:45:21.178768+00:00'."""
    text = "# Memory\n\nName: foo\nSaved: 2026-05-18T06:45:21.178768+00:00\n\nbody"
    ts = extract_timestamp(text)
    assert ts == _epoch("2026-05-18T06:45:21.178768+00:00")


def test_frontmatter_date_only() -> None:
    """transcript frontmatter: 'date: 2026-05-27'."""
    text = "---\nchannel: cl-2\ndate: 2026-05-27\n---\n\nalice: hi"
    ts = extract_timestamp(text)
    assert ts == _epoch("2026-05-27")


def test_labeled_line_wins_over_filename() -> None:
    text = "Saved: 2026-01-02T00:00:00+00:00\nblah"
    ts = extract_timestamp(text, path="/x/2099-12-31.md")
    assert ts == _epoch("2026-01-02T00:00:00+00:00")


def test_filename_date_fallback_when_no_labeled_line() -> None:
    ts = extract_timestamp("just prose, no dates here", path="/logs/2026-05-27.md")
    assert ts == _epoch("2026-05-27")


def test_mtime_fallback(tmp_path) -> None:
    p = tmp_path / "undated.md"
    p.write_text("no date anywhere")
    ts = extract_timestamp("no date anywhere", path=str(p))
    assert ts == p.stat().st_mtime


def test_none_when_no_text_date_and_no_path() -> None:
    assert extract_timestamp("no date, no path") is None


def test_date_deep_in_body_does_not_win() -> None:
    """A date mentioned far down in prose must not be treated as the doc date."""
    body = "\n".join(["line %d" % i for i in range(40)] + ["Saved: 2020-01-01"])
    assert extract_timestamp(body) is None  # past the 15-line head window


def test_never_raises_on_garbage() -> None:
    assert extract_timestamp("Saved: not-a-date\ndate: also-bad") is None


def test_markdown_adapter_populates_timestamp(tmp_path) -> None:
    p = tmp_path / "memory-1.md"
    p.write_text("Name: x\nSaved: 2026-05-18T06:45:21+00:00\n\nbody text here")
    docs = list(MarkdownAdapter().load(str(p)))
    assert len(docs) == 1
    assert docs[0].timestamp == _epoch("2026-05-18T06:45:21+00:00")
