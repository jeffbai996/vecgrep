"""Fresh doc ids read as entry ids, not corpus names: a short base36 suffix."""
from __future__ import annotations

import re
from pathlib import Path

from vecgrep.backend.write import proposal as P


def test_doc_id_suffix_is_short_base36(tmp_path: Path) -> None:
    doc_id = P.next_doc_id(tmp_path, "external")
    prefix, suffix = doc_id.split("-", 1)
    assert prefix == "external"
    assert re.fullmatch(r"[0-9a-z]{8,13}", suffix), doc_id
    assert P._ID_RE.match(doc_id)


def test_doc_ids_stay_unique_on_disk_collision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(P.time, "time_ns", lambda: 1_793_492_323_840_462_021)
    first = P.next_doc_id(tmp_path, "external")
    (tmp_path / f"{first}.md").write_text("x", encoding="utf-8")
    second = P.next_doc_id(tmp_path, "external")
    assert second != first
    assert P._ID_RE.match(second)


def test_doc_ids_order_by_time(tmp_path: Path, monkeypatch) -> None:
    clock = iter([1_793_492_323_840_462_021, 1_793_492_323_840_462_021 + 10**9])
    monkeypatch.setattr(P.time, "time_ns", lambda: next(clock))
    earlier = P.next_doc_id(tmp_path, "external")
    later = P.next_doc_id(tmp_path, "external")
    assert len(earlier) == len(later)
    assert earlier < later
