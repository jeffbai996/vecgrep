"""BM25 keyword index, per corpus.

Persisted as a pickle next to the Qdrant store. Lives in lockstep with the
vector index — same chunk IDs, same lifecycle (create/upsert/delete-by-source).
Lowercased word-token split, no stemming. Predictable across languages
without dragging in nltk.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

# Match runs of letters/digits, treating underscore and CamelCase as
# token boundaries so identifiers like `sharpe_ratio` and `getUserName`
# are searchable as their constituent words. Pure prose is unaffected.
_TOKEN = re.compile(r"[A-Za-z]+|\d+", re.UNICODE)
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for chunk in _TOKEN.findall(text):
        for piece in _CAMEL_SPLIT.split(chunk):
            if piece:
                out.append(piece.lower())
    return out


@dataclass
class _CorpusIndex:
    ids: list[str] = field(default_factory=list)
    docs: list[list[str]] = field(default_factory=list)
    payloads: list[dict] = field(default_factory=list)
    # source_id -> list of array indices, so delete-by-source is O(n) once.
    by_source: dict[str, list[int]] = field(default_factory=dict)


class BM25Store:
    def __init__(self, root: Path | None) -> None:
        # root=None -> ephemeral (in-memory only).
        self.root = root
        self._cache: dict[str, _CorpusIndex] = {}
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, corpus: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{corpus}.pkl"

    def _load(self, corpus: str) -> _CorpusIndex:
        if corpus in self._cache:
            return self._cache[corpus]
        p = self._path(corpus)
        if p and p.exists():
            try:
                idx = pickle.loads(p.read_bytes())
            except Exception:
                idx = _CorpusIndex()
        else:
            idx = _CorpusIndex()
        self._cache[corpus] = idx
        return idx

    def _persist(self, corpus: str) -> None:
        p = self._path(corpus)
        if p is None:
            return
        idx = self._cache[corpus]
        p.write_bytes(pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL))

    def upsert(self, corpus: str, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        if not ids:
            return
        idx = self._load(corpus)
        for i, (cid, text, payload) in enumerate(zip(ids, texts, payloads)):
            arr_pos = len(idx.ids)
            idx.ids.append(cid)
            idx.docs.append(tokenize(text))
            idx.payloads.append(payload)
            sid = payload.get("source_id", "")
            idx.by_source.setdefault(sid, []).append(arr_pos)
        self._persist(corpus)

    def delete_by_source(self, corpus: str, source_id: str) -> None:
        idx = self._load(corpus)
        positions = set(idx.by_source.pop(source_id, []))
        if not positions:
            return
        # Rebuild parallel arrays without those positions, then re-derive
        # the by_source map. Simpler than splice math; BM25 needs full
        # rebuild on every change anyway.
        new = _CorpusIndex()
        for i, cid in enumerate(idx.ids):
            if i in positions:
                continue
            new_pos = len(new.ids)
            new.ids.append(cid)
            new.docs.append(idx.docs[i])
            new.payloads.append(idx.payloads[i])
            sid = idx.payloads[i].get("source_id", "")
            new.by_source.setdefault(sid, []).append(new_pos)
        self._cache[corpus] = new
        self._persist(corpus)

    def drop(self, corpus: str) -> None:
        self._cache.pop(corpus, None)
        p = self._path(corpus)
        if p and p.exists():
            p.unlink()

    def search(self, corpus: str, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        idx = self._load(corpus)
        if not idx.docs:
            return []
        bm25 = BM25Okapi(idx.docs)
        scores = bm25.get_scores(tokenize(query))
        # argsort desc; take top_k positives only.
        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0),
            reverse=True,
        )[:top_k]
        return [(idx.ids[i], float(s), idx.payloads[i]) for s, i in ranked]
