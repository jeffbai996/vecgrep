from __future__ import annotations

import hmac
from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import get_settings
from ..embed import EmbedBackendError
from ..ingestion.adapters import AdapterError
from ..rerank import RerankerError
from ..service import VecgrepService
from ..store import CorpusError
from .schemas import (
    Calibration,
    ChunkWindow,
    ConfigOut,
    CorpusOut,
    DecayRequest,
    IndexRequest,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchStub,
    TimelineGroup,
    TimelineRequest,
    WeightRequest,
)


# Server-side caps on chunk-window expansion. Keeps a leaked unauthed endpoint
# from being abused to dump entire huge sources via repeated calls.
_DEFAULT_CHUNK_WINDOW = 2000
_MAX_CHUNK_WINDOW = 20000

def require_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token gate. No-op when settings.api_token is unset."""
    expected = get_settings().api_token
    if not expected:
        return
    # Strip the CONFIGURED token too, not just the client's. A trailing newline
    # from an env or file read (extremely common) otherwise mismatches every
    # valid request — locking you out of your own server with a 403 that's hard
    # to diagnose. compare_digest keeps the check constant-time.
    expected = expected.strip()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization[len("Bearer ") :].strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


# Health is intentionally public — load balancers and watchdogs need it
# without credentials. Everything else gets the gate via Depends.
public_router = APIRouter(prefix="/api")
router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


_SERVICE: VecgrepService | None = None
def _service() -> VecgrepService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = VecgrepService()
    return _SERVICE


@public_router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    s = get_settings()
    return ConfigOut(
        home=str(s.home),
        ollama_url=s.ollama_url,
        embed_model=s.embed_model,
        openai_configured=bool(s.openai_api_key),
        api_host=s.api_host,
        api_port=s.api_port,
        default_top_k=s.default_top_k,
    )


@router.get("/corpora", response_model=list[CorpusOut])
def list_corpora() -> list[CorpusOut]:
    svc = _service()
    return [CorpusOut(**asdict(c)) for c in svc.list_corpora()]


@router.delete("/corpora/{name}")
def delete_corpus(name: str) -> dict:
    svc = _service()
    try:
        svc.delete_corpus(name)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": name}


@router.get("/corpora/{name}/filters")
def corpus_filters(name: str) -> dict:
    """Describe the filter expressions available for this corpus."""
    svc = _service()
    try:
        return svc.filterable_fields(name)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/corpora/{name}/decay", response_model=CorpusOut)
def set_decay(name: str, req: DecayRequest) -> CorpusOut:
    svc = _service()
    try:
        corpus = svc.set_decay(name, req.half_life_days)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return CorpusOut(**asdict(corpus))


@router.post("/corpora/{name}/weight", response_model=CorpusOut)
def set_rank_weight(name: str, req: WeightRequest) -> CorpusOut:
    svc = _service()
    try:
        corpus = svc.set_rank_weight(name, req.weight)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return CorpusOut(**asdict(corpus))


@router.delete("/corpora/{name}/source/{id:path}")
def delete_source(name: str, id: str) -> dict:
    svc = _service()
    try:
        svc.delete_source(name, id)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": id}


@router.post("/index", response_model=IndexResponse)
def index(req: IndexRequest) -> IndexResponse:
    svc = _service()
    try:
        docs, chunks, skipped = svc.index(
            req.source, req.corpus, req.chunker, force=req.force, include=req.include
        )
    except AdapterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmbedBackendError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except CorpusError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IndexResponse(docs=docs, chunks=chunks, skipped=skipped)


def _hit_out(r) -> SearchHit:
    return SearchHit(
        similarity_pct=r.similarity_pct,
        chunk=r.chunk,
        context_before=r.context_before,
        context_after=r.context_after,
        source_id=r.source_id,
        corpus=r.corpus,
        metadata=r.metadata,
        chunk_id=r.chunk_id,
        matched_by=r.matched_by,
        doc_timestamp=r.doc_timestamp,
        line_start=r.line_start,
        line_end=r.line_end,
        anchor=r.anchor,
        relevance_pct=r.relevance_pct,
        relevance_label=r.relevance_label,
        explain=r.explain or {},
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    svc = _service()
    if req.mode not in ("hybrid", "vector", "bm25"):
        raise HTTPException(status_code=400, detail=f"Unknown search mode: {req.mode}")
    common = dict(
        mode=req.mode,
        rerank=req.rerank,
        rerank_model=req.rerank_model,
        filters=req.filters or None,
        explain=req.explain,
    )
    try:
        if req.budget:
            # Breadth mode: full head + one-line stub tail. Expand a stub via
            # GET /api/chunk/{corpus}/{chunk_id}.
            full, stubs = svc.search_budgeted(
                req.query,
                req.corpus,
                full_k=req.full_k,
                token_ceiling=req.token_ceiling,
                **common,
            )
            return SearchResponse(
                hits=[_hit_out(r) for r in full],
                stubs=[
                    SearchStub(
                        chunk_id=s.chunk_id,
                        corpus=s.corpus,
                        source_id=s.source_id,
                        doc_timestamp=s.doc_timestamp,
                        snippet=s.snippet,
                        similarity_pct=s.similarity_pct,
                    )
                    for s in stubs
                ],
                calibration=Calibration(**svc.calibration(req.corpus)),
            )
        results = svc.search(req.query, req.corpus, req.top_k, **common)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmbedBackendError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RerankerError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(
        hits=[_hit_out(r) for r in results],
        calibration=Calibration(**svc.calibration(req.corpus)),
    )


@router.post("/timeline", response_model=list[TimelineGroup])
def timeline(req: TimelineRequest) -> list[TimelineGroup]:
    """'What happened?' mode: contiguous chronological slices grouped by
    source file, transcript slices parsed into (speaker, time, text) events."""
    svc = _service()
    if req.mode not in ("hybrid", "vector", "bm25"):
        raise HTTPException(status_code=400, detail=f"Unknown search mode: {req.mode}")
    try:
        groups = svc.timeline(
            req.query,
            req.corpus,
            top_k=req.top_k,
            max_groups=req.max_groups,
            padding=req.padding,
            mode=req.mode,
            filters=req.filters or None,
        )
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmbedBackendError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [TimelineGroup(**g) for g in groups]


@router.get("/oauth/status")
def oauth_status() -> dict:
    """State for the squad inventory's OAuth panel. Token VALUES never leave
    the store — clients, counts, issuer only."""
    from ..config import get_settings
    s = get_settings()
    issuer = getattr(s, "oauth_issuer_url", None)
    enabled = bool(getattr(s, "oauth_enabled", False) and issuer)
    if not enabled:
        return {"enabled": False}
    from ...mcp.server import _shared_provider
    prov = _shared_provider()
    return {
        "enabled": True,
        "issuer": issuer,
        "scopes": list(prov.valid_scopes),
        "clients": [
            {
                "client_id": c.client_id,
                "name": getattr(c, "client_name", None) or c.client_id,
                "redirect_uris": [str(u) for u in (c.redirect_uris or [])],
            }
            for c in prov._clients.values()
        ],
        "tokens": prov.store.counts(),
    }


@router.post("/oauth/revoke_client")
def oauth_revoke_client(req: dict) -> dict:
    """Inventory revoke button: kill every token a client holds."""
    client_id = (req or {}).get("client_id") or ""
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id required")
    from ...mcp.server import _shared_provider
    return {"revoked": _shared_provider().store.revoke_client(client_id)}


@router.get("/related/{corpus}/{chunk_id}")
def related(corpus: str, chunk_id: str, top_k: int = 8) -> dict:
    """Nearest neighbours of an existing chunk (query-by-example)."""
    svc = _service()
    try:
        results = svc.related(chunk_id, corpus, top_k=top_k)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"hits": [_hit_out(r).model_dump() for r in results]}


@router.post("/compare")
def compare(req: dict) -> dict:
    """Temporal diff: {query, corpus, a_after?, a_before?, b_after?,
    b_before?, top_k?}. Same window grammar as after:/before: filters."""
    try:
        out = _service().compare(
            req["query"], req["corpus"],
            a_after=req.get("a_after"), a_before=req.get("a_before"),
            b_after=req.get("b_after"), b_before=req.get("b_before"),
            top_k=int(req.get("top_k") or 8),
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"missing field: {e}")
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    for side in ("a", "b"):
        out[side]["results"] = [
            _hit_out(r).model_dump() for r in out[side]["results"]
        ]
    return out


@router.get("/stats/{corpus}")
def corpus_stats(corpus: str) -> dict:
    """Corpus health snapshot (counts, date coverage, gaps, source sizes)."""
    try:
        return _service().corpus_stats(corpus)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/summarize/{corpus}")
def summarize_corpus(corpus: str, after: str = "", before: str = "",
                     sample: int = 40) -> dict:
    """Rollup: speakers, span, top sources, sampled chunks (explicit flag)."""
    try:
        return _service().summarize_corpus(
            corpus, after=after or None, before=before or None, sample=sample)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/chunk/{corpus}/{chunk_id}", response_model=ChunkWindow)
def get_chunk(corpus: str, chunk_id: str, window: str = str(_DEFAULT_CHUNK_WINDOW)) -> ChunkWindow:
    """Fetch an expanded context window around a chunk.

    `window` is char count per side, or "full" for the whole source. Capped
    at `_MAX_CHUNK_WINDOW` per side to bound response size.
    """
    if window == "full":
        win = -1
    else:
        try:
            win = int(window)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="window must be an int or 'full'")
        if win < 0:
            raise HTTPException(status_code=400, detail="window must be >= 0")
        win = min(win, _MAX_CHUNK_WINDOW)
    svc = _service()
    try:
        data = svc.get_chunk_window(corpus, chunk_id, win)
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if data is None:
        raise HTTPException(status_code=404, detail=f"chunk {chunk_id} not found in {corpus}")
    return ChunkWindow(**data)
