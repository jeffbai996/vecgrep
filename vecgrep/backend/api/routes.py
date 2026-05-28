from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException

from ..config import get_settings
from ..embed import EmbedBackendError
from ..ingestion.adapters import AdapterError
from ..rerank import RerankerError
from ..service import VecgrepService
from ..store import CorpusError
from .schemas import (
    ChunkWindow,
    ConfigOut,
    CorpusOut,
    DecayRequest,
    IndexRequest,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
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
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization[len("Bearer ") :].strip() != expected:
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
        docs, chunks, skipped = svc.index(req.source, req.corpus, req.chunker, force=req.force)
    except AdapterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmbedBackendError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except CorpusError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IndexResponse(docs=docs, chunks=chunks, skipped=skipped)


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    svc = _service()
    if req.mode not in ("hybrid", "vector", "bm25"):
        raise HTTPException(status_code=400, detail=f"Unknown search mode: {req.mode}")
    try:
        results = svc.search(
            req.query,
            req.corpus,
            req.top_k,
            mode=req.mode,
            rerank=req.rerank,
            rerank_model=req.rerank_model,
            filters=req.filters or None,
            explain=req.explain,
        )
    except CorpusError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmbedBackendError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RerankerError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(
        hits=[
            SearchHit(
                similarity_pct=r.similarity_pct,
                chunk=r.chunk,
                context_before=r.context_before,
                context_after=r.context_after,
                source_id=r.source_id,
                corpus=r.corpus,
                metadata=r.metadata,
                chunk_id=r.chunk_id,
                matched_by=r.matched_by,
                explain=r.explain or {},
            )
            for r in results
        ]
    )


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
