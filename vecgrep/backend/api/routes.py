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
    ConfigOut,
    CorpusOut,
    IndexRequest,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

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


def _service() -> VecgrepService:
    return VecgrepService()


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
                matched_by=r.matched_by,
                explain=r.explain or {},
            )
            for r in results
        ]
    )
