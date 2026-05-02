from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    source: str = Field(..., description="File path, directory, or URL")
    corpus: str
    chunker: str = "sentence_window"


class IndexResponse(BaseModel):
    docs: int
    chunks: int


class SearchRequest(BaseModel):
    query: str
    corpus: str | None = None
    top_k: int | None = None
    mode: str = "hybrid"
    rerank: bool = False
    rerank_model: str | None = None


class SearchHit(BaseModel):
    similarity_pct: float
    chunk: str
    context_before: str
    context_after: str
    source_id: str
    corpus: str
    metadata: dict
    matched_by: list[str] = []


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class CorpusOut(BaseModel):
    name: str
    embed_backend: str
    embed_model: str
    dim: int
    chunker: str
    doc_count: int
    chunk_count: int
    sources: list[str]
    created_at: float
    updated_at: float


class ConfigOut(BaseModel):
    home: str
    ollama_url: str
    embed_model: str
    openai_configured: bool
    api_host: str
    api_port: int
    default_top_k: int
