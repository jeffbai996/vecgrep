from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    source: str = Field(..., description="File path, directory, or URL")
    corpus: str
    chunker: str = "sentence_window"
    force: bool = False
    include: str | None = None  # filename glob for directory indexing, e.g. "*.md"


class IndexResponse(BaseModel):
    docs: int
    chunks: int
    skipped: int = 0


class SearchRequest(BaseModel):
    query: str
    corpus: str | None = None
    top_k: int | None = None
    mode: str = "hybrid"
    rerank: bool = False  # opt-in: benchmark showed +127ms tax + quality wash on default-on
    rerank_model: str | None = None
    filters: list[str] = []
    explain: bool = False


class SearchHit(BaseModel):
    similarity_pct: float
    chunk: str
    context_before: str
    context_after: str
    source_id: str
    corpus: str
    metadata: dict
    # Deterministic chunk id — pass to GET /api/chunk/{corpus}/{chunk_id} to
    # fetch a wider context window.
    chunk_id: str = ""
    matched_by: list[str] = []
    explain: dict = {}


class ChunkWindow(BaseModel):
    """Expanded context around a chunk. Returned by GET /api/chunk."""
    corpus: str
    chunk_id: str
    source_id: str
    chunk_start: int
    chunk_end: int
    before: str
    chunk: str
    after: str
    source_length: int
    # Echoes the requested window (in chars) or -1 for full source.
    window: int


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
    decay_half_life_days: float | None = None


class DecayRequest(BaseModel):
    # None / omitted disables decay; a positive number sets the half-life.
    half_life_days: float | None = None


class ConfigOut(BaseModel):
    home: str
    ollama_url: str
    embed_model: str
    openai_configured: bool
    api_host: str
    api_port: int
    default_top_k: int
