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
    # Breadth mode: top full_k hits keep context; the rest degrade to
    # one-line stubs (see SearchStub) until ~token_ceiling estimated tokens.
    budget: bool = False
    full_k: int = 8
    token_ceiling: int = 4000


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
    doc_timestamp: float | None = None
    explain: dict = {}


class SearchStub(BaseModel):
    """A one-line result reference (budget mode's tail tier): no context
    windows. Expand via GET /api/chunk/{corpus}/{chunk_id}."""
    chunk_id: str
    corpus: str
    source_id: str
    doc_timestamp: float | None = None
    snippet: str
    similarity_pct: float


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


class Calibration(BaseModel):
    """The score-display calibration the server used for this corpus's model, so
    the web UI's client-side re-derivation of similarity_pct matches the server
    instead of drifting from a hardcoded default."""
    cosine_center: float
    cosine_slope: float
    bm25_top: float
    bm25_floor: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    # Budget mode only: the stub tail below the full-context hits.
    stubs: list[SearchStub] = []
    calibration: Calibration | None = None


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
