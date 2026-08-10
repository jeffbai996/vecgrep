from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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
    full_k: int = Field(8, ge=1, le=100)
    # Keep the REST API's breadth ceiling aligned with the service default;
    # the human UI explicitly requests 40 for a useful dense first page.
    max_total: int = Field(100, ge=1, le=100)
    token_ceiling: int = Field(4000, ge=100, le=20000)

    @model_validator(mode="after")
    def validate_budget_shape(self) -> "SearchRequest":
        if self.full_k > self.max_total:
            raise ValueError("full_k cannot exceed max_total")
        return self


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
    # Precise anchors: 1-based inclusive line range + compact citation string
    # ("path#L12-L24") so a caller can cite/re-open the exact region.
    line_start: int | None = None
    line_end: int | None = None
    anchor: str = ""
    # Going-forward names: relevance_pct == similarity_pct (compat alias);
    # relevance_label is the qualitative bucket (exact/strong/related/weak).
    relevance_pct: float = 0.0
    relevance_label: str = ""
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


class TimelineRequest(BaseModel):
    query: str
    corpus: str | None = None
    top_k: int = 10
    max_groups: int = 4
    padding: int = 1200
    mode: str = "hybrid"
    filters: list[str] = []


class IncidentRequest(BaseModel):
    query: str
    corpus: str | None = None
    mode: str = "hybrid"
    filters: list[str] = []


class BrowseRequest(BaseModel):
    corpus: str
    channel: str | None = None
    date: str | None = None
    source_path: str | None = None
    since: str | None = None
    until: str | None = None
    tail: int | None = Field(default=100, ge=1, le=1000)


class TimelineEvent(BaseModel):
    speaker: str
    time: str
    text: str


class TimelineGroup(BaseModel):
    """One source file's contiguous slice, parsed into chronological events.
    slice_text is set only when the source isn't a transcript (no events)."""
    corpus: str
    source_id: str
    doc_timestamp: float | None = None
    slice_start: int
    slice_end: int
    events: list[TimelineEvent]
    slice_text: str = ""


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
    rank_weight: float = 1.0


class WeightRequest(BaseModel):
    # None / omitted resets to neutral 1.0; a positive number sets the weight.
    weight: float | None = None


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
