"""MCP server: expose vecgrep as a tool for Claude / Cursor / any MCP client.

Two transports share the same tool logic:

* stdio  — `vecgrep mcp` (default for desktop/CLI MCP clients)
* HTTP   — mounted at /mcp by `vecgrep serve` for remote MCP clients

The MCP SDK is gated behind the `vecgrep[mcp]` extra so the base install
stays slim. Imports stay lazy here on purpose: importing this module must
not require the extra to be installed (the CLI catches the RuntimeError
at the call site).

Tools:
    search          — semantic search across one or all corpora
    list_corpora    — list every persisted corpus and its stats
    get_corpus      — get details for one corpus by name

HTTP transport notes
--------------------
The HTTP path uses FastMCP with json_response=True + stateless_http=True.
This makes the server respond with plain JSON (not SSE) so remote clients
like claude.ai work correctly. Their initial probe sends only
"Accept: application/json"; the StreamableHTTPSessionManager-based
approach returned text/event-stream and the client got either a 406 or
a stalled connection.

    json_response=True    → respond with application/json, not SSE
    stateless_http=True   → no server-side session state; each request
                            is independent. Prevents stale session ID
                            bugs and avoids tying up resources when a
                            client disconnects without sending DELETE.
    dns rebinding off     → requests arrive through Tailscale Funnel with
                            a non-localhost Host header; the default
                            allowlist would reject them.
"""
from __future__ import annotations

import json
from typing import Any

from ..backend.service import VecgrepService


def _require_mcp() -> None:
    """Check that the mcp SDK is installed. Raises RuntimeError with an
    actionable message when the optional extra is missing.
    """
    try:
        import mcp  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MCP server requires the 'mcp' extra. "
            "Install with `pip install vecgrep[mcp]`."
        ) from e


# ---------------------------------------------------------------------------
# Shared tool logic — called by both the stdio and HTTP handlers.
# ---------------------------------------------------------------------------

# One service per settings generation. Tool calls used to build a fresh
# VecgrepService each (registry read + store handles + cache handle per
# call); the REST routes already kept a singleton. Keyed on the Settings
# object identity so a config reload (tests, env change) rebuilds cleanly.
_SVC: tuple[int, VecgrepService] | None = None


def _close_svc(svc: VecgrepService) -> None:
    """Best-effort close of a service's store client — matters for embedded
    qdrant (holds a dir lock + noisy destructor at interpreter shutdown)."""
    try:
        svc.store.client.close()
    except Exception:
        pass


def _svc() -> VecgrepService:
    global _SVC
    from ..backend.config import get_settings

    key = id(get_settings())
    if _SVC is None or _SVC[0] != key:
        if _SVC is not None:
            _close_svc(_SVC[1])  # settings changed — release the old handles
        _SVC = (key, VecgrepService())
    return _SVC[1]


def _reset_service_cache() -> None:
    global _SVC
    if _SVC is not None:
        _close_svc(_SVC[1])
    _SVC = None


import atexit as _atexit
_atexit.register(lambda: _reset_service_cache())


def _result_payload(r) -> dict:
    return {
        "similarity_pct": round(r.similarity_pct, 1),  # compat alias
        "relevance_pct": round(r.relevance_pct, 1),
        "relevance_label": r.relevance_label,
        "corpus": r.corpus,
        "source_id": r.source_id,
        "matched_by": r.matched_by,
        "chunk": r.chunk,
        "chunk_id": r.chunk_id,
        "doc_timestamp": r.doc_timestamp,
        "anchor": r.anchor,
        "line_start": r.line_start,
        "line_end": r.line_end,
        "context_before": r.context_before,
        "context_after": r.context_after,
        # Raw retriever scores for downstream re-calibration.
        "scores": r.explain or {},
    }


# Corpora at or above this many points get cross-encoder rerank by default.
# Below it the candidate pool is small enough that RRF already ranks well and
# the extra latency buys nothing. See the module docstring on _run_search.
RERANK_AUTO_MIN_POINTS = 10_000


def _should_rerank(svc, args: dict) -> bool:
    """Explicit `rerank` in the call always wins; otherwise decide on size.

    Never raises: if the corpus size can't be read for any reason, fall back to
    the old default (off) rather than failing the search.
    """
    explicit = args.get("rerank")
    if explicit is not None:
        return bool(explicit)
    try:
        name = args.get("corpus")
        if not name:
            return False          # cross-corpus search: don't pay it blind
        for c in svc.list_corpora():
            # list_corpora returns Corpus objects; the API mirror is a dict.
            # Handle both, and read chunk_count — the field that actually
            # exists (points/chunks do not, and guessing them would make this
            # silently return False forever).
            if isinstance(c, dict):
                cn, n = c.get("name"), c.get("chunk_count")
            else:
                cn, n = getattr(c, "name", None), getattr(c, "chunk_count", None)
            if cn == name:
                return bool(n and int(n) >= RERANK_AUTO_MIN_POINTS)
    except Exception:
        return False
    return False


def _run_search(args: dict) -> str:
    svc = _svc()
    common = dict(
        corpus_name=args.get("corpus"),
        mode=args.get("mode", "hybrid"),
        rerank=_should_rerank(svc, args),
        filters=args.get("filters") or None,
    )
    if args.get("budget"):
        # Breadth mode: full head + one-line stub tail under a token ceiling.
        # Stubs carry chunk_id — expand any of them with the get_chunk tool.
        full, stubs = svc.search_budgeted(
            args["query"],
            full_k=int(args.get("full_k") or 8),
            token_ceiling=int(args.get("token_ceiling") or 4000),
            **common,
        )
        payload = {
            "full": [_result_payload(r) for r in full],
            "stubs": [
                {
                    "chunk_id": s.chunk_id,
                    "corpus": s.corpus,
                    "source_id": s.source_id,
                    "doc_timestamp": s.doc_timestamp,
                    "snippet": s.snippet,
                    "similarity_pct": round(s.similarity_pct, 1),
                }
                for s in stubs
            ],
        }
        return json.dumps(payload, indent=2)
    results = svc.search(args["query"], top_k=args.get("top_k"), **common)
    return json.dumps([_result_payload(r) for r in results], indent=2)


def _run_get_chunk(corpus: str, chunk_id: str, window: int = 400) -> str:
    """Expand a chunk (typically from a budget-search stub) to its full
    context window. window: chars each side, -1 = whole source."""
    svc = _svc()
    win = svc.get_chunk_window(corpus, chunk_id, window)
    if win is None:
        return json.dumps({"error": f"chunk not found: {corpus}/{chunk_id}"})
    return json.dumps(win, indent=2)


def _run_timeline(args: dict) -> str:
    """'What happened?' mode — contiguous chronological slices grouped by
    source file, transcript slices parsed into speaker/time/text events."""
    svc = _svc()
    groups = svc.timeline(
        args["query"],
        args.get("corpus"),
        top_k=int(args.get("top_k") or 10),
        max_groups=int(args.get("max_groups") or 4),
        mode=args.get("mode", "hybrid"),
        filters=args.get("filters") or None,
    )
    return json.dumps(groups, indent=2, ensure_ascii=False)


def _run_incident(args: dict) -> str:
    """Structured incident answer assembled from search + timeline."""
    svc = _svc()
    inc = svc.incident(
        args["query"],
        args.get("corpus"),
        mode=args.get("mode", "hybrid"),
        filters=args.get("filters") or None,
    )
    if inc is None:
        return json.dumps({"error": "nothing found for that query"})
    return json.dumps(inc, indent=2, ensure_ascii=False)


def _run_browse(args: dict) -> str:
    """Location-first reading: full event sequence for channel/date/path —
    no query, no ranking. At least one selector required."""
    svc = _svc()
    try:
        groups = svc.browse(
            args["corpus"],
            channel=args.get("channel"),
            date=args.get("date"),
            source_path=args.get("source_path"),
            since=args.get("since"),
            until=args.get("until"),
            tail=args.get("tail"),
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(groups, indent=2, ensure_ascii=False)


def _run_get_source(corpus: str, source_id: str) -> str:
    """Whole source document by source_id (text + parsed events)."""
    svc = _svc()
    doc = svc.get_source(corpus, source_id)
    if doc is None:
        return json.dumps({"error": f"source not indexed: {source_id}"})
    return json.dumps(doc, indent=2, ensure_ascii=False)


def _run_related(args: dict) -> str:
    """Nearest neighbours of an existing chunk (query-by-example)."""
    svc = _svc()
    try:
        results = svc.related(
            args["chunk_id"], args["corpus"], top_k=int(args.get("top_k") or 8)
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps(
        [_result_payload(r) for r in results], indent=2, ensure_ascii=False
    )


def _run_compare(args: dict) -> str:
    """Temporal diff: one query, two time windows, source-level delta."""
    svc = _svc()
    try:
        out = svc.compare(
            args["query"],
            args["corpus"],
            a_after=args.get("a_after"),
            a_before=args.get("a_before"),
            b_after=args.get("b_after"),
            b_before=args.get("b_before"),
            top_k=int(args.get("top_k") or 8),
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
    out["a"]["results"] = [_result_payload(r) for r in out["a"]["results"]]
    out["b"]["results"] = [_result_payload(r) for r in out["b"]["results"]]
    return json.dumps(out, indent=2, ensure_ascii=False)


def _run_stats(args: dict) -> str:
    """Corpus health snapshot: counts, date coverage, gaps, source sizes."""
    svc = _svc()
    try:
        return json.dumps(svc.corpus_stats(args["corpus"]), indent=2,
                          ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_summarize(args: dict) -> str:
    """Boot-context rollup: speakers, span, sources, sampled chunks."""
    svc = _svc()
    try:
        return json.dumps(
            svc.summarize_corpus(
                args["corpus"],
                after=args.get("after"),
                before=args.get("before"),
                sample=int(args.get("sample") or 40),
            ),
            indent=2, ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_list_aliases() -> str:
    """Read-only view of the active alias-expansion map."""
    from ..backend.aliases import describe_aliases

    return json.dumps(describe_aliases(), indent=2, ensure_ascii=False)


def _run_list_corpora() -> str:
    svc = _svc()
    corpora = [
        {
            "name": c.name,
            "embed_backend": c.embed_backend,
            "embed_model": c.embed_model,
            "doc_count": c.doc_count,
            "chunk_count": c.chunk_count,
            "chunker": c.chunker,
        }
        for c in svc.list_corpora()
    ]
    return json.dumps(corpora, indent=2)


def _run_get_corpus(name: str) -> str:
    svc = _svc()
    for c in svc.list_corpora():
        if c.name == name:
            detail = {
                "name": c.name,
                "embed_backend": c.embed_backend,
                "embed_model": c.embed_model,
                "dim": c.dim,
                "chunker": c.chunker,
                "doc_count": c.doc_count,
                "chunk_count": c.chunk_count,
                "sources": c.sources,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "decay_half_life_days": c.decay_half_life_days,
                # What `filters` the caller can pass to search — surfaced so
                # filtering by actor/channel/date isn't a guessing game.
                "filterable": svc.filterable_fields(c.name)["filters"],
            }
            return json.dumps(detail, indent=2)
    return json.dumps({"error": f"corpus not found: {name}"})


# Where an agent's proposals land when it doesn't name a corpus. A dedicated
# corpus for agent-contributed entries keeps them separate from human-authored
# and ingested corpora. Override with VECGREP_DEFAULT_PROPOSE_CORPUS.
import os as _os
DEFAULT_PROPOSE_CORPUS = _os.environ.get(
    "VECGREP_DEFAULT_PROPOSE_CORPUS", "claude-ai")


def _allowed_propose_corpora() -> set[str]:
    """The corpora an agent is permitted to propose into. DEFAULT-DENY: only the
    dedicated propose corpus is allowed unless the operator explicitly widens the
    set via VECGREP_PROPOSE_ALLOWED_CORPORA (comma-separated). This is the wall
    that keeps an agent -- or a prompt injection in the untrusted content it
    ingested -- from landing a proposal in a shared/human corpus that a confirm
    would then publish. The default corpus is always allowed."""
    raw = _os.environ.get("VECGREP_PROPOSE_ALLOWED_CORPORA", "").strip()
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    allowed.add(DEFAULT_PROPOSE_CORPUS)
    return allowed


# Cap on a single proposal's content. Without it an oversized blob is accepted,
# then OOMs/times-out the embed at confirm time AFTER the file is already on disk
# (a ghost entry). Reject it up front. Override with VECGREP_MAX_PROPOSAL_BYTES.
MAX_PROPOSAL_CONTENT_BYTES = int(
    _os.environ.get("VECGREP_MAX_PROPOSAL_BYTES", str(1_000_000)))

# How much of the rendered proposal to echo back for review. Long enough to see
# the whole of a normal entry (the old 500 cap hid the body of anything real).
PROPOSE_PREVIEW_CHARS = int(
    _os.environ.get("VECGREP_PROPOSE_PREVIEW_CHARS", str(4000)))


def _write_dir(corpus: str):
    from ..backend.config import get_settings
    return get_settings().home / "write" / corpus


def _corpus_doc_dir(corpus: str):
    """Where THIS corpus's doc files physically live — for the propose
    existence/id checks. A native agent-write corpus (claude-ai, scratch) keeps
    its docs under the write dir. A MIRROR corpus's docs live wherever its
    sources were ingested from (e.g. a dump dir); resolve that from the registry
    so an edit/delete of an existing mirror doc validates against real files,
    not the empty write dir. Generic — derived from the corpus's own sources, no
    per-corpus hardcoding. Falls back to the write dir when there are no sources
    (a brand-new write-only corpus)."""
    from pathlib import Path
    wd = _write_dir(corpus)
    try:
        from ..backend.service import VecgrepService
        for c in VecgrepService().list_corpora():
            if c.name == corpus and getattr(c, "sources", None):
                d = Path(c.sources[0]).parent
                if d.is_dir():
                    return d
    except Exception:
        pass
    return wd


def _pending_store():
    from ..backend.config import get_settings
    from ..backend.write import confirm as _C
    return _C.ProposalStore(get_settings().home / "write" / "_pending")


def _doc_body(corpus: str, doc_id: str) -> str | None:
    """The current BODY (content minus frontmatter) of an existing doc, or None
    if it isn't on disk. Used by propose_edit's patch mode to str-replace the
    body without the caller re-sending it. Reads from where the corpus's docs
    physically live (same resolver the edit existence-check uses)."""
    from ..backend.ingestion.adapters.markdown import parse_frontmatter
    path = _corpus_doc_dir(corpus) / f"{doc_id}.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    # Strip a leading frontmatter block the same way parse_frontmatter detects
    # one, so the patch operates on body only (frontmatter is regenerated from
    # meta on render). No frontmatter → the whole file is body.
    if text.startswith("---") and parse_frontmatter(text):
        end = text[3:].find("\n---")
        if end != -1:
            after = text[3 + end + 4:]        # past the closing '---'
            after = after.split("\n", 1)[1] if "\n" in after else ""
            return after.strip("\n")
    return text.strip("\n")


def _apply_patch(body: str, old_str: str, new_str: str) -> str:
    """Strict single-occurrence str-replace (mirrors the str_replace contract:
    a non-unique match is a hard error, never a silent wrong-edit). Raises
    ValueError on 0 or >1 matches."""
    n = body.count(old_str)
    if n == 0:
        raise ValueError("old_str not found in the doc body")
    if n > 1:
        raise ValueError(
            f"old_str not unique ({n} matches) — add surrounding context so it "
            f"identifies exactly one location")
    return body.replace(old_str, new_str, 1)


def _run_propose(corpus: str, content: str | None, edit_id: str | None = None,
                 source_kind: str | None = None, tags: list[str] | None = None,
                 origin: str = "bot-suggested",
                 old_str: str | None = None, new_str: str | None = None) -> str:
    """PROPOSE an entry (or edit) — WRITES NOTHING. Stores a pending proposal a
    human confirms later (`vecgrep confirm <id>`).

    This is the MCP-facing path: an agent (claude.ai) ingests untrusted content,
    so a direct write would be a prompt-injection → memory-poisoning vector. A
    proposal is inert until a human reviews + confirms it off-protocol. origin
    is 'bot-suggested' here (the wall: bots propose, humans authorize). Returns
    JSON {proposal_id, doc_id, is_edit, preview} or {error}.

    PATCH mode (edit only): pass old_str + new_str INSTEAD of content to fix one
    span without re-sending the whole body. The current body is loaded, the
    single unique occurrence of old_str is replaced, and the result flows
    through this exact same path (same corpus gate, size cap, proposal, confirm
    step). Body-only; frontmatter is preserved. A non-unique old_str is a hard
    error, never a silent mis-edit."""
    from ..backend.write import proposal as _P

    # Enforce the wall at the TOOL boundary, BEFORE creating any directory or
    # pending proposal: the corpus must be agent-writable, and the content must
    # be within the size cap. A rejected proposal leaves nothing on disk.
    allowed = _allowed_propose_corpora()
    if corpus not in allowed:
        return json.dumps({"error": (
            f"corpus {corpus!r} is not agent-writable (default-deny). "
            f"Allowed: {sorted(allowed)}. An operator widens this with "
            f"VECGREP_PROPOSE_ALLOWED_CORPORA. This guard keeps an agent from "
            f"landing a proposal in a shared corpus.")})

    # Patch mode: old_str/new_str (edit only, mutually exclusive with content).
    # Resolve `content` to the patched body here, then fall through to the exact
    # same proposal path a full edit takes — no forked approval logic.
    meta_only = False
    is_patch = old_str is not None or new_str is not None
    if is_patch:
        if content is not None:
            return json.dumps({"error": (
                "pass EITHER content (full overwrite) OR old_str+new_str "
                "(surgical patch), not both.")})
        if not edit_id:
            return json.dumps({"error": "patch mode (old_str/new_str) requires "
                                        "an edit target (doc_id)."})
        if old_str is None or new_str is None:
            return json.dumps({"error": "patch mode needs both old_str and "
                                        "new_str."})
        body = _doc_body(corpus, edit_id)
        if body is None:
            return json.dumps({"error": (
                f"doc {edit_id!r} not found in corpus {corpus!r} — can't patch "
                f"a doc that doesn't exist.")})
        try:
            content = _apply_patch(body, old_str, new_str)
        except ValueError as e:
            return json.dumps({"error": str(e)})
    elif content is None:
        # META-ONLY mode: no content, no patch — but tags/source_kind present.
        # The doc's current body is reused untouched; only frontmatter changes.
        if not (tags or source_kind):
            return json.dumps({"error": "propose needs content (or old_str/"
                                        "new_str for a patch, or tags/"
                                        "source_kind for a meta-only edit)."})
        if not edit_id:
            return json.dumps({"error": "a meta-only edit (tags/source_kind "
                                        "without content) requires an edit "
                                        "target (doc_id)."})
        body = _doc_body(corpus, edit_id)
        if body is None:
            return json.dumps({"error": (
                f"doc {edit_id!r} not found in corpus {corpus!r} — can't retag "
                f"a doc that doesn't exist.")})
        content = body
        meta_only = True

    n_bytes = len(content.encode("utf-8"))
    if n_bytes > MAX_PROPOSAL_CONTENT_BYTES:
        return json.dumps({"error": (
            f"content is {n_bytes} bytes, over the {MAX_PROPOSAL_CONTENT_BYTES}"
            f"-byte proposal cap. Split it into smaller entries.")})

    # An EDIT validates its target against where the corpus's docs actually live
    # (a mirror corpus keeps them in its source dir, not the write dir); a NEW
    # write picks its next id from the write dir.
    corpus_dir = _corpus_doc_dir(corpus) if edit_id else _write_dir(corpus)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta = {"origin": origin}
    if source_kind:
        meta["source_kind"] = source_kind
    if tags:
        meta["tags"] = list(tags)
    if meta_only:
        # Internal routing marker: render_doc's whitelist keeps it out of the
        # written file; the writethrough payload's meta carries it upstream.
        meta["meta_only"] = True
    try:
        pr = _P.propose(corpus, content, corpus_dir, meta=meta, edit_id=edit_id)
    except _P.ProposalError as e:
        return json.dumps({"error": str(e)})
    _pending_store().put(pr)
    result = {
        "proposal_id": pr.proposal_id, "doc_id": pr.doc_id,
        "is_edit": pr.is_edit, "corpus": corpus,
        "status": "pending — a human must run `vecgrep confirm "
                  f"{pr.proposal_id}` to write it",
        "preview": (pr.rendered if len(pr.rendered) <= PROPOSE_PREVIEW_CHARS
                    else pr.rendered[:PROPOSE_PREVIEW_CHARS] + "\n... (truncated)"),
    }
    _fire_propose_hook(pr)
    return json.dumps(result)


def _run_propose_delete(corpus: str, delete_id: str,
                        origin: str = "bot-suggested") -> str:
    """PROPOSE a deletion — REMOVES NOTHING. Stores a pending delete-proposal a
    human confirms later (`vecgrep confirm <id>`). Same default-deny corpus wall
    as _run_propose: an agent can only propose deletes in an agent-writable
    corpus, and nothing is removed until a human authorizes it off-protocol."""
    from ..backend.write import proposal as _P

    allowed = _allowed_propose_corpora()
    if corpus not in allowed:
        return json.dumps({"error": (
            f"corpus {corpus!r} is not agent-writable (default-deny). "
            f"Allowed: {sorted(allowed)}. An operator widens this with "
            f"VECGREP_PROPOSE_ALLOWED_CORPORA. This guard keeps an agent from "
            f"proposing a delete in a shared corpus.")})
    corpus_dir = _corpus_doc_dir(corpus)  # validate against real doc location
    try:
        pr = _P.propose_delete(corpus, delete_id, corpus_dir,
                               meta={"origin": origin})
    except _P.ProposalError as e:
        return json.dumps({"error": str(e)})
    _pending_store().put(pr)
    result = {
        "proposal_id": pr.proposal_id, "doc_id": pr.doc_id,
        "is_delete": True, "corpus": corpus,
        "status": "pending — a human must run `vecgrep confirm "
                  f"{pr.proposal_id}` to delete it",
        "preview": (pr.rendered if len(pr.rendered) <= PROPOSE_PREVIEW_CHARS
                    else pr.rendered[:PROPOSE_PREVIEW_CHARS] + "\n... (truncated)"),
    }
    _fire_propose_hook(pr)
    return json.dumps(result)


# ── Direct write: the ONE unconfirmed write surface ──────────────────────────
# Everything above is propose-only for a reason (an agent ingesting untrusted
# content must not be able to write). This is the operator's deliberate exception
# for the opposite case: the operator themselves saying "save this", where a
# confirm tap per note is pure friction. It is narrow by construction --
# single corpus, append-only, capped -- so the worst case is junk docs in one
# searchable corpus, never a mutation of anything else.
# Read at CALL time, not import time: the MCP server is long-lived, so an
# operator changing a cap should take effect on service restart without a code
# change -- and import-time capture would also make these untestable.
def _direct_write_max_bytes() -> int:
    return int(_os.environ.get("VECGREP_DIRECT_WRITE_MAX_BYTES", str(256_000)))


def _direct_write_max_per_hour() -> int:
    return int(_os.environ.get("VECGREP_DIRECT_WRITE_MAX_PER_HOUR", str(60)))


def _direct_write_corpus() -> str | None:
    """The single corpus direct writes land in, or None (feature off).

    DEFAULT-OFF: vecgrep is public OSS, so a fresh clone must expose no
    unconfirmed write path at all. The operator opts in per-deployment."""
    return _os.environ.get("VECGREP_DIRECT_WRITE_CORPUS", "").strip() or None


def _direct_write_rate_ok(corpus_dir) -> bool:
    """True if fewer than the hourly cap of direct writes exist in the last hour.

    Counts mtimes in the corpus dir rather than keeping a counter file -- the docs
    ARE the ledger, so the cap survives a restart and can't drift out of sync.
    Runaway-loop protection only; a human note-taker never approaches it."""
    import time as _t
    if not corpus_dir.exists():
        return True
    cutoff = _t.time() - 3600
    recent = sum(1 for p in corpus_dir.glob("*.md")
                 if p.stat().st_mtime >= cutoff)
    return recent < _direct_write_max_per_hour()


def _run_direct_write(content: str, source_kind: str | None = None,
                      tags: list[str] | None = None,
                      title: str | None = None) -> str:
    """WRITE an entry immediately, no human confirm, into the ONE corpus the
    operator designated via VECGREP_DIRECT_WRITE_CORPUS.

    NOTE THE SIGNATURE: there is deliberately no `corpus` parameter, and no
    edit/delete/patch target. A blocked parameter can be argued around by a
    determined caller; an absent one cannot be expressed at all. So this tool
    can only ever append to one corpus, and can never modify or remove anything.

    Everything else stays propose-only (see _run_propose): for any other corpus,
    an agent proposes and a human confirms. This path exists because the operator
    is the one asking -- 'save this note' -- not because writes became safe.

    Stamped `origin: agent-direct` so an audit can always tell an unreviewed
    write from a human-confirmed one."""
    from ..backend.write import proposal as _P

    corpus = _direct_write_corpus()
    if not corpus:
        return json.dumps({"error": (
            "direct write is not enabled. An operator enables it for exactly one "
            "corpus with VECGREP_DIRECT_WRITE_CORPUS=<corpus>. Use propose_write "
            "instead — it works for any agent-writable corpus and lands after a "
            "human confirm.")})

    # A write-through routes a CONFIRMED op into an upstream store of record.
    # Pairing that with an unconfirmed write would let an agent mutate that
    # upstream with no human in the loop -- exactly the wall this whole module
    # exists to hold. Refuse before touching anything.
    from ..backend.write.confirm import _writethrough_cmd
    if _writethrough_cmd(corpus) is not None:
        return json.dumps({"error": (
            f"corpus {corpus!r} has a write-through configured, so an "
            f"unconfirmed write would mutate an upstream store with no human "
            f"review. Direct write refuses such a corpus. Point "
            f"VECGREP_DIRECT_WRITE_CORPUS at a plain local corpus, or use "
            f"propose_write for this one.")})

    if not content or not content.strip():
        return json.dumps({"error": "direct write needs non-empty content."})

    n_bytes = len(content.encode("utf-8"))
    if n_bytes > _direct_write_max_bytes():
        return json.dumps({"error": (
            f"content is {n_bytes} bytes, over the {_direct_write_max_bytes()}-byte "
            f"direct-write cap. Split it, or use propose_write for a large entry.")})

    corpus_dir = _write_dir(corpus)
    if not _direct_write_rate_ok(corpus_dir):
        return json.dumps({"error": (
            f"rate cap reached ({_direct_write_max_per_hour()} direct writes/hour "
            f"for {corpus!r}). This is runaway-loop protection; wait, or raise "
            f"VECGREP_DIRECT_WRITE_MAX_PER_HOUR.")})

    from datetime import datetime as _datetime, timezone as _timezone

    corpus_dir.mkdir(parents=True, exist_ok=True)
    doc_id = _P.next_doc_id(corpus_dir, corpus)
    meta = {
        "origin": "agent-direct",
        "status": "active",
        "tier": "normal",
        "corpus": corpus,
        "created_at": _datetime.now(_timezone.utc).isoformat(timespec="seconds"),
    }
    if title:
        meta["title"] = title
    if source_kind:
        meta["source_kind"] = source_kind
    if tags:
        meta["tags"] = list(tags)
    try:
        _P._validate_meta(meta)
    except _P.ProposalError as e:
        return json.dumps({"error": str(e)})

    rendered = _P.render_doc(doc_id, content, meta)
    target = corpus_dir / f"{doc_id}.md"
    target.write_text(rendered)

    # Embed immediately so the entry is searchable now -- the whole point is
    # "save this and be able to find it". Same incremental single-file index the
    # confirm path uses. A failure here leaves a findable doc on disk (truth), so
    # it is reported rather than fatal; a later reindex reconciles.
    index_note = ""
    try:
        _svc().index(str(target), corpus)
    except Exception as e:  # noqa: BLE001 - never lose the write over an embed
        index_note = (f" (not yet searchable: {e}; "
                      f"`vecgrep index` will reconcile)")

    return json.dumps({
        "committed": True, "doc_id": doc_id, "corpus": corpus,
        "path": str(target),
        "status": f"written and indexed{index_note}",
    })


def _run_direct_edit(doc_id: str, content: str | None = None,
                     old_str: str | None = None,
                     new_str: str | None = None) -> str:
    """EDIT an existing entry immediately, no human confirm, in the ONE corpus the
    operator designated. Counterpart to propose_edit.

    Two modes, same as propose_edit: `content` replaces the whole body, or
    old_str + new_str patches one unique span.

    An edit is more dangerous than an append -- an append is noise, an overwrite
    DESTROYS. So this path keeps two guards the write path doesn't need:
      1. a `.bak` of the previous file before every edit, so an unreviewed
         overwrite is always recoverable
      2. `tier: protected` docs are refused outright (human-only, even here)

    Deletes are deliberately absent: propose_delete stays the only route, for
    every corpus. Losing an entry should always cross a human."""
    from ..backend.write import proposal as _P

    corpus = _direct_write_corpus()
    if not corpus:
        return json.dumps({"error": (
            "direct edit is not enabled. An operator enables it for exactly one "
            "corpus with VECGREP_DIRECT_WRITE_CORPUS=<corpus>. Use propose_edit "
            "instead — it works for any agent-writable corpus and lands after a "
            "human confirm.")})

    from ..backend.write.confirm import _writethrough_cmd
    if _writethrough_cmd(corpus) is not None:
        return json.dumps({"error": (
            f"corpus {corpus!r} has a write-through configured, so an "
            f"unconfirmed edit would mutate an upstream store with no human "
            f"review. Direct edit refuses such a corpus; use propose_edit.")})

    is_patch = old_str is not None or new_str is not None
    if is_patch and content is not None:
        return json.dumps({"error": (
            "pass EITHER content (full overwrite) OR old_str+new_str "
            "(surgical patch), not both.")})
    if is_patch and (old_str is None or new_str is None):
        return json.dumps({"error": "patch mode needs both old_str and new_str."})
    if not is_patch and content is None:
        return json.dumps({"error": (
            "direct edit needs content (or old_str/new_str for a patch).")})

    # Resolve the target INSIDE this corpus only. _write_dir(corpus) is the sole
    # directory consulted, so a doc_id belonging to another corpus simply doesn't
    # exist here -- the corpus boundary is structural, not a check to forget.
    target = _write_dir(corpus) / f"{doc_id}.md"
    if not target.exists():
        return json.dumps({"error": (
            f"doc {doc_id!r} not found in corpus {corpus!r} — direct edit only "
            f"touches its own corpus. (Nothing was changed.)")})

    on_disk = target.read_text()
    if "tier: protected" in on_disk:
        return json.dumps({"error": (
            f"{doc_id} is tier: protected — human-only. Use propose_edit so a "
            f"human confirms it with the exact-id ack.")})

    body = _doc_body(corpus, doc_id)
    if body is None:
        return json.dumps({"error": f"could not read the body of {doc_id!r}."})

    if is_patch:
        try:
            new_body = _apply_patch(body, old_str, new_str)
        except ValueError as e:
            return json.dumps({"error": str(e)})
    else:
        new_body = content

    n_bytes = len(new_body.encode("utf-8"))
    if n_bytes > _direct_write_max_bytes():
        return json.dumps({"error": (
            f"result is {n_bytes} bytes, over the {_direct_write_max_bytes()}-byte "
            f"direct-write cap.")})

    # Back up BEFORE mutating: this write had no human review, so the previous
    # body must stay recoverable. Timestamped so repeated edits don't clobber
    # each other's history.
    import time as _t
    bak = target.with_suffix(f".md.bak-{int(_t.time())}")
    bak.write_text(on_disk)

    # Preserve the existing frontmatter, swap only the body, and record that an
    # unreviewed edit touched it (and when) for audit.
    from ..backend.ingestion.adapters.markdown import parse_frontmatter
    meta = dict(parse_frontmatter(on_disk) or {})
    meta.update({"corpus": corpus, "origin": "agent-direct"})
    from datetime import datetime as _datetime, timezone as _timezone
    meta["edited_at"] = _datetime.now(_timezone.utc).isoformat(timespec="seconds")
    rendered = _P.render_doc(doc_id, new_body, meta)
    target.write_text(rendered)

    index_note = ""
    try:
        _svc().index(str(target), corpus)
    except Exception as e:  # noqa: BLE001 - the file on disk is the truth
        index_note = (f" (index stale: {e}; `vecgrep index` will reconcile)")

    return json.dumps({
        "committed": True, "doc_id": doc_id, "corpus": corpus,
        "path": str(target), "backup": str(bak),
        "mode": "patch" if is_patch else "overwrite",
        "status": f"edited and re-indexed{index_note}",
    })


def _fire_propose_hook(pr) -> None:
    """Optionally notify an external command that a proposal was created.

    If VECGREP_PROPOSE_HOOK is set, run it with the proposal JSON on stdin so a
    deployment can surface pending proposals out-of-band (a Discord card, a
    desktop notification, etc.) without vecgrep itself knowing anything about
    those channels — it stays generic. Strictly best-effort: the proposal is
    already safely stored, so a missing/failing hook must never break propose.
    """
    import os
    hook = os.environ.get("VECGREP_PROPOSE_HOOK", "").strip()
    if not hook:
        return
    import subprocess
    payload = json.dumps({
        "proposal_id": pr.proposal_id, "doc_id": pr.doc_id,
        "corpus": pr.corpus, "is_edit": pr.is_edit,
        "target_path": pr.target_path, "preview": pr.rendered[:1000],
        "meta": pr.meta,
    })
    try:
        subprocess.run(
            hook, shell=True, input=payload, text=True,
            timeout=10, capture_output=True,
        )
    except Exception:
        # Notification is a nicety; the write path's correctness does not
        # depend on it. Swallow everything.
        pass


# ---------------------------------------------------------------------------
# stdio transport — low-level Server (well-tested, leave as-is)
# ---------------------------------------------------------------------------

def build_mcp_server() -> Any:
    """Construct the configured MCP Server with the three tools registered.

    Used by the stdio transport. The HTTP transport uses build_http_app()
    which builds a FastMCP instance independently.
    """
    _require_mcp()
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("vecgrep")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description=(
                    "Semantic search across vecgrep corpora. Returns ranked chunks with "
                    "surrounding context. Use this instead of dumping documents into "
                    "context — index once, search per question."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query.",
                        },
                        "corpus": {
                            "type": "string",
                            "description": "Limit to one corpus. Omit to search all corpora.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results (default 5).",
                            "default": 5,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "vector", "bm25"],
                            "description": "Retrieval strategy. Default: hybrid (BM25+vector via RRF).",
                            "default": "hybrid",
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": (
                                "Cross-encoder rerank of the top candidates — "
                                "markedly better on long, fuzzy queries. Omit "
                                "to auto-enable on large corpora (>=10k "
                                "chunks) and skip it on small ones; pass "
                                "true/false to force."
                            ),
                        },
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional HARD-constraint filters. Forms: "
                                "'source:<glob>', 'source_path:<glob>', "
                                "'corpus:<name>', 'meta.<key>=<value>', "
                                "'date:YYYY-MM-DD|today|yesterday', "
                                "'after:<iso>|7d|24h|2w', 'before:<iso>|today', "
                                "'channel:<name>', 'speaker:<name>' (alias "
                                "author:), 'bot:true|false', "
                                "'has:code|table|link'. Prefix any with '-' to "
                                "EXCLUDE (-corpus:scratch). All ANDed. Use "
                                "time filters for 'today'-style questions so "
                                "old lore can't leak in; speaker: for 'what "
                                "did X say' questions."
                            ),
                        },
                        "budget": {
                            "type": "boolean",
                            "description": (
                                "Breadth mode: return the top full_k results WITH "
                                "context plus a one-line stub tail (up to a token "
                                "ceiling). Expand any stub via get_chunk. Best for "
                                "pattern-spotting across many hits."
                            ),
                            "default": False,
                        },
                        "full_k": {
                            "type": "integer",
                            "description": "Budget mode: results with full context (default 8).",
                            "default": 8,
                        },
                        "token_ceiling": {
                            "type": "integer",
                            "description": "Budget mode: approx token cap for the stub tail (default 4000).",
                            "default": 4000,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="timeline",
                description=(
                    "'What happened?' mode: reconstructs the event sequence "
                    "around an incident — contiguous chronological slices "
                    "grouped by source file, with speakers + timestamps "
                    "preserved. Use for narrative/incident questions instead "
                    "of piecing together ranked chunks. Supports the same "
                    "hard filters as search (date:/after:/before:/channel:)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "corpus": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "max_groups": {"type": "integer", "default": 4},
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "vector", "bm25"],
                            "default": "hybrid",
                        },
                        "filters": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="incident",
                description=(
                    "One structured answer for an incident question: title, "
                    "sources, participants, time range, the primary "
                    "chronological timeline, related context kept separate, "
                    "and a confidence label. Assembled from search + "
                    "timeline; same hard filters as search."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "corpus": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "vector", "bm25"],
                            "default": "hybrid",
                        },
                        "filters": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_chunk",
                description=(
                    "Expand a chunk to its surrounding context by chunk_id "
                    "(e.g. from a budget-search stub). window = chars each "
                    "side, -1 for the whole source document."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "window": {"type": "integer", "default": 400},
                    },
                    "required": ["corpus", "chunk_id"],
                },
            ),
            Tool(
                name="browse",
                description=(
                    "Location-first reading, no query: the full event "
                    "sequence for a channel and/or day / date-range / path "
                    "glob. Use when you know WHERE/WHEN rather than what "
                    "words to search for. tail=N returns only the newest N "
                    "events of the selection (the 'last N messages' view). "
                    "Requires at least one selector."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "channel": {"type": "string"},
                        "date": {"type": "string", "description": "YYYY-MM-DD (UTC day)"},
                        "since": {"type": "string", "description": "YYYY-MM-DD — first day included"},
                        "until": {"type": "string", "description": "YYYY-MM-DD — last day included"},
                        "tail": {"type": "integer", "description": "keep only the newest N events"},
                        "source_path": {"type": "string", "description": "fnmatch glob on source_id"},
                    },
                    "required": ["corpus"],
                },
            ),
            Tool(
                name="get_source",
                description=(
                    "Whole source document by source_id (raw text + parsed "
                    "transcript events). Use after a stub or corpus listing "
                    "hands you a source_id; get_chunk is the by-chunk_id "
                    "variant."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "source_id": {"type": "string"},
                    },
                    "required": ["corpus", "source_id"],
                },
            ),
            Tool(
                name="list_aliases",
                description=(
                    "Read-only view of the active alias-expansion map (which "
                    "entity surface forms expand into which). Explains why a "
                    "search matched terms you didn't type."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="related",
                description=(
                    "Nearest neighbours of an EXISTING chunk by its stored "
                    "vector (query-by-example). Feed it a chunk_id from any "
                    "search/stub hit to pull more evidence like it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                    },
                    "required": ["corpus", "chunk_id"],
                },
            ),
            Tool(
                name="compare",
                description=(
                    "Temporal diff: run one query in TWO time windows and get "
                    "both result sets plus the source-level delta. Answers "
                    "'how did we talk about X then vs now'. Window values "
                    "accept ISO dates or relative forms (30d, 2w, today)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "corpus": {"type": "string"},
                        "a_after": {"type": "string"},
                        "a_before": {"type": "string"},
                        "b_after": {"type": "string"},
                        "b_before": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                    },
                    "required": ["query", "corpus"],
                },
            ),
            Tool(
                name="stats",
                description=(
                    "Corpus health snapshot: chunk/doc counts, date coverage "
                    "span, gap days (zero-chunk days inside the span — a "
                    "broken archiver shows here), per-source chunk counts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"corpus": {"type": "string"}},
                    "required": ["corpus"],
                },
            ),
            Tool(
                name="summarize_corpus",
                description=(
                    "Rollup of a corpus (optionally time-windowed): speaker "
                    "tally, date span, top sources, and an evenly-spaced "
                    "sample of chunks to theme. Sampling is explicit in the "
                    "output, never silent."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "corpus": {"type": "string"},
                        "after": {"type": "string"},
                        "before": {"type": "string"},
                        "sample": {"type": "integer", "default": 40},
                    },
                    "required": ["corpus"],
                },
            ),
            Tool(
                name="write",
                description=(
                    "WRITE a new entry immediately — no human confirm. Only "
                    "available for the single corpus the operator opened up "
                    "(VECGREP_DIRECT_WRITE_CORPUS); there is no corpus "
                    "parameter, and it cannot edit or delete anything. Use this "
                    "when the USER asks you to save something. For any other "
                    "corpus use propose_write."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "title": {"type": "string"},
                        "source_kind": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="edit",
                description=(
                    "EDIT an existing entry immediately — no human confirm. Same "
                    "single operator-designated corpus as `write`; no corpus "
                    "parameter. Two modes: `content` replaces the whole body, or "
                    "`old_str`+`new_str` patches one unique span. The previous "
                    "body is backed up automatically. Protected entries are "
                    "refused. Cannot delete — use propose_delete."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "content": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["doc_id"],
                },
            ),
            Tool(
                name="propose_write",
                description=(
                    "PROPOSE a new entry for a corpus. Writes NOTHING — "
                    "creates a pending proposal a human reviews + confirms "
                    "before it's saved. Returns the proposal_id."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "corpus": {"type": "string"},
                        "source_kind": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="propose_edit",
                description=(
                    "PROPOSE an edit to an existing entry by id. Writes "
                    "NOTHING until a human confirms. Returns the proposal_id. "
                    "Pass `content` to overwrite the whole body, OR "
                    "`old_str`+`new_str` for a surgical str-replace patch of one "
                    "unique span (fails if old_str is absent or non-unique)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "content": {"type": "string",
                                    "description": "full new body (overwrite mode)"},
                        "old_str": {"type": "string",
                                    "description": "patch mode: the unique span to replace"},
                        "new_str": {"type": "string",
                                    "description": "patch mode: its replacement"},
                        "corpus": {"type": "string"},
                        "source_kind": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["doc_id"],
                },
            ),
            Tool(
                name="propose_delete",
                description=(
                    "PROPOSE deleting an existing entry by id. Removes "
                    "NOTHING until a human confirms. Returns the proposal_id."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "corpus": {"type": "string"},
                    },
                    "required": ["doc_id"],
                },
            ),
            Tool(
                name="list_corpora",
                description="List every vecgrep corpus and its stats (doc count, chunk count, embedding model).",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_corpus",
                description="Get full metadata for one corpus including its source list.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        args = arguments or {}
        try:
            if name == "search":
                text = _run_search(args)
            elif name == "timeline":
                text = _run_timeline(args)
            elif name == "incident":
                text = _run_incident(args)
            elif name == "get_chunk":
                text = _run_get_chunk(
                    args.get("corpus", ""),
                    args.get("chunk_id", ""),
                    int(args.get("window", 400)),
                )
            elif name == "browse":
                text = _run_browse(args)
            elif name == "get_source":
                text = _run_get_source(args.get("corpus", ""), args.get("source_id", ""))
            elif name == "related":
                text = _run_related(args)
            elif name == "compare":
                text = _run_compare(args)
            elif name == "stats":
                text = _run_stats(args)
            elif name == "summarize_corpus":
                text = _run_summarize(args)
            elif name == "list_aliases":
                text = _run_list_aliases()
            elif name == "write":
                text = _run_direct_write(
                    args.get("content", ""),
                    source_kind=args.get("source_kind"),
                    tags=args.get("tags"), title=args.get("title"),
                )
            elif name == "edit":
                text = _run_direct_edit(
                    args.get("doc_id", ""), content=args.get("content"),
                    old_str=args.get("old_str"), new_str=args.get("new_str"),
                )
            elif name == "propose_write":
                text = _run_propose(
                    args.get("corpus") or DEFAULT_PROPOSE_CORPUS,
                    args.get("content", ""), None,
                    args.get("source_kind"), args.get("tags"),
                )
            elif name == "propose_edit":
                doc_id = args.get("doc_id", "")
                corpus = args.get("corpus") or (
                    doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
                )
                text = _run_propose(
                    corpus, args.get("content"), doc_id,
                    args.get("source_kind"), args.get("tags"),
                    old_str=args.get("old_str"), new_str=args.get("new_str"),
                )
            elif name == "propose_delete":
                doc_id = args.get("doc_id", "")
                corpus = args.get("corpus") or (
                    doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
                )
                text = _run_propose_delete(corpus, doc_id)
            elif name == "list_corpora":
                text = _run_list_corpora()
            elif name == "get_corpus":
                text = _run_get_corpus(args.get("name", ""))
            else:
                text = json.dumps({"error": f"unknown tool: {name}"})
        except Exception as exc:
            text = json.dumps({"error": str(exc)})
        return [TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# HTTP transport — FastMCP with json_response + stateless_http
# ---------------------------------------------------------------------------

# ── OAuth helpers ───────────────────────────────────────────────────────────
# The MCP endpoint lives at <root>/mcp, but OAuth/RFC put discovery .well-known
# at the ORIGIN ROOT. So the auth-SERVER issuer is the root (well-known +
# /authorize + /token + /register land at root, where the SPA-free funnel root
# serves them), while the protected RESOURCE is <root>/mcp (the thing being
# guarded). One shared provider instance backs both the /mcp bearer gate (via
# the sub-app's AuthSettings) and the root discovery routes (mounted by main.py)
# so tokens issued at root validate on /mcp.

_PROVIDER = None  # singleton — same token store for the gate + the root routes


def _shared_provider():
    global _PROVIDER
    if _PROVIDER is None:
        from ..backend.auth.provider import VecgrepOAuthProvider
        _PROVIDER = VecgrepOAuthProvider()
    return _PROVIDER


def _oauth_issuer(issuer_url: str):
    """Auth-server issuer = the ORIGIN ROOT (strip any /mcp path)."""
    from pydantic import AnyHttpUrl
    from urllib.parse import urlsplit
    p = urlsplit(issuer_url)
    return AnyHttpUrl(f"{p.scheme}://{p.netloc}")


def _oauth_resource(issuer_url: str):
    """Protected resource = the MCP endpoint itself (<root>/mcp)."""
    from pydantic import AnyHttpUrl
    from urllib.parse import urlsplit
    p = urlsplit(issuer_url)
    return AnyHttpUrl(f"{p.scheme}://{p.netloc}/mcp")


def _oauth_client_reg():
    """Dynamic Client Registration (RFC 7591): claude.ai self-registers instead
    of needing a pre-shared client_id."""
    from mcp.server.auth.settings import ClientRegistrationOptions
    return ClientRegistrationOptions(
        enabled=True, valid_scopes=["read", "propose"], default_scopes=["read"],
    )


def build_oauth_root_routes(oauth_issuer_url: str) -> list:
    """Return the OAuth discovery + auth routes to mount at the PARENT app ROOT
    (ahead of the SPA catch-all). These serve the .well-known JSON, /authorize,
    /token, /register at the origin root where the SDK advertises them and where
    claude.ai looks. Shares the provider with the /mcp bearer gate."""
    _require_mcp()
    from mcp.server.auth.routes import (
        create_auth_routes, create_protected_resource_routes,
    )
    issuer = _oauth_issuer(oauth_issuer_url)
    resource = _oauth_resource(oauth_issuer_url)
    routes = create_auth_routes(
        provider=_shared_provider(),
        issuer_url=issuer,
        client_registration_options=_oauth_client_reg(),
    )
    routes += create_protected_resource_routes(
        resource_url=resource,
        authorization_servers=[issuer],
        scopes_supported=["read", "propose"],
    )
    return routes


def build_http_app(oauth_issuer_url: str | None = None) -> Any:
    """Build a Starlette ASGI app for the MCP HTTP endpoint.

    oauth_issuer_url: when set, enable the embedded OAuth 2.1 auth server on this
    endpoint (FastMCP mounts the auth routes + bearer middleware). When None, no
    OAuth — the endpoint is network-trusted by the deployment.

    Uses FastMCP instead of the raw StreamableHTTPSessionManager so that:

    1. json_response=True  — respond with application/json, not text/event-stream.
       Remote clients (claude.ai) send "Accept: application/json" on their initial
       probe; the old StreamableHTTPSessionManager returned SSE and the client got
       a stalled or 406 response.

    2. stateless_http=True — no server-side session state. Each HTTP request is
       fully self-contained. Prevents stale session ID bugs where a client caches
       a dead session ID after a server restart and never re-initialises.

    3. DNS rebinding protection disabled — requests arrive through Tailscale Funnel
       with a non-localhost Host header; the default allowlist would reject them.

    The returned Starlette app registers the MCP handler at '/' (via
    streamable_http_path='/'), so the parent FastAPI app's _BearerGatedASGI
    middleware can strip the '/mcp' prefix and delegate correctly.
    """
    _require_mcp()
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.server import TransportSecuritySettings

    # OAuth: when an issuer URL is configured, run the embedded auth server.
    # FastMCP, given `auth` + `auth_server_provider`, auto-mounts /authorize,
    # /token, /.well-known and gates the MCP endpoint with bearer middleware.
    # Off (issuer None) → no auth here; /mcp is network-trusted (the parent app
    # reaches it over localhost/tailnet, or via the OAuth-less secret path).
    fmcp_kwargs: dict = {}
    if oauth_issuer_url:
        from mcp.server.auth.settings import AuthSettings
        fmcp_kwargs["auth"] = AuthSettings(
            issuer_url=_oauth_issuer(oauth_issuer_url),
            resource_server_url=_oauth_resource(oauth_issuer_url),
            required_scopes=["read"],
            client_registration_options=_oauth_client_reg(),
        )
        fmcp_kwargs["auth_server_provider"] = _shared_provider()

    fmcp = FastMCP("vecgrep", **fmcp_kwargs)
    fmcp.settings.json_response = True
    fmcp.settings.stateless_http = True
    # Register at '/' so the parent app's prefix-stripping (_BearerGatedASGI
    # rewrites scope['path'] = '/') routes correctly.
    fmcp.settings.streamable_http_path = "/"
    # Tailscale Funnel forwards requests with a non-localhost Host header.
    fmcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    @fmcp.tool(
        description=(
            "Semantic search across vecgrep corpora. Returns ranked chunks with "
            "surrounding context. Use this instead of dumping documents into "
            "context — index once, search per question."
        )
    )
    def search(
        query: str,
        corpus: str | None = None,
        top_k: int = 5,
        mode: str = "hybrid",
        rerank: bool | None = None,
        filters: list[str] | None = None,
        budget: bool = False,
        full_k: int = 8,
        token_ceiling: int = 4000,
    ) -> str:
        """Natural-language query. corpus: limit to one corpus (omit = all).
        top_k: max results. mode: hybrid|vector|bm25.
        rerank: cross-encoder rerank — markedly better on long, fuzzy
        queries. Omit to auto-enable on large corpora (>=10k chunks)
        and skip it on small ones; pass true/false to force.
        filters: hard constraints — 'source:<glob>', 'source_path:<glob>',
        'corpus:<name>', 'meta.<k>=<v>', 'date:YYYY-MM-DD|today|yesterday',
        'after:<iso>|7d|24h|2w', 'before:<iso>|today', 'channel:<name>',
        'speaker:<name>' (alias author:), 'bot:true|false',
        'has:code|table|link'; prefix with '-' to EXCLUDE. Use time filters
        for 'today'-style questions so old lore can't leak in; speaker: for
        'what did X say'.
        budget: breadth mode — top full_k results WITH context plus a
        one-line stub tail capped at ~token_ceiling tokens; expand any stub
        via get_chunk. Best for pattern-spotting across many hits."""
        return _run_search({
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
            "mode": mode,
            "rerank": rerank,
            "filters": filters,
            "budget": budget,
            "full_k": full_k,
            "token_ceiling": token_ceiling,
        })

    @fmcp.tool(
        description=(
            "'What happened?' mode: reconstructs the event sequence around an "
            "incident — contiguous chronological slices grouped by source "
            "file, speakers + timestamps preserved. Use for narrative/incident "
            "questions instead of piecing together ranked chunks."
        )
    )
    def timeline(
        query: str,
        corpus: str | None = None,
        top_k: int = 10,
        max_groups: int = 4,
        mode: str = "hybrid",
        filters: list[str] | None = None,
    ) -> str:
        """Same hard filters as search (date:/after:/before:/channel:...)."""
        return _run_timeline({
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
            "max_groups": max_groups,
            "mode": mode,
            "filters": filters,
        })

    @fmcp.tool(
        description=(
            "One structured answer for an incident question: title, sources, "
            "participants, time range, primary chronological timeline, "
            "related context separated, confidence label. Same hard filters "
            "as search."
        )
    )
    def incident(
        query: str,
        corpus: str | None = None,
        mode: str = "hybrid",
        filters: list[str] | None = None,
    ) -> str:
        """Assembled from search + timeline primitives."""
        return _run_incident({
            "query": query,
            "corpus": corpus,
            "mode": mode,
            "filters": filters,
        })

    @fmcp.tool(
        description=(
            "Location-first reading, no query: the full event sequence for a "
            "channel and/or day (YYYY-MM-DD) and/or path glob. Use when you "
            "know WHERE/WHEN rather than what words to search for."
        )
    )
    def browse(
        corpus: str,
        channel: str | None = None,
        date: str | None = None,
        source_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tail: int | None = None,
    ) -> str:
        """At least one selector (channel/date/since/until/source_path).
        since/until: inclusive YYYY-MM-DD day range. tail: keep only the
        newest N events of the selection — the 'last N messages' view."""
        return _run_browse({
            "corpus": corpus,
            "channel": channel,
            "date": date,
            "source_path": source_path,
            "since": since,
            "until": until,
            "tail": tail,
        })

    @fmcp.tool(
        description=(
            "Whole source document by source_id (raw text + parsed transcript "
            "events). Use after a stub or corpus listing hands you a "
            "source_id; get_chunk is the by-chunk_id variant."
        )
    )
    def get_source(corpus: str, source_id: str) -> str:
        """source_id exactly as returned by search/stubs/get_corpus."""
        return _run_get_source(corpus, source_id)

    @fmcp.tool(
        description=(
            "Read-only view of the active alias-expansion map. Explains why "
            "a search matched terms you didn't type."
        )
    )
    def list_aliases() -> str:
        """No arguments."""
        return _run_list_aliases()

    @fmcp.tool(
        description=(
            "Nearest neighbours of an EXISTING chunk by its stored vector "
            "(query-by-example). Feed it a chunk_id from any search/stub hit "
            "to pull more evidence like it."
        )
    )
    def related(corpus: str, chunk_id: str, top_k: int = 8) -> str:
        """chunk_id exactly as returned by search/stubs."""
        return _run_related({"corpus": corpus, "chunk_id": chunk_id,
                             "top_k": top_k})

    @fmcp.tool(
        description=(
            "Temporal diff: one query in TWO time windows, both result sets "
            "plus the source-level delta. 'How did we talk about X then vs "
            "now'. Windows accept ISO dates or relative forms (30d, 2w, "
            "today)."
        )
    )
    def compare(query: str, corpus: str, a_after: str = "",
                a_before: str = "", b_after: str = "", b_before: str = "",
                top_k: int = 8) -> str:
        """Empty window bounds mean unbounded on that side."""
        return _run_compare({
            "query": query, "corpus": corpus,
            "a_after": a_after or None, "a_before": a_before or None,
            "b_after": b_after or None, "b_before": b_before or None,
            "top_k": top_k,
        })

    @fmcp.tool(
        description=(
            "Corpus health snapshot: chunk/doc counts, date coverage span, "
            "gap days (a broken archiver shows here), per-source sizes."
        )
    )
    def stats(corpus: str) -> str:
        """One corpus per call."""
        return _run_stats({"corpus": corpus})

    @fmcp.tool(
        description=(
            "Rollup of a corpus (optionally time-windowed): speaker tally, "
            "date span, top sources, evenly-spaced sample of chunks to "
            "theme. Sampling is explicit, never silent."
        )
    )
    def summarize_corpus(corpus: str, after: str = "", before: str = "",
                         sample: int = 40) -> str:
        """after/before accept ISO or relative (7d, today)."""
        return _run_summarize({"corpus": corpus, "after": after or None,
                               "before": before or None, "sample": sample})

    @fmcp.tool(
        description=(
            "Expand a chunk to its surrounding context by chunk_id (e.g. from "
            "a budget-search stub). window = chars each side, -1 for the "
            "whole source document."
        )
    )
    def get_chunk(corpus: str, chunk_id: str, window: int = 400) -> str:
        """corpus + chunk_id identify the chunk (both are on every result and
        stub). window: context chars each side."""
        return _run_get_chunk(corpus, chunk_id, window)

    @fmcp.tool(
        description="List every vecgrep corpus and its stats (doc count, chunk count, embedding model)."
    )
    def list_corpora() -> str:
        """Returns JSON array of corpus metadata."""
        return _run_list_corpora()

    @fmcp.tool(
        description="Get full metadata for one corpus including its source list."
    )
    def get_corpus(name: str) -> str:
        """name: corpus name (from list_corpora)."""
        return _run_get_corpus(name)

    @fmcp.tool(
        description=(
            "WRITE a new entry immediately — no human confirm needed. Lands in "
            "the single corpus the operator opened up for direct writes; there "
            "is no corpus parameter, and this tool cannot edit or delete "
            "anything. USE THIS when the user says 'save this' / 'remember "
            "this' / 'note this down'. For any other corpus, use propose_write."
        )
    )
    def write(content: str, title: str | None = None,
              source_kind: str | None = None,
              tags: list[str] | None = None) -> str:
        """content: entry text. title: optional heading. source_kind:
        insight|fact|correction|journal|decision|memory|todo. tags: optional.
        Commits immediately and returns the doc_id."""
        return _run_direct_write(content, source_kind=source_kind,
                                 tags=tags, title=title)

    @fmcp.tool(
        description=(
            "EDIT an existing entry immediately — no human confirm needed. Same "
            "single direct-write corpus as `write`; no corpus parameter. Two "
            "modes: pass `content` to replace the whole body, OR "
            "`old_str`+`new_str` to patch one unique span without re-sending the "
            "doc. The previous body is backed up automatically, so an edit is "
            "always recoverable. Protected entries are refused; cannot delete "
            "(use propose_delete). For any other corpus, use propose_edit."
        )
    )
    def edit(doc_id: str, content: str | None = None,
             old_str: str | None = None, new_str: str | None = None) -> str:
        """doc_id: the entry's id (e.g. external-123). Either content (full
        overwrite) or old_str+new_str (surgical patch), not both."""
        return _run_direct_edit(doc_id, content=content,
                                old_str=old_str, new_str=new_str)

    @fmcp.tool(
        description=(
            "PROPOSE a new entry for a corpus. Writes NOTHING — creates a pending "
            "proposal a human reviews + confirms before it's saved. Use for "
            "durable notes/facts/decisions worth recalling later. Returns the "
            "proposal_id; tell the user to confirm it."
        )
    )
    def propose_write(content: str, corpus: str | None = None,
                      source_kind: str | None = None,
                      tags: list[str] | None = None) -> str:
        """content: entry text. corpus: target (defaults to the agent's own
        propose corpus, VECGREP_DEFAULT_PROPOSE_CORPUS). source_kind:
        insight|fact|correction|journal|decision|memory|todo. tags: optional.
        Returns the pending proposal (nothing is written until a human
        confirms)."""
        return _run_propose(corpus or DEFAULT_PROPOSE_CORPUS, content, None,
                            source_kind, tags)

    @fmcp.tool(
        description=(
            "PROPOSE an edit to an existing entry (by id, e.g. notes-007). Writes "
            "NOTHING — creates a pending proposal a human confirms before it "
            "overwrites. Returns the proposal_id.\n"
            "Two modes: pass `content` to replace the whole body, OR pass "
            "`old_str`+`new_str` for a SURGICAL PATCH — a str-replace of one "
            "unique span, so you don't re-send the whole doc to fix one line. "
            "The patch fails if old_str is missing or appears more than once "
            "(add surrounding context to make it unique). Patch touches the "
            "body only; frontmatter is preserved. content and old_str are "
            "mutually exclusive. THIRD mode: pass ONLY tags (and/or "
            "source_kind) with no content for a META-ONLY retag — the body "
            "stays untouched, only the metadata changes."
        )
    )
    def propose_edit(doc_id: str, content: str | None = None,
                     corpus: str | None = None,
                     source_kind: str | None = None,
                     tags: list[str] | None = None,
                     old_str: str | None = None,
                     new_str: str | None = None) -> str:
        """doc_id: existing id. content: new full body (overwrite mode).
        old_str/new_str: surgical patch mode (str-replace one unique span).
        corpus: inferred from id prefix if omitted. Proposal only — a human
        must confirm."""
        if corpus is None:
            corpus = doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
        return _run_propose(corpus, content, doc_id, source_kind, tags,
                            old_str=old_str, new_str=new_str)

    @fmcp.tool(
        description=(
            "PROPOSE deleting an existing entry (by id, e.g. notes-007). Removes "
            "NOTHING — creates a pending proposal a human confirms before the doc "
            "+ its embeddings are removed. The preview shows the entry that WILL "
            "be deleted. Returns the proposal_id; tell the user to confirm it."
        )
    )
    def propose_delete(doc_id: str, corpus: str | None = None) -> str:
        """doc_id: existing id to remove. corpus: inferred from id prefix if
        omitted. Proposal only — a human must confirm before anything is
        deleted. Same default-deny corpus gate as propose_write/edit."""
        if corpus is None:
            corpus = doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
        return _run_propose_delete(corpus, doc_id)

    return fmcp.streamable_http_app()


def run() -> None:
    """Entry point for `vecgrep mcp` (stdio transport)."""
    import asyncio

    from mcp.server.stdio import stdio_server

    server = build_mcp_server()

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(main())
