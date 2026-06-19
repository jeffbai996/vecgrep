"""vecgrep CLI.

Talks to the local FastAPI backend if it is running on settings.api_port,
otherwise spins up an in-process VecgrepService. The user shouldn't have
to think about which mode they're in.

Examples:
    vecgrep index ./docs --corpus papers
    vecgrep index https://example.com/x --corpus web
    vecgrep search "rate hikes" --corpus papers
    vecgrep corpora list
    vecgrep corpora delete papers
    vecgrep serve
"""
from __future__ import annotations

import fnmatch
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import httpx

from .. import __version__
from ..backend.config import get_settings
from ..backend.embed import EmbedBackendError
from ..backend.ingestion.adapters import AdapterError
from ..backend.service import VecgrepService
from ..backend.store import CorpusError


def _api_base() -> str:
    s = get_settings()
    return f"http://{s.api_host}:{s.api_port}"


def _auth_headers() -> dict:
    token = get_settings().api_token
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api_alive() -> bool:
    try:
        r = httpx.get(f"{_api_base()}/api/health", timeout=0.4)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _post(path: str, payload: dict) -> Any:
    r = httpx.post(
        f"{_api_base()}{path}",
        json=payload,
        timeout=600.0,
        headers=_auth_headers(),
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise click.ClickException(detail)
    return r.json()


def _get(path: str) -> Any:
    r = httpx.get(f"{_api_base()}{path}", timeout=30.0, headers=_auth_headers())
    if r.status_code >= 400:
        raise click.ClickException(r.text)
    return r.json()


def _delete(path: str) -> Any:
    r = httpx.delete(
        f"{_api_base()}{path}", timeout=30.0, headers=_auth_headers()
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise click.ClickException(detail)
    return r.json()


# ----- output helpers --------------------------------------------------------
def _highlight(text: str) -> str:
    # The match is the whole chunk — bold it via ANSI when stdout is a TTY,
    # otherwise leave plain so pipes stay clean.
    if not sys.stdout.isatty():
        return text
    return f"\033[1m{text}\033[0m"


def _print_results(hits: list[dict], json_out: bool) -> None:
    if json_out:
        click.echo(json.dumps(hits, indent=2))
        return
    if not hits:
        click.echo("no matches.")
        return
    for i, h in enumerate(hits, 1):
        pct = h["similarity_pct"]
        sid = h["source_id"]
        corpus = h["corpus"]
        matched_by = h.get("matched_by") or []
        badge = f"  [{'+'.join(matched_by)}]" if matched_by else ""
        click.echo(f"\n[{i}] {pct:5.1f}%  {corpus}  {sid}{badge}")
        explain = h.get("explain") or {}
        if explain:
            parts: list[str] = []
            if "vector_cosine" in explain:
                parts.append(f"vec={explain['vector_cosine']:.3f}@#{explain.get('vector_rank','?')}")
            if "bm25_score" in explain:
                parts.append(f"bm25={explain['bm25_score']:.3f}@#{explain.get('bm25_rank','?')}")
            if "rrf" in explain:
                parts.append(f"rrf={explain['rrf']:.4f}")
            if "rerank_score" in explain:
                parts.append(f"rerank={explain['rerank_score']:.3f}")
            if parts:
                click.echo(f"    explain: {' '.join(parts)}")
        before = (h.get("context_before") or "").strip()
        after = (h.get("context_after") or "").strip()
        chunk = _highlight(h["chunk"].strip())
        if before:
            click.echo(f"    ... {before}")
        click.echo(f"    {chunk}")
        if after:
            click.echo(f"    {after} ...")


# ----- commands --------------------------------------------------------------
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="vecgrep")
def cli() -> None:
    """grep for meaning, not keywords."""


@cli.command()
@click.argument("source")
@click.option("--corpus", required=True, help="Named corpus to write into.")
@click.option(
    "--chunker",
    default="sentence_window",
    type=click.Choice(["sentence_window", "fixed_token"]),
    show_default=True,
)
@click.option("--ephemeral", is_flag=True, help="Don't persist to disk.")
@click.option(
    "--force",
    is_flag=True,
    help="Re-embed even if content hash matches the previous index.",
)
@click.option(
    "--include",
    default=None,
    help="When indexing a directory, only files whose name matches this glob "
    "(e.g. '*.md' to skip sibling raw files). Applies to directory sources only.",
)
def index(
    source: str, corpus: str, chunker: str, ephemeral: bool, force: bool, include: str | None
) -> None:
    """Index a file, directory, or URL into a corpus.

    Incremental by default: if a source's content hash matches the last
    index, it's skipped. Pass --force to re-embed unconditionally.
    """
    def _format(out: dict | tuple) -> str:
        if isinstance(out, tuple):
            docs, chunks, skipped = out
        else:
            docs, chunks, skipped = out["docs"], out["chunks"], out.get("skipped", 0)
        msg = f"indexed: {docs} doc(s), {chunks} chunk(s)"
        if skipped:
            msg += f", {skipped} unchanged (skipped)"
        return msg

    if ephemeral:
        # Ephemeral runs don't go through the API server (which has its own
        # persistent state). They're always in-process.
        svc = VecgrepService(ephemeral=True)
        try:
            out = svc.index(source, corpus, chunker, force=force, include=include)
        except (AdapterError, CorpusError, EmbedBackendError) as e:
            raise click.ClickException(str(e))
        click.echo(_format(out).replace("indexed", "indexed (ephemeral)"))
        return

    if _api_alive():
        body = {"source": source, "corpus": corpus, "chunker": chunker, "force": force}
        if include:
            body["include"] = include
        out = _post("/api/index", body)
        click.echo(_format(out))
        return

    svc = VecgrepService(ephemeral=False)
    try:
        out = svc.index(source, corpus, chunker, force=force, include=include)
    except (AdapterError, CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
    click.echo(_format(out))


@cli.command()
@click.argument("query")
@click.option("--corpus", default=None, help="Search one corpus (default: all).")
@click.option("--top", "top_k", default=None, type=int, help="Max results.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "vector", "bm25"]),
    show_default=True,
    help="Retrieval strategy. Hybrid fuses BM25 + vector via RRF.",
)
@click.option(
    "--rerank",
    is_flag=True,
    help="Rerank top candidates with a cross-encoder. Needs `pip install vecgrep[rerank]`.",
)
@click.option(
    "--rerank-model",
    default=None,
    help="Override the cross-encoder model (default: BAAI/bge-reranker-base).",
)
@click.option(
    "--filter",
    "filters",
    multiple=True,
    help=(
        "Filter results. Forms: 'source:<glob>', 'corpus:<name>', "
        "'meta.<key>=<value>'. Repeatable; all ANDed."
    ),
)
@click.option(
    "--explain",
    "explain",
    is_flag=True,
    help="Show per-retriever score breakdown for each hit.",
)
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
@click.option(
    "--watch", "-w", is_flag=True,
    help="Re-run the query at --interval and print a diff of source IDs.",
)
@click.option(
    "--interval",
    default=5.0,
    type=float,
    show_default=True,
    help="Seconds between re-runs in --watch mode.",
)
def search(
    query: str,
    corpus: str | None,
    top_k: int | None,
    mode: str,
    rerank: bool,
    rerank_model: str | None,
    filters: tuple[str, ...],
    explain: bool,
    json_out: bool,
    watch: bool,
    interval: float,
) -> None:
    """Semantic search across one or all corpora."""
    filter_list = list(filters)

    def run_once() -> list[dict[str, Any]]:
        if _api_alive():
            out = _post(
                "/api/search",
                {
                    "query": query,
                    "corpus": corpus,
                    "top_k": top_k,
                    "mode": mode,
                    "rerank": rerank,
                    "rerank_model": rerank_model,
                    "filters": filter_list,
                    "explain": explain,
                },
            )
            return out["hits"]
        svc = VecgrepService(ephemeral=False)
        try:
            results = svc.search(
                query, corpus, top_k,
                mode=mode, rerank=rerank, rerank_model=rerank_model,
                filters=filter_list or None, explain=explain,
            )
        except (CorpusError, EmbedBackendError) as e:
            raise click.ClickException(str(e))
        except Exception as e:
            if rerank:
                raise click.ClickException(str(e))
            raise
        return [
            {
                "similarity_pct": r.similarity_pct,
                "chunk": r.chunk,
                "context_before": r.context_before,
                "context_after": r.context_after,
                "source_id": r.source_id,
                "corpus": r.corpus,
                "metadata": r.metadata,
                "matched_by": r.matched_by,
                "explain": r.explain or {},
            }
            for r in results
        ]

    if not watch:
        _print_results(run_once(), json_out)
        return

    # --watch: loop, diff source_ids vs previous iteration, swallow Ctrl-C.
    if json_out:
        click.echo("note: --json is ignored in --watch mode (output is a live diff).", err=True)
    prev_keys: set[tuple[str, str]] | None = None
    iteration = 0
    try:
        while True:
            iteration += 1
            try:
                hits = run_once()
            except KeyboardInterrupt:
                break
            now = datetime.now().strftime("%H:%M:%S")
            click.echo(f"\n--- iteration {iteration} @ {now} — {len(hits)} hit(s) ---")
            cur_keys = {(h["corpus"], h["source_id"]) for h in hits}
            if prev_keys is None:
                # First pass: list every hit, no diff.
                for h in hits:
                    click.echo(f"  {h['similarity_pct']:5.1f}%  {h['corpus']}  {h['source_id']}")
            else:
                added = cur_keys - prev_keys
                removed = prev_keys - cur_keys
                if not added and not removed:
                    click.echo("  (no change)")
                else:
                    by_key = {(h["corpus"], h["source_id"]): h for h in hits}
                    for k in sorted(added):
                        h = by_key[k]
                        click.echo(f"  + {h['similarity_pct']:5.1f}%  {k[0]}  {k[1]}")
                    for k in sorted(removed):
                        click.echo(f"  - ----   {k[0]}  {k[1]}")
            prev_keys = cur_keys
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass


@cli.group()
def corpora() -> None:
    """Manage named corpora."""


@corpora.command("list")
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
def corpora_list(json_out: bool) -> None:
    if _api_alive():
        items = _get("/api/corpora")
    else:
        svc = VecgrepService()
        items = [asdict(c) for c in svc.list_corpora()]
    if json_out:
        click.echo(json.dumps(items, indent=2))
        return
    if not items:
        click.echo("no corpora yet. try `vecgrep index <path> --corpus <name>`.")
        return
    click.echo(f"{'NAME':<20} {'BACKEND':<12} {'MODEL':<28} {'DOCS':>5} {'CHUNKS':>7}")
    for c in items:
        click.echo(
            f"{c['name']:<20} "
            f"{c['embed_backend']:<12} "
            f"{c['embed_model']:<28} "
            f"{c['doc_count']:>5} "
            f"{c['chunk_count']:>7}"
        )


@corpora.command("export")
@click.argument("name")
@click.option(
    "--out",
    "out",
    default=None,
    help="Output path. Defaults to ./<name>.vecgrep.tar.gz",
)
def corpora_export(name: str, out: str | None) -> None:
    """Export a corpus to a portable .tar.gz."""
    dest = Path(out) if out else Path.cwd() / f"{name}.vecgrep.tar.gz"
    svc = VecgrepService()
    try:
        path = svc.export_corpus(name, dest)
    except (CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
    click.echo(f"exported: {path}")


@corpora.command("import")
@click.argument("archive")
@click.option("--rename", default=None, help="Restore under a different corpus name.")
def corpora_import(archive: str, rename: str | None) -> None:
    """Restore a corpus from a .tar.gz produced by `corpora export`."""
    svc = VecgrepService()
    try:
        corpus = svc.import_corpus(Path(archive), rename=rename)
    except CorpusError as e:
        raise click.ClickException(str(e))
    click.echo(f"imported: {corpus.name} ({corpus.doc_count} doc(s), {corpus.chunk_count} chunk(s))")


@corpora.command("migrate")
@click.argument("name")
@click.option(
    "--to-backend",
    required=True,
    type=click.Choice(["ollama", "openai"]),
    help="Target embedding backend.",
)
@click.option("--to-model", default=None, help="Target embedding model. Defaults to backend's default.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def corpora_migrate(name: str, to_backend: str, to_model: str | None, yes: bool) -> None:
    """Re-embed a corpus to a new backend / model.

    Migration re-fetches every original source. URLs and files that no
    longer exist are skipped with a warning. The old corpus is replaced
    in place once the new one is fully built — partial failure leaves
    the original untouched.
    """
    if not yes:
        target = f"{to_backend}/{to_model}" if to_model else to_backend
        click.confirm(
            f"migrate corpus '{name}' to {target}? this re-fetches every source.",
            abort=True,
        )
    svc = VecgrepService()
    try:
        out = svc.migrate_corpus(name, to_backend, to_model)
    except (CorpusError, EmbedBackendError, AdapterError) as e:
        raise click.ClickException(str(e))
    click.echo(
        f"migrated: {out.name} -> {out.embed_backend}/{out.embed_model} "
        f"({out.doc_count} doc(s), {out.chunk_count} chunk(s))"
    )


@corpora.command("delete")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def corpora_delete(name: str, yes: bool) -> None:
    if not yes:
        click.confirm(f"delete corpus '{name}'? this is irreversible.", abort=True)
    if _api_alive():
        _delete(f"/api/corpora/{name}")
    else:
        svc = VecgrepService()
        try:
            svc.delete_corpus(name)
        except CorpusError as e:
            raise click.ClickException(str(e))
    click.echo(f"deleted: {name}")


@corpora.command("decay")
@click.argument("name")
@click.option(
    "--half-life",
    "half_life",
    type=float,
    default=None,
    help="Half-life in days. A hit one half-life old ranks as half as relevant. "
    "Omit to DISABLE decay for this corpus.",
)
@click.option("--off", is_flag=True, help="Disable recency decay for this corpus.")
def corpora_decay(name: str, half_life: float | None, off: bool) -> None:
    """Set or clear a corpus's recency-decay half-life (no re-index needed)."""
    value = None if off else half_life
    if _api_alive():
        c = _post(f"/api/corpora/{name}/decay", {"half_life_days": value})
        hl = c.get("decay_half_life_days")
    else:
        svc = VecgrepService()
        try:
            corpus = svc.set_decay(name, value)
        except CorpusError as e:
            raise click.ClickException(str(e))
        hl = corpus.decay_half_life_days
    if hl:
        click.echo(f"{name}: recency decay half-life = {hl:g} days")
    else:
        click.echo(f"{name}: recency decay disabled")


@cli.command()
@click.option("--host", default=None, help="Override host.")
@click.option("--port", default=None, type=int, help="Override port.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev).")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the FastAPI backend (and serve the web UI if built)."""
    import uvicorn

    s = get_settings()
    # `timeout_keep_alive` defaults to 5s in uvicorn — long-running index
    # calls (which can take minutes on large repos) silently get the
    # connection axed before the registry-write response is delivered.
    # The server side completes the work but the CLI sees httpx.ReadTimeout
    # and the registry never records the new corpus. Bump to 15 min.
    uvicorn.run(
        "vecgrep.backend.main:app",
        host=host or s.api_host,
        port=port or s.api_port,
        reload=reload,
        timeout_keep_alive=900,
    )


@cli.command()
@click.argument("path")
@click.option("--corpus", required=True, help="Named corpus to keep current.")
@click.option(
    "--chunker",
    default="sentence_window",
    type=click.Choice(["sentence_window", "fixed_token"]),
    show_default=True,
)
@click.option(
    "--debounce",
    default=1.0,
    show_default=True,
    type=float,
    help="Coalesce events arriving within this window (seconds).",
)
@click.option(
    "--include",
    default=None,
    help="Only index files whose name matches this glob (e.g. '*.md' to skip "
    "sibling raw files). Applies to the initial pass and to change events.",
)
def watch(path: str, corpus: str, chunker: str, debounce: float, include: str | None) -> None:
    """Watch a directory and re-index on change.

    Re-indexes incrementally — only sources whose content hash changed get
    re-embedded. Press Ctrl+C to stop.

    Note: file-system events depend on the OS. Linux uses inotify; WSL2's
    inotify on /tmp can be flaky on some kernels (fine on /home). macOS
    and most Linux native filesystems work as expected.
    """
    try:
        from watchfiles import watch as _watch
    except ImportError:
        raise click.ClickException(
            "watch requires `watchfiles`. Install with `pip install \"vecgrep[watch]\"`."
        )

    target = Path(path).resolve()
    if not target.is_dir():
        raise click.ClickException(f"watch target must be a directory: {target}")

    filt = f" (include: {include})" if include else ""
    click.echo(f"watching {target} -> corpus '{corpus}'{filt} (Ctrl+C to stop)")
    # Initial pass picks up everything currently on disk.
    _do_index(str(target), corpus, chunker, force=False, include=include)

    try:
        for changes in _watch(str(target), step=int(debounce * 1000)):
            kinds = {kind.name for kind, _ in changes}
            paths = sorted({p for _, p in changes})
            click.echo(f"  ! {len(paths)} change(s) [{','.join(sorted(kinds))}] — processing")
            for change_kind, p in changes:
                # Respect the include glob on per-file events too, so a sibling
                # raw file changing doesn't get indexed into a markdown-only corpus.
                if include and not fnmatch.fnmatch(Path(p).name, include):
                    continue
                try:
                    if change_kind.name == "deleted":
                        _do_delete_source(p, corpus)
                    else:
                        _do_index(p, corpus, chunker, force=False, include=include)
                except click.ClickException as e:
                    # Don't kill the watcher on a transient error — log and keep going.
                    click.echo(f"  error: {e.message}", err=True)
    except KeyboardInterrupt:
        click.echo("\nstopped.")


def _do_delete_source(source: str, corpus: str) -> None:
    if _api_alive():
        _delete(f"/api/corpora/{corpus}/source/{source}")
        click.echo(f"  deleted {source}")
        return
    svc = VecgrepService(ephemeral=False)
    try:
        svc.delete_source(corpus, source)
    except (CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
    click.echo(f"  deleted {source}")


def _do_index(source: str, corpus: str, chunker: str, force: bool, include: str | None = None) -> None:
    """Shared index path used by `index` and `watch`. Prints, doesn't raise."""
    if _api_alive():
        body = {"source": source, "corpus": corpus, "chunker": chunker, "force": force}
        if include:
            body["include"] = include
        out = _post("/api/index", body)
        msg = f"  indexed {out['docs']} doc(s), {out['chunks']} chunk(s)"
        if out.get("skipped"):
            msg += f", {out['skipped']} unchanged"
        click.echo(msg)
        return
    svc = VecgrepService(ephemeral=False)
    try:
        docs, chunks, skipped = svc.index(source, corpus, chunker, force=force, include=include)
    except (AdapterError, CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
    msg = f"  indexed {docs} doc(s), {chunks} chunk(s)"
    if skipped:
        msg += f", {skipped} unchanged"
    click.echo(msg)


@cli.group()
def cache() -> None:
    """Inspect and manage the embedding cache."""


@cache.command("stats")
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
def cache_stats(json_out: bool) -> None:
    """Show cached vector counts per (backend, model)."""
    from ..backend.embed.cache import EmbedCache

    s = get_settings()
    db = s.home / "embed_cache.db"
    if not db.exists():
        click.echo("no cache yet.")
        return
    cache = EmbedCache(db)
    stats = cache.stats()
    if json_out:
        click.echo(json.dumps(stats, indent=2))
        return
    if not stats:
        click.echo("cache is empty.")
        return
    click.echo(f"{'IDENTITY':<40} {'COUNT':>10}")
    for ident, count in sorted(stats.items()):
        click.echo(f"{ident:<40} {count:>10}")


@cache.command("clear")
@click.option("--identity", default=None, help="Clear only this identity (e.g. 'ollama:nomic-embed-text').")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def cache_clear(identity: str | None, yes: bool) -> None:
    """Drop cached vectors. Useful to free disk; embedded vectors will be
    refetched on next embed call.
    """
    from ..backend.embed.cache import EmbedCache

    s = get_settings()
    db = s.home / "embed_cache.db"
    if not db.exists():
        click.echo("no cache to clear.")
        return
    target = f"identity={identity}" if identity else "ALL identities"
    if not yes:
        click.confirm(f"clear cache ({target})?", abort=True)
    cache = EmbedCache(db)
    n = cache.clear(identity)
    click.echo(f"cleared: {n} entries")


@cli.command()
def mcp() -> None:
    """Run vecgrep as an MCP server over stdio.

    Configure your MCP client (Claude Desktop, Cursor, etc.) to launch this
    command. Tools exposed: search, list_corpora, get_corpus.
    """
    try:
        from ..mcp.server import run as run_mcp
    except RuntimeError as e:
        raise click.ClickException(str(e))
    run_mcp()


def _relative_age(ts: float) -> str:
    """Render a unix timestamp as a short relative age ('5m', '3h', '2d').

    Returns 'never' for falsy / zero timestamps so a missing updated_at
    doesn't render as '57y' or similar nonsense.
    """
    if not ts or ts <= 0:
        return "never"
    import time

    delta = time.time() - ts
    if delta < 0:
        return "now"
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86400:
        return f"{int(delta // 3600)}h"
    return f"{int(delta // 86400)}d"


@cli.command()
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
def status(json_out: bool) -> None:
    """One-shot health snapshot: daemon, auth, and per-corpus chunk counts.

    Useful first thing when something looks off — e.g. why a remote MCP
    client is seeing different results than local. If the daemon is up,
    we hit /api/corpora; otherwise we synthesize the view from a local
    VecgrepService so the command works the same either way.
    """
    s = get_settings()
    daemon_alive = _api_alive()

    corpora_err: str | None = None
    corpora: list[dict[str, Any]] = []
    if daemon_alive:
        try:
            corpora = _get("/api/corpora")
        except click.ClickException as e:
            corpora_err = e.message
    else:
        try:
            svc = VecgrepService()
            corpora = [asdict(c) for c in svc.list_corpora()]
        except Exception as e:
            # Most common failure: the qdrant lock file is held by the
            # daemon's process that we just thought was dead — surface
            # that as an explicit error rather than an empty list.
            corpora_err = str(e)

    totals = {
        "docs": sum(c.get("doc_count", 0) for c in corpora),
        "chunks": sum(c.get("chunk_count", 0) for c in corpora),
    }
    payload = {
        "version": __version__,
        "daemon": {
            "alive": daemon_alive,
            "url": _api_base(),
        },
        "auth": {"enabled": bool(s.api_token)},
        "home": str(s.home),
        "embed": {
            "backend": "openai" if s.openai_api_key else "ollama",
            "model": s.embed_model,
        },
        "corpora": corpora,
        "totals": totals,
        "error": corpora_err,
    }

    if json_out:
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    click.echo(f"vecgrep {__version__}")
    click.echo(f"  daemon:  {'up' if daemon_alive else 'down'}    ({payload['daemon']['url']})")
    click.echo(f"  auth:    {'enabled (token set)' if s.api_token else 'disabled (no token)'}")
    click.echo(f"  home:    {payload['home']}")
    click.echo(f"  embed:   {payload['embed']['backend']} / {payload['embed']['model']}")
    if corpora_err:
        click.echo(f"  error:   {corpora_err}")
    click.echo("")
    if not corpora:
        click.echo("no corpora yet. try `vecgrep index <path> --corpus <name>`.")
        return
    click.echo("corpora:")
    click.echo(
        f"  {'NAME':<20} {'BACKEND':<10} {'MODEL':<28} {'DOCS':>6} {'CHUNKS':>8} {'UPDATED':>8}"
    )
    for c in corpora:
        click.echo(
            f"  {c['name']:<20} "
            f"{c['embed_backend']:<10} "
            f"{c['embed_model']:<28} "
            f"{c['doc_count']:>6} "
            f"{c['chunk_count']:>8} "
            f"{_relative_age(c.get('updated_at') or 0):>8}"
        )
    click.echo(
        f"  {'':<20} {'':<10} {'TOTAL':<28} "
        f"{totals['docs']:>6} {totals['chunks']:>8} {'':>8}"
    )


@cli.command()
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
def config(json_out: bool) -> None:
    """Show current resolved configuration."""
    if _api_alive():
        cfg = _get("/api/config")
    else:
        s = get_settings()
        cfg = {
            "home": str(s.home),
            "ollama_url": s.ollama_url,
            "embed_model": s.embed_model,
            "openai_configured": bool(s.openai_api_key),
            "api_host": s.api_host,
            "api_port": s.api_port,
            "default_top_k": s.default_top_k,
        }
    if json_out:
        click.echo(json.dumps(cfg, indent=2))
        return
    for k, v in cfg.items():
        click.echo(f"{k:<20} {v}")


def _write_dir(corpus: str):
    """Per-corpus dir where write-tool markdown docs live (files-are-truth)."""
    return get_settings().home / "write" / corpus


def _do_write(corpus: str, content: str, edit_id: str | None,
              source_kind: str | None, tags: tuple[str, ...]) -> None:
    """Shared propose→confirm for the write/edit CLI. The CLI caller is the
    human, so confirmed_by = the local user (the wall's human-confirm). origin
    is 'human' for the same reason."""
    import getpass
    from ..backend.write import proposal as _P
    from ..backend.write import confirm as _C

    corpus_dir = _write_dir(corpus)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta = {"origin": "human"}
    if source_kind:
        meta["source_kind"] = source_kind
    if tags:
        meta["tags"] = list(tags)
    try:
        pr = _P.propose(corpus, content, corpus_dir, meta=meta, edit_id=edit_id)
    except _P.ProposalError as e:
        raise click.ClickException(str(e))

    store = _C.ProposalStore(get_settings().home / "write" / "_pending")
    store.put(pr)
    svc = VecgrepService(ephemeral=False)
    try:
        res = _C.confirm(pr.proposal_id, store, svc, corpus, corpus_dir,
                         confirmed_by=getpass.getuser(),
                         protected_ack=pr.doc_id)  # CLI user IS the human; auto-ack
    except _C.ConfirmError as e:
        raise click.ClickException(str(e))
    status = "✓" if res.ok else "⚠"
    click.echo(f"{status} {pr.doc_id} → {res.path}")
    if not res.ok:
        click.echo(f"  {res.message}")


@cli.command()
@click.argument("corpus")
@click.argument("content")
@click.option("--source-kind", type=click.Choice(
    ["insight", "fact", "correction", "journal", "decision"]),
    default=None, help="What kind of entry this is (powers filtered retrieval).")
@click.option("--tag", "tags", multiple=True, help="Tag (repeatable).")
def write(corpus: str, content: str, source_kind: str | None, tags: tuple[str, ...]) -> None:
    """Write a NEW entry into a corpus (proposes + confirms in one step — you're
    the human). Assigns the next id, writes the markdown doc, indexes it."""
    _do_write(corpus, content, None, source_kind, tags)


@cli.command()
@click.argument("doc_id")
@click.argument("content")
@click.option("--corpus", default=None, help="Corpus (default: inferred from doc_id prefix).")
@click.option("--source-kind", type=click.Choice(
    ["insight", "fact", "correction", "journal", "decision"]), default=None)
@click.option("--tag", "tags", multiple=True)
def edit(doc_id: str, content: str, corpus: str | None,
         source_kind: str | None, tags: tuple[str, ...]) -> None:
    """Overwrite an existing entry by id (e.g. notes-007). Corpus is inferred
    from the id prefix unless --corpus is given."""
    if corpus is None:
        # doc_id is '<prefix>-NNN'; prefix maps to the corpus dir name.
        corpus = doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
    _do_write(corpus, content, doc_id, source_kind, tags)


@cli.command()
def pending() -> None:
    """List pending write proposals awaiting confirmation (from MCP propose_*)."""
    import glob
    pdir = get_settings().home / "write" / "_pending"
    files = sorted(glob.glob(str(pdir / "*.json")))
    if not files:
        click.echo("No pending proposals.")
        return
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        kind = "edit" if d.get("is_edit") else "new"
        click.echo(f"{d.get('proposal_id')}  [{kind}] {d.get('doc_id')} "
                   f"({d.get('corpus')})  origin={d.get('meta', {}).get('origin', '?')}")


@cli.command()
@click.argument("proposal_id")
@click.option("--ack", default=None,
              help="For protected-tier docs: re-state the exact doc id to confirm.")
def confirm(proposal_id: str, ack: str | None) -> None:
    """Confirm a pending write proposal (the human authorization step). This is
    what turns an MCP-proposed entry into an actual write — bots propose, you
    confirm."""
    import getpass
    from ..backend.write import confirm as _C
    from ..backend.write.proposal import _slug_prefix  # noqa: F401 (kept for parity)

    store = _C.ProposalStore(get_settings().home / "write" / "_pending")
    pr = store.get(proposal_id)
    if pr is None:
        raise click.ClickException(f"No pending proposal {proposal_id!r} "
                                   "(see `vecgrep pending`).")
    corpus = pr.corpus
    corpus_dir = get_settings().home / "write" / corpus
    try:
        res = _C.confirm(proposal_id, store, VecgrepService(ephemeral=False),
                         corpus, corpus_dir, confirmed_by=getpass.getuser(),
                         protected_ack=ack or pr.doc_id)
    except _C.ConfirmError as e:
        raise click.ClickException(str(e))
    status = "✓" if res.ok else "⚠"
    click.echo(f"{status} {res.doc_id} → {res.path}")
    if not res.ok:
        click.echo(f"  {res.message}")


@cli.command()
@click.argument("proposal_id")
def discard(proposal_id: str) -> None:
    """Discard a pending write proposal without writing it (the reject path —
    bots propose, you decline). Removes the pending proposal; nothing is
    written or indexed."""
    from ..backend.write import confirm as _C

    store = _C.ProposalStore(get_settings().home / "write" / "_pending")
    pr = store.get(proposal_id)
    if pr is None:
        raise click.ClickException(f"No pending proposal {proposal_id!r} "
                                   "(see `vecgrep pending`).")
    store.delete(proposal_id)
    click.echo(f"✗ discarded {proposal_id} ({pr.doc_id})")


if __name__ == "__main__":
    cli()
