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

import json
import sys
from dataclasses import asdict
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
def index(source: str, corpus: str, chunker: str, ephemeral: bool, force: bool) -> None:
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
            out = svc.index(source, corpus, chunker, force=force)
        except (AdapterError, CorpusError, EmbedBackendError) as e:
            raise click.ClickException(str(e))
        click.echo(_format(out).replace("indexed", "indexed (ephemeral)"))
        return

    if _api_alive():
        out = _post(
            "/api/index",
            {"source": source, "corpus": corpus, "chunker": chunker, "force": force},
        )
        click.echo(_format(out))
        return

    svc = VecgrepService(ephemeral=False)
    try:
        out = svc.index(source, corpus, chunker, force=force)
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
) -> None:
    """Semantic search across one or all corpora."""
    filter_list = list(filters)
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
        _print_results(out["hits"], json_out)
        return
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
        # RerankerError lives in vecgrep.backend.rerank but importing it
        # eagerly would force the optional dep at CLI start. Catch broadly
        # only when --rerank is on.
        if rerank:
            raise click.ClickException(str(e))
        raise
    _print_results(
        [
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
        ],
        json_out,
    )


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


@cli.command()
@click.option("--host", default=None, help="Override host.")
@click.option("--port", default=None, type=int, help="Override port.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev).")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the FastAPI backend (and serve the web UI if built)."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "vecgrep.backend.main:app",
        host=host or s.api_host,
        port=port or s.api_port,
        reload=reload,
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
def watch(path: str, corpus: str, chunker: str, debounce: float) -> None:
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
            "watch requires `watchfiles`. Install with `pip install watchfiles`."
        )

    target = Path(path).resolve()
    if not target.is_dir():
        raise click.ClickException(f"watch target must be a directory: {target}")

    click.echo(f"watching {target} -> corpus '{corpus}' (Ctrl+C to stop)")
    # Initial pass picks up everything currently on disk.
    _do_index(str(target), corpus, chunker, force=False)

    try:
        for changes in _watch(str(target), step=int(debounce * 1000)):
            kinds = {kind.name for kind, _ in changes}
            paths = sorted({p for _, p in changes})
            click.echo(f"  ! {len(paths)} change(s) [{','.join(sorted(kinds))}] — reindexing")
            try:
                _do_index(str(target), corpus, chunker, force=False)
            except click.ClickException as e:
                # Don't kill the watcher on a transient error — log and keep going.
                click.echo(f"  error: {e.message}", err=True)
    except KeyboardInterrupt:
        click.echo("\nstopped.")


def _do_index(source: str, corpus: str, chunker: str, force: bool) -> None:
    """Shared index path used by `index` and `watch`. Prints, doesn't raise."""
    if _api_alive():
        out = _post(
            "/api/index",
            {"source": source, "corpus": corpus, "chunker": chunker, "force": force},
        )
        msg = f"  indexed {out['docs']} doc(s), {out['chunks']} chunk(s)"
        if out.get("skipped"):
            msg += f", {out['skipped']} unchanged"
        click.echo(msg)
        return
    svc = VecgrepService(ephemeral=False)
    try:
        docs, chunks, skipped = svc.index(source, corpus, chunker, force=force)
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


if __name__ == "__main__":
    cli()
