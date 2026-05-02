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


def _api_alive() -> bool:
    try:
        r = httpx.get(f"{_api_base()}/api/health", timeout=0.4)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _post(path: str, payload: dict) -> Any:
    r = httpx.post(f"{_api_base()}{path}", json=payload, timeout=600.0)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise click.ClickException(detail)
    return r.json()


def _get(path: str) -> Any:
    r = httpx.get(f"{_api_base()}{path}", timeout=30.0)
    if r.status_code >= 400:
        raise click.ClickException(r.text)
    return r.json()


def _delete(path: str) -> Any:
    r = httpx.delete(f"{_api_base()}{path}", timeout=30.0)
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
        click.echo(f"\n[{i}] {pct:5.1f}%  {corpus}  {sid}")
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
def index(source: str, corpus: str, chunker: str, ephemeral: bool) -> None:
    """Index a file, directory, or URL into a corpus."""
    if ephemeral:
        # Ephemeral runs don't go through the API server (which has its own
        # persistent state). They're always in-process.
        svc = VecgrepService(ephemeral=True)
        try:
            docs, chunks = svc.index(source, corpus, chunker)
        except (AdapterError, CorpusError, EmbedBackendError) as e:
            raise click.ClickException(str(e))
        click.echo(f"indexed (ephemeral): {docs} doc(s), {chunks} chunk(s)")
        return

    if _api_alive():
        out = _post("/api/index", {"source": source, "corpus": corpus, "chunker": chunker})
        click.echo(f"indexed: {out['docs']} doc(s), {out['chunks']} chunk(s)")
        return

    # No server running -> run in-process. Persisting to disk because this
    # is not ephemeral; subsequent runs will see the corpus.
    svc = VecgrepService(ephemeral=False)
    try:
        docs, chunks = svc.index(source, corpus, chunker)
    except (AdapterError, CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
    click.echo(f"indexed: {docs} doc(s), {chunks} chunk(s)")


@cli.command()
@click.argument("query")
@click.option("--corpus", default=None, help="Search one corpus (default: all).")
@click.option("--top", "top_k", default=None, type=int, help="Max results.")
@click.option("--json", "json_out", is_flag=True, help="Emit JSON.")
def search(query: str, corpus: str | None, top_k: int | None, json_out: bool) -> None:
    """Semantic search across one or all corpora."""
    if _api_alive():
        out = _post("/api/search", {"query": query, "corpus": corpus, "top_k": top_k})
        _print_results(out["hits"], json_out)
        return
    svc = VecgrepService(ephemeral=False)
    try:
        results = svc.search(query, corpus, top_k)
    except (CorpusError, EmbedBackendError) as e:
        raise click.ClickException(str(e))
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
