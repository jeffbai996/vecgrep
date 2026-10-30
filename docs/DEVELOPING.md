# Developing vecgrep

## Layout

```
vecgrep/
├── vecgrep/                # python package
│   ├── backend/            # FastAPI app + service layer
│   ├── cli/                # Click CLI (entry point: `vecgrep`)
│   └── frontend/           # React + Tailwind, built to dist/
└── pyproject.toml
```

## Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[openai]"
vecgrep --help
```

The CLI auto-detects whether the API server is running. If it is, commands
go over HTTP; if not, they run in-process. Either way the user sees the
same output.

To run the server:

```bash
vecgrep serve --reload
# -> http://127.0.0.1:8765
```

## Frontend

```bash
cd vecgrep/frontend
npm install
npm run dev    # Vite dev server on :5173, proxies /api to backend
npm run build  # outputs to dist/, picked up by FastAPI when serving
```

Two-process dev loop:

1. Terminal A: `vecgrep serve --reload`
2. Terminal B: `cd vecgrep/frontend && npm run dev`

Visit http://localhost:5173. Vite proxies `/api/*` to the FastAPI server.

For a single-process production-ish setup:

1. `cd vecgrep/frontend && npm run build`
2. `vecgrep serve`

The FastAPI app detects `frontend/dist/` and serves the SPA from `/`.

## Adding an adapter

1. Create `vecgrep/backend/ingestion/adapters/<your_adapter>.py`
2. Subclass `Adapter`, implement `matches()` and `load()`
3. Decorate the class with `@register_adapter`
4. Import it in `vecgrep/backend/ingestion/adapters/__init__.py` so the
   import side-effect registers it

The `detect_adapter()` function will pick yours up automatically as long
as `matches()` returns True for the given source string.

## Adding a chunker

1. Create `vecgrep/backend/ingestion/chunkers/<your_chunker>.py`
2. Subclass `Chunker`, implement `chunk()`
3. Add it to the `CHUNKERS` registry in `vecgrep/backend/service.py`
4. Add the name to the `--chunker` CLI choice list in
   `vecgrep/cli/main.py`

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The dev extra includes the image/PDF libraries required by the visual fixture
tests. Add the `mcp` and `code` extras when running the complete HTTP, OAuth
and parser coverage: `pip install -e ".[dev,mcp,code]"`.


Hermetic — no Ollama, no network, no shared `~/.vecgrep`. Every test
gets its own `VECGREP_HOME` under `tmp_path` and a deterministic stub
embedding backend (see `tests/conftest.py`). Coverage spans the service
layer, BM25 store, adapters, embedding cache, export/import, and
backend/model migration — i.e. the surfaces where shipped bugs have
historically lived. New extension points (adapters, chunkers, embed
backends) should land with a unit test in the matching `test_*.py`.

## Lint & Typecheck

```bash
pip install -e ".[dev,mcp,code,openai,watch]"
ruff check .
pyright
```

Install the dev and runtime extras and run `ruff check .` and `pyright`. The
dev extra pins Ruff (`>=0.17.3,<0.18`) and Pyright (`pyright[nodejs]==1.1.408`,
configured in `[tool.pyright]` under `typeCheckingMode = "standard"`) so
neither gate depends on a developer's global tool or Node installation. When
invoking `pyright` outside an activated virtualenv, pass `--pythonpath
./venv/bin/python`. The only excluded Python source is
`tests/fixtures/code_symbols/broken.py`, deliberately invalid input used by
the parser-failure tests.

Broad exception handlers are retained only at documented backend, worker,
cleanup and protocol boundaries. Their local annotations preserve established
failure isolation and error responses; new ordinary application code should
catch the errors it can handle. Per-line `# pyright: ignore[rule]` comments are
restricted to optional `rerank` extra imports (`torch`,
`sentence_transformers`) and third-party stub mismatches, always accompanied by
a one-line reason.
