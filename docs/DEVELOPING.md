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

## Why no tests?

v0.1 is intentionally test-free per spec. Once the API stabilizes (v0.2),
add pytest tests for the service layer first — that's where bugs hide.
