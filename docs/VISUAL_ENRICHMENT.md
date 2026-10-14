# Visual document enrichment

Visual enrichment is optional and asynchronous. Native PDF text continues to
index immediately. Images and scans queue for OCR and visual descriptions.
Search uses the existing corpus embedding model; this does not introduce an
image-vector index or change existing embeddings.

Install `pip install -e '.[visual]'` on the services host and install Tesseract
with the OCR language packs needed for your documents. Enable
`VECGREP_VISUAL_ENABLED=1` in the indexing server environment. Images supported:
PNG, JPEG, WebP, TIFF and BMP. Index them through the normal CLI or API.

Run the consumer on the **same services host and VECGREP_HOME as the indexer**:

```sh
python -m vecgrep.backend.ingestion.visual_worker \
  --vision-url http://vision-worker.example:11434 --model qwen3.5:2b
```

The vision endpoint is an Ollama-compatible `/api/chat` service. Restrict its
network listener to your trusted private network. Requests use one image at a
time, 4096 context tokens, at most 900 output tokens, and a 60-second idle model
expiry. The consumer uses the normal authenticated index API for delivery;
set `VECGREP_API_TOKEN` when that API requires it.

For a bounded manual run use `--once`; inspect jobs with `--status`. Without a
vision endpoint, OCR still runs and its text is delivered, while visual
interpretation remains pending. `--language eng` selects Tesseract languages;
for example `eng+chi_sim` requires both language packs.

The SQLite queue and extracted results live under `VECGREP_HOME/visual/`.
Temporary rendered pages are placed there too and removed after OCR. Originals
stay at their source paths. Images and pages are resized to at most 1800 pixels
on the longest side; a document is limited to 40 pages and a source to 200 MiB.
Larger documents remain pending with an explicit error; split them before
indexing if enrichment is needed. PDFs with selectable text still index it.

Jobs are keyed by source path, source SHA-256 and processor version. Completed
results survive consumer restarts. Changed or missing sources reject stale
results. Failed workers retry with backoff up to one hour. An expired processing
lease can be reclaimed after two hours. Failed index delivery remains pending.
Deleting a source or corpus cancels its delivery targets; delayed results cannot
recreate them. An explicit later index can register a target again.

Keep the originals, queue and index on a services host. A remote vision node
holds only models and temporary request data. Its outage delays new visual
descriptions without preventing native text ingestion, CPU OCR or searches over
completed enrichment. Existing descriptions remain available without that node.

PDF results include page markers. Metadata preserves the original source path
and identifies generated enrichment. Descriptions label themselves as generated;
use the original chart for exact numerical claims. Text printed in an image is
untrusted document content, not instructions for the worker.

Increment `VECGREP_VISUAL_VERSION` when intentionally changing the processing
recipe. Re-index selected sources to enqueue their new version. This does not
scan or backfill existing corpora automatically. Backfill selected PDFs/images
through normal index calls with this feature enabled; avoid `--force`, since
unchanged native text and cached embeddings do not need recomputation.

To disable new work, remove `VECGREP_VISUAL_ENABLED` and stop the consumer.
Previously indexed descriptions stay searchable. Do not re-index enriched PDFs
with enrichment disabled unless you intend to replace their derived text with
native text only.
