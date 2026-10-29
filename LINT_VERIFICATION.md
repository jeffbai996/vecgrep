# Lint acceptance verification

The pre-existing lint integration blocker is cleared: **401 findings before,
0 after**, with no new diagnostics. The cleanup was already present in
`origin/main` at `4fda7b5`; this branch adds the missing verification record.
No source, test, dependency, or lint-configuration changes are introduced here.

## Compared revisions

- Before cleanup: `8b61a556d6a4f893484387405508aace7cc42299`.
- After cleanup: `4fda7b5d975c63cacdd55f1c498b1c1829ff9944`.
- Upstream cleanup commit: `style: make the repository lint gate reproducible`.
- This branch contains one documentation/evidence commit, rather than rewriting
  the already-landed cleanup into rule-family commits.

## Acceptance results

| Check | Before | After |
| --- | --- | --- |
| Whole-repository Ruff 0.17.3 | 401 findings | 0 findings |
| Full pytest suite | 1594 passed, 0 skipped | 1594 passed, 0 skipped |
| Dependency warnings | 1 | 1 |
| Deterministic retrieval | 48 query/store/mode combinations | Byte-identical output |
| Package distributions | Not rebuilt | Wheel and source distribution built |
| Isolated base-wheel smoke | Not installed | Passed |
| Installed Python module bytes | Not installed | 83 modules match source |

The seven tests skipped in the supplied integration receipt run here because
the host-side weekly evaluation script is available. There are no added or
removed test cases. Both full runs use Python 3.12 and identical installed
dependencies. The one warning is the same FastAPI/Starlette `httpx` deprecation
in both runs.

Full-suite times: before 101.80s (0:01:41); after 101.35s (0:01:41).
Both processes exited successfully. Test homes and temporary files were isolated
from live state. Commands were equivalent to:

```bash
PYTHONHASHSEED=0 python -m pytest -q -ra --basetemp=<isolated-temp-directory>
ruff check . --no-cache
python -m build
python -m venv <isolated-wheel-environment>
<isolated-wheel-environment>/bin/pip install <built-wheel>
<isolated-wheel-environment>/bin/python -I scripts/smoke_base_install.py
```

## Deterministic retrieval comparison

The existing `tests.eval_harness` golden queries and `StubEmbed` were used
without changing query expectations. Each revision indexed the same four
public synthetic fixture files into 55 chunks per store. All eight queries
ran in `hybrid`, `vector`, and `bm25` modes against both `pickle` and `sqlite`.

Both processes used a shared fixture path, `PYTHONHASHSEED=0`, and a fixed
service clock (`1768564800`) so paths, chunk IDs, recency, and date filtering
were comparable. The fixture root was normalized to `<fixtures>` in output.
The comparison includes complete ordered result records: source and chunk IDs,
text, context, metadata, scores, similarity, spans, source dates, and metrics.
It does not compare elapsed execution time.

The complete before/after JSON files are byte-identical. Their SHA-256 is:

```text
f9346c7ebeddab0e9fab80e8dc48ed55d988400f404b6f94c652e3c485217f73
```

[The compact evaluation record](docs/lint-eval-20261029.json) contains metrics
for all 48 combinations. The full output and runner are retained with the
local task evidence.

## Package acceptance

`python -m build` produced both `vecgrep-1.2.0.tar.gz` and
`vecgrep-1.2.0-py3-none-any.whl`. An isolated virtual environment installed the
wheel with base dependencies only. Running the repository's base-install smoke
with `python -I` passed, including the bundled frontend, text ingestion and
native PDF ingestion without optional visual or MCP extras. All 83 installed
Python modules were byte-compared with the source and matched.

The base environment logs that optional MCP support is disabled, as expected.
Setuptools reports existing packaging/deprecation warnings; the distribution
build and import checks still pass.

## Retained exceptions and scope limits

[RECON.md](RECON.md) inventories every rule and affected file, including Ruff's
safe/unsafe/manual fixability classification. [DEFERRED.md](DEFERRED.md) lists
85 upstream targeted annotation locations and the malformed parser fixture.
These retained boundaries must not be described as removed exception handling.
The two malformed-input syntax diagnostics are excluded by the upstream
configuration; this branch makes no configuration changes.

No deployment, restart, reindex, live environment edit, landing command,
merge to main, force push, or production data mutation was performed.
