# Maintenance release candidate checklist

The current package version remains `1.2.0`. `CHANGELOG.md` collects the
unreleased candidate; this checklist neither reserves a version nor publishes
anything. Review the accumulated API/UI additions when choosing the next version.

The candidate includes bounded search/MCP admission, corpus exploration,
explicit SQLite lexical storage, import validation, OAuth boundary repairs,
cache test stabilization, decayed ordering, and length-aware reranker batches.
Existing CLI/HTTP/MCP contracts and pickle compatibility remain supported.
SQLite is an opt-in alternative: inspect the running process configuration and
per-corpus sidecars/migration evidence before deciding a deployment's backend.
The README default alone does not establish production state. Conversion and
reconciliation are separate operations requiring verified parity and backups.

Validation gate (use isolated task state):

```sh
python -m pip install -e '.[dev,mcp]' build
VECGREP_HOME="$(mktemp -d)" python -m pytest tests
(cd vecgrep/frontend && npm ci && npm run build)
python -m build
python -m venv /tmp/vecgrep-wheel-smoke
/tmp/vecgrep-wheel-smoke/bin/pip install dist/*.whl
(cd /tmp && /tmp/vecgrep-wheel-smoke/bin/vecgrep --version)
(cd /tmp && /tmp/vecgrep-wheel-smoke/bin/vecgrep --help)
(cd /tmp && VECGREP_HOME="$(mktemp -d)" /tmp/vecgrep-wheel-smoke/bin/python \
  -c 'from vecgrep.backend.main import app; assert app is not None')
```

Record the tested commit/tree, checkout cleanliness, actual suite counts,
individual gate exits, dependency/platform identity and wheel hash in the
private task evidence. A green historical main run does not validate a later
candidate. The workflow performs the same test/frontend/package/smoke checks
on PR and branch pushes. Its publish job requires a version tag or explicit
release dispatch; branch checks alone do not publish a package.

Before release, inspect the candidate's hosted job steps and distribution
contents, ensure the packaged frontend was built from that candidate, and
complete the offline benchmark if making any batching speedup claim. A
`NOT_RUN` benchmark leaves performance unverified; deterministic batch/score
regressions still apply. No new latency claim belongs in release notes without
representative uncached evidence and retrieval/negative-query parity.

Proposed release note wording is the current `Unreleased` changelog. Do not
label a prepared wheel, pushed PR or passing gate as deployed. Creating a
release tag, dispatching publication and updating running services each require
release authorization beyond validation work.
