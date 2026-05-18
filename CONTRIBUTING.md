# Contributing to vecgrep

Thanks for considering a contribution.

## Before you start

- Open an issue first for anything bigger than a bug fix or doc tweak.
- One logical change per PR. Bundling unrelated changes makes review slow and rollbacks painful.

## Workflow

```bash
git clone https://github.com/jeffbai996/vecgrep.git
cd vecgrep
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# run tests
pytest

# format
ruff format .

# lint
ruff check .
```

## Test discipline

New behavior needs a test. Pure functions get unit tests in `tests/`. CLI surface changes get end-to-end tests that drive the actual `vecgrep` binary against a tempdir corpus.

If you add or change a corpus adapter, include a tiny sample input file alongside the test so reviewers can see what shape the adapter expects.

## Commit messages

Conventional commits, one line under ~70 chars:

- `feat: …` new user-visible behavior
- `fix: …` bug fix
- `refactor: …` no behavior change
- `docs: …` documentation only
- `test: …` tests only
- `chore: …` build / deps / CI / housekeeping
- `perf: …` performance work
- `release: …` version bumps

Body in the imperative; explain the *why* not the *what*. Keep one logical change per commit so `git bisect` stays useful.
