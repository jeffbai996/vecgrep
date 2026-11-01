"""Security and compatibility contracts for the release workflow."""

import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"
MANUAL_TAG_REF = (
    "ref: ${{ github.event_name == 'workflow_dispatch' "
    "&& format('refs/tags/{0}', inputs.tag) || github.ref }}"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_workflow_contract() -> None:
    workflow = _workflow_text()

    # Manual releases check out the tag named by the input, and the checkout
    # never persists its credentials.
    checkout_count = workflow.count("uses: actions/checkout@")
    assert checkout_count == 2
    assert workflow.count(MANUAL_TAG_REF) == checkout_count
    assert "ref: ${{ inputs.tag || github.ref }}" not in workflow
    assert workflow.count("persist-credentials: false") == checkout_count

    # Event and publish contract.
    # Branch pushes build main only; every other branch builds through its pull
    # request. Tags stay unfiltered and are never skipped by the docs check.
    assert 'branches:\n      - main\n    tags:\n      - "v*"' in workflow
    assert '"**"' not in workflow
    assert "paths-ignore" not in workflow and "paths:" not in workflow
    assert "needs: changes\n    if: needs.changes.outputs.build == 'true'" in workflow
    assert "workflow_dispatch:" in workflow
    assert "required: true" in workflow
    assert (
        "if: startsWith(github.ref, 'refs/tags/v') "
        "|| github.event_name == 'workflow_dispatch'"
    ) in workflow
    assert "RELEASE_TAG: ${{ inputs.tag || github.ref_name }}" in workflow

    # The test job installs the optional extras; the wheel smoke test keeps a
    # separate base-only environment.
    install = re.search(r'pip install -e "\.\[([^]]+)\]" build', workflow)
    assert install is not None
    assert {"dev", "mcp", "visual", "code"} <= set(install.group(1).split(","))
    assert "cache: pip\n          cache-dependency-path: pyproject.toml" in workflow
    assert "python -m pytest tests" in workflow
    assert "python -m venv /tmp/vecgrep-smoke" in workflow
    assert "/tmp/vecgrep-smoke/bin/pip install dist/*.whl\n" in workflow
    assert '/tmp/vecgrep-smoke/bin/python -I "$GITHUB_WORKSPACE/scripts/smoke_base_install.py"' in workflow


def _scope_script() -> str:
    workflow = _workflow_text()
    block = workflow.split("      - id: scope\n", 1)[1].split("        run: |\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line.strip() and not line.startswith("          "):
            break
        lines.append(line[10:])
    return "\n".join(lines)


def _run_scope(tmp_path: Path, event: str, count: str, files: list[str],
               before: str = "a" * 40) -> str:
    """Run the docs-only check with a stub gh that answers from fixtures."""
    import os
    import subprocess

    (tmp_path / "count").write_text(count)
    (tmp_path / "files").write_text("".join(f + "\n" for f in files))
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir()
    gh.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        f'  *"--jq .changed_files"*) cat "{tmp_path}/count" ;;\n'
        f'  *) cat "{tmp_path}/files" ;;\n'
        "esac\n"
    )
    gh.chmod(0o755)
    out = tmp_path / "out"
    env = {**os.environ, "PATH": f"{gh.parent}:{os.environ['PATH']}",
           "EVENT": event, "PR_NUMBER": "7", "BEFORE": before,
           "GITHUB_REPOSITORY": "o/r", "GITHUB_REF": "refs/heads/main",
           "GITHUB_SHA": "b" * 40, "GITHUB_OUTPUT": str(out)}
    subprocess.run(["bash", "-c", _scope_script()], env=env, check=True,
                   capture_output=True)
    return out.read_text().strip()


def test_docs_only_pull_request_skips_build(tmp_path: Path) -> None:
    assert _run_scope(tmp_path, "pull_request", "2",
                      ["docs/guide.md", "README.md"]) == "build=false"


def test_code_change_builds(tmp_path: Path) -> None:
    assert _run_scope(tmp_path, "pull_request", "2",
                      ["README.md", "vecgrep/cli.py"]) == "build=true"


def test_possibly_truncated_pull_request_builds(tmp_path: Path) -> None:
    # The files API stops at 3000; a docs-only first page proves nothing.
    assert _run_scope(tmp_path, "pull_request", "3000",
                      ["docs/a.md"]) == "build=true"


def test_unknown_file_count_builds(tmp_path: Path) -> None:
    assert _run_scope(tmp_path, "pull_request", "", ["docs/a.md"]) == "build=true"
