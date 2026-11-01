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
