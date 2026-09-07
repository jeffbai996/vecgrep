"""Security and compatibility contracts for the release workflow."""

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"
MANUAL_TAG_REF = (
    "ref: ${{ github.event_name == 'workflow_dispatch' "
    "&& format('refs/tags/{0}', inputs.tag) || github.ref }}"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_manual_release_checkouts_use_an_explicit_tag_ref() -> None:
    workflow = _workflow_text()
    checkout_count = workflow.count("uses: actions/checkout@")

    assert checkout_count == 2
    assert workflow.count(MANUAL_TAG_REF) == checkout_count
    assert "ref: ${{ inputs.tag || github.ref }}" not in workflow


def test_checkout_credentials_are_not_persisted() -> None:
    workflow = _workflow_text()
    checkout_count = workflow.count("uses: actions/checkout@")

    assert workflow.count("persist-credentials: false") == checkout_count


def test_release_event_and_publish_contract_is_preserved() -> None:
    workflow = _workflow_text()

    assert 'branches:\n      - "**"' in workflow
    assert 'tags:\n      - "v*"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "required: true" in workflow
    assert (
        "if: startsWith(github.ref, 'refs/tags/v') "
        "|| github.event_name == 'workflow_dispatch'"
    ) in workflow
    assert "RELEASE_TAG: ${{ inputs.tag || github.ref_name }}" in workflow
