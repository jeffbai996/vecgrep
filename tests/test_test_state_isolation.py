"""The test harness must fail before opening an operator's state."""
from pathlib import Path

import pytest

from vecgrep.backend.config import Settings


def test_default_settings_are_isolated_without_requesting_fixture(tmp_path):
    assert Settings().home.resolve().is_relative_to(tmp_path.resolve())


def test_explicit_production_home_is_rejected():
    with pytest.raises(AssertionError, match="outside the test directory"):
        Settings(home=Path.home() / ".vecgrep")


def test_symlink_escape_is_rejected(tmp_path):
    escaped = tmp_path / "escape"
    escaped.symlink_to(Path.home(), target_is_directory=True)
    with pytest.raises(AssertionError, match="outside the test directory"):
        Settings(home=escaped / ".vecgrep")
