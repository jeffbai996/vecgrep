"""The shared registry of Gemini's bearers: digests only, revocations stick,
and an unreadable file accepts nothing and is never overwritten."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vecgrep.backend.auth.google_bearers import GoogleBearerRegistry, digest

DAY = 86400


def _registry(tmp_path):
    return GoogleBearerRegistry(str(tmp_path / "shared" / "google-bearers.json"))


def test_a_recorded_bearer_is_live_until_its_expiry(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    assert reg.lookup(digest("bearer-1"), now=now)["issuer"] == "one"
    assert reg.lookup(digest("bearer-1"), now=now + 2 * DAY) is None


def test_the_file_holds_digests_never_token_values(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-secret-value"): time.time() + DAY}, issuer="one")

    assert "bearer-secret-value" not in Path(reg.path).read_text()


def test_a_revoked_bearer_is_not_live_and_recording_it_again_does_not_undo_that(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    reg.revoke({digest("bearer-1"): None}, now=now)
    reg.record({digest("bearer-1"): now + DAY}, issuer="two", now=now)

    assert reg.lookup(digest("bearer-1"), now=now) is None
    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_an_unregistered_bearer_with_a_known_expiry_keeps_it_out(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.revoke({digest("bearer-1"): now + DAY}, now=now)
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    assert reg.lookup(digest("bearer-1"), now=now) is None


def test_an_entry_lasts_as_long_as_the_longest_copy_recorded(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.record({digest("bearer-1"): now + 365 * DAY}, issuer="two", now=now)

    assert reg.lookup(digest("bearer-1"), now=now + 100 * DAY) is not None


def test_a_revocation_lasts_as_long_as_the_longest_copy_recorded(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.revoke({digest("bearer-1"): None}, now=now)
    reg.record({digest("bearer-1"): now + 365 * DAY}, issuer="two", now=now)

    reg.record({digest("other"): now + 365 * DAY}, issuer="one", now=now + 28 * DAY)

    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_with_a_later_expiry_extends_the_revocation(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.revoke({digest("bearer-1"): now + 365 * DAY}, now=now)

    reg.record({digest("other"): now + 365 * DAY}, issuer="one", now=now + 28 * DAY)

    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_an_unregistered_bearer_without_an_expiry_records_nothing(tmp_path):
    reg = _registry(tmp_path)
    reg.revoke({digest("bearer-1"): None})

    assert not reg.is_revoked(digest("bearer-1"))


def test_an_unreadable_registry_accepts_nothing_and_is_not_overwritten(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")
    with open(reg.path, "w") as fh:
        fh.write("{torn")

    reg.record({digest("bearer-2"): time.time() + DAY}, issuer="one")

    assert reg.lookup(digest("bearer-1")) is None
    assert reg.lookup(digest("bearer-2")) is None
    assert Path(reg.path).read_text() == "{torn"


def test_expired_entries_are_dropped_on_the_next_write(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("old"): now + 10}, issuer="one", now=now)
    reg.record({digest("new"): now + DAY}, issuer="one", now=now + 20)

    assert list(json.loads(Path(reg.path).read_text())["bearers"]) == [digest("new")]


def test_the_registry_file_is_private(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")

    assert os.stat(reg.path).st_mode & 0o777 == 0o600


def test_the_registry_reaches_the_disk_before_it_replaces_the_old_file(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])

    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")

    assert calls == ["fsync", "replace"]
