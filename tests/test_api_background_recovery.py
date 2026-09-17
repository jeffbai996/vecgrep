from __future__ import annotations

import threading

from vecgrep.backend.api import routes


def test_api_service_does_not_block_first_request_on_recovery(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class FakeService:
        def __init__(self, *, recover_pending):
            assert recover_pending is False

        def recover_pending_mutations(self):
            entered.set()
            assert release.wait(2)
            return ["busy-corpus"]

    monkeypatch.setattr(routes, "VecgrepService", FakeService)
    monkeypatch.setattr(routes, "_SERVICE", None)
    try:
        service = routes._service()
        assert isinstance(service, FakeService)
        assert entered.wait(1)
    finally:
        release.set()
        routes._SERVICE = None
