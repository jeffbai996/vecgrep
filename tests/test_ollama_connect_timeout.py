"""OllamaBackend must fail FAST when the host is unreachable.

The bug (2026-06-18): the httpx client used a single 60s timeout for ALL phases
including connect. When the primary endpoint's host is asleep (IP routable but
no service — e.g. a WoL box that's powered down), a TCP connect gets no RST and
hangs the full 60s before raising. With a primary→fallback failover, that means
every embed waits up to 60s for the dead primary before failing over — far past
any caller timeout, so failover never effectively fires. The connect phase must
time out in ~1-2s; the read phase can stay long (big embed batches are slow).
"""
from __future__ import annotations

import httpx
import pytest

from vecgrep.backend.embed.ollama import OllamaBackend


def test_connect_timeout_is_short_not_sixty_seconds():
    # The client's CONNECT timeout governs how fast a dead host fails over.
    # It must be small; the read/write/pool timeouts can remain generous.
    b = OllamaBackend.__new__(OllamaBackend)
    b.base_url = "http://x:11434"
    b.model = "bge-m3"
    b._client = OllamaBackend._make_client(connect=2.0, read=60.0)
    t = b._client.timeout
    assert t.connect is not None and t.connect <= 5.0   # fails over fast
    assert t.read is not None and t.read >= 30.0        # big embeds still allowed


def test_default_client_has_bounded_connect():
    # A normally-constructed backend (against a reachable stub host won't probe
    # here — just assert the client's connect timeout is bounded, not 60s).
    b = OllamaBackend.__new__(OllamaBackend)
    b._client = OllamaBackend._make_client()
    assert b._client.timeout.connect is not None
    assert b._client.timeout.connect <= 5.0
