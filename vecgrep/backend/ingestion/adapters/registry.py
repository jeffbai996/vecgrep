from __future__ import annotations

from .base import Adapter, AdapterError

ADAPTERS: list[type[Adapter]] = []


def register_adapter(cls: type[Adapter]) -> type[Adapter]:
    if cls not in ADAPTERS:
        ADAPTERS.append(cls)
    return cls


def detect_adapter(source: str) -> type[Adapter]:
    for cls in ADAPTERS:
        if cls.matches(source):
            return cls
    raise AdapterError(f"No adapter matches source: {source}")
