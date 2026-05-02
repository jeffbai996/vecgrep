from .base import Adapter, AdapterError, Document
from .registry import detect_adapter, register_adapter, ADAPTERS
from .plaintext import PlaintextAdapter
from .markdown import MarkdownAdapter
from .pdf import PDFAdapter
from .url import URLAdapter

__all__ = [
    "Adapter",
    "AdapterError",
    "Document",
    "ADAPTERS",
    "detect_adapter",
    "register_adapter",
    "PlaintextAdapter",
    "MarkdownAdapter",
    "PDFAdapter",
    "URLAdapter",
]
