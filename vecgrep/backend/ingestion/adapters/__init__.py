from .base import Adapter, AdapterError, Document
from .chatgpt_export import ChatGPTExportAdapter

# More-specific adapters first so the registry tries them before generic
# plaintext/markdown — `matches()` validates schema so order is correctness-
# neutral, but predictable order makes detection cheaper for the common case.
from .claude_export import ClaudeExportAdapter
from .discord_jsonl import DiscordJSONLAdapter
from .image import ImageAdapter
from .markdown import MarkdownAdapter
from .pdf import PDFAdapter
from .plaintext import PlaintextAdapter
from .registry import ADAPTERS, detect_adapter, register_adapter
from .url import URLAdapter

__all__ = [
    "ADAPTERS",
    "Adapter",
    "AdapterError",
    "ChatGPTExportAdapter",
    "ClaudeExportAdapter",
    "DiscordJSONLAdapter",
    "Document",
    "ImageAdapter",
    "MarkdownAdapter",
    "PDFAdapter",
    "PlaintextAdapter",
    "URLAdapter",
    "detect_adapter",
    "register_adapter",
]
