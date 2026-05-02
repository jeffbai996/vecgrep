from .base import Adapter, AdapterError, Document
from .registry import detect_adapter, register_adapter, ADAPTERS
# More-specific adapters first so the registry tries them before generic
# plaintext/markdown — `matches()` validates schema so order is correctness-
# neutral, but predictable order makes detection cheaper for the common case.
from .claude_export import ClaudeExportAdapter
from .chatgpt_export import ChatGPTExportAdapter
from .discord_jsonl import DiscordJSONLAdapter
from .pdf import PDFAdapter
from .url import URLAdapter
from .markdown import MarkdownAdapter
from .plaintext import PlaintextAdapter

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
    "DiscordJSONLAdapter",
    "ClaudeExportAdapter",
    "ChatGPTExportAdapter",
]
