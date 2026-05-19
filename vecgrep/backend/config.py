"""Runtime config for vecgrep.

Single source of truth: env vars > config.json > defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _home() -> Path:
    return Path(os.environ.get("VECGREP_HOME", Path.home() / ".vecgrep")).expanduser()


@dataclass
class Settings:
    home: Path = field(default_factory=_home)
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    openai_api_key: str | None = None
    openai_embed_model: str = "text-embedding-3-small"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    default_top_k: int = 5
    # If set, all /api/* routes (except /api/health) require a matching
    # `Authorization: Bearer <token>` header. Useful when you bind to 0.0.0.0
    # for Tailscale / LAN access. Unset (None) = no auth, the default.
    api_token: str | None = None
    # If set, use Qdrant in server mode at this URL instead of embedded mode.
    # Embedded mode locks the storage dir to a single process — incompatible
    # with running `vecgrep serve` and `vecgrep watch` simultaneously. Server
    # mode shares one daemon across all clients. Recommended for any setup
    # with concurrent readers/writers. Example: "http://localhost:6333".
    qdrant_url: str | None = None

    @property
    def qdrant_path(self) -> Path:
        return self.home / "qdrant"

    @property
    def corpora_file(self) -> Path:
        return self.home / "corpora.json"

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_settings() -> Settings:
    s = Settings()
    file_cfg = _load_json(s.config_file)
    for key, val in file_cfg.items():
        if hasattr(s, key) and key != "home":
            setattr(s, key, val)

    env_map = {
        "VECGREP_OLLAMA_URL": "ollama_url",
        "VECGREP_EMBED_MODEL": "embed_model",
        "OPENAI_API_KEY": "openai_api_key",
        "VECGREP_OPENAI_EMBED_MODEL": "openai_embed_model",
        "VECGREP_API_HOST": "api_host",
        "VECGREP_API_PORT": "api_port",
        "VECGREP_TOP_K": "default_top_k",
        "VECGREP_API_TOKEN": "api_token",
        "VECGREP_QDRANT_URL": "qdrant_url",
    }
    for env_key, attr in env_map.items():
        if env_key in os.environ:
            val = os.environ[env_key]
            if attr in {"api_port", "default_top_k"}:
                val = int(val)
            setattr(s, attr, val)

    return s


def save_settings(s: Settings) -> None:
    s.ensure_dirs()
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(s).items()}
    payload.pop("home", None)
    s.config_file.write_text(json.dumps(payload, indent=2))


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
        _settings.ensure_dirs()
    return _settings
