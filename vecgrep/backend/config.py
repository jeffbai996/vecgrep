"""Runtime config for vecgrep.

Single source of truth: env vars > config.json > defaults.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Invalid or unsafe runtime configuration update."""


ENV_MAP = {
    "VECGREP_OLLAMA_URL": "ollama_url",
    "VECGREP_OLLAMA_FALLBACK_URL": "ollama_fallback_url",
    "VECGREP_EMBED_MODEL": "embed_model",
    "OPENAI_API_KEY": "openai_api_key",
    "VECGREP_OPENAI_EMBED_MODEL": "openai_embed_model",
    "VECGREP_API_HOST": "api_host",
    "VECGREP_API_PORT": "api_port",
    "VECGREP_TOP_K": "default_top_k",
    "VECGREP_API_TOKEN": "api_token",
    "VECGREP_ADMIN_TOKEN": "admin_token",
    "VECGREP_QDRANT_URL": "qdrant_url",
    "VECGREP_OAUTH_ENABLED": "oauth_enabled",
    "VECGREP_OAUTH_ISSUER_URL": "oauth_issuer_url",
}

EDITABLE_FIELDS = {
    "ollama_url",
    "ollama_fallback_url",
    "embed_model",
    "openai_embed_model",
    "api_host",
    "api_port",
    "default_top_k",
    "oauth_enabled",
    "oauth_issuer_url",
    "qdrant_url",
    "backup_enabled",
    "backup_frequency",
    "backup_time",
    "backup_weekday",
    "backup_destination",
    "backup_retention",
}
SECRET_FIELDS = {"openai_api_key", "api_token", "admin_token"}
STRUCTURAL_FIELDS = {"api_host", "api_port", "qdrant_url", "oauth_enabled", "oauth_issuer_url"}


def _home() -> Path:
    return Path(os.environ.get("VECGREP_HOME", Path.home() / ".vecgrep")).expanduser()


@dataclass
class Settings:
    home: Path = field(default_factory=_home)
    ollama_url: str = "http://localhost:11434"
    # Optional second Ollama endpoint. When set, embed-backend selection probes
    # ollama_url first and transparently falls back to this one if the primary
    # is unreachable, before considering OpenAI. Lets a deployment run a primary
    # (e.g. a GPU box) with a local-host fallback. Unset (None) = no fallback.
    ollama_fallback_url: str | None = None
    embed_model: str = "bge-m3"
    openai_api_key: str | None = None
    openai_embed_model: str = "text-embedding-3-small"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    default_top_k: int = 5
    # If set, all /api/* routes (except /api/health) require a matching
    # `Authorization: Bearer <token>` header. Useful when you bind to 0.0.0.0
    # for Tailscale / LAN access. Unset (None) = no auth, the default.
    api_token: str | None = None
    # Separate credential for the mutation-capable /api/admin surface. Admin
    # routes otherwise accept only requests whose peer socket AND Host header
    # are loopback.
    admin_token: str | None = None
    # OAuth on the /mcp endpoint. When oauth_enabled, vecgrep runs an embedded
    # OAuth 2.1 authorization server (the SDK mounts /authorize, /token,
    # /.well-known and gates /mcp with bearer-token middleware) — so a client
    # that speaks OAuth (claude.ai) can authenticate. oauth_issuer_url is the
    # public base URL the MCP endpoint is reachable at (deployment-specific, set
    # via env — never hardcode). Internal callers reach /api over localhost/
    # tailnet with no token (network-trust); OAuth gates only the public /mcp.
    oauth_enabled: bool = False
    oauth_issuer_url: str | None = None
    # If set, use Qdrant in server mode at this URL instead of embedded mode.
    # Embedded mode locks the storage dir to a single process — incompatible
    # with running `vecgrep serve` and `vecgrep watch` simultaneously. Server
    # mode shares one daemon across all clients. Recommended for any setup
    # with concurrent readers/writers. Example: "http://localhost:6333".
    qdrant_url: str | None = None
    backup_enabled: bool = False
    backup_frequency: str = "daily"
    backup_time: str = "03:00"
    backup_weekday: int = 0
    backup_destination: str | None = None
    backup_retention: int = 7

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

    for env_key, attr in ENV_MAP.items():
        if env_key in os.environ:
            val = os.environ[env_key]
            if attr in {"api_port", "default_top_k", "backup_weekday", "backup_retention"}:
                val = int(val)
            elif attr in {"oauth_enabled", "backup_enabled"}:
                val = val.strip().lower() in ("1", "true", "yes", "on")
            setattr(s, attr, val)

    return s


def config_provenance(settings: Settings | None = None) -> dict[str, str]:
    """Return the effective source for every configurable setting."""
    settings = settings or get_settings()
    file_cfg = _load_json(settings.config_file)
    env_attrs = {attr for env, attr in ENV_MAP.items() if env in os.environ}
    return {
        field_name: (
            "env" if field_name in env_attrs else "file" if field_name in file_cfg else "default"
        )
        for field_name in EDITABLE_FIELDS | SECRET_FIELDS
    }


def _validate_url(name: str, value: str | None) -> None:
    if value is None:
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{name} must be an http or https URL")


def validate_settings(settings: Settings) -> None:
    _validate_url("ollama_url", settings.ollama_url)
    _validate_url("ollama_fallback_url", settings.ollama_fallback_url)
    _validate_url("qdrant_url", settings.qdrant_url)
    _validate_url("oauth_issuer_url", settings.oauth_issuer_url)
    if not 1 <= int(settings.api_port) <= 65535:
        raise ConfigError("api_port must be between 1 and 65535")
    if int(settings.default_top_k) <= 0:
        raise ConfigError("default_top_k must be positive")
    for name in ("embed_model", "openai_embed_model"):
        if not str(getattr(settings, name)).strip():
            raise ConfigError(f"{name} must be non-empty")
    if settings.oauth_enabled and not settings.oauth_issuer_url:
        raise ConfigError("oauth_enabled requires an OAuth issuer URL")
    if settings.backup_frequency not in {"daily", "weekly"}:
        raise ConfigError("backup_frequency must be daily or weekly")
    try:
        hour, minute = (int(part) for part in settings.backup_time.split(":"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigError("backup_time must use HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ConfigError("backup_time must use HH:MM")
    if not 0 <= int(settings.backup_weekday) <= 6:
        raise ConfigError("backup_weekday must be between 0 and 6")
    if int(settings.backup_retention) <= 0:
        raise ConfigError("backup_retention must be positive")


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def update_config(
    updates: dict,
    *,
    confirm_qdrant: str | None = None,
) -> dict:
    """Validate and atomically persist non-secret settings.

    Unknown existing keys are retained for forward compatibility. Fields
    currently controlled by the environment stay read-only.
    """
    unknown = set(updates) - EDITABLE_FIELDS
    if unknown:
        raise ConfigError(f"Unknown or non-editable setting: {sorted(unknown)[0]}")
    env_attrs = {attr for env, attr in ENV_MAP.items() if env in os.environ}
    overridden = set(updates) & env_attrs
    if overridden:
        raise ConfigError(f"Setting is controlled by environment: {sorted(overridden)[0]}")

    current = load_settings()
    if "qdrant_url" in updates and updates["qdrant_url"] != current.qdrant_url:
        if confirm_qdrant != updates["qdrant_url"]:
            raise ConfigError("Qdrant URL change requires typed confirmation of the new URL")

    candidate = Settings(**{
        name: getattr(current, name)
        for name in Settings.__dataclass_fields__
    })
    for name, value in updates.items():
        if name in {"api_port", "default_top_k", "backup_weekday", "backup_retention"} and not isinstance(value, bool):
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{name} must be an integer") from exc
        if name in {"oauth_enabled", "backup_enabled"} and not isinstance(value, bool):
            raise ConfigError(f"{name} must be a boolean")
        if name in {"ollama_fallback_url", "oauth_issuer_url", "qdrant_url", "backup_destination"} and value == "":
            value = None
        setattr(candidate, name, value)
    validate_settings(candidate)

    raw = _load_json(current.config_file)
    raw.update(updates)
    for nullable in ("ollama_fallback_url", "oauth_issuer_url", "qdrant_url", "backup_destination"):
        if raw.get(nullable) == "":
            raw[nullable] = None
    _atomic_write_json(current.config_file, raw)
    return raw


def reset_settings() -> Settings:
    global _settings
    _settings = None
    return get_settings()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
        _settings.ensure_dirs()
    return _settings
