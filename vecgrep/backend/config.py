"""Runtime config for vecgrep.

Single source of truth: env vars > config.json > defaults.
"""
from __future__ import annotations

import ipaddress
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Invalid or unsafe runtime configuration update."""


OAUTH_APPROVAL_ENV = "VECGREP_OAUTH_APPROVAL_TOKEN"

ENV_MAP = {
    "VECGREP_OLLAMA_URL": "ollama_url",
    "VECGREP_OLLAMA_FALLBACK_URL": "ollama_fallback_url",
    "VECGREP_OLLAMA_NUM_BATCH": "ollama_num_batch",
    "VECGREP_EMBED_MODEL": "embed_model",
    "OPENAI_API_KEY": "openai_api_key",
    "VECGREP_OPENAI_EMBED_MODEL": "openai_embed_model",
    "VECGREP_API_HOST": "api_host",
    "VECGREP_API_PORT": "api_port",
    "VECGREP_TOP_K": "default_top_k",
    "VECGREP_API_TOKEN": "api_token",
    "VECGREP_REST_ALLOWED_HOSTS": "rest_allowed_hosts",
    "VECGREP_ADMIN_TOKEN": "admin_token",
    "VECGREP_QDRANT_URL": "qdrant_url",
    "VECGREP_BM25_BACKEND": "bm25_backend",
    "VECGREP_OAUTH_ENABLED": "oauth_enabled",
    "VECGREP_OAUTH_ISSUER_URL": "oauth_issuer_url",
    "VECGREP_OAUTH_LOOPBACK_BYPASS": "oauth_loopback_bypass",
    "VECGREP_OAUTH_TAILSCALE_IDENTITY_BYPASS": "oauth_tailscale_identity_bypass",
    "VECGREP_MCP_ALLOWED_HOSTS": "mcp_allowed_hosts",
    "VECGREP_MCP_ALLOWED_ORIGINS": "mcp_allowed_origins",
    OAUTH_APPROVAL_ENV: "oauth_approval_token",
    "VECGREP_THREAD_POOL_SIZE": "thread_pool_size",
}

EDITABLE_FIELDS = {
    "ollama_url",
    "ollama_fallback_url",
    "ollama_num_batch",
    "embed_model",
    "openai_embed_model",
    "api_host",
    "api_port",
    "rest_allowed_hosts",
    "default_top_k",
    "oauth_enabled",
    "oauth_issuer_url",
    "oauth_loopback_bypass",
    "oauth_tailscale_identity_bypass",
    "mcp_allowed_hosts",
    "mcp_allowed_origins",
    "qdrant_url",
    "bm25_backend",
    "backup_enabled",
    "backup_frequency",
    "backup_time",
    "backup_weekday",
    "backup_destination",
    "backup_retention",
    "cross_corpus_exclude",
    "search_fanout_workers",
}
SECRET_FIELDS = {
    "openai_api_key", "api_token", "admin_token", "oauth_approval_token",
}
STRUCTURAL_FIELDS = {
    "api_host", "api_port", "rest_allowed_hosts", "qdrant_url", "bm25_backend", "oauth_enabled",
    "oauth_issuer_url", "oauth_loopback_bypass", "oauth_tailscale_identity_bypass",
    "mcp_allowed_hosts", "mcp_allowed_origins",
}


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
    # Optional per-request Ollama runner batch bound. Lowering this reduces
    # model working memory without changing weights, context, or vector shape.
    # None preserves the model/runtime default.
    ollama_num_batch: int | None = None
    embed_model: str = "bge-m3"
    openai_api_key: str | None = None
    openai_embed_model: str = "text-embedding-3-small"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    default_top_k: int = 5
    # If set, all /api/* routes (except /api/health) require a matching
    # `Authorization: Bearer <token>` header. A non-loopback bind requires a
    # strong value. Unset (None) remains valid for the default loopback-only
    # service when a proxy exposes only the separately authenticated MCP routes.
    api_token: str | None = None
    # Exact, port-qualified Host values for trusted reverse proxies that expose
    # the tokenless REST API through a separate network-access boundary.
    rest_allowed_hosts: list[str] = field(default_factory=list)
    # Separate credential for the mutation-capable /api/admin surface. Admin
    # routes otherwise accept only requests whose peer socket AND Host header
    # are loopback.
    admin_token: str | None = None
    # OAuth on proxied /mcp requests. When oauth_enabled, vecgrep runs an embedded
    # OAuth 2.1 authorization server (the SDK mounts /authorize, /token,
    # /.well-known and gates /mcp with bearer-token middleware) — so a client
    # that speaks OAuth (claude.ai) can authenticate. oauth_issuer_url is the
    # public base URL the MCP endpoint is reachable at (deployment-specific, set
    # via env — never hardcode). Internal callers reach /api over localhost/
    # tailnet with no token (network-trust); OAuth gates proxied /mcp traffic.
    oauth_enabled: bool = False
    oauth_issuer_url: str | None = None
    # Preserve trusted direct-to-loopback MCP clients while OAuth protects
    # requests that traversed a proxy. The bypass is allowed only when the peer
    # socket is loopback and neither standard forwarding header is present.
    # Set false to require OAuth even for direct local MCP clients.
    oauth_loopback_bypass: bool = True
    # Tailscale Serve strips client-supplied identity headers and adds the
    # authenticated tailnet user's identity. Funnel deliberately supplies no
    # identity headers. Trust that distinction for user-owned remote nodes.
    # oauth_loopback_bypass is the master kill switch for all MCP bypasses.
    oauth_tailscale_identity_bypass: bool = True
    # MCP's DNS-rebinding guard always permits loopback and automatically
    # permits the configured OAuth issuer. Private reverse proxies or browser
    # clients on other origins must be named explicitly here.
    mcp_allowed_hosts: list[str] = field(default_factory=list)
    mcp_allowed_origins: list[str] = field(default_factory=list)
    # Separate owner-presence credential for the public OAuth authorize page.
    # Dynamic client registration authenticates no human by itself; without
    # this gate anyone who discovers the public MCP URL could mint a token.
    oauth_approval_token: str | None = None
    # If set, use Qdrant in server mode at this URL instead of embedded mode.
    # Embedded mode locks the storage dir to a single process — incompatible
    # with running `vecgrep serve` and `vecgrep watch` simultaneously. Server
    # mode shares one daemon across all clients. Recommended for any setup
    # with concurrent readers/writers. Example: "http://localhost:6333".
    qdrant_url: str | None = None
    # Lexical sidecar implementation. Pickle preserves the historical default;
    # SQLite keeps large corpora off the Python heap after an explicit migration.
    bm25_backend: str = "pickle"
    backup_enabled: bool = False
    backup_frequency: str = "daily"
    backup_time: str = "03:00"
    backup_weekday: int = 0
    backup_destination: str | None = None
    backup_retention: int = 7
    # AnyIO's default worker-thread limiter for run_in_threadpool (any sync
    # route/dependency FastAPI bridges to a thread) is sized for a dedicated
    # host, not a lightly-loaded background service — a live instance was
    # observed holding 63 threads under a few requests/minute. 8 comfortably
    # covers real concurrent bursts without carrying that idle overhead.
    thread_pool_size: int = 8
    # Corpora an UNSCOPED search skips, as fnmatch patterns. `eval-*` corpora
    # are side-by-side copies built by vecgrep.eval to price a config change
    # (see vecgrep/eval/__init__.py); fanning out over one double-counts its
    # source corpus -- every hit returns twice, verbatim, eating two slots --
    # and doubles the fan-out cost. Naming a corpus explicitly always reaches
    # it, so the eval harness can still query its own build.
    cross_corpus_exclude: list[str] = field(default_factory=lambda: ["eval-*"])
    # Corpora searched concurrently on an unscoped query. The fan-out was
    # serial, so latency was the SUM of per-corpus cost: measured 16.3s across
    # 8 corpora where the slowest single corpus was 5.3s. 1 restores serial.
    search_fanout_workers: int = 8

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
        for path in (self.home, self.qdrant_path):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except OSError:
                # Non-POSIX filesystems may not implement Unix modes. The
                # service umask still protects files on supported deployments.
                pass


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
            if attr in {"api_port", "default_top_k", "backup_weekday", "backup_retention",
                        "thread_pool_size", "ollama_num_batch"}:
                val = int(val)
            elif attr in {
                "oauth_enabled", "oauth_loopback_bypass",
                "oauth_tailscale_identity_bypass", "backup_enabled",
            }:
                val = val.strip().lower() in ("1", "true", "yes", "on")
            elif attr in {
                "rest_allowed_hosts", "mcp_allowed_hosts", "mcp_allowed_origins",
            }:
                val = [item.strip() for item in val.split(",") if item.strip()]
            setattr(s, attr, val)

    validate_settings(s)
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


def _is_loopback_host(value: str) -> bool:
    if value.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_mcp_allowed_host(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError("MCP allowed hosts must be non-empty strings")
    if "://" in value or any(char.isspace() for char in value):
        raise ConfigError(f"MCP allowed host is not a Host value: {value!r}")
    if "*" in value and not value.endswith(":*"):
        raise ConfigError("MCP allowed hosts support only a terminal :* port wildcard")

    base = value[:-2] if value.endswith(":*") else value
    if "*" in base or any(char in base for char in "/?#@"):
        raise ConfigError(f"MCP allowed host is invalid: {value!r}")
    parsed = urlparse(f"//{base}")
    if not parsed.hostname or parsed.username or parsed.password or parsed.path:
        raise ConfigError(f"MCP allowed host is invalid: {value!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise ConfigError(f"MCP allowed host has an invalid port: {value!r}") from exc


def _validate_mcp_allowed_origin(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError("MCP allowed origins must be non-empty strings")
    if "*" in value and not value.endswith(":*"):
        raise ConfigError("MCP allowed origins support only a terminal :* port wildcard")

    base = value[:-2] if value.endswith(":*") else value
    if "*" in base:
        raise ConfigError(f"MCP allowed origin is invalid: {value!r}")
    parsed = urlparse(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError(f"MCP allowed origin must be an http(s) origin: {value!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise ConfigError(f"MCP allowed origin has an invalid port: {value!r}") from exc


def _validate_rest_allowed_host(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError("REST allowed hosts must be non-empty strings")
    if (
        "://" in value
        or "*" in value
        or any(char.isspace() for char in value)
        or any(char in value for char in "/?#@")
    ):
        raise ConfigError(f"REST allowed host is invalid: {value!r}")
    parsed = urlparse(f"//{value}")
    if not parsed.hostname or parsed.username or parsed.password or parsed.path:
        raise ConfigError(f"REST allowed host is invalid: {value!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(f"REST allowed host has an invalid port: {value!r}") from exc
    if port is None:
        raise ConfigError("REST allowed hosts must include an explicit port")


def validate_settings(settings: Settings) -> None:
    _validate_url("ollama_url", settings.ollama_url)
    _validate_url("ollama_fallback_url", settings.ollama_fallback_url)
    _validate_url("qdrant_url", settings.qdrant_url)
    _validate_url("oauth_issuer_url", settings.oauth_issuer_url)
    if not 1 <= int(settings.api_port) <= 65535:
        raise ConfigError("api_port must be between 1 and 65535")
    if int(settings.default_top_k) <= 0:
        raise ConfigError("default_top_k must be positive")
    if settings.ollama_num_batch is not None and int(settings.ollama_num_batch) <= 0:
        raise ConfigError("ollama_num_batch must be positive")
    if settings.bm25_backend not in {"pickle", "sqlite"}:
        raise ConfigError("bm25_backend must be pickle or sqlite")
    for name in ("embed_model", "openai_embed_model"):
        if not str(getattr(settings, name)).strip():
            raise ConfigError(f"{name} must be non-empty")
    if settings.oauth_enabled and not settings.oauth_issuer_url:
        raise ConfigError("oauth_enabled requires an OAuth issuer URL")
    if not isinstance(settings.rest_allowed_hosts, list):
        raise ConfigError("REST allowed hosts must be a list")
    if not isinstance(settings.mcp_allowed_hosts, list):
        raise ConfigError("MCP allowed hosts must be a list")
    if not isinstance(settings.mcp_allowed_origins, list):
        raise ConfigError("MCP allowed origins must be a list")
    for allowed_host in settings.rest_allowed_hosts:
        _validate_rest_allowed_host(allowed_host)
    for allowed_host in settings.mcp_allowed_hosts:
        _validate_mcp_allowed_host(allowed_host)
    for allowed_origin in settings.mcp_allowed_origins:
        _validate_mcp_allowed_origin(allowed_origin)
    approval_token = (settings.oauth_approval_token or "").strip()
    api_token = (settings.api_token or "").strip()
    if settings.oauth_enabled and len(approval_token) < 32:
        raise ConfigError(
            "oauth_enabled requires VECGREP_OAUTH_APPROVAL_TOKEN with at least 32 characters"
        )
    if not _is_loopback_host(settings.api_host) and len(api_token) < 32:
        raise ConfigError(
            "a non-loopback api_host requires VECGREP_API_TOKEN with at least "
            "32 characters"
        )
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
        if name in {
            "api_port", "default_top_k", "backup_weekday", "backup_retention",
            "ollama_num_batch",
        } and not isinstance(value, bool):
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{name} must be an integer") from exc
        if name in {
            "oauth_enabled", "oauth_loopback_bypass",
            "oauth_tailscale_identity_bypass", "backup_enabled",
        } and not isinstance(value, bool):
            raise ConfigError(f"{name} must be a boolean")
        if name in {
            "ollama_fallback_url", "ollama_num_batch", "oauth_issuer_url",
            "qdrant_url", "backup_destination",
        } and value == "":
            value = None
        setattr(candidate, name, value)
    validate_settings(candidate)

    raw = _load_json(current.config_file)
    raw.update(updates)
    for nullable in (
        "ollama_fallback_url", "ollama_num_batch", "oauth_issuer_url",
        "qdrant_url", "backup_destination",
    ):
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
