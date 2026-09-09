"""
NebulonMind Configuration
==========================================================

This module handles configuration settings for the NebulonMind API.
It mirrors ``NebulonDB``'s ``ndb_host/db/ndb_settings.py``:

* Loads operational settings from a config file (default: ``nebulonmind.cfg``).
* Supports an explicit home override via the ``NEBULONMD_HOME`` environment
  variable (the NebulonMind equivalent of ``NEBULONDB_HOME``).
* Safely resolves variables using ``string.Template`` and ``os.path.expandvars``.
* Secrets (credentials / API keys) are NEVER read from the cfg — they stay in
  ``.env`` / the process environment.

``nmd_host`` never imports the ``ndb_host`` package: every store talks to the
NebulonDB REST service. Connection settings (``NEBULONDB_API_*``) come from
``.env`` at the repository root; the service itself is configured with
``NMD_*`` variables (``NMD_ENV``, ``NMD_API_CORS_ORIGINS``, rate/body limits,
retries, …).
"""

from __future__ import annotations

import os

from pathlib import Path
from string import Template
from configparser import ConfigParser

from typing import Optional, Sequence, Tuple

from dataclasses import dataclass

from nmd_host.utils.env_helpers import env_bool as _env_bool
from nmd_host.utils.env_helpers import env_float as _env_float
from nmd_host.utils.env_helpers import env_int as _env_int
from nmd_host.utils.constants import (
    API_PORT_DEFAULT,
    AUTO_DELETE_CRON_DEFAULT,
    DEFAULT_USERNAME,
    GRACEFUL_SHUTDOWN_SECONDS_DEFAULT,
    NEBULONDB_API_HOST_DEFAULT,
    NEBULONDB_API_PORT_DEFAULT,
)


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - minimal fallback parser
    def load_dotenv(path: str = ".env", override: bool = False, **kwargs) -> bool:
        """Read KEY=VALUE lines from a dotenv file into ``os.environ``."""
        env_file = Path(path)
        if not env_file.exists():
            return False
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or not os.environ.get(key):
                os.environ[key] = value
        return True


# Web console static assets live inside the package (``nmd_host/web_dir``),
# not under ``NMD_HOME`` — the frontend ships with the code, not with runtime
# state.
WEB_DIR = Path(__file__).resolve().parent.parent / "web_dir"


# Keys that are secrets / credentials and therefore are NEVER read from the
# INI config file — they must come from ``.env`` / the process environment.
_SECRET_KEYS = frozenset(
    {
        "NEBULONDB_USERNAME",
        "NEBULONDB_PASSWORD",
        "NMD_LLM_API_KEY",
        "NMD_API_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "NVIDIA_API_KEY",
    }
)


def _coerce_typed(raw: str, type_name: str):
    """Coerce a raw cfg string to the native Python value for a setting type."""
    raw = (raw or "").strip()
    if type_name == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    if type_name == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0
    if type_name == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0
    return raw


# Human-friendly settings schema shown in the dashboard/TUI. Mirrors the
# NebulonDB console config layout: grouped settings with a title, a
# description and typed, labelled keys — not raw INI sections.
SETTINGS_GROUPS: list = [
    {
        "id": "paths",
        "title": "Paths",
        "description": "Filesystem locations used by NebulonMind.",
        "keys": [
            {"key": "nmd_home", "type": "str", "label": "NebulonMind home",
             "hint": "Root directory for .env, logs and the pid file."},
        ],
    },
    {
        "id": "backend",
        "title": "Backend · NebulonDB",
        "description": "How NebulonMind reaches the NebulonDB service.",
        "keys": [
            {"key": "nebulondb_api_host", "type": "str", "label": "Host"},
            {"key": "nebulondb_api_port", "type": "int", "label": "Port"},
            {"key": "nebulondb_api_scheme", "type": "str", "label": "Scheme",
             "hint": "http or https."},
            {"key": "nebulondb_api_connect_timeout", "type": "float", "label": "Connect timeout (s)"},
            {"key": "nebulondb_api_read_timeout", "type": "float", "label": "Read timeout (s)"},
            {"key": "nebulondb_api_write_timeout", "type": "float", "label": "Write timeout (s)"},
        ],
    },
    {
        "id": "server",
        "title": "Server · API",
        "description": "The NebulonMind API service (default 0.0.0.0:9696).",
        "keys": [
            {"key": "nebulondmind_api_host", "type": "str", "label": "Bind host"},
            {"key": "nebulondmind_api_port", "type": "int", "label": "Port"},
            {"key": "nmd_api_workers", "type": "int", "label": "Workers"},
            {"key": "nmd_api_graceful_shutdown_seconds", "type": "int", "label": "Graceful shutdown (s)"},
            {"key": "nmd_api_max_body_bytes", "type": "int", "label": "Max body bytes"},
            {"key": "nmd_api_backend_retries", "type": "int", "label": "Backend retries"},
            {"key": "nmd_api_rate_limit_per_minute", "type": "int", "label": "Rate limit (per minute)"},
            {"key": "nmd_api_max_top_k", "type": "int", "label": "Max top-k"},
            {"key": "nmd_api_max_context_characters", "type": "int", "label": "Max context characters"},
            {"key": "nmd_api_cors_origins", "type": "str", "label": "CORS origins",
             "hint": "Comma-separated origins allowed from the browser."},
            {"key": "nmd_api_allow_plaintext_http", "type": "bool", "label": "Allow plaintext HTTP"},
            {"key": "nmd_env", "type": "str", "label": "Environment"},
            {"key": "nmd_expected_backend_version", "type": "str", "label": "Expected backend version"},
        ],
    },
    {
        "id": "secrets_store",
        "title": "Secrets store",
        "description": "NebulonDB corpus/segment used for credential storage.",
        "keys": [
            {"key": "nmd_secrets_corpus", "type": "str", "label": "Corpus"},
            {"key": "nmd_secrets_segment", "type": "str", "label": "Segment"},
        ],
    },
    {
        "id": "llm",
        "title": "LLM",
        "description": "Provider and model used by the agent and the extractor.",
        "keys": [
            {"key": "nmd_llm_provider", "type": "str", "label": "Provider"},
            {"key": "nmd_llm_model", "type": "str", "label": "Model"},
            {"key": "nmd_llm_base_url", "type": "str", "label": "Base URL (for 'other' provider)"},
            {"key": "nmd_llm_timeout", "type": "int", "label": "Timeout (s)"},
            {"key": "nmd_llm_max_retries", "type": "int", "label": "Max retries"},
            {"key": "nmd_llm_extractor", "type": "bool", "label": "LLM extractor"},
            {"key": "nmd_llm_extractor_lenient", "type": "bool", "label": "Lenient extraction (fallback to rules)"},
        ],
    },
    {
        "id": "lifecycle",
        "title": "Lifecycle",
        "description": "Memory lifecycle defaults (TTL, cleanup).",
        "keys": [
            {"key": "nmd_temporary_ttl_seconds", "type": "int", "label": "Temporary memory TTL (s)"},
            {"key": "nmd_lifecycle_auto_cleanup", "type": "bool", "label": "Auto cleanup"},
            {"key": "nmd_lifecycle_auto_cleanup_cron", "type": "str", "label": "Auto cleanup schedule (cron)"},
        ],
    },
    {
        "id": "retrieval",
        "title": "Retrieval",
        "description": "Default retrieval sizes.",
        "keys": [
            {"key": "nmd_retrieval_top_k", "type": "int", "label": "Top-k"},
            {"key": "nmd_retrieval_candidates", "type": "int", "label": "Candidates"},
        ],
    },
    {
        "id": "ranking",
        "title": "Ranking",
        "description": "Weights for the Step 3 ranking pipeline.",
        "keys": [
            {"key": "nmd_ranking_semantic_weight", "type": "float", "label": "Semantic weight"},
            {"key": "nmd_ranking_importance_weight", "type": "float", "label": "Importance weight"},
            {"key": "nmd_ranking_confidence_weight", "type": "float", "label": "Confidence weight"},
            {"key": "nmd_ranking_recency_weight", "type": "float", "label": "Recency weight"},
            {"key": "nmd_ranking_recency_half_life_days", "type": "int", "label": "Recency half-life (days)"},
        ],
    },
    {
        "id": "context",
        "title": "Context",
        "description": "LLM context builder limits.",
        "keys": [
            {"key": "nmd_context_max_items", "type": "int", "label": "Max items"},
            {"key": "nmd_context_max_characters", "type": "int", "label": "Max characters"},
        ],
    },
    {
        "id": "agent",
        "title": "Agent",
        "description": "Step 6 agent runtime settings.",
        "keys": [
            {"key": "nmd_agent_model", "type": "str", "label": "Model"},
            {"key": "nmd_agent_temperature", "type": "float", "label": "Temperature"},
            {"key": "nmd_agent_max_turns", "type": "int", "label": "Max turns"},
            {"key": "nmd_agent_max_recall", "type": "int", "label": "Max recall"},
            {"key": "nmd_agent_max_sessions", "type": "int", "label": "Max sessions"},
            {"key": "nmd_agent_session_ttl_seconds", "type": "int", "label": "Session TTL (s)"},
            {"key": "nmd_agent_durable_sessions", "type": "bool", "label": "Durable sessions"},
            {"key": "nmd_agent_system_prompt", "type": "str", "label": "System prompt"},
        ],
    },
    {
        "id": "background",
        "title": "Background",
        "description": "Background agents (scheduled jobs).",
        "keys": [
            {"key": "nmd_background_user", "type": "str", "label": "Default user"},
            {"key": "nmd_background_persist_state", "type": "bool", "label": "Persist state"},
        ],
    },
    {
        "id": "runner",
        "title": "Runner",
        "description": "Server launch behaviour.",
        "keys": [
            {"key": "nmd_skip_backend_check", "type": "bool", "label": "Skip backend check"},
        ],
    },
]


# ==========================================================
#        NMDConfig
# ==========================================================

class NMDConfig:
    """
    NebulonMind Configuration Loader

    Loads configuration from a specified config file (default: `nebulonmind.cfg`),
    supports environment overrides, safely resolves variables using
    string.Template and os.path.expandvars.
    """

    def __init__(self, config_path: str | Path = None):
        """
        Initialize the NebulonMind configuration loader.

        Args:
            config_path (str): Path to the configuration file. When ``None``,
                resolves ``NEBULONMD_HOME`` (falling back to the repository
                root derived from this file) and reads ``nebulonmind.cfg``.
        """
        if config_path is None:
            nmd_home = os.environ.get("NEBULONMD_HOME", _repo_root())
            config_path = Path(nmd_home) / "nebulonmind.cfg"
        else:
            config_path = Path(config_path)

        self.config_path = config_path.resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            self._config = ConfigParser()
            self._config.read(self.config_path, encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config file '{self.config_path}': {e}"
            ) from e

        self._validate_sections()
        self._apply_env_override()
        self._load_paths()
        self._load_backend()
        self._load_server()
        self._load_secrets_store()
        self._load_llm()
        self._load_lifecycle()
        self._load_retrieval()
        self._load_ranking()
        self._load_context()
        self._load_agent()
        self._load_background()
        self._load_runner()

    # ------------------------------
    #  Private Utility Methods
    # ------------------------------

    @staticmethod
    def _resolve_path(path_vars: dict, value: str) -> Path:
        """Resolve variables using provided path_vars and environment, return a Path."""
        resolved = os.path.expandvars(Template(value).safe_substitute(path_vars))
        return Path(resolved).resolve()

    def _raw(self, section: str, key: str) -> str:
        return self._config.get(section, key, fallback="")

    def _getint(self, section: str, key: str, fallback: int = 0) -> int:
        try:
            return int(self._raw(section, key).strip() or str(fallback))
        except (ValueError, TypeError):
            return fallback

    def _getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        try:
            return float(self._raw(section, key).strip() or str(fallback))
        except (ValueError, TypeError):
            return fallback

    def _getbool(self, section: str, key: str, fallback: bool = False) -> bool:
        value = self._raw(section, key).strip().lower()
        if value in ("", "none"):
            return fallback
        return value in ("1", "true", "yes", "on")

    def _write(self):
        """Persist current config state back to disk."""
        with self.config_path.open("w", encoding="utf-8") as f:
            self._config.write(f)

    def _validate_sections(self):
        required_sections = ["paths", "backend", "server"]
        for section in required_sections:
            if section not in self._config:
                raise KeyError(
                    f"Missing required section: '{section}' in config file."
                )

        if "nmd_home" not in self._config["paths"]:
            raise KeyError("Missing 'nmd_home' in [paths] section.")

    def current_values(self) -> dict:
        """Return ``{section: {key: value}}`` exactly as stored in the cfg."""
        return {
            section: dict(self._config.items(section))
            for section in self._config.sections()
        }

    def settings_view(self) -> list:
        """Return the grouped, typed settings for the dashboard/TUI.

        Mirrors the NebulonDB console config layout: each group carries a
        title/description and typed, labelled keys with native Python values.
        Secret keys are excluded.
        """
        values = self.current_values()
        groups = []
        for group in SETTINGS_GROUPS:
            section = group["id"]
            keys = []
            for meta in group["keys"]:
                key = meta["key"]
                if key.upper() in _SECRET_KEYS:
                    continue
                raw = values.get(section, {}).get(key, "")
                keys.append(
                    {
                        "key": key,
                        "label": meta.get("label", key),
                        "type": meta.get("type", "str"),
                        "hint": meta.get("hint", ""),
                        "value": _coerce_typed(raw, meta.get("type", "str")),
                    }
                )
            groups.append(
                {
                    "id": section,
                    "title": group["title"],
                    "description": group.get("description", ""),
                    "keys": keys,
                }
            )
        return groups

    def update_config(self, updates: dict) -> list:
        """Update cfg entries and persist the file.

        ``updates`` has the shape ``{section: {key: value}}``. Only sections
        already present in the cfg are accepted (so required sections such as
        ``paths``/``backend``/``server`` can never be dropped), and secret
        keys (_SECRET_KEYS) are rejected — they belong in ``.env``, never in
        the cfg. Values are coerced to strings. Returns the list of
        ``section.key=value`` entries that were written. A restart is needed
        for a running service to fully apply the new values.
        """
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates must be a non-empty {section: {key: value}} mapping")
        updated = []
        for section, keys in updates.items():
            if section not in self._config:
                raise ValueError(
                    f"unknown section '{section}' — existing sections: "
                    f"{', '.join(self._config.sections())}"
                )
            if not isinstance(keys, dict):
                raise ValueError(
                    f"value for section '{section}' must be a {{key: value}} mapping"
                )
            for key, value in keys.items():
                key = str(key).strip()
                if not key:
                    continue
                if key.upper() in _SECRET_KEYS:
                    raise ValueError(
                        f"'{section}.{key}' is a secret — keep it in .env, "
                        "not nebulonmind.cfg"
                    )
                self._config.set(section, key, str(value).strip())
                updated.append(f"{section}.{key}={value}")
        self._write()
        return updated

    def set_background_user(self, username: str) -> None:
        """Persist the active conversation username to ``nebulonmind.cfg``.

        Writes ``NMD_BACKGROUND_USER`` in the ``[background]`` section so the
        next TUI/server launch defaults to the same user.

        Values that are empty or look like a slash command (``/create`` etc.)
        are refused — they are never valid usernames and must not leak into
        the persisted default (which ``_load_cfg`` re-exports to the env).
        """
        username = (username or "").strip()
        if not username or username.startswith("/"):
            return
        if not self._config.has_section("background"):
            self._config.add_section("background")
        self._config.set("background", "NMD_BACKGROUND_USER", username)
        self.NMD_BACKGROUND_USER = username
        self._write()

    def _apply_env_override(self):
        updated = False

        # Override NMD home directory
        env_home = os.environ.get("NEBULONMD_HOME")
        if env_home and self._config.get("paths", "nmd_home") != env_home:
            self._config.set("paths", "nmd_home", env_home)
            updated = True

        if updated:
            self._write()

    # ------------------------------
    #  Load Config Sections
    # ------------------------------

    def _load_paths(self):
        paths = dict(self._config["paths"])
        self.NMD_HOME = Path(
            self._resolve_path(paths, self._config.get("paths", "nmd_home"))
        )
        self.ENV_FILE = self.NMD_HOME / ".env"
        self.WEB_DIR = WEB_DIR
        self.LOG_DIR = self.NMD_HOME / "logs"
        self.PID_FILE = self.NMD_HOME / "nebulonmind.pid"

    def _load_backend(self):
        self.NEBULONDB_API_HOST = self._config.get("backend", "NEBULONDB_API_HOST")
        self.NEBULONDB_API_PORT = self._getint(
            "backend", "NEBULONDB_API_PORT", NEBULONDB_API_PORT_DEFAULT
        )
        self.NEBULONDB_API_SCHEME = self._config.get("backend", "NEBULONDB_API_SCHEME")
        self.NEBULONDB_API_CONNECT_TIMEOUT = self._getfloat(
            "backend", "NEBULONDB_API_CONNECT_TIMEOUT", 5.0
        )
        self.NEBULONDB_API_READ_TIMEOUT = self._getfloat(
            "backend", "NEBULONDB_API_READ_TIMEOUT", 30.0
        )
        self.NEBULONDB_API_WRITE_TIMEOUT = self._getfloat(
            "backend", "NEBULONDB_API_WRITE_TIMEOUT", 60.0
        )

    def _load_server(self):
        self.NEBULONDMIND_API_HOST = self._config.get(
            "server", "NEBULONDMIND_API_HOST"
        )
        self.NEBULONDMIND_API_PORT = self._getint(
            "server", "NEBULONDMIND_API_PORT", API_PORT_DEFAULT
        )
        self.NMD_API_WORKERS = self._getint("server", "NMD_API_WORKERS", 1)
        self.NMD_API_GRACEFUL_SHUTDOWN_SECONDS = self._getint(
            "server",
            "NMD_API_GRACEFUL_SHUTDOWN_SECONDS",
            GRACEFUL_SHUTDOWN_SECONDS_DEFAULT,
        )
        self.NMD_API_MAX_BODY_BYTES = self._getint(
            "server", "NMD_API_MAX_BODY_BYTES", 1_048_576
        )
        self.NMD_API_BACKEND_RETRIES = self._getint(
            "server", "NMD_API_BACKEND_RETRIES", 2
        )
        self.NMD_API_RATE_LIMIT_PER_MINUTE = self._getint(
            "server", "NMD_API_RATE_LIMIT_PER_MINUTE", 0
        )
        self.NMD_API_MAX_TOP_K = self._getint("server", "NMD_API_MAX_TOP_K", 50)
        self.NMD_API_MAX_CONTEXT_CHARACTERS = self._getint(
            "server", "NMD_API_MAX_CONTEXT_CHARACTERS", 100_000
        )
        self.NMD_API_CORS_ORIGINS = self._config.get(
            "server", "NMD_API_CORS_ORIGINS", fallback=""
        )
        self.NMD_API_ALLOW_PLAINTEXT_HTTP = self._getbool(
            "server", "NMD_API_ALLOW_PLAINTEXT_HTTP"
        )
        self.NMD_ENV = self._config.get("server", "NMD_ENV", fallback="development")
        self.NMD_EXPECTED_BACKEND_VERSION = self._config.get(
            "server", "NMD_EXPECTED_BACKEND_VERSION", fallback=""
        )

    def _load_secrets_store(self):
        self.NMD_SECRETS_CORPUS = self._config.get(
            "secrets_store", "NMD_SECRETS_CORPUS", fallback="nmd_Secrets"
        )
        self.NMD_SECRETS_SEGMENT = self._config.get(
            "secrets_store", "NMD_SECRETS_SEGMENT", fallback="Authentication"
        )

    def _load_llm(self):
        self.NMD_LLM_PROVIDER = self._config.get("llm", "NMD_LLM_PROVIDER", fallback="")
        self.NMD_LLM_MODEL = self._config.get("llm", "NMD_LLM_MODEL", fallback="")
        self.NMD_LLM_BASE_URL = self._config.get("llm", "NMD_LLM_BASE_URL", fallback="")
        self.NMD_LLM_TIMEOUT = self._getint("llm", "NMD_LLM_TIMEOUT", 60)
        self.NMD_LLM_MAX_RETRIES = self._getint("llm", "NMD_LLM_MAX_RETRIES", 3)
        self.NMD_LLM_EXTRACTOR = self._getbool("llm", "NMD_LLM_EXTRACTOR")
        self.NMD_LLM_EXTRACTOR_LENIENT = self._getbool("llm", "NMD_LLM_EXTRACTOR_LENIENT")
        self.NMD_LLM_THINKING = self._getbool("llm", "NMD_LLM_THINKING")

    def _load_lifecycle(self):
        self.NMD_TEMPORARY_TTL_SECONDS = self._getint(
            "lifecycle", "NMD_TEMPORARY_TTL_SECONDS", 2_592_000
        )
        self.NMD_LIFECYCLE_AUTO_CLEANUP = self._getbool(
            "lifecycle", "NMD_LIFECYCLE_AUTO_CLEANUP"
        )
        self.NMD_LIFECYCLE_AUTO_CLEANUP_CRON = (
            self._raw("lifecycle", "NMD_LIFECYCLE_AUTO_CLEANUP_CRON").strip()
            or AUTO_DELETE_CRON_DEFAULT
        )

    def _load_retrieval(self):
        self.NMD_RETRIEVAL_TOP_K = self._getint("retrieval", "NMD_RETRIEVAL_TOP_K", 5)
        self.NMD_RETRIEVAL_CANDIDATES = self._getint(
            "retrieval", "NMD_RETRIEVAL_CANDIDATES", 15
        )

    def _load_ranking(self):
        self.NMD_RANKING_SEMANTIC_WEIGHT = self._getfloat(
            "ranking", "NMD_RANKING_SEMANTIC_WEIGHT", 0.50
        )
        self.NMD_RANKING_IMPORTANCE_WEIGHT = self._getfloat(
            "ranking", "NMD_RANKING_IMPORTANCE_WEIGHT", 0.20
        )
        self.NMD_RANKING_CONFIDENCE_WEIGHT = self._getfloat(
            "ranking", "NMD_RANKING_CONFIDENCE_WEIGHT", 0.15
        )
        self.NMD_RANKING_RECENCY_WEIGHT = self._getfloat(
            "ranking", "NMD_RANKING_RECENCY_WEIGHT", 0.15
        )
        self.NMD_RANKING_RECENCY_HALF_LIFE_DAYS = self._getint(
            "ranking", "NMD_RANKING_RECENCY_HALF_LIFE_DAYS", 30
        )

    def _load_context(self):
        self.NMD_CONTEXT_MAX_ITEMS = self._getint("context", "NMD_CONTEXT_MAX_ITEMS", 10)
        self.NMD_CONTEXT_MAX_CHARACTERS = self._getint(
            "context", "NMD_CONTEXT_MAX_CHARACTERS", 6000
        )

    def _load_agent(self):
        self.NMD_AGENT_MODEL = self._config.get("agent", "NMD_AGENT_MODEL", fallback="")
        self.NMD_AGENT_TEMPERATURE = self._getfloat("agent", "NMD_AGENT_TEMPERATURE", 0.0)
        self.NMD_AGENT_MAX_TURNS = self._getint("agent", "NMD_AGENT_MAX_TURNS", 4)
        self.NMD_AGENT_MAX_RECALL = self._getint("agent", "NMD_AGENT_MAX_RECALL", 0)
        self.NMD_AGENT_MAX_SESSIONS = self._getint("agent", "NMD_AGENT_MAX_SESSIONS", 100)
        self.NMD_AGENT_SESSION_TTL_SECONDS = self._getint(
            "agent", "NMD_AGENT_SESSION_TTL_SECONDS", 3600
        )
        self.NMD_AGENT_DURABLE_SESSIONS = self._getbool(
            "agent", "NMD_AGENT_DURABLE_SESSIONS"
        )
        self.NMD_AGENT_SYSTEM_PROMPT = self._config.get(
            "agent", "NMD_AGENT_SYSTEM_PROMPT", fallback=""
        )

    def _load_background(self):
        self.NMD_BACKGROUND_USER = self._config.get(
            "background", "NMD_BACKGROUND_USER", fallback=DEFAULT_USERNAME
        )
        self.NMD_BACKGROUND_PERSIST_STATE = self._getbool(
            "background", "NMD_BACKGROUND_PERSIST_STATE"
        )

    def _load_runner(self):
        self.NMD_SKIP_BACKEND_CHECK = self._getbool("runner", "NMD_SKIP_BACKEND_CHECK")


# ==========================================================
#  Module-level helpers (compatibility with existing callers)
# ==========================================================


def _repo_root() -> Path:
    """Return the repository root derived from this file's location."""
    return Path(__file__).resolve().parents[2]


def _nmd_home() -> Path:
    """Resolve the NebulonMind home directory.

    Mirrors ``NEBULONDB_HOME``: an explicit ``NEBULONMD_HOME`` override
    wins; otherwise fall back to the repository root derived from this file.
    Lets ``.env`` / ``nebulonmind.cfg`` (and web assets) be found no matter
    which directory the process is launched from.
    """
    override = os.environ.get("NEBULONMD_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root()


def _default_env_path() -> Path:
    return _nmd_home() / ".env"


def _default_cfg_path() -> Path:
    return _nmd_home() / "nebulonmind.cfg"


def _load_cfg(cfg_path: Optional[Path] = None, override: bool = False) -> bool:
    """Load non-secret settings from ``nebulonmind.cfg`` into ``os.environ``.

    Mirrors ``NebulonDB``'s ``nebulondb.cfg``: the INI file stores operational
    settings (hosts, ports, weights, sizes), while secrets (username,
    password, API keys) stay in ``.env`` / the process environment and are
    never read from the cfg.
    """
    cfg_file = cfg_path or _default_cfg_path()
    if not cfg_file.exists():
        return False
    parser = ConfigParser()
    try:
        parser.read(cfg_file, encoding="utf-8")
    except Exception:  # defensive: corrupt cfg must not break startup
        return False
    loaded = False
    for section in parser.sections():
        for key, value in parser.items(section):
            key = key.upper()
            if key in _SECRET_KEYS:
                continue
            value = value.strip()
            if override or os.environ.get(key) is None:
                os.environ[key] = value
                loaded = True
    return loaded


# Credentials baked into the sample ``.env`` for local development. These are
# rejected outright in production (see ``validate_production_config``) — never
# a valid production defaults pair.
_SAMPLE_DEV_CREDENTIALS = (("nmd_user_01", "nmd_user_01"),)


# ==========================================================
#        NebulonDBConfig
# ==========================================================

@dataclass(frozen=True)
class NebulonDBConfig:
    """Connection settings for the NebulonDB REST API.

    Timeouts are explicit (P1): ``connect_timeout`` bounds the TCP/SSL
    handshake, ``read_timeout`` bounds each response read, ``write_timeout``
    bounds the upload/response for write-heavy calls. Values are separate so
    operators can tune reads vs writes independently.
    """

    host: str = NEBULONDB_API_HOST_DEFAULT
    port: int = NEBULONDB_API_PORT_DEFAULT
    username: str = ""
    password: str = ""
    scheme: str = "http"
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 60.0

    @property
    def base_url(self) -> str:
        """API base path, e.g. ``http://localhost:6969/api/NebulonDB``."""
        return f"{self.scheme}://{self.host}:{self.port}/api/NebulonDB"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "NebulonDBConfig":
        """Build config from environment variables, seeded by cfg + ``.env``."""
        _load_cfg()
        load_dotenv(str(env_file or _default_env_path()))
        return cls(
            host=os.environ.get("NEBULONDB_API_HOST", NEBULONDB_API_HOST_DEFAULT),
            port=_env_int("NEBULONDB_API_PORT", NEBULONDB_API_PORT_DEFAULT),
            username=os.environ.get("NEBULONDB_USERNAME", ""),
            password=os.environ.get("NEBULONDB_PASSWORD", ""),
            scheme=os.environ.get("NEBULONDB_API_SCHEME", "http"),
            connect_timeout=_env_float("NEBULONDB_API_CONNECT_TIMEOUT", 5.0),
            read_timeout=_env_float("NEBULONDB_API_READ_TIMEOUT", 30.0),
            write_timeout=_env_float("NEBULONDB_API_WRITE_TIMEOUT", 60.0),
        )

    @classmethod
    def from_config(cls, cfg: Optional["NMDConfig"] = None) -> "NebulonDBConfig":
        """Build config from an :class:`NMDConfig` instance (or env fallback)."""
        cfg = cfg or NMDConfig()
        return cls(
            host=cfg.NEBULONDB_API_HOST,
            port=cfg.NEBULONDB_API_PORT,
            username=os.environ.get("NEBULONDB_USERNAME", ""),
            password=os.environ.get("NEBULONDB_PASSWORD", ""),
            scheme=cfg.NEBULONDB_API_SCHEME,
            connect_timeout=cfg.NEBULONDB_API_CONNECT_TIMEOUT,
            read_timeout=cfg.NEBULONDB_API_READ_TIMEOUT,
            write_timeout=cfg.NEBULONDB_API_WRITE_TIMEOUT,
        )


# ==========================================================
#        ServiceConfig
# ==========================================================

@dataclass(frozen=True)
class ServiceConfig:
    """Operational settings for the NebulonMind service itself (port 9696).

    Every field maps to one ``NMD_*`` environment variable so a production
    deployment is fully declarative. ``env`` selects the validation regime;
    production requires real backend credentials, HTTPS and an explicit
    rate limit (fail fast — see ``validate_production_config``).
    """

    env: str = "development"
    cors_origins: Tuple[str, ...] = ()
    max_body_bytes: int = 1_048_576
    max_top_k: int = 50
    max_context_characters: int = 100_000
    rate_limit_per_minute: int = 0
    backend_retries: int = 2
    allow_plaintext_http: bool = False
    expected_backend_version: str = ""
    workers: int = 1
    graceful_shutdown_seconds: int = 30
    secrets_corpus: str = "nmd_Secrets"
    secrets_segment: str = "Authentication"
    llm_extractor: bool = False
    auth_token: str = ""
    background_persist_state: bool = False
    agent_model: str = ""

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "ServiceConfig":
        """Build config from environment variables, seeded by cfg + ``.env``."""
        _load_cfg()
        load_dotenv(str(env_file or _default_env_path()), override=True)
        origins = tuple(
            origin.strip()
            for origin in os.environ.get("NMD_API_CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            env=os.environ.get("NMD_ENV", "development").strip().lower(),
            cors_origins=origins,
            max_body_bytes=_env_int("NMD_API_MAX_BODY_BYTES", 1_048_576),
            max_top_k=_env_int("NMD_API_MAX_TOP_K", 50),
            max_context_characters=_env_int("NMD_API_MAX_CONTEXT_CHARACTERS", 100_000),
            rate_limit_per_minute=_env_int("NMD_API_RATE_LIMIT_PER_MINUTE", 0),
            backend_retries=_env_int("NMD_API_BACKEND_RETRIES", 2),
            allow_plaintext_http=_env_bool("NMD_API_ALLOW_PLAINTEXT_HTTP"),
            expected_backend_version=os.environ.get(
                "NMD_EXPECTED_BACKEND_VERSION", ""
            ).strip(),
            workers=_env_int("NMD_API_WORKERS", 1),
            graceful_shutdown_seconds=_env_int("NMD_API_GRACEFUL_SHUTDOWN_SECONDS", 30),
            secrets_corpus=os.environ.get(
                "NMD_SECRETS_CORPUS", "nmd_Secrets"
            ).strip() or "nmd_Secrets",
            secrets_segment=os.environ.get(
                "NMD_SECRETS_SEGMENT", "Authentication"
            ).strip() or "Authentication",
            llm_extractor=_env_bool("NMD_LLM_EXTRACTOR"),
            auth_token=os.environ.get("NMD_API_AUTH_TOKEN", "").strip(),
            background_persist_state=_env_bool("NMD_BACKGROUND_PERSIST_STATE"),
            agent_model=os.environ.get("NMD_AGENT_MODEL", "").strip(),
        )

    @classmethod
    def from_config(cls, cfg: Optional["NMDConfig"] = None) -> "ServiceConfig":
        """Build config from an :class:`NMDConfig` instance (or env fallback)."""
        cfg = cfg or NMDConfig()
        origins = tuple(
            origin.strip()
            for origin in cfg.NMD_API_CORS_ORIGINS.split(",")
            if origin.strip()
        )
        return cls(
            env=cfg.NMD_ENV.strip().lower(),
            cors_origins=origins,
            max_body_bytes=cfg.NMD_API_MAX_BODY_BYTES,
            max_top_k=cfg.NMD_API_MAX_TOP_K,
            max_context_characters=cfg.NMD_API_MAX_CONTEXT_CHARACTERS,
            rate_limit_per_minute=cfg.NMD_API_RATE_LIMIT_PER_MINUTE,
            backend_retries=cfg.NMD_API_BACKEND_RETRIES,
            allow_plaintext_http=cfg.NMD_API_ALLOW_PLAINTEXT_HTTP,
            expected_backend_version=cfg.NMD_EXPECTED_BACKEND_VERSION.strip(),
            workers=cfg.NMD_API_WORKERS,
            graceful_shutdown_seconds=cfg.NMD_API_GRACEFUL_SHUTDOWN_SECONDS,
            secrets_corpus=cfg.NMD_SECRETS_CORPUS.strip() or "nmd_Secrets",
            secrets_segment=cfg.NMD_SECRETS_SEGMENT.strip() or "Authentication",
            llm_extractor=cfg.NMD_LLM_EXTRACTOR,
            auth_token=os.environ.get("NMD_API_AUTH_TOKEN", "").strip(),
            background_persist_state=cfg.NMD_BACKGROUND_PERSIST_STATE,
            agent_model=cfg.NMD_AGENT_MODEL.strip(),
        )


# ==========================================================
#        Production Validation
# ==========================================================

def validate_production_config(
    service: ServiceConfig,
    backend: Optional[NebulonDBConfig] = None,
) -> Sequence[str]:
    """Fail-fast validation of production settings (P2).

    Returns a list of human-readable problems (empty when the configuration
    is acceptable). Only enforced when ``NMD_ENV=production``; the service
    refuses to boot with any problem rather than run with a weakened posture.

    Rules:
    * NebulonDB credentials are set and not the sample ``nmd_user_01/nmd_user_01``.
    * Backend calls use HTTPS unless ``NMD_API_ALLOW_PLAINTEXT_HTTP=true``
      (explicit override for isolated networks with TLS at the proxy).
    * A rate limit is configured (``NMD_API_RATE_LIMIT_PER_MINUTE > 0``).

    The API itself authenticates only when ``NMD_API_AUTH_TOKEN`` is set
    (open by default for internal networks) — NebulonDB access always stays
    authenticated inside ``NebulonMind``.
    """
    problems: list[str] = []
    if service.env != "production":
        return problems
    if backend is not None:
        if not backend.username or not backend.password:
            problems.append("NEBULONDB_USERNAME and NEBULONDB_PASSWORD must be set")
        if (backend.username, backend.password) in _SAMPLE_DEV_CREDENTIALS:
            problems.append(
                "sample NEBULONDB_* credentials (nmd_user_01/nmd_user_01) are not allowed "
                "in production"
            )
        if backend.scheme != "https" and not service.allow_plaintext_http:
            problems.append(
                "NEBULONDB_API_SCHEME must be https in production (set "
                "NMD_API_ALLOW_PLAINTEXT_HTTP=true only for isolated networks)"
            )
    if service.rate_limit_per_minute <= 0:
        problems.append("NMD_API_RATE_LIMIT_PER_MINUTE must be > 0 in production")
    return problems


__all__ = [
    "NMDConfig",
    "NebulonDBConfig",
    "ServiceConfig",
    "validate_production_config",
    "_nmd_home",
    "_load_cfg",
]
