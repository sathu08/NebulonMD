"""NebulonMind API System Management.

Mirrors ``ndb_host/api/routes/dashboard.py`` from NebulonDB: serves the
NebulonMind web console (``web_dir``) and exposes a small runtime-config
view. Unlike NebulonDB's ``nebulondb.cfg``, NebulonMind configuration is
``.env``-driven, so writes go through the dotenv file at the repository root
and require a restart of the service to fully apply.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ...core.config import (
    NebulonDBConfig,
    ServiceConfig,
    WEB_DIR,
    _nmd_home,
    load_dotenv,
)
from nmd_host.utils.constants import API_HOST_DEFAULT, API_PORT_DEFAULT

logger = logging.getLogger("nmd_host.api.routes.dashboard")

# ==========================================================
#        Web Console Location
# ==========================================================
# Static assets ship inside the package (``nmd_host/web_dir``); see
# ``nmd_host.core.config.WEB_DIR``.

# ==========================================================
#        API Router
# ==========================================================

router = APIRouter()

# ==========================================================
#        Serve Web Console (index.html)
# ==========================================================


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/index", response_class=HTMLResponse, include_in_schema=False)
async def index_page() -> FileResponse:
    """Serve the NebulonMind web console page."""
    index_html = WEB_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html)


# ==========================================================
#        Serve Web Console (alias)
# ==========================================================


@router.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_page() -> FileResponse:
    """Alias that serves ``web_dir/index.html`` at ``/api/NebulonMind/web``."""
    index_html = WEB_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html)


# ==========================================================
#        Get Configuration
# ==========================================================


@router.get("/config")
async def get_config() -> dict:
    """Return the active NebulonMind + NebulonDB connection settings.

    Credentials are never included — the API has no authentication layer and
    connection secrets stay inside ``NebulonMind``.
    """
    service = ServiceConfig.from_env()
    backend = NebulonDBConfig.from_env()
    host = os.environ.get("NEBULONDMIND_API_HOST", API_HOST_DEFAULT)
    port = int(os.environ.get("NEBULONDMIND_API_PORT", str(API_PORT_DEFAULT)))
    return {
        "service": {
            "app_name": "NebulonMind",
            "env": service.env,
            "host": host,
            "port": port,
            "workers": service.workers,
            "graceful_shutdown_seconds": service.graceful_shutdown_seconds,
            "url": f"http://{host}:{port}",
        },
        "api": {
            "cors_origins": list(service.cors_origins),
            "max_body_bytes": service.max_body_bytes,
            "max_top_k": service.max_top_k,
            "max_context_characters": service.max_context_characters,
            "rate_limit_per_minute": service.rate_limit_per_minute,
            "backend_retries": service.backend_retries,
            "allow_plaintext_http": service.allow_plaintext_http,
        },
        "backend": {
            "host": backend.host,
            "port": backend.port,
            "scheme": backend.scheme,
            "expected_backend_version": service.expected_backend_version,
        },
        "web": {
            "dir": str(WEB_DIR),
            "index": (WEB_DIR / "index.html").exists(),
        },
    }


# ==========================================================
#        Update Configuration
# ==========================================================


@router.put("/config")
async def update_config(payload: dict) -> dict:
    """Update select ``NMD_*`` / ``NEBULONDB_API_*`` values in ``.env``.

    The payload shape mirrors NebulonDB's dashboard (``{config: {section:
    {key: value}}}``). Only allow-listed environment variables are written
    and values are coerced to strings. A service restart is required for the
    running process to pick up the new values.
    """
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        raise HTTPException(
            status_code=400,
            detail="payload must include a non-empty 'config' mapping",
        )

    def resolve_env(section: str, key: str) -> str | None:
        mapping = {
            "service": {
                "host": "NEBULONDMIND_API_HOST",
                "port": "NEBULONDMIND_API_PORT",
                "workers": "NMD_API_WORKERS",
                "env": "NMD_ENV",
                "graceful_shutdown_seconds": "NMD_API_GRACEFUL_SHUTDOWN_SECONDS",
            },
            "api": {
                "cors_origins": "NMD_API_CORS_ORIGINS",
                "max_body_bytes": "NMD_API_MAX_BODY_BYTES",
                "max_top_k": "NMD_API_MAX_TOP_K",
                "max_context_characters": "NMD_API_MAX_CONTEXT_CHARACTERS",
                "rate_limit_per_minute": "NMD_API_RATE_LIMIT_PER_MINUTE",
                "backend_retries": "NMD_API_BACKEND_RETRIES",
                "allow_plaintext_http": "NMD_API_ALLOW_PLAINTEXT_HTTP",
            },
            "backend": {
                "host": "NEBULONDB_API_HOST",
                "port": "NEBULONDB_API_PORT",
                "scheme": "NEBULONDB_API_SCHEME",
            },
            "llm": {
                "provider": "NMD_LLM_PROVIDER",
                "model": "NMD_LLM_MODEL",
                "api_key": "NMD_LLM_API_KEY",
                "base_url": "NMD_LLM_BASE_URL",
            },
        }
        return mapping.get(section, {}).get(key)

    env_path = _nmd_home() / ".env"
    env_lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    env_index: dict[str, int] = {}
    for idx, line in enumerate(env_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        env_index[key] = idx

    updates: list[str] = []
    for section, values in config.items():
        for key, value in values.items():
            env_var = resolve_env(str(section), str(key))
            if env_var is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown config key '{section}.{key}'",
                )
            line = f"{env_var}={value}"
            if env_var in env_index:
                env_lines[env_index[env_var]] = line
            else:
                env_lines.append(line)
                env_index[env_var] = len(env_lines) - 1
            updates.append(line)

    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    load_dotenv(env_path, override=True)
    logger.info("Configuration updated in .env: %s", ", ".join(updates))

    return {
        "success": True,
        "message": "Configuration updated successfully.",
        "updated": updates,
    }