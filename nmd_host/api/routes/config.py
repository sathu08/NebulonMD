"""NebulonMind configuration (``nebulonmind.cfg``) read/write API.

Mirrors ``ndb_host/api/routes/dashboard.py`` from NebulonDB: exposes a
grouped, typed settings view (``SETTINGS_GROUPS``) so both the TUI and the
web console can inspect and edit operational settings without touching the
file by hand. Secrets are never stored here — they stay in ``.env`` / the
process environment (see ``_SECRET_KEYS``). Writes validate against the live
parser and refuse unknown sections and secret keys; a service restart is
required for a running process to fully pick up the new values.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from fastapi import APIRouter, HTTPException, Body

from ...core.config import NMDConfig, _load_cfg, _nmd_home

logger = logging.getLogger("nmd_host.api.routes.config")

router = APIRouter()

CFG_FILE = _nmd_home() / "nebulonmind.cfg"


def _cfg() -> NMDConfig:
    """Return a fresh NMDConfig so edits always reflect the current file."""
    if not CFG_FILE.exists():
        raise HTTPException(
            status_code=404, detail=f"config file not found: {CFG_FILE}"
        )
    try:
        return NMDConfig(CFG_FILE)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"could not load config: {exc}"
        ) from exc


# ==========================================================
#        Get Configuration
# ==========================================================


@router.get("/cfg")
async def get_cfg() -> dict:
    """Return the ``nebulonmind.cfg`` settings grouped and typed.

    Shape mirrors the NebulonDB console: ``data.groups`` is a list of
    ``{id, title, description, keys: [{key, label, type, hint, value}]}``
    with native (str/int/float/bool) values. Secret keys are excluded.
    """
    cfg = _cfg()
    return {
        "success": True,
        "message": "config loaded",
        "data": {
            "config_path": str(cfg.config_path),
            "groups": cfg.settings_view(),
            "restart_required": True,
        },
    }


# ==========================================================
#        Update Configuration
# ==========================================================


@router.put("/cfg")
async def update_cfg(
    payload: dict = Body(..., description="{'config': {section: {key: value}}}"),
) -> dict:
    """Update ``nebulonmind.cfg`` entries.

    ``config`` is ``{section: {key: value}}``; every section must already
    exist in the cfg and secret keys are rejected. Returns the list of
    written ``section.key=value`` entries. The running service needs a
    restart to fully apply the new values.
    """
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        raise HTTPException(
            status_code=400,
            detail="payload must include a non-empty 'config' mapping",
        )
    cfg = _cfg()
    try:
        updated = cfg.update_config(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _load_cfg(CFG_FILE, override=True)
    logger.info("nebulonmind.cfg updated: %s", ", ".join(updated))
    return {
        "success": True,
        "message": "nebulonmind.cfg updated; non-secret settings applied live.",
        "data": {"updated": updated, "live": True, "restart_required": False},
    }


# ==========================================================
#        Restart Service
# ==========================================================


@router.post("/restart")
async def restart_service() -> dict:
    """Trigger a graceful background restart of the NebulonMind service.

    The current process cannot restart itself in-process (stopping the PID
    file process kills it before ``start_server`` can run), so a detached
    helper subprocess is spawned instead. It waits a short grace period for
    the HTTP response to flush, then stops the running server and starts a
    replacement. Returns immediately with a 202-style acknowledgement.
    """
    home = str(_nmd_home())
    code = (
        "import time; time.sleep(6); "
        "from nmd_host.tui.commands import restart_server; "
        "restart_server()"
    )
    try:
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=home,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"could not schedule restart: {exc}"
        ) from exc
    logger.info("service restart scheduled")
    return {
        "success": True,
        "message": "Service restart scheduled. The console will reconnect shortly.",
        "data": {"restarting": True},
    }