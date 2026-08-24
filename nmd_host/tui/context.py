# nmd_host/tui/context.py
"""Shared runtime context for the NebulonMind terminal UI.

Mirrors ``ndb_host/tui/context.py``: exposes a single ``cfg``
(:class:`nmd_host.core.config.NMDConfig`), the resolved home paths, a
plain ``logging`` logger, and the ``enable_tui_mode`` switch that silences
console stream handlers so logs never corrupt the Textual screen.
"""

from __future__ import annotations

import sys
import logging

from nmd_host.core.config import NMDConfig


# ==========================================================
#        Shared Runtime Context (cfg / logger / pid file)
# ==========================================================

cfg = NMDConfig()

NMD_HOME = cfg.NMD_HOME
PID_FILE = NMD_HOME / "nebulonmind.pid"
LOG_DIR = NMD_HOME / "logs"
ENV_FILE = NMD_HOME / ".env"

logger = logging.getLogger("nmd_host.tui")

is_tui = False


def enable_tui_mode() -> None:
    """Disable console stream logging so nothing corrupts the Textual screen."""
    global is_tui
    is_tui = True
    for log in (
        logger,
        logging.getLogger("nmd_host"),
        logging.getLogger("nmd_host.access"),
        logging.getLogger("uvicorn"),
    ):
        for handler in list(log.handlers):
            if isinstance(handler, logging.StreamHandler):
                log.removeHandler(handler)


def tui_mode() -> bool:
    """Return whether the TUI is currently active."""
    return is_tui


def setup_nebulonmind_paths() -> None:
    """Ensure the NebulonMind home directory is importable."""
    home = NMD_HOME.resolve()
    if not home.is_dir():
        raise EnvironmentError(
            f"invalid NebulonMind home: {home} (set NEBULONMD_HOME)"
        )
    if str(home) not in sys.path:
        sys.path.append(str(home))


def _probe_host(host: str) -> str:
    """0.0.0.0 / :: are bind targets, not valid connect targets."""
    if host in ("0.0.0.0", "::", "", None):
        return "127.0.0.1"
    return host


def _backend_probe_host() -> str:
    from .commands import NEBULONDB_API_HOST

    return _probe_host(NEBULONDB_API_HOST)
