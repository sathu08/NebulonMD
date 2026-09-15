# nmd_host/tui/server_ops.py
"""TUI-friendly server operations for NebulonMind.

These wrappers return ``(ok: bool, message: str)`` tuples and never raise
or ``sys.exit``, so the Textual app can render the outcome as a
notification. The underlying process lifecycle lives in :mod:`.commands`.
"""

from __future__ import annotations

import time
from typing import Tuple

import requests

from . import commands

from .context import ENV_FILE, PID_FILE, logger
from .helpers import is_port_open


def _probe(host: str) -> str:
    from .context import _probe_host
    return _probe_host(host)

def server_status() -> Tuple[bool, bool, str]:
    """Return ``(running, starting, pid_label)`` for the NebulonMind API.

    ``starting`` is True when a live PID file exists but the port is not yet
    reachable (server still booting).
    """
    running = is_port_open(_probe(commands.HOST), commands.PORT)
    starting = False
    pid_label = "Not Found"
    if PID_FILE.exists():
        pid_label = "Present"
        if not running:
            try:
                pid = int(PID_FILE.read_text().strip())
                from .helpers import is_pid_alive

                starting = is_pid_alive(pid)
            except (ValueError, OSError):
                starting = False
    return running, starting, pid_label

def backend_status() -> Tuple[bool, str]:
    """Return ``(reachable, host:port)`` for the NebulonDB backend."""
    host = commands.NEBULONDB_API_HOST
    port = commands.NEBULONDB_API_PORT
    reachable = is_port_open(_probe(host), port)
    return reachable, f"{host}:{port}"

def credentials_ready() -> bool:
    return commands.backend_credentials_present()

def is_startable() -> Tuple[bool, str]:
    """Whether the API can be started, with the reason when it cannot."""
    if commands.is_server_running(commands.HOST, commands.PORT):
        return False, "Server is already running."
    if not commands.SKIP_BACKEND_CHECK:
        reachable, addr = backend_status()
        if not reachable:
            return False, f"NebulonDB backend unreachable at {addr}."
    if not credentials_ready():
        return False, "Backend credentials missing from .env."
    return True, ""

def start_server() -> Tuple[bool, str]:
    running = is_port_open(_probe(commands.HOST), commands.PORT)
    if running:
        return False, f"Server already listening on {commands.HOST}:{commands.PORT}."
    ok = commands.start_server()
    return (True, "Server started.") if ok else (False, "Server was not started.")

def stop_server(force: bool = False) -> Tuple[bool, str]:
    ok = commands.stop_server(force=force)
    return (True, "Server stopped.") if ok else (False, "Server could not be stopped.")

def restart_server(force: bool = False) -> Tuple[bool, str]:
    ok = commands.restart_server(force=force)
    return (True, "Server restarted.") if ok else (False, "Server was not restarted.")

def save_credentials(username: str, password: str) -> Tuple[bool, str]:
    if not username or not password:
        return False, "Username and password cannot be empty."
    ok = commands.save_backend_credentials(username, password)
    return (True, "Credentials saved to .env.") if ok else (False, "Failed to save credentials.")

def wait_until(predicate, attempts: int = 12, interval: float = 1.0) -> bool:
    """Poll ``predicate()`` every ``interval`` up to ``attempts`` times."""
    for _ in range(attempts):
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ==========================================================
#  nebulonmd.cfg (via the API — same source the website uses)
# ==========================================================

def config_url(path: str = "cfg") -> str:
    """HTTP URL of the config management endpoint."""
    return f"http://{_probe(commands.HOST)}:{commands.PORT}/api/NebulonMind/config/{path}"


def fetch_config() -> Tuple[bool, dict]:
    """Fetch the grouped ``nebulonmd.cfg`` settings via the API.

    Returns ``(ok, data)`` where ``data`` is the API ``data`` block
    (``groups: [{id, title, description, keys: [...]}]``) on success,
    else ``{}``.
    """
    try:
        resp = requests.get(config_url(), timeout=(5, 20))
        body = resp.json()
        if resp.ok and body.get("success"):
            return True, body.get("data") or {}
        return False, {}
    except requests.RequestException:
        return False, {}
    except ValueError:
        return False, {}


def save_config(updates: dict) -> Tuple[bool, str]:
    """Persist ``{section: {key: value}}`` updates to nebulonmd.cfg via the API."""
    if not updates:
        return False, "No changes to save."
    try:
        resp = requests.put(config_url(), json={"config": updates}, timeout=(5, 20))
        body = resp.json()
    except requests.RequestException as exc:
        return False, f"could not reach API: {exc}"
    except ValueError:
        return False, f"unexpected API response (HTTP {resp.status_code})."
    if resp.ok and body.get("success"):
        return True, body.get("message", "Configuration saved.")
    detail = body.get("detail") or body.get("message") or f"HTTP {resp.status_code}"
    return False, str(detail)

__all__ = [
    "server_status",
    "backend_status",
    "credentials_ready",
    "is_startable",
    "start_server",
    "stop_server",
    "restart_server",
    "save_credentials",
    "wait_until",
    "fetch_config",
    "save_config",
    "config_url",
    "ENV_FILE",
    "logger",
]