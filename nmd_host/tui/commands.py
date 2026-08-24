"""Server lifecycle commands for the NebulonMind terminal UI.

``start_server`` / ``stop_server`` / ``restart_server`` manage the
NebulonMind API service as a background uvicorn subprocess, using the
process helpers from :mod:`nmd_host.tui.helpers`. ``start_server`` refuses
to boot when the NebulonDB backend is unreachable and verifies backend
credentials are present in ``.env`` (prompting to create them when missing).
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import platform
import subprocess

from getpass import getpass

from nmd_host.core.config import _load_cfg, _nmd_home
from nmd_host.utils.constants import (
    API_HOST_DEFAULT,
    API_PORT_DEFAULT,
    APP_NAME,
    NEBULONDB_API_HOST_DEFAULT,
    NEBULONDB_API_PORT_DEFAULT,
    NEBULONMIND_BANNER,
    SERVER_GETTING_READY_MESSAGE,
)
from .helpers import (
    find_pid_on_port,
    is_pid_alive,
    is_port_open,
    kill_process,
    kill_process_tree,
)


NMD_HOME = _nmd_home()

_load_cfg()

HOST = os.environ.get("NEBULONDMIND_API_HOST", API_HOST_DEFAULT)
PORT = int(os.environ.get("NEBULONDMIND_API_PORT", str(API_PORT_DEFAULT)))

LOG_DIR = NMD_HOME / "logs"
PID_FILE = NMD_HOME / "nebulonmind.pid"
SERVER_MODULE = "nmd_host.tui.serve"

ENV_FILE = NMD_HOME / ".env"

NEBULONDB_API_HOST = os.environ.get("NEBULONDB_API_HOST", NEBULONDB_API_HOST_DEFAULT)
NEBULONDB_API_PORT = int(
    os.environ.get("NEBULONDB_API_PORT", str(NEBULONDB_API_PORT_DEFAULT))
)
SKIP_BACKEND_CHECK = os.environ.get("NMD_SKIP_BACKEND_CHECK", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Directory names never descended into while clearing ``__pycache__`` —
# virtualenvs / tool caches hold thousands of bytecode dirs that are not ours.
PYCACHE_PRUNE_DIRS = {".git", ".venv", "venv", "node_modules", ".mypy_cache"}


def _out(message: str) -> None:
    """Write to the console only when not inside the Textual TUI.

    The TUI renders its own status/notifications; raw prints would corrupt
    the screen. The CLI path keeps the existing ``print`` behaviour.
    """
    if not _in_tui():
        print(message)


def _in_tui() -> bool:
    try:
        from .context import tui_mode
        return tui_mode()
    except Exception:
        return False


def _clear_pycache() -> int:
    """Delete every ``__pycache__`` directory under ``NMD_HOME``.

    Keeps stale bytecode from shadowing freshly edited sources between
    restarts. Hidden/foreign trees (git, virtualenvs, node_modules) are
    pruned so only NebulonMind's own caches are touched. Returns the
    number of directories removed.
    """
    removed = 0
    for dirpath, dirnames, _files in os.walk(NMD_HOME, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in PYCACHE_PRUNE_DIRS]
        for name in list(dirnames):
            if name == "__pycache__":
                shutil.rmtree(os.path.join(dirpath, name), ignore_errors=True)
                dirnames.remove(name)
                removed += 1
    return removed


def is_server_running(host: str, port: int) -> bool:
    """Return True if a server is already listening on ``host:port``."""
    return is_port_open(host, port)


def is_nebulondb_reachable() -> bool:
    """Return True if the NebulonDB backend is reachable on its configured address:port."""
    return is_port_open(NEBULONDB_API_HOST, NEBULONDB_API_PORT)


def _read_env() -> dict:
    """Return ``{KEY: value}`` entries currently stored in ``.env``."""
    entries: dict = {}
    if not ENV_FILE.exists():
        return entries
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        entries[key.strip()] = value.strip().strip('"').strip("'")
    return entries


def _write_env_key(key: str, value: str) -> None:
    """Write (or update) a single ``KEY=value`` in ``.env``, preserving comments."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, _ = stripped.partition("=")
        if k.strip() == key:
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backend_credentials_present() -> bool:
    """Return True when ``.env`` holds non-empty NEBULONDB_USERNAME/PASSWORD."""
    if not ENV_FILE.exists():
        return False
    entries = _read_env()
    return bool(
        entries.get("NEBULONDB_USERNAME", "").strip()
        and entries.get("NEBULONDB_PASSWORD", "").strip()
    )


def ensure_backend_credentials() -> bool:
    """Ensure ``.env`` exists and holds ``NEBULONDB_USERNAME``/``NEBULONDB_PASSWORD``.

    In the CLI, missing credentials prompt the user interactively and save
    them to ``.env`` (returns False when declined). In the TUI the prompt is
    handled by a modal screen, so here we only report availability.
    """
    if backend_credentials_present():
        _out(f"Backend credentials found in {ENV_FILE.name}.")
        return True
    if _in_tui():
        return False
    if not ENV_FILE.exists():
        _out(f".env file not found at: {ENV_FILE}")
        _out("NebulonMind needs NEBULONDB_USERNAME and NEBULONDB_PASSWORD to connect.")
    else:
        _out(f".env exists but NEBULONDB_USERNAME / NEBULONDB_PASSWORD are missing.")

    _out()
    choice = input(
        "Enter credentials now (saved to .env) or create them manually? "
        "[enter to type them / 'skip']: "
    ).strip().lower()

    if choice == "skip":
        _out(
            "Please add NEBULONDB_USERNAME and NEBULONDB_PASSWORD to .env "
            "and re-run. NebulonMind will not start without them."
        )
        return False

    username = input("NEBULONDB_USERNAME: ").strip()
    if sys.stdin.isatty():
        password = getpass("NEBULONDB_PASSWORD: ")
    else:
        password = input("NEBULONDB_PASSWORD: ")
    if not username or not password:
        _out("ERROR: username and password cannot be empty.")
        return False

    return save_backend_credentials(username, password)


def save_backend_credentials(username: str, password: str) -> bool:
    """Persist backend credentials to ``.env``; returns True on success."""
    if not username or not password:
        return False
    if not ENV_FILE.exists():
        ENV_FILE.write_text("", encoding="utf-8")
    _write_env_key("NEBULONDB_USERNAME", username)
    _write_env_key("NEBULONDB_PASSWORD", password)
    _out(f"Credentials saved to {ENV_FILE.name}.")
    return True


def start_server(foreground: bool = False) -> bool:
    # ----- Handle stale PID file -----
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_pid_alive(pid):
                _out(f"Server is already running (PID {pid}).")
                return False
            else:
                _out(f"Found stale PID file (PID {pid}). Removing it.")
                PID_FILE.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            PID_FILE.unlink(missing_ok=True)

    if is_server_running(HOST, PORT):
        _out(f"Server is already listening on {HOST}:{PORT}.")
        return False

    # ----- NebulonDB reachability check -----
    if not SKIP_BACKEND_CHECK and not is_nebulondb_reachable():
        _out("")
        _out("ERROR ⛔  THE NEBULONDB SERVER IS NOT UP")
        _out("	Please make the NebulonDB server up and running first.")
        _out(f"	NebulonDB is expected at: {NEBULONDB_API_HOST}:{NEBULONDB_API_PORT}")
        _out("")
        _out("	How to start NebulonDB:")
        _out("		NEBULONDB_HOME=... python nebulondb.py start")
        _out("		(binds 0.0.0.0:6969; port in `nebulondb.cfg` `[server] port`)")
        _out("")
        _out("NebulonMind will NOT start until the NebulonDB server is up.")
        _out("")
        return False

    # ----- .env / backend credentials check -----
    if not ensure_backend_credentials():
        return False

    _out(NEBULONMIND_BANNER.rstrip())
    _out(f"Starting {APP_NAME} server...")
    removed = _clear_pycache()
    if removed:
        _out(f"Cleared {removed} __pycache__ folder(s).")
    _out(SERVER_GETTING_READY_MESSAGE)
    _out(f"NebulonMind API will listen on {HOST}:{PORT}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / time.strftime("nebulonmind-%Y%m%d.log")

    cmd = [sys.executable, "-m", SERVER_MODULE]

    if foreground:
        # ---- Foreground mode: output to terminal, block until stopped ----
        stdout = None
        stderr = None
    else:
        # ---- Background mode: always persist uvicorn logs to a file ----
        stdout = open(log_file, "a")
        stderr = open(log_file, "a")

    kwargs = {
        "stdout": stdout,
        "stderr": stderr,
        "stdin": subprocess.DEVNULL,
        "cwd": NMD_HOME,
    }
    if platform.system() != "Windows":
        kwargs["start_new_session"] = True
    if platform.system() == "Windows" and not foreground:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    process = subprocess.Popen(cmd, **kwargs)
    PID_FILE.write_text(str(process.pid))
    _out(f"Server started with PID {process.pid}.")
    _out(f"API available at: http://{HOST}:{PORT}/api/NebulonMind/health")
    _out(f"Web console available at: http://{HOST}:{PORT}/api/NebulonMind/dashboard/")
    _out(f"Logs written to: {log_file}")

    if foreground:
        _out("Running in foreground. Press Ctrl+C to stop.")
        try:
            process.wait()
            _out("Server exited normally.")
        except KeyboardInterrupt:
            _out("\nShutting down server...")
            kill_process(process.pid)
            process.wait()
        finally:
            if PID_FILE.exists():
                PID_FILE.unlink(missing_ok=True)
    # else: background mode – we return immediately

    return True


def stop_server(force: bool = False) -> bool:
    if not PID_FILE.exists():
        _out("PID file not found – server is not managed by this script.")
        if not is_server_running(HOST, PORT):
            return False
        if force:
            pid = find_pid_on_port(PORT)
            if pid is None:
                _out(
                    f"Could not find a process listening on {HOST}:{PORT}. "
                    "Stop it manually."
                )
                return False
            _out(f"Force stopping manually-started server (PID {pid})...")
            kill_process_tree(pid)
            time.sleep(1)
            if not is_server_running(HOST, PORT):
                _out("Server stopped successfully.")
                return True
            else:
                _out(f"Failed to force stop server. Port {HOST}:{PORT} is still in use.")
            return False
        _out(
            f"Server appears to be running on {HOST}:{PORT} but was started "
            "manually. Use 'nebulonmind.py stop --force' to stop it."
        )
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        _out("PID file is corrupt. Removing it.")
        PID_FILE.unlink(missing_ok=True)
        return False

    if not is_pid_alive(pid):
        _out(f"Process with PID {pid} is not running. Cleaning up PID file.")
        PID_FILE.unlink(missing_ok=True)
        return True

    _out(f"Stopping {APP_NAME} (PID {pid})...")
    if force:
        kill_process_tree(pid)
    else:
        kill_process(pid)
    time.sleep(1)
    if not is_pid_alive(pid):
        _out("Server stopped successfully.")
        PID_FILE.unlink(missing_ok=True)
        return True
    else:
        _out(f"Failed to stop server. PID {pid} is still alive.")
    return False


def restart_server(foreground: bool = False, force: bool = False) -> bool:
    if is_server_running(HOST, PORT) or PID_FILE.exists():
        stop_server(force=force)
        time.sleep(2)
    return start_server(foreground)


__all__ = [
    "start_server",
    "stop_server",
    "restart_server",
    "is_server_running",
    "is_nebulondb_reachable",
    "ensure_backend_credentials",
    "backend_credentials_present",
    "save_backend_credentials",
    "APP_NAME",
    "HOST",
    "PORT",
    "PID_FILE",
    "LOG_DIR",
    "ENV_FILE",
]