"""NebulonMind Terminal UI — server management commands and interactive TUI.

A terminal-based UI for managing the NebulonMind API service: process/port
helpers, the start/stop/restart lifecycle commands, the CLI entry point, and
an interactive Textual TUI with a live agent chat panel.
"""

from .helpers import (
    find_pid_on_port,
    find_process_tree_root,
    is_pid_alive,
    is_port_open,
    kill_process,
    kill_process_tree,
)
from .commands import (
    start_server,
    stop_server,
    restart_server,
    is_server_running,
    is_nebulondb_reachable,
    ensure_backend_credentials,
    backend_credentials_present,
    save_backend_credentials,
    HOST,
    PORT,
    APP_NAME,
    NEBULONDB_API_HOST,
    NEBULONDB_API_PORT,
    PID_FILE,
    LOG_DIR,
    ENV_FILE,
)
from .context import (
    cfg,
    NMD_HOME,
    logger,
    enable_tui_mode,
    tui_mode,
    setup_nebulonmind_paths,
    _probe_host,
)

__all__ = [
    # helpers
    "find_pid_on_port",
    "find_process_tree_root",
    "is_pid_alive",
    "is_port_open",
    "kill_process",
    "kill_process_tree",
    # commands
    "start_server",
    "stop_server",
    "restart_server",
    "is_server_running",
    "is_nebulondb_reachable",
    "ensure_backend_credentials",
    "backend_credentials_present",
    "save_backend_credentials",
    "HOST",
    "PORT",
    "APP_NAME",
    "NEBULONDB_API_HOST",
    "NEBULONDB_API_PORT",
    "PID_FILE",
    "LOG_DIR",
    "ENV_FILE",
    # context
    "cfg",
    "NMD_HOME",
    "logger",
    "enable_tui_mode",
    "tui_mode",
    "setup_nebulonmind_paths",
    "_probe_host",
]
