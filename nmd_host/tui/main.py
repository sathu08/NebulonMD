"""Command-line entry point for the NebulonMind terminal UI.

No arguments launches the interactive Textual TUI; otherwise the classic
server lifecycle commands are honoured (``start|stop|restart``).
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        from .app import main as tui_main
        from .commands import HOST, PORT, is_server_running

        if not is_server_running(HOST, PORT):
            print("NebulonMind server is not up. Start it first with:")
            print(f"    nebulonmind start   # API listens on {HOST}:{PORT}")
            sys.exit(1)
        # tui_main()
        return

    command = sys.argv[1].lower()

    if command in ("--help", "-h", "help"):
        from nmd_host.utils.constants import TUI_USAGE

        print(TUI_USAGE)
        sys.exit(0)

    from .commands import restart_server, start_server, stop_server

    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    force = "--force" in sys.argv or "-F" in sys.argv

    if command == "start":
        ok = start_server(foreground=foreground)
    elif command == "stop":
        ok = stop_server(force=force)
    elif command == "restart":
        ok = restart_server(foreground=foreground, force=force)
    else:
        print(f"Invalid command. Usage: nebulonmind.py {{start|stop|restart}} [--foreground|-f] [--force|-F]")
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
