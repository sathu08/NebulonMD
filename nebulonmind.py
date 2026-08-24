"""
NebulonMind Runner
==========================================================

Hosts the NebulonMind API (``nmd_host.tui.serve``, default port 9696) the same
way ``nebulondb.py`` hosts NebulonDB: a background uvicorn subprocess managed
through a PID file with persistent logs.

This script is a thin entry point; all functions live in the terminal UI
package ``nmd_host/tui/``. See :mod:`nmd_host.tui` for the implementation.

Usage:
    python nebulonmind.py                 # launch the interactive TUI
    python nebulonmind.py start           # start the NebulonMind API server (requires NebulonDB up)
    python nebulonmind.py restart         # stop (if running) then start the NebulonMind API server
    python nebulonmind.py stop            # stop the NebulonMind API server

Flags:
    --foreground / -f     run in the foreground
    --force / -F          force-stop
"""

from nmd_host.tui.main import main

if __name__ == "__main__":
    main()
