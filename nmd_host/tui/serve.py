"""Run the NebulonMind API service.

Usage:
    ../NebulonDB/.venv/bin/python -m nmd_host.tui.serve

Listens on ``NMD_API_HOST``/``NMD_API_PORT`` (legacy
``NEBULONDMIND_API_HOST``/``NEBULONDMIND_API_PORT`` still accepted; default
``0.0.0.0:9696``). The NebulonDB backend is reached via
``NDB_API_*`` settings (legacy ``NEBULONDB_API_*`` still accepted;
default ``localhost:6969``). Start-up and
shut-down are handled by the app lifespan in ``create_app``; this module
only wires configuration and uvicorn.

Production:

* Process manager: run FastAPI under uvicorn only — a single
  ``python -m nmd_host.tui.serve`` process is the default (systemd unit:
  deploy/nebulonmind.service).
* Graceful shutdown: ``NMD_API_GRACEFUL_SHUTDOWN_SECONDS`` (default 30)
  bounds how long uvicorn waits for in-flight requests before force-close.
* Structured logs: the ``request_id`` field (length of in-flight request)
  is attached to every log record; the correlation middleware sources it.
"""

from __future__ import annotations

import os
import logging
import uvicorn

from nmd_host.api.server import create_app
from nmd_host.utils.constants import (
    API_HOST_DEFAULT,
    API_PORT_DEFAULT,
    GRACEFUL_SHUTDOWN_SECONDS_DEFAULT,
)


logger = logging.getLogger("nmd_host.tui.serve")
HOST = os.environ.get(
    "NMD_API_HOST", os.environ.get("NEBULONDMIND_API_HOST", API_HOST_DEFAULT)
)
PORT = int(
    os.environ.get(
        "NMD_API_PORT",
        os.environ.get("NEBULONDMIND_API_PORT", str(API_PORT_DEFAULT)),
    )
)
GRACEFUL_SHUTDOWN_SECONDS = int(
    os.environ.get(
        "NMD_API_GRACEFUL_SHUTDOWN_SECONDS", str(GRACEFUL_SHUTDOWN_SECONDS_DEFAULT)
    )
)


def _install_request_id_filter() -> None:
    from nmd_host.api.middleware import RequestIdFilter

    filter_ = RequestIdFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(filter_)
    root.addFilter(filter_)
    logging.getLogger("nmd_host").addFilter(filter_)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s req=%(request_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _install_request_id_filter()
    logger.info("NebulonMind API listening on %s:%s (graceful_shutdown=%ds)", HOST, PORT, GRACEFUL_SHUTDOWN_SECONDS)
    uvicorn.run(
        create_app(),
        host=HOST,
        port=PORT,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        access_log=False,
    )


if __name__ == "__main__":
    main()