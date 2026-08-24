"""Phase 4 — standard API error responses.

Every error goes out as the standard envelope with ``success=false`` and a
human-readable ``message``. Internal details (stack traces, exception
``repr``, backend connection strings, credentials) are logged server-side
but never rendered into the HTTP response — NebulonDB credentials stay
inside ``NebulonMind`` (Phase 4.8).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .tracing import request_id_var

logger = logging.getLogger("nmd_host.api.errors")


def error_response(
    status: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {"success": False, "message": message}
    request_id = request_id or request_id_var.get()
    if request_id and request_id != "-":
        body["request_id"] = request_id
    if data is not None:
        body["data"] = data
    return JSONResponse(status_code=status, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Wire standard error handling onto an app (called from ``create_app``)."""

    @app.exception_handler(StarletteHTTPException)
    def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        response = error_response(exc.status_code, detail)
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        return response

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors: List[Dict[str, Any]] = []
        for err in exc.errors():
            errors.append(
                {
                    "loc": [str(part) for part in err.get("loc", [])],
                    "msg": err.get("msg", "invalid value"),
                    "type": err.get("type", "value_error"),
                }
            )
        return error_response(
            422, "request validation failed", {"errors": errors}
        )

    @app.exception_handler(Exception)
    def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak the exception into the response; log it for operators.
        logger.exception(
            "unhandled error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return error_response(500, "internal server error")


__all__ = ["error_response", "register_exception_handlers"]