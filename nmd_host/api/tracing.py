"""Cross-process tracing glue: the per-request correlation ID.

A contextvar is set by the correlation middleware for every inbound request
and read by log filters (structured logs), error handlers (response body)
and the NebulonDB client (outgoing ``X-Request-ID`` header). It flows through
FastAPI sync endpoints via the same async context, so one ID spans the whole
request/backend chain.
"""

from __future__ import annotations

import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nmd_request_id", default="-"
)

__all__ = ["request_id_var"]