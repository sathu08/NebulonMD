"""Operational middleware for the NebulonMind service (Step 6).

* ``CorrelationIdMiddleware`` — accept or mint a per-request ID, echo it in
  the ``X-Request-ID`` response header and publish it into the tracing
  contextvar so logs, error envelopes and outbound backend calls share it.
* ``BodyLimitMiddleware`` — reject oversized request bodies with 413 before
  they reach the application (declared Content-Length fast path + enforced
  while streaming).
* ``RateLimitMiddleware`` — in-process token-bucket limiter per client IP
  (429 with ``Retry-After``). Single-process semantics: each worker keeps its
  own buckets; for strict global limits put a gateway in front.
* ``MetricsRegistry`` — dependency-free counters/latency, rendered in
  Prometheus text format by the ``/metrics`` endpoint.

Everything is dependency-free (stdlib only) so the service adds no
third-party runtime packages.
"""

from __future__ import annotations

import logging
import time
import hmac
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Tuple

from starlette.responses import JSONResponse, Response

from .tracing import request_id_var

logger = logging.getLogger("nmd_host.api.middleware")


def _envelope(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"success": False, "message": message}
    )


# ---------------------------------------------------------------------- #
# Correlatio IDs                                                         #
# ---------------------------------------------------------------------- #

class CorrelationIdMiddleware:
    """Mint (or honour) an ``X-Request-ID`` per HTTP request."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        request_id = None
        for name, value in headers:
            if name.lower() == b"x-request-id":
                request_id = value.decode("latin-1", "replace")[:128] or None
                break
        request_id = request_id or f"nmd-{int(time.time() * 1000)}-{id(scope):x}"
        token = request_id_var.set(request_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every log record.

    Attach to the ``nmd_host`` logger (or root) once — records from all child
    loggers then carry ``request_id`` for the formatter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


# ---------------------------------------------------------------------- #
# Body size limit                                                       #
# ---------------------------------------------------------------------- #

class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Reject bodies larger than ``max_bytes`` with a 413 envelope."""

    def __init__(self, app, max_bytes: int = 1_048_576,
                 metrics: Optional["MetricsRegistry"] = None) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        self.metrics = metrics

    async def _reject(self, scope, send) -> None:
        if self.metrics is not None:
            self.metrics.counter("body_rejected")
        response = _envelope(413, f"request body exceeds {self.max_bytes} bytes")
        await response(scope, receive=None, send=send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or []:
            if name.lower() == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(scope, send)
                        return
                except ValueError:
                    pass
                break
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await self._reject(scope, send)


# ---------------------------------------------------------------------- #
# Rate limiting                                                          #
# ---------------------------------------------------------------------- #

class RateLimitMiddleware:
    """Token-bucket rate limiter keyed by client IP (per process).

    ``per_minute <= 0`` disables the limiter. Refills at
    ``per_minute / 60`` tokens/second up to a burst capacity of
    ``per_minute``. Rejections return 429 + ``Retry-After``.
    """

    def __init__(self, app, per_minute: int = 0,
                 metrics: Optional["MetricsRegistry"] = None) -> None:
        self.app = app
        self.per_minute = int(per_minute)
        self.metrics = metrics
        self._tokens: DefaultDict[str, Tuple[float, float]] = defaultdict(
            lambda: (float(self.per_minute), time.monotonic())
        )

    @staticmethod
    def _client_key(scope) -> str:
        if not scope.get("client"):
            return "unknown"
        host = scope["client"][0]
        return str(host)

    def _allow(self, key: str) -> bool:
        tokens, last = self._tokens[key]
        now = time.monotonic()
        refill = (now - last) * (self.per_minute / 60.0)
        tokens = min(float(self.per_minute), tokens + refill)
        if tokens < 1.0:
            self._tokens[key] = (tokens, now)
            return False
        self._tokens[key] = (tokens - 1.0, now)
        return True

    async def __call__(self, scope, receive, send):
        if self.per_minute <= 0:
            await self.app(scope, receive, send)
            return
        key = self._client_key(scope)
        if self._allow(key):
            await self.app(scope, receive, send)
            return
        if self.metrics is not None:
            self.metrics.counter("rate_limited")
        logger.warning("rate limit exceeded for client %s", key)
        response = _envelope(
            429, "rate limit exceeded; try again later"
        )
        response.headers["Retry-After"] = "60"
        await response(scope, receive=None, send=send)


# ---------------------------------------------------------------------- #
# Authentication (optional shared secret)                                #
# ---------------------------------------------------------------------- #

class AuthMiddleware:
    """Optional shared-secret authentication for the API.

    Disabled when ``token`` is empty (the historical open-internal posture).
    When ``NMD_API_AUTH_TOKEN`` is set, every request must present either
    ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``
    (constant-time comparison); missing/bad credentials get a 401 envelope.
    Health probes (``/health``, ``/health/live``, ``/health/ready``),
    ``/metrics``, the OpenAPI docs and the web console page/asset loads
    stay public so orchestrators, monitoring and the browser can reach them
    without a secret. Every data route — including the console's
    ``GET/PUT .../config`` — requires the token. CORS preflights (OPTIONS)
    also pass without a token.
    """

    _PUBLIC_PREFIXES = (
        "/api/NebulonMind/health",
        "/metrics",
        "/openapi.json",
        "/docs",
        "/redoc",
    )

    _STATIC_EXTENSIONS = (
        ".html",
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".map",
    )
    _INDEX_PATHS = (
        "/api/NebulonMind/dashboard",
        "/api/NebulonMind/dashboard/",
        "/api/NebulonMind/dashboard/index",
        "/api/NebulonMind/dashboard/web",
    )

    def __init__(self, app, token: str = "") -> None:
        self.app = app
        self._token = (token or "").strip()

    @classmethod
    def _is_public(cls, method: str, path: str) -> bool:
        """Whether a request may pass without credentials.

        Health probes, metrics, OpenAPI docs, CORS preflights (OPTIONS) and
        the console's static page loads (a browser does not attach a Bearer
        token to asset requests) are public. Data routes — including the
        console's own ``/config`` — require the token while it is enabled.
        """
        if method == "OPTIONS":
            return True
        if path.startswith(cls._PUBLIC_PREFIXES):
            return True
        if method not in ("GET", "HEAD"):
            return False
        return path in cls._INDEX_PATHS or path.endswith(cls._STATIC_EXTENSIONS)

    @staticmethod
    def _token_from_scope(scope) -> str:
        for name, value in scope.get("headers") or []:
            lower = name.lower()
            if lower == b"authorization":
                scheme, _, credentials = value.decode("latin-1", "replace").partition(" ")
                if scheme.lower() == "bearer" and credentials:
                    return credentials.strip()
            elif lower == b"x-api-key":
                return value.decode("latin-1", "replace").strip()
        return ""

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._token:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "")
        if self._is_public(method, path):
            await self.app(scope, receive, send)
            return
        presented = self._token_from_scope(scope)
        if presented and hmac.compare_digest(presented, self._token):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            status_code=401,
            content={"success": False, "message": "authentication required"},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive=None, send=send)


# ---------------------------------------------------------------------- #
# Metrics                                                                #
# ---------------------------------------------------------------------- #

class MetricsRegistry:
    """Dependency-free operational metrics (counters + latency aggregates)."""

    def __init__(self) -> None:
        self._requests: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
        self._latency: DefaultDict[Tuple[str, str], Tuple[float, int]] = defaultdict(
            lambda: (0.0, 0)
        )
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._started = time.time()

    def record(self, method: str, route: str, status: int, seconds: float) -> None:
        self._requests[(method, route, str(status))] += 1
        total, count = self._latency[(method, route)]
        self._latency[(method, route)] = (total + seconds, count + 1)

    def counter(self, name: str, delta: int = 1) -> None:
        self._counters[name] += delta

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")

    def render(self, version: str = "") -> str:
        lines = [
            "# HELP nmd_up Whether the NebulonMind service process is running.",
            "# TYPE nmd_up gauge",
            "nmd_up 1",
        ]
        if version:
            lines.append("# HELP nmd_version_info NebulonMind service version.")
            lines.append("# TYPE nmd_version_info gauge")
            lines.append(f'nmd_version_info{{version="{self._escape(version)}"}} 1')
        lines.append("# HELP nmd_http_requests_total HTTP requests by method/route/status.")
        lines.append("# TYPE nmd_http_requests_total counter")
        for (method, route, status), count in sorted(self._requests.items()):
            lines.append(
                f'nmd_http_requests_total{{method="{method}",route="{self._escape(route)}",'
                f'status="{status}"}} {count}'
            )
        lines.append(
            "# HELP nmd_http_request_duration_seconds_sum Total request latency by "
            "method/route."
        )
        lines.append("# TYPE nmd_http_request_duration_seconds_sum counter")
        for (method, route), (total, count) in sorted(self._latency.items()):
            if count == 0:
                continue
            lines.append(
                f'nmd_http_request_duration_seconds_sum{{method="{method}",route="{self._escape(route)}"}} '
                f"{total:.6f}"
            )
        lines.append(
            "# HELP nmd_http_request_duration_seconds_count Request count by method/route."
        )
        lines.append("# TYPE nmd_http_request_duration_seconds_count counter")
        for (method, route), (_, count) in sorted(self._latency.items()):
            if count == 0:
                continue
            lines.append(
                f'nmd_http_request_duration_seconds_count{{method="{method}",route="{self._escape(route)}"}} '
                f"{count}"
            )
        for name in ("rate_limited", "body_rejected"):
            lines.append(f"# HELP nmd_{name}_total Rejections by reason.")
            lines.append(f"# TYPE nmd_{name}_total counter")
            lines.append(f"nmd_{name}_total {self._counters[name]}")
        lines.append(f"# HELP nmd_uptime_seconds Seconds since service start.")
        lines.append("# TYPE nmd_uptime_seconds gauge")
        lines.append(f"nmd_uptime_seconds {time.time() - self._started:.0f}")
        return "\n".join(lines) + "\n"


class MetricsMiddleware:
    """Record method/route/status counts and latency for every response."""

    def __init__(self, app, registry: "MetricsRegistry") -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status_holder: Dict[str, Any] = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            route_path = getattr(route, "path", None) or scope.get("path", "")
            method = scope.get("method", "?")
            self.registry.record(
                method, route_path, status_holder["status"],
                time.monotonic() - started,
            )


__all__ = [
    "AuthMiddleware",
    "BodyLimitMiddleware",
    "CorrelationIdMiddleware",
    "MetricsMiddleware",
    "MetricsRegistry",
    "RateLimitMiddleware",
    "RequestIdFilter",
]