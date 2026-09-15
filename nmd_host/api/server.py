"""NebulonMind API (FastAPI) — the NebulonMind service itself.

This is the service the user asked to expose on port 9696: it wraps the
``NebulonMind`` facade (truth + vectors + graph over the NebulonDB REST API)
and exposes ``/api/NebulonMind/...`` endpoints. It is *not* the NebulonDB
server — that stays on its own port (``NEBULONDB_API_PORT``, default 6969).

Phase 4 layout:
    api/schemas.py  — request/response models (pure presentation)
    api/errors.py   — standard error envelope, internals never leak
    api/service.py  — per-user ServiceBundles + injectable providers
    api/server.py   — routes: Memory / Recall / Context / Intelligence /
                      Relationships / Service

Every route talks exclusively to its user's ``ServiceBundle`` (repository +
lifecycle manager): CRUD goes through the repository, retrieval and
context through the existing lifecycle pipeline, intelligence through the
existing decision bridge — no business logic is re-implemented here.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Message, Receive, Scope, Send

from ..core.config import (
    NebulonDBConfig,
    ServiceConfig,
    validate_production_config,
)
from nmd_host.core.models import RetentionPolicy
from nmd_host.core.registry import UserNotFoundError
from nmd_host.utils.constants import (
    AUTO_DELETE_CRON_DEFAULT,
    DEFAULT_USERNAME,
    MEMORY_CONSOLIDATION_CRON_DEFAULT,
    WEEKLY_SUMMARY_CRON_DEFAULT,
)
from nmd_host.utils.env_helpers import env_bool as _env_bool
from nmd_host.agent.schemas import AgentChatRequest
from nmd_host.agent.session import AgentSession, AgentSessionManager
from nmd_host.intelligence.converter import _temporary_expires_at, candidate_to_memory
from nmd_host.lifecycle.manager import IngestionResult
from .errors import register_exception_handlers
from .middleware import (
    AuthMiddleware,
    BodyLimitMiddleware,
    CorrelationIdMiddleware,
    MetricsMiddleware,
    MetricsRegistry,
    RateLimitMiddleware,
    RequestIdFilter,
)
from .schemas import (
    AgentChatEnvelope,
    AgentSessionCreate,
    AgentSessionData,
    AgentSessionEnvelope,
    AgentSessionListData,
    AgentSessionListEnvelope,
    BackgroundAgentEnvelope,
    BackgroundAgentReport,
    BackgroundJobStatus,
    BackgroundMemoryRunRequest,
    BackgroundSchedulerStatus,
    BackgroundStatusEnvelope,
    BackgroundTaskRunRequest,
    ContextData,
    ContextEnvelope,
    DecideData,
    DecideEnvelope,
    DeleteData,
    DeleteEnvelope,
    EvaluationData,
    EvaluationDatasetData,
    EvaluationDatasetEnvelope,
    EvaluationItemData,
    EvaluationMetrics,
    EvaluationRunEnvelope,
    EvaluationRunRequest,
    HealthData,
    HealthEnvelope,
    IngestReport,
    IntelligenceDecideRequest,
    IntelligenceProcessRequest,
    LLMStatusData,
    LLMStatusEnvelope,
    MemoryCreate,
    MemoryData,
    MemoryEnvelope,
    MemoryUpdate,
    ProcessData,
    ProcessEnvelope,
    RelateData,
    RelateEnvelope,
    SearchData,
    SearchEnvelope,
    UserCreateRequest,
    UserData,
    UserEnvelope,
)
from .service import (
    DefaultServiceProvider,
    ServiceBundle,
    ServiceProvider,
)
from .routes.dashboard import WEB_DIR, router as dashboard_router
from .routes.config import router as config_router
from .routes.chats import router as chats_router

logger = logging.getLogger("nmd_host.api.server")

SERVICE_VERSION = "v0.1"

TAGS = [
    {"name": "Memory", "description": "CRUD for Memory objects (Phase 4.2) — truth + vectors + graph writes via the repository."},
    {"name": "Recall", "description": "Semantic recall through the lifecycle pipeline (Phase 4.3) — one retrieval system."},
    {"name": "Context", "description": "Bounded, provenance-carrying LLM context (Phase 4.4) — the existing MemoryContextBuilder."},
    {"name": "Intelligence", "description": "Conversation → memory decisions and ingestion (Phase 4.5) — decision/ingestion flow, no storage bypass."},
    {"name": "Relationships", "description": "Entity linking through the existing relate() functionality (Phase 4.6)."},
    {"name": "Agent", "description": "Agent Runtime: stateless tool-calling chat over the memory tools (remember/recall)."},
    {"name": "Evaluation", "description": "Agent Evaluation: benchmark the agent over a dataset (retrieval, tool selection, answer correctness, hallucination, latency, tokens)."},
    {"name": "Background", "description": "Background Agents: nightly memory consolidation (Memory Agent) and weekly work summaries (Task Agent), scheduled in-process; manual run triggers + scheduler status."},
    {"name": "User", "description": "Explicit username registration: a username is mapped to an opaque user_id via /user/create_user; /user/setup switches to an already-registered username. Unregistered usernames are rejected everywhere."},
    {"name": "Service", "description": "Health checks and service metadata (Phase 4.12)."},
]

_KEY_ENV_VARS = (
    "NMD_LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DASHSCOPE_API_KEY",
)


def _redact_llm_error(message: str) -> str:
    """Sanitize an LLM provider error: never leak API keys in responses."""
    import os

    text = (message or "")[:300]
    for var in _KEY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            text = text.replace(value, "<redacted>")
            text = text.replace(value[:8] + "..." + value[-4:], "<redacted>")
    return text


def _background_default_user() -> str:
    """User segment the scheduled background jobs run for.

    Configurable via ``NMD_BACKGROUND_USER`` (default ``nmd_user_01``). The
    API manual triggers accept any ``user_id``; only the *automatic* nightly
    consolidation / weekly summary run for this segment.
    """
    import os

    default = os.environ.get("NMD_BACKGROUND_USER", DEFAULT_USERNAME).strip() or DEFAULT_USERNAME
    if default.startswith("/"):
        default = DEFAULT_USERNAME
    return default


def _build_extractor(config: ServiceConfig):
    """Return an ``LLMExtractor`` when ``NMD_LLM_EXTRACTOR=true``, else None.

    Falls back to the rule-based extractor (None) whenever the LLM provider
    is missing, misconfigured or fails to construct — the decision pipeline
    never becomes unavailable because of the toggle.
    """
    if not config.llm_extractor:
        return None
    try:
        from ..intelligence.llm_extractor import LLMExtractor
        from ..intelligence.providers import provider_from_env

        return LLMExtractor(provider_from_env())
    except Exception as exc:  # provider unavailable/misconfigured → rules
        logger.warning(
            "LLM extractor requested but unavailable (%s); using rule extractor",
            exc,
        )
        return None


def _intelligence(bundle, extractor):
    from nmd_host.intelligence.bridge import MemoryIntelligence

    return MemoryIntelligence(
        bundle.repository, user_id=bundle.user_id, extractor=extractor
    )


def _decide_with_fallback(bundle, conversation, extractor):
    """Decide with the given extractor, falling back to rules.

    Falls back to the deterministic rule extractor when the LLM extractor
    either raises or returns *no* decisions for non-trivial input — the
    pipeline never silently "forgets" a turn because the model returned an
    empty payload.

    Returns ``(decisions, intelligence)`` so callers can also inspect the
    engine's version of the verdicts (e.g. include-rejected mode).
    """
    intelligence = _intelligence(bundle, extractor)
    try:
        decisions = intelligence.decide(conversation)
    except Exception as exc:
        if extractor is None:
            raise
        logger.warning(
            "LLM extraction failed (%s); falling back to rule extraction", exc
        )
        intelligence = _intelligence(bundle, None)
        decisions = intelligence.decide(conversation)
    if extractor is not None and not decisions:
        logger.warning(
            "LLM extraction returned no decisions; falling back to rule extraction"
        )
        intelligence = _intelligence(bundle, None)
        decisions = intelligence.decide(conversation)
    return decisions, intelligence


class ConsoleAssetCacheMiddleware:
    """Do not let browsers cache console assets: every refresh revalidates,
    so a fixed ``main.js`` is picked up without a manual cache purge."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            is_console = (
                path in ("/", "/index", "/api/NebulonMind/web")
                or path == "/api/NebulonMind/dashboard"
                or path.startswith("/api/NebulonMind/dashboard/")
            )
            if is_console:

                async def send_no_cache(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        headers = [
                            (k, v)
                            for k, v in message.get("headers", [])
                            if k.lower() != b"cache-control"
                        ]
                        headers.append((b"cache-control", b"no-cache"))
                        message = {**message, "headers": headers}
                    await send(message)

                await self.app(scope, receive, send_no_cache)
                return
        await self.app(scope, receive, send)


def create_app(
    provider: Optional[ServiceProvider] = None,
    config: Optional[ServiceConfig] = None,
) -> FastAPI:
    """Build the NebulonMind API app.

    ``provider`` may be injected (tests use ``InMemoryServiceProvider`` so
    no NebulonDB is required); the default is the production
    ``DefaultServiceProvider`` backed by the NebulonDB REST API. ``config``
    may be injected (tests/operators); the default is
    ``ServiceConfig.from_env()`` which is validated fail-fast whenever
    ``NMD_ENV=production`` (P2 — invalid production configuration refuses to
    boot rather than running with a weakened posture).
    """
    config = config or ServiceConfig.from_env()
    provider = provider or DefaultServiceProvider(
        backend_retries=config.backend_retries
    )

    problems = validate_production_config(config, NebulonDBConfig.from_env())
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            f"invalid production configuration (NMD_ENV=production): {details}"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: reachability probe is advisory — if the backend is down
        # the service still boots, /health/live stays green and
        # /health/ready reports not-ready (503) until NebulonDB returns.
        if provider.verify_backend():
            logger.info("NebulonMind API started; NebulonDB backend reachable")
        else:
            logger.warning(
                "NebulonMind API started; NebulonDB backend unreachable "
                "(readiness reports not-ready until it returns)"
            )
        if config.expected_backend_version:
            if provider.name != "nebulondb":
                logger.info(
                    "NMD_EXPECTED_BACKEND_VERSION=%s set but provider has no "
                    "backend version endpoint (in-memory provider); skipped",
                    config.expected_backend_version,
                )
            else:
                try:
                    provider.verify_backend()
                    logger.info(
                        "backend compatibility check passed "
                        "(expected NMD_EXPECTED_BACKEND_VERSION=%s; backend "
                        "has no version endpoint, verified reachability)",
                        config.expected_backend_version,
                    )
                except Exception:  # pragma: no cover - defensive startup
                    logger.warning(
                        "backend compatibility check failed; startup policy is "
                        "advisory (readiness will report not-ready)"
                    )
        # Background agents: start the scheduler with the service.
        try:
            await app.state.scheduler.start()
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("background scheduler failed to start")
        # Username registry: ensure the identity corpus/segment exist and
        # register the configured background user so scheduled jobs run. This
        # is separate from per-request registration — any *client* username is
        # still registered explicitly via /user/create_user (never auto-seeded).
        try:
            provider.bootstrap()
            default_user = _background_default_user()
            if default_user:
                provider.create_user(default_user)
        except Exception:  # pragma: no cover - defensive startup
            logger.warning("username registry bootstrap failed (retried per request)")
            if provider.verify_backend():
                logger.exception("username registry bootstrap failed with backend up")
        yield
        # Shutdown: stop accepting new requests and release every user's
        # repository/mind resources (in-flight requests are allowed to
        # finish by uvicorn's graceful shutdown -- see serve.py).
        try:
            await app.state.scheduler.stop()
        except Exception:  # pragma: no cover - defensive shutdown
            logger.exception("background scheduler failed to stop cleanly")
        provider.close()
        app.state.sessions.clear()
        logger.info("NebulonMind API shut down; provider closed")

    app = FastAPI(
        title="NebulonMind API",
        description=(
            "Memory layer service for AI assistants, built on NebulonDB. "
            "Every write/read is persisted through the NebulonDB REST API "
            "and ranked through the lifecycle pipeline; decision "
            "endpoints are available under /intelligence. "
            "All responses use the StandardResponse envelope "
            "{success, message, data}. Clients identify by username: they "
            "register explicitly via /user/create_user (which returns the "
            "opaque user_id) and supply that username on every other route — "
            "unregistered usernames are rejected (403). There is no password "
            "and no auth middleware; NebulonDB credentials stay inside "
            "NebulonMind (never leave the service, never echoed by health, "
            "metrics, OpenAPI or config). Health, metrics and OpenAPI docs "
            "are public."
        ),
        version=SERVICE_VERSION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    metrics = MetricsRegistry()
    app.state.metrics = metrics
    app.state.config = config
    app.state.provider = provider

    # Background agents scheduler (registered jobs below, once
    # ``_bundle`` exists; started/stopped with the service lifecycle).
    from ..agents import BackgroundScheduler

    # Durable scheduler state (``NMD_BACKGROUND_PERSIST_STATE=true``) persists
    # last-run/results in NebulonDB so a restart does not reset job history.
    # Needs the production provider's shared backend client; skipped otherwise.
    backend_client = getattr(provider, "client", None)
    job_state_store = None
    if config.background_persist_state and backend_client is not None:
        from ..agents.state_store import NebulonDBJobStateStore

        job_state_store = NebulonDBJobStateStore(backend_client)
    scheduler = BackgroundScheduler(store=job_state_store)
    app.state.scheduler = scheduler

    from ..agent.config import AgentConfig

    _agent_config = AgentConfig.from_env()
    app.state.agent_config = _agent_config

    # Durable agent sessions (``NMD_AGENT_DURABLE_SESSIONS=true``): a
    # NebulonDB-backed store lets the in-memory SessionManager re-hydrate
    # sessions after a restart. Supports only the production provider
    # (no client under the in-memory test provider).
    session_store = None
    if _agent_config.durable_sessions and backend_client is not None:
        from ..agent.session_store import NebulonDBSessionStore

        session_store = NebulonDBSessionStore(backend_client)
    app.state.sessions = AgentSessionManager(
        max_sessions_per_user=_agent_config.max_sessions_per_user,
        session_ttl_seconds=float(_agent_config.session_ttl_seconds),
        store=session_store,
    )

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "X-Request-ID",
            ],
        )
    # Middleware order (first added = innermost): metrics → body limit →
    # rate limit → auth → correlation ID (outermost).
    app.add_middleware(MetricsMiddleware, registry=metrics)
    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_body_bytes, metrics=metrics)
    app.add_middleware(
        RateLimitMiddleware,
        per_minute=config.rate_limit_per_minute,
        metrics=metrics,
    )
    app.add_middleware(AuthMiddleware, token=config.auth_token)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(ConsoleAssetCacheMiddleware)

    _request_id_filter = RequestIdFilter()
    nmd_logger = logging.getLogger("nmd_host")
    if not any(
        isinstance(f, RequestIdFilter) for f in nmd_logger.filters
    ):
        nmd_logger.addFilter(_request_id_filter)

    register_exception_handlers(app)

    def _bundle(user_id: str) -> ServiceBundle:
        try:
            return provider.bundle(user_id)
        except UserNotFoundError as exc:
            raise HTTPException(
                status_code=403,
                detail=str(exc),
                headers={"WWW-Authenticate": "NebulonMindUser"},
            ) from exc

    # Default background jobs: nightly memory consolidation
    # (2:00 AM), a weekly summary (Sundays 9:00 AM) and the auto-delete
    # sweep for expired memories (daily, cron from nebulonmd.cfg) — all
    # for the configured default user. Manual triggers hit the endpoints
    # below.
    def _register_background_jobs() -> None:
        from ..agents import MemoryAgent, TaskAgent

        default_user = _background_default_user()
        auto_delete_cron = (
            os.environ.get("NMD_LIFECYCLE_AUTO_CLEANUP_CRON", "").strip()
            or AUTO_DELETE_CRON_DEFAULT
        )

        def memory_job() -> dict:
            bundle = _bundle(default_user)
            return MemoryAgent(
                bundle.repository,
                bundle.manager,
                user_id=default_user,
                mode="consolidate",
            ).run()

        def task_job() -> dict:
            bundle = _bundle(default_user)
            return TaskAgent(
                bundle.repository,
                bundle.manager,
                user_id=default_user,
                days=7,
            ).run()

        def auto_delete_job() -> dict:
            # Re-read on every run so a cfg change + restart applies without
            # re-registering; PERMANENT/ARCHIVED memories are never touched.
            if not _env_bool("NMD_LIFECYCLE_AUTO_CLEANUP", False):
                return {
                    "agent": "auto_delete",
                    "user_id": default_user,
                    "skipped": True,
                    "reason": "NMD_LIFECYCLE_AUTO_CLEANUP is disabled",
                }
            bundle = _bundle(default_user)
            deleted = bundle.manager.cleanup(user_id=default_user)
            return {
                "agent": "auto_delete",
                "user_id": default_user,
                "skipped": False,
                "deleted_memory_ids_count": deleted,
            }

        scheduler.register(
            "memory",
            memory_job,
            schedule=os.environ.get(
                "NMD_MEMORY_CONSOLIDATION_CRON", MEMORY_CONSOLIDATION_CRON_DEFAULT
            ),
        )
        scheduler.register(
            "task_weekly_summary",
            task_job,
            schedule=os.environ.get(
                "NMD_WEEKLY_SUMMARY_CRON", WEEKLY_SUMMARY_CRON_DEFAULT
            ),
        )
        scheduler.register(
            "auto_delete_expired",
            auto_delete_job,
            schedule=auto_delete_cron,
        )

    _register_background_jobs()

    def _require_memory(bundle: ServiceBundle, memory_id: str):
        memory = bundle.repository.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id!r} not found")
        return memory

    # ------------------------------------------------------------------ #
    # Service (4.12): liveness + readiness + metrics                   #
    # ------------------------------------------------------------------ #

    def _health_data(backend: str = "down") -> HealthData:
        return HealthData(
            service="NebulonMind",
            version=app.version,
            provider=provider.name,
            backend=backend,
            minds=provider.user_count(),
        )

    @app.get(
        "/api/NebulonMind/health/live",
        tags=["Service"],
        summary="Liveness probe",
        description=(
            "Answers 200 as long as the process is up and accepting "
            "requests. Never depends on the backend — a dead NebulonDB must "
            "not take the whole service out of the load balancer."
        ),
        response_model=HealthEnvelope,
    )
    def health_live() -> HealthEnvelope:
        return HealthEnvelope(
            message="alive",
            data=_health_data(
                "up" if provider.verify_backend() else "in-memory"
                if provider.name == "in-memory" else "down"
            ),
        )

    @app.get(
        "/api/NebulonMind/health/ready",
        tags=["Service"],
        summary="Readiness probe",
        description=(
            "Answers 200 only when the service can do useful work: for the "
            "NebulonDB provider this requires the backend to be reachable "
            "and authenticated. Returns 503 while NebulonDB is unavailable — "
            "the load balancer then drains this instance. The service still "
            "boots with the backend down (startup policy) and recovers "
            "without a restart once the backend returns."
        ),
        response_model=HealthEnvelope,
        responses={503: {"description": "not ready: backend unavailable"}},
    )
    def health_ready() -> HealthEnvelope:
        if provider.verify_backend():
            return HealthEnvelope(
                message="ready", data=_health_data("up")
            )
        if provider.name == "in-memory":
            return HealthEnvelope(
                message="ready", data=_health_data("in-memory")
            )
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content=HealthEnvelope(
                success=False,
                message="not ready: NebulonDB backend unavailable",
                data=_health_data("down"),
            ).model_dump(),
        )

    @app.get(
        "/api/NebulonMind/health",
        tags=["Service"],
        summary="Service + backend health (compat: liveness + backend info)",
        description=(
            "Compatibility alias that merged probes historically: reports "
            "service metadata and backend reachability exactly like the "
            "liveness probe. New deployments should use /health/live and "
            "/health/ready for unambiguous semantics. Connection settings "
            "and credentials are never included."
        ),
        response_model=HealthEnvelope,
    )
    def health() -> HealthEnvelope:
        return HealthEnvelope(
            message="NebulonMind service running",
            data=_health_data(
                "up"
                if provider.verify_backend()
                else ("down" if provider.name == "nebulondb" else "in-memory")
            ),
        )

    @app.get(
        "/metrics",
        tags=["Service"],
        summary="Prometheus metrics (text exposition)",
        include_in_schema=False,
    )
    def metrics_endpoint() -> PlainTextResponse:
        return PlainTextResponse(
            metrics.render(SERVICE_VERSION),
            media_type="text/plain; version=0.0.4",
        )

    # ------------------------------------------------------------------ #
    # User registry (explicit registration)                              #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/user/create_user",
        status_code=201,
        tags=["User"],
        summary="Register a username (explicit, idempotent)",
        description=(
            "Registers a username, generating the opaque ``user_id`` "
            "(``unique_id``) it maps to. Idempotent: re-calling with an "
            "already-registered username returns the stored user_id with "
            "``created=false``. Usernames are *never* auto-registered — every "
            "other endpoint interprets its ``user_id`` query parameter as a "
            "username and rejects unregistered ones (403). No password is "
            "used; a username alone identifies the client."
        ),
        response_model=UserEnvelope,
        responses={
            403: {"description": "username registration rejected"},
        },
    )
    def create_user(
        request: UserCreateRequest = Body(..., description="Username to register"),
    ) -> UserEnvelope:
        username = request.username
        try:
            user_id, created = provider.create_user(username)
        except Exception as exc:  # defensive: wrap generic registry failures
            logger.warning("create_user(%r) failed: %s", username, exc)
            raise HTTPException(
                status_code=503, detail="user registry unavailable"
            ) from exc
        return UserEnvelope(
            message=(
                f"user {username!r} created" if created
                else f"user {username!r} already registered"
            ),
            data=UserData(username=username, user_id=user_id, created=created),
        )

    @app.get(
        "/api/NebulonMind/user/resolve",
        tags=["User"],
        summary="Resolve a registered username (never auto-registers)",
        description=(
            "Looks up the opaque ``user_id`` mapped to an already-registered "
            "username. Unlike ``/user/create_user`` this endpoint NEVER "
            "registers: an unregistered username returns 404 so the client "
            "can offer the ``/create`` flow instead of silently creating."
        ),
        response_model=UserEnvelope,
        responses={404: {"description": "username not registered"}},
    )
    def resolve_user(
        username: str = Query(..., min_length=1, description="Registered username"),
    ) -> UserEnvelope:
        try:
            user_id = provider.registry().resolve(username)
        except UserNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"user {username!r} not registered; call /user/create_user",
            ) from None
        return UserEnvelope(
            message=f"user {username!r} resolved",
            data=UserData(username=username, user_id=user_id, created=False),
        )

    @app.post(
        "/api/NebulonMind/user/setup",
        tags=["User"],
        summary="Switch to an already-registered username (never auto-registers)",
        description=(
            "POST twin of ``/user/resolve``: activates an existing username "
            "and returns its ``user_id``. Unlike ``/user/create_user`` this "
            "endpoint NEVER registers — an unregistered username returns 404 "
            "(via the standard envelope) so the client can offer the "
            "``/create`` flow instead. Idempotent and side-effect free."
        ),
        response_model=UserEnvelope,
        responses={404: {"description": "username not registered"}},
    )
    def setup_user(
        request: UserCreateRequest = Body(..., description="Username to activate"),
    ) -> UserEnvelope:
        username = request.username
        try:
            user_id = provider.registry().resolve(username)
        except UserNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"user {username!r} not registered; call /user/create_user",
            ) from None
        return UserEnvelope(
            message=f"user {username!r} active",
            data=UserData(username=username, user_id=user_id, created=False),
        )

    # ------------------------------------------------------------------ #
    # Memory API (4.2)                                                   #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/memory",
        status_code=201,
        tags=["Memory"],
        summary="Store a memory",
        description=(
            "Persists a Memory across truth (COSMOS), meaning (ORBIT "
            "vectors) and relationships (ORBIT Mesh). The memory is pinned "
            "to the ``user_id`` query parameter. With ``gate=true`` the same "
            "ingest gate as ``/intelligence/process`` runs first: "
            "expired memories are refused, TEMPORARY memories without an "
            "expiry are stamped with the ``NMD_TEMPORARY_TTL_SECONDS`` "
            "default, and a duplicate resolves to the existing memory "
            "(HTTP 200, idempotent) instead of being stored again."
        ),
        response_model=MemoryEnvelope,
    )
    def store_memory(
        memory: MemoryCreate = Body(..., description="The memory to store"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
        gate: bool = Query(
            False,
            description=(
                "Run the ingest gate (expiry + retention + "
                "deduplication) before storing. Idempotent: a duplicate "
                "returns the existing memory with HTTP 200."
            ),
        ),
    ) -> MemoryEnvelope:
        bundle = _bundle(user_id)
        memory.user_id = bundle.user_id
        superseded_after: List[str] = []
        if gate:
            if (
                memory.lifecycle.retention_policy is RetentionPolicy.TEMPORARY
                and memory.lifecycle.expires_at is None
            ):
                memory.lifecycle.expires_at = _temporary_expires_at()
            verdict = bundle.manager.ingest(memory, user_id=bundle.user_id)
            if verdict.action == "DUPLICATE":
                existing = verdict.existing_memory or memory
                return JSONResponse(
                    status_code=200,
                    content=MemoryEnvelope(
                        message=f"{verdict.reason}; existing memory returned",
                        data=MemoryData(memory=existing),
                    ).model_dump(mode="json"),
                )
            if verdict.action in ("INVALID", "EXPIRED"):
                raise HTTPException(status_code=422, detail=verdict.reason)
            superseded_after = verdict.superseded_memory_ids
        try:
            stored = bundle.repository.create(memory)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if superseded_after:
            bundle.manager.archive_superseded(stored.memory_id or "", superseded_after)
        message = (
            f"memory stored ({len(superseded_after)} conflicting "
            "memory/memories superseded)"
            if superseded_after
            else "memory stored"
        )
        return MemoryEnvelope(message=message, data=MemoryData(memory=stored))

    @app.get(
        "/api/NebulonMind/memory/{memory_id}",
        tags=["Memory"],
        summary="Get a memory",
        response_model=MemoryEnvelope,
    )
    def get_memory(
        memory_id: str,
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> MemoryEnvelope:
        memory = _require_memory(_bundle(user_id), memory_id)
        return MemoryEnvelope(message="ok", data=MemoryData(memory=memory))

    @app.put(
        "/api/NebulonMind/memory/{memory_id}",
        tags=["Memory"],
        summary="Update a memory",
        description=(
            "Partially updates a memory: only the provided fields are "
            "replaced, everything else is preserved."
        ),
        response_model=MemoryEnvelope,
    )
    def update_memory(
        memory_id: str,
        update: MemoryUpdate = Body(..., description="Fields to replace"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> MemoryEnvelope:
        data = update.model_dump(exclude_unset=True, mode="json")
        try:
            updated = _bundle(user_id).repository.update(memory_id, data)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"memory {memory_id!r} not found")
        return MemoryEnvelope(message="memory updated", data=MemoryData(memory=updated))

    @app.delete(
        "/api/NebulonMind/memory/{memory_id}",
        tags=["Memory"],
        summary="Delete a memory",
        description="Deletes in the inverse write order: graph → vector → truth.",
        response_model=DeleteEnvelope,
    )
    def delete_memory(
        memory_id: str,
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> DeleteEnvelope:
        deleted = _bundle(user_id).repository.delete(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"memory {memory_id!r} not found")
        return DeleteEnvelope(
            message="memory deleted", data=DeleteData(memory_id=memory_id)
        )

    # ------------------------------------------------------------------ #
    # Recall API (4.3)                                                   #
    # ------------------------------------------------------------------ #

    @app.get(
        "/api/NebulonMind/search",
        tags=["Recall"],
        summary="Semantic recall (lifecycle-ranked)",
        description=(
            "Retrieves memories through the single lifecycle pipeline "
            "(retention filter → ranking → deduplication → top-k). With "
            "``expand=true`` candidates first pass through depth-1 graph "
            "expansion before the same ranking pipeline."
        ),
        response_model=SearchEnvelope,
    )
    def search(
        query: str = Query(..., min_length=1, description="Natural-language query"),
        top_k: int = Query(5, ge=1, le=config.max_top_k, description="Number of memories to return"),
        expand: bool = Query(False, description="depth-1 graph expansion"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> SearchEnvelope:
        bundle = _bundle(user_id)
        if expand:
            candidates = bundle.recall_candidates(query, top_k * 3, expand=True)
            results = bundle.manager.retriever.filter_and_rank(
                query, candidates, top_k=top_k, user_id=bundle.user_id
            )
        else:
            results = bundle.manager.retrieve(query, top_k=top_k, user_id=bundle.user_id)
        return SearchEnvelope(
            message=f"{len(results)} memory(ies) recalled",
            data=SearchData(query=query, top_k=top_k, expand=expand, results=results),
        )

    # ------------------------------------------------------------------ #
    # Context API (4.4)                                                  #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/memory/context",
        tags=["Context"],
        summary="Build bounded LLM context for a query",
        description=(
            "Retrieves through the lifecycle pipeline, then formats a bounded, "
            "provenance-carrying context string with the existing "
            "MemoryContextBuilder (no call to any LLM)."
        ),
        response_model=ContextEnvelope,
    )
    def build_context(
        query: str = Query(..., min_length=1, description="Natural-language query"),
        top_k: int = Query(5, ge=1, le=config.max_top_k),
        max_items: int = Query(10, ge=1, le=config.max_top_k),
        max_characters: int = Query(
            6000, ge=100, le=config.max_context_characters
        ),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> ContextEnvelope:
        bundle = _bundle(user_id)
        memories = bundle.manager.retrieve(query, top_k=top_k, user_id=bundle.user_id)
        context = bundle.manager.build_context(
            memories,
            max_items=max_items,
            max_characters=max_characters,
        )
        return ContextEnvelope(
            message="context built",
            data=ContextData(
                query=query,
                context=context,
                memory_ids=[m.memory_id for m in memories],
            ),
        )

    # ------------------------------------------------------------------ #
    # Intelligence API (4.5)                                             #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/intelligence/decide",
        tags=["Intelligence"],
        summary="Conversation → memory decisions (no storage)",
        description=(
            "Runs the existing MemoryDecisionEngine over a conversation and "
            "returns validated decisions. Nothing is persisted."
        ),
        response_model=DecideEnvelope,
    )
    def decide(
        request: IntelligenceDecideRequest = Body(..., description="Conversation input"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> DecideEnvelope:
        bundle = _bundle(user_id)
        conversation = request.to_conversation()
        accepted, intelligence = _decide_with_fallback(
            bundle, conversation, _build_extractor(config)
        )
        rejected: Optional[List] = None
        if request.include_rejected:
            rejected = [
                d for d in intelligence.engine.decide(conversation) if not d.should_remember
            ]
        return DecideEnvelope(
            message=f"{len(accepted)} decision(s)",
            data=DecideData(decisions=accepted, rejected=rejected),
        )

    @app.post(
        "/api/NebulonMind/intelligence/process",
        tags=["Intelligence"],
        summary="Conversation → decisions → lifecycle gate → store",
        description=(
            "Decides over a conversation, gates each candidate "
            "through the lifecycle manager's ingest() and stores "
            "only the STORE-approved memories. DUPLICATE / EXPIRED / INVALID "
            "candidates are reported with their reason and never stored."
        ),
        response_model=ProcessEnvelope,
    )
    def process(
        request: IntelligenceProcessRequest = Body(..., description="Conversation input"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> ProcessEnvelope:
        bundle = _bundle(user_id)
        conversation = request.to_conversation()
        decisions, _ = _decide_with_fallback(
            bundle, conversation, _build_extractor(config)
        )
        ingestions: List[IngestReport] = []
        if request.persist:
            for decision in decisions:
                memory = candidate_to_memory(decision.candidate, bundle.user_id)
                verdict: IngestionResult = bundle.manager.ingest(
                    memory, user_id=bundle.user_id
                )
                existing_id = (
                    verdict.existing_memory.memory_id
                    if verdict.existing_memory is not None
                    else None
                )
                report = IngestReport(
                    action=verdict.action,
                    reason=verdict.reason,
                    existing_memory_id=existing_id,
                    superseded_memory_ids=verdict.superseded_memory_ids,
                )
                if verdict.action in ("STORE", "SUPERSEDE"):
                    stored = bundle.repository.create(memory)
                    if verdict.action == "SUPERSEDE":
                        bundle.manager.archive_superseded(
                            stored.memory_id or "", verdict.superseded_memory_ids
                        )
                    report.memory_id = stored.memory_id
                ingestions.append(report)
        return ProcessEnvelope(
            message=(
                f"{sum(1 for r in ingestions if r.action in ('STORE', 'SUPERSEDE'))} "
                "memory(ies) stored"
            ),
            data=ProcessData(decisions=decisions, ingestions=ingestions),
        )

    # ------------------------------------------------------------------ #
    # Relationship API (4.6)                                             #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/memory/{memory_id}/relate",
        tags=["Relationships"],
        summary="Link a memory to an entity",
        description="Adds a memory → entity edge in the graph (idempotent).",
        response_model=RelateEnvelope,
    )
    def relate_memory(
        memory_id: str,
        entity: str = Query(..., min_length=1, description="Entity label to link"),
        relation: str = Query(
            "HAS_ENTITY", min_length=1, description="Edge relation label"
        ),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> RelateEnvelope:
        bundle = _bundle(user_id)
        _require_memory(bundle, memory_id)
        bundle.relate(memory_id, entity, relation)
        return RelateEnvelope(
            message=f"linked {memory_id} -> {entity}",
            data=RelateData(memory_id=memory_id, entity=entity, relation=relation),
        )

    # ------------------------------------------------------------------ #
    # Agent API                                                          #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/agent/chat",
        tags=["Agent"],
        summary="Agent chat: tool-calling loop over the user's memory",
        description=(
            "Runs the AgentRuntime: an LLM (NMD_LLM_PROVIDER) either "
            "answers directly or calls the memory tools — remember persists "
            "new memories (decision → lifecycle gate → NebulonDB), recall searches "
            "existing ones. The runtime is stateless: prior turns arrive via "
            "{messages}, persistent state lives only in NebulonDB."
        ),
        response_model=AgentChatEnvelope,
        responses={503: {"description": "agent unavailable: LLM provider not configured"}},
    )
    def agent_chat(
        request: AgentChatRequest = Body(..., description="Agent chat input"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> AgentChatEnvelope:
        bundle = _bundle(user_id)
        from ..agent import AgentRuntime, build_memory_toolkit
        from ..agent.schemas import AgentMessage
        from ..intelligence.providers import LLMProviderError, provider_from_env, provider_from_env_with_model

        # Resolve the session transcript (if a session_id was supplied): an
        # unknown/closed session is not an error for chat — it degrades to a
        # stateless one-shot as if the client passed its own history.
        resolved_user = bundle.user_id
        session = None
        prior: List[AgentMessage] = list(request.messages or [])
        if request.session_id:
            session = app.state.sessions.get(resolved_user, request.session_id)
            if session is None:
                logger.info(
                    "agent chat session %s unknown/closed for %s; treating as stateless",
                    request.session_id, user_id,
                )
            else:
                prior = list(session.transcript) + prior

        # Pre-retrieval grounding: surface stored memory relevant to the
        # current utterance before the LLM turn (same lifecycle pipeline the
        # recall tool uses). This anchors answers to what is actually stored
        # even when the model would otherwise answer without recalling. A
        # no-op when nothing relevant is stored, so the agent stays free.
        grounding_memories = bundle.manager.retrieve(
            request.text, top_k=min(5, config.max_top_k), user_id=resolved_user
        )
        if grounding_memories:
            grounding_context = bundle.manager.build_context(
                grounding_memories, max_items=5, max_characters=2000
            )
            if grounding_context:
                prior = [
                    AgentMessage(
                        role="tool",
                        content=f"relevant memories:\n{grounding_context}",
                    ),
                    *prior,
                ]

        try:
            agent_model = config.agent_model if config.agent_model else None
            llm = provider_from_env_with_model(model=agent_model)
        except Exception as exc:
            logger.warning("agent chat unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="agent unavailable: LLM provider not configured"
                " (set NMD_LLM_PROVIDER and the matching key/model)",
            ) from exc
        tools = build_memory_toolkit(
            bundle.repository,
            bundle.manager,
            resolved_user,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            extractor=_build_extractor(config),
        )
        runtime = AgentRuntime(llm, tools)
        try:
            data = runtime.chat(request.text, messages=prior)
        except LLMProviderError as exc:
            logger.warning("agent chat LLM call failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="agent LLM call failed (timeout / unavailable / rate-limited)",
            ) from exc
        if session is not None and data.transcript:
            app.state.sessions.append_messages(
                resolved_user, session.session_id, data.transcript
            )
        # Auto-save the transcript into the DB chat history (best-effort):
        # session/conversation-scoped API chats share one history record
        # (id = session_id) visible under /chats. Console chats carry no
        # session_id and keep their own POST /chats saves — no duplicates.
        # A history failure never fails the chat response.
        chat_key = request.session_id or request.conversation_id
        if chat_key and data.transcript:
            try:
                from ..stores.chat_history_store import history_store_for, transcript_to_chat
                history_store_for(provider, user_id).save(
                    transcript_to_chat(chat_key, user_id, data.transcript)
                )
            except Exception as exc:
                logger.warning(
                    "agent chat history auto-save failed for %s: %s", user_id, exc
                )
        return AgentChatEnvelope(message="agent replied", data=data)

    # ------------------------------------------------------------------ #
    # Agent sessions                                                       #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/agent/session",
        status_code=201,
        tags=["Agent"],
        summary="Create an agent session",
        description=(
            "Starts a logical session for the user. Sessions live in process "
            "memory only (bounded by NMD_AGENT_MAX_SESSIONS and "
            "NMD_AGENT_SESSION_TTL_SECONDS); durable memory content still "
            "flows through NebulonDB, never a new persistent store. Pass the "
            "returned session_id to /agent/chat to keep a multi-turn "
            "conversation coherent without resending history."
        ),
        response_model=AgentSessionEnvelope,
        responses={400: {"description": "session capacity exhausted for this user"}},
    )
    def create_agent_session(
        body: AgentSessionCreate = Body(default_factory=AgentSessionCreate),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> AgentSessionEnvelope:
        resolved_user = _bundle(user_id).user_id
        try:
            session = app.state.sessions.create(resolved_user, body.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return AgentSessionEnvelope(
            message="agent session created", data=_session_data(session)
        )

    @app.get(
        "/api/NebulonMind/agent/session/{session_id}",
        tags=["Agent"],
        summary="Get an agent session",
        description=(
            "Returns the session if it exists, is active and not expired. "
            "404 for unknown, closed or expired sessions."
        ),
        response_model=AgentSessionEnvelope,
        responses={404: {"description": "session not found / inactive"}},
    )
    def get_agent_session(
        session_id: str,
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> AgentSessionEnvelope:
        resolved_user = _bundle(user_id).user_id
        session = app.state.sessions.get(resolved_user, session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"agent session {session_id!r} not found (or closed/expired)"
            )
        return AgentSessionEnvelope(
            message="ok", data=_session_data(session)
        )

    @app.get(
        "/api/NebulonMind/agent/sessions",
        tags=["Agent"],
        summary="List a user's active agent sessions",
        description=(
            "Returns the summary of every live session for the user "
            "(no transcripts — use GET /agent/session/{id} for a single one)."
        ),
        response_model=AgentSessionListEnvelope,
    )
    def list_agent_sessions(
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> AgentSessionListEnvelope:
        resolved_user = _bundle(user_id).user_id
        session_data = [
            _session_data(session)
            for session in app.state.sessions.list_active(resolved_user)
        ]
        return AgentSessionListEnvelope(
            message=f"{len(session_data)} active session(s)",
            data=AgentSessionListData(sessions=session_data),
        )

    @app.delete(
        "/api/NebulonMind/agent/session/{session_id}",
        tags=["Agent"],
        summary="Close an agent session",
        description=(
            "Closes and drops the session (in-memory). Its memories stay in "
            "NebulonDB — closing a session never deletes stored memories."
        ),
        response_model=DeleteEnvelope,
        responses={404: {"description": "session not found / inactive"}},
    )
    def close_agent_session(
        session_id: str,
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> DeleteEnvelope:
        resolved_user = _bundle(user_id).user_id
        if not app.state.sessions.close(resolved_user, session_id):
            raise HTTPException(
                status_code=404, detail=f"agent session {session_id!r} not found (or already closed)"
            )
        return DeleteEnvelope(
            message="agent session closed",
            data=DeleteData(memory_id=session_id),
        )

    def _session_data(session: "AgentSession") -> AgentSessionData:
        return AgentSessionData(
            session_id=session.session_id,
            user_id=session.user_id,
            created_at=session.created_at,
            last_activity=session.last_activity,
            status=session.status,
            metadata=session.metadata,
            message_count=len(session.transcript),
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #

    @app.get(
        "/api/NebulonMind/evaluation/dataset",
        tags=["Evaluation"],
        summary="Bundled evaluation dataset",
        description=(
            "Returns the bundled benchmark dataset (memory_test_v1) that "
            "POST /evaluation/run evaluates by default."
        ),
        response_model=EvaluationDatasetEnvelope,
    )
    def evaluation_dataset() -> EvaluationDatasetEnvelope:
        from ..evaluation.runner import DATASET_PATH, load_dataset

        name = "memory_test_v1"
        description = ""
        if DATASET_PATH.exists():
            import json

            raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            name = raw.get("name", name)
            description = raw.get("description", "")
        return EvaluationDatasetEnvelope(
            message="dataset loaded",
            data=EvaluationDatasetData(
                name=name,
                description=description,
                items=load_dataset(),
            ),
        )

    @app.post(
        "/api/NebulonMind/evaluation/run",
        tags=["Evaluation"],
        summary="Run the agent evaluation",
        description=(
            "Runs the agent over the bundled benchmark dataset (or an "
            "inline ``items`` list) and reports metrics: tool selection "
            "accuracy, retrieval accuracy, answer correctness, hallucination "
            "rate, latency and token usage. Requires a configured LLM provider "
            "(NMD_LLM_PROVIDER + key/model). The report is also saved under "
            "evaluation/reports/."
        ),
        response_model=EvaluationRunEnvelope,
        responses={503: {"description": "LLM provider not configured"}},
    )
    def evaluation_run(
        request: EvaluationRunRequest = Body(..., description="Evaluation input"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> EvaluationRunEnvelope:
        from ..evaluation.runner import (
            EvaluationRunner,
            load_dataset,
            save_report,
        )
        from ..intelligence.providers import LLMProviderError, provider_from_env

        try:
            llm = provider_from_env()
        except Exception as exc:
            logger.warning("evaluation run unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "evaluation requires an LLM provider (set NMD_LLM_PROVIDER "
                    "and the matching key/model)"
                ),
            ) from exc
        bundle = _bundle(user_id)
        runner = EvaluationRunner(
            llm, bundle.repository, bundle.manager, user_id=bundle.user_id,
            top_k=request.top_k,
        )
        items = request.items if request.items is not None else load_dataset()
        report = runner.run(items[: request.max_items], seed=request.seed)
        try:
            save_report(report, dataset_name=request.dataset)
        except Exception as exc:  # defensive: reports are best-effort
            logger.warning("evaluation report save failed: %s", exc)
        return EvaluationRunEnvelope(
            message="evaluation complete",
            data=EvaluationData(
                dataset=request.dataset,
                metrics=EvaluationMetrics(**report["metrics"]),
                results=[
                    EvaluationItemData(**item) for item in report["results"]
                ],
            ),
        )

    # ------------------------------------------------------------------ #
    # Background Agents                                                  #
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/NebulonMind/background/memory/run",
        tags=["Background"],
        summary="Run the Memory Agent once (manual trigger)",
        description=(
            "Runs the existing MemoryConsolidator over the user's memories "
            "and reports KEEP / UPDATE / MERGE / IGNORE decisions. In "
            "``consolidate`` mode it also persists one merged memory per "
            "MERGE decision. The agent never deletes memories."
        ),
        response_model=BackgroundAgentEnvelope,
    )
    def background_memory_run(
        request: BackgroundMemoryRunRequest = Body(...,
            description="Memory Agent run options"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> BackgroundAgentEnvelope:
        from ..agents import MemoryAgent

        bundle = _bundle(user_id)
        report = MemoryAgent(
            bundle.repository,
            bundle.manager,
            user_id=bundle.user_id,
            mode=request.mode,
        ).run()
        return BackgroundAgentEnvelope(
            message="memory agent ran",
            data=BackgroundAgentReport(
                agent="memory",
                user_id=bundle.user_id,
                run_at=report["run_at"],
                data=report,
            ),
        )

    @app.post(
        "/api/NebulonMind/background/task/run",
        tags=["Background"],
        summary="Run the Task Agent once (manual trigger)",
        description=(
            "Generates a weekly summary of the user's recent memories, "
            "grouped by category, and persists it as a memory through the "
            "ingest gate (so it is retrievable like any other memory)."
        ),
        response_model=BackgroundAgentEnvelope,
    )
    def background_task_run(
        request: BackgroundTaskRunRequest = Body(...,
            description="Task Agent run options"),
        user_id: str = Query(DEFAULT_USERNAME, description="Registered NebulonMind username"),
    ) -> BackgroundAgentEnvelope:
        from ..agents import TaskAgent

        bundle = _bundle(user_id)
        report = TaskAgent(
            bundle.repository,
            bundle.manager,
            user_id=bundle.user_id,
            days=request.days,
        ).run()
        return BackgroundAgentEnvelope(
            message="task agent ran",
            data=BackgroundAgentReport(
                agent="task",
                user_id=bundle.user_id,
                run_at=report["run_at"],
                data=report,
            ),
        )

    @app.get(
        "/api/NebulonMind/background/status",
        tags=["Background"],
        summary="Background scheduler status",
        description=(
            "Lists the registered background jobs (schedule, last run, "
            "runs, last error). State is in-memory and lost on restart; "
            "no persistent store is introduced."
        ),
        response_model=BackgroundStatusEnvelope,
    )
    def background_status() -> BackgroundStatusEnvelope:
        snapshot = scheduler.snapshot()
        return BackgroundStatusEnvelope(
            message="background scheduler status",
            data=BackgroundSchedulerStatus(
                running=snapshot["running"],
                poll_seconds=snapshot["poll_seconds"],
                jobs=[BackgroundJobStatus(**job) for job in snapshot["jobs"]],
            ),
        )

    # ------------------------------------------------------------------ #
    # LLM status                                                         #
    # ------------------------------------------------------------------ #

    @app.get(
        "/api/NebulonMind/llm/status",
        tags=["Agent"],
        summary="LLM provider status",
        description=(
            "Reports which LLM provider is configured (NMD_LLM_PROVIDER), "
            "whether it constructed successfully, and the active model. "
            "Never returns keys or other credentials."
        ),
        response_model=LLMStatusEnvelope,
    )
    def llm_status() -> LLMStatusEnvelope:
        import os

        from ..intelligence.providers import provider_from_env

        name = os.environ.get("NMD_LLM_PROVIDER", "").strip()
        provider = None
        error = None
        if name:
            try:
                provider = provider_from_env()
            except Exception as exc:
                error = _redact_llm_error(str(exc))
        return LLMStatusEnvelope(
            message="ok",
            data=LLMStatusData(
                provider=name,
                configured=provider is not None,
                model=getattr(provider, "model", None) if provider else None,
                error=error,
            ),
        )

    # ------------------------------------------------------------------ #
    # Web console + system management (dashboard)                         #
    # ------------------------------------------------------------------ #

    app.include_router(
        dashboard_router,
        prefix="/api/NebulonMind/dashboard",
        tags=["Dashboard"],
    )
    app.include_router(
        config_router,
        prefix="/api/NebulonMind/config",
        tags=["Config"],
    )
    app.include_router(
        chats_router,
        prefix="/api/NebulonMind",
        tags=["Chats"],
    )
    # Serve CSS/JS (and any other) console assets from web_dir under the
    # same prefix; the router routes registered above win on conflict.
    app.mount(
        "/api/NebulonMind/dashboard",
        StaticFiles(directory=WEB_DIR, html=True),
        name="web_console",
    )

    return app
