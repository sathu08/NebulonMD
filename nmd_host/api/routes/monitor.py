"""nmd_monitor API routes — traces / trace detail / stats (P0).

Mounted in ``server.py`` under ``/api/NebulonMind/monitor``. Same
conventions as ``routes/chats.py``: ``user_id``-scoped, ``{success,
message, data}`` envelope, invalid user -> 400, unknown user -> 403 via
the provider bundle (never auto-registered).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ...core.models import FeedbackRequest
from ...monitor.store import monitor_store_for, MonitorFeedback

logger = logging.getLogger("nmd_host.api.routes.monitor")

router = APIRouter()


def _user_id(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id or user_id.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid user_id")
    return user_id


def _store(request: Request, username: str):
    return monitor_store_for(request.app.state.provider, username)


def _parse_ok(ok: Optional[str]) -> Optional[bool]:
    if ok is None or ok == "":
        return None
    lowered = ok.strip().lower()
    if lowered in ("1", "true", "yes"):
        return True
    if lowered in ("0", "false", "no"):
        return False
    raise HTTPException(status_code=400, detail="ok must be true/false")


@router.get(
    "/traces",
    tags=["Monitor"],
    summary="List a user's agent traces (newest first)",
)
async def list_traces(
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
    q: str = Query("", description="Substring filter over input/answer"),
    tool: str = Query("", description="Only traces that used this tool"),
    ok: Optional[str] = Query(None, description="true/false outcome filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    user = _user_id(user_id)
    try:
        traces = _store(request, user).list(
            q=q, tool=tool.strip(), ok=_parse_ok(ok), limit=limit, offset=offset
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("monitor traces read failed for %s: %s", user, exc)
        raise HTTPException(status_code=503, detail="monitor store unavailable") from exc
    return {
        "success": True,
        "message": f"{len(traces)} traces for {user!r}",
        "data": {"user_id": user, "traces": [t.model_dump() for t in traces]},
    }


@router.get(
    "/trace/{trace_id}",
    tags=["Monitor"],
    summary="Fetch a single agent trace with spans",
)
async def get_trace(
    trace_id: str,
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
) -> dict:
    user = _user_id(user_id)
    trace = _store(request, user).get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    return {
        "success": True,
        "message": "trace loaded",
        "data": {"user_id": user, "trace": trace.model_dump()},
    }


@router.get(
    "/stats",
    tags=["Monitor"],
    summary="Trace aggregates (counts, latency, tool usage)",
)
async def get_stats(
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
    days: int = Query(7, ge=1, le=365),
) -> dict:
    user = _user_id(user_id)
    since_ms = int((time.time() - days * 86400) * 1000)
    stats = _store(request, user).stats(since_ms=since_ms)
    return {
        "success": True,
        "message": f"monitor stats for {user!r} (last {days}d)",
        "data": {"user_id": user, "days": days, "stats": stats.model_dump()},
    }


@router.post(
    "/feedback",
    tags=["Monitor"],
    summary="Submit feedback for a trace",
)
async def post_feedback(
    request: Request,
    feedback_data: FeedbackRequest,
    user_id: str = Query(..., description="Registered NebulonMind username"),
) -> dict:
    user = _user_id(user_id)
    feedback = MonitorFeedback(
        user_id=user,
        trace_id=feedback_data.trace_id,
        score=feedback_data.score,
        tag=feedback_data.tag,
        comment=feedback_data.comment,
        by=user,
    )
    try:
        success = _store(request, user).save_feedback(feedback)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("monitor feedback save failed for %s: %s", user, exc)
        raise HTTPException(status_code=503, detail="monitor store unavailable") from exc
    if not success:
        raise HTTPException(status_code=500, detail="feedback save failed")
    return {
        "success": True,
        "message": f"feedback saved for trace {feedback_data.trace_id!r}",
        "data": {"user_id": user, "feedback": feedback.model_dump()},
    }


@router.get(
    "/feedback",
    tags=["Monitor"],
    summary="Get feedback entries for a user",
)
async def get_feedback(
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
    trace_id: str = Query("", description="Filter by specific trace ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    user = _user_id(user_id)
    try:
        feedback = _store(request, user).get_feedback(
            trace_id=trace_id, limit=limit, offset=offset
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("monitor feedback read failed for %s: %s", user, exc)
        raise HTTPException(status_code=503, detail="monitor store unavailable") from exc
    return {
        "success": True,
        "message": f"{len(feedback)} feedback entries for {user!r}",
        "data": {"user_id": user, "feedback": [f.model_dump() for f in feedback]},
    }
