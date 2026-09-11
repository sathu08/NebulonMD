"""NebulonMind per-user chat transcript API.

Mirrors the console's client-side chat-history concept on the server:
``/chats`` lists, saves, retrieves and deletes chat transcripts keyed by the
``user_id`` (username) query parameter, so each user sees exactly their own
history from any browser.

Transcripts persist in the ``mind_chats`` COSMOS corpus (one
``user_{unique_id}`` segment per registered user) via ``ChatHistoryStore`` —
no local files. Unknown usernames resolve to their own name-keyed partition
(lenient, as before) instead of an error.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from ...stores.chat_history_store import history_store_for

logger = logging.getLogger("nmd_host.api.routes.chats")

router = APIRouter()


def _user_id(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id or user_id.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid user_id")
    return user_id


def _history_store(request: Request, username: str):
    """Chat history store for a username (DB-backed or in-memory fallback)."""
    return history_store_for(request.app.state.provider, username)


@router.get(
    "/chats",
    tags=["Chats"],
    summary="List a user's saved chat transcripts (newest first)",
)
async def get_chats(
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
) -> dict:
    user = _user_id(user_id)
    store = _history_store(request, user)
    chats = store.list()
    return {
        "success": True,
        "message": f"{len(chats)} chats for {user!r}",
        "data": {"user_id": user, "chats": chats},
    }


@router.get(
    "/chats/{chat_id}",
    tags=["Chats"],
    summary="Fetch a single chat transcript",
)
async def get_chat(
    chat_id: str,
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
) -> dict:
    user = _user_id(user_id)
    chat = _history_store(request, user).get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"chat {chat_id!r} not found")
    return {
        "success": True,
        "message": "chat loaded",
        "data": {"user_id": user, "chat": chat},
    }


@router.post(
    "/chats",
    status_code=201,
    tags=["Chats"],
    summary="Save/overwrite a chat transcript",
)
async def save_chat(request: Request, payload: dict) -> dict:
    chat = payload.get("chat")
    if not isinstance(chat, dict) or not chat.get("id"):
        raise HTTPException(
            status_code=400, detail="payload must include {'chat': {id, title, messages}}"
        )
    user = _user_id(payload.get("user_id", ""))
    try:
        record = _history_store(request, user).save(chat)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("chat %s saved for %s (%d messages)", record["id"], user,
                len(record["messages"]))
    return {
        "success": True,
        "message": "chat saved",
        "data": {"user_id": user, "chat": record},
    }


@router.delete(
    "/chats/{chat_id}",
    tags=["Chats"],
    summary="Delete a chat transcript",
)
async def delete_chat(
    chat_id: str,
    request: Request,
    user_id: str = Query(..., description="Registered NebulonMind username"),
) -> dict:
    user = _user_id(user_id)
    deleted = _history_store(request, user).delete(chat_id)
    return {
        "success": True,
        "message": "chat deleted" if deleted else "chat not found",
        "data": {"user_id": user, "chat_id": chat_id, "deleted": deleted},
    }
