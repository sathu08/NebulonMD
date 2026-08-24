"""Per-user chat transcript persistence on the local filesystem.

Transcripts are stored as JSON under ``<NMD_HOME>/chat_history/<username>/<id>.json``.
Keying by username means every registered user only ever sees their own chats,
from any browser/client. Filenames are sanitised so usernames can never
escape the store directory (no path traversal).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..core.config import _nmd_home

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe(part: str, default: str = "unknown") -> str:
    """Sanitise a path segment (username or chat id) to ``[A-Za-z0-9_.-]``."""
    cleaned = _SAFE_RE.sub("", (part or "")).strip(".")
    if not cleaned or cleaned in {"", ".", ".."}:
        return default
    return cleaned


def _root() -> Path:
    return _nmd_home() / "chat_history"


def _user_dir(username: str) -> Path:
    return _root() / _safe(username)


def _chat_path(username: str, chat_id: str) -> Path:
    return _user_dir(username) / f"{_safe(chat_id, 'chat')}.json"


def list_chats(username: str) -> list[dict[str, Any]]:
    """Return all chats for a username, newest first."""
    user_dir = _user_dir(username)
    if not user_dir.is_dir():
        return []
    chats = []
    for path in sorted(user_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("id"):
            chats.append(record)
    chats.sort(key=lambda c: c.get("updatedAt", c.get("createdAt", 0)), reverse=True)
    return chats


def get_chat(username: str, chat_id: str) -> dict[str, Any] | None:
    """Return a single chat record (or ``None`` if missing/invalid)."""
    path = _chat_path(username, chat_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def save_chat(username: str, chat: dict[str, Any]) -> dict[str, Any]:
    """Persist/overwrite a chat record. Returns the stored record."""
    chat_id = _safe(str(chat.get("id", "")), default="chat")
    messages = chat.get("messages")
    if not isinstance(messages, list):
        messages = []
    title = str(chat.get("title") or "").strip() or "Untitled chat"
    now = int(time.time() * 1000)
    existing = get_chat(username, chat_id)
    record = {
        "id": chat_id,
        "user": str(chat.get("user") or username),
        "title": title[:120],
        "createdAt": int(chat.get("createdAt") or (existing or {}).get("createdAt") or now),
        "updatedAt": now,
        "messages": [
            {
                "role": ("assistant" if str(m.get("role")) == "assistant" else "user"),
                "content": str(m.get("content") or ""),
            }
            for m in messages
            if isinstance(m, dict) and (m.get("role") in ("user", "assistant"))
        ],
    }
    user_dir = _user_dir(username)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = _chat_path(username, chat_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)
    return record


def delete_chat(username: str, chat_id: str) -> bool:
    """Delete a chat record. Returns ``True`` if a file was removed."""
    path = _chat_path(username, chat_id)
    if path.is_file():
        path.unlink()
        return True
    return False