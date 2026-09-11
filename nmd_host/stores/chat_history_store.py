"""ChatHistoryStore — COSMOS corpus ``mind_chats`` via the NebulonDB API.

One document per chat transcript; the segment is ``user_{user_id}``
(multi-tenant isolation). The full chat record
``{id, user, title, createdAt, updatedAt, messages}`` is JSON-encoded into
the ``text`` column with ``doc_type="chat_history"``.

History is COSMOS-only by design: transcripts are exact-fetched for the
console (``/chats``), never embedded, so raw chit-chat can never pollute
semantic recall or the LLM context pipeline.

``InMemoryChatHistoryStore`` mirrors the same interface over a dict for
offline tests / the in-memory provider (no NebulonDB required).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from nmd_host.api import NebulonDBClient


def _normalize_chat(username: str, chat: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate + normalize a chat record (same contract as the legacy file store)."""
    chat_id = str(chat.get("id", "") or "chat")
    messages = chat.get("messages")
    if not isinstance(messages, list):
        messages = []
    title = str(chat.get("title") or "").strip() or "Untitled chat"
    now = int(time.time() * 1000)
    return {
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


def _parse_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        chat = json.loads(record["text"])
    except (KeyError, TypeError, ValueError):
        return None
    return chat if isinstance(chat, dict) and chat.get("id") else None


class ChatHistoryStore:
    CORPUS = "mind_chats"
    DOC_TYPE = "chat_history"

    def __init__(self, client: NebulonDBClient, user_id: str, username: str = "") -> None:
        self._api = client
        self._user_id = user_id
        self._username = username or user_id
        self._segment = f"user_{user_id}"

    @property
    def segment(self) -> str:
        return self._segment

    def list(self) -> List[Dict[str, Any]]:
        chats = [c for c in (self._parse_all()) if c]
        chats.sort(key=lambda c: c.get("updatedAt", c.get("createdAt", 0)), reverse=True)
        return chats

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        for chat in self._parse_all():
            if chat.get("id") == chat_id:
                return chat
        return None

    def save(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(chat, dict) or not chat.get("id"):
            raise ValueError("chat payload must include {'id', 'title', 'messages'}")
        record = _normalize_chat(self._username, chat, self.get(str(chat.get("id"))))
        self.delete(record["id"])
        self._api.load_segment(
            self.CORPUS,
            self._segment,
            "cosmos",
            records=[{"text": json.dumps(record)}],
            set_columns=["text"],
            lang_type="en",
            doc_type=self.DOC_TYPE,
            metadata={"app": "nebulonmind", "message_count": len(record["messages"])},
        )
        return record

    def delete(self, chat_id: str) -> bool:
        for record in self._api.get_data(self.CORPUS, self._segment, "cosmos"):
            chat = _parse_record(record)
            if chat and chat.get("id") == chat_id:
                return self._api.delete_record(
                    self.CORPUS, self._segment, "cosmos", record["_id"]
                )
        return False

    def _parse_all(self) -> List[Dict[str, Any]]:
        chats: List[Dict[str, Any]] = []
        for record in self._api.get_data(self.CORPUS, self._segment, "cosmos"):
            chat = _parse_record(record)
            if chat:
                chats.append(chat)
        return chats


class InMemoryChatHistoryStore:
    """Dict-backed chat history for offline tests (same interface)."""

    def __init__(self, username: str = "") -> None:
        self._username = username
        self._chats: Dict[str, Dict[str, Any]] = {}

    def list(self) -> List[Dict[str, Any]]:
        chats = list(self._chats.values())
        chats.sort(key=lambda c: c.get("updatedAt", c.get("createdAt", 0)), reverse=True)
        return chats

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return self._chats.get(chat_id)

    def save(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(chat, dict) or not chat.get("id"):
            raise ValueError("chat payload must include {'id', 'title', 'messages'}")
        record = _normalize_chat(self._username, chat, self._chats.get(str(chat.get("id"))))
        self._chats[record["id"]] = record
        return record

    def delete(self, chat_id: str) -> bool:
        return self._chats.pop(chat_id, None) is not None


def history_store_for(provider: Any, username: str):
    """Chat history store for a username under any provider.

    Registered usernames resolve to their opaque ``unique_id`` partition;
    anything else falls back to the name itself (lenient per-name history,
    never an error). DB-backed when the provider has a backend client,
    in-memory otherwise (stores cached on the provider instance).
    """
    try:
        partition = provider.registry().resolve(username)
    except Exception:
        partition = username
    client = getattr(provider, "client", None)
    if client is not None:
        return ChatHistoryStore(client, partition, username)
    stores = provider.__dict__.setdefault("_chat_history_stores", {})
    if partition not in stores:
        stores[partition] = InMemoryChatHistoryStore(username)
    return stores[partition]


def transcript_to_chat(
    chat_id: str,
    username: str,
    messages: List[Any],
    title: str = "",
) -> Dict[str, Any]:
    """Build a ``/chats``-compatible record from agent transcript messages.

    Only ``user``/``assistant`` turns are kept (``tool`` spans are internal
    runtime detail); the title comes from the first user turn.
    """
    kept = [
        {"role": m.role, "content": m.content}
        for m in messages
        if getattr(m, "role", None) in ("user", "assistant")
    ]
    if not title:
        first_user = next((m["content"] for m in kept if m["role"] == "user"), "")
        title = (first_user[:40] or "Untitled chat")
    return {"id": chat_id, "user": username, "title": title, "messages": kept}
