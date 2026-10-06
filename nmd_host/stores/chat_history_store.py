"""ChatHistoryStore — COSMOS corpus ``mind_chats`` via the NebulonDB API.

One **record per message**: every user/assistant message is its own JSON
document ``{kind: "message", mid, chat_id, role, content, idx}`` plus one
header document per chat ``{kind: "chat", id, user, title, createdAt,
updatedAt, message_count}``. The segment is ``user_{user_id}``
(multi-tenant isolation).

Combining rule (exact duplicates only): the message key is
``sha256(chat_id + role + content)`` — identical texts share one record,
first occurrence wins, order preserved. Re-saving a transcript reconciles
(diff): only genuinely new messages are inserted, removed ones deleted,
unchanged ones untouched — so 11 auto-saves of a growing chat cost ~11
message writes, not 11 full-transcript rewrites.

Legacy full-transcript docs (``{id, ..., messages[]}`` with no ``kind``)
are read-compatible and superseded (deleted) on the next save of that
chat.

History is COSMOS-only by design: transcripts are exact-fetched for the
console (``/chats``), never embedded, so raw chit-chat can never pollute
semantic recall or the LLM context pipeline.

``InMemoryChatHistoryStore`` mirrors the same interface over dicts for
offline tests / the in-memory provider (no NebulonDB required).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

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


def _message_key(chat_id: str, role: str, content: str) -> str:
    """Stable id for one message: identical texts share one record."""
    return hashlib.sha256(
        f"{chat_id}\x00{role}\x00{content}".encode("utf-8")
    ).hexdigest()


def _dedupe_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop exact duplicates (same role + same text), first occurrence wins."""
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for m in messages:
        key = (m.get("role", "user"), m.get("content", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"role": key[0], "content": key[1]})
    return unique


def _parse_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(record["text"])
    except (KeyError, TypeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _split_records(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Partition raw records into chat headers, message records, legacy docs.

    Returns ``(headers, messages, legacy)`` where headers maps chat_id →
    header doc, messages maps chat_id → mid → message doc, and legacy maps
    chat_id → [raw records] of old full-transcript docs.
    """
    headers: Dict[str, Dict[str, Any]] = {}
    messages: Dict[str, Dict[str, Dict[str, Any]]] = {}
    legacy: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        doc = _parse_record(record)
        if not doc:
            continue
        kind = doc.get("kind")
        if kind == "message" and doc.get("chat_id") and doc.get("mid"):
            messages.setdefault(str(doc["chat_id"]), {})[str(doc["mid"])] = {
                **doc, "_rid": record.get("_id", record.get("id")),
            }
        elif kind == "chat" and doc.get("id"):
            headers[str(doc["id"])] = {**doc, "_rid": record.get("_id", record.get("id"))}
        elif isinstance(doc.get("messages"), list) and doc.get("id"):
            legacy.setdefault(str(doc["id"]), []).append(record)
    return headers, messages, legacy


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

    def _load_records(self) -> List[Dict[str, Any]]:
        try:
            return self._api.get_data(self.CORPUS, self._segment, "cosmos")
        except Exception:
            return []

    def _delete_rid(self, rid: Any) -> bool:
        if rid is None:
            return False
        try:
            return self._api.delete_record(self.CORPUS, self._segment, "cosmos", rid)
        except Exception:
            return False

    def _assemble(
        self,
        headers: Dict[str, Dict[str, Any]],
        messages: Dict[str, Dict[str, Dict[str, Any]]],
        legacy: Dict[str, List[Dict[str, Any]]],
        chat_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rebuild chat dicts (header + ordered messages) from stored parts."""
        ids = [chat_id] if chat_id else list(
            set(headers) | set(messages) | set(legacy)
        )
        chats: List[Dict[str, Any]] = []
        for cid in ids:
            header = headers.get(cid)
            if header is None:
                # Legacy full-transcript doc stands in as the header.
                legacy_docs = [
                    _parse_record(r) for r in legacy.get(cid, [])
                ]
                legacy_docs = [d for d in legacy_docs if d]
                if not legacy_docs:
                    continue
                header = max(
                    legacy_docs,
                    key=lambda d: int(d.get("updatedAt") or 0),
                )
            stored = messages.get(cid, {})
            if stored:
                ordered = sorted(
                    stored.values(), key=lambda m: int(m.get("idx", 0))
                )
                msgs = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in ordered
                ]
            else:
                msgs = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in header.get("messages", [])
                    if isinstance(m, dict)
                ]
            chats.append(
                {
                    "id": cid,
                    "user": str(header.get("user") or self._username),
                    "title": str(header.get("title") or "Untitled chat"),
                    "createdAt": int(header.get("createdAt") or 0),
                    "updatedAt": int(header.get("updatedAt") or 0),
                    "messages": msgs,
                }
            )
        chats.sort(key=lambda c: c.get("updatedAt", 0), reverse=True)
        return chats

    def list(self) -> List[Dict[str, Any]]:
        records = self._load_records()
        headers, messages, legacy = _split_records(records)
        return self._assemble(headers, messages, legacy)

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        records = self._load_records()
        headers, messages, legacy = _split_records(records)
        chats = self._assemble(headers, messages, legacy, str(chat_id))
        return chats[0] if chats else None

    def save(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(chat, dict) or not chat.get("id"):
            raise ValueError("chat payload must include {'id', 'title', 'messages'}")
        try:
            ensure = getattr(self._api, "ensure_corpus", None)
            if callable(ensure):
                ensure(self.CORPUS, "cosmos")
        except Exception:
            pass
        incoming = _normalize_chat(self._username, chat, self.get(str(chat.get("id"))))
        cid = incoming["id"]
        deduped = _dedupe_messages(incoming["messages"])
        wanted = {
            _message_key(cid, m["role"], m["content"]): m for m in deduped
        }

        records = self._load_records()
        headers, messages, legacy = _split_records(records)
        stored = messages.get(cid, {})
        old_header = headers.get(cid)

        # No-op when nothing changed (same title + same message set, no
        # legacy docs left to migrate): zero writes, zero tombstones.
        if (
            not legacy.get(cid)
            and old_header is not None
            and str(old_header.get("title") or "") == incoming["title"]
            and set(stored) == set(wanted)
        ):
            return self._assemble(headers, messages, legacy, cid)[0]

        # Reconcile: delete superseded legacy docs, removed messages and
        # the stale header; insert only genuinely new messages + header.
        for record in legacy.get(cid, []):
            self._delete_rid(record.get("_id", record.get("id")))
        for mid, doc in stored.items():
            if mid not in wanted:
                self._delete_rid(doc.get("_rid"))
        if old_header is not None:
            self._delete_rid(old_header.get("_rid"))

        new_docs: List[Dict[str, Any]] = []
        for idx, (mid, m) in enumerate(wanted.items()):
            if mid in stored:
                continue
            new_docs.append(
                {
                    "kind": "message",
                    "mid": mid,
                    "chat_id": cid,
                    "role": m["role"],
                    "content": m["content"],
                    "idx": idx,
                }
            )
        header_doc = {
            "kind": "chat",
            "id": cid,
            "user": incoming["user"],
            "title": incoming["title"],
            "createdAt": incoming["createdAt"],
            "updatedAt": incoming["updatedAt"],
            "message_count": len(deduped),
        }
        payload = [
            {"text": json.dumps(header_doc)},
            *({"text": json.dumps(d)} for d in new_docs),
        ]
        self._api.load_segment(
            self.CORPUS,
            self._segment,
            "cosmos",
            records=payload,
            set_columns=["text"],
            lang_type="en",
            doc_type=self.DOC_TYPE,
            metadata={"app": "nebulonmind", "message_count": len(deduped)},
        )
        return {**header_doc, "messages": deduped}

    def delete(self, chat_id: str) -> bool:
        cid = str(chat_id)
        records = self._load_records()
        headers, messages, legacy = _split_records(records)
        removed = False
        for record in legacy.get(cid, []):
            removed = self._delete_rid(record.get("_id", record.get("id"))) or removed
        for doc in messages.get(cid, {}).values():
            removed = self._delete_rid(doc.get("_rid")) or removed
        if cid in headers:
            removed = self._delete_rid(headers[cid].get("_rid")) or removed
        return removed


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
        record["messages"] = _dedupe_messages(record["messages"])
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
