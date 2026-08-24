"""TruthStore — COSMOS corpus ``mind_truth`` via the NebulonDB API.

One document per memory; the segment is ``user_{user_id}`` (multi-tenant
isolation). No embedding is stored here.

The COSMOS engine only persists ``{text, lang, type, created_at}`` fields
and assigns its own integer ``_id`` per record, so the full truth document
is JSON-encoded into the ``text`` column and lookups scan the segment,
parsing each record and matching on ``memory_id``. The API has no upsert,
so ``create`` deletes any existing record for the memory first, keeping
re-stores idempotent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from nmd_host.api import NebulonDBClient


def _parse_doc(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(record["text"])
    except (KeyError, TypeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


class TruthStore:
    CORPUS = "mind_truth"

    def __init__(self, client: NebulonDBClient, user_id: str) -> None:
        self._api = client
        self._segment = f"user_{user_id}"

    @property
    def segment(self) -> str:
        return self._segment

    def create(self, doc: Dict[str, Any]) -> None:
        memory_id = doc.get("memory_id")
        if not memory_id:
            raise ValueError("Truth doc must contain 'memory_id'")
        self.delete(memory_id)
        self._api.load_segment(
            self.CORPUS,
            self._segment,
            "cosmos",
            records=[{"text": json.dumps(doc)}],
            set_columns=["text"],
        )

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        for doc in self.read_all():
            if doc.get("memory_id") == memory_id:
                return doc
        return None

    def update(self, doc: Dict[str, Any]) -> None:
        if not doc.get("memory_id"):
            raise ValueError("Truth doc must contain 'memory_id' for an update")
        self.create(doc)

    def delete(self, memory_id: str) -> bool:
        for record in self._api.get_data(self.CORPUS, self._segment, "cosmos"):
            doc = _parse_doc(record)
            if doc and doc.get("memory_id") == memory_id:
                return self._api.delete_record(
                    self.CORPUS, self._segment, "cosmos", record["_id"]
                )
        return False

    def read_all(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for record in self._api.get_data(self.CORPUS, self._segment, "cosmos"):
            doc = _parse_doc(record)
            if doc:
                docs.append(doc)
        return docs

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
