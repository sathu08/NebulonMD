"""VectorStore — ORBIT corpus ``mind_semantic`` (segment ``user_{user_id}``) via API.

One embedding per memory. Embeddings are computed server-side
(``is_precomputed=False``); the record's ``name`` column becomes the
``metadata["label"]`` which carries the ``memory_id``. Reads are normalised
to ``{id, score, text, metadata: {memory_id, text}}``. The ORBIT engine only
persists ``{lang, type, created_at, label, text}`` metadata, so richer
metadata passed by callers is intentionally dropped server-side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nmd_host.api import NebulonDBClient


class VectorStore:
    CORPUS = "mind_semantic"

    def __init__(self, client: NebulonDBClient, user_id: str) -> None:
        self._api = client
        self._segment = f"user_{user_id}"

    @property
    def segment(self) -> str:
        return self._segment

    def insert(self, memory_id: str, text: str) -> int:
        """Insert a new embedding (computed server-side); returns the record id."""
        self._api.load_segment(
            self.CORPUS,
            self._segment,
            "orbit",
            records=[{"name": memory_id, "text": text}],
            set_columns=["text"],
            is_precomputed=False,
        )
        record_id = self.record_id_of(memory_id)
        if record_id is None:
            raise RuntimeError(f"vector insert failed for {memory_id}")
        return record_id

    def update(self, memory_id: str, text: str) -> None:
        """Replace the embedding/metadata of an existing memory (idempotent)."""
        record_id = self.record_id_of(memory_id)
        if record_id is None:
            self.insert(memory_id, text)
            return
        self._api.delete_record(self.CORPUS, self._segment, "orbit", record_id)
        self.insert(memory_id, text)

    def delete(self, memory_id: str) -> bool:
        record_id = self.record_id_of(memory_id)
        if record_id is None:
            return False
        return self._api.delete_record(self.CORPUS, self._segment, "orbit", record_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Nearest neighbours; each hit is {id, score, text, metadata}."""
        hits = self._api.search_segment(
            self.CORPUS,
            self._segment,
            "orbit",
            search_item=query,
            top_matches=top_k,
            mode="auto",
            rank=False,
        )
        normalized: List[Dict[str, Any]] = []
        for hit in hits:
            metadata = hit.get("metadata") or {}
            label = metadata.get("label") or hit.get("label")
            normalized.append(
                {
                    "id": hit.get("id"),
                    "score": hit.get("score"),
                    "text": hit.get("text"),
                    "metadata": {
                        "memory_id": label,
                        "text": metadata.get("text", hit.get("text")),
                    },
                }
            )
        return normalized

    def record_id_of(self, memory_id: str) -> Optional[int]:
        """Orbit record id for a memory, or None if it has no embedding."""
        for record in self._api.get_data(self.CORPUS, self._segment, "orbit"):
            if (record.get("label") == memory_id) or (
                (record.get("metadata") or {}).get("label") == memory_id
            ):
                return record.get("id")
        return None

    def count(self) -> int:
        return int(self._api.segment_stats(self.CORPUS, self._segment, "orbit").get("vector_count", 0))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
