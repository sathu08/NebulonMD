"""NebulonMind facade.

``store(memory)`` orchestrates a single write across truth (COSMOS),
meaning (ORBIT vectors) and relationships (ORBIT Mesh) through the
NebulonDB REST API, compensating partial writes on failure.
``recall(query)`` searches the server-side vector index and hydrates
from truth.
"""

from __future__ import annotations

import uuid
import logging

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import Memory
from .config import NebulonDBConfig

from nmd_host.stores import GraphStore, TruthStore, VectorStore

logger = logging.getLogger("nmd_host.core.mind")


class NebulonMind:
    """Memory layer for one user, backed by a NebulonDB server.

    ``storage_root``/``embedder``/``rank_config``/``reset`` are accepted for
    backward compatibility but ignored: embeddings are computed server-side
    and data lives in NebulonDB, not on the local filesystem.
    """

    def __init__(
        self,
        storage_root=None,
        user_id: str = "user_001",
        embedder=None,
        rank_config=None,
        reset: bool = False,
        client: Optional["NebulonDBClient"] = None,
    ) -> None:
        self.user_id = user_id
        if client is None:
            from nmd_host.api import NebulonDBClient

            client = NebulonDBClient(NebulonDBConfig.from_env())
        self._api = client

        self.truth = TruthStore(self._api, user_id)
        self.vector = VectorStore(self._api, user_id)
        self.graph = GraphStore(self._api, user_id, truth=self.truth)

    # ------------------------------------------------------------------ #
    # Write path                                                         #
    # ------------------------------------------------------------------ #

    def store(self, memory: Memory) -> Memory:
        if memory.user_id != self.user_id:
            raise ValueError(
                f"memory.user_id {memory.user_id!r} != mind user {self.user_id!r}"
            )
        if memory.memory_id is None:
            memory.memory_id = self._new_memory_id()
        memory.stamp_provenance()
        now = datetime.now(timezone.utc)
        if memory.lifecycle.created_at is None:
            memory.lifecycle.created_at = now
        memory.lifecycle.updated_at = now

        doc = memory.to_truth_doc()
        self.truth.create(doc)
        try:
            self.vector.update(
                memory.memory_id,
                memory.content.text,
                {
                    "app": "nebulonmind",
                    "memory_id": memory.memory_id,
                    "category": memory.classification.category,
                },
                memory.classification.lang or "en",
            )
            self.graph.relate_memory(memory)
        except Exception as exc:
            self._compensate(memory.memory_id)
            logger.error(
                "Memory store failed: memory_id=%s operation=%s error_type=%s",
                memory.memory_id,
                "store",
                type(exc).__name__,
            )
            raise
        return memory

    def update(self, memory_id: str, data: Dict[str, Any]) -> Memory:
        existing = self.get(memory_id)
        if existing is None:
            raise KeyError(memory_id)
        updated = Memory.model_validate({**existing.model_dump(), **data})
        updated.memory_id = memory_id
        updated.user_id = self.user_id
        updated.lifecycle.updated_at = datetime.now(timezone.utc)
        updated.stamp_provenance()

        doc = updated.to_truth_doc()
        self.truth.create(doc)
        try:
            self.vector.update(
                memory_id,
                updated.content.text,
                {
                    "app": "nebulonmind",
                    "memory_id": memory_id,
                    "category": updated.classification.category,
                },
                updated.classification.lang or "en",
            )
            self.graph.relate_memory(updated)
        except Exception as exc:
            self._compensate(memory_id)
            logger.error(
                "Memory update failed: memory_id=%s operation=%s error_type=%s",
                memory_id,
                "update",
                type(exc).__name__,
            )
            raise
        return updated

    def delete(self, memory_id: str) -> bool:
        # Delete order: graph -> vector -> truth (inverse of the write order).
        self.graph.delete_memory(memory_id)
        self.vector.delete(memory_id)
        return self.truth.delete(memory_id)

    def relate(self, memory_id: str, entity: str, relation: str = GraphStore.HAS_ENTITY) -> None:
        """Link an existing memory to an entity label in the graph."""
        if self.truth.get(memory_id) is None:
            raise KeyError(memory_id)
        self.graph.link(memory_id, entity, relation)

    # ------------------------------------------------------------------ #
    # Read path                                                          #
    # ------------------------------------------------------------------ #

    def get(self, memory_id: str) -> Optional[Memory]:
        doc = self.truth.get(memory_id)
        return Memory.from_truth_doc(doc) if doc else None

    def list_all(self) -> List[Memory]:
        """All of this mind's memories, hydrated from the truth store."""
        return [Memory.from_truth_doc(doc) for doc in self.truth.read_all()]

    def recall(self, query: str, top_k: int = 5, expand: bool = False) -> List[Memory]:
        """Semantic candidate projection for Step 3 retrieval.

        Server-side embedding -> top-k memory_ids (k × candidate
        multiplier) -> hydrated Memory objects. This is only the candidate
        provider: the Step 3 ``MemoryRetriever`` applies the retention
        filter -> ranking -> deduplication -> top-k pipeline on top.
        With ``expand=True``, memories linked to the same entities as the
        top hits are added (depth-1 graph expansion) *before* the same
        Step 3 pipeline runs.
        """
        hits = self.vector.search(query, top_k=top_k)
        memory_ids: List[str] = []
        for hit in hits:
            memory_id = (hit.get("metadata") or {}).get("memory_id")
            if memory_id:
                memory_ids.append(memory_id)

        if expand:
            memory_ids = self._expand(memory_ids, top_k)

        results: List[Memory] = []
        seen = set()
        for memory_id in memory_ids:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            memory = self.get(memory_id)
            if memory is not None:
                results.append(memory)
        return results

    def close(self) -> None:
        pass

    def __enter__(self) -> "NebulonMind":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _new_memory_id(self) -> str:
        return f"mem_{uuid.uuid4().hex[:12]}"

    def _compensate(self, memory_id: str) -> None:
        """Undo partial writes (delete order: graph -> vector -> truth)."""
        try:
            self.graph.delete_memory(memory_id)
        except Exception:
            pass
        try:
            self.vector.delete(memory_id)
        except Exception:
            pass
        try:
            self.truth.delete(memory_id)
        except Exception:
            pass

    def _expand(self, seed_ids: List[str], limit: int) -> List[str]:
        """Depth-1 expansion: memories sharing an entity with a seed memory."""
        extra: List[str] = []
        for seed in seed_ids:
            for entity in self.graph.entities_of(seed):
                for memory_id in self.graph.memories_of(entity):
                    if memory_id not in seed_ids and memory_id not in extra:
                        extra.append(memory_id)
                        if len(seed_ids) + len(extra) >= limit * 3:
                            break
                if len(seed_ids) + len(extra) >= limit * 3:
                    break
        return seed_ids + extra
