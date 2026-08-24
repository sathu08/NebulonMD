"""GraphStore — ORBIT Mesh on the same ``mind_semantic`` corpus as vectors.

Entities are nodes resolved by label (auto-created on first use). Each memory
is itself a node labelled with its ``memory_id`` and linked to its entities
via ``HAS_ENTITY`` edges; user-defined relationships become directed
entity → entity edges. Edge insertion is idempotent, so re-storing a memory
never duplicates edges.

Writes go through ``mesh_load_graph`` with label references. Reads are
derived from the truth store, because the API exposes no label→node-id
resolution; removing a memory's vector record (``VectorStore.delete``)
removes its mesh node and incident edges, so ``delete_memory`` is a no-op
here.
"""

from __future__ import annotations

from typing import List, Optional

from .truth_store import TruthStore
from nmd_host.core.models import Memory
from nmd_host.api import NebulonDBClient


class GraphStore:
    HAS_ENTITY = "HAS_ENTITY"
    CORPUS = "mind_semantic"

    def __init__(
        self,
        client: NebulonDBClient,
        user_id: str,
        truth: Optional[TruthStore] = None,
    ) -> None:
        self._api = client
        self._segment = f"user_{user_id}"
        self._truth = truth

    def relate_memory(self, memory: Memory) -> None:
        """Write all relationships + memory→entity links (idempotent).

        Relationship edges carry ``memory_id`` as provenance metadata. The
        NebulonDB Mesh only persists ``from``/``to``/``relation``/``weight``
        and ignores extra keys (verified: ``edges`` is ``list[dict]``), so
        the authoritative provenance record lives in the truth document;
        this key is advisory and forward-compatible.
        """
        edges = [
            {
                "from": rel.source,
                "to": rel.target,
                "relation": rel.relation,
                "memory_id": rel.source_memory_id or memory.memory_id,
            }
            for rel in memory.relationships
        ]
        edges += [
            {"from": memory.memory_id, "to": entity, "relation": self.HAS_ENTITY}
            for entity in memory.entities
        ]
        if edges:
            self._api.mesh_load_graph(self.CORPUS, self._segment, "orbit", edges)

    def link(self, memory_id: str, entity: str, relation: str = HAS_ENTITY) -> None:
        """Add a directed edge memory → entity (idempotent)."""
        self._api.mesh_load_graph(
            self.CORPUS,
            self._segment,
            "orbit",
            [{"from": memory_id, "to": entity, "relation": relation}],
        )
        if self._truth is not None:
            doc = self._truth.get(memory_id)
            if doc is not None and entity not in (doc.get("entities") or []):
                doc.setdefault("entities", []).append(entity)
                self._truth.create(doc)

    def entities_of(self, memory_id: str) -> List[str]:
        """Entity labels directly attached to a memory (from the truth doc)."""
        if self._truth is None:
            return []
        doc = self._truth.get(memory_id)
        return list(doc.get("entities") or []) if doc else []

    def memories_of(self, entity_label: str) -> List[str]:
        """memory_ids linked to an entity label via HAS_ENTITY."""
        if self._truth is None:
            return []
        return [
            doc["memory_id"]
            for doc in self._truth.read_all()
            if entity_label in (doc.get("entities") or [])
        ]

    def delete_memory(self, memory_id: str) -> None:
        """No-op: ``VectorStore.delete`` removes the mesh node and its edges.

        Deletion semantics (verified in NebulonDB's orbit orchestrator):
        deleting a record removes only that memory's mesh node and its
        incident edges. Shared entity nodes (e.g. ``Sathya``/``Python`` used
        by other memories) are NOT removed, so deleting one memory can never
        orphan or delete entities still referenced by another.
        """

    def count_edges(self) -> int:
        return int(self._api.segment_stats(self.CORPUS, self._segment, "orbit").get("edge_count", 0))

    def count_nodes(self) -> int:
        return int(self._api.segment_stats(self.CORPUS, self._segment, "orbit").get("node_count", 0))
