"""Repository layer.

``MemoryRepository`` is the only interface the AI layer talks to — never the
stores directly. Two implementations:

* ``InMemoryRepository`` — dict-backed fake for tests without a database.
* ``NebulonMindRepository`` — API-backed: every operation is delegated to a
  ``NebulonMind``, so all persistence goes through the NebulonDB REST API.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Set

from .models import Memory


class MemoryRepository(Protocol):
    """The only interface the AI layer talks to — never the stores directly."""

    def create(self, memory: Memory) -> Memory:
        ...

    def get(self, memory_id: str) -> Optional[Memory]:
        ...

    def update(self, memory_id: str, data: Dict[str, Any]) -> Memory:
        ...

    def delete(self, memory_id: str) -> bool:
        ...

    def list_all(self, user_id: Optional[str] = None) -> List[Memory]:
        """Enumerate memories (used by lifecycle cleanup/consolidation)."""

    def search(self, query: str, top_k: int = 5) -> List[Memory]:
        ...

    def relate(self, memory_id: str, entity: str) -> None:
        ...

    def close(self) -> None:
        ...


def _tokens(*texts: Optional[str]) -> Set[str]:
    tokens: Set[str] = set()
    for text in texts:
        if not text:
            continue
        for word in text.lower().split():
            tokens.add(word.strip(".,;:!?'\"()[]{}"))
    return {t for t in tokens if t}


class InMemoryRepository:
    """Fake repository: dict-backed memories + adjacency of entities.

    Search is a simple token-overlap scorer on text + summary + entities,
    enough to exercise callers without any database.
    """

    def __init__(self) -> None:
        self._memories: Dict[str, Memory] = {}
        self._entity_memories: Dict[str, Set[str]] = defaultdict(set)

    def create(self, memory: Memory) -> Memory:
        if memory.memory_id is None:
            memory.memory_id = f"mem_fake_{len(self._memories) + 1:05d}"
        memory.stamp_provenance()
        self._memories[memory.memory_id] = memory.model_copy(deep=True)
        for entity in memory.entities:
            self._entity_memories[entity].add(memory.memory_id)
        return self._memories[memory.memory_id]

    def get(self, memory_id: str) -> Optional[Memory]:
        memory = self._memories.get(memory_id)
        return memory.model_copy(deep=True) if memory else None

    def update(self, memory_id: str, data: Dict[str, Any]) -> Memory:
        if memory_id not in self._memories:
            raise KeyError(memory_id)
        merged = {**self._memories[memory_id].model_dump(), **data}
        merged["memory_id"] = memory_id
        updated = Memory.model_validate(merged)
        updated.stamp_provenance()
        self._memories[memory_id] = updated
        for entity in updated.entities:
            self._entity_memories[entity].add(memory_id)
        return updated.model_copy(deep=True)

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._memories:
            return False
        del self._memories[memory_id]
        return True

    def list_all(self, user_id: Optional[str] = None) -> List[Memory]:
        memories = list(self._memories.values())
        if user_id is not None:
            memories = [m for m in memories if m.user_id == user_id]
        return [m.model_copy(deep=True) for m in memories]

    def search(self, query: str, top_k: int = 5) -> List[Memory]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        scored: List[tuple] = []
        for memory in self._memories.values():
            text = " ".join(
                filter(None, [
                    memory.content.text,
                    memory.content.summary,
                    " ".join(memory.entities),
                ])
            )
            overlap = len(q_tokens & _tokens(text))
            if overlap:
                scored.append((overlap, memory))
        scored.sort(key=lambda item: (-item[0], item[1].memory_id or ""))
        return [m.model_copy(deep=True) for _, m in scored[:top_k]]

    def relate(self, memory_id: str, entity: str) -> None:
        memory = self._memories.get(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        if entity not in memory.entities:
            memory.entities.append(entity)
            self._entity_memories[entity].add(memory_id)

    def close(self) -> None:
        self._memories.clear()
        self._entity_memories.clear()


class NebulonMindRepository:
    """API-backed repository: implements ``MemoryRepository`` by delegating
    every operation to a ``NebulonMind`` (truth + vectors + graph over REST).

    Search maps to ``NebulonMind.recall`` (server-side embedding), so this
    implementation needs no local model or storage.
    """

    def __init__(self, mind) -> None:
        self._mind = mind

    def create(self, memory: Memory) -> Memory:
        return self._mind.store(memory)

    def get(self, memory_id: str) -> Optional[Memory]:
        return self._mind.get(memory_id)

    def update(self, memory_id: str, data: Dict[str, Any]) -> Memory:
        return self._mind.update(memory_id, data)

    def delete(self, memory_id: str) -> bool:
        return self._mind.delete(memory_id)

    def list_all(self, user_id: Optional[str] = None) -> List[Memory]:
        memories = self._mind.list_all()
        if user_id is not None:
            memories = [m for m in memories if m.user_id == user_id]
        return memories

    def search(self, query: str, top_k: int = 5) -> List[Memory]:
        return self._mind.recall(query, top_k=top_k)

    def relate(self, memory_id: str, entity: str) -> None:
        self._mind.relate(memory_id, entity)

    def close(self) -> None:
        self._mind.close()
