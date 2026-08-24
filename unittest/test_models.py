import pytest
from pydantic import ValidationError

from nmd_host.core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    MemoryType,
    RetentionPolicy,
)
from conftest import make_memory


def test_memory_defaults():
    memory = make_memory("My name is Sathya")
    assert memory.classification.memory_type is MemoryType.SEMANTIC
    assert memory.classification.category == "general"
    assert memory.importance.score == 0.5
    assert memory.importance.priority.value == "medium"
    assert memory.lifecycle.retention_policy is RetentionPolicy.TEMPORARY
    assert memory.entities == []
    assert memory.relationships == []


def test_text_required():
    with pytest.raises(ValidationError):
        Memory(user_id="user_001", content=MemoryContent(text=""))
    with pytest.raises(ValidationError):
        Memory(user_id="user_001", content=MemoryContent(text="   "))


def test_importance_range():
    with pytest.raises(ValidationError):
        make_memory("x", score=1.5)
    with pytest.raises(ValidationError):
        make_memory("x", confidence=-0.1)


def test_memory_type_values():
    for value in MemoryType:
        memory = make_memory("x", memory_type=value)
        assert memory.classification.memory_type is value


def test_truth_doc_roundtrip():
    memory = make_memory("My name is Sathya", entities=["Sathya"])
    memory.memory_id = "mem_001"
    doc = memory.to_truth_doc()
    assert doc["memory_id"] == "mem_001"
    assert doc["content"]["text"] == "My name is Sathya"
    restored = Memory.from_truth_doc(doc)
    assert restored == memory
    assert restored.memory_id == "mem_001"


def test_truth_doc_json_safe():
    memory = make_memory("x", entities=["Sathya"])
    memory.memory_id = "mem_002"
    doc = memory.to_truth_doc()
    assert isinstance(doc["user_id"], str)
    assert isinstance(doc["memory_id"], str)
    assert isinstance(doc["importance"]["score"], float)
