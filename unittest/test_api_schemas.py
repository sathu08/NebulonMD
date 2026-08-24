"""Phase 4.1 — API schema tests.

The API request/response models must stay presentation-only: ``Memory``
semantics are unchanged, Step 2/Step 3 objects are reused, and every
documented model carries OpenAPI-visible examples.
"""

import pytest
from pydantic import ValidationError

from nmd_host.api.schemas import (
    Envelope,
    HealthData,
    IngestReport,
    IntelligenceDecideRequest,
    IntelligenceProcessRequest,
    MemoryCreate,
    MemoryUpdate,
    ProcessData,
)
from nmd_host.core.models import Memory, MemoryStatus
from nmd_host.intelligence.schemas import Conversation


# ---------------------------------------------------------------------- #
# 4.1 — request models                                                   #
# ---------------------------------------------------------------------- #


def test_memory_create_is_a_memory_with_unchanged_semantics():
    memory = MemoryCreate(
        user_id="user_001",
        content={"text": "My name is Sathya"},
        memory_id="mem_custom",  # client-chosen ids are still accepted
    )
    assert isinstance(memory, Memory)
    assert memory.memory_id == "mem_custom"
    # defaults unchanged from the core model
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.classification.memory_type.value == "semantic"


def test_memory_create_round_trips_like_core_memory():
    memory = MemoryCreate.model_validate(
        Memory(user_id="u", content={"text": "hello"}).model_dump()
    )
    assert memory.to_truth_doc()["content"]["text"] == "hello"
    assert memory.memory_id is None  # server assigns on store


def test_memory_update_only_optional_fields():
    update = MemoryUpdate(status="archived")
    dumped = update.model_dump(exclude_unset=True)
    assert dumped == {"status": "archived"}  # absent fields not sent


def test_memory_update_accepts_full_partial_payload():
    update = MemoryUpdate(
        status="archived",
        importance={"score": 0.9},
        entities=["Python"],
    )
    data = update.model_dump(exclude_unset=True, mode="json")
    assert data["importance"] == {"score": 0.9}
    assert data["entities"] == ["Python"]


def test_memory_update_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        MemoryUpdate(unknown_field="x")


# ---------------------------------------------------------------------- #
# 4.1 — intelligence input models                                        #
# ---------------------------------------------------------------------- #


def test_intelligence_input_requires_text_or_conversation():
    with pytest.raises(ValidationError):
        IntelligenceDecideRequest()
    with pytest.raises(ValidationError):
        IntelligenceDecideRequest(text="")
    with pytest.raises(ValidationError):
        IntelligenceDecideRequest(conversation=Conversation(turns=[]))
    with pytest.raises(ValidationError):
        IntelligenceProcessRequest(text="   ")  # stripped to empty


def test_intelligence_input_resolves_conversation():
    request = IntelligenceDecideRequest(text="My name is Sathya.")
    conversation = request.to_conversation()
    assert isinstance(conversation, Conversation)
    assert conversation.turns[0].content == "My name is Sathya."

    multi = IntelligenceProcessRequest(
        conversation={
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
    )
    assert len(multi.to_conversation().turns) == 2


def test_process_request_defaults_persist_true():
    assert IntelligenceProcessRequest(text="hi").persist is True
    assert IntelligenceProcessRequest(text="hi", persist=False).persist is False


# ---------------------------------------------------------------------- #
# 4.1 — response models                                                  #
# ---------------------------------------------------------------------- #


def test_envelope_generic_shape():
    envelope = Envelope[HealthData](message="ok", data=HealthData(backend="up"))
    assert envelope.model_dump() == {
        "success": True,
        "message": "ok",
        "data": {"service": "NebulonMind", "version": "", "provider": "",
                 "backend": "up", "minds": 0},
    }


def test_ingest_report_literals():
    report = IngestReport(action="STORE", memory_id="mem_1")
    assert report.model_dump()["action"] == "STORE"
    with pytest.raises(ValidationError):
        IngestReport(action="MAYBE")  # only the gate's four actions


def test_process_data_accepts_empty_lists():
    data = ProcessData(decisions=[], ingestions=[])
    assert data.ingestions == []


def test_health_data_backend_literal_validation():
    with pytest.raises(ValidationError):
        HealthData(backend="maybe")


# ---------------------------------------------------------------------- #
# 4.1 — OpenAPI-visible examples (part of the schema contract)           #
# ---------------------------------------------------------------------- #


def test_create_request_has_example():
    schema = MemoryCreate.model_json_schema()
    assert schema.get("examples"), "MemoryCreate must carry an example"
    example = schema["examples"][0]
    assert example["content"]["text"] == "My name is Sathya"
    assert "user_id" in example


def test_update_request_has_example():
    schema = MemoryUpdate.model_json_schema()
    assert schema.get("examples")
    assert "status" in schema["examples"][0]


def test_intelligence_requests_have_examples():
    assert IntelligenceDecideRequest.model_json_schema().get("examples")
    assert IntelligenceProcessRequest.model_json_schema().get("examples")


def test_documented_models_carry_descriptions():
    assert MemoryCreate.__doc__ and "memory_id" in MemoryCreate.__doc__
    assert IngestReport.__doc__ and "gate" in IngestReport.__doc__