import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from nmd_host.api import NebulonDBClient, NebulonDBError
from nmd_host.core.config import NebulonDBConfig
from nmd_host.core.models import Classification, Importance, Memory, MemoryContent, MemoryType


@pytest.fixture(scope="session")
def api_config():
    config = NebulonDBConfig.from_env()
    assert config.username and config.password, (
        ".env must set NEBULONDB_USERNAME and NEBULONDB_PASSWORD"
    )
    return config


@pytest.fixture(scope="session")
def api_client(api_config):
    client = NebulonDBClient(api_config)
    try:
        client.verify()
    except NebulonDBError as exc:
        pytest.skip(f"NebulonDB API unavailable at {api_config.base_url}: {exc}")
    return client


@pytest.fixture
def unique_user():
    return "user_" + uuid.uuid4().hex[:12]


def make_memory(
    text: str,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    user_id: str = "user_001",
    entities=None,
    relationships=None,
    **kwargs,
) -> Memory:
    return Memory(
        user_id=user_id,
        content=MemoryContent(text=text),
        classification=Classification(memory_type=memory_type),
        importance=Importance(**kwargs),
        entities=list(entities or []),
        relationships=list(relationships or []),
    )
