"""Config live-apply + LLM credential (.env) API tests (fully offline)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.routes import config as config_routes
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig, _load_cfg

_REPO = Path(__file__).resolve().parents[1]
REAL_CFG = (_REPO / "nebulonmind.cfg").read_text(encoding="utf-8")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBULONMD_HOME", str(tmp_path))
    cfg_path = tmp_path / "nebulonmind.cfg"
    cfg_path.write_text(REAL_CFG, encoding="utf-8")
    monkeypatch.setattr(config_routes, "CFG_FILE", cfg_path)
    _load_cfg(cfg_path, override=True)
    provider = InMemoryServiceProvider()
    provider.create_user("nmd_user_01")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as c:
        yield c


def test_cfg_update_applies_live(client):
    resp = client.put(
        "/api/NebulonMind/config/cfg",
        json={"config": {"llm": {"nmd_llm_provider": "ollama"}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["live"] is True
    assert os.environ.get("NMD_LLM_PROVIDER") == "ollama"

    fetched = client.get("/api/NebulonMind/config/cfg").json()
    groups = {g["id"]: g for g in fetched["data"]["groups"]}
    llm_values = {k["key"]: k["value"] for k in groups["llm"]["keys"]}
    assert llm_values["nmd_llm_provider"] == "ollama"


def test_llm_creds_written_to_dotenv(client, tmp_path):
    resp = client.put(
        "/api/NebulonMind/dashboard/config",
        json={"config": {"llm": {"model": "gpt-test", "api_key": "sk-test"}}},
    )
    assert resp.status_code == 200
    assert os.environ.get("NMD_LLM_MODEL") == "gpt-test"
    env_file = tmp_path / ".env"
    assert env_file.exists()
    content = env_file.read_text(encoding="utf-8")
    assert "NMD_LLM_MODEL=gpt-test" in content
    assert "NMD_LLM_API_KEY=sk-test" in content
    assert env_file.read_text(encoding="utf-8").count("NMD_LLM_MODEL") == 1