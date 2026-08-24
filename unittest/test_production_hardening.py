"""Step 6 — production-hardening behaviour tests (P1/P2, fully offline).

Verifies the *semantics* documented in changes.txt:

* liveness vs readiness split (P1) — live stays green with a dead backend,
  ready reports 503 until the backend returns
* bounded retry policy (P1) — transient backend failures are retried for
  idempotent ops only; 4xx / application rejections never are
* explicit connect/read/write timeouts (P1) are parsed and passed through
* fail-fast production configuration validation (P2): sample credentials,
  plaintext HTTP, missing rate limit all block startup (the API has no
  authentication layer, so no API key check)
* backend version/compatibility check wiring (P2, best-effort)
* /metrics observability surface (P1)
* graceful shutdown: provider resources are released on app exit (P1)
"""

import logging

import pytest
import requests
from fastapi.testclient import TestClient

from nmd_host.api.client import NebulonDBClient, NebulonDBError
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import (
    NebulonDBConfig,
    ServiceConfig,
    validate_production_config,
)

API = "/api/NebulonMind"


class _DownProvider(InMemoryServiceProvider):
    """Backend-shaped provider whose backend is unreachable."""

    name = "nebulondb"

    def verify_backend(self) -> bool:
        return False


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


class _FlakyTransport:
    """Counts calls; fails with a transport error until ``succeed_after``."""

    def __init__(self, succeed_after=3, status=200, body=None):
        self.calls = 0
        self.succeed_after = succeed_after
        self.status = status
        self.body = body or {"success": True, "data": {}}
        self.last_timeout = None
        self.last_headers = None

    def request(self, method, url, json=None, timeout=None, headers=None):
        self.calls += 1
        self.last_timeout = timeout
        self.last_headers = headers
        if self.calls < self.succeed_after:
            raise requests.ConnectionError("flaky: backend down")
        return _Response(self.body, self.status)


# ---------------------------------------------------------------------- #
# Liveness vs readiness (P1)                                             #
# ---------------------------------------------------------------------- #


@pytest.fixture
def down_client():
    app = create_app(provider=_DownProvider(), config=ServiceConfig())
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_stays_green_with_dead_backend(down_client):
    resp = down_client.get(f"{API}/health/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["backend"] == "down"


def test_readiness_503_while_backend_down(down_client):
    resp = down_client.get(f"{API}/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["success"] is False
    assert "NebulonDB" in body["message"]


def test_readiness_200_for_in_memory_provider():
    app = create_app(provider=InMemoryServiceProvider(), config=ServiceConfig())
    with TestClient(app) as client:
        assert client.get(f"{API}/health/ready").status_code == 200


def test_compat_health_alias_still_works(down_client):
    resp = down_client.get(f"{API}/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["backend"] == "down"


# ---------------------------------------------------------------------- #
# Retry policy (P1)                                                      #
# ---------------------------------------------------------------------- #


def _client_with(transport) -> NebulonDBClient:
    client = NebulonDBClient(
        NebulonDBConfig(connect_timeout=1.0, read_timeout=2.0, write_timeout=5.0),
        backend_retries=2,
    )
    client._session = transport
    return client


def test_transient_failures_retried_then_succeed():
    transport = _FlakyTransport(succeed_after=3)
    client = _client_with(transport)
    result = client.verify()
    assert result["success"] is True
    assert transport.calls == 3


def test_4xx_never_retried():
    transport = _FlakyTransport(succeed_after=1, status=404)
    client = _client_with(transport)
    with pytest.raises(NebulonDBError):
        client.verify()
    assert transport.calls == 1


def test_application_rejection_never_retried():
    transport = _FlakyTransport(
        succeed_after=1, status=200, body={"success": False, "message": "rejected"}
    )
    client = _client_with(transport)
    with pytest.raises(NebulonDBError, match="rejected"):
        client.verify()
    assert transport.calls == 1


def test_writes_are_not_retried():
    transport = _FlakyTransport(succeed_after=99)
    client = _client_with(transport)
    with pytest.raises(NebulonDBError):
        client.load_segment("c", "s", "cosmos", [{"text": "x"}], [])
    assert transport.calls == 1


def test_explicit_timeouts_passed_through():
    transport = _FlakyTransport(succeed_after=1)
    client = _client_with(transport)
    client.verify()
    assert transport.last_timeout == (1.0, 2.0)  # (connect, read)
    transport2 = _FlakyTransport(succeed_after=1)
    client2 = _client_with(transport2)
    client2.load_segment("c", "s", "cosmos", [{"text": "x"}], [])
    assert transport2.last_timeout == (1.0, 5.0)  # (connect, write)


def test_timeout_config_parsed_from_env(monkeypatch):
    monkeypatch.setenv("NEBULONDB_API_CONNECT_TIMEOUT", "2.5")
    monkeypatch.setenv("NEBULONDB_API_READ_TIMEOUT", "12")
    monkeypatch.delenv("NEBULONDB_API_HOST", raising=False)
    config = NebulonDBConfig.from_env()
    assert config.connect_timeout == 2.5
    assert config.read_timeout == 12.0
    assert config.write_timeout == 60.0


# ---------------------------------------------------------------------- #
# Production configuration validation (P2)                               #
# ---------------------------------------------------------------------- #


def _prod_service(**overrides) -> ServiceConfig:
    base = dict(env="production", rate_limit_per_minute=60)
    base.update(overrides)
    return ServiceConfig(**base)


def _prod_backend(**overrides) -> NebulonDBConfig:
    base = dict(username="op", password="real-secret", scheme="https")
    base.update(overrides)
    return NebulonDBConfig(**base)


def test_valid_production_config_passes():
    problems = validate_production_config(_prod_service(), _prod_backend())
    assert problems == []


def test_sample_credentials_rejected_in_production():
    problems = validate_production_config(
        _prod_service(), _prod_backend(username="sathya", password="sathya")
    )
    assert any("sathya" in p for p in problems)


def test_plaintext_backend_rejected_in_production():
    problems = validate_production_config(
        _prod_service(), _prod_backend(scheme="http")
    )
    assert any("https" in p for p in problems)
    problems = validate_production_config(
        _prod_service(allow_plaintext_http=True), _prod_backend(scheme="http")
    )
    assert not any("https" in p for p in problems)


def test_missing_rate_limit_rejected_in_production():
    problems = validate_production_config(
        _prod_service(rate_limit_per_minute=0), _prod_backend()
    )
    assert any("NMD_API_RATE_LIMIT_PER_MINUTE" in p for p in problems)


def test_create_app_fails_fast_when_production_invalid(monkeypatch):
    monkeypatch.setenv("NMD_ENV", "production")
    monkeypatch.setenv("NEBULONDB_USERNAME", "sathya")
    monkeypatch.setenv("NEBULONDB_PASSWORD", "sathya")
    monkeypatch.setenv("NEBULONDB_API_SCHEME", "http")
    monkeypatch.setenv("NMD_API_RATE_LIMIT_PER_MINUTE", "60")
    with pytest.raises(RuntimeError, match="invalid production configuration"):
        create_app(provider=InMemoryServiceProvider())


def test_create_app_starts_with_valid_production_config(monkeypatch):
    monkeypatch.setenv("NMD_ENV", "production")
    monkeypatch.setenv("NEBULONDB_USERNAME", "realop")
    monkeypatch.setenv("NEBULONDB_PASSWORD", "realsecret")
    monkeypatch.setenv("NEBULONDB_API_SCHEME", "https")
    monkeypatch.setenv("NMD_API_RATE_LIMIT_PER_MINUTE", "60")
    app = create_app(provider=InMemoryServiceProvider())
    with TestClient(app) as client:
        resp = client.get(f"{API}/health/live")
        assert resp.status_code == 200


# ---------------------------------------------------------------------- #
# Backend compatibility (P2, best-effort)                                #
# ---------------------------------------------------------------------- #


def test_expected_backend_version_does_not_block_startup():
    app = create_app(
        provider=InMemoryServiceProvider(),
        config=ServiceConfig(expected_backend_version="1.0"),
    )
    with TestClient(app) as client:
        assert client.get(f"{API}/health/live").status_code == 200


class _UpProvider(InMemoryServiceProvider):
    """Backend-shaped provider whose backend is reachable."""

    name = "nebulondb"

    def verify_backend(self) -> bool:
        return True


def test_backend_compat_check_runs_against_reachable_backend():
    app = create_app(
        provider=_UpProvider(),
        config=ServiceConfig(expected_backend_version="1.0"),
    )
    with TestClient(app) as client:
        assert client.get(f"{API}/health/ready").status_code == 200


# ---------------------------------------------------------------------- #
# Observability (P1)                                                     #
# ---------------------------------------------------------------------- #


def test_metrics_endpoint_reports_request_counters_and_uptime():
    app = create_app(provider=InMemoryServiceProvider(), config=ServiceConfig())
    with TestClient(app) as client:
        client.get(f"{API}/search", params={"query": "hi"})
        client.get(f"{API}/health/live")
        text = client.get("/metrics").text
    assert 'nmd_http_requests_total{method="GET",route="/api/NebulonMind/search",status="200"}' in text
    assert "nmd_up 1" in text
    assert 'nmd_version_info{version=' in text
    assert "nmd_uptime_seconds" in text
    assert "nmd_auth_rejected_total" not in text


# ---------------------------------------------------------------------- #
# Graceful shutdown (P1)                                                 #
# ---------------------------------------------------------------------- #


def test_shutdown_releases_provider_resources():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider, config=ServiceConfig())
    provider.create_user("user_001")
    with TestClient(app) as client:
        client.post(
            f"{API}/memory",
            json={"content": {"text": "hello"}, "user_id": "user_001"},
            params={"user_id": "user_001"},
        )
        assert provider.user_count() == 1
    # TestClient exit runs the lifespan shutdown → provider.close()
    assert provider.user_count() == 0