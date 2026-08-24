"""Step 6 — production security test suite (P2, fully offline).

The NebulonMind API authenticates only when ``NMD_API_AUTH_TOKEN`` is set
(open internal service by default); NebulonDB credentials live only inside
``NebulonMind`` and must never leak through responses. This suite covers:

* secret leakage (backend credentials never appear in responses)
* CORS: allowed origins get headers, disallowed origins do not
* oversized request bodies → 413, malformed input → 422
* per-client rate limiting → 429
* correlation ID echo
* optional shared-secret auth (Bearer / X-API-Key, public console assets)
"""

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig

ALLOWED_ORIGIN = "https://app.example.com"
DISALLOWED_ORIGIN = "https://evil.example.org"

MEMORY_URL = "/api/NebulonMind/memory"
SEARCH_URL = "/api/NebulonMind/search"


def _config(**overrides) -> ServiceConfig:
    base = dict(
        cors_origins=(ALLOWED_ORIGIN,),
        rate_limit_per_minute=0,
        max_body_bytes=4096,
    )
    base.update(overrides)
    return ServiceConfig(**base)


@pytest.fixture
def client():
    app = create_app(provider=InMemoryServiceProvider(), config=_config())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def authed_client():
    app = create_app(
        provider=InMemoryServiceProvider(),
        config=_config(auth_token="sekret-token"),
    )
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------- #
# Secret leakage (P0/P2)                                                 #
# ---------------------------------------------------------------------- #


def test_backend_credentials_never_appear_in_responses(client):
    health = client.get("/api/NebulonMind/health")
    text = str(health.json()) + client.get("/metrics").text
    for secret in ("sathya", "6969", "NEBULONDB_", "password"):
        assert secret not in text


def test_error_envelopes_do_not_echo_request_details(client):
    resp = client.get(
        SEARCH_URL,
        params={"query": "hi", "top_k": "not-an-int"},
    )
    body = resp.json()
    assert body["success"] is False
    assert "password" not in str(body)


# ---------------------------------------------------------------------- #
# CORS (P0)                                                              #
# ---------------------------------------------------------------------- #


def test_cors_allowed_origin_gets_headers(client):
    resp = client.options(
        MEMORY_URL,
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


def test_cors_disallowed_origin_gets_no_headers(client):
    resp = client.options(
        MEMORY_URL,
        headers={
            "Origin": DISALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------- #
# Request limits (P1) + malformed input                                  #
# ---------------------------------------------------------------------- #


def test_oversized_body_rejected_413(client):
    payload = {
        "content": {"text": "x" * 8000},
        "classification": {"memory_type": "semantic"},
    }
    resp = client.post(MEMORY_URL, json=payload)
    assert resp.status_code == 413
    assert resp.json()["success"] is False


def test_oversized_body_rejected_when_streamed(client):
    resp = client.post(
        MEMORY_URL,
        content=b'{"content": {"text": "' + b"y" * 8000 + b'"}}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_malformed_input_422(client):
    resp = client.post(MEMORY_URL, json={"content": "not-an-object"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["data"]["errors"]


def test_top_k_bounded_by_config(client):
    resp = client.get(
        SEARCH_URL, params={"query": "hi", "top_k": 500}
    )
    assert resp.status_code == 422  # le=config.max_top_k (50)


# ---------------------------------------------------------------------- #
# Rate limiting (P1)                                                     #
# ---------------------------------------------------------------------- #


def test_rate_limit_429_after_budget_exhausted():
    app = create_app(
        provider=InMemoryServiceProvider(),
        config=_config(rate_limit_per_minute=5),
    )
    with TestClient(app) as client:
        statuses = []
        for _ in range(6):
            resp = client.get(SEARCH_URL, params={"query": "hi"})
            statuses.append(resp.status_code)
        assert statuses[:5] == [200] * 5
        assert statuses[5] == 429
        assert resp.headers.get("retry-after") == "60"
        assert resp.json()["success"] is False


def test_rate_limiter_disabled_by_default():
    app = create_app(provider=InMemoryServiceProvider(), config=_config())
    with TestClient(app) as client:
        for _ in range(12):
            assert (
                client.get(SEARCH_URL, params={"query": "hi"})
                .status_code
                == 200
            )


# ---------------------------------------------------------------------- #
# Correlation ID (P1)                                                    #
# ---------------------------------------------------------------------- #


def test_request_id_echoed_and_honoured(client):
    resp = client.get(SEARCH_URL, params={"query": "hi"})
    assert resp.headers.get("x-request-id")
    supplied = "trace-my-own-id"
    resp = client.get(
        SEARCH_URL,
        params={"query": "hi"},
        headers={"X-Request-ID": supplied},
    )
    assert resp.headers.get("x-request-id") == supplied


# ---------------------------------------------------------------------- #
# Auth (optional shared secret, NMD_API_AUTH_TOKEN)                     #
# ---------------------------------------------------------------------- #


def test_auth_required_when_token_configured(authed_client):
    resp = authed_client.get(SEARCH_URL, params={"query": "hi"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "auth" in body["message"]
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_auth_rejects_wrong_token(authed_client):
    resp = authed_client.get(
        SEARCH_URL,
        params={"query": "hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_auth_accepts_bearer_token(authed_client):
    resp = authed_client.get(
        SEARCH_URL,
        params={"query": "hi"},
        headers={"Authorization": "Bearer sekret-token"},
    )
    assert resp.status_code == 200


def test_auth_accepts_x_api_key_header(authed_client):
    resp = authed_client.get(
        SEARCH_URL,
        params={"query": "hi"},
        headers={"X-API-Key": "sekret-token"},
    )
    assert resp.status_code == 200


def test_auth_disabled_when_no_token_configured(client):
    assert client.get(SEARCH_URL, params={"query": "hi"}).status_code == 200


def test_auth_keeps_health_and_metrics_public(authed_client):
    assert authed_client.get("/api/NebulonMind/health").status_code == 200
    assert authed_client.get("/api/NebulonMind/health/live").status_code == 200
    assert authed_client.get("/api/NebulonMind/health/ready").status_code == 200
    assert authed_client.get("/metrics").status_code == 200


def test_auth_passes_cors_preflight(authed_client):
    resp = authed_client.options(
        MEMORY_URL,
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200


def test_auth_keeps_console_page_public(authed_client):
    for path in (
        "/api/NebulonMind/dashboard",
        "/api/NebulonMind/dashboard/",
        "/api/NebulonMind/dashboard/index",
        "/api/NebulonMind/dashboard/web",
    ):
        assert authed_client.get(path).status_code == 200, path


def test_auth_keeps_console_assets_public(authed_client):
    assert (
        authed_client.get("/api/NebulonMind/dashboard/css/main.css").status_code
        == 200
    )
    assert (
        authed_client.get("/api/NebulonMind/dashboard/js/main.js").status_code == 200
    )


def test_auth_still_protects_console_config(authed_client):
    assert authed_client.get("/api/NebulonMind/dashboard/config").status_code == 401
    assert authed_client.put("/api/NebulonMind/dashboard/config").status_code == 401