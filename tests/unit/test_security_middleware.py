from fastapi.testclient import TestClient

from app.core.config import get_settings


def _client(monkeypatch, **env) -> TestClient:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def test_defaults_keep_endpoints_open(client):
    response = client.post(
        "/api/v1/train",
        json={"dataset_id": "ds_none", "targets": [{"name": "y", "task": "regression"}]},
    )
    assert response.status_code == 404


def test_auth_requires_header(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, AUTH_ENABLED="true", API_KEYS="secret-key")
    response = c.post("/api/v1/train", json={"dataset_id": "ds_none"})
    assert response.status_code == 401


def test_auth_rejects_wrong_key(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, AUTH_ENABLED="true", API_KEYS="secret-key")
    response = c.post("/api/v1/train", json={"dataset_id": "ds_none"}, headers={"X-API-Key": "bad"})
    assert response.status_code == 403
    assert response.json()["error"] == "Forbidden"


def test_auth_accepts_valid_key(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, AUTH_ENABLED="true", API_KEYS="secret-key")
    response = c.post(
        "/api/v1/train",
        json={"dataset_id": "ds_none", "targets": [{"name": "y", "task": "regression"}]},
        headers={"X-API-Key": "secret-key"},
    )
    assert response.status_code == 404


def test_reads_stay_public_when_auth_enabled(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, AUTH_ENABLED="true", API_KEYS="k1,k2")
    assert c.get("/api/v1/status/job_x").status_code == 404
    assert c.get("/api/v1/health").status_code == 200


def test_auth_without_keys_fails_closed(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, AUTH_ENABLED="true")
    response = c.post("/api/v1/train", json={"dataset_id": "ds_none"}, headers={"X-API-Key": "whatever"})
    assert response.status_code == 503


def test_rate_limit_returns_429_with_retry_after(isolated_data_dir, monkeypatch):
    c = _client(
        monkeypatch,
        RATE_LIMIT_ENABLED="true",
        RATE_LIMIT_MAX_REQUESTS="2",
        RATE_LIMIT_WINDOW_SECONDS="60",
    )
    assert c.get("/api/v1/status/a").status_code == 404
    assert c.get("/api/v1/status/b").status_code == 404
    third = c.get("/api/v1/status/c")
    assert third.status_code == 429
    assert third.json()["error"] == "RateLimitExceeded"
    assert int(third.headers["Retry-After"]) >= 1


def test_health_exempt_from_rate_limit(isolated_data_dir, monkeypatch):
    c = _client(
        monkeypatch,
        RATE_LIMIT_ENABLED="true",
        RATE_LIMIT_MAX_REQUESTS="1",
        RATE_LIMIT_WINDOW_SECONDS="60",
    )
    for _ in range(3):
        assert c.get("/api/v1/health").status_code == 200
