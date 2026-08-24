from fastapi.testclient import TestClient

from app.core.config import get_settings


def _client(monkeypatch, **env) -> TestClient:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def test_security_headers_on_api_responses(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"


def test_headers_on_error_responses(client):
    response = client.get("/api/v1/status/job_missing")
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_docs_exempt_from_csp(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in {key.lower() for key in response.headers}


def test_hsts_absent_by_default_and_opt_in(client, isolated_data_dir, monkeypatch):
    assert "strict-transport-security" not in {k.lower() for k in client.get("/api/v1/health").headers}

    hardened = _client(monkeypatch, HSTS_ENABLED="true")
    hsts = hardened.get("/api/v1/health").headers["Strict-Transport-Security"]
    assert "max-age=31536000" in hsts


def test_cors_disabled_by_default(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch)
    preflight = c.options(
        "/api/v1/train",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in preflight.headers}

    simple = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in simple.headers}


def test_cors_allows_configured_origin_only(isolated_data_dir, monkeypatch):
    c = _client(monkeypatch, CORS_ORIGINS="http://good.example")
    allowed = c.options(
        "/api/v1/train",
        headers={"Origin": "http://good.example", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers["access-control-allow-origin"] == "http://good.example"

    denied = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in denied.headers}
