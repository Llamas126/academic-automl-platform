import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1.predict import content_length_exceeds
from app.core.config import get_settings
from app.core.exceptions import PayloadTooLargeError


def _client(monkeypatch, **env) -> TestClient:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def _small_csv(rows: int = 10) -> bytes:
    buffer = io.StringIO()
    buffer.write("x1,x2,category,score_continuous,grade\n")
    for i in range(rows):
        buffer.write(f"{i * 0.1},{5 + i * 0.2},{'A' if i % 2 else 'B'},{i},low\n")
    return buffer.getvalue().encode()


def _upload(client: TestClient, csv_bytes: bytes, name: str = "synthetic.csv"):
    return client.post("/api/v1/datasets/analyze-dataset", files={"file": (name, csv_bytes, "text/csv")})


def test_traversal_filename_is_contained(client, isolated_data_dir):
    response = _upload(client, _small_csv(), name="../../evil.csv")
    assert response.status_code == 201, response.text

    dataset_id = response.json()["dataset_id"]
    assert (isolated_data_dir / "uploads" / dataset_id / "evil.csv").exists()
    escapes = [path for path in isolated_data_dir.rglob("evil.csv") if dataset_id not in path.parts]
    assert not escapes


def test_windows_style_traversal_is_contained(client, isolated_data_dir):
    response = _upload(client, _small_csv(), name="..\\..\\win_escape.csv")
    assert response.status_code == 201, response.text

    dataset_id = response.json()["dataset_id"]
    assert (isolated_data_dir / "uploads" / dataset_id / "win_escape.csv").exists()
    assert not (isolated_data_dir / "win_escape.csv").exists()


def test_invalid_dataset_id_rejected(client):
    assert client.get("/api/v1/datasets/bad$id").status_code == 422
    assert client.delete("/api/v1/datasets/bad$id").status_code == 422


def test_invalid_job_id_rejected(client):
    assert client.get("/api/v1/status/bad%5Cid").status_code == 422
    assert client.get("/api/v1/download/bad%5Cid").status_code == 422
    response = client.post("/api/v1/predict/bad%5Cid", json={"records": [{"x1": 1}]})
    assert response.status_code == 422


def test_oversize_upload_aborts_with_413(isolated_data_dir, monkeypatch):
    client = _client(monkeypatch, MAX_UPLOAD_MB="1")
    big_payload = b"x,y\n" + b"1,2\n" * 400_000
    response = _upload(client, big_payload)
    assert response.status_code == 413
    assert response.json()["error"] == "PayloadTooLargeError"


def test_row_limit_rejects_dataset(isolated_data_dir, monkeypatch):
    client = _client(monkeypatch, MAX_DATASET_ROWS="5")
    response = _upload(client, _small_csv(rows=10))
    assert response.status_code == 422
    assert "rows" in response.json()["detail"]


def test_predict_records_limit(isolated_data_dir, monkeypatch, mixed_targets_csv):
    client = _client(monkeypatch, PREDICT_MAX_RECORDS="1")
    upload = _upload(client, mixed_targets_csv)
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    train = client.post(
        "/api/v1/train",
        json={
            "dataset_id": dataset_id,
            "targets": [{"name": "score_continuous", "task": "regression"}],
            "models": ["ridge_regression"],
            "cv": {"n_folds": 2, "mode": "single"},
            "hpo": {"enabled": False},
            "explain": {"shap": False},
        },
    )
    assert train.status_code == 202
    job_id = train.json()["job_id"]
    status = client.get(f"/api/v1/status/{job_id}").json()
    assert status["state"] == "SUCCESS", status

    response = client.post(
        f"/api/v1/predict/{job_id}",
        json={
            "records": [
                {"x1": 0.1, "x2": 5.0, "category": "A"},
                {"x1": 0.2, "x2": 5.1, "category": "B"},
            ]
        },
    )
    assert response.status_code == 422
    assert "limit" in response.json()["detail"]


def test_content_length_guard_unit():
    max_bytes = 100
    assert content_length_exceeds(SimpleNamespace(headers={"content-length": "101"}), max_bytes)
    assert not content_length_exceeds(SimpleNamespace(headers={"content-length": "100"}), max_bytes)
    assert not content_length_exceeds(SimpleNamespace(headers={}), max_bytes)
    assert not content_length_exceeds(SimpleNamespace(headers={"content-length": "abc"}), max_bytes)


def test_capped_reader_stops_early():
    from app.services.dataset_service import UPLOAD_CHUNK_BYTES, read_upload_capped

    class FakeUpload:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, size):
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    upload = FakeUpload([b"a" * UPLOAD_CHUNK_BYTES, b"b" * 10])
    with pytest.raises(PayloadTooLargeError):
        asyncio.run(read_upload_capped(upload, UPLOAD_CHUNK_BYTES))
