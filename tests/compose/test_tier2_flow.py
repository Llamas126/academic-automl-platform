"""Tier-2 smoke flow against the composed microservice stack.

Run inside the ``tier2`` compose service once every container is healthy.
Exercises the full user journey through the gateway proxy: upload ->
train -> poll -> download -> predict-from-file.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import httpx
import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_TIER2") != "1",
        reason="tier2 flow runs only against the composed stack (RUN_TIER2=1)",
    )
]

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8000")
POLL_TIMEOUT_S = int(os.environ.get("TIER2_POLL_TIMEOUT", "300"))


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=GATEWAY_URL, timeout=60.0) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def regression_csv() -> bytes:
    rng = np.random.default_rng(11)
    n = 90
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(3, 1.5, n)
    df = pd.DataFrame(
        {
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "y": np.round(2 * x1 - 1.5 * x2 + rng.normal(0, 0.25, n), 4),
        }
    )
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _wait_job(client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/status/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["state"] in ("SUCCESS", "FAILURE"):
            return payload
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish within {POLL_TIMEOUT_S}s")


def test_gateway_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_train_download_predict_flow(client, regression_csv):
    upload = client.post(
        "/api/v1/datasets/analyze-dataset",
        files={"file": ("regression.csv", regression_csv, "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    train = client.post(
        "/api/v1/train",
        json={
            "dataset_id": dataset_id,
            "targets": [{"name": "y"}],
            "cv": {"mode": "single", "n_folds": 3},
            "hpo": {"enabled": False},
            "explain": {"shap": False},
        },
    )
    assert train.status_code == 202, train.text
    job_id = train.json()["job_id"]

    final = _wait_job(client, job_id)
    assert final["state"] == "SUCCESS", final

    summary = client.get(f"/api/v1/results/{job_id}")
    assert summary.status_code == 200
    assert summary.json()["targets"][0]["target"] == "y"

    bundle = client.get(f"/api/v1/download/{job_id}")
    assert bundle.status_code == 200
    assert bundle.headers["content-type"] in (
        "application/zip",
        "application/x-zip-compressed",
    )

    predict = client.post(f"/api/v1/predict/{job_id}", json={"records": [{"x1": 0.5, "x2": 2.0}]})
    assert predict.status_code == 200, predict.text
    assert len(predict.json()["predictions"]) == 1
