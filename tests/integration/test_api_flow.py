def _upload(client, csv_bytes, name="synthetic.csv"):
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": (name, csv_bytes, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()


def test_full_train_flow_eager(client, mixed_targets_csv):
    profile = _upload(client, mixed_targets_csv)
    dataset_id = profile["dataset_id"]

    train_response = client.post(
        "/api/v1/train",
        json={
            "dataset_id": dataset_id,
            "targets": [
                {"name": "score_continuous", "task": "regression"},
                {"name": "grade", "task": "classification"},
            ],
            "models": ["ridge_regression", "logistic_regression"],
            "cv": {"n_folds": 3, "mode": "single", "random_state": 42},
            "hpo": {"enabled": False},
            "explain": {"shap": False},
        },
    )
    assert train_response.status_code == 202
    body = train_response.json()
    job_id = body["job_id"]
    assert body["status_url"] == f"/api/v1/status/{job_id}"

    status_response = client.get(f"/api/v1/status/{job_id}")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["state"] in ("SUCCESS", "FAILURE")
    if status_body["state"] == "FAILURE":
        raise AssertionError(f"job failed: {status_body['error']}")
    assert status_body["download_ready"] is True

    download = client.get(f"/api/v1/download/{job_id}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.content[:2] == b"PK"


def test_train_with_unknown_dataset_404(client):
    response = client.post(
        "/api/v1/train",
        json={
            "dataset_id": "ds_missing",
            "targets": [{"name": "y", "task": "regression"}],
        },
    )
    assert response.status_code == 404


def test_train_with_unknown_target_column(client, mixed_targets_csv):
    profile = _upload(client, mixed_targets_csv)
    response = client.post(
        "/api/v1/train",
        json={
            "dataset_id": profile["dataset_id"],
            "targets": [{"name": "not_a_column", "task": "auto"}],
        },
    )
    assert response.status_code == 422


def test_train_with_unknown_model_key(client, mixed_targets_csv):
    profile = _upload(client, mixed_targets_csv)
    response = client.post(
        "/api/v1/train",
        json={
            "dataset_id": profile["dataset_id"],
            "targets": [{"name": "grade", "task": "classification"}],
            "models": ["fake_net"],
        },
    )
    assert response.status_code == 422


def test_status_of_missing_job_404(client):
    assert client.get("/api/v1/status/job_nope").status_code == 404


def test_download_of_incomplete_job_conflict(client, mixed_targets_csv):
    from app.services.job_service import get_job_store

    profile = _upload(client, mixed_targets_csv)
    job_id = "job_pending_test"
    get_job_store().create(job_id, profile["dataset_id"], {})
    response = client.get(f"/api/v1/download/{job_id}")
    assert response.status_code == 409
