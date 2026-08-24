import io

TRAIN_PAYLOAD_TEMPLATE = {
    "targets": [
        {"name": "score_continuous", "task": "regression"},
        {"name": "grade", "task": "classification"},
    ],
    "models": ["ridge_regression", "logistic_regression"],
    "cv": {"n_folds": 3, "mode": "single", "random_state": 42},
    "hpo": {"enabled": False},
    "explain": {"shap": False},
}


def _upload(client, csv_bytes) -> str:
    response = client.post(
        "/api/v1/datasets/analyze-dataset",
        files={"file": ("synthetic.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def _train(client, dataset_id) -> str:
    payload = dict(TRAIN_PAYLOAD_TEMPLATE)
    payload["dataset_id"] = dataset_id
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = client.get(f"/api/v1/status/{job_id}").json()
    assert status["state"] == "SUCCESS", status
    return job_id


def test_predict_json_records_with_target_subset(client, mixed_targets_csv):
    job_id = _train(client, _upload(client, mixed_targets_csv))
    records = [
        {"x1": 0.5, "x2": 5.0, "category": "A"},
        {"x1": -1.0, "x2": 6.5, "category": "B"},
    ]
    response = client.post(f"/api/v1/predict/{job_id}?targets=grade", json={"records": records})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["job_id"] == job_id
    assert body["n_rows"] == 2
    assert set(body["targets"]) == {"grade"}
    assert sorted(body["targets"]["grade"]["classes"]) == ["high", "low", "mid"]

    row = body["predictions"][0]
    assert row["row_index"] == 0
    assert row["grade"] in {"low", "mid", "high"}
    probs = row["grade_probabilities"]
    assert set(probs) == {"low", "mid", "high"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert "score_continuous" not in row


def test_predict_file_upload_all_targets(client, mixed_targets_csv):
    job_id = _train(client, _upload(client, mixed_targets_csv))
    buffer = io.StringIO()
    buffer.write("x1,x2,category\n0.2,4.8,A\n-0.7,5.5,C\n")
    response = client.post(
        f"/api/v1/predict/{job_id}",
        files={"file": ("batch.csv", buffer.getvalue().encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["n_rows"] == 2
    assert set(body["targets"]) == {"score_continuous", "grade"}
    assert isinstance(body["predictions"][0]["score_continuous"], float)
    assert body["predictions"][1]["row_index"] == 1


def test_predict_unknown_job_404(client):
    response = client.post("/api/v1/predict/job_nope", json={"records": [{"x1": 1}]})
    assert response.status_code == 404


def test_predict_unfinished_job_409(client, mixed_targets_csv):
    from app.services.job_service import get_job_store

    dataset_id = _upload(client, mixed_targets_csv)
    get_job_store().create("job_pending_pred", dataset_id, {})
    response = client.post(
        "/api/v1/predict/job_pending_pred",
        json={"records": [{"x1": 1, "x2": 2, "category": "A"}]},
    )
    assert response.status_code == 409


def test_predict_missing_column_422(client, mixed_targets_csv):
    job_id = _train(client, _upload(client, mixed_targets_csv))
    response = client.post(f"/api/v1/predict/{job_id}", json={"records": [{"x1": 0.1}]})
    assert response.status_code == 422
    assert "x2" in response.json()["detail"]


def test_predict_empty_records_422(client):
    response = client.post("/api/v1/predict/job_any", json={"records": []})
    assert response.status_code == 422


def test_predict_invalid_json_body_422(client):
    response = client.post(
        "/api/v1/predict/job_any",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
