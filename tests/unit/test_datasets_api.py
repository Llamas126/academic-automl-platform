def test_analyze_dataset_roundtrip(client, mixed_targets_csv):
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("synthetic.csv", mixed_targets_csv, "text/csv")},
    )
    assert response.status_code == 201
    profile = response.json()

    assert profile["n_rows"] == 120
    assert profile["n_cols"] == 5
    roles = {c["name"]: c["role"] for c in profile["columns"]}
    assert roles["x1"] == "numeric"
    assert roles["category"] == "categorical"

    suggestions = {s["name"]: s["suggested_task"] for s in profile["target_suggestions"]}
    assert suggestions["score_continuous"] == "regression"
    assert suggestions["grade"] == "classification"

    dataset_id = profile["dataset_id"]
    fetched = client.get(f"/api/v1/datasets/{dataset_id}")
    assert fetched.status_code == 200
    assert fetched.json()["dataset_id"] == dataset_id


def test_analyze_dataset_rejects_bad_extension(client):
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_get_missing_dataset_returns_404(client):
    response = client.get("/api/v1/datasets/ds_does_not_exist")
    assert response.status_code == 404
