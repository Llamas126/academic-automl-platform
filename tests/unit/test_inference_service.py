import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ConflictError, NotFoundError, UnsupportedFileError, ValidationFailedError
from app.services import inference_service
from app.services.artifact_service import save_final_models
from app.services.inference_service import resolve_job_artifacts, run_inference
from app.services.job_service import get_job_store, mark_success


def _train_pipelines(df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline

    from app.ml.preprocessing.builders import build_preprocessor

    features = ["x1", "cat"]

    reg = Pipeline(
        [
            ("preprocess", build_preprocessor(["x1"], ["cat"], scale_numeric=False)),
            ("estimator", LinearRegression()),
        ]
    )
    reg.fit(df[features], df["y_num"])

    clf = Pipeline(
        [
            ("preprocess", build_preprocessor(["x1"], ["cat"], scale_numeric=False)),
            ("estimator", LogisticRegression()),
        ]
    )
    clf.fit(df[features], df["y_bin"])
    return reg, clf


@pytest.fixture
def trained_job(isolated_data_dir):
    rng = np.random.default_rng(11)
    n = 60
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "cat": rng.choice(["A", "B"], n)})
    df["y_num"] = 2.0 * df["x1"] + np.where(df["cat"] == "B", 1.0, 0.0) + rng.normal(0, 0.05, n)
    df["y_bin"] = (df["x1"] + rng.normal(0, 0.2, n) > 0).astype(int)

    reg, clf = _train_pipelines(df)
    artifacts = {
        "y_num": SimpleNamespace(
            final_pipeline=reg,
            task="regression",
            best_entry_key="linear_regression",
            best_params={},
            label_classes=None,
        ),
        "y_bin": SimpleNamespace(
            final_pipeline=clf,
            task="classification",
            best_entry_key="logistic_regression",
            best_params={},
            label_classes=["neg", "pos"],
        ),
    }
    job_id = "job_inf_test"
    out_dir = isolated_data_dir / "jobs" / job_id / "output"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True)
    save_final_models(artifacts, models_dir)

    summary = {
        "feature_columns": ["x1", "cat"],
        "targets": [
            {
                "target": "y_num",
                "task": "regression",
                "best_model": "Linear Regression",
                "label_encoder_classes": None,
            },
            {
                "target": "y_bin",
                "task": "classification",
                "best_model": "Logistic Regression",
                "label_encoder_classes": ["neg", "pos"],
            },
        ],
    }
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "results_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    store = get_job_store()
    store.create(job_id, "ds_src", {})
    mark_success(store, job_id)
    return job_id, df


def test_predict_row_alignment_with_probabilities(trained_job):
    job_id, df = trained_job
    records = df.iloc[:3][["x1", "cat"]].to_dict(orient="records")

    response = run_inference(job_id, pd.DataFrame(records))

    assert response.job_id == job_id
    assert response.n_rows == 3
    assert [row["row_index"] for row in response.predictions] == [0, 1, 2]
    assert set(response.targets) == {"y_num", "y_bin"}
    assert response.targets["y_num"].model == "Linear Regression"
    assert response.targets["y_bin"].classes == ["neg", "pos"]

    first = response.predictions[0]
    assert isinstance(first["y_num"], float)
    assert first["y_bin"] in {"neg", "pos"}
    probs = first["y_bin_probabilities"]
    assert set(probs) == {"neg", "pos"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_subset_targets(trained_job):
    job_id, df = trained_job
    response = inference_service.run_inference_from_records(
        job_id, df.iloc[:2].to_dict(orient="records"), targets=["y_num"]
    )
    assert response.n_rows == 2
    assert set(response.targets) == {"y_num"}
    assert all("y_bin" not in row for row in response.predictions)


def test_unknown_target_raises(trained_job):
    job_id, df = trained_job
    with pytest.raises(ValidationFailedError, match="unknown target"):
        run_inference(job_id, df.iloc[:2], targets=["nope"])


def test_missing_required_column_raises(trained_job):
    job_id, df = trained_job
    broken = df.iloc[:2].drop(columns=["x1"])
    with pytest.raises(ValidationFailedError, match="x1"):
        run_inference(job_id, broken)


def test_extra_columns_are_dropped_with_warning(trained_job):
    job_id, df = trained_job
    padded = df.iloc[:2].copy()
    padded["junk_col"] = 1
    response = run_inference(job_id, padded)
    assert any("junk_col" in warning for warning in response.warnings)
    assert response.n_rows == 2


def test_unknown_job_not_found(isolated_data_dir):
    frame = pd.DataFrame({"x1": [0.1], "cat": ["A"]})
    with pytest.raises(NotFoundError, match="job_missing"):
        run_inference("job_missing", frame)


def test_unfinished_job_conflict(isolated_data_dir):
    get_job_store().create("job_pending_inf", "ds_x", {})
    frame = pd.DataFrame({"x1": [0.1], "cat": ["A"]})
    with pytest.raises(ConflictError, match="not ready"):
        run_inference("job_pending_inf", frame)


def test_missing_artifacts_not_found(isolated_data_dir):
    store = get_job_store()
    store.create("job_no_artifacts", "ds_x", {})
    mark_success(store, "job_no_artifacts")
    frame = pd.DataFrame({"x1": [0.1], "cat": ["A"]})
    with pytest.raises(NotFoundError, match="artifacts"):
        run_inference("job_no_artifacts", frame)


def test_resolve_artifacts_meta(trained_job):
    job_id, _df = trained_job
    artifacts = resolve_job_artifacts(job_id)
    assert artifacts.feature_columns == ["x1", "cat"]
    assert set(artifacts.models) == {"y_num", "y_bin"}
    assert artifacts.target_meta["y_bin"].model == "Logistic Regression"


def test_file_input_csv(trained_job):
    import io

    job_id, df = trained_job
    buffer = io.StringIO()
    df.iloc[:2][["x1", "cat"]].to_csv(buffer, index=False)
    response = inference_service.run_inference_from_file(
        job_id, "batch.csv", buffer.getvalue().encode(), targets=["y_num"]
    )
    assert response.n_rows == 2
    assert all(isinstance(row["y_num"], float) for row in response.predictions)


def test_file_input_rejects_bad_extension(trained_job):
    job_id, _df = trained_job
    with pytest.raises(UnsupportedFileError):
        inference_service.run_inference_from_file(job_id, "payload.exe", b"bytes")
