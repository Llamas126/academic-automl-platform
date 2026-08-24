import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.core.exceptions import ExportError, NotFoundError
from app.services.artifact_service import (
    MANIFEST_NAME,
    model_digest,
    save_final_models,
    verify_model_integrity,
    write_model_manifest,
)
from app.services.inference_service import run_inference
from app.services.job_service import get_job_store, mark_success


@pytest.fixture
def artifact_job(isolated_data_dir):
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({"x1": rng.normal(0, 1, n)})
    df["y"] = 2.0 * df["x1"] + rng.normal(0, 0.1, n)

    pipe = Pipeline([("estimator", LinearRegression())])
    pipe.fit(df[["x1"]], df["y"])

    artifacts = {
        "y": SimpleNamespace(
            final_pipeline=pipe,
            task="regression",
            best_entry_key="linear_regression",
            best_params={},
            label_classes=None,
        )
    }
    job_id = "job_integrity_test"
    models_dir = isolated_data_dir / "jobs" / job_id / "output" / "models"
    models_dir.mkdir(parents=True)
    save_final_models(artifacts, models_dir)

    out_dir = isolated_data_dir / "jobs" / job_id / "output"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    summary = {
        "feature_columns": ["x1"],
        "targets": [
            {
                "target": "y",
                "task": "regression",
                "best_model": "Linear Regression",
                "label_encoder_classes": None,
            }
        ],
    }
    (metrics_dir / "results_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    store = get_job_store()
    store.create(job_id, "ds_src", {})
    mark_success(store, job_id)
    return job_id, models_dir


def test_manifest_roundtrip(artifact_job):
    _job_id, models_dir = artifact_job
    manifest = write_model_manifest(models_dir)
    assert manifest.name == MANIFEST_NAME
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split("  ")
    assert name == "y_best_model.joblib"
    assert digest == model_digest(models_dir / name)
    verify_model_integrity(models_dir)


def test_tampered_artifact_fails_verification(artifact_job):
    _job_id, models_dir = artifact_job
    write_model_manifest(models_dir)
    target = models_dir / "y_best_model.joblib"
    target.write_bytes(target.read_bytes() + b"\x00tamper")

    with pytest.raises(ExportError, match="integrity verification failed"):
        verify_model_integrity(models_dir)


def test_tampered_artifact_blocks_inference(artifact_job):
    job_id, models_dir = artifact_job
    write_model_manifest(models_dir)
    target = models_dir / "y_best_model.joblib"
    target.write_bytes(b"not a pickle at all")

    with pytest.raises(ExportError):
        run_inference(job_id, pd.DataFrame({"x1": [0.5]}))


def test_if_present_allows_legacy_artifacts(artifact_job):
    _job_id, models_dir = artifact_job
    assert not (models_dir / MANIFEST_NAME).exists()
    verify_model_integrity(models_dir)


def test_strict_mode_requires_manifest(artifact_job, monkeypatch):
    _job_id, models_dir = artifact_job
    monkeypatch.setenv("ARTIFACT_INTEGRITY", "strict")
    get_settings.cache_clear()
    try:
        with pytest.raises(ExportError, match="manifest is missing"):
            verify_model_integrity(models_dir)
    finally:
        get_settings.cache_clear()


def test_untracked_artifact_rejected(artifact_job):
    _job_id, models_dir = artifact_job
    write_model_manifest(models_dir)
    import joblib

    joblib.dump({"rogue": True}, models_dir / "rogue_best_model.joblib")

    with pytest.raises(ExportError, match="not covered by integrity manifest"):
        verify_model_integrity(models_dir)


def test_hmac_signing_roundtrip(artifact_job, monkeypatch):
    _job_id, models_dir = artifact_job
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", "s3cret-key")
    get_settings.cache_clear()
    try:
        write_model_manifest(models_dir)
        verify_model_integrity(models_dir)
        signed_digest = (models_dir / MANIFEST_NAME).read_text().strip().split("  ")[0]
        assert signed_digest != model_digest(models_dir / "y_best_model.joblib")

        target = models_dir / "y_best_model.joblib"
        target.write_bytes(b"corrupted")
        with pytest.raises(ExportError, match="integrity verification failed"):
            verify_model_integrity(models_dir)
    finally:
        get_settings.cache_clear()


def test_off_mode_skips_verification(artifact_job, monkeypatch):
    _job_id, models_dir = artifact_job
    monkeypatch.setenv("ARTIFACT_INTEGRITY", "off")
    get_settings.cache_clear()
    try:
        target = models_dir / "y_best_model.joblib"
        target.write_bytes(b"corrupted")
        verify_model_integrity(models_dir)
    finally:
        get_settings.cache_clear()


def test_missing_models_dir_still_404(isolated_data_dir):
    store = get_job_store()
    store.create("job_no_models", "ds_x", {})
    mark_success(store, "job_no_models")
    from app.core.exceptions import ConflictError

    with pytest.raises((NotFoundError, ConflictError)):
        run_inference("job_no_models", pd.DataFrame({"x1": [0.5]}))
