from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from automl_shared.contract import TEMPORAL_CONTEXT_NAME
from automl_shared.schemas.training import (
    CVConfig,
    HPOConfig,
    TargetIn,
    TemporalFeatureConfig,
    TrainRequest,
)
from automl_shared.storage.local_storage import storage
from automl_shared.temporal import add_temporal_features

from app.ml.contract import persist_training_output
from app.ml.pipeline import run_training
from app.services.artifact_service import save_final_models, write_model_manifest
from app.services.inference_service import resolve_job_artifacts, run_inference
from app.services.job_service import get_job_store, mark_success


class _LagPlusOneRegressor:
    """Minimal estimator that exposes the recursive feedback loop explicitly."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X["y_lag_1"], dtype=float) + 1.0


def _temporal_frame(n: int = 90, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rain = np.round(rng.uniform(0.0, 10.0, n), 4)
    target = np.round(
        3.0 + 0.5 * rain + 0.2 * np.sin(np.arange(n) / 5.0) + rng.normal(0.0, 0.05, n),
        4,
    )
    return pd.DataFrame({"fecha": dates, "lluvia_mm": rain, "y": target})


def _timeseries_request(
    temporal: TemporalFeatureConfig, *, gap: int = 0, targets: list[TargetIn] | None = None
) -> TrainRequest:
    return TrainRequest(
        dataset_id="ds_temporal",
        targets=targets or [TargetIn(name="y", task="regression")],
        models=["linear_regression"],
        cv=CVConfig(
            n_folds=3,
            mode="single",
            scheme="timeseries",
            gap=gap,
            date_column="fecha",
        ),
        temporal=temporal,
        hpo=HPOConfig(enabled=False),
    )


def _persist_trained_job(
    isolated_data_dir, frame: pd.DataFrame, request: TrainRequest, job_id: str
):
    output = run_training(request, "ds_temporal", "temporal.csv", frame, job_id)
    persist_training_output(
        output,
        storage.job_work_dir(job_id),
        storage.job_out_dir(job_id),
    )
    store = get_job_store()
    store.create(job_id, "ds_temporal", {})
    mark_success(store, job_id)
    return output


def _create_manual_recursive_job(job_id: str = "job_recursive") -> None:
    out_dir = storage.job_out_dir(job_id)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True)
    artifacts = {
        "y": SimpleNamespace(
            final_pipeline=_LagPlusOneRegressor(),
            task="regression",
            best_entry_key="lag_plus_one",
            best_params={},
            label_classes=None,
        )
    }
    save_final_models(artifacts, models_dir)
    write_model_manifest(models_dir)

    temporal = {
        "enabled": False,
        "date_column": "fecha",
        "exogenous_columns": [],
        "target_columns": ["y"],
        "lags": [],
        "rolling_windows": [],
        "rolling_aggs": [],
        "include_current_day": True,
        "target_lags": [1],
        "drop_zero_targets": False,
        "max_lookback": 1,
        "generated_columns": ["y_lag_1"],
    }
    summary = {
        "feature_columns": ["y_lag_1"],
        "temporal": temporal,
        "targets": [
            {
                "target": "y",
                "task": "regression",
                "best_model": "LagPlusOne",
                "label_encoder_classes": None,
            }
        ],
    }
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "results_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    context = {
        "schema_version": 1,
        "config": {
            key: value
            for key, value in temporal.items()
            if key not in {"max_lookback", "generated_columns"}
        },
        "date_column": "fecha",
        "max_lookback": 1,
        "history": [{"fecha": "2024-01-03", "y": 30.0}],
    }
    (models_dir / TEMPORAL_CONTEXT_NAME).write_text(json.dumps(context), encoding="utf-8")

    store = get_job_store()
    store.create(job_id, "ds_temporal", {})
    mark_success(store, job_id)


def test_raw_rows_with_saved_history_match_precomputed_features(isolated_data_dir):
    frame = _temporal_frame()
    train_frame = frame.iloc[:80].reset_index(drop=True)
    config = TemporalFeatureConfig(
        enabled=True,
        exogenous_columns=["lluvia_mm"],
        lags=[1, 2],
        rolling_windows=[3],
        rolling_aggs=["sum"],
        include_current_day=False,
    )
    _persist_trained_job(
        isolated_data_dir,
        train_frame,
        _timeseries_request(config),
        "job_temporal_equiv",
    )

    artifacts = resolve_job_artifacts("job_temporal_equiv")
    assert artifacts.temporal is not None
    assert artifacts.temporal.max_lookback == 3
    assert len(artifacts.temporal.history) == 3

    future_raw = frame.iloc[80:83][["fecha", "lluvia_mm"]].reset_index(drop=True)
    reconstructed = add_temporal_features(frame.copy(), artifacts.temporal.config)
    precomputed = reconstructed.iloc[80:83][artifacts.feature_columns].reset_index(drop=True)
    response_precomputed = run_inference("job_temporal_equiv", precomputed)

    response_raw = run_inference("job_temporal_equiv", future_raw)
    assert response_precomputed.n_rows == response_raw.n_rows == 3
    expected = [row["y"] for row in response_precomputed.predictions]
    assert [row["y"] for row in response_raw.predictions] == pytest.approx(expected)

    explicit_history = train_frame.tail(3).reset_index(drop=True)
    response_explicit = run_inference(
        "job_temporal_equiv", future_raw, history=explicit_history
    )
    assert [row["y"] for row in response_explicit.predictions] == pytest.approx(expected)


def test_recursive_inference_feeds_predictions_back_into_target_lags(isolated_data_dir):
    _create_manual_recursive_job()
    records = [
        {"fecha": "2024-01-04"},
        {"fecha": "2024-01-05"},
        {"fecha": "2024-01-06"},
    ]

    response = run_inference(
        "job_recursive",
        pd.DataFrame.from_records(records),
        horizon=3,
        mode="recursive",
    )

    assert response.n_rows == 3
    assert [row["y"] for row in response.predictions] == pytest.approx([31.0, 32.0, 33.0])


def test_predict_api_accepts_recursive_temporal_payload(client):
    _create_manual_recursive_job("job_recursive_api")
    response = client.post(
        "/api/v1/predict/job_recursive_api",
        json={
            "records": [
                {"fecha": "2024-01-04"},
                {"fecha": "2024-01-05"},
                {"fecha": "2024-01-06"},
            ],
            "horizon": 3,
            "mode": "recursive",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["y"] for row in body["predictions"]] == pytest.approx([31.0, 32.0, 33.0])


def test_drop_zero_targets_keeps_daily_lag_continuity(isolated_data_dir):
    frame = _temporal_frame(n=60)
    frame.loc[[10, 11, 20], "y"] = 0.0
    config = TemporalFeatureConfig(
        enabled=False,
        target_columns=["y"],
        target_lags=[1],
        drop_zero_targets=True,
    )
    output = _persist_trained_job(
        isolated_data_dir,
        frame,
        _timeseries_request(config, gap=1),
        "job_drop_zero",
    )

    assert output.summary.temporal["drop_zero_targets"] is True
    assert output.summary.temporal["dropped_zero_rows"] == 3
    assert output.summary.n_samples == len(frame) - 3

    predictions = output.predictions_frame
    assert predictions is not None
    dates = pd.to_datetime(predictions["fecha"])
    assert frame["fecha"].iloc[10] not in set(dates)

    date_12 = frame["fecha"].iloc[12]
    row_12 = predictions.loc[dates == date_12].iloc[0]
    assert row_12["y_lag_1"] == pytest.approx(0.0)

    date_9 = frame["fecha"].iloc[9]
    row_9 = predictions.loc[dates == date_9].iloc[0]
    assert row_9["y_lag_1"] == pytest.approx(frame["y"].iloc[8])
