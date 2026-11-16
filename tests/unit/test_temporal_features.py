from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from automl_shared.core.exceptions import ValidationFailedError

from app.ml.pipeline import run_training
from app.ml.preprocessing.temporal import (
    add_temporal_features,
    assert_causal,
    rolling_feature,
)
from app.schemas.training import (
    CVConfig,
    HPOConfig,
    TargetIn,
    TemporalFeatureConfig,
    TrainRequest,
)

COLLECTED = "PESO (PRODUCCIÓN)"
HARVEST = "Peso_Recolectado_KG"


def _series_frame(n: int = 80, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "fecha": dates,
            "lluvia_mm": np.round(rng.uniform(0.0, 12.0, n), 4),
            "temp_media_c": np.round(rng.uniform(18.0, 32.0, n), 4),
            HARVEST: np.round(rng.uniform(0.0, 9.0, n), 4),
        }
    )


def _temporal_config(**overrides) -> TemporalFeatureConfig:
    payload = {
        "enabled": False,
        "target_columns": [HARVEST],
        "target_lags": [],
    }
    payload.update(overrides)
    return TemporalFeatureConfig(**payload)


def _request(*, temporal: TemporalFeatureConfig, cv: CVConfig, targets=None) -> TrainRequest:
    return TrainRequest(
        dataset_id="ds_temporal",
        targets=targets or [TargetIn(name=HARVEST, task="regression")],
        models=["linear_regression"],
        cv=cv,
        temporal=temporal,
        hpo=HPOConfig(enabled=False),
    )


def _timeseries_cv(**overrides) -> CVConfig:
    payload = {
        "n_folds": 3,
        "mode": "single",
        "scheme": "timeseries",
        "gap": 2,
        "date_column": "fecha",
        "random_state": 42,
    }
    payload.update(overrides)
    return CVConfig(**payload)


def test_lag_features_use_row_shift():
    frame = _series_frame(10)
    config = TemporalFeatureConfig(
        enabled=True,
        exogenous_columns=["lluvia_mm"],
        lags=[1, 3],
        rolling_windows=[],
    )

    out = add_temporal_features(frame, config)

    assert pd.isna(out["lluvia_mm_lag_1"].iloc[0])
    assert out["lluvia_mm_lag_1"].iloc[1] == pytest.approx(frame["lluvia_mm"].iloc[0])
    assert out["lluvia_mm_lag_3"].iloc[3] == pytest.approx(frame["lluvia_mm"].iloc[0])
    assert out["lluvia_mm_lag_3"].iloc[3:].notna().all()


def test_rolling_window_include_current_day_flag():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    current = rolling_feature(series, 2, "sum")
    operational = rolling_feature(series, 2, "sum", include_current_day=False)

    assert current.tolist() == [1.0, 3.0, 5.0, 7.0]
    assert pd.isna(operational.iloc[0])
    assert operational.iloc[1] == pytest.approx(1.0)
    assert operational.iloc[2] == pytest.approx(3.0)
    assert operational.iloc[3] == pytest.approx(5.0)


def test_rolling_features_only_use_past_and_current_rows():
    frame = _series_frame(30)
    config = TemporalFeatureConfig(
        enabled=True,
        exogenous_columns=["lluvia_mm"],
        lags=[1, 7],
        rolling_windows=[7, 14],
        rolling_aggs=["sum", "mean", "max", "min"],
        include_current_day=False,
    )

    out = add_temporal_features(frame, config)

    expected = rolling_feature(
        frame["lluvia_mm"].reset_index(drop=True), 14, "mean", include_current_day=False
    )
    assert np.allclose(
        out["lluvia_mm_mean_14d"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_assert_causal_accepts_causal_lags_rolling_and_target_lags():
    frame = _series_frame(90)
    config = TemporalFeatureConfig(
        enabled=True,
        date_column="fecha",
        exogenous_columns=["lluvia_mm", "temp_media_c"],
        lags=[1, 3, 7, 14, 30],
        rolling_windows=[7, 14, 30],
        rolling_aggs=["sum", "mean", "max", "min"],
        target_columns=[HARVEST],
        target_lags=[1, 2, 3, 7],
    )

    assert_causal(frame, lambda data: add_temporal_features(data, config))


def test_assert_causal_detects_forward_looking_features():
    frame = _series_frame(20)

    def leaky_builder(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["tomorrow_rain"] = data["lluvia_mm"].shift(-1)
        return out

    with pytest.raises(AssertionError, match="future rows changed"):
        assert_causal(frame, leaky_builder)


def test_target_lags_require_timeseries_scheme():
    frame = _series_frame(60)
    request = _request(
        temporal=_temporal_config(target_lags=[1, 2]),
        cv=CVConfig(n_folds=3, mode="single", scheme="kfold"),
    )

    with pytest.raises(ValidationFailedError, match="scheme='timeseries'"):
        run_training(request, "ds_temporal", "temporal.csv", frame, "job_kfold_lags")


def test_target_lags_require_gap_at_least_max_lag():
    frame = _series_frame(60)
    request = _request(
        temporal=_temporal_config(target_lags=[1, 3, 7]),
        cv=_timeseries_cv(gap=3),
    )

    with pytest.raises(ValidationFailedError, match="cv.gap >= 7"):
        run_training(request, "ds_temporal", "temporal.csv", frame, "job_small_gap")


def test_temporal_features_require_a_date_column():
    frame = _series_frame(40).drop(columns="fecha")
    request = _request(
        temporal=_temporal_config(enabled=True, exogenous_columns=["lluvia_mm"], lags=[1]),
        cv=CVConfig(n_folds=3, mode="single"),
    )

    with pytest.raises(ValidationFailedError, match="date column"):
        run_training(request, "ds_temporal", "temporal.csv", frame, "job_no_date")


def test_pipeline_generates_exogenous_lags_and_rolling_windows():
    frame = _series_frame(80)
    request = _request(
        temporal=_temporal_config(
            enabled=True,
            exogenous_columns=["lluvia_mm"],
            lags=[1, 7],
            rolling_windows=[7],
            rolling_aggs=["sum"],
            include_current_day=False,
        ),
        cv=_timeseries_cv(),
    )

    output = run_training(request, "ds_temporal", "temporal.csv", frame, "job_exogenous")

    features = output.summary.feature_columns
    assert "lluvia_mm_lag_1" in features
    assert "lluvia_mm_lag_7" in features
    assert "lluvia_mm_sum_7d" in features
    assert output.summary.temporal["include_current_day"] is False
    assert set(output.summary.temporal["generated_columns"]) >= {
        "lluvia_mm_lag_1",
        "lluvia_mm_lag_7",
        "lluvia_mm_sum_7d",
    }


def test_pipeline_auto_selects_numeric_exogenous_columns():
    frame = _series_frame(60)
    request = _request(
        temporal=_temporal_config(enabled=True, lags=[1], rolling_windows=[]),
        cv=_timeseries_cv(),
    )

    output = run_training(request, "ds_temporal", "temporal.csv", frame, "job_auto_exogenous")

    features = set(output.summary.feature_columns)
    assert "lluvia_mm_lag_1" in features
    assert "temp_media_c_lag_1" in features
    assert f"{HARVEST}_lag_1" not in features


def test_target_lags_support_dynamic_mango_target_names():
    frame = _series_frame(90)
    frame[COLLECTED] = np.round(np.clip(frame[HARVEST] * 0.8 + 1.0, 0.0, None), 4)
    request = _request(
        temporal=_temporal_config(
            target_columns=[COLLECTED, HARVEST],
            target_lags=[1, 2],
        ),
        cv=_timeseries_cv(date_column="fecha"),
        targets=[
            TargetIn(name=COLLECTED, task="regression"),
            TargetIn(name=HARVEST, task="regression"),
        ],
    )

    output = run_training(request, "ds_temporal", "mango.csv", frame, "job_dynamic_targets")

    features = set(output.summary.feature_columns)
    assert f"{COLLECTED}_lag_1" in features
    assert f"{COLLECTED}_lag_2" in features
    assert f"{HARVEST}_lag_1" in features
    assert f"{HARVEST}_lag_2" in features
    assert output.summary.temporal["target_columns"] == [COLLECTED, HARVEST]
    assert output.summary.temporal["target_lags"] == [1, 2]
    assert {target.target for target in output.summary.targets} == {COLLECTED, HARVEST}
    assert len(output.predictions_frame) == len(frame)


def test_target_columns_default_to_request_targets():
    frame = _series_frame(70)
    request = _request(
        temporal=_temporal_config(target_columns=[], target_lags=[1, 3]),
        cv=_timeseries_cv(gap=3),
    )

    output = run_training(request, "ds_temporal", "default_targets.csv", frame, "job_default_targets")

    assert output.summary.temporal["target_columns"] == [HARVEST]
    assert f"{HARVEST}_lag_3" in output.summary.feature_columns


def test_kfold_with_exogenous_temporal_features_warns():
    frame = _series_frame(60)
    request = _request(
        temporal=_temporal_config(
            enabled=True,
            exogenous_columns=["lluvia_mm"],
            lags=[1],
            rolling_windows=[],
        ),
        cv=CVConfig(n_folds=3, mode="single", scheme="kfold"),
    )

    output = run_training(request, "ds_temporal", "temporal.csv", frame, "job_kfold_warning")

    assert any("cv.scheme='kfold'" in warning for warning in output.summary.warnings)


def test_mango_climate_windows_reuse_the_shared_rolling_helper():
    from ingestion.domain.mango.features import add_climate_features

    frame = _series_frame(25)
    frame = frame.rename(columns={"lluvia_mm": "precipitacion_mm", "temp_media_c": "temp_media_c"})
    frame["temp_max_c"] = frame["temp_media_c"] + 3.0
    frame["temp_min_c"] = frame["temp_media_c"] - 3.0
    frame["humedad_relativa_pct"] = np.linspace(50.0, 90.0, len(frame))

    out = add_climate_features(frame)

    expected = rolling_feature(frame["precipitacion_mm"], 7, "sum")
    assert np.allclose(
        out["precipitacion_7d_mm"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
    )
