"""Causal temporal feature engineering shared by every service.

The helpers in this module only look backwards in time. A feature produced for
row ``t`` uses information available up to day ``t`` (or up to ``t - 1`` when
``include_current_day`` is false), which makes them safe to compute once,
before cross-validation, without leaking future rows.

They live in :mod:`automl_shared` so both the ML core and the ingestion domain
can reuse the exact same implementation without importing each other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from automl_shared.core.exceptions import ValidationFailedError
from automl_shared.schemas.training import TemporalFeatureConfig

RollingAgg = Literal["sum", "mean", "max", "min"]

SUPPORTED_ROLLING_AGGS: tuple[str, ...] = ("sum", "mean", "max", "min")

__all__ = [
    "SUPPORTED_ROLLING_AGGS",
    "add_temporal_features",
    "assert_causal",
    "lag_features",
    "max_lookback",
    "rolling_feature",
    "rolling_features",
    "temporal_feature_names",
]


def max_lookback(config: TemporalFeatureConfig) -> int:
    """Maximum number of past rows needed to rebuild a feature row.

    Mirrors the configured horizons exactly: ``max(lags + rolling_windows +
    target_lags)``. The result is persisted with the run so inference can load
    just enough raw training history.
    """
    horizons = [*config.lags, *config.rolling_windows, *config.target_lags]
    return max(horizons, default=0)


def _numeric_series(frame: pd.DataFrame, column: str, role: str) -> pd.Series:
    if column not in frame.columns:
        raise ValidationFailedError(f"temporal {role} column '{column}' not found in dataset")
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    converted = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and not converted.notna().any():
        raise ValidationFailedError(f"temporal {role} column '{column}' is not numeric")
    return converted.astype("float64")


def _guard_generated_name(frame: pd.DataFrame, name: str) -> None:
    if name in frame.columns:
        raise ValidationFailedError(
            f"temporal feature '{name}' already exists; rename the source column or drop the existing one"
        )


def rolling_feature(
    series: pd.Series,
    window: int,
    agg: str = "mean",
    *,
    include_current_day: bool = True,
) -> pd.Series:
    """Causal rolling aggregate for a single series.

    ``include_current_day=True`` aggregates ``t - window + 1 .. t``.
    ``include_current_day=False`` shifts one row first, so the window becomes
    ``t - window .. t - 1`` and can be used operationally before the current
    day's exogenous value is known.
    """
    if window < 1:
        raise ValidationFailedError(f"rolling window must be >= 1, got {window}")
    if agg not in SUPPORTED_ROLLING_AGGS:
        raise ValidationFailedError(
            f"unsupported rolling aggregation '{agg}'; use one of {list(SUPPORTED_ROLLING_AGGS)}"
        )
    values = series if include_current_day else series.shift(1)
    return values.rolling(window=window, min_periods=1).agg(agg)


def lag_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    *,
    role: str = "exogenous",
) -> pd.DataFrame:
    """Append ``column_lag_k`` columns built with ``series.shift(k)``.

    Rows must already be in chronological order; the shift is row-based, which
    is what ``shift(k)`` means for a gapless daily series.
    """
    out = frame.copy()
    for column in columns:
        values = _numeric_series(out, column, role)
        for lag in lags:
            if lag < 1:
                raise ValidationFailedError(f"lag must be >= 1, got {lag}")
            name = f"{column}_lag_{lag}"
            _guard_generated_name(out, name)
            out[name] = values.shift(lag)
    return out


def rolling_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
    aggs: Sequence[str],
    *,
    include_current_day: bool = True,
    role: str = "exogenous",
) -> pd.DataFrame:
    """Append causal rolling aggregates for every ``column × window × agg``."""
    out = frame.copy()
    for column in columns:
        values = _numeric_series(out, column, role)
        for window in windows:
            for agg in aggs:
                name = f"{column}_{agg}_{window}d"
                _guard_generated_name(out, name)
                out[name] = rolling_feature(
                    values, window, agg, include_current_day=include_current_day
                )
    return out


def temporal_feature_names(
    columns: Iterable[str],
    lags: Sequence[int] = (),
    windows: Sequence[int] = (),
    aggs: Sequence[str] = (),
    *,
    role: str = "exogenous",
) -> list[str]:
    """Names that :func:`add_temporal_features` would generate for these lists."""
    names: list[str] = []
    for column in columns:
        for lag in lags:
            names.append(f"{column}_lag_{lag}")
        for window in windows:
            for agg in aggs:
                names.append(f"{column}_{agg}_{window}d")
    return names


def _check_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted({value for value in values if list(values).count(value) > 1})
    if duplicates:
        raise ValidationFailedError(f"duplicate {label}: {duplicates}")


def add_temporal_features(
    frame: pd.DataFrame,
    config: TemporalFeatureConfig,
) -> pd.DataFrame:
    """Apply the configured causal lags and rolling windows to ``frame``.

    * ``config.enabled`` gates exogenous lags/rolling windows.
    * ``config.target_lags`` is independent: a non-empty list always applies,
      even when ``enabled`` is false, because the caller asked for it explicitly.
    * ``config.date_column``, when present, sorts rows chronologically first.
    """
    out = frame.copy()
    if config.date_column is not None:
        if config.date_column not in out.columns:
            raise ValidationFailedError(
                f"temporal date column '{config.date_column}' not found in dataset"
            )
        out[config.date_column] = pd.to_datetime(
            out[config.date_column], errors="coerce", format="mixed"
        )
        out = out.sort_values(
            config.date_column, kind="stable", na_position="last"
        ).reset_index(drop=True)

    exogenous = list(config.exogenous_columns)
    targets = list(config.target_columns)
    _check_unique(exogenous, "exogenous columns")
    _check_unique(targets, "target columns")
    overlap = sorted(set(exogenous) & set(targets))
    if overlap:
        raise ValidationFailedError(
            f"columns cannot be both exogenous and target-lagged: {overlap}"
        )

    if config.enabled and (config.lags or config.rolling_windows):
        if not exogenous:
            raise ValidationFailedError(
                "temporal.enabled requires temporal.exogenous_columns or at least one "
                "numeric non-target column to derive them from"
            )
        if config.lags:
            out = lag_features(out, exogenous, config.lags, role="exogenous")
        if config.rolling_windows:
            out = rolling_features(
                out,
                exogenous,
                config.rolling_windows,
                config.rolling_aggs,
                include_current_day=config.include_current_day,
                role="exogenous",
            )

    if config.target_lags:
        if not targets:
            raise ValidationFailedError(
                "temporal.target_lags requires temporal.target_columns (or targets in the request)"
            )
        out = lag_features(out, targets, config.target_lags, role="target")

    return out


def assert_causal(
    frame: pd.DataFrame,
    build: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    split_fraction: float = 0.5,
    perturbation: float = 1e6,
    atol: float = 1e-9,
) -> None:
    """Assert that mutating future rows cannot change any past feature value.

    ``frame`` must be sorted chronologically and ``build`` must be a
    deterministic callable that returns the original columns plus engineered
    features. Raises :class:`AssertionError` when a past value changes.
    """
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("split_fraction must be between 0 and 1")
    n_rows = len(frame)
    if n_rows < 2:
        raise ValueError("assert_causal needs at least two rows")
    cutoff = max(1, int(n_rows * split_fraction))

    baseline = build(frame.copy())
    mutated = frame.copy()
    numeric_columns = [
        column for column in mutated.columns if pd.api.types.is_numeric_dtype(mutated[column])
    ]
    if not numeric_columns:
        raise ValueError("assert_causal needs at least one numeric column to perturb")
    future_index = mutated.index[cutoff:]
    mutated.loc[future_index, numeric_columns] = (
        mutated.loc[future_index, numeric_columns] + perturbation
    )
    changed = build(mutated)

    if list(baseline.columns) != list(changed.columns):
        raise AssertionError("future rows changed the generated feature set")

    past_baseline = baseline.iloc[:cutoff]
    past_changed = changed.iloc[:cutoff]
    for column in baseline.columns:
        left = past_baseline[column]
        right = past_changed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"future rows changed past feature '{column}'")
        elif not left.astype(object).equals(right.astype(object)):
            raise AssertionError(f"future rows changed past feature '{column}'")
