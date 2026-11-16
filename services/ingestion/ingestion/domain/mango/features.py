"""Derived climate and calendar features for the mango canonical dataset.

Only same-day or strictly past windows are used, so no target-derived value
leaks into the feature matrix. Rows must be sorted by ``fecha`` with no gaps;
the canonical builder guarantees that before calling these helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError

GDD_BASE_C = 10.0
RAIN_THRESHOLD_MM = 1.0
HEAT_STRESS_C = 35.0
WINDOW_SHORT = 7
WINDOW_LONG = 14
DAYS_PER_YEAR = 365.25

CLIMATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "temp_rango_c",
    "grados_dia_crecimiento",
    "precipitacion_7d_mm",
    "precipitacion_14d_mm",
    "temp_max_7d_c",
    "humedad_media_7d_pct",
    "dias_sin_lluvia",
    "estres_calor_7d",
)

CALENDAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "anio",
    "mes",
    "dia_del_ano",
    "dia_semana",
    "sin_doy",
    "cos_doy",
)

_REQUIRED_CLIMATE = (
    "temp_media_c",
    "temp_max_c",
    "temp_min_c",
    "precipitacion_mm",
    "humedad_relativa_pct",
)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add year/month/day-of-year/sinusoidal seasonality columns."""
    out = frame.copy()
    dates = pd.to_datetime(out["fecha"])
    day_of_year = dates.dt.dayofyear
    angle = 2.0 * np.pi * day_of_year / DAYS_PER_YEAR

    out["anio"] = dates.dt.year.astype("int64")
    out["mes"] = dates.dt.month.astype("int64")
    out["dia_del_ano"] = day_of_year.astype("int64")
    out["dia_semana"] = dates.dt.dayofweek.astype("int64")
    out["sin_doy"] = np.sin(angle)
    out["cos_doy"] = np.cos(angle)
    return out


def add_climate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add growing-degree days, rainfall windows and heat/dry-spell counters."""
    missing = [column for column in _REQUIRED_CLIMATE if column not in frame.columns]
    if missing:
        raise ValidationFailedError(f"climate features need columns: {missing}")

    out = frame.copy()
    out["temp_rango_c"] = out["temp_max_c"] - out["temp_min_c"]
    out["grados_dia_crecimiento"] = (out["temp_media_c"] - GDD_BASE_C).clip(lower=0.0)

    rain = out["precipitacion_mm"]
    out["precipitacion_7d_mm"] = rain.rolling(WINDOW_SHORT, min_periods=1).sum()
    out["precipitacion_14d_mm"] = rain.rolling(WINDOW_LONG, min_periods=1).sum()
    out["temp_max_7d_c"] = out["temp_max_c"].rolling(WINDOW_SHORT, min_periods=1).max()
    out["humedad_media_7d_pct"] = out["humedad_relativa_pct"].rolling(WINDOW_SHORT, min_periods=1).mean()

    is_dry = (rain < RAIN_THRESHOLD_MM).fillna(False)
    dry_groups = (~is_dry).cumsum()
    out["dias_sin_lluvia"] = is_dry.groupby(dry_groups).cumsum().astype(float)

    is_hot = (out["temp_max_c"] >= HEAT_STRESS_C).astype(float).where(out["temp_max_c"].notna())
    out["estres_calor_7d"] = is_hot.rolling(WINDOW_SHORT, min_periods=1).sum()

    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply every derived-feature helper in order."""
    return add_calendar_features(add_climate_features(frame))
