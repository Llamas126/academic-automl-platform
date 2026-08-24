from __future__ import annotations

import logging
import math
from datetime import UTC

import numpy as np
import pandas as pd

from automl_shared.schemas.dataset import ColumnProfile, ColumnRole, DatasetProfile, TargetSuggestion

logger = logging.getLogger(__name__)

BOOLEAN_TOKENS = {"true", "false", "yes", "no", "si", "sí", "y", "n", "t", "f"}
DATETIME_PARSE_SAMPLE = 200


def _parse_datetime(series: pd.Series, sample: pd.Series) -> pd.Series | None:
    try:
        parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            parsed_sample = pd.to_datetime(sample, errors="coerce")
        except (ValueError, TypeError):
            return None
    if len(sample) == 0 or parsed_sample.notna().mean() < 0.9:
        return None
    try:
        converted = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        converted = pd.to_datetime(series, errors="coerce")
    if converted.notna().mean() < 0.9:
        return None
    return converted


def _numeric_stats(converted: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(converted, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
    }


def _categorical_stats(series: pd.Series) -> dict[str, str | None]:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return {}
    top = counts.index[0]
    if isinstance(top, float) and math.isnan(top):
        top = ""
    return {"top_value": str(top), "top_freq": f"{counts.iloc[0]}"}


def detect_column_role(series: pd.Series) -> tuple[ColumnRole, pd.Series, bool]:
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return ColumnRole.text_free, series, True

    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.boolean, series.astype(bool), False

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.datetime, series, True

    if pd.api.types.is_numeric_dtype(series):
        return ColumnRole.numeric, series, False

    sample = series.dropna().astype(str).head(DATETIME_PARSE_SAMPLE)
    normalized = set(sample.str.strip().str.lower())

    if normalized and normalized.issubset(BOOLEAN_TOKENS):
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                    "si": True,
                    "sí": True,
                    "y": True,
                    "n": False,
                    "t": True,
                    "f": False,
                }
            )
        )
        if mapped.notna().all():
            return ColumnRole.boolean, mapped.astype("boolean"), False

    converted_dt = _parse_datetime(series, sample)
    if converted_dt is not None:
        return ColumnRole.datetime, converted_dt, True

    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().mean() >= 0.9:
        return ColumnRole.numeric, coerced, False

    n_rows = len(series)
    n_unique = int(series.nunique(dropna=True))
    as_str = series.dropna().astype(str)
    looks_atomic = (~as_str.str.contains(r"\s", regex=True)).mean() >= 0.95
    unique_ratio = n_unique / max(non_null_count, 1)
    if unique_ratio >= 0.95 and n_rows >= 20 and looks_atomic and as_str.str.len().mean() <= 64:
        return ColumnRole.identifier, series, True

    threshold = max(50, 0.05 * n_rows)
    if n_unique <= threshold:
        return ColumnRole.categorical, series.astype(str), False

    return ColumnRole.text_free, series, True


def profile_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> DatasetProfile:
    columns: list[ColumnProfile] = []
    roles: dict[str, tuple[ColumnRole, bool]] = {}

    for name in df.columns:
        series = df[name]
        role, converted, excluded = detect_column_role(series)
        roles[str(name)] = (role, excluded)

        missing_pct = round(float(series.isna().mean()) * 100, 2)
        if role == ColumnRole.numeric:
            stats = _numeric_stats(converted)
        elif role in (ColumnRole.categorical, ColumnRole.boolean):
            stats = _categorical_stats(converted)
        elif role == ColumnRole.datetime:
            stats = {
                "min": str(converted.min()),
                "max": str(converted.max()),
            }
        else:
            stats = {}

        columns.append(
            ColumnProfile(
                name=str(name),
                role=role,
                pandas_dtype=str(converted.dtype),
                missing_pct=missing_pct,
                n_unique=int(series.nunique(dropna=True)),
                stats=stats,
                excluded_from_features=excluded,
            )
        )

    suggestions = suggest_targets(df, roles)
    from datetime import datetime

    return DatasetProfile(
        dataset_id=dataset_id,
        filename=filename,
        created_at=datetime.now(UTC),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=columns,
        target_suggestions=suggestions,
    )


def suggest_targets(df: pd.DataFrame, roles: dict[str, tuple[ColumnRole, bool]]) -> list[TargetSuggestion]:
    suggestions: list[TargetSuggestion] = []
    for name, (role, excluded) in roles.items():
        if excluded or role not in (ColumnRole.numeric, ColumnRole.categorical, ColumnRole.boolean):
            continue
        series = df[name].dropna()
        n_unique = int(series.nunique())
        if n_unique < 2:
            continue
        if role == ColumnRole.numeric:
            if n_unique > 20 and n_unique > 0.05 * len(series):
                suggestions.append(
                    TargetSuggestion(
                        name=name,
                        suggested_task="regression",
                        rationale=f"continuous numeric variable with {n_unique} distinct values",
                    )
                )
            elif 2 <= n_unique <= 10:
                suggestions.append(
                    TargetSuggestion(
                        name=name,
                        suggested_task="classification",
                        rationale=f"numeric-coded label with {n_unique} classes",
                    )
                )
        else:
            if n_unique <= 20:
                suggestions.append(
                    TargetSuggestion(
                        name=name,
                        suggested_task="classification",
                        rationale=f"categorical variable with {n_unique} classes",
                    )
                )
    return suggestions


def feature_columns(profile: DatasetProfile) -> list[str]:
    return [c.name for c in profile.columns if not c.excluded_from_features]


def numeric_feature_names(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    names: list[str] = []
    for name in df.columns:
        if name in exclude:
            continue
        role, converted, excluded = detect_column_role(df[name])
        if not excluded and role == ColumnRole.numeric:
            names.append(name)
    return names


def categorical_feature_names(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    names: list[str] = []
    for name in df.columns:
        if name in exclude:
            continue
        role, converted, excluded = detect_column_role(df[name])
        if not excluded and role in (ColumnRole.categorical, ColumnRole.boolean):
            names.append(name)
    return names


def coerce_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in out.columns:
        _, converted, _excluded = detect_column_role(out[name])
        out[name] = converted
    return out


def infer_target_task(series: pd.Series) -> tuple[str, str]:
    role, converted, excluded = detect_column_role(series)
    n_unique = int(converted.dropna().nunique())
    if excluded:
        return "", f"column '{series.name}' cannot be used as a target"
    if role == ColumnRole.numeric:
        if n_unique > 20 or n_unique > 0.05 * len(converted.dropna()):
            return "regression", ""
        return "classification", ""
    if role in (ColumnRole.categorical, ColumnRole.boolean):
        if 2 <= n_unique <= 20:
            return "classification", ""
        if n_unique < 2:
            return "", "target has a single class"
        return "", f"too many classes ({n_unique}) for classification"
    return "", f"column of type '{role.value}' cannot be used as a target"


def assert_finite_targets(y: pd.Series, task: str) -> None:
    if task != "regression":
        return
    arr = pd.to_numeric(y.dropna(), errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"target '{y.name}' contains non-numeric values")
    if not np.isfinite(arr).all():
        raise ValueError(f"target '{y.name}' contains infinite values")
