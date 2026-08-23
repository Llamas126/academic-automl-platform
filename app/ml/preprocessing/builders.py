from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_columns(
    numeric_cols: list[str], categorical_cols: list[str], exclude: set[str]
) -> tuple[list[str], list[str]]:
    numeric = [c for c in numeric_cols if c not in exclude]
    categorical = [c for c in categorical_cols if c not in exclude]
    return numeric, categorical


def prepare_feature_frame(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        out[col] = series.astype("float64")
    for col in categorical_cols:
        values = df[col]
        if hasattr(values, "cat"):
            values = values.astype(object)
        out[col] = values.astype(object).where(values.notna(), other=np.nan).astype(object)
        out[col] = out[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return out


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer | None:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), list(numeric_cols)))
    if categorical_cols:
        cat_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.01),
            ),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), list(categorical_cols)))
    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop")
