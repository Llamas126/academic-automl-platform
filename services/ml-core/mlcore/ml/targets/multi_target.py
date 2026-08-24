from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from automl_shared.core.exceptions import ValidationFailedError
from automl_shared.profiling.type_detector import (
    assert_finite_targets,
    detect_column_role,
    infer_target_task,
)
from automl_shared.schemas.dataset import ColumnRole
from automl_shared.schemas.training import TargetIn


@dataclass
class TargetSpec:
    name: str
    task: str
    y_raw: pd.Series = field(repr=False, default=None)
    y_encoded: pd.Series = field(repr=False, default=None)
    label_encoder_classes: list[str] | None = None


@dataclass
class RunContext:
    X: pd.DataFrame
    specs: list[TargetSpec]
    numeric_cols: list[str]
    categorical_cols: list[str]
    stratification_labels: np.ndarray | None
    warnings: list[str]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def resolve_targets(df: pd.DataFrame, requests: list[TargetIn]) -> RunContext:
    warnings: list[str] = []
    target_names: list[str] = []
    for request in requests:
        if request.name not in df.columns:
            raise ValidationFailedError(f"target column '{request.name}' not found in dataset")
        if request.name in target_names:
            raise ValidationFailedError(f"duplicate target '{request.name}'")
        target_names.append(request.name)

    working = df.copy()
    for col in working.columns:
        _, converted, _excluded = detect_column_role(working[col])
        working[col] = converted

    complete_mask = working[target_names].notna().all(axis=1).to_numpy()
    dropped_rows = int((~complete_mask).sum())
    if dropped_rows:
        warnings.append(f"{dropped_rows} rows dropped due to missing values in selected targets")
        working = working.loc[complete_mask].reset_index(drop=True)

    specs: list[TargetSpec] = []
    for request in requests:
        series = working[request.name]
        task = request.task
        error: str | None = None
        if task == "auto":
            task, err = infer_target_task(series)
            error = err or None
        elif task == "regression":
            try:
                assert_finite_targets(series, "regression")
                series = pd.to_numeric(series, errors="raise")
            except ValueError as exc:
                error = str(exc)
        else:
            n_unique = int(series.dropna().nunique())
            if n_unique < 2:
                error = f"target '{request.name}' has a single class"
            elif n_unique > 50:
                error = f"target '{request.name}' has too many classes ({n_unique})"
        if error:
            raise ValidationFailedError(f"invalid target '{request.name}': {error}")

        spec = TargetSpec(name=request.name, task=task)
        if task == "classification":
            encoder = LabelEncoder()
            encoded = pd.Series(encoder.fit_transform(series.astype(str)), index=series.index)
            spec.y_raw = series.astype(str)
            spec.y_encoded = encoded
            spec.label_encoder_classes = [str(c) for c in encoder.classes_]
            if len(encoder.classes_) < 2:
                raise ValidationFailedError(
                    f"target '{request.name}' requires at least two classes after row filtering"
                )
        else:
            spec.y_raw = pd.to_numeric(series, errors="coerce")
            spec.y_encoded = spec.y_raw

        specs.append(spec)

    excluded_columns: set[str] = set(target_names)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in working.columns:
        if col in excluded_columns:
            continue
        role, converted, is_excluded = detect_column_role(working[col])
        working[col] = converted
        if is_excluded or role in (ColumnRole.datetime, ColumnRole.identifier, ColumnRole.text_free):
            if role in (ColumnRole.datetime, ColumnRole.text_free) and not is_excluded:
                warnings.append(f"column '{col}' ({role.value}) excluded from features")
            continue
        if role == ColumnRole.numeric:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols:
        raise ValidationFailedError("no usable feature columns remain after excluding targets")

    from mlcore.ml.preprocessing.builders import prepare_feature_frame
    from mlcore.ml.validation.kfold import build_combined_stratification

    stratification = build_combined_stratification(specs)
    raw_X = working[numeric_cols + categorical_cols].copy()
    X = prepare_feature_frame(raw_X, numeric_cols, categorical_cols)

    return RunContext(
        X=X,
        specs=specs,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        stratification_labels=stratification,
        warnings=warnings,
    )
