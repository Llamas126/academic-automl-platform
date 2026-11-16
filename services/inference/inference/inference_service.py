from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from automl_shared.artifacts import load_model, verify_model_integrity
from automl_shared.contract import TEMPORAL_CONTEXT_NAME
from automl_shared.core.config import get_settings
from automl_shared.core.exceptions import (
    ConflictError,
    ExportError,
    NotFoundError,
    UnsupportedFileError,
    ValidationFailedError,
)
from automl_shared.jobs.job_store import get_job_store
from automl_shared.schemas.jobs import JobState
from automl_shared.schemas.predict import PredictResponse, PredictTargetMeta
from automl_shared.schemas.training import TemporalFeatureConfig
from automl_shared.storage.local_storage import storage
from automl_shared.tabular import ALLOWED_EXTENSIONS, read_tabular
from automl_shared.temporal import add_temporal_features, max_lookback

logger = logging.getLogger(__name__)

_INFERENCE_MARKER = "__automl_inference_target_row__"
_INPUT_INDEX = "__automl_inference_input_index__"

_TEMPORAL_CONFIG_FIELDS = (
    "enabled",
    "date_column",
    "exogenous_columns",
    "target_columns",
    "lags",
    "rolling_windows",
    "rolling_aggs",
    "include_current_day",
    "target_lags",
    "drop_zero_targets",
)


@dataclass
class TemporalContext:
    """Persisted state needed to rebuild temporal features at inference time."""

    config: TemporalFeatureConfig
    date_column: str | None
    max_lookback: int
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class JobArtifacts:
    job_id: str
    feature_columns: list[str]
    models: dict[str, dict]
    target_meta: dict[str, PredictTargetMeta] = field(default_factory=dict)
    temporal: TemporalContext | None = None


def _load_temporal_context(summary: dict, models_dir: Path) -> TemporalContext | None:
    summary_temporal = summary.get("temporal") or {}
    context_payload: dict[str, Any] = {}
    context_path = models_dir / TEMPORAL_CONTEXT_NAME
    if context_path.exists():
        try:
            context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("could not read temporal context at %s", context_path)

    config_raw: dict[str, Any] = dict(context_payload.get("config") or {})
    config_raw.update(
        {
            key: value
            for key, value in summary_temporal.items()
            if key in _TEMPORAL_CONFIG_FIELDS
        }
    )
    if not config_raw:
        return None
    active = bool(config_raw.get("enabled")) or bool(config_raw.get("target_lags"))
    if not active:
        return None

    config = TemporalFeatureConfig(
        **{key: value for key, value in config_raw.items() if key in _TEMPORAL_CONFIG_FIELDS}
    )
    date_column = config.date_column or context_payload.get("date_column")
    lookback = int(
        summary_temporal.get("max_lookback")
        or context_payload.get("max_lookback")
        or max_lookback(config)
    )
    history = pd.DataFrame.from_records(context_payload.get("history") or [])
    return TemporalContext(
        config=config.model_copy(update={"date_column": date_column}),
        date_column=date_column,
        max_lookback=lookback,
        history=history,
    )


def resolve_job_artifacts(job_id: str) -> JobArtifacts:
    record = get_job_store().get(job_id)
    if record is None:
        raise NotFoundError(f"job '{job_id}' not found")
    state = record.get("state")
    if state != JobState.SUCCESS.value:
        raise ConflictError(f"job '{job_id}' is not ready for inference (state={state})")

    out_dir = storage.job_out_dir(job_id)
    models_dir = out_dir / "models"
    summary_path = out_dir / "metrics" / "results_summary.json"
    if not summary_path.exists() or not models_dir.exists():
        raise NotFoundError(f"inference artifacts for job '{job_id}' are missing")

    try:
        verify_model_integrity(models_dir)
    except ExportError:
        logger.error("artifact integrity verification failed for job %s", job_id)
        raise

    summary = storage.load_json(summary_path)
    models: dict[str, dict] = {}
    for path in sorted(models_dir.glob("*_best_model.joblib")):
        payload = load_model(path)
        target_name = str(payload.get("target") or path.name[: -len("_best_model.joblib")])
        models[target_name] = payload
    if not models:
        raise NotFoundError(f"job '{job_id}' has no serialized models")

    meta: dict[str, PredictTargetMeta] = {}
    for entry in summary.get("targets", []):
        name = entry.get("target")
        if name in models:
            meta[name] = PredictTargetMeta(
                target=str(name),
                task=str(entry.get("task") or models[name].get("task", "")),
                model=entry.get("best_model"),
                classes=entry.get("label_encoder_classes"),
            )
    return JobArtifacts(
        job_id=job_id,
        feature_columns=[str(col) for col in summary.get("feature_columns", [])],
        models=models,
        target_meta=meta,
        temporal=_load_temporal_context(summary, models_dir),
    )


def _meta_for(artifacts: JobArtifacts, name: str) -> PredictTargetMeta:
    payload = artifacts.models[name]
    label_classes = payload.get("label_classes")
    return artifacts.target_meta.get(name) or PredictTargetMeta(
        target=name,
        task=str(payload.get("task", "")),
        classes=[str(c) for c in label_classes] if label_classes else None,
    )


def _predict_single(
    artifacts: JobArtifacts, name: str, feature_row: pd.DataFrame
) -> tuple[Any, dict[str, float] | None, Any]:
    payload = artifacts.models[name]
    pipeline = payload["pipeline"]
    task = str(payload.get("task", ""))
    label_classes = payload.get("label_classes")
    raw = np.asarray(pipeline.predict(feature_row))[0]

    probabilities: dict[str, float] | None = None
    if task == "classification":
        decoded: Any = str(label_classes[int(raw)]) if label_classes else int(raw)
        if label_classes and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(feature_row)[0]
            probabilities = {
                str(label_classes[index]): float(proba[index])
                for index in range(len(label_classes))
            }
    else:
        decoded = float(raw)
    return decoded, probabilities, raw


def _predict_matrix(
    artifacts: JobArtifacts, X: pd.DataFrame, requested: list[str]
) -> tuple[list[dict[str, Any]], dict[str, PredictTargetMeta], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = [{"row_index": i} for i in range(len(X))]
    metas: dict[str, PredictTargetMeta] = {}
    raw_predictions: dict[str, np.ndarray] = {}

    for name in requested:
        payload = artifacts.models[name]
        pipeline = payload["pipeline"]
        task = str(payload.get("task", ""))
        label_classes = payload.get("label_classes")
        predictions = np.asarray(pipeline.predict(X))
        proba = pipeline.predict_proba(X) if task == "classification" else None
        raw_predictions[name] = predictions

        for i in range(len(X)):
            if task == "classification":
                code = int(predictions[i])
                rows[i][name] = str(label_classes[code]) if label_classes else code
            else:
                rows[i][name] = float(predictions[i])
        if proba is not None and label_classes:
            proba_key = f"{name}_probabilities"
            for i in range(len(X)):
                rows[i][proba_key] = {
                    str(label_classes[class_index]): float(proba[i, class_index])
                    for class_index in range(len(label_classes))
                }
        metas[name] = _meta_for(artifacts, name)

    return rows, metas, raw_predictions


def _as_history_frame(history: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame | None:
    if history is None:
        return None
    if isinstance(history, pd.DataFrame):
        return history.copy()
    return pd.DataFrame.from_records(history)


def _sort_by_date(frame: pd.DataFrame, date_column: str | None) -> pd.DataFrame:
    if not date_column:
        return frame.reset_index(drop=True)
    if date_column not in frame.columns:
        raise ValidationFailedError(
            f"temporal inference requires the date column '{date_column}' in the payload or history"
        )
    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce", format="mixed")
    return out.sort_values(date_column, kind="stable", na_position="last").reset_index(drop=True)


def _build_combined_temporal_frame(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    temporal: TemporalContext,
    warnings: list[str],
) -> tuple[pd.DataFrame, list[int], str | None]:
    config = temporal.config
    date_column = config.date_column or temporal.date_column
    target_df = target_df.copy()
    history_df = history_df.copy() if history_df is not None else pd.DataFrame()

    required_target: list[str] = []
    if date_column:
        required_target.append(date_column)
    if config.enabled and (config.lags or config.rolling_windows):
        required_target.extend(config.exogenous_columns)
    missing_target = [column for column in required_target if column not in target_df.columns]
    if missing_target:
        raise ValidationFailedError(
            "temporal inference needs raw column(s) "
            f"{missing_target}; send raw covariates and the date, or every precomputed feature column"
        )

    required_history = set(required_target)
    if config.target_lags:
        required_history.update(config.target_columns)
    if not history_df.empty:
        missing_history = sorted(
            column for column in required_history if column not in history_df.columns
        )
        if missing_history:
            raise ValidationFailedError(
                f"temporal history is missing raw column(s): {missing_history}"
            )
    elif temporal.max_lookback > 0:
        warnings.append(
            "no temporal history was supplied and the job has none saved; lag/rolling "
            "features for early rows may be imputed"
        )

    for column in config.target_columns:
        if column not in target_df.columns:
            target_df[column] = np.nan
    target_df[_INPUT_INDEX] = np.arange(len(target_df))

    history_part = (
        history_df.assign(**{_INFERENCE_MARKER: False}) if not history_df.empty else history_df
    )
    target_part = target_df.assign(**{_INFERENCE_MARKER: True})
    combined = pd.concat([history_part, target_part], ignore_index=True, sort=False)
    combined = _sort_by_date(combined, date_column)

    target_positions = [
        int(position) for position in np.flatnonzero(combined[_INFERENCE_MARKER].to_numpy())
    ]
    if not target_positions:
        raise ValidationFailedError("temporal inference could not isolate the target rows")
    if temporal.max_lookback and target_positions[0] < temporal.max_lookback:
        warnings.append(
            f"only {target_positions[0]} prior row(s) available before the first target row; "
            f"max_lookback={temporal.max_lookback}, so early lag/rolling values may be imputed"
        )
    return combined, target_positions, date_column


def _apply_temporal_features(
    combined: pd.DataFrame, temporal: TemporalContext, date_column: str | None
) -> pd.DataFrame:
    # ``combined`` is already chronological; avoid a second sort/reset that would
    # invalidate the target-row positions.
    config = temporal.config
    if date_column:
        config = config.model_copy(update={"date_column": None})
    return add_temporal_features(combined.copy(), config)


def _extract_target_features(
    features: pd.DataFrame, artifacts: JobArtifacts, target_positions: list[int]
) -> pd.DataFrame:
    missing = [column for column in artifacts.feature_columns if column not in features.columns]
    if missing:
        raise ValidationFailedError(
            "temporal inference could not rebuild feature column(s) "
            f"{missing}; include the required raw columns/history or send the precomputed features"
        )
    return features.iloc[target_positions][artifacts.feature_columns].reset_index(drop=True)


def _restore_input_order(
    rows: list[dict[str, Any]], combined: pd.DataFrame, target_positions: list[int]
) -> list[dict[str, Any]]:
    """Map chronological predictions back to the caller's original row order."""
    input_indices = (
        combined.iloc[target_positions][_INPUT_INDEX].astype(int).tolist()
    )
    ordered: list[dict[str, Any] | None] = [None] * len(rows)
    for position, input_index in enumerate(input_indices):
        row = dict(rows[position])
        row["row_index"] = int(input_index)
        ordered[input_index] = row
    return [row for row in ordered if row is not None]


def _should_recurse(
    temporal: TemporalContext,
    target_df: pd.DataFrame,
    horizon: int | None,
    mode: str,
) -> bool:
    if mode == "recursive":
        return True
    if mode == "one-step-ahead":
        return False
    config = temporal.config
    if not config.target_lags:
        return False
    steps = horizon or len(target_df)
    if steps <= 1:
        return False
    return any(
        column not in target_df.columns or target_df[column].isna().any()
        for column in config.target_columns
    )


def _target_value_for_lag(raw: Any, payload: dict) -> float | None:
    task = str(payload.get("task", ""))
    label_classes = payload.get("label_classes")
    value = raw
    if task == "classification" and label_classes:
        try:
            value = label_classes[int(raw)]
        except (IndexError, TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_recursive_inference(
    artifacts: JobArtifacts,
    temporal: TemporalContext,
    combined: pd.DataFrame,
    target_positions: list[int],
    requested: list[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, PredictTargetMeta]]:
    config = temporal.config
    feature_config = (
        config.model_copy(update={"date_column": None}) if config.date_column else config
    )
    fill_targets = list(
        dict.fromkeys([*requested, *[t for t in config.target_columns if t in artifacts.models]])
    )
    rows: list[dict[str, Any]] = [{"row_index": i} for i in range(len(target_positions))]
    metas: dict[str, PredictTargetMeta] = {}
    warned_classification = False

    for step, position in enumerate(target_positions):
        features = add_temporal_features(combined.copy(), feature_config)
        missing = [
            column for column in artifacts.feature_columns if column not in features.columns
        ]
        if missing:
            raise ValidationFailedError(
                "temporal inference could not rebuild feature column(s) "
                f"{missing}; include the required raw columns/history or send the precomputed features"
            )
        feature_row = features.iloc[[position]][artifacts.feature_columns]

        for name in fill_targets:
            decoded, probabilities, raw = _predict_single(artifacts, name, feature_row)
            if name in requested:
                rows[step][name] = decoded
                if probabilities:
                    rows[step][f"{name}_probabilities"] = probabilities
                metas[name] = _meta_for(artifacts, name)
            if name in config.target_columns:
                numeric = _target_value_for_lag(raw, artifacts.models[name])
                if numeric is None:
                    if not warned_classification:
                        warnings.append(
                            "recursive inference could not feed a non-numeric classification "
                            "prediction back into target lags; later lag values may be imputed"
                        )
                        warned_classification = True
                else:
                    combined.at[position, name] = numeric

    return rows, metas


def run_inference(
    job_id: str,
    df: pd.DataFrame,
    targets: list[str] | None = None,
    *,
    history: pd.DataFrame | list[dict[str, Any]] | None = None,
    horizon: int | None = None,
    mode: str = "auto",
) -> PredictResponse:
    artifacts = resolve_job_artifacts(job_id)
    if df.empty:
        raise ValidationFailedError("inference input contains no rows")

    settings = get_settings()
    if len(df) > settings.predict_max_records:
        raise ValidationFailedError(
            f"inference input has {len(df)} rows; limit is {settings.predict_max_records}"
        )

    if mode not in {"auto", "one-step-ahead", "recursive"}:
        raise ValidationFailedError(
            f"unknown temporal inference mode '{mode}'; use 'auto', 'one-step-ahead' or 'recursive'"
        )

    warnings: list[str] = []
    if not artifacts.feature_columns:
        raise ValidationFailedError(f"run summary for job '{job_id}' has no feature columns")
    missing = [col for col in artifacts.feature_columns if col not in df.columns]

    requested = list(artifacts.models.keys())
    if targets:
        unknown = [name for name in targets if name not in artifacts.models]
        if unknown:
            raise ValidationFailedError(
                f"unknown target(s) for job '{job_id}': {unknown}; available: {sorted(artifacts.models)}"
            )
        requested = targets

    if not missing:
        # Option A: the payload already carries every training feature column.
        if history is not None:
            warnings.append(
                "history was ignored because the payload already contains every feature column"
            )
        temporal_needs_warning = (
            artifacts.temporal is not None
            and bool(artifacts.temporal.config.target_lags)
            and (mode == "recursive" or bool(horizon and horizon > 1))
        )
        if temporal_needs_warning:
            warnings.append(
                "the payload already contains precomputed target lags; recursive inference was "
                "not applied. Send raw rows without target columns to rebuild them step by step."
            )
        extra = [col for col in df.columns if col not in artifacts.feature_columns]
        if extra:
            warnings.append(f"ignored extra column(s): {extra}")
        X = df[artifacts.feature_columns]
        rows, metas, _raw = _predict_matrix(artifacts, X, requested)
    else:
        # Option B: rebuild the feature matrix from raw covariates plus history.
        temporal = artifacts.temporal
        if temporal is None:
            raise ValidationFailedError(f"missing required feature column(s): {missing}")

        history_frame = _as_history_frame(history)
        if history_frame is None:
            history_frame = temporal.history
        combined, target_positions, date_column = _build_combined_temporal_frame(
            df, history_frame, temporal, warnings
        )
        recursive = _should_recurse(temporal, df, horizon, mode)

        if recursive:
            rows, metas = _run_recursive_inference(
                artifacts, temporal, combined, target_positions, requested, warnings
            )
            rows = _restore_input_order(rows, combined, target_positions)
        else:
            if temporal.config.target_lags and mode == "one-step-ahead" and len(target_positions) > 1:
                warnings.append(
                    "mode='one-step-ahead' with multiple rows keeps missing intermediate target "
                    "values as NaN; use mode='recursive' for multi-step forecasts."
                )
            features = _apply_temporal_features(combined, temporal, date_column)
            X = _extract_target_features(features, artifacts, target_positions)
            rows, metas, _raw = _predict_matrix(artifacts, X, requested)
            rows = _restore_input_order(rows, combined, target_positions)

    logger.info("inference on job %s: %d row(s), targets=%s", job_id, len(rows), requested)
    return PredictResponse(
        job_id=job_id,
        n_rows=len(rows),
        targets=metas,
        predictions=rows,
        warnings=warnings,
    )


def run_inference_from_records(
    job_id: str,
    records: list[dict[str, Any]],
    targets: list[str] | None = None,
    *,
    history: list[dict[str, Any]] | pd.DataFrame | None = None,
    horizon: int | None = None,
    mode: str = "auto",
) -> PredictResponse:
    return run_inference(
        job_id,
        pd.DataFrame.from_records(records),
        targets,
        history=history,
        horizon=horizon,
        mode=mode,
    )


def run_inference_from_file(
    job_id: str,
    filename: str,
    content: bytes,
    targets: list[str] | None = None,
    *,
    history: list[dict[str, Any]] | pd.DataFrame | None = None,
    horizon: int | None = None,
    mode: str = "auto",
) -> PredictResponse:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"file type '{suffix}' not allowed; supported: {sorted(ALLOWED_EXTENSIONS)}"
        )
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValidationFailedError(f"file exceeds upload limit of {settings.max_upload_mb} MB")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        df = read_tabular(tmp_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            os.unlink(tmp_path)
    return run_inference(
        job_id,
        df,
        targets,
        history=history,
        horizon=horizon,
        mode=mode,
    )
