from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from automl_shared.artifacts import load_model, verify_model_integrity
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
from automl_shared.storage.local_storage import storage
from automl_shared.tabular import ALLOWED_EXTENSIONS, read_tabular

logger = logging.getLogger(__name__)


@dataclass
class JobArtifacts:
    job_id: str
    feature_columns: list[str]
    models: dict[str, dict]
    target_meta: dict[str, PredictTargetMeta] = field(default_factory=dict)


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
    )


def run_inference(job_id: str, df: pd.DataFrame, targets: list[str] | None = None) -> PredictResponse:
    artifacts = resolve_job_artifacts(job_id)
    if df.empty:
        raise ValidationFailedError("inference input contains no rows")

    settings = get_settings()
    if len(df) > settings.predict_max_records:
        raise ValidationFailedError(
            f"inference input has {len(df)} rows; limit is {settings.predict_max_records}"
        )

    warnings: list[str] = []
    if not artifacts.feature_columns:
        raise ValidationFailedError(f"run summary for job '{job_id}' has no feature columns")
    missing = [col for col in artifacts.feature_columns if col not in df.columns]
    if missing:
        raise ValidationFailedError(f"missing required feature column(s): {missing}")
    extra = [col for col in df.columns if col not in artifacts.feature_columns]
    if extra:
        warnings.append(f"ignored extra column(s): {extra}")

    requested = list(artifacts.models.keys())
    if targets:
        unknown = [name for name in targets if name not in artifacts.models]
        if unknown:
            raise ValidationFailedError(
                f"unknown target(s) for job '{job_id}': {unknown}; available: {sorted(artifacts.models)}"
            )
        requested = targets

    X = df[artifacts.feature_columns]
    rows: list[dict[str, Any]] = [{"row_index": i} for i in range(len(X))]
    metas: dict[str, PredictTargetMeta] = {}

    for name in requested:
        payload = artifacts.models[name]
        pipeline = payload["pipeline"]
        task = str(payload.get("task", ""))
        label_classes = payload.get("label_classes")
        predictions = np.asarray(pipeline.predict(X))
        proba = pipeline.predict_proba(X) if task == "classification" else None

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

        metas[name] = artifacts.target_meta.get(name) or PredictTargetMeta(
            target=name,
            task=task,
            classes=[str(c) for c in label_classes] if label_classes else None,
        )

    logger.info("inference on job %s: %d row(s), targets=%s", job_id, len(X), requested)
    return PredictResponse(
        job_id=job_id,
        n_rows=len(X),
        targets=metas,
        predictions=rows,
        warnings=warnings,
    )


def run_inference_from_records(
    job_id: str,
    records: list[dict[str, Any]],
    targets: list[str] | None = None,
) -> PredictResponse:
    return run_inference(job_id, pd.DataFrame.from_records(records), targets)


def run_inference_from_file(
    job_id: str,
    filename: str,
    content: bytes,
    targets: list[str] | None = None,
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
    return run_inference(job_id, df, targets)
