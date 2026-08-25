"""Producer side of the training contract (see automl_shared.contract)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from automl_shared.contract import (
    BACKGROUND_NAME,
    META_NAME,
    PER_FOLD_RELPATH,
    PREDICTIONS_RELPATH,
    SUMMARY_RELPATH,
    contract_complete,
    eval_npz_name,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_persisted",
    "persist_training_output",
]


def persist_training_output(output, work_dir: Path, out_dir: Path) -> None:
    """Write the training contract: base package artifacts + worker inputs."""
    from automl_shared.artifacts import save_final_models, write_model_manifest

    metrics_dir = out_dir / "metrics"
    data_dir = out_dir / "data"
    models_dir = out_dir / "models"
    for directory in (metrics_dir, data_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (out_dir / SUMMARY_RELPATH).write_text(
        json.dumps(output.summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_per_fold_csv(output.per_fold_records, out_dir / PER_FOLD_RELPATH)
    if output.predictions_frame is not None:
        output.predictions_frame.to_csv(out_dir / PREDICTIONS_RELPATH, index=False)
    save_final_models(output.artifacts, models_dir)
    write_model_manifest(models_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    targets_meta: list[dict[str, Any]] = []
    for target_name, artifact in output.artifacts.items():
        arrays: dict[str, np.ndarray] = {}
        if artifact.y_true_encoded is not None:
            arrays["y_true"] = artifact.y_true_encoded
        if artifact.oof_predictions is not None:
            arrays["oof_predictions"] = artifact.oof_predictions
        if artifact.oof_probabilities is not None:
            arrays["oof_probabilities"] = artifact.oof_probabilities
        np.savez_compressed(work_dir / eval_npz_name(target_name), **arrays)
        targets_meta.append(
            {
                "target": target_name,
                "task": artifact.task,
                "best_entry_key": artifact.best_entry_key,
                "explainer_kind": artifact.explainer_kind,
                "label_classes": artifact.label_classes,
                "confusion_matrix": artifact.confusion_matrix,
                "confusion_labels": artifact.confusion_labels,
                "citation_key": _citation_key_for(artifact),
            }
        )
    meta = {
        "schema_version": 1,
        "job_id": output.summary.job_id,
        "feature_columns": output.summary.feature_columns,
        "targets": targets_meta,
    }
    (work_dir / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if output.background_frame is not None:
        output.background_frame.to_csv(work_dir / BACKGROUND_NAME, index=False)
    logger.info("persisted training contract for job %s into %s", output.summary.job_id, work_dir)


def _citation_key_for(artifact) -> str | None:
    try:
        from mlcore.ml.models.registry import get_entry

        entry = get_entry(artifact.task, artifact.best_entry_key)
    except Exception:
        return None
    return getattr(entry, "citation_key", None)


def ensure_persisted(output, work_dir: Path, out_dir: Path) -> None:
    """Persist base artifacts once when running against a live PipelineOutput."""
    if contract_complete(work_dir, out_dir):
        return
    persist_training_output(output, work_dir, out_dir)


def _write_per_fold_csv(per_fold_records: dict[str, list[dict]], path: Path) -> None:
    rows: list[dict] = []
    for target, records in per_fold_records.items():
        for record in records:
            flat: dict = {"target": target}
            flat.update({k: v for k, v in record.items() if k not in ("params",)})
            params = record.get("params")
            if params:
                flat["params"] = json.dumps(params)
            rows.append(flat)
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
