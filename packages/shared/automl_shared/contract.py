"""File-based inter-service training contract (consumer side).

The ML core persists every intermediate artifact a downstream stage needs
into the job directories (see ``mlcore.contract.persist_training_output``).
This module is the stable read-side of that contract: explainability and
exporter workers depend only on ``automl_shared``, never on the trainer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl_shared.schemas.results import RunSummary

META_NAME = "training_meta.json"
BACKGROUND_NAME = "background.csv"
SUMMARY_RELPATH = Path("metrics") / "results_summary.json"
PER_FOLD_RELPATH = Path("metrics") / "per_fold.csv"
PREDICTIONS_RELPATH = Path("data") / "dataset_with_predictions.csv"


def safe_target_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]


def eval_npz_name(target: str) -> str:
    return f"eval_{safe_target_stem(target)}.npz"


def contract_complete(work_dir: Path, out_dir: Path) -> bool:
    return (out_dir / SUMMARY_RELPATH).exists() and (work_dir / META_NAME).exists()


def load_training_meta(work_dir: Path) -> dict:
    path = work_dir / META_NAME
    if not path.exists():
        raise FileNotFoundError(f"training contract missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(out_dir: Path) -> RunSummary:
    return RunSummary.model_validate_json((out_dir / SUMMARY_RELPATH).read_text(encoding="utf-8"))


def load_per_fold_records(out_dir: Path) -> dict[str, list[dict]]:
    path = out_dir / PER_FOLD_RELPATH
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    records: dict[str, list[dict]] = {}
    for row in frame.to_dict(orient="records"):
        target = str(row.pop("target"))
        cleaned = {k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
        params = cleaned.get("params")
        if isinstance(params, str):
            try:
                cleaned["params"] = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                pass
        records.setdefault(target, []).append(cleaned)
    return records


def load_background_frame(work_dir: Path) -> pd.DataFrame | None:
    path = work_dir / BACKGROUND_NAME
    if not path.exists():
        return None
    return pd.read_csv(path)


class ExportTargetView:
    """Duck-typed stand-in for TargetArtifacts rebuilt from the contract."""

    def __init__(
        self,
        target: str,
        task: str,
        best_entry_key: str,
        explainer_kind: str,
        label_classes: list[str] | None,
        confusion_matrix: list[list[int]] | None,
        confusion_labels: list[str] | None,
        y_true_encoded: np.ndarray | None,
        oof_predictions: np.ndarray | None,
        oof_probabilities: np.ndarray | None,
        citation_key: str | None = None,
        final_pipeline=None,
    ) -> None:
        self.target = target
        self.task = task
        self.best_entry_key = best_entry_key
        self.explainer_kind = explainer_kind
        self.label_classes = label_classes
        self.confusion_matrix = confusion_matrix
        self.confusion_labels = confusion_labels
        self.y_true_encoded = y_true_encoded
        self.oof_predictions = oof_predictions
        self.oof_probabilities = oof_probabilities
        self.citation_key = citation_key
        self.final_pipeline = final_pipeline


def load_target_views(
    work_dir: Path, models_dir: Path, with_pipelines: bool = False
) -> dict[str, ExportTargetView]:
    import joblib

    meta = load_training_meta(work_dir)
    views: dict[str, ExportTargetView] = {}
    for entry in meta["targets"]:
        name = entry["target"]
        pipeline = None
        if with_pipelines:
            payload = joblib.load(models_dir / f"{safe_target_stem(name)}_best_model.joblib")
            pipeline = payload.get("pipeline")
        data = np.load(work_dir / eval_npz_name(name))
        views[name] = ExportTargetView(
            target=name,
            task=entry["task"],
            best_entry_key=entry["best_entry_key"],
            explainer_kind=entry["explainer_kind"],
            label_classes=entry["label_classes"],
            confusion_matrix=entry["confusion_matrix"],
            confusion_labels=entry["confusion_labels"],
            y_true_encoded=data["y_true"] if "y_true" in data.files else None,
            oof_predictions=data["oof_predictions"] if "oof_predictions" in data.files else None,
            oof_probabilities=data["oof_probabilities"] if "oof_probabilities" in data.files else None,
            citation_key=entry.get("citation_key"),
            final_pipeline=pipeline,
        )
    return views
