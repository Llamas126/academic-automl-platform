from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.results import RunSummary


def save_final_models(artifacts: dict, models_dir: Path) -> None:
    import joblib

    for target_name, artifact in artifacts.items():
        if artifact.final_pipeline is None:
            continue
        safe_target = _safe_name(target_name)
        payload = {
            "pipeline": artifact.final_pipeline,
            "target": target_name,
            "task": artifact.task,
            "model_key": artifact.best_entry_key,
            "hyperparameters": artifact.best_params,
            "label_classes": artifact.label_classes,
        }
        joblib.dump(payload, models_dir / f"{safe_target}_best_model.joblib")


def save_run_summary(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")


def load_model(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
