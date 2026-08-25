"""SHAP rendering entrypoint for the explainability stage.

Loads the verified model artifacts and the background sample written by the
ML core, renders SHAP figures into the job output tree, and degrades
gracefully: any failure is logged and reported through a status file instead
of failing the orchestration chain.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from automl_shared.artifacts import verify_model_integrity
from automl_shared.contract import load_background_frame, load_training_meta, safe_target_stem
from automl_shared.schemas.training import TrainRequest

logger = logging.getLogger(__name__)

SHAP_STATUS_NAME = "shap_status.json"


def render_shap_figures_if_requested(work_dir: Path, out_dir: Path, request: TrainRequest) -> dict[str, int]:
    """Render SHAP figures when ``request.explain.shap`` is enabled.

    Returns a mapping target -> png count. Never raises: on error it writes
    ``work/shap_status.json`` with ``ok=false`` so downstream stages can rely
    on filesystem state only.
    """
    if not request.explain.shap:
        return {}
    try:
        counts = _render_all(work_dir, out_dir, request)
        _write_status(work_dir, ok=True, rendered=counts)
        return counts
    except Exception as exc:
        logger.warning("SHAP stage failed for job: %s", exc, exc_info=True)
        _write_status(work_dir, ok=False, error=str(exc))
        return {}


def _render_all(work_dir: Path, out_dir: Path, request: TrainRequest) -> dict[str, int]:
    from explainability.shap_engine import build_shap_figures

    models_dir = out_dir / "models"
    verify_model_integrity(models_dir)
    meta = load_training_meta(work_dir)
    background = load_background_frame(work_dir)
    if background is None:
        raise FileNotFoundError("background sample missing from training contract")

    import joblib

    counts: dict[str, int] = {}
    for entry in meta["targets"]:
        name = entry["target"]
        payload_path = models_dir / f"{safe_target_stem(name)}_best_model.joblib"
        payload = joblib.load(payload_path)
        pipeline = payload.get("pipeline")
        if pipeline is None:
            continue
        shap_dir = out_dir / "figures" / name.replace("/", "_") / "shap"
        written = build_shap_figures(
            pipeline=pipeline,
            X_background=background,
            explainer_kind=entry["explainer_kind"],
            target_name=name,
            out_dir=shap_dir,
            dpi=request.export.figures_dpi,
            include_eps=request.export.include_eps,
            include_dependence=request.export.include_shap_dependence,
            max_display=request.explain.max_display or 20,
        )
        png_count = sum(1 for p in written if p.suffix == ".png")
        if png_count:
            counts[name] = png_count
    return counts


def _write_status(
    work_dir: Path,
    *,
    ok: bool,
    rendered: dict[str, int] | None = None,
    error: str = "",
) -> None:
    payload = {"ok": ok}
    if rendered is not None:
        payload["rendered"] = rendered
    if error:
        payload["error"] = error[:500]
    (work_dir / SHAP_STATUS_NAME).write_text(json.dumps(payload), encoding="utf-8")
