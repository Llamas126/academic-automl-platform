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

import numpy as np
from automl_shared.artifacts import verify_model_integrity
from automl_shared.contract import load_background_frame, load_training_meta, safe_target_stem
from automl_shared.schemas.training import TrainRequest

logger = logging.getLogger(__name__)

SHAP_STATUS_NAME = "shap_status.json"

INTERPRETABILITY_DIR_NAME = "interpretability"


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
        render_interpretability_profiles_if_requested(work_dir, out_dir, request)
        _write_status(work_dir, ok=True, rendered=counts)
        return counts
    except Exception as exc:
        logger.warning("SHAP stage failed for job: %s", exc, exc_info=True)
        _write_status(work_dir, ok=False, error=str(exc))
        return {}


def render_interpretability_profiles_if_requested(
    work_dir: Path, out_dir: Path, request: TrainRequest
) -> int:
    """Persist per-model interpretability profiles for every evaluated candidate.

    The profiles carry the coefficient / SHAP / feature-importance values the
    exporter needs to render ``coef_plot_*``, ``shap_importance_*``,
    ``shap_beeswarm_*`` and the consolidated comparison matrix. Returns the
    number of targets profiled; never raises.
    """
    if not request.explain.shap:
        return 0
    try:
        return _render_profiles_all(work_dir, out_dir, request)
    except Exception as exc:
        logger.warning("interpretability profiling failed for job: %s", exc, exc_info=True)
        return 0


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


def _render_profiles_all(work_dir: Path, out_dir: Path, request: TrainRequest) -> int:
    import joblib

    from explainability.shap_engine import build_interpretability_profile

    models_dir = out_dir / "models"
    verify_model_integrity(models_dir)
    meta = load_training_meta(work_dir)
    background = load_background_frame(work_dir)
    if background is None:
        raise FileNotFoundError("background sample missing from training contract")

    interp_dir = work_dir / INTERPRETABILITY_DIR_NAME
    interp_dir.mkdir(parents=True, exist_ok=True)
    profiled_targets = 0

    for target_meta in meta.get("targets", []):
        name = target_meta.get("target")
        if not name:
            continue
        safe = safe_target_stem(name)
        best_key = target_meta.get("best_entry_key")
        models_meta = list(target_meta.get("models") or [])
        if best_key and not any(m.get("key") == best_key for m in models_meta):
            models_meta.insert(
                0,
                {
                    "key": best_key,
                    "rank": 1,
                    "explainer_kind": target_meta.get("explainer_kind"),
                },
            )

        profiles: list[dict] = []
        for model_info in models_meta:
            key = model_info.get("key")
            if not key:
                continue
            if key == best_key:
                payload_path = models_dir / f"{safe}_best_model.joblib"
            else:
                payload_path = models_dir / f"{safe}_{safe_target_stem(key)}.joblib"
            if not payload_path.exists():
                logger.warning("candidate model artifact missing: %s", payload_path.name)
                continue
            payload = joblib.load(payload_path)
            pipeline = payload.get("pipeline")
            if pipeline is None:
                continue
            kind = (
                model_info.get("explainer_kind")
                or payload.get("explainer_kind")
                or "kernel"
            )
            profile = build_interpretability_profile(pipeline, background, kind)
            profile["key"] = key
            profile["display_name"] = model_info.get("display_name") or payload.get("display_name") or key
            profile["rank"] = model_info.get("rank") or 0
            profiles.append(profile)

        if not profiles:
            continue

        feature_names = [str(n) for n in profiles[0].get("feature_names") or []]
        serialized: list[dict] = []
        for profile in profiles:
            shap_values = profile.pop("shap_values", None)
            shap_eval_data = profile.pop("shap_eval_data", None)
            stem = f"{safe}_{safe_target_stem(profile['key'])}"
            if not profile.get("multiclass") and shap_values is not None:
                array_path = interp_dir / f"{stem}.npz"
                np.savez_compressed(
                    array_path,
                    shap_values=np.asarray(shap_values, dtype=np.float32),
                    eval_data=np.asarray(shap_eval_data, dtype=np.float32),
                )
                profile["shap_file"] = f"{stem}.npz"
            else:
                profile["shap_file"] = None
            profile.pop("intrinsic_importance", None)
            profile.pop("feature_names", None)
            profile["importance"] = _jsonable(profile.get("importance"))
            profile["coefficients"] = _jsonable(profile.get("coefficients"))
            serialized.append(profile)

        index = {
            "target": name,
            "task": target_meta.get("task"),
            "feature_names": feature_names,
            "models": serialized,
        }
        index_path = interp_dir / f"{safe}.json"
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
        profiled_targets += 1

    logger.info("persisted interpretability profiles for %d target(s) under %s", profiled_targets, interp_dir)
    return profiled_targets


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).tolist()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


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
