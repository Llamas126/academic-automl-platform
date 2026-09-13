from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    pipeline,
    X_background: pd.DataFrame,
    explainer_kind: str,
    max_eval_rows: int = 500,
) -> tuple[np.ndarray, bool, np.ndarray, list[str]] | None:
    import shap

    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["estimator"]

    background = preprocessor.transform(X_background)
    if hasattr(background, "toarray"):
        background = background.toarray()
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    eval_data = preprocessor.transform(X_background)
    if hasattr(eval_data, "toarray"):
        eval_data = eval_data.toarray()

    if explainer_kind == "tree":
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(background, check_additivity=False)
        values, multiclass = _normalize_values(raw)
        return values, multiclass, eval_data, feature_names

    if explainer_kind == "linear":
        explainer = shap.LinearExplainer(estimator, background)
        raw = explainer.shap_values(background)
        values, multiclass = _normalize_values(raw)
        return values, multiclass, eval_data, feature_names

    predict_fn = (
        (lambda d: estimator.predict_proba(d)) if hasattr(estimator, "predict_proba") else estimator.predict
    )
    sampled_bg = shap.sample(background, min(len(background), 100), random_state=0)
    explainer = shap.KernelExplainer(predict_fn, sampled_bg)
    n_eval = min(len(eval_data), max_eval_rows)
    summary_data = shap.sample(eval_data, n_eval, random_state=0)
    raw = explainer.shap_values(summary_data, nsamples="auto", silent=True)
    values, multiclass = _normalize_values(raw)
    return values, multiclass, summary_data, feature_names


def _normalize_values(values) -> tuple[np.ndarray, bool]:
    if isinstance(values, list):
        stacked = np.stack([np.asarray(v) for v in values])
        return np.abs(stacked).mean(axis=0), True
    arr = np.asarray(values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=-1), True
    return arr, False


def build_interpretability_profile(
    pipeline,
    X_background: pd.DataFrame,
    explainer_kind: str,
) -> dict:
    """Extract the per-model interpretability values the exporter plots from.

    Linear estimators expose raw ``coef_`` (beta weights); tree estimators are
    explained through SHAP. When SHAP cannot be computed the profile falls back
    to the estimator's intrinsic ``feature_importances_`` so the exporter can
    still render an importance figure.

    Returns a dict whose arrays are consumed either as JSON lists (coefficients,
    importance) or persisted separately as npz (raw SHAP + eval arrays needed
    by the beeswarm figure).
    """
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["estimator"]

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    profile: dict = {
        "kind": explainer_kind,
        "feature_names": feature_names,
        "coefficients": None,
        "importance": None,
        "intrinsic_importance": None,
        "source": "intrinsic",
        "multiclass": False,
        "has_shap": False,
        "shap_values": None,
        "shap_eval_data": None,
    }

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        profile["coefficients"] = coef
        if coef.ndim == 1:
            profile["importance"] = np.abs(coef)
        else:
            profile["importance"] = np.abs(coef).max(axis=0)

    if hasattr(estimator, "feature_importances_"):
        profile["intrinsic_importance"] = np.asarray(estimator.feature_importances_)

    if explainer_kind in ("tree", "linear"):
        try:
            result = compute_shap_values(pipeline, X_background, explainer_kind)
        except Exception as exc:
            logger.warning("SHAP values unavailable for %s: %s", explainer_kind, exc)
            result = None
        if result is not None:
            values, multiclass, eval_data, shap_names = result
            profile["feature_names"] = shap_names
            profile["multiclass"] = multiclass
            profile["has_shap"] = True
            profile["source"] = "shap"
            profile["importance"] = np.abs(values).mean(axis=0)
            profile["shap_values"] = values
            profile["shap_eval_data"] = eval_data

    if profile["importance"] is None and profile["intrinsic_importance"] is not None:
        profile["importance"] = profile["intrinsic_importance"]
        profile["source"] = "intrinsic"
    return profile
