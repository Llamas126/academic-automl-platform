from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from automl_shared.figures import save_figure

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
        return np.abs(arr).mean(axis=0), True
    return arr, False


def build_shap_figures(
    pipeline,
    X_background: pd.DataFrame,
    explainer_kind: str,
    target_name: str,
    out_dir: Path,
    dpi: int = 300,
    include_eps: bool = True,
    include_dependence: bool = True,
    max_display: int = 20,
    seed: int = 42,
) -> list[Path]:
    try:
        result = compute_shap_values(pipeline, X_background, explainer_kind)
        if result is None:
            return []
        values, was_multiclass, eval_data, feature_names = result

        written: list[Path] = []
        importance = np.abs(values).mean(axis=0)
        order = np.argsort(importance)[::-1][:max_display]
        safe_target = _safe(target_name)

        bar_fig = _bar_figure(importance[order], [feature_names[i] for i in order], target_name)
        written.extend(
            save_figure(bar_fig, out_dir, f"shap_importance_{safe_target}", include_eps)
        )

        if not was_multiclass:
            beeswarm_fig = _beeswarm_figure(
                values[:, order], eval_data[:, order], [feature_names[i] for i in order], target_name
            )
            written.extend(
                save_figure(beeswarm_fig, out_dir, f"shap_beeswarm_{safe_target}", include_eps)
            )
            if include_dependence and len(order) > 0:
                top_index = int(order[0])
                dep_fig = _dependence_figure(
                    values, eval_data, top_index, feature_names[top_index], target_name
                )
                stem = f"shap_dependence_{_safe(feature_names[top_index])[:40]}_{safe_target}"
                written.extend(save_figure(dep_fig, out_dir, stem, include_eps))
        else:
            logger.info("multiclass SHAP: only global importance plot generated for %s", target_name)
        return written
    except Exception as exc:
        logger.warning("SHAP analysis failed for target '%s': %s", target_name, exc, exc_info=True)
        return []


def _bar_figure(importance: np.ndarray, names: list[str], target: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, max(2.8, 0.32 * len(names) + 1.2)))
    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, importance, align="center", color="#3b6ea5")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten_feature(n) for n in names])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance (SHAP): {target}")
    return fig


def _beeswarm_figure(values, data, names, target):
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.34 * len(names) + 1.4)))
    rng = np.random.default_rng(0)
    for row_index in range(min(len(names), values.shape[1])):
        column = values[:, row_index]
        feature_column = data[:, row_index]
        spread = float(np.ptp(feature_column))
        norm = (feature_column - feature_column.min()) / (spread + 1e-12)
        jitter = rng.uniform(-0.18, 0.18, size=len(column))
        ax.scatter(
            column,
            np.full(len(column), row_index) + jitter,
            c=norm,
            cmap="coolwarm",
            s=9,
            edgecolors="none",
        )
    ax.set_yticks(range(min(len(names), values.shape[1])))
    ax.set_yticklabels([_shorten_feature(n) for n in names])
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"SHAP summary: {target}")
    return fig


def _dependence_figure(values, data, feature_index, feature_name, target):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(data[:, feature_index], values[:, feature_index], s=12, edgecolors="none")
    ax.set_xlabel(_shorten_feature(feature_name))
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP dependence: {feature_name} ({target})")
    return fig


def _shorten_feature(name: str, width: int = 34) -> str:
    clean = name.replace("numeric__", "").replace("categorical__", "")
    return clean if len(clean) <= width else clean[: width - 1] + "…"


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]
