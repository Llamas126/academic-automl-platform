"""Diagnostic figure builders for the scientific bundle."""
from __future__ import annotations

import json
import logging
import re

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from automl_shared.figures import configure_publication_style, save_figure  # noqa: F401
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)

__all__ = [
    "configure_publication_style",
    "coefficients_figure",
    "confusion_matrix_figure",
    "correlation_heatmap_figure",
    "decision_tree_flowchart_figure",
    "feature_importance_figure",
    "friendly_feature_names",
    "metric_boxplot_figure",
    "model_feature_importance_comparison_figure",
    "predicted_vs_actual_figure",
    "residuals_figure",
    "roc_curves_figure",
    "save_figure",
    "shap_beeswarm_figure",
]


def confusion_matrix_figure(
    cm: list[list[int]],
    labels: list[str],
    title: str,
) -> plt.Figure:
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return fig


def roc_curves_figure(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    n_classes = len(class_names)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[1]} (AUC = {roc_auc:.3f})")
    else:
        for class_index in range(n_classes):
            binary = (y_true == class_index).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                continue
            fpr, tpr, _ = roc_curve(binary, proba[:, class_index])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_names[class_index]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    return fig


def predicted_vs_actual_figure(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.scatter(y_true, y_pred, s=14, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="gray", linewidth=1.0)
    ax.set_xlabel(f"Observed {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Out-of-fold predictions: {target}")
    return fig


def residuals_figure(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> plt.Figure:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fitted = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(fitted, residuals, s=14, edgecolors="none")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual analysis: {target}")
    return fig


def metric_boxplot_figure(per_fold_records: list[dict], metric: str, models: list[str]) -> plt.Figure:
    data: list[list[float]] = []
    labels: list[str] = []
    for model in models:
        values = [
            r[metric]
            for r in per_fold_records
            if r.get("model") == model and r.get(metric) is not None and np.isfinite(r.get(metric))
        ]
        if values:
            data.append([float(v) for v in values])
            labels.append(model)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if data:
        ax.boxplot(
            data,
            tick_labels=[_shorten(label, 22) for label in labels],
        )
        ax.tick_params(axis="x", rotation=20)
    else:
        logger.warning(
            "boxplot for metric '%s': no valid fold data for models %s "
            "(%d records examined)",
            metric,
            models,
            len(per_fold_records),
        )
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution across folds")
    return fig


def correlation_heatmap_figure(df: pd.DataFrame, max_cols: int = 12) -> plt.Figure:
    numeric = df.select_dtypes(include=[np.number]).iloc[:, :max_cols]
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Feature correlation matrix")
    return fig


def coefficients_figure(coef, feature_names: list[str], model_name: str) -> plt.Figure:
    """Horizontal bar chart of the model's fitted coefficients (beta weights)."""
    coef_arr = np.asarray(coef, dtype=float)
    n_features = coef_arr.shape[-1]
    feature_names = feature_names[:n_features]
    if coef_arr.ndim == 1:
        matrix = coef_arr.reshape(1, -1)
        class_labels: list[str] = []
    else:
        matrix = coef_arr
        class_labels = [f"class {index}" for index in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.34 * n_features + 1.2)))
    y_pos = np.arange(n_features)
    if matrix.shape[0] == 1:
        ax.barh(y_pos, matrix[0], align="center", color="#3b6ea5")
    else:
        width = 0.8 / matrix.shape[0]
        for row_index, coefficients in enumerate(matrix):
            offset = (row_index - (matrix.shape[0] - 1) / 2) * width
            ax.barh(
                y_pos + offset,
                coefficients,
                height=width,
                align="center",
                label=class_labels[row_index],
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(_clean_feature(name), 30) for name in feature_names])
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient (beta)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Coefficients: {model_name}")
    if class_labels:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    return fig


def feature_importance_figure(
    importance, feature_names: list[str], model_name: str, source: str = "shap"
) -> plt.Figure:
    """Horizontal bar chart of per-feature importance, sorted descending."""
    importance_arr = np.asarray(importance, dtype=float)
    order = np.argsort(importance_arr)[::-1]
    sorted_importances = importance_arr[order]
    labels = [_shorten(_clean_feature(name), 34) for name in np.asarray(feature_names)[order]]
    xlabel = "Mean |SHAP value|" if source == "shap" else "Importance"

    fig, ax = plt.subplots(figsize=(6.2, max(2.8, 0.32 * len(sorted_importances) + 1.2)))
    y_pos = np.arange(len(sorted_importances))[::-1]
    ax.barh(y_pos, sorted_importances, align="center", color="#3b6ea5")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Feature importance: {model_name}")
    return fig


def shap_beeswarm_figure(
    shap_values, feature_values, feature_names: list[str], model_name: str
) -> plt.Figure:
    """Jittered scatter of SHAP contributions colored by feature magnitude."""
    values = np.asarray(shap_values, dtype=float)
    data = np.asarray(feature_values, dtype=float)
    names = list(feature_names[: values.shape[1]])
    order = np.argsort(np.abs(values).mean(axis=0))[::-1]
    values = values[:, order]
    data = data[:, order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.34 * len(names) + 1.4)))
    rng = np.random.default_rng(0)
    for row_index in range(len(names)):
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
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_shorten(_clean_feature(name), 34) for name in names])
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"SHAP summary: {model_name}")
    return fig


def model_feature_importance_comparison_figure(
    profiles: list[dict], feature_names: list[str]
) -> plt.Figure:
    """Heatmap of per-model importance normalized to [0, 1] per model.

    Each ``profile`` requires ``importance`` (aligned to ``feature_names``) and
    a label passed as ``display_name`` or ``key``. Values are max-normalized so
    linear (coefficient/SHAP) and tree (SHAP) importances share one scale.
    """
    if not profiles:
        logger.warning("comparison matrix requested without any model profiles")
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.axis("off")
        ax.set_title("Model feature importance comparison")
        ax.text(0.5, 0.5, "No interpretability data available", ha="center", va="center")
        return fig

    matrix = np.zeros((len(feature_names), len(profiles)))
    labels: list[str] = []
    for column_index, profile in enumerate(profiles):
        importance = np.asarray(profile.get("importance"), dtype=float)
        importance = importance[: len(feature_names)]
        denom = float(np.max(np.abs(importance))) if len(importance) else 1.0
        matrix[:, column_index] = np.abs(importance) / max(denom, 1e-12)
        labels.append(profile.get("display_name") or profile.get("key") or f"model {column_index + 1}")

    row_order = np.argsort(matrix.max(axis=1))[::-1]
    matrix = matrix[row_order]
    ordered_features = [feature_names[i] for i in row_order]

    fig, ax = plt.subplots(
        figsize=(max(5.5, 0.9 * len(labels) + 2.5), max(3.5, 0.38 * len(ordered_features) + 1.5))
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Relative importance (0-1)"},
        xticklabels=[_shorten(label, 24) for label in labels],
        yticklabels=[_shorten(_clean_feature(name), 26) for name in ordered_features],
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    ax.set_title("Model feature importance comparison (max-normalized)")
    for column_index in range(matrix.shape[1]):
        top_row = int(np.argmax(matrix[:, column_index]))
        ax.add_patch(
            plt.Rectangle(
                (column_index, top_row),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.4,
            )
        )
    return fig


def decision_tree_flowchart_figure(
    estimator,
    feature_names: list[str],
    model_name: str,
    *,
    target: str | None = None,
    task: str = "regression",
    class_names: list[str] | None = None,
    max_depth: int | None = 2,
) -> plt.Figure | None:
    """Render a single decision tree as a plain-language "Sí / No" flowchart.

    Intended for non-technical audiences: the selected model's first tree is
    drawn like an infographic. Split nodes show the real climate/variable name
    (e.g. ``temp_min < 26.3``) with ``Sí`` / ``No`` connectors, while leaves
    show the resulting prediction. Returns ``None`` when the model is not a
    supported tree estimator or its structure cannot be normalized.
    """
    tree = _normalize_first_tree(estimator)
    if tree is None:
        logger.info(
            "skipping decision tree flowchart for %s: estimator %r is not a supported tree",
            model_name,
            type(estimator).__name__,
        )
        return None

    cleaned_features = [_clean_feature(name) for name in feature_names]
    limited = _limit_tree_depth(tree, max_depth if max_depth is not None else 0)
    leaves = [n for n in limited.values() if n["is_leaf"]]
    n_leaves = max(len(leaves), 1)

    width = 2.2 * n_leaves
    height = 1.55 * (max((n["depth"] for n in limited.values()), default=1) + 1) + 0.8
    fig, ax = plt.subplots(figsize=(min(width, 16.0), min(height, 10.0)))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    positions = _compute_tree_positions(limited, width)
    node_size = min(0.85, max(0.5, 10.5 / n_leaves))
    font = _node_font(n_leaves)

    if tree.get("subtitle") is not None:
        ax.text(
            width / 2,
            height - 0.18,
            tree["subtitle"],
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4a4a4a",
            style="italic",
        )

    for node in limited.values():
        if node["is_leaf"]:
            continue
        left = positions.get(node["left"])
        right = positions.get(node["right"])
        parent_pos = positions[node["id"]]
        for child_pos, label, color in (
            (left, "Sí", "#2e8540"),
            (right, "No", "#b32d2e"),
        ):
            if child_pos is None:
                continue
            cnx, cny = parent_pos[0], parent_pos[1] - node_size / 2
            chx, chy = child_pos[0], child_pos[1] + node_size / 2
            ax.plot([cnx, cnx], [cny, (cny + chy) / 2], color="#9aa0a6", linewidth=1.0)
            ax.plot([cnx, chx], [(cny + chy) / 2, (cny + chy) / 2], color="#9aa0a6", linewidth=1.0)
            ax.plot([chx, chx], [(cny + chy) / 2, chy], color="#9aa0a6", linewidth=1.0)
            ax.text(
                (cnx + chx) / 2,
                (cny + chy) / 2 + 0.06,
                label,
                ha="center",
                va="bottom",
                fontsize=font + 1,
                fontweight="bold",
                color=color,
            )

    for node in limited.values():
        x, y = positions[node["id"]]
        box_x = x - node_size / 2
        box_y = y - node_size / 2
        if node["is_leaf"]:
            text = _format_leaf(node, task, class_names)
            facecolor = "#dff0d8"
            edgecolor = "#2e8540"
        else:
            text = _format_decision(node, cleaned_features)
            facecolor = "#dbe9f7"
            edgecolor = "#2b6cb0"
        ax.add_patch(
            FancyBboxPatch(
                (box_x, box_y),
                node_size,
                node_size,
                boxstyle="round,pad=0.04,rounding_size=0.06",
                linewidth=1.2,
                edgecolor=edgecolor,
                facecolor=facecolor,
            )
        )
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=font,
            wrap=True,
        )

    title = f"Cómo decide el modelo: {model_name}"
    if target is not None:
        title = f"Cómo decide el modelo ({target}): {model_name}"
    ax.text(width / 2, height - 0.28, title, ha="center", va="top", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    return fig


def _normalize_first_tree(estimator) -> dict | None:
    """Extract the first tree as a normalized node map shared by all renderers."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError:
        return None

    try:
        if _is_xgboost(estimator):
            return _tree_from_xgboost(estimator)
        if isinstance(estimator, (DecisionTreeClassifier, DecisionTreeRegressor)):
            return _tree_from_sklearn(estimator)
        if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
            return _tree_from_sklearn(estimator.estimators_[0])
        if isinstance(estimator, (GradientBoostingClassifier, GradientBoostingRegressor)):
            return _tree_from_sklearn(estimator.estimators_[0, 0])
        if isinstance(estimator, (LGBMClassifier, LGBMRegressor)):
            return _tree_from_lightgbm(estimator)
    except Exception as exc:
        logger.warning("could not normalize tree for %r: %s", type(estimator).__name__, exc)
        return None
    return None


def _tree_from_xgboost(estimator) -> dict | None:
    booster = estimator.get_booster()
    dumps = booster.get_dump(dump_format="json", with_stats=True)
    if not dumps:
        return None
    root = json.loads(dumps[0])
    nodes: dict[int, dict] = {}
    _walk_xgboost(root, nodes)
    if not nodes:
        return None
    fitted_names = getattr(booster, "feature_names", None)
    if fitted_names:
        for node in nodes.values():
            if not node["is_leaf"]:
                node["feature"] = str(fitted_names[node["split_idx"]])
    return {"root": 0, "nodes": nodes, "subtitle": "Primer árbol del modelo (nivel de detalle simplificado)"}


def _walk_xgboost(raw: dict, nodes: dict[int, dict]) -> None:
    node_id = int(raw["nodeid"])
    if "leaf" in raw:
        nodes[node_id] = {
            "id": node_id,
            "is_leaf": True,
            "is_xgb": True,
            "depth": int(raw.get("depth", 0)),
            "value": float(raw["leaf"]),
            "cover": float(raw.get("cover", 0.0)),
            "left": None,
            "right": None,
            "split_idx": None,
            "feature": None,
            "threshold": None,
        }
        return
    split_idx = _xgboost_feature_index(str(raw["split"]))
    nodes[node_id] = {
        "id": node_id,
        "is_leaf": False,
        "depth": int(raw.get("depth", 0)),
        "split_idx": split_idx,
        "feature": None,
        "threshold": _parse_threshold(raw.get("split_condition")),
        "cover": float(raw.get("cover", 0.0)),
        "left": int(raw["yes"]),
        "right": int(raw["no"]),
        "value": None,
    }
    for child in raw.get("children") or []:
        _walk_xgboost(child, nodes)


def _xgboost_feature_index(split: str) -> int:
    match = re.fullmatch(r"f(\d+)", split)
    if match:
        return int(match.group(1))
    return int(split) if split.isdigit() else 0


def _tree_from_sklearn(tree) -> dict | None:
    attrs = tree.tree_
    children_left = np.asarray(attrs.children_left)
    children_right = np.asarray(attrs.children_right)
    features = np.asarray(attrs.feature)
    thresholds = np.asarray(attrs.threshold)
    values = np.asarray(attrs.value)
    depth = _sklearn_depths(children_left, children_right)

    nodes: dict[int, dict] = {}
    for node_id in range(children_left.shape[0]):
        left = int(children_left[node_id])
        right = int(children_right[node_id])
        is_leaf = left < 0 and right < 0
        nodes[node_id] = {
            "id": node_id,
            "is_leaf": is_leaf,
            "depth": depth[node_id],
            "split_idx": None if is_leaf else int(features[node_id]),
            "feature": None,
            "threshold": None if is_leaf else float(thresholds[node_id]),
            "cover": float(attrs.weighted_n_node_samples[node_id]),
            "left": None if is_leaf else left,
            "right": None if is_leaf else right,
            "value": values[node_id],
        }
    return {"root": 0, "nodes": nodes, "subtitle": None}


def _sklearn_depths(children_left, children_right) -> dict[int, int]:
    depths: dict[int, int] = {0: 0}
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        if node_id in depths and depths[node_id] != depth:
            continue
        for child in (int(children_left[node_id]), int(children_right[node_id])):
            if child >= 0 and child not in depths:
                depths[child] = depth + 1
                stack.append((child, depth + 1))
    return depths


def _tree_from_lightgbm(estimator) -> dict | None:
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        return None
    dump = booster.dump_model()
    tree_info = dump.get("tree_info") or []
    if not tree_info:
        return None
    structure = tree_info[0].get("tree_structure")
    if structure is None:
        return None
    nodes: dict[int, dict] = {}
    _walk_lightgbm(structure, nodes)
    return {"root": 0, "nodes": nodes, "subtitle": "Primer árbol del modelo (nivel de detalle simplificado)"}


def _walk_lightgbm(raw: dict, nodes: dict[int, dict]) -> None:
    node_id = int(raw.get("node_index") or len(nodes))
    if raw.get("leaf_value") is not None:
        nodes[node_id] = {
            "id": node_id,
            "is_leaf": True,
            "depth": int(raw.get("depth", 0)),
            "value": float(raw["leaf_value"]),
            "cover": float(raw.get("internal_count") or raw.get("leaf_count") or 0.0),
            "left": None,
            "right": None,
            "split_idx": None,
            "feature": str(raw.get("split_feature") or ""),
            "threshold": None,
        }
        return
    feature = str(raw.get("split_feature") or "")
    split_idx = raw.get("split_feature_id")
    if split_idx is None:
        split_idx = int(feature) if feature.isdigit() else 0
    nodes[node_id] = {
        "id": node_id,
        "is_leaf": False,
        "depth": int(raw.get("depth", 0)),
        "split_idx": int(split_idx),
        "feature": feature,
        "threshold": float(raw.get("threshold")),
        "cover": float(raw.get("internal_count") or 0.0),
        "left": None,
        "right": None,
        "value": None,
    }
    for branch, tag in (("left_child", "left"), ("right_child", "right")):
        child = raw.get(branch)
        if child is None:
            continue
        kid = int(child.get("node_index") or len(nodes))
        nodes[node_id][tag] = kid
        _walk_lightgbm(child, nodes)


def _limit_tree_depth(tree: dict, max_depth: int) -> dict:
    nodes = tree["nodes"]
    if max_depth <= 0:
        return nodes
    shallow = {
        node_id: dict(node)
        for node_id, node in nodes.items()
        if node["depth"] <= max_depth or node["is_leaf"]
    }
    for node in shallow.values():
        if node["depth"] == max_depth:
            node["left"] = None
            node["right"] = None
            node["is_leaf"] = True
    return shallow


def _parse_threshold(value) -> float | None:
    if value is None or value == "nan":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_decision(node: dict, feature_names: list[str]) -> str:
    names = friendly_feature_names(feature_names)
    name = node.get("feature")
    if name is None or name == "":
        idx = node["split_idx"]
        name = names[idx] if 0 <= idx < len(names) else f"variable {idx}"
    display = _friendly_feature_name(name, names)
    threshold = node.get("threshold")
    if threshold is None:
        condition = _one_hot_condition(display)
        if condition is not None:
            return f"¿{condition}?"
        return f"¿{_shorten(display, 22)}?"
    condition = _one_hot_condition(display)
    if condition is not None and threshold in (0.0, 1.0):
        return f"¿{condition}?"
    return f"{_format_feature_label(display)} < {_format_number(threshold)}?"


def _friendly_feature_name(name: str, feature_names: list[str]) -> str:
    """Resolve the most human-readable feature label for a split node.

    XGBoost often trains without explicit feature names (``f0``, ``f1``, ...)
    because the sklearn wrapper receives a preprocessed ndarray. We recover the
    real variables by position and prefer the original column name for numeric
    columns, keeping the encoded suffix only for one-hot categorical columns.
    """
    raw = str(name)
    if raw in feature_names:
        return raw
    index = _split_feature_index(raw)
    if index is not None and 0 <= index < len(feature_names):
        return feature_names[index]
    return raw


def _split_feature_index(raw: str) -> int | None:
    if raw.startswith("f") and raw[1:].lstrip("-").isdigit():
        return int(raw[1:])
    if raw.isdigit():
        return int(raw)
    return None


def _one_hot_condition(name: str) -> str | None:
    """Render a one-hot encoded split as its real category value.

    XGBoost encodes categorical splits as ``split_condition: 1`` (or ``nan``
    with the split feature itself named ``categorical__<var>_<value>``); the
    honest human-readable form is ``<var> = <value>``.
    """
    raw = str(name)
    if not raw.startswith("categorical__"):
        return None
    base = raw[len("categorical__") :]
    variable, separator, value = base.rpartition("_")
    if separator and value:
        return f"{variable} = {value}"
    return None


def friendly_feature_names(feature_names: list[str]) -> list[str]:
    """Human-readable labels for the preprocessed feature list.

    ``numeric__temp_min`` -> ``temp_min`` (the real column),
    ``categorical__region_este`` -> ``region_este`` (kept so categorical splits
    can still be recognised and rendered as ``region = este``).
    """
    return [_clean_feature(name) for name in feature_names]


def _format_feature_label(name: str) -> str:
    if name.startswith("categorical__") or name.startswith("numeric__"):
        return _clean_feature(name)
    return name


def _format_leaf(node: dict, task: str, class_names: list[str] | None) -> str:
    value = node.get("value")
    if value is None:
        return f"Pred. ≈ {_format_number(0.0)}"
    arr = np.asarray(value)
    if task == "classification" and class_names:
        if node.get("is_xgb") or arr.ndim == 0 or arr.size == 1:
            raw_leaf = float(arr.reshape(-1)[0])
            proba = 1.0 / (1.0 + np.exp(-raw_leaf))
            return f"Prob. ≈ {_format_number(proba)}"
        flat = arr.reshape(-1)
        best = int(np.argmax(flat))
        label = class_names[best] if best < len(class_names) else f"clase {best}"
        share = float(flat[best])
        if share > 0 and share <= 1:
            return f"Pred.: {_shorten(label, 16)} ({share * 100:.0f}%)"
        return f"Pred.: {_shorten(label, 18)}"
    return f"Pred. ≈ {_format_number(float(arr.reshape(-1)[0]))}"


def _compute_tree_positions(nodes: dict, width: float) -> dict[int, tuple[float, float]]:
    leaves: list[tuple[int, int]] = []
    _collect_leaves(nodes, 0, leaves)
    n_leaves = max(len(leaves), 1)
    positions: dict[int, tuple[float, float]] = {}
    depths = {n["depth"] for n in nodes.values()}
    max_depth = max(depths, default=0)
    top = 0.78
    for index, (node_id, depth) in enumerate(leaves):
        x = (index + 0.5) * width / n_leaves
        y = top + (max_depth - depth) * 1.1
        positions[node_id] = (x, y)
    for depth in range(max_depth - 1, -1, -1):
        nodes_at_level = [n for n in nodes.values() if n["depth"] == depth and not n["is_leaf"]]
        for node in nodes_at_level:
            left = positions.get(node["left"])
            right = positions.get(node["right"])
            if left and right:
                x = (left[0] + right[0]) / 2
            elif left:
                x = left[0]
            elif right:
                x = right[0]
            else:
                x = width / 2
            positions[node["id"]] = (x, top + (max_depth - depth) * 1.1)
    for node in nodes.values():
        if node["id"] not in positions:
            x = positions.get(node["left"], (width / 2, 0))[0] if node.get("left") in positions else width / 2
            positions[node["id"]] = (x, top + (max_depth - node["depth"]) * 1.1)
    return positions


def _collect_leaves(nodes: dict, node_id: int, out: list[tuple[int, int]]) -> int:
    node = nodes.get(node_id)
    if node is None:
        return 0
    if node["is_leaf"] or (node["left"] is None and node["right"] is None):
        out.append((node_id, node["depth"]))
        return 1
    count = 0
    for child in (node.get("left"), node.get("right")):
        if child is not None and child in nodes:
            count += _collect_leaves(nodes, child, out)
    if count == 0:
        out.append((node_id, node["depth"]))
        count = 1
    return count


def _node_font(n_leaves: int) -> float:
    if n_leaves <= 5:
        return 9.5
    if n_leaves <= 8:
        return 8.5
    return 7.5


def _format_number(value: float | None) -> str:
    if value is None:
        return "?"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _is_xgboost(estimator) -> bool:
    module = type(estimator).__module__
    if module.startswith("xgboost.") or module.startswith("xgboost"):
        return hasattr(estimator, "get_booster")
    return False


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _clean_feature(name: str) -> str:
    return str(name).replace("numeric__", "").replace("categorical__", "")
