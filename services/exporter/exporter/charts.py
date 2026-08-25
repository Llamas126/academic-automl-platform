"""Diagnostic figure builders for the scientific bundle."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from automl_shared.figures import configure_publication_style, save_figure  # noqa: F401
from sklearn.metrics import auc, roc_curve

__all__ = [
    "configure_publication_style",
    "confusion_matrix_figure",
    "correlation_heatmap_figure",
    "metric_boxplot_figure",
    "predicted_vs_actual_figure",
    "residuals_figure",
    "roc_curves_figure",
    "save_figure",
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
            tick_label_kwargs={"rotation": 20, "ha": "right"},
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


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
