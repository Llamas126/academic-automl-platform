from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC: dict[str, str] = {
    "classification": "f1_macro",
    "regression": "rmse",
}

METRIC_DIRECTIONS: dict[str, str] = {
    "accuracy": "max",
    "balanced_accuracy": "max",
    "precision_macro": "max",
    "recall_macro": "max",
    "f1_macro": "max",
    "f1_weighted": "max",
    "roc_auc": "max",
    "matthews_corrcoef": "max",
    "cohen_kappa": "max",
    "rmse": "min",
    "mae": "min",
    "r2": "max",
    "adjusted_r2": "max",
    "mape": "min",
    "max_error": "min",
}


def primary_metric(task: str) -> str:
    return PRIMARY_METRIC[task]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float | None]:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    average = "binary" if len(labels) <= 2 else "macro"
    metrics: dict[str, float | None] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    try:
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except ValueError:
        metrics["precision_macro"] = None
        metrics["recall_macro"] = None
        metrics["f1_macro"] = None
        metrics["f1_weighted"] = None
    try:
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics["matthews_corrcoef"] = None
    try:
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    except ValueError:
        metrics["cohen_kappa"] = None
    metrics["roc_auc"] = _safe_roc_auc(y_true, y_proba)
    return metrics


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray | None) -> float | None:
    if y_proba is None:
        return None
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    try:
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        if len(unique) < y_proba.shape[1]:
            mask = np.isin(y_true, unique)
            return float(roc_auc_score(y_true[mask], y_proba[mask], multi_class="ovr", average="macro"))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        return None


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    denom = n - n_features - 1
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / denom if denom > 0 else None
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else None
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mape": mape,
        "max_error": float(max_error(y_true, y_pred)),
    }


def format_error_margin(mae: float, mape: float | None, unit: str = "kg") -> str | None:
    """Human-readable 'Margen de Error Promedio': ``±X.XX <unit> (Y.YY%)``.

    Combines MAE (absolute average deviation, in the target units) and MAPE
    (relative deviation, percent) into a single interpretable string. Returns
    ``None`` when MAPE is unavailable or either value is not finite.
    """
    if not math.isfinite(mae) or mape is None or not math.isfinite(mape):
        return None
    unit_text = f" {unit}" if unit else ""
    return f"±{mae:,.2f}{unit_text} ({mape:.1f}%)"


def aggregate_metric(name: str, folds_values: list[float | None]) -> dict:
    direction = METRIC_DIRECTIONS.get(name, "max")
    clean = [v for v in folds_values if v is not None and np.isfinite(v)]
    if not clean:
        return {
            "metric": name,
            "direction": direction,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "folds": list(folds_values),
        }
    arr = np.asarray(clean, dtype=float)
    k = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if k > 1 else 0.0
    half_width = 0.0
    if k > 1 and std > 0:
        tcrit = float(student_t.ppf(0.975, df=k - 1))
        half_width = tcrit * std / math.sqrt(k)
    return {
        "metric": name,
        "direction": direction,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "folds": [float(v) if v is not None else None for v in folds_values],
    }


def better_than(value_a: float, value_b: float, direction: str) -> bool:
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return np.isfinite(value_a)
    return value_a > value_b if direction == "max" else value_a < value_b


def confusion_matrix_aggregated(y_true: np.ndarray, y_pred: np.ndarray):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return confusion_matrix(y_true, y_pred, labels=labels).tolist(), [str(v) for v in labels]
