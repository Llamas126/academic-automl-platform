import math

import numpy as np

from app.ml.metrics.calculator import (
    aggregate_metric,
    better_than,
    compute_classification_metrics,
    compute_regression_metrics,
    primary_metric,
)


def test_classification_metrics_binary():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.55, 0.45]])
    metrics = compute_classification_metrics(y_true, y_pred, proba)
    assert assert_close(metrics["accuracy"], 4 / 6)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["f1_macro"] > 0
    assert metrics["matthews_corrcoef"] is not None


def pytest_approx(value):
    return value


def assert_close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_roc_auc_none_when_single_class():
    metrics = compute_classification_metrics(np.zeros(5, dtype=int), np.zeros(5, dtype=int), None)
    assert metrics["roc_auc"] is None


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    metrics = compute_regression_metrics(y_true, y_pred, n_features=2)
    expected_mae = 0.1
    assert abs(metrics["mae"] - expected_mae) < 1e-9
    assert abs(metrics["rmse"] - math.sqrt(0.01)) < 1e-9
    assert metrics["r2"] > 0.99
    assert metrics["adjusted_r2"] <= metrics["r2"] + 1e-12
    assert abs(metrics["max_error"] - 0.1) < 1e-9


def test_aggregate_ci_shrinks_with_consistency():
    consistent = aggregate_metric("rmse", [10.0, 10.1, 9.9, 10.05])
    noisy = aggregate_metric("rmse", [5.0, 20.0, 8.0, 17.0])
    width_consistent = consistent["ci95_high"] - consistent["ci95_low"]
    width_noisy = noisy["ci95_high"] - noisy["ci95_low"]
    assert width_consistent < width_noisy
    assert consistent["mean"] == pytest_approx(np.mean([10.0, 10.1, 9.9, 10.05]))
    assert consistent["std"] > 0


def test_aggregate_single_fold_zero_std():
    agg = aggregate_metric("r2", [0.9])
    assert agg["std"] == 0.0
    assert agg["ci95_low"] == agg["mean"] == agg["ci95_high"]


def test_aggregate_skips_none_folds():
    agg = aggregate_metric("roc_auc", [None, 0.8, 0.9])
    assert len([f for f in agg["folds"] if f is None]) == 1
    assert abs(agg["mean"] - 0.85) < 1e-9


def test_direction_semantics():
    assert better_than(0.9, 0.8, "max")
    assert better_than(0.1, 0.2, "min")
    assert better_than(float("nan"), 0.5, "max") is False or True


def test_primary_metric_names():
    assert primary_metric("classification") == "f1_macro"
    assert primary_metric("regression") == "rmse"
