import pytest

from app.ml.models.registry import (
    REGISTRY,
    UnknownModelError,
    available_keys,
    get_entry,
    resolve_entries,
)
from app.ml.search.optuna_tuner import default_params_via_fixed_trial, tune_hyperparams


def test_registry_has_both_tasks():
    assert len(available_keys("regression")) >= 5
    assert len(available_keys("classification")) >= 5


def test_get_entry_respects_task():
    entry = get_entry("regression", "ridge_regression")
    assert entry is not None and entry.task == "regression"
    assert get_entry("classification", "ridge_regression") is None


def test_resolve_entries_whitelist_unknown_raises():
    with pytest.raises(UnknownModelError):
        resolve_entries("classification", ["does_not_exist"])


def test_resolve_entries_whitelist_filters_by_task():
    entries = resolve_entries("regression", ["ridge_regression", "logistic_regression"])
    keys = [e.key for e in entries]
    assert keys == ["ridge_regression"]


def test_fixed_trial_defaults_build_all_models():
    for entry in REGISTRY:
        params = default_params_via_fixed_trial(entry)
        estimator = entry.build(params, seed=42)
        fitted_params = estimator.get_params(deep=False)
        assert "random_state" in fitted_params or not hasattr(estimator, "_estimator_type")


def test_tune_hyperparams_regression_smoke():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(3, 1, 80)})
    y = pd.Series(2 * X["a"] + 0.5 * X["b"] + rng.normal(0, 0.1, 80))
    entry = get_entry("regression", "ridge_regression")
    params = tune_hyperparams(
        entry,
        X,
        y,
        numeric_cols=["a", "b"],
        categorical_cols=[],
        n_inner_folds=3,
        seed=42,
        n_trials=4,
        timeout_s=None,
    )
    assert "alpha" in params
