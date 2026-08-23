from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

SKLEARN_KEY = "pedregosa2011scikit"
XGB_KEY = "chen2016xgboost"
LGBM_KEY = "ke2017lightgbm"


class ModelEntry:
    def __init__(
        self,
        key: str,
        display_name: str,
        task: str,
        citation_key: str,
        space: Callable[[Any], dict[str, Any]],
        build: Callable[[dict[str, Any]], Any],
        needs_scaling: bool = False,
        explainer: str = "kernel",
    ) -> None:
        self.key = key
        self.display_name = display_name
        self.task = task
        self.citation_key = citation_key
        self.space = space
        self.build_fn = build
        self.needs_scaling = needs_scaling
        self.explainer = explainer

    def build(self, params: dict[str, Any], seed: int) -> Any:
        estimator = self.build_fn(params)
        try:
            if "random_state" in estimator.get_params(deep=False):
                estimator.set_params(random_state=seed)
            if "random_seed" in estimator.get_params(deep=False):
                estimator.set_params(random_seed=seed)
        except Exception:
            pass
        return estimator


def _rf_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16, 24]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _gbrt_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        "linear_regression",
        "Linear Regression",
        "regression",
        SKLEARN_KEY,
        lambda _t: {},
        lambda p: LinearRegression(),
        needs_scaling=False,
        explainer="linear",
    ),
    ModelEntry(
        "ridge_regression",
        "Ridge Regression",
        "regression",
        SKLEARN_KEY,
        lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1e2, log=True)},
        lambda p: Ridge(alpha=p["alpha"]),
        needs_scaling=True,
        explainer="linear",
    ),
    ModelEntry(
        "lasso_regression",
        "Lasso Regression",
        "regression",
        SKLEARN_KEY,
        lambda t: {
            "alpha": t.suggest_float("alpha", 1e-4, 1e1, log=True),
            "max_iter": 50000,
        },
        lambda p: Lasso(alpha=p["alpha"], max_iter=p["max_iter"]),
        needs_scaling=True,
        explainer="linear",
    ),
    ModelEntry(
        "random_forest_regressor",
        "Random Forest Regressor",
        "regression",
        SKLEARN_KEY,
        _rf_space,
        lambda p: RandomForestRegressor(n_jobs=-1, **p),
        explainer="tree",
    ),
    ModelEntry(
        "gradient_boosting_regressor",
        "Gradient Boosting Regressor",
        "regression",
        SKLEARN_KEY,
        _gbrt_space,
        lambda p: GradientBoostingRegressor(**p),
        explainer="tree",
    ),
    ModelEntry(
        "xgboost_regressor",
        "XGBoost Regressor",
        "regression",
        XGB_KEY,
        _xgb_space,
        lambda p: XGBRegressor(tree_method="hist", verbosity=0, n_jobs=-1, **p),
        explainer="tree",
    ),
    ModelEntry(
        "lightgbm_regressor",
        "LightGBM Regressor",
        "regression",
        LGBM_KEY,
        _lgbm_space,
        lambda p: LGBMRegressor(verbose=-1, n_jobs=-1, **p),
        explainer="tree",
    ),
    ModelEntry(
        "logistic_regression",
        "Logistic Regression",
        "classification",
        SKLEARN_KEY,
        lambda t: {"C": t.suggest_float("C", 1e-3, 1e3, log=True)},
        lambda p: LogisticRegression(C=p["C"], max_iter=5000),
        needs_scaling=True,
        explainer="linear",
    ),
    ModelEntry(
        "random_forest_classifier",
        "Random Forest Classifier",
        "classification",
        SKLEARN_KEY,
        _rf_space,
        lambda p: RandomForestClassifier(n_jobs=-1, class_weight="balanced", **p),
        explainer="tree",
    ),
    ModelEntry(
        "gradient_boosting_classifier",
        "Gradient Boosting Classifier",
        "classification",
        SKLEARN_KEY,
        _gbrt_space,
        lambda p: GradientBoostingClassifier(**p),
        explainer="tree",
    ),
    ModelEntry(
        "xgboost_classifier",
        "XGBoost Classifier",
        "classification",
        XGB_KEY,
        _xgb_space,
        lambda p: XGBClassifier(tree_method="hist", verbosity=0, n_jobs=-1, eval_metric="logloss", **p),
        explainer="tree",
    ),
    ModelEntry(
        "lightgbm_classifier",
        "LightGBM Classifier",
        "classification",
        LGBM_KEY,
        _lgbm_space,
        lambda p: LGBMClassifier(verbose=-1, n_jobs=-1, class_weight="balanced", **p),
        explainer="tree",
    ),
]


class UnknownModelError(Exception):
    pass


def get_entry(task: str, key: str) -> ModelEntry | None:
    for entry in REGISTRY:
        if entry.task == task and entry.key == key:
            return entry
    return None


def available_keys(task: str) -> list[str]:
    return [e.key for e in REGISTRY if e.task == task]


def resolve_entries(task: str, whitelist: list[str] | None) -> list[ModelEntry]:
    candidates = [e for e in REGISTRY if e.task == task]
    if whitelist is None:
        return candidates
    unknown = [k for k in whitelist if k not in {e.key for e in REGISTRY}]
    if unknown:
        raise UnknownModelError(
            f"unknown model key(s): {unknown}; valid keys per task: "
            f"classification={available_keys('classification')}, "
            f"regression={available_keys('regression')}"
        )
    selected = [e for e in candidates if e.key in whitelist]
    return selected
