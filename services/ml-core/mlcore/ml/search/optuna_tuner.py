from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from mlcore.ml.models.registry import ModelEntry
from mlcore.ml.preprocessing.builders import build_preprocessor
from mlcore.ml.validation.kfold import make_folds

logger = logging.getLogger(__name__)


class TuningFailure(Exception):
    pass


def _score_max(y_pred: np.ndarray, y_true: np.ndarray, task_hint: str) -> float:
    from sklearn.metrics import f1_score, r2_score

    if task_hint == "classification":
        labels = np.unique(y_true)
        average = "binary" if len(labels) <= 2 else "macro"
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    return float(r2_score(y_true, y_pred))


def _score_min(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class _DeterministicDefaults:
    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        value = float(low)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        value = int(low)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list) -> Any:
        value = choices[0]
        self.params[name] = value
        return value


def default_params_via_fixed_trial(entry: ModelEntry) -> dict[str, Any]:
    shim = _DeterministicDefaults()
    params = entry.space(shim)
    return {**shim.params, **params}


def tune_hyperparams(
    entry: ModelEntry,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_inner_folds: int,
    seed: int,
    n_trials: int,
    timeout_s: int | None,
) -> dict[str, Any]:
    y_arr = np.asarray(y)
    stratify = y_arr if entry.task == "classification" else None
    folds = make_folds(len(X), n_inner_folds, seed, stratify_labels=stratify)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial: optuna.Trial) -> float:
        params = entry.space(trial)
        values: list[float] = []
        for train_idx, test_idx in folds:
            estimator = entry.build(params, seed)
            preprocessor = build_preprocessor(
                numeric_cols, categorical_cols, scale_numeric=entry.needs_scaling
            )
            pipe = Pipeline([("preprocess", preprocessor), ("estimator", estimator)])
            pipe.fit(X.iloc[train_idx], y_arr[train_idx])
            predictions = pipe.predict(X.iloc[test_idx])
            if entry.task == "classification":
                values.append(_score_max(predictions, y_arr[test_idx], "classification"))
            else:
                values.append(-_score_min(predictions, y_arr[test_idx]))
        return float(np.mean(values)) if values else float("-inf")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        catch=(Exception,),
        show_progress_bar=False,
        gc_after_trial=False,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise TuningFailure(f"all {n_trials} Optuna trials failed for model '{entry.key}'")
    best_trial = max(completed, key=lambda t: t.value)
    logger.info(
        "HPO %s/%s: best value %.4f in %d trials",
        entry.task,
        entry.key,
        best_trial.value,
        len(completed),
    )
    return dict(best_trial.params)
