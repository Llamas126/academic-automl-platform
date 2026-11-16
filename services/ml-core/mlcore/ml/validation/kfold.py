from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from automl_shared.core.exceptions import ValidationFailedError
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

logger = logging.getLogger(__name__)


def _folds_from_splitter(splitter, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(np.zeros(n_rows), None):
        folds.append((np.asarray(train_idx), np.asarray(test_idx)))
    return folds


def _stratified_folds(
    labels: np.ndarray | pd.Series, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    values = pd.Series(labels).to_numpy()
    class_counts = pd.Series(values).value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < n_folds:
        return None
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in splitter.split(np.zeros(len(values)), values)]


def _timeseries_folds(n_rows: int, n_folds: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds that never put future rows in the training set.

    ``gap`` rows are dropped between the end of each training block and the
    start of its test block, which is the barrier needed by lagged features.
    """
    try:
        splitter = TimeSeriesSplit(n_splits=n_folds, gap=gap)
        folds = _folds_from_splitter(splitter, n_rows)
    except ValueError as exc:
        raise ValidationFailedError(
            f"timeseries cross-validation needs more rows for {n_folds} folds "
            f"with gap={gap}: {exc}"
        ) from exc
    if not folds or any(len(train) == 0 or len(test) == 0 for train, test in folds):
        raise ValidationFailedError(
            f"timeseries cross-validation produced empty folds for {n_rows} rows, "
            f"{n_folds} folds and gap={gap}"
        )
    return folds


def make_folds(
    n_rows: int,
    n_folds: int,
    seed: int,
    stratify_labels: np.ndarray | pd.Series | None = None,
    *,
    scheme: str = "kfold",
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if scheme == "timeseries":
        if stratify_labels is not None:
            logger.info("timeseries scheme ignores stratification to preserve chronological order")
        return _timeseries_folds(n_rows, n_folds, max(0, int(gap)))
    if scheme != "kfold":
        raise ValidationFailedError(f"unknown cross-validation scheme '{scheme}'")
    if stratify_labels is not None:
        folds = _stratified_folds(stratify_labels, n_folds, seed)
        if folds is not None:
            return folds
        logger.info(
            "stratification not feasible with %d folds; falling back to shuffled KFold",
            n_folds,
        )
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return _folds_from_splitter(splitter, n_rows)


def build_combined_stratification(specs) -> np.ndarray | None:
    classification_specs = [s for s in specs if s.task == "classification"]
    if not classification_specs:
        return None
    encoded_columns = [s.y_encoded.to_numpy() for s in classification_specs]
    combined = np.array(["|".join(map(str, row)) for row in zip(*encoded_columns, strict=False)])
    return combined


def fold_report(folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, int]]:
    return [{"fold": i, "n_train": int(len(tr)), "n_test": int(len(te))} for i, (tr, te) in enumerate(folds)]
