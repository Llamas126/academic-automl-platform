import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ValidationFailedError
from app.ml.validation.kfold import make_folds


def test_folds_deterministic_with_seed():
    labels = np.repeat([0, 1, 2, 3], 25)
    folds_a = make_folds(100, 5, 42, stratify_labels=labels)
    folds_b = make_folds(100, 5, 42, stratify_labels=labels)
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b, strict=False):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_stratified_folds_preserve_class_balance():
    labels = np.repeat([0, 1], 50)
    folds = make_folds(100, 5, 42, stratify_labels=labels)
    for _train_idx, test_idx in folds:
        counts = pd.Series(labels[test_idx]).value_counts()
        assert list(counts) == [10, 10]


def test_regression_uses_shuffled_kfold():
    folds = make_folds(90, 3, 7, stratify_labels=None)
    assert len(folds) == 3
    all_test = np.concatenate([te for _tr, te in folds])
    assert sorted(all_test.tolist()) == list(range(90))


def test_fallback_when_class_too_small():
    labels = np.array([0] * 95 + [1])
    folds = make_folds(96, 5, 42, stratify_labels=labels)
    all_test = np.concatenate([te for _tr, te in folds])
    assert len(all_test) == 96
    assert sorted(all_test.tolist()) == list(range(96))


def test_folds_partition_dataset():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=200)
    folds = make_folds(200, 4, 11, stratify_labels=labels)
    seen = []
    for _tr, te in folds:
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(200))


def test_timeseries_folds_respect_chronology_and_gap():
    folds = make_folds(100, 5, 42, scheme="timeseries", gap=3)

    assert len(folds) == 5
    previous_test_start = -1
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert np.all(train_idx < test_idx.min())
        assert test_idx.min() >= train_idx.max() + 1 + 3
        assert np.all(np.diff(test_idx) == 1)
        assert test_idx.min() > previous_test_start
        previous_test_start = int(test_idx.min())


def test_timeseries_folds_expand_the_training_window():
    folds = make_folds(120, 4, 42, scheme="timeseries")

    train_sizes = [len(train_idx) for train_idx, _test_idx in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] > 0


def test_timeseries_ignores_stratification_and_keeps_order():
    labels = np.repeat([0, 1], 50)

    folds = make_folds(100, 5, 42, stratify_labels=labels, scheme="timeseries", gap=0)

    for train_idx, test_idx in folds:
        assert np.all(train_idx < test_idx.min())


def test_timeseries_rejects_insufficient_rows():
    with pytest.raises(ValidationFailedError):
        make_folds(5, 5, 42, scheme="timeseries")


def test_unknown_scheme_rejected():
    with pytest.raises(ValidationFailedError):
        make_folds(100, 5, 42, scheme="bogus")
