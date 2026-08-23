import numpy as np
import pandas as pd
import pytest

from app.ml.profiling.type_detector import (
    assert_finite_targets,
    detect_column_role,
    infer_target_task,
    profile_dataframe,
    suggest_targets,
)
from app.schemas.dataset import ColumnRole


def _series(values):
    return pd.Series(values)


def test_numeric_dtype_detected():
    role, converted, excluded = detect_column_role(_series([1.0, 2.5, 3.1, np.nan]))
    assert role == ColumnRole.numeric
    assert not excluded
    assert pd.api.types.is_numeric_dtype(converted)


def test_string_numbers_coerced_to_numeric():
    role, converted, excluded = detect_column_role(_series(["1", "2", "3", "4", "x"]))
    assert role == ColumnRole.numeric
    assert not excluded
    assert converted.isna().sum() == 1


def test_datetime_strings_detected():
    values = ["2023-01-01", "2023-02-01", "2023-03-05", "2023-04-17"]
    role, converted, excluded = detect_column_role(_series(values))
    assert role == ColumnRole.datetime
    assert excluded
    assert pd.api.types.is_datetime64_any_dtype(converted)


def test_boolean_tokens_detected():
    role, converted, _ = detect_column_role(_series(["yes", "no", "YES", "no", "yes"]))
    assert role == ColumnRole.boolean
    assert set(converted.dropna().unique()) <= {True, False}


def test_identifier_column_excluded():
    values = [f"id_{i}" for i in range(50)]
    role, _, excluded = detect_column_role(_series(values))
    assert role == ColumnRole.identifier
    assert excluded


def test_low_cardinality_categorical():
    values = ["a", "b"] * 30 + [None]
    role, _, excluded = detect_column_role(_series(values))
    assert role == ColumnRole.categorical
    assert not excluded


def test_free_text_high_cardinality():
    values = [f"this is sentence number {i} about topic {i % 7}" for i in range(200)]
    role, _, excluded = detect_column_role(_series(values))
    assert role == ColumnRole.text_free
    assert excluded


def test_all_null_column_excluded():
    role, _, excluded = detect_column_role(_series([np.nan, None, np.nan]))
    assert excluded


def test_profile_dataframe_shape_and_suggestions():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "price": rng.normal(100, 20, 120),
            "quality": rng.choice(["low", "mid", "high"], 120),
            "record_id": [f"r{i:04d}" for i in range(120)],
            "notes": [f"text blob {i} lorem" for i in range(120)],
        }
    )
    profile = profile_dataframe(df, dataset_id="ds_test", filename="t.csv")
    assert profile.n_rows == 120 and profile.n_cols == 4
    roles = {c.name: c.role for c in profile.columns}
    assert roles["price"] == ColumnRole.numeric
    assert roles["quality"] == ColumnRole.categorical
    assert roles["record_id"] == ColumnRole.identifier
    tasks = {s.name: s.suggested_task for s in suggest_targets(df, roles)}
    assert tasks["price"] == "regression"
    assert tasks["quality"] == "classification"


def test_infer_target_task_regression():
    y = pd.Series(np.random.default_rng(0).normal(size=150), name="y")
    task, error = infer_target_task(y)
    assert task == "regression" and not error


def test_infer_target_task_binary_numeric_becomes_classification():
    y = pd.Series([0, 1] * 60, name="flag")
    task, _error = infer_target_task(y)
    assert task == "classification"


def test_infer_target_task_rejects_single_class():
    y = pd.Series(["a"] * 10, name="only")
    task, error = infer_target_task(y)
    assert not task and error


def test_assert_finite_targets_rejects_nan_regression():
    with pytest.raises(ValueError):
        assert_finite_targets(pd.Series([1.0, np.nan, 3.0]), "regression")


def test_assert_finite_targets_rejects_inf_regression():
    with pytest.raises(ValueError):
        assert_finite_targets(pd.Series([1.0, np.inf, 3.0]), "regression")
