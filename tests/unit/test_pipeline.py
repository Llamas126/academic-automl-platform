import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ValidationFailedError
from app.ml.pipeline import run_training
from app.schemas.training import CVConfig, HPOConfig, TargetIn, TrainRequest


@pytest.fixture
def tiny_mixed_df():
    rng = np.random.default_rng(11)
    n = 90
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    category = rng.choice(["red", "green", "blue"], n)
    df = pd.DataFrame(
        {
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "category": category,
            "value": np.round(3 * x1 - 2 * x2 + 5 + rng.normal(0, 0.4, n), 4),
            "label": pd.cut(
                x1 + rng.normal(0, 0.25, n), bins=[-np.inf, -0.33, 0.33, np.inf], labels=["lo", "mid", "hi"]
            ).astype(str),
        }
    )
    return df


def _request(**overrides):
    payload = dict(
        dataset_id="ds_test",
        targets=[
            TargetIn(name="value", task="regression"),
            TargetIn(name="label", task="classification"),
        ],
        models=[
            "ridge_regression",
            "random_forest_regressor",
            "logistic_regression",
            "random_forest_classifier",
        ],
        cv=CVConfig(n_folds=3, mode="single", random_state=42),
        hpo=HPOConfig(enabled=False),
    )
    payload.update(overrides)
    return TrainRequest(**payload)


def test_run_training_single_mode_no_hpo(tiny_mixed_df):
    output = run_training(_request(), "ds_test", "tiny.csv", tiny_mixed_df, "job_test")
    summary = output.summary

    assert {t.target: t.task for t in summary.targets} == {"value": "regression", "label": "classification"}
    for target_results in summary.targets:
        ranks = sorted(m.rank for m in target_results.models)
        assert ranks == [1, 2]
        best = next(m for m in target_results.models if m.rank == 1)
        assert target_results.best_model == best.model
        for metric in best.metrics:
            assert metric.ci95_low <= metric.mean + 1e-12 <= metric.ci95_high + 1e-12
        if target_results.task == "regression":
            assert all(
                m.avg_error_margin is not None and m.avg_error_margin.startswith("±")
                for m in target_results.models
            )
        else:
            assert all(m.avg_error_margin is None for m in target_results.models)

    pred_frame = output.predictions_frame
    assert len(pred_frame) == len(tiny_mixed_df)
    assert "fold_id" in pred_frame.columns
    assert set(pred_frame["fold_id"].unique()) <= {0, 1, 2}
    assert (pred_frame["fold_id"] >= 0).all()
    assert "value_predicted" in pred_frame.columns
    assert "label_predicted" in pred_frame.columns

    artifacts = output.artifacts
    assert artifacts["value"].final_pipeline is not None
    assert artifacts["label"].confusion_matrix is not None
    assert sum(map(sum, artifacts["label"].confusion_matrix)) == len(tiny_mixed_df)


def test_run_training_drops_rows_with_missing_targets():
    df = _tiny_with_missing()
    request = _request()
    request.targets = [TargetIn(name="value", task="auto")]
    output = run_training(request, "ds_x", "t.csv", df, "job_missing")
    assert any("rows dropped" in w for w in output.summary.warnings)
    assert output.summary.n_samples < len(df)


def test_predictions_frame_keeps_datetime_context_column():
    df = _tiny_mixed_with_date()
    output = run_training(_request(), "ds_test", "tiny_date.csv", df, "job_date")

    assert "fecha" not in {t.target for t in output.summary.targets}
    assert "fecha" not in set(output.summary.feature_columns)

    pred_frame = output.predictions_frame
    assert "fecha" in pred_frame.columns
    assert len(pred_frame) == len(df)
    expected = pd.to_datetime(df["fecha"], errors="coerce")
    pd.testing.assert_series_equal(
        pd.to_datetime(pred_frame["fecha"], errors="coerce").reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_run_training_drops_rows_with_missing_targets_and_keeps_date():
    df = _tiny_mixed_with_date()
    df.loc[[3, 17], "value"] = np.nan
    request = _request()
    request.targets = [TargetIn(name="value", task="auto")]
    output = run_training(request, "ds_x", "t_date.csv", df, "job_missing_date")

    pred_frame = output.predictions_frame
    assert "fecha" in pred_frame.columns
    assert len(pred_frame) < len(df)
    assert "fold_id" in pred_frame.columns
    assert len(pred_frame) == output.summary.n_samples


def test_auto_task_inference(tiny_mixed_df):
    request = TrainRequest(
        dataset_id="ds_test",
        targets=[TargetIn(name="value"), TargetIn(name="label")],
        models=["ridge_regression", "logistic_regression"],
        cv=CVConfig(n_folds=3, mode="single"),
        hpo=HPOConfig(enabled=False),
    )
    output = run_training(request, "ds_test", "tiny.csv", tiny_mixed_df, "job_auto")
    tasks = {t.target: t.task for t in output.summary.targets}
    assert tasks == {"value": "regression", "label": "classification"}


def test_unknown_model_key_rejected(tiny_mixed_df):
    with pytest.raises(ValidationFailedError):
        run_training(
            _request(models=["nonexistent_model"]),
            "ds_test",
            "tiny.csv",
            tiny_mixed_df,
            "job_bad",
        )


def test_nested_mode_with_hpo(tiny_mixed_df):
    request = TrainRequest(
        dataset_id="ds_test",
        targets=[TargetIn(name="value", task="regression"), TargetIn(name="label", task="classification")],
        models=["ridge_regression", "logistic_regression"],
        cv=CVConfig(n_folds=2, mode="nested", n_inner_folds=2, random_state=7),
        hpo=HPOConfig(enabled=True, n_trials=3, timeout_s=None),
    )
    output = run_training(request, "ds_test", "tiny.csv", tiny_mixed_df, "job_nested")
    summary = output.summary
    assert summary.cv["mode"] == "nested"
    for target_results in summary.targets:
        assert target_results.best_params
        per_fold_with_params = [r for r in output.per_fold_records[target_results.target] if "params" in r]
        assert len(per_fold_with_params) == 2
    stages_seen = []
    run_training(
        request,
        "ds_test",
        "tiny.csv",
        tiny_mixed_df,
        "job_progress",
        progress_cb=lambda stage, frac: stages_seen.append((stage, round(frac, 2))),
    )
    assert stages_seen[0][0] == "profiling"
    assert stages_seen[-1][0] == "done"
    fractions = [f for _stage, f in stages_seen]
    assert all(b >= a for a, b in zip(fractions, fractions[1:], strict=False))
    assert 0.0 <= fractions[-1] <= 1.0


def test_determinism_same_seed_same_results(tiny_mixed_df):
    request = _request()
    out_a = run_training(request, "ds_test", "tiny.csv", tiny_mixed_df, "job_a")
    out_b = run_training(_request(), "ds_test", "tiny.csv", tiny_mixed_df, "job_b")
    means_a = {t.target: t.models[0].primary_mean for t in out_a.summary.targets}
    means_b = {t.target: t.models[0].primary_mean for t in out_b.summary.targets}
    assert means_a.keys() == means_b.keys()
    for key in means_a:
        assert abs(means_a[key] - means_b[key]) < 1e-9


def test_duplicate_target_rejected(tiny_mixed_df):
    with pytest.raises(ValidationFailedError):
        run_training(
            _request(targets=[TargetIn(name="value"), TargetIn(name="value")]),
            "ds_test",
            "tiny.csv",
            tiny_mixed_df,
            "job_dup",
        )


def _tiny_with_missing():
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "value": rng.normal(size=n) + 10,
        }
    )
    df.loc[[3, 17, 41], "value"] = np.nan
    return df


def _tiny_mixed_with_date():
    rng = np.random.default_rng(11)
    n = 90
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2023-01-01", periods=n, freq="D"),
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "category": ["red", "green", "blue"] * (n // 3) + ["red", "green"][: n % 3],
            "value": np.round(3 * x1 - 2 * x2 + 5 + rng.normal(0, 0.4, n), 4),
            "label": pd.cut(
                x1 + rng.normal(0, 0.25, n), bins=[-np.inf, -0.33, 0.33, np.inf], labels=["lo", "mid", "hi"]
            ).astype(str),
        }
    )
    return df
