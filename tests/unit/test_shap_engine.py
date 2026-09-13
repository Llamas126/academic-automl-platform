import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app.ml.explainability.shap_engine import (
    build_interpretability_profile,
    compute_shap_values,
)
from app.ml.preprocessing.builders import build_preprocessor


def _fitted_ridge_pipeline():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
            "cat": rng.choice(["a", "b", "c"], 120),
        }
    )
    y = 3 * X["x1"] - 2 * X["x2"] + rng.normal(0, 0.1, 120)
    pre = build_preprocessor(["x1", "x2"], ["cat"], scale_numeric=False)
    pipe = Pipeline([("preprocess", pre), ("estimator", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    return pipe, X


def _training_meta(targets: list[dict]) -> dict:
    return {"schema_version": 1, "targets": targets}


def _write_contract(tmp_path, targets: list[dict], pipelines: dict[str, object]) -> Path:
    import joblib
    from automl_shared.contract import META_NAME, safe_target_stem

    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "job.db").write_text("mock-db", encoding="utf-8")
    (tmp_path / META_NAME).write_text(json.dumps(_training_meta(targets)), encoding="utf-8")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    for name, pipe in pipelines.items():
        joblib.dump({"pipeline": pipe}, tmp_path / "models" / f"{safe_target_stem(name)}_best_model.joblib")
    return tmp_path


def test_compute_shap_values_linear_returns_arrays():
    pipe, X = _fitted_ridge_pipeline()
    result = compute_shap_values(pipe, X.head(60), "linear")
    assert result is not None
    values, multiclass, eval_data, feature_names = result
    assert multiclass is False
    assert values.shape == eval_data.shape
    assert len(feature_names) == values.shape[1]


def test_render_profiles_persists_without_legacy_figures(tmp_path):
    from explainability.rendering import render_interpretability_profiles_if_requested

    from app.schemas.training import TrainRequest

    pipe, X = _fitted_ridge_pipeline()
    _write_contract(
        tmp_path,
        [{"target": "score", "best_entry_key": "ridge_regression", "explainer_kind": "linear"}],
        {"score": pipe},
    )
    X.to_csv(tmp_path / "background.csv", index=False)
    request = TrainRequest(
        dataset_id="ds_test",
        targets=[{"name": "score", "task": "regression"}],
        models=["ridge_regression"],
    )

    profiled = render_interpretability_profiles_if_requested(
        work_dir=tmp_path, out_dir=tmp_path, request=request
    )
    assert profiled == 1

    profiled_file = tmp_path / "interpretability" / "score.json"
    assert profiled_file.exists()
    index = json.loads(profiled_file.read_text(encoding="utf-8"))
    assert index["target"] == "score"
    assert len(index["models"]) == 1
    assert index["models"][0]["coefficients"] is not None

    legacy = list(tmp_path.rglob("shap_importance_*.png")) + list(tmp_path.rglob("shap_beeswarm_*.png"))
    assert legacy == [], "legacy SHAP figures must no longer be generated"


def _fitted_rf_multiclass_pipeline():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=150),
            "x2": rng.normal(size=150),
            "cat": rng.choice(["a", "b", "c"], 150),
        }
    )
    y = rng.choice(["low", "mid", "high"], 150)
    pre = build_preprocessor(["x1", "x2"], ["cat"], scale_numeric=False)
    pipe = Pipeline(
        [("preprocess", pre), ("estimator", RandomForestClassifier(n_estimators=5, random_state=0))]
    )
    pipe.fit(X, y)
    return pipe, X


def test_interpretability_profile_linear_exposes_coefficients():
    pipe, X = _fitted_ridge_pipeline()
    profile = build_interpretability_profile(pipe, X.head(60), "linear")
    assert profile["kind"] == "linear"
    assert profile["coefficients"] is not None
    assert np.asarray(profile["coefficients"]).shape[-1] == len(profile["feature_names"])
    assert profile["importance"] is not None
    assert np.asarray(profile["importance"]).shape[0] == len(profile["feature_names"])


def test_interpretability_profile_tree_exposes_shap_arrays():
    pipe, X = _fitted_rf_multiclass_pipeline()
    profile = build_interpretability_profile(pipe, X.head(60), "tree")
    assert profile["kind"] == "tree"
    assert profile["multiclass"] is True
    assert profile["has_shap"] is True
    assert profile["importance"] is not None
    assert np.asarray(profile["importance"]).shape[0] == len(profile["feature_names"])
    assert profile["shap_values"] is None or np.asarray(profile["shap_values"]).shape[-1] == len(
        profile["feature_names"]
    )


def test_interpretability_profile_importance_aligned_to_features():
    pipe, X = _fitted_rf_multiclass_pipeline()
    profile = build_interpretability_profile(pipe, X.head(60), "tree")
    assert np.asarray(profile["importance"]).ndim == 1
    assert np.asarray(profile["importance"]).shape == (len(profile["feature_names"]),)
