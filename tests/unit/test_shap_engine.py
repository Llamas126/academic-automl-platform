import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app.ml.explainability.shap_engine import (
    build_interpretability_profile,
    build_shap_figures,
)
from app.ml.preprocessing.builders import build_preprocessor
from app.services import figure_service


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


def test_shap_linear_produces_png_and_eps(tmp_path):
    pipe, X = _fitted_ridge_pipeline()
    figure_service.configure_publication_style(300)
    written = build_shap_figures(
        pipe,
        X.head(60),
        explainer_kind="linear",
        target_name="score",
        out_dir=tmp_path,
        include_eps=True,
        include_dependence=True,
        max_display=10,
    )
    stems = {p.name for p in written}
    assert "shap_importance_score.png" in stems
    assert "shap_importance_score.eps" in stems
    assert any(n.startswith("shap_beeswarm") for n in stems)
    for path in written:
        assert path.stat().st_size > 500


def test_shap_failure_degrades_gracefully(tmp_path):
    from sklearn.dummy import DummyRegressor

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=40)})
    y = rng.normal(size=40)
    pipe = Pipeline([("preprocess", build_preprocessor(["a"], [], False)), ("estimator", DummyRegressor())])
    pipe.fit(X, y)
    written = build_shap_figures(pipe, X, "tree", "y", tmp_path, max_display=5)
    assert written == []


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
