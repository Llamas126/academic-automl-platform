import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app.ml.explainability.shap_engine import build_shap_figures
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
