import matplotlib.pyplot as plt
import numpy as np
from exporter.charts import (
    coefficients_figure,
    decision_tree_flowchart_figure,
    feature_importance_figure,
    model_feature_importance_comparison_figure,
    shap_beeswarm_figure,
)


def _close(fig):
    plt.close(fig)


def test_coefficients_figure_1d():
    fig = coefficients_figure(np.array([0.5, -1.2, 0.3]), ["x1", "x2", "x3"], "Ridge Regression")
    ax = fig.axes[0]
    assert "Coefficient (beta)" in ax.get_xlabel()
    assert len(ax.containers) > 0
    _close(fig)


def test_coefficients_figure_multiclass():
    coef = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.0]])
    fig = coefficients_figure(coef, ["x1", "x2"], "Logistic Regression")
    ax = fig.axes[0]
    assert len(ax.get_legend().get_texts()) == 3
    _close(fig)


def test_feature_importance_figure_sorts_descending():
    fig = feature_importance_figure(
        np.array([0.1, 0.9, 0.5]), ["low", "top", "mid"], "Random Forest", source="shap"
    )
    ax = fig.axes[0]
    assert "Mean |SHAP value|" in ax.get_xlabel()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[0] == "top"
    _close(fig)


def test_shap_beeswarm_figure():
    rng = np.random.default_rng(3)
    shap_values = rng.normal(size=(50, 4))
    feature_values = rng.normal(size=(50, 4))
    fig = shap_beeswarm_figure(shap_values, feature_values, ["a", "b", "c", "d"], "GBRT")
    ax = fig.axes[0]
    assert "SHAP value" in ax.get_xlabel()
    collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
    assert collections and len(collections) == 4
    _close(fig)


def test_comparison_figure_normalizes_within_model():
    profiles = [
        {"key": "ridge", "display_name": "Ridge", "importance": np.array([2.0, 1.0, 0.5])},
        {"key": "rf", "display_name": "Random Forest", "importance": np.array([0.1, 0.6, 0.3])},
    ]
    fig = model_feature_importance_comparison_figure(profiles, ["x1", "x2", "x3"])
    ax = fig.axes[0]
    assert len(ax.collections) > 0
    _close(fig)


def test_comparison_figure_empty_profiles():
    fig = model_feature_importance_comparison_figure([], ["x1", "x2"])
    assert len(fig.axes) == 1
    _close(fig)


def _small_mixed_tree_data():
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "temp_min": rng.normal(size=n),
            "temp_media": rng.normal(size=n),
            "region": rng.choice(["norte", "sur", "este"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    y = 2 * df["temp_min"] - 0.4 * df["temp_media"] + (df["region"] == "este").astype(int)
    target = y + rng.normal(scale=0.1, size=n)
    classes = (target > 0).astype(int)

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["temp_min", "temp_media"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["region"],
            ),
        ]
    )
    Xt = preprocessor.fit_transform(df)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    return Xt, feature_names, target, classes


def test_decision_tree_flowchart_xgboost_uses_clean_names():
    from xgboost import XGBClassifier

    Xt, feature_names, _target, classes = _small_mixed_tree_data()
    estimator = XGBClassifier(tree_method="hist", verbosity=0, max_depth=5, n_estimators=2).fit(
        Xt, classes
    )
    fig = decision_tree_flowchart_figure(
        estimator,
        feature_names,
        "XGBoost Classifier",
        target="churn",
        task="classification",
        class_names=["no", "si"],
        max_depth=2,
    )
    assert fig is not None
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("Sí" in text for text in texts)
    assert any("No" in text for text in texts)
    assert any("temp_min" in text for text in texts)
    assert not any("numeric__" in text for text in texts)
    _close(fig)


def test_decision_tree_flowchart_one_hot_condition_readable():
    from xgboost import XGBClassifier

    Xt, feature_names, _target, classes = _small_mixed_tree_data()
    estimator = XGBClassifier(tree_method="hist", verbosity=0, max_depth=5, n_estimators=1).fit(
        Xt, classes
    )
    fig = decision_tree_flowchart_figure(
        estimator,
        feature_names,
        "XGBoost Classifier",
        target="churn",
        task="classification",
        class_names=["no", "si"],
        max_depth=2,
    )
    assert fig is not None
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("region" in text for text in texts)
    assert not any("categorical__" in text for text in texts)
    _close(fig)


def test_decision_tree_flowchart_sklearn_forest():
    from sklearn.ensemble import RandomForestRegressor

    Xt, feature_names, target, _classes = _small_mixed_tree_data()
    estimator = RandomForestRegressor(max_depth=5, n_estimators=2, random_state=0).fit(Xt, target)
    fig = decision_tree_flowchart_figure(
        estimator,
        feature_names,
        "Random Forest",
        target="produccion",
        task="regression",
        max_depth=2,
    )
    assert fig is not None
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("temp_min" in text for text in texts)
    assert any("≈" in text for text in texts)
    _close(fig)


def test_decision_tree_flowchart_non_tree_returns_none():
    from sklearn.linear_model import Ridge

    Xt, feature_names, target, _classes = _small_mixed_tree_data()
    estimator = Ridge().fit(Xt, target)
    fig = decision_tree_flowchart_figure(estimator, feature_names, "Ridge", task="regression")
    assert fig is None


def test_decision_tree_flowchart_spreads_four_leaves_horizontally():
    from matplotlib.patches import FancyBboxPatch
    from sklearn.ensemble import RandomForestRegressor

    Xt, feature_names, target, _classes = _small_mixed_tree_data()
    estimator = RandomForestRegressor(max_depth=5, n_estimators=2, random_state=0).fit(Xt, target)
    fig = decision_tree_flowchart_figure(
        estimator,
        feature_names,
        "Random Forest",
        target="produccion",
        task="regression",
        max_depth=2,
    )
    assert fig is not None
    ax = fig.axes[0]
    boxes = [(box.get_x(), box.get_y()) for box in ax.patches if isinstance(box, FancyBboxPatch)]
    boxes.sort(key=lambda point: float(point[1]))
    leaf_row = boxes[:4]
    leaf_x_edges = sorted(round(float(x), 2) for x, _ in leaf_row)
    expected_gap = round(12.0 / 4.0, 2)
    assert [round(leaf_x_edges[i + 1] - leaf_x_edges[i], 2) for i in range(len(leaf_x_edges) - 1)] == [
        expected_gap
    ] * 3, leaf_x_edges
    assert leaf_x_edges[0] < 12.0 / 8.0, leaf_x_edges
    assert len({round(float(y), 2) for _, y in leaf_row}) == 1, "pruned leaves must share a row"
    _close(fig)


def test_decision_tree_flowchart_canvas_is_12x6():
    from sklearn.ensemble import RandomForestRegressor

    Xt, feature_names, target, _classes = _small_mixed_tree_data()
    estimator = RandomForestRegressor(max_depth=5, n_estimators=2, random_state=0).fit(Xt, target)
    fig = decision_tree_flowchart_figure(
        estimator,
        feature_names,
        "Random Forest",
        target="produccion",
        task="regression",
        max_depth=2,
    )
    assert fig is not None
    assert fig.get_size_inches().tolist() == [12.0, 6.0]
    _close(fig)
