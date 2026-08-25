import matplotlib.pyplot as plt
from exporter.charts import metric_boxplot_figure


def _fold_records(models: list[str], metric: str, values_per_model: dict[str, list[float]]):
    records = []
    for model in models:
        for fold_idx, val in enumerate(values_per_model.get(model, [])):
            records.append({"model": model, "fold": fold_idx, metric: val})
    return records


def test_boxplot_valid_data():
    records = _fold_records(
        ["RF", "XGB"], "rmse", {"RF": [10.0, 12.0, 11.0], "XGB": [9.0, 11.0, 10.5]}
    )
    fig = metric_boxplot_figure(records, "rmse", ["RF", "XGB"])
    ax = fig.axes[0]
    assert ax.get_ylabel() == "rmse"
    assert "rmse distribution across folds" in ax.get_title()
    assert len(ax.lines) > 0
    plt.close(fig)


def test_boxplot_metric_not_in_records():
    records = _fold_records(["RF"], "rmse", {"RF": [10.0, 12.0]})
    fig = metric_boxplot_figure(records, "nonexistent_metric", ["RF"])
    ax = fig.axes[0]
    assert len(ax.lines) == 0
    plt.close(fig)


def test_boxplot_none_and_nan_filtered():
    records = [
        {"model": "RF", "fold": 0, "rmse": 10.0},
        {"model": "RF", "fold": 1, "rmse": None},
        {"model": "RF", "fold": 2, "rmse": float("nan")},
        {"model": "RF", "fold": 3, "rmse": 11.0},
    ]
    fig = metric_boxplot_figure(records, "rmse", ["RF"])
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)


def test_boxplot_empty_records():
    fig = metric_boxplot_figure([], "rmse", [])
    ax = fig.axes[0]
    assert ax.get_ylabel() == "rmse"
    plt.close(fig)


def test_boxplot_single_model():
    records = _fold_records(["RF"], "f1_macro", {"RF": [0.85, 0.87, 0.86, 0.88, 0.84]})
    fig = metric_boxplot_figure(records, "f1_macro", ["RF"])
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)


def test_boxplot_many_models():
    models = [f"Model_{i}" for i in range(20)]
    values = {m: [float(i) for i in range(5)] for m in models}
    records = _fold_records(models, "rmse", values)
    fig = metric_boxplot_figure(records, "rmse", models)
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close(fig)
