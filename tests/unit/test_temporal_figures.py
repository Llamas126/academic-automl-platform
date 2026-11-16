import matplotlib.pyplot as plt
import numpy as np
from exporter.charts import predicted_vs_actual_figure, residuals_figure, roc_curves_figure


def _close(fig):
    plt.close(fig)


def test_diagnostic_figures_ignore_untested_nan_rows():
    y_true = np.array([1.0, 2.0, 3.0, np.nan])
    y_pred = np.array([1.1, np.nan, 3.2, 2.5])

    pva = predicted_vs_actual_figure(y_true, y_pred, "value")
    res = residuals_figure(y_true, y_pred, "value")

    assert pva.axes[0].get_title() == "Out-of-fold predictions: value"
    assert res.axes[0].get_title() == "Residual analysis: value"
    _close(pva)
    _close(res)


def test_roc_figure_ignores_untested_nan_rows():
    y_true = np.array([0, 1, 0, 1])
    proba = np.array([[0.8, 0.2], [np.nan, np.nan], [0.3, 0.7], [0.1, 0.9]])

    fig = roc_curves_figure(y_true, proba, ["no", "yes"], "ROC")

    assert fig.axes[0].get_title() == "ROC"
    assert len(fig.axes[0].lines) >= 1
    _close(fig)


def test_roc_figure_handles_no_evaluated_rows():
    y_true = np.array([0, 1])
    proba = np.full((2, 2), np.nan)

    fig = roc_curves_figure(y_true, proba, ["no", "yes"], "ROC")

    assert fig.axes[0].get_title() == "ROC"
    _close(fig)
