"""Compatibility shim: figure rendering + diagnostic chart builders."""
from automl_shared.figures import (
    _PUBLICATION_RC,
    configure_publication_style,
    save_figure,
)
from exporter.charts import (
    confusion_matrix_figure,
    correlation_heatmap_figure,
    metric_boxplot_figure,
    predicted_vs_actual_figure,
    residuals_figure,
    roc_curves_figure,
)

__all__ = [
    "_PUBLICATION_RC",
    "configure_publication_style",
    "confusion_matrix_figure",
    "correlation_heatmap_figure",
    "metric_boxplot_figure",
    "predicted_vs_actual_figure",
    "residuals_figure",
    "roc_curves_figure",
    "save_figure",
]
