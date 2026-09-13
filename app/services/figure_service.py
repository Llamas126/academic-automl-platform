"""Compatibility shim: figure rendering + diagnostic chart builders."""
from automl_shared.figures import (
    _PUBLICATION_RC,
    configure_publication_style,
    save_figure,
)
from exporter.charts import (
    coefficients_figure,
    confusion_matrix_figure,
    correlation_heatmap_figure,
    feature_importance_figure,
    metric_boxplot_figure,
    model_feature_importance_comparison_figure,
    predicted_vs_actual_figure,
    residuals_figure,
    roc_curves_figure,
    shap_beeswarm_figure,
)

__all__ = [
    "_PUBLICATION_RC",
    "coefficients_figure",
    "configure_publication_style",
    "confusion_matrix_figure",
    "correlation_heatmap_figure",
    "feature_importance_figure",
    "metric_boxplot_figure",
    "model_feature_importance_comparison_figure",
    "predicted_vs_actual_figure",
    "residuals_figure",
    "roc_curves_figure",
    "save_figure",
    "shap_beeswarm_figure",
]
