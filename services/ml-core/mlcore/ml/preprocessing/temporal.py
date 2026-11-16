"""Causal temporal feature engineering for the ML core service.

The implementation lives in :mod:`automl_shared.temporal` so the ingestion
domain can reuse the same causal primitives without importing ``mlcore``. This
module is the public entry point used by the training pipeline.
"""

from automl_shared.temporal import (
    SUPPORTED_ROLLING_AGGS,
    add_temporal_features,
    assert_causal,
    lag_features,
    rolling_feature,
    rolling_features,
    temporal_feature_names,
)

__all__ = [
    "SUPPORTED_ROLLING_AGGS",
    "add_temporal_features",
    "assert_causal",
    "lag_features",
    "rolling_feature",
    "rolling_features",
    "temporal_feature_names",
]
