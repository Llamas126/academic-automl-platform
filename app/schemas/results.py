from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MetricAggregate(BaseModel):
    metric: str
    direction: Literal["max", "min"]
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    folds: list[float]


class ModelResult(BaseModel):
    model: str
    citation_key: str | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    primary_metric: str
    primary_mean: float
    rank: int
    metrics: list[MetricAggregate]


class TargetResults(BaseModel):
    target: str
    task: Literal["classification", "regression"]
    label_encoder_classes: list[str] | None = None
    best_model: str
    best_params: dict[str, Any] = Field(default_factory=dict)
    models: list[ModelResult]


class RunSummary(BaseModel):
    job_id: str
    dataset_id: str
    filename: str
    n_samples: int
    n_features: int
    feature_columns: list[str]
    cv: dict[str, Any]
    hpo: dict[str, Any]
    targets: list[TargetResults]
    warnings: list[str] = Field(default_factory=list)
    library_versions: dict[str, str] = Field(default_factory=dict)
    runtime_s: float = 0.0
    finished_at: datetime | None = None
