from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ColumnRole(StrEnum):
    numeric = "numeric"
    categorical = "categorical"
    boolean = "boolean"
    datetime = "datetime"
    identifier = "identifier"
    text_free = "text_free"


class ColumnProfile(BaseModel):
    name: str
    role: ColumnRole
    pandas_dtype: str
    missing_pct: float = Field(ge=0.0, le=100.0)
    n_unique: int
    stats: dict[str, float | str | None] = Field(default_factory=dict)
    excluded_from_features: bool = False


class TargetSuggestion(BaseModel):
    name: str
    suggested_task: Literal["classification", "regression"]
    rationale: str


class DatasetProfile(BaseModel):
    dataset_id: str
    filename: str
    created_at: datetime
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    target_suggestions: list[TargetSuggestion]
