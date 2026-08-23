from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetIn(BaseModel):
    name: str
    task: Literal["auto", "classification", "regression"] = "auto"


class CVConfig(BaseModel):
    n_folds: int = Field(5, ge=2, le=20)
    mode: Literal["nested", "single"] = "nested"
    n_inner_folds: int = Field(3, ge=2, le=10)
    random_state: int = 42


class HPOConfig(BaseModel):
    enabled: bool = True
    n_trials: int = Field(25, ge=1, le=500)
    timeout_s: int | None = Field(3600, ge=1)


class ExplainConfig(BaseModel):
    shap: bool = True
    background_samples: int = Field(100, ge=10, le=1000)
    max_display: int = Field(20, ge=3, le=50)


class ExportConfig(BaseModel):
    figures_dpi: int = Field(300, ge=72, le=600)
    include_eps: bool = True
    include_shap_dependence: bool = True


class TrainRequest(BaseModel):
    dataset_id: str
    targets: list[TargetIn] = Field(min_length=1)
    models: list[str] | None = None
    cv: CVConfig = CVConfig()
    hpo: HPOConfig = HPOConfig()
    explain: ExplainConfig = ExplainConfig()
    export: ExportConfig = ExportConfig()
