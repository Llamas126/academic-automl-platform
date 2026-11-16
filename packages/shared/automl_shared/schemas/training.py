from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class TargetIn(BaseModel):
    name: str
    task: Literal["auto", "classification", "regression"] = "auto"
    unit: str | None = None


class CVConfig(BaseModel):
    n_folds: int = Field(5, ge=2, le=20)
    mode: Literal["nested", "single"] = "nested"
    n_inner_folds: int = Field(3, ge=2, le=10)
    random_state: int = 42
    #: ``kfold`` shuffles rows; ``timeseries`` keeps chronological order.
    scheme: Literal["kfold", "timeseries"] = "kfold"
    #: Rows dropped between train and test in ``timeseries`` (leakage barrier).
    gap: int = Field(0, ge=0)
    #: Optional datetime column used to order rows for ``timeseries``.
    date_column: str | None = None


class TemporalFeatureConfig(BaseModel):
    """Causal lags and rolling windows applied before cross-validation.

    ``enabled`` gates the exogenous features. ``target_lags`` is independent:
    a non-empty list always applies, because it explicitly asks for harvest
    autocorrelation. Target lags require ``cv.scheme="timeseries"`` and
    ``cv.gap >= max(target_lags)``, which the training pipeline enforces.
    """

    enabled: bool = False
    #: Datetime column used to order rows. Falls back to ``cv.date_column``.
    date_column: str | None = None
    #: Numeric exogenous columns to lag and roll. Empty means auto-detect when enabled.
    exogenous_columns: list[str] = Field(default_factory=list)
    #: Target columns to lag. Empty means use every target in the request.
    target_columns: list[str] = Field(default_factory=list)
    #: Short and phenological horizons, in days.
    lags: list[int] = Field(default_factory=lambda: [1, 3, 7, 14, 30])
    #: Rolling window sizes, in days.
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 14, 30])
    rolling_aggs: list[Literal["sum", "mean", "max", "min"]] = Field(
        default_factory=lambda: ["sum", "mean", "max", "min"]
    )
    #: ``False`` shifts the window to ``t-w .. t-1`` for operational inference.
    include_current_day: bool = True
    #: Harvest autocorrelation lags. Requires a temporal CV with a sufficient gap.
    target_lags: list[int] = Field(default_factory=list)
    #: Drop rows where every configured target is zero *after* computing temporal features.
    #: Defaults to ``False`` for backward compatibility; useful for zero-inflated harvest data.
    drop_zero_targets: bool = False

    @field_validator("lags", "rolling_windows", "target_lags")
    @classmethod
    def _positive_unique_ints(cls, value: list[int], info: ValidationInfo) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        if any(item < 1 for item in value):
            raise ValueError(f"{info.field_name} entries must be >= 1")
        return value

    @field_validator("exogenous_columns", "target_columns")
    @classmethod
    def _unique_columns(cls, value: list[str], info: ValidationInfo) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return value


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
    temporal: TemporalFeatureConfig = Field(default_factory=TemporalFeatureConfig)
    hpo: HPOConfig = HPOConfig()
    explain: ExplainConfig = ExplainConfig()
    export: ExportConfig = ExportConfig()
