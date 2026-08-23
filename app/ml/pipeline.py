from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.exceptions import ValidationFailedError
from app.ml.metrics.calculator import (
    METRIC_DIRECTIONS,
    aggregate_metric,
    compute_classification_metrics,
    compute_regression_metrics,
    primary_metric,
)
from app.ml.models.registry import ModelEntry, UnknownModelError, resolve_entries
from app.ml.preprocessing.builders import build_preprocessor
from app.ml.search.optuna_tuner import (
    TuningFailure,
    default_params_via_fixed_trial,
    tune_hyperparams,
)
from app.ml.targets.multi_target import RunContext, TargetSpec, resolve_targets
from app.ml.validation.kfold import fold_report, make_folds
from app.schemas.results import MetricAggregate, ModelResult, RunSummary, TargetResults
from app.schemas.training import TrainRequest

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]

_PROGRESS_BASE = 0.05
_PROGRESS_SPAN = 0.94


class ProgressTracker:
    def __init__(self, total_units: float, cb: ProgressCallback) -> None:
        self.total = max(total_units, 1e-9)
        self.done = 0.0
        self.cb = cb

    def emit(self, stage: str) -> None:
        fraction = min(_PROGRESS_BASE + _PROGRESS_SPAN * self.done / self.total, 0.995)
        self.cb(stage, round(fraction, 4))

    def complete(self, amount: float) -> None:
        self.done = min(self.done + amount, self.total)


@dataclass
class TargetArtifacts:
    target: str
    task: str
    final_pipeline: Pipeline | None
    best_entry_key: str
    best_params: dict
    explainer_kind: str = "kernel"
    label_classes: list[str] | None = None
    y_true_encoded: np.ndarray | None = None
    oof_predictions: np.ndarray | None = None
    oof_probabilities: np.ndarray | None = None
    confusion_matrix: list[list[int]] | None = None
    confusion_labels: list[str] | None = None


@dataclass
class PipelineOutput:
    summary: RunSummary
    artifacts: dict[str, TargetArtifacts] = field(default_factory=dict)
    predictions_frame: pd.DataFrame | None = None
    fold_assignments: np.ndarray | None = None
    background_frame: pd.DataFrame | None = None
    per_fold_records: dict[str, list[dict]] = field(default_factory=dict)


def _fit_pipeline(
    entry: ModelEntry,
    params: dict,
    seed: int,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=entry.needs_scaling)
    estimator = entry.build(params, seed)
    return Pipeline([("preprocess", preprocessor), ("estimator", estimator)])


def _evaluate_single_mode(
    entry: ModelEntry,
    spec: TargetSpec,
    ctx: RunContext,
    request: TrainRequest,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tracker: ProgressTracker,
) -> tuple[ModelResult, list[dict], np.ndarray, np.ndarray | None]:
    seed = request.cv.random_state
    y_true_full = spec.y_encoded.to_numpy()
    n_rows = len(ctx.X)

    if request.hpo.enabled:
        params = tune_hyperparams(
            entry,
            ctx.X,
            y_true_full,
            ctx.numeric_cols,
            ctx.categorical_cols,
            n_inner_folds=request.cv.n_inner_folds,
            seed=seed,
            n_trials=request.hpo.n_trials,
            timeout_s=request.hpo.timeout_s,
        )
    else:
        params = default_params_via_fixed_trial(entry)

    oof_pred = np.empty(n_rows, dtype=float)
    oof_proba = (
        np.full((n_rows, len(spec.label_encoder_classes)), np.nan) if spec.task == "classification" else None
    )
    fold_records: list[dict] = []
    metric_names: set[str] = set()

    for fold_index, (train_idx, test_idx) in enumerate(folds):
        tracker.emit(f"cv:{spec.name}:{entry.key} (fold {fold_index + 1}/{len(folds)})")
        error_msg: str | None = None
        predictions = proba = None
        try:
            pipe = _fit_pipeline(entry, params, seed, ctx.numeric_cols, ctx.categorical_cols)
            pipe.fit(ctx.X.iloc[train_idx], y_true_full[train_idx])
            predictions = pipe.predict(ctx.X.iloc[test_idx])
            proba = pipe.predict_proba(ctx.X.iloc[test_idx]) if spec.task == "classification" else None
        except Exception as exc:
            logger.warning("fold %d failed for %s/%s: %s", fold_index, spec.name, entry.key, exc)
            error_msg = str(exc)

        if error_msg is None:
            y_test = y_true_full[test_idx]
            metrics = _compute_metrics_for(spec.task, y_test, predictions, proba, ctx.n_features)
            metric_names.update(k for k, v in metrics.items() if v is not None)
            oof_pred[test_idx] = predictions
            if proba is not None:
                oof_proba[test_idx] = proba
            fold_records.append({"model": entry.key, "fold": fold_index, **_round_metrics(metrics)})
        else:
            fold_records.append({"model": entry.key, "fold": fold_index, "error": error_msg})
        tracker.complete(1.0)

    aggregates = [
        MetricAggregate(**aggregate_metric(name, [r.get(name) for r in fold_records if "error" not in r]))
        for name in sorted(metric_names)
    ]
    primary = primary_metric(spec.task)
    primary_agg = next((a for a in aggregates if a.metric == primary), None)
    if primary_agg is None or not np.isfinite(primary_agg.mean):
        raise TuningFailure(f"no successful folds for '{entry.key}' on target '{spec.name}'")

    result = ModelResult(
        model=entry.display_name,
        citation_key=entry.citation_key,
        best_params=params,
        primary_metric=primary,
        primary_mean=primary_agg.mean,
        rank=0,
        metrics=aggregates,
    )
    return result, fold_records, oof_pred, oof_proba


def _evaluate_nested_mode(
    entry: ModelEntry,
    spec: TargetSpec,
    ctx: RunContext,
    request: TrainRequest,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tracker: ProgressTracker,
) -> tuple[ModelResult, list[dict], np.ndarray, np.ndarray | None]:
    seed = request.cv.random_state
    y_true_full = spec.y_encoded.to_numpy()
    n_rows = len(ctx.X)

    oof_pred = np.empty(n_rows, dtype=float)
    oof_proba = (
        np.full((n_rows, len(spec.label_encoder_classes)), np.nan) if spec.task == "classification" else None
    )
    fold_records: list[dict] = []
    metric_names: set[str] = set()

    for fold_index, (train_idx, test_idx) in enumerate(folds):
        params_fold: dict = {}
        error_msg: str | None = None
        predictions = proba = None
        try:
            if request.hpo.enabled:
                tracker.emit(f"hpo:{spec.name}:{entry.key} (outer fold {fold_index + 1})")
                params_fold = tune_hyperparams(
                    entry,
                    ctx.X.iloc[train_idx],
                    y_true_full[train_idx],
                    ctx.numeric_cols,
                    ctx.categorical_cols,
                    n_inner_folds=request.cv.n_inner_folds,
                    seed=seed + fold_index,
                    n_trials=request.hpo.n_trials,
                    timeout_s=request.hpo.timeout_s,
                )
                tracker.complete(0.5)
            else:
                params_fold = default_params_via_fixed_trial(entry)
            tracker.emit(f"cv:{spec.name}:{entry.key} (outer fold {fold_index + 1})")
            pipe = _fit_pipeline(entry, params_fold, seed, ctx.numeric_cols, ctx.categorical_cols)
            pipe.fit(ctx.X.iloc[train_idx], y_true_full[train_idx])
            predictions = pipe.predict(ctx.X.iloc[test_idx])
            proba = pipe.predict_proba(ctx.X.iloc[test_idx]) if spec.task == "classification" else None
        except Exception as exc:
            logger.warning("outer fold %d failed for %s/%s: %s", fold_index, spec.name, entry.key, exc)
            error_msg = str(exc)

        if error_msg is None:
            y_test = y_true_full[test_idx]
            metrics = _compute_metrics_for(spec.task, y_test, predictions, proba, ctx.n_features)
            metric_names.update(k for k, v in metrics.items() if v is not None)
            oof_pred[test_idx] = predictions
            if proba is not None:
                oof_proba[test_idx] = proba
            fold_records.append(
                {"model": entry.key, "fold": fold_index, "params": params_fold, **_round_metrics(metrics)}
            )
        else:
            fold_records.append({"model": entry.key, "fold": fold_index, "error": error_msg})
        tracker.complete(0.5 if request.hpo.enabled else 1.0)

    aggregates = [
        MetricAggregate(**aggregate_metric(name, [r.get(name) for r in fold_records if "error" not in r]))
        for name in sorted(metric_names)
    ]
    primary = primary_metric(spec.task)
    primary_agg = next((a for a in aggregates if a.metric == primary), None)
    if primary_agg is None or not np.isfinite(primary_agg.mean):
        raise TuningFailure(f"no successful outer folds for '{entry.key}' on target '{spec.name}'")

    result = ModelResult(
        model=entry.display_name,
        citation_key=entry.citation_key,
        best_params={},
        primary_metric=primary,
        primary_mean=primary_agg.mean,
        rank=0,
        metrics=aggregates,
    )
    result.best_params = {}
    return result, fold_records, oof_pred, oof_proba


def _compute_metrics_for(task, y_true, y_pred, proba, n_features):
    if task == "classification":
        return compute_classification_metrics(y_true, y_pred, proba)
    return compute_regression_metrics(y_true, y_pred, n_features)


def _round_metrics(metrics: dict) -> dict:
    return {k: (None if v is None else float(v)) for k, v in metrics.items()}


def run_training(
    request: TrainRequest,
    dataset_id: str,
    filename: str,
    df_raw: pd.DataFrame,
    job_id: str,
    progress_cb: ProgressCallback | None = None,
) -> PipelineOutput:
    started = time.monotonic()
    cb: ProgressCallback = progress_cb or (lambda stage, frac: None)
    cb("profiling", 0.01)

    ctx = resolve_targets(df_raw, request.targets)
    for warning in ctx.warnings:
        logger.info("[job %s] %s", job_id, warning)

    try:
        resolve_entries("classification", request.models)
        resolve_entries("regression", request.models)
    except UnknownModelError as exc:
        raise ValidationFailedError(str(exc)) from exc

    folds = make_folds(
        len(ctx.X),
        request.cv.n_folds,
        request.cv.random_state,
        stratify_labels=ctx.stratification_labels,
    )
    cb("folding", _PROGRESS_BASE)

    plan: list[tuple[TargetSpec, list[ModelEntry]]] = []
    for spec in ctx.specs:
        entries = resolve_entries(spec.task, request.models)
        if not entries:
            raise ValidationFailedError(
                f"no candidate models available for task '{spec.task}' with the requested filter"
            )
        plan.append((spec, entries))

    total_work = float(sum(len(entries) * request.cv.n_folds + 1 for _spec, entries in plan))
    tracker = ProgressTracker(total_work, cb)

    results_by_target: list[TargetResults] = []
    artifacts: dict[str, TargetArtifacts] = {}
    per_fold_all: dict[str, list[dict]] = {}
    warnings_out = list(ctx.warnings)
    fold_assignment = np.empty(len(ctx.X), dtype=int)
    fold_assignment.fill(-1)
    for fold_index, (_tr, te) in enumerate(folds):
        fold_assignment[te] = fold_index

    for spec, entries in plan:
        model_results: list[ModelResult] = []
        fold_records_target: list[dict] = []
        oof_by_model: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        direction = METRIC_DIRECTIONS[primary_metric(spec.task)]

        for entry in entries:
            evaluator = _evaluate_nested_mode if request.cv.mode == "nested" else _evaluate_single_mode
            result, fold_records, oof_pred, oof_proba = evaluator(entry, spec, ctx, request, folds, tracker)
            model_results.append(result)
            fold_records_target.extend(fold_records)
            oof_by_model[result.model] = (oof_pred, oof_proba)

        ranked = sorted(model_results, key=lambda r: r.primary_mean, reverse=(direction == "max"))
        for position, model_result in enumerate(ranked, start=1):
            model_result.rank = position
        best_result = ranked[0]
        winning_entry = next(e for e in entries if e.display_name == best_result.model)
        best_key = winning_entry.key
        best_pred, best_proba = oof_by_model[best_result.model]

        if request.cv.mode == "nested":
            if request.hpo.enabled:
                try:
                    final_params = tune_hyperparams(
                        winning_entry,
                        ctx.X,
                        spec.y_encoded.to_numpy(),
                        ctx.numeric_cols,
                        ctx.categorical_cols,
                        n_inner_folds=request.cv.n_inner_folds,
                        seed=request.cv.random_state + 9999,
                        n_trials=request.hpo.n_trials,
                        timeout_s=request.hpo.timeout_s,
                    )
                except TuningFailure as exc:
                    warnings_out.append(str(exc))
                    final_params = default_params_via_fixed_trial(winning_entry)
            else:
                final_params = default_params_via_fixed_trial(winning_entry)
        else:
            final_params = best_result.best_params

        final_pipe = _fit_pipeline(
            winning_entry, final_params, request.cv.random_state, ctx.numeric_cols, ctx.categorical_cols
        )
        final_pipe.fit(ctx.X, spec.y_encoded.to_numpy())

        cm: list[list[int]] | None = None
        cm_labels: list[str] | None = None
        if spec.task == "classification":
            from app.ml.metrics.calculator import confusion_matrix_aggregated

            cm, cm_labels = confusion_matrix_aggregated(spec.y_encoded.to_numpy(), best_pred)

        tracker.emit(f"final-model:{spec.name}")
        artifacts[spec.name] = TargetArtifacts(
            target=spec.name,
            task=spec.task,
            final_pipeline=final_pipe,
            best_entry_key=best_key,
            best_params=final_params,
            explainer_kind=winning_entry.explainer,
            label_classes=spec.label_encoder_classes,
            y_true_encoded=spec.y_encoded.to_numpy(),
            oof_predictions=best_pred,
            oof_probabilities=best_proba,
            confusion_matrix=cm,
            confusion_labels=cm_labels,
        )
        tracker.complete(1.0)

        results_by_target.append(
            TargetResults(
                target=spec.name,
                task=spec.task,
                label_encoder_classes=spec.label_encoder_classes,
                best_model=best_result.model,
                best_params=final_params,
                models=ranked,
            )
        )
        per_fold_all[spec.name] = fold_records_target

    predictions_frame = _build_predictions_frame(ctx, artifacts, fold_assignment)
    runtime = time.monotonic() - started
    summary = RunSummary(
        job_id=job_id,
        dataset_id=dataset_id,
        filename=filename,
        n_samples=len(ctx.X),
        n_features=ctx.n_features,
        feature_columns=ctx.feature_names(),
        cv={
            "mode": request.cv.mode,
            "n_folds": request.cv.n_folds,
            "n_inner_folds": request.cv.n_inner_folds,
            "random_state": request.cv.random_state,
            "fold_sizes": fold_report(folds),
        },
        hpo={
            "enabled": request.hpo.enabled,
            "n_trials": request.hpo.n_trials,
            "timeout_s": request.hpo.timeout_s,
        },
        targets=results_by_target,
        warnings=warnings_out,
        library_versions=_library_versions(),
        runtime_s=round(runtime, 2),
        finished_at=datetime.now(UTC),
    )
    cb("done", 1.0)
    background_n = min(len(ctx.X), 1000)
    if background_n < len(ctx.X):
        rng = np.random.default_rng(request.cv.random_state)
        background_indices = np.sort(rng.choice(len(ctx.X), size=background_n, replace=False))
        background_frame = ctx.X.iloc[background_indices]
    else:
        background_frame = ctx.X
    return PipelineOutput(
        summary=summary,
        artifacts=artifacts,
        predictions_frame=predictions_frame,
        fold_assignments=fold_assignment,
        background_frame=background_frame.copy(),
        per_fold_records=per_fold_all,
    )


def _build_predictions_frame(
    ctx: RunContext, artifacts: dict[str, TargetArtifacts], fold_assignment: np.ndarray
) -> pd.DataFrame:
    out = pd.DataFrame(index=ctx.X.index)
    for col in ctx.feature_names():
        out[col] = ctx.X[col].values
    out["fold_id"] = fold_assignment
    for spec in ctx.specs:
        artifact = artifacts.get(spec.name)
        if artifact is None or artifact.oof_predictions is None:
            continue
        pred_col = f"{spec.name}_predicted"
        if spec.task == "classification" and artifact.label_classes:
            codes = artifact.oof_predictions.astype(int)
            out[pred_col] = [artifact.label_classes[int(c)] for c in codes]
            if artifact.oof_probabilities is not None:
                for class_index, class_label in enumerate(artifact.label_classes):
                    safe_label = str(class_label).replace(" ", "_")
                    out[f"{spec.name}_proba_{safe_label}"] = artifact.oof_probabilities[:, class_index]
        else:
            out[pred_col] = artifact.oof_predictions.astype(float)
    return out


def _library_versions() -> dict[str, str]:
    from importlib.metadata import version

    wanted = [
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "optuna",
        "shap",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "fastapi",
        "celery",
    ]
    versions = {}
    for package in wanted:
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "unknown"
    return versions
