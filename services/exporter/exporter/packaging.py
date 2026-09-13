"""Render the scientific bundle from the persisted training contract.

This worker is a pure consumer: every input arrives as files under the job
directories written by the ML core (and optionally enriched by the
explainability stage).
"""
from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
from automl_shared.contract import (
    load_per_fold_records,
    load_summary,
    load_target_views,
)
from automl_shared.figures import configure_publication_style
from automl_shared.schemas.results import RunSummary
from automl_shared.schemas.training import TrainRequest

logger = logging.getLogger(__name__)


def package_from_contract(
    job_id: str,
    request: TrainRequest,
    work_dir: Path,
    out_dir: Path,
    zip_path: Path,
    progress_cb=None,
) -> Path:
    cb = progress_cb or (lambda stage, frac: None)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    data_dir = out_dir / "data"
    models_dir = out_dir / "models"
    latex_tables_dir = out_dir / "latex" / "tables"
    bib_dir = out_dir / "bib"
    report_dir = out_dir / "report"
    for directory in (metrics_dir, data_dir, models_dir, latex_tables_dir, bib_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cb("export:metrics", 0.86)
    configure_publication_style(request.export.figures_dpi)

    summary = load_summary(out_dir)
    per_fold_records = load_per_fold_records(out_dir)
    artifacts = load_target_views(work_dir, models_dir)

    figure_paths_by_target = _export_diagnostics_figures(out_dir, artifacts, per_fold_records, request)
    interpretability_paths = _export_interpretability_figures(work_dir, out_dir, request)
    for target, rels in interpretability_paths.items():
        figure_paths_by_target.setdefault(target, []).extend(rels)
    tree_paths = _export_decision_tree_figures(out_dir, artifacts, request)
    for target, rels in tree_paths.items():
        figure_paths_by_target.setdefault(target, []).extend(rels)
    figure_paths_by_target = {
        target: [f"../{rel}" for rel in rels] for target, rels in figure_paths_by_target.items()
    }

    shap_notes: dict[str, str] = {}
    if request.explain.shap:
        shap_notes = _scan_shap_notes(out_dir, artifacts)
        if not shap_notes:
            summary.warnings.append("SHAP analysis was requested but produced no figures.")

    cb("export:tables", 0.93)
    tables_by_target = _export_latex_tables(summary, latex_tables_dir)

    citation_keys = _collect_citation_keys(artifacts)
    from exporter.services.bibtex_service import references_bibtex

    bib_text = references_bibtex(citation_keys, summary.library_versions)
    (bib_dir / "references.bib").write_text(bib_text, encoding="utf-8")

    cb("export:report", 0.95)
    tex_path, pdf_path = _export_report(
        summary, tables_by_target, figure_paths_by_target, shap_notes, report_dir
    )

    manifest_path = out_dir / "MANIFEST.txt"
    _write_manifest(manifest_path, job_id, summary, tex_path, pdf_path)

    cb("export:zip", 0.97)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(out_dir))
    logger.info("packaged job %s -> %s", job_id, zip_path)
    return zip_path


def _scan_shap_notes(out_dir: Path, artifacts: dict) -> dict[str, str]:
    """Describe pre-rendered SHAP figures produced by the explainability stage."""
    notes: dict[str, str] = {}
    for target_name, artifact in artifacts.items():
        shap_dir = out_dir / "figures" / target_name.replace("/", "_") / "shap"
        png_count = len(list(shap_dir.glob("*.png")))
        if png_count:
            notes[target_name] = (
                f"{png_count} SHAP figure(s) generated under "
                f"figures/{target_name}/shap/ (explainer={artifact.explainer_kind})."
            )
    return notes


def _export_diagnostics_figures(
    out_dir: Path, artifacts: dict, per_fold_records: dict[str, list[dict]], request: TrainRequest
) -> dict[str, list[str]]:
    from exporter.charts import (
        confusion_matrix_figure,
        metric_boxplot_figure,
        predicted_vs_actual_figure,
        residuals_figure,
        roc_curves_figure,
        save_figure,
    )

    include_eps = request.export.include_eps
    relative_paths: dict[str, list[str]] = {}

    for target_name, artifact in artifacts.items():
        fig_dir = out_dir / "figures" / target_name.replace("/", "_")
        written: list[Path] = []
        if artifact.task == "classification" and artifact.confusion_matrix is not None:
            labels = artifact.confusion_labels or artifact.label_classes or []
            cm_fig = confusion_matrix_figure(
                artifact.confusion_matrix, labels, f"Confusion matrix: {target_name}"
            )
            written.extend(
                save_figure(cm_fig, fig_dir, f"confusion_matrix_{target_name}", include_eps)
            )
            if artifact.oof_probabilities is not None and artifact.y_true_encoded is not None and labels:
                roc_fig = roc_curves_figure(
                    np.asarray(artifact.y_true_encoded),
                    artifact.oof_probabilities,
                    list(labels),
                    f"ROC curves (OOF): {target_name}",
                )
                written.extend(save_figure(roc_fig, fig_dir, f"roc_curves_{target_name}", include_eps))
        elif artifact.task == "regression":
            if artifact.y_true_encoded is not None and artifact.oof_predictions is not None:
                y_true = artifact.y_true_encoded.astype(float)
                y_pred = artifact.oof_predictions.astype(float)
                pva = predicted_vs_actual_figure(y_true, y_pred, target_name)
                written.extend(
                    save_figure(pva, fig_dir, f"pred_vs_actual_{target_name}", include_eps)
                )
                res = residuals_figure(y_true, y_pred, target_name)
                written.extend(save_figure(res, fig_dir, f"residuals_{target_name}", include_eps))

        records = per_fold_records.get(target_name, [])
        models = sorted({r.get("model") for r in records if r.get("model")})
        primary = _resolve_primary_metric(records, artifact)
        box_fig = metric_boxplot_figure(records, primary, models) if models else None
        if box_fig is not None:
            safe_stem = _safe_stem(target_name)
            written.extend(
                save_figure(box_fig, fig_dir, f"boxplot_{primary}_{safe_stem}", include_eps)
            )

        relative_paths[target_name] = [
            f"figures/{p.parent.name}/{p.name}" for p in written if p.suffix == ".png"
        ]
    return relative_paths


def _export_decision_tree_figures(
    out_dir: Path, artifacts: dict, request: TrainRequest
) -> dict[str, list[str]]:
    """Render the simplified "Sí / No" decision-tree flowchart for the winner.

    Loads the best model artifact, resolves the fitted tree estimator and
    draws the first tree as an infographic with real feature names (e.g.
    ``temp_min < 26.3``). The PNG (and optional EPS) ships under
    ``figures/<target>/decision_tree_flowchart_<model_key>.png`` and is linked
    into the LaTeX report. Always degrades gracefully: unsupported or
    non-tree models are skipped.
    """
    import joblib

    from exporter.charts import decision_tree_flowchart_figure, save_figure

    include_eps = request.export.include_eps
    relative_paths: dict[str, list[str]] = {}
    models_dir = out_dir / "models"

    for target_name, artifact in artifacts.items():
        best_key = artifact.best_entry_key
        if not best_key:
            continue
        payload_path = models_dir / f"{_safe_stem(target_name)}_best_model.joblib"
        if not payload_path.exists():
            logger.info("best model artifact missing for %s; skipping tree flowchart", target_name)
            continue
        try:
            payload = joblib.load(payload_path)
            pipeline = payload.get("pipeline")
        except Exception as exc:
            logger.warning("could not load best model for %s: %s", target_name, exc)
            continue
        if pipeline is None:
            continue

        estimator = pipeline.named_steps.get("estimator")
        preprocessor = pipeline.named_steps.get("preprocess")
        if estimator is None or not _is_tree_estimator(estimator):
            continue

        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
            except Exception:
                logger.warning("preprocessed feature names unavailable for %s", target_name)
                continue

        display_name = payload.get("display_name") or best_key
        class_names = payload.get("label_classes") or artifact.label_classes
        task = payload.get("task") or artifact.task

        fig = decision_tree_flowchart_figure(
            estimator,
            feature_names,
            display_name,
            target=target_name,
            task=task,
            class_names=class_names,
            max_depth=2,
        )
        if fig is None:
            continue
        fig_dir = out_dir / "figures" / target_name.replace("/", "_")
        written = save_figure(
            fig, fig_dir, f"decision_tree_flowchart_{_safe_stem(best_key)}", include_eps
        )
        relative_paths[target_name] = [
            f"figures/{p.parent.name}/{p.name}" for p in written if p.suffix == ".png"
        ]
        logger.info("exported decision tree flowchart for %s (%s)", target_name, best_key)
    return relative_paths


def _is_tree_estimator(estimator) -> bool:
    module = type(estimator).__module__ or ""
    if module.split(".")[0] in ("xgboost", "lightgbm"):
        return True
    return _is_sklearn_tree(estimator)


def _is_sklearn_tree(estimator) -> bool:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return isinstance(
        estimator,
        (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        ),
    )


def _export_interpretability_figures(
    work_dir: Path, out_dir: Path, request: TrainRequest
) -> dict[str, list[str]]:
    """Render per-model interpretability figures for every evaluated candidate.

    The explainability stage persists compact value profiles (coefficients,
    SHAP importance, raw SHAP arrays) under ``work/interpretability``; this
    function turns them into ``coef_plot_*``, ``shap_importance_*``,
    ``shap_beeswarm_*`` and the consolidated ``model_feature_importance_
    comparison`` figures so they ship inside the report bundle.
    """
    from exporter.charts import (
        coefficients_figure,
        feature_importance_figure,
        model_feature_importance_comparison_figure,
        save_figure,
        shap_beeswarm_figure,
    )

    interp_dir = work_dir / "interpretability"
    if not interp_dir.exists():
        logger.info("interpretability profiles not found in %s; skipping per-model figures", interp_dir)
        return {}

    include_eps = request.export.include_eps
    relative_paths: dict[str, list[str]] = {}

    for profile_file in sorted(interp_dir.glob("*.json")):
        try:
            payload = json.loads(profile_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read interpretability profile %s: %s", profile_file.name, exc)
            continue
        target_name = payload.get("target")
        if not target_name:
            continue
        feature_names = payload.get("feature_names") or []
        fig_dir = out_dir / "figures" / target_name.replace("/", "_")
        written: list[Path] = []
        profiles_for_matrix: list[dict] = []

        for model_entry in payload.get("models") or []:
            key = model_entry.get("key")
            if not key:
                continue
            display_name = model_entry.get("display_name") or key
            kind = model_entry.get("kind")
            importance = model_entry.get("importance")
            if importance is None:
                continue
            importance_arr = np.asarray(importance, dtype=float)
            profiles_for_matrix.append(
                {"key": key, "display_name": display_name, "importance": importance_arr}
            )

            coefficients = model_entry.get("coefficients")
            if kind == "linear" and coefficients is not None:
                coef_fig = coefficients_figure(
                    np.asarray(coefficients, dtype=float), feature_names, display_name
                )
                written.extend(save_figure(coef_fig, fig_dir, f"coef_plot_{key}", include_eps))
            elif kind == "tree":
                source = model_entry.get("source", "intrinsic")
                imp_fig = feature_importance_figure(
                    importance_arr, feature_names, display_name, source=source
                )
                written.extend(save_figure(imp_fig, fig_dir, f"shap_importance_{key}", include_eps))
                if not model_entry.get("multiclass") and model_entry.get("shap_file"):
                    try:
                        arrays = np.load(interp_dir / model_entry["shap_file"])
                        swarm_fig = shap_beeswarm_figure(
                            arrays["shap_values"], arrays["eval_data"], feature_names, display_name
                        )
                        written.extend(save_figure(swarm_fig, fig_dir, f"shap_beeswarm_{key}", include_eps))
                    except (OSError, KeyError) as exc:
                        logger.warning("beeswarm skipped for %s: %s", key, exc)

        if profiles_for_matrix:
            matrix_fig = model_feature_importance_comparison_figure(profiles_for_matrix, feature_names)
            written.extend(
                save_figure(matrix_fig, fig_dir, "model_feature_importance_comparison", include_eps)
            )

        relative_paths[target_name] = [
            f"figures/{p.parent.name}/{p.name}" for p in written if p.suffix == ".png"
        ]
    return relative_paths


def _export_latex_tables(summary: RunSummary, latex_tables_dir: Path) -> dict[str, str]:
    from exporter.services.latex_service import metrics_table_tex

    tables: dict[str, str] = {}
    n_folds = int(summary.cv.get("n_folds", 5))
    mode = str(summary.cv.get("mode", "single"))
    for target_result in summary.targets:
        fragment = metrics_table_tex(target_result, summary.n_samples, mode, n_folds)
        file_name = f"table_{_safe_stem(target_result.target)}.tex"
        (latex_tables_dir / file_name).write_text(fragment, encoding="utf-8")
        tables[target_result.target] = f"\\input{{../latex/tables/{file_name}}}"
    return tables


def _collect_citation_keys(artifacts: dict) -> set[str]:
    keys: set[str] = set()
    for artifact in artifacts.values():
        citation_key = getattr(artifact, "citation_key", None)
        if citation_key:
            keys.add(citation_key)
    keys.update({"pedregosa2011scikit", "akiba2019optuna"})
    return keys


def _export_report(summary, tables_by_target, figure_paths_by_target, shap_notes, report_dir: Path):
    from exporter.services.pdf_report_service import build_report

    try:
        return build_report(
            summary=summary,
            tables_by_target=tables_by_target,
            figure_paths_by_target=figure_paths_by_target,
            shap_notes=shap_notes,
            run_warnings=list(summary.warnings),
            out_dir=report_dir,
        )
    except Exception as exc:
        logger.warning("report generation failed: %s", exc, exc_info=True)
        summary.warnings.append(f"Report generation failed: {exc}")
        fallback = report_dir / "report.tex"
        fallback.write_text("% report generation failed\n", encoding="utf-8")
        return fallback, None


_TASK_PRIMARY_METRIC: dict[str, str] = {
    "classification": "f1_macro",
    "regression": "rmse",
}


def _resolve_primary_metric(records: list[dict], artifact) -> str:
    """Resolve the primary metric name with a 3-level fallback chain."""
    if records and "primary_metric" in records[0]:
        return str(records[0]["primary_metric"])
    default = _TASK_PRIMARY_METRIC.get(artifact.task)
    if default:
        return default
    logger.warning(
        "Could not resolve primary metric for target '%s' (task=%s); "
        "falling back to best_entry_key '%s'",
        getattr(artifact, "target", "?"),
        artifact.task,
        artifact.best_entry_key,
    )
    return artifact.best_entry_key


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "_" for ch in name)[:50]


def _write_manifest(
    path: Path,
    job_id: str,
    summary,
    tex_path=None,
    pdf_path=None,
) -> None:
    lines = [
        "Academic AutoML Platform - Run Manifest",
        "=" * 48,
        f"job_id: {job_id}",
        f"dataset: {summary.filename} ({summary.n_samples} x {summary.n_features})",
        f"cv: mode={summary.cv['mode']} folds={summary.cv['n_folds']} seed={summary.cv['random_state']}",
        f"hpo: enabled={summary.hpo['enabled']} trials={summary.hpo['n_trials']}",
        f"runtime_s: {summary.runtime_s}",
    ]
    if pdf_path is not None:
        lines.append("pdf: compiled with tectonic/pdflatex")
    else:
        lines.append("pdf: not available (LaTeX engine missing); see report/report.tex")
    lines.extend(["", "Targets:"])
    for target in summary.targets:
        params_json = json.dumps(target.best_params)
        lines.append(f"  - {target.target} [{target.task}] best={target.best_model} params={params_json}")
    if summary.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  ! {w}" for w in summary.warnings)
    lines.append("")
    lines.append("Library versions:")
    lines.extend(f"  {k}=={v}" for k, v in sorted(summary.library_versions.items()))
    path.write_text("\n".join(lines), encoding="utf-8")
