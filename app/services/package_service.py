from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas.training import TrainRequest
from app.storage.local_storage import new_id

logger = logging.getLogger(__name__)


def export_and_package(
    job_id: str,
    output,
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
    from app.services.figure_service import configure_publication_style

    configure_publication_style(request.export.figures_dpi)
    summary = output.summary
    (metrics_dir / "results_summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_per_fold_csv(output.per_fold_records, metrics_dir / "per_fold.csv")
    if output.predictions_frame is not None:
        output.predictions_frame.to_csv(data_dir / "dataset_with_predictions.csv", index=False)

    from app.services.artifact_service import save_final_models, write_model_manifest

    save_final_models(output.artifacts, models_dir)
    write_model_manifest(models_dir)

    cb("export:figures", 0.9)
    figure_paths_by_target = _export_diagnostics_figures(out_dir, output, request)
    figure_paths_by_target = {
        target: [f"../{rel}" for rel in rels] for target, rels in figure_paths_by_target.items()
    }

    shap_notes: dict[str, str] = {}
    if request.explain.shap and output.background_frame is not None:
        shap_notes = _export_shap_figures(out_dir, output, request)
        if not shap_notes:
            summary.warnings.append("SHAP analysis was requested but produced no figures.")

    cb("export:tables", 0.93)
    tables_by_target = _export_latex_tables(summary, latex_tables_dir, output)

    citation_keys = _collect_citation_keys(output)
    from app.services.bibtex_service import references_bibtex

    bib_text = references_bibtex(citation_keys, summary.library_versions)
    (bib_dir / "references.bib").write_text(bib_text, encoding="utf-8")

    cb("export:report", 0.95)
    tex_path, pdf_path = _export_report(
        summary, tables_by_target, figure_paths_by_target, shap_notes, report_dir
    )

    manifest_path = out_dir / "MANIFEST.txt"
    _write_manifest(manifest_path, job_id, request, summary, tex_path, pdf_path)

    cb("export:zip", 0.97)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(out_dir))
    logger.info("packaged job %s -> %s", job_id, zip_path)
    return zip_path


def _export_diagnostics_figures(out_dir: Path, output, request: TrainRequest) -> dict[str, list[str]]:
    from app.services import figure_service

    include_eps = request.export.include_eps
    relative_paths: dict[str, list[str]] = {}

    for target_name, artifact in output.artifacts.items():
        fig_dir = out_dir / "figures" / target_name.replace("/", "_")
        written: list[Path] = []
        if artifact.task == "classification" and artifact.confusion_matrix is not None:
            labels = artifact.confusion_labels or artifact.label_classes or []
            cm_fig = figure_service.confusion_matrix_figure(
                artifact.confusion_matrix, labels, f"Confusion matrix: {target_name}"
            )
            written.extend(
                figure_service.save_figure(cm_fig, fig_dir, f"confusion_matrix_{target_name}", include_eps)
            )
            if artifact.oof_probabilities is not None and artifact.y_true_encoded is not None and labels:
                roc_fig = figure_service.roc_curves_figure(
                    np.asarray(artifact.y_true_encoded),
                    artifact.oof_probabilities,
                    list(labels),
                    f"ROC curves (OOF): {target_name}",
                )
                written.extend(
                    figure_service.save_figure(roc_fig, fig_dir, f"roc_curves_{target_name}", include_eps)
                )
        elif artifact.task == "regression":
            if artifact.y_true_encoded is not None and artifact.oof_predictions is not None:
                y_true = artifact.y_true_encoded.astype(float)
                y_pred = artifact.oof_predictions.astype(float)
                pva = figure_service.predicted_vs_actual_figure(y_true, y_pred, target_name)
                written.extend(
                    figure_service.save_figure(pva, fig_dir, f"pred_vs_actual_{target_name}", include_eps)
                )
                res = figure_service.residuals_figure(y_true, y_pred, target_name)
                written.extend(
                    figure_service.save_figure(res, fig_dir, f"residuals_{target_name}", include_eps)
                )

        records = output.per_fold_records.get(target_name, [])
        models = sorted({r.get("model") for r in records if r.get("model")})
        primary = (
            next(iter(records))["primary_metric"]
            if records and "primary_metric" in records[0]
            else artifact.best_entry_key
        )
        box_fig = figure_service.metric_boxplot_figure(records, primary, models) if models else None
        if box_fig is not None:
            written.extend(
                figure_service.save_figure(box_fig, fig_dir, f"boxplot_{primary}_{target_name}", include_eps)
            )

        relative_paths[target_name] = [
            f"figures/{p.parent.name}/{p.name}" for p in written if p.suffix == ".png"
        ]
    return relative_paths


def _export_shap_figures(out_dir: Path, output, request: TrainRequest) -> dict[str, str]:
    from app.ml.explainability.shap_engine import build_shap_figures
    from app.services.figure_service import configure_publication_style

    configure_publication_style(request.export.figures_dpi)
    notes: dict[str, str] = {}
    for target_name, artifact in output.artifacts.items():
        if artifact.final_pipeline is None:
            continue
        shap_dir = out_dir / "figures" / target_name.replace("/", "_") / "shap"
        written = build_shap_figures(
            pipeline=artifact.final_pipeline,
            X_background=output.background_frame,
            explainer_kind=artifact.explainer_kind,
            target_name=target_name,
            out_dir=shap_dir,
            dpi=request.export.figures_dpi,
            include_eps=request.export.include_eps,
            include_dependence=request.export.include_shap_dependence,
            max_display=request.explain.max_display or 20,
        )
        png_count = sum(1 for p in written if p.suffix == ".png")
        if png_count:
            notes[target_name] = (
                f"{png_count} SHAP figure(s) generated under "
                f"figures/{target_name}/shap/ (explainer={artifact.explainer_kind})."
            )
    return notes


def _export_latex_tables(summary, latex_tables_dir: Path, output) -> dict[str, str]:
    from app.services.latex_service import metrics_table_tex

    tables: dict[str, str] = {}
    n_folds = int(summary.cv.get("n_folds", 5))
    mode = str(summary.cv.get("mode", "single"))
    for target_result in summary.targets:
        fragment = metrics_table_tex(target_result, summary.n_samples, mode, n_folds)
        file_name = f"table_{_safe_stem(target_result.target)}.tex"
        (latex_tables_dir / file_name).write_text(fragment, encoding="utf-8")
        tables[target_result.target] = f"\\input{{../latex/tables/{file_name}}}"
    return tables


def _collect_citation_keys(output) -> set[str]:
    from app.ml.models.registry import get_entry

    keys: set[str] = set()
    for artifact in output.artifacts.values():
        try:
            entry = get_entry(artifact.task, artifact.best_entry_key)
        except Exception:
            continue
        if getattr(entry, "citation_key", None):
            keys.add(entry.citation_key)
    keys.update({"pedregosa2011scikit", "akiba2019optuna"})
    return keys


def _export_report(summary, tables_by_target, figure_paths_by_target, shap_notes, report_dir: Path):
    from app.services.pdf_report_service import build_report

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


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "_" for ch in name)[:50]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_per_fold_csv(per_fold_records: dict[str, list[dict]], path: Path) -> None:
    rows: list[dict] = []
    for target, records in per_fold_records.items():
        for record in records:
            flat: dict = {"target": target}
            flat.update({k: v for k, v in record.items() if k not in ("params",)})
            params = record.get("params")
            if params:
                flat["params"] = json.dumps(params)
            rows.append(flat)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _write_manifest(
    path: Path,
    job_id: str,
    request: TrainRequest,
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


def new_job_id() -> str:
    return new_id("job")
