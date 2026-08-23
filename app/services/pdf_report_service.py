from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.core.config import get_settings
from app.schemas.results import RunSummary
from app.services import latex_service

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=False,
        autoescape=False,
    )
    return env


def _params_to_tex(params: dict) -> str:
    if not params:
        return "defaults (no tuning)"
    items = [
        f"\\texttt{{{latex_service.escape_latex(key)}}}={latex_service.escape_latex(value)}"
        for key, value in sorted(params.items())
    ]
    return ", ".join(items)


def render_report_tex(
    summary: RunSummary,
    tables_by_target: dict[str, str],
    figure_paths_by_target: dict[str, list[str]],
    shap_notes: dict[str, str],
    warnings: list[str],
) -> str:
    env = _jinja_env()
    template = env.get_template("report.tex.j2")

    algorithm_names = sorted({model.model for result in summary.targets for model in result.models})
    primary_metric = (
        summary.targets[0].models[0].primary_metric
        if summary.targets and summary.targets[0].models
        else "score"
    )

    results_payload = []
    for target_result in summary.targets:
        results_payload.append(
            {
                "target": latex_service.escape_latex(target_result.target),
                "task": target_result.task,
                "best_model": latex_service.escape_latex(target_result.best_model),
                "best_params_tex": _params_to_tex(target_result.best_params or {}),
                "table": tables_by_target[target_result.target],
                "figures": figure_paths_by_target.get(target_result.target, []),
                "shap_note": shap_notes.get(target_result.target),
            }
        )

    context = {
        "reportdate": (summary.finished_at or datetime.now(UTC)).strftime("%B %d, %Y"),
        "nsamples": summary.n_samples,
        "nfeatures": summary.n_features,
        "nfeaturesused": len(summary.feature_columns),
        "ntargets": len(summary.targets),
        "targetplural": "" if len(summary.targets) == 1 else "s",
        "cvmode": summary.cv.get("mode", "single"),
        "nfolds": summary.cv.get("n_folds", 5),
        "nmodels": len(algorithm_names),
        "primarymetric": primary_metric,
        "algorithmlist": ", ".join(algorithm_names[:8]) + ("\\dots" if len(algorithm_names) > 8 else ""),
        "results": results_payload,
        "warnings": [latex_service.escape_latex(w) for w in warnings],
    }
    return template.render(**context)


def find_pdf_compiler() -> str | None:
    settings = get_settings()
    if settings.pdf_engine == "none":
        return None
    compiler = shutil.which("tectonic")
    if compiler is None and settings.pdf_engine != "tectonic":
        compiler = shutil.which("pdflatex")
    return compiler


def compile_pdf(tex_path: Path, timeout_s: int = 180) -> Path | None:
    compiler = find_pdf_compiler()
    if compiler is None:
        logger.info("no LaTeX engine found; shipping TeX sources without PDF")
        return None
    try:
        subprocess.run(
            [compiler, tex_path.name],
            cwd=str(tex_path.parent),
            capture_output=True,
            timeout=timeout_s,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("PDF compilation failed (%s); shipping TeX sources", exc)
        return None
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        return pdf_path
    return None


def build_report(
    summary: RunSummary,
    tables_by_target: dict[str, str],
    figure_paths_by_target: dict[str, list[str]],
    shap_notes: dict[str, str],
    run_warnings: list[str],
    out_dir: Path,
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_text = render_report_tex(summary, tables_by_target, figure_paths_by_target, shap_notes, run_warnings)
    tex_path = out_dir / "report.tex"
    tex_path.write_text(tex_text, encoding="utf-8")
    pdf_path = compile_pdf(tex_path)
    return tex_path, pdf_path
