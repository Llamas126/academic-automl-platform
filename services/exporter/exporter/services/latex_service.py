from __future__ import annotations

import math

from automl_shared.schemas.results import ModelResult, TargetResults

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
}


def escape_latex(text: str) -> str:
    out = []
    for char in str(text):
        if char in LATEX_SPECIALS:
            replacement = LATEX_SPECIALS[char]
            out.append(replacement)
        else:
            out.append(char)
    return "".join(out).replace(r"\textbackslash\{}", r"\textbackslash{}")


def decimals_for(std: float) -> int:
    if not math.isfinite(std) or std <= 0:
        return 2
    exponent = math.floor(math.log10(abs(std)))
    return max(0, 2 - exponent)


def format_pm(mean: float, std: float) -> str:
    decimals = decimals_for(std)
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def metric_arrow(direction: str) -> str:
    return r"$\uparrow$" if direction == "max" else r"$\downarrow$"


def _safe_label(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "_" for ch in name).lower()


def best_values_by_metric(models: list[ModelResult]) -> dict[str, tuple[float, int]]:
    best: dict[str, tuple[float, int]] = {}
    for model in models:
        for aggregate in model.metrics:
            direction = aggregate.direction
            value = aggregate.mean
            current = best.get(aggregate.metric)
            if current is None or _is_better(value, current[0], direction):
                best[aggregate.metric] = (value, model.rank)
    return best


def _is_better(candidate: float, incumbent: float, direction: str) -> bool:
    if not (math.isfinite(candidate)):
        return False
    if not math.isfinite(incumbent):
        return True
    return candidate > incumbent if direction == "max" else candidate < incumbent


def metrics_table_tex(target_results: TargetResults, n_samples: int, cv_mode: str, n_folds: int) -> str:
    ordered_metrics: list[str] = []
    seen: set[str] = set()
    for model in sorted(target_results.models, key=lambda m: m.rank):
        for aggregate in model.metrics:
            if aggregate.metric not in seen:
                seen.add(aggregate.metric)
                ordered_metrics.append(aggregate.metric)

    directions = {agg.metric: agg.direction for model in target_results.models for agg in model.metrics}

    best = best_values_by_metric(target_results.models)
    label = f"tab:{target_results.task}_{_safe_label(target_results.target)}"
    caption = (
        f"Out-of-fold test performance across {n_folds} "
        f"{'nested' if cv_mode == 'nested' else 'standard'} cross-validation folds "
        f"for target \\texttt{{{escape_latex(target_results.target)}}} "
        f"(n={n_samples}; mean $\\pm$ standard deviation; "
        "best mean per metric in bold)."
    )

    header_cells = ["Model"] + [
        f"{escape_latex(metric)} {metric_arrow(directions[metric])}" for metric in ordered_metrics
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(ordered_metrics) + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    for model in sorted(target_results.models, key=lambda m: m.rank):
        cells = [escape_latex(model.model)]
        by_metric = {agg.metric: agg for agg in model.metrics}
        for metric in ordered_metrics:
            agg = by_metric.get(metric)
            if agg is None:
                cells.append("--")
                continue
            cell = format_pm(agg.mean, agg.std)
            best_entry = best.get(metric)
            if best_entry is not None and best_entry[1] == model.rank and math.isfinite(agg.mean):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def standalone_table_document(table_fragment: str, title: str = "Table preview") -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage[margin=2.5cm]{geometry}",
            "\\begin{document}",
            f"\\section*{{{escape_latex(title)}}}",
            table_fragment,
            "\\end{document}",
        ]
    )
