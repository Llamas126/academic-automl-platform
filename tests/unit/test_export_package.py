import json
import zipfile

import pandas as pd
import pytest

from app.ml.pipeline import run_training
from app.schemas.training import CVConfig, HPOConfig, TargetIn, TrainRequest
from app.services.bibtex_service import references_bibtex
from app.services.latex_service import (
    decimals_for,
    escape_latex,
    format_pm,
    metrics_table_tex,
)
from app.services.package_service import export_and_package


@pytest.fixture
def tiny_frame():
    rng = pd.Series(range(90))
    return pd.DataFrame(
        {
            "x1": rng.map(lambda v: v * 0.1),
            "grp": ["a", "b", "c"] * 30,
            "score_continuous": rng.map(lambda v: 2.0 + 0.5 * v),
            "grade": ["low", "mid", "high"] * 30,
        }
    )


def _request(models: list[str]) -> TrainRequest:
    return TrainRequest(
        dataset_id="ds_test",
        targets=[
            TargetIn(name="score_continuous", task="regression"),
            TargetIn(name="grade", task="classification"),
        ],
        models=models,
        cv=CVConfig(n_folds=3, mode="single", random_state=7),
        hpo=HPOConfig(enabled=False),
    )


def test_escape_latex_specials():
    assert escape_latex("a_b & 50%") == r"a\_b \& 50\%"
    assert escape_latex("100$") == r"100\$"


def test_format_pm_adaptive_decimals():
    assert decimals_for(0.45) == 3
    text = format_pm(12.3104, 0.45)
    assert "$\\pm$" in text
    assert text.startswith("12.310")


def test_metrics_table_tex_structure(tiny_frame):
    from app.schemas.results import MetricAggregate, ModelResult, TargetResults

    def agg(mean: float, std: float) -> MetricAggregate:
        return MetricAggregate(
            metric="rmse",
            direction="min",
            mean=mean,
            std=std,
            ci95_low=mean - std,
            ci95_high=mean + std,
            folds=[mean - 0.1, mean + 0.1],
        )

    result = TargetResults(
        target="score",
        task="regression",
        best_model="ridge",
        best_params={},
        models=[
            ModelResult(
                model="ridge",
                primary_metric="rmse",
                primary_mean=1.5,
                rank=1,
                metrics=[agg(1.5, 0.1)],
            ),
            ModelResult(
                model="lasso",
                primary_metric="rmse",
                primary_mean=2.0,
                rank=2,
                metrics=[agg(2.0, 0.2)],
            ),
        ],
    )
    tex = metrics_table_tex(result, n_samples=90, cv_mode="nested", n_folds=3)
    assert "\\begin{tabular}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\textbf{" in tex
    assert "\\label{tab:regression_score}" in tex


def test_references_bibtex_includes_versions_and_skips_unknown():
    bib = references_bibtex(
        {"pedregosa2011scikit", "not_a_key"},
        {"scikit-learn": "1.9.0"},
    )
    assert "@article{pedregosa2011scikit," in bib
    assert "Version 1.9.0" in bib
    assert "not_a_key" not in bib


def test_export_and_package_zip_layout(tmp_path, tiny_frame):
    request = _request(
        [
            "ridge_regression",
            "random_forest_regressor",
            "logistic_regression",
            "random_forest_classifier",
        ]
    )
    output = run_training(
        request,
        dataset_id="ds_test",
        filename="tiny.csv",
        df_raw=tiny_frame,
        job_id="job_x",
    )
    out_dir = tmp_path / "out"
    zip_path = export_and_package(
        job_id="job_x",
        output=output,
        request=request,
        work_dir=tmp_path / "work",
        out_dir=out_dir,
        zip_path=tmp_path / "bundle.zip",
    )

    names = set(zipfile.ZipFile(zip_path).namelist())
    required = {
        "MANIFEST.txt",
        "metrics/results_summary.json",
        "metrics/per_fold.csv",
        "data/dataset_with_predictions.csv",
        "bib/references.bib",
        "report/report.tex",
    }
    for target_dir in ("score_continuous", "grade"):
        if target_dir == "grade":
            required |= {f"figures/{target_dir}/confusion_matrix_{target_dir}.png"}
        else:
            required |= {f"figures/{target_dir}/pred_vs_actual_{target_dir}.png"}
        required |= {f"models/{target_dir}_best_model.joblib"}
    missing = {name for name in required if name not in names}
    assert not missing, f"missing entries: {missing}"

    summary = json.loads(zipfile.ZipFile(zip_path).read("metrics/results_summary.json"))
    assert len(summary["targets"]) == 2

    table_names = [n for n in names if n.startswith("latex/tables/table_")]
    assert len(table_names) == 2

    shap_pngs = [n for n in names if "shap" in n and n.endswith(".png")]
    assert shap_pngs, "expected SHAP figures in bundle"

    report_tex = zipfile.ZipFile(zip_path).read("report/report.tex").decode("utf-8")
    assert "\\begin{abstract}" in report_tex
    assert "\\bibliography{../bib/references}" in report_tex
