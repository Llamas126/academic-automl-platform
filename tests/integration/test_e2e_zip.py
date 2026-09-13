import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "data"
FIXTURE_CSV = FIXTURES_DIR / "mixed_targets.csv"

TRAIN_PAYLOAD = {
    "dataset_id": "",
    "targets": [
        {"name": "score_continuous", "task": "regression"},
        {"name": "passed_threshold", "task": "classification"},
    ],
    "models": [
        "ridge_regression",
        "random_forest_regressor",
        "logistic_regression",
        "random_forest_classifier",
    ],
    "cv": {"n_folds": 3, "mode": "single", "random_state": 42},
    "hpo": {"enabled": False},
    "explain": {"shap": True},
    "export": {"figures_dpi": 150, "include_eps": False},
}


def _ensure_fixture() -> None:
    if not FIXTURE_CSV.exists():
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(FIXTURES_DIR.parent / "make_fixtures.py"), "--out", str(FIXTURES_DIR)],
            check=True,
        )


def _upload_csv(client: TestClient) -> str:
    with open(FIXTURE_CSV, "rb") as handle:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("mixed.csv", handle, "text/csv")},
        )
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def test_end_to_end_zip_bundle(client: TestClient):
    _ensure_fixture()
    dataset_id = _upload_csv(client)

    payload = dict(TRAIN_PAYLOAD)
    payload["dataset_id"] = dataset_id
    train_response = client.post("/api/v1/train", json=payload)
    assert train_response.status_code == 202, train_response.text
    job_id = train_response.json()["job_id"]

    status = client.get(f"/api/v1/status/{job_id}").json()
    assert status["state"] == "SUCCESS", status
    assert 0.0 <= status["progress"] <= 1.0
    assert status["stage"] == "done"
    assert status["download_ready"] is True

    download = client.get(f"/api/v1/download/{job_id}")
    assert download.status_code == 200
    bundle = zipfile.ZipFile(io.BytesIO(download.content))
    names = set(bundle.namelist())

    expected_prefixes = (
        "MANIFEST.txt",
        "metrics/results_summary.json",
        "metrics/per_fold.csv",
        "data/dataset_with_predictions.csv",
        "bib/references.bib",
        "report/report.tex",
        "latex/tables/",
        "figures/score_continuous/",
        "figures/passed_threshold/",
        "models/score_continuous_best_model.joblib",
        "models/passed_threshold_best_model.joblib",
    )
    missing = [prefix for prefix in expected_prefixes if not any(n.startswith(prefix) for n in names)]
    assert not missing, f"missing: {missing}"

    shap_figures = [n for n in names if "/shap/" in n and n.endswith(".png")]
    assert shap_figures, f"no SHAP figures in {sorted(names)}"

    def _figs_for(target: str) -> list[str]:
        return [n for n in names if n.startswith(f"figures/{target}/") and n.endswith(".png")]

    for target in ("score_continuous", "passed_threshold"):
        figs = _figs_for(target)
        basenames = {n.rsplit("/", 1)[-1] for n in figs}
        assert "model_feature_importance_comparison.png" in basenames, figs
        assert any(n.startswith("coef_plot_") for n in basenames), figs
        assert any(n.startswith("shap_importance_") for n in basenames), figs

    score_figs = {n.rsplit("/", 1)[-1] for n in _figs_for("score_continuous")}
    assert any(n.startswith("shap_beeswarm_") for n in score_figs), score_figs

    tree_flowcharts = [n for n in names if "/decision_tree_flowchart_" in n and n.endswith(".png")]
    assert (
        "figures/passed_threshold/decision_tree_flowchart_random_forest_classifier.png" in tree_flowcharts
    ), f"expected the decision-tree flowchart for the tree winner (passed_threshold): {tree_flowcharts}"

    model_paths = [n for n in names if n.startswith("models/") and n.endswith(".joblib")]
    best_models = [n for n in model_paths if n.endswith("_best_model.joblib")]
    candidate_models = [n for n in model_paths if not n.endswith("_best_model.joblib")]
    assert len(best_models) == 2, model_paths
    assert len(candidate_models) == 2, model_paths
    for target in ("score_continuous", "passed_threshold"):
        assert any(n.startswith(f"models/{target}_") for n in candidate_models), candidate_models

    manifest = bundle.read("MANIFEST.txt").decode("utf-8")
    assert "job_id" in manifest and "Library versions" in manifest

    report_tex = bundle.read("report/report.tex").decode("utf-8")
    assert "../figures/passed_threshold/decision_tree_flowchart_random_forest_classifier.png" in report_tex

    summary = json.loads(bundle.read("metrics/results_summary.json"))
    tasks = {t["target"]: t["task"] for t in summary["targets"]}
    assert tasks == {"score_continuous": "regression", "passed_threshold": "classification"}

    predictions_csv = bundle.read("data/dataset_with_predictions.csv").decode("utf-8")
    header = predictions_csv.splitlines()[0]
    assert "score_continuous_predicted" in header
    assert "passed_threshold_predicted" in header
