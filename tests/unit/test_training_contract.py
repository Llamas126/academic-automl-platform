import json
import zipfile

import pandas as pd
import pytest

from app.ml.contract import (
    META_NAME,
    load_background_frame,
    load_summary,
    load_target_views,
    persist_training_output,
)
from app.ml.pipeline import run_training
from app.schemas.training import CVConfig, HPOConfig, TargetIn, TrainRequest
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


def _request() -> TrainRequest:
    return TrainRequest(
        dataset_id="ds_test",
        targets=[
            TargetIn(name="score_continuous", task="regression"),
            TargetIn(name="grade", task="classification"),
        ],
        models=[
            "ridge_regression",
            "random_forest_regressor",
            "logistic_regression",
            "random_forest_classifier",
        ],
        cv=CVConfig(n_folds=3, mode="single", random_state=7),
        hpo=HPOConfig(enabled=False),
        export={"figures_dpi": 150, "include_eps": False},
    )


def test_persist_contract_writes_worker_inputs(tmp_path, tiny_frame):
    request = _request()
    output = run_training(request, "ds_test", "tiny.csv", tiny_frame, "job_c")
    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"

    persist_training_output(output, work_dir, out_dir)

    meta = json.loads((work_dir / META_NAME).read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert {t["target"] for t in meta["targets"]} == {"score_continuous", "grade"}
    for entry in meta["targets"]:
        assert entry["explainer_kind"]
        assert entry["best_entry_key"]

    background = load_background_frame(work_dir)
    assert background is not None
    assert len(background) == len(output.background_frame)

    views = load_target_views(work_dir, out_dir / "models")
    for name, view in views.items():
        original = output.artifacts[name]
        assert view.task == original.task
        assert view.best_entry_key == original.best_entry_key
        assert view.explainer_kind == original.explainer_kind
        assert view.confusion_matrix == original.confusion_matrix
        assert view.final_pipeline is None
    pipelines_view = load_target_views(work_dir, out_dir / "models", with_pipelines=True)
    assert pipelines_view["grade"].final_pipeline is not None

    summary = load_summary(out_dir)
    assert summary.job_id == "job_c"
    assert {t.target for t in summary.targets} == {"score_continuous", "grade"}


def test_file_mode_export_matches_in_memory_export(tmp_path, tiny_frame):
    request = _request()
    output = run_training(request, "ds_test", "tiny.csv", tiny_frame, "job_eq")

    memory_out = tmp_path / "mem_out"
    memory_zip = export_and_package(
        job_id="job_eq",
        output=output,
        request=request,
        work_dir=tmp_path / "mem_work",
        out_dir=memory_out,
        zip_path=tmp_path / "memory.zip",
    )

    file_work = tmp_path / "file_work"
    file_out = tmp_path / "file_out"
    persist_training_output(output, file_work, file_out)
    file_zip = export_and_package(
        job_id="job_eq",
        output=None,
        request=request,
        work_dir=file_work,
        out_dir=file_out,
        zip_path=tmp_path / "files.zip",
    )

    memory_names = set(zipfile.ZipFile(memory_zip).namelist())
    file_names = set(zipfile.ZipFile(file_zip).namelist())
    assert memory_names == file_names

    memory_bundle = zipfile.ZipFile(memory_zip)
    file_bundle = zipfile.ZipFile(file_zip)
    assert (
        memory_bundle.read("metrics/results_summary.json")
        == file_bundle.read("metrics/results_summary.json")
    )
    assert memory_bundle.read("MANIFEST.txt") == file_bundle.read("MANIFEST.txt")
    assert memory_bundle.read("metrics/per_fold.csv") == file_bundle.read("metrics/per_fold.csv")
