import json
import struct
import zlib


def _tiny_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + chunk(b"IEND", b"")


def _summary_payload() -> dict:
    return {
        "job_id": "job_res1",
        "dataset_id": "ds_1",
        "filename": "mixed.csv",
        "n_samples": 120,
        "n_features": 4,
        "feature_columns": ["x1", "x2", "category"],
        "cv": {"n_folds": 2, "mode": "single"},
        "hpo": {"enabled": False},
        "targets": [
            {
                "target": "y",
                "task": "regression",
                "label_encoder_classes": None,
                "best_model": "Ridge Regression",
                "best_params": {"alpha": 1.0},
                "models": [
                    {
                        "model": "Ridge Regression",
                        "citation_key": None,
                        "best_params": {"alpha": 1.0},
                        "primary_metric": "r2",
                        "primary_mean": 0.9,
                        "rank": 1,
                        "metrics": [
                            {
                                "metric": "r2",
                                "direction": "max",
                                "mean": 0.9,
                                "std": 0.02,
                                "ci95_low": 0.85,
                                "ci95_high": 0.95,
                                "folds": [0.88, 0.92],
                            }
                        ],
                    }
                ],
            }
        ],
        "warnings": [],
        "library_versions": {"scikit-learn": "1.9.0"},
        "runtime_s": 1.5,
    }


def _seed_success_job(data_dir, job_id: str = "job_res1", with_figures: bool = True) -> None:
    from app.services.job_service import get_job_store

    get_job_store().create(job_id=job_id, dataset_id="ds_1", request_payload={})
    get_job_store().update(job_id, state="SUCCESS", progress=1.0, download_ready=True)

    out_dir = data_dir / "jobs" / job_id / "output"
    (out_dir / "metrics").mkdir(parents=True)
    (out_dir / "metrics" / "results_summary.json").write_text(json.dumps(_summary_payload()))
    if with_figures:
        shap_dir = out_dir / "figures" / "y" / "shap"
        shap_dir.mkdir(parents=True)
        (shap_dir / "summary_plot.png").write_bytes(_tiny_png())


def test_results_endpoint_returns_summary(client, isolated_data_dir):
    _seed_success_job(isolated_data_dir)
    response = client.get("/api/v1/results/job_res1")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job_res1"
    assert body["targets"][0]["best_model"] == "Ridge Regression"
    assert body["n_features"] == 4


def test_results_requires_success_state(client, isolated_data_dir):
    from app.services.job_service import get_job_store

    get_job_store().create(job_id="job_run1", dataset_id="ds", request_payload={})
    response = client.get("/api/v1/results/job_run1")
    assert response.status_code == 409


def test_results_unknown_job_404(client):
    response = client.get("/api/v1/results/does_not_exist")
    assert response.status_code == 404


def test_figures_listing_and_serving(client, isolated_data_dir):
    _seed_success_job(isolated_data_dir)
    listing = client.get("/api/v1/results/job_res1/figures")
    assert listing.status_code == 200
    assert listing.json()["figures"] == ["y/shap/summary_plot.png"]

    served = client.get("/api/v1/results/job_res1/figures/y/shap/summary_plot.png")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content.startswith(b"\x89PNG")


def test_figure_traversal_and_extension_blocked(client, isolated_data_dir):
    _seed_success_job(isolated_data_dir)

    # HTTP clients normalize "../" before sending; the raw request collapses to an unrouted path.
    assert (
        client.get("/api/v1/results/job_res1/figures/../../metrics/results_summary.json").status_code == 404
    )

    import pytest

    from app.api.v1.results import get_figure
    from app.core.exceptions import NotFoundError, ValidationFailedError

    # True escape above the figures root -> blocked by containment guard.
    with pytest.raises(ValidationFailedError):
        get_figure("job_res1", "y/../../../metrics/results_summary.json")
    # Stays inside the tree but wrong extension -> blocked by suffix check.
    with pytest.raises(NotFoundError):
        get_figure("job_res1", "y/shap/../../metrics/results_summary.json")

    wrong_ext_url = "/api/v1/results/job_res1/figures/y/shap/../..%2Fmetrics%2Fresults_summary.json"
    assert client.get(wrong_ext_url).status_code in {404, 422}

    missing = client.get("/api/v1/results/job_res1/figures/y/nope.png")
    assert missing.status_code == 404


def test_figures_empty_when_shap_disabled(client, isolated_data_dir):
    _seed_success_job(isolated_data_dir, with_figures=False)
    listing = client.get("/api/v1/results/job_res1/figures")
    assert listing.status_code == 200
    assert listing.json()["figures"] == []
