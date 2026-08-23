import os

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    os.environ["DATA_DIR"] = str(data_dir)
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()
    os.environ.pop("DATA_DIR", None)


@pytest.fixture
def client(isolated_data_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


@pytest.fixture
def mixed_targets_csv() -> bytes:
    import io

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 120
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    category = rng.choice(["A", "B", "C"], n)
    df = pd.DataFrame(
        {
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "category": category,
            "score_continuous": np.round(3 * x1 + 2 * x2 + rng.normal(0, 0.3, n), 4),
            "grade": pd.cut(
                x1 + 0.5 * (x2 - 5) / 2 + rng.normal(0, 0.2, n),
                bins=[-np.inf, -0.5, 0.8, np.inf],
                labels=["low", "mid", "high"],
            ).astype(str),
        }
    )
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()
