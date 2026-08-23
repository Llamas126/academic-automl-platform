"""Generate synthetic fixtures for end-to-end testing.

Usage:
    python tests/fixtures/make_fixtures.py [--out data/fixtures]

Creates:
    mixed_targets.csv   - regression + binary classification targets
    multiclass.csv      - 3-class classification target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_mixed(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 12, n_rows)
    x2 = rng.normal(0, 1, n_rows)
    grp = rng.choice(["alpha", "beta", "gamma"], n_rows, p=[0.5, 0.3, 0.2])
    noise = rng.normal(0, 4, n_rows)

    group_effect = np.select([grp == "alpha", grp == "beta"], [5.0, -3.0], default=0.0)
    score = 0.8 * x1 + 10 * x2 + group_effect + noise
    passed = ((score > score.mean()) & (x2 > -0.5)).astype(int)

    df = pd.DataFrame(
        {
            "participant_id": [f"P{idx:05d}" for idx in range(1, n_rows + 1)],
            "site_notes": rng.choice(["north campus lab", "south annex"], n_rows),
            "age_years": np.clip(rng.normal(35, 10, n_rows), 18, 80).round().astype(int),
            "biomarker_x1": x1.round(3),
            "z_score_x2": x2.round(4),
            "treatment_group": grp,
            "score_continuous": score.round(3),
            "passed_threshold": np.where(passed == 1, "yes", "no"),
        }
    )
    missing_idx = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
    df.loc[missing_idx[: len(missing_idx) // 2], "biomarker_x1"] = np.nan
    df.loc[missing_idx[len(missing_idx) // 2 :], "treatment_group"] = None
    return df


def make_multiclass(n_rows: int = 360, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    labels = rng.integers(0, 3, n_rows)
    points = centers[labels] + rng.normal(0, 1.1, (n_rows, 2))
    species = np.array(["setosa_a", "viridis_b", "candidus_c"])[labels]
    return pd.DataFrame(
        {
            "sensor_1": points[:, 0].round(4),
            "sensor_2": points[:, 1].round(4),
            "batch": rng.choice(["b1", "b2"], n_rows),
            "species": species,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/fixtures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_mixed(seed=args.seed).to_csv(out_dir / "mixed_targets.csv", index=False)
    make_multiclass(seed=args.seed).to_csv(out_dir / "multiclass.csv", index=False)
    print(f"fixtures written to {out_dir}")


if __name__ == "__main__":
    main()
