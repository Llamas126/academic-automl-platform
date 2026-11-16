"""Generate synthetic fixtures for end-to-end testing.

Usage:
    python tests/fixtures/make_fixtures.py [--out data/fixtures]

Creates:
    mixed_targets.csv   - regression + binary classification targets
    multiclass.csv      - 3-class classification target
    nasa_power_mango.csv - NASA POWER daily climate export (with preamble)
    mango_local.csv     - farm harvest records (``FECHA``, day-first)
    mango_loss.csv      - farm loss records (``Peso_Perdido_KG``)
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


def make_nasa_power_csv(start: str = "2024-01-11", n_days: int = 120, seed: int = 42) -> str:
    """NASA POWER-shaped CSV with a free-text preamble and ``-999`` gaps."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    season = np.sin(2 * np.pi * (dates.dayofyear - 60) / 365.25)
    frame = pd.DataFrame(
        {
            "YEAR": dates.year,
            "DOY": dates.dayofyear,
            "T2M": (27.0 + 2.5 * season + rng.normal(0, 0.4, n_days)).round(2),
            "T2M_MAX": (31.0 + 3.0 * season + rng.normal(0, 0.5, n_days)).round(2),
            "T2M_MIN": (23.0 + 2.0 * season + rng.normal(0, 0.5, n_days)).round(2),
            "T2MDEW": (21.0 + 1.5 * season + rng.normal(0, 0.6, n_days)).round(2),
            "PRECTOTCORR": np.clip(rng.gamma(0.7, 3.0, n_days), 0, None).round(2),
            "RH2M": np.clip(78.0 + rng.normal(0, 6, n_days), 30, 100).round(1),
            "WS2M": np.clip(rng.normal(2.5, 0.8, n_days), 0.1, None).round(2),
            "ALLSKY_SFC_SW_DWN": np.clip(rng.normal(18.0, 3.0, n_days), 0, None).round(2),
        }
    )
    if n_days > 5:
        frame.loc[frame.index[5], "PRECTOTCORR"] = -999
    if n_days > 17:
        frame.loc[frame.index[17], "RH2M"] = -999

    header = "\n".join(
        [
            "-BEGIN HEADER-",
            "NASA/POWER Source Native Resolution Daily Data ",
            f"Dates (month/day/year): {dates[0]:%m/%d/%Y} through {dates[-1]:%m/%d/%Y} in LST",
            "Location: latitude  10.85   longitude -74.78 ",
            "The value for missing source data that cannot be computed or is outside of "
            "the sources availability range: -999 ",
            "-END HEADER-",
        ]
    )
    return header + "\n" + frame.to_csv(index=False)


def make_mango_local_records(start: str = "2024-01-11", n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Farm harvest records using the real export headers (``FECHA``, day-first)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    weights = np.where(rng.random(n_days) < 0.45, np.round(rng.gamma(2.0, 12.0, n_days), 2), 0.0)
    return pd.DataFrame(
        {
            "FECHA": dates.strftime("%d-%m-%Y"),
            "PESO (PRODUCCIÓN)": weights,
            "LOTE": rng.choice(["A", "B"], n_days),
        }
    )


def make_mango_loss_records(start: str = "2024-01-11", n_days: int = 120, seed: int = 11) -> pd.DataFrame:
    """Loss records, deliberately on different days than the harvest file."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    weights = np.where(rng.random(n_days) < 0.2, np.round(rng.gamma(2.0, 6.0, n_days), 2), 0.0)
    return pd.DataFrame(
        {
            "FECHA": dates.strftime("%d/%m/%Y"),
            "Peso_Perdido_KG": weights,
            "Motivo": rng.choice(["impacto", "sol", "pudricion"], n_days),
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

    (out_dir / "nasa_power_mango.csv").write_text(make_nasa_power_csv(seed=args.seed), encoding="utf-8")
    make_mango_local_records(seed=args.seed).to_csv(out_dir / "mango_local.csv", index=False)
    make_mango_loss_records(seed=args.seed).to_csv(out_dir / "mango_loss.csv", index=False)
    print(f"fixtures written to {out_dir}")


if __name__ == "__main__":
    main()
