from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_PUBLICATION_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
}


def configure_publication_style(dpi: int = 300) -> None:
    rc = dict(_PUBLICATION_RC)
    rc["savefig.dpi"] = dpi
    plt.rcParams.update(rc)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, include_eps: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, format="png")
    written.append(png_path)
    if include_eps:
        eps_path = out_dir / f"{stem}.eps"
        try:
            fig.savefig(eps_path, format="eps")
            written.append(eps_path)
        except Exception as exc:
            logger.warning("EPS export failed for %s: %s", stem, exc)
    plt.close(fig)
    return written
