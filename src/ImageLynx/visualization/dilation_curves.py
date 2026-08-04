"""Curves from a pericyte dilation and inlet-pressure sweep."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#: What is plotted against dilation: result column -> (file stem, axis label, title).
CURVES = {
    "equivalent_resistance": (
        "resistance_vs_pericyte_dilation",
        "Equivalent resistance (Pa.s/m^3)",
        "Resistance vs pericyte dilation",
    ),
    "total_inlet_flow": (
        "flow_vs_pericyte_dilation",
        "Total inlet flow (m^3/s)",
        "Flow vs pericyte dilation",
    ),
}


def plot_dilation_curves(
    results: Iterable[Mapping[str, object]], output_dir: Path | str
) -> dict[str, str]:
    """Plot each swept quantity against dilation, one line per inlet pressure.

    Returns the written paths, keyed ``<column>_plot_path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_pressure: dict[int, list[Mapping[str, object]]] = {}
    for row in results:
        by_pressure.setdefault(int(row["inlet_pressure_pa"]), []).append(row)
    for rows in by_pressure.values():
        rows.sort(key=lambda r: int(r["dilation_percent"]))

    written: dict[str, str] = {}
    for column, (stem, y_label, title) in CURVES.items():
        figure, axes = plt.subplots(figsize=(8, 5))
        for pressure in sorted(by_pressure):
            rows = by_pressure[pressure]
            axes.plot(
                [int(r["dilation_percent"]) for r in rows],
                [float(r[column]) for r in rows],
                marker="o",
                linewidth=2.0,
                label=f"{pressure} Pa inlet",
            )
        axes.set_xlabel("Pericyte dilation (%)")
        axes.set_ylabel(y_label)
        axes.set_title(title)
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()

        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written[f"{column}_plot_path"] = str(path)

    logger.info(f"Dilation curves saved to: {', '.join(sorted(written.values()))}")
    return written
