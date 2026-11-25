"""Curves from a pericyte dilation and inlet-pressure sweep."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from haemolynx.visualization.perturbation_plots import (
    SweepAxisSpec,
    plot_sweep_curves,
)

#: What is plotted against dilation: result column -> (file stem, axis label, title).
#: Kept for importers that still read this table; plotting goes through
#: :func:`plot_sweep_curves` with the pericyte axis below.
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

_PERICYTE_AXIS = SweepAxisSpec(
    x_key="dilation_percent",
    x_label="Pericyte dilation (%)",
    series_key="inlet_pressure_pa",
    series_label="{value} Pa inlet",
    stem_suffix="vs_pericyte_dilation",
    title_subject="pericyte dilation",
)


def plot_dilation_curves(
    results: Iterable[Mapping[str, object]], output_dir: Path | str
) -> dict[str, str]:
    """Plot each swept quantity against dilation, one line per inlet pressure.

    Returns the written paths, keyed ``<column>_plot_path``.
    """
    return plot_sweep_curves(results, output_dir, axis=_PERICYTE_AXIS)
