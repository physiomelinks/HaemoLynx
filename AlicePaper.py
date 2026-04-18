"""Alice paper plotting utilities for pericyte dilation sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def graph(records: Iterable[dict], output_dir: Path | str) -> dict:
    """Plot resistance and flow curves vs pericyte dilation for each inlet pressure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict]] = {}
    for row in records:
        pressure = int(row["inlet_pressure_pa"])
        grouped.setdefault(pressure, []).append(row)

    for pressure_rows in grouped.values():
        pressure_rows.sort(key=lambda r: int(r["dilation_percent"]))

    resistance_plot_path = output_path / "alice_resistance_vs_pericyte_dilation.png"
    flow_plot_path = output_path / "alice_flow_vs_pericyte_dilation.png"

    fig_res, ax_res = plt.subplots(figsize=(8, 5))
    for pressure in sorted(grouped.keys()):
        rows = grouped[pressure]
        x = [int(r["dilation_percent"]) for r in rows]
        y = [float(r["equivalent_resistance"]) for r in rows]
        ax_res.plot(x, y, marker="o", linewidth=2.0, label=f"{pressure} Pa inlet")
    ax_res.set_xlabel("Pericyte dilation (%)")
    ax_res.set_ylabel("Equivalent resistance")
    ax_res.set_title("Resistance vs Pericyte Dilation")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend()
    fig_res.tight_layout()
    fig_res.savefig(resistance_plot_path, dpi=200)
    plt.close(fig_res)

    fig_flow, ax_flow = plt.subplots(figsize=(8, 5))
    for pressure in sorted(grouped.keys()):
        rows = grouped[pressure]
        x = [int(r["dilation_percent"]) for r in rows]
        y = [float(r["total_inlet_flow"]) for r in rows]
        ax_flow.plot(x, y, marker="o", linewidth=2.0, label=f"{pressure} Pa inlet")
    ax_flow.set_xlabel("Pericyte dilation (%)")
    ax_flow.set_ylabel("Total inlet flow")
    ax_flow.set_title("Flow vs Pericyte Dilation")
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend()
    fig_flow.tight_layout()
    fig_flow.savefig(flow_plot_path, dpi=200)
    plt.close(fig_flow)

    return {
        "resistance_plot_path": str(resistance_plot_path),
        "flow_plot_path": str(flow_plot_path),
    }
