"""Alice paper plotting utilities for pericyte dilation sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

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


def _is_capillary_branch_order(branch_order: object, capillary_branch_prefix: str) -> bool:
    if branch_order is None:
        return False
    return str(branch_order).startswith(str(capillary_branch_prefix))


def _is_arteriole_branch_order(branch_order: object, arteriole_branch_prefix: str) -> bool:
    if branch_order is None:
        return False
    return str(branch_order).startswith(str(arteriole_branch_prefix))


def _scale_passive_capillary_edge_fwhm_diameters(
    graph_with_branch_orders,
    *,
    capillary_dilation_factor: float,
    capillary_branch_prefix: str,
) -> None:
    """Scale only capillary edge passive diameters in-place."""
    for _, _, _, edge_data in graph_with_branch_orders.edges(keys=True, data=True):
        if not _is_capillary_branch_order(edge_data.get("branch_order"), capillary_branch_prefix):
            continue
        fwhm_diameter_um = edge_data.get("fwhm_diameter_um")
        if fwhm_diameter_um is None:
            continue
        if float(fwhm_diameter_um) <= 0.0:
            continue
        edge_data["fwhm_diameter_um"] = float(fwhm_diameter_um) * capillary_dilation_factor


def _scale_passive_capillary_branch_diameters(
    diameter_by_branch_order: Mapping[object, float],
    *,
    capillary_dilation_factor: float,
    capillary_branch_prefix: str,
) -> dict[object, float]:
    """Return branch-order diameters with only capillary classes scaled."""
    out = dict(diameter_by_branch_order)
    for branch_order, diameter_um in list(out.items()):
        if not _is_capillary_branch_order(branch_order, capillary_branch_prefix):
            continue
        out[branch_order] = float(diameter_um) * capillary_dilation_factor
    return out


def _shift_passive_arteriole_edge_fwhm_diameters(
    graph_with_branch_orders,
    *,
    arteriole_diameter_delta_um: float,
    arteriole_branch_prefix: str,
) -> None:
    """Shift only arteriole edge passive diameters in-place by a fixed delta."""
    for u, v, key, edge_data in graph_with_branch_orders.edges(keys=True, data=True):
        if not _is_arteriole_branch_order(edge_data.get("branch_order"), arteriole_branch_prefix):
            continue
        fwhm_diameter_um = edge_data.get("fwhm_diameter_um")
        if fwhm_diameter_um is None:
            continue
        if float(fwhm_diameter_um) <= 0.0:
            continue
        shifted = float(fwhm_diameter_um) + float(arteriole_diameter_delta_um)
        if shifted <= 0.0:
            raise ValueError(
                "arteriole_diameter_delta_um yields non-positive arteriole edge diameter "
                f"for edge ({u}, {v}, {key})."
            )
        edge_data["fwhm_diameter_um"] = shifted


def _shift_passive_arteriole_branch_diameters(
    diameter_by_branch_order: Mapping[object, float],
    *,
    arteriole_diameter_delta_um: float,
    arteriole_branch_prefix: str,
) -> dict[object, float]:
    """Return branch-order diameters with only arterioles shifted by a fixed delta."""
    out = dict(diameter_by_branch_order)
    for branch_order, diameter_um in list(out.items()):
        if not _is_arteriole_branch_order(branch_order, arteriole_branch_prefix):
            continue
        shifted = float(diameter_um) + float(arteriole_diameter_delta_um)
        if shifted <= 0.0:
            raise ValueError(
                "arteriole_diameter_delta_um yields non-positive branch-order diameter "
                f"for branch '{branch_order}'."
            )
        out[branch_order] = shifted
    return out


def passive_capillary_diameter_beforeafter(
    *,
    graph_with_branch_orders,
    diameter_by_branch_order: Mapping[object, float],
    poiseuille_model,
    solve_pressure_and_boundary_flow: Callable[..., Mapping[str, object]],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
    inlet_pressures_pa: Sequence[int],
    output_p_bc: float,
    capillary_dilation_percent: float,
    output_dir: Path | str,
    capillary_branch_prefix: str = "B",
) -> dict:
    """Compare flow before/after capillary-only passive dilation and plot changes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capillary_dilation_factor = 1.0 + (float(capillary_dilation_percent) / 100.0)
    if capillary_dilation_factor <= 0.0:
        raise ValueError(
            "capillary_dilation_percent produces non-positive diameters; "
            f"got percent={capillary_dilation_percent}."
        )

    baseline_graph = graph_with_branch_orders.copy()
    dilated_graph = graph_with_branch_orders.copy()
    _scale_passive_capillary_edge_fwhm_diameters(
        dilated_graph,
        capillary_dilation_factor=capillary_dilation_factor,
        capillary_branch_prefix=capillary_branch_prefix,
    )

    baseline_diameter_by_branch_order = dict(diameter_by_branch_order)
    dilated_diameter_by_branch_order = _scale_passive_capillary_branch_diameters(
        diameter_by_branch_order,
        capillary_dilation_factor=capillary_dilation_factor,
        capillary_branch_prefix=capillary_branch_prefix,
    )

    baseline_graph, _ = poiseuille_model.set_poiseuille_weights(
        baseline_graph,
        baseline_diameter_by_branch_order,
        prefer_edge_fwhm_diameter=True,
    )
    dilated_graph, _ = poiseuille_model.set_poiseuille_weights(
        dilated_graph,
        dilated_diameter_by_branch_order,
        prefer_edge_fwhm_diameter=True,
    )

    import numpy as np
    from ImageLynx import haemodynamics

    baseline_conductance, baseline_node_list = haemodynamics.build_conductance_matrix_from_graph(
        baseline_graph
    )
    dilated_conductance, dilated_node_list = haemodynamics.build_conductance_matrix_from_graph(
        dilated_graph
    )

    rows: list[dict[str, float]] = []
    for inlet_pressure_pa in inlet_pressures_pa:
        baseline = solve_pressure_and_boundary_flow(
            conductance=baseline_conductance,
            node_list=baseline_node_list,
            input_p_bc=float(inlet_pressure_pa),
            output_p_bc=float(output_p_bc),
            starting_nodes=list(starting_nodes),
            output_nodes=list(output_nodes),
        )
        dilated = solve_pressure_and_boundary_flow(
            conductance=dilated_conductance,
            node_list=dilated_node_list,
            input_p_bc=float(inlet_pressure_pa),
            output_p_bc=float(output_p_bc),
            starting_nodes=list(starting_nodes),
            output_nodes=list(output_nodes),
        )
        baseline_flow = float(baseline["total_inlet_flow"])
        dilated_flow = float(dilated["total_inlet_flow"])
        delta_flow = dilated_flow - baseline_flow
        delta_percent = np.nan
        if not np.isclose(baseline_flow, 0.0):
            delta_percent = (delta_flow / baseline_flow) * 100.0
        rows.append(
            {
                "inlet_pressure_pa": float(inlet_pressure_pa),
                "baseline_total_inlet_flow": baseline_flow,
                "capillary_dilated_total_inlet_flow": dilated_flow,
                "flow_delta": delta_flow,
                "flow_delta_percent": float(delta_percent),
            }
        )

    flow_change_plot_path = (
        output_path
        / f"alice_flow_change_capillary_passive_dilation_{int(round(capillary_dilation_percent))}pct.png"
    )

    x = [float(r["inlet_pressure_pa"]) for r in rows]
    baseline_y = [float(r["baseline_total_inlet_flow"]) for r in rows]
    dilated_y = [float(r["capillary_dilated_total_inlet_flow"]) for r in rows]
    delta_y = [float(r["flow_delta"]) for r in rows]

    fig, (ax_flow, ax_delta) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_flow.plot(x, baseline_y, marker="o", linewidth=2.0, label="Baseline (no capillary dilation)")
    ax_flow.plot(
        x,
        dilated_y,
        marker="o",
        linewidth=2.0,
        label=f"Capillary passive dilation +{float(capillary_dilation_percent):.1f}%",
    )
    ax_flow.set_ylabel("Total inlet flow")
    ax_flow.set_title("Flow Before vs After Capillary-Only Passive Dilation")
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend()

    ax_delta.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax_delta.plot(x, delta_y, marker="o", linewidth=2.0, color="tab:green")
    ax_delta.set_xlabel("Inlet pressure (Pa)")
    ax_delta.set_ylabel("Flow change")
    ax_delta.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(flow_change_plot_path, dpi=200)
    plt.close(fig)

    return {
        "flow_change_plot_path": str(flow_change_plot_path),
        "capillary_dilation_percent": float(capillary_dilation_percent),
        "capillary_dilation_factor": float(capillary_dilation_factor),
        "rows": rows,
    }


def passive_arteriole_diameter_beforeafter(
    *,
    graph_with_branch_orders,
    diameter_by_branch_order: Mapping[object, float],
    poiseuille_model,
    solve_pressure_and_boundary_flow: Callable[..., Mapping[str, object]],
    starting_nodes: Sequence[int],
    output_nodes: Sequence[int],
    inlet_pressures_pa: Sequence[int],
    output_p_bc: float,
    arteriole_diameter_delta_um: float,
    output_dir: Path | str,
    arteriole_branch_prefix: str = "Art",
) -> dict:
    """Compare flow before/after arteriole-only passive diameter shift."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline_graph = graph_with_branch_orders.copy()
    shifted_graph = graph_with_branch_orders.copy()
    _shift_passive_arteriole_edge_fwhm_diameters(
        shifted_graph,
        arteriole_diameter_delta_um=float(arteriole_diameter_delta_um),
        arteriole_branch_prefix=arteriole_branch_prefix,
    )

    baseline_diameter_by_branch_order = dict(diameter_by_branch_order)
    shifted_diameter_by_branch_order = _shift_passive_arteriole_branch_diameters(
        diameter_by_branch_order,
        arteriole_diameter_delta_um=float(arteriole_diameter_delta_um),
        arteriole_branch_prefix=arteriole_branch_prefix,
    )

    baseline_graph, _ = poiseuille_model.set_poiseuille_weights(
        baseline_graph,
        baseline_diameter_by_branch_order,
        prefer_edge_fwhm_diameter=True,
    )
    shifted_graph, _ = poiseuille_model.set_poiseuille_weights(
        shifted_graph,
        shifted_diameter_by_branch_order,
        prefer_edge_fwhm_diameter=True,
    )

    import numpy as np
    from ImageLynx import haemodynamics

    baseline_conductance, baseline_node_list = haemodynamics.build_conductance_matrix_from_graph(
        baseline_graph
    )
    shifted_conductance, shifted_node_list = haemodynamics.build_conductance_matrix_from_graph(
        shifted_graph
    )

    rows: list[dict[str, float]] = []
    for inlet_pressure_pa in inlet_pressures_pa:
        baseline = solve_pressure_and_boundary_flow(
            conductance=baseline_conductance,
            node_list=baseline_node_list,
            input_p_bc=float(inlet_pressure_pa),
            output_p_bc=float(output_p_bc),
            starting_nodes=list(starting_nodes),
            output_nodes=list(output_nodes),
        )
        shifted = solve_pressure_and_boundary_flow(
            conductance=shifted_conductance,
            node_list=shifted_node_list,
            input_p_bc=float(inlet_pressure_pa),
            output_p_bc=float(output_p_bc),
            starting_nodes=list(starting_nodes),
            output_nodes=list(output_nodes),
        )
        baseline_flow = float(baseline["total_inlet_flow"])
        shifted_flow = float(shifted["total_inlet_flow"])
        delta_flow = shifted_flow - baseline_flow
        delta_percent = np.nan
        if not np.isclose(baseline_flow, 0.0):
            delta_percent = (delta_flow / baseline_flow) * 100.0
        rows.append(
            {
                "inlet_pressure_pa": float(inlet_pressure_pa),
                "baseline_total_inlet_flow": baseline_flow,
                "arteriole_shifted_total_inlet_flow": shifted_flow,
                "flow_delta": delta_flow,
                "flow_delta_percent": float(delta_percent),
            }
        )

    delta_abs_tag = f"{abs(float(arteriole_diameter_delta_um)):.2f}".replace(".", "p")
    change_label = "dilation" if float(arteriole_diameter_delta_um) >= 0.0 else "constriction"
    flow_change_plot_path = (
        output_path / f"alice_flow_change_arteriole_passive_{change_label}_{delta_abs_tag}um.png"
    )

    x = [float(r["inlet_pressure_pa"]) for r in rows]
    baseline_y = [float(r["baseline_total_inlet_flow"]) for r in rows]
    shifted_y = [float(r["arteriole_shifted_total_inlet_flow"]) for r in rows]
    delta_y = [float(r["flow_delta"]) for r in rows]

    fig, (ax_flow, ax_delta) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_flow.plot(x, baseline_y, marker="o", linewidth=2.0, label="Baseline (no arteriole change)")
    ax_flow.plot(
        x,
        shifted_y,
        marker="o",
        linewidth=2.0,
        label=f"Arteriole passive delta {float(arteriole_diameter_delta_um):+.2f} um",
    )
    ax_flow.set_ylabel("Total inlet flow")
    ax_flow.set_title("Flow Before vs After Arteriole-Only Passive Diameter Change")
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend()

    ax_delta.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax_delta.plot(x, delta_y, marker="o", linewidth=2.0, color="tab:orange")
    ax_delta.set_xlabel("Inlet pressure (Pa)")
    ax_delta.set_ylabel("Flow change")
    ax_delta.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(flow_change_plot_path, dpi=200)
    plt.close(fig)

    return {
        "flow_change_plot_path": str(flow_change_plot_path),
        "arteriole_diameter_delta_um": float(arteriole_diameter_delta_um),
        "rows": rows,
    }
