"""Pericyte dilation and inlet-pressure sweep, on a synthetic network."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from haemolynx.haemodynamics.pericyte_sweep import (  # noqa: E402
    run_pericyte_dilation_pressure_sweep,
)
from haemolynx.visualization.dilation_curves import plot_dilation_curves  # noqa: E402


def _build_synthetic_network() -> tuple[
    nx.MultiGraph,
    list[int],
    list[int],
    list[int],
    list[int],
    dict[str, float],
]:
    """Create a small synthetic vessel network with explicit role assignments."""
    G = nx.MultiGraph()
    positions = {
        0: (0.0, 0.0, 0.0),    # input
        1: (0.0, 0.0, 10.0),   # arteriole->capillary transition
        2: (0.0, 0.0, 20.0),   # capillary
        3: (0.0, 0.0, 30.0),   # capillary->venule transition
        4: (0.0, 0.0, 40.0),   # output
    }
    for node_id, pos in positions.items():
        G.add_node(node_id, pos=np.asarray(pos, dtype=float))

    # Edge attributes include branch orders and FWHM diameters.
    # The sweep scales these diameters to emulate progressive pericyte dilation.
    G.add_edge(
        0,
        1,
        key=0,
        length=10.0,
        branch_order="Art1",
        fwhm_diameter_um=7.0,
        voxels=[(0, 0, 0), (0, 0, 10)],
    )
    G.add_edge(
        1,
        2,
        key=0,
        length=10.0,
        branch_order="B01",
        fwhm_diameter_um=5.0,
        voxels=[(0, 0, 10), (0, 0, 20)],
    )
    G.add_edge(
        2,
        3,
        key=0,
        length=10.0,
        branch_order="B01",
        fwhm_diameter_um=5.0,
        voxels=[(0, 0, 20), (0, 0, 30)],
    )
    G.add_edge(
        3,
        4,
        key=0,
        length=10.0,
        branch_order="Ven1",
        fwhm_diameter_um=8.0,
        voxels=[(0, 0, 30), (0, 0, 40)],
    )

    inlet_nodes = [0]
    outlet_nodes = [4]
    arteriole_boundary_nodes = [1]
    venule_boundary_nodes = [3]
    diameter_by_branch_order = {
        "Art1": 7.0,
        "B01": 5.0,
        "Ven1": 8.0,
    }
    return (
        G,
        inlet_nodes,
        outlet_nodes,
        arteriole_boundary_nodes,
        venule_boundary_nodes,
        diameter_by_branch_order,
    )


def _run_synthetic_sweep(
    output_dir: Path,
    *,
    min_dilation_percent: int,
    max_dilation_percent: int,
) -> dict:
    """Run a reduced sweep and return its results and output paths."""
    (
        G,
        inlet_nodes,
        outlet_nodes,
        arteriole_boundary_nodes,
        venule_boundary_nodes,
        diameter_by_branch_order,
    ) = _build_synthetic_network()

    # Assigned transition-node sets are part of the synthetic model specification.
    assert arteriole_boundary_nodes == [1]
    assert venule_boundary_nodes == [3]

    settings = {
        "diameter_by_branch_order": diameter_by_branch_order,
        "outlet_p_bc": 1000.0,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "pericyte_dilation_min_percent": min_dilation_percent,
        "pericyte_dilation_max_percent": max_dilation_percent,
        "pericyte_dilation_step_percent": 1,
        "inlet_pressure_min_pa": 4500,
        "inlet_pressure_max_pa": 6000,
        "inlet_pressure_step_pa": 500,
    }
    sweep = run_pericyte_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=output_dir,
    )
    sweep["plot_outputs"] = plot_dilation_curves(sweep["results"], output_dir)
    return sweep


def _assert_sweep_outputs_valid(
    sweep: dict,
    *,
    expected_dilation_points: int,
    expected_pressure_points: int = 4,
) -> None:
    """Validate sweep row counts and expected trend directions."""
    rows = sweep["results"]
    assert len(rows) == expected_dilation_points * expected_pressure_points
    csv_path = Path(sweep["csv_path"])
    resistance_plot = Path(sweep["plot_outputs"]["equivalent_resistance_plot_path"])
    flow_plot = Path(sweep["plot_outputs"]["total_inlet_flow_plot_path"])

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert resistance_plot.exists() and resistance_plot.stat().st_size > 0
    assert flow_plot.exists() and flow_plot.stat().st_size > 0

    # For each inlet pressure, resistance should decrease with dilation,
    # and inlet flow should increase with dilation.
    for pressure in (4500, 5000, 5500, 6000):
        subset = [r for r in rows if int(r["inlet_pressure_pa"]) == pressure]
        subset.sort(key=lambda r: int(r["dilation_percent"]))
        resistances = [float(r["equivalent_resistance"]) for r in subset]
        flows = [float(r["total_inlet_flow"]) for r in subset]
        assert resistances[0] > resistances[-1]
        assert flows[0] < flows[-1]


def test_dilation_sweep_on_synthetic_network(tmp_path: Path):
    """Run a reduced sweep and verify the CSV and the two curve plots."""
    sweep = _run_synthetic_sweep(
        tmp_path,
        min_dilation_percent=1,
        max_dilation_percent=3,
    )
    _assert_sweep_outputs_valid(
        sweep,
        expected_dilation_points=3,
    )


def test_arteriole_dilation_sweep_scales_only_arterioles(tmp_path: Path):
    """Whole-branch arteriole percent sweep leaves capillaries untouched."""
    from haemolynx.haemodynamics.pericyte_sweep import (
        run_arteriole_dilation_pressure_sweep,
    )

    G, inlet_nodes, outlet_nodes, _a, _v, diameters = _build_synthetic_network()
    baseline_caps = [
        float(data["fwhm_diameter_um"])
        for _u, _v, data in G.edges(data=True)
        if data["branch_order"] == "B01"
    ]
    settings = {
        "diameter_by_branch_order": diameters,
        "outlet_p_bc": 1000.0,
        "inlet_p_bc": 4500.0,
        "use_fwhm_edge_diameters": True,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "arteriole_dilation_min_percent": 0,
        "arteriole_dilation_max_percent": 20,
        "arteriole_dilation_step_percent": 20,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
    }
    sweep = run_arteriole_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path,
        sweep_dilation=True,
        sweep_pressure=False,
    )
    assert Path(sweep["csv_path"]).name == "arteriole_dilation_sweep.csv"
    assert len(sweep["results"]) == 2  # 0% and 20%
    # Baseline graph unchanged; capillaries still match.
    after_caps = [
        float(data["fwhm_diameter_um"])
        for _u, _v, data in G.edges(data=True)
        if data["branch_order"] == "B01"
    ]
    assert after_caps == baseline_caps
    # Wider arterioles lower network resistance.
    by_percent = {int(r["dilation_percent"]): r for r in sweep["results"]}
    assert by_percent[20]["equivalent_resistance"] < by_percent[0]["equivalent_resistance"]


def test_pressure_and_arteriole_sweep_varies_both_axes(tmp_path: Path):
    from haemolynx.haemodynamics.pericyte_sweep import (
        run_arteriole_dilation_pressure_sweep,
    )

    G, inlet_nodes, outlet_nodes, _a, _v, diameters = _build_synthetic_network()
    settings = {
        "diameter_by_branch_order": diameters,
        "outlet_p_bc": 1000.0,
        "use_fwhm_edge_diameters": True,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "arteriole_dilation_min_percent": 0,
        "arteriole_dilation_max_percent": 10,
        "arteriole_dilation_step_percent": 10,
        "inlet_pressure_min_pa": 4500,
        "inlet_pressure_max_pa": 5000,
        "inlet_pressure_step_pa": 500,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
    }
    sweep = run_arteriole_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path,
        sweep_dilation=True,
        sweep_pressure=True,
    )
    assert Path(sweep["csv_path"]).name == "arteriole_dilation_pressure_sweep.csv"
    assert len(sweep["results"]) == 4  # 2 dilations x 2 pressures
    percents = {int(r["dilation_percent"]) for r in sweep["results"]}
    pressures = {int(r["inlet_pressure_pa"]) for r in sweep["results"]}
    assert percents == {0, 10}
    assert pressures == {4500, 5000}


if __name__ == "__main__":
    demo_output_dir = REPO_ROOT / "examples" / "outputs" / "synthetic_dilation_sweep"
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    sweep = _run_synthetic_sweep(
        demo_output_dir,
        min_dilation_percent=1,
        max_dilation_percent=30,
    )
    _assert_sweep_outputs_valid(
        sweep,
        expected_dilation_points=30,
    )
    print("Synthetic dilation sweep completed.")
    print(f"CSV: {sweep['csv_path']}")
    print(f"Resistance plot: {sweep['plot_outputs']['equivalent_resistance_plot_path']}")
    print(f"Flow plot: {sweep['plot_outputs']['total_inlet_flow_plot_path']}")
