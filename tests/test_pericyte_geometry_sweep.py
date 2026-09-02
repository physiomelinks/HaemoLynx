"""Pericyte spacing and length sweeps at fixed dilation and pressure."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics.pericyte_geometry_sweep import (  # noqa: E402
    run_pericyte_geometry_sweep,
)
from haemolynx.haemodynamics.perturbations import (  # noqa: E402
    PERICYTE_LENGTH_SWEEP_SETTINGS,
    PERICYTE_SPACING_SWEEP_SETTINGS,
    SETTINGS_FOR_TYPE,
)


EDGE_LENGTH_UM = 400.0
DIAMETERS = {"Art1": 20.0, "B01": 6.0, "Ven1": 20.0}


def _network() -> nx.MultiGraph:
    """Long capillary bed so spacing / length changes fit multiple sites."""
    graph = nx.MultiGraph()
    orders = ("Art1", "B01", "B01", "Ven1")
    for node in range(len(orders) + 1):
        graph.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    for node, branch_order in enumerate(orders):
        graph.add_edge(
            node,
            node + 1,
            key=0,
            length=EDGE_LENGTH_UM,
            branch_order=branch_order,
            voxels=[
                [0.0, 0.0, node * EDGE_LENGTH_UM],
                [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM],
            ],
        )
    return graph


def _base_settings(**extra) -> dict:
    settings = {
        "diameter_by_branch_order": dict(DIAMETERS),
        "constriction_by_branch_order": {order: 0.5 for order in DIAMETERS},
        "outlet_p_bc": 1000.0,
        "inlet_p_bc": 4500.0,
        "use_fwhm_edge_diameters": False,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "use_probabilistic_pericyte_constriction": False,
        "pericyte_constriction_probability": 1.0,
        "pericyte_geometry_dilation_percent": 0,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "constriction_spacing_min_um": 50.0,
        "constriction_spacing_max_um": 150.0,
        "constriction_spacing_step_um": 50.0,
        "constriction_length_min_um": 20.0,
        "constriction_length_max_um": 60.0,
        "constriction_length_step_um": 20.0,
    }
    settings.update(extra)
    return settings


def test_settings_for_type_expose_geometry_axes_and_fixed_knobs():
    spacing = set(SETTINGS_FOR_TYPE["pericyte_spacing_sweep"])
    length = set(SETTINGS_FOR_TYPE["pericyte_length_sweep"])
    assert set(PERICYTE_SPACING_SWEEP_SETTINGS) <= spacing
    assert set(PERICYTE_LENGTH_SWEEP_SETTINGS) <= length
    assert "constriction_length_um" in spacing
    assert "constriction_spacing_um" not in spacing
    assert "constriction_spacing_um" in length
    assert "constriction_length_um" not in length
    assert "pericyte_geometry_dilation_percent" in spacing
    assert "pericyte_geometry_dilation_percent" in length
    assert "constriction_by_branch_order" in spacing
    assert "constriction_by_branch_order" in length
    assert "pericyte_constriction_factor" in spacing
    assert "pericyte_constriction_factor" in length


def test_spacing_sweep_changes_results_with_length_and_pressure_fixed(
    tmp_path: Path,
):
    G = _network()
    settings = _base_settings(
        constriction_length_um=40.0,
        constriction_spacing_min_um=50.0,
        constriction_spacing_max_um=150.0,
        constriction_spacing_step_um=50.0,
        pericyte_geometry_dilation_percent=0,
        inlet_p_bc=4500.0,
    )
    sweep = run_pericyte_geometry_sweep(
        G,
        settings,
        inlet_nodes=[0],
        outlet_nodes=[4],
        output_dir=tmp_path,
        sweep_axis="spacing",
    )
    assert Path(sweep["csv_path"]).name == "pericyte_spacing_sweep.csv"
    rows = sweep["results"]
    assert len(rows) == 3
    spacings = [float(r["constriction_spacing_um"]) for r in rows]
    assert spacings == pytest.approx([50.0, 100.0, 150.0])
    lengths = {float(r["constriction_length_um"]) for r in rows}
    pressures = {int(r["inlet_pressure_pa"]) for r in rows}
    dilations = {int(r["dilation_percent"]) for r in rows}
    assert lengths == {40.0}
    assert pressures == {4500}
    assert dilations == {0}
    resistances = [float(r["equivalent_resistance"]) for r in rows]
    assert resistances[0] != resistances[-1]
    # Denser sites (smaller spacing) raise network resistance when sites narrow.
    assert resistances[0] > resistances[-1]


def test_length_sweep_changes_results_with_spacing_and_pressure_fixed(
    tmp_path: Path,
):
    G = _network()
    settings = _base_settings(
        constriction_spacing_um=100.0,
        constriction_length_min_um=20.0,
        constriction_length_max_um=60.0,
        constriction_length_step_um=20.0,
        pericyte_geometry_dilation_percent=0,
        inlet_p_bc=4500.0,
    )
    sweep = run_pericyte_geometry_sweep(
        G,
        settings,
        inlet_nodes=[0],
        outlet_nodes=[4],
        output_dir=tmp_path,
        sweep_axis="length",
    )
    assert Path(sweep["csv_path"]).name == "pericyte_length_sweep.csv"
    rows = sweep["results"]
    assert len(rows) == 3
    lengths = [float(r["constriction_length_um"]) for r in rows]
    assert lengths == pytest.approx([20.0, 40.0, 60.0])
    spacings = {float(r["constriction_spacing_um"]) for r in rows}
    pressures = {int(r["inlet_pressure_pa"]) for r in rows}
    dilations = {int(r["dilation_percent"]) for r in rows}
    assert spacings == {100.0}
    assert pressures == {4500}
    assert dilations == {0}
    resistances = [float(r["equivalent_resistance"]) for r in rows]
    assert resistances[0] != resistances[-1]
    # Longer constriction plateaus raise resistance.
    assert resistances[-1] > resistances[0]


def test_geometry_sweep_keeps_fixed_dilation_percent_on_every_row(
    tmp_path: Path,
):
    G = _network()
    settings = _base_settings(
        pericyte_geometry_dilation_percent=10,
        constriction_spacing_min_um=80.0,
        constriction_spacing_max_um=120.0,
        constriction_spacing_step_um=40.0,
    )
    sweep = run_pericyte_geometry_sweep(
        G,
        settings,
        inlet_nodes=[0],
        outlet_nodes=[4],
        output_dir=tmp_path,
        sweep_axis="spacing",
    )
    for row in sweep["results"]:
        assert int(row["dilation_percent"]) == 10
        assert float(row["dilation_factor"]) == pytest.approx(1.1)
        assert int(row["inlet_pressure_pa"]) == 4500


def test_invalid_axis_raises():
    with pytest.raises(ValueError, match="sweep_axis"):
        run_pericyte_geometry_sweep(
            _network(),
            _base_settings(),
            inlet_nodes=[0],
            outlet_nodes=[4],
            output_dir=".",
            sweep_axis="pressure",  # type: ignore[arg-type]
        )
