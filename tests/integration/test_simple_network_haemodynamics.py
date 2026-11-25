"""End-to-end checks for the minimal graph -> BCs -> flow -> VTK example."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics.poiseuille import (  # noqa: E402
    CAPILLARY_REGIME_MAX_DIAMETER_UM,
    LARGE_VESSEL_VISCOSITY_PA_S,
    UM_PER_M,
)

pytestmark = pytest.mark.integration


def _load_example_module():
    module_path = REPO_ROOT / "examples" / "simple_network_haemodynamics.py"
    spec = importlib.util.spec_from_file_location("simple_network_haemodynamics", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example_module()


@pytest.fixture(scope="module")
def settings(example, tmp_path_factory):
    """Settings exactly as a real run gets them: the config file on disk."""
    from haemolynx.parsers import load_config

    return load_config(
        example.CONFIG_PATH,
        example.SCHEMA,
        overrides={"output_dir": tmp_path_factory.mktemp("simple_network")},
    )


@pytest.fixture(scope="module")
def run_result(example, settings):
    return example.main(settings)


def test_example_network_is_well_formed(example, settings):
    G = example.build_example_network()

    import networkx as nx

    assert nx.is_connected(G)
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == 9
    for u, v, data in G.edges(data=True):
        assert data["length"] > 0
        assert data["branch_order"] in settings["diameter_by_branch_order"]
        assert len(data["voxels"]) == 2
    # Length must be the physical node separation, in um.
    assert G[0][1][0]["length"] == pytest.approx(100.0)


def test_every_vessel_gets_resistance_and_matching_conductance(run_result):
    G = run_result["graph"]
    for u, v, data in G.edges(data=True):
        assert data["resistance"] > 0
        assert data["conductance"] == pytest.approx(1.0 / data["resistance"], rel=1e-12)


def test_large_vessels_are_covered_by_the_law_rather_than_a_constant(
    settings, run_result
):
    """The 20 um arteriole is above the capillary limit, and no longer a stub.

    It used to take a flat 3.5 mPa.s, because the old default law had nothing
    to say between 7 um and the macroscale. The default is Pries now, which is
    fitted across 3.3-1978 um, so the arteriole gets a viscosity that depends
    on its diameter like every other vessel.
    """
    from haemolynx.haemodynamics.viscosity import pries_in_vitro_viscosity

    G = run_result["graph"]
    diameter = settings["diameter_by_branch_order"]["Art1"]
    assert diameter > CAPILLARY_REGIME_MAX_DIAMETER_UM

    length_m = G[0][1][0]["length"] / UM_PER_M
    diameter_m = diameter / UM_PER_M
    expected = (
        128.0 * pries_in_vitro_viscosity(diameter) * length_m
    ) / (math.pi * diameter_m**4)
    assert G[0][1][0]["resistance"] == pytest.approx(expected, rel=1e-12)
    assert pries_in_vitro_viscosity(diameter) != LARGE_VESSEL_VISCOSITY_PA_S


def test_pressures_stay_within_the_boundary_conditions(settings, run_result):
    pressure = run_result["flow_result"]["pressure"]
    assert pressure.min() == pytest.approx(settings["outlet_pressure_pa"])
    assert pressure.max() == pytest.approx(settings["inlet_pressure_pa"])


def test_flow_is_conserved_at_every_internal_node(example, run_result):
    """Kirchhoff's current law: only the two boundary nodes may source flow."""
    from haemolynx import haemodynamics

    G = run_result["graph"]
    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    pressure = run_result["flow_result"]["pressure"]
    inlet_flow = run_result["inlet_flow_m3_s"]

    for idx, node_id in enumerate(node_list):
        if node_id in (run_result["inlet_nodes"] + run_result["outlet_nodes"]):
            continue
        net_flow = float(np.sum(conductance[idx, :] * (pressure[idx] - pressure)))
        assert abs(net_flow) < 1e-9 * abs(inlet_flow)


def test_inlet_flow_matches_the_effective_resistance(settings, run_result):
    """Ohm's law across the whole network, computed two independent ways."""
    pressure_drop = settings["inlet_pressure_pa"] - settings["outlet_pressure_pa"]
    expected_flow = pressure_drop / run_result["effective_resistance"]
    assert run_result["inlet_flow_m3_s"] == pytest.approx(expected_flow, rel=1e-9)
    assert run_result["inlet_flow_m3_s"] > 0


def test_flow_is_physiologically_plausible(run_result):
    """A small capillary bed at ~37 mmHg drop should carry O(1-100) nL/min."""
    flow_nl_min = run_result["inlet_flow_m3_s"] * 6.0e13
    assert 1.0 <= flow_nl_min <= 100.0


def test_boundary_selection_picks_the_terminal_nodes(run_result):
    """The `select_boundary_nodes_by_method` call must resolve to the two ends."""
    G = run_result["graph"]
    terminals = {node for node, degree in G.degree() if degree == 1}

    assert run_result["inlet_nodes"] == [0]
    assert run_result["outlet_nodes"] == [7]
    # Pinning an interior junction would make it inject or remove flow.
    assert set(run_result["inlet_nodes"]) <= terminals
    assert set(run_result["outlet_nodes"]) <= terminals
    assert not set(run_result["inlet_nodes"]) & set(run_result["outlet_nodes"])


def test_running_the_example_no_longer_warns_about_a_placeholder(
    example, settings, tmp_path
):
    """This run has 20 um and 30 um vessels, and used to warn about every one.

    That warning was the old default law admitting it had nothing for the
    7-100 um band. The default covers it now, so a run of ordinary arterioles
    is quiet -- which is the whole of issue #90.
    """
    import warnings as _warnings

    from haemolynx.haemodynamics.poiseuille import PlaceholderViscosityWarning

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", PlaceholderViscosityWarning)
        example.main({**settings, "output_dir": tmp_path / "quiet"})


def test_vtk_files_are_written_with_flow_fields(run_result):
    vtk_export = run_result["vtk_export"]
    # One export, after the solve: the flows ride along as edge attributes.
    flow_path = Path(vtk_export["vessels_path"])
    assert flow_path.exists() and flow_path.stat().st_size > 0
    assert Path(vtk_export["nodes_path"]).exists()

    vessels = pv.read(str(flow_path))
    assert vessels.n_cells == run_result["graph"].number_of_edges()
    for field in ("resistance", "conductance", "pressure_drop", "flow_signed", "flow_abs"):
        assert field in vessels.cell_data
        assert np.all(np.isfinite(vessels.cell_data[field]))

    # Signed flows are drawn from the same solve as the reported inlet flow.
    assert np.max(vessels.cell_data["flow_abs"]) == pytest.approx(
        run_result["inlet_flow_m3_s"], rel=1e-9
    )
