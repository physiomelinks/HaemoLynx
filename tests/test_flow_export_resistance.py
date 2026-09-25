"""The flow export divides by the VTK resistance array, so that array must hold real values.

``solve_flow_from_conductance_matrix`` solves pressures from the conductance matrix (built from
the graph), then writes each cell's flow as pressure drop / the ``resistance`` cell array of the
vessels file. Since aecc53d the carotid body path writes no resistance before ``graph_to_vtk``,
so that array was all NaN, every exported flow was NaN, and the pipeline copied those NaNs back
into the graph's ``flow_abs``. Tier 3 then stopped on its unit check with a message blaming the
conversion factor, because NaN never equals NaN.
"""
import logging

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    _edge_flows_um3_per_s,
    map_vessels_to_grid,
)
from ImageLynx.haemodynamics.resistance import (
    build_conductance_matrix_from_graph,
    solve_flow_from_conductance_matrix,
)
from ImageLynx.visualization.vtk_io import graph_to_vtk


def _chain(resistances):
    """Three edges in a line along z, 20 um each, with centreline voxels."""
    G = nx.MultiGraph()
    for n in range(len(resistances) + 1):
        G.add_node(n, pos=np.array([20.0 * n, 0.0, 0.0]))
    for n, r in enumerate(resistances):
        attrs = dict(length=20.0, assigned_diameter_um=6.0,
                     voxels=[np.array([20.0 * n + t, 0.0, 0.0]) for t in (0.0, 10.0, 20.0)])
        if r is not None:
            attrs["resistance"] = r
        G.add_edge(n, n + 1, key=0, **attrs)
    return G


def _export(G, tmp_path):
    vtk_export = graph_to_vtk(G, tmp_path / "net")
    conductance, node_list = build_conductance_matrix_from_graph(G)
    last = G.number_of_nodes() - 1
    return conductance, node_list, vtk_export, last


def test_a_file_with_nan_resistance_is_refused_not_turned_into_nan_flow(tmp_path):
    # The graph has resistances (so the matrix and pressures are fine); the file was written
    # before they existed, which is the carotid body order that broke.
    G = _chain([None, None, None])
    vtk_export = graph_to_vtk(G, tmp_path / "net")
    for _, _, d in G.edges(data=True):
        d["resistance"] = 2.0
    conductance, node_list = build_conductance_matrix_from_graph(G)
    with pytest.raises(ValueError, match="None of the 3 cells .* has a finite, positive resistance"):
        solve_flow_from_conductance_matrix(conductance, node_list, 100.0, 0.0, [0], [3], vtk_export)


@pytest.mark.parametrize("bad", [np.nan, 0.0])
def test_a_cell_without_a_usable_resistance_gets_nan_flow_and_a_warning(tmp_path, caplog, bad):
    # Same policy as build_conductance_matrix_from_graph, which drops such an edge with a
    # warning. A zero resistance used to give an infinite flow.
    G = _chain([2.0, 2.0, 2.0])
    conductance, node_list, vtk_export, last = _export(G, tmp_path)
    import pyvista as pv
    mesh = pv.read(vtk_export["vessels_path"])
    mesh.cell_data["resistance"] = np.array([2.0, bad, 2.0])
    mesh.save(vtk_export["vessels_path"])
    with caplog.at_level(logging.WARNING, logger="ImageLynx.haemodynamics.resistance"):
        flow, _ = solve_flow_from_conductance_matrix(
            conductance, node_list, 100.0, 0.0, [0], [last], vtk_export)
    assert np.isnan(flow["flow_abs"][1])
    np.testing.assert_allclose(flow["flow_abs"][[0, 2]], 100.0 / 6.0, rtol=1e-9)
    assert any("1 of 3 cells" in r.getMessage() for r in caplog.records)


def test_flow_is_the_pressure_drop_over_the_file_resistance(tmp_path):
    # Series chain, R = 1, 2, 3: one flow Q = 100 / 6 through every edge.
    G = _chain([1.0, 2.0, 3.0])
    conductance, node_list, vtk_export, last = _export(G, tmp_path)
    flow, _ = solve_flow_from_conductance_matrix(
        conductance, node_list, 100.0, 0.0, [0], [last], vtk_export)
    np.testing.assert_allclose(flow["flow_abs"], 100.0 / 6.0, rtol=1e-9)


def _flowing_graph(flow_abs):
    G = _chain([1.0])
    d = G[0][1][0]
    d.update(flow_abs=flow_abs, flow_signed=flow_abs, hematocrit=0.45)
    return G


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_grid_mapping_names_a_nan_flow(bad):
    G = _flowing_graph(bad)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="NaN or infinite 'flow_abs'"):
        map_vessels_to_grid(G, grid)


def test_the_march_names_a_nan_flow_instead_of_blaming_the_unit_factor():
    # The pipeline's copy-back set flow_abs to NaN while the rheology solver's flow_signed
    # stayed finite, so the edge was in the DAG with a NaN flow.
    DAG = nx.MultiDiGraph()
    DAG.add_edge(0, 1, key=0, flow_abs=np.nan, flow_signed=1.0)
    cells = {0: [{"edge": (0, 1, 0), "flow": np.nan, "surface_area": 1.0}]}
    with pytest.raises(ValueError, match="NaN or infinite 'flow_abs'"):
        _edge_flows_um3_per_s(DAG, cells, 1.0)
