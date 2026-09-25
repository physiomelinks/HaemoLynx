"""Loose end A3: the post-solve VTK rheology arrays must not carry invented values.

A cell that matched no graph edge used to keep pre-filled values of haematocrit 0.45,
viscosity 1.2 cP and wall shear stress 0, and a matched edge missing an attribute got the
same through .get(..., default). Those would look real in ParaView. Only resistance was NaN.
"""
import logging

import networkx as nx
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
C = pytest.importorskip("carotid_image_to_model")

SOLVED = {"hematocrit": 0.38, "viscosity": 2.7, "wall_shear_stress_pa": 1.9, "resistance": 5.5}


def _mesh(edges):
    """One two-point line cell per (u, v, key), carrying the id arrays graph_to_vtk writes."""
    points, lines = [], []
    for i, _ in enumerate(edges):
        points += [[0.0, float(i), 0.0], [1.0, float(i), 0.0]]
        lines += [2, 2 * i, 2 * i + 1]
    mesh = pv.PolyData(np.asarray(points), lines=np.asarray(lines))
    mesh.cell_data["edge_u"] = np.array([e[0] for e in edges], dtype=np.int64)
    mesh.cell_data["edge_v"] = np.array([e[1] for e in edges], dtype=np.int64)
    mesh.cell_data["edge_key"] = np.array([e[2] for e in edges], dtype=np.int64)
    return mesh


def _graph(**attrs):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, **attrs)
    return G


def test_a_matched_edge_carries_its_solved_values():
    arrays = C._rheology_cell_arrays(_graph(**SOLVED), _mesh([(0, 1, 0)]))
    for name, value in SOLVED.items():
        assert arrays[name][0] == value


def test_an_unmatched_cell_is_nan_not_a_plausible_default():
    arrays = C._rheology_cell_arrays(_graph(**SOLVED), _mesh([(0, 1, 0), (7, 8, 0)]))
    for name in SOLVED:
        assert np.isnan(arrays[name][1]), f"unmatched cell got {name} = {arrays[name][1]}"


def test_a_matched_edge_without_wall_shear_stress_is_nan():
    """WSS is only set inside the rheology update loop, so an early stop leaves it unset."""
    partial = {k: v for k, v in SOLVED.items() if k != "wall_shear_stress_pa"}
    arrays = C._rheology_cell_arrays(_graph(**partial), _mesh([(0, 1, 0)]))
    assert np.isnan(arrays["wall_shear_stress_pa"][0])
    assert arrays["hematocrit"][0] == SOLVED["hematocrit"]


def test_unmatched_and_missing_values_are_reported(caplog):
    partial = {k: v for k, v in SOLVED.items() if k != "viscosity"}
    with caplog.at_level(logging.WARNING):
        C._rheology_cell_arrays(_graph(**partial), _mesh([(0, 1, 0), (7, 8, 0), (8, 9, 0)]))
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "2 of 3 cells match no graph edge" in message
    assert "'viscosity': 1" in message


def test_a_fully_matched_mesh_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        C._rheology_cell_arrays(_graph(**SOLVED), _mesh([(0, 1, 0)]))
    assert not caplog.records
