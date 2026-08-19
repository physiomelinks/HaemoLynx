"""An edge's flow must be shared among the cells it crosses, not repeated in each.

map_vessels_to_grid recorded the whole edge flow against every cell the edge passed through,
so an edge crossing five cells injected five times its own oxygen. Refining the grid makes an
edge cross more cells, so the total source grew with resolution: measured on WKY-C, the summed
s_incoming went 8.87e6, 1.20e7, 1.54e7, 1.80e7 at 10, 6, 4 and 3 um, in exact proportion to the
mean cells crossed per edge. That is the reason section 2.3's PO2 rose with refinement.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import PerfusionGrid, map_vessels_to_grid


def _one_long_edge(n_points=40, span=280.0):
    """A single straight edge crossing many cells, so the effect is unmissable."""
    G = nx.MultiGraph()
    pts = [np.array([t, 150.0, 150.0]) for t in np.linspace(10.0, span, n_points)]
    G.add_node(1, pos=pts[0])
    G.add_node(2, pos=pts[-1])
    G.add_edge(1, 2, key=0, length=float(span - 10.0), flow_abs=4.0,
               assigned_diameter_um=8.0, voxels=pts)
    return G


def test_length_fractions_sum_to_one_across_the_cells_an_edge_crosses():
    G = _one_long_edge()
    for res in (30.0, 15.0, 8.0):
        grid = PerfusionGrid(G, (res, res, res))
        mapping = map_vessels_to_grid(G, grid)
        total = sum(v["length_fraction"] for cell in mapping.values() for v in cell)
        assert total == pytest.approx(1.0, rel=1e-9), f"res={res}: {total}"


def test_the_delivered_flow_does_not_grow_when_the_grid_is_refined():
    """The quantity that was scaling with resolution."""
    G = _one_long_edge()
    totals = {}
    for res in (30.0, 15.0, 8.0, 5.0):
        grid = PerfusionGrid(G, (res, res, res))
        mapping = map_vessels_to_grid(G, grid)
        totals[res] = sum(v["flow"] * v["length_fraction"]
                          for cell in mapping.values() for v in cell)
    values = list(totals.values())
    assert max(values) / min(values) < 1.0001, totals


def test_an_edge_crossing_more_cells_gets_a_smaller_share_in_each():
    G = _one_long_edge()
    coarse = map_vessels_to_grid(G, PerfusionGrid(G, (30.0, 30.0, 30.0)))
    fine = map_vessels_to_grid(G, PerfusionGrid(G, (8.0, 8.0, 8.0)))
    assert len(fine) > len(coarse)
    max_coarse = max(v["length_fraction"] for c in coarse.values() for v in c)
    max_fine = max(v["length_fraction"] for c in fine.values() for v in c)
    assert max_fine < max_coarse


def test_the_full_edge_flow_is_still_recorded_for_callers_that_want_it():
    """`flow` stays the edge's own flow; the share is a separate field."""
    G = _one_long_edge()
    mapping = map_vessels_to_grid(G, PerfusionGrid(G, (30.0, 30.0, 30.0)))
    flows = {v["flow"] for cell in mapping.values() for v in cell}
    assert len(flows) == 1


def test_an_edge_wholly_inside_one_cell_keeps_its_whole_share():
    G = nx.MultiGraph()
    pts = [np.array([100.0, 100.0, 100.0]), np.array([102.0, 100.0, 100.0])]
    G.add_node(1, pos=pts[0]); G.add_node(2, pos=pts[1])
    G.add_edge(1, 2, key=0, length=2.0, flow_abs=3.0, assigned_diameter_um=8.0, voxels=pts)
    mapping = map_vessels_to_grid(G, PerfusionGrid(G, (50.0, 50.0, 50.0)))
    shares = [v["length_fraction"] for cell in mapping.values() for v in cell]
    assert sum(shares) == pytest.approx(1.0)


def test_the_oxygen_source_is_grid_independent_end_to_end():
    """What section 2.3 actually consumes."""
    from ImageLynx.haemodynamics.perfusion import build_adr_matrix

    class Cfg:
        sigma_diff, M_max, k_reduce, C_arterial = 1.5e-9, 0.05, 0.1, 0.13

    G = _one_long_edge()
    sums = {}
    for res in (30.0, 15.0, 8.0):
        grid = PerfusionGrid(G, (res, res, res))
        _A, q, s = build_adr_matrix(grid, map_vessels_to_grid(G, grid), Cfg())
        sums[res] = (float(q.sum()), float(s.sum()))
    q_vals = [v[0] for v in sums.values()]
    s_vals = [v[1] for v in sums.values()]
    assert max(q_vals) / min(q_vals) < 1.0001, sums
    assert max(s_vals) / min(s_vals) < 1.0001, sums
