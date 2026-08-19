"""Joining a segmented tissue mask to the perfusion grid.

H2 §2.3 asks for "a higher metabolic rate for TH-positive voxels, and a lower rate for the
surrounding stroma". The ADR solver takes a single scalar ``M_max``, so the mask has to become
a per-cell array first. The grid is coarse relative to the segmentation, roughly 154 mask
voxels to a 10 µm cell, so a cell is rarely all tissue or all stroma and sampling the mask at
the cell centre would discard almost all of it.
"""
import warnings

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import PerfusionGrid
from ImageLynx.haemodynamics.tissue_regions import (
    blend_per_cell_rate,
    mask_fraction_per_cell,
)

VOX = (1.8639, 1.866, 1.866)


def _grid(extent_um=40.0, res_um=10.0):
    """A grid spanning [0, extent] on each axis, from a two-node graph."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([extent_um, extent_um, extent_um]))
    G.add_edge(0, 1, length=extent_um)
    return PerfusionGrid(G, (res_um, res_um, res_um))


def test_a_fully_masked_volume_gives_a_fraction_of_one_everywhere_inside():
    grid = _grid()
    shape = tuple(int(np.ceil(40.0 / v)) + 4 for v in VOX)
    mask = np.ones(shape, bool)
    frac = mask_fraction_per_cell(mask, grid, VOX)

    assert frac.shape == (grid.n_cells,)
    interior = frac[frac > 0]
    assert interior.size > 0
    assert np.allclose(interior.max(), 1.0, atol=0.05)


def test_an_empty_mask_gives_zero_everywhere():
    grid = _grid()
    mask = np.zeros((30, 30, 30), bool)
    assert not mask_fraction_per_cell(mask, grid, VOX).any()


def test_the_fraction_is_the_occupied_volume_not_a_centre_sample():
    """Half a cell masked must read about 0.5, not 0 or 1."""
    grid = _grid(extent_um=20.0, res_um=20.0)      # a single cell, near enough
    n = 24
    mask = np.zeros((n, n, n), bool)
    mask[: n // 2] = True                           # lower half in z
    frac = mask_fraction_per_cell(mask, grid, VOX)
    occupied = frac[frac > 0]
    assert occupied.size >= 1
    assert 0.3 < occupied.max() < 0.7, f"got {occupied.max()}"


def test_fractions_never_exceed_one():
    grid = _grid(res_um=3.0)                        # finer than the voxel size
    mask = np.ones((30, 30, 30), bool)
    frac = mask_fraction_per_cell(mask, grid, VOX)
    assert frac.max() <= 1.0 + 1e-9


def test_mask_voxels_outside_the_grid_are_dropped_not_wrapped():
    """A wrapped index would silently deposit distal tissue into cell 0."""
    grid = _grid(extent_um=20.0)
    n = 200                                          # far larger than the grid
    mask = np.zeros((n, n, n), bool)
    mask[-5:, -5:, -5:] = True                       # entirely beyond the grid
    assert not mask_fraction_per_cell(mask, grid, VOX).any()


def test_the_join_respects_the_grid_origin():
    """Grid bounds come from graph positions and need not start at zero."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([100.0, 100.0, 100.0]))
    G.add_node(1, pos=np.array([140.0, 140.0, 140.0]))
    G.add_edge(0, 1, length=40.0)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))

    n = 100
    mask = np.zeros((n, n, n), bool)
    mask[50:60, 50:60, 50:60] = True                 # about 93 to 112 um
    frac = mask_fraction_per_cell(mask, grid, VOX)
    assert frac.any(), "mask overlapping the grid produced nothing"
    assert frac[frac > 0].size < grid.n_cells, "a local mask filled the whole grid"


def test_rate_blending_is_linear_between_the_two_rates():
    frac = np.array([0.0, 0.25, 0.5, 1.0])
    rate = blend_per_cell_rate(frac, tissue_rate=0.05, stroma_rate=0.01)
    assert rate[0] == pytest.approx(0.01)
    assert rate[-1] == pytest.approx(0.05)
    assert rate[2] == pytest.approx(0.03)
    assert np.all(np.diff(rate) > 0)


def test_rate_blending_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError, match="fraction"):
        blend_per_cell_rate(np.array([0.0, 1.5]), tissue_rate=0.05, stroma_rate=0.01)


def test_a_uniform_rate_reproduces_the_scalar_case():
    """The array path must not change the answer when both rates are equal."""
    frac = np.linspace(0, 1, 17)
    rate = blend_per_cell_rate(frac, tissue_rate=0.05, stroma_rate=0.05)
    assert np.allclose(rate, 0.05)


def test_the_solver_accepts_a_per_cell_rate_array():
    """M_max is used elementwise against PO2, so an array must broadcast unchanged."""
    n = 12
    m_scalar = 0.05
    m_array = np.full(n, 0.05)
    po2 = np.linspace(1.0, 90.0, n)
    k = 0.5
    assert np.allclose(m_scalar * (1.0 - np.exp(-k * po2)),
                       m_array * (1.0 - np.exp(-k * po2)))


def test_the_linear_index_matches_the_grids_own_convention():
    """PerfusionGrid indexes z-fastest. A cubic grid cannot tell that from x-fastest.

    The grid's own get_cell_index is the oracle: if this module disagrees with it, tissue is
    deposited in the wrong cell and the metabolic map is transposed without any error.
    """
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([30.0, 60.0, 90.0]))       # deliberately non-cubic
    G.add_edge(0, 1, length=1.0)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    assert len(set(grid.dims)) > 1, "grid must be non-cubic for this test to bite"

    # One small blob, well inside a single cell, away from any cell boundary.
    target_um = np.array([5.0, 35.0, 65.0])
    idx = np.round(target_um / np.asarray(VOX)).astype(int)
    mask = np.zeros(tuple(int(np.ceil(e / v)) + 6 for e, v in zip((30, 60, 90), VOX)), bool)
    mask[idx[0], idx[1], idx[2]] = True

    frac = mask_fraction_per_cell(mask, grid, VOX)
    assert frac.sum() > 0
    assert int(np.argmax(frac)) == int(grid.get_cell_index(target_um))


def test_origin_um_places_the_mask_in_physical_space():
    """The mask's first voxel corner is not always the graph's origin."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([100.0, 100.0, 100.0]))
    G.add_node(1, pos=np.array([140.0, 140.0, 140.0]))
    G.add_edge(0, 1, length=40.0)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))

    mask = np.zeros((12, 12, 12), bool)
    mask[2:8, 2:8, 2:8] = True                            # about 4 to 15 um from its own origin

    assert not mask_fraction_per_cell(mask, grid, VOX).any(), (
        "a mask at the coordinate origin must not land in a grid spanning 100 to 140 um")
    shifted = mask_fraction_per_cell(mask, grid, VOX, origin_um=(100.0, 100.0, 100.0))
    assert shifted.any(), "origin_um did not move the mask into the grid"


# --- The graph-side join: which edges lie inside the tissue -------------------------------

from ImageLynx.haemodynamics.tissue_regions import edge_tissue_fraction   # noqa: E402


def _edge_graph(polylines):
    G = nx.MultiGraph()
    for i, (name, pts) in enumerate(polylines.items()):
        a, b = f"{name}_a", f"{name}_b"
        pts = np.asarray(pts, dtype=float)
        G.add_node(a, pos=pts[0])
        G.add_node(b, pos=pts[-1])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        G.add_edge(a, b, key=0, voxels=[tuple(p) for p in pts], length=float(seg))
    return G


def test_an_edge_inside_the_mask_is_wholly_inside():
    mask = np.ones((40, 40, 40), bool)
    G = _edge_graph({"e": [(10.0, 10.0, 10.0), (10.0, 30.0, 10.0)]})
    frac = edge_tissue_fraction(G, mask, VOX)
    assert list(frac.values())[0] == pytest.approx(1.0)


def test_an_edge_outside_the_mask_is_wholly_outside():
    mask = np.zeros((40, 40, 40), bool)
    G = _edge_graph({"e": [(10.0, 10.0, 10.0), (10.0, 30.0, 10.0)]})
    assert list(edge_tissue_fraction(G, mask, VOX).values())[0] == pytest.approx(0.0)


def test_an_edge_crossing_the_boundary_is_partly_inside():
    mask = np.zeros((60, 60, 60), bool)
    mask[:, :30, :] = True                       # tissue below y index 30, i.e. y < 56 um
    G = _edge_graph({"e": [(20.0, 10.0, 20.0), (20.0, 100.0, 20.0)]})
    frac = list(edge_tissue_fraction(G, mask, VOX).values())[0]
    assert 0.3 < frac < 0.7, f"got {frac}"


def test_the_whole_centreline_is_sampled_not_just_the_endpoints():
    """An edge whose ends are in stroma but whose middle runs through tissue.

    Sampling endpoints alone would call this edge entirely extra-glomus, which is exactly the
    penetrating capillary section 2.1 is about.
    """
    mask = np.zeros((60, 60, 60), bool)
    mask[:, 25:35, :] = True                     # a slab in the middle
    G = _edge_graph({"e": [(20.0, 10.0, 20.0), (20.0, 56.0, 20.0), (20.0, 100.0, 20.0)]})
    frac = list(edge_tissue_fraction(G, mask, VOX).values())[0]
    assert frac > 0.05, "the mid-edge tissue crossing was missed"
    assert frac < 0.5


def test_the_fraction_is_weighted_by_length_not_by_point_count():
    """Many closely spaced vertices outside must not outvote one long run inside.

    Counting vertices rather than length would call this edge 2% inside; by length it is 90%.
    The stored polylines are not uniformly spaced, so the two genuinely differ.
    """
    mask = np.zeros((120, 120, 120), bool)
    mask[:, 40:, :] = True                       # tissue above y index 40, i.e. y > 74.6 um
    dense_outside = [(20.0, y, 20.0) for y in np.linspace(60.0, 74.0, 40)]   # 14 um, 40 pts
    long_inside = [(20.0, 200.0, 20.0)]                                      # 126 um, 1 pt
    G = _edge_graph({"e": dense_outside + long_inside})

    frac = list(edge_tissue_fraction(G, mask, VOX).values())[0]
    assert frac > 0.8, f"length weighting failed: {frac}"
    assert frac < 0.95


def test_centreline_points_outside_the_array_count_as_outside():
    mask = np.ones((20, 20, 20), bool)
    G = _edge_graph({"inside": [(10.0, 10.0, 10.0), (10.0, 20.0, 10.0)],
                     "beyond": [(10.0, 500.0, 10.0), (10.0, 600.0, 10.0)]})
    frac = edge_tissue_fraction(G, mask, VOX)
    vals = {k[0]: v for k, v in frac.items()}
    assert vals["inside_a"] == pytest.approx(1.0)
    assert vals["beyond_a"] == pytest.approx(0.0)


def test_an_edge_without_a_polyline_falls_back_to_its_endpoints():
    mask = np.ones((40, 40, 40), bool)
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array([10.0, 10.0, 10.0]))
    G.add_node("b", pos=np.array([10.0, 30.0, 10.0]))
    G.add_edge("a", "b", key=0, length=20.0)          # no 'voxels'
    assert list(edge_tissue_fraction(G, mask, VOX).values())[0] == pytest.approx(1.0)


def test_every_edge_is_reported_exactly_once_keyed_by_u_v_key():
    mask = np.ones((40, 40, 40), bool)
    G = _edge_graph({"p": [(10.0, 10.0, 10.0), (10.0, 30.0, 10.0)],
                     "q": [(12.0, 10.0, 10.0), (12.0, 30.0, 10.0)]})
    frac = edge_tissue_fraction(G, mask, VOX)
    assert len(frac) == G.number_of_edges()
    assert all(len(k) == 3 for k in frac)


def test_tissue_falling_outside_the_grid_is_reported_not_just_dropped():
    """Dropping is correct; dropping quietly is not.

    The grid comes from the graph's node bounding box, so a specimen whose vessels stop short
    of the region edge gets a grid smaller than the mask. Every returned fraction is then
    valid and describes less tissue than was handed in, which nothing downstream can detect.
    Two of the six carotid body specimens lose 4.35% and 7.54% of their glomus volume this way.
    """
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array([0.0, 0.0, 0.0]))
    G.add_node("b", pos=np.array([20.0, 20.0, 20.0]))
    grid = PerfusionGrid(G, (4.0, 4.0, 4.0))          # spans about -2 to 22 um

    mask = np.zeros((30, 8, 8), dtype=bool)
    mask[:, 2:5, 2:5] = True                          # runs well past the grid in z

    with pytest.warns(RuntimeWarning, match="fall outside the grid"):
        fraction = mask_fraction_per_cell(mask, grid, (1.0, 1.0, 1.0))

    assert fraction.shape == (grid.n_cells,)
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))


def test_a_mask_that_fits_the_grid_warns_about_nothing():
    G = nx.MultiGraph()
    G.add_node("a", pos=np.array([0.0, 0.0, 0.0]))
    G.add_node("b", pos=np.array([20.0, 20.0, 20.0]))
    grid = PerfusionGrid(G, (4.0, 4.0, 4.0))

    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:10, 4:10, 4:10] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mask_fraction_per_cell(mask, grid, (1.0, 1.0, 1.0))
