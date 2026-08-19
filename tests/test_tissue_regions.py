"""Joining a segmented tissue mask to the perfusion grid.

H2 §2.3 asks for "a higher metabolic rate for TH-positive voxels, and a lower rate for the
surrounding stroma". The ADR solver takes a single scalar ``M_max``, so the mask has to become
a per-cell array first. The grid is coarse relative to the segmentation, roughly 154 mask
voxels to a 10 µm cell, so a cell is rarely all tissue or all stroma and sampling the mask at
the cell centre would discard almost all of it.
"""
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
