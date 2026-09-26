"""Open item 30: ``map_vessels_to_grid(..., vessel_mapping="cross_section")``.

The centreline mapping puts a vessel only in the cells its centreline crosses. The cross-section
mapping sweeps each centreline point over a disc of the vessel's radius, perpendicular to the
local tangent, and shares the edge's length and wall area by the part of the disc each cell
holds. These tests pin what that must and must not change.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    map_vessels_to_grid,
)


def _graph(points, diameter, flow=1.0, hematocrit=0.45):
    points = np.asarray(points, dtype=float)
    G = nx.MultiGraph()
    G.add_node(0, pos=points[0])
    G.add_node(1, pos=points[-1])
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    G.add_edge(0, 1, voxels=points, length=length, assigned_diameter_um=diameter,
               hematocrit=hematocrit, flow_abs=flow)
    return G


def _line(start, stop, n):
    return np.linspace(np.asarray(start, float), np.asarray(stop, float), n)


def _grid(G, h, lo, hi):
    return PerfusionGrid(G, (h, h, h), bounds_zyx=(lo, hi))


def _by_edge(cells):
    """Per edge: {cell: entry}."""
    out = {}
    for idx, vessels in cells.items():
        for item in vessels:
            out.setdefault(item["edge"], {})[idx] = item
    return out


def test_unknown_mapping_raises():
    G = _graph(_line((0, 5, 5), (10, 5, 5), 11), diameter=2.0)
    grid = _grid(G, 1.0, (0, 0, 0), (10, 10, 10))
    with pytest.raises(ValueError, match="vessel_mapping"):
        map_vessels_to_grid(G, grid, vessel_mapping="ball")


def test_shares_sum_to_one_and_wall_area_is_conserved():
    """Spreading moves where an edge deposits, never how much."""
    pts = _line((2.0, 20.0, 20.0), (38.0, 26.0, 31.0), 40)
    G = _graph(pts, diameter=8.0)
    grid = _grid(G, 1.0, (0, 0, 0), (40, 45, 45))
    for mapping in ("centreline", "cross_section"):
        entries = _by_edge(map_vessels_to_grid(G, grid, vessel_mapping=mapping))[(0, 1, 0)]
        assert sum(e["length_fraction"] for e in entries.values()) == pytest.approx(1.0)
        length = sum(e["length"] for e in entries.values())
        area = sum(e["surface_area"] for e in entries.values())
        assert area == pytest.approx(2.0 * np.pi * 4.0 * length)
        assert len({e["flow"] for e in entries.values()}) == 1   # flow is shared, not split


def test_a_vessel_inside_one_cell_maps_as_its_centreline():
    """A 1 um vessel on a 10 um grid, centreline on cell centres: the disc never leaves the cell."""
    pts = _line((5.0, 15.0, 25.0), (45.0, 15.0, 25.0), 41)
    G = _graph(pts, diameter=1.0)
    grid = _grid(G, 10.0, (0, 0, 0), (50, 40, 40))
    line = _by_edge(map_vessels_to_grid(G, grid))[(0, 1, 0)]
    disc = _by_edge(map_vessels_to_grid(G, grid, vessel_mapping="cross_section"))[(0, 1, 0)]
    assert set(line) == set(disc)
    for idx in line:
        assert disc[idx]["length_fraction"] == pytest.approx(line[idx]["length_fraction"])
        assert disc[idx]["surface_area"] == pytest.approx(line[idx]["surface_area"])


def _covered_cells(G, grid):
    entries = _by_edge(map_vessels_to_grid(G, grid, vessel_mapping="cross_section"))[(0, 1, 0)]
    centres = np.array([grid.get_xyz_from_index(i) for i in entries])
    share = np.array([e["length_fraction"] for e in entries.values()])
    return centres, share


def test_an_axial_vessel_covers_its_lumen_and_nothing_else():
    """9 um vessel along z on a 1 um grid: each slice holds pi r^2 of lumen, none beyond r."""
    radius, h = 4.5, 1.0
    pts = _line((0.5, 20.0, 20.0), (19.5, 20.0, 20.0), 20)
    G = _graph(pts, diameter=2 * radius)
    grid = _grid(G, h, (0, 0, 0), (20, 40, 40))
    centres, share = _covered_cells(G, grid)

    lateral = np.hypot(centres[:, 1] - 20.0, centres[:, 2] - 20.0)
    assert lateral.max() <= radius + h * np.sqrt(0.5)
    # Rim cells hold part of the disc, so count cells by the share they hold: a full cell's
    # share is the largest.
    per_slice = share.sum() / share.max() / 20
    assert per_slice == pytest.approx(np.pi * radius ** 2 / h ** 2, rel=0.02)
    # Cells near the axis hold a whole lattice cell's worth of disc; the rim holds less.
    inner = lateral < radius - h
    assert np.allclose(share[inner], share[inner].max())
    assert share[~inner].max() <= share[inner].max() + 1e-12


def test_the_disc_is_perpendicular_to_a_diagonal_vessel():
    """Along (1, 1, 1): nothing beyond r of the axis, and the share spread as a disc's area is."""
    radius, h = 3.0, 1.0
    axis = np.ones(3) / np.sqrt(3.0)
    start = np.array([8.0, 8.0, 8.0])
    pts = start + np.outer(np.linspace(0.0, 30.0, 61), axis)
    G = _graph(pts, diameter=2 * radius)
    grid = _grid(G, h, (0, 0, 0), (40, 40, 40))
    centres, share = _covered_cells(G, grid)

    rel = centres - start
    along = rel @ axis
    lateral = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    assert lateral.max() <= radius + h * np.sqrt(3.0) / 2
    # A uniform disc has mean radius 2r/3. A disc tilted off the perpendicular reaches further
    # from the axis; a line (the centreline mapping) sits near zero.
    mean_lateral = np.average(lateral, weights=share)
    assert mean_lateral == pytest.approx(2.0 * radius / 3.0, rel=0.1)


def test_coincident_centreline_points_raise():
    """No tangent, so no cross-section plane: raised rather than guessed."""
    pts = np.array([[1.0, 5.0, 5.0], [2.0, 5.0, 5.0], [2.0, 5.0, 5.0], [2.0, 5.0, 5.0]])
    G = _graph(pts, diameter=2.0)
    grid = _grid(G, 1.0, (0, 0, 0), (10, 10, 10))
    with pytest.raises(ValueError, match="tangent"):
        map_vessels_to_grid(G, grid, vessel_mapping="cross_section")
    map_vessels_to_grid(G, grid)  # the centreline mapping needs no tangent


def test_lumen_outside_the_grid_is_dropped_and_the_rest_renormalised():
    """Same rule as the centreline mapping: drop, never clip, then shares sum to one."""
    pts = _line((5.0, 1.0, 20.0), (35.0, 1.0, 20.0), 31)   # hugs the y = 0 face
    G = _graph(pts, diameter=6.0)
    grid = _grid(G, 1.0, (0, 0, 0), (40, 40, 40))
    centres, share = _covered_cells(G, grid)
    assert centres[:, 1].min() >= grid.min_xyz[1]
    assert share.sum() == pytest.approx(1.0)


def test_the_centreline_default_is_unchanged():
    """No argument means the centreline mapping, entry for entry."""
    pts = _line((2.0, 20.0, 20.0), (38.0, 26.0, 31.0), 40)
    G = _graph(pts, diameter=8.0)
    grid = _grid(G, 2.0, (0, 0, 0), (40, 45, 45))
    assert map_vessels_to_grid(G, grid) == map_vessels_to_grid(
        G, grid, vessel_mapping="centreline")
