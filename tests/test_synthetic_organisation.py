"""Where a network's vessels lie and which way they run
(``synthetic/organisation.py``), on hand-built networks whose answers are
known, and the comparison rows it feeds."""
from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from haemolynx.synthetic.compare import organisation_rows
from haemolynx.synthetic.organisation import (
    Organisation,
    cell_index,
    correlation,
    grid_shape,
    measure_organisation,
    profile,
)
from haemolynx.synthetic.template import NetworkTemplate
from synthetic_fixtures import fixture_template, tiny_hierarchical_graph, TINY_EXTENT_UM


def _line(a, b, step=0.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
    return [list(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n)]


def _graph(edges, extent):
    """*edges*: (start, end, label) in (z, y, x) um."""
    G = nx.MultiGraph()
    index: dict = {}
    for a, b, label in edges:
        for p in (a, b):
            key = tuple(float(v) for v in p)
            if key not in index:
                index[key] = len(index)
                G.add_node(index[key], pos=np.asarray(key))
        G.add_edge(index[tuple(map(float, a))], index[tuple(map(float, b))], voxels=_line(a, b), branch_order=label)
    G.graph["image_extent_um"] = tuple(extent)
    return G


EXTENT = (100.0, 200.0, 200.0)


def _slab():
    """Vertical capillaries 10 um apart in x and y, running z 0 to 40 um:
    every one along z, all in the lower of the box's two layers of cells."""
    edges = []
    for y in np.arange(5.0, 200.0, 10.0):
        for x in np.arange(5.0, 200.0, 10.0):
            edges.append(((0.0, y, x), (40.0, y, x), "B01"))
    return _graph(edges, EXTENT)


def _lattice(step=20.0):
    """A cubic lattice of capillaries filling the box: a third of the
    length along each axis."""
    edges = []
    zs, ys, xs = (np.arange(10.0, e, step) for e in EXTENT)
    for z in zs:
        for y in ys:
            for x in xs:
                if z + step < EXTENT[0]:
                    edges.append(((z, y, x), (z + step, y, x), "B01"))
                if y + step < EXTENT[1]:
                    edges.append(((z, y, x), (z, y + step, x), "B01"))
                if x + step < EXTENT[2]:
                    edges.append(((z, y, x), (z, y, x + step), "B01"))
    return _graph(edges, EXTENT)


def test_the_grid_is_about_fifty_microns_a_cell_and_shared_by_scale():
    assert grid_shape((100.0, 300.0, 500.0)) == (2, 6, 10)
    assert grid_shape((10.0, 10.0, 10.0)) == (1, 1, 1)
    assert grid_shape((5000.0, 5000.0, 5000.0)) == (12, 12, 12)
    points = np.array([[0.0, 0.0, 0.0], [99.99, 199.9, 199.9], [100.0, 200.0, 200.0], [60.0, 10.0, 150.0]])
    cells = cell_index(points, EXTENT, (2, 4, 4))
    assert cells.tolist() == [0, 31, 31, 16 + 0 * 4 + 3]


def test_a_slab_of_vertical_capillaries_reads_as_vertical_and_in_its_layer():
    org = measure_organisation(_slab(), EXTENT)
    assert org.shape == (2, 4, 4)
    assert org.axis_share == pytest.approx((1.0, 0.0, 0.0))
    assert org.global_orientation[0] == pytest.approx(1.0)  # zz
    depth = profile(org.field("capillary_length_um"), 0)
    assert depth[0] == pytest.approx(400 * 40.0, rel=1e-6) and depth[1] == 0.0
    nodes = org.field("capillary_nodes")
    assert nodes.sum() == 800 and nodes[1].sum() == 0  # 400 bottoms and 400 tops, all below z = 50
    assert org.capillary_curvature_deg == pytest.approx(0.0, abs=1e-6)  # straight vessels
    # The empty upper layer holds the global tensor, so the bed has a direction there too.
    assert org.orientation[-1] == org.global_orientation
    assert org.beyond_far > 0.0 and org.within_near < 1.0


def test_a_cubic_lattice_reads_as_every_way_alike_and_covering_its_box():
    org = measure_organisation(_lattice(), EXTENT)
    assert org.axis_share == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=0.03)
    assert org.within_near > 0.95 and org.beyond_far == 0.0


def test_a_trunk_running_straight_turns_by_nothing_and_a_bent_one_by_its_bend():
    side = ((10.0, 100.0, 60.0), (10.0, 160.0, 60.0), "B01")
    straight = _graph([((10.0, 100.0, 0.0), (10.0, 100.0, 60.0), "Large_Art1"),
                       ((10.0, 100.0, 60.0), (10.0, 100.0, 120.0), "Large_Art2"), side], EXTENT)
    assert measure_organisation(straight, EXTENT).trunk_turn_deg == pytest.approx(0.0, abs=1e-6)
    bend = math.radians(30.0)
    end = (10.0, 100.0 + 60.0 * math.sin(bend), 60.0 + 60.0 * math.cos(bend))
    bent = _graph([((10.0, 100.0, 0.0), (10.0, 100.0, 60.0), "Large_Art1"),
                   ((10.0, 100.0, 60.0), end, "Large_Art2"), side], EXTENT)
    org = measure_organisation(bent, EXTENT)
    assert org.trunk_turn_deg == pytest.approx(30.0, abs=1e-6)
    assert org.field("arterial_length_um").sum() == pytest.approx(120.0, rel=1e-6)


def test_a_polyline_stored_backwards_reads_the_same():
    G = _slab()
    for *_e, d in G.edges(data=True):
        d["voxels"] = d["voxels"][::-1]
    a, b = measure_organisation(_slab(), EXTENT), measure_organisation(G, EXTENT)
    assert a == b


def test_a_known_zigzag_curves_by_its_angle():
    # Straight 10 um chords, each turned 20 degrees from the last, in the y-x plane.
    points = [np.array([50.0, 20.0, 20.0])]
    heading = 0.0
    for i in range(12):
        heading += math.radians(20.0) * (1 if i % 2 else -1)
        points.append(points[-1] + 10.0 * np.array([0.0, math.sin(heading), math.cos(heading)]))
    path = []
    for a, b in zip(points[:-1], points[1:]):
        path.extend(_line(a, b)[:-1])
    path.append(list(points[-1]))
    G = nx.MultiGraph()
    G.add_node(0, pos=points[0])
    G.add_node(1, pos=points[-1])
    G.add_edge(0, 1, voxels=path, branch_order="B01")
    org = measure_organisation(G, EXTENT)
    assert org.capillary_curvature_deg == pytest.approx(20.0, abs=3.0)


def test_the_capillary_vertices_and_their_spacing_are_recorded():
    org = measure_organisation(_slab(), EXTENT)
    positions = np.asarray(org.capillary_node_positions_rel) * np.asarray(EXTENT)
    assert len(positions) == 800
    assert set(np.round(positions[:, 0], 6)) == {0.0, 40.0}
    assert org.capillary_node_spacing_um == pytest.approx(10.0)  # the grid the vessels stand on
    old = org.to_dict()
    for name in ("capillary_node_positions_rel", "capillary_node_spacing_um", "chord_axis_share_by_cell",
                 "chord_axis_share", "long_runs", "chord_by_class_um", "tree_leaves_by_order", "trunk_straightness",
                 "tree_reach_um"):
        old.pop(name)
    loaded = Organisation.from_dict(old)
    assert loaded.capillary_node_positions_rel is None and loaded.capillary_node_spacing_um is None
    assert loaded.long_runs is None and loaded.chord_by_class_um is None and loaded.chord_axis_share_by_cell is None


def _columns(extent=(200.0, 200.0, 200.0)):
    """Four columns of capillaries joined end to end, z 10 to 160 (spanning
    150 um), and one short one, z 10 to 60."""
    edges = []
    for y, x in ((50.0, 50.0), (50.0, 150.0), (150.0, 50.0), (150.0, 150.0)):
        for z0, z1 in ((10.0, 60.0), (60.0, 110.0), (110.0, 160.0)):
            edges.append(((z0, y, x), (z1, y, x), "B01"))
    edges.append(((10.0, 100.0, 100.0), (60.0, 100.0, 100.0), "B01"))
    return _graph(edges, extent)


def test_chord_directions_runs_and_lengths_are_read_off_the_network():
    """Harvey, 2026-10-25: E14.5's capillary columns, read from whatever
    template has them, never built in."""
    slab = measure_organisation(_slab(), EXTENT)
    assert slab.chord_axis_share == pytest.approx((1.0, 0.0, 0.0))
    occupied = [s for s, n in zip(slab.chord_axis_share_by_cell, slab.capillary_nodes) if n]
    assert all(s == pytest.approx((1.0, 0.0, 0.0)) for s in occupied)
    assert slab.long_runs == (0, 0, 0)  # 40 um vessels, none joined end to end
    assert slab.chord_by_class_um == (pytest.approx(40.0), None, None, None)
    columns = measure_organisation(_columns(), (200.0, 200.0, 200.0))
    assert columns.long_runs == (4, 0, 0)  # the short column spans 50 um
    lattice = measure_organisation(_lattice(), EXTENT)
    # Lines of 20 um vessels: along z they span 80 um, along y and x 180.
    assert lattice.long_runs == (0, 50, 50)
    assert lattice.chord_by_class_um == tuple(pytest.approx(20.0) for _ in range(3)) + (None,)
    # 400 vessels along z, 450 along y and along x, all 20 um.
    assert lattice.chord_axis_share == pytest.approx((400 / 1300, 450 / 1300, 450 / 1300))


def test_the_trees_leaves_and_straightness_are_read_off_the_trees():
    """A trunk from the root at x 0 running straight along x in three
    vessels, a 30 um side branch off each of its two junctions: three leaves
    (the trunk's end and the two branches), the trunk's end straight from
    the root (1), each branch's chord over its path below 1."""
    from haemolynx.synthetic.organisation import tree_shape

    positions = {0: (10.0, 100.0, 0.0), 1: (10.0, 100.0, 60.0), 2: (10.0, 100.0, 120.0), 3: (10.0, 100.0, 180.0),
                 4: (10.0, 130.0, 60.0), 5: (10.0, 130.0, 120.0)}
    edges = [(0, 1, "Large_Art1", 60.0), (1, 2, "Large_Art2", 60.0), (2, 3, "Large_Art3", 60.0),
             (1, 4, "Large_Art2", 30.0), (2, 5, "Large_Art3", 30.0)]
    positions = {k: np.asarray(v) for k, v in positions.items()}
    leaves, straightness, reach = tree_shape(edges, positions, roots=[0])
    assert leaves == (("Large_Art2", 1), ("Large_Art3", 2))
    chords = [180.0, math.hypot(60.0, 30.0), math.hypot(120.0, 30.0)]
    assert reach == (("arterial", pytest.approx(float(np.median(chords))), pytest.approx(180.0)),)
    branch_1 = math.hypot(60.0, 30.0) / 90.0
    branch_2 = math.hypot(120.0, 30.0) / 150.0
    assert straightness == pytest.approx(float(np.median([1.0, branch_1, branch_2])))
    # Without the roots named, the order-1 vessel's free end is the root.
    assert tree_shape(edges, positions) == (leaves, straightness, reach)


def test_chord_class_is_the_cone_round_an_axis_or_none():
    from haemolynx.synthetic.organisation import chord_class

    assert chord_class(np.array([10.0, 1.0, 1.0])) == 0
    assert chord_class(np.array([0.0, -5.0, 0.5])) == 1
    assert chord_class(np.array([1.0, 1.0, 1.0])) == 3  # 55 degrees from every axis
    assert chord_class(np.array([0.0, 0.0, 0.0])) == 3


def test_no_capillaries_no_organisation_and_json_round_trips():
    G = _graph([((10.0, 100.0, 0.0), (10.0, 100.0, 60.0), "Large_Art1")], EXTENT)
    assert measure_organisation(G, EXTENT) is None
    org = measure_organisation(_slab(), EXTENT)
    assert Organisation.from_dict(org.to_dict()) == org


def test_a_template_carries_its_organisation_and_an_older_one_loads_without(tmp_path):
    from haemolynx.synthetic.template import measure_template
    from synthetic_fixtures import TINY_ROLES

    T = measure_template(
        tiny_hierarchical_graph(), inlet_nodes=TINY_ROLES["inlet"], outlet_nodes=TINY_ROLES["outlet"]
    )
    assert T.organisation is not None and T.organisation.shape == grid_shape(TINY_EXTENT_UM)
    assert NetworkTemplate.from_dict(T.to_dict()) == T
    old = T.to_dict()
    old.pop("organisation")
    assert NetworkTemplate.from_dict(old).organisation is None
    assert T.scaled(2.0).organisation == T.organisation  # cells are shares of the extent


def test_organisation_rows_score_a_network_against_its_template():
    slab, lattice = measure_organisation(_slab(), EXTENT), measure_organisation(_lattice(), EXTENT)
    rows = {r.characteristic: r for r in organisation_rows(slab, slab)}
    assert all(r.within in (True, None) for r in rows.values())
    assert rows["capillary length by cell (correlation)"].synthetic == pytest.approx(1.0)
    rows = {r.characteristic: r for r in organisation_rows(slab, lattice)}
    assert rows["capillary length within 30 degrees of z (share)"].within is False
    assert rows["box beyond 50 um of a capillary"].within is False
    assert rows["capillary length along z (correlation)"].within is False
    # Another grid (a box drawn at another scale): the fields cannot be lined up.
    other = replace(lattice, shape=(1, 1, 1))
    rows = {r.characteristic: r for r in organisation_rows(slab, other)}
    assert rows["capillary length by cell (correlation)"].within is None
    assert organisation_rows(None, None) == []


def test_correlation_is_none_for_a_flat_field():
    assert correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert fixture_template().organisation is None
