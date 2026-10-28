"""A box a synthetic network leaves empty (``synthetic/hole.py``,
``PLAN_synthetic_surround.md`` phase 2): its geometry, and a network drawn
round one -- nothing of it inside, the template's counts for the rest of the
box -- and one drawn without, exactly as before."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.synthetic import GRAPH_TEMPLATE_RECORD, GenerationOptions, Hole, centred_hole, generate_network
from haemolynx.synthetic import bed as bed_module
from haemolynx.synthetic.generate import validate_synthetic_graph
from haemolynx.synthetic.hole import GRAPH_HOLE
from haemolynx.synthetic.polylines import bend_to_lengths
from haemolynx.synthetic.trees import exit_fraction, fit_fraction
from synthetic_fixtures import fixture_template

HOLE = Hole((10.0, 10.0, 10.0), (20.0, 20.0, 20.0))


# -- the geometry --------------------------------------------------------------


def test_a_point_is_in_the_hole_only_inside_its_faces():
    points = np.array([[15.0, 15.0, 15.0], [10.0, 15.0, 15.0], [9.0, 15.0, 15.0], [15.0, 15.0, 25.0]])
    assert HOLE.contains(points).tolist() == [True, False, False, False]  # on a face is not in it
    assert HOLE.contains(points, margin=1.5).tolist() == [True, True, True, False]
    assert bool(HOLE.contains(np.array([15.0, 15.0, 15.0])))


@pytest.mark.parametrize(
    "start, end, crosses",
    [
        ((15, 15, 0), (15, 15, 30), True),  # straight through
        ((15, 15, 0), (15, 15, 12), True),  # into it
        ((10, 15, 0), (10, 15, 30), False),  # along a face
        ((0, 0, 0), (10, 10, 10), False),  # to a corner
        ((15, 15, 0), (15, 15, 10), False),  # ending on a face
        ((15, 25, 0), (15, 25, 30), False),  # past it
    ],
)
def test_a_segment_crosses_the_hole_only_through_its_inside(start, end, crosses):
    assert HOLE.crosses(start, end) is crosses
    assert bool(HOLE.crosses_each(np.array([start]), np.array([end]))[0]) is crosses


def test_the_batch_crossing_test_agrees_with_the_one_at_a_time():
    rng = np.random.default_rng(0)
    starts, ends = rng.uniform(0.0, 30.0, size=(400, 3)), rng.uniform(0.0, 30.0, size=(400, 3))
    ends[:50, 0] = starts[:50, 0]  # some flat along an axis
    assert HOLE.crosses_each(starts, ends).tolist() == [HOLE.crosses(a, b) for a, b in zip(starts, ends)]
    assert HOLE.crosses_each(starts, ends, clearance=2.0).tolist() == [
        HOLE.crosses(a, b, clearance=2.0) for a, b in zip(starts, ends)
    ]


def test_a_segment_is_cut_short_where_it_comes_within_the_margin():
    assert HOLE.entry_fraction((15, 15, 0), (15, 15, 20)) == pytest.approx(0.5)
    assert HOLE.entry_fraction((15, 15, 0), (15, 15, 20), clearance=2.0) == pytest.approx(0.4)
    assert HOLE.entry_fraction((15, 25, 0), (15, 25, 20)) == 1.0
    # From within the margin already: where it enters the hole itself.
    assert HOLE.entry_fraction((15, 15, 9), (15, 15, 19), clearance=2.0) == pytest.approx(0.1)
    extent = np.array([30.0, 30.0, 30.0])
    start, end = np.array([15.0, 15.0, 0.0]), np.array([15.0, 15.0, 20.0])
    assert fit_fraction(start, end, extent) == exit_fraction(start, end, extent) == 1.0
    assert fit_fraction(start, end, extent, HOLE, 2.0) == pytest.approx(0.4)


def test_a_path_cutting_a_corner_between_its_points_enters_the_hole():
    corner = np.array([[[15.0, 21.0, 18.0], [15.0, 18.0, 21.0]]])  # both points outside; the step cuts its edge
    assert not HOLE.contains(corner[0]).any()
    assert HOLE.entered_by(corner).tolist() == [True]
    assert HOLE.depth(corner)[0] > 0.0  # read along the step
    assert HOLE.depth(corner, subdivisions=1)[0] == 0.0  # not at the points alone
    assert HOLE.entered_by(np.zeros((0, 2, 3))).tolist() == [] and HOLE.depth(np.zeros((0, 2, 3))).tolist() == []


def test_a_point_is_pushed_out_along_its_face_and_kept_in_the_box():
    through_z = Hole((0.0, 10.0, 10.0), (30.0, 20.0, 20.0))  # an xy surround's: the whole depth
    pushed = through_z.push_out((0.0, 12.0, 15.0), axes=(1, 2), margin=1.0, extent_um=(30.0, 30.0, 30.0))
    assert pushed.tolist() == [0.0, 9.0, 15.0]  # on its z face, the nearest way off the hole
    assert through_z.push_out((0.0, 25.0, 15.0), axes=(1, 2), margin=1.0).tolist() == [0.0, 25.0, 15.0]
    wall = Hole((-1.0, -1.0, 10.0), (31.0, 31.0, 20.0))  # across the whole box but x
    assert wall.push_out((15.0, 15.0, 12.0), axes=(1, 2), extent_um=(30.0, 30.0, 30.0)).tolist() == [15.0, 15.0, 10.0]
    # No way out along z or y that stays in the box: left where it was.
    assert wall.push_out((15.0, 15.0, 12.0), axes=(0, 1), extent_um=(30.0, 30.0, 30.0)).tolist() == [15.0, 15.0, 12.0]


def test_a_holes_share_is_of_the_box_it_lies_in():
    assert HOLE.share_of((30.0, 30.0, 30.0)) == pytest.approx(1000.0 / 27000.0)
    assert Hole((20.0, 20.0, 20.0), (40.0, 40.0, 40.0)).volume_in((30.0, 30.0, 30.0)) == pytest.approx(1000.0)
    assert HOLE.grown(2.0) == Hole((8.0, 8.0, 8.0), (22.0, 22.0, 22.0))
    centred = centred_hole((120.0, 400.0, 400.0), (120.0, 200.0, 200.0))
    assert centred == Hole((0.0, 100.0, 100.0), (120.0, 300.0, 300.0))
    assert Hole.from_record(centred.to_record()) == centred and Hole.from_record(None) is None
    with pytest.raises(ValueError, match="positive extent"):
        Hole((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="does not fit"):
        centred_hole((100.0, 100.0, 100.0), (120.0, 50.0, 50.0))


def test_a_template_round_a_hole_draws_the_rest_of_its_counts():
    template = fixture_template()
    rest = template.around_hole(0.25)
    assert rest.classes["capillary"].count == 120 and rest.classes["capillary"].mean_length_um == 50.0
    assert [o.edge_count for o in rest.orders] == [round(o.edge_count * 0.75) for o in template.orders]
    assert rest.loops.independent_loops == round(51 * 0.75)
    assert rest.inlets == template.inlets and rest.dead_ends.leaves_image == template.dead_ends.leaves_image
    assert rest.extent_um == template.extent_um
    assert any("25% of the box left to a hole" in n for n in rest.provenance.notes)
    assert template.around_hole(0.0) is template
    with pytest.raises(ValueError):
        template.around_hole(1.0)


# -- the pieces of the generator -------------------------------------------------


def test_the_bed_drops_its_vertices_in_the_hole_and_its_edges_across_it():
    points = np.array(
        [[15.0, 15.0, 15.0], [15.0, 15.0, 5.0], [15.0, 15.0, 25.0], [5.0, 5.0, 5.0], [15.0, 15.0, 21.0], [25.0, 25.0, 25.0]]
    )
    pairs = [(1, 2), (1, 3), (3, 2), (2, 5), (0, 3)]
    kept, kept_pairs = bed_module._out_of_hole(points, pairs, HOLE, 2.0, 10, np.random.default_rng(0))
    # The vertex in it and the one within 2 um of it go.
    assert kept.tolist() == [[15.0, 15.0, 5.0], [15.0, 15.0, 25.0], [5.0, 5.0, 5.0], [25.0, 25.0, 25.0]]
    # 1-2 ran straight through it, 3-2 through its corner (by (12.5, 12.5, 17.5)); 0-3 lost a vertex.
    assert kept_pairs == [(0, 2), (1, 3)]
    thinned, _none = bed_module._out_of_hole(points, None, HOLE, 2.0, 2, np.random.default_rng(0))
    assert len(thinned) == 2


def test_a_bow_is_kept_out_of_the_hole_or_drawn_straight():
    rng = np.random.default_rng(3)
    starts = rng.uniform(0.0, 30.0, size=(300, 3))
    starts = starts[~HOLE.contains(starts, 1.0)]
    ends = np.clip(starts + rng.normal(0.0, 6.0, size=starts.shape), 0.0, 30.0)
    keep = ~HOLE.crosses_each(starts, ends) & ~HOLE.contains(ends, 1.0)
    starts, ends = starts[keep], ends[keep]
    chords = np.linalg.norm(ends - starts, axis=1)
    targets = chords * 1.6
    paths = bend_to_lengths(starts, ends, targets, rng, extent_um=np.array([30.0, 30.0, 30.0]), hole=HOLE)
    lengths = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths])
    assert not HOLE.entered_by(np.stack(paths)).any()
    assert np.allclose(lengths, targets, rtol=1e-6)  # turned off it, at their length
    # Without the hole, some of the same bows pass through it.
    free = bend_to_lengths(starts, ends, targets, np.random.default_rng(3), extent_um=np.array([30.0, 30.0, 30.0]))
    assert HOLE.entered_by(np.stack(free)).sum() > 0


def test_a_bow_no_turn_keeps_out_runs_along_its_chord():
    """One box cannot close round a chord from every side, so a hole that
    every turn enters is a stand-in."""

    class Everywhere:
        def depth(self, paths):
            return np.ones(len(paths))

        def entered_by(self, paths):
            return np.ones(len(paths), dtype=bool)

    starts, ends = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]), np.array([[1.0, 1.0, 9.0], [5.0, 5.0, 6.0]])
    paths = bend_to_lengths(starts, ends, np.array([12.0, 1.0]), np.random.default_rng(0), extent_um=np.array([10.0] * 3), hole=Everywhere())
    assert np.allclose(paths[0][:, :2], 1.0) and np.linalg.norm(np.diff(paths[0], axis=0), axis=1).sum() == pytest.approx(8.0)
    assert len(paths[1]) == 2  # a target no longer than its chord was straight already


# -- a network drawn round a hole --------------------------------------------------


@pytest.fixture(scope="module", params=[("xy", "delaunay", 1), ("xy", "voronoi", 2), ("xyz", "delaunay", 3), ("xyz", "voronoi", 1)])
def round_a_hole(request):
    axes, bed, seed = request.param
    template = fixture_template()
    hole = centred_hole(template.scaled(2.0, axes).extent_um, template.extent_um)
    G = generate_network(template, seed=seed, options=GenerationOptions(scale=2.0, scale_axes=axes, bed_model=bed, hole=hole))
    return template, hole, G


def test_nothing_drawn_lies_in_the_hole(round_a_hole):
    _template, hole, G = round_a_hole
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes])
    assert not hole.contains(positions).any()
    for *_e, data in G.edges(data=True):
        assert not hole.contains(np.asarray(data["voxels"], float)).any()
    # Nor a node nearer it than half the margin the generator keeps.
    spacing = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["spacing_um"]
    assert not hole.contains(positions, 0.5 * bed_module.FACE_MARGIN_SPACINGS * spacing).any()
    assert Hole.from_record(G.graph[GRAPH_HOLE]) == hole


def test_the_rest_of_the_box_gets_the_templates_counts(round_a_hole):
    template, hole, G = round_a_hole
    record = G.graph[GRAPH_TEMPLATE_RECORD]
    generated = record["generated"]
    scaled = template.scaled(2.0, record["options"]["scale_axes"])
    rest = scaled.around_hole(hole.share_of(scaled.extent_um))
    assert generated["hole_share"] == pytest.approx(hole.share_of(scaled.extent_um))
    assert record["template"]["classes"]["capillary"]["count"] == rest.classes["capillary"].count
    joined = generated["dead_end_pairs_joined"]
    wanted = rest.classes["capillary"].count - joined
    assert abs(generated["capillaries"] - wanted) <= 4
    # The bed's spacing is the rest of the box's, not the whole box's.
    vertices = wanted - (rest.loops.independent_loops + joined) - (rest.inlet_count + rest.outlet_count) + 1
    rest_volume = np.prod(scaled.extent_um) - hole.volume_in(scaled.extent_um)
    assert generated["spacing_um"] == pytest.approx((rest_volume / vertices) ** (1 / 3), rel=1e-12)


def test_the_validation_refuses_a_network_reaching_into_its_hole(round_a_hole):
    _template, hole, G = round_a_hole
    H = G.copy()
    node = next(iter(H.nodes))
    H.nodes[node]["pos"] = 0.5 * (hole.lo + hole.hi)
    with pytest.raises(ValueError, match="lies in the hole"):
        validate_synthetic_graph(H)


# -- and one drawn without ------------------------------------------------------------


def _same_network(a, b) -> bool:
    if sorted(a.nodes) != sorted(b.nodes) or sorted(a.edges(keys=True)) != sorted(b.edges(keys=True)):
        return False
    if any(not np.array_equal(a.nodes[n]["pos"], b.nodes[n]["pos"]) for n in a.nodes):
        return False
    for u, v, k, data in a.edges(keys=True, data=True):
        other = b.edges[u, v, k]
        if data.keys() != other.keys() or data["voxels"] != other["voxels"]:
            return False
        if any(data[name] != other[name] for name in data if name != "voxels"):
            return False
    return True


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"bed_model": "voronoi"},
        {"scale": 1.5, "scale_axes": "xy"},
        {"open_ends": "template", "interior_dead_ends": True},
    ],
)
def test_a_hole_outside_the_box_draws_the_network_drawn_without_one(options):
    """Every hole check is asked and excludes nothing: the network is the
    one drawn without a hole, to the bit. (That one is HEAD's: checked
    against a git archive of 3e819c4d on nine networks, fixture and E14.5,
    both beds, 2026-10-28.)"""
    template = fixture_template()
    far = Hole((1e5, 1e5, 1e5), (1e5 + 1.0, 1e5 + 1.0, 1e5 + 1.0))
    without = generate_network(template, seed=4, options=GenerationOptions(**options))
    beside = generate_network(template, seed=4, options=GenerationOptions(**options, hole=far))
    assert _same_network(without, beside)
    assert GRAPH_HOLE not in without.graph and GRAPH_HOLE in beside.graph


def test_without_a_hole_no_hole_check_is_asked(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a hole was asked about")

    for name in ("contains", "crosses", "crosses_each", "entry_fraction", "depth", "entered_by", "push_out", "share_of"):
        monkeypatch.setattr(Hole, name, refuse)
    G = generate_network(fixture_template(), seed=5, options=GenerationOptions(bed_model="voronoi", scale=1.3))
    assert G.number_of_edges() > 100
