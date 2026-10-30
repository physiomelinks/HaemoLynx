"""The image's open ends read for a synthetic network drawn round it, and
the network drawn round it and joined to it (``synthetic/surround.py``,
``PLAN_synthetic_surround.md`` phases 1 and 3)."""
from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.dead_ends import DEAD_END_KIND, LEAVES_IMAGE
from haemolynx.haemodynamics.open_ends import DRAINS, FEEDS
from haemolynx.synthetic import surround
from haemolynx.synthetic.generate import (
    JOIN_BED,
    JOIN_TRUNK,
    SURROUND_JOIN,
    SYNTHETIC_CLASS,
    SYNTHETIC_ROLES,
    TRUNK_MIN_OUTWARD,
    GenerationOptions,
    Join,
    boundary_nodes,
    drawn_template,
    generate_network,
)
from haemolynx.synthetic.hole import GRAPH_HOLE, Hole, centred_hole
from haemolynx.synthetic.surround import (
    ARTERIAL,
    BED,
    CAPILLARY,
    CONTINUATIONS,
    IMAGE_NODE_ID,
    IMAGE_OFFSET,
    IMAGE_PART,
    NETWORK_PART,
    SURROUND_CONNECTOR,
    SURROUND_RECORD,
    SURROUNDED,
    SYNTHETIC_PART,
    TRUNK,
    UNKNOWN,
    VENOUS,
    ImageOpenEnd,
    read_open_ends,
    surround_network,
    validate_surrounded_graph,
)
from haemolynx.synthetic.template import GRAPH_NETWORK_SOURCE, GRAPH_TEMPLATE_RECORD
from synthetic_fixtures import fixture_template

CENTRE = (50.0, 50.0, 50.0)


def _star(arms: dict) -> nx.MultiGraph:
    """A junction at the box's centre with one vessel to each of *arms*
    (``{node: (tip position, edge attributes, kind)}``)."""
    G = nx.MultiGraph(image_extent_um=(100.0, 100.0, 100.0))
    G.add_node(0, pos=np.asarray(CENTRE))
    for node, (tip, attributes, kind) in arms.items():
        G.add_node(node, pos=np.asarray(tip, float), **({DEAD_END_KIND: kind} if kind else {}))
        voxels = [list(np.asarray(CENTRE) + (np.asarray(tip) - CENTRE) * t) for t in np.linspace(0, 1, 11)]
        G.add_edge(0, node, voxels=voxels, length=float(np.linalg.norm(np.asarray(tip) - CENTRE)), **attributes)
    return G


def _fixture() -> nx.MultiGraph:
    return _star({
        1: ((50.0, 50.0, 100.0), {"branch_order": "Art2", "diameter_um": 12.0}, LEAVES_IMAGE),
        2: ((0.0, 50.0, 50.0), {"branch_order": "Ven1", "diameter_um": 20.0}, LEAVES_IMAGE),
        3: ((50.0, 100.0, 50.0), {"branch_order": "B3", "diameter_um": 5.0}, LEAVES_IMAGE),
        4: ((50.0, 50.0, 30.0), {"branch_order": "B2", "diameter_um": 5.0}, "blind"),
    })


def test_each_open_end_is_read_with_its_face_heading_width_and_class():
    ends = {e.node: e for e in read_open_ends(_fixture())}
    assert set(ends) == {1, 2, 3}  # the blind end inside the tissue is not one
    art, ven, cap = ends[1], ends[2], ends[3]
    assert (art.face, ven.face, cap.face) == ("x1", "z0", "y1")
    assert art.heading == pytest.approx((0.0, 0.0, 1.0)) and ven.heading == pytest.approx((-1.0, 0.0, 0.0))
    assert (art.diameter_um, art.order, art.category, art.side) == (12.0, "Art2", "arteriole", ARTERIAL)
    assert (ven.order, ven.category, ven.side) == ("Ven1", "venule", VENOUS)
    assert (cap.order, cap.category, cap.side) == ("B3", "capillary", CAPILLARY)
    assert art.position_um == (50.0, 50.0, 100.0)
    # Unsolved: no direction, and none is needed.
    assert all(e.direction is None and e.designated is None for e in ends.values())
    assert (art.continuation, ven.continuation, cap.continuation) == (TRUNK, TRUNK, BED)


@pytest.mark.parametrize("direction", [FEEDS, DRAINS, None])
@pytest.mark.parametrize(
    "order, expected",
    [("Art2", TRUNK), ("Large_Art1", TRUNK), ("Ven1", TRUNK), ("Large_Ven2", TRUNK), ("B3", BED)],
)
def test_every_arteriole_and_venule_runs_on_as_a_trunk_whichever_way_it_flowed(monkeypatch, order, expected, direction):
    """D5 (Harvey, 2026-10-27): the solved direction was right at no more
    tree ends than a coin on the cut-out harness, so it decides nothing."""
    G = _star({1: ((50.0, 50.0, 100.0), {"branch_order": order, "diameter_um": 10.0}, LEAVES_IMAGE)})
    monkeypatch.setattr(surround, "open_end_directions", lambda _G: {} if direction is None else {1: direction})
    (end,) = read_open_ends(G)
    assert end.direction == direction and end.continuation == expected
    assert CONTINUATIONS == (BED, TRUNK)


def test_designated_inlets_and_outlets_run_on_as_trunks():
    G = _fixture()
    ends = {e.node: e for e in read_open_ends(G, inlets=[1], outlets=[2, 4])}
    assert ends[1].designated == "inlet" and ends[1].continuation == TRUNK
    assert ends[2].designated == "outlet" and ends[2].continuation == TRUNK
    # A designated dead end inside the tissue is read too.
    assert ends[4].designated == "outlet" and ends[4].face == "x0"


def test_a_synthetic_class_stands_in_for_a_branch_order():
    G = _star({
        1: ((50.0, 50.0, 100.0), {"synthetic_class": "venule", "diameter_um": 9.0}, LEAVES_IMAGE),
        2: ((50.0, 0.0, 50.0), {"diameter_um": 9.0}, LEAVES_IMAGE),
    })
    ends = {e.node: e for e in read_open_ends(G)}
    assert (ends[1].order, ends[1].category, ends[1].side) == (None, "venule", VENOUS)
    assert (ends[2].category, ends[2].side, ends[2].continuation) == (None, UNKNOWN, BED)


def test_the_extent_comes_from_the_graph_or_the_caller():
    G = _fixture()
    del G.graph["image_extent_um"]
    with pytest.raises(ValueError, match="image_extent_um"):
        read_open_ends(G)
    assert {e.node for e in read_open_ends(G, extent_um=(100.0, 100.0, 100.0))} == {1, 2, 3}


def test_an_open_end_is_a_plain_record():
    end = ImageOpenEnd(1, (0.0, 0.0, 0.0), "z0", None, None, None, None, None)
    assert end.side == UNKNOWN and end.continuation == BED


# -- phase 3: the network drawn round the image and joined to it ------------

SCALE = 2.0


def _hole():
    T = fixture_template()
    return centred_hole(T.scaled(SCALE).extent_um, T.extent_um)


def _joined_shell(*joins):
    T = fixture_template()
    return generate_network(T, seed=3, options=GenerationOptions(scale=SCALE, hole=_hole()), joins=list(joins))


def _join_node(G, key):
    (node,) = [n for n, d in G.nodes(data=True) if d.get(SURROUND_JOIN) == key]
    return node


def _trunk(G, tip):
    """The trunk from *tip* to its inlet or outlet: the run of vessels of its
    first one's class and width."""
    (first,) = [d for *_e, d in G.edges(tip, data=True)]
    path = [tip]
    while not {"inlet", "outlet"} & set(G.nodes[path[-1]].get(SYNTHETIC_ROLES, ())):
        (step,) = [
            b for _a, b, d in G.edges(path[-1], data=True)
            if b not in path and d[SYNTHETIC_CLASS] == first[SYNTHETIC_CLASS] and d["diameter_um"] == first["diameter_um"]
        ]
        path.append(step)
    return path


def test_a_trunk_runs_from_its_join_out_to_the_box_handing_vessels_to_the_bed():
    hole = _hole()
    start = (float(hole.lo[0] + 40.0), float(hole.lo[1] + 90.0), float(hole.lo[2]))
    join = Join(start, JOIN_TRUNK, key="a", side="arterial", heading=(0.0, 0.0, -1.0), diameter_um=9.0, category="arteriole", order="Art2")
    G = _joined_shell(join)
    tip = _join_node(G, "a")
    assert np.array_equal(G.nodes[tip]["pos"], np.asarray(start)) and G.degree(tip) == 1
    path = _trunk(G, tip)
    assert len(path) >= 3  # two pieces at least: a joint between hands a vessel to the bed
    assert all(np.allclose(G.nodes[n]["pos"][:2], start[:2]) for n in path)
    assert G.nodes[path[-1]]["pos"][2] == 0.0 and "inlet" in G.nodes[path[-1]][SYNTHETIC_ROLES]
    assert all(G.get_edge_data(a, b)[0]["diameter_um"] == 9.0 for a, b in zip(path[:-1], path[1:]))
    for joint in path[1:-1]:
        assert "arteriole_boundary" in G.nodes[joint][SYNTHETIC_ROLES] and G.degree(joint) >= 3
    generated = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["joins"]
    assert generated == {"bed": 0, "trunks": {"arterial": 1, "venous": 0}, "trunk_edges": len(path) - 1}
    # The trunk's end is one of the box's inlets, in place of one the
    # template would have grown a tree from.
    assert len(boundary_nodes(G)["inlet"]) == len(fixture_template().scaled(SCALE).inlets)


def test_a_trunk_of_an_order_cut_to_a_sliver_keeps_its_joints_a_bed_vertex_apart():
    """Regression: a template measured from an image can hold an order only
    as the sliver its face cut (E14.5's harness: one vessel of 0.44 um). Its
    trunk was cut into pieces that long, 400 joints each wanting a bed
    vessel, and the few bed vertices near them took 46-78 vessels each: the
    junction trades over them took 42 and 96 minutes."""
    T = fixture_template()
    sliver = replace(T, orders=tuple(
        replace(o, mean_length_um=0.44, mean_tortuosity=1.0) if o.label == "Art2" else o for o in T.orders
    ))
    hole = _hole()
    start = (float(hole.lo[0] + 40.0), float(hole.lo[1] + 90.0), float(hole.lo[2]))
    join = Join(start, JOIN_TRUNK, key="a", side="arterial", heading=(0.0, 0.0, -1.0), diameter_um=9.0, category="arteriole", order="Art2")
    G = generate_network(sliver, seed=3, options=GenerationOptions(scale=SCALE, hole=hole), joins=[join])
    path = _trunk(G, _join_node(G, "a"))
    min_separation = 0.5 * G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["spacing_um"]
    reach = start[2]
    assert 3 <= len(path) <= round(reach / min_separation) + 1  # not 0.44 um apart: 227 pieces
    assert all(G.degree(joint) == 3 for joint in path[1:-1])
    assert max(d for _n, d in G.degree()) <= 4


def test_a_trunk_lying_along_the_holes_face_is_turned_out_of_it():
    hole = _hole()
    start = (float(hole.lo[0] + 40.0), float(hole.lo[1] + 90.0), float(hole.hi[2]))
    G = _joined_shell(Join(start, JOIN_TRUNK, key="v", side="venous", heading=(0.0, 1.0, 0.0), category="venule"))
    path = _trunk(G, _join_node(G, "v"))
    direction = G.nodes[path[-1]]["pos"] - np.asarray(start)
    assert direction[2] / np.linalg.norm(direction) >= TRUNK_MIN_OUTWARD - 1e-9
    assert "outlet" in G.nodes[path[-1]][SYNTHETIC_ROLES]
    # Its width is its class's, the join giving none.
    assert G.get_edge_data(path[0], path[1])[0]["diameter_um"] == fixture_template().classes["venule"].mean_diameter_um


def test_a_bed_join_is_joined_by_one_capillary():
    hole = _hole()
    start = (float(hole.hi[0]), float(hole.lo[1] + 50.0), float(hole.lo[2] + 70.0))
    G = _joined_shell(Join(start, JOIN_BED, key=17))
    tip = _join_node(G, 17)
    assert np.array_equal(G.nodes[tip]["pos"], np.asarray(start)) and G.degree(tip) == 1
    assert G.get_edge_data(*next(iter(G.edges(tip))))[0][SYNTHETIC_CLASS] == "capillary"
    assert G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["joins"]["bed"] == 1
    assert len(boundary_nodes(G)["inlet"]) == len(fixture_template().scaled(SCALE).inlets)


@pytest.fixture(scope="module")
def image_run():
    """A synthetic network standing in for an image run: its open ends cut by
    every face, its inlets and outlets on the x faces."""
    T = fixture_template()
    image = generate_network(T, seed=7, options=GenerationOptions(open_ends="template"))
    return T, image, boundary_nodes(image)


def _surround(image_run, **options):
    T, image, roles = image_run
    return surround_network(
        image, T, inlets=roles["inlet"], outlets=roles["outlet"],
        roles={k: v for k, v in roles.items() if k not in ("inlet", "outlet")},
        options=GenerationOptions(**{"scale": SCALE, **options}), seed=3,
    )


@pytest.fixture(scope="module", params=["xy", "xyz"])
def surrounded(request, image_run):
    return request.param, _surround(image_run, scale_axes=request.param)


def test_the_image_sits_unchanged_in_the_middle_of_the_box(image_run, surrounded):
    T, image, _roles = image_run
    axes, G = surrounded
    offset = (np.asarray(T.scaled(SCALE, axes).extent_um) - np.asarray(T.extent_um)) / 2.0
    assert np.allclose(G.graph[IMAGE_OFFSET], offset)
    assert G.graph[GRAPH_NETWORK_SOURCE] == SURROUNDED
    for node, data in image.nodes(data=True):
        assert np.array_equal(G.nodes[node]["pos"], data["pos"] + offset)
        assert G.nodes[node][IMAGE_NODE_ID] == node
    assert all(G.edges[u, v, k][NETWORK_PART] == IMAGE_PART for u, v, k in image.edges(keys=True))
    drawn = [d for *_e, d in G.edges(data=True) if d[NETWORK_PART] == SYNTHETIC_PART]
    assert len(drawn) == G.number_of_edges() - image.number_of_edges() > 0
    validate_surrounded_graph(G, image)


def test_every_open_end_is_joined_unless_the_box_stops_at_its_face(image_run, surrounded):
    _T, image, roles = image_run
    axes, G = surrounded
    record = G.graph[SURROUND_RECORD]
    ends = read_open_ends(image, inlets=roles["inlet"], outlets=roles["outlet"])
    on_z = {e.node for e in ends if e.face[0] == "z"}
    if axes == "xy":
        # The box is the image's depth: an end cut by a z face stays open.
        assert on_z and set(record["left_open"]) == on_z
        assert all(G.degree(n) == 1 and G.nodes[n][DEAD_END_KIND] == image.nodes[n][DEAD_END_KIND] for n in on_z)
    else:
        assert record["left_open"] == []
    joined = {e.node for e in ends} - set(record["left_open"])
    assert set(record["joined"]) == joined
    assert all(G.degree(n) >= 2 and DEAD_END_KIND not in G.nodes[n] for n in joined)
    trunks = sum(1 for e in ends if e.node in joined and e.continuation == TRUNK)
    assert trunks and (record["trunks"], record["bed_joins"]) == (trunks, len(joined) - trunks)
    assert nx.is_connected(G)


def test_the_arterioles_and_venules_run_on_to_new_inlets_and_outlets_on_the_box(image_run, surrounded):
    T, _image, roles = image_run
    axes, G = surrounded
    outer = np.asarray(T.scaled(SCALE, axes).extent_um)
    found = boundary_nodes(G)
    # The image's own inlets and outlets are joined: their roles are now
    # their trunks' ends', on the box, beside the template's own.
    assert not set(found["inlet"]) & set(roles["inlet"]) and not set(found["outlet"]) & set(roles["outlet"])
    drawn = drawn_template(T, G.graph[GRAPH_TEMPLATE_RECORD]["options"])
    trunks = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["joins"]["trunks"]
    assert len(found["inlet"]) == max(len(drawn.inlets), trunks["arterial"])
    assert len(found["outlet"]) == max(len(drawn.outlets), trunks["venous"])
    for node in found["inlet"] + found["outlet"]:
        position = G.nodes[node]["pos"]
        assert np.min(np.concatenate([position, outer - position])) <= 1e-6
    # The image's other boundary nodes keep their roles.
    assert roles["arteriole_boundary"]
    assert all("arteriole_boundary" in G.nodes[n][SYNTHETIC_ROLES] for n in roles["arteriole_boundary"])


def test_a_tip_short_of_its_face_is_drawn_on_to_it(image_run):
    T, image, roles = image_run
    image = image.copy()
    end = next(e for e in read_open_ends(image) if e.continuation == BED)
    axis = "zyx".index(end.face[0])
    inward = np.zeros(3)
    inward[axis] = 3.0 if end.face.endswith("0") else -3.0
    (u, v, k), = image.edges(end.node, keys=True)
    voxels = np.asarray(image.edges[u, v, k]["voxels"], float)
    position = image.nodes[end.node]["pos"]
    voxels[0 if np.linalg.norm(voxels[0] - position) < np.linalg.norm(voxels[-1] - position) else -1] += inward
    image.edges[u, v, k]["voxels"] = voxels.tolist()
    image.nodes[end.node]["pos"] = position + inward
    G = _surround((T, image, roles), scale_axes="xyz")
    assert G.graph[SURROUND_RECORD]["connectors"] == 1
    (connector,) = [(a, b, d) for a, b, d in G.edges(data=True) if d.get(SURROUND_CONNECTOR)]
    assert end.node in connector[:2] and connector[2][NETWORK_PART] == SYNTHETIC_PART
    assert connector[2]["length"] == pytest.approx(3.0)
    assert connector[2]["diameter_um"] == image.edges[u, v, k]["diameter_um"]


def test_a_connector_carries_its_vessels_mask_tags_so_the_large_tier_runs_on_across_it(image_run):
    """Regression: the connector from an image vessel's tip to its face took
    the vessel's class but not its large-mask tag. The large tier is followed
    along tagged vessels from the trunk's inlet, so it stopped there, and on
    E14.5 every one of the image's 86 large vessels lost its Large_ label."""
    from haemolynx.graph.branch_order import LARGE_MASK_VESSEL_TYPE

    T, image, roles = image_run
    image = image.copy()
    inlet = roles["inlet"][0]
    (u, v, k), = image.edges(inlet, keys=True)
    image.edges[u, v, k][LARGE_MASK_VESSEL_TYPE] = "arteriole"
    end = next(e for e in read_open_ends(image, inlets=roles["inlet"], outlets=roles["outlet"]) if e.node == inlet)
    axis = "zyx".index(end.face[0])
    inward = np.zeros(3)
    inward[axis] = 3.0 if end.face.endswith("0") else -3.0
    voxels = np.asarray(image.edges[u, v, k]["voxels"], float)
    position = image.nodes[inlet]["pos"]
    voxels[0 if np.linalg.norm(voxels[0] - position) < np.linalg.norm(voxels[-1] - position) else -1] += inward
    image.edges[u, v, k]["voxels"] = voxels.tolist()
    image.nodes[inlet]["pos"] = position + inward
    G = _surround((T, image, roles), scale_axes="xyz")
    (connector,) = [d for _a, _b, d in G.edges(inlet, data=True) if d.get(SURROUND_CONNECTOR)]
    assert connector[LARGE_MASK_VESSEL_TYPE] == "arteriole"
    assert connector[SYNTHETIC_CLASS] == image.edges[u, v, k][SYNTHETIC_CLASS]


def test_with_no_trees_of_their_own_beyond_the_image_only_the_trunks_reach_the_box(image_run):
    G = _surround(image_run, arteriole_density=0.0, venule_density=0.0)
    trunks = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["joins"]["trunks"]
    found = boundary_nodes(G)
    assert len(found["inlet"]) == trunks["arterial"] > 0 and len(found["outlet"]) == trunks["venous"] > 0


def test_a_surround_is_drawn_again_from_its_seed(image_run):
    a, b = _surround(image_run), _surround(image_run)
    assert sorted(a.edges(keys=True)) == sorted(b.edges(keys=True))
    assert all(np.array_equal(a.nodes[n]["pos"], b.nodes[n]["pos"]) for n in a.nodes)


def test_the_image_and_the_shells_plan_count_the_surrounds_nodes(image_run, surrounded):
    """What preflight checks the dense solvers' memory on before the draw."""
    T, image, roles = image_run
    axes, G = surrounded
    plan, _joins = surround.planned_surround_bed(
        image, T, inlets=roles["inlet"], outlets=roles["outlet"],
        options=GenerationOptions(scale=SCALE, scale_axes=axes),
    )
    assert G.number_of_nodes() == pytest.approx(image.number_of_nodes() + plan.nodes, rel=0.03)


def test_a_surround_needs_a_larger_box_and_int_nodes(image_run):
    T, image, _roles = image_run
    with pytest.raises(ValueError, match="synthetic_scale above 1"):
        surround_network(image, T, options=GenerationOptions(scale=1.0))
    with pytest.raises(ValueError, match="ints"):
        surround_network(nx.relabel_nodes(image, str), T, options=GenerationOptions(scale=SCALE))


def test_the_validation_refuses_a_broken_surround(image_run, surrounded):
    _T, image, _roles = image_run
    _axes, G = surrounded
    node = next(iter(image.nodes))
    moved = G.copy()
    moved.nodes[node]["pos"] = moved.nodes[node]["pos"] + 1.0
    with pytest.raises(ValueError, match="moved"):
        validate_surrounded_graph(moved, image)
    inside = G.copy()
    hole = Hole.from_record(G.graph[GRAPH_HOLE])
    inside.add_node(max(G.nodes) + 1, pos=(hole.lo + hole.hi) / 2.0)
    with pytest.raises(ValueError, match="lies in the hole"):
        validate_surrounded_graph(inside, image)
    loose = G.copy()
    tip = G.graph[SURROUND_RECORD]["joined"][0]
    loose.remove_edges_from([(a, b, k) for a, b, k, d in loose.edges(tip, keys=True, data=True) if d[NETWORK_PART] == SYNTHETIC_PART])
    with pytest.raises(ValueError, match="was not joined"):
        validate_surrounded_graph(loose, image)
