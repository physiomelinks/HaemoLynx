"""The synthetic network generator (``synthetic/generate.py`` and the modules
it drives: ``polylines``, ``trees``, ``bed``)."""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from scipy.spatial import cKDTree

from haemolynx.graph._helpers import calculate_path_length
from haemolynx.graph.branch_order import assign_vessel_branch_orders
from haemolynx.graph.dead_ends import BLIND, DEAD_END_KIND, DEAD_END_KIND_SOURCE, LEAVES_IMAGE, SOURCE_SYNTHETIC
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes
from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_SYNTHETIC, EDGE_DIAMETER_BASIS
from haemolynx.statistics import compute_vessel_statistics_by_measure
from haemolynx.statistics.bifurcation import fitted_murray_exponent
from haemolynx.synthetic import bed as bed_module
from haemolynx.synthetic import polylines, trees
from haemolynx.synthetic.compare import compare_templates, comparison_summary
from haemolynx.synthetic.trees import face_distance_um
from haemolynx.synthetic.generate import (
    DEFAULT_SEED,
    SYNTHETIC,
    SYNTHETIC_CLASS,
    SYNTHETIC_ROLES,
    GenerationOptions,
    boundary_nodes,
    drawn_template,
    extension_share,
    face_extension_shares,
    generate_network,
    planned_bed,
    validate_synthetic_graph,
)
from haemolynx.synthetic.hole import centred_hole
from haemolynx.synthetic.template import (
    FACES,
    GRAPH_NETWORK_SOURCE,
    GRAPH_TEMPLATE_RECORD,
    Endpoint,
    ROLES,
    OrderTarget,
    measure_template,
)
from haemolynx.synthetic.organisation import ISOTROPIC, Organisation, cell_index, grid_shape
from synthetic_fixtures import fixture_template

SEED = 7
#: The template as it was imaged: its vessels leaving the box and its dead
#: ends inside the tissue drawn too. The defaults (Harvey, 2026-10-25: D2a,
#: D3a) leave both out, so its loops and boundary nodes are not the template's.
FAITHFUL = GenerationOptions(interior_dead_ends=True, open_ends="template")


@pytest.fixture(scope="module")
def generated():
    return generate_network(fixture_template(), seed=SEED)


@pytest.fixture(scope="module")
def faithful():
    return generate_network(fixture_template(), seed=SEED, options=FAITHFUL)


def _labelled(G: nx.MultiGraph) -> dict:
    roles = boundary_nodes(G)
    assign_vessel_branch_orders(
        G,
        roles["inlet"],
        outlet_nodes=roles["outlet"],
        arteriole_boundary_nodes=roles["arteriole_boundary"],
        venule_boundary_nodes=roles["venule_boundary"],
        large_arteriole_boundary_nodes=roles["large_arteriole_boundary"],
        large_venule_boundary_nodes=roles["large_venule_boundary"],
    )
    return roles


def _measured(G: nx.MultiGraph):
    roles = _labelled(G)
    return measure_template(
        G,
        inlet_nodes=roles["inlet"],
        outlet_nodes=roles["outlet"],
        arteriole_boundary_nodes=roles["arteriole_boundary"],
        venule_boundary_nodes=roles["venule_boundary"],
        samples=False,
    )


# -- the contract --------------------------------------------------------------


def test_the_generated_graph_keeps_the_contract(generated):
    G = generated
    validate_synthetic_graph(G, fixture_template())
    assert isinstance(G, nx.MultiGraph)
    assert all(isinstance(n, int) for n in G)
    extent = np.asarray(G.graph["image_extent_um"])
    assert np.allclose(extent, fixture_template().extent_um)
    assert G.graph["image_voxel_size_zyx"] == fixture_template().voxel_size_zyx
    assert G.graph["image_voxel_size_xyz"] == tuple(reversed(fixture_template().voxel_size_zyx))
    assert G.graph[GRAPH_NETWORK_SOURCE] == SYNTHETIC
    record = G.graph[GRAPH_TEMPLATE_RECORD]
    assert record["seed"] == SEED
    assert record["generated"]["edges"] == G.number_of_edges()
    assert record["template"]["classes"]["capillary"]["samples"] is None
    for node, data in G.nodes(data=True):
        assert data["pos"].shape == (3,)
        assert np.all(data["pos"] >= 0) and np.all(data["pos"] <= extent + 1e-9)
        assert G.degree(node) != 2
    for u, v, data in G.edges(data=True):
        voxels = np.asarray(data["voxels"])
        assert voxels.shape[1] == 3 and len(voxels) >= 2
        assert max(np.linalg.norm(np.diff(voxels, axis=0), axis=1)) <= 0.5 + 1e-9
        assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
        assert data["diameter_um"] > 0
        assert data["diameter_source"] == DIAMETER_SOURCE_SYNTHETIC
        assert "branch_order" not in data
        assert data[SYNTHETIC_CLASS] in {"arteriole", "capillary", "venule"}
    assert_no_forbidden_edge_attributes(G)
    assert nx.is_connected(G)


def test_every_dead_end_has_a_kind_matching_its_place(generated):
    G = generated
    extent = np.asarray(G.graph["image_extent_um"])
    for node in G:
        if G.degree(node) != 1:
            assert DEAD_END_KIND not in G.nodes[node]
            continue
        pos = G.nodes[node]["pos"]
        on_face = min(pos.min(), (extent - pos).min()) <= 1e-9
        assert G.nodes[node][DEAD_END_KIND] == (LEAVES_IMAGE if on_face else BLIND)
        assert G.nodes[node][DEAD_END_KIND_SOURCE] == SOURCE_SYNTHETIC


def test_roles_and_endpoints_are_the_templates(faithful):
    generated = faithful
    T = fixture_template()
    roles = boundary_nodes(generated)
    assert set(roles) == set(ROLES)
    assert len(roles["inlet"]) == 2 and len(roles["outlet"]) == 2
    assert len(roles["arteriole_boundary"]) == T.arteriole_boundary_count
    assert len(roles["venule_boundary"]) == T.venule_boundary_count
    assert roles["large_arteriole_boundary"] == [] and roles["large_venule_boundary"] == []
    for node in roles["inlet"]:
        assert generated.nodes[node]["pos"][2] == 0.0  # on x0
    for node in roles["outlet"]:
        assert generated.nodes[node]["pos"][2] == pytest.approx(T.extent_um[2])  # on x1
    for role in ("arteriole_boundary", "venule_boundary"):
        for node in roles[role]:
            assert generated.degree(node) >= 3
            assert generated.nodes[node][SYNTHETIC_ROLES] == (role,)


def test_without_interior_dead_ends_every_tree_leaf_joins_the_bed(generated):
    """The defaults: a tree leaf the template ended blind hands over to the
    bed, so there are as many boundary nodes as the template's or more."""
    T = fixture_template()
    roles = boundary_nodes(generated)
    assert len(roles["arteriole_boundary"]) >= T.arteriole_boundary_count
    assert len(roles["venule_boundary"]) >= T.venule_boundary_count
    tree = {u for u, v, d in generated.edges(data=True) if d[SYNTHETIC_CLASS] != "capillary"} | {
        v for u, v, d in generated.edges(data=True) if d[SYNTHETIC_CLASS] != "capillary"
    }
    ends = [n for n in tree if generated.degree(n) == 1]
    assert set(ends) <= set(roles["inlet"]) | set(roles["outlet"])


def test_the_validator_catches_a_broken_contract(generated):
    G = generated.copy()
    first = next(iter(G.edges(keys=True)))
    G.edges[first]["branch_order"] = "Art1"
    with pytest.raises(ValueError, match="already has a branch order"):
        validate_synthetic_graph(G)
    G = generated.copy()
    G.edges[first]["weight"] = 1.0
    with pytest.raises(ValueError, match="weight"):
        validate_synthetic_graph(G)
    G = generated.copy()
    G.graph.pop("image_extent_um")
    with pytest.raises(ValueError, match="image_extent_um"):
        validate_synthetic_graph(G)


def test_every_drawn_vessel_carries_the_templates_diameter_basis(generated):
    T = replace(fixture_template(), diameter_basis="endothelial")
    G = generate_network(T, seed=SEED)
    assert {data[EDGE_DIAMETER_BASIS] for *_e, data in G.edges(data=True)} == {"endothelial"}
    assert G.graph[GRAPH_TEMPLATE_RECORD]["template"]["diameter_basis"] == "endothelial"
    # The same draws: the basis is a mark, not a draw.
    assert [d["diameter_um"] for *_e, d in G.edges(data=True)] == [d["diameter_um"] for *_e, d in generated.edges(data=True)]
    # A template that names no basis leaves every vessel on the run's own.
    assert fixture_template().diameter_basis is None
    assert not any(EDGE_DIAMETER_BASIS in data for *_e, data in generated.edges(data=True))
    validate_synthetic_graph(G, T)
    with pytest.raises(ValueError, match="diameter basis is not the template's"):
        validate_synthetic_graph(G, replace(T, diameter_basis="plasma_column"))


# -- determinism ---------------------------------------------------------------


def test_the_same_seed_draws_the_same_network(generated):
    again = generate_network(fixture_template(), seed=SEED)
    assert list(again.nodes) == list(generated.nodes)
    assert sorted(again.edges(keys=True)) == sorted(generated.edges(keys=True))
    for node in generated:
        assert np.array_equal(again.nodes[node]["pos"], generated.nodes[node]["pos"])
    for u, v, k, data in generated.edges(keys=True, data=True):
        assert again.edges[u, v, k]["voxels"] == data["voxels"]
        assert again.edges[u, v, k]["diameter_um"] == data["diameter_um"]


def test_another_seed_draws_another_network(generated):
    other = generate_network(fixture_template(), seed=SEED + 1)
    assert sorted(other.edges(keys=True)) != sorted(generated.edges(keys=True)) or any(
        not np.array_equal(other.nodes[n]["pos"], generated.nodes[n]["pos"]) for n in generated if n in other
    )


def test_a_generator_wins_over_the_seed_and_none_draws_fresh():
    rng = np.random.default_rng(3)
    one = generate_network(fixture_template(), seed=99, rng=rng)
    two = generate_network(fixture_template(), seed=99, rng=np.random.default_rng(3))
    assert sorted(one.edges(keys=True)) == sorted(two.edges(keys=True))
    fresh = generate_network(fixture_template(), seed=None)
    assert fresh.number_of_edges() > 0
    assert DEFAULT_SEED == 20240917


# -- what is hit exactly -------------------------------------------------------


def test_branch_orders_come_out_as_the_templates_with_exact_tree_counts(generated):
    G = generated.copy()
    _labelled(G)
    counts = Counter(d["branch_order"] for *_e, d in G.edges(data=True))
    for order in fixture_template().orders:
        label = order.label.replace("BO", "B0")
        assert counts[label] == order.edge_count, (label, counts)
    for *_e, d in G.edges(data=True):
        prefix = {"arteriole": "Art", "venule": "Ven", "capillary": "B"}[d[SYNTHETIC_CLASS]]
        assert d["branch_order"].startswith(prefix), d
    assert sum(counts[k] for k in counts if k.startswith("B")) == G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["capillaries"]


def test_murray_holds_exactly_at_every_tree_junction(generated):
    G = generated.copy()
    _labelled(G)
    exponent = fixture_template().murray_exponent
    checked = 0
    for node in G:
        incident = [(v, d) for _u, v, d in G.edges(node, data=True)]
        if G.degree(node) < 3 or any(d[SYNTHETIC_CLASS] == "capillary" for _v, d in incident):
            continue
        orders = sorted(incident, key=lambda item: int("".join(ch for ch in item[1]["branch_order"] if ch.isdigit())))
        parent, daughters = orders[0][1]["diameter_um"], [d["diameter_um"] for _v, d in orders[1:]]
        fitted = fitted_murray_exponent(parent, daughters)
        assert fitted == pytest.approx(exponent, abs=1e-6)
        checked += 1
    assert checked >= 6


def test_capillary_count_loops_and_dead_ends_are_the_templates(faithful):
    generated = faithful
    T = fixture_template()
    record = generated.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert abs(record["capillaries"] - T.classes["capillary"].count) <= 3
    loops = generated.number_of_edges() - generated.number_of_nodes() + 1
    assert abs(loops - T.loops.independent_loops) <= 2
    kinds = Counter(generated.nodes[n][DEAD_END_KIND] for n in generated if generated.degree(n) == 1)
    assert abs(kinds[LEAVES_IMAGE] - T.dead_ends.leaves_image) <= 2
    assert abs(kinds[BLIND] - T.dead_ends.interior) <= 2
    assert set(Counter(dict(generated.degree()).values())) <= {1, 3, 4}


def test_the_defaults_join_the_dead_ends_left_out_in_pairs(generated):
    """A closed network without interior dead ends: its only dead ends are
    its inlets and outlets. The fixture's 4 ends leaving the image (less
    its 4 inlets and outlets) and 8 inside the tissue are joined in 6 pairs:
    6 capillaries fewer, 6 loops more, the capillaries' length kept."""
    T = fixture_template()
    record = generated.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert record["dead_end_pairs_joined"] == 6
    assert abs(record["capillaries"] - (T.classes["capillary"].count - 6)) <= 3
    kinds = Counter(generated.nodes[n][DEAD_END_KIND] for n in generated if generated.degree(n) == 1)
    roles = boundary_nodes(generated)
    assert kinds[LEAVES_IMAGE] == sum(1 for n in [*roles["inlet"], *roles["outlet"]] if generated.degree(n) == 1)
    assert kinds[BLIND] == 0
    loops = generated.number_of_edges() - generated.number_of_nodes() + 1
    assert abs(loops - (T.loops.independent_loops + 6)) <= 2
    capillary_length = sum(d["length"] for *_e, d in generated.edges(data=True) if d[SYNTHETIC_CLASS] == "capillary")
    assert capillary_length == pytest.approx(T.classes["capillary"].total_length_um, rel=0.05)


def test_dead_end_pairs_joined_counts_what_is_not_drawn():
    from haemolynx.synthetic.generate import dead_end_pairs_joined

    T = fixture_template()  # 8 ends leaving the image (4 of them inlets and outlets), 8 inside
    assert dead_end_pairs_joined(T, GenerationOptions()) == 6
    assert dead_end_pairs_joined(T, FAITHFUL) == 0
    assert dead_end_pairs_joined(T, GenerationOptions(interior_dead_ends=True)) == 2
    assert dead_end_pairs_joined(T, GenerationOptions(open_ends="template")) == 4
    assert dead_end_pairs_joined(T, {"open_ends": "closed", "interior_dead_ends": False}) == 6  # a record's dict
    assert dead_end_pairs_joined(T, {}) == 0


def test_with_the_templates_open_ends_its_count_leaves_the_box():
    T = fixture_template()
    G = generate_network(T, seed=SEED, options=FAITHFUL)
    kinds = Counter(G.nodes[n][DEAD_END_KIND] for n in G if G.degree(n) == 1)
    assert abs(kinds[LEAVES_IMAGE] - T.dead_ends.leaves_image) <= 2
    assert abs(kinds[BLIND] - T.dead_ends.interior) <= 2


def test_without_interior_dead_ends_every_dead_end_is_on_a_face():
    G = generate_network(fixture_template(), seed=SEED, options=GenerationOptions(interior_dead_ends=False, open_ends="template"))
    kinds = Counter(G.nodes[n][DEAD_END_KIND] for n in G if G.degree(n) == 1)
    assert kinds[BLIND] == 0
    assert kinds[LEAVES_IMAGE] >= 4


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_closed_network_without_interior_dead_ends_ends_only_at_its_inlets_and_outlets(seed):
    """Harvey, 2026-10-25 (D2a, D3a). A bed vertex meant to be a junction
    used to be left now and then with one vessel, a dead end nobody asked for."""
    G = generate_network(fixture_template(), seed=seed, options=GenerationOptions(interior_dead_ends=False))
    roles = boundary_nodes(G)
    ends = {n for n in G if G.degree(n) == 1}
    assert ends <= set(roles["inlet"]) | set(roles["outlet"])


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_a_closed_network_keeps_every_vessel_in_its_box(seed):
    T = fixture_template()
    G = generate_network(T, seed=seed)
    assert G.graph[GRAPH_TEMPLATE_RECORD]["options"]["open_ends"] == "closed"
    extent = np.asarray(T.extent_um)
    roots = set(boundary_nodes(G)["inlet"]) | set(boundary_nodes(G)["outlet"])
    for n, data in G.nodes(data=True):
        if n not in roots:
            assert np.all(data["pos"] > 0.0) and np.all(data["pos"] < extent), (n, data["pos"])
    for *_e, d in G.edges(data=True):
        path = np.asarray(d["voxels"])
        assert np.all(path >= -1e-9) and np.all(path <= extent + 1e-9)
    G_open = generate_network(T, seed=seed, options=GenerationOptions(open_ends="template"))
    on_face = [n for n in G_open if n not in set(boundary_nodes(G_open)["inlet"]) | set(boundary_nodes(G_open)["outlet"])
               and np.min(np.concatenate([G_open.nodes[n]["pos"], extent - G_open.nodes[n]["pos"]])) < 1e-6]
    assert on_face  # the template's open ends are on its faces


def test_a_closed_trees_child_heading_out_is_turned_back_at_its_length():
    """A tip 5 um from the x face heading straight at it: every plane's
    children leave the box. Open, they are cut at the face; closed, they are
    turned back off it at their full chord."""
    extent = np.array([100.0, 100.0, 100.0])
    tip = trees.Tip(node=0, position=np.array([50.0, 50.0, 95.0]), direction=np.array([0.0, 0.0, 1.0]), diameter_um=5.0, generation=1)
    occupancy = trees._Occupancy([tip.position])
    chords, deflections = [20.0, 20.0], [math.radians(30.0), math.radians(30.0)]
    for closed in (False, True):
        children = trees._place_children(
            tip, chords, deflections, extent, occupancy, 20.0, 5.0, np.random.default_rng(0), None, closed
        )
        for end, direction, chord, cut in children:
            assert np.all(end >= 0.0) and np.all(end <= extent)
            if closed:
                assert not cut and chord == pytest.approx(20.0) and direction[2] < 0.0
            else:
                assert cut and end[2] == pytest.approx(100.0) and chord < 20.0


def test_open_ends_must_be_one_of_the_choices():
    from haemolynx.pipeline import default_schema
    from haemolynx.synthetic import OPEN_ENDS

    with pytest.raises(ValueError, match="open_ends"):
        generate_network(fixture_template(), seed=SEED, options=GenerationOptions(open_ends="sometimes"))
    assert default_schema()["synthetic_open_ends"].choices == OPEN_ENDS
    assert default_schema()["synthetic_open_ends"].default == "closed"


def test_scale_grows_the_box_and_the_counts():
    T = fixture_template()
    G = generate_network(T, seed=SEED, options=GenerationOptions(scale=1.5))
    assert np.allclose(G.graph["image_extent_um"], np.asarray(T.extent_um) * 1.5)
    capillaries = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["capillaries"]
    assert capillaries == pytest.approx(T.classes["capillary"].count * 1.5**3, rel=0.05)
    assert len(boundary_nodes(G)["inlet"]) == max(1, round(2 * 1.5**2))


def test_scale_along_x_and_y_grows_the_box_but_keeps_its_depth():
    T = fixture_template()
    G = generate_network(T, seed=SEED, options=GenerationOptions(scale=1.5, scale_axes="xy"))
    extent = np.asarray(G.graph["image_extent_um"], float)
    assert np.allclose(extent, np.asarray(T.extent_um) * (1.0, 1.5, 1.5))
    positions = np.array([G.nodes[n]["pos"] for n in G.nodes], float)
    assert positions[:, 0].min() >= 0.0 and positions[:, 0].max() <= T.extent_um[0] + 1e-9
    assert positions[:, 1:].max() > max(T.extent_um[1:])  # the network fills the wider box
    record = G.graph[GRAPH_TEMPLATE_RECORD]
    assert record["options"]["scale_axes"] == "xy"
    assert record["template"]["extent_um"] == list(extent)
    assert record["generated"]["capillaries"] == pytest.approx(T.classes["capillary"].count * 1.5**2, rel=0.05)
    # The inlets' x face grows as y alone.
    assert len(boundary_nodes(G)["inlet"]) == round(2 * 1.5)
    validate_synthetic_graph(G)
    with pytest.raises(ValueError, match="scale axes"):
        generate_network(T, seed=SEED, options=GenerationOptions(scale=1.5, scale_axes="yz"))


def test_the_bed_is_planned_by_the_templates_own_arithmetic():
    """The fixture's docstring counts it by hand: 160 capillaries less 51
    loops less 4 roots, plus one; closed and without interior dead ends, 6
    pairs of its dead ends are joined instead (a capillary fewer, a loop
    more each)."""
    T = fixture_template()
    plan = planned_bed(T, GenerationOptions(open_ends="template", interior_dead_ends=True))
    assert (plan.capillaries, plan.loops, plan.roots, plan.joined, plan.vertices) == (160, 51, 4, 0, 106)
    plan = planned_bed(T)
    assert (plan.capillaries, plan.loops, plan.joined, plan.vertices) == (154, 57, 6, 94)
    # The network drawn is spaced by it.
    G = generate_network(T, seed=SEED)
    spacing = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["spacing_um"]
    assert spacing == pytest.approx((np.prod(T.extent_um) / 94) ** (1.0 / 3.0))


@pytest.mark.parametrize(
    "options",
    [GenerationOptions(), GenerationOptions(scale=1.5, scale_axes="xy"), GenerationOptions(scale=2.0)],
    ids=["template's box", "1.5x along xy", "2x"],
)
def test_the_plan_counts_the_nodes_the_network_is_drawn_with(options):
    """Its vessels, the trees' and the bed's, less its loops, plus one: what
    preflight checks the dense solvers' memory on before anything is drawn."""
    T = fixture_template()
    plan = planned_bed(T, options)
    trees = sum(target.count for key, target in plan.template.classes.items() if key != "capillary")
    assert trees > 0 and plan.nodes == trees + plan.capillaries - plan.loops + 1
    G = generate_network(T, seed=SEED, options=options)
    assert G.number_of_nodes() == pytest.approx(plan.nodes, rel=0.03)


def test_the_extension_is_the_box_beyond_the_templates_own():
    own = (100.0, 200.0, 200.0)
    # Twice as wide every way: 7/8 of the volume, 3/4 of each face.
    shares = face_extension_shares(own, (200.0, 400.0, 400.0))
    assert shares["interior"] == pytest.approx(0.875) == extension_share(own, (200.0, 400.0, 400.0))
    assert all(shares[face] == pytest.approx(0.75) for face in FACES)
    # Along x and y: the z faces grow four times, the others twice.
    shares = face_extension_shares(own, (100.0, 400.0, 400.0))
    assert shares["interior"] == pytest.approx(0.75)
    assert shares["z0"] == shares["z1"] == pytest.approx(0.75)
    assert all(shares[face] == pytest.approx(0.5) for face in ("y0", "y1", "x0", "x1"))
    assert set(face_extension_shares(own, own).values()) == {0.0}


def test_round_a_hole_the_templates_size_all_of_the_box_drawn_is_extension():
    own = (100.0, 200.0, 200.0)
    hole = centred_hole((200.0, 400.0, 400.0), own)
    assert set(face_extension_shares(own, (200.0, 400.0, 400.0), hole).values()) == {1.0}
    # Along x and y the hole runs the box's depth: the z faces are the
    # template's where the hole meets them.
    hole = centred_hole((100.0, 400.0, 400.0), own)
    shares = face_extension_shares(own, (100.0, 400.0, 400.0), hole)
    assert shares["interior"] == 1.0
    assert shares["z0"] == shares["z1"] == pytest.approx(0.75)
    assert all(shares[face] == 1.0 for face in ("y0", "y1", "x0", "x1"))


def test_the_densities_scale_the_trees_beyond_the_templates_box():
    T = fixture_template()
    options = GenerationOptions(scale=2.0, scale_axes="xy", arteriole_density=0.5, venule_density=2.0)
    drawn = drawn_template(T, options)
    scaled = T.scaled(2.0, "xy")
    # Three quarters of the box is beyond the template's: 1 + 0.75 (d - 1).
    assert drawn.classes["arteriole"].count == round(scaled.classes["arteriole"].count * 0.625) == 30
    assert drawn.classes["venule"].count == round(scaled.classes["venule"].count * 1.75) == 42
    assert drawn.classes["capillary"] == scaled.classes["capillary"]
    # The inlets' and outlets' x faces are half beyond: 1 + 0.5 (d - 1).
    assert len(drawn.inlets) == round(4 * 0.75) == 3 and len(drawn.outlets) == round(4 * 1.5) == 6
    # The record's options (a dict) read the same.
    assert drawn_template(T, {"scale": 2.0, "scale_axes": "xy", "arteriole_density": 0.5, "venule_density": 2.0}) == drawn
    assert drawn_template(T, GenerationOptions(scale=2.0, scale_axes="xy")) == scaled
    assert drawn_template(T, None) is T

    G = generate_network(T, seed=SEED, options=options)
    classes = Counter(d[SYNTHETIC_CLASS] for *_e, d in G.edges(data=True))
    assert classes["arteriole"] == 30 and classes["venule"] == 42
    assert len(boundary_nodes(G)["inlet"]) == 3 and len(boundary_nodes(G)["outlet"]) == 6
    assert G.graph[GRAPH_TEMPLATE_RECORD]["options"]["arteriole_density"] == 0.5
    validate_synthetic_graph(G)


def test_no_trees_beyond_the_templates_box_leaves_the_templates_own():
    T = fixture_template()
    G = generate_network(T, seed=SEED, options=GenerationOptions(scale=2.0, arteriole_density=0.0, venule_density=0.0))
    classes = Counter(d[SYNTHETIC_CLASS] for *_e, d in G.edges(data=True))
    assert classes["arteriole"] == T.classes["arteriole"].count and classes["venule"] == T.classes["venule"].count
    assert len(boundary_nodes(G)["inlet"]) == len(T.inlets) and len(boundary_nodes(G)["outlet"]) == len(T.outlets)
    assert classes["capillary"] == pytest.approx(T.scaled(2.0).classes["capillary"].count, rel=0.05)
    validate_synthetic_graph(G)


def test_at_the_templates_size_the_densities_change_nothing():
    T = fixture_template()
    plain = generate_network(T, seed=SEED)
    dense = generate_network(T, seed=SEED, options=GenerationOptions(arteriole_density=3.0, venule_density=0.0))
    assert drawn_template(T, GenerationOptions(arteriole_density=3.0, venule_density=0.0)) is T
    assert sorted(plain.edges(keys=True)) == sorted(dense.edges(keys=True))
    assert all(np.array_equal(plain.nodes[n]["pos"], dense.nodes[n]["pos"]) for n in plain.nodes)


# -- the acceptance test: every banded characteristic within tolerance -------


def test_the_generated_network_measures_as_its_template(faithful):
    generated = faithful
    options = generated.graph[GRAPH_TEMPLATE_RECORD]["options"]
    rows = compare_templates(fixture_template(), _measured(generated.copy()), options=options)
    summary = comparison_summary(rows)
    assert summary["fitted"] >= 50
    outside = [r for r in rows if r.within is False]
    # One characteristic may miss at a single seed (a class of six vessels'
    # median, say); the generator is not allowed to miss anything exact.
    assert all(r.tolerance != "exact" for r in outside), [(r.section, r.characteristic) for r in outside]
    assert len(outside) <= 1, [(r.section, r.characteristic, r.template, r.synthetic) for r in outside]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_rows_hold_at_every_seed(seed):
    for options in (FAITHFUL, GenerationOptions()):
        G = generate_network(fixture_template(), seed=seed, options=options)
        rows = compare_templates(fixture_template(), _measured(G), options=G.graph[GRAPH_TEMPLATE_RECORD]["options"])
        exact = [r for r in rows if r.tolerance == "exact"]
        assert exact and all(r.within is not False for r in exact), [(r.section, r.characteristic) for r in exact if r.within is False]


def test_with_the_defaults_only_the_loops_leave_their_bands(generated):
    options = generated.graph[GRAPH_TEMPLATE_RECORD]["options"]
    rows = compare_templates(fixture_template(), _measured(generated.copy()), options=options)
    outside = {(r.section, r.characteristic) for r in rows if r.within is False}
    assert outside <= {("loops", "independent loops"), ("loops", "independent loops per mm^3")}, outside
    reported = {r.characteristic: r.tolerance for r in rows if r.tolerance.startswith(("not drawn", "not held"))}
    assert reported["leaving the image"] == "not drawn (a closed network)"
    assert reported["inside the tissue"] == "not drawn (no dead ends inside the tissue)"
    assert reported["arteriole_boundary nodes"].startswith("not held (every tree leaf joins the bed")


def test_the_generated_network_runs_through_the_statistics(generated):
    G = generated.copy()
    roles = _labelled(G)
    sections = compute_vessel_statistics_by_measure(
        G,
        node_positions={n: G.nodes[n]["pos"] for n in G},
        voxel_size=G.graph["image_voxel_size_zyx"],
        image_dimensions=(121, 205, 205),
        enabled_measures=frozenset({"basic", "vessel_classes", "dead_ends", "loop_hierarchy", "murray_law"}),
        inlet_nodes=roles["inlet"],
        outlet_nodes=roles["outlet"],
    )
    assert sections["basic"]["Total Edges"] == G.number_of_edges()
    assert sections["vessel_classes"]["Vessel Classes"]["Capillaries"]["Vessel Count"] == G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["capillaries"]


# -- fallbacks and other schemes -----------------------------------------------


def test_a_venous_side_without_orders_attaches_the_outlets_to_the_bed():
    T = fixture_template()
    T = replace(T, orders=tuple(o for o in T.orders if o.category != "venule"), classes={k: v for k, v in T.classes.items() if k != "venule"})
    G = generate_network(T, seed=SEED)
    roles = boundary_nodes(G)
    assert roles["outlet"] == roles["venule_boundary"]
    for node in roles["outlet"]:
        assert G.degree(node) == 1
    _labelled(G)
    labels = Counter(d["branch_order"] for *_e, d in G.edges(data=True))
    assert not any(k.startswith("Ven") for k in labels)
    assert labels["Art1"] == 2


def test_a_capillary_only_template_has_no_trees():
    T = fixture_template()
    T = replace(
        T,
        scheme="capillary",
        orders=(),
        classes={"capillary": T.classes["capillary"]},
        arteriole_boundary_count=0,
        venule_boundary_count=0,
    )
    G = generate_network(T, seed=SEED)
    roles = boundary_nodes(G)
    assert roles["arteriole_boundary"] == [] and roles["venule_boundary"] == []
    assert all(d[SYNTHETIC_CLASS] == "capillary" for *_e, d in G.edges(data=True))
    _labelled(G)
    assert all(d["branch_order"].startswith("B") for *_e, d in G.edges(data=True))


def test_a_large_tier_on_one_side_hands_over_at_its_boundary():
    T = fixture_template()
    # One large arteriole from each inlet, each bifurcating into the small
    # tier: Large_Art1 x2, then Art1 x4, Art2 x8, Art3 x12.
    large = OrderTarget("Large_Art1", "large_arteriole", 1, 2, 60.0, 1.05, 30.0, 20.0, 0.1)
    small = tuple(
        replace(o, edge_count=o.edge_count * 2) if o.category == "arteriole" else o for o in T.orders
    )
    T = replace(
        T,
        orders=(large,) + small,
        classes={
            **T.classes,
            "arteriole": replace(T.classes["arteriole"], count=24, total_length_um=24 * 34.0),
            "large_arteriole": replace(T.classes["arteriole"], category="large_arteriole", count=2, total_length_um=120.0, mean_length_um=60.0, mean_diameter_um=20.0),
        },
        arteriole_boundary_count=12,
        large_arteriole_boundary_count=2,
    )
    G = generate_network(T, seed=SEED, options=FAITHFUL)
    roles = boundary_nodes(G)
    assert len(roles["large_arteriole_boundary"]) == 2
    assert len(roles["arteriole_boundary"]) == 12
    # The hierarchy reads the Large tier from the vessels tagged as lying in
    # a large mask (graph.branch_order.LARGE_MASK_VESSEL_TYPE): the generator
    # tags its large-tier vessels with their side, and nothing else.
    from haemolynx.graph import branch_order as branch_order_module
    from haemolynx.synthetic.generate import LARGE_MASK_VESSEL_TYPE

    assert getattr(branch_order_module, "LARGE_MASK_VESSEL_TYPE", LARGE_MASK_VESSEL_TYPE) == LARGE_MASK_VESSEL_TYPE
    tagged = {d.get(LARGE_MASK_VESSEL_TYPE) for *_e, d in G.edges(data=True) if d[SYNTHETIC_CLASS] == "large_arteriole"}
    assert tagged == {"arteriole"}
    assert all(LARGE_MASK_VESSEL_TYPE not in d for *_e, d in G.edges(data=True) if d[SYNTHETIC_CLASS] != "large_arteriole")
    _labelled(G)
    labels = Counter(d["branch_order"] for *_e, d in G.edges(data=True))
    assert labels["Large_Art1"] == 2 and labels["Art1"] == 4 and labels["Art3"] == 12
    assert all(d[SYNTHETIC_CLASS] == "large_arteriole" for *_e, d in G.edges(data=True) if d["branch_order"] == "Large_Art1")


def _large_tier_only_template():
    """The fixture template as a run with large-vessel masks alone leaves one
    (E14.5's): Large_Art/Large_Ven orders straight into the capillary bed, no
    Art/Ven tier, the hand-off nodes counted as large-vessel boundary nodes."""
    T = fixture_template()
    renamed = {"arteriole": ("large_arteriole", "Large_Art"), "venule": ("large_venule", "Large_Ven")}
    orders = tuple(
        replace(o, label=f"{renamed[o.category][1]}{o.index}", category=renamed[o.category][0]) for o in T.orders
    )
    classes = {
        "capillary": T.classes["capillary"],
        "large_arteriole": replace(T.classes["arteriole"], category="large_arteriole"),
        "large_venule": replace(T.classes["venule"], category="large_venule"),
    }
    return replace(
        T,
        orders=orders,
        classes=classes,
        arteriole_boundary_count=0,
        venule_boundary_count=0,
        large_arteriole_boundary_count=6,
        large_venule_boundary_count=4,
    )


def test_a_large_tier_alone_hands_over_to_the_bed_at_large_boundary_nodes():
    """Regression (E14.5 as a template, 2026-10-24): with no Art/Ven tier the
    large tier's leaves were marked arteriole/venule boundary nodes, so the
    run labelled the large tier Art/Ven and counted no large boundary nodes."""
    T = _large_tier_only_template()
    G = generate_network(T, seed=SEED, options=FAITHFUL)
    roles = boundary_nodes(G)
    assert roles["arteriole_boundary"] == [] and roles["venule_boundary"] == []
    assert len(roles["large_arteriole_boundary"]) == 6 and len(roles["large_venule_boundary"]) == 4
    summary = assign_vessel_branch_orders(
        G,
        roles["inlet"],
        outlet_nodes=roles["outlet"],
        large_arteriole_boundary_nodes=roles["large_arteriole_boundary"],
        large_venule_boundary_nodes=roles["large_venule_boundary"],
        expects_hierarchical=False,
    )
    assert summary["mode"] == "large_tier"
    labels = Counter(d["branch_order"] for *_e, d in G.edges(data=True))
    trees = {label: n for label, n in labels.items() if not label.startswith("B")}
    assert trees == {o.label: o.edge_count for o in T.orders}
    assert not any(label.startswith(("Art", "Ven")) for label in labels)
    for *_e, d in G.edges(data=True):
        expected = "capillary" if d["branch_order"].startswith("B") else ("large_arteriole" if "Art" in d["branch_order"] else "large_venule")
        assert d[SYNTHETIC_CLASS] == expected


def _trunk_template():
    """A large tier shaped like E14.5's: two trunks per side over six orders
    at a flat width (an order's mean hardly falls), no Art/Ven tier."""
    T = _large_tier_only_template()
    orders = []
    for side, label, width in (("large_arteriole", "Large_Art", 20.0), ("large_venule", "Large_Ven", 22.0)):
        for index in range(1, 7):
            orders.append(OrderTarget(f"{label}{index}", side, index, 2, 30.0, 1.1, 40.0, width + (index % 2), 0.1))
    classes = {
        **T.classes,
        "large_arteriole": replace(T.classes["large_arteriole"], count=12, total_length_um=360.0, mean_length_um=30.0, median_length_um=29.0, mean_diameter_um=20.5),
        "large_venule": replace(T.classes["large_venule"], count=12, total_length_um=360.0, mean_length_um=30.0, median_length_um=29.0, mean_diameter_um=22.5),
    }
    return replace(T, orders=tuple(orders), classes=classes, large_arteriole_boundary_count=12, large_venule_boundary_count=12)


def test_trunk_children_per_tip_lets_a_trunk_go_on_as_one():
    assert trees.trunk_children_per_tip(2, 2) == [1, 1]
    assert trees.trunk_children_per_tip(3, 2) == [1, 1]
    assert trees.trunk_children_per_tip(2, 3) == [2, 1]
    assert trees.trunk_children_per_tip(2, 6) == [3, 3]
    assert trees.trunk_children_per_tip(2, 0) == [] and trees.trunk_children_per_tip(0, 2) == []
    notes: list = []
    assert trees.trunk_children_per_tip(1, 5, notes, "Large_Art3") == [5] and notes


def test_a_trunk_hands_side_branches_to_the_bed_at_its_orders_widths():
    """Harvey, 2026-10-24: the large tier is a trunk with side branches, its
    widths the template's per order. Murray's law at its exponent shrank a
    trunk of 15 orders below a micron on E14.5."""
    T = _trunk_template()
    G = generate_network(T, seed=SEED)
    roles = boundary_nodes(G)
    summary = assign_vessel_branch_orders(
        G,
        roles["inlet"],
        outlet_nodes=roles["outlet"],
        large_arteriole_boundary_nodes=roles["large_arteriole_boundary"],
        large_venule_boundary_nodes=roles["large_venule_boundary"],
        expects_hierarchical=False,
    )
    assert summary["mode"] == "large_tier"
    labels = Counter(d["branch_order"] for *_e, d in G.edges(data=True))
    assert {k: v for k, v in labels.items() if not k.startswith("B")} == {o.label: o.edge_count for o in T.orders}
    # Each order's widths are its own: their mean is the order's exactly.
    for order in T.orders:
        widths = [d["diameter_um"] for *_e, d in G.edges(data=True) if d["branch_order"] == order.label]
        assert np.mean(widths) == pytest.approx(order.mean_diameter_um, rel=1e-9)
    # A trunk going on as one vessel hands a side branch (a capillary) to the bed.
    handoffs = [
        n for n in roles["large_arteriole_boundary"] + roles["large_venule_boundary"]
        if sorted(d[SYNTHETIC_CLASS] for *_e, d in G.edges(n, data=True)).count("capillary") == 1
        and G.degree(n) == 3
    ]
    assert len(handoffs) >= 8  # two trunks a side, five orders on
    for n in handoffs:
        classes = Counter(d[SYNTHETIC_CLASS] for *_e, d in G.edges(n, data=True))
        assert classes["capillary"] == 1 and sum(classes.values()) == 3


def test_the_tree_rescale_stops_at_the_templates_longest_and_most_tortuous():
    """Harvey, 2026-10-24: no class-length rescale stretches a vessel past
    the template's longest or most tortuous; the shortfall is noted. (One
    vessel left of a tier was stretched to 904 um, tortuosity 14.6.)"""
    from haemolynx.synthetic import generate as generate_module
    from haemolynx.synthetic.draft import TREE, NetworkDraft

    T = fixture_template()
    T = replace(T, classes={**T.classes, "arteriole": replace(T.classes["arteriole"], total_length_um=5000.0)})
    longest, most_tortuous = generate_module._tree_length_caps(T.classes["arteriole"], T, "arteriole")
    assert longest == pytest.approx(trees.LENGTH_CLIP[1] * 32.0)
    assert 1.0 < most_tortuous < 1.5  # mean 1.1, two sigmas up the draw's spread
    draft = NetworkDraft()
    a = draft.add_node([0.0, 0.0, 0.0])
    ends = [draft.add_node([0.0, 0.0, 40.0 * (i + 1)]) for i in range(3)]
    for end, chord in zip(ends, (40.0, 80.0, 120.0)):
        draft.add_edge(a, end, category="arteriole", order="Art1", length_um=chord * 1.05, diameter_um=10.0, part=TREE)
    generate_module._rescale_tree_lengths(draft, T)
    for edge, chord in zip(draft.edges, (40.0, 80.0, 120.0)):
        assert edge.length_um == pytest.approx(min(longest, chord * most_tortuous))
    assert any("longer or more tortuous than any of the template's" in n for n in draft.notes)


def test_the_tree_rescale_reads_the_graph_sources_own_extremes():
    from haemolynx.synthetic import generate as generate_module
    from haemolynx.synthetic.template import Samples

    T = fixture_template()
    cls = replace(T.classes["arteriole"], samples=Samples(lengths_um=(10.0, 55.0), tortuosities=(1.02, 1.4)))
    assert generate_module._tree_length_caps(cls, T, "arteriole") == (55.0, 1.4)


def test_a_first_vessel_longer_than_the_box_is_shortened_inside_it():
    """Harvey, 2026-10-24: a root's first vessel stays in the box -- re-aimed,
    else shortened -- since cut at a face its tree would be that one vessel
    (E14.5's graph source drew the one inlet's out through a face)."""
    T = fixture_template()
    T = replace(
        T,
        inlets=(Endpoint("x0", (0.5, 0.5, 0.0)),),
        orders=tuple(replace(o, mean_length_um=4000.0) if o.label == "Art1" else o for o in T.orders),
    )
    G = generate_network(T, seed=SEED)  # validates: one piece
    roles = boundary_nodes(G)
    inlet = roles["inlet"][0]
    (first,) = list(G.neighbors(inlet))
    extent = np.asarray(T.extent_um)
    assert face_distance_um(np.asarray(G.nodes[first]["pos"]), extent) > 1.0  # inside, not on a face
    assert G.degree(first) >= 3  # its tree grows on from it
    labels = {d[SYNTHETIC_CLASS] for *_e, d in G.edges(first, data=True)}
    assert labels == {"arteriole"}
    notes = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["notes"]
    assert any("has no room in the box" in n for n in notes)


def test_a_first_vessel_that_fits_is_drawn_as_before():
    """Re-aiming only happens when the direction drawn leaves the box: the
    fixture's first vessels fit, and their draws are the ones they were."""
    G = generate_network(fixture_template(), seed=SEED)
    notes = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["notes"]
    assert not any("has no room in the box" in n for n in notes)


def test_a_tree_whose_every_leaf_is_at_a_face_keeps_one_on_the_bed():
    """Regression (5dda9ac9): a side with no leaf inside the box returned
    before the rule that each tree keeps a leaf on the bed, and its tree hung
    loose. The leaf at the face joins the bed instead."""
    from haemolynx.synthetic import generate as generate_module
    from haemolynx.synthetic.draft import NetworkDraft
    from haemolynx.synthetic.trees import SideResult, Tip

    draft = NetworkDraft()
    root = draft.add_node(np.array([10.0, 10.0, 0.0]), "inlet")
    end = draft.add_node(np.array([0.0, 10.0, 30.0]))
    tip = Tip(node=end, position=np.array([0.0, 10.0, 30.0]), direction=np.array([0.0, 0.0, 1.0]), diameter_um=10.0, generation=1, root=root, at_face=True)
    side = generate_module._Side("arterial", [(root, "x0")], boundary_role="arteriole_boundary", boundary_count=3)
    side.result = SideResult(side="arterial", leaves=[tip], nodes=[end], trees={root: [end]})
    chosen = generate_module._choose_boundary_leaves(side, True, np.random.default_rng(0), draft)
    assert chosen == [end] and side.face == []


def test_a_template_without_capillaries_or_inlets_is_refused():
    T = fixture_template()
    with pytest.raises(ValueError, match="no capillaries"):
        generate_network(replace(T, classes={k: v for k, v in T.classes.items() if k != "capillary"}), seed=1)
    with pytest.raises(ValueError, match="inlets"):
        generate_network(replace(T, inlets=()), seed=1)


@pytest.mark.slow
def test_a_ten_thousand_vessel_template_generates_in_seconds():
    """The plan's bound. Measured 7-10 s here (2026-10-24, three seeds, the
    validation included); it was 11-12 s, half of it the validation, before
    the validation stopped calling np.allclose on every polyline end."""
    T = fixture_template().scaled(3.76)
    start = time.perf_counter()
    G = generate_network(T, seed=0)
    seconds = time.perf_counter() - start
    assert seconds < 30.0
    record = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert G.number_of_edges() > 9000
    assert record["capillaries"] == pytest.approx(T.classes["capillary"].count, rel=0.03)
    assert set(Counter(dict(G.degree()).values())) <= {1, 3, 4, 5}


# -- polylines -----------------------------------------------------------------


def test_bend_to_lengths_hits_each_target_with_the_ends_fixed():
    rng = np.random.default_rng(0)
    starts = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [5.0, 5.0, 5.0]])
    ends = np.array([[0.0, 0.0, 40.0], [10.0, 60.0, 30.0], [5.0, 5.0, 35.0]])
    targets = np.array([48.0, 40.0, 30.0])  # bent, straight (equal), straight (shorter than the chord)
    paths = polylines.bend_to_lengths(starts, ends, targets, rng)
    assert np.allclose(paths[0][0], starts[0]) and np.allclose(paths[0][-1], ends[0])
    assert calculate_path_length(paths[0].tolist()) == pytest.approx(48.0, rel=1e-9)
    assert len(paths[1]) == 2 and len(paths[2]) == 2
    assert calculate_path_length(paths[2].tolist()) == pytest.approx(30.0)


def test_a_bent_polyline_leaves_its_ends_along_the_chord():
    rng = np.random.default_rng(1)
    path = polylines.bend_to_lengths([[0.0, 0.0, 0.0]], [[0.0, 0.0, 60.0]], [66.0], rng, points=33)[0]
    tangent = path[1] - path[0]
    cos = tangent[2] / np.linalg.norm(tangent)
    assert math.degrees(math.acos(min(cos, 1.0))) < 5.0


def test_bend_path_keeps_its_interior_points():
    rng = np.random.default_rng(2)
    via = np.array([10.0, 10.0, 20.0])
    path = polylines.bend_path([np.zeros(3), via, np.array([0.0, 0.0, 40.0])], 60.0, rng)
    assert any(np.allclose(p, via) for p in path)
    assert calculate_path_length(path.tolist()) == pytest.approx(60.0, rel=1e-9)


def test_densify_paths_keeps_lengths_and_vertices():
    paths = [np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 3.0, 10.0]]), np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.2]])]
    voxels, lengths = polylines.densify_paths(paths, step_um=0.5)
    assert lengths == pytest.approx([11.0, 0.2])
    assert calculate_path_length(voxels[0]) == pytest.approx(11.0)
    assert voxels[0][0] == [0.0, 0.0, 0.0] and voxels[0][-1] == [0.0, 3.0, 10.0]
    assert [0.0, 3.0, 4.0] in voxels[0]
    assert max(np.linalg.norm(np.diff(np.asarray(voxels[0]), axis=0), axis=1)) <= 0.5 + 1e-12
    assert voxels[1] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.2]]


# -- trees ---------------------------------------------------------------------


def test_murray_ratios_sum_to_one_and_the_fit_inverts_the_mean_ratio():
    for a in (0.3, 0.7, 1.0):
        ratios = trees.murray_ratios(a, 3.0)
        assert np.sum(ratios**3) == pytest.approx(1.0)
        assert ratios[1] == pytest.approx(a * ratios[0])
        assert trees.fit_asymmetry(trees.mean_child_ratio(a, 3.0), 3.0) == pytest.approx(a, abs=1e-6)
    notes: list = []
    assert trees.fit_asymmetry(0.95, 3.0, notes) == 1.0 and notes
    assert trees.fit_asymmetry(0.1, 3.0) == trees.MIN_ASYMMETRY
    assert trees.fit_asymmetry(None, 3.0) == 0.8


def test_children_per_tip_matches_the_wanted_count():
    assert trees.children_per_tip(4, 8) == [2, 2, 2, 2]
    assert trees.children_per_tip(4, 7) == [3, 2, 2]  # three tips grow, one of them three children
    assert sum(trees.children_per_tip(2, 9)) == 9 and max(trees.children_per_tip(2, 9)) == 5
    notes: list = []
    assert trees.children_per_tip(3, 1, notes, "Art2") == [2] and notes
    assert trees.children_per_tip(0, 5) == []


def test_murray_deflections_bend_the_narrower_daughter_more():
    # Daughters obeying Murray's law (9^3 + 6.47^3 = 10^3): the optimum
    # deflects the narrower one about twice as far.
    shares = trees.murray_deflections(10.0, [9.0, (1000.0 - 729.0) ** (1.0 / 3.0)])
    assert shares.sum() == pytest.approx(1.0)
    assert shares[1] > 1.5 * shares[0]
    assert trees.murray_deflections(10.0, [7.0, 7.0]) == pytest.approx([0.5, 0.5])
    assert trees.murray_deflections(10.0, [8.0, 5.0]) == pytest.approx([0.5, 0.5])  # not Murray's: no optimum


def test_exit_fraction_and_face_snapping():
    extent = np.array([100.0, 200.0, 300.0])
    assert trees.exit_fraction(np.array([50.0, 50.0, 50.0]), np.array([50.0, 50.0, 80.0]), extent) == 1.0
    assert trees.exit_fraction(np.array([50.0, 50.0, 290.0]), np.array([50.0, 50.0, 310.0]), extent) == pytest.approx(0.5)
    assert trees.exit_fraction(np.array([10.0, 50.0, 50.0]), np.array([-10.0, 50.0, 50.0]), extent) == pytest.approx(0.5)
    snapped = trees.snap_to_face(np.array([50.0, 50.0, 299.9999]), extent)
    assert snapped[2] == 300.0
    assert trees.inward_normal("x0").tolist() == [0.0, 0.0, 1.0]
    assert trees.inward_normal("z1").tolist() == [-1.0, 0.0, 0.0]
    assert trees.inward_normal("interior") is None


# -- the bed's arithmetic ------------------------------------------------------


def test_degree_sequence_targets_satisfy_the_handshake_lemma():
    targets = bed_module.degree_sequence_targets(
        capillary_count=160, loops=51, root_count=4, attached_caps=[2] * 10, dead_ends=12
    )
    assert targets.vertices == 160 - 51 - 4 + 1
    assert targets.degree_counts == {1: 12, 3: 88, 4: 6}
    assert sum(d * n for d, n in targets.degree_counts.items()) == 2 * 160 - 20
    assert targets.loops_shortfall == 0 and targets.notes == ()


def test_degree_sequence_targets_give_on_loops_not_dead_ends():
    targets = bed_module.degree_sequence_targets(
        capillary_count=160, loops=40, root_count=4, attached_caps=[2] * 10, dead_ends=12
    )
    assert targets.dead_ends == 12
    assert targets.vertices < 160 - 40 - 4 + 1
    assert any("loops are drawn" in n for n in targets.notes)
    assert sum(d * n for d, n in targets.degree_counts.items()) == 2 * 160 - 20


def test_degree_sequence_targets_cap_the_junction_degree():
    targets = bed_module.degree_sequence_targets(
        capillary_count=200, loops=150, root_count=2, attached_caps=[], dead_ends=0, max_degree=4
    )
    assert max(targets.degree_counts) <= 4
    assert targets.loops_shortfall > 0


def _star_beside_a_dead_end():
    """The largest piece: a junction (0) with four dead ends, one of them (4)
    nearest the other piece. The other piece: a junction (5) whose three
    vessels all end blind -- what the fill left on E14.5. Every vertex is
    full, and the star's junction has a dead end, not a junction, nearest."""
    points = np.array(
        [
            [0.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, -10.0, 0.0], [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0], [18.0, 0.0, 0.0], [14.0, 3.0, 0.0], [14.0, -3.0, 0.0],
        ]
    )
    caps = np.array([4, 1, 1, 1, 1, 3, 1, 1, 1])
    fill = bed_module._Fill(points, caps, np.zeros(len(points), bool), np.random.default_rng(0), 75.0, 10.0)
    for i, j in ((0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8)):
        fill.connect(i, j)
    return fill


def test_a_piece_whose_junctions_have_dead_ends_nearest_is_still_joined():
    """Regression (E14.5 as a template, 2026-10-24): the repair looked only at
    each vertex's nearest vertex of the largest piece; a dead end there, and
    no spare vessel at the star's junction, left the star apart and the
    network failed its contract."""
    fill = _star_beside_a_dead_end()
    edges_before = len(fill.edges)
    repairs, joined = bed_module.join_pieces(fill, range(9))
    assert (repairs, joined) == (1, True)
    assert fill.edges[edges_before:] == [(0, 5)]  # junction to junction, past dead end 4
    G = nx.Graph(fill.edges)
    assert nx.is_connected(G)
    # One edge for one piece: the loop count (E - N + pieces) is unchanged.
    assert G.number_of_edges() - G.number_of_nodes() + 1 == edges_before - 9 + 2


def test_a_join_never_runs_through_a_dead_end():
    fill = _star_beside_a_dead_end()
    bed_module.join_pieces(fill, range(9))
    for end in (1, 2, 3, 4, 6, 7, 8):
        assert fill.degree[end] == 1


# -- the bed laid out to the template's organisation (Harvey 2026-10-25: D4, D6)


def _organisation(shape, nodes, orientation=ISOTROPIC, arterial=None, venous=None):
    """A hand-set layout: capillary vertices by cell, one direction
    everywhere, the trees' length by cell (none unless given)."""
    n = int(np.prod(shape))
    return Organisation(
        shape=tuple(shape),
        capillary_nodes=tuple(float(v) for v in nodes),
        capillary_length_um=tuple(float(v) for v in nodes),
        arterial_length_um=tuple(float(v) for v in (arterial or (0.0,) * n)),
        venous_length_um=tuple(float(v) for v in (venous or (0.0,) * n)),
        orientation=(tuple(orientation),) * n,
        global_orientation=tuple(orientation),
        axis_share=(1 / 3, 1 / 3, 1 / 3),
        within_near=0.5,
        beyond_far=0.1,
        trunk_turn_deg=None,
        capillary_curvature_deg=None,
        tree_curvature_deg=None,
    )


def _lower_layer(shape):
    """Every capillary vertex in the lowest layer of cells in z."""
    n = int(np.prod(shape))
    return [1.0 if i < n // shape[0] else 0.0 for i in range(n)]


def _capillaries(G):
    return [d for *_e, d in G.edges(data=True) if d[SYNTHETIC_CLASS] == "capillary"]


def _axis_shares(G):
    """Share of capillary length within 30 degrees of z, y and x."""
    along, total = np.zeros(3), 0.0
    for d in _capillaries(G):
        steps = np.diff(np.asarray(d["voxels"], dtype=float), axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        ok = lengths > 0
        units = steps[ok] / lengths[ok, None]
        along += (np.abs(units) >= math.cos(math.radians(30.0))).T @ lengths[ok]
        total += float(lengths[ok].sum())
    return along / total


def test_cells_share_the_vertices_by_largest_remainder():
    assert bed_module._allocate(10, [1.0, 1.0, 1.0]).tolist() == [4, 3, 3]
    assert bed_module._allocate(7, [3.0, 0.0, 1.0]).tolist() == [5, 0, 2]
    assert bed_module._allocate(4, [0.0, 0.0]).tolist() == [2, 2]  # nothing to go by: even


def test_density_vertices_follow_the_templates_cells():
    extent = np.array([100.0, 100.0, 100.0])
    rng = np.random.default_rng(0)
    org = _organisation((2, 2, 2), _lower_layer((2, 2, 2)))
    points = bed_module.density_vertices(extent, 200, rng=rng, keep_clear=None, spacing_um=10.0, organisation=org)
    assert len(points) == 200
    assert np.all(points[:, 0] < 50.0)  # none where the template has none
    cells = Counter(cell_index(points, extent, (2, 2, 2)).tolist())
    assert sorted(cells) == [0, 1, 2, 3] and min(cells.values()) >= 40
    # Three times the vertices below as above: three quarters of them below.
    org = _organisation((2, 1, 1), [3.0, 1.0])
    points = bed_module.density_vertices(extent, 400, rng=rng, keep_clear=None, spacing_um=10.0, organisation=org)
    assert np.mean(points[:, 0] < 50.0) == pytest.approx(0.75, abs=0.03)


def test_local_spacing_is_the_cells_own():
    extent = np.array([100.0, 100.0, 100.0])
    org = _organisation((1, 1, 2), [1.0, 1.0])
    points = np.array([[50.0, 50.0, 10.0]] * 8 + [[50.0, 50.0, 90.0]])
    spacing = bed_module.local_spacings(points, extent, org, 10.0)
    assert spacing[:8] == pytest.approx([(5.0e5 / 8) ** (1 / 3)] * 8)
    assert spacing[8] == pytest.approx(5.0e5 ** (1 / 3))
    assert bed_module.local_spacings(points, extent, None, 10.0).tolist() == [10.0] * 9


def test_a_template_without_an_organisation_gets_an_even_bed():
    """The hand-set fixtures carry no organisation: the bed is a lattice over
    the whole box, its spacing the box over the bed's vertices."""
    T = fixture_template()
    G = generate_network(T, seed=SEED)
    generated = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert generated["bed_layout"] == "even"
    # Capillaries less loops less inlets and outlets, plus one; each pair of
    # dead ends joined is a capillary fewer and a loop more.
    joined = generated["dead_end_pairs_joined"]
    vertices = (T.classes["capillary"].count - joined) - (T.loops.independent_loops + joined) - 4 + 1
    assert generated["spacing_um"] == pytest.approx((120.0 * 200.0 * 200.0 / vertices) ** (1 / 3))


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_a_bed_laid_out_to_a_layered_template_stays_in_its_layer(seed):
    T = fixture_template()
    shape = grid_shape(T.extent_um)
    G = generate_network(replace(T, organisation=_organisation(shape, _lower_layer(shape))), seed=seed)
    # A hand-set layout records no vertices of its own: the cell field.
    assert G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["bed_layout"] == "template cells"
    capillaries = _capillaries(G)
    total = sum(d["length"] for d in capillaries)
    below = sum(d["length"] * np.mean(np.asarray(d["voxels"])[:, 0] < 60.0) for d in capillaries)
    # The few vessels joining the trees above to the layer run out of it.
    assert below / total > 0.85
    validate_synthetic_graph(G)


def test_a_bed_runs_the_way_the_templates_capillaries_run():
    T = fixture_template()
    shape = grid_shape(T.extent_um)
    even = [1.0] * int(np.prod(shape))
    shares = {
        name: np.mean([
            _axis_shares(generate_network(replace(T, organisation=_organisation(shape, even, orient)), seed=seed))
            for seed in (7, 8, 9)
        ], axis=0)
        for name, orient in (("isotropic", ISOTROPIC), ("z", (1.0, 0, 0, 0, 0, 0)), ("x", (0, 0, 1.0, 0, 0, 0)))
    }
    # An isotropic network has 13.4% of its length within 30 degrees of each axis.
    assert shares["isotropic"] == pytest.approx([0.134] * 3, abs=0.04)
    assert shares["z"][0] > shares["isotropic"][0] + 0.1
    assert shares["x"][2] > shares["isotropic"][2] + 0.1
    assert shares["z"][2] < shares["isotropic"][2] and shares["x"][0] < shares["isotropic"][0]


def test_a_bed_in_a_box_too_big_for_it_keeps_the_templates_capillary_lengths_where_they_are():
    """The density the template's capillaries have where they are, not the
    box's: spread evenly over a box four times their own (decision 3's case),
    the fixture's capillaries come out 40% longer than its 50 um; laid out to
    a template that holds them in one layer, at its length."""
    T = replace(fixture_template(), extent_um=(240.0, 400.0, 400.0), image_shape=(241, 409, 409))
    shape = grid_shape(T.extent_um)
    laid_out = generate_network(replace(T, organisation=_organisation(shape, _lower_layer(shape))), seed=SEED)
    lengths = [d["length"] for d in _capillaries(laid_out)]
    assert np.mean(lengths) == pytest.approx(50.0, rel=0.10)
    assert sum(lengths) == pytest.approx(160 * 50.0, rel=0.10)
    even = [d["length"] for d in _capillaries(generate_network(T, seed=SEED))]
    assert np.mean(even) > 1.25 * 50.0


def test_a_voronoi_bed_follows_the_templates_layers():
    T = fixture_template()
    shape = grid_shape(T.extent_um)
    G = generate_network(
        replace(T, organisation=_organisation(shape, _lower_layer(shape))), seed=SEED, options=GenerationOptions(bed_model="voronoi")
    )
    capillaries = _capillaries(G)
    total = sum(d["length"] for d in capillaries)
    below = sum(d["length"] * np.mean(np.asarray(d["voxels"])[:, 0] < 60.0) for d in capillaries)
    assert below / total > 0.75
    validate_synthetic_graph(G)


def test_depths_like_template_moves_points_rank_for_rank_onto_the_templates():
    z = np.array([50.0, 10.0, 90.0, 30.0, 70.0])
    template = [0.05, 0.1, 0.15, 0.2, 0.25]  # every junction in the bottom quarter
    out = bed_module.depths_like_template(z, template, 100.0, 2.0)
    assert list(np.argsort(out)) == list(np.argsort(z))  # the order in depth is kept
    assert out.tolist() == pytest.approx([15.0, 5.0, 25.0, 10.0, 20.0])
    # Kept off the faces.
    assert bed_module.depths_like_template(z, [0.0] * 5, 100.0, 2.0).min() == pytest.approx(2.0)


def test_a_voronoi_beds_corners_take_the_templates_junction_depths():
    """Harvey, grid D1e: the template's junctions lie in its plexus; a
    tessellation puts its corners above and below a sheet of seeds."""
    T = fixture_template()
    shape = grid_shape(T.extent_um)
    plexus = tuple(float(v) for v in np.linspace(0.02, 0.18, 200))
    organisation = replace(_organisation(shape, _lower_layer(shape)), capillary_junction_depths_rel=plexus)
    shares = {}
    for on in (True, False):
        bed_module.VORONOI_DEPTHS_FROM_TEMPLATE = on
        try:
            G = generate_network(replace(T, organisation=organisation), seed=SEED, options=GenerationOptions(bed_model="voronoi"))
        finally:
            bed_module.VORONOI_DEPTHS_FROM_TEMPLATE = True
        bed_junctions = [
            n for n in G if G.degree(n) >= 3
            and all(d.get(SYNTHETIC_CLASS) == "capillary" for *_e, d in G.edges(n, data=True))
        ]
        z = np.array([G.nodes[n]["pos"][0] for n in bed_junctions])
        shares[on] = float(np.mean(z < 0.2 * T.extent_um[0]))
        if on:
            validate_synthetic_graph(G)
    assert shares[True] > 0.85
    assert shares[True] > shares[False] + 0.2


# -- the trees grown in the template's tree layer (Harvey 2026-10-25: D5a) ------


def test_a_tree_layer_reads_each_sides_tree_cells():
    extent = (100.0, 100.0, 100.0)
    arterial = [10.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    org = _organisation((2, 2, 2), [1.0] * 8, arterial=arterial)
    T = replace(fixture_template(), extent_um=extent, organisation=org)
    layer = trees.TreeLayer.of(T, "arterial")
    # Over the median of the cells holding any (25 um), capped at 1; the
    # cells round them a share of their best neighbour.
    assert layer.value.ravel().tolist() == pytest.approx([0.4, 1.0] + [trees.LAYER_NEIGHBOUR_SHARE] * 6)
    assert layer.at(np.array([25.0, 25.0, 75.0])) == pytest.approx(1.0)  # cell 1's centre
    assert layer.at(np.array([25.0, 25.0, 25.0])) == pytest.approx(0.4)
    assert layer.at(np.array([25.0, 25.0, 50.0])) == pytest.approx(0.7)  # between them
    # A side the template has no tree on reads both sides'.
    assert trees.TreeLayer.of(T, "venous").value.ravel().tolist() == layer.value.ravel().tolist()
    assert trees.TreeLayer.of(replace(T, organisation=_organisation((2, 2, 2), [1.0] * 8)), "arterial") is None
    assert trees.TreeLayer.of(fixture_template(), "arterial") is None


def test_room_ahead_is_the_distance_to_the_face_along_the_heading():
    extent = np.array([100.0, 100.0, 100.0])
    point = np.array([10.0, 20.0, 30.0])
    assert trees.room_ahead_um(point, np.array([0.0, 0.0, 1.0]), extent) == pytest.approx(70.0)
    assert trees.room_ahead_um(point, np.array([-1.0, 0.0, 0.0]), extent) == pytest.approx(10.0)
    diagonal = np.array([-1.0, -1.0, 0.0]) / math.sqrt(2.0)
    assert trees.room_ahead_um(point, diagonal, extent) == pytest.approx(10.0 * math.sqrt(2.0))


def _tree_share_below(G, z_um):
    inside = total = 0.0
    for *_e, d in G.edges(data=True):
        if d[SYNTHETIC_CLASS] in ("arteriole", "venule", "large_arteriole", "large_venule"):
            path = np.asarray(d["voxels"], dtype=float)
            lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            inside += float(lengths[(path[1:, 0] + path[:-1, 0]) / 2 < z_um].sum())
            total += float(lengths.sum())
    return inside / total


def _layered_trees_template():
    T = fixture_template()
    shape = grid_shape(T.extent_um)
    lower = _lower_layer(shape)
    return replace(T, organisation=_organisation(shape, [1.0] * len(lower), arterial=lower, venous=lower))


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_trees_grow_in_the_templates_tree_layer(seed):
    """The fixture's inlets and outlets sit at 40% and 60% of its depth: a
    template whose trees lie in its lower half gets trees there."""
    layered = generate_network(_layered_trees_template(), seed=seed)
    assert _tree_share_below(layered, 60.0) > 0.7
    validate_synthetic_graph(layered)


def test_trees_without_a_layer_fall_anywhere():
    """One with no layout keeps them off the faces wherever they fall:
    about half below (0.43 to 0.63 over seeds 1 to 8 on 2026-10-25)."""
    seeds = (7, 8, 9)
    layered = np.mean([_tree_share_below(generate_network(_layered_trees_template(), seed=s), 60.0) for s in seeds])
    anywhere = np.mean([_tree_share_below(generate_network(fixture_template(), seed=s), 60.0) for s in seeds])
    assert layered > anywhere + 0.25


# -- the bed drawn from the template's own vertices; curves (2026-10-25) --------


def _measured_template(seed=SEED):
    """A template measured off a generated network: it records its
    capillary vertices and their spacing, as a run's does."""
    G = generate_network(fixture_template(), seed=seed, options=FAITHFUL)
    T = _measured(G)
    return replace(T, extent_um=fixture_template().extent_um)


def test_template_vertices_are_the_templates_moved_by_their_spacing():
    T = _measured_template()
    org = T.organisation
    assert org.capillary_node_spacing_um > 0
    extent = np.asarray(T.extent_um)
    source = np.asarray(org.capillary_node_positions_rel) * extent
    points = bed_module.template_vertices(
        extent, 80, rng=np.random.default_rng(0), keep_clear=None, spacing_um=20.0, organisation=org
    )
    assert len(points) == 80 and np.all(points > 0) and np.all(points < extent)
    nearest, _ = cKDTree(source).query(points)
    # A normal step of the spacing over three axes: most within twice it.
    assert np.mean(nearest < 2.0 * org.capillary_node_spacing_um) > 0.9
    assert bed_module.drawn_from_template_vertices(org, True)
    assert not bed_module.drawn_from_template_vertices(org, False)  # a scaled box
    assert not bed_module.drawn_from_template_vertices(replace(org, capillary_node_positions_rel=None), True)


def test_a_bed_at_the_templates_size_is_drawn_from_its_vertices_a_scaled_one_from_its_cells():
    T = _measured_template()
    G = generate_network(T, seed=SEED)
    assert G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["bed_layout"] == "template vertices"
    # The record keeps the layout's fields, not the template's vertices.
    assert G.graph[GRAPH_TEMPLATE_RECORD]["template"]["organisation"]["capillary_node_positions_rel"] is None
    scaled = generate_network(T, seed=SEED, options=GenerationOptions(scale=1.3))
    assert scaled.graph[GRAPH_TEMPLATE_RECORD]["generated"]["bed_layout"] == "template cells"
    validate_synthetic_graph(G)
    validate_synthetic_graph(scaled)


def test_one_lobe_is_the_bow_and_more_lobes_curve_more_at_one_length():
    s = np.linspace(0.0, 1.0, 33)
    assert polylines.lobes(s, 1) == pytest.approx(np.sin(np.pi * s) ** 2)
    for k in (2, 3, 4):
        lobe = polylines.lobes(s, k)
        assert lobe[0] == pytest.approx(0.0) and lobe[-1] == pytest.approx(0.0, abs=1e-12)
    curvature = {}
    for target in (float("nan"), 26.0):
        path = polylines.bend_to_lengths(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 40.0]], [42.0], np.random.default_rng(0), curvature_deg=np.array([target])
        )[0]
        assert polylines._lengths(path[None])[0] == pytest.approx(42.0, rel=1e-6)
        curvature[str(target)] = polylines._path_curvature(path)
    # One bow of tortuosity 1.05 turns about 20 degrees per 10 um; asked
    # for 26, the vessel takes more lobes and comes nearer.
    assert abs(curvature["26.0"] - 26.0) < abs(curvature["nan"] - 26.0)
    assert curvature["26.0"] == pytest.approx(26.0, abs=3.0)


def test_each_direction_is_judged_against_its_own_chord():
    """E14.5's vertical capillaries reach past nearer junctions (1.4 times
    the mean chord): judged against one wanted length the fill never drew
    them."""
    org = replace(_organisation((1, 1, 1), [1.0]), chord_by_class_um=(40.0, None, 20.0, 30.0))
    ratios = bed_module.class_chord_ratios(org)
    assert ratios.tolist() == pytest.approx([4 / 3, 1.0, 2 / 3, 1.0])
    assert bed_module.class_chord_ratios(_organisation((1, 1, 1), [1.0])) is None
    points = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
    for chords, expected in ((ratios, (0.0, 0.0)), (None, (10.0, 10.0))):
        fill = bed_module._Fill(
            points, np.array([3, 1, 1]), np.zeros(3, dtype=bool), np.random.default_rng(0), 90.0, 30.0,
            target_chord_um=30.0, class_chords=chords,
        )
        vertical, across = fill.score(0, 1, 40.0), fill.score(0, 2, 20.0)
        assert vertical == pytest.approx(expected[0], abs=2.0 + 1e-9) and across == pytest.approx(expected[1], abs=2.0 + 1e-9)
        if chords is not None:
            assert vertical == 0.0 and across == 0.0


def _columns_template():
    """The fixture with the layout of capillary columns rising from a plexus:
    at every (y, x) of a 40 um grid, vessels from z 5 to 60 and 60 to 115,
    the plexus joining neighbours at z 5."""
    from haemolynx.synthetic.organisation import measure_organisation

    T = fixture_template()
    edges = []
    for y in np.arange(20.0, 200.0, 40.0):
        for x in np.arange(20.0, 200.0, 40.0):
            edges.append(((5.0, y, x), (60.0, y, x)))
            edges.append(((60.0, y, x), (115.0, y, x)))
            if x + 40.0 < 200.0:
                edges.append(((5.0, y, x), (5.0, y, x + 40.0)))
            if y + 40.0 < 200.0:
                edges.append(((5.0, y, x), (5.0, y + 40.0, x)))
    G = nx.MultiGraph()
    index: dict = {}
    for a, b in edges:
        for p in (a, b):
            if p not in index:
                index[p] = len(index)
                G.add_node(index[p], pos=np.asarray(p))
        G.add_edge(index[a], index[b], voxels=[list(a), list(b)], branch_order="B01")
    return replace(T, organisation=measure_organisation(G, T.extent_um))


def test_a_template_with_columns_gets_columns():
    from haemolynx.synthetic.organisation import measure_organisation

    columns = _columns_template()
    assert columns.organisation.long_runs[0] == 25 and columns.organisation.chord_axis_share[0] > 0.6
    drawn = {"columns": [], "none": []}
    for seed in (7, 8, 9):
        for name, T in (("columns", columns), ("none", fixture_template())):
            G = generate_network(T, seed=seed, options=FAITHFUL)
            _labelled(G)
            drawn[name].append(measure_organisation(G, T.extent_um))
    share = {name: np.mean([o.chord_axis_share[0] for o in orgs]) for name, orgs in drawn.items()}
    runs = {name: sum(o.long_runs[0] for o in orgs) for name, orgs in drawn.items()}
    assert share["columns"] > share["none"] + 0.12
    assert runs["columns"] >= runs["none"] + 4


def test_capillaries_keep_the_templates_tortuosity_within_its_length():
    """Lengths from the length distribution were clamped to their chords
    where they fell short, and E14.5's capillaries came out straight (grid
    D1b): each is now its chord times a tortuosity drawn from the template,
    the excess scaled alike so the class total is the template's."""
    from haemolynx.synthetic.generate import MAX_TORTUOSITY_SCALE, _capillary_lengths_by_tortuosity
    from haemolynx.synthetic.template import Samples

    def template(total):
        T = fixture_template()
        cls = replace(T.classes["capillary"], total_length_um=total, samples=Samples(tortuosities=(1.2,)))
        return replace(T, classes={**T.classes, "capillary": cls})

    chords = np.full(4, 30.0)
    none = np.zeros(4, dtype=bool)
    for total, expected in ((144.0, 36.0), (130.0, 32.5)):
        notes: list = []
        lengths = _capillary_lengths_by_tortuosity(template(total), chords, none, None, np.random.default_rng(0), notes)
        assert lengths == pytest.approx([expected] * 4) and not notes  # every vessel curves alike
    notes = []
    lengths = _capillary_lengths_by_tortuosity(template(110.0), chords, none, None, np.random.default_rng(0), notes)
    assert lengths == pytest.approx([30.0] * 4) and "drawn straight" in notes[0]
    notes = []
    lengths = _capillary_lengths_by_tortuosity(template(400.0), chords, none, None, np.random.default_rng(0), notes)
    assert lengths == pytest.approx([30.0 * (1 + 0.2 * MAX_TORTUOSITY_SCALE)] * 4) and "times the template's tortuosity" in notes[0]
    # A blind chain keeps the length drawn for it; the rest share what is left.
    blind = np.array([True, False, False, False])
    lengths = _capillary_lengths_by_tortuosity(
        template(144.0), chords, blind, np.array([40.0, np.nan, np.nan, np.nan]), np.random.default_rng(0), []
    )
    assert lengths[0] == 40.0 and lengths[1:] == pytest.approx([(144.0 - 40.0) / 3] * 3)


def test_the_trees_end_where_the_templates_do():
    """Harvey, grid D1b: the template's trees have side branches, ending
    where they hand over to capillaries; the drawn trunks handed every side
    branch to the bed. Each order's vessels now end the tree as often as the
    template's do."""
    from haemolynx.synthetic.organisation import measure_organisation

    T = _measured_template()
    assert T.organisation.tree_leaves_by_order
    for seed in (7, 8):
        G = generate_network(T, seed=seed, options=FAITHFUL)
        roles = _labelled(G)
        S = measure_organisation(G, T.extent_um, roots=roles["inlet"] + roles["outlet"])
        assert S.tree_leaves_by_order == T.organisation.tree_leaves_by_order
    # Asked for one more Art2 leaf, one more Art2 tip ends there.
    asked = dict(T.organisation.tree_leaves_by_order)
    asked["Art2"] = asked.get("Art2", 0) + 1
    more = replace(T, organisation=replace(T.organisation, tree_leaves_by_order=tuple(sorted(asked.items()))))
    G = generate_network(more, seed=7, options=FAITHFUL)
    roles = _labelled(G)
    drawn = dict(measure_organisation(G, T.extent_um, roots=roles["inlet"] + roles["outlet"]).tree_leaves_by_order)
    assert drawn["Art2"] == asked["Art2"]


def test_the_trees_are_grown_to_the_templates_reach():
    """Harvey's grid D1c picked the drawn networks whose venous trees reach
    across the box as the template's do: of several trees grown per side,
    the one whose leaves' median and furthest reach miss the template's
    least is kept, and the others leave nothing behind."""
    import copy

    from haemolynx.synthetic import generate as generate_module
    from haemolynx.synthetic.draft import NetworkDraft

    T = _measured_template()
    reach = dict((s, (m, f)) for s, m, f in T.organisation.tree_reach_um)
    assert set(reach) == {"arterial", "venous"}
    extent = np.asarray(T.extent_um)
    draft = NetworkDraft()
    roots = generate_module._place_roots(draft, T, extent, np.random.default_rng(0))
    before = copy.deepcopy(draft)
    kwargs = dict(spacing_um=40.0, extent_um=extent, min_separation_um=20.0)
    kept = generate_module._grow_side_to_reach(draft, T, "venous", roots["outlet"], np.random.default_rng(5), **kwargs)
    median, far = reach["venous"]
    misses = []
    for seed in np.random.default_rng(5).integers(0, 2**62, size=generate_module.TREE_CANDIDATES):
        trial = copy.deepcopy(before)
        result = trees.grow_side(trial, T, "venous", roots["outlet"], rng=np.random.default_rng(int(seed)), **kwargs)
        chords = generate_module._leaf_reach(trial, result)
        misses.append((abs(np.median(chords) - median) / median + abs(max(chords) - far) / far, len(trial.positions)))
    best = min(misses, key=lambda m: m[0])
    chords = generate_module._leaf_reach(draft, kept)
    assert abs(np.median(chords) - median) / median + abs(max(chords) - far) / far == pytest.approx(best[0])
    assert len(draft.positions) == best[1]  # the candidates grown and rolled back left nothing
    assert len(draft.edges) == sum(1 for e in draft.edges if e.part == "tree")


def test_a_trunk_keeps_to_its_trees_course():
    """At a junction the plane is chosen so the trunk (the least deflected
    child) turns towards the tree's course from its root."""
    extent = np.array([200.0, 200.0, 200.0])
    tip = trees.Tip(node=0, position=np.array([100.0, 100.0, 100.0]), direction=np.array([0.0, 0.0, 1.0]),
                    diameter_um=8.0, generation=2)
    occupancy = trees._Occupancy([tip.position])
    course = np.array([0.0, 1.0, 0.0])
    deflections = [0.3, 0.9]
    children = trees._place_children(
        tip, [20.0, 20.0], deflections, extent, occupancy, 30.0, 5.0, np.random.default_rng(0), None, False,
        course, 10.0,
    )
    trunk_direction = children[0][1]
    # The best of the eight planes leans the trunk at most sin(0.3) towards the course.
    assert float(trunk_direction @ course) > 0.9 * math.sin(0.3)


def test_curvature_targets_leave_a_template_without_a_layout_alone():
    """No organisation, no target: the polylines are the bows they were."""
    G = generate_network(fixture_template(), seed=SEED, options=FAITHFUL)
    assert all(len(d["voxels"]) >= 2 for *_e, d in G.edges(data=True))
    record = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert record["bed_layout"] == "even"


# -- the Voronoi bed (bed_model="voronoi", Harvey 2026-10-24) -----------------


def test_circumcentres_are_equidistant_and_a_flat_tetrahedron_has_none():
    tetrahedra = np.array([
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])
    centres = bed_module._circumcentres(tetrahedra)
    assert centres[0] == pytest.approx([2.0, 3.0, 4.0])
    distances = np.linalg.norm(tetrahedra[0] - centres[0], axis=1)
    assert distances == pytest.approx([distances[0]] * 4)
    assert np.isnan(centres[1]).all()


def test_voronoi_vertices_are_corners_joined_along_cell_edges():
    rng = np.random.default_rng(3)
    extent = np.array([200.0, 200.0, 200.0])
    points, edges = bed_module.voronoi_vertices(extent, 300, rng=rng, keep_clear=None, spacing_um=200.0 / 300 ** (1 / 3))
    assert len(points) == 300
    assert np.all(points > 0) and np.all(points < extent)
    assert all(0 <= a < b < 300 for a, b in edges) and len(set(edges)) == len(edges)
    degree = np.bincount(np.array(edges).ravel(), minlength=300)
    # A tessellation's corners meet four cell edges; contracting its
    # shortest edges merges a few into corners of five or six.
    assert np.median(degree) == 4
    assert np.mean(degree) == pytest.approx(4.0, abs=0.6)


def test_a_voronoi_bed_keeps_the_templates_counts_along_cell_edges():
    T = fixture_template()
    voronoi = replace(FAITHFUL, bed_model="voronoi")
    G = generate_network(T, seed=SEED, options=voronoi)
    record = G.graph[GRAPH_TEMPLATE_RECORD]
    generated = record["generated"]
    assert record["options"]["bed_model"] == "voronoi"
    assert generated["capillaries"] == pytest.approx(160, rel=0.03)
    assert generated["loops"] == pytest.approx(51, abs=2)
    assert generated["bed_face_ends"] == generated["bed_face_ends_wanted"]
    assert generated["bed_blind_ends"] == generated["bed_blind_ends_wanted"]
    # Most of the bed's own vessels run along the tessellation's cell edges.
    assert generated["fill_phases"]["voronoi_edges"] >= 0.6 * generated["fill_phases"]["bed_only_edges"]
    again = generate_network(T, seed=SEED, options=voronoi)
    assert sorted(again.edges(keys=True)) == sorted(G.edges(keys=True))
    delaunay = generate_network(T, seed=SEED, options=FAITHFUL)
    assert "voronoi_edges" not in delaunay.graph[GRAPH_TEMPLATE_RECORD]["generated"]["fill_phases"]


def test_an_unknown_bed_model_is_refused():
    with pytest.raises(ValueError, match="bed model"):
        generate_network(fixture_template(), seed=SEED, options=GenerationOptions(bed_model="lattice"))


def test_the_schemas_bed_models_are_the_generators():
    from haemolynx.pipeline import default_schema

    assert default_schema()["synthetic_bed_model"].choices == bed_module.BED_MODELS
    assert default_schema()["synthetic_bed_model"].default == bed_module.BED_MODEL_DELAUNAY


def test_lattice_vertices_fill_the_box_and_keep_clear():
    rng = np.random.default_rng(0)
    extent = np.array([50.0, 100.0, 100.0])
    clear = np.array([[25.0, 50.0, 50.0]])
    points = bed_module.lattice_vertices(extent, 80, rng=rng, keep_clear=clear, spacing_um=20.0)
    assert len(points) <= 80
    assert np.all(points >= 0) and np.all(points <= extent)
    assert np.min(np.linalg.norm(points - clear, axis=1)) >= 0.35 * 20.0


def test_candidate_edges_are_delaunay_edges_with_lengths():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(30, 3))
    edges = bed_module.candidate_edges(points)
    assert edges and all(i < j for i, j, _l in edges)
    for i, j, length in edges[:10]:
        assert length == pytest.approx(np.linalg.norm(points[i] - points[j]))
    few = bed_module.candidate_edges(points[:3])
    assert sorted((i, j) for i, j, _l in few) == [(0, 1), (0, 2), (1, 2)]
