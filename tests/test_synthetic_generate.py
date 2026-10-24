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
from haemolynx.synthetic.generate import (
    DEFAULT_SEED,
    SYNTHETIC,
    SYNTHETIC_CLASS,
    SYNTHETIC_ROLES,
    GenerationOptions,
    boundary_nodes,
    generate_network,
    validate_synthetic_graph,
)
from haemolynx.synthetic.template import (
    GRAPH_NETWORK_SOURCE,
    GRAPH_TEMPLATE_RECORD,
    ROLES,
    OrderTarget,
    measure_template,
)
from synthetic_fixtures import fixture_template

SEED = 7


@pytest.fixture(scope="module")
def generated():
    return generate_network(fixture_template(), seed=SEED)


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


def test_roles_and_endpoints_are_the_templates(generated):
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


def test_capillary_count_loops_and_dead_ends_are_the_templates(generated):
    T = fixture_template()
    record = generated.graph[GRAPH_TEMPLATE_RECORD]["generated"]
    assert abs(record["capillaries"] - T.classes["capillary"].count) <= 3
    loops = generated.number_of_edges() - generated.number_of_nodes() + 1
    assert abs(loops - T.loops.independent_loops) <= 2
    kinds = Counter(generated.nodes[n][DEAD_END_KIND] for n in generated if generated.degree(n) == 1)
    assert abs(kinds[LEAVES_IMAGE] - T.dead_ends.leaves_image) <= 2
    assert abs(kinds[BLIND] - T.dead_ends.interior) <= 2
    assert set(Counter(dict(generated.degree()).values())) <= {1, 3, 4}


def test_without_interior_dead_ends_every_dead_end_is_on_a_face():
    G = generate_network(fixture_template(), seed=SEED, options=GenerationOptions(interior_dead_ends=False))
    kinds = Counter(G.nodes[n][DEAD_END_KIND] for n in G if G.degree(n) == 1)
    assert kinds[BLIND] == 0
    assert kinds[LEAVES_IMAGE] >= 4


def test_scale_grows_the_box_and_the_counts():
    T = fixture_template()
    G = generate_network(T, seed=SEED, options=GenerationOptions(scale=1.5))
    assert np.allclose(G.graph["image_extent_um"], np.asarray(T.extent_um) * 1.5)
    capillaries = G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["capillaries"]
    assert capillaries == pytest.approx(T.classes["capillary"].count * 1.5**3, rel=0.05)
    assert len(boundary_nodes(G)["inlet"]) == max(1, round(2 * 1.5**2))


# -- the acceptance test: every banded characteristic within tolerance -------


def test_the_generated_network_measures_as_its_template(generated):
    rows = compare_templates(fixture_template(), _measured(generated.copy()))
    summary = comparison_summary(rows)
    assert summary["fitted"] >= 50
    outside = [r for r in rows if r.within is False]
    # One characteristic may miss at a single seed (a class of six vessels'
    # median, say); the generator is not allowed to miss anything exact.
    assert all(r.tolerance != "exact" for r in outside), [(r.section, r.characteristic) for r in outside]
    assert len(outside) <= 1, [(r.section, r.characteristic, r.template, r.synthetic) for r in outside]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_rows_hold_at_every_seed(seed):
    G = generate_network(fixture_template(), seed=seed)
    rows = compare_templates(fixture_template(), _measured(G))
    exact = [r for r in rows if r.tolerance == "exact"]
    assert exact and all(r.within is not False for r in exact), [(r.section, r.characteristic) for r in exact if r.within is False]


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
    G = generate_network(T, seed=SEED)
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
