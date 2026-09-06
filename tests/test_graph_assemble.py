"""Tests for graph.assemble.build_graph_from_skeleton, the topology orchestrator.

`build_graph_from_skeleton` runs eleven topology steps in a fixed order and is
the only caller of most of them. The things that can go wrong at this level are
orchestration mistakes rather than algorithm mistakes: a step dropped or run out
of order, a threshold routed to the wrong step, or a `step_callback` label that
no longer matches what the pipeline keys its per-step artifacts on.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("skan")

from haemolynx.graph import STEP_LABELS, build_graph_from_skeleton
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes

# Coarse z, fine x, all three distinct so a (z, y, x) / (x, y, z) swap shows up.
VOXEL_SIZE_ZYX = (2.0, 0.5, 0.4)

#: Emitted in this order after each topology step. `haemolynx.pipeline.stages`
#: special-cases "..._pass1" by name when it writes per-step plots, so these are
#: a contract, not an implementation detail.
EXPECTED_STEP_LABELS = [
    "build_graph_segment_skan_stitched_loops",
    "reconnect_secondary_loop_edges",
    "optimise_graph_topology_fixed",
    "smart_multigraph_degree2_removal_pass1",
    "collapse_node_clusters",
    "smart_multigraph_degree2_removal_post_collapse",
    "prune_vascular_stubs",
    "smart_multigraph_degree2_removal_post_prune",
    "remove_edges_for_self_connected_nodes",
    "reconnect_orphan_and_dangling_nodes",
    "smart_multigraph_degree2_removal_post_orphan_reconnect",
]


def _t_skeleton() -> np.ndarray:
    """A T: a trunk along z with one side branch along y, three free ends.

    The arms are long enough that no two free ends fall within the default
    reconnection thresholds, so the assembled topology is the drawn topology.
    """
    skeleton = np.zeros((40, 40, 40), dtype=bool)
    skeleton[2:38, 20, 20] = True
    skeleton[20, 21:38, 20] = True
    return skeleton


def _build(skeleton, **kwargs):
    params = dict(
        voxel_size=VOXEL_SIZE_ZYX,
        min_stub_length=0.0,
        cluster_collapse_distance=0.0,
    )
    params.update(kwargs)
    return build_graph_from_skeleton(skeleton, **params)


# --- topology ---------------------------------------------------------------


def test_a_branching_skeleton_becomes_one_junction_and_three_free_ends():
    """Node and edge counts are the whole point of the assembly; pin them exactly."""
    G = _build(_t_skeleton())

    assert isinstance(G, nx.MultiGraph)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    degrees = sorted(degree for _node, degree in G.degree())
    assert degrees == [1, 1, 1, 3]


def test_no_degree_two_nodes_survive_the_cleanup_passes():
    """Un-merged degree-2 nodes split one vessel into several resistors in series."""
    G = _build(_t_skeleton())

    assert [node for node, degree in G.degree() if degree == 2] == []


def test_every_edge_carries_the_geometry_downstream_stages_need():
    G = _build(_t_skeleton())

    for _u, _v, data in G.edges(data=True):
        assert float(data["length"]) > 0.0
        voxels = np.asarray(data["voxels"], dtype=float)
        assert voxels.ndim == 2 and voxels.shape[1] == 3


def test_the_assembled_graph_has_no_forbidden_weight_attribute():
    """'weight' meant length at build time and conductance after haemodynamics."""
    G = _build(_t_skeleton())
    assert_no_forbidden_edge_attributes(G)


# --- physical units ---------------------------------------------------------


def test_edge_lengths_and_positions_use_the_per_axis_spacing_given():
    """The trunk steps along z (2.0 um per voxel) and the branch along y (0.5 um).

    A (z, y, x) / (x, y, z) swap would measure the trunk with the 0.4 um x
    spacing and the numbers below could not come out.
    """
    G = _build(_t_skeleton())

    lengths = sorted(float(data["length"]) for _u, _v, data in G.edges(data=True))
    # Trunk splits at the junction: 18 steps one way, 17 the other; branch is 17 in y.
    assert lengths == pytest.approx([17 * 0.5, 17 * 2.0, 18 * 2.0], rel=1e-9)

    z_positions = sorted(float(data["pos"][0]) for _node, data in G.nodes(data=True))
    assert z_positions == pytest.approx([2 * 2.0, 20 * 2.0, 20 * 2.0, 37 * 2.0])


def test_the_voxel_size_is_recorded_on_the_graph_for_later_stages():
    """Statistics and mask lookups read this back rather than being told again."""
    G = _build(_t_skeleton())
    assert tuple(G.graph["voxel_size"]) == pytest.approx(VOXEL_SIZE_ZYX)


def test_an_isotropic_default_voxel_size_measures_in_voxels():
    skeleton = np.zeros((20, 20, 20), dtype=bool)
    skeleton[2:15, 10, 10] = True

    G = build_graph_from_skeleton(skeleton, min_stub_length=0.0, cluster_collapse_distance=0.0)

    assert G.number_of_edges() == 1
    assert float(next(iter(G.edges(data=True)))[2]["length"]) == pytest.approx(12.0)


# --- parameter routing ------------------------------------------------------


def test_min_stub_length_reaches_the_pruning_step():
    """A threshold wired to the wrong step leaves spurious terminals in the network.

    The 4 um side branch is a pruning candidate; the 35 um trunk is not.
    """
    skeleton = np.zeros((40, 40, 40), dtype=bool)
    skeleton[2:38, 20, 20] = True
    skeleton[20, 21:25, 20] = True

    kept = build_graph_from_skeleton(
        skeleton, voxel_size=(1.0, 1.0, 1.0),
        min_stub_length=0.0, cluster_collapse_distance=0.0,
    )
    pruned = build_graph_from_skeleton(
        skeleton, voxel_size=(1.0, 1.0, 1.0),
        min_stub_length=10.0, cluster_collapse_distance=0.0,
    )

    assert kept.number_of_edges() == 3
    assert sorted(degree for _n, degree in kept.degree()) == [1, 1, 1, 3]
    # Pruning the stub also strands the junction, which the degree-2 pass merges away.
    assert pruned.number_of_nodes() == 2
    assert pruned.number_of_edges() == 1
    assert float(next(iter(pruned.edges(data=True)))[2]["length"]) == pytest.approx(35.0)


# --- cluster_collapse_method routing -----------------------------------------


def test_cluster_collapse_method_defaults_to_the_unmodified_legacy_behaviour(monkeypatch):
    """Neither opt-in method may change a single caller's behaviour unless
    explicitly asked for -- see direction_aware_collapse's and
    persistence_collapse's own docstrings on why each exists as a fully
    separate, removable module."""
    import haemolynx.graph.assemble as assemble_module

    def must_not_be_called(name):
        def _raise(*_args, **_kwargs):
            raise AssertionError(f"{name} ran even though cluster_collapse_method was not set")
        return _raise

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_direction_aware",
        must_not_be_called("collapse_node_clusters_direction_aware"),
    )
    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_persistence",
        must_not_be_called("collapse_node_clusters_persistence"),
    )

    _build(_t_skeleton())


def test_cluster_collapse_method_direction_aware_reaches_the_direction_aware_collapse(monkeypatch):
    import haemolynx.graph.assemble as assemble_module

    calls = []

    def fake_direction_aware(
        G, *, distance_threshold, max_radial_dispersion,
        min_degree_for_dispersion_check, tangent_length_um, debug,
    ):
        calls.append(
            (max_radial_dispersion, min_degree_for_dispersion_check, tangent_length_um)
        )
        return assemble_module.collapse_node_clusters(G, distance_threshold=distance_threshold, debug=debug)

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_direction_aware", fake_direction_aware
    )

    _build(
        _t_skeleton(),
        cluster_collapse_method="direction_aware",
        cluster_collapse_max_radial_dispersion=0.7,
        cluster_collapse_direction_aware_min_degree=9,
        cluster_collapse_direction_aware_tangent_length_um=15.0,
    )

    assert calls == [(pytest.approx(0.7), 9, pytest.approx(15.0))]


def test_cluster_collapse_method_distance_only_never_calls_direction_aware_collapse(monkeypatch):
    import haemolynx.graph.assemble as assemble_module

    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("must not be called for distance_only")

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_direction_aware", must_not_be_called
    )

    _build(_t_skeleton(), cluster_collapse_method="distance_only")


def test_cluster_collapse_method_persistence_reaches_the_persistence_collapse(monkeypatch):
    import haemolynx.graph.assemble as assemble_module

    calls = []

    def fake_persistence(G, *, distance_threshold, search_radius_multiple, debug):
        calls.append(search_radius_multiple)
        return assemble_module.collapse_node_clusters(G, distance_threshold=distance_threshold, debug=debug)

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_persistence", fake_persistence
    )

    _build(
        _t_skeleton(),
        cluster_collapse_method="persistence",
        cluster_collapse_persistence_search_multiple=4.5,
    )

    assert calls == [pytest.approx(4.5)]


def test_cluster_collapse_method_distance_only_never_calls_persistence_collapse(monkeypatch):
    import haemolynx.graph.assemble as assemble_module

    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("must not be called for distance_only")

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_persistence", must_not_be_called
    )

    _build(_t_skeleton(), cluster_collapse_method="distance_only")


def test_cluster_collapse_method_direction_aware_never_calls_persistence_collapse(monkeypatch):
    import haemolynx.graph.assemble as assemble_module

    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("must not be called for direction_aware")

    monkeypatch.setattr(
        assemble_module, "collapse_node_clusters_persistence", must_not_be_called
    )

    _build(_t_skeleton(), cluster_collapse_method="direction_aware")


def test_an_unknown_cluster_collapse_method_is_rejected():
    with pytest.raises(ValueError, match="cluster_collapse_method"):
        _build(_t_skeleton(), cluster_collapse_method="nonsense")


# --- step callback contract -------------------------------------------------


def test_the_step_callback_fires_once_per_step_in_a_fixed_order():
    """The pipeline writes one artifact per label; a dropped step loses its snapshot."""
    seen: list[str] = []

    G = _build(_t_skeleton(), step_callback=lambda graph, label: seen.append(label))

    assert seen == EXPECTED_STEP_LABELS
    assert G.number_of_nodes() == 4


def test_the_published_step_labels_are_the_steps_a_build_actually_fires():
    """A progress bar sizes itself from `STEP_LABELS` before the first fires.

    Declaring the list separately from the calls that emit it is only safe if
    the two cannot drift, which is what this pins.
    """
    seen: list[str] = []

    _build(_t_skeleton(), step_callback=lambda graph, label: seen.append(label))

    assert list(STEP_LABELS) == seen == EXPECTED_STEP_LABELS


def test_step_labels_are_unique_so_snapshots_do_not_overwrite_each_other():
    """The three degree-2 passes must stay distinguishable by name."""
    seen: list[str] = []
    _build(_t_skeleton(), step_callback=lambda graph, label: seen.append(label))

    assert len(set(seen)) == len(seen)
    assert sum(label.startswith("smart_multigraph_degree2_removal") for label in seen) == 4


def test_the_callback_receives_the_live_graph_at_that_step():
    """A callback handed a label but no graph could not snapshot anything."""
    observed: list[tuple[str, int]] = []

    def record(graph, label):
        assert isinstance(graph, nx.MultiGraph)
        observed.append((label, graph.number_of_nodes()))

    G = _build(_t_skeleton(), step_callback=record)

    assert observed[0][1] > 0
    assert observed[-1] == (EXPECTED_STEP_LABELS[-1], G.number_of_nodes())


def test_building_without_a_callback_is_supported():
    G = _build(_t_skeleton(), step_callback=None)
    assert G.number_of_edges() == 3


# --- debug output -----------------------------------------------------------


def test_degree2_diagnostics_are_logged_only_in_debug_mode(caplog):
    """The report is the tool for chasing un-merged degree-2 nodes; it must be gated."""
    with caplog.at_level(logging.DEBUG, logger="haemolynx.graph"):
        _build(_t_skeleton(), debug=False)
        quiet = caplog.text
        caplog.clear()

        _build(_t_skeleton(), debug=True)
        verbose = caplog.text

    assert "DEGREE-2" in verbose.upper()
    assert "DEGREE-2" not in quiet.upper()
