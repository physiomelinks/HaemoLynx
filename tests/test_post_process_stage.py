"""The Post processing stage: a hand-edited network brought in line with the run.

The stage sits between Haemodynamics and Perturbations, so the network is
edited solved. The panel's Post processing tab edits it
(``haemolynx.graph.post_processing``), each edit marking what it touched;
`post_process` then does to those vessels what the stages before Haemodynamics
did to every vessel -- a length, a branch order, a diameter by the run's own
methods, and a zero-resistance bridge where one opens into a thick vessel --
and `solve_post_processed` solves the haemodynamics again on the result. An
unedited network passes through exactly as it was, and is not solved again.

The graphs are test_diameter_assignment's straight Art1-B01-B01-Ven1 chain,
driven through the real Diameters, Haemodynamics and Post processing stages.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx import graph as graph_module  # noqa: E402
from haemolynx.graph import (  # noqa: E402
    IS_ZERO_RESISTANCE,
    add_vessel_between,
    calculate_path_length,
    delete_vessels,
    has_pending_edits,
    mark_edited,
    set_vessel_diameter,
)
from haemolynx.graph.post_processing import APPLIED, EDITED  # noqa: E402
from haemolynx.pipeline import stages  # noqa: E402
from haemolynx.pipeline.progress import STAGES  # noqa: E402
from haemolynx.pipeline.stages import (  # noqa: E402
    PipelineResume,
    assign_diameters,
    build_haemodynamic_model,
    post_process,
    run_pipeline_stages,
)

from test_diameter_assignment import (  # noqa: E402
    BRANCH_ORDERS,
    EDGE_LENGTH_UM,
    SCHEMA,
    _boundaries,
    _settings,
    _vessel_network,
)

LAST = len(BRANCH_ORDERS)  # the chain's outlet


def _diameters_run(tmp_path, graph=None, **settings):
    """The chain through the real Diameters stage: settings, network, boundaries, model."""
    run_settings = _settings(tmp_path, **settings)
    network = _vessel_network(tmp_path, graph=graph)
    boundaries = _boundaries()
    model = assign_diameters(run_settings, network, boundaries, SCHEMA)
    return run_settings, network, boundaries, model


def _table_diameter(run_settings, order) -> float:
    """What the run's own branch-order table gives *order*."""
    from haemolynx.haemodynamics.poiseuille import table_diameter_for_order

    config = stages._haemodynamics_apply_config(run_settings, SCHEMA, voxel_size_zyx=(1.0, 1.0, 1.0))
    return float(table_diameter_for_order(config.diameter("diameter_by_branch_order"), order))


def _pendant(G, from_node=2, *, diameter_um=None):
    """A vessel drawn by hand from *from_node* to a new node 50 um off the chain."""
    start = np.asarray(G.nodes[from_node]["pos"], dtype=float)
    end = start + np.asarray([50.0, 0.0, 0.0])
    new = max(G.nodes) + 1
    G.add_node(new, pos=end)
    middle = start + np.asarray([25.0, 10.0, 0.0])
    edge = add_vessel_between(G, from_node, new, [start, middle, end], diameter_um=diameter_um)
    return edge, new


# --- the stage is part of the run -------------------------------------------


def test_post_processing_is_the_stage_between_haemodynamics_and_perturbations():
    calls = [stage.call for stage in STAGES if stage.call]
    at = calls.index("post_process")
    assert calls[at - 1] == "solve"
    assert calls[at + 1] == "run_perturbations"
    assert stages.STAGE_CALLS == tuple(calls)
    titles = [stage.title for stage in STAGES]
    assert titles.index("6. Haemodynamics") < titles.index("7. Post processing")


# --- an unedited network ----------------------------------------------------


def test_an_unedited_network_passes_through_untouched(tmp_path, monkeypatch):
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    graph_before = model.graph
    attributes_before = copy.deepcopy(
        {(u, v, k): dict(d) for u, v, k, d in graph_before.edges(keys=True, data=True)}
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("an unedited network must not be measured again")

    monkeypatch.setattr(stages, "assign_edge_diameters", refuse)
    monkeypatch.setattr(stages, "_assign_branch_orders", refuse)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert out is model and out.graph is graph_before
    assert {
        (u, v, k): dict(d) for u, v, k, d in out.graph.edges(keys=True, data=True)
    } == attributes_before
    assert APPLIED not in out.graph.graph


def test_an_unedited_pass_changes_no_resistance(tmp_path):
    """Pausing and continuing without an edit models the same network as not pausing."""
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    straight = copy.deepcopy(model)
    through = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    left = build_haemodynamic_model(run_settings, straight, SCHEMA).graph
    right = build_haemodynamic_model(run_settings, through, SCHEMA).graph
    for u, v, k, data in left.edges(keys=True, data=True):
        assert right.edges[u, v, k]["resistance"] == pytest.approx(data["resistance"])


# --- a vessel drawn by hand -----------------------------------------------------


def test_a_vessel_drawn_by_hand_gets_a_length_a_branch_order_and_a_table_diameter(tmp_path):
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    (u, v, k), _new = _pendant(model.graph)
    model.graph.edges[u, v, k]["length"] = 1.0  # stale on purpose
    assert "branch_order" not in model.graph.edges[u, v, k]

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[u, v, k]
    assert data["length"] == pytest.approx(calculate_path_length(data["voxels"]))
    assert data["branch_order"], "the new vessel has a branch order"
    assert data["diameter_source"] == "table"
    assert data["diameter_um"] == pytest.approx(
        _table_diameter(run_settings, data["branch_order"])
    )
    assert not has_pending_edits(out.graph)
    assert EDITED not in data
    assert out.graph.graph[APPLIED] is True


def test_the_diameter_an_edit_gave_stays_as_a_provisional_override(tmp_path):
    """No method measured it: the neighbours' mean beats the branch-order table."""
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    (u, v, k), _new = _pendant(model.graph, diameter_um=11.0)

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[u, v, k]
    assert data["diameter_source"] == "override"
    assert data["diameter_um"] == pytest.approx(11.0)


def test_a_measurement_replaces_the_provisional_diameter(tmp_path, monkeypatch):
    run_settings, network, boundaries, model = _diameters_run(
        tmp_path, use_edt_diameter_crosscheck=True,
        edt_diameter_prefer_over_table_on_fwhm_failure=True,
    )
    (u, v, k), _new = _pendant(model.graph, diameter_um=11.0)
    measured: list = []

    def fake_edt(G, _config, *, mask_volume=None, edges=None):
        measured.append(sorted(tuple(sorted(e[:2])) for e in edges))
        for a, b, key in edges:
            G.edges[a, b, key]["edt_diameter_um"] = 7.5
        return {"edges_measured": len(edges), "edges_skipped": []}

    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_edt_diameters", fake_edt)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[u, v, k]
    assert measured == [[tuple(sorted((u, v)))]], "only the edited vessel is measured"
    assert data["diameter_source"] == "edt_mask"
    assert data["diameter_um"] == pytest.approx(7.5)
    # Every other vessel keeps what Diameters gave it.
    for a, b, key, other in out.graph.edges(keys=True, data=True):
        if (a, b, key) != (u, v, k):
            assert other["diameter_source"] == "table"


def test_a_measurement_carried_over_from_a_cut_vessel_is_measured_again(tmp_path):
    """A half keeps the whole vessel's diameter only as a provisional one."""
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    G = model.graph
    data = G.edges[1, 2, 0]
    data["fwhm_diameter_um"] = 33.0  # as if FWHM had measured the whole vessel
    data["fwhm_status"] = "measured"
    node = graph_module.split_vessel_at(G, (1, 2, 0), (0.0, 0.0, 1.5 * EDGE_LENGTH_UM))

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    for half in ((1, node), (node, 2)):
        (key,) = out.graph[half[0]][half[1]]
        half_data = out.graph.edges[half[0], half[1], key]
        assert "fwhm_diameter_um" not in half_data and "fwhm_status" not in half_data
        assert half_data["diameter_source"] == "override"
        assert half_data["diameter_um"] == pytest.approx(_table_diameter(run_settings, "B01"))
        assert half_data["length"] == pytest.approx(EDGE_LENGTH_UM / 2)


def test_a_merge_a_deletion_leaves_is_measured_as_one_vessel(tmp_path):
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    (u, v, k), new = _pendant(model.graph)
    post_process(run_settings, model, boundaries, SCHEMA, network=network)

    # Deleting the pendant leaves node 2 a pass-through: 1-2 and 2-3 merge.
    delete_vessels(model.graph, [(u, v, k)], protected={0, LAST})
    assert not model.graph.has_node(2) and model.graph.has_edge(1, 3)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    (key,) = out.graph[1][3]
    merged = out.graph.edges[1, 3, key]
    assert merged["length"] == pytest.approx(2 * EDGE_LENGTH_UM)
    assert merged["branch_order"]
    assert merged["diameter_um"] > 0


def test_a_deletion_alone_restamps_table_diameters_from_the_new_orders(tmp_path):
    """Nothing to measure, but a branch order an edit moved must move its diameter."""
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    G = model.graph
    right_order = G.edges[1, 2, 0]["branch_order"]
    G.edges[1, 2, 0]["branch_order"] = "Ven9"  # stale, as a changed topology leaves it
    G.edges[1, 2, 0]["diameter_um"] = 99.0
    mark_edited(G)  # a deletion elsewhere: pending, no vessel marked

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert out.graph.edges[1, 2, 0]["branch_order"] == right_order
    assert out.graph.edges[1, 2, 0]["diameter_um"] == pytest.approx(
        _table_diameter(run_settings, right_order)
    )


# --- boundary nodes ---------------------------------------------------------


def test_boundary_nodes_an_edit_removed_leave_the_lists(tmp_path):
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    G = model.graph
    G.add_edge(0, 10, voxels=[(0.0, 0.0, 0.0), (0.0, 50.0, 0.0)], length=50.0)
    G.nodes[10]["pos"] = np.asarray([0.0, 50.0, 0.0])
    boundaries.inlet_nodes.append(10)
    boundaries.resistance_node_pair = (10, LAST)
    run_settings["inlet_nodes"] = list(boundaries.inlet_nodes)
    G.remove_node(10)  # as a prune would
    mark_edited(G)

    post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert boundaries.inlet_nodes == [0]
    assert run_settings["inlet_nodes"] == [0]
    assert boundaries.resistance_node_pair == (0, LAST)


def test_an_edit_that_removes_every_inlet_is_refused(tmp_path):
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    G = model.graph
    G.remove_node(0)
    mark_edited(G)

    with pytest.raises(ValueError, match="every inlet or every outlet"):
        post_process(run_settings, model, boundaries, SCHEMA, network=network)


# --- thick-vessel bridges ---------------------------------------------------

#: The thick vessel, 4 um in radius round y = z = 5, runs from x = THICK_FROM_UM
#: on through the end of a 1-um grid.
THICK_FROM_UM = 30.0
THICK_RADIUS_UM = 4.0


def _thick_network(*, along_axis: bool = False):
    """A thick vessel's centreline 2-3 (x 40..55), and a thin vessel 0-1 the
    hand-drawn vessel 1-2 joins to it: from the side (along y, 1 at y = 15),
    or *along_axis*, carrying on from the thick vessel's end (x 0..20)."""
    G = nx.MultiGraph()
    if along_axis:
        positions = {0: (5.0, 5.0, 0.0), 1: (5.0, 5.0, 20.0), 2: (5.0, 5.0, 40.0), 3: (5.0, 5.0, 55.0)}
    else:
        positions = {0: (5.0, 25.0, 40.0), 1: (5.0, 15.0, 40.0), 2: (5.0, 5.0, 40.0), 3: (5.0, 5.0, 55.0)}
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for a, b in ((0, 1), (2, 3), (1, 2)):
        voxels = [positions[a], positions[b]]
        if (a, b) == (1, 2):
            continue  # the vessel drawn by hand, below
        G.add_edge(a, b, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01")
    return G


def _thick_mask():
    z, y, _x = np.indices((11, 11, 60))
    mask = (z - 5) ** 2 + (y - 5) ** 2 <= THICK_RADIUS_UM ** 2
    mask[:, :, : int(THICK_FROM_UM)] = False
    return mask


def _thick_run(tmp_path, *, volume_mask=True, along_axis=False, **settings):
    options = {
        "use_thick_vessel_skeletonisation": True,
        "inlet_nodes": [0],
        "outlet_nodes": [3],
        "arteriole_boundary_nodes": [0],
        "venule_boundary_nodes": [3],
        **settings,
    }
    run_settings = _settings(tmp_path, **options)
    G = _thick_network(along_axis=along_axis)
    network = _vessel_network(tmp_path, graph=G)
    network.volume.thick_vessel_mask = _thick_mask() if volume_mask else None
    boundaries = SimpleNamespace(
        inlet_nodes=[0], outlet_nodes=[3], arteriole_boundary_nodes=[0],
        venule_boundary_nodes=[3], large_arteriole_boundary_nodes=[],
        large_venule_boundary_nodes=[], resistance_node_pair=(0, 3), graph=G,
    )
    model = stages.HaemodynamicModel(graph=G)
    edge = add_vessel_between(G, 1, 2)
    return run_settings, network, boundaries, model, edge


def _pieces_between(G, a, b):
    """The chain of edges the vessel drawn from *a* to *b* became, in order."""
    path = nx.shortest_path(G, a, b)
    pieces = []
    for x, y in zip(path[:-1], path[1:]):
        (key,) = G[x][y]
        pieces.append((x, y, G.edges[x, y, key]))
    return pieces


def test_a_vessel_drawn_into_a_thick_vessel_becomes_a_bridge_inside_it(tmp_path):
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path)

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    # The drawn vessel up to the thick vessel's boundary, and one bridge from
    # there to the thick vessel's centreline.
    pieces = _pieces_between(out.graph, 1, 2)
    assert len(pieces) == 2
    (_a, joint, outside), (_j, _b, bridge) = pieces
    assert not outside.get(IS_ZERO_RESISTANCE)
    assert bridge.get(IS_ZERO_RESISTANCE)
    joint_y = float(out.graph.nodes[joint]["pos"][1])
    assert abs(joint_y - (5.0 + THICK_RADIUS_UM)) <= 1.0
    assert bridge["length"] <= THICK_RADIUS_UM + 1.0
    assert outside["length"] + bridge["length"] == pytest.approx(10.0)
    assert outside["branch_order"] and outside["diameter_um"] > 0
    assert EDITED not in outside
    # A bridge is no vessel of any order, and is not measured.
    assert "branch_order" not in bridge
    assert bridge["diameter_source"] == "bridge" and bridge["diameter_um"] > 0
    assert EDITED not in bridge
    # The thick vessel's own centreline, entirely inside, is no bridge.
    (key,) = out.graph[2][3]
    assert not out.graph.edges[2, 3, key].get(IS_ZERO_RESISTANCE)
    assert out.graph is model.graph is boundaries.graph is network.graph


def test_a_gap_in_the_thick_centreline_beside_an_edit_is_drawn_and_measured(tmp_path):
    """The tree fill in post-processing marks what it draws edited, so the
    stage measures it with the run's own methods as it does a vessel drawn
    by hand: it used to be left to the branch-order table's diameter."""
    from haemolynx.graph.thick_vessel_junctions import FROM_TREE

    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path)
    G = model.graph
    # The graph's thick centreline broken from x = 44 to 50.
    G.remove_edge(2, 3)
    G.add_node(4, pos=np.asarray((5.0, 5.0, 44.0)))
    G.add_node(5, pos=np.asarray((5.0, 5.0, 50.0)))
    for a, b in ((2, 4), (5, 3)):
        voxels = [tuple(G.nodes[a]["pos"]), tuple(G.nodes[b]["pos"])]
        G.add_edge(a, b, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01")

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    drawn = [d for *_e, d in out.graph.edges(data=True) if d.get(FROM_TREE)]
    assert drawn and sum(d["length"] for d in drawn) == pytest.approx(6.0, abs=1.5)
    # Measured with the drawn vessel's own two pieces: up to the wall, and its bridge.
    assert out.results["post_process"]["edited_vessels"] == 2 + len(drawn)


def test_a_vessel_drawn_on_along_a_thick_vessels_line_from_its_end_gets_no_bridge(tmp_path):
    """The thick vessel narrowing into the drawn one (Harvey, 2026-10-13:
    bridges along a vessel's own line should not be there): the drawn vessel
    stays vessel all the way, where it used to get a bridge from the thick
    region's end to the tree."""
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path, along_axis=True)

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    pieces = _pieces_between(out.graph, 1, 2)
    assert not any(d.get(IS_ZERO_RESISTANCE) for *_e, d in pieces)
    assert sum(d["length"] for *_e, d in pieces) == pytest.approx(20.0)


def test_the_bridge_gets_the_negligible_resistance_graph_building_bridges_get(tmp_path):
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    solved = build_haemodynamic_model(run_settings, out, SCHEMA).graph

    from haemolynx.haemodynamics.apply import ZERO_RESISTANCE_FRACTION

    real = [d["resistance"] for *_e, d in solved.edges(data=True) if not d.get(IS_ZERO_RESISTANCE)]
    bridges = [d["resistance"] for *_e, d in solved.edges(data=True) if d.get(IS_ZERO_RESISTANCE)]
    assert bridges == [pytest.approx(min(real) * ZERO_RESISTANCE_FRACTION)]


def test_the_thick_region_comes_from_the_resume_when_the_volume_has_none(tmp_path, monkeypatch):
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path, volume_mask=False)
    monkeypatch.setattr(
        stages, "thick_vessel_mask_from_image",
        lambda *_a, **_k: pytest.fail("the resume's region should have been used"),
    )

    out = post_process(
        run_settings, model, boundaries, SCHEMA, network=network, thick_vessel_mask=_thick_mask()
    )

    assert sum(bool(d.get(IS_ZERO_RESISTANCE)) for *_e, d in _pieces_between(out.graph, 1, 2)) == 1


def test_without_a_region_to_hand_it_is_found_again_in_the_segmentation(tmp_path, monkeypatch):
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path, volume_mask=False)
    asked: list = []

    def found_again(settings, image, voxel_size_xyz, crop, *, cache):
        asked.append((image is network.volume.image, tuple(voxel_size_xyz), crop))
        return _thick_mask()

    monkeypatch.setattr(stages, "thick_vessel_mask_from_image", found_again)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    # The image was not cropped on import, so the vessel masks are not either.
    assert asked == [(True, (1.0, 1.0, 1.0), None)]
    assert sum(bool(d.get(IS_ZERO_RESISTANCE)) for *_e, d in _pieces_between(out.graph, 1, 2)) == 1


def test_no_bridge_without_thickness_gated_skeletonisation(tmp_path, monkeypatch):
    run_settings, network, boundaries, model, _edge = _thick_run(
        tmp_path, use_thick_vessel_skeletonisation=False
    )
    monkeypatch.setattr(
        stages, "thick_vessel_mask_from_image",
        lambda *_a, **_k: pytest.fail("no thick region is looked for when the option is off"),
    )

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    (piece,) = _pieces_between(out.graph, 1, 2)
    assert not piece[2].get(IS_ZERO_RESISTANCE)


# --- pausing and picking the run up again -----------------------------------


def _stub_early_stages(monkeypatch, tmp_path):
    """Segment to Boundaries return the chain; Perturbations and Export do nothing."""
    network = _vessel_network(tmp_path)

    monkeypatch.setattr(stages, "segment", lambda *a, **k: SimpleNamespace(image_path=None))
    monkeypatch.setattr(stages, "skeletonise", lambda *a, **k: network.volume)
    monkeypatch.setattr(stages, "build_network", lambda *a, **k: network)

    def boundaries_for(settings, net):
        found = _boundaries()
        found.graph = net.graph
        return found

    monkeypatch.setattr(stages, "assign_boundaries", boundaries_for)
    monkeypatch.setattr(stages, "run_perturbations", lambda *a, **k: stages.PerturbationRun())
    monkeypatch.setattr(stages, "export_results", lambda *a, **k: None)
    return network


def _resume_from_post_processing(graph) -> PipelineResume:
    """What the panel's Continue / Regenerate hands the run: *graph*, the chain's boundaries."""
    return PipelineResume(
        start_from="post_process",
        graph=graph,
        inlet_nodes=(0,),
        outlet_nodes=(LAST,),
        arteriole_boundary_nodes=(0,),
        venule_boundary_nodes=(LAST,),
        resistance_node_pair=(0, LAST),
    )


def _shortcut(G):
    """A vessel drawn by hand from node 1 straight to node 3, beside the chain."""
    a, b = np.asarray(G.nodes[1]["pos"]), np.asarray(G.nodes[3]["pos"])
    return add_vessel_between(G, 1, 3, [a, (a + b) / 2 + [30.0, 0.0, 0.0], b])


def test_a_run_pauses_after_haemodynamics_with_the_network_solved(tmp_path, monkeypatch):
    """The pause comes after the solve, so the network is edited with its flows."""
    _stub_early_stages(monkeypatch, tmp_path)
    ran: list[str] = []
    paused = run_pipeline_stages(
        _settings(tmp_path),
        SCHEMA,
        on_stage_output=lambda stage, _output: ran.append(stage),
        stop_after="solve",
    )

    assert ran == [
        "segment", "skeletonise", "build_network", "assign_boundaries",
        "assign_diameters", "build_haemodynamic_model", "solve",
    ]
    for *_e, data in paused.edges(data=True):
        assert np.isfinite(data["resistance"]) and abs(data["flow_signed"]) > 0
    assert all("pressure" in data for _n, data in paused.nodes(data=True))


def test_a_run_paused_after_haemodynamics_and_continued_solves_as_one_run(tmp_path, monkeypatch):
    options = {}
    _stub_early_stages(monkeypatch, tmp_path / "straight")
    straight = run_pipeline_stages(_settings(tmp_path / "straight", **options), SCHEMA)

    _stub_early_stages(monkeypatch, tmp_path / "paused")
    paused = run_pipeline_stages(
        _settings(tmp_path / "paused", **options), SCHEMA, stop_after="solve"
    )
    continued = run_pipeline_stages(
        _settings(tmp_path / "paused", **options),
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(copy.deepcopy(paused)),
    )

    for node, data in straight.nodes(data=True):
        assert continued.nodes[node]["pressure"] == pytest.approx(data["pressure"])
    for u, v, k, data in straight.edges(keys=True, data=True):
        assert continued.edges[u, v, k]["resistance"] == pytest.approx(data["resistance"])
        assert continued.edges[u, v, k]["flow_signed"] == pytest.approx(data["flow_signed"])


def test_a_resumed_network_drops_the_statistics_it_carried(tmp_path, monkeypatch):
    """Regenerate starts from the network on screen, which carries what the
    last export measured -- of the network before the edit. The run drops it,
    so one paused before its export returns no statistics rather than old ones."""
    _stub_early_stages(monkeypatch, tmp_path)
    paused = run_pipeline_stages(_settings(tmp_path), SCHEMA, stop_after="solve")
    paused.graph[stages.STATISTICS_REPORT] = {
        "mode": "fast", "csv": None, "sections": {"basic": {"Total Nodes": 99}}
    }

    resumed = run_pipeline_stages(
        _settings(tmp_path),
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(paused),
        stop_after="post_process",
    )

    assert stages.STATISTICS_REPORT not in resumed.graph


def test_an_unedited_network_is_not_solved_again(tmp_path, monkeypatch):
    """Continue without an edit keeps the solve, including what the CSV needs.

    Post processing is after the solve, so Continue does not solve again, and
    neither does Regenerate when nothing was edited. The two resistances and
    the haematocrit-distribution report have to travel with the graph: they
    used to exist only on the Solution, and the statistics CSV then lost its
    Haemodynamics section.
    """
    import pickle

    run_settings = _settings(tmp_path, haematocrit_model="distributed_iterative")
    _stub_early_stages(monkeypatch, tmp_path)
    paused_out: dict = {}
    paused = run_pipeline_stages(
        run_settings, SCHEMA, stop_after="solve", on_stage_output=paused_out.__setitem__
    )
    solved = paused_out["solve"]
    assert solved.equivalent_resistance is not None
    assert solved.network_resistance is not None
    assert solved.statistics["haematocrit_distribution"]

    def refuse(*_args, **_kwargs):
        raise AssertionError("an unedited network must not be solved again")

    monkeypatch.setattr(stages, "solve", refuse)
    monkeypatch.setattr(stages, "build_haemodynamic_model", refuse)
    outputs: dict = {}
    # The panel's checkpoint is a pickle of the graph, not a live Solution.
    run_pipeline_stages(
        run_settings,
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(pickle.loads(pickle.dumps(paused))),
        on_stage_output=outputs.__setitem__,
    )

    continued = outputs["post_process"]
    assert continued.post_processed is None
    assert continued.equivalent_resistance == pytest.approx(solved.equivalent_resistance)
    assert continued.network_resistance == pytest.approx(solved.network_resistance)
    assert continued.equivalent_resistance_nodes == solved.equivalent_resistance_nodes
    assert (
        continued.statistics["haematocrit_distribution"]
        == solved.statistics["haematocrit_distribution"]
    )
    assert "Haemodynamics" in stages.resistance_statistics(continued)


def test_a_vessel_drawn_while_paused_is_solved_with_the_rest(tmp_path, monkeypatch):
    """Regenerating reruns Haemodynamics on the edited network, so the new
    vessel carries a resistance and a flow, and the chain it bypasses carries
    less than it did before the edit."""
    options = {}
    _stub_early_stages(monkeypatch, tmp_path)
    paused = run_pipeline_stages(_settings(tmp_path, **options), SCHEMA, stop_after="solve")
    edited = copy.deepcopy(paused)
    edge = _shortcut(edited)
    bypassed_before = abs(paused.edges[1, 2, 0]["flow_signed"])

    outputs: dict = {}
    solved = run_pipeline_stages(
        _settings(tmp_path, **options),
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(edited),
        on_stage_output=outputs.__setitem__,
    )

    data = solved.edges[edge]
    assert data["branch_order"] and data["diameter_um"] > 0
    assert np.isfinite(data["resistance"]) and data["resistance"] > 0
    assert abs(data["flow_signed"]) > 0, "blood takes the new vessel too"
    assert abs(solved.edges[1, 2, 0]["flow_signed"]) < bypassed_before
    # What the stage hands the viewer is the solve it ran again.
    solution = outputs["post_process"]
    assert solution.graph is solved
    assert solution.post_processed["edited_vessels"] == 1
    assert len(solution.pressure) == solved.number_of_nodes()


def test_perturbations_and_the_export_get_the_solve_run_again(tmp_path, monkeypatch):
    _stub_early_stages(monkeypatch, tmp_path)
    paused = run_pipeline_stages(_settings(tmp_path), SCHEMA, stop_after="solve")
    edited = copy.deepcopy(paused)
    edge = _shortcut(edited)
    handed: dict = {}

    def perturb(settings, model, *_args, **_kwargs):
        handed["perturbed"] = model.graph
        return stages.PerturbationRun()

    monkeypatch.setattr(stages, "run_perturbations", perturb)
    monkeypatch.setattr(
        stages, "export_results",
        lambda settings, network, model, solution: handed.update(exported=solution),
    )

    run_pipeline_stages(
        _settings(tmp_path),
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(edited),
    )

    assert abs(handed["perturbed"].edges[edge]["flow_signed"]) > 0
    assert handed["exported"].graph is handed["perturbed"]
    assert handed["exported"].post_processed["edited_vessels"] == 1


def test_solve_post_processed_runs_the_network_handling_on_the_edited_network(
    tmp_path, monkeypatch
):
    """The Haemodynamics tab's "Network handling" applies to the edited network too."""
    run_settings, network, boundaries, model = _diameters_run(tmp_path)
    (u, v, k), _new = _pendant(model.graph)
    model = post_process(run_settings, model, boundaries, SCHEMA, network=network)
    handled: list = []
    real = stages.apply_network_handling

    def spy(settings, handed_model, *args, **kwargs):
        handled.append(handed_model.graph.has_edge(u, v, k))
        return real(settings, handed_model, *args, **kwargs)

    monkeypatch.setattr(stages, "apply_network_handling", spy)
    solution = stages.solve_post_processed(
        run_settings, model, boundaries, SCHEMA, network=network
    )

    assert handled == [True]
    assert np.isfinite(solution.graph.edges[u, v, k]["resistance"])
    assert solution.post_processed == model.results["post_process"]


# --- a resumed run keeps the Large_Art/Large_Ven hand-off nodes ----------------------


def test_a_resume_hands_the_large_vessel_hand_off_nodes_on(tmp_path):
    """Regression: only assign_boundaries finds them, so a run resumed past it --
    which assigns branch orders again -- used to lose the Large tier."""
    run_settings = _settings(tmp_path)
    run_settings["large_arteriole_boundary_nodes"][:] = []
    run_settings["large_venule_boundary_nodes"][:] = []
    resume = PipelineResume(
        start_from="post_process",
        inlet_nodes=(0,),
        outlet_nodes=(LAST,),
        large_arteriole_boundary_nodes=(1,),
        large_venule_boundary_nodes=(3,),
    )

    stages._fill_boundary_settings(run_settings, resume)
    boundaries = stages._boundaries_from_resume(resume, _vessel_network(tmp_path))

    assert run_settings["large_arteriole_boundary_nodes"] == [1]
    assert run_settings["large_venule_boundary_nodes"] == [3]
    assert boundaries.large_arteriole_boundary_nodes == [1]
    assert boundaries.large_venule_boundary_nodes == [3]


# --- a vessel drawn in a large vessel mask is tagged as lying in it ----------------


def _large_tier_run(tmp_path, *, gap_at_z=25.0):
    """test_branch_order_hierarchy's branching large-vessel network through the
    real Boundaries and Diameters stages, its large arteriole broken at
    *gap_at_z*: the trunk a0-a1-a2 reaches the inlet, the piece beyond it
    (m-a3) is a fragment lying in the arteriole mask, joined to nothing --
    the E14.5 large-mask vessels the Large tier missed. Returns settings,
    network, boundaries, model and the node ids."""
    from test_branch_order_hierarchy import _branching_large_vessel_network, _large_tier_settings

    network, ids = _branching_large_vessel_network()
    G = network.graph
    a2, a3 = ids["a2"], ids["a3"]
    G.remove_edge(a2, a3)
    ids["m"] = max(G.nodes) + 1
    m_pos = (gap_at_z, 4.0, 5.0)
    G.add_node(ids["m"], pos=m_pos)
    G.add_edge(ids["m"], a3, voxels=[m_pos, G.nodes[a3]["pos"]],
               length=float(G.nodes[a3]["pos"][0] - gap_at_z))
    run_settings = _large_tier_settings(tmp_path)
    run_settings["run_haemodynamics"] = False
    boundaries = stages.assign_boundaries(run_settings, network)
    model = assign_diameters(run_settings, network, boundaries, SCHEMA)
    return run_settings, network, boundaries, model, ids


def _large_tags(G) -> dict[tuple, str | None]:
    return {
        (u, v, k): data.get(graph_module.LARGE_MASK_VESSEL_TYPE)
        for u, v, k, data in G.edges(keys=True, data=True)
    }


def test_a_vessel_drawn_along_the_large_arteriole_stays_large_art(tmp_path):
    """Regression: a vessel drawn by hand carried no large-mask tag, and the
    Large tier is the tagged vessels joined to the inlet through tagged
    vessels -- so a vessel drawn to close a gap in the large arteriole was
    not Large_Art, nor was the trunk it joined back on."""
    run_settings, network, boundaries, model, ids = _large_tier_run(tmp_path)
    G = model.graph
    assert run_settings["inlet_nodes"] == [ids["a0"]]
    assert G.edges[ids["m"], ids["a3"], 0].get("branch_order") is None  # the fragment
    drawn = add_vessel_between(G, ids["a2"], ids["m"])

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    G = out.graph
    assert G.edges[ids["a1"], ids["a2"], 0]["branch_order"] == "Large_Art2"
    assert G.edges[drawn].get("branch_order") == "Large_Art3"
    assert G.edges[ids["m"], ids["a3"], 0].get("branch_order") == "Large_Art4"
    assert G.edges[drawn][graph_module.LARGE_MASK_VESSEL_TYPE] == "arteriole"


def test_a_vessel_drawn_under_covered_between_two_large_vessels_takes_their_tag(tmp_path):
    """The Boundaries stage's own rule, gap-filling included: a vessel only a
    third in the arteriole mask, between two vessels lying in it, takes the
    tag (GAP_FILL_FRACTION_OF_THRESHOLD of the 0.5 overlap threshold)."""
    run_settings, network, boundaries, model, ids = _large_tier_run(tmp_path)
    G = model.graph
    a2, m = ids["a2"], ids["m"]
    start, end = np.asarray(G.nodes[a2]["pos"]), np.asarray(G.nodes[m]["pos"])
    drawn = add_vessel_between(G, a2, m, [start, (start + end) / 2 + [0.0, 0.0, 8.0], end])

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert out.graph.edges[drawn][graph_module.LARGE_MASK_VESSEL_TYPE] == "arteriole"
    assert out.graph.edges[drawn]["branch_order"] == "Large_Art3"


def test_an_edited_vessel_outside_the_large_masks_loses_the_tag_and_no_other_vessel_changes(
    tmp_path,
):
    """A split copies its parent's attributes and a merge takes its first
    vessel's, so an edited vessel can carry a large-mask tag it does not lie
    in: it is judged again, and every vessel not edited keeps the tag the
    Boundaries stage gave it."""
    run_settings, network, boundaries, model, ids = _large_tier_run(tmp_path)
    G = model.graph
    start = np.asarray(G.nodes[ids["a2"]]["pos"], dtype=float)
    new = max(G.nodes) + 1
    G.add_node(new, pos=start + [6.0, 0.0, 11.0])
    drawn = add_vessel_between(G, ids["a2"], new)
    G.edges[drawn][graph_module.LARGE_MASK_VESSEL_TYPE] = "arteriole"  # as a split copies it
    before = {edge: tag for edge, tag in _large_tags(G).items() if edge != drawn}
    assert set(before.values()) == {"arteriole", "venule", None}

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert graph_module.LARGE_MASK_VESSEL_TYPE not in out.graph.edges[drawn]
    assert not out.graph.edges[drawn]["branch_order"].startswith("Large_")
    after = _large_tags(out.graph)
    assert {edge: after[edge] for edge in before} == before


def test_without_the_large_masks_the_edited_vessels_keep_their_tags_and_it_says_so(
    tmp_path, caplog
):
    run_settings, network, boundaries, model, ids = _large_tier_run(tmp_path)
    split_piece = (ids["a1"], ids["a2"], 0)
    mark_edited(model.graph, [split_piece])
    boundaries.large_arteriole_mask = boundaries.large_venule_mask = None

    with caplog.at_level("WARNING", logger="haemolynx.pipeline.stages"):
        out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert out.graph.edges[split_piece][graph_module.LARGE_MASK_VESSEL_TYPE] == "arteriole"
    assert out.graph.edges[split_piece]["branch_order"] == "Large_Art2"
    assert "no large vessel masks to read" in caplog.text


# --- the small-mask arteriole/venule boundary nodes follow the edits ---------------


def _small_boundary_run(tmp_path, *, side_arteriole=False, **overrides):
    """test_branching_large_vessel_network through the real Boundaries and
    Diameters stages, with a second small arteriole's mask beside the large
    arteriole at z = 25: a node p on the trunk there, and a capillary from f,
    across the mask from p (x = 14), into the bed at c. With
    *side_arteriole* the vessel p-f lies in the network when Boundaries
    runs, so p and f are arteriole boundary nodes; without it, it is the
    vessel a hand edit draws. Returns settings, network, boundaries, model
    and the node ids."""
    from test_branch_order_hierarchy import (
        _branching_large_vessel_network,
        _large_tier_settings,
        _segment,
    )

    network, ids = _branching_large_vessel_network()
    G = network.graph
    G.remove_edge(ids["a2"], ids["a3"])
    ids["p"], ids["f"] = max(G.nodes) + 1, max(G.nodes) + 2
    G.add_node(ids["p"], pos=(25.0, 4.0, 5.0))
    G.add_node(ids["f"], pos=(25.0, 4.0, 14.0))
    vessels = [("a2", "p"), ("p", "a3"), ("f", "c")] + ([("p", "f")] if side_arteriole else [])
    for a, b in vessels:
        start, end = G.nodes[ids[a]]["pos"], G.nodes[ids[b]]["pos"]
        G.add_edge(ids[a], ids[b], voxels=_segment(start, end),
                   length=float(np.linalg.norm(np.subtract(end, start))))
    network.small_arteriole_mask[24:27, 3:6, 8:14] = True
    run_settings = _large_tier_settings(tmp_path)
    run_settings["run_haemodynamics"] = False
    run_settings.update(overrides)
    boundaries = stages.assign_boundaries(run_settings, network)
    model = assign_diameters(run_settings, network, boundaries, SCHEMA)
    return run_settings, network, boundaries, model, ids


def _boundary_lists(run_settings, boundaries, ids) -> dict[str, list[str]]:
    """The arteriole and venule boundary lists by node name, checking the
    settings' lists (which branch orders are read from) say the same."""
    name = {node: label for label, node in ids.items()}
    lists = {}
    for key in ("arteriole_boundary_nodes", "venule_boundary_nodes"):
        assert list(run_settings[key]) == list(getattr(boundaries, key))
        lists[key] = [name[node] for node in getattr(boundaries, key)]
    return lists


def _protected(run_settings) -> list:
    """The nodes the panel will not let a deletion strand or merge away."""
    return [
        *run_settings["inlet_nodes"], *run_settings["outlet_nodes"],
        *run_settings["arteriole_boundary_nodes"], *run_settings["venule_boundary_nodes"],
    ]


def test_a_side_arteriole_drawn_through_the_small_mask_ends_at_arteriole_boundary_nodes(
    tmp_path,
):
    """Regression: post_process only kept the boundary lists to the nodes
    still in the graph, so a vessel drawn from the large arteriole through a
    small arteriole's mask into the bed had no boundary node at its far end
    -- it came out a capillary, and the bed beyond it was counted from
    elsewhere. Now its ends are judged as the Boundaries stage judged every
    vessel's: it lies in the mask, so f, where it meets the capillary, and
    p, where it leaves the large arteriole, are arteriole boundary nodes, as
    a1 and s1 are for the small arteriole Boundaries found."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    before = _boundary_lists(run_settings, boundaries, ids)
    assert before == {
        "arteriole_boundary_nodes": ["a1", "s1"],
        "venule_boundary_nodes": ["v1", "w1"],
    }
    drawn = add_vessel_between(model.graph, ids["p"], ids["f"])

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    G = out.graph
    assert G.edges[drawn][graph_module.SMALL_MASK_VESSEL_TYPE] == "arteriole"
    assert _boundary_lists(run_settings, boundaries, ids) == {
        "arteriole_boundary_nodes": ["a1", "s1", "p", "f"],  # the others where they were
        "venule_boundary_nodes": ["v1", "w1"],
    }
    assert G.edges[drawn]["branch_order"] == "Art1"
    assert G.edges[ids["f"], ids["c"], 0]["branch_order"] == "B01"
    assert G.edges[ids["a2"], ids["p"], 0]["branch_order"] == "Large_Art3"
    assert G.edges[ids["a1"], ids["s1"], 0]["branch_order"] == "Art1"
    assert out.results["post_process"]["boundary_nodes"]["arteriole_boundary_nodes"] == [
        ids["a1"], ids["s1"], ids["p"], ids["f"],
    ]


def test_deleting_a_small_arteriole_drops_the_boundary_nodes_at_its_ends(tmp_path):
    """A deletion leaves no vessel to mark edited, but the nodes it took a
    vessel from are judged again: with p-f gone, p is a node on the large
    arteriole and f a capillary's, neither where a small arteriole meets
    the rest. The small arteriole Boundaries found elsewhere keeps its."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(
        tmp_path, side_arteriole=True
    )
    assert sorted(_boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"]) == [
        "a1", "f", "p", "s1",
    ]
    assert model.graph.edges[ids["f"], ids["c"], 0]["branch_order"] == "B01"
    delete_vessels(model.graph, [(ids["p"], ids["f"], 0)], protected=_protected(run_settings))

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert sorted(_boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"]) == [
        "a1", "s1",
    ]
    assert _boundary_lists(run_settings, boundaries, ids)["venule_boundary_nodes"] == ["v1", "w1"]
    assert out.graph.edges[ids["f"], ids["c"], 0]["branch_order"] == "B02"


def test_boundary_nodes_the_edits_do_not_touch_stay_as_they_are(tmp_path):
    """Only the nodes an edit touched are judged again. A boundary node the
    masks' rule would not give -- here one the hop-distance fallback could
    have added beside an outlet the masks miss -- stays where it is when
    the edits are elsewhere, and so does every list's order."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    boundaries.venule_boundary_nodes.append(ids["v3"])
    assert run_settings["venule_boundary_nodes"] is boundaries.venule_boundary_nodes
    add_vessel_between(model.graph, ids["p"], ids["f"])

    post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert _boundary_lists(run_settings, boundaries, ids)["venule_boundary_nodes"] == [
        "v1", "w1", "v3",
    ]


def test_an_edit_never_leaves_a_boundary_list_empty(tmp_path, caplog):
    """Deleting the network's only small arteriole would leave no arteriole
    boundary node, and with none the branch orders stop being hierarchical
    (or a strict run fails): the list is kept as it was, and it says so."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    delete_vessels(model.graph, [(ids["a1"], ids["s1"], 0)], protected=_protected(run_settings))

    with caplog.at_level("WARNING", logger="haemolynx.pipeline.stages"):
        out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert _boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"] == [
        "a1", "s1",
    ]
    assert "leave no arteriole boundary node" in caplog.text
    assert out.graph.edges[ids["w1"], ids["v1"], 0]["branch_order"] == "Ven1"


@pytest.mark.parametrize("fallback", [True, False])
def test_a_vessel_drawn_from_an_inlet_the_masks_miss_gets_the_fallback_boundary_node(
    tmp_path, fallback
):
    """With no Large tier the inlets' own vessels lie in no small mask, so
    the hop-distance fallback makes the node a vessel from each an arteriole
    boundary node -- the Boundaries stage put p there, a vessel from the
    inlet a3 (a1, a vessel from a0, is the small arteriole's already). A
    vessel drawn from a0, clear of every mask, ends where the fallback,
    placed on the edited network, says: at c, one vessel from the inlet, so
    it is Art1 and stops there. With the fallback off nothing marks its end."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(
        tmp_path,
        assign_large_vessel_branch_orders=False,
        small_vessel_boundary_fallback_to_hop_distance=fallback,
    )
    before = _boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"]
    assert sorted(run_settings["inlet_nodes"]) == sorted([ids["a0"], ids["a3"]])
    assert before == (["a1", "s1", "p"] if fallback else ["a1", "s1"])
    a0, c = model.graph.nodes[ids["a0"]]["pos"], model.graph.nodes[ids["c"]]["pos"]
    drawn = add_vessel_between(model.graph, ids["a0"], ids["c"], [a0, (0.0, 4.0, 15.0), c])

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    arterioles = _boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"]
    assert graph_module.SMALL_MASK_VESSEL_TYPE not in out.graph.edges[drawn]
    if fallback:
        assert arterioles == [*before, "c"]
        assert out.graph.edges[drawn]["branch_order"] == "Art1"
    else:
        assert arterioles == before


def test_without_the_small_masks_the_boundary_nodes_stay_and_it_says_so(tmp_path, caplog):
    """A run resumed from a checkpoint saved before the small masks were
    carried has none to judge an edited vessel by."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    boundaries.small_arteriole_mask = boundaries.small_venule_mask = None
    add_vessel_between(model.graph, ids["p"], ids["f"])

    with caplog.at_level("WARNING", logger="haemolynx.pipeline.stages"):
        post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert _boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"] == [
        "a1", "s1",
    ]
    assert "no small vessel masks to read" in caplog.text


def test_a_network_without_small_mask_tags_keeps_its_boundary_nodes_and_says_so(
    tmp_path, caplog
):
    """A graph whose Boundaries stage ran before the tags were written: the
    vessels beside an edit cannot be read, so nothing is judged from them --
    every touched node would read as off the masks and be dropped."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    for *_, data in model.graph.edges(keys=True, data=True):
        data.pop(graph_module.SMALL_MASK_VESSEL_TYPE, None)
    drawn = add_vessel_between(model.graph, ids["p"], ids["f"])  # in the mask

    with caplog.at_level("WARNING", logger="haemolynx.pipeline.stages"):
        post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert _boundary_lists(run_settings, boundaries, ids)["arteriole_boundary_nodes"] == [
        "a1", "s1",
    ]
    assert graph_module.SMALL_MASK_VESSEL_TYPE not in model.graph.edges[drawn]
    assert "no vessel carries the small-mask tag" in caplog.text


def test_the_boundaries_stage_tags_the_vessels_its_boundary_nodes_come_from(tmp_path):
    """The tags are the labelling the boundary nodes were read from: read
    back by the same rule, they give the same nodes. A run whose boundary
    nodes do not come from the small masks clears any a resumed graph kept."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(
        tmp_path, side_arteriole=True, small_vessel_boundary_fallback_to_hop_distance=False
    )
    G = model.graph
    tags = {
        frozenset((u, v)): data[graph_module.SMALL_MASK_VESSEL_TYPE]
        for u, v, data in G.edges(data=True)
        if graph_module.SMALL_MASK_VESSEL_TYPE in data
    }
    assert tags == {
        frozenset((ids["a1"], ids["s1"])): "arteriole",
        frozenset((ids["p"], ids["f"])): "arteriole",
        frozenset((ids["w1"], ids["v1"])): "venule",
    }
    from_tags = graph_module.boundary_nodes_from_small_mask_tags(G)
    assert sorted(from_tags["arteriole"]) == sorted(run_settings["arteriole_boundary_nodes"])
    assert sorted(from_tags["venule"]) == sorted(run_settings["venule_boundary_nodes"])

    run_settings["use_small_vessel_masks_for_boundary_assignment"] = False
    stages.assign_boundaries(run_settings, network)
    assert network.graph is G
    assert not any(graph_module.SMALL_MASK_VESSEL_TYPE in data for *_, data in G.edges(data=True))


@pytest.mark.parametrize("dilation_um", [0.0, 6.0])
def test_tagging_every_vessel_gives_the_boundaries_stage_labelling(tmp_path, dilation_um):
    """tag_edges_in_small_vessel_masks is the Boundaries stage's own rule --
    the threshold, the dilation steps, the gap fill: handed every vessel of
    an untagged graph, it tags them as the Boundaries stage labelled them."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(
        tmp_path, side_arteriole=True, small_vessel_mask_dilation_microns=dilation_um
    )
    G = model.graph
    expected = {
        (u, v, k): data.pop(graph_module.SMALL_MASK_VESSEL_TYPE, None)
        for u, v, k, data in G.edges(keys=True, data=True)
    }
    assert sum(tag is not None for tag in expected.values()) >= 3

    counts = graph_module.tag_edges_in_small_vessel_masks(
        G,
        boundaries.small_arteriole_mask,
        boundaries.small_venule_mask,
        list(expected),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        minimum_overlap_fraction=run_settings["small_vessel_mask_min_overlap_fraction"],
        max_dilation_microns=dilation_um,
    )

    got = {
        (u, v, k): data.get(graph_module.SMALL_MASK_VESSEL_TYPE)
        for u, v, k, data in G.edges(keys=True, data=True)
    }
    assert got == expected
    assert counts["arteriole_edge_count"] == sum(tag == "arteriole" for tag in expected.values())


# --- a drawn vessel read against the segmentation ---------------------------


def _segmentation_along(G, shape, radius_vox=1, *, wide=(), wide_radius_vox=3):
    """A mask of *shape* with a tube of *radius_vox* (Chebyshev) round every
    vessel's path (walked every half micron: a straight vessel is stored as
    its two ends) -- *wide_radius_vox* round the vessels in *wide* -- on
    the fixture's 1 um grid."""
    from haemolynx.preprocessing.bridge_mask_support import _densify

    mask = np.zeros(shape, dtype=bool)
    for u, v, data in G.edges(data=True):
        r = wide_radius_vox if (u, v) in wide or (v, u) in wide else radius_vox
        for point in _densify(np.asarray(data["voxels"], dtype=float), 0.5):
            z, y, x = (int(round(c)) for c in point)
            mask[
                max(z - r, 0): z + r + 1, max(y - r, 0): y + r + 1, max(x - r, 0): x + r + 1
            ] = True
    return mask


def _drawn_records(out):
    return out.results["post_process"]["drawn_vessels"]


def test_a_vessel_drawn_through_the_segmentation_is_recorded_as_all_in_it(tmp_path):
    """The reader graph building judges its bridges with, applied to each
    vessel a hand edit drew: p-f lies along the small arteriole's mask, so
    all of it is in the segmentation, no background is crossed, and no
    other centreline runs beside it in its lumen."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    drawn = add_vessel_between(model.graph, ids["p"], ids["f"])
    network.volume.image = _segmentation_along(model.graph, network.volume.image.shape)

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[drawn]
    assert data[graph_module.DRAWN_MASK_FRACTION] == pytest.approx(1.0)
    assert data[graph_module.DRAWN_BACKGROUND_RUN_UM] == pytest.approx(0.0)
    assert data[graph_module.DRAWN_BESIDE_VESSEL] is False
    (record,) = _drawn_records(out)
    assert record == {
        "edge": drawn, "length_um": pytest.approx(9.0), "mask_fraction": pytest.approx(1.0),
        "background_run_um": pytest.approx(0.0), "beside_vessel": False,
        "within_bridge_gate": True,
    }
    assert "drawn_vessels" in out.results["post_process"]
    # The halves and tags the edit also touched are not vessels anybody drew.
    for edge, other in out.graph.edges.items():
        if edge != drawn:
            assert graph_module.DRAWN_MASK_FRACTION not in other


def test_a_vessel_drawn_across_background_records_the_run_and_is_kept(tmp_path, caplog):
    """p-f drawn where the segmentation shows no vessel between the large
    arteriole's tube and the capillary's: 6 um of background in one run,
    a third of it in the mask, which graph building's bridge gate would
    have refused -- recorded and said, and the vessel kept as drawn."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    network.volume.image = _segmentation_along(model.graph, network.volume.image.shape)
    drawn = add_vessel_between(model.graph, ids["p"], ids["f"])

    with caplog.at_level("INFO", logger="haemolynx.pipeline.stages"):
        out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[drawn]
    assert out.graph.has_edge(*drawn) and data["branch_order"]
    assert 0.25 < data[graph_module.DRAWN_MASK_FRACTION] < 0.45
    assert data[graph_module.DRAWN_BACKGROUND_RUN_UM] == pytest.approx(6.0, abs=1.0)
    assert data[graph_module.DRAWN_BESIDE_VESSEL] is False
    (record,) = _drawn_records(out)
    assert record["edge"] == drawn and record["within_bridge_gate"] is False
    assert "1 partly off it" in caplog.text
    assert "of background in one run at most" in caplog.text


def test_a_vessel_drawn_beside_a_vessel_in_its_lumen_says_so(tmp_path):
    """A second strand drawn 2 um off the large arteriole's centreline,
    inside its 7-voxel-wide tube from a1 to a3: all in the segmentation,
    and beside a vessel the network already has."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    G = model.graph
    a1, a3 = G.nodes[ids["a1"]]["pos"], G.nodes[ids["a3"]]["pos"]
    network.volume.image = _segmentation_along(
        G, network.volume.image.shape,
        wide=((ids["a1"], ids["a2"]), (ids["a2"], ids["p"]), (ids["p"], ids["a3"])),
    )
    path = [a1, (a1[0] + 1.0, 4.0, 7.0), (a3[0] - 1.0, 4.0, 7.0), a3]
    drawn = add_vessel_between(G, ids["a1"], ids["a3"], path)

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[drawn]
    assert data[graph_module.DRAWN_MASK_FRACTION] == pytest.approx(1.0)
    assert data[graph_module.DRAWN_BESIDE_VESSEL] is True
    (record,) = _drawn_records(out)
    assert record["beside_vessel"] is True and record["within_bridge_gate"] is True
    # The strand it runs beside is not marked: it is the vessel the run built.
    assert graph_module.DRAWN_BESIDE_VESSEL not in out.graph.edges[ids["a2"], ids["p"], 0]


def test_without_a_segmentation_no_drawn_vessel_is_read(tmp_path):
    """A synthetic run has no mask: the attributes are not written and the
    summary has no list, rather than a vessel read against nothing."""
    run_settings, network, boundaries, model, ids = _small_boundary_run(tmp_path)
    drawn = add_vessel_between(model.graph, ids["p"], ids["f"])
    network.volume.image = None

    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert graph_module.DRAWN_MASK_FRACTION not in out.graph.edges[drawn]
    assert "drawn_vessels" not in out.results["post_process"]


def test_a_resume_hands_the_small_masks_the_labelling_read_on(tmp_path):
    """The panel's Regenerate on the Post processing tab resumes the run:
    its boundaries come from the resume, which carries the small masks as
    the Boundaries stage's labelling read them, never the network's own."""
    _settings, network, *_ = _small_boundary_run(tmp_path)
    cleaned = np.zeros_like(network.small_arteriole_mask)
    resume = PipelineResume(
        start_from="post_process",
        graph=network.graph,
        small_arteriole_mask=cleaned,
        small_venule_mask=cleaned,
    )

    boundaries = stages._boundaries_from_resume(resume, network)

    assert boundaries.small_arteriole_mask is cleaned
    assert boundaries.small_venule_mask is cleaned
    without = stages._boundaries_from_resume(PipelineResume(start_from="post_process"), network)
    assert without.small_arteriole_mask is None


def test_a_diameter_set_by_hand_survives_post_processing_and_moves_the_resistance(tmp_path, monkeypatch):
    """Set diameter on the review lists: the vessel is not measured again --
    a measurement would replace the hand value -- and its resistance follows
    the new width when the haemodynamics is built again."""
    run_settings, network, boundaries, model = _diameters_run(
        tmp_path, use_edt_diameter_crosscheck=True,
        edt_diameter_prefer_over_table_on_fwhm_failure=True,
    )
    model = build_haemodynamic_model(run_settings, model, SCHEMA)
    G = model.graph
    u, v, k = next(iter(G.edges(keys=True)))
    G.edges[u, v, k]["edt_diameter_um"] = 7.5  # a measurement it could fall back on
    before = G.edges[u, v, k]["resistance"]
    width = 2.0 * G.edges[u, v, k]["diameter_um"]
    measured: list = []
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply._measure_edt_diameters",
        lambda G, _config, *, mask_volume=None, edges=None: measured.append(edges) or {},
    )

    set_vessel_diameter(G, [(u, v, k)], width)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    data = out.graph.edges[u, v, k]
    assert measured == []
    assert data["diameter_source"] == "override"
    assert data["diameter_um"] == pytest.approx(width)
    out = build_haemodynamic_model(run_settings, out, SCHEMA)
    assert out.graph.edges[u, v, k]["resistance"] < before / 8.0


def test_the_region_found_again_is_kept_for_the_next_post_process(tmp_path, monkeypatch):
    """Finding the region in the segmentation is minutes on a real stack, so
    the one found is kept beside the run's skeleton (thick_vessel_region_cache)
    and a later post_process on the same segmentation reads it back."""
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path, volume_mask=False)
    real = stages.preprocessing.thick_vessel_region
    found = []

    def spy(*args, **kwargs):
        found.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(stages.preprocessing, "thick_vessel_region", spy)
    post_process(run_settings, model, boundaries, SCHEMA, network=network)
    assert found == [1]
    cache = stages.thick_vessel_region_cache(run_settings, network.volume.output_dir)
    assert cache.sidecar.exists()

    again = _thick_run(tmp_path, volume_mask=False)
    post_process(again[0], again[3], again[2], SCHEMA, network=again[1])
    assert found == [1]
