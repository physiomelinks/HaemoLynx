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

#: The thick vessel fills x >= THICK_FROM_UM of a 1-um grid.
THICK_FROM_UM = 30.0


def _thick_network():
    """A thin vessel 0-1 (x 0..20) and a thick one's centreline 2-3 (x 40..55)."""
    G = nx.MultiGraph()
    positions = {0: (5.0, 5.0, 0.0), 1: (5.0, 5.0, 20.0), 2: (5.0, 5.0, 40.0), 3: (5.0, 5.0, 55.0)}
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for a, b in ((0, 1), (2, 3), (1, 2)):
        voxels = [positions[a], positions[b]]
        if (a, b) == (1, 2):
            continue  # the vessel drawn by hand, below
        G.add_edge(a, b, voxels=voxels, length=calculate_path_length(voxels), branch_order="B01")
    return G


def _thick_mask():
    mask = np.zeros((10, 10, 60), dtype=bool)
    mask[:, :, int(THICK_FROM_UM):] = True
    return mask


def _thick_run(tmp_path, *, volume_mask=True, **settings):
    options = {
        "use_thick_vessel_skeletonisation": True,
        "inlet_nodes": [0],
        "outlet_nodes": [3],
        "arteriole_boundary_nodes": [0],
        "venule_boundary_nodes": [3],
        **settings,
    }
    run_settings = _settings(tmp_path, **options)
    G = _thick_network()
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

    pieces = _pieces_between(out.graph, 1, 2)
    assert len(pieces) == 2, "split once, where it enters the thick vessel"
    (a, joint, outside), (_joint, b, inside) = pieces
    assert not outside.get(IS_ZERO_RESISTANCE)
    assert inside.get(IS_ZERO_RESISTANCE)
    joint_x = float(out.graph.nodes[joint]["pos"][2])
    assert abs(joint_x - THICK_FROM_UM) <= 1.0
    assert outside["length"] + inside["length"] == pytest.approx(20.0)
    for piece in (outside, inside):
        assert piece["branch_order"] and piece["diameter_um"] > 0
        assert EDITED not in piece
    # The thick vessel's own centreline, entirely inside, is no bridge.
    (key,) = out.graph[2][3]
    assert not out.graph.edges[2, 3, key].get(IS_ZERO_RESISTANCE)
    assert out.graph is model.graph is boundaries.graph is network.graph


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

    assert len(_pieces_between(out.graph, 1, 2)) == 2


def test_without_a_region_to_hand_it_is_found_again_in_the_segmentation(tmp_path, monkeypatch):
    run_settings, network, boundaries, model, _edge = _thick_run(tmp_path, volume_mask=False)
    asked: list = []

    def found_again(settings, image, voxel_size_xyz):
        asked.append((image is network.volume.image, tuple(voxel_size_xyz)))
        return _thick_mask()

    monkeypatch.setattr(stages, "thick_vessel_mask_from_image", found_again)
    out = post_process(run_settings, model, boundaries, SCHEMA, network=network)

    assert asked == [(True, (1.0, 1.0, 1.0))]
    assert len(_pieces_between(out.graph, 1, 2)) == 2


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


def test_an_unedited_network_is_not_solved_again(tmp_path, monkeypatch):
    """Continue without an edit: the solve the pause made is the run's."""
    _stub_early_stages(monkeypatch, tmp_path)
    paused = run_pipeline_stages(_settings(tmp_path), SCHEMA, stop_after="solve")

    def refuse(*_args, **_kwargs):
        raise AssertionError("an unedited network must not be solved again")

    monkeypatch.setattr(stages, "solve", refuse)
    monkeypatch.setattr(stages, "build_haemodynamic_model", refuse)
    outputs: dict = {}
    run_pipeline_stages(
        _settings(tmp_path),
        SCHEMA,
        start_from="post_process",
        resume=_resume_from_post_processing(copy.deepcopy(paused)),
        on_stage_output=outputs.__setitem__,
    )

    assert outputs["post_process"].post_processed is None


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
