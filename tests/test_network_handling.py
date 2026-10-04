"""Network handling: what is done to the assigned network before its model.

The "Network handling" settings sit on the "6. Haemodynamics" tab, and run at
the start of that stage (``apply_network_handling``) rather than at the end of
boundary assignment -- so changing one there and choosing "Run from this
stage" actually takes effect.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from haemolynx import graph as graph_module
from haemolynx.gui.tabs import assign_to_stages, tabs_for
from haemolynx.pipeline import apply_network_handling, default_schema, stages
from haemolynx.pipeline.progress import STAGES
from haemolynx.pipeline.stages import (
    BoundaryNodes,
    HaemodynamicModel,
    PipelineResume,
    VesselNetwork,
    run_pipeline_stages,
)

SCHEMA = default_schema()
#: The checkbox that "Remove disconnected and dead-end branches/trees" replaced.
OLD_PRUNE = "remove_disconnected_io_components_after_final_assignment"
REMOVE = {"boundary_handling": "remove_disconnected"}
LEAVE = {"boundary_handling": "leave_unsolved"}


def _two_components() -> nx.MultiGraph:
    """0-1-2 runs from inlet 0 to outlet 2; 10-11 has an inlet and no outlet."""
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(10, 11)
    return graph


def _boundaries(graph, **extra) -> BoundaryNodes:
    return BoundaryNodes(
        inlet_nodes=[10, 0],
        outlet_nodes=[2],
        arteriole_boundary_nodes=[1, 11],
        venule_boundary_nodes=[1],
        resistance_node_pair=(10, 2),
        graph=graph,
        **extra,
    )


# --- where the settings are ------------------------------------------------


def test_boundary_handling_is_in_the_network_handling_section():
    assert SCHEMA["boundary_handling"].section == "Network handling"


def test_boundary_handling_leaves_unsolved_or_removes():
    from haemolynx.gui.form import widget_type_for

    setting = SCHEMA["boundary_handling"]
    assert setting.kind == "choice"
    assert tuple(setting.choices) == ("leave_unsolved", "remove_disconnected")
    # Off by default, as the checkbox it replaced was.
    assert setting.default == "leave_unsolved"
    assert setting.requires == ()
    assert setting.advanced
    assert widget_type_for(setting) == "ComboBox"


def test_the_drop_down_reads_as_sentences():
    from haemolynx.gui.form import field_for

    field = field_for(SCHEMA["boundary_handling"])
    assert field.options["choices"] == [
        ("Leave unsolved", "leave_unsolved"),
        ("Remove disconnected and dead-end branches/trees", "remove_disconnected"),
    ]


def test_the_prune_toggle_is_now_a_choice_of_the_drop_down():
    assert OLD_PRUNE not in SCHEMA
    assert OLD_PRUNE in SCHEMA["boundary_handling"].replaces


def test_network_handling_is_the_last_box_on_the_haemodynamics_tab():
    owner = assign_to_stages(SCHEMA)
    assert owner["boundary_handling"] == "6. Haemodynamics"
    (tab,) = [tab for tab in tabs_for(SCHEMA) if tab.stage.title == "6. Haemodynamics"]
    sections = list(dict.fromkeys(SCHEMA[row.name].section for row in tab.fields))
    assert sections[-1] == "Network handling"
    names = [row.name for row in tab.fields if SCHEMA[row.name].section == "Network handling"]
    assert names == ["boundary_handling"]


def test_nothing_about_pruning_is_on_the_boundaries_tab():
    from haemolynx.gui.form import visible_vessel_mask_settings

    shown = visible_vessel_mask_settings(SCHEMA, {"automated_vessel_assignment": True})
    assert OLD_PRUNE not in shown and "boundary_handling" not in shown


# --- configs written before the drop-down ---------------------------------------


def _load(tmp_path, text: str) -> dict:
    from haemolynx.parsers import load_config

    path = tmp_path / "old.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path, SCHEMA)


def test_an_old_config_with_the_toggle_under_vessel_masks_still_loads(tmp_path):
    loaded = _load(tmp_path, f"vessel_masks:\n  {OLD_PRUNE}: true\n")

    assert loaded["boundary_handling"] == "remove_disconnected"
    assert OLD_PRUNE not in loaded


@pytest.mark.parametrize(
    "lines, expected",
    [
        # Saved while the checkbox sat beside a drop-down that only said None.
        ([f"{OLD_PRUNE}: true", "boundary_handling: None"], "remove_disconnected"),
        ([f"{OLD_PRUNE}: false", "boundary_handling: None"], "leave_unsolved"),
        (["boundary_handling: None"], "leave_unsolved"),
        ([f"{OLD_PRUNE}: false"], "leave_unsolved"),
        # A choice the file made itself is not overridden by the old toggle.
        ([f"{OLD_PRUNE}: false", "boundary_handling: remove_disconnected"], "remove_disconnected"),
    ],
)
def test_an_old_network_handling_box_loads_as_the_drop_down(tmp_path, lines, expected):
    text = "network_handling:\n" + "".join(f"  {line}\n" for line in lines)

    assert _load(tmp_path, text)["boundary_handling"] == expected


def test_an_old_saved_run_upgrades_before_reaching_the_form():
    """A .haemorun's settings go straight into the panel's rows, which only
    hold today's names and values."""
    saved = {OLD_PRUNE: True, "boundary_handling": "None", "inlet_p_bc": 5.0}

    assert SCHEMA.upgrade(saved) == {"boundary_handling": "remove_disconnected", "inlet_p_bc": 5.0}
    assert SCHEMA.upgrade({"boundary_handling": "None"}) == {"boundary_handling": "leave_unsolved"}


# --- what the pruning does ----------------------------------------------------


@pytest.mark.parametrize("settings", [LEAVE, {}], ids=["leave_unsolved", "unset"])
def test_leave_unsolved_leaves_the_network_alone(settings):
    graph = _two_components()
    model = HaemodynamicModel(graph=graph)
    boundaries = _boundaries(graph)

    result = apply_network_handling(dict(settings), model, boundaries)

    assert result is model and model.graph is graph
    assert boundaries.inlet_nodes == [10, 0]


def test_remove_drops_components_without_both_an_inlet_and_an_outlet():
    graph = _two_components()
    model = HaemodynamicModel(graph=graph)
    network = VesselNetwork(graph=graph, volume=None)
    boundaries = _boundaries(graph)
    settings = {
        **REMOVE,
        # assign_boundaries hands these very lists over; a resume copies them.
        "inlet_nodes": boundaries.inlet_nodes,
        "outlet_nodes": [2],
        "arteriole_boundary_nodes": [1, 11],
    }

    apply_network_handling(settings, model, boundaries, network)

    assert sorted(model.graph.nodes) == [0, 1, 2]
    assert boundaries.graph is model.graph and network.graph is model.graph
    assert boundaries.inlet_nodes == [0]
    assert boundaries.arteriole_boundary_nodes == [1]
    assert settings["inlet_nodes"] == [0]
    assert settings["arteriole_boundary_nodes"] == [1]
    # The pair named a node that is gone, so it is chosen again.
    assert boundaries.resistance_node_pair == (0, 2)


def test_a_pair_still_in_the_network_is_kept():
    graph = _two_components()
    boundaries = _boundaries(graph)
    boundaries.resistance_node_pair = (0, 2)

    apply_network_handling(dict(REMOVE), HaemodynamicModel(graph=graph), boundaries)

    assert boundaries.resistance_node_pair == (0, 2)


def test_nothing_left_to_solve_is_an_error():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(10, 11)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[11], graph=graph)

    with pytest.raises(ValueError, match="no valid boundary nodes"):
        apply_network_handling(dict(REMOVE), HaemodynamicModel(graph=graph), boundaries)


# --- dead ends off a perfused network -------------------------------------------


def _vessels(graph) -> list[tuple]:
    return sorted((min(u, v), max(u, v), k) for u, v, k in graph.edges(keys=True))


def _remove(graph, inlets, outlets, **lists) -> tuple[HaemodynamicModel, BoundaryNodes]:
    model = HaemodynamicModel(graph=graph)
    boundaries = BoundaryNodes(inlet_nodes=list(inlets), outlet_nodes=list(outlets),
                               graph=graph, **lists)
    apply_network_handling(dict(REMOVE), model, boundaries)
    return model, boundaries


def _chain(*nodes) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    nx.add_path(graph, nodes, length=10.0)
    return graph


def test_remove_drops_a_dead_end_tree_off_a_perfused_path():
    """The reported case: a tree leaving a perfused vessel and reaching no
    outlet sits in the inlet-outlet component, so a component test kept it."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(1, 4), (4, 5), (4, 6), (6, 7)], length=10.0)

    model, _boundaries = _remove(graph, [0], [3])

    assert _vessels(model.graph) == [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
    assert sorted(model.graph.nodes) == [0, 1, 2, 3]
    # The graph handed in -- a stage checkpoint, on a rerun -- is untouched.
    assert graph.number_of_edges() == 7


def test_remove_drops_a_dead_end_loop_hanging_off_one_node():
    """A loop joined to the rest at node 2 alone, its stem 4-5, and a
    self-loop: every pressure on them equals node 2's (or 1's), so no flow."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(2, 4), (4, 5), (5, 6), (6, 7), (7, 5), (1, 1)], length=10.0)

    model, _boundaries = _remove(graph, [0], [3])

    assert _vessels(model.graph) == [(0, 1, 0), (1, 2, 0), (2, 3, 0)]


def test_remove_keeps_loops_on_the_inlet_to_outlet_route():
    """Parallel vessels and a detour both carry flow between inlet and outlet."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edge(1, 2, length=12.0)  # a parallel vessel: key 1
    graph.add_edges_from([(1, 4), (4, 2)], length=10.0)

    model, _boundaries = _remove(graph, [0], [3])

    assert model.graph is graph
    assert _vessels(model.graph) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (1, 4, 0), (2, 3, 0),
                                     (2, 4, 0)]


def test_remove_keeps_every_inlet_and_outlet_branch_of_several():
    """Inlets 0 and 4, outlets 3 and 6: each one's branch is kept, a dead end
    off an inlet's branch (4-8) or an outlet's (5-7) is not."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(4, 1), (2, 5), (5, 6), (5, 7), (4, 8)], length=10.0)

    model, boundaries = _remove(graph, [0, 4], [3, 6])

    assert _vessels(model.graph) == [(0, 1, 0), (1, 2, 0), (1, 4, 0), (2, 3, 0), (2, 5, 0),
                                     (5, 6, 0)]
    assert boundaries.inlet_nodes == [0, 4] and boundaries.outlet_nodes == [3, 6]


def test_remove_still_drops_components_without_an_inlet_and_an_outlet():
    graph = _chain(0, 1, 2)
    graph.add_edges_from([(10, 11), (11, 12), (20, 21)], length=10.0)

    model, boundaries = _remove(graph, [0, 10], [2])

    assert _vessels(model.graph) == [(0, 1, 0), (1, 2, 0)]
    assert boundaries.inlet_nodes == [0]


def test_remove_reports_dead_ends_and_components_apart(caplog):
    import logging

    graph = _chain(0, 1, 2)
    graph.add_edges_from([(1, 3), (3, 4)], length=15.0)  # dead end: 2 vessels, 30 um
    graph.add_edge(10, 11, length=7.0)  # a component with an inlet only

    with caplog.at_level(logging.INFO, logger="haemolynx"):
        _remove(graph, [0, 10], [2])

    (message,) = [r.getMessage() for r in caplog.records if "remove_disconnected" in r.getMessage()]
    assert "1 disconnected component(s)" in message
    assert "(1 vessel(s), 2 node(s), 7.0 um)" in message
    assert "2 dead-end vessel(s) that reach no outlet (2 node(s), 30.0 um)" in message
    assert "2 vessel(s) and 3 node(s) remain" in message


def test_the_removal_counts_what_it_removed_without_touching_the_graph():
    graph = _chain(0, 1, 2)
    graph.add_edges_from([(1, 3), (3, 4)], length=15.0)
    graph.add_edge(1, 1, length=5.0)
    graph.add_edge(10, 11, length=7.0)
    graph.add_node(30)  # an isolated node: a component of its own

    pruned, stats = graph_module.remove_vessels_off_inlet_outlet_paths(graph, [0, 10], [2])

    assert _vessels(pruned) == [(0, 1, 0), (1, 2, 0)]
    assert stats == {
        "removed_components": 2,
        "removed_component_vessels": 1,
        "removed_component_nodes": 3,
        "removed_component_length_um": 7.0,
        "removed_dead_end_vessels": 3,
        "removed_dead_end_nodes": 2,
        "removed_dead_end_length_um": 35.0,
        "remaining_nodes": 3,
        "remaining_vessels": 2,
    }
    assert graph.number_of_edges() == 6 and graph.number_of_nodes() == 8


def test_boundary_lists_follow_the_dead_ends_removed():
    """A boundary node on a dead end goes from every list, on *boundaries*
    and in the settings both; the resistance pair is chosen again."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(1, 4), (4, 5)], length=10.0)
    model = HaemodynamicModel(graph=graph)
    network = VesselNetwork(graph=graph, volume=None)
    boundaries = BoundaryNodes(
        inlet_nodes=[0],
        outlet_nodes=[3],
        arteriole_boundary_nodes=[1, 4],
        venule_boundary_nodes=[2, 5],
        large_arteriole_boundary_nodes=[4],
        large_venule_boundary_nodes=[2],
        resistance_node_pair=(0, 5),
        graph=graph,
    )
    settings = {
        **REMOVE,
        "inlet_nodes": boundaries.inlet_nodes,
        "arteriole_boundary_nodes": [1, 4],
        "large_arteriole_boundary_nodes": [4],
    }

    apply_network_handling(settings, model, boundaries, network)

    assert boundaries.graph is model.graph and network.graph is model.graph
    assert boundaries.arteriole_boundary_nodes == [1]
    assert boundaries.venule_boundary_nodes == [2]
    assert boundaries.large_arteriole_boundary_nodes == []
    assert boundaries.large_venule_boundary_nodes == [2]
    assert settings["arteriole_boundary_nodes"] == [1]
    assert settings["large_arteriole_boundary_nodes"] == []
    assert boundaries.resistance_node_pair == (0, 3)


def test_the_kept_edges_with_dead_ends_are_exactly_what_the_removal_keeps():
    graph = _chain(0, 1, 2, 3)
    graph.add_edge(1, 2)  # a parallel vessel keeps its own key
    graph.add_edges_from([(2, 4), (4, 5), (5, 6), (6, 4), (3, 3), (20, 21)])
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], graph=graph)

    kept = stages.edges_network_handling_keeps(dict(REMOVE), graph, boundaries)
    pruned, _stats = graph_module.remove_vessels_off_inlet_outlet_paths(graph, [0], [3])

    assert kept == list(pruned.edges(keys=True))
    assert _vessels(pruned) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
    assert stages.edges_network_handling_keeps(dict(LEAVE), graph, boundaries) is None


def test_leave_unsolved_keeps_dead_ends():
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(1, 4), (4, 5), (2, 6), (6, 7), (7, 2)], length=10.0)
    model = HaemodynamicModel(graph=graph)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], graph=graph)

    apply_network_handling(dict(LEAVE), model, boundaries)

    assert model.graph is graph and graph.number_of_edges() == 8


# --- it runs in the haemodynamics stage ---------------------------------------


def _stub_stages(monkeypatch):
    seen: dict[str, object] = {}

    def stub(name):
        def call(*args, **_kwargs):
            if name == "build_haemodynamic_model":
                seen["model_graph"] = args[1].graph
                return args[1]
            if name == "solve":
                seen["solve_boundaries"] = args[2]
            return SimpleNamespace(
                name=f"{name}-result", graph=None,
                large_arteriole_mask=None, large_venule_mask=None,
            )

        return call

    for stage in STAGES:
        if stage.call:
            monkeypatch.setattr(stages, stage.call, stub(stage.call))
    return seen


def test_run_from_the_haemodynamics_tab_applies_the_pruning(monkeypatch):
    """The point of moving it: the boundaries checkpoint is unpruned, and a
    rerun from Haemodynamics must still prune it."""
    seen = _stub_stages(monkeypatch)
    graph = _two_components()
    resume = PipelineResume(
        start_from="build_haemodynamic_model",
        graph=graph,
        inlet_nodes=(10, 0),
        outlet_nodes=(2,),
    )

    run_pipeline_stages(
        dict(REMOVE),
        schema=None,
        start_from="build_haemodynamic_model",
        resume=resume,
    )

    assert sorted(seen["model_graph"].nodes) == [0, 1, 2]
    assert seen["solve_boundaries"].inlet_nodes == [0]


# --- the diameters stage calibrates on what will be kept ---------------------


def test_the_kept_edges_are_exactly_what_the_removal_keeps():
    graph = _two_components()
    graph.add_edge(1, 2)  # a parallel vessel keeps its own key
    graph.add_edge(20, 21)  # an outlet and no inlet

    kept = graph_module.edges_with_connected_io(graph, [10, 0], [2, 21])
    pruned, _stats = graph_module.remove_components_without_connected_io(graph, [10, 0], [2, 21])

    assert sorted(kept) == sorted(pruned.edges(keys=True)) == [(0, 1, 0), (1, 2, 0), (1, 2, 1)]


def _diameters_stage(tmp_path, monkeypatch, handling):
    """The diameters stage on a chain from inlet to outlet plus a stray vessel
    with neither, recording which vessels it asks the measurements to
    calibrate on."""
    from test_diameter_assignment import _boundaries, _network, _settings, _vessel_network

    graph = _network()
    for node, z in ((90, 0.0), (91, 50.0)):
        graph.add_node(node, pos=[z, 100.0, 0.0])
    graph.add_edge(90, 91, key=0, branch_order="B01", length=50.0,
                   voxels=[[0.0, 100.0, 0.0], [50.0, 100.0, 0.0]])
    calibrated = []

    def fake_assign(G, _config, *, mask_volume=None, calibration_edges=None):
        calibrated.append(calibration_edges)
        return G, {}, None

    monkeypatch.setattr(stages, "assign_edge_diameters", fake_assign)
    model = stages.assign_diameters(
        _settings(tmp_path, **handling), _vessel_network(tmp_path, graph), _boundaries(), SCHEMA
    )
    return model, calibrated


def test_the_diameters_stage_calibrates_on_the_vessels_remove_disconnected_keeps(
    tmp_path, monkeypatch
):
    """The FWHM lines over a stray segmented blob: its vessel was measured,
    then removed at the start of the haemodynamics stage -- after its widths
    had joined the decoy check and the blur estimate the kept vessels'
    widths were fitted with."""
    model, calibrated = _diameters_stage(tmp_path, monkeypatch, REMOVE)

    assert sorted(calibrated[0]) == [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]
    # The removal itself stays where it was, at the start of Haemodynamics.
    assert model.graph.has_edge(90, 91)


def test_leave_unsolved_calibrates_on_every_vessel(tmp_path, monkeypatch):
    model, calibrated = _diameters_stage(tmp_path, monkeypatch, LEAVE)

    assert calibrated == [None]
    assert model.graph.has_edge(90, 91)


# --- haematocrit_junction_rule=split_junctions ---------------------------------

SPLIT = {
    "run_haemodynamics": True,
    "haematocrit_model": "distributed_iterative",
    "haematocrit_junction_rule": "split_junctions",
}


def _four_way() -> nx.MultiGraph:
    """Inlet 0 -> junction 1 -> outlets 2, 3, 4; 3 and 4 leave 1 close together."""
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 50.0),
        2: (0.0, 60.0, 80.0),
        3: (0.0, -10.0, 110.0),
        4: (0.0, -40.0, 100.0),
    }
    graph = nx.MultiGraph()
    for node, pos in positions.items():
        graph.add_node(node, pos=pos)
    for v, diameter in zip((1, 2, 3, 4), (15.0, 12.0, 8.0, 10.0)):
        u = 0 if v == 1 else 1
        graph.add_edge(u, v, diameter_um=diameter, branch_order="B01", length=50.0)
    return graph


def test_split_junctions_splits_every_four_way_on_a_copy():
    graph = _four_way()
    model = HaemodynamicModel(graph=graph)
    network = VesselNetwork(graph=graph, volume=None)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3, 4], graph=graph)

    apply_network_handling(dict(SPLIT), model, boundaries, network)

    assert model.graph is not graph
    assert boundaries.graph is model.graph and network.graph is model.graph
    assert max(degree for _node, degree in model.graph.degree()) == 3
    assert model.graph.number_of_edges() == 5
    # The graph handed in -- a stage checkpoint, on a rerun -- is untouched.
    assert graph.degree(1) == 4 and graph.number_of_edges() == 4
    assert boundaries.inlet_nodes == [0] and boundaries.outlet_nodes == [2, 3, 4]


@pytest.mark.parametrize(
    "settings",
    [
        {**SPLIT, "haematocrit_junction_rule": "no_separation"},
        {**SPLIT, "haematocrit_junction_rule": "sequential_bifurcations"},
        # The rule belongs to the distributed model; a fixed haematocrit has
        # nothing for it to decide.
        {**SPLIT, "haematocrit_model": "fixed"},
        {**SPLIT, "run_haemodynamics": False},
    ],
)
def test_only_split_junctions_with_distributed_haematocrit_changes_the_network(settings):
    graph = _four_way()
    model = HaemodynamicModel(graph=graph)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3, 4], graph=graph)

    apply_network_handling(settings, model, boundaries)

    assert model.graph is graph and graph.degree(1) == 4


@pytest.mark.parametrize(
    "setting, expected",
    [
        # 0, the default: the junction's mean diameter, (15 + 12 + 8 + 10) / 4.
        (0.0, 11.25),
        (20.0, 20.0),
    ],
)
def test_the_connector_length_setting_reaches_the_split(setting, expected):
    graph = _four_way()
    model = HaemodynamicModel(graph=graph)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3, 4], graph=graph)

    apply_network_handling(
        {**SPLIT, "haematocrit_split_connector_length_um": setting}, model, boundaries
    )

    (connector,) = [
        data for _u, _v, data in model.graph.edges(data=True) if data.get("junction_split_connector")
    ]
    assert connector["length"] == pytest.approx(expected)


def test_the_connector_length_only_shows_with_split_junctions():
    setting = SCHEMA["haematocrit_split_connector_length_um"]
    assert "haematocrit_junction_rule=split_junctions" in setting.requires
    assert setting.default == 0.0 and setting.unit == "um" and setting.advanced


def test_split_junctions_leaves_a_network_of_bifurcations_alone():
    graph = _four_way()
    graph.remove_edge(1, 4)
    model = HaemodynamicModel(graph=graph)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3], graph=graph)

    apply_network_handling(dict(SPLIT), model, boundaries)

    assert model.graph is graph


def test_run_from_the_haemodynamics_tab_splits_the_junctions(monkeypatch):
    """The checkpoint a rerun resumes from is unsplit; the split happens in
    this stage, so choosing split_junctions there and rerunning takes effect."""
    seen = _stub_stages(monkeypatch)
    graph = _four_way()
    resume = PipelineResume(
        start_from="build_haemodynamic_model",
        graph=graph,
        inlet_nodes=(0,),
        outlet_nodes=(2, 3, 4),
    )

    run_pipeline_stages(
        dict(SPLIT),
        schema=None,
        start_from="build_haemodynamic_model",
        resume=resume,
    )

    assert max(degree for _node, degree in seen["model_graph"].degree()) == 3
    assert graph.degree(1) == 4


def test_assign_boundaries_no_longer_prunes():
    """It used to prune at its own end; the step now belongs to Haemodynamics."""
    import inspect

    source = inspect.getsource(stages.assign_boundaries)
    assert OLD_PRUNE not in source and "boundary_handling" not in source


# --- leave_unsolved: the solve marks what it could not solve -------------------


def _bifurcation_and_strays(dead_ends: bool = False) -> nx.MultiGraph:
    """0 (inlet) -> 1 -> {2, 3} (outlets); 10-11-12 has inlet 10 and no
    outlet; 20-21 has no boundary at all. Real diameters, lengths in um.

    With *dead_ends*, also a dead-end branch 1-4-5 and a loop 2-6-7-2
    joined to the rest at node 2 alone."""
    import numpy as np

    graph = nx.MultiGraph()
    positions = {
        0: (0, 0, 0), 1: (0, 0, 50), 2: (0, 40, 90), 3: (0, -40, 90),
        10: (0, 200, 0), 11: (0, 200, 50), 12: (0, 200, 100),
        20: (0, 400, 0), 21: (0, 400, 50),
    }
    vessels = [
        (0, 1, "B01", 15.0), (1, 2, "B02", 20.0), (1, 3, "B03", 8.0),
        (10, 11, "B02", 10.0), (11, 12, "B03", 8.0), (20, 21, "B03", 8.0),
    ]
    if dead_ends:
        positions.update({4: (0, 0, 100), 5: (0, 0, 150), 6: (0, 60, 130), 7: (0, 30, 140)})
        vessels += [
            (1, 4, "B03", 8.0), (4, 5, "B03", 8.0),
            (2, 6, "B03", 8.0), (6, 7, "B03", 8.0), (7, 2, "B03", 8.0),
        ]
    for node, pos in positions.items():
        graph.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v, order, diameter in vessels:
        length = float(np.linalg.norm(graph.nodes[u]["pos"] - graph.nodes[v]["pos"]))
        graph.add_edge(u, v, key=0, length=length, branch_order=order, diameter_um=diameter,
                       voxels=[list(graph.nodes[u]["pos"]), list(graph.nodes[v]["pos"])])
    graph.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    return graph


def _solve_with_strays(dead_ends: bool = False, **overrides):
    from haemolynx.pipeline import resolve_settings
    from haemolynx.pipeline.stages import build_haemodynamic_model, solve

    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "run_haemodynamics": True,
            "inlet_p_bc": 1000.0,
            "outlet_p_bc": 0.0,
            "inlet_nodes": [0, 10],
            "outlet_nodes": [2, 3],
            "all_diams_const": False,
            "diameter_by_branch_order": {"B01": 15.0, "B02": 20.0, "B03": 8.0},
        }
    )
    values.update(overrides)
    settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    graph = _bifurcation_and_strays(dead_ends)
    boundaries = BoundaryNodes(
        inlet_nodes=settings["inlet_nodes"],
        outlet_nodes=settings["outlet_nodes"],
        resistance_node_pair=(0, 2),
        graph=graph,
    )
    model = apply_network_handling(settings, HaemodynamicModel(graph=graph), boundaries)
    model = build_haemodynamic_model(settings, model, SCHEMA)
    return solve(settings, model, boundaries, SCHEMA)


def _solved_by_edge(graph) -> dict:
    from haemolynx.graph import FLOW_SOLVED

    return {(min(u, v), max(u, v)): data[FLOW_SOLVED] for u, v, data in graph.edges(data=True)}


@pytest.mark.parametrize("haematocrit_model", ["fixed", "distributed_iterative"])
def test_leave_unsolved_solves_what_it_can_and_marks_the_rest(haematocrit_model, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="haemolynx"):
        solution = _solve_with_strays(**LEAVE, haematocrit_model=haematocrit_model)

    assert _solved_by_edge(solution.graph) == {
        (0, 1): True, (1, 2): True, (1, 3): True,
        (10, 11): False, (11, 12): False, (20, 21): False,
    }
    # Every edge still carries a flow; the mark is what says which mean anything.
    assert all("flow_abs" in data for _u, _v, data in solution.graph.edges(data=True))
    assert solution.graph.edges[0, 1, 0]["flow_abs"] > 0
    assert any("3 of 6 vessel(s)" in record.getMessage() for record in caplog.records)


def test_remove_disconnected_leaves_nothing_unsolved():
    solution = _solve_with_strays(**REMOVE)

    assert _solved_by_edge(solution.graph) == {(0, 1): True, (1, 2): True, (1, 3): True}


@pytest.mark.parametrize("haematocrit_model", ["fixed", "distributed_iterative"])
def test_remove_disconnected_solves_the_network_without_its_dead_ends(haematocrit_model):
    solution = _solve_with_strays(dead_ends=True, **REMOVE, haematocrit_model=haematocrit_model)

    assert _solved_by_edge(solution.graph) == {(0, 1): True, (1, 2): True, (1, 3): True}
    inflow = solution.graph.edges[0, 1, 0]["flow_abs"]
    assert inflow > 0
    assert inflow == pytest.approx(
        solution.graph.edges[1, 2, 0]["flow_abs"] + solution.graph.edges[1, 3, 0]["flow_abs"]
    )


DEAD_ENDS = [(1, 4), (4, 5), (2, 6), (6, 7), (2, 7)]


@pytest.mark.parametrize("haematocrit_model", ["fixed", "distributed_iterative"])
def test_leave_unsolved_marks_dead_ends_unsolved(haematocrit_model, caplog):
    """A dead end on a component with an inlet and an outlet carries no flow
    either: it is marked unsolved, like a component without both."""
    import logging

    with caplog.at_level(logging.INFO, logger="haemolynx"):
        solution = _solve_with_strays(dead_ends=True, **LEAVE, haematocrit_model=haematocrit_model)

    assert _solved_by_edge(solution.graph) == {
        (0, 1): True, (1, 2): True, (1, 3): True,
        **{edge: False for edge in DEAD_ENDS},
        (10, 11): False, (11, 12): False, (20, 21): False,
    }
    inflow = solution.graph.edges[0, 1, 0]["flow_abs"]
    assert inflow > 0
    for u, v in DEAD_ENDS:
        assert solution.graph.edges[u, v, 0]["flow_abs"] == pytest.approx(0.0, abs=1e-9 * inflow)
    assert any("8 of 11 vessel(s) lie on no inlet-to-outlet path" in r.getMessage()
               for r in caplog.records)


def test_the_viewer_blanks_a_dead_ends_flow_columns():
    """The solve's mark is what the vessels layers read: a dead end is
    named Unsolved and left out of every flow-based colouring."""
    import numpy as np

    from haemolynx.gui.results import (
        FLOW_SOLUTION,
        flow_solution_values,
        mask_unsolved_flow_columns,
    )

    graph = _solve_with_strays(dead_ends=True, **LEAVE).graph
    edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    index = np.arange(len(edges))
    columns = {
        "edge_index": index,
        "flow_abs": [data["flow_abs"] for _u, _v, data in graph.edges(data=True)],
        "pressure_drop": [data["pressure_drop"] for _u, _v, data in graph.edges(data=True)],
        "length": [data["length"] for _u, _v, data in graph.edges(data=True)],
        FLOW_SOLUTION: flow_solution_values(graph, index),
    }
    mask_unsolved_flow_columns(columns)

    for row, edge in enumerate(edges):
        if edge in DEAD_ENDS:
            assert columns[FLOW_SOLUTION][row] == "Unsolved"
            assert np.isnan(columns["flow_abs"][row]) and np.isnan(columns["pressure_drop"][row])
        assert np.isfinite(columns["length"][row])
    perfused = [row for row, edge in enumerate(edges) if edge in {(0, 1), (1, 2), (1, 3)}]
    assert all(columns[FLOW_SOLUTION][row] == "Solved" for row in perfused)
    assert all(np.isfinite(columns["flow_abs"][row]) for row in perfused)


def test_marking_reads_parallel_vessels_and_simple_graphs():
    from haemolynx.graph import FLOW_SOLVED, mark_flow_solved_edges

    multi = _chain(0, 1, 2)
    multi.add_edge(1, 2)  # parallel: on the route
    multi.add_edges_from([(2, 3), (1, 1)])  # a dead end and a self-loop
    assert mark_flow_solved_edges(multi, [0], [2]) == 2
    assert {(u, v, k): d[FLOW_SOLVED] for u, v, k, d in multi.edges(keys=True, data=True)} == {
        (0, 1, 0): True, (1, 2, 0): True, (1, 2, 1): True, (2, 3, 0): False, (1, 1, 0): False,
    }

    simple = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 1), (2, 4)])
    assert mark_flow_solved_edges(simple, [0], [3]) == 1
    assert simple.edges[2, 4][FLOW_SOLVED] is False
    assert simple.edges[1, 2][FLOW_SOLVED] is True


def test_no_solve_marks_nothing():
    from haemolynx.graph import FLOW_SOLVED

    solution = _solve_with_strays(**LEAVE, run_haemodynamics=False)

    assert not any(FLOW_SOLVED in data for _u, _v, data in solution.graph.edges(data=True))


# --- the Post processing tab's "Prune" button: the same rule -----------------


def test_the_prune_button_removes_dead_ends_and_keeps_parallel_routes():
    """Inlet 0 -> 1 => 2 -> outlet 3, with 1-2 twice and a detour 1-8-2;
    a dead-end tree 1-4-5 / 4-6, a loop 2-7-9-2 hanging off node 2 and a
    piece 20-21 with inlet 20 only."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edge(1, 2, length=10.0)
    graph.add_edges_from(
        [(1, 8), (8, 2), (1, 4), (4, 5), (4, 6), (2, 7), (7, 9), (9, 2), (20, 21)], length=10.0
    )

    pruned, stats = graph_module.prune_disconnected_branches(graph, [0, 20], [3])

    assert _vessels(pruned) == [(0, 1, 0), (1, 2, 0), (1, 2, 1), (1, 8, 0), (2, 3, 0),
                                (2, 8, 0)]
    assert stats["removed_components"] == 1
    assert stats["removed_dead_end_vessels"] == 6
    assert stats["removed_vessels"] == 7
    assert stats["removed_nodes"] == 7
    assert stats["removed_boundary_nodes"] == [20]
    assert graph_module.has_pending_edits(pruned)
    assert graph.number_of_edges() == 13  # the graph on screen is left as it was


def test_the_prune_button_and_network_handling_keep_the_same_vessels():
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(1, 4), (4, 5), (2, 6), (6, 7), (7, 2), (3, 3), (20, 21)])
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], graph=graph)

    pruned, _stats = graph_module.prune_disconnected_branches(graph, [0], [3])

    assert stages.edges_network_handling_keeps(dict(REMOVE), graph, boundaries) == list(
        pruned.edges(keys=True)
    )


def test_boundary_lists_follow_a_prune_on_regenerate():
    """The button only prunes the graph; the post-processing stage then
    trims every boundary list to it, as network handling does."""
    graph = _chain(0, 1, 2, 3)
    graph.add_edges_from([(1, 4), (4, 5)], length=10.0)
    pruned, _stats = graph_module.prune_disconnected_branches(graph, [0], [3])
    boundaries = BoundaryNodes(
        inlet_nodes=[0], outlet_nodes=[3],
        arteriole_boundary_nodes=[1, 4], venule_boundary_nodes=[2, 5],
        large_arteriole_boundary_nodes=[4], resistance_node_pair=(0, 5), graph=pruned,
    )
    settings = {"arteriole_boundary_nodes": [1, 4]}

    stages._trim_boundary_lists(settings, boundaries, pruned)

    assert boundaries.arteriole_boundary_nodes == [1]
    assert boundaries.venule_boundary_nodes == [2]
    assert boundaries.large_arteriole_boundary_nodes == []
    assert settings["arteriole_boundary_nodes"] == [1]
    assert boundaries.resistance_node_pair == (0, 3)
