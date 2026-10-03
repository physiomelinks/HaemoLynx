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
#: The checkbox that "Remove disconnected branches/trees" replaced.
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
        ("Remove disconnected branches/trees", "remove_disconnected"),
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


def _bifurcation_and_strays() -> nx.MultiGraph:
    """0 (inlet) -> 1 -> {2, 3} (outlets); 10-11-12 has inlet 10 and no
    outlet; 20-21 has no boundary at all. Real diameters, lengths in um."""
    import numpy as np

    graph = nx.MultiGraph()
    positions = {
        0: (0, 0, 0), 1: (0, 0, 50), 2: (0, 40, 90), 3: (0, -40, 90),
        10: (0, 200, 0), 11: (0, 200, 50), 12: (0, 200, 100),
        20: (0, 400, 0), 21: (0, 400, 50),
    }
    for node, pos in positions.items():
        graph.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v, order, diameter in [
        (0, 1, "B01", 15.0), (1, 2, "B02", 20.0), (1, 3, "B03", 8.0),
        (10, 11, "B02", 10.0), (11, 12, "B03", 8.0), (20, 21, "B03", 8.0),
    ]:
        length = float(np.linalg.norm(graph.nodes[u]["pos"] - graph.nodes[v]["pos"]))
        graph.add_edge(u, v, key=0, length=length, branch_order=order, diameter_um=diameter,
                       voxels=[list(graph.nodes[u]["pos"]), list(graph.nodes[v]["pos"])])
    graph.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    return graph


def _solve_with_strays(**overrides):
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
    graph = _bifurcation_and_strays()
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


def test_no_solve_marks_nothing():
    from haemolynx.graph import FLOW_SOLVED

    solution = _solve_with_strays(**LEAVE, run_haemodynamics=False)

    assert not any(FLOW_SOLVED in data for _u, _v, data in solution.graph.edges(data=True))
