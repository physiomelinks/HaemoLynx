"""Network handling: what is done to the assigned network before its model.

The "Network handling" settings sit on the "7. Haemodynamics" tab, and run at
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
PRUNE = "remove_disconnected_io_components_after_final_assignment"


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


def test_both_settings_are_in_the_network_handling_section():
    assert SCHEMA[PRUNE].section == "Network handling"
    assert SCHEMA["boundary_handling"].section == "Network handling"


def test_boundary_handling_is_a_drop_down_with_only_none_for_now():
    from haemolynx.gui.form import widget_type_for

    setting = SCHEMA["boundary_handling"]
    assert setting.kind == "choice"
    assert tuple(setting.choices) == ("None",)
    assert setting.default == "None"
    assert widget_type_for(setting) == "ComboBox"


def test_the_prune_toggle_still_needs_automated_assignment():
    assert SCHEMA[PRUNE].requires == ("automated_vessel_assignment",)


def test_network_handling_is_the_last_box_on_the_haemodynamics_tab():
    owner = assign_to_stages(SCHEMA)
    assert owner[PRUNE] == "7. Haemodynamics"
    assert owner["boundary_handling"] == "7. Haemodynamics"
    (tab,) = [tab for tab in tabs_for(SCHEMA) if tab.stage.title == "7. Haemodynamics"]
    sections = list(dict.fromkeys(SCHEMA[row.name].section for row in tab.fields))
    assert sections[-1] == "Network handling"
    names = [row.name for row in tab.fields if SCHEMA[row.name].section == "Network handling"]
    assert names == [PRUNE, "boundary_handling"]


def test_the_prune_toggle_left_the_boundaries_tab():
    from haemolynx.gui.form import visible_vessel_mask_settings

    shown = visible_vessel_mask_settings(SCHEMA, {"automated_vessel_assignment": True})
    assert PRUNE not in shown


def test_an_old_config_with_the_toggle_under_vessel_masks_still_loads(tmp_path):
    from haemolynx.parsers import load_config

    path = tmp_path / "old.yaml"
    path.write_text(f"vessel_masks:\n  {PRUNE}: true\n", encoding="utf-8")

    assert load_config(path, SCHEMA)[PRUNE] is True


# --- what the pruning does ----------------------------------------------------


def test_off_leaves_the_network_alone():
    graph = _two_components()
    model = HaemodynamicModel(graph=graph)
    boundaries = _boundaries(graph)

    result = apply_network_handling({PRUNE: False}, model, boundaries)

    assert result is model and model.graph is graph
    assert boundaries.inlet_nodes == [10, 0]


def test_on_drops_components_without_both_an_inlet_and_an_outlet():
    graph = _two_components()
    model = HaemodynamicModel(graph=graph)
    network = VesselNetwork(graph=graph, volume=None)
    boundaries = _boundaries(graph)
    settings = {
        PRUNE: True,
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

    apply_network_handling({PRUNE: True}, HaemodynamicModel(graph=graph), boundaries)

    assert boundaries.resistance_node_pair == (0, 2)


def test_nothing_left_to_solve_is_an_error():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(10, 11)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[11], graph=graph)

    with pytest.raises(ValueError, match="no valid boundary nodes"):
        apply_network_handling({PRUNE: True}, HaemodynamicModel(graph=graph), boundaries)


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
        {PRUNE: True},
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

    assert PRUNE not in inspect.getsource(stages.assign_boundaries)
