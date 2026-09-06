"""What a run promises about the graph it is working on.

The napari panel builds the vessel geometry once, when `build_network` hands
its network over, and every later stage only replaces the feature table -- 774
nodes and 33,000 points are not rebuilt five times for a run. That is only
correct because no stage after `build_network` adds or removes anything: they
write attributes and nothing else.

Nothing else states that. A stage that started editing topology -- splitting an
edge at a constriction, dropping an unreachable component -- would leave the
viewer showing a network that no longer exists, and would do it silently. So it
is stated here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from haemolynx.graph import assert_no_forbidden_edge_attributes, detect_cartwheel_hubs
from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.stages import TOPOLOGY_STEP

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"


@pytest.mark.slow
@pytest.mark.integration
def test_no_stage_after_the_graph_is_built_changes_its_topology(tmp_path):
    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": FIXTURE,
            "vtk_output_prefix": tmp_path / "run",
            "plot_dir": tmp_path / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)

    seen: dict[str, tuple[frozenset, frozenset]] = {}

    def remember(stage: str, output) -> None:
        # The topology steps hand over the graph mid-repair -- changing it is
        # exactly what they are for -- so only the stages are compared. (Their
        # output is the graph itself, whose `.graph` is its metadata dict, so
        # they would not survive the check below either.)
        if stage.startswith(TOPOLOGY_STEP):
            return
        graph = getattr(output, "graph", None)
        if graph is None or not hasattr(graph, "nodes"):
            return
        seen[stage] = (
            frozenset(graph.nodes),
            frozenset(graph.edges(keys=True)),
        )

    run_pipeline_stages(settings, schema, on_stage_output=remember)

    assert "build_network" in seen, "no stage handed over a graph"
    built_nodes, built_edges = seen["build_network"]
    assert built_edges, "the fixture produced no vessels; the test proves nothing"

    for stage, (nodes, edges) in seen.items():
        assert nodes == built_nodes, (
            f"{stage} changed the graph's nodes. The napari panel builds the "
            "geometry once at build_network and only updates features after; a "
            "stage that edits topology needs that design revisited."
        )
        assert edges == built_edges, f"{stage} changed the graph's edges, likewise"


@pytest.mark.slow
@pytest.mark.integration
def test_the_cartwheel_hub_guard_never_changes_the_graph(tmp_path, caplog):
    """detect_cartwheel_hub_artifacts is diagnostic-only: turning it on must
    produce byte-identical topology to a run with it off (the default).

    Also pins that the fixture/threshold combination actually gets the guard
    to flag a hub (via the log line it emits), not just that "on" and "off"
    happen to agree -- without this, a regression that silently made
    detect_cartwheel_hubs always return [] (the same class of bug as a
    schema-legal cartwheel_hub_tangent_length_um of 0.0 used to cause) would
    still leave "on" and "off" topologies trivially equal, and this test
    would keep passing with zero signal that the diagnostic path ever ran.
    """
    schema = default_schema()

    def _settings(**overrides):
        values = {setting.name: setting.default for setting in schema}
        values.update(
            {
                "input_path": FIXTURE,
                "vtk_output_prefix": tmp_path / overrides.pop("run_name"),
                "plot_dir": tmp_path / "plots",
                "statistics": False,
                "show_plots_in_ide": False,
                "interactive_plots": False,
            }
        )
        values.update(overrides)
        return resolve_settings(values, schema=schema, config_path=None)

    def _built_graph_topology(settings) -> tuple[frozenset, frozenset]:
        captured = {}

        def remember(stage: str, output) -> None:
            if stage == "build_network":
                captured["graph"] = output.graph

        run_pipeline_stages(settings, schema, on_stage_output=remember)
        graph = captured["graph"]
        return frozenset(graph.nodes), frozenset(graph.edges(keys=True))

    off = _settings(run_name="off", detect_cartwheel_hub_artifacts=False)
    on = _settings(
        run_name="on",
        detect_cartwheel_hub_artifacts=True,
        # Low enough to actually flag something on this fixture, proving the
        # check ran rather than trivially finding nothing to warn about.
        cartwheel_hub_min_degree=3,
        cartwheel_hub_max_radial_dispersion=1.0,
    )

    topology_off = _built_graph_topology(off)
    with caplog.at_level(logging.WARNING, logger="haemolynx.pipeline.stages"):
        topology_on = _built_graph_topology(on)

    assert topology_off == topology_on
    assert any(
        "Cartwheel hub guard" in record.getMessage() and "flagged" in record.getMessage()
        for record in caplog.records
    ), "the guard never actually flagged a hub, so this test proves nothing about it running"


@pytest.mark.slow
@pytest.mark.integration
def test_cluster_collapse_direction_aware_runs_end_to_end_and_never_flags_more_hubs(tmp_path):
    """cluster_collapse_method=direction_aware wired all the way through a
    real run: produces a valid graph (not just on the small hand-built
    fixtures in test_direction_aware_collapse.py), and never leaves *more*
    cartwheel-shaped hubs behind than the legacy distance_only method does
    on the same input -- the property the whole feature exists for. A small,
    already-clean fixture may not exercise the pathology heavily enough for
    this to also assert *fewer* hubs; that reduction is what the real,
    densely-braided dataset this feature was written for is expected to show.
    """
    schema = default_schema()

    def _settings(**overrides):
        values = {setting.name: setting.default for setting in schema}
        values.update(
            {
                "input_path": FIXTURE,
                "vtk_output_prefix": tmp_path / overrides.pop("run_name"),
                "plot_dir": tmp_path / "plots",
                "statistics": False,
                "show_plots_in_ide": False,
                "interactive_plots": False,
                "detect_cartwheel_hub_artifacts": True,
                "cartwheel_hub_min_degree": 3,
                "cartwheel_hub_max_radial_dispersion": 1.0,
            }
        )
        values.update(overrides)
        return resolve_settings(values, schema=schema, config_path=None)

    def _built_graph(settings):
        captured = {}

        def remember(stage: str, output) -> None:
            if stage == "build_network":
                captured["graph"] = output.graph

        run_pipeline_stages(settings, schema, on_stage_output=remember)
        return captured["graph"]

    distance_only = _settings(run_name="distance_only", cluster_collapse_method="distance_only")
    direction_aware = _settings(run_name="direction_aware", cluster_collapse_method="direction_aware")

    graph_distance_only = _built_graph(distance_only)
    graph_direction_aware = _built_graph(direction_aware)

    assert graph_direction_aware.number_of_nodes() > 0
    assert graph_direction_aware.number_of_edges() > 0
    assert_no_forbidden_edge_attributes(graph_direction_aware)

    hubs_distance_only = detect_cartwheel_hubs(
        graph_distance_only, min_degree=3, max_radial_dispersion=1.0
    )
    hubs_direction_aware = detect_cartwheel_hubs(
        graph_direction_aware, min_degree=3, max_radial_dispersion=1.0
    )
    assert len(hubs_direction_aware) <= len(hubs_distance_only)


@pytest.mark.slow
@pytest.mark.integration
def test_cluster_collapse_persistence_runs_end_to_end_and_never_flags_more_hubs(tmp_path):
    """cluster_collapse_method=persistence wired all the way through a real
    run, mirroring the direction_aware check above: produces a valid graph,
    and never leaves more cartwheel-shaped hubs behind than distance_only
    does on the same input. See test_persistence_collapse.py for the hand-
    built fixtures that show the method actually splitting a cartwheel-
    shaped cluster this small, already-clean fixture may not exercise
    heavily enough to also assert *fewer* hubs here.
    """
    schema = default_schema()

    def _settings(**overrides):
        values = {setting.name: setting.default for setting in schema}
        values.update(
            {
                "input_path": FIXTURE,
                "vtk_output_prefix": tmp_path / overrides.pop("run_name"),
                "plot_dir": tmp_path / "plots",
                "statistics": False,
                "show_plots_in_ide": False,
                "interactive_plots": False,
                "detect_cartwheel_hub_artifacts": True,
                "cartwheel_hub_min_degree": 3,
                "cartwheel_hub_max_radial_dispersion": 1.0,
            }
        )
        values.update(overrides)
        return resolve_settings(values, schema=schema, config_path=None)

    def _built_graph(settings):
        captured = {}

        def remember(stage: str, output) -> None:
            if stage == "build_network":
                captured["graph"] = output.graph

        run_pipeline_stages(settings, schema, on_stage_output=remember)
        return captured["graph"]

    distance_only = _settings(run_name="distance_only", cluster_collapse_method="distance_only")
    persistence = _settings(run_name="persistence", cluster_collapse_method="persistence")

    graph_distance_only = _built_graph(distance_only)
    graph_persistence = _built_graph(persistence)

    assert graph_persistence.number_of_nodes() > 0
    assert graph_persistence.number_of_edges() > 0
    assert_no_forbidden_edge_attributes(graph_persistence)

    hubs_distance_only = detect_cartwheel_hubs(
        graph_distance_only, min_degree=3, max_radial_dispersion=1.0
    )
    hubs_persistence = detect_cartwheel_hubs(
        graph_persistence, min_degree=3, max_radial_dispersion=1.0
    )
    assert len(hubs_persistence) <= len(hubs_distance_only)
