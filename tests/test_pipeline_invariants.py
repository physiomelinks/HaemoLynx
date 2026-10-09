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

import haemolynx.graph as graph_package
from haemolynx.graph import (
    assert_no_forbidden_edge_attributes,
    detect_cartwheel_hubs,
    duplicate_parallel_edges,
    duplicate_vessel_routes,
)
from haemolynx.graph.assemble import mask_lumen_test
from haemolynx.io.load import _to_binary_volume_for_skeletonization
from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.stages import TOPOLOGY_STEP

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"
NERVE_FIXTURE = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"


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
@pytest.mark.parametrize("collapse_method", ["distance_only", "direction_aware"])
def test_no_vessel_is_built_twice_between_the_same_two_nodes(tmp_path, collapse_method):
    """No two edges joining the same two nodes, and no route through a
    degree-2 node beside an edge, run through one lumen: one vessel, drawn
    twice. Regression test: on this capillary bed distance_only collapse
    left two such pairs; on the E14.5 MCA stack direction-aware collapse and
    degree-2 merging left hundreds, drawn in the viewer as parallel vessels
    inside one segmented branch."""
    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": NERVE_FIXTURE,
            "vtk_output_prefix": tmp_path / "run",
            "plot_dir": tmp_path / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
            "cluster_collapse_method": collapse_method,
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)
    built = {}

    def keep(stage: str, output) -> None:
        if stage == "build_network":
            built["network"] = output

    run_pipeline_stages(settings, schema, on_stage_output=keep, stop_after="build_network")

    network = built["network"]
    G = network.graph
    assert G.number_of_edges() > 100, "the fixture's capillary bed did not build; the test proves nothing"
    inside_lumen = mask_lumen_test(
        _to_binary_volume_for_skeletonization(network.volume.image), network.volume.voxel_size_zyx
    )
    assert duplicate_parallel_edges(G, inside_lumen) == []
    assert duplicate_vessel_routes(G, inside_lumen) == []


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


@pytest.mark.slow
@pytest.mark.integration
def test_consistency_warn_below_settings_choose_the_right_log_level(
    tmp_path, caplog, monkeypatch
):
    """Stage-wiring regression test for the three *_consistency_warn_below
    settings (skeleton_mask_, skeleton_graph_, graph_mask_consistency_warn_
    below): each is read from the right settings-dict key, compared with
    the right operator, and drives WARNING vs INFO correctly.

    None of tests/test_skeleton_consistency_diagnostics.py's unit tests can
    catch a bug here: they exercise the diagnose_*/format_* functions
    directly, never the settings lookup and log-level choice around them in
    pipeline/stages.py. That gap is exactly where the last review's bug
    lived -- a `requires` naming the wrong prerequisite setting -- and unit
    tests on the functions alone stayed green through it.
    """
    schema = default_schema()
    checks = (
        ("skeleton_mask_consistency_warn_below", "Skeleton/mask consistency"),
        ("skeleton_graph_consistency_warn_below", "Skeleton/graph consistency"),
        ("graph_mask_consistency_warn_below", "Graph/mask consistency"),
    )

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

    # The finished graph can trace every voxel of this fixture's skeleton
    # (a stub ending at the image face is kept, not pruned), and a coverage of
    # exactly 1.0 never warns. So the skeleton/graph check measures a copy of
    # the real graph missing one edge: below 1.0 whatever the graph traces,
    # and the settings lookup and comparison under test are unchanged.
    real_skeleton_graph_check = graph_package.diagnose_skeleton_graph_consistency

    def _graph_missing_one_edge(G, skeleton, **kwargs):
        partial = G.copy()
        partial.remove_edge(*next(iter(partial.edges(keys=True))))
        return real_skeleton_graph_check(partial, skeleton, **kwargs)

    monkeypatch.setattr(
        graph_package, "diagnose_skeleton_graph_consistency", _graph_missing_one_edge
    )

    # 1.0 is above any coverage_fraction this real, noisy fixture achieves
    # for the two mask checks (all measured well under 1.0 elsewhere in this
    # test suite), and above the one-edge-short skeleton check's -- guaranteed
    # to warn on all three at once.
    warn_high = _settings(run_name="warn_high", **{name: 1.0 for name, _ in checks})
    # 0.0 is at or below any achievable coverage_fraction -- guaranteed to
    # never warn on any of the three.
    warn_low = _settings(run_name="warn_low", **{name: 0.0 for name, _ in checks})

    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.stages"):
        caplog.clear()
        run_pipeline_stages(warn_high, schema)
        high_records = list(caplog.records)

        caplog.clear()
        run_pipeline_stages(warn_low, schema)
        low_records = list(caplog.records)

    for name, substring in checks:
        assert any(
            r.levelname == "WARNING" and substring in r.getMessage() for r in high_records
        ), f"{name}=1.0 should have produced a WARNING containing {substring!r}"
        assert any(
            r.levelname == "INFO" and substring in r.getMessage() for r in low_records
        ), f"{name}=0.0 should have produced an INFO line containing {substring!r}"
        assert not any(
            r.levelname == "WARNING" and substring in r.getMessage() for r in low_records
        ), f"{name}=0.0 must never produce a WARNING containing {substring!r}"


@pytest.mark.slow
@pytest.mark.integration
def test_missing_vessel_warn_below_settings_choose_the_right_log_level(tmp_path, caplog):
    """Stage-wiring regression test for skeleton_missing_vessel_warn_below
    and graph_missing_vessel_warn_below, mirroring the consistency test
    above -- same rationale: a bug in the settings lookup/log-level choice
    around diagnose_vessels_missing_from_skeleton/_graph would not be
    caught by tests/test_skeleton_consistency_diagnostics.py's unit tests
    on those functions alone.

    missing_vessel_min_voxels=1 is set on both runs so the fixture's ~56
    single-voxel noise specks count as vessels too, guaranteeing at least
    one genuine miss regardless of collapse method -- at the default
    min_vessel_voxels=2 this fixture has only one real vessel, which the
    skeleton/graph do explain, so there would be nothing to warn about at
    warn_below=1.0 either.
    """
    schema = default_schema()
    checks = (
        ("skeleton_missing_vessel_warn_below", "Vessels missing from skeleton"),
        ("graph_missing_vessel_warn_below", "Vessels missing from graph"),
    )

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
                "missing_vessel_min_voxels": 1,
            }
        )
        values.update(overrides)
        return resolve_settings(values, schema=schema, config_path=None)

    # 1.0 is the schema default, and this fixture's noise specks (once
    # counted as vessels via min_vessel_voxels=1) are guaranteed missing.
    warn_high = _settings(run_name="warn_high", **{name: 1.0 for name, _ in checks})
    # 0.0 is at or below any achievable explained_vessel_fraction --
    # guaranteed to never warn on either.
    warn_low = _settings(run_name="warn_low", **{name: 0.0 for name, _ in checks})

    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.stages"):
        caplog.clear()
        run_pipeline_stages(warn_high, schema)
        high_records = list(caplog.records)

        caplog.clear()
        run_pipeline_stages(warn_low, schema)
        low_records = list(caplog.records)

    for name, substring in checks:
        assert any(
            r.levelname == "WARNING" and substring in r.getMessage() for r in high_records
        ), f"{name}=1.0 should have produced a WARNING containing {substring!r}"
        assert any(
            r.levelname == "INFO" and substring in r.getMessage() for r in low_records
        ), f"{name}=0.0 should have produced an INFO line containing {substring!r}"
        assert not any(
            r.levelname == "WARNING" and substring in r.getMessage() for r in low_records
        ), f"{name}=0.0 must never produce a WARNING containing {substring!r}"


@pytest.mark.slow
@pytest.mark.integration
def test_the_built_network_draws_each_lumen_once(tmp_path):
    """What graph building leaves in one segmented vessel that is not a
    vessel (``graph.diagnose_lumen_artefacts``, on the graph build_network
    hands over): no loop lying inside one lumen, no dead end whose tip is in
    another vessel's lumen, beside another centreline or mostly off the
    mask, and no two edges through one lumen unless taking either out would
    cut vessels off (``lumen_loops.cuts_off_vessels``, the only pairs the
    end-of-build clean-up keeps). The one-centreline-per-lumen work's
    regression guard: on E14.5 the build used to leave 124 such pairs and 12
    loops inside one lumen."""
    from haemolynx.graph import diagnose_lumen_artefacts
    from haemolynx.graph.lumen_loops import cuts_off_vessels

    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": NERVE_FIXTURE,
            "vtk_output_prefix": tmp_path / "run",
            "plot_dir": tmp_path / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)
    built = {}

    def keep(stage: str, output) -> None:
        if stage == "build_network":
            built["network"] = output

    run_pipeline_stages(settings, schema, on_stage_output=keep, stop_after="build_network")

    network = built["network"]
    G = network.graph
    assert G.number_of_edges() > 100, "the fixture's capillary bed did not build; the test proves nothing"
    report = diagnose_lumen_artefacts(
        G,
        _to_binary_volume_for_skeletonization(network.volume.image),
        voxel_size_zyx=network.volume.voxel_size_zyx,
    )
    assert report["short_loop_count"] > 0, "no loops to judge; the loop check proves nothing"
    assert report["loops_inside_one_lumen"] == 0
    for kind in ("inside_other_lumen", "beside_vessel", "off_mask"):
        assert report["dead_end_counts"][kind] == 0, kind
    for left, right in report["duplicate_pairs"]:
        assert cuts_off_vessels(G, left) and cuts_off_vessels(G, right), (left, right)
