"""Unit tests for the branch comparison tool's reporting logic.

These drive `scripts/compare_branches.py`'s comparison and report code with two
small synthetic graphs -- no pipeline run, no git worktree, no large image --
so the part a reviewer trusts is checked in milliseconds. The full comparison
itself is not a test and never runs in CI.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import networkx as nx
import pytest

from branch_comparison import metrics, runner
from branch_comparison.report import (
    ComparisonReport,
    PlotPair,
    SideReport,
    render_html,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures: two small graphs that differ in a known way
# ---------------------------------------------------------------------------


def _chain_graph(lengths, *, extra_attributes=None):
    """A path graph whose edges carry the given lengths, in microns."""
    graph = nx.MultiGraph()
    position = 0.0
    graph.add_node(0, pos=(0.0, 0.0, 0.0))
    for index, length in enumerate(lengths):
        position += length
        graph.add_node(index + 1, pos=(position, 0.0, 0.0))
        graph.add_edge(index, index + 1, length=float(length), **(extra_attributes or {}))
    return graph


@pytest.fixture
def reference_graph():
    """Four nodes, three edges, one branch point once the spur is added."""
    graph = _chain_graph([10.0, 20.0, 30.0], extra_attributes={"resistance": 2.0})
    graph.add_node(4, pos=(20.0, 5.0, 0.0))
    graph.add_edge(1, 4, length=5.0, resistance=1.0)
    return graph


@pytest.fixture
def current_graph():
    """The same shape, but one edge is longer and the flow solve ran."""
    graph = _chain_graph([10.0, 25.0, 30.0], extra_attributes={"resistance": 2.0})
    graph.add_node(4, pos=(20.0, 5.0, 0.0))
    graph.add_edge(1, 4, length=5.0, resistance=1.0)
    for *_, data in graph.edges(data=True):
        data["flow_abs"] = 1.5
    return graph


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------


def test_graph_metrics_reports_the_headline_numbers(reference_graph):
    values = metrics.graph_metrics(reference_graph)

    assert values["nodes"] == 5
    assert values["edges"] == 4
    assert values["total_edge_length"] == pytest.approx(65.0)
    assert values["mean_edge_length"] == pytest.approx(65.0 / 4)
    assert values["median_edge_length"] == pytest.approx(15.0)
    assert values["max_edge_length"] == pytest.approx(30.0)
    assert values["branching_points"] == 1
    assert values["average_degree"] == pytest.approx(8 / 5)


def test_graph_metrics_leaves_lengths_unset_when_no_edge_has_one():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)

    values = metrics.graph_metrics(graph)

    assert values["edges"] == 1
    assert values["total_edge_length"] is None
    assert values["mean_edge_length"] is None


def test_compare_metrics_computes_deltas_and_flags_only_real_changes(
    current_graph, reference_graph
):
    rows = metrics.compare_metrics(
        metrics.graph_metrics(current_graph), metrics.graph_metrics(reference_graph)
    )
    by_name = {row.name: row for row in rows}

    assert [row.name for row in rows][:2] == ["Nodes", "Edges"]
    assert by_name["Nodes"].differs is False
    assert by_name["Total edge length"].current == pytest.approx(70.0)
    assert by_name["Total edge length"].reference == pytest.approx(65.0)
    assert by_name["Total edge length"].delta == pytest.approx(5.0)
    assert by_name["Total edge length"].percent_change == pytest.approx(
        100.0 * 5.0 / 65.0
    )
    assert by_name["Total edge length"].flagged is True
    assert by_name["Branching points (degree >= 3)"].flagged is False
    assert metrics.any_differences(rows) is True


def test_metric_row_handles_missing_sides_and_informational_rows():
    missing = metrics.MetricRow("only current", 3.0, None)
    assert missing.differs is True
    assert missing.delta is None

    both_absent = metrics.MetricRow("absent", None, None)
    assert both_absent.differs is False

    runtime = metrics.MetricRow("Runtime", 20.0, 30.0, informational=True)
    assert runtime.differs is True
    assert runtime.flagged is False
    assert metrics.any_differences([runtime]) is False


# ---------------------------------------------------------------------------
# Per-stage divergence
# ---------------------------------------------------------------------------


def _stage_sequence(labels, graphs):
    return [
        (label, metrics.graph_fingerprint(graph))
        for label, graph in zip(labels, graphs)
    ]


def test_first_differing_stage_names_the_earliest_divergence(
    current_graph, reference_graph
):
    labels = ["build", "reconnect", "collapse", "prune"]
    same = _chain_graph([1.0, 2.0])
    current = _stage_sequence(labels, [same, same, current_graph, current_graph])
    reference = _stage_sequence(labels, [same, same, reference_graph, reference_graph])

    diffs = metrics.compare_stages(current, reference)
    first = metrics.first_differing_stage(diffs)

    assert [diff.status for diff in diffs] == ["same", "same", "differs", "differs"]
    assert first is not None
    assert first.label == "collapse"
    assert "edge lengths" in first.reasons


def test_stages_identical_on_both_sides_report_no_divergence():
    graph = _chain_graph([1.0, 2.0])
    stages = _stage_sequence(["build", "prune"], [graph, graph])

    diffs = metrics.compare_stages(stages, stages)

    assert all(diff.status == "same" for diff in diffs)
    assert metrics.first_differing_stage(diffs) is None


def test_a_stage_only_one_side_ran_is_reported_not_dropped():
    graph = _chain_graph([1.0])
    current = _stage_sequence(["build", "new_step"], [graph, graph])
    reference = _stage_sequence(["build", "old_step"], [graph, graph])

    diffs = metrics.compare_stages(current, reference)

    assert [(d.label, d.status) for d in diffs] == [
        ("build", "same"),
        ("new_step", "only_current"),
        ("old_step", "only_reference"),
    ]
    assert metrics.first_differing_stage(diffs).label == "new_step"


def test_fingerprint_sees_node_positions_that_counts_alone_would_miss():
    moved = _chain_graph([10.0, 20.0])
    same_counts = nx.MultiGraph()
    same_counts.add_node(0, pos=(0.0, 0.0, 0.0))
    same_counts.add_node(1, pos=(10.0, 0.0, 0.0))
    same_counts.add_node(2, pos=(30.0, 7.0, 0.0))
    same_counts.add_edge(0, 1, length=10.0)
    same_counts.add_edge(1, 2, length=20.0)

    left = metrics.graph_fingerprint(moved)
    right = metrics.graph_fingerprint(same_counts)

    assert left.nodes == right.nodes and left.edges == right.edges
    assert left.differences(right) == ["node positions"]


def test_stage_order_mismatch_is_detected():
    graph = _chain_graph([1.0])
    forwards = _stage_sequence(["a", "b"], [graph, graph])
    backwards = _stage_sequence(["b", "a"], [graph, graph])

    assert metrics.stage_order_mismatch(forwards, forwards) is False
    assert metrics.stage_order_mismatch(forwards, backwards) is True


def test_stage_snapshots_are_ordered_by_when_the_run_wrote_them(tmp_path):
    graph = _chain_graph([1.0])
    for index, label in enumerate(["third", "first", "second"]):
        path = tmp_path / f"stack_graph_after_{label}.pkl"
        path.write_bytes(pickle.dumps(graph))
        # Filename order is alphabetical; write order is what the run means.
        os.utime(path, ns=(0, (index + 1) * 10**9))

    labels = [
        metrics.stage_label(path, "stack")
        for path in metrics.discover_stage_snapshots(tmp_path, "stack")
    ]

    assert labels == ["third", "first", "second"]
    assert [label for label, _ in metrics.fingerprint_stage_files(tmp_path, "stack")] == labels


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------


def test_edge_attributes_only_one_side_produced_are_listed(
    current_graph, reference_graph
):
    comparison = metrics.compare_edge_attributes(
        metrics.edge_attribute_summary(current_graph),
        metrics.edge_attribute_summary(reference_graph),
    )

    assert comparison.only_current == ["flow_abs"]
    assert comparison.only_reference == []
    assert comparison.differs is True

    shared = {row.name: row for row in comparison.shared_rows}
    assert shared["resistance.mean"].flagged is False
    assert shared["length.max"].current == pytest.approx(30.0)
    assert shared["length.mean"].flagged is True


# ---------------------------------------------------------------------------
# Statistics CSV
# ---------------------------------------------------------------------------


def _write_statistics_csv(path: Path, timestamp: str, total_length: str) -> Path:
    path.write_text(
        "Section,Metric,Value,Unit,Notes\n"
        f"Metadata,Exported At (UTC),{timestamp},,Generated by HaemoLynx.\n"
        "Metadata,Metric Count,2,,Number of metric rows.\n"
        f"Network,Total Length,{total_length},microns,\n",
        encoding="utf-8",
    )
    return path


def test_statistics_csv_comparison_ignores_the_export_timestamp(tmp_path):
    current = _write_statistics_csv(tmp_path / "a.csv", "2026-01-01T00:00:00+00:00", "70")
    reference = _write_statistics_csv(tmp_path / "b.csv", "2026-08-04T09:30:00+00:00", "65")

    rows = metrics.compare_statistics_csv(
        metrics.read_statistics_csv(current), metrics.read_statistics_csv(reference)
    )
    by_name = {row.name: row for row in rows}

    assert not any("Exported At" in name for name in by_name)
    assert by_name["Network / Total Length [microns]"].delta == pytest.approx(5.0)
    assert by_name["Network / Total Length [microns]"].flagged is True
    assert by_name["Metadata / Metric Count"].flagged is False


def test_missing_statistics_csv_reads_as_empty_rather_than_raising(tmp_path):
    assert metrics.read_statistics_csv(tmp_path / "absent.csv") == {}


# ---------------------------------------------------------------------------
# Report assembly and rendering
# ---------------------------------------------------------------------------


def _side(label, ref, **overrides):
    values = dict(
        label=label,
        ref=ref,
        commit="abc1234",
        checkout=f"/tmp/{label}",
        ok=True,
        runtime_seconds=12.5,
        api_style="settings-dict",
        applied_settings=138,
        final_graph_source="graph returned by image_to_model_pipeline()",
    )
    values.update(overrides)
    return SideReport(**values)


def _artefacts(graph, stages):
    return runner.SideArtefacts(
        metrics=metrics.graph_metrics(graph),
        edge_attributes=metrics.edge_attribute_summary(graph),
        stages=stages,
        statistics={},
        branch_statistics={},
        vtk={},
        plots={},
    )


@pytest.fixture
def comparison(current_graph, reference_graph):
    labels = ["build", "collapse"]
    unchanged = _chain_graph([1.0, 2.0])
    report = runner.build_report(
        current=_side("current", "feature/x"),
        reference=_side("reference", "main"),
        current_artefacts=_artefacts(
            current_graph, _stage_sequence(labels, [unchanged, current_graph])
        ),
        reference_artefacts=_artefacts(
            reference_graph, _stage_sequence(labels, [unchanged, reference_graph])
        ),
        image_path="/data/Nerve_capillaries.tif",
    )
    report.plots = runner.pair_plots(
        {"final_graph.png": "current/plots/final_graph.png",
         "only_here.png": "current/plots/only_here.png"},
        {"final_graph.png": "reference/plots/final_graph.png"},
    )
    return report


def test_build_report_finds_the_first_differing_stage(comparison):
    assert comparison.complete is True
    assert comparison.differs is True
    assert comparison.first_stage_difference.label == "collapse"
    assert comparison.status_line().startswith("COMPLETE")
    assert "different numbers" in comparison.status_line()


def test_runtime_difference_alone_is_not_reported_as_a_change(
    current_graph, reference_graph
):
    stages = _stage_sequence(["build"], [reference_graph])
    report = runner.build_report(
        current=_side("current", "feature/x", runtime_seconds=10.0),
        reference=_side("reference", "main", runtime_seconds=99.0),
        current_artefacts=_artefacts(reference_graph, stages),
        reference_artefacts=_artefacts(reference_graph, stages),
        image_path="/data/image.tif",
    )

    assert report.runtime_row.differs is True
    assert report.runtime_row.flagged is False
    assert report.differs is False
    assert "identical numbers" in report.status_line()


def test_markdown_names_the_first_differing_stage_and_marks_changes(comparison):
    text = render_markdown(comparison)

    assert "First differing stage: `collapse`" in text
    assert "**Total edge length**" in text
    assert "CHANGED" in text
    assert "| Nodes |  | 5 | 5 | 0 | +0% |  |" in text
    assert "`flow_abs`" in text


def test_html_shows_both_sides_of_every_plot(comparison):
    page = render_html(comparison)

    assert '<title>Branch comparison: feature/x vs main</title>' in page
    assert 'src="current/plots/final_graph.png"' in page
    assert 'src="reference/plots/final_graph.png"' in page
    # A plot only one side produced still gets a slot, marked as missing.
    assert 'src="current/plots/only_here.png"' in page
    assert page.count("not produced") == 1
    assert "First differing stage" in page
    assert 'class="changed"' in page


def test_html_escapes_values_rather_than_injecting_them(current_graph):
    stages = _stage_sequence(["build"], [current_graph])
    report = runner.build_report(
        current=_side("current", "<script>alert(1)</script>"),
        reference=_side("reference", "main"),
        current_artefacts=_artefacts(current_graph, stages),
        reference_artefacts=_artefacts(current_graph, stages),
        image_path="/data/image.tif",
    )

    page = render_html(report)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_a_failed_side_is_reported_and_no_tables_are_shown(current_graph):
    report = runner.build_report(
        current=_side("current", "feature/x"),
        reference=_side(
            "reference", "main", ok=False, error="FileNotFoundError: image missing"
        ),
        current_artefacts=_artefacts(
            current_graph, _stage_sequence(["build"], [current_graph])
        ),
        reference_artefacts=runner.SideArtefacts(),
        image_path="/data/image.tif",
    )

    assert report.complete is False
    assert "INCOMPLETE" in report.status_line()
    assert "reference (main) run failed" in report.status_line()

    text = render_markdown(report)
    assert "FileNotFoundError: image missing" in text
    assert "Graph metrics" not in text
    assert "Per-stage divergence" not in text
    assert "No comparison" in text

    page = render_html(report)
    assert "No comparison" in page
    assert "Graph metrics" not in page


def test_settings_a_branch_cannot_apply_make_the_report_incomplete(current_graph):
    stages = _stage_sequence(["build"], [current_graph])
    report = runner.build_report(
        current=_side("current", "feature/x"),
        reference=_side(
            "reference",
            "old-branch",
            unapplied_required={"min_stub_length": "no parameter of this name"},
        ),
        current_artefacts=_artefacts(current_graph, stages),
        reference_artefacts=_artefacts(current_graph, stages),
        image_path="/data/image.tif",
    )

    assert report.complete is False
    assert "min_stub_length" in report.status_line()
    assert "not the same experiment" in report.status_line()
    assert "min_stub_length" in render_markdown(report)


def test_settings_a_branch_simply_lacks_are_caveats_not_failures(current_graph):
    stages = _stage_sequence(["build"], [current_graph])
    report = runner.build_report(
        current=_side("current", "feature/x"),
        reference=_side(
            "reference",
            "old-branch",
            unapplied_optional={"vtk_export": "no setting of this name"},
        ),
        current_artefacts=_artefacts(current_graph, stages),
        reference_artefacts=_artefacts(current_graph, stages),
        image_path="/data/image.tif",
    )

    assert report.complete is True
    assert report.caveats == 1
    assert "do not exist on one side" in report.status_line()
    assert "vtk_export" in render_markdown(report)


def test_differing_graph_capture_points_are_warned_about(current_graph):
    stages = _stage_sequence(["build"], [current_graph])
    report = runner.build_report(
        current=_side("current", "feature/x"),
        reference=_side(
            "reference", "old-branch", final_graph_source="graph passed to the VTK export"
        ),
        current_artefacts=_artefacts(current_graph, stages),
        reference_artefacts=_artefacts(current_graph, stages),
        image_path="/data/image.tif",
    )

    assert any("captured at different points" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Artefact collection
# ---------------------------------------------------------------------------


def test_collect_side_reads_the_graphs_a_run_left_behind(tmp_path, current_graph):
    paths = runner.SidePaths("current", tmp_path / "current")
    paths.output_dir.mkdir(parents=True)
    paths.graph_path.write_bytes(pickle.dumps(current_graph))
    (paths.output_dir / "stack_graph_after_build.pkl").write_bytes(
        pickle.dumps(current_graph)
    )
    paths.plot_dir.mkdir(parents=True)
    (paths.plot_dir / "final_graph.png").write_bytes(b"not really a png")
    (paths.plot_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    artefacts = runner.collect_side(paths, "stack", root=tmp_path)

    assert artefacts.metrics["edges"] == 4
    assert [label for label, _ in artefacts.stages] == ["build"]
    assert artefacts.plots == {"final_graph.png": "current/plots/final_graph.png"}
    assert artefacts.vtk == {}


def test_vtk_parts_put_the_usual_exports_first_and_keep_the_rest():
    assert runner.ordered_vtk_parts(
        {"nodes": None, "vessels": None, "vessels_flow": None}, {"pericytes": None}
    ) == ["vessels", "pericytes", "nodes", "vessels_flow"]


def test_vtk_comparison_reports_a_mesh_only_one_side_exported():
    comparison = metrics.compare_vtk(
        "vessels",
        {"n_points": 10, "n_cells": 4, "cell_arrays": {"resistance": {"n": 4, "mean": 2.0}}},
        None,
    )

    assert comparison.missing == (False, True)
    assert comparison.only_current == ["resistance"]
    assert comparison.differs is True


def test_pair_plots_covers_every_filename_either_side_produced():
    pairs = runner.pair_plots({"a.png": "current/a.png"}, {"b.png": "reference/b.png"})

    assert pairs == [
        PlotPair("a.png", "current/a.png", None),
        PlotPair("b.png", None, "reference/b.png"),
    ]


def test_report_dataclass_defaults_are_safe_to_render():
    report = ComparisonReport(
        current=_side("current", "feature/x"), reference=_side("reference", "main")
    )

    assert render_markdown(report)
    assert render_html(report)


# --- the config defines the experiment, not this tool -------------------------


def test_the_tool_pins_nothing_that_defines_the_experiment():
    """Which stages run and how they are tuned is the config's business.

    Pinning it here let the tool run something neither branch would: while
    `resistance_pipeline_config.yaml` could not run at all -- it named an image
    absent from the repository and gave no outlet nodes -- these overrides
    replaced every setting that would have failed, so nothing said so.
    """
    from branch_comparison import run_settings

    experiment = {
        "do_skeletonize", "do_graph_building", "run_haemodynamics",
        "skeleton_closing_radius", "skeleton_bridge_gap_size",
        "skeleton_min_branch_length", "skeleton_max_bridge_distance",
        "skeleton_min_component_percent", "inlet_node_selection_method",
        "outlet_node_selection_method", "inlet_node_volumes",
        "outlet_node_volumes", "inlet_p_bc", "outlet_p_bc", "min_stub_length",
    }
    pinned = set(run_settings.REQUIRED_SETTINGS) | set(run_settings.BEST_EFFORT_SETTINGS)
    assert not (pinned & experiment), (
        f"these define the experiment and belong in the config: "
        f"{sorted(pinned & experiment)}"
    )


def test_the_tool_still_pins_what_the_report_cannot_do_without():
    """Statistics, because the report is largely a diff of those CSVs."""
    from branch_comparison import run_settings

    assert run_settings.REQUIRED_SETTINGS["statistics"] is True


def test_the_tool_still_forces_an_unattended_run():
    """The shipped config asks for a browser tab and a window to close."""
    from branch_comparison import run_settings

    for name in ("show_plots_in_ide", "hold_ide_plots_open", "interactive_plots"):
        assert run_settings.REQUIRED_SETTINGS[name] is False, name
    assert run_settings.REQUIRED_SETTINGS["ide_plot_mode"] == "none"


def test_build_settings_leaves_the_boundaries_to_the_config():
    """No boundary boxes unless --smoke asks for the fixture's own."""
    from branch_comparison import run_settings

    settings, required = run_settings.build_settings(
        image_path="/tmp/image.tif",
        plot_dir="/tmp/plots",
        vtk_output_prefix="/tmp/run",
    )
    for name in ("inlet_node_volumes", "outlet_node_volumes",
                 "inlet_node_selection_method", "outlet_node_selection_method"):
        assert name not in settings, f"{name} should come from the config"
        assert name not in required


def test_smoke_boundaries_are_still_applied_when_asked_for():
    """The 48-voxel fixture needs its own boxes; the config's select nothing."""
    from branch_comparison import run_settings

    settings, required = run_settings.build_settings(
        image_path="/tmp/image.tif",
        plot_dir="/tmp/plots",
        vtk_output_prefix="/tmp/run",
        boundary_settings=run_settings.SMOKE_BOUNDARY_SETTINGS,
    )
    assert settings["outlet_node_volumes"] == [[[0, 0, 28], [47, 47, 47]]]
    assert "outlet_node_volumes" in required
