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

from pathlib import Path

import pytest

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
