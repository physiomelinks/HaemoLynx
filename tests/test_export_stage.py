"""The Export stage (``pipeline.stages.export_results``): what it writes, and where.

Each test is a regression for a defect the Export tab review found
(2026-11-02): VTK files written only with haemodynamics, the pericytes file
ignoring the run's constriction spacing, a resumed run's files split between
two folders, the weighted betweenness reports computed whatever the
statistics toggles said and on a graph the statistics do not measure, and the
panel's note claiming files that were never written.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from haemolynx import statistics  # noqa: E402
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges  # noqa: E402
from haemolynx.pipeline import default_schema, stages  # noqa: E402

IMAGE_SHAPE = (8, 10, 12)
STEM = "export_run"


def _straight_vessel_graph(length_um: float = 100.0) -> nx.MultiGraph:
    """Two vessels in a line, the first *length_um* long, both with a width."""
    G = nx.MultiGraph()
    positions = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, length_um), 2: (0.0, 0.0, length_um + 10.0)}
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for u, v in ((0, 1), (1, 2)):
        start, end = np.asarray(positions[u]), np.asarray(positions[v])
        G.add_edge(
            u, v, key=0,
            length=float(np.linalg.norm(end - start)),
            diameter_um=6.0,
            branch_order="B01",
            voxels=[tuple(start), tuple(end)],
        )
    return G


def _forked_graph_with_a_bridge() -> nx.MultiGraph:
    """A junction whose third vessel opens into a thick vessel through a bridge."""
    G = nx.MultiGraph()
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 4.0),
        2: (0.0, 3.0, 8.0),
        3: (0.0, -3.0, 8.0),
        4: (0.0, -3.0, 9.0),
        5: (0.0, -3.0, 20.0),
        6: (0.0, 6.0, 20.0),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for u, v in ((0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)):
        start, end = np.asarray(positions[u]), np.asarray(positions[v])
        G.add_edge(
            u, v, key=0,
            length=float(np.linalg.norm(end - start)),
            branch_order="B01",
            voxels=[tuple(start), tuple(end)],
        )
    G.edges[3, 4, 0][IS_ZERO_RESISTANCE] = True
    return G


def _settings(tmp_path: Path, **overrides) -> dict:
    settings = default_schema().defaults()
    settings.update(
        {
            "input_path": tmp_path / f"{STEM}.tif",
            "output_dir": tmp_path,
            "statistics": False,
            "run_haemodynamics": False,
            "measurement_3d_to_cell_mask": False,
            "vtk_export": False,
            "visualize_vtk": False,
            "visualize_results": False,
            "export_citations": False,
        }
    )
    settings.update(overrides)
    return settings


def _export(tmp_path: Path, graph: nx.MultiGraph | None = None, *, volume_dir=None, **overrides):
    """Run ``export_results`` on *graph*, its volume's folder *volume_dir*
    (*tmp_path* by default); returns the graph and the solution."""
    settings = _settings(tmp_path, **overrides)
    G = _straight_vessel_graph() if graph is None else graph
    volume = stages.SkeletonisedVolume(
        image=np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        skeleton=np.zeros(IMAGE_SHAPE, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path if volume_dir is None else volume_dir,
    )
    solution = stages.export_results(
        settings,
        stages.VesselNetwork(graph=G, volume=volume),
        stages.HaemodynamicModel(graph=G),
        stages.Solution(),
    )
    return G, solution


# --- VTK ----------------------------------------------------------------------


def test_the_vtk_files_are_written_without_haemodynamics(tmp_path):
    """Geometry, diameters and branch orders need no solve; vtk_export used to
    require run_haemodynamics, so a statistics-only run had no VTK at all."""
    _G, solution = _export(tmp_path, vtk_export=True, run_haemodynamics=False)

    vessels = pv.read(tmp_path / f"{STEM}_vessels.vtp")
    assert vessels.n_cells == 2
    assert np.allclose(vessels.cell_data["diameter_um"], [6.0, 6.0])
    assert np.allclose(vessels.cell_data["length_um"], [100.0, 10.0])
    # Nothing was solved, so no flow array is written.
    assert "flow_signed" not in vessels.cell_data
    assert (tmp_path / f"{STEM}_nodes.vtp").is_file()
    assert "VTK" in solution.exported


def test_the_vtk_export_no_longer_needs_haemodynamics_in_the_schema():
    schema = default_schema()
    assert "run_haemodynamics" not in schema["vtk_export"].requires
    assert schema["visualize_vtk"].requires == ("vtk_export",)


def test_the_pericytes_file_follows_the_runs_constriction_spacing(tmp_path):
    """The export passed no spacing, so the file always held a site every
    100 um from 20 um, whatever constriction_spacing_um said."""
    _export(tmp_path, vtk_export=True, constriction_spacing_um=30.0, constriction_length_um=40.0)

    sites = pv.read(tmp_path / f"{STEM}_pericytes.vtp")
    on_the_long_vessel = np.sort(sites.points[sites.point_data["edge_v"] == 1][:, 2])
    assert np.allclose(on_the_long_vessel, [20.0, 50.0, 80.0])


@pytest.mark.parametrize("spacing,length", [(0.0, 40.0), (100.0, 0.0)])
def test_a_spacing_or_length_of_zero_writes_an_empty_pericytes_file(tmp_path, spacing, length):
    """The schema allows 0, which places no periodic site; it must not stop the export."""
    _export(tmp_path, vtk_export=True, constriction_spacing_um=spacing, constriction_length_um=length)

    sites = pv.read(tmp_path / f"{STEM}_pericytes.vtp")
    assert sites.n_points == 0
    assert (tmp_path / f"{STEM}_vessels.vtp").is_file()


# --- one folder ---------------------------------------------------------------


def test_the_vtk_files_go_with_the_statistics_into_the_runs_folder(tmp_path):
    """The VTK files were written to settings["output_dir"], everything else to
    the network's folder; a resumed run with another output folder split them."""
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    _export(
        tmp_path, volume_dir=run_folder, output_dir=tmp_path / "elsewhere",
        vtk_export=True, statistics=True,
    )

    assert (run_folder / f"{STEM}_vessels.vtp").is_file()
    assert (run_folder / f"{STEM}_statistics.csv").is_file()
    assert not (tmp_path / "elsewhere").exists()


def test_a_resumed_volume_writes_into_the_folder_the_run_names_now(tmp_path):
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    old = stages.SkeletonisedVolume(
        image=image, skeleton=None, voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0), output_dir=tmp_path / "first",
    )

    moved = stages._volume_writing_to(old, tmp_path / "second")

    assert moved.output_dir == tmp_path / "second"
    assert (tmp_path / "second").is_dir()
    # The image is the same array, and the checkpoint's own volume is untouched.
    assert moved.image is image
    assert old.output_dir == tmp_path / "first"
    assert stages._volume_writing_to(old, tmp_path / "first") is old


# --- weighted betweenness and communities --------------------------------------


def test_the_weighted_reports_are_left_out_with_betweenness_and_communities_off(tmp_path):
    """Three betweenness passes ran whatever statistics_betweenness and
    statistics_community said."""
    G, _solution = _export(
        tmp_path, statistics=True, statistics_betweenness=False, statistics_community=False
    )

    assert (tmp_path / f"{STEM}_statistics.csv").is_file()
    assert list(tmp_path.glob("*_betweenness_communities_*.json")) == []
    assert stages.WEIGHTED_MEASURES_SECTION not in G.graph[stages.STATISTICS_REPORT]["sections"]


@pytest.mark.parametrize("on", ["statistics_betweenness", "statistics_community"])
def test_either_measure_writes_the_weighted_reports(tmp_path, on):
    off = {"statistics_betweenness": False, "statistics_community": False}
    _export(tmp_path, statistics=True, **{**off, on: True})

    assert len(list(tmp_path.glob(f"{STEM}_betweenness_communities_*.json"))) == 3


def test_the_weighted_reports_measure_the_network_the_statistics_measure(tmp_path):
    """The statistics see a bridge's two ends as one node; the JSONs saw the bridge."""
    _export(tmp_path, _forked_graph_with_a_bridge(), statistics=True)

    written = json.loads((tmp_path / f"{STEM}_betweenness_communities_edge_length.json").read_text())
    contracted = contract_bridges(_forked_graph_with_a_bridge())[0]
    expected = statistics.compute_weighted_betweenness_summary(
        contracted, source_attr="length", inverse_source_attr=False
    )
    assert written["Betweenness"]["Betweenness Max"] == pytest.approx(expected["Betweenness Max"])
    assert [entry["node"] for entry in written["Betweenness"]["Betweenness Top Nodes"]] == [
        entry["node"] for entry in expected["Betweenness Top Nodes"]
    ]


def test_a_perturbations_weighted_reports_measure_the_network_its_statistics_measure(tmp_path):
    """Regression: a perturbation's JSONs still saw the bridge the baseline's
    no longer did."""
    from haemolynx.visualization.perturbation_plots import (
        export_non_sweep_perturbation_artifacts,
    )

    G = _forked_graph_with_a_bridge()
    export_non_sweep_perturbation_artifacts(
        G, tmp_path, _settings(tmp_path), name_stem="dilated", betweenness_and_communities=True,
    )

    written = json.loads((tmp_path / "dilated_betweenness_communities_edge_length.json").read_text())
    contracted = contract_bridges(_forked_graph_with_a_bridge())[0]
    expected = statistics.compute_weighted_betweenness_summary(
        contracted, source_attr="length", inverse_source_attr=False
    )
    assert written["Betweenness"]["Betweenness Max"] == pytest.approx(expected["Betweenness Max"])
    assert [entry["node"] for entry in written["Betweenness"]["Betweenness Top Nodes"]] == [
        entry["node"] for entry in expected["Betweenness Top Nodes"]
    ]
    # The bridge itself is left on the perturbation's graph.
    assert G.edges[3, 4, 0][IS_ZERO_RESISTANCE]


def test_no_partition_is_shared_across_a_bridge(tmp_path):
    """The vessels layer partitions the graph with its bridges, the statistics
    without; one shared partition counted bridge ends the statistics lack."""
    settings = _settings(
        tmp_path, statistics=True, compute_vascular_communities=True,
        vascular_community_weighting="topology",
    )
    with_bridge = _forked_graph_with_a_bridge()
    without = _forked_graph_with_a_bridge()
    without.edges[3, 4, 0].pop(IS_ZERO_RESISTANCE)

    shared = dict(topology_report=True, weighted_report=True)
    assert stages.shared_vascular_community_partition(settings, with_bridge, **shared) is None
    assert stages.shared_vascular_community_partition(settings, without, **shared) is not None


# --- what the panel is told -------------------------------------------------------


def test_the_export_records_what_it_wrote(tmp_path):
    """The panel's note said "Wrote the VTK, statistics and plots." whatever was written."""
    _G, nothing = _export(tmp_path / "a")
    assert nothing.exported == ()

    _G, some = _export(tmp_path / "b", statistics=True, export_citations=True)
    assert some.exported == ("statistics", "citations")


def test_the_citations_drop_greedy_modularity_above_its_node_limit(tmp_path):
    """Above 1500 nodes every partition but full mode's is connected components."""
    from haemolynx.graph.communities import DEFAULT_MAX_NODES_EXACT
    from haemolynx.pipeline.citations import render_citations

    settings = _settings(tmp_path, statistics=True, statistics_mode="fast")
    name = "Greedy modularity communities"

    assert name in render_citations(settings)
    assert name in render_citations(settings, network_nodes=DEFAULT_MAX_NODES_EXACT)
    assert name not in render_citations(settings, network_nodes=DEFAULT_MAX_NODES_EXACT + 1)
    full = {**settings, "statistics_mode": "full"}
    assert name in render_citations(full, network_nodes=DEFAULT_MAX_NODES_EXACT + 1)
