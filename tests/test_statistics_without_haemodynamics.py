"""Statistics on, haemodynamics off: the one branch that reached missing names.

With `run_haemodynamics=False` there are no edge resistances to weight paths
by, so `export_results` fills the resistance half of the betweenness/community
report with "N/A" strings and computes the edge-length half itself, through
`statistics.compute_weighted_betweenness_summary` and
`statistics.compute_weighted_communities_summary`. Neither was re-exported from
`statistics/__init__.py`, so that configuration -- and only that configuration,
since `compute_betweenness_and_community_measurements` is what the solved run
calls -- died with AttributeError after the statistics CSVs had been written.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

import haemolynx
from haemolynx.pipeline import default_schema, stages

IMAGE_SHAPE = (8, 10, 12)

#: What `export_results` calls through the `statistics` namespace. Kept as
#: literal names so the contract is readable, not derived from the source.
NAMES_THE_EXPORT_STAGE_CALLS = [
    "compute_comprehensive_vessel_statistics",
    "export_statistics_to_csv",
    "compute_branch_order_statistics",
    "export_branch_order_statistics_to_csv",
    "compute_betweenness_and_community_measurements",
    "compute_weighted_betweenness_summary",
    "compute_weighted_communities_summary",
    "run_3d_measurement_to_cell_mask",
]


@pytest.mark.parametrize("name", NAMES_THE_EXPORT_STAGE_CALLS)
def test_the_export_stage_can_reach_the_functions_it_calls(name):
    assert hasattr(haemolynx.statistics, name), (
        f"pipeline/stages.py calls statistics.{name}(...) but the statistics "
        "package does not re-export it"
    )
    assert callable(getattr(haemolynx.statistics, name))


def _forked_vessel_graph() -> nx.MultiGraph:
    """Four nodes around one junction, so betweenness is not uniformly zero."""
    G = nx.MultiGraph()
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (4.0, 0.0, 0.0),
        2: (8.0, 3.0, 0.0),
        3: (8.0, -3.0, 0.0),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    for u, v in ((0, 1), (1, 2), (1, 3)):
        start = np.asarray(positions[u])
        end = np.asarray(positions[v])
        G.add_edge(
            u,
            v,
            key=0,
            length=float(np.linalg.norm(end - start)),
            branch_order="B01",
            voxels=[tuple(start), tuple(end)],
        )
    return G


def _export_with_haemodynamics_off(tmp_path):
    settings = default_schema().defaults()
    settings.update(
        {
            "input_path": tmp_path / "no_haemodynamics.tif",
            "statistics": True,
            "statistics_mode": "fast",
            "run_haemodynamics": False,
            "measurement_3d_to_cell_mask": False,
            "vtk_export": False,
            "visualize_vtk": False,
            "visualize_results": False,
        }
    )
    volume = stages.SkeletonisedVolume(
        image=np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        skeleton=np.zeros(IMAGE_SHAPE, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    G = _forked_vessel_graph()
    stages.export_results(
        settings,
        stages.VesselNetwork(graph=G, volume=volume),
        stages.HaemodynamicModel(graph=G),
        stages.Solution(),
    )
    return G


def test_the_export_stage_runs_and_writes_both_reports_without_haemodynamics(tmp_path):
    G = _export_with_haemodynamics_off(tmp_path)

    stem = "no_haemodynamics"
    assert (tmp_path / f"{stem}_statistics.csv").exists()
    assert (tmp_path / f"{stem}_branch_statistics.csv").exists()

    resistance = json.loads(
        (tmp_path / f"{stem}_betweenness_communities_resistance.json").read_text()
    )
    length = json.loads(
        (tmp_path / f"{stem}_betweenness_communities_edge_length.json").read_text()
    )

    # No resistances exist to weight by, so that half is explicitly not a number.
    assert resistance["Betweenness"]["Betweenness Method"].startswith("N/A")
    assert resistance["Communities"]["Community Method"].startswith("N/A")

    # The edge-length half is measured, and measured from `length` -- the
    # junction is the only node any shortest path passes through.
    assert length["Betweenness"]["Betweenness Method"] == "exact_weighted"
    assert length["Betweenness"]["Betweenness Max"] > 0.0
    assert length["Betweenness"]["Betweenness Top Nodes"][0]["node"] == 1
    assert length["Communities"]["Community Count"] >= 1

    # `export_results` reports on the graph; it must not have edited it.
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3


def test_the_edge_length_report_matches_calling_the_functions_directly(tmp_path):
    """It is the same numbers by the same route, not a namespace-only smoke test."""
    _export_with_haemodynamics_off(tmp_path)

    length = json.loads(
        (tmp_path / "no_haemodynamics_betweenness_communities_edge_length.json").read_text()
    )
    expected_betweenness = haemolynx.statistics.compute_weighted_betweenness_summary(
        _forked_vessel_graph(), source_attr="length", inverse_source_attr=False
    )
    expected_communities = haemolynx.statistics.compute_weighted_communities_summary(
        _forked_vessel_graph(), source_attr="length", inverse_source_attr=False
    )

    assert length["Betweenness"]["Betweenness Mean"] == pytest.approx(
        expected_betweenness["Betweenness Mean"]
    )
    assert length["Betweenness"]["Betweenness Max"] == pytest.approx(
        expected_betweenness["Betweenness Max"]
    )
    assert length["Communities"]["Community Count"] == expected_communities["Community Count"]
