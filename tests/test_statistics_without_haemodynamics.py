"""Statistics on, haemodynamics off: the one branch that reached missing names.

With `run_haemodynamics=False` there is neither a resistance nor a solved flow
to weight paths by, so `export_results` fills those two-thirds of the
betweenness/community report with "N/A" strings and computes the edge-length
third itself, through `statistics.compute_weighted_betweenness_summary` and
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


def _export_with_haemodynamics_off(tmp_path, **overrides):
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
    settings.update(overrides)
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


def test_the_export_stage_runs_and_writes_all_three_reports_without_haemodynamics(tmp_path):
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
    flow = json.loads(
        (tmp_path / f"{stem}_betweenness_communities_edge_flow.json").read_text()
    )

    # No resistances exist to weight by, so that third is explicitly not a number.
    assert resistance["Betweenness"]["Betweenness Method"].startswith("N/A")
    assert resistance["Communities"]["Community Method"].startswith("N/A")

    # No haemodynamics means no solved flow either -- same as resistance.
    assert flow["Betweenness"]["Betweenness Method"].startswith("N/A")
    assert flow["Communities"]["Community Method"].startswith("N/A")

    # The edge-length third is measured, and measured from `length` -- the
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


def test_the_statistics_csv_includes_the_new_topology_measures(tmp_path):
    _export_with_haemodynamics_off(tmp_path)
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    for label in (
        "Cyclomatic Number",
        "Degree Assortativity",
        "Rich Club Coefficient",
        "Max Core Number",
        "Flow Hierarchy",
    ):
        assert label in csv_text


def test_export_results_passes_inlet_outlet_nodes_to_network_robustness(tmp_path):
    """Regression: export_results must forward settings["inlet_nodes"]/
    ["outlet_nodes"] (already resolved to concrete node IDs by
    assign_boundaries earlier in the pipeline) to compute_comprehensive_
    vessel_statistics, so network_robustness's perfusion-critical bridge/
    articulation-point classification actually runs on a real pipeline
    settings dict, not just when called directly."""
    _export_with_haemodynamics_off(
        tmp_path, inlet_nodes=[0], outlet_nodes=[2],
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert "Critical Bridge Edge Count" in csv_text
    assert "Critical Articulation Point Count" in csv_text


def test_compute_vascular_communities_off_by_default_leaves_the_graph_untouched(tmp_path):
    G = _export_with_haemodynamics_off(tmp_path)
    assert all("vascular_community" not in data for _u, _v, data in G.edges(data=True))


def test_compute_vascular_communities_on_writes_domain_labels_without_haemodynamics(tmp_path):
    """The default weighting, "topology", needs no resistance or solved flow
    -- it must work in the same haemodynamics-off configuration this whole
    file exercises."""
    from haemolynx.graph.communities import BOUNDARY_LABEL

    G = _export_with_haemodynamics_off(tmp_path, compute_vascular_communities=True)
    labels = {data.get("vascular_community") for _u, _v, data in G.edges(data=True)}
    assert labels
    assert all(label is not None for label in labels)
    assert all(label == BOUNDARY_LABEL or label.startswith("D") for label in labels)


def test_statistics_network_analysis_off_removes_every_network_measure_from_the_csv(tmp_path):
    """statistics_network_analysis is the GUI's own "Connectivity/Network
    Analysis" master toggle (nested under statistics) -- turning it off must
    drop every measure it groups from the statistics CSV even though each
    individual statistics_<measure> flag is still True, the same
    "outer gate checked explicitly" enforcement use_edt_diameter_crosscheck's
    children get from haemodynamics.apply.assign_edge_diameters."""
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    _export_with_haemodynamics_off(on_dir, input_path=on_dir / "on.tif")
    _export_with_haemodynamics_off(
        off_dir, input_path=off_dir / "off.tif", statistics_network_analysis=False,
    )

    on_csv = (on_dir / "on_statistics.csv").read_text()
    off_csv = (off_dir / "off_statistics.csv").read_text()

    for label in (
        "Cyclomatic Number", "Degree Assortativity", "Rich Club Coefficient",
        "Max Core Number", "Flow Hierarchy", "Bridge Edge Count",
        "Path Efficiency", "Community Count", "Betweenness Mean",
    ):
        assert label in on_csv, label
        assert label not in off_csv, label

    # Nothing outside the new group is affected.
    assert "Total Nodes" in off_csv
    assert "Average Tortuosity Index" in off_csv


def test_statistics_network_analysis_on_still_respects_each_measures_own_toggle(tmp_path):
    """The master being on does not force every child measure on -- each
    keeps its own independent statistics_<measure> toggle underneath it."""
    G = _export_with_haemodynamics_off(
        tmp_path,
        statistics_network_analysis=True,
        statistics_betweenness=False,
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert "Cyclomatic Number" in csv_text  # still on
    assert "Betweenness Mean" not in csv_text  # individually turned off
    assert G.number_of_nodes() == 4


def test_matching_vascular_community_weighting_reuses_the_statistics_partition(tmp_path, monkeypatch):
    """Regression: when vascular_community_weighting matches a weighting the
    statistics report already computes (here "length", the one weighting
    available with haemodynamics off), export_results must compute that
    partition once and share it, not run greedy modularity on the same
    graph twice."""
    import haemolynx.graph as graph_pkg
    from haemolynx.graph import communities as gc
    from haemolynx.statistics import network_measures as nm

    calls = []
    original = gc.communities_for_weighting

    def _spy(graph_arg, weighting, *args, **kwargs):
        calls.append(weighting)
        return original(graph_arg, weighting, *args, **kwargs)

    # Three separate module-level bindings of the same function
    # (pipeline/stages.py via the haemolynx.graph package, graph/communities.py's
    # own internal calls, statistics/network_measures.py's own import) --
    # patch all three so a regression anywhere still shows up as an extra call.
    monkeypatch.setattr(graph_pkg, "communities_for_weighting", _spy)
    monkeypatch.setattr(gc, "communities_for_weighting", _spy)
    monkeypatch.setattr(nm, "communities_for_weighting", _spy)
    _export_with_haemodynamics_off(
        tmp_path,
        compute_vascular_communities=True,
        vascular_community_weighting="length",
    )

    assert calls.count("length") == 1


def test_a_disabled_statistics_measure_setting_is_missing_from_the_exported_csv(tmp_path):
    """Stage-wiring regression test for the statistics_<measure> settings:
    each is read from the right settings-dict key (statistics_murray_law,
    not e.g. a typo'd murray_law) and actually reaches
    compute_comprehensive_vessel_statistics's enabled_measures. Unit tests
    on that function alone (tests/test_statistics.py) cannot catch a bug in
    this wiring, only in the function itself.
    """
    with_dir = tmp_path / "with"
    without_dir = tmp_path / "without"
    with_dir.mkdir()
    without_dir.mkdir()
    _export_with_haemodynamics_off(with_dir, input_path=with_dir / "with.tif")
    _export_with_haemodynamics_off(
        without_dir, input_path=without_dir / "without.tif",
        statistics_murray_law=False,
    )

    with_csv = (with_dir / "with_statistics.csv").read_text()
    without_csv = (without_dir / "without_statistics.csv").read_text()

    assert "Murray" in with_csv
    assert "Murray" not in without_csv
    # Nothing else should have been affected by disabling just this one.
    assert "Fractal Dimension" in without_csv
    assert "Tortuosity" in without_csv
