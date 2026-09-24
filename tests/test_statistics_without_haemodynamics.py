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


def test_export_results_writes_bottleneck_and_shunt_results_onto_the_vessels(tmp_path):
    """The real export stage, with inlet/outlet nodes: the report gets the
    bottleneck/shunt rows, and the graph the vessels layer is built from
    gets the per-vessel attributes. Resistance weighting is asked for but
    haemodynamics is off, so both fall back to length and say so."""
    G = _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        statistics_route_weighting="resistance",
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert "Bottleneck Min-Cut Edge Count" in csv_text
    assert "Shunt Pathway Count" in csv_text
    assert "length (resistance unavailable)" in csv_text

    # 0-1-2 is the only route; 1-3 is a dead-end spur off it.
    assert G[0][1][0]["bottleneck_min_cut"] == "min_cut" or G[1][2][0]["bottleneck_min_cut"] == "min_cut"
    assert G[1][3][0]["bottleneck_route_share"] == pytest.approx(0.0)
    assert G[0][1][0]["bottleneck_route_share"] == pytest.approx(1.0)
    assert np.isnan(G[1][3][0]["shunt_route_ratio"])
    assert {data["shunt"] for _u, _v, data in G.edges(data=True)} == {"not_shunt"}


def test_export_results_skips_bottlenecks_and_shunts_when_toggled_off(tmp_path):
    G = _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        statistics_bottlenecks=False,
        statistics_shunts=False,
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    assert "Bottleneck" not in csv_text
    assert "Shunt" not in csv_text
    assert all(
        "bottleneck_min_cut" not in data and "shunt" not in data
        for _u, _v, data in G.edges(data=True)
    )


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


def test_compute_vascular_communities_runs_with_statistics_off(tmp_path):
    """Vascular communities colour the vessels layer; they are not part of
    the statistics report, so turning Statistics off must not switch them off."""
    G = _export_with_haemodynamics_off(
        tmp_path, statistics=False, compute_vascular_communities=True
    )
    assert not (tmp_path / "no_haemodynamics_statistics.csv").exists()
    labels = {data.get("vascular_community") for _u, _v, data in G.edges(data=True)}
    assert labels and None not in labels


def _spy_on_partitioning(monkeypatch):
    """Count every call that partitions the graph, across each module-level
    binding of the partitioning functions."""
    import networkx as nx

    import haemolynx.graph as graph_pkg
    from haemolynx.graph import communities as gc
    from haemolynx.statistics import network_measures as nm

    calls = []
    original_for_weighting = gc.communities_for_weighting
    original_greedy = nx.community.greedy_modularity_communities

    def _for_weighting(graph_arg, weighting, *args, **kwargs):
        calls.append(("for_weighting", weighting))
        return original_for_weighting(graph_arg, weighting, *args, **kwargs)

    def _greedy(*args, **kwargs):
        calls.append(("greedy", None))
        return original_greedy(*args, **kwargs)

    for module in (graph_pkg, gc, nm):
        monkeypatch.setattr(module, "communities_for_weighting", _for_weighting)
    monkeypatch.setattr(nx.community, "greedy_modularity_communities", _greedy)
    monkeypatch.setattr(nm, "greedy_modularity_communities", _greedy)
    return calls


def test_topology_vascular_communities_reuse_the_statistics_partition_in_fast_mode(
    tmp_path, monkeypatch
):
    """The default "topology" weighting is exactly the partition the
    statistics report's own "community" measure computes, so with both on it
    is computed once and shared -- and they agree on the community count."""
    calls = _spy_on_partitioning(monkeypatch)
    G = _export_with_haemodynamics_off(
        tmp_path, compute_vascular_communities=True, statistics_mode="fast"
    )
    assert calls.count(("for_weighting", "topology")) == 1

    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    domains = {
        data["vascular_community"]
        for _u, _v, data in G.edges(data=True)
        if data["vascular_community"].startswith("D")
    }
    count_row = next(line for line in csv_text.splitlines() if ",Community Count," in line)
    assert int(float(count_row.split(",")[2])) >= len(domains)


def test_topology_vascular_communities_reuse_the_statistics_partition_in_full_mode(
    tmp_path, monkeypatch
):
    calls = _spy_on_partitioning(monkeypatch)
    _export_with_haemodynamics_off(
        tmp_path, compute_vascular_communities=True, statistics_mode="full"
    )
    assert calls.count(("greedy", None)) == 1
    assert ("for_weighting", "topology") not in calls


def test_topology_vascular_communities_partition_on_their_own_without_the_report(
    tmp_path, monkeypatch
):
    """Nothing to reuse when the report's community measure is off."""
    calls = _spy_on_partitioning(monkeypatch)
    G = _export_with_haemodynamics_off(
        tmp_path, compute_vascular_communities=True, statistics_community=False
    )
    assert calls.count(("for_weighting", "topology")) == 1
    assert all("vascular_community" in data for _u, _v, data in G.edges(data=True))


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


#: The eight annotating analyses beyond bottlenecks/shunts, with the per-vessel
#: attributes each writes and one report row each always adds when it runs.
NETWORK_ANALYSES = {
    "statistics_occlusion_impact": (("occlusion_flow_loss", "occlusion_hypoperfused_length_um"), "Occlusion Impact Method"),
    "statistics_occlusion_curves": ((), "Occlusion Curves Route Weighting"),
    "statistics_current_flow": (("current_flow_share",), "Equivalent Inlet-Outlet Resistance"),
    "statistics_perfusion_territories": (
        ("arteriolar_territory", "venular_territory", "watershed", "arteriolar_watershed_margin"),
        "Arteriolar Territory Count",
    ),
    "statistics_transit_time": (("transit_time_s", "arrival_time_s"), "Transit Time Status"),
    "statistics_loop_hierarchy": (("loop_length_um",), "Vessels On No Loop"),
    "statistics_strahler": (("strahler_order",), "Strahler Method"),
    "statistics_algebraic_connectivity": (("fiedler_value", "fiedler_side"), "Algebraic Connectivity lambda_2"),
}


def test_export_results_runs_every_network_analysis_and_annotates_the_vessels(tmp_path):
    """All eight on, with inlet 0 and outlet 2 on the forked 0-1-2 (+ spur 1-3)
    graph and haemodynamics off: each adds its report rows, and every one that
    can run writes its per-vessel attributes. Transit time needs a solved
    flow, so it reports why not and writes nothing."""
    G = _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        **{name: True for name in NETWORK_ANALYSES},
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    for name, (attributes, row) in NETWORK_ANALYSES.items():
        assert row in csv_text, name
        for attribute in attributes:
            present = all(attribute in data for _u, _v, data in G.edges(data=True))
            assert present == (name != "statistics_transit_time"), (name, attribute)
    assert "needs a solved flow" in csv_text

    # 0-1-2 carries everything and blocking either vessel stops it; the spur nothing.
    assert G[0][1][0]["current_flow_share"] == pytest.approx(1.0)
    assert G[1][3][0]["current_flow_share"] == pytest.approx(0.0)
    assert G[0][1][0]["occlusion_flow_loss"] == pytest.approx(1.0)
    assert G[1][3][0]["occlusion_flow_loss"] == pytest.approx(0.0)
    # A tree: no vessel is on a loop.
    assert all(np.isnan(data["loop_length_um"]) for _u, _v, data in G.edges(data=True))


def test_export_results_skips_every_network_analysis_when_toggled_off(tmp_path):
    G = _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        **{name: False for name in NETWORK_ANALYSES},
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    for name, (attributes, row) in NETWORK_ANALYSES.items():
        assert row not in csv_text, name
        assert all(
            attribute not in data for attribute in attributes for _u, _v, data in G.edges(data=True)
        ), name


def test_the_network_analysis_master_toggle_gates_the_new_analyses_too(tmp_path):
    G = _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        statistics_network_analysis=False,
        **{name: True for name in NETWORK_ANALYSES},
    )
    csv_text = (tmp_path / "no_haemodynamics_statistics.csv").read_text()
    for name, (attributes, row) in NETWORK_ANALYSES.items():
        assert row not in csv_text, name
    assert all("current_flow_share" not in data for _u, _v, data in G.edges(data=True))


def test_export_results_passes_the_occlusion_settings_through(tmp_path, monkeypatch):
    from haemolynx.statistics import comprehensive

    seen = {}
    real = comprehensive.compute_occlusion_curves

    def spy(*args, **kwargs):
        seen["max_fraction"] = kwargs["max_fraction"]
        return real(*args, **kwargs)

    monkeypatch.setattr(comprehensive, "compute_occlusion_curves", spy)
    real_impact = comprehensive.compute_single_vessel_occlusion_impact

    def spy_impact(*args, **kwargs):
        seen["hypoperfusion_fraction"] = kwargs["hypoperfusion_fraction"]
        return real_impact(*args, **kwargs)

    monkeypatch.setattr(comprehensive, "compute_single_vessel_occlusion_impact", spy_impact)
    _export_with_haemodynamics_off(
        tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[2],
        statistics_occlusion_impact=True,
        statistics_occlusion_curves=True,
        statistics_occlusion_curve_max_fraction=0.3,
        statistics_occlusion_hypoperfusion_fraction=0.8,
    )
    assert seen == {"max_fraction": 0.3, "hypoperfusion_fraction": 0.8}
