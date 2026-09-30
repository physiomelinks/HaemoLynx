"""Tests for statistics module."""
import pytest
import numpy as np
import networkx as nx

from ImageLynx.statistics import (
    compute_basic_statistics,
    compute_tortuosity_measures,
    compute_branching_statistics,
    compute_tree_asymmetry,
    compute_fractal_dimension,
    compute_path_efficiency,
    compute_vessel_density,
    compute_comprehensive_vessel_statistics,
)


def test_compute_basic_statistics(simple_graph):
    s = compute_basic_statistics(simple_graph, False)
    assert s["Total Nodes"] == 3
    assert s["Total Edges"] == 2


def test_compute_tortuosity_measures(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_tortuosity_measures(simple_graph, pos, False)
    assert "Average Tortuosity Index" in s


def test_compute_branching_statistics(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_branching_statistics(simple_graph, pos)
    assert "Average Branching Angle (degrees)" in s


def test_branching_points_are_counted_on_the_multigraph_not_its_simple_copy():
    """A node joined twice to one neighbour and once to another meets three vessels.

    Merging the parallel pair drops it to degree 2. The per-edge CSV, Figure 3 and
    cb_h1_vtk.py count it as a junction, so the printed stats must too (open item 40).
    """
    G = nx.MultiGraph()
    pos = {0: (0.0, 0.0, 0.0), 1: (10.0, 0.0, 0.0), 2: (0.0, 10.0, 0.0)}
    for n, p in pos.items():
        G.add_node(n, pos=p)
    G.add_edge(0, 1, length=10.0)
    G.add_edge(0, 1, length=12.0)
    G.add_edge(0, 2, length=10.0)

    merged = compute_branching_statistics(nx.Graph(G), pos)
    counted = compute_branching_statistics(nx.Graph(G), pos, count_graph=G)
    assert merged["Number of Branching Points"] == 0
    assert counted["Number of Branching Points"] == 1

    full = compute_comprehensive_vessel_statistics(G, pos)
    assert full["Number of Branching Points"] == 1


def test_compute_tree_asymmetry(simple_graph):
    s = compute_tree_asymmetry(simple_graph)
    assert "Tree Asymmetry Index" in s


def test_compute_fractal_dimension(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_fractal_dimension(simple_graph, pos)
    assert "Fractal Dimension" in s


def test_compute_path_efficiency(simple_graph):
    s = compute_path_efficiency(simple_graph, False)
    assert "Path Efficiency" in s


def test_compute_vessel_density(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_vessel_density(
        simple_graph, pos, (1, 1, 1), (10, 10, 10), False
    )
    assert "Total Vessel Length (microns)" in s


def test_compute_comprehensive_vessel_statistics(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_comprehensive_vessel_statistics(
        simple_graph, node_positions=pos, voxel_size=(1, 1, 1),
        image_dimensions=(10, 10, 10),
    )
    assert "Total Nodes" in s
    assert "Fractal Dimension" in s
import csv

import networkx as nx
import numpy as np
import pytest

from ImageLynx.statistics import (
    PER_EDGE_MORPHOMETRY_COLUMNS,
    compute_tortuosity_measures,
    export_per_edge_morphometry,
    write_per_edge_morphometry_csv,
)


def _morphometry_graph():
    """Edges spanning every provenance combination the export has to keep separable."""
    G = nx.MultiGraph()
    for n, pos in {0: (0, 0, 0), 1: (0, 0, 10), 2: (0, 10, 10), 3: (0, 20, 10)}.items():
        G.add_node(n, pos=np.array(pos, dtype=float))

    # A measured, smoothed, genuine edge with a detour: path 14 um over a 10 um separation.
    G.add_edge(0, 1, length=14.0, weight=14.0, branch_order="B01",
               voxels=[[0, 0, 0], [0, 3, 5], [0, 0, 10]],
               assigned_diameter_um=6.37, diameter_provenance="measured_edt",
               edt_diameter_um=6.37, fwhm_diameter_um=8.20,
               centreline_smoothing="bspline", edt_junction_trim="trimmed")
    # A fabricated diameter on an unsmoothed edge.
    G.add_edge(1, 2, length=10.0, weight=10.0, branch_order="B02",
               voxels=[[0, 0, 10], [0, 5, 10], [0, 10, 10]],
               assigned_diameter_um=4.0, diameter_provenance="synthetic_branch_order",
               centreline_smoothing="raw_fallback",
               edt_junction_trim="untrimmed_too_short")
    # A 2-point reconnection: tortuosity 1.0 by construction, not anatomy.
    G.add_edge(2, 3, length=10.0, weight=10.0, branch_order="B02",
               voxels=[[0, 10, 10], [0, 20, 10]],
               assigned_diameter_um=4.0, diameter_provenance="synthetic_branch_order",
               centreline_smoothing="raw_too_short", reconnected=True)
    return G


def test_per_edge_export_emits_one_row_per_edge_with_every_column():
    G = _morphometry_graph()
    rows = export_per_edge_morphometry(G)

    assert len(rows) == G.number_of_edges()
    for row in rows:
        assert set(row) == set(PER_EDGE_MORPHOMETRY_COLUMNS)


def test_per_edge_export_keeps_provenance_with_each_measurement():
    """A distribution mixing measured and fabricated values is not a measurement."""
    rows = export_per_edge_morphometry(_morphometry_graph())

    measured = [r for r in rows if r["diameter_provenance"] == "measured_edt"]
    synthetic = [r for r in rows if r["diameter_provenance"] == "synthetic_branch_order"]
    assert len(measured) == 1 and len(synthetic) == 2

    # Both raw estimators survive, so the FWHM/EDT comparison is reproducible from the table.
    assert measured[0]["edt_diameter_um"] == pytest.approx(6.37)
    assert measured[0]["fwhm_diameter_um"] == pytest.approx(8.20)
    assert synthetic[0]["edt_diameter_um"] is None


def test_per_edge_export_separates_the_smoothing_provenances():
    """section 1.4's numerator mixes operators; the tortuosity column has to be stratifiable."""
    rows = export_per_edge_morphometry(_morphometry_graph())
    by_tag = {r["centreline_smoothing"]: r for r in rows}

    assert set(by_tag) == {"bspline", "raw_fallback", "raw_too_short"}
    # The reconnection edge is straight by construction, and flagged as such twice over.
    assert by_tag["raw_too_short"]["tortuosity"] == pytest.approx(1.0)
    assert by_tag["raw_too_short"]["reconnected"] is True
    assert by_tag["raw_too_short"]["n_centreline_points"] == 2
    assert by_tag["bspline"]["reconnected"] is False


def test_per_edge_tortuosity_matches_the_summary_it_is_derived_from():
    """The table and the mean must not be able to disagree about an edge."""
    G = _morphometry_graph()
    positions = nx.get_node_attributes(G, "pos")

    rows = export_per_edge_morphometry(G, positions, True)
    summary = compute_tortuosity_measures(G, positions, True)

    values = [r["tortuosity"] for r in rows if r["tortuosity"] is not None]
    assert summary["Average Tortuosity Index"] == pytest.approx(np.mean(values))
    # 14 um of path across a 10 um separation.
    assert rows[0]["tortuosity"] == pytest.approx(1.4)
    assert rows[0]["curvature"] == pytest.approx((14.0 - 10.0) / 14.0)


def test_per_edge_csv_round_trips_with_a_stable_column_order(tmp_path):
    """A written table is an analysis input, so its schema has to be fixed."""
    rows = export_per_edge_morphometry(_morphometry_graph())
    path = write_per_edge_morphometry_csv(rows, tmp_path / "nested" / "per_edge.csv")

    assert path.exists()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(PER_EDGE_MORPHOMETRY_COLUMNS)
        written = list(reader)

    assert len(written) == len(rows)
    assert float(written[0]["tortuosity"]) == pytest.approx(1.4)
    assert written[2]["centreline_smoothing"] == "raw_too_short"


def test_tortuosity_summary_still_handles_missing_positions():
    assert compute_tortuosity_measures(_morphometry_graph(), None, True) == {
        "Average Tortuosity Index": "N/A (no position data)",
        "Average Curvature": "N/A (no position data)",
    }


def test_per_edge_export_carries_the_junction_trim_provenance():
    """An untrimmable edge's radius is inflated by its junction, and has to stay identifiable.

    Segments shorter than twice the exclusion keep their untrimmed median rather than being
    discarded, because dropping them would delete the short inter-junction capillaries that
    section 1.2 is a claim about. That trade only holds if the affected rows can be told
    apart in the exported distribution - otherwise a known-biased radius is pooled with
    corrected ones and nothing downstream can see it.
    """
    rows = export_per_edge_morphometry(_morphometry_graph())
    by_trim = {r["edt_junction_trim"]: r for r in rows}

    assert "edt_junction_trim" in PER_EDGE_MORPHOMETRY_COLUMNS
    assert by_trim["trimmed"]["edt_diameter_um"] == pytest.approx(6.37)
    assert "untrimmed_too_short" in by_trim
    # Edges never measured by EDT carry no trim tag at all, rather than a misleading one.
    assert None in by_trim


def test_image_volume_applies_the_voxel_size(simple_graph):
    """Re-run notes item 10: the image volume ignored the voxel size, 6.5x too small on CB."""
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_vessel_density(simple_graph, pos, (2.0, 1.0, 0.5), (10, 10, 10), False)
    assert s["Total Image Volume (micron³)"] == pytest.approx(1000.0)
    assert s["Vessel Density in Whole Image (microns/micron³)"] == pytest.approx(2.0 / 1000.0)


def test_image_volume_without_a_voxel_size_raises(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    with pytest.raises(ValueError, match="voxel_size"):
        compute_vessel_density(simple_graph, pos, None, (10, 10, 10), False)
    with pytest.raises(ValueError, match="voxel_size"):
        compute_comprehensive_vessel_statistics(
            simple_graph, node_positions=pos, image_dimensions=(10, 10, 10))


def test_no_image_dimensions_needs_no_voxel_size(simple_graph):
    pos = nx.get_node_attributes(simple_graph, "pos")
    s = compute_vessel_density(simple_graph, pos, None, None, False)
    assert s["Vessel Density in Whole Image (microns/micron³)"] == "N/A (no image dimension data)"


def test_bounding_box_volume_is_named_for_what_it_is():
    """The node-extent box is not a tissue volume, so its keys say "bounding box"."""
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([2.0, 3.0, 4.0]))
    G.add_edge(0, 1, length=6.0)
    s = compute_vessel_density(G, nx.get_node_attributes(G, "pos"), (1, 1, 1), None, False)
    assert s["Node Bounding-Box Volume (micron³)"] == pytest.approx(24.0)
    assert s["Vessel Density in Node Bounding Box (microns/micron³)"] == pytest.approx(0.25)
    assert "Vessel-Occupied Volume (micron³)" not in s
    assert "Vessel Density in Tissue (microns/micron³)" not in s


@pytest.mark.parametrize("script", [
    "carotid_image_to_model.py",
    "resistance_network_pipeline.py",
    "resistance_network_pipeline_for_Alice.py",
])
def test_pipelines_pass_the_voxel_size_to_the_statistics(script):
    import ast
    from pathlib import Path

    source = (Path(__file__).parent.parent / "examples" / script).read_text()
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_comprehensive_vessel_statistics"
    ]
    assert calls, f"{script} no longer calls compute_comprehensive_vessel_statistics"
    for call in calls:
        assert "voxel_size" in {kw.arg for kw in call.keywords}, (
            f"{script}:{call.lineno} computes statistics without voxel_size")
