"""The statistics cross-check (``scripts/stats_cross_check.py``, release review
``stats.8a``): its table and readers, and the comparison it records on the
synthetic dense capillary bed, pinned.

On the bed every number has a known answer (``design_segments``): skan, which
reads the mask literally, must find the bed as segmented, and HaemoLynx must
part from it only where one of its own definitions says so -- each difference
pinned to the step that makes it. maskel and VesselVio need environments of
their own (Python >= 3.13; GPL-3, run as a program) and are not run here; their
readers are tested on files in their formats.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skan")

from stats_cross_check import haemolynx_side, maskel_side, skan_side, vesselvio_files
from stats_cross_check.measures import (
    Measurements,
    comparison_table,
    read_measurements,
    relative_difference,
    summary,
    tortuosity_of,
    weighted_median,
    write_measurements,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from dense_capillary_fixtures import (  # noqa: E402
    PAIR_YX_UM,
    PLANE_Z_UM,
    centreline_samples,
    dense_capillary_bed,
    design_segments,
    design_topology,
)


def _measurements(tool="t", lengths=(10.0, 20.0), chords=(10.0, 10.0), diameters=(4.0, np.nan), **kw):
    tortuosity, loops = tortuosity_of(lengths, chords)
    return Measurements(
        tool=tool, total_length_um=float(sum(lengths)), segments=len(lengths),
        branch_points=kw.get("branch_points", 1), end_points=kw.get("end_points", 2),
        segment_lengths_um=np.asarray(lengths), tortuosity=tortuosity,
        diameters_um=np.asarray(diameters), loops=loops,
    )


# --- the pure parts ---------------------------------------------------------


def test_a_loop_has_no_tortuosity_and_is_counted():
    tortuosity, loops = tortuosity_of([10.0, 30.0, 12.0], [10.0, 0.0, 6.0])
    np.testing.assert_array_equal(tortuosity, [1.0, np.nan, 2.0])
    assert loops == 1


def test_per_segment_arrays_must_run_segment_by_segment():
    with pytest.raises(ValueError, match="different lengths"):
        Measurements(tool="t", total_length_um=1.0, segments=2, branch_points=0, end_points=2,
                     segment_lengths_um=[1.0, 2.0], tortuosity=[1.0], diameters_um=[1.0, 2.0])


def test_length_weighted_median_is_the_value_at_a_typical_micron():
    # One 90 um vessel at 8 um and nine 1 um pieces at 4 um: most pieces are
    # 4 um, most of the centreline is 8 um.
    values = np.array([8.0] + [4.0] * 9 + [np.nan])
    lengths = np.array([90.0] + [1.0] * 9 + [5.0])
    assert weighted_median(values, lengths) == 8.0
    assert np.nanmedian(values) == 4.0
    assert np.isnan(weighted_median(np.array([np.nan]), np.array([1.0])))


def test_summary_reads_the_distributions_without_nan():
    m = _measurements(lengths=(10.0, 20.0, 30.0, 50.0), chords=(10.0, 10.0, 15.0, 40.0),
                      diameters=(4.0, 5.0, np.nan, 6.0))
    row = summary(m)
    assert row["Total centreline length (um)"] == 110.0
    assert row["Segment length, median (um)"] == 25.0
    assert row["Tortuosity, median"] == 1.625
    assert row["Tortuosity, length-weighted median"] == 1.25
    assert row["Segments with tortuosity > 1.1"] == 3
    assert row["Diameter, median (um)"] == 5.0
    assert row["Diameter, length-weighted median (um)"] == 6.0
    assert row["Segments with a diameter"] == 3


def test_table_gives_each_tool_its_difference_from_haemolynx_and_the_design():
    reference = _measurements("HaemoLynx", lengths=(10.0, 10.0))
    other = _measurements("skan", lengths=(10.0, 12.0))
    table = comparison_table([reference, other], truth={"Total centreline length (um)": 25.0})
    line = next(row for row in table.splitlines() if row.startswith("| Total centreline length"))
    assert line == "| Total centreline length (um) | 25.0 | 20.0 | 22.0 (+10%) |"
    assert relative_difference(1.0, 0.0) != relative_difference(1.0, 0.0)  # nan


def test_measurements_round_trip_through_json(tmp_path):
    m = _measurements("skan", diameters=(4.0, 5.0))
    m.extra = {"Junction-pixel clusters": 3}
    write_measurements([m], tmp_path / "m.json")
    back = read_measurements(tmp_path / "m.json")[0]
    assert back.tool == "skan" and back.extra == {"Junction-pixel clusters": 3}
    np.testing.assert_array_equal(back.diameters_um, m.diameters_um)


def test_maskel_output_is_summed_over_its_objects():
    data = {
        "maskel_version": "2.0.2",
        "summary": [
            {"total_length": 30.0, "num_edges": 2, "num_bifurcations": 1, "num_endpoints": 2},
            {"total_length": 5.0, "num_edges": 1, "num_bifurcations": 0, "num_endpoints": 2},
        ],
        "branches": [
            {"branch-distance": 10.0, "euclidean-distance": 10.0, "mean_diameter": 4.0},
            {"branch-distance": 20.0, "euclidean-distance": 0.0, "mean_diameter": None},
            {"branch-distance": 5.0, "euclidean-distance": 4.0, "mean_diameter": 3.0},
        ],
    }
    m = maskel_side.measurements(data)
    assert (m.total_length_um, m.segments, m.branch_points, m.end_points) == (35.0, 3, 1, 4)
    assert m.loops == 1
    np.testing.assert_array_equal(m.tortuosity, [1.0, np.nan, 1.25])
    np.testing.assert_array_equal(m.diameters_um, [4.0, np.nan, 3.0])


def _write_vesselvio(folder: Path, name: str, segments, network_length: float) -> None:
    """A results folder as VesselVio writes it: the segment CSV (a header line
    naming the file, rows with the blank lines csv.writer leaves on Windows)
    and the dataset workbook (a title row, then the header)."""
    openpyxl = pytest.importorskip("openpyxl")
    (folder / "Segment Results").mkdir(parents=True)
    with (folder / "Segment Results" / f"{name}.csv").open("w", newline="") as f:
        f.write(f"Filename:,{name},ROI Name:,None\r\n\r\n")
        writer = csv.writer(f, lineterminator="\r\n\r\n")
        writer.writerow(["Segment ID", "Volume", "Length", "Surface Area", "Tortuosity",
                         "Mean Radius", "Max Radius", "Min Radius", "Radius Std. Dev."])
        for i, (length, tortuosity, radius) in enumerate(segments):
            writer.writerow([float(i), 1.0, length, 1.0, tortuosity, radius, radius, radius, 0.0])
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Main Results"
    sheet.append(["", "", "Main Results"])
    sheet.append(["File Name", "ROI Name", "roi_Volume", "Volume", "Network Length", "Surface Area",
                  "Branchpoints", "Endpoints", "Number of Segments"])
    sheet.append(["other", "None", "NA", 1.0, 999.0, 1.0, 9, 9, 9])
    sheet.append([name, "None", "NA", 1.0, network_length, 1.0, 1, 3, len(segments)])
    workbook.save(folder / "VesselVio 3D Dataset Analysis Results.xlsx")


def test_vesselvio_results_are_read_as_vesselvio_writes_them(tmp_path):
    _write_vesselvio(tmp_path, "bed", [(10.0, 1.0, 2.0), (12.0, 1.2, 2.5), (30.0, 0.0, 3.0)], 52.0)
    m = vesselvio_files.load(tmp_path, "bed")
    assert (m.total_length_um, m.segments, m.branch_points, m.end_points) == (52.0, 3, 1, 3)
    # VesselVio's tortuosity of a loop is 0: counted, not averaged.
    assert m.loops == 1
    np.testing.assert_array_equal(m.tortuosity, [1.0, 1.2, np.nan])
    assert m.diameters_um.tolist() == [4.0, 5.0, 6.0]


def test_vesselvio_segments_that_do_not_add_up_to_its_network_length_are_refused(tmp_path):
    _write_vesselvio(tmp_path, "bed", [(10.0, 1.0, 2.0)], 52.0)
    with pytest.raises(ValueError, match="workbook says"):
        vesselvio_files.load(tmp_path, "bed")


# --- the bed's design ---------------------------------------------------------


def test_the_bed_as_designed_and_as_segmented():
    designed = design_segments()
    assert len(designed) == 70
    assert math.isclose(sum(s.length_um for s in designed), 1200.8, abs_tol=1e-6)
    assert design_topology(designed) == {"branch_points": 31, "end_points": 22}
    # Each dropout cuts one vessel in two and takes a voxel's width with it.
    segmented = design_segments(as_segmented=True)
    assert len(segmented) == 72
    assert math.isclose(sum(s.length_um for s in segmented), 1200.8 - 2 * 0.98, abs_tol=1e-6)
    assert design_topology(segmented) == {"branch_points": 31, "end_points": 26}


# --- the comparison on the bed, pinned ------------------------------------------


@pytest.fixture(scope="module")
def bed_comparison(tmp_path_factory):
    import tifffile

    out = tmp_path_factory.mktemp("stats_cross_check")
    bed = dense_capillary_bed()
    mask_path = out / "dense_capillary_bed.tif"
    tifffile.imwrite(mask_path, bed.mask.astype(np.uint8) * 255)
    haemolynx, build, csv_path = haemolynx_side.run(mask_path, (0.98, 0.98, 1.0), out / "haemolynx")
    skan = skan_side.run(bed.mask, bed.voxel_size_zyx)
    return bed, haemolynx, build, csv_path, skan, dict(haemolynx_side.ledger(build, skan))


@pytest.mark.integration
@pytest.mark.slow
def test_skan_reads_the_bed_as_segmented(bed_comparison):
    *_rest, skan, _ledger = bed_comparison
    segmented = design_segments(as_segmented=True)
    assert (skan.segments, skan.branch_points, skan.end_points) == (72, 31, 26)
    # Every free end of a Lee skeleton stops short of where its vessel ends:
    # 20 vessels cut by the faces, each 1-2 um short, and two blind tips.
    shortfall = sum(s.length_um for s in segmented) - skan.total_length_um
    assert 30.0 < shortfall < 60.0
    assert skan.extra["Junction-pixel clusters"] == 31
    assert np.nanmedian(skan.tortuosity) == pytest.approx(1.0, abs=1e-6)
    # The distance transform reads to the nearest background voxel's centre,
    # half a voxel past the wall: a 5 um capillary reads 5.5-6 um.
    assert 5.4 < np.median(skan.diameters_um) < 6.0


@pytest.mark.integration
@pytest.mark.slow
def test_haemolynx_statistics_csv_on_the_bed_is_pinned(bed_comparison):
    _bed, haemolynx, _build, csv_path, skan, _ledger = bed_comparison
    rows = haemolynx_side.read_statistics_csv(csv_path)
    # If these move, a definition changed: rerun scripts/stats_cross_check.py
    # and update stats.8a's evidence in RELEASE_REVIEW.md.
    assert (haemolynx.segments, haemolynx.branch_points, haemolynx.end_points) == (66, 27, 20)
    assert haemolynx.total_length_um == pytest.approx(1086.0, rel=0.01)
    assert float(rows["Total Edge Length"]) == pytest.approx(haemolynx.total_length_um, rel=1e-5)
    assert np.nanmean(haemolynx.tortuosity) == pytest.approx(1.0, abs=0.02)
    # The mask's own cross-section, read away from junctions: the design's 5 um.
    assert np.nanmedian(haemolynx.diameters_um) == pytest.approx(5.0, rel=0.05)
    assert np.nanmedian(haemolynx.diameters_um) < np.median(skan.diameters_um) - 0.4


@pytest.mark.integration
@pytest.mark.slow
def test_each_difference_from_skan_is_one_haemolynx_definition(bed_comparison):
    bed, haemolynx, build, _csv, skan, ledger = bed_comparison
    raw = ledger["skan on the mask's Lee skeleton"]
    cleaned = ledger["skan on HaemoLynx's cleaned skeleton (skeleton cleaning)"]
    first = ledger["topology step: build_graph_segment_skan_stitched_loops"]
    collapsed = ledger["topology step: collapse_node_clusters"]
    consolidated = ledger["topology step: consolidate_lumen"]
    smoothed = ledger["centreline smoothing (alone, on the last step's graph)"]
    final = ledger["statistics: zero-resistance bridges contracted"]

    # Skeleton cleaning bridges the two one-voxel dropouts (four ends, two
    # segments fewer) and, closing and thinning again, pulls each free end
    # back a voxel or two.
    assert (raw["end_points"] - cleaned["end_points"], raw["segments"] - cleaned["segments"]) == (4, 2)
    assert -40.0 < cleaned["length"] - raw["length"] < -10.0
    # On one skeleton, HaemoLynx's graph and skan measure the same length.
    assert first["length"] == pytest.approx(cleaned["length"], rel=1e-3)
    assert first["segments"] == cleaned["segments"]
    # Cluster collapse folds the wide vessel's two short face stubs into their junctions.
    assert (cleaned["end_points"] - collapsed["end_points"], cleaned["segments"] - collapsed["segments"]) == (2, 2)
    # The short-loop rule (200 um round a hole under 10 um) takes out both
    # parallel vessels, 4 um apart: four junctions and their length go, and
    # the four nodes they leave beside 4-way crossings stay (degree 2 is
    # merged only away from a junction of degree 4 or more).
    assert collapsed["branch_points"] - consolidated["branch_points"] == 4
    assert 35.0 < collapsed["length"] - consolidated["length"] < 45.0
    G = build.graph
    points = np.vstack([np.asarray(d["voxels"], float) for *_e, d in G.edges(data=True)])
    for start, end in PAIR_YX_UM:
        line = centreline_samples(np.array([PLANE_Z_UM, *start]), np.array([PLANE_Z_UM, *end]))
        nearest = np.min(np.linalg.norm(line[:, None, :] - points[None, :, :], axis=2), axis=1)
        assert np.mean(nearest < 2.0) < 0.5
    assert sum(1 for n in G if G.degree(n) == 2) == 4
    # Smoothing straight centrelines changes little; nothing else is left.
    assert abs(smoothed["length"] - consolidated["length"]) < 0.01 * consolidated["length"]
    assert final == {
        "length": pytest.approx(haemolynx.total_length_um, rel=1e-5),
        "segments": haemolynx.segments,
        "branch_points": haemolynx.branch_points,
        "end_points": haemolynx.end_points,
    }


@pytest.mark.integration
@pytest.mark.slow
def test_same_diameter_definition_same_answer(bed_comparison):
    """HaemoLynx's inscribed-radius method is the definition skan reads: the
    two agree, so the cross-section's lower reading is its definition."""
    _bed, haemolynx, _build, _csv, skan, _ledger = bed_comparison
    median, _weighted = haemolynx.extra[haemolynx_side.INSCRIBED]
    assert median == pytest.approx(np.median(skan.diameters_um), rel=0.08)
    assert median > np.nanmedian(haemolynx.diameters_um) + 0.4


@pytest.mark.integration
@pytest.mark.slow
def test_the_skeleton_left_out_of_every_drawn_vessel_is_the_two_parallel_vessels(bed_comparison):
    """Everything else HaemoLynx's graph leaves out of the raw skeleton lies in
    the lumen of a vessel it drew; what does not is the vessels the short-loop
    rule removed."""
    from scipy.spatial import cKDTree

    bed, _haemolynx, build, _csv, _skan, _ledger = bed_comparison
    raw = skan_side.lee_skeleton(bed.mask)
    voxels, inside = haemolynx_side.in_drawn_vessels(build, raw)
    assert len(voxels) == np.count_nonzero(raw)
    assert 0.02 < 1.0 - inside.mean() < 0.06
    pair = np.vstack([
        centreline_samples(np.array([PLANE_Z_UM, *start]), np.array([PLANE_Z_UM, *end]))
        for start, end in PAIR_YX_UM
    ])
    assert np.all(cKDTree(pair).query(voxels[~inside])[0] < 1.5)
    reach = haemolynx_side.skeleton_in_haemolynx_lumen(build, raw)
    assert reach["share in a drawn vessel"] == pytest.approx(inside.mean())
