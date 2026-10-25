"""How the vessels lie: junction shape, tissue coverage and vessel columns
(``statistics/layout.py``), on networks whose answers are known by hand."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.statistics import compute_vessel_statistics_by_measure, layout
from haemolynx.statistics.layout import (
    compute_junction_shape,
    compute_tissue_coverage,
    compute_vessel_columns,
    junction_angle_triples,
)


def _network(edges, extent=None):
    """A MultiGraph of straight (or given) polylines: each edge a list of points."""
    G = nx.MultiGraph()
    index: dict = {}

    def node(p):
        key = tuple(round(float(c), 6) for c in p)
        if key not in index:
            index[key] = len(index)
            G.add_node(index[key], pos=np.asarray(key, float))
        return index[key]

    for points in edges:
        points = [np.asarray(p, float) for p in points]
        G.add_edge(node(points[0]), node(points[-1]), voxels=[list(p) for p in points],
                   length=float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:]))))
    if extent is not None:
        G.graph["image_extent_um"] = tuple(float(e) for e in extent)
    return G


def _spoke(centre, degrees, length=30.0, axis_plane=(1, 2)):
    a = math.radians(degrees)
    end = np.asarray(centre, float).copy()
    end[axis_plane[0]] += length * math.cos(a)
    end[axis_plane[1]] += length * math.sin(a)
    return [centre, tuple(end)]


def test_a_flat_symmetric_y_is_wide_flat_and_not_acute():
    c = (50.0, 50.0, 50.0)
    G = _network([_spoke(c, 0), _spoke(c, 120), _spoke(c, 240)])
    rows = compute_junction_shape(G)
    assert rows["Three-Vessel Junctions Measured"] == 1
    assert rows["Junction Smallest Angle Median (degrees)"] == pytest.approx(120.0, abs=1e-6)
    assert rows["Junction Largest Angle Median (degrees)"] == pytest.approx(120.0, abs=1e-6)
    assert rows["Junction Angle Sum Median (degrees)"] == pytest.approx(360.0, abs=1e-6)
    assert rows["Acute Junction Share (fraction)"] == 0.0


def test_two_vessels_leaving_together_are_an_acute_pair():
    c = (50.0, 50.0, 50.0)
    G = _network([_spoke(c, 0), _spoke(c, 30), _spoke(c, 195)])
    (smallest, middle, largest), = junction_angle_triples(G)
    assert smallest == pytest.approx(30.0, abs=1e-6)
    assert largest == pytest.approx(180.0 - 15.0, abs=1e-6)
    assert compute_junction_shape(G)["Acute Junction Share (fraction)"] == 1.0


def test_a_corner_of_three_axes_is_not_flat():
    c = np.array([50.0, 50.0, 50.0])
    G = _network([[c, c + (30, 0, 0)], [c, c + (0, 30, 0)], [c, c + (0, 0, 30)]])
    assert compute_junction_shape(G)["Junction Angle Sum Median (degrees)"] == pytest.approx(270.0, abs=1e-6)


def test_a_vessel_is_read_along_its_first_microns_not_its_chord():
    """It leaves along x and turns after 10 um: read 8 um out, it runs along x."""
    c = np.array([50.0, 50.0, 50.0])
    bent = [c, c + (0, 0, 10), c + (0, 30, 10)]
    G = _network([bent, [c, c + (0, 0, -30)], [c, c + (0, 30, 0)]])
    triples = junction_angle_triples(G)
    # x against -x 180, x against y 90, -x against y 90: the bend is not read.
    assert sorted(triples[0]) == pytest.approx([90.0, 90.0, 180.0], abs=1e-6)
    assert layout.JUNCTION_DIRECTION_REACH_UM == 8.0


def test_four_vessel_junctions_are_counted_not_measured():
    c = (50.0, 50.0, 50.0)
    G = _network([_spoke(c, a) for a in (0, 90, 180, 270)])
    rows = compute_junction_shape(G)
    assert rows["Three-Vessel Junctions Measured"] == 0
    assert rows["Junctions of Four or More Vessels"] == 1
    assert rows["Acute Junction Share (fraction)"].startswith("N/A")


def test_coverage_of_one_vessel_through_a_box_by_hand():
    """A vessel along x through (y, z) = (50, 50) of a 100 um box, probes on a
    10 um grid: 12 of each 100 (y, z) probes lie within 20 um, 20 beyond 50."""
    G = _network([[(50.0, 50.0, 0.0), (50.0, 50.0, 100.0)]], extent=(100.0, 100.0, 100.0))
    rows = compute_tissue_coverage(G)
    assert rows["Coverage Probe Points"] == 1000
    assert rows["Coverage Box"] == "image"
    assert rows["Box Within 20 um of a Vessel (fraction)"] == pytest.approx(0.12)
    assert rows["Box Beyond 50 um of a Vessel (fraction)"] == pytest.approx(0.20)
    # The vessel is sampled every 2 um and the probes sit 1 um between two
    # samples along x, so each distance reads up to 0.1 um long.
    offsets = np.arange(5.0, 100.0, 10.0) - 50.0
    by_hand = np.hypot(*np.meshgrid(offsets, offsets)).ravel()
    assert rows["Median Distance to Nearest Vessel (microns)"] == pytest.approx(np.median(by_hand), abs=0.1)
    assert rows["90th Percentile Distance to Nearest Vessel (microns)"] == pytest.approx(np.percentile(by_hand, 90), abs=0.1)


def test_without_an_image_box_the_vessels_bound_it():
    G = _network([[(0.0, 0.0, 0.0), (0.0, 0.0, 100.0)], [(0.0, 0.0, 0.0), (40.0, 40.0, 0.0)]])
    assert compute_tissue_coverage(G)["Coverage Box"] == "network bounding box"


def test_a_large_box_is_read_at_a_seeded_sample(monkeypatch):
    monkeypatch.setattr(layout, "COVERAGE_MAX_PROBES", 500)
    G = _network([[(50.0, 50.0, 0.0), (50.0, 50.0, 100.0)]], extent=(100.0, 100.0, 100.0))
    first, again = compute_tissue_coverage(G), compute_tissue_coverage(G)
    assert first["Coverage Probe Points"] == 500
    assert first == again


def _columns(xy, height=60.0):
    return [[(0.0, y, x), (height, y, x)] for y, x in xy]


def test_evenly_spaced_columns_read_above_one_and_bunched_ones_below():
    lattice = [(y, x) for y in range(10, 100, 20) for x in range(10, 100, 20)]
    even = compute_vessel_columns(_network(_columns(lattice), extent=(60.0, 100.0, 100.0)))
    assert even["Column Vessels (within 30 degrees of z)"] == 25
    assert even["Column Spacing Median (microns)"] == pytest.approx(20.0)
    assert even["Column Spacing Clark-Evans Ratio"] > 1.5
    assert even["Columns per mm² of Image Plane"] == pytest.approx(25 / (100 * 100) * 1e6)
    pairs = [(y + dy, x) for y, x in lattice[:12] for dy in (0.0, 2.0)]
    bunched = compute_vessel_columns(_network(_columns(pairs), extent=(60.0, 100.0, 100.0)))
    assert bunched["Column Spacing Clark-Evans Ratio"] < 0.5


def test_a_column_is_within_thirty_degrees_of_z_and_its_share_is_by_length():
    tilt = lambda deg, y: [(0.0, y, 50.0), (40.0 * math.cos(math.radians(deg)), y + 40.0 * math.sin(math.radians(deg)), 50.0)]  # noqa: E731
    G = _network([tilt(25, 10.0), tilt(40, 60.0), [(0.0, 90.0, 0.0), (0.0, 90.0, 40.0)]], extent=(60.0, 100.0, 100.0))
    rows = compute_vessel_columns(G)
    assert rows["Column Vessels (within 30 degrees of z)"] == 1
    assert rows["Column Share of Vessel Length (fraction)"] == pytest.approx(40.0 / 120.0)
    assert rows["Column Spacing Clark-Evans Ratio"].startswith("N/A")


def test_the_three_measures_are_sections_of_the_statistics():
    c = (50.0, 50.0, 50.0)
    G = _network([_spoke(c, 0), _spoke(c, 120), _spoke(c, 240)], extent=(100.0, 100.0, 100.0))
    sections = compute_vessel_statistics_by_measure(G, enabled_measures=frozenset({"junction_shape", "tissue_coverage", "vessel_columns"}))
    assert list(sections) == ["junction_shape", "tissue_coverage", "vessel_columns"]
    assert sections["junction_shape"]["Junction Smallest Angle Median (degrees)"] == pytest.approx(120.0)


def test_each_measure_has_its_own_switch():
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import statistics_arguments

    schema = default_schema()
    values = schema.defaults()
    for name in ("junction_shape", "tissue_coverage", "vessel_columns"):
        assert schema[f"statistics_{name}"].default is True
    values["statistics_tissue_coverage"] = False
    enabled = statistics_arguments(values)["enabled_measures"]
    assert {"junction_shape", "vessel_columns"} <= enabled and "tissue_coverage" not in enabled
