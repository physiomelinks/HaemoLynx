"""Measurements around the pericytes of a pericyte mask (``measure_pericytes``)
and the perturbation that re-solves a network narrowed as they measure it
(``pericyte_measured_constriction``).

The segmentation here is a capillary as a tube of known width, narrowed over a
known stretch, so each reading, each junction distance and each resistance
has an answer worked out by hand.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics.constriction import profile_resistance
from haemolynx.haemodynamics.edt_diameter import (
    mask_diameters_at_points,
    measure_edge_diameters_from_binary_mask,
)
from haemolynx.haemodynamics.pericyte_measurements import (
    BRANCH_COLUMNS,
    PERICYTE_COLUMNS,
    PROFILE_COLUMNS,
    DiameterProfile,
    PlacedPericyte,
    branch_resistances,
    junction_distances,
    measure_pericytes,
    write_pericyte_measurements,
)
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel
from haemolynx.pipeline import default_schema, stages
from haemolynx.pipeline.checks import check_pericyte_measurements

SCHEMA = default_schema()

#: The capillary runs along x at (z, y) = (10, 10), from x = 10 (node 0, a
#: dead end) to x = 110 (node 1, where two short branches leave: a junction).
SHAPE = (21, 21, 141)
START_X, END_X = 10, 110
#: Its radius in voxels (1 um each), and the narrower radius over |x - 60| <= 6.
RADIUS, NARROW_RADIUS, NARROW_HALF_LENGTH = 3.0, 2.0, 6
#: The cross-section's area-equivalent diameters of those two discs of voxels
#: (29 and 13 voxels).
WIDE_UM = 2.0 * np.sqrt(29.0 / np.pi)
NARROW_UM = 2.0 * np.sqrt(13.0 / np.pi)


def _segmentation(narrowed: bool = True) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    zz, yy = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    for x in range(START_X, END_X + 1):
        radius = NARROW_RADIUS if narrowed and abs(x - 60) <= NARROW_HALF_LENGTH else RADIUS
        mask[:, :, x] |= (((zz - 10) ** 2 + (yy - 10) ** 2) <= radius**2).astype(np.uint8)
    return mask


def _graph(*, reversed_voxels: bool = False, junction_at_start: bool = False) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([10.0, 10.0, float(START_X)]))
    G.add_node(1, pos=np.array([10.0, 10.0, float(END_X)]))
    G.add_node(2, pos=np.array([10.0, 0.0, float(END_X)]))
    G.add_node(3, pos=np.array([10.0, 20.0, float(END_X)]))
    voxels = [np.array([10.0, 10.0, float(x)]) for x in range(START_X, END_X + 1)]
    if reversed_voxels:
        voxels = voxels[::-1]  # traced from node 1 to node 0
    G.add_edge(0, 1, key=0, voxels=voxels, length=float(END_X - START_X), branch_order="B01", diameter_um=6.0)
    for other in (2, 3):
        G.add_edge(1, other, key=0, voxels=[G.nodes[1]["pos"], G.nodes[other]["pos"]], length=10.0,
                   branch_order="B02", diameter_um=6.0)
    if junction_at_start:
        # A thick vessel's bridge at the dead end: the capillary opens into it there.
        G.add_node(4, pos=np.array([10.0, 10.0, 0.0]))
        G.add_edge(0, 4, key=0, voxels=[G.nodes[0]["pos"], G.nodes[4]["pos"]], length=10.0,
                   **{IS_ZERO_RESISTANCE: True})
    return G


def _pericytes(*blobs: tuple[slice, slice, slice]) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    for blob in blobs:
        mask[blob] = 1
    return mask


def _beside(x: int) -> tuple[slice, slice, slice]:
    """A 4 x 4 x 5 pericyte (5.3 um across) beside the capillary at *x*, its
    centroid 2.1 um off the centreline."""
    return (slice(11, 15), slice(7, 11), slice(x - 2, x + 3))


def _measure(G, pericytes, segmentation, **kwargs):
    kwargs.setdefault("model", PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0))
    return measure_pericytes(G, pericytes, (1.0, 1.0, 1.0), vessel_mask=segmentation, **kwargs)


# --- where each pericyte sits ---------------------------------------------------


def test_each_pericyte_is_placed_at_its_midline_on_the_nearest_capillary():
    result = _measure(_graph(), _pericytes(_beside(60)), _segmentation())

    (pericyte,) = result.pericytes
    assert pericyte.placed and pericyte.number == 1
    assert pericyte.edge == (0, 1, 0) and pericyte.branch_order == "B01"
    assert pericyte.arc_position_um == pytest.approx(50.0)
    assert pericyte.vessel_length_um == pytest.approx(100.0)
    assert pericyte.midline_um == pytest.approx((10.0, 10.0, 60.0))
    assert pericyte.distance_to_vessel_um == pytest.approx(np.hypot(2.5, 1.5))


def test_a_pericyte_that_is_not_placed_says_why():
    far = (slice(17, 21), slice(15, 19), slice(38, 43))  # 9 um off the centreline
    small = (slice(10, 12), slice(8, 10), slice(80, 82))  # 8 voxels: 2.5 um across
    large = (slice(4, 17), slice(4, 17), slice(90, 104))  # far wider than 12 um
    result = _measure(_graph(), _pericytes(_beside(60), far, small, large), _segmentation())

    reasons = {p.number: p.reason for p in result.pericytes}
    placed = [p.number for p in result.pericytes if p.placed]
    assert len(placed) == 1 and reasons[placed[0]] == ""
    assert sorted(reasons.values()) == sorted(
        ["", "too far from a capillary", "smaller than a pericyte", "larger than a pericyte"]
    )


def test_a_network_without_capillaries_places_no_pericyte():
    G = _graph()
    for _u, _v, data in G.edges(data=True):
        data["branch_order"] = "Art1"
    result = _measure(G, _pericytes(_beside(60)), _segmentation())
    assert [p.reason for p in result.pericytes] == ["no capillary in the network"]
    assert result.profiles == {} and result.branches == {}


# --- the diameter profile -----------------------------------------------------------


def test_a_profile_reads_every_step_each_way_out_to_its_reach():
    result = _measure(_graph(), _pericytes(_beside(60)), _segmentation(), step_um=2.0, reach_um=20.0)

    profile = result.profiles[1]
    np.testing.assert_allclose(profile.offsets_um, np.arange(-20.0, 21.0, 2.0))
    np.testing.assert_allclose(profile.arc_positions_um, 50.0 + np.arange(-20.0, 21.0, 2.0))
    np.testing.assert_allclose(profile.points_um[:, 2], 60.0 + np.arange(-20.0, 21.0, 2.0))


def test_a_profile_stops_where_the_vessel_ends():
    """At arc 90 on a 100 um vessel, 50 um each way reaches only 10 um toward v."""
    result = _measure(_graph(), _pericytes(_beside(100)), _segmentation(), step_um=2.0, reach_um=50.0)

    profile = result.profiles[1]
    assert profile.offsets_um.min() == pytest.approx(-50.0)
    assert profile.offsets_um.max() == pytest.approx(10.0)
    assert len(profile.offsets_um) == 31
    assert profile.arc_positions_um.max() <= 100.0 + 1e-9


def test_a_profile_reads_the_segmentations_width_and_its_narrowing():
    result = _measure(_graph(), _pericytes(_beside(60)), _segmentation(), step_um=2.0, reach_um=30.0)

    profile = result.profiles[1]
    by_offset = dict(zip(profile.offsets_um.round(6), profile.diameters_um))
    assert profile.diameter_at_pericyte_um == pytest.approx(NARROW_UM, rel=1e-6)
    for offset in (-4.0, -2.0, 2.0, 4.0):
        assert by_offset[offset] == pytest.approx(NARROW_UM, rel=1e-6)
    for offset in (-30.0, -20.0, -12.0, 12.0, 20.0, 30.0):
        assert by_offset[offset] == pytest.approx(WIDE_UM, rel=1e-6)
    assert set(profile.methods) == {"cross_section"}
    assert profile.narrowest() == pytest.approx((NARROW_UM, profile.offsets_um[np.argmin(profile.diameters_um)]))


def test_the_point_reader_reads_as_the_edge_reader_reads():
    """One reader of the mask's width: the edge's own samples, read at the same
    points, give the same widths."""
    segmentation = _segmentation()
    G = _graph()
    summary = measure_edge_diameters_from_binary_mask(
        G, binary_mask=segmentation.astype(bool), voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=10.0, edges=[(0, 1, 0)], min_resolvable_diameter_um=0.0,
    )
    assert summary["edges_measured"] == 1
    samples = np.asarray(G.edges[0, 1, 0]["edt_diameter_samples_um"])
    xs = np.linspace(START_X, END_X, len(samples))
    points = np.column_stack([np.full(len(xs), 10.0), np.full(len(xs), 10.0), xs])
    tangents = np.tile([0.0, 0.0, 1.0], (len(xs), 1))
    ((diameters, methods),) = mask_diameters_at_points(segmentation.astype(bool), (1.0, 1.0, 1.0), [(points, tangents)])
    sectioned = np.asarray([method == "cross_section" for method in methods])
    np.testing.assert_allclose(diameters[sectioned], samples[: len(diameters)][sectioned])


@pytest.mark.parametrize("reversed_voxels", [False, True])
def test_offsets_run_negative_toward_u_whichever_way_the_centreline_was_stored(reversed_voxels):
    result = _measure(_graph(reversed_voxels=reversed_voxels), _pericytes(_beside(40)), _segmentation(),
                      reach_um=10.0)

    (pericyte,) = result.pericytes
    assert pericyte.edge == (0, 1, 0)  # u is node 0, at x = 10
    assert pericyte.arc_position_um == pytest.approx(30.0)
    profile = result.profiles[1]
    toward_u = profile.points_um[np.isclose(profile.offsets_um, -10.0)][0]
    assert toward_u[2] == pytest.approx(30.0)


# --- the nearest junction -----------------------------------------------------------


def test_the_nearest_junction_is_the_nearer_end_that_is_a_junction():
    """Node 0 is a dead end 20 um away; node 1, where three vessels meet, 80 um."""
    result = _measure(_graph(), _pericytes(_beside(30)), _segmentation(), profiles=False)

    junction = result.junctions[1]
    assert junction.node == 1 and junction.end == "v" and junction.vessels == 3
    assert junction.along_vessel_um == pytest.approx(80.0)
    assert junction.straight_line_um == pytest.approx(80.0)


def test_a_vessel_opening_into_a_thick_vessel_has_its_junction_there():
    result = _measure(_graph(junction_at_start=True), _pericytes(_beside(30)), _segmentation(), profiles=False)

    junction = result.junctions[1]
    assert junction.node == 0 and junction.end == "u"
    assert junction.along_vessel_um == pytest.approx(20.0)


def test_a_vessel_with_no_junction_at_either_end_has_no_junction_distance():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([10.0, 10.0, 10.0]))
    G.add_node(1, pos=np.array([10.0, 10.0, 110.0]))
    G.add_edge(0, 1, key=0, voxels=[G.nodes[0]["pos"], G.nodes[1]["pos"]], length=100.0, branch_order="B01")
    placed = PlacedPericyte(
        number=1, centroid_um=(10.0, 10.0, 60.0), equivalent_diameter_um=6.0, edge=(0, 1, 0),
        arc_position_um=50.0, vessel_length_um=100.0, midline_um=(10.0, 10.0, 60.0),
    )
    junction = junction_distances(G, [placed])[1]
    assert junction.node is None and junction.along_vessel_um is None


# --- the branch's resistances -------------------------------------------------------


def _straight_branch(length: float = 100.0, diameter: float = 6.0) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, length]))
    G.add_edge(0, 1, key=0, voxels=[G.nodes[0]["pos"], G.nodes[1]["pos"]], length=length,
               branch_order="B01", diameter_um=diameter, discharge_haematocrit=0.3)
    return G


def _profile(number: int, at: float, reach: float, width: float, step: float = 2.0) -> DiameterProfile:
    offsets = np.arange(-reach, reach + step / 2, step)
    positions = at + offsets
    pericyte = PlacedPericyte(
        number=number, centroid_um=(0.0, 0.0, at), equivalent_diameter_um=6.0, edge=(0, 1, 0),
        arc_position_um=at, vessel_length_um=100.0, midline_um=(0.0, 0.0, at),
    )
    return DiameterProfile(
        pericyte=pericyte, offsets_um=offsets, arc_positions_um=positions,
        points_um=np.column_stack([np.zeros_like(positions), np.zeros_like(positions), positions]),
        diameters_um=np.full(len(offsets), float(width)), methods=("cross_section",) * len(offsets),
    )


CONSTANT = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0, viscosity_law="constant")


def test_the_profile_integral_of_an_even_tube_is_poiseuilles():
    """The one integral the sites' constriction and a measured profile share."""
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    positions = np.linspace(0.0, 80.0, 321)
    assert profile_resistance(positions, np.full(len(positions), 5.0)) == pytest.approx(
        model.resistance_of_uniform_segment(80.0, 5.0), rel=1e-12
    )


def test_a_uniform_branch_keeps_its_poiseuille_resistance():
    G = _straight_branch()
    (branch,) = branch_resistances(G, {1: _profile(1, 50.0, 30.0, 6.0)}, model=CONSTANT).values()

    expected = CONSTANT.resistance_of_uniform_segment(100.0, 6.0, haematocrit=0.3)
    assert branch.uniform_resistance == pytest.approx(expected, rel=1e-12)
    assert branch.constricted_resistance == pytest.approx(expected, rel=1e-9)
    assert branch.ratio == pytest.approx(1.0, rel=1e-9)
    assert branch.haematocrit == pytest.approx(0.3)


def test_two_pericytes_on_one_branch_each_narrow_their_own_stretch():
    """P1 reads 3 um over 10-50 um, P2 6 um over 40-100 um. The median reading
    is 6 um, so P1's stretch is half the branch's 6 um: where both cover a
    point, the nearer pericyte's reading counts -- P1's up to 50 um. With a
    constant viscosity the resistance is (60 + 40 * 2^4) / 100 of uniform."""
    G = _straight_branch()
    profiles = {1: _profile(1, 30.0, 20.0, 3.0), 2: _profile(2, 70.0, 30.0, 6.0)}
    (branch,) = branch_resistances(G, profiles, model=CONSTANT).values()

    assert branch.pericytes == (1, 2)
    assert branch.reference_diameter_um == pytest.approx(6.0)
    assert branch.narrowest_diameter_um == pytest.approx(3.0)
    assert branch.ratio == pytest.approx((60.0 + 40.0 * 2.0**4) / 100.0, rel=1e-2)


def test_a_narrowing_ramp_integrates_as_poiseuille_says():
    """Readings 6 um but 3 um at the midline: the diameter ramps linearly over
    the 2 um each side, whose resistance has a closed form."""
    G = _straight_branch()
    profile = _profile(1, 50.0, 10.0, 6.0)
    widths = profile.diameters_um.copy()
    widths[np.isclose(profile.offsets_um, 0.0)] = 3.0
    profile = DiameterProfile(
        pericyte=profile.pericyte, offsets_um=profile.offsets_um, arc_positions_um=profile.arc_positions_um,
        points_um=profile.points_um, diameters_um=widths, methods=profile.methods,
    )
    (branch,) = branch_resistances(G, {1: profile}, model=CONSTANT).values()

    # Per unit of 128 mu / pi: 96 um at d = 6, and two linear 2 um ramps 3 -> 6.
    def ramp(length, d1, d2):
        return length / (3.0 * (d2 - d1)) * (1.0 / d1**3 - 1.0 / d2**3)

    expected = (96.0 / 6.0**4 + 2.0 * ramp(2.0, 3.0, 6.0)) / (100.0 / 6.0**4)
    assert branch.ratio == pytest.approx(expected, rel=2e-3)


def test_readings_that_are_no_closed_section_do_not_narrow_the_branch():
    """A junction's or a dead end's plane reads the inscribed radius: left out."""
    G = _straight_branch()
    profile = _profile(1, 50.0, 10.0, 6.0)
    widths = profile.diameters_um.copy()
    widths[0] = 2.0
    methods = ("inscribed_radius",) + profile.methods[1:]
    profile = DiameterProfile(
        pericyte=profile.pericyte, offsets_um=profile.offsets_um, arc_positions_um=profile.arc_positions_um,
        points_um=profile.points_um, diameters_um=widths, methods=methods,
    )
    (branch,) = branch_resistances(G, {1: profile}, model=CONSTANT).values()
    assert branch.ratio == pytest.approx(1.0, rel=1e-9)
    assert branch.resolved_readings == len(widths) - 1


def test_the_measured_narrowing_raises_the_branchs_resistance():
    """The tube narrowed to 2/3 over 12 um: about 0.88 + 0.12 * 1.5^4 of uniform,
    and a little more for the readings either side (2 um apart)."""
    result = _measure(
        _graph(), _pericytes(_beside(60)), _segmentation(), step_um=2.0, reach_um=30.0, model=CONSTANT,
    )
    branch = result.branches[(0, 1, 0)]
    narrowing = 0.88 + 0.12 * (WIDE_UM / NARROW_UM) ** 4
    assert narrowing < branch.ratio < narrowing + 0.04 * (WIDE_UM / NARROW_UM) ** 4
    unnarrowed = _measure(
        _graph(), _pericytes(_beside(60)), _segmentation(narrowed=False), step_um=2.0, reach_um=30.0,
        model=CONSTANT,
    ).branches[(0, 1, 0)]
    assert unnarrowed.ratio == pytest.approx(1.0, rel=1e-9)


def test_a_profile_read_mostly_inside_the_narrowing_still_reads_it_as_one():
    """Regression: with a 10 um reach most readings are the narrowing's, and
    their own median took it for the branch's width -- the narrowing came out
    a widening (ratio 0.94). The width is the branch's along its whole length."""
    result = _measure(
        _graph(), _pericytes(_beside(60)), _segmentation(), step_um=2.0, reach_um=10.0, model=CONSTANT,
    )
    branch = result.branches[(0, 1, 0)]
    assert branch.reference_diameter_um == pytest.approx(WIDE_UM, rel=1e-6)
    assert branch.ratio > 0.88 + 0.12 * (WIDE_UM / NARROW_UM) ** 4


# --- the CSVs ---------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def test_the_csvs_hold_a_row_per_pericyte_per_reading_and_per_branch(tmp_path):
    far = (slice(17, 21), slice(15, 19), slice(38, 43))
    result = _measure(_graph(), _pericytes(_beside(60), far), _segmentation(), step_um=2.0, reach_um=10.0)
    written = write_pericyte_measurements(result, tmp_path, "run")

    # Components are numbered in the mask's scan order (z first): the one
    # beside the vessel (z 11-14) is 1, the far one (z 17-20) 2.
    summary = tmp_path / "run_pericytes.csv"
    profile = tmp_path / "run_pericyte_profiles" / "pericyte_0001.csv"
    branches = tmp_path / "run_pericyte_branches.csv"
    assert written == [summary, profile, branches]
    assert _header(summary) == list(PERICYTE_COLUMNS)
    assert _header(profile) == list(PROFILE_COLUMNS)
    assert _header(branches) == list(BRANCH_COLUMNS)

    rows = {row["pericyte"]: row for row in _read(summary)}
    placed, unplaced = rows["1"], rows["2"]
    assert placed["placed"] == "True" and placed["profile_file"] == "pericyte_0001.csv"
    assert float(placed["junction_distance_along_vessel_um"]) == pytest.approx(50.0)
    assert float(placed["branch_resistance_ratio"]) > 1.0
    assert unplaced["placed"] == "False" and unplaced["reason"] == "too far from a capillary"
    assert unplaced["branch_id"] == "" and unplaced["profile_file"] == ""

    readings = _read(profile)
    assert [float(row["offset_um"]) for row in readings] == list(np.arange(-10.0, 11.0, 2.0))
    assert readings[0]["toward_node"] == "0" and readings[-1]["toward_node"] == "1"
    middle = readings[5]
    assert float(middle["diameter_change_um"]) == pytest.approx(0.0)
    assert float(readings[0]["diameter_ratio"]) == pytest.approx(WIDE_UM / NARROW_UM, rel=1e-6)

    (branch,) = _read(branches)
    assert branch["branch_id"] == "0" and branch["pericytes"] == "1"
    assert json.loads(branch["pericyte_numbers"]) == [1]
    assert float(branch["length_um"]) == pytest.approx(100.0)


def test_a_measurement_switched_off_leaves_its_columns_blank(tmp_path):
    result = _measure(_graph(), _pericytes(_beside(60)), _segmentation(), profiles=False)
    write_pericyte_measurements(result, tmp_path, "run")

    (row,) = _read(tmp_path / "run_pericytes.csv")
    assert row["profile_file"] == "" and row["branch_uniform_resistance_pa_s_per_m3"] == ""
    assert float(row["junction_distance_along_vessel_um"]) == pytest.approx(50.0)
    assert not (tmp_path / "run_pericyte_profiles").exists()
    assert not (tmp_path / "run_pericyte_branches.csv").exists()


# --- the export stage and preflight -----------------------------------------------------


def _export_inputs(tmp_path: Path, **overrides):
    mask_path = tmp_path / "pericytes.tif"
    tifffile.imwrite(mask_path, _pericytes(_beside(60)))
    settings = SCHEMA.defaults()
    settings.update(
        input_path=tmp_path / "seg.tif",
        run_haemodynamics=False,
        vtk_export=False,
        visualize_vtk=False,
        visualize_results=False,
        export_citations=False,
        diameter_by_branch_order={"B01": 6.0, "B02": 6.0},
        measure_pericytes=True,
        pericyte_measurement_mask_path=mask_path,
        pericyte_profile_reach_um=20.0,
    )
    settings.update(overrides)
    volume = stages.SkeletonisedVolume(
        image=_segmentation(),
        skeleton=np.zeros(SHAPE, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    G = _graph()
    return settings, stages.VesselNetwork(graph=G, volume=volume), G


def test_the_export_stage_writes_the_pericyte_measurements(tmp_path):
    settings, network, G = _export_inputs(tmp_path)
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), stages.Solution())

    (row,) = _read(tmp_path / "seg_pericytes.csv")
    assert row["placed"] == "True"
    assert float(row["diameter_at_pericyte_um"]) == pytest.approx(NARROW_UM, rel=1e-6)
    assert int(row["profile_readings"]) == 21
    assert float(row["junction_distance_along_vessel_um"]) == pytest.approx(50.0)
    assert float(row["branch_resistance_ratio"]) > 1.0
    assert (tmp_path / "seg_pericyte_profiles" / "pericyte_0001.csv").is_file()
    assert (tmp_path / "seg_pericyte_branches.csv").is_file()


def test_the_export_stage_measures_nothing_while_it_is_off(tmp_path):
    settings, network, G = _export_inputs(tmp_path, measure_pericytes=False)
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), stages.Solution())
    assert not (tmp_path / "seg_pericytes.csv").exists()


def test_the_export_stage_refuses_a_run_with_no_mask(tmp_path):
    settings, network, G = _export_inputs(tmp_path, pericyte_measurement_mask_path=None)
    with pytest.raises(ValueError, match="pericyte_measurement_mask_path"):
        stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), stages.Solution())


def test_preflight_refuses_a_size_range_that_admits_no_pericyte():
    settings = {**SCHEMA.defaults(), "measure_pericytes": True,
                "pericyte_measurement_min_diameter_um": 10.0, "pericyte_measurement_max_diameter_um": 5.0}
    report = check_pericyte_measurements(settings)
    assert any("pericyte_measurement_min_diameter_um" in line for line in report.errors)
    settings["pericyte_measurement_max_diameter_um"] = 12.0
    assert not check_pericyte_measurements(settings).errors


def test_preflight_warns_when_both_measurements_are_off():
    settings = {**SCHEMA.defaults(), "measure_pericytes": True,
                "measure_pericyte_diameter_profiles": False, "measure_pericyte_junction_distances": False}
    assert check_pericyte_measurements(settings).warnings


def test_a_synthetic_network_measures_no_pericytes():
    """It has no image: preflight refuses it (tests/test_synthetic_stage.py),
    and the check here leaves it to that."""
    settings = {**SCHEMA.defaults(), "network_source": "synthetic", "measure_pericytes": True,
                "pericyte_measurement_min_diameter_um": 10.0, "pericyte_measurement_max_diameter_um": 5.0}
    assert not check_pericyte_measurements(settings).errors


def test_the_profile_settings_belong_to_their_own_box():
    names = [setting.name for setting in SCHEMA if setting.section == "Pericyte measurements"]
    assert names[:2] == ["measure_pericytes", "pericyte_measurement_mask_path"]
    assert {"pericyte_profile_step_um", "pericyte_profile_reach_um",
            "measure_pericyte_diameter_profiles", "measure_pericyte_junction_distances"} <= set(names)
    assert SCHEMA["pericyte_profile_step_um"].default == 2.0
    assert SCHEMA["pericyte_profile_reach_um"].default == 50.0
    assert SCHEMA["pericyte_profile_step_um"].unit == "um"


# --- the measured-constriction perturbation -------------------------------------------


def _perturbation_inputs(tmp_path: Path, segmentation: np.ndarray, overrides=None):
    from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, resolve_settings

    mask_path = tmp_path / "pericytes.tif"
    tifffile.imwrite(mask_path, _pericytes(_beside(60)))
    G = _graph()
    for _u, _v, data in G.edges(data=True):
        data["branch_order"] = "B01"
    G, _results = PoiseuilleModel(
        constriction_length=40.0, constriction_spacing=100.0, viscosity_law="constant"
    ).set_poiseuille_resistances(G, {"B01": 6.0})
    entry = {
        "name": "as_measured",
        "type": "pericyte_measured_constriction",
        "overrides": {"use_pericyte_mask_constriction": True, "pericyte_mask_path": str(mask_path), **(overrides or {})},
    }
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        input_path=tmp_path / "seg.tif",
        vtk_output_prefix=tmp_path / "out" / "run",
        plot_dir=tmp_path / "plots",
        run_haemodynamics=True,
        run_perturbations=True,
        perturbations=[entry],
        diameter_by_branch_order={"B01": 6.0},
        measure_diameters_from_raw_image=False,
        viscosity_law="constant",
        pericyte_profile_reach_um=30.0,
    )
    settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3], resistance_node_pair=(0, 2))
    network = stages.VesselNetwork(
        graph=G,
        volume=stages.SkeletonisedVolume(
            image=segmentation, skeleton=np.zeros(SHAPE, dtype=bool), voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0), output_dir=tmp_path,
        ),
    )
    return settings, HaemodynamicModel(graph=G), boundaries, network


def test_the_perturbation_narrows_each_branch_as_measured_and_raises_the_networks_resistance(tmp_path):
    from haemolynx.pipeline import run_perturbations

    settings, model, boundaries, network = _perturbation_inputs(tmp_path, _segmentation())
    baseline = dict(((u, v, k), d["resistance"]) for u, v, k, d in model.graph.edges(keys=True, data=True))
    run = run_perturbations(settings, model, boundaries, SCHEMA, network=network)

    (result,) = run.results
    assert result.error is None, result.error
    (branch,) = _read(result.output_dir / "as_measured_pericyte_branches.csv")
    ratio = float(branch["resistance_ratio"])
    assert ratio > 1.4
    narrowed = result.graph.edges[0, 1, 0]
    assert narrowed["resistance"] == pytest.approx(baseline[(0, 1, 0)] * ratio, rel=1e-9)
    assert narrowed["pericyte_count_assigned"] == 1
    assert narrowed["pericyte_centers_um"] == pytest.approx([50.0])
    for edge in ((1, 2, 0), (1, 3, 0)):
        assert result.graph.edges[edge]["resistance"] == pytest.approx(baseline[edge])
    assert result.summary["equivalent_resistance"] > run.baseline["equivalent_resistance"]
    # The baseline is untouched.
    assert model.graph.edges[0, 1, 0]["resistance"] == pytest.approx(baseline[(0, 1, 0)])


def test_the_perturbation_leaves_an_unnarrowed_network_as_it_was(tmp_path):
    from haemolynx.pipeline import run_perturbations

    settings, model, boundaries, network = _perturbation_inputs(tmp_path, _segmentation(narrowed=False))
    run = run_perturbations(settings, model, boundaries, SCHEMA, network=network)

    (result,) = run.results
    assert result.error is None, result.error
    assert result.summary["equivalent_resistance"] == pytest.approx(run.baseline["equivalent_resistance"], rel=1e-9)


def test_the_perturbation_needs_its_mask(tmp_path):
    from haemolynx.haemodynamics.perturbations import perturbation_problems

    settings, *_ = _perturbation_inputs(tmp_path, _segmentation(), overrides={"use_pericyte_mask_constriction": False})
    assert any("reads no mask" in line for line in perturbation_problems(settings, SCHEMA))


def test_choosing_the_type_in_the_panel_starts_it_reading_a_mask():
    from haemolynx.gui.perturbation_editing import (
        display_name_for_type,
        rows_for_type,
        starting_values_for_type,
    )

    assert starting_values_for_type("pericyte_measured_constriction") == {"use_pericyte_mask_constriction": True}
    assert starting_values_for_type("pericyte_diameter_change") == {}
    assert "pericyte_mask_path" in rows_for_type("pericyte_measured_constriction")
    assert display_name_for_type("pericyte_measured_constriction") == "pericyte constriction as measured"


# --- the batch and the citations -----------------------------------------------------


def test_a_batch_fills_both_pericyte_masks_from_its_folder_of_pericyte_masks(tmp_path):
    from haemolynx.io.batch_matching import BatchFolders, BatchMatch
    from haemolynx.pipeline.batch import item_settings

    mask = tmp_path / "image_pericytes.tif"
    match = BatchMatch(
        segmented=tmp_path / "image.tif", raw=None, folders=BatchFolders(segmented=tmp_path),
        files={"pericyte": mask},
    )
    settings = item_settings(SCHEMA.defaults(), match, tmp_path / "out")
    assert settings["pericyte_mask_path"] == mask
    assert settings["pericyte_measurement_mask_path"] == mask


def test_a_batch_refuses_one_pericyte_measurement_mask_for_every_image(tmp_path):
    from haemolynx.pipeline.batch import check_batch

    mask = tmp_path / "pericytes.tif"
    mask.write_bytes(b"")
    settings = {**SCHEMA.defaults(), "measure_pericytes": True, "pericyte_measurement_mask_path": mask}
    report = check_batch(settings, SCHEMA)
    assert any("pericyte_measurement_mask_path names one file" in line for line in report.errors)


def test_the_measurements_mask_reader_is_cited(tmp_path):
    from haemolynx.pipeline.citations import _image_readers

    settings = {**SCHEMA.defaults(), "measure_pericytes": True,
                "pericyte_measurement_mask_path": tmp_path / "pericytes.h5"}
    assert "h5py" in _image_readers(settings)


# --- the raw image's FWHM as the profile's reading ---------------------------------


def _raw(narrowed: bool = True) -> np.ndarray:
    """The tube as a lumen label would image it: bright inside, dim tissue
    around, blurred by a 1 um PSF."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.where(_segmentation(narrowed) > 0, 1000.0, 100.0), sigma=1.0).astype(np.float32)


def test_reading_fwhm_at_positions_reads_as_the_edges_own_samples():
    """One FWHM reader: an edge read at the arc lengths its own regular
    samples fall at gives those samples' widths, exactly."""
    from haemolynx.haemodynamics.fwhm_diameter import measure_edge_diameters_fwhm_from_raw_tiff

    arguments = dict(
        raw_volume=_raw(), voxel_size_zyx=(1.0, 1.0, 1.0), sample_spacing_along_edge_um=10.0,
        transverse_profile_step_um=0.25, transverse_half_extent_um=8.0, diameter_guess_um=6.0,
        diameter_guess_edge_attribute=None, edges=[(0, 1, 0)],
    )
    regular = _graph()
    measure_edge_diameters_fwhm_from_raw_tiff(regular, **arguments)
    samples = regular.edges[0, 1, 0]["fwhm_diameter_samples_um"]
    assert len(samples) >= 8

    at = _graph()
    summary = measure_edge_diameters_fwhm_from_raw_tiff(
        at, **arguments, at_positions={(0, 1, 0): np.linspace(0.0, 100.0, 11)}
    )
    readings = summary["readings_at_positions"][(0, 1, 0)]
    assert len(readings) == 11
    assert [r for r in readings if np.isfinite(r)] == samples


def test_reading_fwhm_at_positions_keys_the_edge_as_asked():
    from haemolynx.haemodynamics.fwhm_diameter import measure_edge_diameters_fwhm_from_raw_tiff

    summary = measure_edge_diameters_fwhm_from_raw_tiff(
        _graph(), raw_volume=_raw(), voxel_size_zyx=(1.0, 1.0, 1.0), sample_spacing_along_edge_um=10.0,
        transverse_profile_step_um=0.25, transverse_half_extent_um=8.0, edges=[(1, 0, 0)],
        at_positions={(1, 0, 0): [50.0]},
    )
    assert list(summary["readings_at_positions"]) == [(1, 0, 0)]


def test_the_export_stage_reads_the_profile_off_the_raw_image(tmp_path):
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, _raw())
    settings, network, G = _export_inputs(
        tmp_path, raw_image_path=raw_path, raw_image_stain="plasma_column",
        pericyte_profile_reading="raw_image_fwhm", pericyte_profile_reach_um=30.0,
    )
    stages.export_results(settings, network, stages.HaemodynamicModel(graph=G), stages.Solution())

    readings = _read(tmp_path / "seg_pericyte_profiles" / "pericyte_0001.csv")
    methods = {row["diameter_method"] for row in readings}
    assert methods <= {"fwhm", ""} and "fwhm" in methods
    width = {float(row["offset_um"]): number(row["diameter_um"]) for row in readings}
    far = [width[o] for o in (-30.0, -20.0, 20.0, 30.0) if np.isfinite(width[o])]
    near = [width[o] for o in (-2.0, 0.0, 2.0) if np.isfinite(width[o])]
    assert far and near
    assert np.median(far) == pytest.approx(WIDE_UM, abs=1.5)
    assert np.median(near) < np.median(far) - 1.0
    (branch,) = _read(tmp_path / "seg_pericyte_branches.csv")
    assert float(branch["reference_diameter_um"]) == pytest.approx(np.median(far), abs=1.0)
    assert float(branch["resistance_ratio"]) > 1.2


def test_the_raw_image_reading_needs_a_raw_image():
    settings = {**SCHEMA.defaults(), "measure_pericytes": True, "pericyte_profile_reading": "raw_image_fwhm"}
    assert any("raw_image_path is not set" in line for line in check_pericyte_measurements(settings).errors)
    settings["raw_image_path"] = "raw.tif"
    assert not check_pericyte_measurements(settings).errors
    settings["raw_image_stain"] = "endothelial"
    assert any("wall" in line for line in check_pericyte_measurements(settings).warnings)


def test_the_reading_choices_are_the_modules():
    from haemolynx.haemodynamics.pericyte_measurements import PROFILE_READINGS

    assert tuple(SCHEMA["pericyte_profile_reading"].choices) == PROFILE_READINGS
    assert SCHEMA["pericyte_profile_reading"].default == "segmentation"


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def test_the_perturbation_reads_the_raw_image_when_the_profiles_do(tmp_path):
    from haemolynx.pipeline import run_perturbations

    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, _raw())
    settings, model, boundaries, network = _perturbation_inputs(tmp_path, _segmentation())
    settings.update(raw_image_path=raw_path, raw_image_stain="plasma_column", pericyte_profile_reading="raw_image_fwhm")
    run = run_perturbations(settings, model, boundaries, SCHEMA, network=network)

    (result,) = run.results
    assert result.error is None, result.error
    (branch,) = _read(result.output_dir / "as_measured_pericyte_branches.csv")
    assert float(branch["resistance_ratio"]) > 1.2
    assert result.summary["equivalent_resistance"] > run.baseline["equivalent_resistance"]
