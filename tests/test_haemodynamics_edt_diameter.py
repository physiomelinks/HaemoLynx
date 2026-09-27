"""Per-edge diameter from the segmentation mask's own EDT inscribed radius.

Mirrors ``tests/test_haemodynamics_automated_fwhm.py``'s structure. Unlike
the FWHM path, nothing here touches ``scipy.optimize.curve_fit`` or any
other LAPACK-backed call -- ``distance_transform_edt`` is a pure geometric
algorithm -- so every test in this file is a real, fully-executed check.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics.edt_diameter import measure_edge_diameters_from_binary_mask


def _cylinder_mask(shape, *, z, y, radius, x0, x1):
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


def _straight_edge_graph(x0, x1, *, z=10.0, y=10.0) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([z, y, float(x0)]))
    graph.add_node(1, pos=np.array([z, y, float(x1)]))
    voxels = [(z, y, float(x)) for x in range(int(x0), int(x1) + 1)]
    graph.add_edge(0, 1, key=0, length=float(x1 - x0), voxels=voxels)
    return graph


def test_measure_edge_diameters_from_binary_mask_straight_cylinder():
    shape = (20, 20, 30)
    mask = _cylinder_mask(shape, z=10, y=10, radius=4.0, x0=2, x1=27)
    graph = _straight_edge_graph(2, 27)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    assert summary["edges_measured"] == 1
    assert not summary["edges_skipped"]
    measured = graph[0][1][0]["edt_diameter_um"]
    # EDT reads the discretised mask's own boundary, not the analytic
    # radius=4.0 threshold exactly -- a small margin absorbs that rounding.
    assert abs(measured - 8.0) < 0.5


def test_measure_edge_diameters_from_binary_mask_reads_physical_voxel_size():
    """The same index-space mask reads a different physical diameter once
    voxel_size_zyx changes -- proving this reads physical units via
    ``physical_points_to_continuous_indices``, not raw voxel counts
    (matching the analogous FWHM convention test). The graph's own node/
    voxel coordinates are physical microns (this module's documented
    convention), so a coarser z spacing needs the *same* voxel-index
    position expressed as a *larger* physical z value."""
    shape = (10, 10, 30)
    mask = _cylinder_mask(shape, z=5, y=5, radius=3.0, x0=2, x1=27)

    def _graph_for(voxel_size_z: float) -> nx.MultiGraph:
        z_phys = 5.0 * voxel_size_z  # physical position of voxel-index z=5
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([z_phys, 5.0, 2.0]))
        graph.add_node(1, pos=np.array([z_phys, 5.0, 27.0]))
        voxels = [(z_phys, 5.0, float(x)) for x in range(2, 28)]
        graph.add_edge(0, 1, key=0, length=25.0, voxels=voxels)
        return graph

    graph_fine = _graph_for(1.0)
    measure_edge_diameters_from_binary_mask(
        graph_fine, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )
    graph_coarse = _graph_for(2.0)
    measure_edge_diameters_from_binary_mask(
        graph_coarse, binary_mask=mask, voxel_size_zyx=(2.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )
    fine = graph_fine[0][1][0]["edt_diameter_um"]
    coarse = graph_coarse[0][1][0]["edt_diameter_um"]
    assert coarse > fine  # coarser z spacing makes the z-direction escape route physically longer


def test_edt_diameter_junction_exclusion_removes_inflated_estimate():
    """A reading right at a branch point returns the junction's own larger
    body, not the branch's true radius -- a real, documented bias for a short
    branch off a wide hub. Without exclusion the short branch below reads
    far wider than its true 4.0 (dominated by the radius-10 hub); with the
    exclusion reaching past the hub's own radius (10 um, the default) it
    reads 4.0.

    It used to exclude only 8 um -- still inside the hub -- and got 4.0 by
    coincidence: the inscribed radius read 6.0 at the sample in the hub and
    2.0 at the branch's flat end (where an inscribed sphere is halved), and
    their median was 4.0. The cross-section reads that end at 4.0."""
    shape = (30, 30, 40)
    zz, yy, xx = np.indices(shape, dtype=float)
    hub = (zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 10.0**2
    branch_radial = np.sqrt((zz - 15) ** 2 + (yy - 15) ** 2)
    branch = (branch_radial <= 2.0) & (xx >= 15) & (xx <= 25)
    mask = hub | branch

    def _build_graph() -> nx.MultiGraph:
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([15.0, 15.0, 15.0]))
        graph.add_node(1, pos=np.array([15.0, 15.0, 25.0]))
        graph.add_edge(
            0, 1, key=0, length=10.0,
            voxels=[(15.0, 15.0, float(x)) for x in range(15, 26)],
        )
        # A second edge on node 0 so it is a real bifurcation (degree > 1).
        graph.add_node(2, pos=np.array([15.0, 15.0, 0.0]))
        graph.add_edge(
            0, 2, key=0, length=15.0,
            voxels=[(15.0, 15.0, float(x)) for x in range(0, 16)],
        )
        return graph

    unexcluded = _build_graph()
    measure_edge_diameters_from_binary_mask(
        unexcluded, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, branch_endpoint_exclusion_um=0.0,
    )
    excluded = _build_graph()
    measure_edge_diameters_from_binary_mask(
        excluded, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, branch_endpoint_exclusion_um=10.0,
    )

    inflated = unexcluded[0][1][0]["edt_diameter_um"]
    corrected = excluded[0][1][0]["edt_diameter_um"]
    assert inflated > 8.0  # dominated by the hub's own radius-10 sphere
    assert corrected == pytest.approx(4.0, abs=0.1)
    assert corrected < inflated


def test_measure_edge_diameters_skips_an_edge_with_no_voxels():
    graph = nx.MultiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, key=0, length=5.0)  # no `voxels` attribute at all
    mask = np.ones((5, 5, 5), dtype=bool)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=1.0,
    )

    assert summary["edges_measured"] == 0
    assert summary["edges_skipped"] == [(0, 1, 0, "no_voxels")]
    assert "edt_diameter_um" not in graph[0][1][0]


def test_measure_edge_diameters_skips_a_zero_length_edge():
    graph = nx.MultiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(
        0, 1, key=0, length=0.0, voxels=[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
    )
    mask = np.ones((5, 5, 5), dtype=bool)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=1.0,
    )

    assert summary["edges_measured"] == 0
    assert summary["edges_skipped"] == [(0, 1, 0, "zero_length")]


def test_measure_edge_diameters_skips_an_edge_entirely_outside_the_mask():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)  # empty mask
    graph = _straight_edge_graph(1, 8, z=5.0, y=5.0)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=1.0,
    )

    assert summary["edges_measured"] == 0
    assert summary["edges_skipped"][0][3] == "edt_failed"


def test_measure_edge_diameters_uses_the_same_aggregation_helper_as_fwhm():
    """`aggregation="mean"` must reach the actual computation, not just
    exist on the signature -- pins the shared aggregation policy with
    `automated._aggregate_edge_diameter`."""
    shape = (30, 30, 40)
    zz, yy, xx = np.indices(shape, dtype=float)
    hub = (zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 10.0**2
    branch_radial = np.sqrt((zz - 15) ** 2 + (yy - 15) ** 2)
    branch = (branch_radial <= 2.0) & (xx >= 15) & (xx <= 25)
    mask = hub | branch

    def _build_graph() -> nx.MultiGraph:
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([15.0, 15.0, 15.0]))
        graph.add_node(1, pos=np.array([15.0, 15.0, 25.0]))
        graph.add_edge(
            0, 1, key=0, length=10.0,
            voxels=[(15.0, 15.0, float(x)) for x in range(15, 26)],
        )
        graph.add_node(2, pos=np.array([15.0, 15.0, 0.0]))
        graph.add_edge(
            0, 2, key=0, length=15.0,
            voxels=[(15.0, 15.0, float(x)) for x in range(0, 16)],
        )
        return graph

    g_median = _build_graph()
    measure_edge_diameters_from_binary_mask(
        g_median, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, aggregation="median",
    )
    g_mean = _build_graph()
    measure_edge_diameters_from_binary_mask(
        g_mean, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, aggregation="mean",
    )
    assert g_median[0][1][0]["edt_diameter_um"] != pytest.approx(
        g_mean[0][1][0]["edt_diameter_um"]
    )


# --- every vessel gets a width -------------------------------------------------------


def _short_edge_between_junctions(length):
    """An edge of *length* um whose both ends are bifurcations (degree 3)."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([10.0, 10.0, 10.0]))
    graph.add_node(1, pos=np.array([10.0, 10.0, 10.0 + length]))
    graph.add_edge(
        0, 1, key=0, length=float(length),
        voxels=[(10.0, 10.0, 10.0 + x) for x in np.linspace(0.0, length, int(length) + 1)],
    )
    for node, x in ((0, 10.0), (1, 10.0 + length)):
        for k, dz in enumerate((-3.0, 3.0)):
            other = f"{node}-{k}"
            graph.add_node(other, pos=np.array([10.0 + dz, 10.0, x]))
            graph.add_edge(node, other, key=0, length=3.0,
                           voxels=[(10.0, 10.0, x), (10.0 + dz, 10.0, x)])
    return graph


def test_an_edge_shorter_than_its_junction_zones_is_read_across_its_middle():
    """Regression: a 3um capillary between two junctions, sampled only at its
    two ends, lost both to the 10um zones and got no width at all -- 238 of
    3191 edges on a real run, left to the branch-order table. It is read
    across its middle half, several times, not once at its midpoint."""
    mask = _cylinder_mask((20, 20, 30), z=10, y=10, radius=3.0, x0=0, x1=29)
    graph = _short_edge_between_junctions(3.0)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, branch_endpoint_exclusion_um=10.0,
    )

    data = graph[0][1][0]
    assert (0, 1, 0, "excluded_near_branch") not in summary["edges_skipped"]
    assert data["edt_short_edge_readings"] == 4  # 0.75..2.25 um, 0.5 um apart
    assert len(data["edt_diameter_samples_um"]) == 4
    assert data["edt_diameter_um"] == pytest.approx(6.0, abs=0.6)


def test_one_bad_spot_on_a_short_edge_is_outvoted():
    """The sub-2 um vessels of a real network were mostly short edges between
    junctions read once, at the midpoint; where that one reading hit a notch
    in the mask, a 4-5 um vessel was read under 2 um. Read at five points, the
    notch is one reading of five and the median is the vessel's own."""
    mask = _cylinder_mask((20, 20, 30), z=10, y=10, radius=3.0, x0=0, x1=29)
    mask[:, 10:, 13] = False  # over half the section gone, one voxel long, at the midpoint
    graph = _short_edge_between_junctions(6.0)  # x = 10..16, midpoint 13

    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, branch_endpoint_exclusion_um=10.0,
    )

    data = graph[0][1][0]
    readings = data["edt_diameter_samples_um"]
    assert data["edt_short_edge_readings"] == 5
    assert min(readings) < 0.8 * 6.0  # the notch, which alone used to be the answer
    assert data["edt_diameter_um"] == pytest.approx(6.0, abs=0.6)


@pytest.mark.parametrize(
    ("start_is_branch", "end_is_branch", "window"),
    [(True, True, (2.5, 7.5)), (True, False, (5.0, 10.0)), (False, True, (0.0, 5.0))],
)
def test_an_edge_its_junction_zones_cover_is_read_on_the_half_farthest_from_them(
    start_is_branch, end_is_branch, window
):
    from haemolynx.haemodynamics.edt_diameter import _short_edge_targets

    targets = _short_edge_targets(
        10.0,
        start=12.0 if start_is_branch else 0.0,  # a 12 um zone on a 10 um edge
        end=-2.0 if end_is_branch else 10.0,
        start_is_branch=start_is_branch,
        end_is_branch=end_is_branch,
    )

    assert targets == pytest.approx(np.linspace(*window, 5))


def test_what_a_junction_zone_leaves_of_a_short_edge_is_all_that_is_read():
    """A short branch off a wide hub: its middle half is inside the hub, so
    what the 10 um zone leaves -- however little -- is read, not the middle."""
    from haemolynx.haemodynamics.edt_diameter import _short_edge_targets

    assert _short_edge_targets(
        10.4, start=10.0, end=10.4, start_is_branch=True, end_is_branch=False
    ) == pytest.approx([10.2])
    assert _short_edge_targets(
        12.0, start=10.0, end=12.0, start_is_branch=True, end_is_branch=False
    ) == pytest.approx([10.0, 10.5, 11.0, 11.5, 12.0])


def test_an_off_mask_centreline_climbs_across_its_vessel_not_along_it():
    """A centreline just outside a thin branch, near where the branch leaves a
    wide hub: climbing along the branch reaches the hub's body, so the search
    only climbs across it."""
    from haemolynx.haemodynamics.edt_diameter import _nearby_radius
    from haemolynx.preprocessing.thick_vessels import inscribed_radius_map
    from scipy.ndimage import map_coordinates

    shape = (30, 30, 40)
    zz, yy, xx = np.indices(shape, dtype=float)
    hub = (zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 12) ** 2 <= 8.0**2
    branch = (np.hypot(zz - 15, yy - 15) <= 2.0) & (xx >= 12) & (xx <= 35)
    radius_map = inscribed_radius_map(hub | branch, (1.0, 1.0, 1.0))

    def sample(idx):
        return map_coordinates(radius_map, np.asarray(idx).T, order=1)

    start = np.array([[15.0, 18.0, 20.0]])  # 1 um outside the branch, at the hub's edge
    assert sample(start)[0] == 0.0

    radius = _nearby_radius(sample, start, (1.0, 1.0, 1.0), 3.0, np.array([[0.0, 0.0, 1.0]]))
    wrong_way = _nearby_radius(sample, start, (1.0, 1.0, 1.0), 3.0, np.array([[0.0, 1.0, 0.0]]))

    assert 2.0 * radius[0] == pytest.approx(4.0, abs=1.0)
    assert wrong_way[0] > radius[0] + 2.0  # let loose along the branch, it reaches the hub


# --- widths the mask cannot give ------------------------------------------------
#
# A strand of mask a voxel thick is a vessel the segmentation caught a sliver
# of: its reading (~1.4 um) is the voxel's, not the vessel's, and as a d^4
# resistance it is hundreds of times its neighbours'.


def _strand(shape=(20, 20, 30), *, x0=2, x1=27):
    mask = np.zeros(shape, dtype=bool)
    mask[10, 10, x0:x1 + 1] = True
    return mask


def test_a_one_voxel_strand_is_left_for_the_table():
    graph = _straight_edge_graph(4, 25)

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=_strand(), voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    data = graph[0][1][0]
    assert summary["edges_skipped"] == [(0, 1, 0, "below_resolvable_diameter")]
    assert "edt_diameter_um" not in data
    assert data["edt_unresolved_diameter_um"] == pytest.approx(1.4, abs=0.2)


def test_an_unresolved_edge_takes_the_branch_order_table_diameter():
    """The EDT fallback skips it, so the chain goes on to the table."""
    from haemolynx.haemodynamics.poiseuille import (
        DIAMETER_SOURCE_TABLE,
        stamp_edge_diameters,
    )

    graph = _straight_edge_graph(4, 25)
    graph[0][1][0]["branch_order"] = "B01"
    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=_strand(), voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    stamp_edge_diameters(graph, {"B01": 5.0}, use_edt_fallback=True)

    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_TABLE
    assert graph[0][1][0]["diameter_um"] == pytest.approx(5.0)


def test_a_zero_floor_keeps_every_reading():
    graph = _straight_edge_graph(4, 25)

    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=_strand(), voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0, min_resolvable_diameter_um=0.0,
    )

    assert graph[0][1][0]["edt_diameter_um"] == pytest.approx(1.4, abs=0.2)


def test_readings_on_a_strand_are_left_out_of_a_measured_vessel_median():
    """Four readings on a 6 um vessel, eight on the strand it narrows to: the
    median of all twelve is the strand's; of the resolvable ones, the vessel's."""
    mask = _cylinder_mask((20, 20, 30), z=10, y=10, radius=3.0, x0=0, x1=10)
    mask |= _strand(x0=11, x1=29)
    graph = _straight_edge_graph(4, 26)  # readings at x = 4, 6, ..., 26

    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    data = graph[0][1][0]
    assert data["edt_unresolved_samples"] == 8
    assert len(data["edt_diameter_samples_um"]) == 4
    assert data["edt_diameter_um"] == pytest.approx(6.0, abs=0.6)


def test_remeasuring_an_edge_clears_what_the_last_measurement_wrote():
    """An edge measured once with no floor, then again with one, must not keep
    the first run's width when the second leaves it unmeasured."""
    graph = _straight_edge_graph(4, 25)
    for floor in (0.0, 1.5):
        measure_edge_diameters_from_binary_mask(
            graph, binary_mask=_strand(), voxel_size_zyx=(1.0, 1.0, 1.0),
            sample_spacing_along_edge_um=2.0, min_resolvable_diameter_um=floor,
        )

    data = graph[0][1][0]
    assert "edt_diameter_um" not in data
    assert "edt_diameter_samples_um" not in data
    assert "edt_diameter_method" not in data


def test_a_negative_floor_is_rejected():
    with pytest.raises(ValueError, match="min_resolvable_diameter_um"):
        measure_edge_diameters_from_binary_mask(
            _straight_edge_graph(4, 25), binary_mask=_strand(), voxel_size_zyx=(1.0, 1.0, 1.0),
            sample_spacing_along_edge_um=2.0, min_resolvable_diameter_um=-1.0,
        )


def test_a_centreline_just_off_its_mask_finds_its_own_vessel():
    """A smoothed or reconnected centreline can run a voxel or two outside its
    segmentation, where the inscribed radius is 0: it looks round for the
    vessel instead of reporting no width."""
    mask = _cylinder_mask((20, 20, 30), z=10, y=10, radius=3.0, x0=0, x1=29)
    graph = _straight_edge_graph(4, 25, z=10.0, y=14.0)  # 1um outside the r=3 wall

    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    data = graph[0][1][0]
    assert data["edt_off_centreline_samples"] > 0
    assert data["edt_diameter_um"] == pytest.approx(6.0, abs=0.6)


def test_a_centreline_far_from_any_vessel_still_gets_no_width():
    """The search is local: a centreline 6um from the nearest vessel is not
    that vessel, and must not borrow its width."""
    mask = _cylinder_mask((30, 30, 30), z=10, y=10, radius=3.0, x0=0, x1=29)
    graph = _straight_edge_graph(4, 25, z=10.0, y=19.0)  # 6um outside the wall

    summary = measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=2.0,
    )

    assert summary["edges_skipped"][0][3] == "edt_failed"


# --- the cross-section method -------------------------------------------------
#
# The inscribed radius read an axis-aligned 2-3 um vessel 30-40% wide and an
# oblique, flattened or off-centre one up to ~30% narrow -- a d^4 error of
# 0.3x to 3.7x in resistance. The section's own area does not care which way
# the vessel runs, where in its voxel it sits, or whether it is round.

_VOXEL = (1.0, 0.98, 0.98)  # z, y, x: the real run's


def _tube(diameter, direction, *, ellipse=1.0, shift=(0.0, 0.0, 0.0), offset_um=0.0, shape=(56, 56, 56)):
    """A straight vessel of *diameter* through the volume, and its centreline;
    *ellipse* flattens it keeping the area, *offset_um* moves the centreline
    sideways off the axis."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    points = np.stack(np.indices(shape), axis=-1) * np.asarray(_VOXEL)
    centre = np.asarray(shape) * np.asarray(_VOXEL) / 2 + np.asarray(shift)
    rel = points - centre
    along = rel @ direction
    across = rel - along[..., None] * direction
    reference = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    a = diameter / 2 * np.sqrt(ellipse)
    b = diameter / 2 / np.sqrt(ellipse)
    mask = ((across @ first / a) ** 2 + (across @ second / b) ** 2 <= 1.0) & (np.abs(along) < 22)
    line = centre + np.linspace(-18, 18, 37)[:, None] * direction + offset_um * first
    return mask, line


def _edt_diameter(mask, line, method="cross_section"):
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, voxels=[tuple(point) for point in line])
    measure_edge_diameters_from_binary_mask(
        graph, binary_mask=mask, voxel_size_zyx=_VOXEL, sample_spacing_along_edge_um=2.0,
        method=method,
    )
    return graph.edges[0, 1, 0]


@pytest.mark.parametrize("direction", [(1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (2, 1, 0.5)])
@pytest.mark.parametrize("diameter", [3.0, 6.0])
def test_the_cross_section_reads_a_vessel_of_known_size_at_any_orientation(direction, diameter):
    """Within what the mask itself allows: a 3 um vessel centred on a voxel
    is nine voxels, 22% more area than the circle -- the section reads the
    mask's area, which averages true only over where the vessel sits."""
    tolerance = 0.12 if diameter < 4 else 0.06
    for shift in ((0.0, 0.0, 0.0), (0.3, -0.2, 0.41)):
        edge = _edt_diameter(*_tube(diameter, direction, shift=shift))
        assert edge["edt_diameter_method"] == "cross_section"
        assert edge["edt_diameter_um"] == pytest.approx(diameter, rel=tolerance)


def test_the_cross_section_reads_a_flattened_or_off_centre_vessel_the_inscribed_radius_does_not():
    """A 2:1 section's inscribed circle is its narrow width; a centreline half
    a voxel off the axis reads the radius short. The area is neither."""
    for case in (dict(ellipse=2.0), dict(offset_um=0.5)):
        mask, line = _tube(6.0, (1, 1, 1), **case)
        assert _edt_diameter(mask, line)["edt_diameter_um"] == pytest.approx(6.0, rel=0.06)
        assert _edt_diameter(mask, line, "inscribed_radius")["edt_diameter_um"] < 0.9 * 6.0


def test_touching_vessels_are_measured_one_at_a_time():
    """Two vessels whose sections touch at a neck: the whole joined piece
    read 40% wide; split at the neck, the centreline's own lobe reads true."""
    first, line = _tube(6.0, (1, 0, 0))
    second, _ = _tube(6.0, (1, 0, 0), shift=(0.0, 0.0, 0.95 * 6.0))
    assert _edt_diameter(first | second, line)["edt_diameter_um"] == pytest.approx(6.0, rel=0.05)


def test_an_edge_with_no_closed_section_falls_back_to_its_inscribed_radius():
    """A sheet one voxel thick runs out of every plane across it: no section
    closes, so the edge keeps the inscribed-radius width, and says so."""
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[20, :, :] = True
    line = [(20.0 * 1.0, 10.0 * 0.98, x * 0.98) for x in range(5, 35)]

    edge = _edt_diameter(mask, np.asarray(line))

    assert edge["edt_diameter_method"] == "inscribed_radius"
    assert edge["edt_diameter_um"] > 0


def test_an_unknown_method_is_refused():
    mask, line = _tube(4.0, (1, 0, 0))
    with pytest.raises(ValueError, match="cross_section"):
        _edt_diameter(mask, line, method="thickest")
