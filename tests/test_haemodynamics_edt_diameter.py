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
    """A raw EDT reading right at a branch point returns the junction's own
    larger inscribed sphere, not the branch's true radius -- a real,
    documented bias for a short branch off a wide hub. Confirmed
    numerically first (not asserted blind): without exclusion the short
    branch below reads ~11.0 (heavily biased by the radius-10 hub);
    with an 8 um exclusion zone it reads exactly 4.0, the branch's true
    diameter (radius 2)."""
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
        sample_spacing_along_edge_um=2.0, branch_endpoint_exclusion_um=8.0,
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
