"""Tests for 3D vessel length measurements on synthetic data."""
import math

import numpy as np
import pytest

from ImageLynx.graph import build_graph_segment_skan_stitched_loops
from ImageLynx.graph._helpers import calculate_path_length


def _build_synthetic_vascular_bed() -> np.ndarray:
    """Create a branched 3D skeleton with analytically known branch lengths."""
    skel = np.zeros((24, 24, 24), dtype=bool)
    center = (10, 10, 10)
    skel[center] = True

    # Axis-aligned branches.
    for x in range(11, 14):
        skel[x, 10, 10] = True
    for y in range(9, 5, -1):
        skel[10, y, 10] = True
    for z in range(11, 16):
        skel[10, 10, z] = True

    # Diagonal branches in 3D.
    for step in range(1, 4):
        skel[10 + step, 10 + step, 10] = True
    for step in range(1, 3):
        skel[10 + step, 10 + step, 10 + step] = True

    return skel


def test_synthetic_3d_branch_lengths_are_measured_correctly():
    skan = pytest.importorskip("skan")
    skel = _build_synthetic_vascular_bed()
    voxel_size = (2.0, 1.5, 1.0)

    sk = skan.csr.Skeleton(skel)
    graph, _, _ = build_graph_segment_skan_stitched_loops(
        sk,
        skel,
        voxel_size=voxel_size,
    )

    expected_total_length = sum(
        [
            3 * voxel_size[0],  # +x arm
            4 * voxel_size[1],  # -y arm
            5 * voxel_size[2],  # +z arm
            3 * math.hypot(voxel_size[0], voxel_size[1]),  # xy diagonal
            2
            * math.sqrt(
                voxel_size[0] ** 2 + voxel_size[1] ** 2 + voxel_size[2] ** 2
            ),  # xyz diagonal
        ]
    )

    measured_lengths = sorted(
        float(data["length"]) for _, _, _, data in graph.edges(keys=True, data=True)
    )
    # Also verify consistency between stored length and explicit voxel-path length.
    for _, _, _, data in graph.edges(keys=True, data=True):
        assert float(data["length"]) == pytest.approx(
            calculate_path_length(data["voxels"]),
            rel=1e-9,
            abs=1e-9,
        )

    # Segmentation/stitching may split expected branches into extra edge segments,
    # but total traversed path length should remain close to the analytic ground truth.
    total_measured_length = sum(measured_lengths)
    assert 0.85 * expected_total_length <= total_measured_length <= 1.05 * expected_total_length
