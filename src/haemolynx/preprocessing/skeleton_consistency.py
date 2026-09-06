"""How well a skeleton represents the mask it was extracted from.

A companion diagnostic to graph.diagnostics' skeleton-vs-graph consistency
check, which looks one stage later, at the graph against the skeleton it was
built from -- this one looks at the skeleton against the binary mask it came
from, before any graph topology exists to compare it against. Both exist to
surface skeletonisation/topology-repair quality loss in the GUI log instead
of it only showing up later as an unexplained missing vessel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .thick_vessels import inscribed_radius_map


def diagnose_skeleton_mask_consistency(
    skeleton: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Fraction of *mask* the skeleton actually runs through.

    A mask voxel counts as "explained" when the nearest skeleton voxel is no
    farther from it than that point's own inscribed radius -- the local
    vessel half-thickness, from the same EDT
    :func:`thick_vessels.inscribed_radius_map` uses for thickness gating. A
    skeleton passing further than a vessel's own half-width from part of its
    own interior is not running through that part of the mask. A whole
    region skeletonisation dropped, an off-centre ridge, or a vessel
    binarised too loosely for its centreline to be recovered cleanly would
    all show up here as a low fraction -- before that loss reaches the
    graph at all.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        return {
            "mask_voxel_count": 0,
            "explained_voxel_count": 0,
            "coverage_fraction": 1.0,
        }

    spacing = tuple(float(v) for v in voxel_size_zyx)
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    local_radius = inscribed_radius_map(mask_bool, spacing)
    if skeleton_bool.any():
        distance_to_skeleton = distance_transform_edt(~skeleton_bool, sampling=spacing)
    else:
        distance_to_skeleton = np.full(mask_bool.shape, np.inf)

    explained = mask_bool & (distance_to_skeleton <= local_radius)
    explained_voxel_count = int(explained.sum())
    return {
        "mask_voxel_count": mask_voxel_count,
        "explained_voxel_count": explained_voxel_count,
        "coverage_fraction": explained_voxel_count / mask_voxel_count,
    }


def format_skeleton_mask_consistency_report(report: dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_skeleton_mask_consistency`."""
    return (
        "Skeleton/mask consistency: "
        f"{report.get('explained_voxel_count', 0)} of "
        f"{report.get('mask_voxel_count', 0)} segmented-image voxels are "
        f"within their own local radius of the skeleton "
        f"({report.get('coverage_fraction', 1.0):.1%})."
    )
