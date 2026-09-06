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


def _explained_by_local_radius(
    source_bool: np.ndarray,
    mask_bool: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Which *mask_bool* voxels count as "explained" by *source_bool*.

    A mask voxel is explained when the nearest True voxel of *source_bool*
    is no farther from it than that point's own local inscribed radius
    (:func:`thick_vessels.inscribed_radius_map`), plus one voxel diagonal's
    worth of grid-discretisation slack -- see
    :func:`diagnose_skeleton_mask_consistency`'s own docstring for the
    empirical justification of that margin.

    Shared by :func:`diagnose_skeleton_mask_consistency` (*source_bool* is
    the skeleton) and ``graph.diagnostics.diagnose_graph_mask_consistency``
    (*source_bool* is the graph's own rasterised edges) -- both compare
    against the same kind of segmented-image mask with the identical
    local-radius-plus-margin geometry, so a fix to that geometry (like the
    margin itself) only has to be made once.
    """
    spacing = tuple(float(v) for v in voxel_size_zyx)
    local_radius = inscribed_radius_map(mask_bool, spacing)
    discretisation_margin = float(np.linalg.norm(spacing))
    if source_bool.any():
        distance_to_source = distance_transform_edt(~source_bool, sampling=spacing)
    else:
        distance_to_source = np.full(mask_bool.shape, np.inf)
    return mask_bool & (distance_to_source <= local_radius + discretisation_margin)


def diagnose_skeleton_mask_consistency(
    skeleton: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Fraction of *mask* the skeleton actually runs through.

    A mask voxel counts as "explained" when the nearest skeleton voxel is no
    farther from it than that point's own inscribed radius, plus one voxel
    diagonal's worth of grid-discretisation slack (see below) -- the local
    vessel half-thickness, from the same EDT
    :func:`thick_vessels.inscribed_radius_map` uses for thickness gating. A
    skeleton passing further than that from part of a vessel's own interior
    is not running through that part of the mask. A whole region
    skeletonisation dropped, an off-centre ridge, or a vessel binarised too
    loosely for its centreline to be recovered cleanly would all show up
    here as a low fraction -- before that loss reaches the graph at all.

    The one-voxel-diagonal margin is not slack added to hide a real problem;
    it corrects a genuine miscalibration this module's own tests caught on a
    real fixture. Microvascular capillaries are frequently only 1-2 voxels
    wide, where the inscribed radius is itself close to its theoretical
    minimum of one voxel -- and a discretised medial axis running along one
    face of, say, a 2x2-voxel cross-section is, correctly, up to a full
    voxel diagonal from the section's opposite corner, even though there is
    no single-voxel-wide line that could do better. Comparing against the
    bare inscribed radius on such a fixture read a skeleton that visibly
    tracked the mask well as only 46% "explained"; adding one voxel's worth
    of margin (accounting for a half-voxel of placement slack on each of the
    skeleton's own discrete position and the mask's own EDT-based radius
    estimate) raised that to 93-95%, in line with what the skeleton actually
    looks like. Wider vessels are unaffected in relative terms, since the
    margin is a small, fixed addition next to their much larger radius.

    *mask* is read the same way skeletonisation itself reads it -- via
    :func:`io.load._to_binary_volume_for_skeletonization`, not a plain
    ``!= 0`` test. The image this compares against is whatever was loaded,
    not necessarily an already-clean 0/1 mask (a noisy or grayscale-ish
    input is thresholded internally before skeletonising it), so treating
    every nonzero voxel as foreground can read almost the entire volume as
    "mask" on such an input and make a perfectly healthy skeleton look like
    it explains almost none of it -- the same real fixture above has only
    1,230 of its 110,592 voxels genuinely foreground once thresholded this
    way, not the 105,594 a bare ``!= 0`` reads. (Imported locally: ``io.load``
    imports ``preprocessing.skeleton`` at module scope, so importing it back
    at this module's own top level would be circular.)
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        return {
            "mask_voxel_count": 0,
            "explained_voxel_count": 0,
            "coverage_fraction": 1.0,
        }

    skeleton_bool = np.asarray(skeleton, dtype=bool)
    explained = _explained_by_local_radius(
        skeleton_bool, mask_bool, voxel_size_zyx=voxel_size_zyx
    )
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
        f"within their own local radius (plus discretisation margin) of the "
        f"skeleton ({report.get('coverage_fraction', 1.0):.1%})."
    )
