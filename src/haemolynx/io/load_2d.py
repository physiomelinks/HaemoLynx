"""Load a 2D image into the pipeline as a single-slice (z=1) volume.

Every stage downstream of loading works in canonical ``(z, y, x)`` volumes
(see CLAUDE.md's axis-order convention); a genuinely 2D image has no z axis
at all, a different shape from anything the rest of the pipeline was
written to expect. skimage's own Lee skeletonisation and this package's
EDT/graph-building logic are both dimension-generic and handle a
single-slice ``(1, y, x)`` volume correctly -- confirmed directly: a
synthetic 2D vessel network skeletonises and builds into the expected
graph topology once represented that way. So the fix here is narrow: read
the file exactly the way :mod:`haemolynx.io.load` already does (identical
metadata handling -- x/y voxel size from the TIFF/H5's own tags, the same
``voxel_size_policy``/override machinery every other input goes through),
and promote a 2D result to ``(1, y, x)`` instead of leaving it 2D (the
TIFF path today) or raising (the H5 path today).

The promotion is not free: most of this pipeline's genuinely 3D-specific
features (thickness-gated skeletonisation, the cartwheel-hub and
thick-vessel braid guards, tube-mesh rendering) were built and tested
against real z-extent, not flat data. They have not been verified against
a single-slice volume, so a warning is logged -- and recorded on the voxel
metadata status as ``promoted_from_2d`` -- whenever this actually promotes
something, rather than silently treating a 2D image as business as usual.

``pipeline.stages._load_volume_for_skeletonise`` calls
:func:`load_image_with_voxel_size_2d_aware` directly, so a 2D file reaches
the pipeline through the exact same ``input_path`` / ``image_axis_order``
settings and GUI row every other input does -- there is no separate
control for this.
"""
from __future__ import annotations

import logging

import numpy as np

from .axis_order import CANONICAL_AXIS_ORDER
from .load import load_3d_h5_with_voxel_size, load_3d_tif_with_voxel_size

logger = logging.getLogger(__name__)


def promote_2d_to_single_slice_volume(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(image, was_2d)``.

    *image* is returned unchanged unless it is 2D, in which case it is
    reshaped to ``(1, *image.shape)`` -- a single-slice ``(z, y, x)``
    volume with ``z=1``, the shape every other stage already expects.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[np.newaxis, ...], True
    return arr, False


def _voxel_meta_status_for_promotion(
    filepath: str, image: np.ndarray, was_2d: bool, voxel_meta_status: dict[str, object]
) -> dict[str, object]:
    if not was_2d:
        return voxel_meta_status
    logger.warning(
        "%s is a 2D image (%d x %d); loading it as a single-slice volume "
        "(z=1) rather than rejecting it. Most of this pipeline's "
        "genuinely 3D-specific features (thickness-gated skeletonisation, "
        "the cartwheel-hub and thick-vessel braid guards, tube-mesh "
        "rendering) were built and tested against real z-extent, not "
        "flat data, and have not been verified against a single-slice "
        "volume -- check results carefully.",
        filepath,
        image.shape[1],
        image.shape[2],
    )
    return {**voxel_meta_status, "promoted_from_2d": True}


def load_image_with_voxel_size_2d_aware(
    filepath: str,
    *,
    input_format: str,
    axis_order: str = CANONICAL_AXIS_ORDER,
    h5_dataset_name: str | None = None,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Load a TIFF or H5 image exactly the way the ordinary 3D loaders do,
    promoting a genuinely 2D result to a single-slice volume instead of
    leaving it 2D (the TIFF path) or raising (the H5 path).

    *input_format* selects the reader the same way
    ``pipeline.stages._load_volume_for_skeletonise`` already does --
    ``"tif"``/``"tiff"`` or ``"h5"``. Voxel size and its metadata status
    come back exactly as the underlying loader reports them, plus
    ``promoted_from_2d`` in the status dict when this actually promoted
    something (missing z metadata, e.g. no z-resolution tag on a 2D TIFF,
    already falls back to 1.0 through the same path a 3D file with no
    z tag uses -- nothing extra needed for that here).
    """
    fmt = input_format.strip().lower()
    if fmt in {"tif", "tiff"}:
        image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status = (
            load_3d_tif_with_voxel_size(filepath, axis_order=axis_order)
        )
    elif fmt == "h5":
        image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status = (
            load_3d_h5_with_voxel_size(
                filepath,
                h5_dataset_name,
                axis_order=axis_order,
                allow_2d=True,
            )
        )
    else:
        raise ValueError("input_format must be 'tif', 'tiff', or 'h5'.")

    image, was_2d = promote_2d_to_single_slice_volume(image)
    voxel_meta_status = _voxel_meta_status_for_promotion(
        filepath, image, was_2d, voxel_meta_status
    )
    return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status
