"""The CB pipeline's vessel mask, as one function other stages can call.

`examples/carotid_image_to_model.py` builds its mask in two places: `_apply_preprocessing_filters`
(pad, hysteresis, hole filling, unpad) and `_preprocess_local_mask` (closing, keep the largest
component). The threshold selector reads its calibre on a plain ``p >= t`` cut, 5.27 um at 0.95
where the network reads a median of 6.70 um; it prints the same statistic on this mask (d_net)
so the gap is visible at every threshold (open item 41).

This is a copy of that recipe at the CB defaults, not a refactor of the pipeline, so the batch
outputs stay byte-identical. ``tests/test_vessel_mask.py`` checks the two agree.
"""
import numpy as np

from .image import hysteresis_threshold
from .skeleton import close_binary_mask, fill_holes_3d, keep_largest_mask_components


def build_vessel_mask(
    probabilities: np.ndarray,
    low: float,
    high: float,
    *,
    pad_z: int = 10,
    closing_radius: int = 1,
    n_components: int = 1,
    connectivity: int = 3,
) -> np.ndarray:
    """Hysteresis mask with the pipeline's padding, hole filling, closing and pruning.

    The defaults are the CB pipeline's: the "caged" boundary mode pads 10 slices in z by edge
    replication, ``SkeletonConfig.closing_radius`` is 1, ``prune_mask_before`` keeps 1
    component, and ``component_connectivity`` is 3. On a supersampled field, scale ``pad_z``
    and ``closing_radius`` by the factor so they keep the same physical size.
    """
    if high <= low:
        raise ValueError(
            f"Hysteresis seed ({high}) must be above the flood threshold ({low}).")
    image = np.asarray(probabilities)
    if pad_z > 0:
        image = np.pad(image, ((pad_z, pad_z), (0, 0), (0, 0)), mode="edge")
    binary = hysteresis_threshold(image, low=low, high=high)
    binary = fill_holes_3d(binary)
    if pad_z > 0:
        binary = binary[pad_z:-pad_z]
    if closing_radius > 0:
        binary = close_binary_mask(binary, radius=closing_radius)
    if n_components > 0:
        binary = keep_largest_mask_components(
            binary, n_components=n_components, connectivity=connectivity)
    return binary
