"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from skimage.morphology import remove_small_objects, skeletonize
logger = logging.getLogger(__name__)

def _resolve_component_connectivity(ndim: int, connectivity: int | None) -> int:
    """Return valid component-connectivity in [1, ndim]."""
    if connectivity is None:
        return ndim
    return max(1, min(int(connectivity), ndim))


def _filter_components_by_total_fraction(
    skeleton: np.ndarray,
    min_component_fraction: float,
    component_connectivity: int | None = None,
) -> np.ndarray:
    """Keep only components with size >= min_component_fraction of total voxels."""
    skeleton_bool = skeleton.astype(bool)
    total_voxels = int(skeleton_bool.sum())
    if total_voxels == 0 or min_component_fraction <= 0.0:
        return skeleton_bool

    conn = _resolve_component_connectivity(skeleton_bool.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton_bool.ndim, conn)
    labeled, n_components = label(skeleton_bool, structure=structure)
    if n_components == 0:
        return skeleton_bool

    min_component_size = int(np.ceil(min_component_fraction * total_voxels))
    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(component_sizes >= min_component_size)[0]
    keep_labels = keep_labels[keep_labels != 0]  # exclude background

    if keep_labels.size == 0:
        logger.warning(
            "Component fraction %.4f removed all components (min size=%d). "
            "Returning original skeleton.",
            min_component_fraction,
            min_component_size,
        )
        return skeleton_bool

    return np.isin(labeled, keep_labels)


def print_skeleton_connectivity_stats(
    name: str,
    skeleton: np.ndarray,
    component_connectivity: int | None = None,
) -> None:
    """Print concise connectivity diagnostics for a 2D/3D skeleton."""
    skeleton_bool = skeleton.astype(bool)
    voxel_count = int(skeleton_bool.sum())
    conn = _resolve_component_connectivity(skeleton_bool.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton_bool.ndim, conn)
    labeled, n_components = label(skeleton_bool, structure=structure)
    if n_components == 0:
        print(f"[skeleton:{name}] empty skeleton (0 foreground voxels).")
        return

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    sorted_sizes = np.sort(component_sizes[1:])[::-1]
    largest = int(sorted_sizes[0]) if sorted_sizes.size else 0
    largest_fraction = (largest / voxel_count) if voxel_count else 0.0
    top_sizes = sorted_sizes[:10].tolist()

    print(
        f"[skeleton:{name}] shape={skeleton.shape}, dtype={skeleton.dtype}, "
        f"voxels={voxel_count}, components={int(n_components)}, "
        f"largest={largest} ({largest_fraction:.2%} of voxels)"
    )
    print(f"[skeleton:{name}] top component sizes (up to 10): {top_sizes}")

def _skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Skeletonize 3D binary image. Uses skeletonize with Lee method."""
    return skeletonize(img, method="lee")


def bridge_gaps(binary_skeleton: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill small gaps in a binary mask using a distance-transform dilation.

    Every background voxel within *max_gap* voxels of any foreground voxel is
    set to foreground.  Equivalent to morphological dilation with radius
    *max_gap*.
    """
    dist = distance_transform_edt(~binary_skeleton)
    fill_mask = (dist <= max_gap) & (~binary_skeleton)
    return binary_skeleton | fill_mask


def close_binary_mask(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    """Morphologically close a binary mask to seal small gaps.

    Applies binary dilation followed by binary erosion (closing) using a
    ball-shaped structuring element of the given *radius*.  Unlike plain
    dilation, closing does not permanently expand object boundaries — it only
    fills concavities and bridges narrow gaps smaller than the structuring
    element.

    Parameters
    ----------
    binary:
        Input boolean array (2D or 3D).
    radius:
        Number of erosion/dilation iterations.  Larger values bridge wider
        gaps but risk merging genuinely distinct structures.
    """
    from scipy.ndimage import binary_closing, generate_binary_structure

    if radius <= 0:
        return binary
    struct = generate_binary_structure(binary.ndim, 1)
    return binary_closing(binary.astype(bool), structure=struct, iterations=radius)


def skeletonize_3d_safe(img: np.ndarray) -> np.ndarray:
    """Safe 3D skeletonization wrapper."""
    return _skeletonize_3d(img.astype(bool))


def _draw_line_3d(array: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Set voxels along the straight line from *start* to *end* to True."""
    n_steps = max(int(np.linalg.norm(end.astype(float) - start.astype(float))) + 1, 2)
    for t in np.linspace(0.0, 1.0, n_steps):
        pt = np.round(start + t * (end - start)).astype(int)
        pt = np.clip(pt, 0, np.array(array.shape) - 1)
        array[tuple(pt)] = True


def connect_skeleton_components(
    skeleton: np.ndarray,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
) -> np.ndarray:
    """Bridge isolated skeleton components to the main (largest) component.

    After pruning, a skeleton may contain small isolated fragments that are
    genuinely connected in the original vessel image but were separated by the
    ``remove_small_objects`` step.  This function:

    1. Labels connected components.
    2. For every non-main component whose nearest voxel is within
       *max_bridge_distance* of the main component, draws a straight voxel
       line to reconnect it.
    3. Re-skeletonizes the result to restore a single-pixel-wide skeleton.

    Parameters
    ----------
    skeleton:
        Boolean skeleton array.
    max_bridge_distance:
        Maximum voxel distance allowed for bridging.  Isolated fragments
        further away than this are left untouched.
    """
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    labeled, n_components = label(skeleton, structure=structure)
    if n_components <= 1:
        return skeleton

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # exclude background
    main_label = int(np.argmax(component_sizes))

    result = skeleton.copy()
    main_coords = np.argwhere(labeled == main_label)
    tree = cKDTree(main_coords)

    bridged = 0
    for comp_id in range(1, n_components + 1):
        if comp_id == main_label:
            continue
        comp_coords = np.argwhere(labeled == comp_id)
        if len(comp_coords) == 0:
            continue
        dists, idxs = tree.query(comp_coords)
        nearest_comp_idx = int(np.argmin(dists))
        min_dist = float(dists[nearest_comp_idx])
        if min_dist > max_bridge_distance:
            continue
        start = comp_coords[nearest_comp_idx]
        end = main_coords[int(idxs[nearest_comp_idx])]
        _draw_line_3d(result, start, end)
        bridged += 1

    if bridged:
        logger.debug("Bridged %d isolated skeleton component(s).", bridged)
        result = _skeletonize_3d(result)

    return result.astype(bool)


def preprocess_skeleton_for_graph(
    skeleton_image: np.ndarray,
    min_branch_length: int = 5,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    min_component_fraction: float = 0.0,
) -> np.ndarray:
    """Remove small objects, re-skeletonize, and reconnect isolated fragments.

    Parameters
    ----------
    skeleton_image:
        Raw boolean skeleton from :func:`skeletonize_3d_safe`.
    min_branch_length:
        Connected components with fewer than this many voxels are removed
        before re-skeletonizing (reduces degree-2 noise nodes).
    max_bridge_distance:
        After pruning, isolated components whose nearest voxel is within this
        many voxels of the main skeleton are bridged back in.  Set to 0 to
        disable reconnection.
    component_connectivity:
        Connectivity used when identifying connected components. Defaults to
        full neighborhood (8-neighbor in 2D, 26-neighbor in 3D), which
        preserves diagonal links as connected.
    min_component_fraction:
        Minimum fraction (0.0-1.0) of total skeleton voxels required for a
        connected component to be retained. For example, 0.05 keeps only
        components with at least 5% of all skeleton voxels.
    """
    conn = _resolve_component_connectivity(skeleton_image.ndim, component_connectivity)
    cleaned = remove_small_objects(
        skeleton_image.astype(bool),
        min_size=min_branch_length,
        connectivity=conn,
    )
    if min_component_fraction > 0.0:
        cleaned = _filter_components_by_total_fraction(
            cleaned,
            min_component_fraction=min_component_fraction,
            component_connectivity=conn,
        )
    cleaned = _skeletonize_3d(cleaned.astype(bool))
    if max_bridge_distance > 0:
        cleaned = connect_skeleton_components(
            cleaned.astype(bool),
            max_bridge_distance=max_bridge_distance,
            component_connectivity=conn,
        )
    return cleaned.astype(bool)
