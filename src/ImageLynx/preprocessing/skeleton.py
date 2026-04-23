"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
import logging

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
    maximum_filter,
    uniform_filter,
)
from skimage.morphology import remove_small_objects, skeletonize
from skimage.transform import resize
logger = logging.getLogger(__name__)

def rescale_and_skeletonize_3d(
    binary_volume: np.ndarray, downsample_factor: float = 2.0
) -> np.ndarray:
    """Perform faster skeletonization by downsampling, skeletonizing, and upscaling.

    Parameters
    ----------
    binary_volume:
        Input boolean 3D array.
    downsample_factor:
        Factor to downsample by (e.g., 2.0 reduces each dimension by half).
    """
    if downsample_factor <= 1.0:
        return skeletonize_3d(binary_volume)

    # 1. Manual Max Pooling (most robust for binary downsampling)
    f = int(np.round(downsample_factor))
    
    # Pad to multiple of f
    pad_width = [ (0, (f - dim % f) % f) for dim in binary_volume.shape ]
    padded = np.pad(binary_volume, pad_width, mode='constant', constant_values=False)
    
    new_shape = [ dim // f for dim in padded.shape ]
    small_vol = padded.reshape(
        new_shape[0], f, 
        new_shape[1], f, 
        new_shape[2], f
    ).max(axis=(1, 3, 5))

    # 2. Skeletonize the small volume
    small_skel = skeletonize_3d(small_vol)

    # 3. Upscale back to original size
    thick_skel = resize(
        small_skel.astype(float),
        binary_volume.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5

    # 3.5 Expansion pass to ensure robustness during final thinning
    struct = generate_binary_structure(binary_volume.ndim, 1)
    thick_skel = binary_dilation(thick_skel, structure=struct)
    
    # 4. Final thinning pass
    return skeletonize_3d(thick_skel)


def keep_largest_mask_components(
    binary_mask: np.ndarray,
    n_components: int = 1,
    connectivity: int | None = None,
) -> np.ndarray:
    """Keep only the N largest connected components in a binary mask.

    Parameters
    ----------
    binary_mask:
        Input boolean mask.
    n_components:
        Number of largest components to keep.
    connectivity:
        Connectivity for labeling (e.g., 3 for 26-neighbor in 3D).
    """
    if not binary_mask.any():
        return binary_mask

    conn = _resolve_component_connectivity(binary_mask.ndim, connectivity)
    struct = generate_binary_structure(binary_mask.ndim, conn)
    labeled, n_found = label(binary_mask, structure=struct)

    if n_found <= n_components:
        return binary_mask

    sizes = np.bincount(labeled.ravel())
    # Sort labels by size descending, exclude background (index 0)
    sorted_labels = np.argsort(sizes[1:])[::-1] + 1
    keep_labels = sorted_labels[:n_components]

    return np.isin(labeled, keep_labels)


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


def fill_holes_3d(binary: np.ndarray) -> np.ndarray:
    """Fill internal holes in a 3D binary mask.

    Parameters
    ----------
    binary:
        Input boolean 3D array.
    """
    logger.info("Filling holes in 3D binary mask.")
    return binary_fill_holes(binary)


def skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Safe 3D skeletonization wrapper."""
    return skeletonize(img.astype(bool))

def _draw_line_3d(array: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Set voxels along the straight line from *start* to *end* to True."""
    n_steps = max(int(np.linalg.norm(end.astype(float) - start.astype(float))) + 1, 2)
    for t in np.linspace(0.0, 1.0, n_steps):
        pt = np.round(start + t * (end - start)).astype(int)
        pt = np.clip(pt, 0, np.array(array.shape) - 1)
        array[tuple(pt)] = True


def skeletonize_voxel_bundles_into_paths(
    binary_mask: np.ndarray,
    scan_size: int | tuple[int, ...] = 9,
    density_fraction: float = 0.35,
    max_connections_per_hub: int = 8,
    hub_min_spacing: int | None = None,
) -> np.ndarray:
    """Scan for dense local volumes and collapse each into a hub node.

    Strategy:
    1. Compute local voxel density with a sliding window of ``scan_size``.
    2. Detect dense hubs where local density >= ``density_fraction``.
    3. Pick local density peaks (spatially separated) as hub centers.
    4. Remove dense hub regions from the thin skeleton, insert one center node,
       and reconnect in/out paths using directional boundary links to avoid
       excessive overlap.

    Parameters
    ----------
    binary_mask:
        Boolean foreground mask/skeleton candidate.
    scan_size:
        Sliding window size used to estimate local density (int or per-axis
        tuple). Odd values are preferred.
    density_fraction:
        Mark a location as dense when its local foreground fraction is at least
        this threshold (0.0-1.0).
    max_connections_per_hub:
        Max number of directional in/out links reconnected to each hub.
    hub_min_spacing:
        Minimum Euclidean spacing between selected hub centers. Defaults to
        about half the smallest scan window dimension.
    """
    mask = binary_mask.astype(bool)
    if not mask.any():
        return mask

    if isinstance(scan_size, int):
        scan = (max(3, int(scan_size)),) * mask.ndim
    else:
        if len(scan_size) != mask.ndim:
            raise ValueError(
                f"scan_size must have {mask.ndim} dimensions, got {len(scan_size)}"
            )
        scan = tuple(max(3, int(s)) for s in scan_size)
    density_fraction = float(np.clip(density_fraction, 0.0, 1.0))
    max_connections_per_hub = max(1, int(max_connections_per_hub))
    if hub_min_spacing is None:
        hub_min_spacing = max(1, int(min(scan) / 2))

    base_skeleton = skeletonize_3d(mask)
    density = uniform_filter(mask.astype(np.float32), size=scan, mode="constant")
    dense_volume = density >= density_fraction
    if not dense_volume.any():
        return base_skeleton.astype(bool)

    peak_map = dense_volume & (density == maximum_filter(density, size=scan, mode="nearest"))
    peak_coords = np.argwhere(peak_map)
    if peak_coords.size == 0:
        return base_skeleton.astype(bool)

    order = np.argsort(density[tuple(peak_coords.T)])[::-1]
    selected_hubs: list[np.ndarray] = []
    for idx in order:
        candidate = peak_coords[idx]
        if all(np.linalg.norm(candidate - existing) >= hub_min_spacing for existing in selected_hubs):
            selected_hubs.append(candidate)

    result = base_skeleton.astype(bool).copy()
    shape = np.array(mask.shape)
    half_window = np.array(scan) // 2
    structure = generate_binary_structure(mask.ndim, 1)

    for hub in selected_hubs:
        lo = np.maximum(hub - half_window, 0)
        hi = np.minimum(hub + half_window + 1, shape)
        slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(mask.ndim))
        local_dense = np.zeros_like(mask, dtype=bool)
        local_dense[slices] = dense_volume[slices]
        if not local_dense.any():
            continue

        local_mask_coords = np.argwhere(mask[slices])
        if local_mask_coords.size == 0:
            continue
        local_center = hub - lo
        nearest = int(
            np.argmin(np.sum((local_mask_coords - local_center.astype(float)) ** 2, axis=1))
        )
        center = (local_mask_coords[nearest] + lo).astype(int)
        center_t = tuple(center.tolist())

        shell = binary_dilation(local_dense, structure=structure) & ~local_dense
        boundary_points = np.argwhere(result & shell)

        result[local_dense] = False
        result[center_t] = True

        if boundary_points.size == 0:
            continue

        by_direction: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        for pt in boundary_points:
            vec = pt - center
            direction = tuple(np.sign(vec).astype(int).tolist())
            if all(v == 0 for v in direction):
                continue
            dist2 = float(np.dot(vec, vec))
            prev = by_direction.get(direction)
            if prev is None or dist2 > prev[0]:
                by_direction[direction] = (dist2, pt)

        chosen = sorted(by_direction.values(), key=lambda item: item[0], reverse=True)[
            :max_connections_per_hub
        ]
        for _, endpoint in chosen:
            _draw_line_3d(result, center, endpoint)

    return skeletonize_3d(result.astype(bool)).astype(bool)


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
        result = skeletonize_3d(result)

    return result.astype(bool)


def preprocess_skeleton_for_graph(
    skeleton_image: np.ndarray,
    min_branch_length: int = 5,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    min_component_fraction: float = 0.0,
    bundle_scan_size: int | tuple[int, ...] = 9,
    bundle_density_fraction: float = 0.35,
    bundle_max_connections_per_hub: int = 8,
    bundle_hub_min_spacing: int | None = None,
    closing_radius: int = 0,
    bridge_gap_size: int = 0,
) -> np.ndarray:
    """Remove small objects, re-skeletonize, and reconnect isolated fragments.

    Parameters
    ----------
    skeleton_image:
        Raw boolean skeleton from :func:`skeletonize_3d`.
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
    bundle_scan_size:
        Sliding window size used to detect local dense bundles.
    bundle_density_fraction:
        Local foreground density threshold for treating a region as a bundle.
    bundle_max_connections_per_hub:
        Max directional links retained when reconnecting paths to each hub.
    bundle_hub_min_spacing:
        Minimum spacing between neighboring dense hub centers.
    closing_radius:
        Optional morphological closing radius.
    bridge_gap_size:
        Optional gap bridging distance.
    """
    if closing_radius > 0:
        skeleton_image = close_binary_mask(skeleton_image, radius=closing_radius)
        
    if bridge_gap_size > 0:
        skeleton_image = bridge_gaps(skeleton_image, max_gap=bridge_gap_size)

    conn = _resolve_component_connectivity(skeleton_image.ndim, component_connectivity)
    cleaned = remove_small_objects(
        skeleton_image.astype(bool),
        min_size=min_branch_length,
        connectivity=conn,
    )
    # Refine dense local bundles into single hub nodes with clean in/out links.
    cleaned = skeletonize_voxel_bundles_into_paths(
        cleaned,
        scan_size=bundle_scan_size,
        density_fraction=bundle_density_fraction,
        max_connections_per_hub=bundle_max_connections_per_hub,
        hub_min_spacing=bundle_hub_min_spacing,
    )

    if min_component_fraction > 0.0:
        cleaned = _filter_components_by_total_fraction(
            cleaned,
            min_component_fraction=min_component_fraction,
            component_connectivity=conn,
        )
    cleaned = skeletonize_3d(cleaned.astype(bool))
    if max_bridge_distance > 0:
        cleaned = connect_skeleton_components(
            cleaned.astype(bool),
            max_bridge_distance=max_bridge_distance,
            component_connectivity=conn,
        )
    return cleaned.astype(bool)
