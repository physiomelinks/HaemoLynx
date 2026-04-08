"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
import logging

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
    maximum_filter,
    uniform_filter,
)
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


def skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Safe 3D skeletonization wrapper."""
    return skeletonize(img.astype(bool), method="lee").astype(bool)

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
    """Bridge nearby skeleton components with straight voxel lines.

    Unlike a main-component-only strategy, this function considers *all*
    pairwise inter-component gaps.  A greedy union-find approach bridges the
    closest pairs first, avoiding redundant connections once two components
    have already been merged.

    Parameters
    ----------
    skeleton:
        Boolean skeleton array.
    max_bridge_distance:
        Maximum voxel distance allowed for bridging.  Component pairs
        further apart than this are left disconnected.
    """
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    labeled, n_components = label(skeleton, structure=structure)
    if n_components <= 1:
        return skeleton

    comp_coords: dict[int, np.ndarray] = {}
    comp_trees: dict[int, cKDTree] = {}
    for comp_id in range(1, n_components + 1):
        coords = np.argwhere(labeled == comp_id)
        if len(coords) == 0:
            continue
        comp_coords[comp_id] = coords
        comp_trees[comp_id] = cKDTree(coords)

    # Union-find helpers
    _parent: dict[int, int] = {c: c for c in comp_coords}

    def _find(x: int) -> int:
        while _parent[x] != x:
            _parent[x] = _parent[_parent[x]]
            x = _parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            _parent[ra] = rb

    # Collect candidate bridges (distance, start, end, comp_a, comp_b)
    comp_ids = sorted(comp_coords.keys())
    candidates: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
    for i, cid_a in enumerate(comp_ids):
        for cid_b in comp_ids[i + 1 :]:
            dists, idxs = comp_trees[cid_b].query(comp_coords[cid_a])
            nearest_idx = int(np.argmin(dists))
            min_dist = float(dists[nearest_idx])
            if min_dist <= max_bridge_distance:
                start = comp_coords[cid_a][nearest_idx]
                end = comp_coords[cid_b][int(idxs[nearest_idx])]
                candidates.append((min_dist, start, end, cid_a, cid_b))

    candidates.sort(key=lambda c: c[0])

    result = skeleton.copy()
    bridged = 0
    for _, start, end, cid_a, cid_b in candidates:
        if _find(cid_a) == _find(cid_b):
            continue
        _draw_line_3d(result, start, end)
        _union(cid_a, cid_b)
        bridged += 1

    if bridged:
        logger.debug("Bridged %d skeleton component pair(s).", bridged)
        result = skeletonize_3d(result)

    return result.astype(bool)


def preprocess_skeleton_for_graph(
    skeleton_image: np.ndarray,
    min_branch_length: int = 5,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    min_component_fraction: float = 0.0,
    closing_radius: int = 0,
    bridge_gap_size: int = 0,
    bundle_scan_size: int | tuple[int, ...] = 9,
    bundle_density_fraction: float = 0.35,
    bundle_max_connections_per_hub: int = 8,
    bundle_hub_min_spacing: int | None = None,
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
    closing_radius:
        Morphological closing iterations applied before re-skeletonization.
        Seals narrow gaps without permanently expanding boundaries. Set to 0
        to disable.
    bridge_gap_size:
        Maximum distance (in voxels) for the dilation-based gap filler
        applied before re-skeletonization. Every background voxel within this
        distance of a foreground voxel is set to foreground, then the result
        is re-skeletonized. Set to 0 to disable.
    bundle_scan_size:
        Sliding window size used to detect local dense bundles.
    bundle_density_fraction:
        Local foreground density threshold for treating a region as a bundle.
    bundle_max_connections_per_hub:
        Max directional links retained when reconnecting paths to each hub.
    bundle_hub_min_spacing:
        Minimum spacing between neighboring dense hub centers.
    """
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

    # Morphological closing seals narrow gaps without expanding boundaries.
    if closing_radius > 0:
        cleaned = close_binary_mask(cleaned, radius=closing_radius)

    # Dilation-based gap filling reconnects nearby foreground regions.
    if bridge_gap_size > 0:
        cleaned = bridge_gaps(cleaned.astype(bool), max_gap=bridge_gap_size)

    cleaned = skeletonize_3d(cleaned.astype(bool))

    # Bridge remaining disconnected components BEFORE filtering by size so
    # that small fragments get a chance to merge rather than being discarded.
    if max_bridge_distance > 0:
        cleaned = connect_skeleton_components(
            cleaned.astype(bool),
            max_bridge_distance=max_bridge_distance,
            component_connectivity=conn,
        )

    if min_component_fraction > 0.0:
        cleaned = _filter_components_by_total_fraction(
            cleaned,
            min_component_fraction=min_component_fraction,
            component_connectivity=conn,
        )

    return cleaned.astype(bool)
