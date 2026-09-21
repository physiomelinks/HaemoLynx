"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

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

from .memmap_support import new_memmap_array, release_memmap_array, temporary_memmap_array

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


@dataclass(frozen=True)
class SkeletonConnectivityStats:
    """Connected-component summary of a binary skeleton, largest component first."""

    shape: tuple[int, ...]
    voxel_count: int
    n_components: int
    largest_component_size: int
    largest_fraction: float
    component_sizes: tuple[int, ...]


def compute_skeleton_connectivity_stats(
    skeleton: np.ndarray,
    component_connectivity: int | None = None,
) -> SkeletonConnectivityStats:
    """Connected-component sizes of *skeleton*, largest first.

    Labelling a full stack is not free, so a caller that only wants this
    occasionally in a log line should keep using
    :func:`log_skeleton_connectivity_stats`, which skips the computation
    entirely when INFO logging is off. A caller that needs the numbers
    themselves -- the settings optimiser scoring a cleanup candidate -- calls
    this directly.
    """
    # `astype` copies a whole volume even when it is already boolean.
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    voxel_count = int(skeleton_bool.sum())
    if voxel_count == 0:
        return SkeletonConnectivityStats(
            shape=tuple(skeleton.shape),
            voxel_count=0,
            n_components=0,
            largest_component_size=0,
            largest_fraction=0.0,
            component_sizes=(),
        )

    conn = _resolve_component_connectivity(skeleton_bool.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton_bool.ndim, conn)
    labeled, n_components = label(skeleton_bool, structure=structure)

    # Only foreground voxels carry a component label, so counting those is the
    # same tally as counting the whole volume -- minus the background at index
    # 0, which is zeroed here anyway. On a sparse skeleton that is thousands of
    # voxels rather than hundreds of millions.
    component_sizes = np.bincount(labeled[skeleton_bool], minlength=n_components + 1)
    component_sizes[0] = 0
    sorted_sizes = np.sort(component_sizes[1:])[::-1]
    largest = int(sorted_sizes[0]) if sorted_sizes.size else 0
    largest_fraction = (largest / voxel_count) if voxel_count else 0.0
    return SkeletonConnectivityStats(
        shape=tuple(skeleton.shape),
        voxel_count=voxel_count,
        n_components=int(n_components),
        largest_component_size=largest,
        largest_fraction=largest_fraction,
        component_sizes=tuple(int(s) for s in sorted_sizes),
    )


def log_skeleton_connectivity_stats(
    name: str,
    skeleton: np.ndarray,
    component_connectivity: int | None = None,
) -> None:
    """Log concise connectivity diagnostics for a 2D/3D skeleton.

    Diagnostics only, so it does nothing when nothing is listening -- labelling
    a full stack is not free, and a caller that has turned INFO off has said it
    does not want this.
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    stats = compute_skeleton_connectivity_stats(skeleton, component_connectivity)
    if stats.n_components == 0:
        logger.warning(f"[skeleton:{name}] empty skeleton (0 foreground voxels).")
        return

    logger.info(
        f"[skeleton:{name}] shape={stats.shape}, dtype={skeleton.dtype}, "
        f"voxels={stats.voxel_count}, components={stats.n_components}, "
        f"largest={stats.largest_component_size} ({stats.largest_fraction:.2%} of voxels)"
    )
    logger.info(
        f"[skeleton:{name}] top component sizes (up to 10): "
        f"{list(stats.component_sizes[:10])}"
    )


def fill_binary_holes(
    mask: np.ndarray, *, use_memmap: bool = False, memmap_directory: str | Path | None = None
) -> np.ndarray:
    """Fill background regions of *mask* that are enclosed by foreground.

    The same result as :func:`scipy.ndimage.binary_fill_holes`, reached the
    other way round: label the background and keep whatever fails to reach an
    edge of the volume, rather than flood-filling inward from the border. Both
    are one pass, but the flood fill has to propagate through every background
    voxel one dilation at a time, and on a sparse skeleton -- where the
    background is almost the entire volume -- that is much the slower of the
    two.

    Uses face connectivity for the background, as ``binary_fill_holes`` does by
    default, so a diagonal chink counts as a way out for neither.

    The trade is memory for time: labelling holds an int32 per voxel where the
    flood fill holds a bool, so this wants about four times the working set of
    the volume. That is the right way round for the stacks this runs on, but it
    is the reason to reach for the flood fill on a machine short of memory.
    Windows tries int16 first (half the commit); Linux keeps the int32 path
    that was timed there. Both return the same mask.

    *use_memmap*, when True, writes the labelled background to a disk-backed
    buffer instead of a fresh in-RAM one -- this runs unconditionally on
    every load (see :func:`haemolynx.io.load._skeletonize_loaded_volume`), so
    it is the one full-volume, wider-than-boolean allocation a low-memory
    run cannot just disable. *memmap_directory* is forwarded to
    :func:`haemolynx.preprocessing.new_memmap_array`; ignored when
    *use_memmap* is False.
    """
    mask = np.asarray(mask, dtype=bool)
    background_labels, n_labels = _label_inverted_background(
        ~mask, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    if n_labels == 0:
        if use_memmap and isinstance(background_labels, np.memmap):
            release_memmap_array(background_labels)
        return mask

    reaches_edge = np.zeros(n_labels + 1, dtype=bool)
    for axis in range(mask.ndim):
        for face in (0, -1):
            face_slice = [slice(None)] * mask.ndim
            face_slice[axis] = face
            reaches_edge[background_labels[tuple(face_slice)]] = True
    # Label 0 is the foreground itself, never a hole to fill.
    reaches_edge[0] = True
    result = mask | ~reaches_edge[background_labels]
    if use_memmap and isinstance(background_labels, np.memmap):
        release_memmap_array(background_labels)
    return result


def _label_inverted_background(
    inverted: np.ndarray,
    *,
    platform: str | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Label background components; compact dtype on Windows only.

    *use_memmap*, when True, allocates the labelled array (and, on Windows,
    its int32 fallback) as a disk-backed buffer -- the caller
    (:func:`fill_binary_holes`) releases it once it has read what it needs.
    *memmap_directory* is forwarded to :func:`new_memmap_array`.
    """
    if platform is None:
        platform = sys.platform
    if platform == "win32":
        compact = (
            new_memmap_array(inverted.shape, np.int16, directory=memmap_directory)
            if use_memmap
            else np.empty(inverted.shape, dtype=np.int16)
        )
        try:
            n_labels = int(label(inverted, output=compact))
            return compact, n_labels
        except (RuntimeError, TypeError, ValueError):
            if use_memmap:
                release_memmap_array(compact)
    if use_memmap:
        labeled = new_memmap_array(inverted.shape, np.int32, directory=memmap_directory)
        n_labels = int(label(inverted, output=labeled))
        return labeled, n_labels
    labeled, n_labels = label(inverted)
    return labeled, int(n_labels)


def _euclidean_ball(radius: int) -> np.ndarray:
    """Offsets whose Euclidean distance from the centre is at most *radius*."""
    span = np.arange(-radius, radius + 1)
    grids = np.meshgrid(*([span] * 3), indexing="ij")
    return sum(g.astype(np.int64) ** 2 for g in grids) <= radius * radius


#: Largest *max_gap* for which :func:`bridge_gaps` dilates rather than measures.
#: A ball footprint has ``(2r+1)^3`` elements and a dilation costs the volume
#: times that, while the distance transform costs the volume whatever the
#: radius, so the two cross over. Measured on a 329-million-voxel stack: radius
#: 1 dilates 17x faster, radius 3 still 1.7x faster, radius 4 is 0.8x -- slower
#: -- and radius 8 is nine times slower. Hence 3.
MAX_BALL_DILATION_RADIUS = 3


def bridge_gaps(
    binary_skeleton: np.ndarray,
    max_gap: int = 4,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Fill small gaps in a binary mask.

    Every background voxel within *max_gap* voxels of any foreground voxel is
    set to foreground -- that is, dilation by a Euclidean ball of radius
    *max_gap*. Two ways to get there, and which is cheaper depends on the
    radius: dilating by that ball directly, or measuring an exact distance for
    every voxel in the volume and thresholding it. Small radii dilate; from
    :data:`MAX_BALL_DILATION_RADIUS` up the footprint grows faster than the
    transform does and the transform wins. Both return the same mask, so this
    only decides how long it takes.

    *use_memmap*, when True and the distance-transform path is taken, writes
    the transform to a disk-backed buffer instead of a fresh in-RAM
    ``float64`` array -- 8 bytes per voxel, the widest full-volume allocation
    anywhere in this module. Only reached when ``max_gap`` exceeds
    :data:`MAX_BALL_DILATION_RADIUS` (not the pipeline's own default).
    *memmap_directory* is forwarded to :func:`temporary_memmap_array`;
    ignored when *use_memmap* is False.
    """
    if max_gap <= 0:
        return binary_skeleton
    max_gap = int(max_gap)
    if max_gap <= MAX_BALL_DILATION_RADIUS:
        return binary_dilation(binary_skeleton, structure=_euclidean_ball(max_gap))
    inverted = ~binary_skeleton
    if use_memmap:
        with temporary_memmap_array(
            binary_skeleton.shape, np.float64, directory=memmap_directory
        ) as distance:
            distance_transform_edt(inverted, distances=distance)
            return binary_skeleton | ((distance <= max_gap) & inverted)
    distance = distance_transform_edt(inverted)
    return binary_skeleton | ((distance <= max_gap) & inverted)


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


def skeletonize_volume(img: np.ndarray) -> np.ndarray:
    """Skeletonize a 2D/3D binary volume (Lee method; replaces legacy skeletonize_3d)."""
    return skeletonize(img.astype(bool), method="lee")


def skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`skeletonize_volume`."""
    return skeletonize_volume(img)

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

    base_skeleton = skeletonize_volume(mask)
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

    return skeletonize_volume(result.astype(bool)).astype(bool)


#: Padding (voxels) added around a candidate bridge's own bounding box when
#: cropping the segmentation mask for :func:`_bridge_path_through_mask` --
#: enough slack for the router to curve around a nearby obstruction without
#: needing the full-volume windowed-cost-budget machinery
#: `graph/reconnect.py` uses for its much larger search space (a skeleton
#: bridge is already bounded by ``max_bridge_distance``).
_BRIDGE_MASK_WINDOW_PAD = 3


def _bridge_path_through_mask(
    mask: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray | None:
    """A* path from *start* to *end* that prefers staying inside *mask*.

    Same cost-field convention as `graph.reconnect`'s own skeleton-proximity
    router (``1 + distance_transform_edt(~reference) ** 2``), but built from
    the segmentation mask instead of the skeleton, and only over a small
    local window around the two endpoints. Returns ``None`` (never raises) on
    any failure -- a shape mismatch, an out-of-bounds window, or a routing
    exception -- so the caller can fall back to a straight line.
    """
    try:
        from skimage.graph import route_through_array

        lo = np.maximum(np.minimum(start, end) - _BRIDGE_MASK_WINDOW_PAD, 0)
        hi = np.minimum(
            np.maximum(start, end) + _BRIDGE_MASK_WINDOW_PAD + 1, np.array(mask.shape)
        )
        if np.any(hi <= lo):
            return None
        window = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        cost = 1 + distance_transform_edt(~window.astype(bool)) ** 2
        start_local = tuple((start - lo).astype(int))
        end_local = tuple((end - lo).astype(int))
        path_coords, _cost = route_through_array(
            cost, start_local, end_local, fully_connected=True
        )
        if not path_coords:
            return None
        return np.asarray(path_coords, dtype=int) + lo
    except Exception:
        logger.debug("Mask-weighted bridge routing failed; using a straight line.", exc_info=True)
        return None


def connect_skeleton_components(
    skeleton: np.ndarray,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    *,
    z_distance_weight: float = 1.0,
    segmentation_mask: np.ndarray | None = None,
    weight_by_segmentation: bool = False,
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
    z_distance_weight:
        Multiplies the z-component of every inter-voxel distance used both
        to pick the nearest pair and to compare against
        *max_bridge_distance* -- the z and xy axes are otherwise treated as
        equally spaced regardless of the actual voxel size. 1.0 (default)
        reproduces that voxel-isotropic behaviour exactly; above 1.0 a given
        z-gap counts for more, discouraging bridges that reach mostly
        through z relative to xy; below 1.0 does the reverse.
    segmentation_mask:
        Optional binary mask of the real segmented tissue, same shape as
        *skeleton*. Only used when *weight_by_segmentation* is also true.
    weight_by_segmentation:
        When true and *segmentation_mask* is given, each accepted bridge is
        drawn by routing through the mask (preferring to stay inside real
        segmented signal) rather than an unconditional straight line --
        see :func:`_bridge_path_through_mask`. A bridge within
        *max_bridge_distance* is always drawn either way; this only changes
        the path's shape, never whether two components get connected.
    """
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    labeled, n_components = label(skeleton, structure=structure)
    if n_components <= 1:
        return skeleton

    z_weight = float(z_distance_weight)
    axis_weights = np.array([z_weight, 1.0, 1.0], dtype=float)

    # One pass over the labels, then split by component. Asking
    # `labeled == comp_id` per component instead re-reads the whole volume once
    # for every component, which on a full stack with a hundred-odd fragments
    # was the bulk of this function's cost.
    coords_all = np.argwhere(labeled)
    labels_all = labeled[tuple(coords_all.T)]
    order = np.argsort(labels_all, kind="stable")
    coords_all = coords_all[order]
    labels_all = labels_all[order]
    starts = np.searchsorted(labels_all, np.arange(1, n_components + 2))

    comp_coords: dict[int, np.ndarray] = {}
    comp_trees: dict[int, cKDTree] = {}
    for comp_id in range(1, n_components + 1):
        lo, hi = int(starts[comp_id - 1]), int(starts[comp_id])
        if hi <= lo:
            continue
        coords = coords_all[lo:hi]
        comp_coords[comp_id] = coords
        # The tree is built (and queried) in z-weighted space so nearest-pair
        # selection and the distance cutoff both respect `z_distance_weight`;
        # the original, unscaled `coords` above are what actually get drawn.
        comp_trees[comp_id] = cKDTree(coords * axis_weights)

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
            dists, idxs = comp_trees[cid_b].query(comp_coords[cid_a] * axis_weights)
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
        path = None
        if weight_by_segmentation and segmentation_mask is not None:
            if segmentation_mask.shape == skeleton.shape:
                path = _bridge_path_through_mask(segmentation_mask, start, end)
        if path is not None:
            result[tuple(path.T)] = True
        else:
            _draw_line_3d(result, start, end)
        _union(cid_a, cid_b)
        bridged += 1

    if bridged:
        logger.debug("Bridged %d skeleton component pair(s).", bridged)
        result = skeletonize_volume(result)

    return result.astype(bool)


def inter_component_gap_distances(
    skeleton: np.ndarray,
    component_connectivity: int | None = None,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    z_distance_weight: float = 1.0,
) -> np.ndarray:
    """Nearest-neighbour distance between every pair of skeleton components.

    The same union of per-component :class:`~scipy.spatial.cKDTree` nearest-pair
    queries :func:`connect_skeleton_components` uses to decide what to bridge,
    without a distance cutoff and without drawing anything -- just the
    distribution of gaps that exist. Distances are in the units of
    *voxel_size_zyx* (microns when given, voxels when left at the default).

    A skeleton with 0 or 1 components has no gaps, so this returns an empty
    array rather than raising -- the caller (candidate-value generation for the
    settings optimiser) always wants a percentile of this array, and
    ``np.percentile`` of an empty array raises, so it is on the caller to check
    ``.size`` first and fall back to a schema default.

    *z_distance_weight* is the same extra z-axis multiplier
    :func:`connect_skeleton_components` accepts -- applied on top of
    *voxel_size_zyx* -- so a caller generating candidate distance thresholds
    for that function sees the same metric it will actually be compared
    against.
    """
    from scipy.spatial import cKDTree

    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    labeled, n_components = label(skeleton, structure=structure)
    if n_components <= 1:
        return np.array([], dtype=float)

    spacing = np.asarray(voxel_size_zyx, dtype=float) * np.array(
        [float(z_distance_weight), 1.0, 1.0]
    )
    coords_all = np.argwhere(labeled)
    labels_all = labeled[tuple(coords_all.T)]
    order = np.argsort(labels_all, kind="stable")
    coords_all = coords_all[order]
    labels_all = labels_all[order]
    starts = np.searchsorted(labels_all, np.arange(1, n_components + 2))

    comp_coords: dict[int, np.ndarray] = {}
    comp_trees: dict[int, cKDTree] = {}
    for comp_id in range(1, n_components + 1):
        lo, hi = int(starts[comp_id - 1]), int(starts[comp_id])
        if hi <= lo:
            continue
        physical = coords_all[lo:hi] * spacing
        comp_coords[comp_id] = physical
        comp_trees[comp_id] = cKDTree(physical)

    comp_ids = sorted(comp_coords.keys())
    distances: list[float] = []
    for i, cid_a in enumerate(comp_ids):
        for cid_b in comp_ids[i + 1:]:
            dists, _ = comp_trees[cid_b].query(comp_coords[cid_a])
            distances.append(float(dists.min()))

    return np.sort(np.asarray(distances, dtype=float))


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
    *,
    bridge_z_distance_weight: float = 1.0,
    segmentation_mask: np.ndarray | None = None,
    bridge_weight_by_segmentation: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Remove small objects, re-skeletonize, and reconnect isolated fragments.

    Parameters
    ----------
    skeleton_image:
        Raw boolean skeleton from :func:`skeletonize_volume` (or initial load).
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
    bridge_z_distance_weight, segmentation_mask, bridge_weight_by_segmentation:
        Forwarded to :func:`connect_skeleton_components` as
        ``z_distance_weight``, ``segmentation_mask``, and
        ``weight_by_segmentation`` respectively -- see its own docstring.
    use_memmap, memmap_directory:
        Forwarded to :func:`bridge_gaps`, whose distance-transform path (only
        taken when ``bridge_gap_size`` exceeds
        :data:`MAX_BALL_DILATION_RADIUS`) is the one full-volume ``float64``
        allocation this function can make.
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
        cleaned = bridge_gaps(
            cleaned.astype(bool),
            max_gap=bridge_gap_size,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )

    cleaned = skeletonize_volume(cleaned.astype(bool))

    # Bridge remaining disconnected components BEFORE filtering by size so
    # that small fragments get a chance to merge rather than being discarded.
    if max_bridge_distance > 0:
        cleaned = connect_skeleton_components(
            cleaned.astype(bool),
            max_bridge_distance=max_bridge_distance,
            component_connectivity=conn,
            z_distance_weight=bridge_z_distance_weight,
            segmentation_mask=segmentation_mask,
            weight_by_segmentation=bridge_weight_by_segmentation,
        )

    if min_component_fraction > 0.0:
        cleaned = _filter_components_by_total_fraction(
            cleaned,
            min_component_fraction=min_component_fraction,
            component_connectivity=conn,
        )

    return cleaned.astype(bool)
