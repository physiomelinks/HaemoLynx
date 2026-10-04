"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
    maximum_filter,
    uniform_filter,
)
from skimage.morphology import remove_small_objects, skeletonize

from .bridge_mask_support import (
    DEFAULT_MAX_BACKGROUND_GAP_UM,
    DEFAULT_MIN_MASK_FRACTION,
    MaskSupport,
)
from .memmap_support import (
    LOW_MEMORY_BLOCK_VOXELS,
    argwhere_by_slab,
    bincount_by_slab,
    iter_blocks,
    map_blockwise,
    new_memmap_array,
    release_memmap_array,
    release_superseded,
    slab_step,
    temporary_memmap_array,
)

logger = logging.getLogger(__name__)

def _resolve_component_connectivity(ndim: int, connectivity: int | None) -> int:
    """Return valid component-connectivity in [1, ndim]."""
    if connectivity is None:
        return ndim
    return max(1, min(int(connectivity), ndim))


@contextmanager
def _labeled_components(
    mask: np.ndarray,
    structure: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
):
    """Yield ``(labeled, n_components)`` for ``label(mask, structure=structure)``.

    Calling ``scipy.ndimage.label`` with no ``output=`` argument -- as every
    caller here used to -- always allocates a fresh, plain-RAM array the
    size of *mask*, regardless of how *mask* itself is backed. With
    *use_memmap* True, the labelled array is a transient memmap instead,
    released (if this block does not raise first) when the ``with`` block
    exits; do not let *labeled* escape it. Unlike
    :func:`_label_inverted_background`, no Windows-specific compact dtype
    is attempted -- that saving was worth its own complexity only for
    ``fill_binary_holes``, which runs unconditionally on every load.
    """
    if not use_memmap:
        labeled, n_components = label(mask, structure=structure)
        yield labeled, n_components
        return
    with temporary_memmap_array(mask.shape, np.int32, directory=memmap_directory) as labeled:
        n_components = int(label(mask, structure=structure, output=labeled))
        yield labeled, n_components


def _filter_components_by_total_fraction(
    skeleton: np.ndarray,
    min_component_fraction: float,
    component_connectivity: int | None = None,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Keep only components with size >= min_component_fraction of total voxels.

    *use_memmap*, when True, backs the labelled array (see
    :func:`_labeled_components`) and, if any component is actually dropped,
    the returned mask with disk-backed buffers instead of fresh in-RAM
    ones; *memmap_directory* is forwarded to both.
    """
    skeleton_bool = np.asanyarray(skeleton, dtype=bool)
    total_voxels = int(skeleton_bool.sum())
    if total_voxels == 0 or min_component_fraction <= 0.0:
        return skeleton_bool

    conn = _resolve_component_connectivity(skeleton_bool.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton_bool.ndim, conn)
    with _labeled_components(
        skeleton_bool, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as (labeled, n_components):
        if n_components == 0:
            return skeleton_bool

        min_component_size = int(np.ceil(min_component_fraction * total_voxels))
        component_sizes = (
            bincount_by_slab(labeled) if use_memmap else np.bincount(labeled.ravel())
        )
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

        if use_memmap:
            result = new_memmap_array(skeleton_bool.shape, bool, directory=memmap_directory)
            _combine_per_slice(result, lambda lb: np.isin(lb, keep_labels), labeled)
            return result
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
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> SkeletonConnectivityStats:
    """Connected-component sizes of *skeleton*, largest first.

    Labelling a full stack is not free, so a caller that only wants this
    occasionally in a log line should keep using
    :func:`log_skeleton_connectivity_stats`, which skips the computation
    entirely when INFO logging is off. A caller that needs the numbers
    themselves -- the settings optimiser scoring a cleanup candidate -- calls
    this directly.

    *use_memmap* labels into a disk-backed array (see
    :func:`_labeled_components`): scipy's own label array for a volume of
    2**31 voxels or more is int64, eight times the skeleton's own size.
    """
    # `astype` copies a whole volume even when it is already boolean.
    skeleton_bool = np.asanyarray(skeleton, dtype=bool)
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
    with _labeled_components(
        skeleton_bool, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as (labeled, n_components):
        # Only foreground voxels carry a component label, so counting those is
        # the same tally as counting the whole volume -- minus the background
        # at index 0, which is zeroed here anyway. On a sparse skeleton that is
        # thousands of voxels rather than hundreds of millions.
        component_sizes = (
            bincount_by_slab(labeled, where=skeleton_bool, minlength=n_components + 1)
            if use_memmap
            else np.bincount(labeled[skeleton_bool], minlength=n_components + 1)
        )
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
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> None:
    """Log concise connectivity diagnostics for a 2D/3D skeleton.

    Diagnostics only, so it does nothing when nothing is listening -- labelling
    a full stack is not free, and a caller that has turned INFO off has said it
    does not want this. *use_memmap* and *memmap_directory* are forwarded to
    :func:`compute_skeleton_connectivity_stats`.
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    stats = compute_skeleton_connectivity_stats(
        skeleton,
        component_connectivity,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
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


def _combine_per_slice(out: np.ndarray, op, *arrays: np.ndarray) -> None:
    """Write ``op(*slab)`` into ``out[slab]`` for every slab of whole z-slices
    (axis 0) -- *op* must be elementwise. A slab rather than one slice per
    call: a stack of thousands of small slices otherwise pays numpy's
    per-call overhead thousands of times.

    Shared by every caller that needs to combine same-shaped boolean
    arrays into a *pre-allocated* ``out`` (memmap or not) one slice at a
    time -- neither numpy's unary ``~`` nor its binary ``|``/``&`` write
    into a memmap unless explicitly told to, so a single whole-volume
    expression always allocates a fresh plain array regardless of how any
    of *arrays* is itself backed. Callers that don't need a memmap-backed
    result skip this and just write the whole-volume expression directly,
    since that stays the faster, simpler path when nothing here needs to
    be disk-backed.
    """
    step = slab_step(out.shape)
    for start in range(0, out.shape[0], step):
        out[start:start + step] = op(
            *(np.asarray(arr[start:start + step]) for arr in arrays)
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

    *use_memmap*, when True, writes the labelled background, the inverted
    mask fed to it, and this function's own returned mask to disk-backed
    buffers instead of fresh in-RAM ones -- this runs unconditionally on
    every load (see :func:`haemolynx.io.load._skeletonize_loaded_volume`), so
    it is the one full-volume, wider-than-boolean allocation a low-memory
    run cannot just disable. The inversion and the final combine are done
    one slice (axis 0) at a time via :func:`_combine_per_slice` rather than
    as a single ``~mask`` / ``mask | ...`` expression over the whole volume
    -- either one would otherwise allocate a fresh full-size plain array no
    matter how *mask* itself is backed. *memmap_directory* is forwarded to
    :func:`haemolynx.preprocessing.new_memmap_array`; both are ignored when
    *use_memmap* is False.

    The inverted mask is a transient buffer, scoped to a ``with
    temporary_memmap_array(...)`` block that also covers the labelling
    call -- not just allocated ahead of a separate ``try/finally``, so a
    failure anywhere in between (including while writing it slice by
    slice) still releases its backing file rather than leaking it.
    """
    mask = np.asanyarray(mask, dtype=bool)
    if use_memmap:
        with temporary_memmap_array(mask.shape, bool, directory=memmap_directory) as inverted:
            _combine_per_slice(inverted, lambda m: ~m, mask)
            background_labels, n_labels = _label_inverted_background(
                inverted, use_memmap=use_memmap, memmap_directory=memmap_directory
            )
    else:
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
    if use_memmap:
        result = new_memmap_array(mask.shape, bool, directory=memmap_directory)
        _combine_per_slice(result, lambda m, bl: m | ~reaches_edge[bl], mask, background_labels)
    else:
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


def _finest_axis_spacing(voxel_size_zyx, ndim: int) -> np.ndarray:
    """Per-axis spacing in units of the finest axis (1 there, more along a
    coarser one); all ones without *voxel_size_zyx*."""
    if voxel_size_zyx is None:
        return np.ones(ndim)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    return spacing / float(spacing.min())


def _physical_footprint(radius: int, relative_spacing: np.ndarray, *, norm: int) -> np.ndarray:
    """Offsets within *radius* finest-axis voxels of the centre, measured in
    physical units: a coarser axis reaches ``radius / its relative spacing``
    voxels, possibly none. *norm* 2 is a ball (Euclidean), 1 a diamond
    (city-block -- what iterating a 6-connected cross builds)."""
    reach = np.floor(float(radius) / relative_spacing + 1e-9).astype(int)
    spans = [np.arange(-r, r + 1) * s for r, s in zip(reach, relative_spacing)]
    grids = np.meshgrid(*spans, indexing="ij")
    if norm == 1:
        return sum(np.abs(g) for g in grids) <= radius + 1e-9
    return sum(g * g for g in grids) <= radius * radius + 1e-9


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
    voxel_size_zyx: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    within: np.ndarray | None = None,
) -> np.ndarray:
    """Fill small gaps in a binary mask.

    Every background voxel within *max_gap* voxels of the finest axis of any
    foreground voxel is set to foreground -- that is, dilation by a Euclidean
    ball of radius ``max_gap * min(voxel_size_zyx)`` microns, reaching fewer
    voxels along a coarser axis. Counted in voxels of every axis alike, a gap
    of 3 bridged 12 um in z but 3 um in-plane on a 0.5 x 0.5 x 2 um stack,
    fusing vessels stacked in z. Two ways to get there, and which is cheaper
    depends on the radius: dilating by that ball directly, or measuring an
    exact distance for every voxel in the volume and thresholding it. Small
    radii dilate; from :data:`MAX_BALL_DILATION_RADIUS` up the footprint grows
    faster than the transform does and the transform wins. Both return the
    same mask, so this only decides how long it takes. Without
    *voxel_size_zyx* every voxel is a unit cube.

    *use_memmap*, when True, runs either path one padded block at a time
    (:func:`map_blockwise`) into a new memmap in *memmap_directory*. Both
    only ever look *max_gap* voxels away, so a halo of *max_gap* makes every
    block exact. Handing scipy a memmap as the distance transform's
    ``distances=`` output is not enough on its own: scipy still builds its
    feature-transform indices, an ``np.indices`` grid and float64 copies of
    both internally -- roughly 70 bytes per voxel of plain RAM whatever the
    output array is.

    *within*, a boolean array of the same shape, is where the fill may add
    voxels: one outside it is added nowhere (the segmentation, so a gap
    through background stays a gap).
    """
    if max_gap <= 0:
        return binary_skeleton
    max_gap = int(max_gap)
    relative = _finest_axis_spacing(voxel_size_zyx, np.ndim(binary_skeleton))
    if use_memmap:
        grown = _bridge_gaps_blockwise(binary_skeleton, max_gap, relative, memmap_directory)
    elif max_gap <= MAX_BALL_DILATION_RADIUS:
        grown = binary_dilation(
            binary_skeleton, structure=_physical_footprint(max_gap, relative, norm=2)
        )
    # scipy's distance transform of an array with no background voxel is not
    # all-infinite but measured from a corner: nothing must dilate to nothing.
    elif not binary_skeleton.any():
        return binary_skeleton.copy()
    else:
        inverted = ~binary_skeleton
        distance = distance_transform_edt(inverted, sampling=relative)
        grown = binary_skeleton | ((distance <= max_gap) & inverted)
    return _additions_within(grown, binary_skeleton, within)


def _additions_within(grown: np.ndarray, original: np.ndarray, within: np.ndarray | None) -> np.ndarray:
    """*grown* with every voxel it added outside *within* taken back out; in
    place on a memmap, which is the step's own fresh output."""
    if within is None:
        return grown
    if isinstance(grown, np.memmap):
        _combine_per_slice(grown, lambda g, o, w: g & (o | w), grown, original, within)
        return grown
    return grown & (np.asarray(original, dtype=bool) | np.asarray(within, dtype=bool))


def _restore_thinned_away(before: np.ndarray, after: np.ndarray, reach_voxels: float) -> int:
    """Put back, into *after*, each 26-connected piece of the skeleton
    *before* that no voxel of *after* lies within *reach_voxels* of. Growth
    held to the mask can leave a piece as thin as its vessel, and Lee
    thinning erases some such solids outright (a 4 x 4 voxel rod, for one);
    a vessel is then better kept unjoined than lost. Returns how many."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    coords = argwhere_by_slab(before)
    if len(coords) == 0:
        return 0
    kept = argwhere_by_slab(after)
    if len(kept):
        distance, _ = cKDTree(kept).query(coords, distance_upper_bound=float(reach_voxels) + 1e-6)
        reached = np.isfinite(distance)
    else:
        reached = np.zeros(len(coords), dtype=bool)
    if reached.all():
        return 0
    pairs = cKDTree(coords).query_pairs(r=np.sqrt(3.0) + 1e-6, output_type="ndarray")
    n = len(coords)
    links = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_pieces, labels = connected_components(links, directed=False)
    lost = np.bincount(labels, weights=reached, minlength=n_pieces) == 0
    restore = coords[lost[labels]]
    after[tuple(restore.T)] = True
    return int(np.count_nonzero(lost))


def _bridge_gaps_blockwise(
    binary_skeleton: np.ndarray,
    max_gap: int,
    relative_spacing: np.ndarray,
    memmap_directory: str | Path | None,
) -> np.memmap:
    skeleton_bool = np.asanyarray(binary_skeleton, dtype=bool)
    result = new_memmap_array(skeleton_bool.shape, bool, directory=memmap_directory)
    if max_gap <= MAX_BALL_DILATION_RADIUS:
        ball = _physical_footprint(max_gap, relative_spacing, norm=2)
        return map_blockwise(
            skeleton_bool,
            lambda block: binary_dilation(block, structure=ball),
            result,
            halo=max_gap,
        )

    def within_gap(block: np.ndarray) -> np.ndarray:
        # scipy's distance transform of an array with no background voxel
        # at all is not all-infinite but garbage measured from a corner --
        # and a block with no skeleton in it is the common case here.
        if not block.any():
            return block.copy()
        inverted = ~block
        distance = distance_transform_edt(inverted, sampling=relative_spacing)
        return block | ((distance <= max_gap) & inverted)

    return map_blockwise(skeleton_bool, within_gap, result, halo=max_gap)


def close_binary_mask(
    binary: np.ndarray,
    radius: int = 2,
    *,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    within: np.ndarray | None = None,
) -> np.ndarray:
    """Morphologically close a binary mask to seal small gaps.

    Applies binary dilation followed by binary erosion (closing) using a
    diamond-shaped structuring element of the given *radius* in voxels of the
    finest axis -- ``radius * min(voxel_size_zyx)`` microns, reaching fewer
    voxels along a coarser axis. On isotropic voxels that is *radius*
    iterations of a 6-connected cross, as before; counted in voxels of every
    axis alike, a radius of 2 closed 8 um gaps in z against 2 um in-plane on
    a 0.5 x 0.5 x 2 um stack. Unlike plain dilation, closing does not
    permanently expand object boundaries — it only fills concavities and
    bridges narrow gaps smaller than the structuring element.

    Parameters
    ----------
    binary:
        Input boolean array (3D, canonical ``(z, y, x)``, or 2D).
    radius:
        Reach of the closing in voxels of the finest axis. Larger values
        bridge wider gaps but risk merging genuinely distinct structures.
    voxel_size_zyx:
        Per-axis spacing of *binary*; ``None`` treats every voxel as a cube.
    use_memmap, memmap_directory:
        When *use_memmap* is True, close one padded block at a time
        (:func:`map_blockwise`) into a new memmap in *memmap_directory*
        instead of dilating, then eroding, the whole volume in RAM. A voxel's
        closed value depends only on the input within ``2 * radius`` voxels
        of it (``radius`` for the dilation, ``radius`` more for the erosion
        that reads it), so that halo makes every block exact.
    within:
        Where the closing may add voxels, as :func:`bridge_gaps` takes it;
        ``None`` places no limit.
    """
    from scipy.ndimage import binary_closing

    if radius <= 0:
        return binary
    radius = int(radius)
    struct = _physical_footprint(
        radius, _finest_axis_spacing(voxel_size_zyx, binary.ndim), norm=1
    )
    if use_memmap:
        binary_bool = np.asanyarray(binary, dtype=bool)
        result = new_memmap_array(binary_bool.shape, bool, directory=memmap_directory)
        closed = map_blockwise(
            binary_bool,
            lambda block: binary_closing(block, structure=struct),
            result,
            halo=2 * radius,
        )
        return _additions_within(closed, binary_bool, within)
    return _additions_within(
        binary_closing(binary.astype(bool), structure=struct), binary, within
    )


def skeletonize_volume(img: np.ndarray) -> np.ndarray:
    """Skeletonize a 2D/3D binary volume (Lee method; replaces legacy skeletonize_3d)."""
    return skeletonize(img.astype(bool), method="lee")


def skeletonize_3d(img: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`skeletonize_volume`."""
    return skeletonize_volume(img)


def _skeletonize_component_into(
    component_mask: np.ndarray,
    out: np.ndarray,
    *,
    tile_large_components: bool,
    tile_max_voxels: int,
    tile_halo_voxels: int,
) -> None:
    """Skeletonize *component_mask*, OR-ing the result into *out* (same
    shape as *component_mask* -- typically a view into a larger memmap,
    e.g. ``result[bbox]``).

    With *tile_large_components* False, or *component_mask* already at or
    below *tile_max_voxels*, this is exactly
    ``out |= skeletonize(component_mask, method="lee")`` -- no tiling.

    Otherwise, splits *component_mask* along axis 0 (Z) into slabs no
    larger than *tile_max_voxels*, each padded by *tile_halo_voxels* of
    neighbouring context on either side before skeletonizing, with only
    the un-padded core written into *out*. Unlike
    :func:`skeletonize_by_component`'s own per-component split, this is
    **not** provably exact -- Lee thinning is iterative, with no hard
    bound on how far a boundary effect could in principle propagate, so
    the halo is a heuristic margin, not a proof. A halo generously larger
    than the largest vessel radius in the data keeps results
    indistinguishable from the monolithic computation in practice.

    At the component's own true Z edges, the padding window clips to the
    real boundary rather than extending past it -- the same "outside is
    background" treatment ``skimage.morphology.skeletonize``'s own
    internal zero-padding already applies at the outer boundary of a
    monolithic call, so an edge slab sees exactly what the monolithic call
    would have seen there.

    Works for any *component_mask* dimensionality, not just 3D: "axis 0"
    is whatever the caller's own first axis means (Z for this pipeline's
    canonical volumes), and the per-tile voxel budget is derived from
    ``size // depth`` rather than unpacking a fixed number of axes.
    """
    if not tile_large_components or component_mask.size <= tile_max_voxels:
        out |= skeletonize(component_mask, method="lee")
        return
    depth = component_mask.shape[0]
    per_slice_voxels = max(1, component_mask.size // max(1, depth))
    tile_depth = max(1, tile_max_voxels // per_slice_voxels)
    for start in range(0, depth, tile_depth):
        end = min(start + tile_depth, depth)
        pad_start = max(0, start - tile_halo_voxels)
        pad_end = min(depth, end + tile_halo_voxels)
        padded = skeletonize(component_mask[pad_start:pad_end], method="lee")
        core_lo = start - pad_start
        out[start:end] |= padded[core_lo : core_lo + (end - start)]


def skeletonize_by_component(
    mask: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
) -> np.ndarray:
    """Lee-skeletonize *mask*, one connected component at a time, in each
    component's own tight bounding box, instead of running
    ``skimage.morphology.skeletonize`` on the whole volume in one call.

    With *use_memmap* False (the default), this is exactly
    ``skimage.morphology.skeletonize(mask, method="lee")`` -- that call is
    made directly, unchanged, so nothing differs for anyone not using
    ``use_memmap_loading``.

    With *use_memmap* True: Lee thinning's simple-point classification for
    a voxel depends only on information that can propagate through
    connected foreground, so a component with no foreground path to any
    other component is provably unaffected by anything outside itself --
    this is not an approximation. Splitting first and skeletonizing each
    component's own (typically far smaller) bounding box, with any other,
    spatially-nearby-but-unconnected component's voxels inside that box
    masked out, gives skimage's own algorithm a small in-RAM array to work
    with regardless of how large the whole volume is, and writes each
    result into a shared disk-backed output. Only actually reduces peak
    memory when *mask* splits into components meaningfully smaller than
    the whole volume -- one single network filling most of the volume gets
    exactly one "component" the same size as before, no worse, just the
    modest extra cost of labelling first.

    Labels with the widest possible connectivity for *mask*'s
    dimensionality (``generate_binary_structure(mask.ndim, mask.ndim)``)
    regardless of any other connectivity setting elsewhere in this
    pipeline: two components must be independent under skeletonize's own
    (full-neighbourhood) notion of adjacency for the exactness argument
    above to hold, and a narrower connectivity here could split two
    components that skeletonize's own algorithm would still treat as
    touching, corrupting the result rather than just failing to help.
    *memmap_directory* is forwarded to :func:`new_memmap_array`; both are
    ignored when *use_memmap* is False. The label array is a transient
    buffer, scoped to a ``with temporary_memmap_array(...)`` block that
    also covers the per-component loop -- released whether or not that
    loop raises partway through, rather than leaking on a failure. Each
    component's own boolean mask (``labeled[bbox] == component_id``) is
    a second transient memmap of its own, filled one slice at a time via
    :func:`_combine_per_slice` rather than as a single whole-bbox
    expression -- otherwise that comparison would allocate a fresh,
    plain-RAM array the size of the bbox regardless of how *mask* itself
    is backed, which for the single-giant-component case is (almost) the
    whole volume: the exact case *tile_large_components* below exists to
    shrink, undone if reaching it still costs one full-volume allocation.

    *tile_large_components*, *tile_max_voxels* and *tile_halo_voxels* are
    forwarded to :func:`_skeletonize_component_into` for each component in
    turn -- see its own docstring. Defaulted so that leaving them alone
    reproduces this function's own pre-tiling behaviour exactly: a
    component larger than the whole volume normally is is skeletonized as
    one piece regardless, same as before tiling existed.
    """
    if not use_memmap:
        return skeletonize(np.asanyarray(mask, dtype=bool), method="lee")

    mask = np.asanyarray(mask, dtype=bool)
    footprint = generate_binary_structure(mask.ndim, mask.ndim)
    result = new_memmap_array(mask.shape, bool, directory=memmap_directory)
    with temporary_memmap_array(mask.shape, np.int32, directory=memmap_directory) as labeled:
        n_labels = label(mask, footprint, output=labeled)
        if n_labels == 0:
            return result
        for component_id, bbox in enumerate(find_objects(labeled, max_label=n_labels), start=1):
            if bbox is None:
                continue
            labeled_component = labeled[bbox]
            if labeled_component.size <= LOW_MEMORY_BLOCK_VOXELS:
                # Small enough to compare in RAM: a temp file per component
                # made a noisy mask with thousands of specks crawl.
                _skeletonize_component_into(
                    np.asarray(labeled_component) == component_id,
                    result[bbox],
                    tile_large_components=tile_large_components,
                    tile_max_voxels=tile_max_voxels,
                    tile_halo_voxels=tile_halo_voxels,
                )
                continue
            # Not `labeled_component == component_id`: that comparison
            # always allocates a fresh, plain-RAM array the size of the
            # bbox, regardless of how `labeled` itself is backed -- for
            # the single-giant-component case Tier 2 tiling exists for,
            # that bbox *is* (almost) the whole volume, defeating memmap
            # right before the one call tiling is supposed to shrink.
            with temporary_memmap_array(
                labeled_component.shape, bool, directory=memmap_directory
            ) as component_mask:
                _combine_per_slice(
                    component_mask, lambda lb: lb == component_id, labeled_component
                )
                _skeletonize_component_into(
                    component_mask,
                    result[bbox],
                    tile_large_components=tile_large_components,
                    tile_max_voxels=tile_max_voxels,
                    tile_halo_voxels=tile_halo_voxels,
                )
    return result


def _draw_line_3d(array: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Set voxels along the straight line from *start* to *end* to True."""
    n_steps = max(int(np.linalg.norm(end.astype(float) - start.astype(float))) + 1, 2)
    for t in np.linspace(0.0, 1.0, n_steps):
        pt = np.round(start + t * (end - start)).astype(int)
        pt = np.clip(pt, 0, np.array(array.shape) - 1)
        array[tuple(pt)] = True


def _select_hub_centres(
    peak_coords: np.ndarray,
    peak_density: np.ndarray,
    hub_min_spacing: float,
    relative_spacing: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Density peaks, densest first, keeping each only if it is at least
    *hub_min_spacing* from every hub already kept -- measured in voxels of
    the finest axis, each axis weighted by *relative_spacing* (see
    :func:`_finest_axis_spacing`).

    Each kept hub is filed in a grid of cells a hair wider than
    *hub_min_spacing*, so a peak is checked only against the hubs in its own
    cell and the ones next to it: any hub nearer than the spacing is in one
    of those. Checking every hub kept so far made this quadratic, and a
    skeleton's local density takes few distinct values, so its maxima tie
    across whole plateaus of peaks -- on a 2x grid of the E14.5 stack, bundle
    refinement took 24 minutes of a 29-minute optimiser run. Distances are
    the same arithmetic on the same integer coordinates as before, so the
    hubs kept are exactly the same.
    """
    weights = np.ones(peak_coords.shape[1]) if relative_spacing is None else relative_spacing
    order = np.argsort(peak_density)[::-1]
    selected_hubs: list[np.ndarray] = []
    if not hub_min_spacing > 0:
        return [peak_coords[idx] for idx in order]
    cell = float(hub_min_spacing) * (1.0 + 1e-9)
    cell_of = np.floor(peak_coords * np.asarray(weights, dtype=float) / cell).astype(np.int64)
    neighbours = np.array(np.meshgrid(*([(-1, 0, 1)] * peak_coords.shape[1]), indexing="ij"))
    neighbours = neighbours.reshape(peak_coords.shape[1], -1).T
    kept_in_cell: dict[tuple[int, ...], list[np.ndarray]] = {}
    for idx in order:
        candidate = peak_coords[idx]
        home = cell_of[idx]
        if all(
            np.linalg.norm((candidate - existing) * weights) >= hub_min_spacing
            for offset in neighbours
            for existing in kept_in_cell.get(tuple((home + offset).tolist()), ())
        ):
            selected_hubs.append(candidate)
            kept_in_cell.setdefault(tuple(home.tolist()), []).append(candidate)
    return selected_hubs


def _odd_at_least_three(value: float) -> int:
    return max(3, 2 * int(round((float(value) - 1.0) / 2.0)) + 1)


def bundle_scan_window(
    scan_size: int | tuple[int, ...], voxel_size_zyx, ndim: int = 3
) -> tuple[int, ...]:
    """The per-axis window :func:`skeletonize_voxel_bundles_into_paths` scans
    with: an int *scan_size* is voxels of the finest axis, the same physical
    width on every axis (odd, at least 3); a tuple is taken per axis as given.
    Anything else that measures local density "at a scan size" -- the
    optimiser's candidates and its cost -- uses this window, not
    ``size=scan_size``.
    """
    if isinstance(scan_size, (int, np.integer)):
        relative = _finest_axis_spacing(voxel_size_zyx, ndim)
        return tuple(_odd_at_least_three(int(scan_size) / r) for r in relative)
    if len(scan_size) != ndim:
        raise ValueError(f"scan_size must have {ndim} dimensions, got {len(scan_size)}")
    return tuple(max(3, int(s)) for s in scan_size)


def _draw_hub_links(
    result: np.ndarray,
    center: np.ndarray,
    boundary_points: np.ndarray,
    max_connections_per_hub: int,
) -> None:
    """Link *center* to every branch leaving the hub, at most
    *max_connections_per_hub* of them, farthest first.

    A branch is one 26-connected group of *boundary_points* (the skeleton
    voxels in the shell just outside the hub's window), and its link goes to
    the group's farthest point. Links used to be kept one per sign of the
    direction vector -- one per octant -- so two branches leaving in the same
    octant got one link between them, and the other, its voxels inside the
    window already erased, was left cut off.
    """
    for target in _hub_link_targets(center, boundary_points, max_connections_per_hub):
        _draw_line_3d(result, center, target)


def _hub_link_targets(
    center: np.ndarray,
    boundary_points: np.ndarray,
    max_connections_per_hub: int,
    accept_link=None,
) -> list[np.ndarray]:
    """The voxels :func:`_draw_hub_links` links *center* to, farthest branch
    first; with *accept_link* ``(center, target) -> bool``, only the links it
    accepts, still at most *max_connections_per_hub* of them."""
    from scipy.spatial import cKDTree

    points = np.asarray(boundary_points)
    points = points[np.any(points != np.asarray(center), axis=1)]
    if points.shape[0] == 0:
        return []
    parent = list(range(points.shape[0]))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # sqrt(3): 26-adjacency, the far corner of the 3x3x3 block and no further.
    for i, j in cKDTree(points).query_pairs(r=1.7321):
        parent[find(i)] = find(j)
    farthest: dict[int, tuple[float, int]] = {}
    for i, point in enumerate(points):
        vec = point - center
        dist2 = float(np.dot(vec, vec))
        root = find(i)
        if root not in farthest or dist2 > farthest[root][0]:
            farthest[root] = (dist2, i)

    branches = sorted(farthest.values(), key=lambda item: (-item[0], item[1]))
    if accept_link is not None:
        branches = [item for item in branches if accept_link(center, points[item[1]])]
    if len(branches) > max_connections_per_hub:
        logger.debug(
            "Bundle hub at %s has %d branches; linking the %d farthest.",
            tuple(int(v) for v in center),
            len(branches),
            max_connections_per_hub,
        )
    return [points[index] for _, index in branches[:max_connections_per_hub]]


def _collapse_hubs(
    result: np.ndarray,
    mask: np.ndarray,
    dense_volume: np.ndarray,
    selected_hubs: list[np.ndarray],
    scan: tuple[int, ...],
    max_connections_per_hub: int,
    accept_link=None,
) -> None:
    """Collapse each hub's dense window of *result* to one centre voxel,
    relinked to the skeleton just outside it, in place.

    With *accept_link* ``(center, target) -> bool`` (voxel indices), a link
    it refuses is not drawn, and a hub left fewer than two links is not
    collapsed at all: erasing its window would cut off the branches it could
    not relink.

    Each hub reads and writes only its own scan window plus one voxel -- the
    boundary shell's dilation reaches no further -- so this works in that
    box rather than on a volume-sized array per hub: the same shell and the
    same boundary points in the same (C) order, hence the same links, as the
    whole-volume version this replaced, which allocated, dilated and scanned
    the entire volume once for every hub.
    """
    shape = np.array(mask.shape)
    half_window = np.array(scan) // 2
    structure = generate_binary_structure(mask.ndim, 1)

    for hub in selected_hubs:
        lo = np.maximum(hub - half_window, 0)
        hi = np.minimum(hub + half_window + 1, shape)
        slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(mask.ndim))
        box_lo = np.maximum(lo - 1, 0)
        box_hi = np.minimum(hi + 1, shape)
        box = tuple(slice(int(box_lo[d]), int(box_hi[d])) for d in range(mask.ndim))
        window_in_box = tuple(
            slice(int(lo[d] - box_lo[d]), int(hi[d] - box_lo[d])) for d in range(mask.ndim)
        )
        local_dense = np.zeros(tuple(int(v) for v in box_hi - box_lo), dtype=bool)
        local_dense[window_in_box] = dense_volume[slices]
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
        result_box = result[box]
        boundary_points = np.argwhere(np.asarray(result_box) & shell) + box_lo

        if accept_link is not None:
            targets = _hub_link_targets(
                center, boundary_points, max_connections_per_hub, accept_link
            )
            if len(targets) < 2:
                continue
            result_box[local_dense] = False
            result[center_t] = True
            for target in targets:
                _draw_line_3d(result, center, target)
            continue

        result_box[local_dense] = False
        result[center_t] = True

        if boundary_points.size == 0:
            continue
        _draw_hub_links(result, center, boundary_points, max_connections_per_hub)


def skeletonize_voxel_bundles_into_paths(
    binary_mask: np.ndarray,
    scan_size: int | tuple[int, ...] = 9,
    density_fraction: float = 0.35,
    max_connections_per_hub: int = 8,
    hub_min_spacing: int | None = None,
    *,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
    mask_support: MaskSupport | None = None,
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
        Sliding window size used to estimate local density: a per-axis tuple
        of voxels, or an int in voxels of the finest axis -- the same
        physical width on every axis, so fewer voxels (odd, at least 3) along
        a coarser one. Counted as voxels on every axis, a window of 9 spanned
        18 um in z but 4.5 um in-plane on a 0.5 x 0.5 x 2 um stack, and
        vessels stacked in z read as one bundle. Odd values are preferred.
    density_fraction:
        Mark a location as dense when its local foreground fraction is at least
        this threshold (0.0-1.0).
    max_connections_per_hub:
        Max number of directional in/out links reconnected to each hub.
    hub_min_spacing:
        Minimum spacing between selected hub centers, in voxels of the finest
        axis, measured physically. Defaults to about half the window's
        physical width.
    voxel_size_zyx:
        Per-axis spacing of *binary_mask*; ``None`` treats every voxel as a
        cube.
    use_memmap, memmap_directory, tile_large_components, tile_max_voxels, tile_halo_voxels:
        With *use_memmap* True, see
        :func:`_skeletonize_voxel_bundles_into_paths_low_memory` -- the same
        steps without any whole-volume plain-RAM array. The tiling settings
        are forwarded to its two skeletonize calls.
    mask_support:
        A :class:`~haemolynx.preprocessing.bridge_mask_support.MaskSupport`
        over the segmentation: a hub's link is drawn only where it accepts
        the straight line, and a hub left fewer than two is not collapsed.
        ``None`` links every branch, as before.
    """
    mask = np.asanyarray(binary_mask, dtype=bool) if use_memmap else binary_mask.astype(bool)
    if not mask.any():
        return mask

    relative = _finest_axis_spacing(voxel_size_zyx, mask.ndim)
    scan = bundle_scan_window(scan_size, voxel_size_zyx, mask.ndim)
    density_fraction = float(np.clip(density_fraction, 0.0, 1.0))
    max_connections_per_hub = max(1, int(max_connections_per_hub))
    if hub_min_spacing is None:
        hub_min_spacing = max(1, int(min(s * r for s, r in zip(scan, relative)) / 2))
    accept_link = _hub_link_gate(mask_support)

    if use_memmap:
        return _skeletonize_voxel_bundles_into_paths_low_memory(
            mask,
            scan,
            density_fraction,
            max_connections_per_hub,
            hub_min_spacing,
            relative_spacing=relative,
            memmap_directory=memmap_directory,
            tile_large_components=tile_large_components,
            tile_max_voxels=tile_max_voxels,
            tile_halo_voxels=tile_halo_voxels,
            accept_link=accept_link,
        )

    base_skeleton = skeletonize_volume(mask)
    density = uniform_filter(mask.astype(np.float32), size=scan, mode="constant")
    dense_volume = density >= density_fraction
    if not dense_volume.any():
        return base_skeleton.astype(bool)

    peak_map = dense_volume & (density == maximum_filter(density, size=scan, mode="nearest"))
    peak_coords = np.argwhere(peak_map)
    if peak_coords.size == 0:
        return base_skeleton.astype(bool)

    selected_hubs = _select_hub_centres(
        peak_coords, density[tuple(peak_coords.T)], hub_min_spacing, relative
    )

    result = base_skeleton.astype(bool).copy()
    _collapse_hubs(
        result, mask, dense_volume, selected_hubs, scan, max_connections_per_hub, accept_link
    )

    return skeletonize_volume(result.astype(bool)).astype(bool)


def _hub_link_gate(mask_support: MaskSupport | None):
    """``(center, target) -> bool`` for :func:`_collapse_hubs`: whether
    *mask_support* accepts the straight link between two voxels."""
    if mask_support is None:
        return None
    spacing = np.asarray(mask_support.voxel_size_zyx, dtype=float)

    def accept_link(center: np.ndarray, target: np.ndarray) -> bool:
        return mask_support.accepts(np.vstack([center, target]).astype(float) * spacing)

    return accept_link


def _skeletonize_voxel_bundles_into_paths_low_memory(
    mask: np.ndarray,
    scan: tuple[int, ...],
    density_fraction: float,
    max_connections_per_hub: int,
    hub_min_spacing: float,
    *,
    relative_spacing: np.ndarray | None = None,
    memmap_directory: str | Path | None,
    tile_large_components: bool,
    tile_max_voxels: int,
    tile_halo_voxels: int,
    accept_link=None,
) -> np.ndarray:
    """:func:`skeletonize_voxel_bundles_into_paths` without a whole-volume
    array in RAM.

    - Both skeletonize calls go through :func:`skeletonize_by_component`
      (disk-backed, optionally tiled) rather than one plain Lee call on the
      whole volume.
    - Density and its peaks are computed one padded block at a time. A
      voxel's density reads the mask within half a scan window of it, and
      whether it is a peak reads the density within half a window more, so
      a halo of two half-windows reproduces both on every block's core.
      scipy's running-mean filter accumulates from the start of each line it
      is handed, which a block starts somewhere else, so a float32 density
      can differ from a whole-volume call in its last bit -- a threshold or
      peak tie landing exactly on such a difference could come out the
      other way.
    - Peaks are put back in whole-volume (C) order before being ranked, so
      hubs are chosen from exactly the list a whole-volume ``np.argwhere``
      would give.
    - Each hub touches only its own scan window plus one voxel (the
      boundary-shell dilation's reach), instead of a fresh full-volume
      ``zeros_like`` array per hub.
    """
    def skeletonize_all(volume: np.ndarray) -> np.ndarray:
        return skeletonize_by_component(
            volume,
            use_memmap=True,
            memmap_directory=memmap_directory,
            tile_large_components=tile_large_components,
            tile_max_voxels=tile_max_voxels,
            tile_halo_voxels=tile_halo_voxels,
        )

    base_skeleton = skeletonize_all(mask)
    half = tuple(int(s) // 2 for s in scan)
    with temporary_memmap_array(mask.shape, bool, directory=memmap_directory) as dense_volume:
        peak_coord_parts: list[np.ndarray] = []
        peak_density_parts: list[np.ndarray] = []
        any_dense = False
        for padded, inner, core in iter_blocks(mask.shape, halo=2 * max(half)):
            block = np.asarray(mask[padded])
            density = uniform_filter(block.astype(np.float32), size=scan, mode="constant")
            dense = density >= density_fraction
            dense_volume[core] = dense[inner]
            if not dense[inner].any():
                continue
            any_dense = True
            peaks = dense & (density == maximum_filter(density, size=scan, mode="nearest"))
            local = np.argwhere(peaks[inner])
            if local.size:
                peak_density_parts.append(density[inner][tuple(local.T)])
                peak_coord_parts.append(local + np.array([s.start for s in core]))

        if not any_dense or not peak_coord_parts:
            return base_skeleton

        peak_coords = np.concatenate(peak_coord_parts)
        peak_density = np.concatenate(peak_density_parts)
        c_order = np.lexsort(peak_coords.T[::-1])
        selected_hubs = _select_hub_centres(
            peak_coords[c_order], peak_density[c_order], hub_min_spacing, relative_spacing
        )

        result = base_skeleton
        _collapse_hubs(
            result, mask, dense_volume, selected_hubs, scan, max_connections_per_hub,
            accept_link,
        )

    final = skeletonize_all(result)
    if final is not result:
        release_memmap_array(result)
    return final


#: Padding (voxels) added around a candidate bridge's own bounding box when
#: cropping the segmentation mask for :func:`_bridge_path_through_mask` --
#: enough slack for the router to curve around a nearby obstruction without
#: needing the full-volume windowed-cost-budget machinery
#: `graph/reconnect.py` uses for its much larger search space (a skeleton
#: bridge is already bounded by ``max_bridge_distance``).
_BRIDGE_MASK_WINDOW_PAD = 3


def _bridge_path_through_mask(
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    relative_spacing: np.ndarray | None = None,
) -> np.ndarray | None:
    """A* path from *start* to *end* that prefers staying inside *mask*.

    Same cost-field convention as `graph.reconnect`'s own skeleton-proximity
    router (``1 + distance_transform_edt(~reference) ** 2``), but built from
    the segmentation mask instead of the skeleton, and only over a small
    local window around the two endpoints. Distances and steps are physical
    (*relative_spacing*, see :func:`_finest_axis_spacing`): costed per voxel,
    a step along a coarse z covered four times the tissue for the same price,
    and bridges went the long way round through it. Returns ``None`` (never
    raises) on any failure -- a shape mismatch, an out-of-bounds window, or a
    routing exception -- so the caller can fall back to a straight line.
    """
    try:
        from skimage.graph import MCP_Geometric

        relative = np.ones(3) if relative_spacing is None else np.asarray(relative_spacing, float)

        lo = np.maximum(np.minimum(start, end) - _BRIDGE_MASK_WINDOW_PAD, 0)
        hi = np.minimum(
            np.maximum(start, end) + _BRIDGE_MASK_WINDOW_PAD + 1, np.array(mask.shape)
        )
        if np.any(hi <= lo):
            return None
        window = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        cost = 1 + distance_transform_edt(~window.astype(bool), sampling=tuple(relative)) ** 2
        start_local = tuple(int(v) for v in (start - lo))
        end_local = tuple(int(v) for v in (end - lo))
        router = MCP_Geometric(cost, fully_connected=True, sampling=tuple(relative))
        router.find_costs([start_local], [end_local])
        path_coords = router.traceback(end_local)
        if not path_coords:
            return None
        return np.asarray(path_coords, dtype=int) + lo
    except Exception:
        logger.debug("Mask-weighted bridge routing failed; using a straight line.", exc_info=True)
        return None


#: How many skeleton steps back from a tip its direction is read over: enough
#: to see past the last voxel's staircase, few enough to stay on the branch.
_TIP_TANGENT_STEPS = 4

#: The default for ``min_facing_cosine``: a tip must point within this much of
#: the gap it would bridge (cos 60 degrees). Loose, because a tip's direction
#: is read from a few voxels.
DEFAULT_BRIDGE_MIN_FACING_COSINE = 0.5


def _bridge_metric(
    ndim: int,
    voxel_size_zyx: tuple[float, float, float] | None,
    z_distance_weight: float,
    max_bridge_distance: float,
) -> tuple[np.ndarray, float]:
    """``(per-axis weights, reach)`` of :func:`connect_skeleton_components`:
    a voxel index times *weights* is its position in (z-weighted) microns, and
    *reach* is how far a bridge may go in that space."""
    spacing = np.asarray(voxel_size_zyx if voxel_size_zyx is not None else (1.0,) * ndim, dtype=float)
    weights = spacing * np.array([float(z_distance_weight), 1.0, 1.0])
    return weights, float(max_bridge_distance) * float(spacing.min())


def _neighbour_lists(index_tree, coords: np.ndarray) -> list[list[int]]:
    """Each skeleton voxel's 26-neighbours, as indices into *coords*."""
    # sqrt(3) reaches the far corner of the 3x3x3 block and nothing beyond it.
    neighbours = index_tree.query_ball_point(coords, r=1.7321)
    return [[j for j in found if j != i] for i, found in enumerate(neighbours)]


def _tip_directions(
    coords: np.ndarray, neighbours: list[list[int]], weights: np.ndarray
) -> dict[int, np.ndarray]:
    """``{tip index: unit direction}`` for every tip that has one.

    A tip is a voxel with exactly one neighbour -- a branch end. Its direction
    points out of the branch: from the voxel up to :data:`_TIP_TANGENT_STEPS`
    steps back along it to the tip, measured in *weights* (physical, z
    weighted) space. A voxel with no neighbour is a speck with no direction.
    """
    directions: dict[int, np.ndarray] = {}
    for tip, around in enumerate(neighbours):
        if len(around) != 1:
            continue
        previous, current = tip, around[0]
        for _ in range(_TIP_TANGENT_STEPS - 1):
            onward = [j for j in neighbours[current] if j != previous]
            if len(onward) != 1:
                break
            previous, current = current, onward[0]
        vector = (coords[tip] - coords[current]) * weights
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            directions[tip] = vector / norm
    return directions


@dataclass(frozen=True)
class _TipGeometry:
    """A skeleton's voxels in (z-weighted) microns, a KD-tree over them, and
    each branch end's direction (:func:`_tip_directions`)."""

    weighted: np.ndarray
    tree: Any
    directions: dict[int, np.ndarray]

    @classmethod
    def of(cls, coords: np.ndarray, weights: np.ndarray) -> "_TipGeometry":
        from scipy.spatial import cKDTree

        weighted = coords * weights
        neighbours = _neighbour_lists(cKDTree(coords), coords)
        return cls(weighted, cKDTree(weighted), _tip_directions(coords, neighbours, weights))


#: Targets a tip's bridge is put to the mask test for before the tip is given
#: up on: the nearest few voxels ahead, not every one within reach.
_MAX_MASK_TESTED_TARGETS_PER_TIP = 8


def _straight_voxel_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The voxels :func:`_draw_line_3d` sets between *start* and *end*."""
    n_steps = max(int(np.linalg.norm(end.astype(float) - start.astype(float))) + 1, 2)
    t = np.linspace(0.0, 1.0, n_steps)[:, None]
    return np.round(start + t * (end - start)).astype(int)


class _SkeletonBridgeGate:
    """The mask test :func:`connect_skeleton_components` puts each bridge
    to: its path (straight, or routed through the mask -- whichever
    *routed_first* tries first, then the other) must pass *support*'s
    background test and must not run beside the skeleton, or beside a bridge
    drawn before it, in the same lumen. The path it accepted is the one
    drawn."""

    def __init__(self, support: MaskSupport, coords: np.ndarray, *, routed_first: bool):
        self.support = support
        self.spacing = np.asarray(support.voxel_size_zyx, dtype=float)
        self.relative = self.spacing / float(self.spacing.min())
        self.points = coords * self.spacing
        self.routed_first = routed_first
        self._tree = None
        self.paths: dict[tuple[int, int], np.ndarray] = {}
        self.drawn: list[np.ndarray] = []
        self.refused_background = 0
        self.refused_duplicate = 0

    def accept(self, coords: np.ndarray, tip: int, end: int) -> bool:
        start, stop = coords[tip], coords[end]
        straight = _straight_voxel_path(start, stop)

        def routed():
            return _bridge_path_through_mask(self.support.mask, start, stop, self.relative)

        order = (routed, lambda: straight) if self.routed_first else (lambda: straight, routed)
        for make in order:
            path = make()
            if path is not None and self.support.accepts(path * self.spacing):
                break
        else:
            self.refused_background += 1
            return False
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        if self.support.shadows(path * self.spacing, self._tree, self.points):
            self.refused_duplicate += 1
            return False
        self.paths[(tip, end)] = path
        return True

    def path_to_draw(self, tip: int, end: int) -> np.ndarray | None:
        """The accepted path, or ``None`` when it runs beside a bridge
        already drawn in the same lumen."""
        path = self.paths[(tip, end)]
        physical = path * self.spacing
        if self.drawn:
            reach = 2.0 * float(np.max(self.support.radius(physical))) + 1.0
            lo, hi = physical.min(axis=0) - reach, physical.max(axis=0) + reach
            nearby = [
                drawn for drawn in self.drawn
                if np.all(drawn.max(axis=0) >= lo) and np.all(drawn.min(axis=0) <= hi)
            ]
            if nearby and self.support.shadows(physical, None, np.vstack(nearby)):
                self.refused_duplicate += 1
                return None
        self.drawn.append(physical)
        return path


def _bridge_candidates(
    tips: _TipGeometry,
    labels: np.ndarray,
    reach: float,
    min_facing_cosine: float,
    *,
    source: np.ndarray | None = None,
    target: np.ndarray | None = None,
    accept=None,
) -> tuple[list[tuple[float, int, int]], int]:
    """The bridges :func:`connect_skeleton_components` may draw, shortest
    first, as ``(length, tip, target)`` indices into the skeleton's voxels,
    and how many tips had another component within reach but none ahead.

    Each tip goes to its nearest voxel of another component (*labels*) that is
    within *reach* and ahead of it, and that, if itself a tip, faces back.
    *source* and *target*, boolean over the voxels, restrict which tips may
    start a bridge and which voxels it may end on. *accept* ``(tip, target)
    -> bool``, when given, may refuse a target, and the tip then tries its
    next nearest (up to :data:`_MAX_MASK_TESTED_TARGETS_PER_TIP` of them); a
    tip whose every target was refused is not counted as declined.
    """
    weighted, directions = tips.weighted, tips.directions
    candidates: list[tuple[float, int, int]] = []
    declined = 0
    for tip, heading in directions.items():
        if source is not None and not source[tip]:
            continue
        nearby = np.asarray(tips.tree.query_ball_point(weighted[tip], r=reach * (1 + 1e-9) + 1e-9), dtype=int)
        nearby = nearby[labels[nearby] != labels[tip]]
        if target is not None:
            nearby = nearby[target[nearby]]
        if nearby.size == 0:
            continue
        vectors = weighted[nearby] - weighted[tip]
        distances = np.linalg.norm(vectors, axis=1)
        # Elementwise, not `vectors @ heading`: see thick_vessels._matvec_3x3.
        along = vectors[:, 0] * heading[0] + vectors[:, 1] * heading[1] + vectors[:, 2] * heading[2]
        facing = along >= float(min_facing_cosine) * distances
        tested = 0
        for end in nearby[facing][np.argsort(distances[facing], kind="stable")]:
            back = directions.get(int(end))
            if back is not None:
                gap = weighted[tip] - weighted[int(end)]
                if float(np.dot(back, gap)) < float(min_facing_cosine) * float(np.linalg.norm(gap)):
                    continue
            if accept is not None:
                if tested >= _MAX_MASK_TESTED_TARGETS_PER_TIP:
                    break
                tested += 1
                if not accept(tip, int(end)):
                    continue
            candidates.append((float(np.linalg.norm(weighted[int(end)] - weighted[tip])), tip, int(end)))
            break
        else:
            if not tested:
                declined += 1
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return candidates, declined


def connect_skeleton_components(
    skeleton: np.ndarray,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    *,
    z_distance_weight: float = 1.0,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    min_facing_cosine: float = DEFAULT_BRIDGE_MIN_FACING_COSINE,
    segmentation_mask: np.ndarray | None = None,
    weight_by_segmentation: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
    mask_support: MaskSupport | None = None,
) -> np.ndarray:
    """Bridge a branch end of one skeleton component to a nearby voxel of
    another, where the end points at it.

    Every bridge starts at a *tip* -- a voxel with one skeleton neighbour --
    and goes to the nearest voxel of a different component that lies within
    reach and within the tip's own heading (the cosine between the tip's
    direction and the bridge at least *min_facing_cosine*). If that voxel is
    itself a tip, it must face back the same way. So a vessel broken in two,
    or one ending just short of another, is joined; two parallel vessels a
    few voxels apart, neither of which ends there, are not. This used to join
    the nearest pair of voxels between any two components, wherever they
    were, which cross-linked side-by-side capillaries. A component with no
    tip (a loop) can still be bridged *to*; a lone voxel, having no
    direction, is never bridged from.

    Candidates are taken shortest first, and a greedy union-find skips any
    whose two components have already been joined.

    Parameters
    ----------
    skeleton:
        Boolean skeleton array.
    max_bridge_distance:
        How far a bridge may reach, in voxels of the finest axis: in microns
        that is ``max_bridge_distance * min(voxel_size_zyx)``, and every
        distance is measured in microns, so a gap along a coarse z axis counts
        for its real length rather than as a few voxels. With
        *voxel_size_zyx* left ``None`` every voxel is a unit cube, as before.
    z_distance_weight:
        Multiplies the z component of every distance, on top of the physical
        spacing: above 1.0 a z gap counts for more, below 1.0 for less. 1.0
        (the default) measures plain microns.
    voxel_size_zyx:
        Per-axis spacing of *skeleton*, in microns.
    min_facing_cosine:
        How nearly a tip must point along the bridge it would make, as the
        cosine of the angle between them. -1 accepts any direction.
    segmentation_mask:
        Optional binary mask of the real segmented tissue, same shape as
        *skeleton*. Only used when *weight_by_segmentation* is also true.
    weight_by_segmentation:
        When true and *segmentation_mask* is given, each accepted bridge is
        drawn by routing through the mask (preferring to stay inside real
        segmented signal) rather than an unconditional straight line --
        see :func:`_bridge_path_through_mask`. On its own this only changes
        the path's shape, never whether two components get connected.
    mask_support:
        A :class:`~haemolynx.preprocessing.bridge_mask_support.MaskSupport`
        over the segmentation (same shape as *skeleton*). A bridge is then
        drawn only along a path it accepts -- the straight line, or the route
        through the mask, tried first when *weight_by_segmentation* is set --
        and only when that path does not run beside the skeleton, or another
        bridge, in the same lumen. A tip refused its nearest target tries the
        next. ``None`` draws every bridge within reach, as before.
    use_memmap, memmap_directory:
        Back the labelled array (see :func:`_labeled_components`), the
        working copy of *skeleton* that bridges are drawn into, and (via
        :func:`skeletonize_by_component`) the re-skeletonize step after
        bridging, with disk-backed buffers instead of fresh in-RAM ones.
        The coordinate list and KD-trees below are sized to the sparse
        skeleton's own foreground voxel count, not the volume.
    tile_large_components, tile_max_voxels, tile_halo_voxels:
        Forwarded to the re-skeletonize step's own
        :func:`skeletonize_by_component` call, only reached when a bridge
        was actually drawn -- see its own docstring.
    """
    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    weights, reach = _bridge_metric(
        skeleton.ndim, voxel_size_zyx, z_distance_weight, max_bridge_distance
    )
    with _labeled_components(
        skeleton, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as (labeled, n_components):
        if n_components <= 1:
            return skeleton

        coords = np.argwhere(labeled)
        labels = np.asarray(labeled[tuple(coords.T)])
        tips = _TipGeometry.of(coords, weights)
        directions = tips.directions
        gate = None
        if mask_support is not None and np.shape(mask_support.mask) == skeleton.shape:
            gate = _SkeletonBridgeGate(
                mask_support, coords, routed_first=bool(weight_by_segmentation)
            )
        candidates, declined = _bridge_candidates(
            tips, labels, reach, min_facing_cosine,
            accept=None if gate is None else (lambda tip, end: gate.accept(coords, tip, end)),
        )

        parent: dict[int, int] = {int(c): int(c) for c in np.unique(labels)}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        skeleton_bool = np.asanyarray(skeleton, dtype=bool)
        if use_memmap:
            result = new_memmap_array(skeleton_bool.shape, bool, directory=memmap_directory)
            _combine_per_slice(result, lambda s: s, skeleton_bool)
        else:
            result = skeleton_bool.copy()
        bridged = 0
        for _, tip, target in candidates:
            comp_a, comp_b = _find(int(labels[tip])), _find(int(labels[target]))
            if comp_a == comp_b:
                continue
            start, end = coords[tip], coords[target]
            path = None
            if gate is not None:
                path = gate.path_to_draw(tip, target)
                if path is None:
                    continue
            elif weight_by_segmentation and segmentation_mask is not None:
                if segmentation_mask.shape == skeleton.shape:
                    path = _bridge_path_through_mask(
                        segmentation_mask, start, end,
                        _finest_axis_spacing(voxel_size_zyx, skeleton.ndim),
                    )
            if path is not None:
                result[tuple(path.T)] = True
            else:
                _draw_line_3d(result, start, end)
            parent[comp_a] = comp_b
            bridged += 1

    logger.info(
        "Skeleton bridging: %d bridge(s) drawn from %d branch end(s) within %.3g um; "
        "%d end(s) had another component within reach but not ahead of them.",
        bridged,
        len(directions),
        reach,
        declined,
    )
    if gate is not None:
        logger.info(
            "Skeleton bridging: refused %d bridge(s) crossing background and %d running "
            "beside a vessel in the same lumen.",
            gate.refused_background,
            gate.refused_duplicate,
        )
    # `labeled` is released above (or was never disk-backed) before the
    # re-skeletonize step's own, potentially large, working buffers exist.
    if bridged:
        bridged_copy = result
        result = skeletonize_by_component(
            bridged_copy,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
            tile_large_components=tile_large_components,
            tile_max_voxels=tile_max_voxels,
            tile_halo_voxels=tile_halo_voxels,
        )
        release_superseded(bridged_copy, result, keep=skeleton)

    return np.asanyarray(result, dtype=bool)


#: Points in one KD-tree query batch from which it is spread over every core.
_PARALLEL_QUERY_POINTS = 5_000


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

    comp_coords: list[np.ndarray] = []
    for comp_id in range(1, n_components + 1):
        lo, hi = int(starts[comp_id - 1]), int(starts[comp_id])
        if hi > lo:
            comp_coords.append(coords_all[lo:hi] * spacing)

    # Each pair's gap is its smaller component's voxels queried against the
    # larger one's tree (the earlier one on a size tie). In (size, label)
    # order that is every component queried against each later one's tree,
    # so each tree is queried once, with all the earlier components' voxels
    # together, instead of once per pair -- the same queries, a small fraction
    # of the calls. A component's gap to the tree is the minimum over its own
    # run of the results.
    by_size = sorted(range(len(comp_coords)), key=lambda k: len(comp_coords[k]))
    ordered = [comp_coords[k] for k in by_size]
    points = np.concatenate(ordered)
    run_starts = np.cumsum([0] + [len(c) for c in ordered])
    distances = []
    for k in range(1, len(ordered)):
        queried = points[: run_starts[k]]
        # Threads only pay for themselves on a big batch (same results).
        workers = -1 if len(queried) >= _PARALLEL_QUERY_POINTS else 1
        dists, _ = cKDTree(ordered[k]).query(queried, workers=workers)
        distances.append(np.minimum.reduceat(dists, run_starts[:k]))

    return np.sort(np.concatenate(distances)) if distances else np.array([], dtype=float)


def _small_object_survival_by_size(sizes: np.ndarray, min_size: int) -> np.ndarray:
    """For each value in *sizes* (component voxel counts), whether a
    component of that size would survive
    ``skimage.morphology.remove_small_objects(..., min_size=min_size)``.

    Answered by asking skimage itself, on a tiny synthetic 1D array holding
    one isolated run of each *distinct* size, rather than hard-coding this
    installed skimage version's exact threshold comparison. That comparison
    is a deprecated parameter (``min_size``, in favour of an inclusive
    ``max_size``) whose semantics have already changed once between skimage
    releases and are only pinned here as ``scikit-image>=0.24`` with no
    upper bound -- confirmed empirically that this installed version
    removes a component of exactly *min_size* voxels (inclusive), which
    contradicts the exclusive threshold the deprecated parameter's own
    docstring still describes. Cost is proportional to the number of
    distinct sizes, never to the volume :func:`drop_small_components` is
    protecting RAM for.
    """
    if sizes.size == 0:
        return np.zeros(0, dtype=bool)
    unique_sizes, inverse = np.unique(sizes, return_inverse=True)
    synthetic = np.zeros(int(unique_sizes.sum()) + unique_sizes.size, dtype=bool)
    starts = np.empty(unique_sizes.size, dtype=np.int64)
    pos = 0
    for i, size in enumerate(unique_sizes):
        pos += 1  # a 1-voxel gap keeps each run its own connected component
        starts[i] = pos
        synthetic[pos : pos + int(size)] = True
        pos += int(size)
    filtered = remove_small_objects(synthetic, min_size=min_size, connectivity=1)
    return filtered[starts][inverse]


def _skeleton_component_lengths(
    coords: np.ndarray, labels: np.ndarray, n_components: int, relative_spacing: np.ndarray
) -> np.ndarray:
    """Each component's length in voxels of the finest axis, index 0 unused:
    every voxel counts for the mean physical length, per unit of voxel step,
    of the steps to its 26-neighbours -- 1 for a step along the finest axis,
    the axis's relative spacing along a coarser one (a lone voxel counts 1).
    On cube voxels every voxel counts exactly 1: the voxel count."""
    from scipy.spatial import cKDTree

    weights = np.ones(len(coords))
    neighbours = _neighbour_lists(cKDTree(coords), coords)
    for index, around in enumerate(neighbours):
        if not around:
            continue
        steps = coords[around] - coords[index]
        voxel_steps = np.sqrt(np.sum(steps * steps, axis=1))
        physical = np.sqrt(np.sum((steps * relative_spacing) ** 2, axis=1))
        weights[index] = float(np.mean(physical / voxel_steps))
    return np.bincount(labels, weights=weights, minlength=n_components + 1)


def _surviving_components(
    labeled: np.ndarray,
    n_components: int,
    min_size: int,
    relative_spacing: np.ndarray,
    *,
    coords: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """``survives[label]`` (index 0 unused, False): whether each component is
    at least *min_size* long in voxels of the finest axis, by skimage's own
    rule (:func:`_small_object_survival_by_size`). On cube voxels the length is
    the voxel count, exactly as ``remove_small_objects`` counts it."""
    if coords is None:
        coords = np.argwhere(labeled)
        labels = np.asarray(labeled[tuple(coords.T)])
    if np.allclose(relative_spacing, 1.0):
        sizes = np.bincount(labels, minlength=n_components + 1)
    else:
        sizes = np.rint(
            _skeleton_component_lengths(coords, labels, n_components, relative_spacing)
        ).astype(np.int64)
    survives = np.zeros(n_components + 1, dtype=bool)
    survives[1:] = _small_object_survival_by_size(sizes[1:], min_size)
    return survives


def drop_small_components(
    mask: np.ndarray,
    *,
    min_size: int,
    connectivity: int = 1,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Remove connected components smaller than *min_size* voxels.

    With an anisotropic *voxel_size_zyx*, for a skeleton: smaller than
    *min_size* voxels of the finest axis in length, each voxel counting for
    the physical steps to its neighbours (see
    :func:`_skeleton_component_lengths`) -- three voxels running along a
    2 um z are 6 um of vessel, three along a 0.5 um x 1.5 um, and a count
    treated them alike. On cube voxels, or without a voxel size, exactly the
    voxel count, as below.

    With *use_memmap* False (the default), this is exactly
    ``skimage.morphology.remove_small_objects(mask, min_size=min_size,
    connectivity=connectivity)`` -- that call is made directly, unchanged,
    so nothing about this function's behaviour differs from before it
    existed for anyone not using ``use_memmap_loading``.

    With *use_memmap* True: skimage's own implementation always allocates
    two fresh full-volume buffers regardless of how *mask* is backed --
    ``out = ar.copy()`` and ``ccs = np.zeros_like(ar, dtype=np.int32)`` --
    silently defeating ``use_memmap_loading`` for whichever caller passes it
    a memmap-backed skeleton or mask. This computes the identical result
    with the label array disk-backed and transient (scoped to a ``with
    temporary_memmap_array(...)`` block, released whether or not the block
    raises) and the final mask disk-backed and persistent, combined one
    z-slice (axis 0) at a time via :func:`_combine_per_slice` so no
    full-volume plain array is ever allocated. *memmap_directory* is
    forwarded to :func:`new_memmap_array`; both are ignored when
    *use_memmap* is False. The size-threshold decision itself is delegated
    back to skimage (see :func:`_small_object_survival_by_size`) rather
    than assumed, so this tracks whatever the *use_memmap* False branch
    above actually does on whichever skimage version is installed.
    """
    relative = _finest_axis_spacing(voxel_size_zyx, np.ndim(mask))
    if not np.allclose(relative, 1.0):
        mask = np.asanyarray(mask, dtype=bool)
        structure = generate_binary_structure(mask.ndim, connectivity)
        with _labeled_components(
            mask, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
        ) as (labeled, n_components):
            if n_components == 0:
                return mask
            survives = _surviving_components(labeled, n_components, min_size, relative)
            if survives[1:].all():
                return mask
            if use_memmap:
                result = new_memmap_array(mask.shape, bool, directory=memmap_directory)
                _combine_per_slice(result, lambda m, lb: m & survives[lb], mask, labeled)
                return result
            return mask & survives[labeled]
    if not use_memmap:
        return remove_small_objects(
            np.asanyarray(mask, dtype=bool), min_size=min_size, connectivity=connectivity
        )

    mask = np.asanyarray(mask, dtype=bool)
    footprint = generate_binary_structure(mask.ndim, connectivity)
    result = new_memmap_array(mask.shape, bool, directory=memmap_directory)
    with temporary_memmap_array(mask.shape, np.int32, directory=memmap_directory) as labeled:
        label(mask, footprint, output=labeled)
        component_sizes = bincount_by_slab(labeled)
        too_small = np.zeros(component_sizes.shape, dtype=bool)
        too_small[1:] = ~_small_object_survival_by_size(component_sizes[1:], min_size)
        _combine_per_slice(result, lambda m, lb: m & ~too_small[lb], mask, labeled)
    return result


def _drop_small_components_bridging_cannot_join(
    skeleton: np.ndarray,
    *,
    min_size: int,
    weights: np.ndarray,
    reach: float,
    min_facing_cosine: float,
    connectivity: int,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Remove the components :func:`drop_small_components` would, except one
    that :func:`connect_skeleton_components` could bridge to a component big
    enough to keep -- a short piece of a broken vessel, in line with it.

    *weights* and *reach* are :func:`_bridge_metric`'s. A speck of noise goes
    here, before closing and ``bridge_gaps`` can grow it into a blob whose
    skeleton is long enough to pass the size filter run after bridging, and so
    survive as a component of its own or as a spur off a vessel's side.
    """
    skeleton = np.asanyarray(skeleton, dtype=bool)
    structure = generate_binary_structure(skeleton.ndim, connectivity)
    with _labeled_components(
        skeleton, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as (labeled, n_components):
        if n_components == 0:
            return skeleton
        coords = np.argwhere(labeled)
        labels = np.asarray(labeled[tuple(coords.T)])
        drop = ~_surviving_components(
            labeled, n_components, min_size, _finest_axis_spacing(voxel_size_zyx, skeleton.ndim),
            coords=coords, labels=labels,
        )
        drop[0] = False
        if not drop.any():
            return skeleton
        small = drop[labels]
        if reach > 0.0 and not small.all():
            tips = _TipGeometry.of(coords, weights)
            joined = [
                labels[tip]
                for _, tip, _end in _bridge_candidates(
                    tips, labels, reach, min_facing_cosine, source=small, target=~small
                )[0]
            ] + [
                labels[end]
                for _, _tip, end in _bridge_candidates(
                    tips, labels, reach, min_facing_cosine, source=~small, target=small
                )[0]
            ]
            drop[np.asarray(joined, dtype=np.intp)] = False
            if not drop.any():
                return skeleton
        if use_memmap:
            result = new_memmap_array(skeleton.shape, bool, directory=memmap_directory)
            _combine_per_slice(result, lambda m, lb: m & ~drop[lb], skeleton, labeled)
            return result
        return skeleton & ~drop[labeled]


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
    bridge_min_facing_cosine: float = DEFAULT_BRIDGE_MIN_FACING_COSINE,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    segmentation_mask: np.ndarray | None = None,
    bridge_weight_by_segmentation: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
    bridge_require_mask_support: bool = True,
    bridge_max_background_gap_um: float = DEFAULT_MAX_BACKGROUND_GAP_UM,
    bridge_min_mask_fraction: float = DEFAULT_MIN_MASK_FRACTION,
) -> np.ndarray:
    """Refine dense bundles, close and re-skeletonize, reconnect fragments,
    then remove what is still too small.

    Parameters
    ----------
    skeleton_image:
        Raw boolean skeleton from :func:`skeletonize_volume` (or initial load).
    min_branch_length:
        Connected components with fewer than this many voxels are removed.
        One that bridging could join to a component big enough to keep is
        kept until after bridging, so a short piece of a broken vessel gets
        its chance to be joined back first (removed up front, a larger value
        deleted exactly the fragments bridging exists to reconnect). The rest
        go up front, before closing and ``bridge_gaps`` can grow a speck of
        noise into a fragment that survives the filter on its own.
    max_bridge_distance:
        How far a fragment's branch end may be from another component, in
        voxels of the finest axis, to be bridged to it -- see
        :func:`connect_skeleton_components`. Set to 0 to disable reconnection.
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
    bridge_z_distance_weight, bridge_min_facing_cosine, voxel_size_zyx, segmentation_mask, bridge_weight_by_segmentation:
        Forwarded to :func:`connect_skeleton_components` as
        ``z_distance_weight``, ``min_facing_cosine``, ``voxel_size_zyx``,
        ``segmentation_mask``, and ``weight_by_segmentation`` respectively --
        see its own docstring.
    use_memmap, memmap_directory:
        Forwarded to :func:`drop_small_components`; to :func:`bridge_gaps`, whose distance-transform path
        (only taken when ``bridge_gap_size`` exceeds
        :data:`MAX_BALL_DILATION_RADIUS`) is a full-volume ``float64``
        allocation this function can also redirect to disk; to the
        re-skeletonize step after bundle collapse, via
        :func:`skeletonize_by_component` rather than the plain
        :func:`skeletonize_volume` -- bundle collapse changes individual
        voxels but never the volume's shape, so that second Lee-thinning
        call is exactly as large as the first one this pipeline already
        protects, not a smaller, already-cheap operation; to
        :func:`connect_skeleton_components` (reached whenever
        ``max_bridge_distance`` is nonzero, the default), whose own
        connected-component labelling and bridged-copy buffer are the same
        shape of allocation; and to
        :func:`_filter_components_by_total_fraction` (only reached when
        ``min_component_fraction`` is set), for the same reason.
    tile_large_components, tile_max_voxels, tile_halo_voxels:
        Forwarded to the re-skeletonize step's own
        :func:`skeletonize_by_component` call, and to
        :func:`connect_skeleton_components`'s own re-skeletonize step
        (reached only once a bridge is actually drawn) -- see either
        docstring. :func:`_filter_components_by_total_fraction` has no
        skeletonize call of its own, so these do not apply there.
    bridge_require_mask_support, bridge_max_background_gap_um, bridge_min_mask_fraction:
        With a *segmentation_mask* of the skeleton's shape and
        *bridge_require_mask_support* on, everything this function joins is
        held to the segmentation: closing and ``bridge_gaps`` add voxels only
        within one voxel of it, and a bundle hub's link and a
        :func:`connect_skeleton_components` bridge are drawn only along a
        path crossing at most *bridge_max_background_gap_um* of background in
        one run with at least *bridge_min_mask_fraction* of it in the mask
        (a :class:`~haemolynx.preprocessing.bridge_mask_support.MaskSupport`).
        Without a mask, or off, nothing changes.
    """
    conn = _resolve_component_connectivity(skeleton_image.ndim, component_connectivity)
    mask_support = None
    within = None
    if (
        bridge_require_mask_support
        and segmentation_mask is not None
        and np.shape(segmentation_mask) == np.shape(skeleton_image)
    ):
        segmentation_bool = np.asanyarray(segmentation_mask, dtype=bool)
        mask_support = MaskSupport(
            segmentation_bool,
            voxel_size_zyx if voxel_size_zyx is not None else (1.0,) * skeleton_image.ndim,
            max_background_gap_um=bridge_max_background_gap_um,
            min_mask_fraction=bridge_min_mask_fraction,
        )
        if closing_radius > 0 or bridge_gap_size > 0:
            within = bridge_gaps(
                segmentation_bool,
                max_gap=1,
                voxel_size_zyx=voxel_size_zyx,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
    tiling = {
        "tile_large_components": tile_large_components,
        "tile_max_voxels": tile_max_voxels,
        "tile_halo_voxels": tile_halo_voxels,
    }
    cleaned = skeleton_image
    before_growth = None

    def advance(new: np.ndarray) -> np.ndarray:
        # Under use_memmap every step below writes a fresh volume-sized file;
        # the one it replaces is this function's own and nobody else's, so
        # it is deleted now rather than left in the temp directory.
        if cleaned is not before_growth:
            release_superseded(cleaned, new, keep=skeleton_image)
        return new

    # Specks go before closing and bridge_gaps can grow them into fragments
    # long enough to pass the size filter at the end; a short piece that
    # bridging could join to a vessel stays for connect_skeleton_components.
    weights, reach = _bridge_metric(
        skeleton_image.ndim, voxel_size_zyx, bridge_z_distance_weight, max(max_bridge_distance, 0)
    )
    cleaned = advance(
        _drop_small_components_bridging_cannot_join(
            cleaned,
            min_size=min_branch_length,
            weights=weights,
            reach=reach,
            min_facing_cosine=bridge_min_facing_cosine,
            connectivity=conn,
            voxel_size_zyx=voxel_size_zyx,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    )

    # Refine dense local bundles into single hub nodes with clean in/out links.
    cleaned = advance(
        skeletonize_voxel_bundles_into_paths(
            cleaned,
            scan_size=bundle_scan_size,
            density_fraction=bundle_density_fraction,
            max_connections_per_hub=bundle_max_connections_per_hub,
            hub_min_spacing=bundle_hub_min_spacing,
            voxel_size_zyx=voxel_size_zyx,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
            mask_support=mask_support,
            **tiling,
        )
    )
    if within is not None:
        before_growth = cleaned

    # Morphological closing seals narrow gaps without expanding boundaries.
    if closing_radius > 0:
        cleaned = advance(
            close_binary_mask(
                cleaned,
                radius=closing_radius,
                voxel_size_zyx=voxel_size_zyx,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
                within=within,
            )
        )

    # Dilation-based gap filling reconnects nearby foreground regions.
    if bridge_gap_size > 0:
        cleaned = advance(
            bridge_gaps(
                np.asanyarray(cleaned, dtype=bool),
                max_gap=bridge_gap_size,
                voxel_size_zyx=voxel_size_zyx,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
                within=within,
            )
        )
    if isinstance(within, np.memmap):
        release_memmap_array(within)

    # Not skeletonize_volume: that is the plain, untiled, single-component
    # skimage.morphology.skeletonize call this whole module exists to avoid
    # on a volume too large for it -- calling it here on the *whole* volume
    # (bundle-collapse can change individual voxels but never the shape)
    # would silently throw away every bit of use_memmap/tiling protection
    # the first skeletonize call in this pipeline already got, for a
    # second, identical-in-kind allocation.
    cleaned = advance(
        skeletonize_by_component(
            np.asanyarray(cleaned, dtype=bool),
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
            **tiling,
        )
    )
    if before_growth is not None:
        restored = _restore_thinned_away(
            before_growth, cleaned, reach_voxels=closing_radius + bridge_gap_size + 1
        )
        if restored:
            logger.info(
                f"Mask-held closing/gap filling: {restored} skeleton piece(s) thinned "
                "away entirely were put back as they were."
            )
        release_superseded(before_growth, cleaned, keep=skeleton_image)
        before_growth = None

    # Bridge remaining disconnected components BEFORE filtering by size --
    # both filters below -- so that small fragments get a chance to merge
    # rather than being discarded.
    if max_bridge_distance > 0:
        cleaned = advance(
            connect_skeleton_components(
                np.asanyarray(cleaned, dtype=bool),
                max_bridge_distance=max_bridge_distance,
                component_connectivity=conn,
                z_distance_weight=bridge_z_distance_weight,
                voxel_size_zyx=voxel_size_zyx,
                min_facing_cosine=bridge_min_facing_cosine,
                segmentation_mask=segmentation_mask,
                weight_by_segmentation=bridge_weight_by_segmentation,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
                mask_support=mask_support,
                **tiling,
            )
        )

    cleaned = advance(
        drop_small_components(
            cleaned,
            min_size=min_branch_length,
            connectivity=conn,
            voxel_size_zyx=voxel_size_zyx,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    )

    if min_component_fraction > 0.0:
        cleaned = advance(
            _filter_components_by_total_fraction(
                cleaned,
                min_component_fraction=min_component_fraction,
                component_connectivity=conn,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )

    return np.asanyarray(cleaned, dtype=bool)
