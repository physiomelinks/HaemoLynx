"""Skeleton operations: bridging gaps, skeletonization, cleaning."""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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

from .memmap_support import (
    LOW_MEMORY_BLOCK_VOXELS,
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

    *use_memmap*, when True, runs either path one padded block at a time
    (:func:`map_blockwise`) into a new memmap in *memmap_directory*. Both
    only ever look *max_gap* voxels away, so a halo of *max_gap* makes every
    block exact. Handing scipy a memmap as the distance transform's
    ``distances=`` output is not enough on its own: scipy still builds its
    feature-transform indices, an ``np.indices`` grid and float64 copies of
    both internally -- roughly 70 bytes per voxel of plain RAM whatever the
    output array is.
    """
    if max_gap <= 0:
        return binary_skeleton
    max_gap = int(max_gap)
    if use_memmap:
        return _bridge_gaps_blockwise(binary_skeleton, max_gap, memmap_directory)
    if max_gap <= MAX_BALL_DILATION_RADIUS:
        return binary_dilation(binary_skeleton, structure=_euclidean_ball(max_gap))
    # scipy's distance transform of an array with no background voxel is not
    # all-infinite but measured from a corner: nothing must dilate to nothing.
    if not binary_skeleton.any():
        return binary_skeleton.copy()
    inverted = ~binary_skeleton
    distance = distance_transform_edt(inverted)
    return binary_skeleton | ((distance <= max_gap) & inverted)


def _bridge_gaps_blockwise(
    binary_skeleton: np.ndarray, max_gap: int, memmap_directory: str | Path | None
) -> np.memmap:
    skeleton_bool = np.asanyarray(binary_skeleton, dtype=bool)
    result = new_memmap_array(skeleton_bool.shape, bool, directory=memmap_directory)
    if max_gap <= MAX_BALL_DILATION_RADIUS:
        ball = _euclidean_ball(max_gap)
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
        return block | ((distance_transform_edt(inverted) <= max_gap) & inverted)

    return map_blockwise(skeleton_bool, within_gap, result, halo=max_gap)


def close_binary_mask(
    binary: np.ndarray,
    radius: int = 2,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
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
    use_memmap, memmap_directory:
        When *use_memmap* is True, close one padded block at a time
        (:func:`map_blockwise`) into a new memmap in *memmap_directory*
        instead of dilating, then eroding, the whole volume in RAM. A voxel's
        closed value depends only on the input within ``2 * radius`` voxels
        of it (``radius`` for the dilation, ``radius`` more for the erosion
        that reads it), so that halo makes every block exact.
    """
    from scipy.ndimage import binary_closing, generate_binary_structure

    if radius <= 0:
        return binary
    struct = generate_binary_structure(binary.ndim, 1)
    if use_memmap:
        binary_bool = np.asanyarray(binary, dtype=bool)
        result = new_memmap_array(binary_bool.shape, bool, directory=memmap_directory)
        return map_blockwise(
            binary_bool,
            lambda block: binary_closing(block, structure=struct, iterations=radius),
            result,
            halo=2 * radius,
        )
    return binary_closing(binary.astype(bool), structure=struct, iterations=radius)


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
    peak_coords: np.ndarray, peak_density: np.ndarray, hub_min_spacing: float
) -> list[np.ndarray]:
    """Density peaks, densest first, keeping each only if it is at least
    *hub_min_spacing* from every hub already kept."""
    order = np.argsort(peak_density)[::-1]
    selected_hubs: list[np.ndarray] = []
    for idx in order:
        candidate = peak_coords[idx]
        if all(np.linalg.norm(candidate - existing) >= hub_min_spacing for existing in selected_hubs):
            selected_hubs.append(candidate)
    return selected_hubs


def _draw_hub_links(
    result: np.ndarray,
    center: np.ndarray,
    boundary_points: np.ndarray,
    max_connections_per_hub: int,
) -> None:
    """Link *center* to the farthest boundary point in each direction, at
    most *max_connections_per_hub* of them, farthest first."""
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


def _collapse_hubs(
    result: np.ndarray,
    mask: np.ndarray,
    dense_volume: np.ndarray,
    selected_hubs: list[np.ndarray],
    scan: tuple[int, ...],
    max_connections_per_hub: int,
) -> None:
    """Collapse each hub's dense window of *result* to one centre voxel,
    relinked to the skeleton just outside it, in place.

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
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
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
    use_memmap, memmap_directory, tile_large_components, tile_max_voxels, tile_halo_voxels:
        With *use_memmap* True, see
        :func:`_skeletonize_voxel_bundles_into_paths_low_memory` -- the same
        steps without any whole-volume plain-RAM array. The tiling settings
        are forwarded to its two skeletonize calls.
    """
    mask = np.asanyarray(binary_mask, dtype=bool) if use_memmap else binary_mask.astype(bool)
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

    if use_memmap:
        return _skeletonize_voxel_bundles_into_paths_low_memory(
            mask,
            scan,
            density_fraction,
            max_connections_per_hub,
            hub_min_spacing,
            memmap_directory=memmap_directory,
            tile_large_components=tile_large_components,
            tile_max_voxels=tile_max_voxels,
            tile_halo_voxels=tile_halo_voxels,
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
        peak_coords, density[tuple(peak_coords.T)], hub_min_spacing
    )

    result = base_skeleton.astype(bool).copy()
    _collapse_hubs(result, mask, dense_volume, selected_hubs, scan, max_connections_per_hub)

    return skeletonize_volume(result.astype(bool)).astype(bool)


def _skeletonize_voxel_bundles_into_paths_low_memory(
    mask: np.ndarray,
    scan: tuple[int, ...],
    density_fraction: float,
    max_connections_per_hub: int,
    hub_min_spacing: float,
    *,
    memmap_directory: str | Path | None,
    tile_large_components: bool,
    tile_max_voxels: int,
    tile_halo_voxels: int,
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
            peak_coords[c_order], peak_density[c_order], hub_min_spacing
        )

        result = base_skeleton
        _collapse_hubs(
            result, mask, dense_volume, selected_hubs, scan, max_connections_per_hub
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


def _pairs_within_reach(
    comp_ids: list[int], weighted: dict[int, np.ndarray], reach: float
) -> Iterator[tuple[int, int]]:
    """Every ``(a, b)``, ``a`` before ``b`` in *comp_ids*, in the same order a
    double loop over them gives -- less the pairs whose bounding boxes are
    more than *reach* apart along some axis. No voxel pair of those can be
    within *reach*: each coordinate difference is at least that gap (float
    subtraction is monotonic), and a distance is at least any one of its
    coordinate differences. So skipping them never drops a bridge.
    """
    if not comp_ids:
        return
    lo = np.stack([weighted[cid].min(axis=0) for cid in comp_ids])
    hi = np.stack([weighted[cid].max(axis=0) for cid in comp_ids])
    for i, cid_a in enumerate(comp_ids):
        later = slice(i + 1, None)
        apart = ((lo[later] - hi[i]) > reach) | ((lo[i] - hi[later]) > reach)
        for j in np.flatnonzero(~apart.any(axis=1)):
            yield cid_a, comp_ids[i + 1 + int(j)]


def _nearest_voxel_pair(
    points_a: np.ndarray,
    tree_a,
    points_b: np.ndarray,
    tree_b,
    *,
    max_distance: float,
) -> tuple[float, int, int] | None:
    """``(distance, i, j)`` for the closest voxels ``points_a[i]``,
    ``points_b[j]``, or None if they are further apart than *max_distance*.

    Exactly what querying every point of *a* against *b*'s tree and taking the
    first minimum gives -- the same ``i`` and ``j`` on a tie -- without doing
    that for every point of a large component: the minimum distance comes
    from whichever side is smaller, and only the points of *a* that could
    reach it are queried the original way to pick ``i`` and ``j``.
    """
    bound = float(max_distance) * (1 + 1e-9) + 1e-9
    if len(points_a) <= len(points_b):
        dists, _ = tree_b.query(points_a, distance_upper_bound=bound)
        min_dist = float(dists.min())
        if not min_dist <= max_distance:
            return None
        candidates = np.flatnonzero(dists == min_dist)[:1]
    else:
        dists, _ = tree_a.query(points_b, distance_upper_bound=bound)
        min_dist = float(dists.min())
        if not min_dist <= max_distance:
            return None
        # Any point of a at the minimum is within it of one of these.
        near = tree_a.query_ball_point(
            points_b[dists == min_dist], r=min_dist * (1 + 1e-9) + 1e-9
        )
        candidates = np.unique(np.concatenate([np.asarray(n, dtype=np.intp) for n in near]))
    exact, idx = tree_b.query(points_a[candidates])
    first = int(np.flatnonzero(exact == min_dist)[0])
    return min_dist, int(candidates[first]), int(idx[first])


def connect_skeleton_components(
    skeleton: np.ndarray,
    max_bridge_distance: int = 20,
    component_connectivity: int | None = None,
    *,
    z_distance_weight: float = 1.0,
    segmentation_mask: np.ndarray | None = None,
    weight_by_segmentation: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
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
    use_memmap, memmap_directory:
        Back the labelled array (see :func:`_labeled_components`), the
        working copy of *skeleton* that bridges are drawn into, and (via
        :func:`skeletonize_by_component`) the re-skeletonize step after
        bridging, with disk-backed buffers instead of fresh in-RAM ones.
        The coordinate lists and KD-trees below are sized to the sparse
        skeleton's own foreground voxel count, not the volume, so only
        these three are worth redirecting.
    tile_large_components, tile_max_voxels, tile_halo_voxels:
        Forwarded to the re-skeletonize step's own
        :func:`skeletonize_by_component` call, only reached when a bridge
        was actually drawn -- see its own docstring.
    """
    from scipy.spatial import cKDTree

    conn = _resolve_component_connectivity(skeleton.ndim, component_connectivity)
    structure = generate_binary_structure(skeleton.ndim, conn)
    with _labeled_components(
        skeleton, structure, use_memmap=use_memmap, memmap_directory=memmap_directory
    ) as (labeled, n_components):
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
        weighted = {cid: comp_coords[cid] * axis_weights for cid in comp_ids}
        candidates: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
        for cid_a, cid_b in _pairs_within_reach(comp_ids, weighted, max_bridge_distance):
            found = _nearest_voxel_pair(
                weighted[cid_a],
                comp_trees[cid_a],
                weighted[cid_b],
                comp_trees[cid_b],
                max_distance=max_bridge_distance,
            )
            if found is not None:
                min_dist, start_idx, end_idx = found
                start = comp_coords[cid_a][start_idx]
                end = comp_coords[cid_b][end_idx]
                candidates.append((min_dist, start, end, cid_a, cid_b))

        candidates.sort(key=lambda c: c[0])

        skeleton_bool = np.asanyarray(skeleton, dtype=bool)
        if use_memmap:
            result = new_memmap_array(skeleton_bool.shape, bool, directory=memmap_directory)
            _combine_per_slice(result, lambda s: s, skeleton_bool)
        else:
            result = skeleton_bool.copy()
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

    # `labeled` is released above (or was never disk-backed) before the
    # re-skeletonize step's own, potentially large, working buffers exist.
    if bridged:
        logger.debug("Bridged %d skeleton component pair(s).", bridged)
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


def drop_small_components(
    mask: np.ndarray,
    *,
    min_size: int,
    connectivity: int = 1,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Remove connected components smaller than *min_size* voxels.

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
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
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
        Forwarded to :func:`drop_small_components` (the very first step
        below); to :func:`bridge_gaps`, whose distance-transform path
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
    """
    conn = _resolve_component_connectivity(skeleton_image.ndim, component_connectivity)
    tiling = {
        "tile_large_components": tile_large_components,
        "tile_max_voxels": tile_max_voxels,
        "tile_halo_voxels": tile_halo_voxels,
    }
    cleaned = skeleton_image

    def advance(new: np.ndarray) -> np.ndarray:
        # Under use_memmap every step below writes a fresh volume-sized file;
        # the one it replaces is this function's own and nobody else's, so
        # it is deleted now rather than left in the temp directory.
        release_superseded(cleaned, new, keep=skeleton_image)
        return new

    cleaned = advance(
        drop_small_components(
            cleaned,
            min_size=min_branch_length,
            connectivity=conn,
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
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
            **tiling,
        )
    )

    # Morphological closing seals narrow gaps without expanding boundaries.
    if closing_radius > 0:
        cleaned = advance(
            close_binary_mask(
                cleaned,
                radius=closing_radius,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )

    # Dilation-based gap filling reconnects nearby foreground regions.
    if bridge_gap_size > 0:
        cleaned = advance(
            bridge_gaps(
                np.asanyarray(cleaned, dtype=bool),
                max_gap=bridge_gap_size,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
            )
        )

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

    # Bridge remaining disconnected components BEFORE filtering by size so
    # that small fragments get a chance to merge rather than being discarded.
    if max_bridge_distance > 0:
        cleaned = advance(
            connect_skeleton_components(
                np.asanyarray(cleaned, dtype=bool),
                max_bridge_distance=max_bridge_distance,
                component_connectivity=conn,
                z_distance_weight=bridge_z_distance_weight,
                segmentation_mask=segmentation_mask,
                weight_by_segmentation=bridge_weight_by_segmentation,
                use_memmap=use_memmap,
                memmap_directory=memmap_directory,
                **tiling,
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
