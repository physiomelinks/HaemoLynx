"""Recover vessels the segmentation has and the graph lost.

Every gate that keeps a bridge from crossing background can also leave a
vessel the mask clearly has with no edge through it: a fragment the skeleton
filter dropped, a branch whose bridge was refused. This step finds the mask
the finished graph does not cover, traces a centreline through each piece big
enough to be a vessel, and joins it to the network through the mask -- under
the same mask-support test and same-lumen duplicate guard as every reconnect
(``reconnect.MaskBridges``), so a recovered piece is never a second strand
beside a vessel the graph already has.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import convolve, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import SAME_LUMEN_MARGIN_UM, MaskSupport
from haemolynx.preprocessing.memmap_support import LOW_MEMORY_BLOCK_VOXELS, slab_step

from ._helpers import (
    EdgeSampleIndex,
    duplicates_existing_vessel,
    edge_id,
    edge_sample_points,
    next_node_id,
)
from .build import GAP_BRIDGE_TANGENT_LENGTH_UM, gap_bridge_continues_terminal
from .edit import delete_edge_and_collapse, insert_node_on_edge
from .lumen_loops import LOOP_SEARCH_UM, loop_inside_one_lumen
from .reconnect import route_through_mask

logger = logging.getLogger(__name__)

#: Smallest uncovered piece of mask worth tracing, in cubic microns.
DEFAULT_MIN_REGION_VOLUME_UM3 = 30.0

#: Shortest centreline a recovered piece may add, in microns.
DEFAULT_MIN_LENGTH_UM = 5.0

#: An uncovered piece whose widest point is no more than this many voxels of
#: the finest axis from background is a sliver of a covered vessel's wall,
#: not a vessel of its own.
MIN_INSCRIBED_RADIUS_VOXELS = 1.5

#: Nearest centreline samples each mask voxel is judged against -- and every
#: further sample exactly as far as the last of them (:func:`_nearest_with_ties`).
_COVERAGE_NEIGHBOURS = 4

#: Two sample distances closer than this (um) are one distance.
_TIE_UM = 1e-9

#: How far out from a centreline, in lumen radii (+ the margin), its lumen's
#: cross-section is followed to the wall. A flattened lumen's far side lies
#: well past its radius -- the radius is the distance to the *nearest* wall --
#: and on E14.5 79% of the mask the radius rule left uncovered had a straight
#: line through the mask to its nearest centreline, a median 2 radii out, 90%
#: within 3.2. Recovery traced it as a second strand. A branch the graph lost
#: runs on past this reach, so its far part stays uncovered and is recovered.
CROSS_SECTION_REACH_RADII = 4.0

#: Directions round a centreline sample its cross-section is followed in:
#: enough that the outline of a thin, flat lumen keeps its corners.
_CROSS_SECTION_RAYS = 32

#: Samples whose cross-sections are followed at once (memory, not result).
_CROSS_SECTION_CHUNK = 2048


@dataclass
class _CrossSections:
    """Each centreline sample's plane across the vessel (unit vectors *e1*,
    *e2* perpendicular to the tangent *t*) and how far the mask runs from the
    sample in each of :data:`_CROSS_SECTION_RAYS` directions in that plane."""

    t: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    reach: np.ndarray


def _sample_tangents(index: EdgeSampleIndex) -> np.ndarray:
    """A unit tangent at every sample, read along the sample's own edge (an
    index holds each edge's samples one after another)."""
    points = index.points
    tangents = np.zeros_like(points)
    owners = index.owners
    start = 0
    for stop in range(1, len(points) + 1):
        if stop < len(points) and owners[stop] == owners[start]:
            continue
        block = points[start:stop]
        if len(block) > 1:
            tangents[start:stop] = np.gradient(block, axis=0)
        start = stop
    norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.where(norm > 1e-9, tangents / np.maximum(norm, 1e-9), np.array([1.0, 0.0, 0.0]))
    return tangents


def _follow_rays(inside, points: np.ndarray, t: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                 radii: np.ndarray, caps: np.ndarray, step: float) -> np.ndarray:
    """How far the mask runs from each sample along each of
    :data:`_CROSS_SECTION_RAYS` rays in its plane: the last of the distances
    ``step, 2 step, ...`` no further than the sample's cap at which the mask
    holds the point, and the points a lumen radius ahead of and behind it
    along the vessel (0 when the first is already out). *inside* answers
    whether physical points are in the mask.

    Each ray is followed only until it leaves the mask or passes its cap, so
    the lookups made are those of the outline itself, not of every distance
    up to the widest vessel's; the answer is per sample and per ray, the
    same whichever samples are followed together.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, _CROSS_SECTION_RAYS, endpoint=False)
    directions = (np.cos(angles)[None, :, None] * e1[:, None, :]
                  + np.sin(angles)[None, :, None] * e2[:, None, :])
    # A flattened lumen's far side runs on along the vessel; a branch
    # leaving it is only as wide as itself. The outline stops where the
    # mask is not there a lumen radius ahead and behind as well, so a
    # branch narrower than its vessel is left uncovered, to be recovered.
    along = radii[:, None] * t
    reach = np.zeros((len(points), _CROSS_SECTION_RAYS))
    alive = np.ones((len(points), _CROSS_SECTION_RAYS), dtype=bool)
    previous = 0.0
    i = 1
    while True:
        s, r = np.nonzero(alive)
        if not len(s):
            return reach
        # The same multiples of step whatever the chunk: i * step, never a
        # running sum.
        d = i * step
        at = points[s] + d * directions[s, r]
        offset = along[s]
        held = inside(np.concatenate([at, at + offset, at - offset])).reshape(3, len(s)).all(axis=0)
        held &= d <= caps[s]  # past its cap every ray is out, so this ends
        gone = ~held
        reach[s[gone], r[gone]] = previous
        alive[s[gone], r[gone]] = False
        previous = d
        i += 1


def _cross_section_task(bits: np.ndarray, payload) -> np.ndarray:
    """:func:`_follow_rays` for one chunk of samples, reading the mask from
    its packed bits -- a module-level function, so worker processes can run it."""
    from haemolynx.preprocessing.bridge_mask_support import PackedMask, _points_in_mask

    points, t, e1, e2, radii, caps, step, spacing, shape = payload
    packed = PackedMask(bits, shape)
    return _follow_rays(
        lambda p: _points_in_mask(p, packed, spacing), points, t, e1, e2, radii, caps, step
    )


#: Fewer centreline samples than this have their cross-sections followed in
#: this process. Starting the workers costs about ten seconds (each imports
#: the package afresh), and one process follows roughly 6,000 samples a
#: second: E14.5's 66,000 stay here, a 16 G-voxel stack's millions go out.
_WORKER_MIN_SAMPLES = 400_000


def _cross_sections(index: EdgeSampleIndex, radii: np.ndarray, support: MaskSupport, *,
                    reach_radii: float, margin_um: float, workers: int | None = None) -> _CrossSections:
    from haemolynx.preprocessing.bridge_mask_support import _sample_step
    from haemolynx.preprocessing.shared_volume_tasks import default_worker_count, run_volume_tasks

    t = _sample_tangents(index)
    helper = np.eye(3)[np.argmin(np.abs(t), axis=1)]
    e1 = np.cross(t, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(t, e1)
    step = _sample_step(support.voxel_size_zyx)
    caps = reach_radii * radii + margin_um
    reach = np.zeros((len(radii), _CROSS_SECTION_RAYS))
    if not len(radii):
        return _CrossSections(t, e1, e2, reach)
    # Samples whose rays run about as far are followed together, so a chunk
    # is not kept going by its one widest vessel; the lookups go to the
    # mask's bits, and the chunks to worker processes when there are enough
    # of them to repay starting one. The result does not depend on either.
    order = np.argsort(caps, kind="stable")
    packed = support.pack()
    spacing = support.voxel_size_zyx
    payloads = []
    for lo in range(0, len(order), _CROSS_SECTION_CHUNK):
        chunk = order[lo:lo + _CROSS_SECTION_CHUNK]
        payloads.append((index.points[chunk], t[chunk], e1[chunk], e2[chunk], radii[chunk],
                         caps[chunk], step, spacing, packed.shape))
    if workers is None:
        workers = default_worker_count() if len(radii) >= _WORKER_MIN_SAMPLES else 1
    # The bits the workers map go where the run keeps its volumes, not %TEMP%.
    results = run_volume_tasks(
        _cross_section_task, payloads, volume=packed.bits, workers=int(workers), min_payloads=2,
        memmap_directory=getattr(support, "memmap_directory", None),
    )
    for lo, result in zip(range(0, len(order), _CROSS_SECTION_CHUNK), results):
        reach[order[lo:lo + _CROSS_SECTION_CHUNK]] = result
    return _CrossSections(t, e1, e2, reach)


def _inside_cross_section(points_um, candidates, sections: _CrossSections, sample_points,
                          *, axial_um: float, margin_um: float) -> np.ndarray:
    """Whether each point lies inside the cross-section outline of any of its
    candidate samples (``candidates``: ``(n, k)`` indices, the sample count
    where there is none): within *axial_um* of the sample's plane and no
    further from it than the mask runs in that direction, read between the
    two rays either side."""
    n, k = candidates.shape
    covered = np.zeros(n, dtype=bool)
    rays = sections.reach.shape[1]
    sector = 2.0 * np.pi / rays
    for j in range(k):
        index = candidates[:, j]
        valid = index < len(sample_points)
        if not valid.any():
            continue
        at = np.flatnonzero(valid & ~covered)
        if not len(at):
            break
        s = index[at]
        v = points_um[at] - sample_points[s]
        axial = np.sum(v * sections.t[s], axis=1)
        w = v - axial[:, None] * sections.t[s]
        rho = np.linalg.norm(w, axis=1)
        theta = np.mod(np.arctan2(np.sum(w * sections.e2[s], axis=1), np.sum(w * sections.e1[s], axis=1)), 2.0 * np.pi)
        lower = np.floor(theta / sector).astype(int) % rays
        upper = (lower + 1) % rays
        fraction = theta / sector - np.floor(theta / sector)
        limit = (1.0 - fraction) * sections.reach[s, lower] + fraction * sections.reach[s, upper]
        covered[at] = (np.abs(axial) <= axial_um) & (rho <= limit + margin_um)
    return covered

#: How far past a covered lumen, in microns of slack (distance minus that
#: lumen's radius), a recovered end may look for a vessel to join. The
#: traced end sits inside the uncovered mask, and the nearest sample is
#: often not the vessel the branch continues.
_MASK_JOIN_REACH_UM = 16.0

#: Distinct vessels tried at one end, nearest lumen first.
_JOIN_TARGETS = 6

#: Finest-axis voxels of room for a recovery join to follow a bend in the
#: mask, rather than only the straight chord between its ends.
_RECOVERY_ROUTE_PAD = 12


@dataclass
class _Piece:
    """One uncovered region's centreline: paths between voxel-keyed ends."""

    paths: list[np.ndarray] = field(default_factory=list)
    ends: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)


def _nearest_with_ties(tree: cKDTree, points: np.ndarray, k: int, upper: float):
    """Each point's *k* nearest samples within *upper*, with every further
    sample exactly as far as the *k*-th: blocks of ``(rows, distance, index)``,
    ``inf`` and ``tree.n`` where a row has fewer.

    A junction holds one sample per edge at one point, each with its own
    edge's cross-section. Which of those tied samples a plain *k*-nearest
    query hands back depends on how the tree split every sample in the image,
    so a change anywhere changed which edges judged a voxel here -- the mask
    left uncovered differed 280 um from any change on E14.5
    (PLAN_dale_review_fixes.md, item 2). Rows without a tie at the *k*-th
    come back *k* wide; the few with one, in blocks as wide as they need.
    """
    count = int(tree.n)
    wide = min(k + 1, count)
    distance, nearest = tree.query(points, k=wide, distance_upper_bound=upper, workers=-1)
    distance, nearest = distance.reshape(len(points), wide), nearest.reshape(len(points), wide)
    if wide == k:
        yield np.arange(len(points)), distance, nearest
        return
    tied = np.isfinite(distance[:, k]) & (distance[:, k] <= distance[:, k - 1] + _TIE_UM)
    plain = np.flatnonzero(~tied)
    yield plain, distance[plain, :k], nearest[plain, :k]
    rows = np.flatnonzero(tied)
    while len(rows):
        wide = min(2 * wide, count)
        distance, nearest = tree.query(points[rows], k=wide, distance_upper_bound=upper, workers=-1)
        distance, nearest = distance.reshape(len(rows), wide), nearest.reshape(len(rows), wide)
        kth = distance[:, k - 1:k]
        more = (wide < count) & np.isfinite(distance[:, -1]) & (distance[:, -1] <= kth[:, 0] + _TIE_UM)
        done = ~more
        beyond = distance[done] > kth[done] + _TIE_UM
        yield rows[done], np.where(beyond, np.inf, distance[done]), np.where(beyond, count, nearest[done])
        rows = rows[more]


def _centreline_samples(G: nx.MultiGraph, support: MaskSupport):
    index = EdgeSampleIndex(G, step_um=float(min(support.voxel_size_zyx)))
    radii = support.radius(index.points) if len(index.points) else np.zeros(0)
    return index, radii


def uncovered_mask_voxels(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    block_voxels: int = LOW_MEMORY_BLOCK_VOXELS,
    cross_section_reach_radii: float = CROSS_SECTION_REACH_RADII,
    z_range: tuple[int, int] | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Voxel indices of the mask no centreline sample of *G* covers. A sample
    ``p`` covers a voxel within ``radius(p) + margin_um`` of it, and -- with a
    positive *cross_section_reach_radii* -- one inside its lumen's
    cross-section: in the plane across the vessel at ``p``, no further out than
    the mask runs from ``p`` in that direction, followed to at most that many
    radii (+ *margin_um*; :data:`CROSS_SECTION_REACH_RADII`). The radius is
    the distance to the nearest wall, so the radius rule alone leaves the far
    side of every flattened lumen uncovered. 0 keeps the radius rule only.
    A voxel is judged by its :data:`_COVERAGE_NEIGHBOURS` nearest samples and
    every sample tied with the last of them, so the answer does not depend on
    the order the graph's edges come in, or on edges elsewhere.
    Read a slab of whole z-slices at a time, so the mask may be a memmap;
    *z_range* ``(first, stop)`` reads only those slices. *workers* is how
    many processes follow the cross-sections (by default several, once there
    are :data:`_WORKER_MIN_SAMPLES` samples); the answer is the same with any."""
    mask = support.mask
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    index, radii = _centreline_samples(G, support)
    sections = None
    if index.tree is not None and cross_section_reach_radii > 0:
        sections = _cross_sections(index, radii, support, reach_radii=cross_section_reach_radii,
                                   margin_um=margin_um, workers=workers)
    step = slab_step(mask.shape, block_voxels) if np.ndim(mask) == 3 else mask.shape[0]
    uncovered = []
    reach = float(radii.max()) + margin_um if len(radii) else 0.0
    section_reach = float(sections.reach.max()) + margin_um if sections is not None else 0.0
    axial = float(spacing.min())
    first, stop = (0, mask.shape[0]) if z_range is None else (
        max(int(z_range[0]), 0), min(int(z_range[1]), mask.shape[0])
    )
    for start in range(first, stop, step):
        coords = np.argwhere(np.asarray(mask[start:min(start + step, stop)], dtype=bool))
        if not len(coords):
            continue
        coords[:, 0] += start
        if index.tree is None:
            uncovered.append(coords)
            continue
        k = min(_COVERAGE_NEIGHBOURS, len(index.points))
        held = np.zeros(len(coords), dtype=bool)
        for rows, distance, nearest in _nearest_with_ties(index.tree, coords * spacing, k, reach + 1e-9):
            found = nearest < len(radii)
            slack = np.where(found, distance - radii[np.minimum(nearest, len(radii) - 1)], np.inf)
            held[rows] = np.any(slack <= margin_um + 1e-9, axis=1)
        left = coords[~held]
        if sections is not None and len(left):
            points = left * spacing
            inside = np.zeros(len(left), dtype=bool)
            for rows, _d, candidates in _nearest_with_ties(index.tree, points, k, section_reach + 1e-9):
                inside[rows] = _inside_cross_section(points[rows], candidates, sections, index.points,
                                                     axial_um=axial, margin_um=margin_um)
            left = left[~inside]
        uncovered.append(left)
    return np.concatenate(uncovered) if uncovered else np.empty((0, np.ndim(mask)), dtype=np.intp)


def _components(coords: np.ndarray, *, block_voxels: int = LOW_MEMORY_BLOCK_VOXELS) -> list[np.ndarray]:
    """26-connected groups of voxel indices ``(z, y, x)``: the groups in the
    order their first voxel comes in *coords*, each in *coords*' own order.

    Labelled a slab of z-slices at a time, as a dense block over the slab's
    bounding box of at most *block_voxels* (``scipy.ndimage.label``), the
    slabs joined wherever a voxel in one's last slice touches one in the
    next's first. The KD-tree pair search this replaces held every pair of
    neighbouring voxels at once -- 13 a voxel in a solid region, a hundred
    bytes each -- which is gigabytes for the uncovered mask of a large stack,
    and took six times as long on two million voxels.
    """
    coords = np.asarray(coords, dtype=np.intp).reshape(-1, 3)
    n = len(coords)
    if n == 0:
        return []
    lo = coords.min(axis=0)
    extent = coords.max(axis=0) - lo + 1
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    local = coords[order] - lo
    z = local[:, 0]
    slice_voxels = int(extent[1]) * int(extent[2])
    dz = max(1, int(block_voxels) // max(slice_voxels, 1))
    structure = np.ones((3, 3, 3), dtype=bool)
    ids = np.zeros(n, dtype=np.intp)
    links = []
    offset = 0
    previous = None  # the slab before's last slice, as ids; None across an empty slab
    for z0 in range(0, int(extent[0]), dz):
        z1 = min(z0 + dz, int(extent[0]))
        a, b = np.searchsorted(z, [z0, z1])
        if a == b:
            previous = None
            continue
        slab = local[a:b]
        block = np.zeros((z1 - z0, int(extent[1]), int(extent[2])), dtype=bool)
        block[slab[:, 0] - z0, slab[:, 1], slab[:, 2]] = True
        labelled, count = label(block, structure=structure)
        ids[a:b] = labelled[slab[:, 0] - z0, slab[:, 1], slab[:, 2]] + offset
        if previous is not None:
            first = labelled[0]
            rows, cols = first.shape
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = previous[max(0, -dy):rows - max(0, dy), max(0, -dx):cols - max(0, dx)]
                    c = first[max(0, dy):rows - max(0, -dy), max(0, dx):cols - max(0, -dx)]
                    both = (p > 0) & (c > 0)
                    if both.any():
                        links.append(np.column_stack([p[both], c[both] + offset]))
        last = labelled[-1]
        previous = np.where(last > 0, last + offset, 0)
        offset += int(count)
    if links:
        pairs = np.concatenate(links)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(offset + 1,) * 2)
        _, group_of_id = connected_components(graph, directed=False)
    else:
        group_of_id = np.arange(offset + 1)
    group = np.empty(n, dtype=np.intp)
    group[order] = group_of_id[ids]
    groups, first_index, dense = np.unique(group, return_index=True, return_inverse=True)
    rank = np.empty(len(groups), dtype=np.intp)
    rank[np.argsort(first_index, kind="stable")] = np.arange(len(groups))
    final = rank[dense.reshape(-1)]
    out = np.argsort(final, kind="stable")
    bounds = np.flatnonzero(np.diff(final[out])) + 1
    return np.split(coords[out], bounds)


#: Voxels of an uncovered region :func:`uncovered_mask_regions` hands back to
#: draw it: enough to show its shape, few enough to draw at once.
REGION_SAMPLE_VOXELS = 2000


@dataclass(frozen=True)
class UncoveredRegion:
    """One piece of the mask no centreline covers."""

    voxel_count: int
    volume_um3: float
    #: Its middle, physical microns ``(z, y, x)``.
    centre_um: tuple[float, float, float]
    #: Its widest point's distance to background, in microns.
    radius_um: float
    #: Up to :data:`REGION_SAMPLE_VOXELS` of its voxels, physical microns.
    sample_um: np.ndarray = field(compare=False, repr=False)


def uncovered_mask_regions(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    min_region_volume_um3: float = DEFAULT_MIN_REGION_VOLUME_UM3,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    z_range: tuple[int, int] | None = None,
) -> list[UncoveredRegion]:
    """The pieces of mask :func:`recover_uncovered_mask_vessels` would trace,
    largest first -- read-only: 26-connected regions of
    :func:`uncovered_mask_voxels` holding at least *min_region_volume_um3*
    and wider somewhere than :data:`MIN_INSCRIBED_RADIUS_VOXELS` (a thinner
    one is a sliver of a covered vessel's wall). On a graph recovery has
    already run on, each is a segmented vessel it could not join."""
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    finest = float(spacing.min())
    rng = np.random.default_rng(0)
    regions = []
    for region in _components(uncovered_mask_voxels(G, support, margin_um=margin_um, z_range=z_range)):
        if len(region) * voxel_volume < float(min_region_volume_um3):
            continue
        points = region * spacing
        radius = float(np.max(support.radius(points)))
        if radius < MIN_INSCRIBED_RADIUS_VOXELS * finest:
            continue
        if len(points) > REGION_SAMPLE_VOXELS:
            points = points[np.sort(rng.choice(len(points), REGION_SAMPLE_VOXELS, replace=False))]
        regions.append(UncoveredRegion(
            voxel_count=int(len(region)),
            volume_um3=float(len(region) * voxel_volume),
            centre_um=tuple(float(c) for c in (region * spacing).mean(axis=0)),  # type: ignore[arg-type]
            radius_um=radius,
            sample_um=points,
        ))
    regions.sort(key=lambda r: -r.voxel_count)
    return regions


def _trace_piece(
    region: np.ndarray, support: MaskSupport, pad: int, *, skeleton: np.ndarray | None = None
) -> _Piece:
    """The centreline where it runs through *region*, as skan paths: the
    run's *skeleton* when given, else the mask's own (Lee thinning of the
    whole mask round the region)."""
    if skeleton is not None:
        return _trace_skeleton_in_region(region, skeleton, support.voxel_size_zyx)
    from skan import csr

    shape = np.asarray(support.mask.shape)
    lo = np.maximum(region.min(axis=0) - pad, 0)
    hi = np.minimum(region.max(axis=0) + pad + 1, shape)
    window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    if skeleton is not None:
        skeleton = np.array(skeleton[window], dtype=bool)
    else:
        from skimage.morphology import skeletonize

        skeleton = skeletonize(np.asarray(support.mask[window], dtype=bool), method="lee").astype(bool)
    inside = np.zeros_like(skeleton)
    inside[tuple((region - lo).T)] = True
    skeleton &= inside
    piece = _Piece()
    neighbours = convolve(skeleton.astype(np.uint8), np.ones((3,) * skeleton.ndim, np.uint8), mode="constant")
    # A component every voxel of which has two neighbours is a closed ring:
    # its one path ends where it starts, a self-loop the caller drops, and
    # skan sizes its path buffer too small for a ring of a few voxels
    # ("index pointer size 1 should be 2").
    labels, n_components = label(skeleton, structure=np.ones((3,) * skeleton.ndim, bool))
    if n_components:
        not_ring = np.zeros(n_components + 1, dtype=bool)
        not_ring[labels[skeleton & (neighbours != 3)]] = True
        skeleton &= not_ring[labels]
    # skan raises on a skeleton whose every voxel is isolated.
    if not np.any(skeleton & (neighbours > 1)):
        return piece
    sk = csr.Skeleton(skeleton)
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    for i in range(sk.n_paths):
        coords = np.rint(sk.path_coordinates(i)).astype(int) + lo
        if len(coords) < 2:
            continue
        piece.paths.append(coords * spacing)
        piece.ends.append((tuple(coords[0].tolist()), tuple(coords[-1].tolist())))
    return piece


def _trace_skeleton_in_region(region: np.ndarray, skeleton: np.ndarray, voxel_size_zyx) -> _Piece:
    """:func:`_trace_piece` along the run's *skeleton*: its voxels inside
    *region*, as skan paths, from those voxels alone.

    What the window round the region gave -- rings dropped, skan's paths in
    the same order -- without the window: its bounding box, padded, at some
    12 bytes a voxel, was a volume-sized allocation for a long region on a
    large stack (``graph.build.skan_skeleton``'s coordinate graph, which
    gives skan's own paths)."""
    from skan import csr

    from .build import _SkeletonPaths, _skeleton_adjacency_from_coordinates

    piece = _Piece()
    region = np.asarray(region, dtype=np.intp).reshape(-1, skeleton.ndim)
    coords = region[np.asarray(skeleton[tuple(region.T)], dtype=bool)]
    if not len(coords):
        return piece
    # Each voxel once, in C order, as the window's voxels were numbered.
    coords = np.unique(coords, axis=0)
    adjacency = _skeleton_adjacency_from_coordinates(coords, skeleton.shape)
    degree = np.diff(adjacency.indptr)
    # A component every voxel of which has two neighbours is a closed ring:
    # its one path ends where it starts, a self-loop the caller drops, and
    # skan sizes its path buffer too small for a ring of a few voxels.
    n_components, labels = connected_components(adjacency, directed=False)
    not_ring = np.zeros(n_components, dtype=bool)
    not_ring[labels[degree != 2]] = True
    keep = not_ring[labels]
    # skan raises on a skeleton whose every voxel is isolated.
    if not np.any(keep & (degree > 0)):
        return piece
    if not keep.all():
        coords = coords[keep]
        adjacency = _skeleton_adjacency_from_coordinates(coords, skeleton.shape)
    paths = csr._build_skeleton_path_graph(csr.csr_to_nbgraph(csr._mst_junctions(adjacency)))
    sk = _SkeletonPaths(paths, coords)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    for i in range(sk.n_paths):
        path = np.asarray(sk.path_coordinates(i), dtype=int)
        if len(path) < 2:
            continue
        piece.paths.append(path * spacing)
        piece.ends.append((tuple(path[0].tolist()), tuple(path[-1].tolist())))
    return piece


def _strands(paths):
    """Accepted paths grouped by the ends they share.

    A path the mask test refused is already absent, so the two sides of a
    broken centreline are separate strands. Each has to join the network on
    its own: adding one because the other joined would leave an island.
    """
    parent: dict[tuple[int, ...], tuple[int, ...]] = {}

    def find(end: tuple[int, ...]) -> tuple[int, ...]:
        parent.setdefault(end, end)
        root = end
        while parent[root] != root:
            root = parent[root]
        while parent[end] != root:
            parent[end], end = root, parent[end]
        return root

    def union(a: tuple[int, ...], b: tuple[int, ...]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _path, a, b, _background in paths:
        union(a, b)
    groups: dict[tuple[int, ...], list] = {}
    for item in paths:
        groups.setdefault(find(item[1]), []).append(item)
    return list(groups.values())


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


def _leaves_image_at(point_um: np.ndarray, support: MaskSupport, outward: np.ndarray | None = None) -> bool:
    """Whether a recovered strand's free end at *point_um*, heading
    *outward*, is a vessel the image cut through: the dead-end definition's
    ``leaves_image`` (``graph.dead_ends``), with the tip's lumen radius read
    from the mask -- it touches a face, or the mask runs on from it to a face
    within a few of its radii. A centreline often stops a few voxels short of
    a face the tube itself runs into; such a spur is a vessel, and stays."""
    from .dead_ends import image_extent_um, leaves_image

    tip = np.asarray(point_um, dtype=float)
    return leaves_image(
        tip,
        image_extent_um(support.mask.shape, support.voxel_size_zyx),
        heading=outward,
        support=support,
        tip_radius_um=float(support.radius(tip.reshape(1, 3))[0]),
        voxel_size_zyx=support.voxel_size_zyx,
    )[0]


def _outward_at(strand, end, spacing) -> tuple[np.ndarray, np.ndarray | None]:
    """The tip position and the unit vector pointing out of it, from the
    strand path that ends there."""
    spacing = np.asarray(spacing, dtype=float)
    end_um = np.asarray(end, dtype=float) * spacing
    for path, a, b, _background in strand:
        if end not in (a, b):
            continue
        points = np.asarray(path, dtype=float)
        points = points if end == a else points[::-1]
        if len(points) < 2:
            return end_um, None
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        behind = int(np.searchsorted(arc, 3.0))
        behind = min(max(behind, 1), len(points) - 1)
        vector = points[0] - points[behind]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return points[0], None
        return points[0], vector / norm
    return end_um, None


def _trim_recovered_spur(G: nx.MultiGraph, tip) -> int:
    """Delete the recovered chain hanging from *tip*, stopping at a node that
    still has another connection. A split left with nothing recovered on it
    is collapsed by :func:`delete_edge_and_collapse`."""
    removed = 0
    node = tip
    while G.has_node(node) and G.degree(node) == 1:
        neighbour = next(G.neighbors(node))
        key = next((k for k, data in G[node][neighbour].items() if data.get("recovered")), None)
        if key is None:
            break
        delete_edge_and_collapse(G, node, neighbour, key)
        removed += 1
        node = neighbour
    return removed


def _voxels_of(data) -> np.ndarray:
    voxels = data.get("voxels")
    return np.asarray([] if voxels is None else voxels, dtype=float).reshape(-1, 3)


def _oriented(path: np.ndarray, a, from_end) -> np.ndarray:
    return path if from_end == a else path[::-1]


def _without_loops_in_one_lumen(strand, support: MaskSupport):
    """``(kept, dropped)``: a strand's ``(path, a, b, background)`` paths less
    each one that closes a loop inside one lumen with the rest.

    Paths are taken longest first, so what such a loop loses is its shortest
    path. A path closing a loop round tissue is kept; the kept paths are in
    their original order.
    """
    kept_graph = nx.MultiGraph()
    keep = set()
    for i in sorted(range(len(strand)), key=lambda i: -_path_length(strand[i][0])):
        path, a, b, _background = strand[i]
        try:
            route = nx.shortest_path(kept_graph, b, a, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            route = None
        if route is not None:
            pieces = [path]
            for x, y in zip(route[:-1], route[1:]):
                data = min(kept_graph[x][y].values(), key=lambda d: d["length"])
                pieces.append(_oriented(data["path"], data["a"], x))
            if loop_inside_one_lumen(np.vstack(pieces), support):
                continue
        kept_graph.add_edge(a, b, length=_path_length(path), path=path, a=a)
        keep.add(i)
    return [strand[i] for i in sorted(keep)], len(strand) - len(keep)


class _NodeIds:
    """New numeric node ids, in the order :func:`next_node_id` would give
    them, without rescanning the graph for each. An id is never handed out
    twice, so one whose node is taken out again is not reused."""

    def __init__(self, G: nx.MultiGraph):
        self.next = next_node_id(G, set())

    def take(self) -> int:
        node, self.next = self.next, self.next + 1
        return node


#: Centreline samples a join is chosen among: the nearest this many within
#: reach, before the lumen slack ranks them.
_NEAREST_SAMPLES = 48

#: Recovered samples kept loose -- searched by their distance to the point
#: outright -- before they are put in a tree: building one costs more than
#: searching this many.
_LOOSE_SAMPLES = 2048


class _RecoveredSamples:
    """The centreline samples of the branches recovered so far in this pass,
    added a path at a time and searchable at once.

    One KD-tree over all of them, rebuilt after every strand, cost the size
    of everything recovered so far per strand -- a fifth of recovery's time
    on a 43,000-edge graph, and growing with it. Here the newest samples lie
    loose until there are :data:`_LOOSE_SAMPLES`, then go into a tree, merged
    with the trees before it while those are no bigger: trees double in
    size, each sample is indexed about ``log2(n / _LOOSE_SAMPLES)`` times,
    and a query reads every tree and the loose samples. The answer is the
    k nearest within reach, as one tree over all of them would give.
    """

    def __init__(self):
        self._loose: list[tuple[np.ndarray, np.ndarray, list]] = []
        self._loose_count = 0
        self._trees: list[tuple[cKDTree, np.ndarray, np.ndarray, list]] = []
        self.count = 0
        self.max_radius = 0.0

    def add(self, points: np.ndarray, radii: np.ndarray, owners: list) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(points):
            return
        radii = np.asarray(radii, dtype=float).reshape(-1)
        self._loose.append((points, radii, list(owners)))
        self._loose_count += len(points)
        self.count += len(points)
        self.max_radius = max(self.max_radius, float(radii.max()))
        if self._loose_count >= _LOOSE_SAMPLES:
            self._pack()

    def _pack(self) -> None:
        points = np.vstack([p for p, _r, _o in self._loose])
        radii = np.concatenate([r for _p, r, _o in self._loose])
        owners = [o for _p, _r, os in self._loose for o in os]
        self._loose, self._loose_count = [], 0
        while self._trees and len(self._trees[-1][1]) <= len(points):
            _tree, p, r, o = self._trees.pop()
            points, radii, owners = np.vstack([p, points]), np.concatenate([r, radii]), o + owners
        self._trees.append((cKDTree(points), points, radii, owners))

    def nearest(self, point_um: np.ndarray, k: int, upper: float) -> list:
        """The *k* nearest samples within *upper* of *point_um*, nearest
        first, each ``(distance, point, radius, owner)``."""
        point_um = np.asarray(point_um, dtype=float).reshape(3)
        found = []
        for tree, points, radii, owners in self._trees:
            distance, nearest = tree.query(point_um, k=min(k, len(points)), distance_upper_bound=upper)
            for d, i in zip(np.atleast_1d(distance), np.atleast_1d(nearest)):
                if np.isfinite(d):
                    found.append((float(d), points[int(i)], float(radii[int(i)]), owners[int(i)]))
        for points, radii, owners in self._loose:
            distance = np.linalg.norm(points - point_um, axis=1)
            for i in np.flatnonzero(distance < upper):
                found.append((float(distance[i]), points[i], float(radii[i]), owners[i]))
        found.sort(key=lambda item: item[0])
        return found[:k]


class _Attachments:
    """Where a recovered end may join the network: the centreline sample
    minimising ``distance - radius(sample)``, within *reach_um* of the
    covered lumen round it. Edges split by an earlier attachment are
    followed to the sub-edge now holding the point."""

    def __init__(self, G, index: EdgeSampleIndex, radii: np.ndarray, reach_um: float):
        self.G = G
        self.index = index
        self.radii = radii
        self.reach_um = float(reach_um)
        self.upper = (float(radii.max()) if len(radii) else 0.0) + self.reach_um
        self.recovered = _RecoveredSamples()

    def targets(self, point_um: np.ndarray, reach_um: float, limit: int = _JOIN_TARGETS):
        """Up to *limit* ``(slack, point, owner)`` attachments within
        *reach_um* of slack, nearest lumen first and one per edge. Samples
        of branches recovered earlier in this pass are included."""
        found = []
        found.extend(self._ranked(
            self.index.tree, self.index.points, self.radii, self.index.owners, point_um, reach_um
        ))
        found.extend(self._ranked_recovered(point_um, reach_um))
        found.sort(key=lambda item: item[0])
        kept, seen = [], set()
        for slack, point, owner in found:
            if owner in seen:
                continue
            seen.add(owner)
            kept.append((slack, point, owner))
            if len(kept) >= limit:
                break
        return kept

    def remember(self, path, owner, support: MaskSupport) -> None:
        """Index *path* so a later uncovered piece can join this branch."""
        from haemolynx.preprocessing.bridge_mask_support import _densify

        samples = _densify(np.asarray(path, dtype=float).reshape(-1, 3), 1.0)
        if len(samples) == 0:
            return
        radii = np.asarray(support.radius(samples), dtype=float).reshape(-1)
        self.recovered.add(samples, radii, [owner] * len(samples))

    def _ranked_recovered(self, point_um, reach_um):
        """:meth:`_ranked` over the samples recovered so far."""
        samples = self.recovered
        if samples.count == 0:
            return []
        upper = samples.max_radius + float(reach_um)
        k = min(_NEAREST_SAMPLES, samples.count)
        ranked = []
        for distance, point, radius, owner in samples.nearest(point_um, k, upper + 1e-9):
            slack = float(distance) - float(radius)
            if slack <= float(reach_um):
                ranked.append((slack, point, owner))
        return ranked

    @staticmethod
    def _ranked(tree, points, radii, owners, point_um, reach_um):
        if tree is None or len(points) == 0:
            return []
        radii = np.asarray(radii, dtype=float)
        upper = float(radii.max()) + float(reach_um)
        k = min(_NEAREST_SAMPLES, len(points))
        distance, nearest = tree.query(point_um, k=k, distance_upper_bound=upper + 1e-9)
        ranked = []
        for d, i in zip(np.atleast_1d(distance), np.atleast_1d(nearest)):
            if not np.isfinite(d) or int(i) >= len(radii):
                continue
            slack = float(d) - float(radii[int(i)])
            if slack <= float(reach_um):
                ranked.append((slack, points[int(i)], owners[int(i)]))
        return ranked

    def nearest(self, point_um: np.ndarray):
        if self.index.tree is None:
            return None
        k = min(16, len(self.index.points))
        distance, nearest = self.index.tree.query(point_um, k=k, distance_upper_bound=self.upper)
        best = None
        for d, i in zip(np.atleast_1d(distance), np.atleast_1d(nearest)):
            if i >= len(self.radii):
                continue
            slack = float(d) - float(self.radii[i])
            if slack <= self.reach_um and (best is None or slack < best[0]):
                best = (slack, int(i))
        if best is None:
            return None
        return self.index.points[best[1]], self.index.owners[best[1]]

    def _current(self, owner, point_um):
        """*owner*, or the piece of it now holding *point_um* once an earlier
        attachment has split it; ``None`` when there is none."""
        u, v, key = owner
        if self.G.has_edge(u, v, key):
            return owner
        return self._sub_edge(u, v, point_um)

    def attach_node(self, owner, point_um, ids: _NodeIds) -> Any:
        """The node *point_um* joins the network at, splitting the edge
        holding it."""
        owner = self._current(owner, point_um)
        if owner is None:
            return None
        u, v, key = owner
        data = self.G.edges[u, v, key]
        voxels = _voxels_of(data)
        if len(voxels) > 1:
            # insert_node_on_edge cuts the path as if it ran u -> v.
            start = voxels[0]
            from_u = np.linalg.norm(start - np.asarray(self.G.nodes[u]["pos"], dtype=float))
            from_v = np.linalg.norm(start - np.asarray(self.G.nodes[v]["pos"], dtype=float))
            data["voxels"] = (voxels[::-1] if from_v < from_u else voxels).tolist()
        node = insert_node_on_edge(self.G, u, v, key, point_um, node_id=ids.next)
        if node not in (u, v):
            ids.take()
            self.G.nodes[node]["_recovery_split"] = True
        return node

    def _polyline(self, x, y, data) -> np.ndarray:
        positions = {n: np.asarray(self.G.nodes[n]["pos"], dtype=float) for n in (x, y)}
        return edge_sample_points(x, y, data, positions)

    def loop_through(self, start, owner, point_um, join: np.ndarray, reach_um: float):
        """The closed path a join from node *start* to *point_um* on *owner*
        would make with the network -- round from *start* to the nearer end
        of *owner*, along it to *point_um*, and back along *join* -- or
        ``None`` when the network holds no way round within *reach_um*."""
        owner = self._current(owner, point_um)
        if owner is None:
            return None
        u, v, key = owner
        distance, routes = nx.single_source_dijkstra(self.G, start, cutoff=reach_um, weight="length")
        along = self._polyline(u, v, self.G.edges[u, v, key])
        cut = int(np.argmin(np.linalg.norm(along - np.asarray(point_um, dtype=float), axis=1)))
        best = None
        for end, piece in ((u, along[: cut + 1]), (v, along[cut:][::-1])):
            if end not in distance:
                continue
            total = distance[end] + _path_length(piece)
            if best is None or total < best[0]:
                best = (total, routes[end], piece)
        if best is None:
            return None
        _total, route, piece = best
        pieces = []
        for x, y in zip(route[:-1], route[1:]):
            data = min(self.G[x][y].values(), key=lambda d: d.get("length", np.inf))
            pieces.append(self._polyline(x, y, data))
        pieces += [piece, np.asarray(join, dtype=float).reshape(-1, 3)[::-1]]
        return np.vstack(pieces)

    def _sub_edge(self, u, v, point_um):
        """The edge now holding *point_um*, among the edges reachable from
        *u* or *v* through nodes an earlier split added.

        Only pieces of the vessel are searched: an earlier join hangs off
        each split node too, and taking the point to be on it attached the
        next strand to that join instead of to the vessel."""
        best, best_distance = None, np.inf
        frontier, seen = [u, v], set()
        while frontier:
            node = frontier.pop()
            if node in seen or not self.G.has_node(node):
                continue
            seen.add(node)
            for _, other, key, data in self.G.edges(node, keys=True, data=True):
                # The join itself, not the recovered centreline it lands on.
                if data.get("reconnected"):
                    continue
                voxels = _voxels_of(data)
                if len(voxels):
                    distance = float(np.min(np.linalg.norm(voxels - point_um, axis=1)))
                    if distance < best_distance:
                        best, best_distance = (node, other, key), distance
                if self.G.nodes[other].get("_recovery_split"):
                    frontier.append(other)
        return best


def _continues_along(G, node, route: np.ndarray) -> bool:
    """Whether *route* leaves *node* the way that vessel was heading.

    Judged a few microns along the route, so a mask path that sets off in
    the right direction and then bends is a join, and one that turns
    straight back is not.
    """
    route = np.asarray(route, dtype=float).reshape(-1, 3)
    if len(route) < 2:
        return True
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    at = int(np.searchsorted(arc, GAP_BRIDGE_TANGENT_LENGTH_UM))
    at = min(max(at, 1), len(route) - 1)
    return gap_bridge_continues_terminal(G, node, route[at])


def _accepted_path(G, support, index, path) -> tuple[bool, float]:
    measured = support.support(path)
    if not support.accepts_support(measured):
        return False, measured.longest_background_um
    if duplicates_existing_vessel(path, G, support.inside, support.radius, index=index):
        return False, measured.longest_background_um
    return True, measured.longest_background_um


def recover_uncovered_mask_vessels(
    G: nx.MultiGraph,
    support: MaskSupport,
    *,
    attach_reach_um: float = 3.0,
    min_region_volume_um3: float = DEFAULT_MIN_REGION_VOLUME_UM3,
    min_length_um: float = DEFAULT_MIN_LENGTH_UM,
    margin_um: float = SAME_LUMEN_MARGIN_UM,
    skeleton: np.ndarray | None = None,
    coverage_reach_radii: float = CROSS_SECTION_REACH_RADII,
    two_ended: bool = False,
) -> nx.MultiGraph:
    """Add, in place, a centreline through each piece of *support*'s mask
    the graph does not cover (:func:`uncovered_mask_voxels`), joined to the
    network.

    *coverage_reach_radii* is how far across its lumen a centreline covers
    the mask (``uncovered_mask_voxels``' *cross_section_reach_radii*); 0 is
    its own radius alone. *two_ended* keeps only what joins the network at
    both ends: recovered edges left hanging from a dead end are taken out
    again, until none is (:func:`keep_two_ended_recovery`). Both are for a
    graph whose interior dead ends were removed first
    (``dead_ends.remove_interior_dead_ends``): followed out to 4 radii a
    lumen counts the mask beside a neighbour as covered, and recovery as it
    is draws mostly dead ends -- on E14.5, 1,376 of the 2,024 edges it drew
    there (2026-10-21).

    With *skeleton* -- the run's own skeleton, voxel-indexed like the mask --
    a piece's centreline is the skeleton where it runs through the piece,
    and a piece the skeleton does not run through is left alone. Without
    it the mask round the piece is Lee-thinned again. On E14.5 (2026-10-13)
    re-thinning drew two thirds of its length where the skeleton has none,
    as combs and mazes across wide lumens, which the clean-up then took out
    again; from the skeleton, recovery took 38 s instead of 97, and the
    final graph had no pair of edges in one lumen (7 before) and 549
    interior dead ends (648), reaching as much of the skeleton.

    A piece is traced when it holds at least *min_region_volume_um3* of mask
    and is somewhere wider than :data:`MIN_INSCRIBED_RADIUS_VOXELS`; a path
    of it shorter than *min_length_um* with both ends free is dropped. Each
    free end is joined, by a route through the mask, to a centreline it can
    reach through that mask, splitting the edge where needed. A strand -- one
    connected run of accepted paths -- is added
    only when one of its own ends joins. A strand with no join, or whose join
    cannot be made, is left out, so recovery never adds an island. Each path
    and join must pass *support*'s mask test and not run beside an existing
    vessel in the same lumen. Every join, the first included, must also
    continue the recovered end (``graph.build.gap_bridge_continues_terminal``)
    and must not close a loop inside one lumen
    (``lumen_loops.loop_inside_one_lumen``). A loop round tissue is kept.
    Each free end may join any of the nearest vessels it can reach through
    the mask, not only the single closest sample: the closest one is often
    beside the branch rather than the vessel it continues. A free end whose
    join would run beside a vessel already in the same lumen, or close a
    loop inside one, is taken back along its recovered chain. An end that
    simply could not be routed, or whose straight chord folded back, stays:
    in a dense bed that end is still the branch. A split that join had
    opened, left with nothing on it, is collapsed. Recovered edges
    carry ``recovered=True``; the joins also
    ``reconnected=True``, ``bridge_kind="recovered"`` and
    ``bridge_background_um``.
    """
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    finest = float(spacing.min())
    if skeleton is not None and tuple(np.shape(skeleton)) != tuple(support.mask.shape):
        raise ValueError(
            f"the skeleton's shape {tuple(np.shape(skeleton))} is not the mask's "
            f"{tuple(support.mask.shape)}: recovery reads one at the other's voxels"
        )
    uncovered = uncovered_mask_voxels(
        G, support, margin_um=margin_um, cross_section_reach_radii=coverage_reach_radii
    )
    if not len(uncovered):
        return G
    index, radii = _centreline_samples(G, support)
    attachments = _Attachments(G, index, radii, margin_um + float(attach_reach_um))
    ids = _NodeIds(G)
    pieces = added_paths = joins = dead_ends = 0
    small = thin = unattached = loops_in_lumen = 0
    join_reach = max(attachments.reach_um, _MASK_JOIN_REACH_UM)

    def _closeness(region: np.ndarray) -> float:
        """Slack from the region's middle to the network it has to join.
        Pieces nearest the network go first, so the next piece can join the
        branch just recovered."""
        centre = np.asarray(region, dtype=float).mean(axis=0) * spacing
        if attachments.index.tree is None:
            return 0.0
        distance, nearest = attachments.index.tree.query(centre, k=1)
        nearest = int(np.atleast_1d(nearest)[0])
        distance = float(np.atleast_1d(distance)[0])
        if not np.isfinite(distance) or nearest >= len(radii):
            return np.inf
        return distance - float(radii[nearest])

    for region in sorted(_components(uncovered), key=_closeness):
        if len(region) * voxel_volume < float(min_region_volume_um3):
            small += 1
            continue
        region_radius = float(np.max(support.radius(region * spacing)))
        if region_radius < MIN_INSCRIBED_RADIUS_VOXELS * finest:
            thin += 1
            continue
        pad = int(np.ceil((region_radius + float(attach_reach_um)) / finest)) + 2
        piece = _trace_piece(region, support, pad, skeleton=skeleton)
        degree: dict[tuple[int, ...], int] = {}
        for a, b in piece.ends:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        paths = []
        for path, (a, b) in zip(piece.paths, piece.ends):
            # A path ending where it starts is Lee thinning round a hole in
            # the mask; as an edge it would be a self-loop, which carries no
            # flow and which the conductance matrix refuses.
            if a == b:
                continue
            if degree[a] == 1 and degree[b] == 1 and _path_length(path) < float(min_length_um):
                continue
            ok, background = _accepted_path(G, support, index, path)
            if ok:
                paths.append((path, a, b, background))
        if not paths or sum(_path_length(p[0]) for p in paths) < float(min_length_um):
            continue
        for strand in _strands(paths):
            # A free end only ever has one path, and a path closing a loop
            # has two at each end, so dropping one leaves the free ends free.
            strand, dropped = _without_loops_in_one_lumen(strand, support)
            loops_in_lumen += dropped
            ends_used = {}
            for _path, a, b, _bg in strand:
                for end in (a, b):
                    ends_used[end] = ends_used.get(end, 0) + 1
            # Free ends, nearest the network first. Several vessels each,
            # not only the closest sample.
            candidates = []
            for end, count in ends_used.items():
                if count != 1:
                    continue
                end_um = np.asarray(end, dtype=float) * spacing
                options = attachments.targets(end_um, join_reach)
                if options:
                    candidates.append((options[0][0], end, end_um, options))
            if not candidates:
                unattached += 1
                continue
            candidates.sort(key=lambda candidate: candidate[0])
            node_of: dict[tuple[int, ...], Any] = {}
            created: list[Any] = []

            def node_for(end):
                if end not in node_of:
                    node = ids.take()
                    G.add_node(node, pos=np.asarray(end, dtype=float) * spacing)
                    node_of[end] = node
                    created.append(node)
                return node_of[end]

            added_edges = []
            for path, a, b, background in strand:
                u, v = node_for(a), node_for(b)
                key = G.add_edge(
                    u, v,
                    length=_path_length(path), voxels=path.tolist(),
                    recovered=True, bridge_background_um=float(background),
                )
                added_edges.append((u, v, key))
            made = []
            joined: set[tuple[int, ...]] = set()
            spur_ends: set[tuple[int, ...]] = set()
            for _slack, end, end_um, options in candidates:
                refused_spur = False
                for _option_slack, point, owner in options:
                    join = route_through_mask(
                        support.mask, end_um, point, spacing, pad=_RECOVERY_ROUTE_PAD
                    )
                    if join is None or not _continues_along(G, node_for(end), join):
                        continue
                    loop = attachments.loop_through(
                        node_for(end), owner, point, join, LOOP_SEARCH_UM
                    )
                    if loop is not None and loop_inside_one_lumen(loop, support):
                        loops_in_lumen += 1
                        refused_spur = True
                        continue
                    measured = support.support(join)
                    if not support.accepts_support(measured):
                        continue
                    if duplicates_existing_vessel(
                        join, G, support.inside, support.radius, index=index
                    ):
                        refused_spur = True
                        continue
                    target = attachments.attach_node(owner, point, ids)
                    if target is None:
                        continue
                    join = np.vstack([join[:-1], np.asarray(G.nodes[target]["pos"], dtype=float)])
                    background = measured.longest_background_um
                    key = G.add_edge(
                        node_for(end), target,
                        length=_path_length(join), voxels=join.tolist(),
                        reconnected=True, recovered=True, bridge_kind="recovered",
                        bridge_background_um=float(background),
                    )
                    made.append((end, join))
                    joined.add(end)
                    joins += 1
                    refused_spur = False
                    break
                if refused_spur and end not in joined:
                    spur_ends.add(end)
            if not made:
                # attach_node returns before it changes the graph, so the
                # strand's own edges are the only thing to take back.
                for u, v, key in added_edges:
                    if G.has_edge(u, v, key):
                        G.remove_edge(u, v, key)
                for node in created:
                    if G.has_node(node) and G.degree(node) == 0:
                        G.remove_node(node)
                unattached += 1
                continue
            for end, count in ends_used.items():
                if count != 1 or end in joined or end not in node_of or end not in spur_ends:
                    continue
                if not G.has_node(node_of[end]) or G.degree(node_of[end]) != 1:
                    continue
                tip, outward = _outward_at(strand, end, spacing)
                if _leaves_image_at(tip, support, outward):
                    continue
                if _trim_recovered_spur(G, node_of[end]) > 0:
                    dead_ends += 1
            # The strand's own edges and its joins all touch its nodes, so
            # they are read from those nodes (ones a trimmed spur took out are
            # skipped), not by scanning the whole graph for every strand.
            remaining, seen = [], set()
            for u, v, key, data in G.edges(list(node_of.values()), keys=True, data=True):
                if data.get("recovered") and edge_id(u, v, key) not in seen:
                    seen.add(edge_id(u, v, key))
                    remaining.append((u, v, key, data))
            kept_joins = [
                join for end, join in made
                if G.has_node(node_of[end]) and G.degree(node_of[end]) > 1
            ]
            joins -= len(made) - len(kept_joins)
            if not remaining:
                unattached += 1
                continue
            for u, v, key, data in remaining:
                path = np.asarray(data["voxels"], dtype=float)
                index.add(path)
                if not data.get("reconnected"):
                    attachments.remember(path, edge_id(u, v, key), support)
            added_paths += sum(1 for _u, _v, _key, data in remaining if not data.get("reconnected"))
            pieces += 1
    for node in list(G.nodes):
        G.nodes[node].pop("_recovery_split", None)
    if two_ended:
        hanging, hanging_um = keep_two_ended_recovery(G)
        G.graph["recovered_one_ended_removed"] = int(G.graph.get("recovered_one_ended_removed", 0)) + hanging
        logger.info(
            "Mask recovery, two-ended: took out %d recovered edge(s), %.0f um, joined to the network "
            "at one end only", hanging, hanging_um,
        )
    G.graph["recovered_joins"] = int(G.graph.get("recovered_joins", 0)) + joins
    G.graph["recovered_dead_ends_removed"] = int(G.graph.get("recovered_dead_ends_removed", 0)) + dead_ends
    logger.info(
        "Mask recovery: %d uncovered mask voxel(s); traced %d piece(s) (%d path(s), %d join(s)); "
        "skipped %d below %.3g um^3, %d too thin to be a vessel, %d with no end to join; "
        "left out %d path(s) and join(s) closing a loop inside one lumen; "
        "removed %d recovered dead end(s) that could not join",
        len(uncovered), pieces, added_paths, joins, small, float(min_region_volume_um3), thin, unattached,
        loops_in_lumen, dead_ends,
    )
    return G


def keep_two_ended_recovery(G: nx.MultiGraph) -> tuple[int, float]:
    """Take out, in place, every recovered edge (``recovered=True``) hanging
    from a dead end, again and again until none is: what stays of recovery
    joins the network at both ends. Dead ends on edges recovery did not draw
    stay. Returns how many edges went, and their length (um)."""
    removed = 0
    length = 0.0
    while True:
        tips = [
            node for node in G.nodes
            if G.degree(node) == 1
            and all(data.get("recovered") for *_e, data in G.edges(node, data=True))
        ]
        if not tips:
            return removed, length
        for node in tips:
            if node not in G or G.degree(node) != 1:
                continue
            for u, v, key, data in list(G.edges(node, keys=True, data=True)):
                length += float(data.get("length") or 0.0)
                G.remove_edge(u, v, key)
                removed += 1
            G.remove_node(node)
        G.remove_nodes_from([node for node in list(G.nodes) if G.degree(node) == 0])


#: An edge this far out of the mask, or further, is the floating strand:
#: most of its centreline is background. A real vessel that merely clips a
#: corner stays.
_OFF_MASK_FRACTION = 0.5


def remove_edges_off_the_mask(G: nx.MultiGraph, support: MaskSupport) -> nx.MultiGraph:
    """Drop, in place, edges whose centreline lies mostly outside the mask.

    A short dropout the bridge test already allows stays, and so does a
    vessel that only leaves the mask in passing. An edge with less than
    half its length in the mask is the strand drawn through empty space;
    it goes, and the two sides are not joined back across the gap. Nodes
    left with no edge are removed.
    """
    doomed = []
    for u, v, key, data in G.edges(keys=True, data=True):
        voxels = data.get("voxels")
        if voxels is not None and len(voxels) >= 2:
            path = voxels
        else:
            path = [
                np.asarray(G.nodes[u]["pos"], dtype=float),
                np.asarray(G.nodes[v]["pos"], dtype=float),
            ]
        measured = support.support(path)
        if measured.inside_fraction < _OFF_MASK_FRACTION and not support.accepts_support(measured):
            doomed.append((u, v, key))
    for u, v, key in doomed:
        if G.has_edge(u, v, key):
            G.remove_edge(u, v, key)
    for node in [n for n in G.nodes if G.degree(n) == 0]:
        G.remove_node(node)
    removed = len(doomed)
    G.graph["edges_off_mask_removed"] = int(G.graph.get("edges_off_mask_removed", 0)) + removed
    if removed:
        logger.info("Removed %d edge(s) that do not lie in the segmented mask", removed)
    return G


__all__ = [
    "DEFAULT_MIN_LENGTH_UM",
    "DEFAULT_MIN_REGION_VOLUME_UM3",
    "REGION_SAMPLE_VOXELS",
    "UncoveredRegion",
    "keep_two_ended_recovery",
    "recover_uncovered_mask_vessels",
    "remove_edges_off_the_mask",
    "uncovered_mask_regions",
    "uncovered_mask_voxels",
]
