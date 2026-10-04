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
from scipy.ndimage import convolve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from haemolynx.preprocessing.bridge_mask_support import SAME_LUMEN_MARGIN_UM, MaskSupport
from haemolynx.preprocessing.memmap_support import LOW_MEMORY_BLOCK_VOXELS, slab_step

from ._helpers import EdgeSampleIndex, duplicates_existing_vessel, next_node_id
from .edit import insert_node_on_edge
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

#: Nearest centreline samples each mask voxel is judged against.
_COVERAGE_NEIGHBOURS = 4


@dataclass
class _Piece:
    """One uncovered region's centreline: paths between voxel-keyed ends."""

    paths: list[np.ndarray] = field(default_factory=list)
    ends: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)


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
) -> np.ndarray:
    """Voxel indices of the mask farther than its lumen radius + *margin_um*
    from every centreline sample of *G*: a mask voxel is covered by a sample
    ``p`` when within ``radius(p) + margin_um`` of it. Read a slab of whole
    z-slices at a time, so the mask may be a memmap."""
    mask = support.mask
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    index, radii = _centreline_samples(G, support)
    step = slab_step(mask.shape, block_voxels) if np.ndim(mask) == 3 else mask.shape[0]
    uncovered = []
    reach = float(radii.max()) + margin_um if len(radii) else 0.0
    for start in range(0, mask.shape[0], step):
        coords = np.argwhere(np.asarray(mask[start:start + step], dtype=bool))
        if not len(coords):
            continue
        coords[:, 0] += start
        if index.tree is None:
            uncovered.append(coords)
            continue
        k = min(_COVERAGE_NEIGHBOURS, len(index.points))
        distance, nearest = index.tree.query(
            coords * spacing, k=k, distance_upper_bound=reach + 1e-9, workers=-1
        )
        distance, nearest = distance.reshape(len(coords), k), nearest.reshape(len(coords), k)
        found = nearest < len(radii)
        slack = np.where(found, distance - radii[np.minimum(nearest, len(radii) - 1)], np.inf)
        uncovered.append(coords[~np.any(slack <= margin_um + 1e-9, axis=1)])
    return np.concatenate(uncovered) if uncovered else np.empty((0, np.ndim(mask)), dtype=np.intp)


def _components(coords: np.ndarray) -> list[np.ndarray]:
    """26-connected groups of voxel indices."""
    if len(coords) == 0:
        return []
    pairs = cKDTree(coords).query_pairs(r=np.sqrt(3.0) + 1e-6, output_type="ndarray")
    n = len(coords)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    return np.split(coords[order], bounds)


def _trace_piece(region: np.ndarray, support: MaskSupport, pad: int) -> _Piece:
    """The mask's own centreline (Lee thinning of the whole mask round the
    region) where it runs through *region*, as skan paths."""
    from skan import csr
    from skimage.morphology import skeletonize

    shape = np.asarray(support.mask.shape)
    lo = np.maximum(region.min(axis=0) - pad, 0)
    hi = np.minimum(region.max(axis=0) + pad + 1, shape)
    window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    skeleton = skeletonize(np.asarray(support.mask[window], dtype=bool), method="lee").astype(bool)
    inside = np.zeros_like(skeleton)
    inside[tuple((region - lo).T)] = True
    skeleton &= inside
    piece = _Piece()
    neighbours = convolve(skeleton.astype(np.uint8), np.ones((3,) * skeleton.ndim, np.uint8), mode="constant")
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


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


def _voxels_of(data) -> np.ndarray:
    voxels = data.get("voxels")
    return np.asarray([] if voxels is None else voxels, dtype=float).reshape(-1, 3)


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

    def attach_node(self, owner, point_um, reserved) -> Any:
        """The node *point_um* joins the network at, splitting the edge
        holding it."""
        u, v, key = owner
        if not self.G.has_edge(u, v, key):
            owner = self._sub_edge(u, v, point_um)
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
        node = insert_node_on_edge(self.G, u, v, key, point_um, reserved_ids=reserved)
        if node not in (u, v):
            self.G.nodes[node]["_recovery_split"] = True
        return node

    def _sub_edge(self, u, v, point_um):
        """The edge now holding *point_um*, among the edges reachable from
        *u* or *v* through nodes an earlier split added."""
        best, best_distance = None, np.inf
        frontier, seen = [u, v], set()
        while frontier:
            node = frontier.pop()
            if node in seen or not self.G.has_node(node):
                continue
            seen.add(node)
            for _, other, key, data in self.G.edges(node, keys=True, data=True):
                voxels = _voxels_of(data)
                if len(voxels):
                    distance = float(np.min(np.linalg.norm(voxels - point_um, axis=1)))
                    if distance < best_distance:
                        best, best_distance = (node, other, key), distance
                if self.G.nodes[other].get("_recovery_split"):
                    frontier.append(other)
        return best


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
) -> nx.MultiGraph:
    """Add, in place, a centreline through each piece of *support*'s mask
    the graph does not cover (:func:`uncovered_mask_voxels`), joined to the
    network.

    A piece is traced when it holds at least *min_region_volume_um3* of mask
    and is somewhere wider than :data:`MIN_INSCRIBED_RADIUS_VOXELS`; a path
    of it shorter than *min_length_um* with both ends free is dropped. Each
    free end is joined, by a route through the mask, to the centreline whose
    covered lumen it lies within *attach_reach_um* of -- splitting that edge
    where needed. A piece is only added when at least one end joins, and
    each path and join must pass *support*'s mask test and not run beside an
    existing vessel in the same lumen. Recovered edges carry
    ``recovered=True``; the joins also ``reconnected=True``,
    ``bridge_kind="recovered"`` and ``bridge_background_um``.
    """
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    finest = float(spacing.min())
    uncovered = uncovered_mask_voxels(G, support, margin_um=margin_um)
    if not len(uncovered):
        return G
    index, radii = _centreline_samples(G, support)
    attachments = _Attachments(G, index, radii, margin_um + float(attach_reach_um))
    reserved: set[Any] = set()
    pieces = added_paths = joins = 0
    small = thin = unattached = 0
    for region in _components(uncovered):
        if len(region) * voxel_volume < float(min_region_volume_um3):
            small += 1
            continue
        region_radius = float(np.max(support.radius(region * spacing)))
        if region_radius < MIN_INSCRIBED_RADIUS_VOXELS * finest:
            thin += 1
            continue
        pad = int(np.ceil((region_radius + float(attach_reach_um)) / finest)) + 2
        piece = _trace_piece(region, support, pad)
        degree: dict[tuple[int, ...], int] = {}
        for a, b in piece.ends:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        paths = []
        for path, (a, b) in zip(piece.paths, piece.ends):
            if degree[a] == 1 and degree[b] == 1 and _path_length(path) < float(min_length_um):
                continue
            ok, background = _accepted_path(G, support, index, path)
            if ok:
                paths.append((path, a, b, background))
        if not paths or sum(_path_length(p[0]) for p in paths) < float(min_length_um):
            continue
        ends_used = {}
        for _path, a, b, _bg in paths:
            for end in (a, b):
                ends_used[end] = ends_used.get(end, 0) + 1
        plans = []
        for end, count in ends_used.items():
            if count != 1:
                continue
            end_um = np.asarray(end, dtype=float) * spacing
            target = attachments.nearest(end_um)
            if target is None:
                continue
            point, owner = target
            join = route_through_mask(support.mask, end_um, point, spacing)
            if join is None:
                continue
            ok, background = _accepted_path(G, support, index, join)
            if ok:
                plans.append((end, point, owner, join, background))
        if not plans:
            unattached += 1
            continue
        node_of: dict[tuple[int, ...], Any] = {}

        def node_for(end):
            if end not in node_of:
                node = next_node_id(G, reserved)
                reserved.add(node)
                G.add_node(node, pos=np.asarray(end, dtype=float) * spacing)
                node_of[end] = node
            return node_of[end]

        for path, a, b, background in paths:
            G.add_edge(
                node_for(a), node_for(b),
                length=_path_length(path), voxels=path.tolist(),
                recovered=True, bridge_background_um=float(background),
            )
            index.add(path)
            added_paths += 1
        for end, point, owner, join, background in plans:
            target = attachments.attach_node(owner, point, reserved)
            if target is None:
                continue
            join = np.vstack([join[:-1], np.asarray(G.nodes[target]["pos"], dtype=float)])
            G.add_edge(
                node_for(end), target,
                length=_path_length(join), voxels=join.tolist(),
                reconnected=True, recovered=True, bridge_kind="recovered",
                bridge_background_um=float(background),
            )
            index.add(join)
            joins += 1
        pieces += 1
    for node in list(G.nodes):
        G.nodes[node].pop("_recovery_split", None)
    logger.info(
        "Mask recovery: %d uncovered mask voxel(s); traced %d piece(s) (%d path(s), %d join(s)); "
        "skipped %d below %.3g um^3, %d too thin to be a vessel, %d with no end to join",
        len(uncovered), pieces, added_paths, joins, small, float(min_region_volume_um3), thin, unattached,
    )
    return G


__all__ = [
    "DEFAULT_MIN_LENGTH_UM",
    "DEFAULT_MIN_REGION_VOLUME_UM3",
    "recover_uncovered_mask_vessels",
    "uncovered_mask_voxels",
]
