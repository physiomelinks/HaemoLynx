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
    edge_sample_points,
    next_node_id,
)
from .build import gap_bridge_continues_terminal
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


def _on_image_face(point_um: np.ndarray, support: MaskSupport, outward: np.ndarray | None = None) -> bool:
    """Whether *point_um* is a vessel the image cut through.

    A tip within one voxel of the finest axis of the volume boundary is.
    So is one whose mask, walked along *outward*, reaches that boundary
    within four voxels of the coarsest axis before it reaches background.
    A centreline often stops a few voxels short of a face the tube itself
    runs into; a tip in the middle of the volume, however continuous the
    mask beyond it, does not.
    """
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    extent = (np.asarray(support.mask.shape, dtype=float) - 1.0) * spacing
    point = np.asarray(point_um, dtype=float)
    if float(np.min(np.minimum(point, extent - point))) <= float(spacing.min()) + 1e-6:
        return True
    if outward is None:
        return False
    direction = np.asarray(outward, dtype=float)
    limits = []
    for axis in range(3):
        if direction[axis] > 1e-8:
            limits.append((extent[axis] - point[axis]) / direction[axis])
        elif direction[axis] < -1e-8:
            limits.append(point[axis] / -direction[axis])
    if not limits:
        return False
    reach = float(min(limits))
    if reach > 4.0 * float(spacing.max()) + 1e-6:
        return False
    if reach <= float(spacing.min()) + 1e-6:
        return True
    step = float(spacing.min())
    samples = point + (np.arange(1, int(np.ceil(reach / step)) + 1) * step)[:, None] * direction
    # The last step can land just past the boundary; that voxel is not the face.
    samples = samples[np.all((samples >= -1e-6) & (samples <= extent + 1e-6), axis=1)]
    return len(samples) > 0 and bool(np.all(support.inside(samples)))


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
                if data.get("recovered"):
                    continue
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
    where needed. A strand -- one connected run of accepted paths -- is added
    only when one of its own ends joins. A strand with no join, or whose join
    cannot be made, is left out, so recovery never adds an island. Each path
    and join must pass *support*'s mask test and not run beside an existing
    vessel in the same lumen. Every join, the first included, must also
    continue the recovered end (``graph.build.gap_bridge_continues_terminal``)
    and must not close a loop inside one lumen
    (``lumen_loops.loop_inside_one_lumen``). A loop round tissue is kept.
    A free end that then has no join is taken back along its recovered chain,
    unless the mask ends there and nothing was in reach, or the end lies on
    the image face. A split that join had opened, left with nothing on it,
    is collapsed. Recovered edges carry ``recovered=True``; the joins also
    ``reconnected=True``, ``bridge_kind="recovered"`` and
    ``bridge_background_um``.
    """
    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxel_volume = float(np.prod(spacing))
    finest = float(spacing.min())
    uncovered = uncovered_mask_voxels(G, support, margin_um=margin_um)
    if not len(uncovered):
        return G
    index, radii = _centreline_samples(G, support)
    attachments = _Attachments(G, index, radii, margin_um + float(attach_reach_um))
    ids = _NodeIds(G)
    pieces = added_paths = joins = dead_ends = 0
    small = thin = unattached = loops_in_lumen = 0
    from haemolynx.graph.assemble import mask_continues_past

    mask_runs_on = mask_continues_past(support)
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
            # Free ends, nearest the network first.
            candidates = []
            for end, count in ends_used.items():
                if count != 1:
                    continue
                end_um = np.asarray(end, dtype=float) * spacing
                target = attachments.nearest(end_um)
                if target is not None:
                    candidates.append((float(np.linalg.norm(target[0] - end_um)), end, end_um, target))
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
            for _distance, end, end_um, (point, owner) in candidates:
                # Every join, including the first. One that folds back on the
                # recovered end, or that closes a loop inside one lumen, is
                # not a vessel meeting a vessel. A loop round tissue is.
                if not gap_bridge_continues_terminal(G, node_for(end), point):
                    continue
                loop = attachments.loop_through(
                    node_for(end), owner, point, np.vstack([end_um, point]), LOOP_SEARCH_UM
                )
                if loop is not None and loop_inside_one_lumen(loop, support):
                    loops_in_lumen += 1
                    continue
                join = route_through_mask(support.mask, end_um, point, spacing)
                if join is None:
                    continue
                ok, background = _accepted_path(G, support, index, join)
                if not ok:
                    continue
                target = attachments.attach_node(owner, point, ids)
                if target is None:
                    continue
                join = np.vstack([join[:-1], np.asarray(G.nodes[target]["pos"], dtype=float)])
                key = G.add_edge(
                    node_for(end), target,
                    length=_path_length(join), voxels=join.tolist(),
                    reconnected=True, recovered=True, bridge_kind="recovered",
                    bridge_background_um=float(background),
                )
                made.append((end, join))
                joined.add(end)
                joins += 1
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
            candidate_ends = {end for _distance, end, _end_um, _target in candidates}
            for end, count in ends_used.items():
                if count != 1 or end in joined or end not in node_of:
                    continue
                if not G.has_node(node_of[end]) or G.degree(node_of[end]) != 1:
                    continue
                tip, outward = _outward_at(strand, end, spacing)
                if _on_image_face(tip, support, outward):
                    continue
                # A real vessel end: the mask stops here and no branch was in reach.
                if end not in candidate_ends and (outward is None or not mask_runs_on(tip, outward)):
                    continue
                if _trim_recovered_spur(G, node_of[end]) > 0:
                    dead_ends += 1
            strand_nodes = set(node_of.values())
            remaining = [
                data for u, v, data in G.edges(data=True)
                if data.get("recovered") and (u in strand_nodes or v in strand_nodes)
            ]
            kept_joins = [
                join for end, join in made
                if G.has_node(node_of[end]) and G.degree(node_of[end]) > 1
            ]
            joins -= len(made) - len(kept_joins)
            if not remaining:
                unattached += 1
                continue
            for data in remaining:
                index.add(np.asarray(data["voxels"], dtype=float))
            added_paths += sum(1 for data in remaining if not data.get("reconnected"))
            pieces += 1
    for node in list(G.nodes):
        G.nodes[node].pop("_recovery_split", None)
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


__all__ = [
    "DEFAULT_MIN_LENGTH_UM",
    "DEFAULT_MIN_REGION_VOLUME_UM3",
    "recover_uncovered_mask_vessels",
    "uncovered_mask_voxels",
]
