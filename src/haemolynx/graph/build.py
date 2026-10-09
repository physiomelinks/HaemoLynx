"""Build vascular graph from skeleton using skan."""
import logging
import os
import time

import numpy as np
import networkx as nx
from scipy.ndimage import generate_binary_structure
from scipy.spatial import cKDTree
import heapq
from itertools import product

from scipy import sparse
from skan import csr

from ._platform import iter_python_work, map_python_work
from .cartwheel_guard import _incident_edge_items, _spoke_direction_and_length
from .reconnect import MaskBridges

logger = logging.getLogger(__name__)

#: Foreground voxels whose 26 neighbours are looked up at once when building
#: the pixel adjacency without the image: bounds the ``(n, 26)`` temporaries.
_ADJACENCY_BATCH_NODES = 500_000

#: A gap bridge stands in for a missing piece of the vessel its terminal ends,
#: so it has to leave the terminal the way that vessel was heading. One that
#: turns further than this from either terminal's end direction folds back:
#: most often onto another spur of the junction the terminal's own spur left,
#: which degree-2 merging then turns into one hairpin edge running out and
#: straight back. Off the voxel lattice's 90 degrees, so a staircase step
#: does not decide it.
MAX_GAP_BRIDGE_TURN_DEG = 80.0

#: How far back along a terminal's own centreline its end direction is read:
#: a few voxel steps, so one staircase step at the tip does not set it.
GAP_BRIDGE_TANGENT_LENGTH_UM = 5.0

#: A terminal at the end of less centreline than this is not bridged at all.
#: skan splits the skeleton at every junction voxel, which leaves 1-3 um arms
#: round a junction's voxel cluster: not vessel ends, and pointing whichever of
#: the lattice's 26 directions the cluster happened to take. Judged by
#: direction alone they reach out to terminals up to the reconnect threshold
#: away; on the E14.5 MCA stack 3,225 of the 3,743 bridges built without a
#: direction check started from one.
MIN_GAP_BRIDGE_END_LENGTH_UM = 3.0


def _terminal_end(G, node, tangent_length_um):
    """``(direction, length)``: the unit vector out of the open end of the
    vessel *node* terminates (``None`` when its edge gives none), and that
    edge's length -- or ``None`` unless *node* has exactly one edge."""
    items = list(_incident_edge_items(G, node))
    if len(items) != 1:
        return None
    neighbor, _key, data = items[0]
    inward, length = _spoke_direction_and_length(
        G, node, neighbor, data, tangent_length_um=tangent_length_um
    )
    return (None if inward is None else -inward), length


def _bridge_continues_end(end, from_pos, to_pos, max_turn_deg, min_end_length_um):
    if end is None:
        return True
    direction, length = end
    if length < min_end_length_um:
        return False
    if direction is None:
        return True
    chord = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    norm = float(np.linalg.norm(chord))
    if norm <= 0.0:
        return True
    return float(chord @ direction) / norm >= np.cos(np.radians(max_turn_deg))


def gap_bridge_continues_terminal(
    G,
    node,
    to_pos,
    *,
    max_turn_deg=MAX_GAP_BRIDGE_TURN_DEG,
    min_end_length_um=MIN_GAP_BRIDGE_END_LENGTH_UM,
    tangent_length_um=GAP_BRIDGE_TANGENT_LENGTH_UM,
):
    """Whether a straight bridge from *node* to *to_pos* continues the vessel
    *node* ends: that vessel is at least *min_end_length_um* of centreline,
    and the bridge turns at most *max_turn_deg* from the way it was heading.

    True for a node that does not end exactly one edge -- a junction, a vessel
    part way along, an orphan -- which has no end direction to continue; and
    for a long enough edge that gives no direction, so a missing direction
    never blocks a bridge.
    """
    return _bridge_continues_end(
        _terminal_end(G, node, tangent_length_um),
        G.nodes[node]["pos"],
        to_pos,
        max_turn_deg,
        min_end_length_um,
    )


def gap_bridge_pairs_following_ends(
    G,
    pairs,
    *,
    max_turn_deg=MAX_GAP_BRIDGE_TURN_DEG,
    min_end_length_um=MIN_GAP_BRIDGE_END_LENGTH_UM,
    tangent_length_um=GAP_BRIDGE_TANGENT_LENGTH_UM,
):
    """The ``(dist, src, tgt)`` candidate bridges that continue the vessels
    at both their ends (see :func:`gap_bridge_continues_terminal`).

    Filtered before any is added, so a terminal whose nearest partner folds
    back stays free for the next one along.
    """
    ends = {}

    def end(node):
        if node not in ends:
            ends[node] = _terminal_end(G, node, tangent_length_um)
        return ends[node]

    kept = []
    for dist, src, tgt in pairs:
        src_pos, tgt_pos = G.nodes[src]["pos"], G.nodes[tgt]["pos"]
        if _bridge_continues_end(
            end(src), src_pos, tgt_pos, max_turn_deg, min_end_length_um
        ) and _bridge_continues_end(end(tgt), tgt_pos, src_pos, max_turn_deg, min_end_length_um):
            kept.append((dist, src, tgt))
    return kept


class _SkeletonPaths:
    """The part of ``skan.csr.Skeleton`` the graph builder reads.

    ``n_paths`` and ``path_coordinates`` only, over the same path matrix and
    coordinates skan would have built, so a caller cannot tell them apart.
    """

    def __init__(self, paths, coordinates):
        self.paths = paths
        self.coordinates = coordinates
        self.n_paths = paths.shape[0]

    def path(self, index):
        start, stop = self.paths.indptr[index:index + 2]
        return self.paths.indices[start:stop]

    def path_coordinates(self, index):
        return self.coordinates[self.path(index)]


def _foreground_neighbours(coords, shape):
    """Each foreground voxel's 26-connected foreground neighbours, a batch of
    voxels at a time.

    Yields ``(source, neighbour, step_length)`` arrays: indices into *coords*
    and the Euclidean length of each step, in voxels. Neighbours are found by
    ``searchsorted`` over the raveled coordinates, so nothing the size of the
    volume is allocated.
    """
    n = len(coords)
    ndim = len(shape)
    padded_shape = np.asarray(shape, dtype=np.int64) + 2
    strides = np.ones(ndim, dtype=np.int64)
    for axis in range(ndim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * padded_shape[axis + 1]
    # C-order argwhere output is already sorted by raveled index, padded or not.
    keys = (coords.astype(np.int64) + 1) @ strides

    offsets = np.array(
        [step for step in product((-1, 0, 1), repeat=ndim) if any(step)],
        dtype=np.int64,
    )
    step_keys = offsets @ strides
    step_lengths = np.linalg.norm(offsets, axis=1)

    for start in range(0, n, _ADJACENCY_BATCH_NODES):
        stop = min(start + _ADJACENCY_BATCH_NODES, n)
        neighbour_keys = keys[start:stop, None] + step_keys[None, :]
        found = np.searchsorted(keys, neighbour_keys)
        np.minimum(found, n - 1, out=found)
        present = keys[found] == neighbour_keys
        source, step = np.nonzero(present)
        yield source + start, found[source, step], step_lengths[step]


def skeleton_has_path(skeleton) -> bool:
    """Whether any two foreground voxels of *skeleton* touch (26-connected).

    skan's paths run between touching voxels, so a skeleton without such a
    pair -- empty, or only isolated voxels -- has no path, and
    ``skan.csr.Skeleton`` cannot be built from it at all (scipy refuses the
    empty path matrix: "index pointer size 0 should be 1"). Stops at the
    first batch of voxels with a neighbour, so on a real skeleton this costs
    little more than finding the foreground.
    """
    coords = np.argwhere(skeleton)
    return any(len(source) for source, _, _ in _foreground_neighbours(coords, skeleton.shape))


def _no_skeleton_paths(ndim):
    return _SkeletonPaths(sparse.csr_matrix((0, 0)), np.empty((0, ndim), dtype=np.int64))


def _skeleton_adjacency_from_coordinates(coords, shape):
    """The 26-connected pixel graph skan builds, from foreground coordinates.

    Edge weights are the Euclidean length of each step, in voxels, as skan's
    own ``pixel_graph`` gives them for a boolean skeleton. Built from
    :func:`_foreground_neighbours`, so nothing the size of the volume is
    allocated: skan instead copies the image to bool and pads it, two
    whole-volume temporaries.
    """
    n = len(coords)
    rows, cols, data = [], [], []
    for source, neighbour, step_length in _foreground_neighbours(coords, shape):
        rows.append(source)
        cols.append(neighbour)
        data.append(step_length)
    adjacency = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return adjacency.tocsr()


def skan_skeleton(skeleton, *, use_memmap=False):
    """The skan skeleton the graph builder walks, optionally without the image.

    With *use_memmap* off this is ``skan.csr.Skeleton(skeleton)``. With it on,
    the same paths are built from the foreground coordinates alone: skan's
    constructor casts the whole volume to bool and pads it, which on a
    memory-mapped volume means two in-RAM copies of the thing that was mapped
    to keep it out of RAM. Only ``n_paths`` and ``path_coordinates`` are
    provided in that mode, the two things the graph builder reads.

    Either way, a skeleton with no two voxels touching (see
    :func:`skeleton_has_path`) has no paths, where skan's own constructor
    raises on it.
    """
    if not use_memmap:
        if not skeleton_has_path(skeleton):
            return _no_skeleton_paths(skeleton.ndim)
        return csr.Skeleton(skeleton)

    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return _no_skeleton_paths(skeleton.ndim)
    adjacency = _skeleton_adjacency_from_coordinates(coords, skeleton.shape)
    if adjacency.nnz == 0:
        return _no_skeleton_paths(skeleton.ndim)
    adjacency = csr._mst_junctions(adjacency)
    paths = csr._build_skeleton_path_graph(csr.csr_to_nbgraph(adjacency))
    return _SkeletonPaths(paths, coords)


def build_graph_segment_skan_stitched_loops(
    sk,
    skeleton_image,
    debug=False,
    reconnect_threshold=3.0,
    max_voxel_graph_size=100000,
    use_spatial_index=True,
    voxel_size=(1.0, 1.0, 1.0),
    max_bridge_turn_deg=MAX_GAP_BRIDGE_TURN_DEG,
    mask_support=None,
):
    """Build NetworkX graph from skan Skeleton with loop detection and terminal reconnection.

    ``voxel_size`` is the spacing of each array axis in canonical ``(z, y, x)``
    order, so node ``pos`` and edge ``voxels`` come out as physical ``(z, y, x)``
    coordinates. Image metadata reports ``(x, y, z)``; convert with
    ``haemolynx.io.voxel_size_zyx_from_xyz`` first.

    Terminals within ``reconnect_threshold`` of each other are joined by a
    straight gap bridge, nearest pairs first, unless the bridge turns more than
    ``max_bridge_turn_deg`` from either terminal's end direction (see
    :data:`MAX_GAP_BRIDGE_TURN_DEG`) or either terminal ends less than
    :data:`MIN_GAP_BRIDGE_END_LENGTH_UM` of centreline; ``None`` bridges every
    pair.

    With *mask_support* (a ``preprocessing.MaskSupport``) a bridge is drawn
    only where ``reconnect.MaskBridges`` accepts it: routed through the mask,
    through no more background than it allows, and not beside another edge
    in the same lumen. A terminal refused one bridge may still take the next
    nearest.
    """
    if sk is None or skeleton_image is None:
        raise ValueError("sk and skeleton_image cannot be None")

    if sk.n_paths == 0:
        logger.warning("No paths found in skeleton")
        return nx.MultiGraph(), [], set()

    paths = [(i, sk.path_coordinates(i)) for i in range(sk.n_paths)]
    skel = skeleton_image
    ndim = skel.ndim
    foreground = np.argwhere(skel)
    voxel_loops = []

    if len(foreground) <= max_voxel_graph_size:
        offsets = np.argwhere(generate_binary_structure(ndim, 1)) - 1
        voxel_graph = nx.Graph()

        def process_pt_batch(pts_batch):
            edges = []
            for pt in pts_batch:
                for off in offsets:
                    nb = pt + off
                    if np.all(nb >= 0) and np.all(nb < skel.shape) and skel[tuple(nb)]:
                        edges.append((tuple(pt), tuple(nb)))
            return edges

        batch_size = min(1000, len(foreground))
        batches = [
            foreground[i : i + batch_size]
            for i in range(0, len(foreground), batch_size)
        ]
        max_workers = min(4, os.cpu_count() or 1)
        for batch_edges in iter_python_work(process_pt_batch, batches, max_workers):
            voxel_graph.add_edges_from(batch_edges)
        logger.info(
            "Voxel graph built: %d nodes, %d edges. Running cycle_basis...",
            voxel_graph.number_of_nodes(), voxel_graph.number_of_edges(),
        )
        try:
            voxel_loops = nx.cycle_basis(voxel_graph)
            logger.info("cycle_basis complete: found %d loops", len(voxel_loops))
            if debug:
                logger.debug("Found %d voxel loops", len(voxel_loops))
        except Exception as e:
            logger.warning("Loop detection failed: %s", e)
            voxel_loops = []
    else:
        if debug:
            logger.warning(
                "Skeleton too large (%d voxels) for loop detection", len(foreground)
            )

    t0 = time.perf_counter()
    loop_vox = set()
    for loop in voxel_loops:
        for v in loop:
            if isinstance(v, (list, tuple, np.ndarray)):
                loop_vox.add(tuple(np.round(v).astype(int)))
    logger.info("loop_vox built (%d voxels) in %.1fs", len(loop_vox), time.perf_counter() - t0)

    def make_segment_safe(pid_path):
        pid, path = pid_path
        if len(path) < 2:
            return None
        path_array = np.array(path)
        if np.any(path_array < 0) or np.any(path_array >= np.array(skel.shape)):
            if debug:
                logger.warning("Path %s contains out-of-bounds coordinates", pid)
            path_array = np.clip(path_array, 0, np.array(skel.shape) - 1)
        segment = [tuple(np.round(p).astype(int)) for p in path_array]
        unique_segment = [segment[0]]
        for i in range(1, len(segment)):
            if segment[i] != segment[i - 1]:
                unique_segment.append(segment[i])
        return unique_segment if len(unique_segment) >= 2 else None

    t0 = time.perf_counter()
    max_workers = min(4, os.cpu_count() or 1)
    segments = [s for s in map_python_work(make_segment_safe, paths, max_workers) if s]
    logger.info("Segments extracted (%d valid) in %.1fs", len(segments), time.perf_counter() - t0)

    if not segments:
        logger.warning("No valid segments found")
        return nx.MultiGraph(), voxel_loops, set()

    # Use MultiGraph so distinct vessel segments between the same two
    # junction nodes are preserved instead of overwritten.
    G = nx.MultiGraph()
    loop_edges = set()
    mapping = {}
    voxel_size_arr = np.asarray(voxel_size, dtype=float)

    for seg_idx, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        u_vox, v_vox = seg[0], seg[-1]
        uid = mapping.setdefault(u_vox, len(mapping))
        vid = mapping.setdefault(v_vox, len(mapping))
        u_pos = np.array(u_vox, dtype=float) * voxel_size_arr
        v_pos = np.array(v_vox, dtype=float) * voxel_size_arr
        if not G.has_node(uid):
            G.add_node(uid, pos=u_pos)
        if not G.has_node(vid):
            G.add_node(vid, pos=v_pos)
        seg_array_phys = np.array(seg, dtype=float) * voxel_size_arr
        if len(seg_array_phys) > 1:
            total_dist = float(np.sum(np.linalg.norm(np.diff(seg_array_phys, axis=0), axis=1)))
        else:
            total_dist = 0.0
        G.add_edge(
            uid,
            vid,
            length=total_dist,
            voxels=seg_array_phys.tolist(),
            segment_id=seg_idx,
        )
        if u_vox in loop_vox and v_vox in loop_vox:
            loop_edges.add(tuple(sorted([uid, vid])))

    logger.info("Graph built: %d nodes, %d edges, %d loop_edges", G.number_of_nodes(), G.number_of_edges(), len(loop_edges))
    if reconnect_threshold and reconnect_threshold > 0:
        terminals = [n for n in G.nodes if G.degree[n] == 1]
        if len(terminals) > 1:
            if use_spatial_index and len(terminals) > 10:
                terminal_coords = np.array([G.nodes[n]["pos"] for n in terminals])
                tree = cKDTree(terminal_coords)
                pairs_indices = tree.query_pairs(reconnect_threshold)
                pairs = []
                for i, j in pairs_indices:
                    src, tgt = terminals[i], terminals[j]
                    edge_norm = tuple(sorted([src, tgt]))
                    if (
                        G.has_edge(src, tgt)
                        or edge_norm in loop_edges
                        or G.degree[src] > 1
                        or G.degree[tgt] > 1
                    ):
                        continue
                    dist = np.linalg.norm(terminal_coords[i] - terminal_coords[j])
                    pairs.append((dist, src, tgt))
            else:
                pairs = []
                for i, src in enumerate(terminals):
                    for j in range(i + 1, len(terminals)):
                        tgt = terminals[j]
                        edge_norm = tuple(sorted([src, tgt]))
                        if (
                            edge_norm in loop_edges
                            or G.degree[src] > 1
                            or G.degree[tgt] > 1
                        ):
                            continue
                        src_pos = np.array(G.nodes[src]["pos"])
                        tgt_pos = np.array(G.nodes[tgt]["pos"])
                        dist = np.linalg.norm(src_pos - tgt_pos)
                        if dist <= reconnect_threshold:
                            pairs.append((dist, src, tgt))
            if max_bridge_turn_deg is not None:
                candidates = len(pairs)
                pairs = gap_bridge_pairs_following_ends(
                    G, pairs, max_turn_deg=max_bridge_turn_deg
                )
                logger.info(
                    "Gap bridges: %d of %d terminal pairs within %.1f um do not continue "
                    "the vessels they would join and are not bridged",
                    candidates - len(pairs), candidates, reconnect_threshold,
                )
            heapq.heapify(pairs)
            bridges = None if mask_support is None else MaskBridges(G, mask_support, "gap")
            reconnected = 0
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                if G.has_edge(src, tgt) or G.degree[src] > 1 or G.degree[tgt] > 1:
                    continue
                src_pos = np.array(G.nodes[src]["pos"])
                tgt_pos = np.array(G.nodes[tgt]["pos"])
                if bridges is None:
                    G.add_edge(
                        src,
                        tgt,
                        length=dist,
                        voxels=[
                            src_pos.tolist(),
                            tgt_pos.tolist(),
                        ],
                        reconnected=True,
                    )
                else:
                    found = bridges.bridge(src_pos, tgt_pos)
                    if found is None:
                        continue
                    path, measured = found
                    G.add_edge(
                        src,
                        tgt,
                        length=float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
                        voxels=path.tolist(),
                        reconnected=True,
                        **bridges.attributes(measured),
                    )
                reconnected += 1
                if debug:
                    logger.debug("Reconnected %s-%s, d=%.2f", src, tgt, dist)
            if bridges is not None:
                bridges.log_summary()
            if debug and reconnected > 0:
                logger.info("Reconnected %d terminal pairs", reconnected)

    isolated_nodes = [n for n in G.nodes if G.degree[n] == 0]
    if isolated_nodes:
        G.remove_nodes_from(isolated_nodes)
        if debug:
            logger.warning("Removed %d isolated nodes", len(isolated_nodes))

    if debug:
        logger.info(
            "Final graph: %d nodes, %d edges, %d loop edges",
            G.number_of_nodes(),
            G.number_of_edges(),
            len(loop_edges),
        )
    G.graph["voxel_size"] = tuple(float(v) for v in voxel_size_arr)
    return G, voxel_loops, loop_edges
