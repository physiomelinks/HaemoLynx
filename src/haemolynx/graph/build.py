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

logger = logging.getLogger(__name__)

#: Foreground voxels whose 26 neighbours are looked up at once when building
#: the pixel adjacency without the image: bounds the ``(n, 26)`` temporaries.
_ADJACENCY_BATCH_NODES = 500_000


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


def _skeleton_adjacency_from_coordinates(coords, shape):
    """The 26-connected pixel graph skan builds, from foreground coordinates.

    Edge weights are the Euclidean length of each step, in voxels, as skan's
    own ``pixel_graph`` gives them for a boolean skeleton. Neighbours are
    found by ``searchsorted`` over the raveled coordinates, so nothing the
    size of the volume is allocated: skan instead copies the image to bool
    and pads it, two whole-volume temporaries.
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

    rows, cols, data = [], [], []
    for start in range(0, n, _ADJACENCY_BATCH_NODES):
        stop = min(start + _ADJACENCY_BATCH_NODES, n)
        neighbour_keys = keys[start:stop, None] + step_keys[None, :]
        found = np.searchsorted(keys, neighbour_keys)
        np.minimum(found, n - 1, out=found)
        present = keys[found] == neighbour_keys
        source, step = np.nonzero(present)
        rows.append(source + start)
        cols.append(found[source, step])
        data.append(step_lengths[step])
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
    """
    if not use_memmap:
        return csr.Skeleton(skeleton)

    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return _SkeletonPaths(
            sparse.csr_matrix((0, 0)), np.empty((0, skeleton.ndim), dtype=np.int64)
        )
    adjacency = _skeleton_adjacency_from_coordinates(coords, skeleton.shape)
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
):
    """Build NetworkX graph from skan Skeleton with loop detection and terminal reconnection.

    ``voxel_size`` is the spacing of each array axis in canonical ``(z, y, x)``
    order, so node ``pos`` and edge ``voxels`` come out as physical ``(z, y, x)``
    coordinates. Image metadata reports ``(x, y, z)``; convert with
    ``haemolynx.io.voxel_size_zyx_from_xyz`` first.
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
            heapq.heapify(pairs)
            reconnected = 0
            while pairs:
                dist, src, tgt = heapq.heappop(pairs)
                if G.has_edge(src, tgt) or G.degree[src] > 1 or G.degree[tgt] > 1:
                    continue
                src_pos = np.array(G.nodes[src]["pos"])
                tgt_pos = np.array(G.nodes[tgt]["pos"])
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
                reconnected += 1
                if debug:
                    logger.debug("Reconnected %s-%s, d=%.2f", src, tgt, dist)
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
