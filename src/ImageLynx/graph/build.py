"""Build vascular graph from skeleton using skan."""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import networkx as nx
from scipy.ndimage import (
    generate_binary_structure,
    label,
    find_objects,
)
from scipy.spatial import cKDTree
import heapq

from ._helpers import RECONNECT_THRESHOLD_UM

logger = logging.getLogger(__name__)


def build_graph_segment_skan_stitched_loops(
    sk,
    skeleton_image,
    debug=False,
    reconnect_threshold=RECONNECT_THRESHOLD_UM,
    max_voxel_graph_size=1000000, # Increased limit because it's now instant
    use_spatial_index=True,
    use_padded_slicing=True, # Ignored, kept for API compatibility
    padding=3, # Ignored, kept for API compatibility
    voxel_size=(1.0, 1.0, 1.0),
):
    """Build NetworkX graph from skan Skeleton with loop detection and terminal reconnection.

    reconnect_threshold is in MICRONS, not voxels: terminals are matched on node "pos",
    which this function stores in physical units. See RECONNECT_THRESHOLD_UM in _helpers.
    """
    if sk is None or skeleton_image is None:
        raise ValueError("sk and skeleton_image cannot be None")

    if sk.n_paths == 0:
        logger.warning("No paths found in skeleton")
        return nx.MultiGraph(), [], set()

    paths = [(i, sk.path_coordinates(i)) for i in range(sk.n_paths)]
    skel = skeleton_image
    voxel_loops = []

    # ARCHITECTURAL SPEEDUP: Use skan's internal CSR graph directly
    # This eliminates the need to build a manual NetworkX voxel graph in a Python loop.
    n_voxels = sk.graph.shape[0]
    if n_voxels > 0:
        if n_voxels > max_voxel_graph_size:
            logger.warning(
                "Skeleton contains %d voxels, exceeding limit for loop detection (%d)",
                n_voxels, max_voxel_graph_size
            )
        else:
            try:
                import igraph as ig
                t_start = time.perf_counter()
                
                # Convert CSR matrix to igraph edges
                # Since sk.graph is symmetric, we take the upper triangle to avoid duplicate edges
                rows, cols = sk.graph.nonzero()
                upper = rows < cols
                edges = list(zip(rows[upper], cols[upper]))
                
                ig_vox = ig.Graph(n=n_voxels, edges=edges)
                
                # Find biconnected components of size >= 3 (loop clusters)
                # igraph's biconnected_components() is implemented in C and is O(V+E)
                bc = ig_vox.biconnected_components()
                for comp in bc:
                    if len(comp) >= 3:
                        # Map internal skan indices back to global (z,y,x) coordinates
                        loop_coords = sk.coordinates[list(comp)]
                        voxel_loops.append([tuple(p.astype(int)) for p in loop_coords])
                
                logger.info(
                    "Direct skan graph loop detection complete: %d nodes, %d edges, %d loop clusters found in %.3fs",
                    n_voxels, len(edges), len(voxel_loops), time.perf_counter() - t_start
                )
            except (ImportError, Exception) as e:
                logger.warning("Fast loop detection failed, falling back to basic extraction: %s", e)

    t0 = time.perf_counter()
    loop_vox = set()
    for loop in voxel_loops:
        for v in loop:
            loop_vox.add(v)
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

    from tqdm import tqdm
    
    t0 = time.perf_counter()
    max_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        segments = [s for s in tqdm(executor.map(make_segment_safe, paths), total=len(paths), desc="Extracting Segments") if s]
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
            weight=max(total_dist, 1e-6),
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
                    weight=max(dist, 1e-6),
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
