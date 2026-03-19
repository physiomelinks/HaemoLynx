"""Build vascular graph from skeleton using skan."""
import logging
import os
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

logger = logging.getLogger(__name__)


def build_graph_segment_skan_stitched_loops(
    sk,
    skeleton_image,
    debug=False,
    reconnect_threshold=3.0,
    max_voxel_graph_size=100000,
    use_spatial_index=True,
    use_padded_slicing=True,
    padding=3,
):
    """Build NetworkX graph from skan Skeleton with loop detection and terminal reconnection."""
    if sk is None or skeleton_image is None:
        raise ValueError("sk and skeleton_image cannot be None")

    if sk.n_paths == 0:
        logger.warning("No paths found in skeleton")
        return nx.Graph(), [], set()

    paths = [(i, sk.path_coordinates(i)) for i in range(sk.n_paths)]
    skel = skeleton_image
    ndim = skel.ndim
    foreground = np.argwhere(skel)
    voxel_loops = []

    if len(foreground) > 0:
        if use_padded_slicing:
            # 1. Label segments (paths) in the skeleton
            # We treat the entire skeleton as one "image" and find cycles within connected clusters
            # For efficiency, we can label the whole skeleton and process each component
            structure = generate_binary_structure(ndim, ndim)
            labeled_skel, n_comp = label(skel, structure=structure)
            slices = find_objects(labeled_skel)

            offsets = np.argwhere(generate_binary_structure(ndim, 1)) - 1

            for comp_id, sl in enumerate(slices, 1):
                if sl is None:
                    continue
                
                # Expand slice with padding
                sl_padded = tuple(
                    slice(max(0, s.start - padding), min(dim, s.stop + padding))
                    for s, dim in zip(sl, skel.shape)
                )
                
                # Extract local crop
                local_skel = skel[sl_padded]
                local_foreground = np.argwhere(local_skel)
                
                # If local component is small enough, find cycles
                if len(local_foreground) <= max_voxel_graph_size:
                    local_graph = nx.Graph()
                    for pt in local_foreground:
                        for off in offsets:
                            nb = pt + off
                            if (
                                np.all(nb >= 0)
                                and np.all(nb < local_skel.shape)
                                and local_skel[tuple(nb)]
                            ):
                                # Map local back to global
                                global_pt = tuple(pt + np.array([s.start for s in sl_padded]))
                                global_nb = tuple(nb + np.array([s.start for s in sl_padded]))
                                local_graph.add_edge(global_pt, global_nb)
                    
                    try:
                        comp_loops = nx.cycle_basis(local_graph)
                        voxel_loops.extend(comp_loops)
                    except Exception as e:
                        logger.warning("Local loop detection failed for comp %d: %s", comp_id, e)
        else:
            # Legacy global approach
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
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for batch_edges in executor.map(process_pt_batch, batches):
                        voxel_graph.add_edges_from(batch_edges)
                try:
                    voxel_loops = nx.cycle_basis(voxel_graph)
                except Exception as e:
                    logger.warning("Loop detection failed: %s", e)
                    voxel_loops = []
            else:
                if debug:
                    logger.warning(
                        "Skeleton too large (%d voxels) for loop detection", len(foreground)
                    )

    if debug:
        logger.debug("Found %d voxel loops", len(voxel_loops))

    loop_vox = set()
    for loop in voxel_loops:
        for v in loop:
            if isinstance(v, (list, tuple, np.ndarray)):
                loop_vox.add(tuple(np.round(v).astype(int)))

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

    max_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        segments = [s for s in executor.map(make_segment_safe, paths) if s]

    if not segments:
        logger.warning("No valid segments found")
        return nx.Graph(), voxel_loops, set()

    G = nx.Graph()
    loop_edges = set()
    mapping = {}

    for seg_idx, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        u_vox, v_vox = seg[0], seg[-1]
        uid = mapping.setdefault(u_vox, len(mapping))
        vid = mapping.setdefault(v_vox, len(mapping))
        u_pos = np.array(u_vox, dtype=float)
        v_pos = np.array(v_vox, dtype=float)
        if not G.has_node(uid):
            G.add_node(uid, pos=u_pos)
        if not G.has_node(vid):
            G.add_node(vid, pos=v_pos)
        seg_array = np.array(seg, dtype=float)
        if len(seg_array) > 1:
            total_dist = float(np.sum(np.linalg.norm(np.diff(seg_array, axis=0), axis=1)))
        else:
            total_dist = 0.0
        G.add_edge(
            uid,
            vid,
            weight=max(total_dist, 1e-6),
            length=len(seg),
            voxels=seg,
            segment_id=seg_idx,
        )
        if u_vox in loop_vox and v_vox in loop_vox:
            loop_edges.add(tuple(sorted([uid, vid])))

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
                        or G.degree[src] >= 3
                        or G.degree[tgt] >= 3
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
                            or G.degree[src] >= 3
                            or G.degree[tgt] >= 3
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
                if G.has_edge(src, tgt) or G.degree[src] >= 3 or G.degree[tgt] >= 3:
                    continue
                src_pos = np.array(G.nodes[src]["pos"])
                tgt_pos = np.array(G.nodes[tgt]["pos"])
                G.add_edge(
                    src,
                    tgt,
                    weight=max(dist, 1e-6),
                    length=2,
                    voxels=[
                        tuple(src_pos.astype(int)),
                        tuple(tgt_pos.astype(int)),
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
    return G, voxel_loops, loop_edges
