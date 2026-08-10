"""Reconnect secondary loop edges with alternative paths."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.distance import directed_hausdorff
from skimage.graph import route_through_array

logger = logging.getLogger(__name__)

#: Voxels of context included around a routing window when its cost field is
#: built. Distances up to this are exact; beyond it the field only has to stay
#: large, which it does.
COST_WINDOW_PAD = 32


def _path_length_3d(points) -> float:
    """Compute 3D polyline length from physical coordinates."""
    if not points or len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def reconnect_secondary_loop_edges(
    G,
    skeleton,
    voxel_size=(1, 1, 1),
    min_length_voxels=30,
    max_length_voxels=6000,
    max_distance=6000.0,
    margin=10,
    k_paths=5,
    min_overlap=0.9,
    min_geom_dev=8.0,
    repulsion_sigma=2.0,
    max_workers=None,
    debug=True,
    max_cache_size=1000,
):
    """Find alternative paths for degree-2 pairs and add as secondary edges."""
    if not isinstance(G, (nx.Graph, nx.MultiGraph)):
        raise ValueError("G must be a NetworkX Graph or MultiGraph")
    if skeleton is None or skeleton.size == 0:
        raise ValueError("skeleton cannot be None or empty")

    if not isinstance(G, nx.MultiGraph):
        if debug:
            logger.info("Converting Graph to MultiGraph")
        G = nx.MultiGraph(G)

    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        logger.warning("No node positions found")
        return G

    deg = dict(G.degree())
    skeleton_copy = skeleton.astype(bool)

    def window_cost(minc, maxc):
        """Routing cost `1 + d^2` over one sub-volume, d = distance to skeleton.

        The transform runs on the window padded by :data:`COST_WINDOW_PAD`
        rather than on the whole stack, because that is the only part the
        router ever reads and an exact whole-volume transform costs the same
        whether one window is wanted or all of them. Distances are exact
        wherever the nearest skeleton voxel lies inside the padded window; past
        the pad they come out larger than the true distance, which only pushes
        the router further away from voxels it already avoids -- an accepted
        path has to lie on the skeleton for `min_overlap` of its length.
        """
        plo = np.maximum(minc - COST_WINDOW_PAD, 0)
        phi = np.minimum(maxc + COST_WINDOW_PAD, skeleton_copy.shape)
        padded = ~skeleton_copy[plo[0]:phi[0], plo[1]:phi[1], plo[2]:phi[2]]
        dist = distance_transform_edt(padded)
        inner = tuple(
            slice(int(minc[d] - plo[d]), int(minc[d] - plo[d] + maxc[d] - minc[d]))
            for d in range(3)
        )
        return 1 + dist[inner] ** 2

    cache_lock = threading.Lock()
    sub_cache = {}
    cache_access_order = []

    def manage_cache(key, value=None):
        with cache_lock:
            if value is not None:
                if key in sub_cache:
                    cache_access_order.remove(key)
                    cache_access_order.append(key)
                else:
                    if len(sub_cache) >= max_cache_size:
                        oldest = cache_access_order.pop(0)
                        del sub_cache[oldest]
                    sub_cache[key] = value
                    cache_access_order.append(key)
                return value
            else:
                if key in sub_cache:
                    cache_access_order.remove(key)
                    cache_access_order.append(key)
                    return sub_cache[key]
                return None

    def make_repulsion_safe(orig_voxels, sub_shape):
        if not orig_voxels or not sub_shape or any(s <= 0 for s in sub_shape):
            return np.zeros(sub_shape, dtype=float)
        mask = np.zeros(sub_shape, dtype=float)
        valid_count = 0
        for coords in orig_voxels:
            if len(coords) >= 3:
                x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
                if 0 <= x < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= z < sub_shape[2]:
                    mask[x, y, z] = 1.0
                    valid_count += 1
        if valid_count == 0:
            return np.zeros(sub_shape, dtype=float)
        try:
            repulsion_field = gaussian_filter(mask, sigma=repulsion_sigma)
            max_repulsion = np.max(repulsion_field)
            if max_repulsion > 0:
                repulsion_field = (repulsion_field / max_repulsion) * 50.0
                high_penalty_mask = mask > 0
                repulsion_field[high_penalty_mask] += 100.0
            return repulsion_field
        except Exception as e:
            logger.warning("Gaussian filter failed: %s", e)
            fallback_mask = mask.copy()
            fallback_mask[mask > 0] = 100.0
            return fallback_mask

    candidates = []
    for u, v, key, data in G.edges(data=True, keys=True):
        if (
            deg[u] == 2
            and deg[v] == 2
            and data.get("voxels")
            and u in pos
            and v in pos
            and not data.get("secondary", False)
        ):
            distance = np.linalg.norm(np.subtract(pos[u], pos[v]))
            if distance <= max_distance:
                candidates.append((u, v, distance))

    seen_pairs = set()
    unique_candidates = []
    for u, v, dist in candidates:
        pair = tuple(sorted([u, v]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_candidates.append((u, v, dist))
    unique_candidates.sort(key=lambda x: x[2])
    pairs = [(u, v) for u, v, _ in unique_candidates]

    if debug:
        logger.info("%d candidate deg-2 pairs", len(pairs))
    
    if len(pairs) == 0:
        return G

    def attempt_reconnect(pair_data):
        u, v, node_positions = pair_data
        try:
            if u not in node_positions or v not in node_positions:
                return None
            pu, pv = np.array(node_positions[u]), np.array(node_positions[v])
            u_vox = np.round(pu / np.array(voxel_size)).astype(int)
            v_vox = np.round(pv / np.array(voxel_size)).astype(int)
            primary_edge_data = None
            for key, edge_data in G[u][v].items():
                if not edge_data.get("secondary", False):
                    primary_edge_data = edge_data
                    break
            if not primary_edge_data:
                return None
            orig_voxels_raw = primary_edge_data.get("voxels", [])
            if not orig_voxels_raw:
                return None
            orig_voxels = []
            for vox in orig_voxels_raw:
                if isinstance(vox, (list, tuple, np.ndarray)) and len(vox) >= 3:
                    vox_coords = np.round(np.array(vox) / np.array(voxel_size)).astype(int)
                    if np.all(vox_coords >= 0) and np.all(vox_coords < skeleton_copy.shape):
                        orig_voxels.append(vox_coords)
            if not orig_voxels:
                return None
            best_paths = []
            for expansion in [0, 10, 25, 50]:
                ext = margin + expansion
                minc = np.maximum(np.minimum(u_vox, v_vox) - ext, 0)
                maxc = np.minimum(np.maximum(u_vox, v_vox) + ext + 1, skeleton_copy.shape)
                if np.any(minc >= maxc):
                    continue
                cache_key = (*minc, *maxc)
                cached_result = manage_cache(cache_key)
                if cached_result is None:
                    try:
                        sub_cost = window_cost(minc, maxc)
                        if sub_cost.size == 0:
                            continue
                        orig_rel = [vox - minc for vox in orig_voxels]
                        repulsion = make_repulsion_safe(orig_rel, sub_cost.shape)
                        sub_cost = sub_cost + repulsion
                        cached_result = (sub_cost, minc)
                        manage_cache(cache_key, cached_result)
                    except Exception as e:
                        if debug:
                            logger.warning("Subvolume creation failed for %s-%s: %s", u, v, e)
                        continue
                sub_cost, minc = cached_result
                ru = u_vox - minc
                rv = v_vox - minc
                if (
                    np.any(ru < 0)
                    or np.any(rv < 0)
                    or np.any(ru >= sub_cost.shape)
                    or np.any(rv >= sub_cost.shape)
                ):
                    continue
                try:
                    path_coords, cost = route_through_array(
                        sub_cost, tuple(ru), tuple(rv), fully_connected=True
                    )
                    if path_coords is None or len(path_coords) < min_length_voxels:
                        continue
                    path_coords = np.array(path_coords)
                    path_length = len(path_coords)
                    if path_length > max_length_voxels:
                        continue
                    abs_coords = path_coords + minc
                    if np.any(abs_coords < 0) or np.any(abs_coords >= skeleton.shape):
                        continue
                    try:
                        x, y, z = abs_coords.T
                        skeleton_hits = skeleton[x, y, z]
                        overlap = np.sum(skeleton_hits) / path_length
                        if overlap < min_overlap:
                            continue
                        orig_coords = np.array(orig_voxels)
                        hausdorff_dist = max(
                            directed_hausdorff(orig_coords, abs_coords)[0],
                            directed_hausdorff(abs_coords, orig_coords)[0],
                        )
                        if hausdorff_dist < min_geom_dev:
                            if debug:
                                logger.debug(
                                    "Path too similar (dev=%.1f < %.1f)",
                                    hausdorff_dist,
                                    min_geom_dev,
                                )
                            continue
                        orig_set = set(tuple(coord) for coord in orig_coords)
                        new_set = set(tuple(coord) for coord in abs_coords)
                        overlap_voxels = len(orig_set.intersection(new_set))
                        path_similarity = overlap_voxels / min(
                            len(orig_set), len(new_set)
                        )
                        if path_similarity > 0.7:
                            if debug:
                                logger.debug(
                                    "Path too similar (voxel overlap=%.2f)",
                                    path_similarity,
                                )
                            continue
                        vox3d = (abs_coords * np.array(voxel_size)).tolist()
                        path_length_3d = _path_length_3d(vox3d)
                        unique_voxels = len(new_set - orig_set)
                        path_novelty = unique_voxels / len(new_set)
                        best_paths.append(
                            {
                                "voxels": vox3d,
                                "overlap": overlap,
                                "deviation": hausdorff_dist,
                                "length": path_length_3d if path_length_3d > 0 else float(path_length),
                                "cost": cost,
                                "novelty": path_novelty,
                                "voxel_similarity": path_similarity,
                            }
                        )
                        if len(best_paths) >= k_paths:
                            break
                    except Exception as e:
                        if debug:
                            logger.warning("Metric calculation failed: %s", e)
                        continue
                except Exception as e:
                    if debug:
                        logger.warning("Pathfinding failed for %s-%s: %s", u, v, e)
                    continue
                if best_paths:
                    break
            if not best_paths:
                return None
            best_paths.sort(
                key=lambda p: (-p["novelty"], -p["overlap"], -p["deviation"])
            )
            return u, v, best_paths[:k_paths]
        except Exception as e:
            if debug:
                logger.error("Attempt failed for %s-%s: %s", u, v, e)
            return None

    added = 0
    edge_lock = threading.Lock()
    max_workers = max_workers or min(4, len(pairs))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pair_data_list = [(u, v, pos) for u, v in pairs]
            future_to_pair = {
                executor.submit(attempt_reconnect, pd): (pd[0], pd[1])
                for pd in pair_data_list
            }
            for future in as_completed(future_to_pair):
                u, v = future_to_pair[future]
                try:
                    result = future.result()
                    if not result:
                        continue
                    u, v, candidates_list = result
                    if debug:
                        logger.info("%s-%s -> %d candidates", u, v, len(candidates_list))
                    with edge_lock:
                        has_secondary = False
                        if G.has_edge(u, v):
                            for key, edge_data in G[u][v].items():
                                if edge_data.get("secondary", False):
                                    has_secondary = True
                                    break
                        if has_secondary:
                            if debug:
                                logger.info("%s-%s already has secondary edge", u, v)
                            continue
                        best = candidates_list[0]
                        G.add_edge(
                            u,
                            v,
                            voxels=best["voxels"],
                            length=best["length"],
                            overlap=best["overlap"],
                            deviation=best["deviation"],
                            novelty=best["novelty"],
                            secondary=True,
                        )
                        added += 1
                        if debug:
                            logger.info(
                                "Added secondary edge %s-%s: novelty=%.2f",
                                u,
                                v,
                                best["novelty"],
                            )
                except Exception as e:
                    if debug:
                        logger.error("Processing failed for %s-%s: %s", u, v, e)
                    continue
    except Exception as e:
        logger.error("Threading failed: %s", e)
        return G

    with cache_lock:
        sub_cache.clear()
        cache_access_order.clear()

    if debug:
        logger.info("Done: added %d secondary edges", added)
    return G
