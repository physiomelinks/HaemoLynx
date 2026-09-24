"""Reconnect secondary loop edges with alternative paths."""
import logging
import threading
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.distance import directed_hausdorff
from skimage.graph import route_through_array

from haemolynx.preprocessing.memmap_support import LOW_MEMORY_BLOCK_VOXELS

from ._platform import nested_native_thread_limit

logger = logging.getLogger(__name__)

#: Voxels of context included around a routing window when its cost field is
#: built. Distances up to this are exact; beyond it the field only has to stay
#: large, which it does.
COST_WINDOW_PAD = 32

#: Padded routing-window voxels all threads together may hold at once under
#: the low-RAM option, and the largest single window it routes. A window costs
#: about 80 bytes per voxel between its distance transform and the router's own
#: arrays, so this keeps routing to roughly 5 GB however many threads run.
LOW_MEMORY_ROUTING_VOXELS = 60_000_000

#: A single padded window this much of the volume or more is not worth doing as
#: a window: at that size it costs about what the whole volume costs, and a
#: second such window costs it again.
GLOBAL_FIELD_WINDOW_FRACTION = 0.25

#: Total padded window volume, as a multiple of the volume, past which the
#: whole-volume field is built instead.
#:
#: Deliberately not 1, even though transforming a volume's worth of windows
#: sounds like the break-even point. It is not, because the windows are
#: transformed on the worker threads below and the transform releases the GIL,
#: so they run several at a time, while one whole-volume transform runs alone.
#: On the nerve stack the windows come to 1.29x the volume and take about 13 s
#: of wall clock against 35 s for the whole-volume field -- so counting raw
#: voxels against the volume declares the windows too expensive at the point
#: where they are still nearly three times cheaper. Setting this to 1.0 did
#: exactly that and made the step three times slower.
#:
#: At 4 the worst case is roughly one whole-volume field's worth of windowing
#: before the fallback and one after, and the observed 1.29x has room to grow.
GLOBAL_FIELD_BUDGET_MULTIPLE = 4.0


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
    max_cache_size=None,
    use_memmap=False,
    low_memory_window_voxels=LOW_MEMORY_ROUTING_VOXELS,
):
    """Find alternative paths for degree-2 pairs and add as secondary edges.

    *use_memmap* is the low-RAM option: *skeleton* is read in place instead of
    copied and the whole-volume cost field is never built. A window's distance
    transform runs one padded block at a time once it is bigger than one
    block (the same values; see ``pointwise_distance``), and the windows the
    threads hold at once are capped at *low_memory_window_voxels* padded
    voxels between them -- a thread waits for room rather than adding to RAM.
    A single window bigger than that is skipped, and counted in the closing
    summary.

    *max_cache_size* is accepted for compatibility and ignored: each pair
    routes through its own windows only once, and a cost field (which carries
    that pair's own repulsion term) is never valid for another pair.
    """
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
    # Read-only either way; under use_memmap a bool memmap is used as is.
    skeleton_copy = (
        np.asanyarray(skeleton, dtype=bool) if use_memmap else skeleton.astype(bool)
    )

    # Windowing is only a saving while the windows stay small: one whole-volume
    # transform costs a fixed amount, and enough window area eventually exceeds
    # it. Two ways that happens -- one window nearly as big as the volume, or a
    # great many smaller ones -- and either switches to the whole-volume field,
    # which is then built once and sliced.
    cost_budget = {"transformed_voxels": 0, "global_field": None}
    cost_lock = threading.Lock()
    volume_voxels = int(skeleton_copy.size)
    window_budget = GLOBAL_FIELD_BUDGET_MULTIPLE * volume_voxels
    single_window_limit = GLOBAL_FIELD_WINDOW_FRACTION * volume_voxels

    def window_cost(minc, maxc):
        """Routing cost `1 + d^2` over one sub-volume, d = distance to skeleton.

        The transform runs on the window padded by :data:`COST_WINDOW_PAD`
        rather than on the whole stack, because that is the only part the
        router ever reads. Distances are exact wherever the nearest skeleton
        voxel lies inside the padded window; past the pad they come out larger
        than the true distance, which only pushes the router further away from
        voxels it already avoids -- an accepted path has to lie on the skeleton
        for `min_overlap` of its length.
        """
        plo = np.maximum(minc - COST_WINDOW_PAD, 0)
        phi = np.minimum(maxc + COST_WINDOW_PAD, skeleton_copy.shape)
        padded_voxels = int(np.prod(np.maximum(phi - plo, 0)))

        if use_memmap:
            if padded_voxels > low_memory_window_voxels:
                return None
            global_field = None
        else:
            global_field = _global_field_or_none(padded_voxels)

        if global_field is not None:
            return global_field[
                minc[0]:maxc[0], minc[1]:maxc[1], minc[2]:maxc[2]
            ].copy()

        crop = skeleton_copy[plo[0]:phi[0], plo[1]:phi[1], plo[2]:phi[2]]
        dist = None
        if use_memmap and padded_voxels > LOW_MEMORY_BLOCK_VOXELS:
            from haemolynx.preprocessing.pointwise_distance import (
                distance_transform_edt_blockwise,
            )

            dist = np.empty(crop.shape, dtype=np.float64)
            try:
                distance_transform_edt_blockwise(crop, dist, feature_value=True)
            except ValueError:
                dist = None  # no skeleton in the window: scipy's own answer
        if dist is None:
            dist = distance_transform_edt(~np.asarray(crop))
        inner = tuple(
            slice(int(minc[d] - plo[d]), int(minc[d] - plo[d] + maxc[d] - minc[d]))
            for d in range(3)
        )
        return 1 + dist[inner] ** 2

    def _global_field_or_none(padded_voxels):
        with cost_lock:
            if cost_budget["global_field"] is None:
                spent = cost_budget["transformed_voxels"] + padded_voxels
                too_big = padded_voxels >= single_window_limit
                too_many = spent > window_budget
                if too_big or too_many:
                    logger.info(
                        "[reconnect] %s; transforming the whole volume once instead",
                        "this routing window covers a quarter of the volume"
                        if too_big
                        else "routing windows have covered "
                        f"{spent / max(volume_voxels, 1):.1f}x the volume",
                    )
                    cost_budget["global_field"] = (
                        1 + distance_transform_edt(~skeleton_copy) ** 2
                    )
            global_field = cost_budget["global_field"]
            if global_field is None:
                cost_budget["transformed_voxels"] += padded_voxels
            return global_field

    # Each category is only ever logged per-occurrence behind `debug` --
    # normally off, tied to verbose_logging -- so a run where every single
    # candidate pair hit the same exception looked identical to a run that
    # found nothing to reconnect. Counted here so a summary can say so
    # regardless of `debug`, the same way set_poiseuille_resistances reports
    # its own skipped-edge counts.
    failure_counts = {
        "subvolume creation failed": 0,
        "pathfinding failed": 0,
        "metric calculation failed": 0,
        "candidate pair raised unexpectedly": 0,
        "result processing raised unexpectedly": 0,
        "routing window skipped: too large for the low-RAM option": 0,
    }
    failure_counts_lock = threading.Lock()

    def record_failure(category: str) -> None:
        with failure_counts_lock:
            failure_counts[category] += 1

    routing_room = threading.Condition()
    routing_in_flight = [0]

    @contextmanager
    def routing_memory(minc, maxc):
        """Hold room for one window's padded voxels while it is built and
        routed (the low-RAM option only)."""
        plo = np.maximum(minc - COST_WINDOW_PAD, 0)
        phi = np.minimum(maxc + COST_WINDOW_PAD, skeleton_copy.shape)
        need = min(int(np.prod(np.maximum(phi - plo, 0))), int(low_memory_window_voxels))
        with routing_room:
            while routing_in_flight[0] and routing_in_flight[0] + need > low_memory_window_voxels:
                routing_room.wait()
            routing_in_flight[0] += need
        try:
            yield
        finally:
            with routing_room:
                routing_in_flight[0] -= need
                routing_room.notify_all()

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
                # Held from the cost field through routing: under the low-RAM
                # option the threads' windows share one voxel budget.
                with (routing_memory(minc, maxc) if use_memmap else nullcontext()):
                    try:
                        sub_cost = window_cost(minc, maxc)
                        if sub_cost is None:
                            record_failure(
                                "routing window skipped: too large for the low-RAM option"
                            )
                            break
                        if sub_cost.size == 0:
                            continue
                        orig_rel = [vox - minc for vox in orig_voxels]
                        repulsion = make_repulsion_safe(orig_rel, sub_cost.shape)
                        sub_cost = sub_cost + repulsion
                    except Exception as e:
                        record_failure("subvolume creation failed")
                        if debug:
                            logger.warning("Subvolume creation failed for %s-%s: %s", u, v, e)
                        continue
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
                            record_failure("metric calculation failed")
                            if debug:
                                logger.warning("Metric calculation failed: %s", e)
                            continue
                    except Exception as e:
                        record_failure("pathfinding failed")
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
            record_failure("candidate pair raised unexpectedly")
            if debug:
                logger.error("Attempt failed for %s-%s: %s", u, v, e)
            return None

    added = 0
    edge_lock = threading.Lock()
    max_workers = max_workers or min(4, len(pairs))
    try:
        with nested_native_thread_limit():
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
                        record_failure("result processing raised unexpectedly")
                        if debug:
                            logger.error("Processing failed for %s-%s: %s", u, v, e)
                        continue
    except Exception as e:
        logger.error("Threading failed: %s", e)
        return G

    # Always reported, unlike the per-occurrence messages above: a run where
    # every candidate pair failed the same way must not look identical to a
    # run that cleanly found nothing to reconnect.
    for category, count in failure_counts.items():
        if count:
            logger.warning(
                "reconnect_secondary_loop_edges: %s (%d occurrence%s; "
                "enable verbose_logging for per-pair detail)",
                category,
                count,
                "" if count == 1 else "s",
            )

    if debug:
        logger.info("Done: added %d secondary edges", added)
    return G
