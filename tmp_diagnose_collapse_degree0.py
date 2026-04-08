#!/usr/bin/env python3
"""Temporary diagnostic script to explain degree-0 nodes from collapse stage.

This script does NOT modify pipeline code. It reproduces the graph up to
`smart_multigraph_degree2_removal_pass1`, then runs an instrumented collapse
pass mirroring `graph.collapse_node_clusters` to report exactly why and where
degree-0 nodes appear.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skan import csr


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
EXAMPLES_DIR = ROOT_DIR / "examples"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from ImageLynx import graph, io, preprocessing  # noqa: E402
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz  # noqa: E402
import resistance_network_pipeline as main_pipeline  # noqa: E402


def _rewire_edges_with_stats(
    G: nx.MultiGraph,
    old_node: int,
    new_node: int,
) -> dict[str, int]:
    stats = {
        "incident_edges_seen": 0,
        "edges_rewired": 0,
        "edges_skipped_neighbor_is_rep": 0,
    }
    old_pos = np.asarray(G.nodes[old_node].get("pos", [0, 0, 0]), dtype=float)
    new_pos = np.asarray(G.nodes[new_node].get("pos", [0, 0, 0]), dtype=float)
    edges = list(G.edges(old_node, data=True, keys=True))
    for u, v, _key, data in edges:
        stats["incident_edges_seen"] += 1
        neighbor = v if u == old_node else u
        if neighbor == new_node:
            stats["edges_skipped_neighbor_is_rep"] += 1
            continue
        patched = dict(data)
        voxels = patched.get("voxels")
        if voxels and len(voxels) >= 2:
            voxels = list(voxels)
            old_voxel = tuple(np.round(old_pos).astype(int))
            new_voxel = tuple(np.round(new_pos).astype(int))
            start_key = tuple(np.round(np.asarray(voxels[0])).astype(int))
            end_key = tuple(np.round(np.asarray(voxels[-1])).astype(int))
            if (
                start_key == old_voxel
                or np.linalg.norm(np.asarray(voxels[0], dtype=float) - old_pos)
                < np.linalg.norm(np.asarray(voxels[-1], dtype=float) - old_pos)
            ):
                voxels[0] = new_voxel
            else:
                voxels[-1] = new_voxel
            patched["voxels"] = voxels
        G.add_edge(new_node, neighbor, **patched)
        stats["edges_rewired"] += 1
    return stats


def _load_or_generate_skeleton(
    settings: dict[str, Any],
    image_path: Path,
    skeleton_path: Path | None,
    force_skeletonize: bool,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    if skeleton_path is not None and skeleton_path.exists() and not force_skeletonize:
        skeleton = np.load(skeleton_path)
        voxel_size = tuple(float(v) for v in settings.get("voxel_size_override_xyz_px_per_um", (1.0, 1.0, 1.0)))
        if voxel_size == (0.0, 0.0, 0.0):
            voxel_size = (1.0, 1.0, 1.0)
        return skeleton, voxel_size, {"source": "existing_file", "skeleton_path": str(skeleton_path)}

    image_path = io.resolve_image_path_with_optional_zip(image_path)
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix in {"tif", "tiff"}:
        _, skeleton, vx, vy, vz, voxel_meta_status = io.load_and_skeletonize_3d_tif(image_path)
    elif suffix == "h5":
        _, skeleton, vx, vy, vz, voxel_meta_status = io.load_and_skeletonize_3d_h5(image_path)
    else:
        raise ValueError(f"Unsupported input format for skeletonization: {suffix}")

    metadata_voxel_size = (float(vx), float(vy), float(vz))
    voxel_size, voxel_size_source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz_px_per_um=settings.get("voxel_size_override_xyz_px_per_um"),
        voxel_size_policy=settings.get("voxel_size_policy"),
    )
    cleaned = preprocessing.preprocess_skeleton_for_graph(
        skeleton,
        min_branch_length=settings.get("skeleton_min_branch_length", 10),
        max_bridge_distance=settings.get("skeleton_max_bridge_distance", 8),
        component_connectivity=settings.get("skeleton_component_connectivity", 26),
        min_component_fraction=settings.get("skeleton_min_component_percent", 0.0) / 100.0,
        closing_radius=settings.get("skeleton_closing_radius", 0),
        bridge_gap_size=settings.get("skeleton_bridge_gap_size", 0),
    )
    meta = {
        "source": "fresh_skeletonization",
        "image_path": str(image_path),
        "voxel_size_source": voxel_size_source,
        "voxel_metadata_status": voxel_meta_status,
    }
    return cleaned, tuple(float(v) for v in voxel_size), meta


def _build_graph_up_to_precollapse(
    skeleton: np.ndarray,
    voxel_size: tuple[float, float, float],
    reconnect_threshold: float,
    verbose_logging: bool,
) -> nx.MultiGraph:
    sk = csr.Skeleton(skeleton)
    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=verbose_logging,
        reconnect_threshold=reconnect_threshold,
        voxel_size=voxel_size,
    )
    G = graph.reconnect_secondary_loop_edges(G, skeleton, voxel_size=voxel_size, debug=verbose_logging)
    G, _ = graph.optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=verbose_logging,
        reconnect_threshold=reconnect_threshold,
    )
    G = graph.smart_multigraph_degree2_removal(G, skeleton, max_degree=4, debug=verbose_logging)
    return G


def _instrumented_collapse(
    G_in: nx.MultiGraph,
    distance_threshold: float,
    max_iterations: int = 10,
) -> tuple[nx.MultiGraph, list[dict[str, Any]]]:
    G = G_in.copy()
    events: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        nodes_with_pos = [
            (n, np.array(G.nodes[n]["pos"], dtype=float))
            for n in G.nodes()
            if "pos" in G.nodes[n]
        ]
        if len(nodes_with_pos) < 2:
            break

        node_ids = [n for n, _ in nodes_with_pos]
        coords = np.array([p for _, p in nodes_with_pos], dtype=float)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(distance_threshold)
        if not pairs:
            break

        proximity = nx.Graph()
        proximity.add_nodes_from(range(len(node_ids)))
        for i, j in pairs:
            proximity.add_edge(i, j)

        merged_this_iter = 0
        for component in nx.connected_components(proximity):
            if len(component) < 2:
                continue
            cluster = [node_ids[i] for i in component if G.has_node(node_ids[i])]
            if len(cluster) < 2:
                continue

            rep = max(cluster, key=lambda n: (G.degree(n), -n))
            others = [n for n in cluster if n != rep]

            cluster_set = set(cluster)
            ext_neighbors_before = set()
            internal_edges_before = 0
            cluster_total_degree = 0
            for n in cluster:
                d = int(G.degree(n))
                cluster_total_degree += d
                for nb in G.neighbors(n):
                    if nb in cluster_set:
                        internal_edges_before += 1
                    else:
                        ext_neighbors_before.add(nb)

            positions = np.array([G.nodes[n]["pos"] for n in cluster if "pos" in G.nodes[n]], dtype=float)
            if positions.size > 0:
                G.nodes[rep]["pos"] = positions.mean(axis=0)

            rewire_stats = {
                "incident_edges_seen": 0,
                "edges_rewired": 0,
                "edges_skipped_neighbor_is_rep": 0,
            }
            removed_nodes = 0
            for other in others:
                if not G.has_node(other):
                    continue
                s = _rewire_edges_with_stats(G, old_node=other, new_node=rep)
                for k in rewire_stats:
                    rewire_stats[k] += s[k]
                G.remove_node(other)
                removed_nodes += 1
                merged_this_iter += 1

            rep_degree_after_rewire = int(G.degree(rep)) if G.has_node(rep) else -1
            event = {
                "iteration": int(iteration + 1),
                "cluster_size": int(len(cluster)),
                "cluster_nodes": [int(n) for n in sorted(cluster)],
                "representative": int(rep),
                "removed_nodes": int(removed_nodes),
                "cluster_total_degree_before": int(cluster_total_degree),
                "external_neighbor_count_before": int(len(ext_neighbors_before)),
                "external_neighbors_before": [int(n) for n in sorted(ext_neighbors_before)],
                "rewire_stats": rewire_stats,
                "rep_degree_after_rewire": rep_degree_after_rewire,
            }
            if rep_degree_after_rewire == 0:
                if len(ext_neighbors_before) == 0:
                    event["likely_cause"] = (
                        "cluster had no external neighbors before collapse; after merging internal-only "
                        "connectivity, representative becomes isolated"
                    )
                else:
                    event["likely_cause"] = (
                        "external connectivity existed but representative still isolated; inspect edge rewiring "
                        "and subsequent self-loop removal"
                    )
            events.append(event)

        if merged_this_iter == 0:
            break

    self_loops = list(nx.selfloop_edges(G, keys=True))
    if self_loops:
        G.remove_edges_from(self_loops)

    return G, events


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose degree-0 emergence during cluster collapse.")
    parser.add_argument("--image-path", type=Path, default=None, help="Override input image path.")
    parser.add_argument(
        "--skeleton-path",
        type=Path,
        default=ROOT_DIR / "examples" / "outputs" / "brain_microvessels_skeleton.npy",
        help="Use existing skeleton file (default uses previously saved brain skeleton).",
    )
    parser.add_argument("--force-skeletonize", action="store_true", help="Force fresh skeletonization.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT_DIR / "tmp_collapse_degree0_diagnostic_report.json",
        help="Where to save the detailed collapse diagnostic report.",
    )
    args = parser.parse_args()

    default_plot_dir = Path(main_pipeline.BASE_PLOT_DIR) / "nerve"
    settings = main_pipeline._build_pipeline_kwargs_from_active_settings(plot_dir=default_plot_dir)
    image_path = Path(args.image_path) if args.image_path is not None else Path(settings["image_path"])
    reconnect_threshold = float(settings.get("graph_reconnect_threshold", 3.0))
    cluster_collapse_distance = float(settings.get("cluster_collapse_distance", 5.0))
    verbose_logging = bool(settings.get("verbose_logging", False))

    skeleton, voxel_size, skeleton_meta = _load_or_generate_skeleton(
        settings=settings,
        image_path=image_path,
        skeleton_path=args.skeleton_path,
        force_skeletonize=bool(args.force_skeletonize),
    )
    G_pre = _build_graph_up_to_precollapse(
        skeleton=skeleton,
        voxel_size=voxel_size,
        reconnect_threshold=reconnect_threshold,
        verbose_logging=verbose_logging,
    )
    pre_degree0 = [int(n) for n in G_pre.nodes if G_pre.degree[n] == 0]
    G_post, events = _instrumented_collapse(
        G_pre,
        distance_threshold=cluster_collapse_distance,
        max_iterations=10,
    )
    post_degree0 = [int(n) for n in G_post.nodes if G_post.degree[n] == 0]
    events_with_isolated_rep = [e for e in events if e["rep_degree_after_rewire"] == 0]
    isolated_with_external_neighbors = [
        e for e in events_with_isolated_rep if e["external_neighbor_count_before"] > 0
    ]
    isolated_internal_only = [
        e for e in events_with_isolated_rep if e["external_neighbor_count_before"] == 0
    ]

    print("=== Collapse-focused degree-0 diagnostic ===")
    print(f"Image path: {image_path}")
    print(f"Skeleton source: {skeleton_meta.get('source')}")
    print(f"Pre-collapse: nodes={G_pre.number_of_nodes()}, edges={G_pre.number_of_edges()}, degree0={len(pre_degree0)}")
    print(f"Post-collapse: nodes={G_post.number_of_nodes()}, edges={G_post.number_of_edges()}, degree0={len(post_degree0)}")
    print(f"Clusters analyzed: {len(events)}")
    print(f"Clusters with isolated representative: {len(events_with_isolated_rep)}")
    print(f"  - Internal-only clusters: {len(isolated_internal_only)}")
    print(f"  - Isolated despite external neighbors: {len(isolated_with_external_neighbors)}")

    if isolated_with_external_neighbors:
        print("\nRepresentative-isolated clusters that DID have external neighbors (inspect first 10):")
        for row in isolated_with_external_neighbors[:10]:
            print(
                f"- rep={row['representative']} cluster_size={row['cluster_size']} "
                f"ext_neighbors={row['external_neighbor_count_before']} "
                f"rewired={row['rewire_stats']['edges_rewired']} "
                f"skipped_to_rep={row['rewire_stats']['edges_skipped_neighbor_is_rep']}"
            )

    report = {
        "settings_summary": {
            "image_path": str(image_path),
            "voxel_size_xyz": list(voxel_size),
            "graph_reconnect_threshold": reconnect_threshold,
            "cluster_collapse_distance": cluster_collapse_distance,
            "skeleton_meta": skeleton_meta,
        },
        "precollapse": {
            "node_count": int(G_pre.number_of_nodes()),
            "edge_count": int(G_pre.number_of_edges()),
            "degree0_nodes": pre_degree0,
        },
        "postcollapse": {
            "node_count": int(G_post.number_of_nodes()),
            "edge_count": int(G_post.number_of_edges()),
            "degree0_nodes": post_degree0,
        },
        "summary": {
            "clusters_analyzed": int(len(events)),
            "clusters_with_isolated_rep": int(len(events_with_isolated_rep)),
            "clusters_isolated_internal_only": int(len(isolated_internal_only)),
            "clusters_isolated_with_external_neighbors": int(len(isolated_with_external_neighbors)),
        },
        "events": events,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"\nDetailed JSON report written to: {args.output_json}")


if __name__ == "__main__":
    main()
