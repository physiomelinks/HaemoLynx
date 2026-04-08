#!/usr/bin/env python3
"""Temporary diagnostic script to trace degree-0 nodes through graph building.

This script intentionally does NOT modify the main pipeline. It replays the
same skeleton->graph stages used in `examples/resistance_network_pipeline.py`,
captures degree-0 nodes at each stage, and reports why any remain at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from skan import csr


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
EXAMPLES_DIR = ROOT_DIR / "examples"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from ImageLynx import graph, io, preprocessing  # noqa: E402
from ImageLynx.graph.validate import validate_skeleton_connection  # noqa: E402
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz  # noqa: E402
import resistance_network_pipeline as main_pipeline  # noqa: E402


def _position_key(pos: Any, decimals: int = 3) -> tuple[float, float, float] | None:
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size != 3:
        return None
    return tuple(float(np.round(v, decimals)) for v in arr)


def _degree_counts(G) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, degree in G.degree():
        counts[degree] = counts.get(degree, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _degree0_nodes(G) -> list[int]:
    return [n for n in G.nodes if G.degree[n] == 0]


def _snapshot_stage(
    stage_name: str,
    G,
    history: list[dict[str, Any]],
    tracker: dict[str, dict[str, Any]],
) -> None:
    degree0 = _degree0_nodes(G)
    entry = {
        "stage": stage_name,
        "node_count": int(G.number_of_nodes()),
        "edge_count": int(G.number_of_edges()),
        "degree_histogram": _degree_counts(G),
        "degree0_count": len(degree0),
        "degree0_nodes": [],
    }
    for node_id in degree0:
        pos = G.nodes[node_id].get("pos")
        key = _position_key(pos)
        row = {"node_id": int(node_id), "pos": None if key is None else list(key)}
        entry["degree0_nodes"].append(row)

        track_key = f"id:{node_id}" if key is None else f"pos:{key}"
        if track_key not in tracker:
            tracker[track_key] = {
                "first_seen_stage": stage_name,
                "stages": [],
                "node_ids": [],
                "pos": None if key is None else list(key),
            }
        tracker[track_key]["stages"].append(stage_name)
        tracker[track_key]["node_ids"].append(int(node_id))
    history.append(entry)


def _diagnose_final_degree0_reasons(
    G,
    skeleton: np.ndarray,
    reconnect_threshold: float,
) -> list[dict[str, Any]]:
    degree0 = _degree0_nodes(G)
    if not degree0:
        return []

    vs = tuple(G.graph.get("voxel_size", (1.0, 1.0, 1.0)))
    valid_targets = [n for n in G.nodes if G.degree[n] >= 1 and "pos" in G.nodes[n]]
    target_positions = None
    target_ids: list[int] = []
    if valid_targets:
        target_ids = [int(n) for n in valid_targets]
        target_positions = np.array([G.nodes[n]["pos"] for n in valid_targets], dtype=float)

    diagnoses: list[dict[str, Any]] = []
    for node_id in degree0:
        node_data = G.nodes[node_id]
        pos = node_data.get("pos")
        node_result: dict[str, Any] = {
            "node_id": int(node_id),
            "pos": None if pos is None else [float(v) for v in np.asarray(pos, dtype=float)],
            "reason": "",
            "nearest_target_node_id": None,
            "nearest_target_distance": None,
            "within_threshold": False,
            "skeleton_path_valid": None,
        }

        if pos is None:
            node_result["reason"] = "node has no 'pos' attribute; reconnection cannot evaluate it"
            diagnoses.append(node_result)
            continue

        if target_positions is None or target_positions.size == 0:
            node_result["reason"] = "no target nodes (degree >= 1 with pos) available for reconnection"
            diagnoses.append(node_result)
            continue

        src = np.asarray(pos, dtype=float)
        distances = np.linalg.norm(target_positions - src, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        nearest_target = target_ids[nearest_idx]
        node_result["nearest_target_node_id"] = nearest_target
        node_result["nearest_target_distance"] = nearest_distance

        if nearest_distance > reconnect_threshold:
            node_result["reason"] = (
                "nearest reconnection candidate is outside reconnect threshold "
                f"({nearest_distance:.3f} > {reconnect_threshold:.3f})"
            )
            diagnoses.append(node_result)
            continue

        node_result["within_threshold"] = True
        target_pos = np.asarray(G.nodes[nearest_target]["pos"], dtype=float)
        valid, _ = validate_skeleton_connection(
            skeleton_data=skeleton,
            pos1=src,
            pos2=target_pos,
            max_gap=reconnect_threshold,
            voxel_size=vs,
        )
        node_result["skeleton_path_valid"] = bool(valid)
        if not valid:
            node_result["reason"] = "candidate is in range, but skeleton-path validation fails"
        else:
            node_result["reason"] = (
                "candidate appears valid; inspect reconnection candidate ordering and constraints "
                "(e.g., candidate pair filtering during heap processing)"
            )
        diagnoses.append(node_result)

    return diagnoses


def _load_or_generate_skeleton(
    settings: dict[str, Any],
    image_path: Path,
    skeleton_path: Path | None,
    force_skeletonize: bool,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    if skeleton_path is not None and skeleton_path.exists() and not force_skeletonize:
        skeleton = np.load(skeleton_path)
        meta = {
            "source": "existing_file",
            "skeleton_path": str(skeleton_path),
            "note": "Loaded skeleton directly; raw skeletonisation stage not rerun.",
        }
        voxel_size = tuple(float(v) for v in settings.get("voxel_size_override_xyz_px_per_um", (1.0, 1.0, 1.0)))
        if voxel_size == (0.0, 0.0, 0.0):
            voxel_size = (1.0, 1.0, 1.0)
        return skeleton, voxel_size, meta

    image_path = io.resolve_image_path_with_optional_zip(image_path)
    input_format = image_path.suffix.lower().lstrip(".")
    if input_format in {"tif", "tiff"}:
        _, skeleton, vx, vy, vz, voxel_meta_status = io.load_and_skeletonize_3d_tif(image_path)
    elif input_format == "h5":
        _, skeleton, vx, vy, vz, voxel_meta_status = io.load_and_skeletonize_3d_h5(image_path)
    else:
        raise ValueError(f"Unsupported input format for skeletonization: {input_format}")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace degree-0 nodes across main graph-building stages.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Override image path; defaults to main pipeline INPUT_PATH.",
    )
    parser.add_argument(
        "--skeleton-path",
        type=Path,
        default=None,
        help="Optional existing skeleton .npy path to skip rerunning skeletonization.",
    )
    parser.add_argument(
        "--force-skeletonize",
        action="store_true",
        help="Force fresh skeletonization even if --skeleton-path exists.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT_DIR / "tmp_degree0_diagnostic_report.json",
        help="Where to write a detailed JSON report.",
    )
    parser.add_argument(
        "--reconnect-threshold",
        type=float,
        default=None,
        help="Override graph reconnect threshold (defaults to settings).",
    )
    args = parser.parse_args()

    default_plot_dir = Path(main_pipeline.BASE_PLOT_DIR) / "nerve"
    settings = main_pipeline._build_pipeline_kwargs_from_active_settings(plot_dir=default_plot_dir)
    image_path = Path(args.image_path) if args.image_path is not None else Path(settings["image_path"])
    reconnect_threshold = float(
        args.reconnect_threshold
        if args.reconnect_threshold is not None
        else settings.get("graph_reconnect_threshold", 3.0)
    )
    final_orphan_reconnect_threshold = float(settings.get("final_orphan_reconnect_threshold", reconnect_threshold))
    cluster_collapse_distance = float(settings.get("cluster_collapse_distance", 5.0))
    min_stub_length = float(settings.get("min_stub_length", 10.0))
    verbose_logging = bool(settings.get("verbose_logging", False))

    skeleton, voxel_size, skeleton_meta = _load_or_generate_skeleton(
        settings=settings,
        image_path=image_path,
        skeleton_path=args.skeleton_path,
        force_skeletonize=bool(args.force_skeletonize),
    )
    print("=== Degree-0 Tracking Diagnostic ===")
    print(f"Image path: {image_path}")
    print(f"Skeleton source: {skeleton_meta.get('source')}")
    print(f"Voxel size (x, y, z): {voxel_size}")
    print(f"Reconnect threshold: {reconnect_threshold}")
    print(f"Final orphan reconnect threshold: {final_orphan_reconnect_threshold}")
    print(f"Cluster collapse distance: {cluster_collapse_distance}")
    print(f"Min stub length: {min_stub_length}")

    sk = csr.Skeleton(skeleton)
    G, voxel_loops, loop_edges = graph.build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=verbose_logging,
        voxel_size=voxel_size,
        reconnect_threshold=reconnect_threshold,
    )

    history: list[dict[str, Any]] = []
    tracker: dict[str, dict[str, Any]] = {}
    _snapshot_stage("build_graph_segment_skan_stitched_loops", G, history, tracker)

    G = graph.reconnect_secondary_loop_edges(
        G, skeleton, voxel_size=voxel_size, debug=verbose_logging
    )
    _snapshot_stage("reconnect_secondary_loop_edges", G, history, tracker)

    G, _ = graph.optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=verbose_logging,
        reconnect_threshold=reconnect_threshold,
    )
    _snapshot_stage("optimise_graph_topology_fixed", G, history, tracker)

    degree2_pass1_max_degree = 4
    degree2_pass2_max_degree = 8
    G = graph.smart_multigraph_degree2_removal(
        G, skeleton, max_degree=degree2_pass1_max_degree, debug=verbose_logging
    )
    _snapshot_stage("smart_multigraph_degree2_removal_pass1", G, history, tracker)

    G = graph.collapse_node_clusters(
        G,
        distance_threshold=cluster_collapse_distance,
        debug=verbose_logging,
    )
    _snapshot_stage("collapse_node_clusters", G, history, tracker)

    G = graph.smart_multigraph_degree2_removal(
        G, skeleton, max_degree=degree2_pass2_max_degree, debug=verbose_logging
    )
    _snapshot_stage("smart_multigraph_degree2_removal_post_collapse", G, history, tracker)

    G = graph.prune_vascular_stubs(G, debug=verbose_logging, min_stub_length=min_stub_length)
    _snapshot_stage("prune_vascular_stubs", G, history, tracker)

    G = graph.smart_multigraph_degree2_removal(
        G, skeleton, max_degree=degree2_pass2_max_degree, debug=verbose_logging
    )
    _snapshot_stage("smart_multigraph_degree2_removal_post_prune", G, history, tracker)

    G = graph.remove_edges_for_self_connected_nodes(G)
    _snapshot_stage("remove_edges_for_self_connected_nodes", G, history, tracker)

    G = graph.reconnect_orphan_and_dangling_nodes(
        G,
        skeleton_data=skeleton,
        reconnect_threshold=final_orphan_reconnect_threshold,
        include_degree1=True,
        max_new_edges_per_node=1,
        validate_reconnections=True,
        debug=verbose_logging,
    )
    _snapshot_stage("reconnect_orphan_and_dangling_nodes", G, history, tracker)

    G = graph.smart_multigraph_degree2_removal(
        G, skeleton, max_degree=degree2_pass1_max_degree, debug=verbose_logging
    )
    _snapshot_stage("smart_multigraph_degree2_removal_post_orphan_reconnect", G, history, tracker)

    final_diagnosis = _diagnose_final_degree0_reasons(
        G,
        skeleton=skeleton,
        reconnect_threshold=final_orphan_reconnect_threshold,
    )

    print("\n=== Stage-by-stage degree-0 counts ===")
    for stage in history:
        print(
            f"- {stage['stage']}: degree0={stage['degree0_count']} "
            f"(nodes={stage['node_count']}, edges={stage['edge_count']})"
        )

    print("\n=== Final degree-0 diagnosis ===")
    if not final_diagnosis:
        print("No degree-0 nodes remain after graph building.")
    else:
        for row in final_diagnosis:
            print(
                f"- node {row['node_id']}: {row['reason']} "
                f"(nearest={row['nearest_target_node_id']}, "
                f"dist={row['nearest_target_distance']}, "
                f"valid={row['skeleton_path_valid']})"
            )

    report = {
        "settings_summary": {
            "image_path": str(image_path),
            "voxel_size_xyz": list(voxel_size),
            "graph_reconnect_threshold": reconnect_threshold,
            "final_orphan_reconnect_threshold": final_orphan_reconnect_threshold,
            "cluster_collapse_distance": cluster_collapse_distance,
            "min_stub_length": min_stub_length,
            "skeleton_meta": skeleton_meta,
        },
        "history": history,
        "tracked_degree0_groups": tracker,
        "final_degree0_diagnosis": final_diagnosis,
    }

    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    print(f"\nDetailed JSON report written to: {output_json}")


if __name__ == "__main__":
    main()
