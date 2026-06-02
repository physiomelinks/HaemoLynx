#!/usr/bin/env python3
# ============================================================
# Pipeline tutorial - companion to tutorials/pipeline_tutorial.ipynb
#
# To adapt this pipeline for your own data, copy this file and
# modify the configuration section below for your use-case.
#
# INPUT REQUIREMENT: pre-segmented binary mask
# ============================================================
# ImageLynx requires a segmented binary volume as input — NOT a raw
# fluorescence image. The input TIFF must already have vessel voxels
# labelled as foreground.
#
# To segment a raw image, use one of:
#   A) ilastik (recommended):
#         from ImageLynx import io
#         segmented_path = io.run_ilastik_headless_segmentation(
#             input_image_path="raw.tif",
#             classifier_path="classifier.ilp",
#             output_path="segmented.tif",
#             ilastik_executable="/path/to/ilastik",
#         )
#   B) Any other tool (Fiji, napari, StarDist, etc.) that produces a
#      3D binary/label TIFF or H5.
#   C) Simple threshold in Python:
#         import tifffile, numpy as np
#         raw = tifffile.imread("raw.tif")
#         segmented = (raw > threshold_value * 255).astype(np.uint8)
#         tifffile.imwrite("segmented.tif", segmented)
#
# The tutorial input (Nerve_capillaries_cropped.tif) is already a
# binary mask (0/255) and does not need segmentation.
# ============================================================
"""Step-by-step ImageLynx pipeline tutorial on Nerve_capillaries_cropped.tif."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz


def _resolve_tutorial_dir() -> Path:
    """Resolve tutorials/ whether run as a script or from a notebook."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        if (cwd / "pipeline_tutorial.ipynb").exists():
            return cwd
        if (cwd / "tutorials" / "pipeline_tutorial.ipynb").exists():
            return cwd / "tutorials"
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "tutorials" / "pipeline_tutorial.ipynb"
            if candidate.exists():
                return parent / "tutorials"
        return cwd


TUTORIAL_DIR = _resolve_tutorial_dir()


def _resolve_repo_root(tutorial_dir: Path) -> Path:
    env_root = os.environ.get("IMAGELYNX_REPO_ROOT")
    if env_root:
        return Path(env_root)
    for start in [
        tutorial_dir,
        tutorial_dir.parent,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]:
        if (start / "src" / "ImageLynx").is_dir() and (
            start / "examples" / "resistance_network_pipeline.py"
        ).is_file():
            return start
    return tutorial_dir.parent


REPO_ROOT = _resolve_repo_root(TUTORIAL_DIR)
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from tutorial_plots import GraphBuildPlotter, show_stage_plots  # noqa: E402
from resistance_pipeline_settings import (  # noqa: E402
    CONSTRICTION_BY_BRANCH_ORDER,
    DIAMETER_BY_BRANCH_ORDER,
    custom_edges,
)

# ---------------------------------------------------------------------------
# Configuration — edit these paths and parameters for your own use-case
# ---------------------------------------------------------------------------
INPUT_TIFF = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"
OUTPUT_DIR = Path(
    os.environ.get(
        "IMAGELYNX_TUTORIAL_OUTPUT_DIR",
        str(TUTORIAL_DIR / "outputs"),
    )
)
PLOT_DIR = Path(
    os.environ.get(
        "IMAGELYNX_TUTORIAL_PLOT_DIR",
        str(TUTORIAL_DIR / "plots"),
    )
)
VTK_PREFIX = OUTPUT_DIR / "nerve_capillaries_tutorial"

SKELETON_CLOSING_RADIUS = 1
SKELETON_BRIDGE_GAP_SIZE = 1
SKELETON_MIN_BRANCH_LENGTH = 3
SKELETON_MAX_BRIDGE_DISTANCE = 2
SKELETON_COMPONENT_CONNECTIVITY = 3
SKELETON_MIN_COMPONENT_PERCENT = 1.0

GRAPH_RECONNECT_THRESHOLD = 10.0
FINAL_ORPHAN_RECONNECT_THRESHOLD = 3.0
CLUSTER_COLLAPSE_DISTANCE = 5.0
MIN_STUB_LENGTH = 3.0

INPUT_P_BC = 1000.0
OUTPUT_P_BC = 500.0

# When True, display saved stage PNGs inline in Jupyter (no-op in plain Python).
SHOW_STAGE_PLOTS = True


def _volume_boundary_boxes(shape: tuple[int, ...]) -> tuple[list, list]:
    cropped_z, cropped_y, cropped_x = shape[:3]
    y_band = max(1, int(0.2 * cropped_y))
    starting_node_volumes = [
        (
            (0.0, 0.0, 0.0),
            (float(cropped_z - 1), float(y_band - 1), float(cropped_x - 1)),
        )
    ]
    output_node_volumes = [
        (
            (0.0, float(cropped_y - y_band), 0.0),
            (float(cropped_z - 1), float(cropped_y - 1), float(cropped_x - 1)),
        )
    ]
    return starting_node_volumes, output_node_volumes


def main() -> None:
    if not INPUT_TIFF.exists():
        raise FileNotFoundError(f"Tutorial input TIFF not found: {INPUT_TIFF}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    stem = INPUT_TIFF.stem
    skeleton_path = OUTPUT_DIR / f"{stem}_skeleton.npy"
    voxel_meta_path = OUTPUT_DIR / f"{stem}_voxel_size.json"
    graph_path = OUTPUT_DIR / f"{stem}_graph.pkl"

    # --- Stage 1: Load and skeletonize ---
    print("\n=== Stage 1: Load and skeletonize ===")
    (
        image,
        skeleton,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_meta_status,
    ) = io.load_and_skeletonize_3d_tif(INPUT_TIFF)
    metadata_voxel_size = (
        float(voxel_size_x),
        float(voxel_size_y),
        float(voxel_size_z),
    )
    voxel_size, voxel_size_source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=voxel_meta_status,
        voxel_size_override_xyz=None,
        voxel_size_policy="auto",
    )
    print(f"Voxel size (x,y,z): {voxel_size} (source={voxel_size_source})")

    preprocessing.print_skeleton_connectivity_stats(
        "raw",
        skeleton,
        component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
    )
    visualization.visualize_skeleton(
        skeleton, save_path=PLOT_DIR / "raw_skeleton.png"
    )

    skeleton = preprocessing.preprocess_skeleton_for_graph(
        skeleton,
        min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
        max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
        component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
        min_component_fraction=SKELETON_MIN_COMPONENT_PERCENT / 100.0,
        closing_radius=SKELETON_CLOSING_RADIUS,
        bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
    )
    preprocessing.print_skeleton_connectivity_stats(
        "cleaned",
        skeleton,
        component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
    )
    np.save(skeleton_path, skeleton)
    voxel_meta_path.write_text(
        json.dumps(
            {
                "voxel_size": voxel_size,
                "voxel_size_source": voxel_size_source,
                "voxel_metadata_status": voxel_meta_status,
            }
        )
    )
    visualization.visualize_skeleton(
        skeleton, save_path=PLOT_DIR / "skeleton_projection.png"
    )
    print(f"Saved skeleton: {skeleton_path}")
    show_stage_plots(
        "Stage 1: Skeleton",
        [
            PLOT_DIR / "raw_skeleton.png",
            PLOT_DIR / "skeleton_projection.png",
        ],
        enabled=SHOW_STAGE_PLOTS,
    )

    starting_node_volumes, output_node_volumes = _volume_boundary_boxes(image.shape)

    # --- Stage 2: Build vascular graph ---
    print("\n=== Stage 2: Build vascular graph ===")

    graph_plotter = GraphBuildPlotter(
        image,
        PLOT_DIR,
        show_inline=SHOW_STAGE_PLOTS,
        label_nodes=False,
    )

    def _print_step(graph_obj: nx.MultiGraph, label: str) -> None:
        print(
            f"  [{label}] nodes={graph_obj.number_of_nodes()}, "
            f"edges={graph_obj.number_of_edges()}"
        )
        graph_plotter(graph_obj, label)

    G = graph.build_graph_from_skeleton(
        skeleton,
        voxel_size=tuple(float(v) for v in voxel_size),
        graph_reconnect_threshold=GRAPH_RECONNECT_THRESHOLD,
        final_orphan_reconnect_threshold=FINAL_ORPHAN_RECONNECT_THRESHOLD,
        cluster_collapse_distance=CLUSTER_COLLAPSE_DISTANCE,
        min_stub_length=MIN_STUB_LENGTH,
        debug=False,
        step_callback=_print_step,
    )
    graph_plotter.display_all("Stage 2: Graph topology steps", enabled=SHOW_STAGE_PLOTS)
    G.graph["image_voxel_size_xyz"] = tuple(float(v) for v in voxel_size)
    with graph_path.open("wb") as fh:
        pickle.dump(G, fh)
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved graph: {graph_path}")

    final_graph_path = PLOT_DIR / "final_graph.png"
    visualization.visualize_edges_and_nodes(
        image,
        G,
        label_nodes=False,
        save_path=final_graph_path,
        show_coordinates_degree_1=True,
        show=False,
    )
    show_stage_plots(
        "Stage 2: Final graph",
        [final_graph_path],
        enabled=SHOW_STAGE_PLOTS,
    )

    # --- Stage 3: Boundary nodes and branch orders ---
    print("\n=== Stage 3: Boundary nodes and branch orders ===")
    starting_nodes = graph.select_boundary_nodes_by_method(
        G,
        image.shape,
        method="volume",
        node_role="input",
        coordinates=[],
        volume_boxes=starting_node_volumes,
    )
    output_nodes = graph.select_boundary_nodes_by_method(
        G,
        image.shape,
        method="volume",
        node_role="output",
        coordinates=[],
        volume_boxes=output_node_volumes,
        exclude_nodes=starting_nodes,
    )
    print(f"Starting nodes: {starting_nodes}")
    print(f"Output nodes: {output_nodes}")
    if not starting_nodes or not output_nodes:
        raise ValueError("No inlet/outlet nodes selected; adjust volume boxes.")

    branch_summary = graph.assign_vessel_branch_orders(
        G,
        starting_nodes,
        output_nodes=output_nodes,
    )
    print(f"Branch order assignment: {branch_summary['mode']}")
    resistance_node_pair = (starting_nodes[0], output_nodes[0])
    print(f"Resistance node pair: {resistance_node_pair}")

    branch_orders_path = PLOT_DIR / "geometry_with_branch_orders.png"
    visualization.visualize_geometry_with_branch_orders(
        image,
        G,
        group_above=8,
        save_path=branch_orders_path,
        show=False,
    )
    show_stage_plots(
        "Stage 3: Branch orders",
        [branch_orders_path],
        enabled=SHOW_STAGE_PLOTS,
    )

    # --- Stage 4: Haemodynamics ---
    print("\n=== Stage 4: Haemodynamics (Poiseuille) ===")
    G, haemo_results = haemodynamics.apply_poiseuille_haemodynamics(
        G,
        diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
        custom_edges=custom_edges,
    )
    print(f"Haemodynamics results: {haemo_results.get('weights', haemo_results)}")

    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    source_node, target_node = resistance_node_pair
    if source_node in node_to_idx and target_node in node_to_idx:
        laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
        two_point_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
            laplacian,
            G,
            source_node,
            target_node,
        )
        print(f"Two-point resistance ({source_node} -> {target_node}): {two_point_resistance}")

    # --- Stage 5: VTK export and flow solve ---
    print("\n=== Stage 5: VTK export and flow solve ===")
    vtk_export = visualization.graph_to_vtk(G, VTK_PREFIX)
    print(f"Vessels VTK: {vtk_export['vessels_path']}")
    flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        INPUT_P_BC,
        OUTPUT_P_BC,
        starting_nodes,
        output_nodes,
        vtk_export,
    )
    print(f"Flow solve complete; VTK with flow: {vtk_export['vessels_path']}")

    # --- Stage 6: Statistics ---
    print("\n=== Stage 6: Statistics ===")
    node_positions = nx.get_node_attributes(G, "pos")
    stats = statistics.compute_comprehensive_vessel_statistics(
        G,
        node_positions=node_positions,
        image_dimensions=image.shape,
        statistics_mode="fast",
    )
    stats_csv_path = OUTPUT_DIR / f"{stem}_statistics.csv"
    statistics.export_statistics_to_csv(stats, stats_csv_path)
    print("Sample statistics:")
    for key, value in list(stats.items())[:8]:
        print(f"  {key}: {value}")
    print(f"Saved statistics CSV: {stats_csv_path}")

    vessels_flow_path = VTK_PREFIX.with_name(VTK_PREFIX.name + "_vessels_flow.vtp")
    if not vessels_flow_path.exists():
        raise FileNotFoundError(f"Expected VTK flow file: {vessels_flow_path}")

    print("\n=== Tutorial complete ===")
    print(f"Graph: {graph_path}")
    print(f"VTK (flow): {vessels_flow_path}")


if __name__ == "__main__":
    main()
