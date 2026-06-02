#!/usr/bin/env python3
# AUTO-GENERATED from tutorials/pipeline_tutorial.ipynb — do not edit manually.
# Regenerate: pytest tests/integration/test_pipeline_tutorial.py

# coding: utf-8

# # ImageLynx pipeline tutorial
# 
# This notebook is the **source of truth** for the step-by-step tutorial on `tests/data/Nerve_capillaries_cropped.tif`. `pipeline_tutorial.py` in this folder is **auto-generated** when you run `pytest tests/integration/test_pipeline_tutorial.py` — edit the notebook, not the `.py` file.
# 
# ## Input requirement: pre-segmented binary mask
# 
# **ImageLynx operates on a segmented binary volume, not raw fluorescence.** The input TIFF must already be a binary mask where vessel voxels are foreground.
# 
# | Stage | What it does |
# |-------|----------------|
# | 0 | Segmentation note — how to go from raw to binary mask |
# | 1 | Load segmented TIFF, skeletonize, preprocess skeleton |
# | 2 | `graph.build_graph_from_skeleton` — topology repair |
# | 3 | Boundary nodes (volume boxes) and branch orders |
# | 4 | Poiseuille conductances and two-point resistance |
# | 5 | VTK export and pressure-driven flow solve |
# | 6 | Network statistics CSV |
# 
# **Plots:** set `SHOW_STAGE_PLOTS = True` in the configuration cell to display figures inline. Stage 2 shows **five milestone** graph-topology overlays plus the final graph (the full 11-step pipeline still runs; intermediate degree-2 steps are not plotted).
# 

# ## Prerequisites
# 
# ```bash
# pip install -e ".[dev]"
# ```
# 
# Run from the repository root or from `tutorials/`.

# In[3]:


from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np


def _resolve_tutorial_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd().resolve()
        if (cwd / "pipeline_tutorial.ipynb").exists():
            return cwd
        if (cwd / "tutorials" / "pipeline_tutorial.ipynb").exists():
            return cwd / "tutorials"
        for parent in [cwd, *cwd.parents]:
            if (parent / "tutorials" / "pipeline_tutorial.ipynb").exists():
                return parent / "tutorials"
        return cwd


TUTORIAL_DIR = _resolve_tutorial_dir()


def _resolve_repo_root(tutorial_dir: Path) -> Path:
    env_root = os.environ.get("IMAGELYNX_REPO_ROOT")
    if env_root:
        return Path(env_root)
    for start in [tutorial_dir, tutorial_dir.parent, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (start / "src" / "ImageLynx").is_dir() and (start / "examples" / "resistance_network_pipeline.py").is_file():
            return start
    return tutorial_dir.parent


REPO_ROOT = _resolve_repo_root(TUTORIAL_DIR)
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
for p in (SRC_DIR, EXAMPLES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz
from resistance_pipeline_settings import (
    DIAMETER_BY_BRANCH_ORDER,
    custom_edges,
)

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))
from tutorial_plots import GraphBuildPlotter, in_jupyter, show_saved_plot, show_stage_plots

print(f"Repository root: {REPO_ROOT}")
print(f"Running in Jupyter: {in_jupyter()}")


# ## Configuration
# 
# Paths and skeleton/graph parameters. Inlet/outlet volume boxes are derived from image shape (top/bottom 20% of Y).

# In[4]:


INPUT_TIFF = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"
OUTPUT_DIR = Path(os.environ.get("IMAGELYNX_TUTORIAL_OUTPUT_DIR", str(TUTORIAL_DIR / "outputs")))
PLOT_DIR = Path(os.environ.get("IMAGELYNX_TUTORIAL_PLOT_DIR", str(TUTORIAL_DIR / "plots"))).resolve()
VTK_PREFIX = OUTPUT_DIR / "nerve_capillaries_tutorial"
OUTPUT_DIR = OUTPUT_DIR.resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

stem = INPUT_TIFF.stem
skeleton_path = OUTPUT_DIR / f"{stem}_skeleton.npy"
voxel_meta_path = OUTPUT_DIR / f"{stem}_voxel_size.json"
graph_path = OUTPUT_DIR / f"{stem}_graph.pkl"

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

# Display saved stage PNGs inline after each section (notebook only).
SHOW_STAGE_PLOTS = True

print(f"Input: {INPUT_TIFF}")
print(f"Outputs: {OUTPUT_DIR}")
print(f"Plots: {PLOT_DIR}")
print(f"Inline stage plots enabled: {SHOW_STAGE_PLOTS and in_jupyter()}")


# ## Stage 0: Segmentation (prerequisite — not run in this notebook)
# 
# ImageLynx requires a **pre-segmented binary volume** as input. The pipeline does not perform segmentation itself; it expects the vessels to already be labelled.
# 
# `tests/data/Nerve_capillaries_cropped.tif` used here is already a binary mask (0 = background, 255 = vessel), so Stage 0 is not executed. For your own raw fluorescence data you must segment it first.
# 
# ### Option A — ilastik (recommended)
# 
# Train a Pixel Classification project in [ilastik](https://www.ilastik.org/) on your image, export a "Simple Segmentation" classifier, then call:
# 
# ```python
# from ImageLynx import io
# 
# segmented_path = io.run_ilastik_headless_segmentation(
#     input_image_path="path/to/raw_image.tif",
#     classifier_path="path/to/classifier.ilp",
#     output_path="path/to/segmented_output.tif",
#     ilastik_executable="/path/to/ilastik",   # or "ilastik.exe" on Windows
# )
# ```
# 
# Pass `segmented_path` as `INPUT_TIFF` in the configuration cell below.
# 
# The full pipeline script (`examples/resistance_network_pipeline.py`) can run this automatically via the `use_ilastik_segmentation=True` parameter.
# 
# ### Option B — any other segmentation tool
# 
# Any tool that produces a 3D TIFF (or H5) binary/label mask works: Fiji, napari, StarDist, cellpose, etc. Save the segmented result and point `INPUT_TIFF` at it.
# 
# ### Option C — thresholding in Python
# 
# For simple cases where global thresholding is sufficient:
# 
# ```python
# import tifffile, numpy as np
# raw = tifffile.imread("path/to/raw_image.tif")
# binary = raw > threshold_value           # adjust threshold to your data
# segmented = (binary * 255).astype(np.uint8)
# tifffile.imwrite("path/to/segmented.tif", segmented)
# ```
# 
# ---
# 
# ## Stage 1: Load and skeletonize
# 
# Load the segmented TIFF, extract a 3D skeleton, clean it with `preprocess_skeleton_for_graph`, and save intermediate artifacts.

# In[5]:


(
    image,
    skeleton,
    voxel_size_x,
    voxel_size_y,
    voxel_size_z,
    voxel_meta_status,
) = io.load_and_skeletonize_3d_tif(INPUT_TIFF)
metadata_voxel_size = (float(voxel_size_x), float(voxel_size_y), float(voxel_size_z))
voxel_size, voxel_size_source = resolve_voxel_size_xyz(
    metadata_voxel_size_xyz=metadata_voxel_size,
    metadata_status=voxel_meta_status,
    voxel_size_override_xyz=None,
    voxel_size_policy="auto",
)
print(f"Image shape: {image.shape[:3]}, voxel size: {voxel_size}")

preprocessing.print_skeleton_connectivity_stats("raw", skeleton, component_connectivity=SKELETON_COMPONENT_CONNECTIVITY)
raw_skeleton_path = PLOT_DIR / "raw_skeleton.png"
visualization.visualize_skeleton(skeleton, save_path=raw_skeleton_path)
if SHOW_STAGE_PLOTS:
    show_saved_plot(raw_skeleton_path, title="Raw skeleton (Z-projection)")

skeleton = preprocessing.preprocess_skeleton_for_graph(
    skeleton,
    min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
    max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
    component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
    min_component_fraction=SKELETON_MIN_COMPONENT_PERCENT / 100.0,
    closing_radius=SKELETON_CLOSING_RADIUS,
    bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
)
preprocessing.print_skeleton_connectivity_stats("cleaned", skeleton, component_connectivity=SKELETON_COMPONENT_CONNECTIVITY)
np.save(skeleton_path, skeleton)
voxel_meta_path.write_text(json.dumps({"voxel_size": voxel_size, "voxel_size_source": voxel_size_source}))
skeleton_projection_path = PLOT_DIR / "skeleton_projection.png"
visualization.visualize_skeleton(skeleton, save_path=skeleton_projection_path)
if SHOW_STAGE_PLOTS:
    show_saved_plot(skeleton_projection_path, title="Cleaned skeleton (Z-projection)")

cropped_z, cropped_y, cropped_x = image.shape[:3]
y_band = max(1, int(0.2 * cropped_y))
starting_node_volumes = [((0.0, 0.0, 0.0), (float(cropped_z - 1), float(y_band - 1), float(cropped_x - 1)))]
output_node_volumes = [((0.0, float(cropped_y - y_band), 0.0), (float(cropped_z - 1), float(cropped_y - 1), float(cropped_x - 1)))]
print(f"Inlet box: {starting_node_volumes[0]}")
print(f"Outlet box: {output_node_volumes[0]}")

show_stage_plots(
    "Stage 1: Skeleton (summary)",
    [raw_skeleton_path, skeleton_projection_path],
    enabled=SHOW_STAGE_PLOTS,
)


# ## Stage 2: Build vascular graph
# 
# `build_graph_from_skeleton` runs the full topology pipeline (11 steps). The tutorial shows **5 milestone overlays** (initial build, after optimisation, after cluster collapse, after stub pruning, after orphan reconnection), then a final graph plot. Intermediate degree-2 cleanup steps are still executed but not plotted.
# 

# In[ ]:


graph_plotter = GraphBuildPlotter(
    image,
    PLOT_DIR,
    show_inline=False,
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
    step_callback=_print_step,
)
G.graph["image_voxel_size_xyz"] = tuple(float(v) for v in voxel_size)
with graph_path.open("wb") as fh:
    pickle.dump(G, fh)
print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

graph_plotter.display_all("Stage 2: Graph topology steps", enabled=SHOW_STAGE_PLOTS)

final_graph_path = PLOT_DIR / "final_graph.png"
visualization.visualize_edges_and_nodes(
    image,
    G,
    label_nodes=False,
    save_path=final_graph_path,
    show_coordinates_degree_1=True,
    show=False,
)
show_stage_plots("Stage 2: Final graph", [final_graph_path], enabled=SHOW_STAGE_PLOTS)


# ## Stage 3: Boundary nodes and branch orders
# 
# Select inlet/outlet nodes inside the volume boxes, then call `graph.assign_vessel_branch_orders` (capillary `B*` orders, or hierarchical `Art*`/`Ven*`/`B*` when boundary sets are provided).

# In[ ]:


starting_nodes = graph.select_boundary_nodes_by_method(
    G, image.shape, method="volume", node_role="input",
    coordinates=[], volume_boxes=starting_node_volumes,
)
output_nodes = graph.select_boundary_nodes_by_method(
    G, image.shape, method="volume", node_role="output",
    coordinates=[], volume_boxes=output_node_volumes, exclude_nodes=starting_nodes,
)
print(f"Starting nodes: {starting_nodes}")
print(f"Output nodes: {output_nodes}")
branch_summary = graph.assign_vessel_branch_orders(
    G, starting_nodes, output_nodes=output_nodes,
)
print(f"Branch order mode: {branch_summary['mode']}")
resistance_node_pair = (starting_nodes[0], output_nodes[0])
print(f"Resistance pair: {resistance_node_pair}")

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

# ## Stage 4: Haemodynamics (Poiseuille)
# 
# Call `haemodynamics.apply_poiseuille_haemodynamics` to assign conductances from branch-order diameters, then build the conductance matrix and compute two-point equivalent resistance.

# In[ ]:


G, haemo_results = haemodynamics.apply_poiseuille_haemodynamics(
    G,
    diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
    custom_edges=custom_edges,
)
print(f"Weight assignment: {haemo_results.get('weights', haemo_results)}")

conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
r = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
    laplacian, G, resistance_node_pair[0], resistance_node_pair[1],
)
print(f"Two-point resistance: {r}")

# ## Stage 5: VTK export and flow solve
# 
# Export vessels/pericytes/nodes to VTK, then solve pressure-driven flow with inlet/outlet boundary conditions.

# In[ ]:


vtk_export = visualization.graph_to_vtk(G, VTK_PREFIX)
flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
    conductance, node_list, INPUT_P_BC, OUTPUT_P_BC,
    starting_nodes, output_nodes, vtk_export,
)
print(f"VTK with flow: {vtk_export['vessels_path']}")

# ## Stage 6: Statistics
# 
# Compute network metrics and export to CSV.

# In[ ]:


node_positions = nx.get_node_attributes(G, "pos")
stats = statistics.compute_comprehensive_vessel_statistics(
    G, node_positions=node_positions, image_dimensions=image.shape, statistics_mode="fast",
)
stats_csv = OUTPUT_DIR / f"{stem}_statistics.csv"
statistics.export_statistics_to_csv(stats, stats_csv)
for key, value in list(stats.items())[:8]:
    print(f"{key}: {value}")
print(f"Saved: {stats_csv}")

# ## Adapting for your own data
# 
# 1. **Copy and edit this notebook** (`pipeline_tutorial.ipynb`), not the generated `.py` file.
# 2. Set `INPUT_TIFF` to your **segmented** binary volume (see Stage 0 if you start from raw images).
# 3. Adjust volume boxes or use `coordinates` / mask-based selection (see `examples/resistance_pipeline_settings.py`).
# 4. Tune skeleton and `build_graph_from_skeleton` parameters for your resolution.
# 5. For FWHM diameters, pericytes, or ilastik: use `examples/resistance_network_pipeline.py` with presets.
# 6. Regenerate `pipeline_tutorial.py` after notebook edits: `pytest tests/integration/test_pipeline_tutorial.py`.
# 
