#!/usr/bin/env python3
# AUTO-GENERATED from tutorials/pipeline_tutorial.ipynb — do not edit manually.
# Regenerate: pytest tests/integration/test_pipeline_tutorial.py

# coding: utf-8

# # ImageLynx pipeline tutorial
# 
# This notebook runs each stage of the image-to-model pipeline **explicitly**, using the public `ImageLynx` API (not a single call to `image_to_model_pipeline`).
# 
# | Stage | What it does |
# |-------|----------------|
# | 0 | **Your raw image** — train an ilastik classifier, segment headlessly |
# | 1 | **Pre-segmented mask** — load (tutorial default or your Stage 0 output), skeletonize |
# | 2 | `graph.build_graph_from_skeleton` — topology repair |
# | 3 | Boundary nodes (volume boxes) and branch orders |
# | 4 | Poiseuille conductances and two-point resistance |
# | 5 | VTK export and pressure-driven flow solve |
# | 6 | Network statistics CSV |
# 
# **Workflow:** Stage 0 uses **your own unsegmented** microscopy volume. Stage 1 onward needs a **binary vessel mask** — either the bundled tutorial TIFF or the mask you produced in Stage 0.
# 
# **Plots:** PNGs under `tutorials/plots/`. Set `SHOW_STAGE_PLOTS = True` for inline figures.
# 
# **Source of truth:** edit **this notebook only**. [`pipeline_tutorial.py`](pipeline_tutorial.py) is auto-generated (do not edit by hand). Regenerate it after notebook changes:
# 
# ```bash
# pytest tests/integration/test_pipeline_tutorial.py
# ```

# ## Prerequisites
# 
# ```bash
# pip install -e ".[dev]"
# ```
# 
# - **ImageLynx** (above) for Stages 1–6.
# - **[ilastik](https://www.ilastik.org/download/)** (separate install) for Stage 0 — only if you segment your own raw image.
# 
# Run from the repository root or from `tutorials/`.

# In[ ]:


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
from ImageLynx.pipeline import resolve_settings
from resistance_pipeline_schema import SCHEMA

# Settings come from the pipeline's config file, so the tutorial and the
# example agree by construction rather than by being kept in step by hand.
# resolve_settings also fills in the tables derived from other settings.
PIPELINE_SETTINGS = resolve_settings(
    schema=SCHEMA, config_path=EXAMPLES_DIR / "resistance_pipeline_config.yaml"
)
DIAMETER_BY_BRANCH_ORDER = PIPELINE_SETTINGS["diameter_by_branch_order"]
custom_edges = PIPELINE_SETTINGS["custom_edges"]

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))
from tutorial_plots import GraphBuildPlotter, in_jupyter, show_stage_plots

print(f"Repository root: {REPO_ROOT}")
print(f"Running in Jupyter: {in_jupyter()}")

# ## Configuration (shared)
# 
# Output directories and skeleton/graph parameters used from Stage 1 onward. **Segmented input path is chosen in Stage 1** (tutorial default or your Stage 0 mask).

# In[ ]:


OUTPUT_DIR = Path(os.environ.get("IMAGELYNX_TUTORIAL_OUTPUT_DIR", str(TUTORIAL_DIR / "outputs"))).resolve()
PLOT_DIR = Path(os.environ.get("IMAGELYNX_TUTORIAL_PLOT_DIR", str(TUTORIAL_DIR / "plots"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Meaning of each input array axis. Volumes load as canonical (z, y, x), so this
# selects which axis is z — the axis projections and overlays look through.
IMAGE_AXIS_ORDER = "zyx"
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

print(f"Outputs: {OUTPUT_DIR}")
print(f"Plots: {PLOT_DIR}")
print(f"Inline stage plots enabled: {SHOW_STAGE_PLOTS and in_jupyter()}")

# ## Stage 0: Segment **your** raw image with ilastik
# 
# ImageLynx Stages 1–6 need a **binary vessel mask** (foreground = vessel). This stage turns **your unsegmented** 3D TIFF into that mask using [ilastik](https://www.ilastik.org/) Pixel Classification.
# 
# > **Training happens in the ilastik GUI** (interactive). **Inference** can run headlessly from the code cell below once you have a saved `.ilp` project.
# 
# ### Step 0.1 — Install ilastik
# 
# 1. Download ilastik from [ilastik.org/download](https://www.ilastik.org/download/) for your OS.
# 2. Note the executable path, e.g. Linux `~/ilastik-1.4.0-Linux/ilastik`, macOS `ilastik.app/Contents/MacOS/ilastik`, Windows `ilastik.exe`.
# 3. Set `ILASTIK_EXECUTABLE` in the code cell to that path (or add it to your `PATH`).
# 
# ### Step 0.2 — Create a Pixel Classification project
# 
# 1. Launch ilastik → **Create new project** → **Pixel Classification**.
# 2. **Input data:** Add your **raw** 3D volume (the same file you will set as `RAW_IMAGE_PATH`). Use TIFF or H5; shape should be `(Z, Y, X)` or `(Y, X, Z)` consistent with how you acquired the data.
# 3. **Features:** Default sigma features are usually fine for vessels; add more scales if capillaries are very thin or very thick.
# 4. **Labels:** Paint at least two classes on representative slices:
#    - **Label 1** — vessel / capillary (foreground)
#    - **Label 2** — background (or tissue you want excluded)
#    Sample multiple Z positions and both bright/dim regions.
# 5. **Live update:** Enable live update and iterate labels until the preview segmentation looks acceptable on several slices.
# 6. **Train:** Click **Train** (or wait for live update) until the classifier stabilises.
# 7. **Save project:** **File → Save project as…** → e.g. `my_vessel_classifier.ilp`. This file is your **classifier** for headless runs.
# 
# ### Step 0.3 — Check export settings (important)
# 
# In the **Prediction** / export section of the Pixel Classification workflow, ensure **Simple Segmentation** is available as an export source (this is what ImageLynx requests headlessly). The saved `.ilp` embeds these settings.
# 
# ### Step 0.4 — Run headless segmentation (code cell below)
# 
# 1. Set `RAW_IMAGE_PATH` to your raw TIFF.
# 2. Set `ILASTIK_CLASSIFIER_PATH` to your `.ilp` file.
# 3. Set `RUN_STAGE_0_ILASTIK = True` and run the cell.
# 
# Output is written to `SEGMENTED_OUTPUT_PATH`. Then enable **your mask in Stage 1** (`USE_CUSTOM_SEGMENTED_IMAGE = True`) and run Stages 1–6.
# 
# ### Skipping Stage 0 (tutorial demo only)
# 
# Leave `RUN_STAGE_0_ILASTIK = False` to skip segmentation and use the bundled pre-segmented `tests/data/Nerve_capillaries_cropped.tif` in Stage 1.
# 
# ### Alternatives to ilastik
# 
# If you already have a mask from Fiji, napari, cellpose, etc., save it as a 3D TIFF and set `USE_CUSTOM_SEGMENTED_IMAGE = True` in Stage 1 with that path — you can skip Stage 0 entirely.

# In[ ]:


# --- Stage 0: edit paths to your data ---
RUN_STAGE_0_ILASTIK = False  # True after you have a trained .ilp classifier

RAW_IMAGE_PATH = Path("/path/to/your/raw_microscopy.tif")
ILASTIK_CLASSIFIER_PATH = Path("/path/to/your_pixel_classifier.ilp")
ILASTIK_EXECUTABLE = os.environ.get("ILASTIK_EXECUTABLE", "ilastik")

SEGMENTED_OUTPUT_DIR = OUTPUT_DIR / "segmentation"
SEGMENTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_OUTPUT_PATH = SEGMENTED_OUTPUT_DIR / f"{RAW_IMAGE_PATH.stem}_segmented.tif"

# Integration tests set this to skip ilastik even if RUN_STAGE_0_ILASTIK were True.
if os.environ.get("IMAGELYNX_SKIP_ILASTIK", "").lower() in ("1", "true", "yes"):
    RUN_STAGE_0_ILASTIK = False

if RUN_STAGE_0_ILASTIK:
    raw_path = RAW_IMAGE_PATH.expanduser().resolve()
    classifier_path = ILASTIK_CLASSIFIER_PATH.expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw image not found: {raw_path}\n"
            "Set RAW_IMAGE_PATH to your unsegmented 3D TIFF."
        )
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found: {classifier_path}\n"
            "Train a Pixel Classification project in ilastik and save a .ilp file."
        )
    print(f"Segmenting: {raw_path}")
    print(f"Classifier: {classifier_path}")
    print(f"Executable: {ILASTIK_EXECUTABLE}")
    segmented_path = io.run_ilastik_headless_segmentation(
        input_image_path=raw_path,
        classifier_path=classifier_path,
        output_path=SEGMENTED_OUTPUT_PATH,
        ilastik_executable=ILASTIK_EXECUTABLE,
    )
    SEGMENTED_OUTPUT_PATH = Path(segmented_path)
    print(f"Saved segmented mask: {SEGMENTED_OUTPUT_PATH}")
    print("Next: set USE_CUSTOM_SEGMENTED_IMAGE = True in Stage 1 and run Stages 1–6.")
else:
    print("Stage 0 skipped (RUN_STAGE_0_ILASTIK=False).")
    print("Stage 1 will use the tutorial default mask unless USE_CUSTOM_SEGMENTED_IMAGE=True.")

# ## Stage 1: Load and skeletonize (pre-segmented mask)
# 
# Stages 1–6 operate on a **binary segmentation**, not raw fluorescence.
# 
# - **Default (no Stage 0):** `tests/data/Nerve_capillaries_cropped.tif` — small cropped mask for this tutorial.
# - **After Stage 0:** set `USE_CUSTOM_SEGMENTED_IMAGE = True` to use `SEGMENTED_OUTPUT_PATH` from ilastik.
# 
# The mask should be foreground voxels (e.g. 255) on background (0). ImageLynx binarizes automatically during skeletonization.

# In[ ]:


# --- Stage 1 input selection ---
DEFAULT_SEGMENTED_TIFF = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"

# True: use your Stage 0 mask (or any other segmented TIFF you specify).
USE_CUSTOM_SEGMENTED_IMAGE = False
CUSTOM_SEGMENTED_TIFF = SEGMENTED_OUTPUT_PATH  # from Stage 0; only used when USE_CUSTOM_SEGMENTED_IMAGE=True

if USE_CUSTOM_SEGMENTED_IMAGE:
    INPUT_TIFF = Path(CUSTOM_SEGMENTED_TIFF).expanduser().resolve()
else:
    env_input = os.environ.get("IMAGELYNX_TUTORIAL_INPUT_TIFF")
    INPUT_TIFF = Path(env_input).expanduser().resolve() if env_input else DEFAULT_SEGMENTED_TIFF.resolve()

if not INPUT_TIFF.exists():
    raise FileNotFoundError(
        f"Segmented input not found: {INPUT_TIFF}\n"
        "Run Stage 0 with ilastik, or set USE_CUSTOM_SEGMENTED_IMAGE=False for the tutorial default."
    )

stem = INPUT_TIFF.stem
skeleton_path = OUTPUT_DIR / f"{stem}_skeleton.npy"
voxel_meta_path = OUTPUT_DIR / f"{stem}_voxel_size.json"
graph_path = OUTPUT_DIR / f"{stem}_graph.pkl"
VTK_PREFIX = OUTPUT_DIR / f"{stem}_tutorial"

print(f"Stage 1 input (segmented): {INPUT_TIFF}")
print(f"USE_CUSTOM_SEGMENTED_IMAGE={USE_CUSTOM_SEGMENTED_IMAGE}")

# In[ ]:


(
    image,
    skeleton,
    voxel_size_x,
    voxel_size_y,
    voxel_size_z,
    voxel_meta_status,
) = io.load_and_skeletonize_3d_tif(INPUT_TIFF, axis_order=IMAGE_AXIS_ORDER)
metadata_voxel_size = (float(voxel_size_x), float(voxel_size_y), float(voxel_size_z))
voxel_size, voxel_size_source = resolve_voxel_size_xyz(
    metadata_voxel_size_xyz=metadata_voxel_size,
    metadata_status=voxel_meta_status,
    voxel_size_override_xyz=None,
    voxel_size_policy="auto",
)
# Metadata reports voxel size as (x, y, z); array axes are canonical (z, y, x).
voxel_size_zyx = io.voxel_size_zyx_from_xyz(voxel_size)
print(f"Image shape: {image.shape[:3]}, voxel size (x, y, z): {voxel_size}")
print(f"Array-axis spacing (z, y, x): {voxel_size_zyx}")

preprocessing.log_skeleton_connectivity_stats("raw", skeleton, component_connectivity=SKELETON_COMPONENT_CONNECTIVITY)
visualization.visualize_skeleton(skeleton, save_path=PLOT_DIR / "raw_skeleton.png")

skeleton = preprocessing.preprocess_skeleton_for_graph(
    skeleton,
    min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
    max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
    component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
    min_component_fraction=SKELETON_MIN_COMPONENT_PERCENT / 100.0,
    closing_radius=SKELETON_CLOSING_RADIUS,
    bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
)
preprocessing.log_skeleton_connectivity_stats("cleaned", skeleton, component_connectivity=SKELETON_COMPONENT_CONNECTIVITY)
np.save(skeleton_path, skeleton)
voxel_meta_path.write_text(json.dumps({"voxel_size": voxel_size, "voxel_size_source": voxel_size_source}))
visualization.visualize_skeleton(skeleton, save_path=PLOT_DIR / "skeleton_projection.png")

cropped_z, cropped_y, cropped_x = image.shape[:3]
y_band = max(1, int(0.2 * cropped_y))
starting_node_volumes = [((0.0, 0.0, 0.0), (float(cropped_z - 1), float(y_band - 1), float(cropped_x - 1)))]
output_node_volumes = [((0.0, float(cropped_y - y_band), 0.0), (float(cropped_z - 1), float(cropped_y - 1), float(cropped_x - 1)))]
print(f"Inlet box: {starting_node_volumes[0]}")
print(f"Outlet box: {output_node_volumes[0]}")

show_stage_plots(
    "Stage 1: Skeleton",
    [PLOT_DIR / "raw_skeleton.png", PLOT_DIR / "skeleton_projection.png"],
    enabled=SHOW_STAGE_PLOTS,
)

# ## Stage 2: Build vascular graph
# 
# `build_graph_from_skeleton` runs the full topology pipeline (skan extraction, loop stitching, degree-2 cleanup, stub pruning, orphan reconnection). The optional callback prints node/edge counts after each step.

# In[ ]:


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
    voxel_size=voxel_size_zyx,
    graph_reconnect_threshold=GRAPH_RECONNECT_THRESHOLD,
    final_orphan_reconnect_threshold=FINAL_ORPHAN_RECONNECT_THRESHOLD,
    cluster_collapse_distance=CLUSTER_COLLAPSE_DISTANCE,
    min_stub_length=MIN_STUB_LENGTH,
    step_callback=_print_step,
)
G.graph["image_voxel_size_xyz"] = tuple(float(v) for v in voxel_size)
G.graph["image_voxel_size_zyx"] = voxel_size_zyx
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

# ## Stage 5: Flow solve and VTK export
# 
# Solve pressure-driven flow with inlet/outlet boundary conditions, put the resulting flows on the graph, then export vessels, pericytes, and nodes to VTK (`.vtp`) in one pass.

# In[ ]:


flow = haemodynamics.solve_flow_from_conductance_matrix(
    conductance, node_list,
    input_p_bc=INPUT_P_BC, output_p_bc=OUTPUT_P_BC,
    starting_nodes=starting_nodes, output_nodes=output_nodes,
)
# Flows go onto the graph, so one export writes vessels and flow together.
haemodynamics.set_edge_flows(G, node_list, flow["pressure"])
vtk_export = visualization.graph_to_vtk(G, VTK_PREFIX)
print(f"VTK with flow: {vtk_export['vessels_path']}")
print(
    "Open the .vtp files in ParaView (or similar) to visualise vessels, nodes, and flow."
)


# ### View VTK outputs in ParaView
# 
# After the cell above finishes, open the exported `.vtp` files in [ParaView](https://www.paraview.org/) or similar VTK software (e.g. napari with a VTK plugin, [3D Slicer](https://www.slicer.org/)):
# 
# - ``stem`_tutorial_vessels.vtp` — vessel centreline geometry and attributes
# - the vessels file also carries **flow** scalars once the solve has run
# - ``stem`_tutorial_nodes.vtp` — graph nodes
# - ``stem`_tutorial_pericytes.vtp` — pericyte points (if present)
# 
# In ParaView: **File → Open**, select the `.vtp` files, click **Apply**, then colour by array names such as `flow` or `branch_order`.

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
# 1. **Edit this notebook** (`pipeline_tutorial.ipynb`) — not the generated `.py`.
# 2. **Stage 0:** set `RAW_IMAGE_PATH`, train ilastik, save `.ilp`, run with `RUN_STAGE_0_ILASTIK = True`.
# 3. **Stage 1:** set `USE_CUSTOM_SEGMENTED_IMAGE = True` to use your segmented mask.
# 4. Adjust inlet/outlet volume boxes or use coordinate/mask-based boundary selection (`examples/resistance_pipeline_config.yaml`).
# 5. Tune skeleton and `build_graph_from_skeleton` parameters for your resolution.
# 6. For FWHM diameters, pericytes, or automated ilastik in one script: `examples/resistance_network_pipeline.py`.
# 7. Regenerate `pipeline_tutorial.py`: `pytest tests/integration/test_pipeline_tutorial.py`.
