#!/usr/bin/env python3
# AUTO-GENERATED from tutorials/pipeline_tutorial.ipynb — do not edit manually.
# Regenerate: pytest tests/integration/test_pipeline_tutorial.py

# coding: utf-8

# # HaemoLynx pipeline tutorial
# 
# This notebook runs the image-to-model pipeline **one stage at a time**, calling
# the same functions `examples/resistance_network_pipeline.py` calls, in the same
# order. Nothing here is a tutorial-only code path: what you run below is the
# pipeline.
# 
# | Stage | Call | What it does |
# |-------|------|--------------|
# | 0 | `segment(settings)` | Pick the mask to analyse; run ilastik first if asked to |
# | 1 | `skeletonise(settings, inputs)` | Load the volume, resolve its voxel size, skeletonise |
# | 2 | `build_network(settings, volume, SCHEMA)` | Skeleton and vessel masks → graph |
# | 3 | `assign_boundaries(settings, network)` | Where flow enters and leaves |
# | 4 | `assign_diameters(...)` → `build_haemodynamic_model(...)` | Branch orders, diameters, resistances |
# | 5 | `solve(settings, model, boundaries)` | Pressures, flows, equivalent resistance |
# | 6 | `export_results(...)` | VTK, statistics, plots |
# 
# **Every stage takes the same `settings` dict**, loaded from a YAML config and
# described by a schema, so changing the run means changing a value in that dict
# — never editing a call. Each stage returns a small dataclass the next one takes:
# `SegmentedInputs` → `SkeletonisedVolume` → `VesselNetwork` → `BoundaryNodes` →
# `HaemodynamicModel` → `Solution`.
# 
# **Plots:** each stage writes its own PNGs to `settings["plot_dir"]`; the cells
# below show them. Set `SHOW_STAGE_PLOTS = False` to skip the inline display.
# 
# **Source of truth:** edit **this notebook only**. [`pipeline_tutorial.py`](pipeline_tutorial.py)
# is auto-generated (do not edit by hand). Regenerate it after notebook changes:
# 
# ```bash
# pytest tests/integration/test_pipeline_tutorial.py
# ```

# ## Prerequisites
# 
# ```bash
# pip install HaemoLynx
# ```
# 
# That is everything Stages 1–6 need: the next cell installs it for you if it is
# missing, and the tutorial runs on a synthetic vessel volume it builds itself,
# so no data download and no repository checkout are required.
# 
# Two optional extras:
# 
# - **Working from a clone?** `pip install -e ".[dev]"` from the repository root
#   instead. The tutorial then picks up the real cropped nerve mask in
#   `tests/data/` and the pipeline's own `examples/resistance_pipeline_config.yaml`
#   rather than the defaults, and you get the per-step graph plots from
#   `tutorials/tutorial_plots.py`.
# - **[ilastik](https://www.ilastik.org/download/)** (a separate program) for
#   Stage 0, only if you want to segment your own raw image.

# In[ ]:


# `from __future__` has to be the first statement in the exported .py, so
# it leads the first code cell rather than the setup cell below.
from __future__ import annotations

# Install HaemoLynx if this kernel does not already have it. Nothing happens
# when it is present, so a clone or an editable install is left alone.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("haemolynx") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "HaemoLynx"], check=True)

import haemolynx

print(f"HaemoLynx {haemolynx.__version__} from {haemolynx.__file__}")


# In[ ]:


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


def _resolve_repo_root(tutorial_dir: Path) -> Path | None:
    """The checkout this notebook sits in, or None when pip-installed."""
    env_root = os.environ.get("HAEMOLYNX_REPO_ROOT")
    if env_root:
        return Path(env_root)
    candidates = [tutorial_dir, tutorial_dir.parent, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for start in candidates:
        if (start / "src" / "haemolynx").is_dir() and (start / "examples").is_dir():
            return start
    return None


REPO_ROOT = _resolve_repo_root(TUTORIAL_DIR)
if REPO_ROOT is not None:
    # In a checkout, prefer the working tree over anything already installed.
    for p in (REPO_ROOT / "src", REPO_ROOT / "examples"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from haemolynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from haemolynx.parsers import configure_console_logging

# The eight stages the example runs, in the order it runs them.
from haemolynx.pipeline import (
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    default_schema,
    export_results,
    preflight,
    resolve_settings,
    segment,
    skeletonise,
    solve,
    write_default_config,
)

SCHEMA = default_schema()

# The stages report their progress through `logging`; send it to the notebook.
configure_console_logging()

# The plot helpers live beside the notebook in the repository. Without them the
# pipeline still runs; only the inline display of the saved figures is skipped.
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))
try:
    from tutorial_plots import in_jupyter, show_stage_plots
except ModuleNotFoundError:
    print("tutorial_plots.py not found (installed, no checkout): figures are saved to disk, not shown inline.")

    def in_jupyter() -> bool:
        return False

    def show_stage_plots(stage_title, paths, *, enabled=True, **kwargs) -> None:
        if enabled:
            existing = [str(p) for p in paths if Path(p).exists()]
            print(f"{stage_title}: {', '.join(existing) if existing else 'no figures'}")

print(f"Repository root: {REPO_ROOT if REPO_ROOT else 'none - running from an installed HaemoLynx'}")
print(f"Running in Jupyter: {in_jupyter()}")


# ## Configuration: one settings dict
# 
# Every stage below reads this dict. It comes from a YAML config file validated
# against `SCHEMA`, which is also what the CLI flags and a future GUI are built
# from — so a setting has one name, one default, and one description wherever it
# appears.
# 
# `TUTORIAL_OVERRIDES` is the only place this notebook differs from the shipped
# configuration. To change what a stage does, change a value here (or edit the
# YAML) — you never edit a call.
# 
# Run `python examples/resistance_network_pipeline.py --list-settings` to see all
# 140 of them, or `write_default_config("my_config.yaml")` to get a documented
# file to edit.

# In[ ]:


OUTPUT_DIR = Path(os.environ.get("HAEMOLYNX_TUTORIAL_OUTPUT_DIR", str(TUTORIAL_DIR / "outputs"))).resolve()
PLOT_DIR = Path(os.environ.get("HAEMOLYNX_TUTORIAL_PLOT_DIR", str(TUTORIAL_DIR / "plots"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# In a checkout, the pipeline example's own config; otherwise the schema writes
# a default one, which is exactly what `write_default_config` gives any
# installed user.
CONFIG_PATH = REPO_ROOT / "examples" / "resistance_pipeline_config.yaml" if REPO_ROOT else None
if CONFIG_PATH is None or not CONFIG_PATH.exists():
    CONFIG_PATH = write_default_config(OUTPUT_DIR / "tutorial_config.yaml")

TUTORIAL_OVERRIDES = {
    # Where everything is written.
    "vtk_output_prefix": OUTPUT_DIR / "tutorial",
    "plot_dir": PLOT_DIR,
    # Meaning of each input array axis. Volumes load as canonical (z, y, x), so
    # this selects which axis is z -- the one projections look through.
    "image_axis_order": "zyx",
    # Skeletonisation: small values suit the small volume this tutorial uses.
    "skeleton_closing_radius": 1,
    "skeleton_bridge_gap_size": 1,
    "skeleton_min_branch_length": 3,
    "skeleton_max_bridge_distance": 2,
    "skeleton_component_connectivity": 3,
    "skeleton_min_component_percent": 1.0,
    # Graph topology repair.
    "graph_reconnect_threshold": 10.0,
    "final_orphan_reconnect_threshold": 3.0,
    "cluster_collapse_distance": 5.0,
    "min_stub_length": 3.0,
    # Boundary conditions: inlet and outlet pressures, in Pa.
    "inlet_p_bc": 1000.0,
    "outlet_p_bc": 500.0,
    # Inlet and outlet are picked by the box they fall in; the boxes are set in
    # Stage 1, once the volume's shape is known.
    "inlet_node_selection_method": "volume",
    "outlet_node_selection_method": "volume",
    "statistics": True,
    "statistics_mode": "fast",
    # Figures are saved and shown by the cells here, so the stages must not try
    # to open blocking windows of their own.
    "show_plots_in_ide": False,
    "hold_ide_plots_open": False,
    "interactive_plots": False,
    "final_render_mode": "2d",
}

# Display each stage's saved PNGs inline (notebook only).
SHOW_STAGE_PLOTS = True

print(f"Settings from: {CONFIG_PATH}")
print(f"Outputs: {OUTPUT_DIR}")
print(f"Plots: {PLOT_DIR}")
print(f"Inline stage plots enabled: {SHOW_STAGE_PLOTS and in_jupyter()}")


# ## Stage 0: Segment **your** raw image with ilastik
# 
# HaemoLynx Stages 1–6 need a **binary vessel mask** (foreground = vessel). This stage turns **your unsegmented** 3D TIFF into that mask using [ilastik](https://www.ilastik.org/) Pixel Classification.
# 
# > **Training happens in the ilastik GUI** (interactive). **Inference** runs headlessly inside `segment()` once you have a saved `.ilp` project — set `use_ilastik_segmentation` and the three paths below, and the stage does the rest.
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
# In the **Prediction** / export section of the Pixel Classification workflow, ensure **Simple Segmentation** is available as an export source (this is what HaemoLynx requests headlessly). The saved `.ilp` embeds these settings.
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


# --- Stage 0: point these at your data, then set RUN_STAGE_0_ILASTIK = True ---
RUN_STAGE_0_ILASTIK = False  # True after you have a trained .ilp classifier

if RUN_STAGE_0_ILASTIK:
    TUTORIAL_OVERRIDES.update(
        {
            "use_ilastik_segmentation": True,
            "ilastik_unsegmented_image_path": Path("/path/to/your/raw_microscopy.tif"),
            "ilastik_classifier_path": Path("/path/to/your_pixel_classifier.ilp"),
            "ilastik_executable": os.environ.get("ILASTIK_EXECUTABLE", "ilastik"),
            "ilastik_output_dir": OUTPUT_DIR / "segmentation",
        }
    )
    print("Stage 0: segment() will run ilastik and analyse the mask it writes.")
else:
    print("Stage 0 skipped (RUN_STAGE_0_ILASTIK=False): Stage 1 analyses an existing mask.")


# ## Stage 1: `segment()` and `skeletonise()`
# 
# `segment()` settles **which** image is analysed — running ilastik when Stage 0
# asked for it, otherwise passing the mask straight through — and returns a
# `SegmentedInputs`. `skeletonise()` then loads that volume, resolves its voxel
# size from the file metadata, and reduces the vessels to a one-voxel-wide
# skeleton, returning a `SkeletonisedVolume`.
# 
# Stages 1–6 operate on a **binary segmentation**, not raw fluorescence. The mask
# should be foreground voxels (e.g. 255) on background (0); HaemoLynx binarises
# it on load.
# 
# - **In a checkout:** `tests/data/Nerve_capillaries_cropped.tif`, a small cropped mask.
# - **Installed, with no data to hand:** a synthetic branching volume built below.
# - **Your own:** set `input_path` in the overrides, or run Stage 0.

# In[ ]:


# --- Which mask to analyse -------------------------------------------------
# In a checkout, the small cropped nerve mask committed for the tests. Without
# one, a synthetic volume built here -- so this notebook runs on a bare
# `pip install HaemoLynx`, with no data to download.
DEFAULT_SEGMENTED_TIFF = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif" if REPO_ROOT else None


def build_synthetic_vessel_volume(path: Path) -> Path:
    """A branching phantom: one trunk, two branches, four free ends.

    Vessels are drawn as thick lines through a 64 x 64 x 64 volume, which is
    enough for every stage below to do something real -- skeletonise, build a
    graph with junctions and free ends, order the branches, and solve a flow.
    """
    import tifffile

    volume = np.zeros((64, 64, 64), dtype=np.uint8)

    def draw(start, end, radius=2):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = int(np.linalg.norm(end - start)) * 4 + 1
        for t in np.linspace(0.0, 1.0, steps):
            z, y, x = np.round(start + t * (end - start)).astype(int)
            z0, z1 = max(0, z - radius), min(volume.shape[0], z + radius + 1)
            y0, y1 = max(0, y - radius), min(volume.shape[1], y + radius + 1)
            x0, x1 = max(0, x - radius), min(volume.shape[2], x + radius + 1)
            volume[z0:z1, y0:y1, x0:x1] = 255

    # Points are (z, y, x). The tree runs along y because the boxes below take
    # the inlet from the first 40% of y and the outlet from the remaining 60%:
    # the trunk starts inside the inlet band and every free end lands in the
    # outlet band.
    draw((32, 5, 32), (32, 30, 32))          # trunk
    draw((32, 30, 32), (32, 50, 14))         # two branches...
    draw((32, 30, 32), (32, 50, 50))
    draw((32, 50, 14), (18, 60, 8))          # ...each splitting once more
    draw((32, 50, 14), (46, 60, 8))
    draw((32, 50, 50), (18, 60, 56))
    draw((32, 50, 50), (46, 60, 56))

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, volume)
    return path


env_input = os.environ.get("HAEMOLYNX_TUTORIAL_INPUT_TIFF")
if RUN_STAGE_0_ILASTIK:
    INPUT_TIFF = None  # segment() fills input_path in from the ilastik run
elif env_input:
    INPUT_TIFF = Path(env_input).expanduser().resolve()
elif DEFAULT_SEGMENTED_TIFF is not None and DEFAULT_SEGMENTED_TIFF.exists():
    INPUT_TIFF = DEFAULT_SEGMENTED_TIFF.resolve()
else:
    INPUT_TIFF = build_synthetic_vessel_volume(OUTPUT_DIR / "synthetic_vessels.tif")
    print("No segmented image to hand, so this run uses the synthetic volume above.")

if INPUT_TIFF is not None:
    TUTORIAL_OVERRIDES["input_path"] = INPUT_TIFF

# Every setting for this run, validated against the schema. `resolve_settings`
# also fills in the tables derived from other settings, such as the diameter
# for each branch order.
settings = resolve_settings(schema=SCHEMA, config_path=CONFIG_PATH, overrides=TUTORIAL_OVERRIDES)

# The pre-run checks the examples run for you: every path a stage will need,
# checked before any work starts.
report = preflight(settings, SCHEMA)
assert report.ok, "settings are not runnable; see the checklist above"


# In[ ]:


inputs = segment(settings)
print(f"Stage 0/1 input: {inputs.image_path} ({inputs.input_format})")

volume = skeletonise(settings, inputs)
print(f"Image shape: {volume.image.shape[:3]}")
print(f"Voxel size (x, y, z): {volume.voxel_size_xyz}")
print(f"Array-axis spacing (z, y, x): {volume.voxel_size_zyx}")
print(f"Skeleton voxels: {int(volume.skeleton.sum())}")

# The inlet and outlet boxes are in physical (z, y, x) MICRONS, not voxel
# indices, so they are built from the volume's own extent. Inlet: the first 40%
# along y. Outlet: the remaining 60%. The cropped nerve fixture's connected
# component sits in the low-y part of the image, so a last-20% outlet band
# misses it and the flow solve has no conductive path between the two roles.
extent_zyx = [
    (dimension - 1) * spacing
    for dimension, spacing in zip(volume.image.shape[:3], volume.voxel_size_zyx)
]
y_split = 0.4 * extent_zyx[1]
settings["inlet_node_volumes"] = [((0.0, 0.0, 0.0), (extent_zyx[0], y_split, extent_zyx[2]))]
settings["outlet_node_volumes"] = [
    ((0.0, y_split, 0.0), (extent_zyx[0], extent_zyx[1], extent_zyx[2]))
]
print(f"Inlet box (um):  {settings['inlet_node_volumes'][0]}")
print(f"Outlet box (um): {settings['outlet_node_volumes'][0]}")

# Both are written by skeletonise(): the skeleton as loaded, and after the
# cleaning pass that bridges gaps and drops the smallest components.
show_stage_plots(
    "Stage 1: Skeleton",
    [PLOT_DIR / "raw_skeleton.png", PLOT_DIR / "skeleton_projection.png"],
    enabled=SHOW_STAGE_PLOTS,
)


# ## Stage 2: `build_network()`
# 
# Turns the skeleton into an `nx.MultiGraph` and repairs its topology in eleven
# steps — stitching loops, reconnecting broken segments, collapsing clusters of
# nearby junctions, pruning stubs, and merging away degree-2 nodes so one vessel
# is one edge rather than a chain of them. It also loads any large/small vessel
# masks the settings name, and returns a `VesselNetwork`.
# 
# The stage saves a plot after each step to `settings["plot_dir"]`, so you can see
# exactly what each one changed.

# In[ ]:


network = build_network(settings, volume, SCHEMA)
G = network.graph
print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Free ends (degree 1): {sum(1 for _n, d in G.degree() if d == 1)}")
print(f"Junctions (degree >= 3): {sum(1 for _n, d in G.degree() if d >= 3)}")

# One overlay per topology step, in the order the steps ran.
step_plots = sorted(PLOT_DIR.glob("graph_after_*.png"))
print(f"Step overlays saved: {len(step_plots)}")
show_stage_plots("Stage 2: Graph topology steps", step_plots, enabled=SHOW_STAGE_PLOTS)


# ## Stage 3: `assign_boundaries()`
# 
# Chooses where flow enters and leaves, from the boxes set in Stage 1, and
# returns a `BoundaryNodes`. Only degree-1 terminals are eligible: pinning a
# pressure on an interior junction would make it inject or remove flow mid-network.
# 
# `inlet_node_selection_method` decides how they are picked, and each method
# reads a different setting:
# 
# | Method | Reads |
# |---|---|
# | `"volume"` | `inlet_node_volumes` — corner pairs in (z, y, x) microns |
# | `"coordinates"` | `inlet_node_coordinates` — each point snaps to the nearest terminal |
# | `"edge_percent"` | `boundary_first_percent` / `boundary_last_percent` / `boundary_axis` — the first and last bands of the network along one axis |
# | `"all_degree_1"` | every terminal in the graph |
# | `"degree_1_from_inlet"` | `boundary_distance_from_inlet_node` — every terminal further than that from a inlet node |
# 
# `"edge_percent"` is the default, and the only one that asks nothing of the
# dataset: it needs no coordinate, box or mask, so it has something to say about
# an image nobody has looked at yet. This notebook overrides it with `"volume"`
# because it knows where its own vessels are.
# 
# With segmented arteriole/venule masks, set `automated_vessel_assignment` instead
# and the terminals come from anatomy rather than geometry.

# In[ ]:


boundaries = assign_boundaries(settings, network)
print(f"Inlet nodes:  {boundaries.inlet_nodes}")
print(f"Outlet nodes: {boundaries.outlet_nodes}")
print(f"Node pair for the equivalent resistance: {boundaries.resistance_node_pair}")
assert boundaries.inlet_nodes and boundaries.outlet_nodes, (
    "no boundary nodes: widen the boxes above, or pick another selection method"
)


# ## Stage 4: `assign_diameters()` and `build_haemodynamic_model()`
# 
# `assign_diameters()` walks out from the inlets to give every edge a
# **branch order**, then uses that order as the key into the diameter table
# (`diameter_by_branch_order`, derived from the settings). With
# `use_fwhm_edge_diameters` on it measures each vessel from the raw image instead
# and the table is only a fallback.
# 
# `build_haemodynamic_model()` turns diameter and length into a **resistance** for
# each edge, using the Poiseuille law with a diameter-dependent apparent
# viscosity, and writes `resistance` (Pa·s/m³) and `conductance` (m³/(Pa·s))
# together. It returns a `HaemodynamicModel`.
# 
# > Vessels between 7 µm and 100 µm raise a `PlaceholderViscosityWarning`: the
# > viscosity there is held constant rather than transitioning smoothly, so treat
# > those resistances as order-of-magnitude (issue #90).

# In[ ]:


diameters = assign_diameters(settings, network, boundaries, SCHEMA)
model = build_haemodynamic_model(settings, diameters)
G = model.graph

# Edges the walk could not reach from an inlet keep no branch order, and are
# left without a resistance rather than being given a made-up one.
orders = sorted({data["branch_order"] for _u, _v, data in G.edges(data=True) if "branch_order" in data})
print(f"Branch orders assigned: {orders[:10]}{' ...' if len(orders) > 10 else ''}")
resistances = [data["resistance"] for _u, _v, data in G.edges(data=True) if "resistance" in data]
print(f"Edges with a resistance: {len(resistances)} of {G.number_of_edges()}")
if resistances:
    print(f"Resistance range (Pa.s/m^3): {min(resistances):.3e} to {max(resistances):.3e}")

# The branch-order figure is drawn by Stage 6, once flows are on the graph too.


# ## Stage 5: `solve()`
# 
# Builds the conductance matrix from the edge conductances, pins `inlet_p_bc` at
# the inlets and `outlet_p_bc` at the outlets, and solves for the pressure at
# every node — then writes the resulting flow onto each edge. With
# `do_equiv_resistance_calculation` on it also computes the two-point resistance
# between the boundary pair, which is the whole network reduced to a single
# number. It returns a `Solution`.

# In[ ]:


solution = solve(settings, model, boundaries)

pressures = solution.pressure
print(f"Solved for {len(solution.node_list)} nodes")
print(f"Pressure range (Pa): {float(np.min(pressures)):.1f} to {float(np.max(pressures)):.1f}")
print(f"Equivalent resistance (Pa.s/m^3): {solution.equivalent_resistance:.6e}")

flows = [abs(data["flow_signed"]) for _u, _v, data in model.graph.edges(data=True) if "flow_signed" in data]
print(f"Edge flows (m^3/s): {min(flows):.3e} to {max(flows):.3e}")


# ## Stage 6: `export_results()`
# 
# The last stage writes everything out: the VTK files (vessels, nodes and
# pericytes, carrying resistance, pressure and flow as cell arrays), the network
# statistics CSV, and the final plots.

# In[ ]:


export_results(settings, network, model, solution)

written = sorted(OUTPUT_DIR.glob("*.vtp")) + sorted(OUTPUT_DIR.glob("*.csv"))
for path in written:
    print(f"  {path.name}  ({path.stat().st_size:,} bytes)")

stats_csv = next((p for p in written if p.suffix == ".csv"), None)
if stats_csv is not None:
    print()
    print(stats_csv.read_text(encoding="utf-8").splitlines()[0])

show_stage_plots(
    "Stage 6: Final network",
    [
        PLOT_DIR / "geometry_with_branch_orders.png",
        PLOT_DIR / "edges_and_nodes_overlay.png",
        PLOT_DIR / "node_degree_distribution.png",
        PLOT_DIR / "final_graph.png",
    ],
    enabled=SHOW_STAGE_PLOTS,
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

# ## Running all six at once
# 
# The example is exactly the cells above, one after the other:
# 
# ```python
# inputs     = segment(settings)
# volume     = skeletonise(settings, inputs)
# network    = build_network(settings, volume, SCHEMA)
# boundaries = assign_boundaries(settings, network)
# diameters  = assign_diameters(settings, network, boundaries, SCHEMA)
# model      = build_haemodynamic_model(settings, diameters)
# solution   = solve(settings, model, boundaries)
# export_results(settings, network, model, solution)
# ```
# 
# `haemolynx.pipeline.run_pipeline_stages(settings, SCHEMA)` runs that sequence
# for you, and `python examples/resistance_network_pipeline.py` runs it from the
# command line against `examples/resistance_pipeline_config.yaml`.
# 
# ## Adapting for your own data
# 
# 1. **Edit this notebook** (`pipeline_tutorial.ipynb`) — not the generated `.py`.
# 2. **Your own mask:** put its path in `TUTORIAL_OVERRIDES["input_path"]`.
# 3. **Your own raw image:** set `RUN_STAGE_0_ILASTIK = True` in Stage 0 and fill in
#    the three ilastik paths; `segment()` runs the classifier and analyses what it writes.
# 4. **Boundaries:** change `inlet_node_selection_method` and the setting it
#    reads (see the table in Stage 3), or set `automated_vessel_assignment` to take
#    them from arteriole/venule masks.
# 5. **Tuning:** every skeleton and graph threshold is a key in `TUTORIAL_OVERRIDES`.
#    `write_default_config("my_config.yaml")` writes all 140 settings with their
#    documentation, ranges and defaults; edit that and pass it as `CONFIG_PATH`.
# 6. **Regenerate** `pipeline_tutorial.py`: `pytest tests/integration/test_pipeline_tutorial.py`.
