# ImageLynx Carotid Pipeline

This folder contains `carotid_image_to_model.py`. It's an end-to-end pipeline that takes 3D image volumes (like micro-CT or light-sheet microscopy scans of blood vessels), turns them into mathematical graphs, and runs haemodynamic flow simulations (calculating blood pressure and resistance) across the network.

## How it works

The pipeline runs in four main steps:

1. **Image Preprocessing:** Takes raw 3D/4D probability maps (e.g., from Ilastik), runs median filters to drop noise, and thresholds the data to get a clean binary mask of the vessels.
2. **Skeletonization:** Thins out the 3D binary tubes into 1D centerlines. It also prunes small artifacts and collapses messy "spiderweb" loops.
3. **Graph Extraction & Optimization:** Builds a mathematical network (nodes and edges) from the skeleton. It merges nodes that are too close, cleans up branching points, and removes unnecessary degree-2 nodes.
4. **Haemodynamic Simulation:** Finds the inlets and outlets, calculates physical resistance for each segment using Poiseuille's law, and solves the linear equations to get pressure and flow everywhere.

---

## Quick Start

### 1. Running the pipeline
Just run the script directly from your terminal:
```bash
python examples/carotid_image_to_model.py
```

### 2. Inputs and Outputs
* **Input:** Out of the box, it looks for a pre-classified `.tif` or `.h5` file (e.g., `C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff`).
* **Outputs:** 
  * **3D Models:** Exported as `.vtp` files in `examples/outputs/resistance_network/`. You can view these in [ParaView](https://www.paraview.org/).
  * **Plots:** 2D network projections and degree distributions end up in `examples/plots/carotid/`.
  * **Graph Data:** The cleaned-up network is saved as a Python pickle (`_graph.pkl`) right next to your input image, so you don't have to rebuild it from scratch next time.

---

## Configuration

Everything is driven by Dataclass configs at the bottom of the script. Tweak these to fit your dataset:

* **`PreprocessingConfig`:** How we go from raw image to binary mask.
  * `enable_hysteresis_threshold`: Uses high/low thresholds to grab faint capillaries connected to confident main vessels.
  * `median_filter_size`: 3D filter size to kill salt-and-pepper noise.
  * `enable_shannon_entropy`: (For Ilastik 4D outputs) Drops voxels where the ML model wasn't confident.
* **`SkeletonConfig`:** How we extract the structure.
  * `downsample_factor`: Crank this up (e.g., `2.0`) to massively speed up skeletonization on huge volumes.
  * `closing_radius`: Smooths out bumpy vessel walls so the skeleton doesn't get "hairy".
* **`GraphConfig`:** Mathematical cleanup.
  * `keep_largest_component_only`: Drops disconnected floating islands of vessels.
* **`HaemodynamicsConfig`:** Physics parameters.
  * `input_p_bc` / `output_p_bc`: Inlet and outlet pressures in Pascals.
* **`PipelineConfig`:** Toggle whole phases. 
  * If you just want to re-run the flow solver with new pressures, set `do_skeletonize=False` and `do_graph_building=False`. It'll just load your `.pkl` and solve.

---

## 3D Visualization & Debugging

Visualizing 3D networks can be tough, so we've wired up [Vedo](https://vedo.embl.es/) for interactive checkpoints. Enable these in `VisualizationConfig`:

* **`visualize_mask_only = True`**: Pops up the cleaned 3D binary mask before skeletonization starts. Great for dialing in your thresholds.
* **`visualize_overlay_preview = True`**: Gives you a 4-panel view showing the raw image, binary mask, raw skeleton, and final optimized graph stacked together. 
  * *Note: Closing this preview window will kill the script (`sys.exit`) so you can tweak things without waiting for the haemodynamics solver to finish.*

---

## Using Ilastik (Optional)

If you want to run headless Ilastik pixel classification before the pipeline starts:

1. Set `RUN_ILASTIK = True` at the top of the script.
2. Point `ILASTIK_BINARY_PATH` to your `run_ilastik.sh`.
3. Point `ILASTIK_PROJECT_PATH` to your trained `.ilp` model.
4. Make sure `RAW_IMAGE_PATH` points to your raw intensity volume.

The script will handle the Ilastik run, grab the probability map, and feed it straight into the pipeline.
