# ImageLynx Carotid Pipeline Example

This directory contains the `carotid_image_to_model.py` script, an end-to-end pipeline designed to convert 3D image volumes (such as micro-CT or light-sheet microscopy of vascular networks) into mathematical graphs, and simulate hemodynamic flow (blood pressure and resistance) through the network.

## Pipeline Overview

The script processes data through four main phases:

1. **Image Preprocessing:** Loads 3D or 4D probability maps (e.g., from Ilastik), applies noise reduction (median filters), and thresholds the data into a clean, solid binary mask representing the vessels.
2. **Skeletonization:** Converts the thick 3D binary tubes into a 1D centerline skeleton, pruning tiny artifacts and collapsing dense "spiderweb" bundles.
3. **Graph Extraction & Topological Optimization:** Extracts a mathematical network (Nodes and Edges) from the skeleton. It rigorously cleans the topology by merging adjacent nodes, resolving "triangle" intersections into clean bifurcations, and removing redundant degree-2 points.
4. **Hemodynamic Simulation:** Automatically identifies boundary inlets and outlets, calculates physical resistance weights using Poiseuille's law, and solves the system of linear equations to determine pressure at every node and flow in every edge.

---

## Quick Start

### 1. Execute the Pipeline
You can run the script directly from your terminal:
```bash
python examples/carotid_image_to_model.py
```

### 2. Expected Inputs and Outputs
* **Input:** By default, the script looks for a pre-classified `.tif` or `.h5` file (e.g., `C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff`).
* **Output:** 
  * **3D Models:** The final solved network is exported as `.vtp` files (PolyData) in the `examples/outputs/resistance_network/` directory. These can be opened and analyzed in [ParaView](https://www.paraview.org/).
  * **2D Plots:** Degree distributions and 2D projections of the network are saved as `.png` files in `examples/plots/carotid/`.
  * **Graph Data:** The optimized mathematical network is saved as a serialized Python object (`_graph.pkl`) next to your input image to save time on future runs.

---

## Configuring the Pipeline (Dataclasses)

The pipeline's behavior is controlled by several structured `Config` dataclasses instantiated at the very bottom of the script. You can modify these to tune the pipeline for your specific dataset:

* **`PreprocessingConfig`:** Controls how the raw image is turned into a binary mask.
  * `enable_hysteresis_threshold`: Uses high/low thresholds to confidently identify vessels while preserving faint connections.
  * `median_filter_size`: Size of the 3D median filter used to remove salt-and-pepper noise.
  * `enable_shannon_entropy`: If your input is a 4D probability map from Ilastik, this uses Shannon Entropy to automatically reject voxels where the ML model was uncertain.
* **`SkeletonConfig`:** Controls the structural extraction of the network.
  * `downsample_factor`: Increase this (e.g., `2.0`) to drastically speed up skeletonization on massive volumes by shrinking the image first.
  * `closing_radius`: Smooths the bumpy outer walls of the binary mask to prevent "hairy" skeletons.
* **`GraphConfig`:** Controls mathematical edge pruning.
  * `keep_largest_component_only`: Ensures only the single main interconnected network is kept, deleting any floating "island" fragments.
* **`HemodynamicsConfig`:** Sets the physical boundary conditions for the flow simulation.
  * `input_p_bc` / `output_p_bc`: The pressure (in Pascals) applied to the automatically detected inlet and outlet nodes.
* **`PipelineConfig`:** Allows you to toggle entire phases on or off. 
  * For example, if you already built the graph and just want to tweak the pressures, set `do_skeletonize=False` and `do_graph_building=False`. The script will instantly load the saved `.pkl` graph and run the flow solver.

---

## Interactive Visualization & Debugging Checkpoints

Because 3D networks are complex, the script includes interactive 3D [Vedo](https://vedo.embl.es/) visualizations to help you debug the data at critical checkpoints. You can enable these inside the `VisualizationConfig`:

* **`visualize_mask_only = True`**: Pops up a 3D window showing the cleaned binary mask *before* the heavy skeletonization step begins. Useful for tuning your thresholds.
* **`visualize_overlay_preview = True`**: A powerful 4-panel checkpoint window showing the raw image, the binary mask, the raw skeleton, and the mathematically optimized graph overlaid on top of each other. 
  * *Note: Closing this preview window will intentionally halt the script (`sys.exit`) so you can inspect intermediate results without waiting for the hemodynamics solver.*

---

## Advanced: Running with Ilastik (Optional)

The script includes an optional wrapper to run headless pixel classification using Ilastik *before* the main pipeline starts. 

To enable this:
1. Set `RUN_ILASTIK = True` at the top of the script.
2. Update `ILASTIK_BINARY_PATH` to point to your local `run_ilastik.sh` executable.
3. Update `ILASTIK_PROJECT_PATH` to point to your trained `.ilp` model file. 
4. Ensure `RAW_IMAGE_PATH` points to your raw intensity volume.

The script will automatically pass the raw image to Ilastik, generate the probability map, and feed it directly into the `carotid_image_to_model` pipeline.
