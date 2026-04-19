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
* **Inputs:** 
  * **Raw or Segmented Volumes:** Out of the box, it accepts `.tif`, `.tiff`, `.h5`, or `.hdf5` files containing 3D image volumes or 4D/5D multi-channel data (e.g., `vesselness_map_probs.tiff`). Voxel sizes are automatically extracted from the file metadata when available.
  * **Auxiliary Masks (Optional):** It can also accept large-vessel or cell-boundary masks (`.tif` or `.h5`) to automate inlet/outlet assignments and distance measurements.
* **Outputs:** 
  * **3D Models:** The solved physical network is exported as interactive `.vtp` (PolyData) files in `examples/outputs/resistance_network/`. Open these in [ParaView](https://www.paraview.org/) to visualize pressure gradients and flow velocities.
  * **Plots & Diagnostics:** 2D skeleton projections, degree distributions, and interactive 3D HTML diagnostic plots are saved to `examples/plots/carotid/`.
  * **Statistical Reports:** Comprehensive vessel statistics, distance metrics, and branch-order summaries are exported as `.csv` files.
  * **Graph Data:** The fully optimized mathematical network is cached as a Python pickle (`_graph.pkl`) and the skeleton as a `.npy` file next to your input image, saving you from repeating the heavy processing steps on future runs.

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

If you want to run headless Ilastik pixel classification before the pipeline starts, you can configure the paths at the top of `carotid_image_to_model.py`:

1. Set `RUN_ILASTIK = True`.
2. Point `ILASTIK_BINARY_PATH` to your local Ilastik installation (e.g., `run_ilastik.sh` or `ilastik.exe`).
3. Point `ILASTIK_PROJECT_PATH` to the trained model included in this repository (e.g., `examples/images/cb_wky_2x2x2_A.ilp`).
4. **Multi-Channel Inputs:** This specific Ilastik model requires two input features. You must provide *both*:
   * `RAW_IMAGE_PATH`: Point this to your raw intensity volume (e.g., `..._vessels.tif`).
   * `FRANGI_IMAGE_PATH`: Point this to your pre-computed vesselness map (e.g., `..._vesselness_map.tif`).

The script will automatically feed both images into Ilastik, generate the probability map, and seamlessly pass the result directly into the main pipeline.
