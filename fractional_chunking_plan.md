# Implementation Plan: Fractional Sub-Volume Chunking

This document outlines the iterative plan to divide the raw vessel probability field into localized sub-volume chunks based on a user-specified fractional resolution.

## Phase 0: Decoupling Image Loading from Preprocessing

**Objective:** Ensure that the fractional chunking algorithm and grid visualization operate strictly on the raw probability field *before* any noise removal, smoothing, or binarization occurs.

1.  **Refactor Image Loader:** Split the existing monolithic `_load_and_preprocess_image` function in `carotid_image_to_model.py` into two distinct functions:
    *   `_load_raw_probability_field`: Handles disk I/O (TIFF/H5), lazy loading with Dask, ROI cropping, and 4D channel/Shannon entropy extraction. Returns the raw, un-thresholded probability float array.
    *   `_preprocess_local_mask`: Handles median filtering, morphological operations, hysteresis thresholding, and hole-filling.
2.  **Execution Reordering:** The main script will call `_load_raw_probability_field` first. Immediately following this, the Phase 1 and 1.5 logic (fractional chunking and VTK export) will execute, allowing the script to prematurely exit before any preprocessing computation is wasted.

## Phase 1: Fractional Math and Grid Discretization

**Objective:** Mathematically divide the global probability field into a 3D grid of chunks based on a fractional input (e.g., 0.2), ensuring support for non-cubic array dimensions and anisotropic voxel resolutions.

### 1. Configuration & Input
*   **PipelineConfig:** Introduce a `chunk_fraction: float` parameter (e.g., `0.2` implies splitting each axis into approximately 5 segments, yielding $5 \times 5 \times 5 = 125$ total chunks).
*   **CLI Integration:** Expose this parameter via `argparse` (e.g., `--chunk-fraction 0.2`) in the main script (`carotid_image_to_model.py`).

### 2. Evenly Distributed, Near-Cubic Grid Calculation
To satisfy the constraint that the collective grid must **strictly equal the loaded field dimensions** (zero padding allowed) while simultaneously ensuring all chunks are **as close to cubic as possible** and **equal in size** (no tiny "remainder" chunks at the edges), we must dynamically calculate the number of subdivisions per axis.

1.  **Define a Target Isotropic Size:** We establish a target baseline size ($S$) derived from the longest axis to preserve biological context.
    *   $S = \max(Z, Y, X) \cdot \text{chunk\_fraction}$
2.  **Calculate Subdivisions per Axis:** We determine how many chunks each axis should be divided into to get as close to $S$ as possible.
    *   $N_z = \max(1, \text{round}(Z / S))$
    *   $N_y = \max(1, \text{round}(Y / S))$
    *   $N_x = \max(1, \text{round}(X / S))$
3.  **Distribute the Grid:** The exact floating-point step size for each axis is calculated as $step_z = Z / N_z$. When generating the bounding boxes, the start and end points for the $i$-th chunk are computed as $\text{round}(i \cdot step_z)$ to $\text{round}((i+1) \cdot step_z)$.

**Why this satisfies all constraints:**
*   **No Padding:** The final chunk's end point is mathematically guaranteed to perfectly hit $Z_{max}, Y_{max}, X_{max}$.
*   **Near-Cubic:** Because $N_z, N_y, N_x$ are mathematically forced to scale proportionally to their respective axis lengths, the resulting chunks are inherently as physically cubic as the discrete voxel grid allows. (e.g. dividing a $100 \times 120 \times 300$ volume using $S=60$ results in $N=(2, 2, 5)$, yielding chunks of $50 \times 60 \times 60$, which is highly isotropic).
*   **Uniform Size (No Slivers):** Because the remainder is evenly distributed across the floating-point step sizes, there are no tiny edge chunks. Every chunk on a given axis will be identical in size (within a maximum variance of exactly 1 voxel due to rounding).

### 3. Bounding Box Generation with Overlap Margins
Simply splitting the array creates hard boundaries. If a vessel crosses a boundary, it will be severed, confusing downstream skeletonization algorithms.
*   We will implement a generator function that yields a dictionary for each chunk containing:
    *   `core_bbox`: The exact non-overlapping mathematical bounds (e.g., $Z: 0 \to 20$).
    *   `padded_bbox`: The physical bounds extracted for processing, which includes a safety `margin` (e.g., $+15$ voxels on all sides). 
    *   `offset`: The global $(Z, Y, X)$ offset of the padded box, crucial for later translating local coordinates back into global space.
*   **Boundary Truncation:** The generator will enforce `min()` and `max()` clamping so that padding at the absolute edges of the global volume (e.g., $Z=0$ or $Z_{max}$) does not trigger `IndexError` exceptions.

### 4. Phase 1: Unit Testing Suite
To ensure the fractional math, uniform size distribution, and bounding box generation are perfectly robust, we will implement a dedicated unit testing suite (`tests/test_fractional_grid.py`).

*   **`test_strict_volume_coverage`:**
    *   **Logic:** Mock a volume of $(117, 234, 489)$ and set `chunk_fraction = 0.2`. Generate the grid.
    *   **Assertion:** Sum the volumes of all generated `core_bbox` chunks. Assert that this sum exactly equals $117 \times 234 \times 489$, proving no zero-padding was added and no voxels were dropped.
*   **`test_uniform_chunk_sizing_variance`:**
    *   **Logic:** Mock an indivisible volume (e.g., $(100, 100, 100)$) with `chunk_fraction = 0.3` (which aims for ~3 chunks per axis).
    *   **Assertion:** Extract the sizes of all chunks along the Z-axis. Assert that the maximum size minus the minimum size is $\le 1$ voxel. This proves the "remainder" was evenly distributed and no "sliver" chunks were created at the boundary.
*   **`test_extreme_anisotropy_near_cubic_resolution`:**
    *   **Logic:** Mock a "needle" volume of $(10, 15, 1000)$ with `chunk_fraction = 0.1`.
    *   **Assertion:** Verify the computed subdivisions ($N_z, N_y, N_x$) resolve to $(1, 1, 10)$. Assert that the resulting chunk shape is $(10, 15, 100)$, demonstrating that the algorithm successfully bypassed the short axes to create the most cubic isotropic shape possible.
*   **`test_margin_truncation_at_absolute_boundaries`:**
    *   **Logic:** Generate padded bounding boxes with a `margin = 20` for a $(100, 100, 100)$ volume.
    *   **Assertion:** Inspect the `padded_bbox` of the very first chunk and very last chunk. Assert that negative indices are clamped to $0$, and indices exceeding $100$ are clamped to $100$, preventing downstream `IndexError` crashes.

## Phase 1.5: Temporary Grid Visualization (VTK Export)

**Objective:** Provide a debugging mechanism to visually confirm the mathematical grid calculation by exporting the raw probability field and the calculated chunk grid to ParaView.

### 1. CLI Toggle and Execution Flow
*   Add a new boolean argument to `argparse` (e.g., `--export-grid-preview`).
*   If this flag is enabled, the pipeline will calculate the bounding boxes, generate the VTK export, and then **immediately terminate/exit** (`sys.exit(0)`), bypassing all downstream preprocessing, skeletonization, and physics modelling.

### 2. Generating the Grid Mask
*   Allocate a 3D numpy array of `uint8` (`grid_mask`) perfectly matching the dimensions of the raw probability field, initialized to zeros.
*   As the orchestrator loops through the generated `core_bbox` definitions, the script will "paint" the boundaries of each chunk.
*   Specifically, the 2D planar faces of every $Z, Y, X$ bounding box will be set to `255`, effectively drawing a 3D wireframe cage that explicitly outlines where the array is being severed.

### 3. Multi-Channel VTK Export
*   Using `pyvista`, initialize a single `pv.ImageData()` object matching the spatial dimensions and physical voxel spacing of the dataset.
*   Add the raw probability field array to the point data (e.g., `vtk_vol.point_data["Probability"]`).
*   Add the newly generated `grid_mask` to the point data as a separate channel (e.g., `vtk_vol.point_data["ChunkGrid"]`).
*   Export this as a `.vti` file. 

**ParaView Verification:** When opened in ParaView, the user can render the `Probability` field as a volume, and use a standard Threshold filter on the `ChunkGrid` channel to instantly overlay the glowing 3D wireframe boxes directly on top of the anatomy, guaranteeing that the calculated grid behaves exactly as requested.

## Phase 2: Localized Preprocessing Optimization and Binary Stitching

**Objective:** Push the preprocessing Bayesian optimization loop (Optuna) into the localized Map-Reduce workers. This allows each individual chunk to independently tune its median filters and hysteresis thresholds to fit its specific local anatomical noise profile.

### 1. Encapsulating the Preprocessing Worker Function
We will construct a self-contained worker function `preprocess_local_chunk(chunk_raw_prob, bbox)` that executes the initial filtering phases on a single padded sub-volume.
*   **Input:** The `chunk_raw_prob` float array (extracted using the `padded_bbox` to include the safety margin) and the `bbox` metadata dictionary.
*   **Preprocessing Optimization:** If `--optimize-preprocessing > 0` is set, a fresh, in-memory Optuna study is launched *exclusively for this chunk*. It tunes the hysteresis thresholds and median filters against this specific local noise profile.
*   **Binary Mask Generation:** The (tuned) preprocessing filters are applied to the raw chunk to create the `local_binary` mask.
*   **Output:** The worker isolates the `core_bbox` voxels of the finalized binary mask (stripping away the padded margin to prevent spatial overlaps) and returns the `local_core_binary` array.

### 2. Map-Reduce Orchestrator (Image Stitching)
The central orchestrator in `carotid_image_to_model.py` will manage the parallel execution and the final reduction.
*   **Parallel Dispatch:** Using `joblib`, the orchestrator will pass the raw probability chunks into the worker function across all available CPU cores concurrently.
*   **Binary Stitching (The Image Reduce):** A massive, empty global array (`stitched_binary_mask`) matching the original image dimensions is pre-allocated. As workers return their `local_core_binary` arrays, the orchestrator slots them into the global array at their exact `core_bbox` coordinates.

### 3. Global Export and Optional Early Exit
Once all chunks are processed and stitched back together:
*   The pipeline will possess a perfectly continuous, locally-optimized global binary mask.
*   This reconstructed mask will be exported via PyVista to a `.vti` file named `..._vessel_mask.vti`.
*   **CLI Toggle:** A new command-line argument (e.g., `--exit-after-mask`) will be added to `argparse`. 
    *   If this flag is provided, the pipeline will **immediately terminate/exit** (`sys.exit(0)`) after exporting the mask. This provides a hard stop for the user to visually verify the locally-optimized stitched binary mask in ParaView before proceeding.
    *   If the flag is *not* provided, the pipeline will seamlessly continue into the downstream global skeletonization and haemodynamic modelling phases using this newly optimized global mask.

### 4. Phase 2: Unit Testing Suite
To guarantee the mathematical accuracy of the isolated image processing and the flawless realignment of the stitched chunks, we will implement the following tests in `tests/test_fractional_preprocessing.py`:

*   **`test_local_worker_margin_stripping`:**
    *   **Logic:** Pass a mocked $50 \times 50 \times 50$ padded probability chunk (where the core is $30 \times 30 \times 30$ and the margin is $10$) into the `preprocess_local_chunk` worker function.
    *   **Assertion:** Verify that the returned `local_core_binary` mask has the exact shape of $(30, 30, 30)$. This proves the worker accurately strips the overlap margin to prevent spatial duplication during stitching.
*   **`test_binary_stitching_continuity`:**
    *   **Logic:** Create a synthetic global probability field containing a solid 3D cylinder that deliberately spans across the boundary of 4 adjacent chunks. Run the orchestrator with `chunk_fraction` enabled.
    *   **Assertion:** Inspect the final `stitched_binary_mask`. Verify that the cylinder remains perfectly continuous and intact. If margin trimming or coordinate slotting failed, there would be a 0-value "gap" or a misaligned shift at the chunk seams.
*   **`test_localized_optuna_invocation`:**
    *   **Logic:** Use Python's `unittest.mock` to spy on the `run_optuna_preprocessing_optimization` function. Execute the Map-Reduce pipeline with `--optimize-preprocessing 5`.
    *   **Assertion:** Verify that the Optuna auto-tuner was explicitly invoked exactly $N$ times (where $N$ is the number of generated chunks), proving that the Bayesian optimization is successfully localized and independently executed for every sub-volume.

---
*Note: This plan will be expanded iteratively to include future phases.*