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

## Phase 1.5: Permanent Grid and Probability Visualization (VTK Export)

**Objective:** Guarantee that users always have access to a visual representation of the grid layout overlaid on the raw probability field. These visualization artifacts are now a mandatory part of the chunking workflow.

### 1. Mandatory Export and CLI Early Exit
*   **Always Export:** Whenever `chunk_fraction < 1.0` is specified, the pipeline will **always** calculate the bounding boxes and explicitly export the underlying visualization data to distinct `.vti` files.
*   **CLI Toggle:** A boolean argument `--export-grid-preview` will be used exclusively to toggle early termination.
    *   If enabled, the script will generate the `.vti` exports and then **immediately terminate/exit** (`sys.exit(0)`).
    *   If disabled, the script generates the exact same `.vti` exports but continues seamlessly into Phase 2.

### 2. Generating the Grid Mask
*   Allocate a 3D numpy array of `uint8` (`grid_mask`) perfectly matching the dimensions of the raw probability field, initialized to zeros.
*   As the orchestrator loops through the generated `core_bbox` definitions, the script will "paint" the boundaries of each chunk.
*   Specifically, the 2D planar faces of every $Z, Y, X$ bounding box will be set to `255`, effectively drawing a 3D wireframe cage that explicitly outlines where the array is being severed.

### 3. Distinct VTK Exports
Using `pyvista`, the script will initialize multiple separate `pv.ImageData()` objects:
*   **Dual Raw Anatomy Exports:** The pipeline will load the original, unprocessed `.tif` image (from `RAW_IMAGE_PATH`). It will explicitly perform *two* separate exports of this biological ground truth regardless of whether `--export-grid-preview` is toggled on or off:
    1.  **Global Volume Export:** The absolute, uncropped, original TIFF volume is exported as `..._raw_anatomy_global.vti`.
    2.  **Sub-Volume Export:** If the user specified an ROI via `--sub-volume`, the pipeline will crop the raw anatomy to perfectly align with the processing region and export it as `..._raw_anatomy_subvolume.vti`. (If no crop is specified, this simply mirrors the global volume).
*   **Raw Probability Export:** The next `.vti` file will contain only the raw float probability field from Ilastik (`..._raw_probability.vti`).
*   **Grid Mask Export:** A `.vti` file will contain only the binary wireframe chunk boundaries (`..._grid_preview.vti`).
*   **Shannon-Entropy Export:** If `enable_shannon_entropy` is active, a final `.vti` file will be generated containing the calculated Shannon-entropy float field (`..._shannon_entropy.vti`).

**ParaView Verification:** When opened in ParaView, the user can load these independent files simultaneously. By superimposing the reconstructed `vessel_mask.vti` (from Phase 2) or the `raw_probability.vti` directly on top of the `raw_anatomy_subvolume.vti`, the user can slice through the Z-axis and quantitatively/qualitatively assess exactly how well the mathematical preprocessing lines up with the true biological boundaries of the raw TIFF image *within the exact processing bounds*. Furthermore, comparing this to the `raw_anatomy_global.vti` allows the user to see the exact macroscopic context of their selected ROI. Rendering the `Shannon-Entropy` volume alongside these further allows the user to visually inform exactly where they should be placing their hard Shannon-entropy rejection threshold.

### 4. Phase 1.5: Unit Testing Suite
To ensure the robustness of the visualization generation and the correct handling of file I/O, the following tests will be implemented in a dedicated test file (e.g., `tests/test_vtk_export.py` or within existing preview tests):

*   **`test_mandatory_vtk_generation`:**
    *   **Logic:** Mock a 3D probability array, a 3D entropy array, and a 3D raw anatomical array. Execute the Phase 1.5 export block with `chunk_fraction = 0.5`.
    *   **Assertion:** Intercept the `pv.ImageData().save()` calls. Verify that exactly five distinct `.vti` files are saved: `_raw_anatomy_global.vti`, `_raw_anatomy_subvolume.vti`, `_raw_probability.vti`, `_grid_preview.vti`, and `_shannon_entropy.vti`.
*   **`test_dual_anatomy_export_alignment`:**
    *   **Logic:** Provide a mock global TIFF volume of $100 \times 100 \times 100$, and configure `--sub-volume 0.5` to target the central $50 \times 50 \times 50$ core.
    *   **Assertion:** Assert that the generated `_raw_anatomy_global.vti` object retains the full $1000000$ points, while the `_raw_anatomy_subvolume.vti` object correctly scales down to exactly $125000$ points matching the sub-volume geometry.
*   **`test_grid_wireframe_painting`:**
    *   **Logic:** Pass a synthesized $100 \times 100 \times 100$ bounding box dictionary into the grid masking algorithm.
    *   **Assertion:** Mathematically verify that the `grid_mask` array contains `255` ONLY on the explicit 2D planar boundaries (e.g., $Z=0, Z=99$, etc.) and that the interior volume remains strictly `0`. This guarantees the wireframe is hollow and accurately demarcates the chunk.
*   **`test_early_termination_toggle`:**
    *   **Logic:** Execute the orchestrator twice using `unittest.mock.patch` on `sys.exit`. First run with `--export-grid-preview` enabled, second run with it disabled.
    *   **Assertion:** Verify that the exports strictly occur *in both runs*, but that `sys.exit(0)` is invoked perfectly after the VTK export ONLY in the first run, explicitly bypassing the exit in the second run to allow the pipeline to proceed to Phase 2.

## Phase 2: Localized Map-Reduce Preprocessing & Dual-Parameter Hysteresis

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