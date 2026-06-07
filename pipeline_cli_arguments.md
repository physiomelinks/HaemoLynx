# ImageLynx Pipeline Command-Line Arguments

The main execution script `examples/carotid_image_to_model.py` accepts several command-line arguments to override default configurations and trigger specific execution phases. This allows for flexible benchmarking, debugging, and auto-tuning without modifying the underlying Python code.

## Execution Control & Early Exits

*   **`--chunk-fraction <float>`**
    *   **Description:** Enables Map-Reduce parallel processing. Divides the global volume into localized sub-volume chunks based on the provided fraction (e.g., `0.25` divides each axis into approximately 4 chunks, yielding 64 chunks total).
    *   **Type:** `float` (Default: `None`)

*   **`--export-grid-preview`**
    *   **Description:** Toggles an early termination immediately after Phase 1.5. If set, the script calculates the chunk bounds, exports the raw probability field and grid wireframes to `.vti` files, and exits cleanly via `sys.exit(0)`.
    *   **Type:** Flag (Boolean)

*   **`--exit-after-mask`**
    *   **Description:** Toggles an early termination immediately after Phase 2. If set, the script executes the localized Map-Reduce workers, stitches the optimized binary masks into a global volume, exports `vessel_mask.vti`, and exits before entering global skeletonization.
    *   **Type:** Flag (Boolean)

## Bayesian Auto-Tuning (Optuna)

*   **`--optimize-preprocessing <int>`**
    *   **Description:** Runs an in-memory Optuna study to automatically tune hysteresis thresholds and median filter sizes. When used alongside `--chunk-fraction`, the optimization executes independently *per chunk*.
    *   **Type:** `int` (Number of trials, Default: `0`)

*   **`--optimize-skeleton <int>`**
    *   **Description:** Runs an Optuna study to tune skeletonization pruning, tortuosity limits, and branch collapsing parameters against an objective loss function.
    *   **Type:** `int` (Number of trials, Default: `0`)

*   **`--optimize-patience <int>`**
    *   **Description:** Overrides the Optuna `EarlyStoppingCallback` patience limit. If the objective score does not improve after $N$ consecutive trials, the study truncates early.
    *   **Type:** `int` (Default: dynamically set by config)

## Configuration & ROI Overrides

*   **`--config <path>`**
    *   **Description:** Path to a YAML file containing dataclass overrides. Allows you to define distinct physical environments (e.g., Normotensive vs Hypertensive) cleanly.
    *   **Type:** `string` (Path)

*   **`--sub-volume <float>`**
    *   **Description:** Quickly crops the raw loaded image to a fractional Region of Interest (ROI) from the center. Useful for fast localized debugging (e.g., `0.1` crops to the central 10% volume).
    *   **Type:** `float` (Range: `0.0` to `1.0`, Default: `None`)

## Physics & Graph Overrides

*   **`--core-resolution {eradicate, stitch, none}`**
    *   **Description:** Overrides the strategy for handling dead-end stubs in the graph core.
    *   **Choices:** `eradicate` (delete dead-ends), `stitch` (reconnect via distance logic), `none` (leave intact).

*   **`--boundary-mode {caged, universal_sink, robin_resistance}`**
    *   **Description:** Overrides the X/Y spatial boundary permeability logic.
    *   **Choices:** `caged` (no flow leaves X/Y), `universal_sink` (Dirichlet 0 pressure), `robin_resistance` (dynamic resistance bleeding).

*   **`--radius-mode {fwhm_radius, edt_radius, constant_radius}`**
    *   **Description:** Overrides how the pipeline computes the physical vessel radius for Poiseuille conductance. `edt_radius` is strongly recommended for stability.
    *   **Choices:** `fwhm_radius`, `edt_radius`, `constant_radius`.
