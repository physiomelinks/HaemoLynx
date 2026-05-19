# Preprocessing Hyperparameter Optimization (HPO) Plan

## Objective
To extend the Optuna Bayesian Optimization framework from the 1D Skeletonization phase into the **3D Voxel Preprocessing** phase. This will allow the AI to automatically discover the optimal mathematical filters and thresholds required to convert the raw continuous probability map (e.g., from Ilastik) into a cohesive, biologically accurate binary mask.

---

## 1. The Search Space (Optuna Trials)
The objective function will expose the `PreprocessingConfig` parameters to the Optuna TPE sampler.
*   **`hysteresis_threshold_low` (Float):** e.g., 0.10 to 0.50
*   **`hysteresis_threshold_high` (Float):** e.g., 0.30 to 0.90
*   **`median_filter_size` (Categorical/Int):** e.g., [0, 3, 5, 7] to remove salt-and-pepper noise.
*   **`morphological_opening_radius` (Int):** e.g., 0 to 3, to break artificial webbing.
*   **`shannon_entropy_threshold` (Float):** e.g., 0.70 to 0.99, determining how much blurry/uncertain tissue is aggressively rejected.

---

## 2. Reference-Free Quantitative Benchmarks (The Cost Function)
Because there is no human-drawn "ground truth" mask to calculate a Dice score against, the AI must score the generated binary mask against the physical properties of the original probability field.

1.  **Probability-Mask Cross-Entropy (Confidence Score):**
    *   *Metric:* Average probability value of all voxels where `mask == True`.
    *   *Goal:* Maximize. Penalizes the AI for setting the threshold so low that it includes highly uncertain (e.g., 15% probability) background noise.
2.  **Boundary Gradient Coincidence (Crispness Score):**
    *   *Metric:* Calculate the 3D gradient (edges) of the raw probability map. Measure the gradient intensity exactly at the outer surface boundary of the resulting binary mask.
    *   *Goal:* Maximize. Ensures the binary wall aligns perfectly with the sharpest physical drop-off in the raw image, penalizing excessive blurring/bloating from median filters.
3.  **Component Fragmentation Penalty (Dust Score):**
    *   *Metric:* `scipy.ndimage.label(mask)[1]` (Total number of isolated 3D components).
    *   *Goal:* Minimize towards 1. Penalizes thresholds that are too aggressive and shatter the continuous capillary bed into thousands of floating dots.

---

## 3. Pipeline Staging & Execution Logic
To prevent the optimizer from "cheating" the Loss function, strict chronological staging must be enforced during the `--optimize-preprocessing` routine.

### Stage 1: The Evaluation Loop (Pre-Pruning)
1.  Optuna generates threshold parameters.
2.  Pipeline applies median filtering, Shannon entropy, and Hysteresis.
3.  **STOP AND SCORE:** The 3 metrics above are evaluated *immediately*. 
4.  *Crucial Rationale:* By scoring fragmentation *before* brute-force deletion, the AI is forced to find thresholds that naturally generate a biologically continuous network.

### Stage 2: Final Injection (Post-Optimization)
1.  Optuna converges and saves `best_preprocessing_params.yaml`.
2.  The main pipeline resumes using these optimal thresholds to generate the volume.
3.  **Now**, `keep_largest_mask_components` is safely executed. Because the thresholding is mathematically perfected, this step acts solely as a gentle broom to remove 1 or 2 true biological background artifacts, rather than acting as a sledgehammer to fix bad thresholding.
4.  The perfected binary mask is passed to the skeletonizer.