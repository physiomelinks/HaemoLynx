# Pipeline Auto-Tuning & Benchmarking Engine

## Overview
This document details the Hyperparameter Optimization (HPO) schemes successfully implemented to automate the structural discovery of biological tissues. The pipeline utilizes **Optuna (Bayesian Optimization)** via Tree-structured Parzen Estimator (TPE) algorithms. It is broken into two strict chronological phases:

1.  **3D Voxel Preprocessing Optimization:** Discovers the perfect filters/thresholds to isolate vessels from a raw probability field.
2.  **1D Skeletonization Optimization:** Discovers the perfect topological pruning and bridging parameters to map the 3D mask.

---

## PART 1: 3D Voxel Preprocessing Optimization

Because there is no human-drawn "ground truth" mask to calculate a Dice score against, the AI must score the generated binary mask against the physical properties of the original probability field.

### The Search Space (Optuna Trials)
The objective function exposes the `PreprocessingConfig` parameters to the Optuna TPE sampler.
*   **`hysteresis_threshold_low/high` (Float):** Determines the strictness of probability masking.
*   **`median_filter_size` (Categorical/Int):** Removes salt-and-pepper noise.
*   **`morphological_opening_radius` (Int):** Breaks artificial webbing.
*   **`shannon_entropy_threshold` (Float):** Determines how much blurry/uncertain tissue is aggressively rejected.

### Reference-Free Quantitative Benchmarks (The Cost Function)
1.  **Probability-Mask Cross-Entropy (Confidence Score):**
    *   *Metric:* Average probability value of all voxels where `mask == True`. Also tracks **Probability Yield** to prevent the AI from cheating by returning a 1-voxel mask.
    *   *Goal:* Maximize. Penalizes the AI for setting the threshold so low that it includes highly uncertain background noise.
2.  **Boundary Gradient Coincidence (Crispness Score):**
    *   *Metric:* Calculates the 3D gradient (edges) of the raw probability map. Measures the gradient intensity exactly at the outer surface boundary of the resulting binary mask.
    *   *Goal:* Maximize. Ensures the binary wall aligns perfectly with the sharpest physical drop-off in the raw image, penalizing excessive blurring from median filters.
3.  **Component Fragmentation Penalty (Dust Score):**
    *   *Metric:* `scipy.ndimage.label(mask)[1]` (Total number of isolated 3D components).
    *   *Goal:* Minimize towards 1. Penalizes thresholds that shatter the continuous capillary bed into thousands of floating dots.

### Anti-Cheating Pipeline Staging
To prevent the optimizer from "cheating" the Loss function, strict chronological staging is enforced during the `--optimize-preprocessing` routine.
1.  **STOP AND SCORE:** The 3 metrics above are evaluated *immediately* after thresholding. By scoring fragmentation *before* brute-force deletion, the AI is forced to find thresholds that naturally generate a biologically continuous network.
2.  **Final Injection:** Once Optuna converges, `keep_largest_mask_components` is safely executed. Because the thresholding is mathematically perfected, this step acts solely as a gentle broom to remove 1 or 2 true biological background artifacts, rather than acting as a sledgehammer to fix bad thresholding.

---

## PART 2: 1D Skeletonization Optimization

### The Quantitative Benchmarking Suite
To objectively score the quality of a 1D skeleton against its 3D parent mask, four distinct mathematical benchmarks were implemented:

1.  **Volumetric Reconstruction (Dice/Jaccard):**
    *   **Mechanism:** Computationally "re-dilates" the 1D skeleton back into a solid 3D tube using its assigned physical radii (e.g., FWHM boundaries).
    *   **Metric:** Voxel-wise Dice Similarity Coefficient (DSC) overlap between the dilated tube and the raw binary mask.
2.  **Completeness & Over-Pruning (Orphaned Volume):**
    *   **Mechanism:** Computes the Euclidean Distance Transform (EDT) *outward* from the 1D skeleton into the tissue mask.
    *   **Metric:** Identifies the fraction of tissue voxels that are further than $X \mu m$ (max capillary radius) from the skeleton. Heavily penalizes aggressive stub-pruning that destroys valid capillary beds.
3.  **Topological Preservation (Euler/Betti Components):**
    *   **Mechanism:** Computes the isolated 3D components of the binary mask and compares them against the cyclomatic complexity ($E - V + C$) of the skeleton graph.
    *   **Metric:** Tracks "Fundamental Loops". Softly penalizes the creation of excessive artificial loops ("spiderwebs") inside thick vessels.
4.  **Centerline Centricity:**
    *   **Mechanism:** Computes the EDT *inward* from the mask walls and samples the distances precisely at the 1D skeleton nodes. 
    *   **Metric:** Evaluates how perfectly the mathematical skeleton tracks the physical ridges/center of the biological tubes.

### The Search Space
The AI modifies the `SkeletonConfig` using strictly constrained, biologically-grounded limits:
*   `min_branch_length`: (0 to 15 $\mu m$) - Stub removal.
*   `max_bridge_distance`: (0 to 8 $\mu m$) - Snapping parallel nodes.
*   `min_component_percent`: (0.01% to 2.0%) - Floating island deletion.
*   `bundle_scan_size`: (3 to 8 $\mu m$) - Spiderweb collapse spheres.

### The Loss Function
The outputs from the benchmarking suite are collapsed into a single scalar value.
$$Loss = w_1(1.0 - DSC) + w_2(\text{Orphaned Fraction}) + w_3(\text{Excess Loops})$$
*   By minimizing this Loss towards `0.0`, the TPE algorithm discovers parameters that simultaneously maximize tissue volume while minimizing hallucinated topology.

---

## Early Stopping & Unit Testing
*   **Early Stopping:** Both Optuna loops utilize a custom `EarlyStoppingCallback` (Patience = 15). If a perfect skeleton or threshold is discovered early, the loop safely aborts to save computational resources.
*   **Unit Testing Suite:** Because running 3D morphological optimization is computationally heavy and stochastic, the Pytest suite (`test_auto_tuner.py` & `test_benchmarking.py`) utilizes **"Mock Pipelines"** and **Procedurally Generated Synthetic Objects**.
    *   Generates mathematical primitives (perfect straight cylinders and toruses) to guarantee sub-second, mathematically exact validation (e.g. asserting DSC perfectly evaluates to 1.0).
    *   Proves the Bayesian search successfully navigates the multidimensional space to discover hidden sweet spots without crashing.
    *   Guarantees `ZeroDivisionError` safeguards when evaluating totally anoxic (empty) biological masks.