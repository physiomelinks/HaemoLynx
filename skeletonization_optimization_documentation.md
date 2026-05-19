# 1D Skeletonization Auto-Tuning & Benchmarking Engine

## Overview
This document details the Hyperparameter Optimization (HPO) scheme successfully implemented to automate the topological cleanup of the 1D vascular centerlines. The pipeline utilizes **Optuna (Bayesian Optimization)** via Tree-structured Parzen Estimator (TPE) algorithms to mathematically discover the perfect pruning, bundling, and skeletonization parameters for a given 3D biological mask.

---

## 1. The Quantitative Benchmarking Suite (`benchmarking.py`)
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

---

## 2. The Optuna Bayesian Loop (`auto_tuner.py`)
The pipeline exposes a command-line flag (`--optimize-skeleton N`) that temporarily hijacks the workflow to run $N$ evolutionary trials.

### The Search Space
The AI modifies the `SkeletonConfig` using strictly constrained, biologically-grounded limits:
*   `min_branch_length`: (0 to 50 $\mu m$) - Stub removal.
*   `max_bridge_distance`: (0 to 20 $\mu m$) - Snapping parallel nodes.
*   `min_component_percent`: (0.01% to 5.0%) - Floating island deletion.
*   `bundle_scan_size`: (3 to 15 $\mu m$) - Spiderweb collapse spheres.

### The Loss Function
The outputs from the benchmarking suite are collapsed into a single scalar value.
$$Loss = w_1(1.0 - DSC) + w_2(\text{Orphaned Fraction}) + w_3(\text{Excess Loops})$$
*   By minimizing this Loss towards `0.0`, the TPE algorithm discovers parameters that simultaneously maximize tissue volume while minimizing hallucinated topology.
*   **Early Stopping:** The loop utilizes a custom `EarlyStoppingCallback` (Patience = 15). If a perfect skeleton is discovered early, the loop safely aborts to save computational resources.

---

## 3. The Unit Testing Suite (`test_auto_tuner.py` & `test_benchmarking.py`)
Because running 3D morphological optimization is computationally heavy and stochastic, the Pytest suite utilizes **"Mock Pipelines"** and **Procedurally Generated Synthetic Objects** to guarantee sub-second, mathematically exact validation.

*   **Perfect Synthetic Assertions:** Generates mathematical primitives (perfect straight cylinders and toruses). Asserts that the benchmarking suite evaluates these objects to exactly `DSC = 1.0` and exactly `Loops = 1`.
*   **Over-Pruning Assertions:** Mathematically cuts a synthetic cylinder perfectly in half. Asserts the Completeness algorithm successfully identifies an `orphaned_volume_fraction` of exactly $0.5$ (50%).
*   **Optuna TPE Discovery Test:** Runs the full Optuna engine against a rigged mock pipeline where a specific parameter (e.g., `min_branch_length = 15`) is mathematically enforced as the global minimum. Asserts that the Bayesian search successfully navigates the multidimensional space to discover this hidden sweet spot without crashing.
*   **Graceful Degradation:** Guarantees `ZeroDivisionError` safeguards when evaluating totally anoxic (empty) biological masks.