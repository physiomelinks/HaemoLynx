# Optimal Filtering Strategies for High-Resolution Capillary Segmentation - Version 4
*A Calibrated Pre-processing, Multi-Channel Machine Learning, and Morphological 1D Graph Extraction Pipeline*

---

## 1. Physical and Geometric Calibration Parameters

Quantitative microvascular network characterization is highly sensitive to voxel resolution, and choosing arbitrary filtering parameters can severely distort vessel geometries and transport calculations [3, 4]. At the specified high-resolution confocal fluorescence microscopy (CFM) resolution, the voxels are nearly perfectly isotropic:

$$\Delta x = 1.8660\ \mu\text{m},\quad \Delta y = 1.8660\ \mu\text{m},\quad \Delta z = 1.8639\ \mu\text{m}$$

With an axial-to-lateral aspect ratio of $1.0011$, this dataset is natively isotropic, completely bypassing the need for standard anisotropic Z-axis interpolation [355, 521]. Based on standard anatomical profiles of the mammalian carotid body (CB) and microvascular network dimensions, we calibrate our pixel-to-physical space conversions as follows:

*   **Capillary Boundaries:** Capillaries typically measure $4.0$ to $7.0\ \mu\text{m}$ in diameter [205, 406]. At this voxel size, a capillary spans **$2.14$ to $3.75$ voxels** in the spatial grid [3].
*   **Feeding Arterioles:** Larger arteriole diameters measure $20.0$ to $30.0\ \mu\text{m}$ [5]. This corresponds to **$10.7$ to $16.1$ voxels** [3].
*   **In-Plane and Volumetric Denoising:** A spatial smoothing kernel with $\sigma = 1.0$ voxel corresponds to a physical radius of $1.866\ \mu\text{m}$ [9, 211]. This is ideal for suppressing shot noise without merging parallel capillary tracts [9, 138].
*   **Background Subtraction Window ($L$):** To prevent clipping the hollow lumens of larger feeding vessels, the sliding window length ($L$) must exceed the maximum expected vessel diameter of $30.0\ \mu\text{m}$ [10, 111]. Calibration dictates $L = 25$ to $30$ pixels (representing a physical field of $46.6$ to $56.0\ \mu\text{m}$) [10, 111].

---

## 2. Phase-by-Phase Image Pre-processing Protocol

This corrected Fiji pre-processing workflow isolates the vascular Lectin channel, eliminates optical noise, compensates for tissue signal decay, standardizes dynamic range, and subsequently pre-calculates multiscale tubular vesselness probabilities to maximize Ilastik classification performance.

### Phase 1: Channel Separation and Metadata Validation
1.  **Split Channels:** Go to **`Image > Color > Split Channels`** [83]. Isolate the Lectin-FITC microvascular channel and close the remaining channels (e.g., TH-positive glomus cell somas or DAPI) to conserve system memory [83].
2.  **Verify Dimensionality:** Ensure Fiji recognizes the volume as a 3D Z-stack. If metadata is misaligned (swapping slices with timeframes), go to **`Image > Properties...`** and restore the proper dimensions [7].
3.  **Filenames Cleaning:** Remove all white spaces, parentheses, or multiple periods from filenames to prevent batch script parsing errors [8].

### Phase 2: Z-Axis Signal Decay & Bleach Correction
Thick, cleared tissue blocks inherently experience light scattering and fluorescent signal decay in deep planes relative to the objective [47, 85].
1.  Navigate to **`Image > Adjust > Bleach Correction`** (or use the *3D ImageJ Suite* plugin) [83, 287].
2.  Select **`Histogram Matching`** as the correction method. This algorithm matches the intensity histogram of each Z-slice to a reference slice [83].
3.  **Close the original stack** and proceed exclusively with the generated, corrected stack (prefixed with `DUP_`) to maintain a clean memory footprint.

### Phase 3: Rolling-Ball Background Subtraction
Cleared tissue imaging frequently captures out-of-focus light haze, uneven illumination, and background autofluorescence from adjacent connective tissues [87, 111].
1.  Go to **`Process > Subtract Background...`** [111, 288].
2.  Set the **`Rolling Ball Radius` to `30.0` pixels** [111]. This value ($\approx 56\ \mu\text{m}$) is calibrated to be larger than your thickest $30\ \mu\text{m}$ arteriole, preventing the erosion of internal vascular signal [10, 111].
3.  Ensure **`Light Background` is unchecked** and **`Process entire stack`** is selected.

### Phase 4: Volumetric Edge-Preserving Denoising
Linear smoothing filters (like Gaussian Blur) blend intensities across sharp structural interfaces, creating a transitional gradient that artificially inflates vessel diameters and merges closely running parallel capillaries [138, 222].
1.  Go to **`Process > Filters > Median 3D...`** [14, 354].
2.  Set **`x = 1`, `y = 1`, and `z = 1`** [14].
3.  *Rationale:* Because your voxels are isotropic, this symmetric 3D Median kernel suppresses high-frequency "salt-and-pepper" camera noise while keeping capillary borders extremely sharp and distinct for subsequent manual training [14].

### Phase 5: Dynamic Range Normalization (Crucial Prior Step)
To standardize intensities between variable WKY (control) and SHR (hypertensive) cohorts and negate staining or perfusion variations, we normalize the raw histograms **prior to any vesselness calculation** [38, 100].
1.  On the preprocessed grayscale stack, go to **`Process > Enhance Contrast...`** [100].
2.  Set **`Saturated Pixels` to `0.35%`** [100].
3.  Check **`Normalize`** and **`Process all slices`** [100]. This rescales the 3D volume to the complete dynamic scale (e.g., $[0, 65535]$ for 16-bit or $[0, 255]$ for 8-bit), creating a uniform intensity feature space and stable gradients for subsequent second-derivative Hessian computations [38, 100].

### Phase 6: Multiscale Tubeness/Frangi Filtering (The Shape-Aware Channel)
To provide the machine learning model with specialized shape-aware priors, we pre-calculate a 3D Tubeness/Frangi map to differentiate tubular capillaries from spherical autofluorescent debris or flat structural boundaries [138, 425]. 

Since Fiji’s native **`Tubeness`** plugin only runs on a single scale (Sigma) at a time, you must manually generate multiple scale maps, stack them, and project them to construct a true multiscale vesselness map. Follow these precise interactive GUI steps:

1.  **Keep your Normalized Grayscale Stack active** (the output from Phase 5). Let's assume its window title is `Normalized_Grayscale`.
2.  **Generate Individual Scale Maps:**
    *   Go to the menu: **`Process > Filters > Tubeness`** [288, 387].
    *   In the **`Tubeness`** dialog box:
        *   Set **`Sigma (pixels)`** to **`1.0`** (calibrated to enhance the finest capillaries of $\approx 1.86\ \mu\text{m}$ radius) [154, 205].
        *   **CRITICAL:** **Uncheck** the box for **`Use calibration`** [7]. (Checking this will make Fiji look for physical microns, which causes scale distortion; unchecking forces Fiji to use our precise, calibrated pixel-to-voxel ratios).
        *   Click **`OK`**. Fiji will process the stack and open a new 32-bit window named `tubeness of Normalized_Grayscale`. Rename this window to **`tubeness_s1.0`** (`Image > Rename...`) and keep it open.
    *   Click back on your original **`Normalized_Grayscale`** stack window to make it active.
    *   Go to **`Process > Filters > Tubeness`** again:
        *   Set **`Sigma (pixels)`** to **`2.0`**. Uncheck **`Use calibration`**. Click **`OK`**. Rename this output window to **`tubeness_s2.0`**.
    *   Make the original **`Normalized_Grayscale`** stack active again and repeat the process for:
        *   **`Sigma = 4.0`** (rename output to **`tubeness_s4.0`**).
        *   **`Sigma = 8.0`** (rename output to **`tubeness_s8.0`** to capture major feeding arterioles) [154].
3.  **Assemble the Multiscale Stack:**
    *   Go to the menu: **`Image > Stacks > Images to Stack`**.
    *   In the dialog box:
        *   Set **`Name`** to **`Scale_Stack`**.
        *   Set **`Title contains`** to **`tubeness`** (this pulls all four of your open scale windows into a single coordinate stack).
        *   Ensure **`Bicubic interpolation`** is unchecked and click **`OK`**.
4.  **Compute the Multiscale Maximum Projection (MIP across scales):**
    *   With the new **`Scale_Stack`** window active, go to **`Image > Stacks > Z Project...`**.
    *   Set **`Start slice`** to **`1`** and **`Stop slice`** to **`4`**.
    *   In the **`Projection type`** dropdown, select **`Max Intensity`**.
    *   Click **`OK`**. This generates a new window representing your true multiscale vesselness map—containing the maximum response for each voxel across all four calibrated sizes.
5.  **Dynamic Range Normalization of the Vesselness Channel:**
    *   The generated max-intensity projection is a 32-bit floating-point image. We must scale it to match the dynamic range of our grayscale channel:
    *   With the projected window active, go to **`Process > Enhance Contrast...`** [100].
    *   Set **`Saturated pixels`** to **`0.35%`** [100].
    *   Check **`Normalize`** and **`Process all slices`** [100]. Click **`OK`**.
    *   Convert the normalized 32-bit map to 16-bit (or 8-bit, matching your grayscale channel type): Go to **`Image > Type > 16-bit`** (or **`Image > Type > 8-bit`**).
    *   Save this final window as **`Multiscale_Vesselness`** [211].

### Phase 7: RAM-Optimized Lateral Tiling and Exporters Setup
To prevent Out-Of-Memory (OOM) crashes in Ilastik while preserving continuous vascular pathways in 3D, we tile the stack laterally [48, 355]:
1.  Go to **`Edit > Selection > Specify...`**.
2.  Set **`Width = 256`** and **`Height = 256`** (representing a physical field-of-view of $\approx 478 \times 478\ \mu\text{m}^2$).
3.  Check **`Slice = 1`** and **`Stack`**.
4.  Duplicate the sub-stack (**`Image > Duplicate...`**, ensuring `Duplicate stack` is checked) and save as a TIFF tile [211].
5.  Increment $X$ and $Y$ coordinates by $256$-pixel offsets and save each tile. **Never crop along the Z-axis**; keeping the entire Z-depth intact is mandatory for preserving 3D topological connectivity [143, 170].

---

## 3. Two-Channel Ilastik Dataset Structuring

Instead of training on a single channel, combine your preprocessed grayscale and Frangi vesselness images into a **two-channel HDF5 (`.h5`)** stack [83, 138]. This hybrid input merges the raw intensity and edge details of the grayscale data with the hand-crafted tubular priors of the Frangi map [60, 138].

### Channel Mapping Layout:
*   **Channel 1 (Grayscale):** Corrected, rolling-ball subtracted, 3D median-smoothed, and normalized intensity data [38, 111, 354].
*   **Channel 2 (Vesselness):** Normalized multiscale Frangi/Tubeness vesselness probability map [48, 100, 138].

### Compilation and Conversion:
1.  In Fiji, open both corresponding grayscale and Frangi tiles [83].
2.  Go to **`Image > Color > Merge Channels...`**, mapping Grayscale to `C1` and Frangi to `C2` [83].
3.  Run the **`convert_tiles_to_hdf5.py`** Python script over the merged TIFFs [11, 12]:
    ```bash
    python convert_tiles_to_hdf5.py --input-dir "/fiji_merged_tiles" --output-dir "/ilastik_inputs" --axes ZCYXS --compression 4
    ```
    *Why this axis tagging?* The script writes the **`ZCYXS`** axis tag metadata directly into the HDF5 header [11, 12]. This guarantees that Ilastik reads the files as a true 3D spatial grid with 2 independent channels, preventing slice-swapping or geometric distortion on import [7, 12].

---

## 4. Multi-Scale Feature Selection in Ilastik

In Ilastik's **Feature Selection** tab, we choose multi-scale Gaussian derivative filters to compute voxel properties [66, 426]. Because our inputs are organized in two distinct channels, we configure the features to capture complementary details [83, 138]:

| Feature Group | Channel 1 (Grayscale) Settings | Channel 2 (Frangi Map) Settings | Physical Calibration Relevance |
| :--- | :--- | :--- | :--- |
| **Color/Intensity** | All Sigmas ($\sigma_0$ to $\sigma_6$) | All Sigmas ($\sigma_0$ to $\sigma_6$) | Captures local fluorescent brightness and pre-calculated vesselness probability [38, 138]. |
| **Edge (Laplacian / Gradient)** | $\sigma = 0.70$ (3D), $\sigma = 1.00$ (3D) | $\sigma = 1.00$ (3D), $\sigma = 3.50$ (2D) | **$\sigma = 0.70$ (3D)** extracts the sharp boundary walls of thin capillaries ($\approx 1.3\ \mu\text{m}$) [66]. **$\sigma = 3.50$** handles larger vessel boundaries [66]. |
| **Texture (Hessian / Tensor)** | $\sigma = 1.00$ (3D), $\sigma = 1.60$ (3D) | $\sigma = 1.60$ (3D), $\sigma = 3.50$ (2D) | Identifies 3D tubular flow directionality [66, 138]. Restricting larger scales ($\ge 3.5$) to 2D prevents CPU memory crashes [66]. |

### Feature Scale Calibration Values:
*   **$\sigma = 0.30$ to $0.70$ (3D):** Sub-voxel to microscopic boundary interfaces [66].
*   **$\sigma = 1.00$ (3D):** Calibrated to your isotropic capillary wall boundaries ($\approx 1.86\ \mu\text{m}$) [66].
*   **$\sigma = 1.60$ (3D):** Captures the whole capillary diameter ($\approx 3.0\ \mu\text{m}$), enforcing 3D structural continuity [66].
*   **$\sigma = 3.50$ to $5.00$ (2D):** Captures larger arteriolar boundaries ($\approx 6.5 - 9.3\ \mu\text{m}$) [66].
*   **$\sigma = 10.00$ (2D):** Captures macroscopic anatomical background transitions ($\approx 18.6\ \mu\text{m}$) [66].

---

## 5. Interactive Machine Learning & Export Configuration

### Training Class Strategy:
Define **two classes** for your random forest classifier [11]:
1.  **Vessels:** Paint thin, short brush strokes down the core lumens of capillaries in 3D [11]. Scroll through the Z-planes using your mouse wheel to keep annotations contiguous in 3D [391].
2.  **Background:** Label the dark background tissue, autofluorescent circular debris, and empty space [11]. Paint specifically close to capillary walls to teach the model to resolve closely running parallel tracts without merging them [138].

### Resolving the Parallel Vessel "Contact" Trap:
In dense, hypervascular carotid body networks (especially in SHR cohorts), closely running parallel capillaries frequently merge in Frangi vesselness maps [138]. Teach Ilastik to "cut" these merged zones by drawing precise **Background** labels in the narrow gap separating the parallel vessels on Channel 1 (Grayscale), while relying on Channel 2 (Frangi) to maintain overall capillary loop continuity [138].

### Prediction Export Settings:
1.  Under the **Prediction Export** tab, set **`Source: Probability Map`** [11]. Preserving probability gradients (as opposed to a simple binary segmentation) is crucial for custom threshold fine-tuning and morphological repair in downstream steps [11, 13].
2.  Select **`Format: HDF5`** and **`Data Type: unsigned 8-bit`** [11].
3.  Enable **`Renormalize`** to scale the raw floating-point probabilities from $[1.0, 2.0]$ to **$[0, 255]$** [11]. In your exported file, a value of $255$ represents a $100\%$ vascular probability [11].
4.  Run the **Batch Processing** tab to apply your trained classifier across your entire WKY and SHR tile directories [11].

---

## 6. Post-processing Strategy for 1D Vascular Graph Extraction

Once your 8-bit probability map tiles are exported, run the following post-processing sequence in Fiji or Python to construct an unbroken, geometrically accurate 3D skeleton for blood flow simulations:

1.  **Re-Import and Binarize:** Re-import your `.h5` probability maps into Fiji and apply a **global threshold cutoff of `128`** (corresponding to $\ge 50\%$ vascular probability) to generate binary masks [11, 13].
2.  **3D Cavity Filling:** Apply **`3D Fill Holes`** (from the *3D ImageJ Suite*) to solidify hollow vessel lumens resulting from inconsistent dye penetration [13, 201].
3.  **Bridge Staining Gaps:** To repair minor capillary discontinuities along the Z-axis:
    *   Apply **`Dilate (3D)`** with a radius of $1$ voxel to slightly expand vessel boundaries [13].
    *   Apply **`Median 3D`** with a radius of $2$ voxels in $X/Y$ and $1$ voxel in $Z$ to bridge micro-gaps [14].
    *   Apply **`Erode (3D)`** using a voxel distance of $1$ voxel to reduce the vessel extension to its original size before the dilation in a) while preserving the new connections between vessels through the median filtering in b) [13, 14].
4.  **Extract 1D Skeletons:** Run Fiji's native **`Analyze Skeleton (2D/3D)`** (with the *prune shortest branch* option enabled) or a Python 3D thinning algorithm to compute continuous, single-pixel-wide centerlines [16, 171, 212].
5.  **Euclidean Distance Mapping:** Apply the **`3D Distance Map`** function (EDT) to the binary vessel mask [17, 173]. Multiply this EDT map with your extracted skeleton centerlines to obtain highly accurate, orientation-independent capillary radii along every point of the network [17, 18, 355].
