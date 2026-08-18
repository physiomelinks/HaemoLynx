# Preprocessing LCFM Carotid Body Z-Stacks for Ilastik Pixel Classification — Version 5

*Corrected pipeline: calibration → conservative pre-processing → multiscale vesselness → multi-channel Ilastik → probability-domain post-processing → 1D graph extraction*

**Supersedes v4.** Every step below has been checked against Fiji's actual menus/dialogs and against the measured contents of `CB3-WKY-CB-{A,B,C}-2x2x2.tif`. A change log versus v4 is in §8.

---

## 0. The constraint that governs everything

Your voxels are **1.8660 × 1.8660 × 1.8639 µm** (axial:lateral = 1.0011, natively isotropic — no z-interpolation needed for shape).

At this sampling:

| Structure | Diameter | Diameter in voxels |
|---|---|---|
| Capillary | 4–7 µm | **2.1 – 3.8** |
| Feeding arteriole | 20–30 µm | 10.7 – 16.1 |

**A capillary is two to four voxels across.** This single fact invalidates most "standard" denoising advice, because any filter with a support comparable to the structure width will destroy it:

* A 3×3×3 median (radius 1) spans 5.6 µm — *wider than a capillary*. Applied to a 2-voxel-wide tube, the majority of every neighbourhood is background, so the tube is erased or fragmented.
* Morphological opening, binary median, and "smooth then threshold" have the same failure mode.
* Any operation that costs you one voxel of radius costs you ~50 % of a capillary's radius.

**Rule for this dataset: prefer no filtering over mild filtering.** Ilastik's Random Forest is explicitly designed to learn from noisy raw intensities using its own multi-scale derivative features. Pre-smoothing removes information it would otherwise exploit, and cannot be undone.

### 0.1 What this resolution is and is not good for

The 2×2×2-binned data is being used deliberately as a **fast iteration substrate** for developing the segmentation → skeletonisation → haemodynamics pipeline. The unbinned ~0.933 µm acquisition exists and is the eventual production input. That is a sound strategy, with one caveat that must be stated in any interim result:

**Radii measured at 1.866 µm are not quantitatively usable.** Radii come from a Euclidean distance transform, quantised in voxel units:

* A capillary radius is 1.1–1.9 voxels here. Half-voxel placement error on a nominal 3.0 µm radius is **±31 %**.
* Poiseuille resistance scales as *r*⁻⁴, so that propagates to a **~3× error in segment resistance**.

At 0.933 µm the same half-voxel error is ±16 %, i.e. ~1.8× on resistance. Still not negligible, but a different regime.

**Consequence for interim work:** treat everything topological — connectivity, branch counts, junction degree, path lengths, tortuosity — as trustworthy at 1.866 µm. Treat absolute radii, and therefore absolute flows and pressures, as **pipeline-validation output only, not biological results**. Relative WKY-vs-SHR comparisons at fixed resolution are more defensible than absolutes, since the quantisation bias is common to both cohorts, but only if capillary diameters are similar between them — which is precisely what you may be trying to test, so do not lean on it.

If you need a better interim number without moving to full resolution, `prob_to_mask.py --refine-radii` (§11) upsamples the probability map ×2 before the EDT. It creates no new information, but it lets the boundary sit at a sub-voxel position and halves the quantisation step. Verified on volume A: the ridge-voxel median radius is stable at 1.87 µm either way, while the quantisation step drops from 1.87 to 0.93 µm.

### 0.2 Transferring this protocol to the unbinned 0.933 µm data

Two things will **not** carry over, and it is worth knowing now rather than discovering it later:

* **Ilastik labels do not transfer.** They are stored in dataset pixel coordinates. Moving to 0.933 µm means relabelling from scratch. Budget for it; do not over-invest in exhaustive labelling at 1.866 µm.
* **The trained classifier does not transfer.** Feature sigmas are in pixels, so the same σ means a different physical scale.

Every parameter expressed in **pixels** must double. Parameters in **physical units or unitless** are unchanged:

| Parameter | § | 1.866 µm (binned) | 0.933 µm (unbinned) |
|---|---|---|---|
| Rolling-ball radius | 3 | 30 px | **60 px** |
| Remove Outliers radius | 4 | 1 px | **2 px** |
| Tubeness σ set | 6.2 | 1.0, 1.4, 2.0, 4.0, 8.0 px | **2.0, 2.8, 4.0, 8.0, 16.0 px** |
| Ilastik feature σ set | 9.1 | 0.7, 1.0, 1.6, 3.5 | **1.6, 3.5, 5.0, 10.0** (+ custom 2.0) |
| `--min-size` | 10 | 50 vx | **400 vx** (8× volume) |
| Enhance Contrast saturation | 5 | 0.02 % | 0.02 % — unchanged |
| Hysteresis thresholds | 10 | 0.70 / 0.30 | 0.70 / 0.30 — unchanged |

> **Tip that removes this whole table:** in the Tubeness dialog, **check `Use calibration information`** and enter σ in **µm** (1.87, 2.61, 3.73, 7.46, 14.93). Those values are then resolution-independent and transfer unchanged. The cost is that your existing σ = 1, 2, 4, 8 px maps would need regenerating — which §2 says you need to do anyway. Ilastik's feature sigmas are always in pixels and will still have to be shifted.

### 0.3 RAM planning for the full-resolution run

Your machine has **31 GB RAM / 20 cores**. At 1.866 µm nothing below is a concern. At 0.933 µm the volume becomes ~1014 × 912 × 870 ≈ **805 M voxels**:

| Item | Size |
|---|---|
| One float32 channel | 3.2 GB |
| 3-channel ilastik input, in RAM | **9.7 GB** |
| `preprocess_cb.py` parent (volume + 2 vesselness results) | **~12.9 GB** — see §13.5, this is the binding constraint on the preprocessing side |
| Ilastik interactive training | fine — features are computed only for the visible block |
| Ilastik full-volume prediction | the binding constraint on the classification side |

Measured at the current binned resolution, for calibration: volume A (101 M voxels) peaks at **12.2 GB and 4.6 min** with the default settings. Full resolution is 8× that in voxels.

Before the full-resolution run: set ilastik's memory cap in `Settings > Preferences` to ~20 GB, drop the thread count to ~8 (20 threads × per-block feature stacks is what actually triggers OOM, not the raw volume), and run prediction **headless** rather than from the GUI. Do not use `--refine-radii` at 0.933 µm — it would need ~77 GB and is pointless once you already have the resolution.

### 0.4 Calibration constants used below

| Quantity | Value | Derivation |
|---|---|---|
| Voxel (x, y) | 1.8660 µm | TIFF `XResolution` |
| Voxel (z) | 1.8639 µm | ImageJ `spacing` |
| σ = 1.0 px | 1.866 µm | — |
| Rolling-ball radius 30 px | 56.0 µm radius | Must exceed the *radius* of the largest bright object; a 30 µm arteriole has a 15 µm radius, so 30 px gives ~3.7× headroom |
| Tile 256 px | 477.7 µm | (not used — see §5) |

> **Note on terminology (v4 error):** `Subtract Background` takes a rolling-ball **radius**, not a "sliding window length *L*". The correct rule is *radius > largest object radius*, not *> largest object diameter*. Both roads lead to ~30 px here, but the reasoning in v4 was wrong.

---

## 0.5 If you are running the Python pipeline (recommended), read §13 instead of §1–8

Sections 1–8 describe the Fiji GUI route and are kept as the reference for *what each step does and why*. `preprocess_cb.py` implements all of them end-to-end, so if you are working in Python the operational instructions are **§13**, and §1–8 become background reading for tuning parameters.

The two routes produce equivalent output. Differences worth knowing:

| | Fiji GUI (§1–8) | `preprocess_cb.py` (§13) |
|---|---|---|
| Steps per volume | ~25 manual dialogs | one command |
| Six volumes | ~2 h of clicking | ~15 min unattended |
| Background subtraction | ImageJ rolling ball | `skimage` rolling ball, parallel over z — bit-identical to serial |
| Vesselness | Tubeness, 5 manual runs | `skimage.filters.sato`, chunked |
| Cross-scale max | `Image Calculator`, manual | automatic, per-scale normalised |
| Reproducibility | whatever you remember doing | `*_qc.json` records every parameter |
| Inspectability | native | `--save-tif` writes calibrated TIFFs to open in Fiji |

---

## 1. Phase 1 — Channel separation and metadata validation

1. **Open and inspect** the 2-channel hyperstack. `Image > Properties...` should read `Channels=2, Slices=435, Frames=1`, unit `µm`, pixel width/height `1.8660`, voxel depth `1.8639`. If channels/slices/frames are transposed, fix them **here** — Bleach Correction and Tubeness both behave differently on frames vs. slices.
2. **`Image > Color > Split Channels`.** Keep C1, the lectin/vascular channel — already extracted as `C1-*_vessels.tif`. C2 is the TH stain for glomus cells and is preprocessed separately; it is **not** part of this protocol's main path.
3. **Optional, low-cost: keep TH as an extra ilastik input channel.** Measured cross-channel correlation is 0.51 at mid-stack, so TH carries genuinely independent information, and structures bright in *both* channels are usually autofluorescent debris rather than lumen. Worth knowing: adding TH as ilastik channel 3 **does not change anything downstream** — ilastik still exports a single vessel probability map, so your skeletonisation pipeline never sees it and its "vessel stains only" assumption is not violated. The only costs are ~30 % more feature computation and having to keep the two channels co-registered. Park it as a tuning knob if debris rejection turns out to be the limiting error; ignore it otherwise.
4. **Filenames:** no spaces, parentheses, or multiple periods. (Your current directory has none — good.)

---

## 2. Phase 2 — Axial intensity correction (revised: usually skip)

**v4 was wrong here, and the damage is in your `DUP_` files.**

v4 prescribed `Image > Adjust > Bleach Correction` → `Histogram Matching`. Measured per-slice means in `C1-CB3-WKY-CB-A-2x2x2_vessels.tif`:

```
z:      0    40    80   120   160   200   240   280   320   360   400
mean: 303   462   559   744   901   934   911   813   658   508   367
```

That is a **symmetric hump**, not exponential decay. It reflects how much carotid body tissue intersects each slice, not photobleaching. Histogram Matching forces every slice to share one histogram, so it multiplied the near-empty end slices by roughly 6× and the dense mid slices by ~2×. In `DUP_C1-CB3-WKY-CB-A-2x2x2_vessels.tif` every slice now has mean ≈ 1500–1850 and **exactly 0.175 % of voxels clipped to 65535** — background noise at z ≈ 0 and z ≈ 434 has been promoted to vessel-level intensity.

### Corrected procedure

1. **Measure before correcting.** `Image > Stacks > Plot Z-axis Profile` on the raw channel.
   * **Monotonic decay** (deep slices dimmer, no rebound) → real attenuation, correct it.
   * **Hump / plateau / any non-monotonic shape** → tissue geometry, **do not correct**.
2. **Default for this dataset: no axial correction.** Proceed straight to Phase 3. The residual gradient is handled two ways, both better than flattening the data:
   * Per-slice rolling-ball subtraction (Phase 3) removes the additive component.
   * **Placing training labels at shallow, mid, and deep z** teaches the Random Forest the gradient directly. This is strictly more robust than destroying it beforehand, because the classifier keeps the depth cue rather than having it erased.
3. **If, and only if, the profile is genuinely monotonic**, use `Image > Adjust > Bleach Correction` → **`Simple Ratio`** (scales each slice by a single factor; preserves relative structure within a slice) and set the background parameter to the modal background of the darkest slice. **Never** `Histogram Matching` on a volume with varying tissue content.
4. Bleach Correction outputs a duplicate prefixed **`DUP_`** (v4 was correct on this). Close the original.

> **Action for your current files:** re-derive from `C1-CB3-WKY-CB-A-2x2x2_vessels.tif`, not from the `DUP_` files. The four existing `*_tubeness_s*.tif` volumes were computed from the clipped `DUP_` stack and should be regenerated.

---

## 3. Phase 3 — Rolling-ball background subtraction

Removes out-of-focus haze, uneven illumination, and connective-tissue autofluorescence. This is the one filtering step that is unambiguously worth doing.

1. `Process > Subtract Background...`
2. **`Rolling ball radius = 30.0`** pixels (= 56 µm). Larger than the 15 µm radius of your thickest arteriole, so lumen interiors are not eaten.
3. Leave **`Light background` unchecked**, **`Create background (don't subtract)` unchecked**, **`Disable smoothing` unchecked** (the default 3×3 smoothing is used only to *estimate* the background and makes it robust to shot noise).
4. Optionally check **`Sliding paraboloid`** — it handles smooth illumination gradients better than a true rolling ball and is a reasonable default for cleared tissue. Compare both on one slice with `Preview`.
5. Click **OK**. ImageJ then pops a separate **`Process Stack?`** dialog — click **`Yes`**. (v4 described a "Process entire stack" checkbox inside the dialog; there is no such checkbox.)

---

## 4. Phase 4 — Denoising (revised: omit, or use Remove Outliers)

**Do not run `Median 3D` with radius 1.** See §0: its 5.6 µm support exceeds a capillary diameter, so it thins and fragments exactly the structures you are segmenting. This was the most damaging routine step in v4.

Choose one:

* **Recommended — do nothing.** Hand the background-subtracted stack straight to Phase 5. Ilastik's Gaussian Smoothing features at σ = 0.3/0.7/1.0 provide denoising *as a feature*, reversibly, and the RF learns how much to trust it.
* **If you have genuine impulse noise** (isolated hot pixels, visible as single-voxel specks): `Process > Noise > Remove Outliers...`, `Radius = 1`, `Threshold = 50`, `Which outliers = Bright`. This replaces only voxels that deviate from the local median by more than the threshold, so tube interiors are untouched. Verify the threshold by previewing on one slice — it should remove specks and visibly nothing else.
* **Never** use a 3D median, a Gaussian with σ ≥ 1 px, or anisotropic diffusion here, at this voxel size.

---

## 5. Phase 5 — Global dynamic-range normalisation

Standardises intensity across WKY/SHR cohorts so one trained classifier transfers between volumes.

1. `Process > Enhance Contrast...`
2. **`Saturated pixels = 0.02`** (not 0.35). Vessels occupy only a few percent of voxels; saturating 0.35 % clips a large share of *vessel core* intensities to full scale, flattening the gradients the Hessian/edge features depend on. Your current `DUP_` file has 0.175 % of voxels pinned at 65535 for exactly this reason.
3. Check **`Normalize`**.
4. Check **`Process all slices`**.
5. **Check `Use stack histogram`.** ← *This option appears only after step 4 and was missing from v4.* Without it, every slice is normalised against **its own** histogram, which re-introduces slice-to-slice intensity jumps, breaks 3D feature continuity, and makes the z-derivative features meaningless. **This is mandatory for 3D work.**
6. Leave `Equalize histogram` **unchecked** (it destroys intensity linearity).

Result: a 16-bit volume spanning [0, 65535] with a single global mapping and negligible clipping.

> **32-bit caveat:** `Normalize` on a 32-bit image rescales to **[0.0, 1.0]**, not [0, 65535] (v4 §5 claimed otherwise). This matters in Phase 6.

---

## 6. Phase 6 — Multiscale vesselness (rewritten; v4's procedure was non-functional)

### 6.0 Why v4's version could not work

v4 asked you to run Tubeness at four sigmas, then:

* `Image > Stacks > Images to Stack` — this concatenates *slices*. Four 435-slice stacks become one 1740-slice stack, **not** a 4-element scale axis.
* `Z Project`, start 1, stop 4 — projects the first four slices, producing **one 2D image** from a 435-slice volume.

Neither step produces a multiscale map. Use §6.2 instead.

### 6.1 A second problem v4 missed: scales are not comparable

Fiji's Tubeness does not γ-normalise across σ. Measured 99th percentiles on your existing outputs (slice 200):

| σ (px) | p99 response |
|---|---|
| 1.0 | 1443 |
| 2.0 | ~2000 |
| 8.0 | 4238 |

A plain voxelwise `Max` is therefore **dominated by σ = 8 nearly everywhere**, which erases the capillary-scale selectivity that was the entire point of the channel. **Each scale map must be normalised to a common range before combining.**

### 6.2 Generating the scale maps

1. Start from the Phase 5 normalised grayscale stack. Rename it for clarity: `Image > Rename...` → `GRAY`.
2. Locate Tubeness. It is a **plugin**, at `Plugins > Process > Tubeness` — *not* `Process > Filters > Tubeness` as v4 stated. If the path differs in your build, press **`L`** (`Help > Find Commands`) and type `Tubeness`.
3. For each σ, select `GRAY`, run Tubeness, and rename the 32-bit output.

   **`Use calibration information`:**
   * **Unchecked** → σ is in **pixels**. 
   * **Checked** → σ is in **µm**.
   
   Either is fine as long as you are consistent; v4's rationale ("unchecking forces Fiji to use our calibrated pixel-to-voxel ratios") is backwards and meaningless. **Use unchecked / pixels**, which is what your existing files used — I verified this from their lateral autocorrelation length (~2 px at σ=1, ~9 px at σ=8), consistent with pixel-unit sigmas.

   | σ (px) | σ (µm) | Targets diameters of… |
   |---|---|---|
   | 1.0 | 1.87 | 3–5 µm — finest capillaries |
   | 1.4 | 2.61 | 5–7 µm — typical capillaries *(add this; the 2× spacing in v4 under-samples the capillary range)* |
   | 2.0 | 3.73 | 7–10 µm — post-capillary venules |
   | 4.0 | 7.46 | 12–20 µm |
   | 8.0 | 14.93 | 25–35 µm — feeding arterioles |

   Rename each output to `TUBE_s1.0`, `TUBE_s1.4`, `TUBE_s2.0`, `TUBE_s4.0`, `TUBE_s8.0`.

   > Your existing σ = 1, 2, 4, 8 set is usable if you don't want to recompute — but regenerate it from the corrected Phase 5 stack, not from the clipped `DUP_` volume (§2).

### 6.3 Per-scale normalisation (new — do not skip)

For **each** `TUBE_s*` window:

1. `Process > Enhance Contrast...`, `Saturated pixels = 0.05`, check `Normalize`, `Process all slices`, **`Use stack histogram`**, OK.
2. The image is 32-bit, so this maps it to **[0.0, 1.0]** — that is what we want, and it is now directly comparable to the other scales.

### 6.4 Combining scales — the correct Fiji operation

Use `Process > Image Calculator...`, which operates voxelwise on two stacks of equal size.

Rather than one flat max, produce **two** vesselness channels. Capillaries and arterioles have opposite requirements, and letting the classifier weight them separately outperforms collapsing them:

**Channel `VESS_FINE`** (capillary scale):
```
Image Calculator:  TUBE_s1.0   Max   TUBE_s1.4   → "Create new window" ✔, "32-bit (float) result" ✔  → tmp
Image Calculator:  tmp         Max   TUBE_s2.0   → ...                                                → VESS_FINE
```

**Channel `VESS_COARSE`** (arteriole scale):
```
Image Calculator:  TUBE_s4.0   Max   TUBE_s8.0   → "Create new window" ✔, "32-bit (float) result" ✔  → VESS_COARSE
```

Each `Image Calculator` run on stacks pops a **`Process all N images?`** dialog — answer **Yes**.

> If you prefer a single vesselness channel, take the Max of all five normalised maps into one `VESS` volume. It is simpler and still correct now that §6.3 has equalised the scales; it is just less informative.

### 6.5 Keep them 32-bit

**Do not convert to 16-bit.** Ilastik reads float32 natively. v4's convert-to-16-bit step invites a silent scaling error, because `Image > Type > 16-bit` rescales using the *current display range* when `Edit > Options > Conversions... > Scale when converting` is enabled, which is not necessarily the data range.

Save each as TIFF: `GRAY.tif`, `VESS_FINE.tif`, `VESS_COARSE.tif` (and `C2.tif` if using it, normalised the same way as `GRAY`).

---

## 7. Phase 7 — Tiling (revised: don't)

**v4's tiling instructions should not be followed for this dataset.**

Your largest volume is 507 × 456 × 435 × 16-bit = **201 MB per channel**; four channels as float32 is ~1.6 GB. Ilastik handles this comfortably on any machine with ≥ 16 GB RAM, and its prediction is internally blockwise regardless.

Tiling actively hurts you:

* **Seam artifacts.** Ilastik features at σ = 5–10 have a support of ±15–30 voxels. Near a tile edge these are computed against zero-padding, so the classifier sees different evidence there — producing a visible probability discontinuity on every tile boundary.
* **Normalisation drift.** The v4 macro normalises **per tile** (`fiji-batch-preprocess.ijm:160`), which directly contradicts the document's own Phase 5 global normalisation. Per-tile normalisation gives every tile a different intensity scale, so intensity features become incomparable across tiles and one classifier cannot serve them all.
* **Broken topology.** Skeletonising per tile and merging afterwards fragments the graph at exactly the seams — fatal for a connectivity-based flow model.

**At 0.933 µm this changes.** 805 M voxels × 3 float32 channels is 9.7 GB before ilastik computes a single feature, on a 31 GB machine. Escalate in this order — the first two are usually enough, and both preserve a single continuous volume:

1. **Lower ilastik's resource limits** (`Settings > Preferences`): memory cap ~20 GB, threads ~8. The usual OOM cause is 20 threads each holding a per-block feature stack, not the volume itself.
2. **Predict headless** rather than from the GUI, which avoids holding the interactive caches alongside the prediction:
   ```bash
   run_ilastik.sh --headless \
       --project=cb_vessels.ilp \
       --export_source="Probabilities" \
       --output_format=hdf5 \
       --output_filename_format="{dataset_dir}/{nickname}_Probabilities.h5" \
       --export_dtype=float32 \
       CB3-WKY-A_ilastik.h5
   ```
3. **Drop feature scales before you drop resolution.** Removing σ = 10.0, then σ = 5.0, then the Structure Tensor family cuts memory roughly linearly and costs far less accuracy than tiling does.
4. **Only then tile**, with **≥ 64 voxel overlap** at 0.933 µm (≥ 2× your largest feature σ in pixels). `Edit > Selection > Specify...` sets **Width, Height, X coordinate, Y coordinate** only — there is no `Slice` or `Stack` field as v4 claimed; full z-depth comes from `Image > Duplicate...` with **`Duplicate stack`** checked and the **Range** left at the full extent.
5. Predict per tile, then **discard the overlap margins and stitch the probability maps back into one volume before thresholding**. Skeletonise only the stitched volume. v4 omitted both the overlap and the stitching, which is what makes its tiling scheme unusable for graph extraction.

---

## 8. Building the Ilastik input

### 8.1 Channel layout

| Channel | Content | Role |
|---|---|---|
| 0 | `GRAY` — background-subtracted, globally normalised lectin | Intensity + boundary evidence |
| 1 | `VESS_FINE` — normalised max over σ = 1.0–2.0 px | Capillary-scale tubular prior |
| 2 | `VESS_COARSE` — normalised max over σ = 4.0–8.0 px | Arteriole-scale tubular prior |
| 3 | `C2` *(optional)* — second acquisition channel, same normalisation | Negative evidence: bright in both channels ⇒ debris/bleed-through, not lumen |

Minimum viable configuration is channels 0 + 1.

### 8.2 Writing the HDF5

**v4's `convert_tiles_to_hdf5.py` does not exist**, and its `--axes ZCYXS` argument is invalid — `S` is not an ilastik axis, and ilastik axis keys are **lowercase** from `{t, c, z, y, x}`. For a `(Z, Y, X, C)` array the correct tag string is **`zyxc`**.

A working script is provided alongside this document as **`make_ilastik_input.py`** (§11). Run:

```bash
python3 make_ilastik_input.py \
    --gray GRAY.tif \
    --vess VESS_FINE.tif VESS_COARSE.tif \
    --out  CB3-WKY-A_ilastik.h5
```

It writes `/data` with shape `(435, 456, 507, C)`, dtype float32, correct `axistags`, and gzip compression.

**Fiji alternative (single channel only):** install the `ilastik` update site, then `Plugins > ilastik > Export HDF5`. This writes valid axistags but does not build multi-channel stacks conveniently. **Avoid** `Image > Color > Merge Channels...` → save TIFF as an intermediate: composite mode reorders axes and can silently apply LUTs.

---

## 9. Ilastik configuration

### 9.1 Feature selection — v4's per-channel table is not buildable

> **Ilastik applies the *same* feature/σ set to *every* input channel.** There is no per-channel configuration, so v4 §4's table (different sigmas for Channel 1 vs Channel 2, 2D for some and 3D for others) cannot be entered into the GUI. Likewise, the 2D-vs-3D choice is a property of the feature set as a whole, not of individual features — you cannot mix them as v4 prescribed.

**Use 3D features throughout.** 2D-per-slice features produce z-anisotropic probability maps, which cause staircase artifacts in the skeleton and systematically biased z-connectivity — unacceptable for a topology-driven flow model.

Selected features (ilastik's fixed σ grid is 0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0):

| Feature | 0.3 | 0.7 | 1.0 | 1.6 | 3.5 | 5.0 | 10.0 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Gaussian Smoothing | ✔ | ✔ | ✔ | ✔ | ✔ | | |
| Laplacian of Gaussian | | ✔ | ✔ | ✔ | ✔ | | |
| Gaussian Gradient Magnitude | | ✔ | ✔ | ✔ | ✔ | | |
| Difference of Gaussians | | ✔ | ✔ | ✔ | | | |
| Structure Tensor Eigenvalues | | ✔ | ✔ | ✔ | | | |
| Hessian of Gaussian Eigenvalues | | ✔ | ✔ | ✔ | ✔ | | |

Rationale at 1.866 µm/voxel:

* **σ = 0.7 px (1.31 µm)** — the capillary wall itself.
* **σ = 1.0 px (1.87 µm)** — capillary radius; the single most important scale here.
* **σ = 1.6 px (2.99 µm)** — whole capillary cross-section; enforces tubular continuity.
* **σ = 3.5 px (6.5 µm)** — arteriole walls.
* **σ = 0.3 px** is below the Nyquist limit of your sampling — it fits noise. Only useful on Gaussian Smoothing as a near-identity feature; do not enable it on any derivative feature.
* **σ = 5.0 and 10.0** (9.3 and 18.7 µm) mostly encode tissue-block-scale context, cost the most RAM, and dilute the RF. Add σ = 5.0 on Gaussian Smoothing only if arterioles are being missed.

That is ~26 features per channel. With 3 channels ≈ 78 features — near the practical ceiling. If prediction is slow, drop Structure Tensor first (it overlaps heavily with Hessian).

### 9.2 Label classes

Two classes:

1. **Vessel** — short strokes down the lumen core. In dense regions label the *core*, not the wall.
2. **Background** — dark tissue, autofluorescent debris, and, critically, **the gaps between adjacent capillaries**.

Labelling discipline that matters more than any parameter above:

* **Label across depth.** Put labels at z ≈ 40, 200, and 380 minimum. This is what makes the classifier robust to the axial gradient you deliberately did *not* flatten in §2.
* **Label the failure modes, not the easy cases.** After the first Live Update, find where the prediction is wrong and label *there*. Twenty corrective strokes beat two hundred confirmatory ones.
* **Include the stack ends.** z ≈ 0–20 and z ≈ 415–434 are sparse and noisy; label background there explicitly or the classifier will hallucinate vessels.
* **Don't over-invest.** Per §0.2, these labels will not transfer to the 0.933 µm data. Label enough to validate the pipeline, not enough to publish.

### 9.2.1 One classifier for all six volumes

You have 3 WKY + 3 SHR at matched acquisition settings. Add **all six as separate datasets in one ilastik project** (`Input Data` tab → add each file) and label across all of them, rather than training six classifiers or training on one and batch-applying.

* This works only because Phase 5 normalises each volume independently against its own stack histogram. That per-volume normalisation is what puts all six on a common intensity scale — it is the step that makes cohort-wide transfer legitimate, so do not skip or vary it.
* **Label both cohorts, and label SHR more.** SHR carotid bodies are hypervascular, so the parallel-capillary merge case of §9.3 is denser and harder there. A classifier trained mostly on WKY will systematically over-merge SHR capillaries — which would manifest as a spurious "fewer, thicker vessels in SHR" result. That is a cohort-correlated artifact that looks exactly like a biological finding, so it is worth guarding against explicitly.
* **Sanity check before batch processing:** vessel volume fraction should be broadly comparable across the three volumes within each cohort. A volume that is a large outlier usually indicates a staining or normalisation problem, not biology.

### 9.3 The parallel-vessel merge problem

Two capillaries 2 voxels apart with a 1-voxel gap are the hardest case in this dataset, and vesselness filters are structurally prone to merging them (a pair of adjacent tubes looks like one thicker tube to a Hessian at σ ≥ 2).

* Draw **Background** labels *in the gap*, on the grayscale channel, at several z positions.
* This is precisely why `VESS_FINE` and `VESS_COARSE` are separate channels: in a merged pair, `VESS_FINE` shows two ridges while `VESS_COARSE` shows one blob. That disagreement is a *learnable feature*, and the RF will use it. Collapsing them into a single max channel throws it away.

### 9.4 Prediction export

1. `Prediction Export` tab → **`Source: Probabilities`**.
2. `Choose Export Image Settings...`:
   * **Transformations → Subregion:** set the **channel (`c`) range to the Vessel class index only** (`0` to `1` if Vessel is your first class). **v4 omitted this.** By default ilastik exports *all* classes, so the output has 2 channels; thresholding channel 1 instead of channel 0 silently gives you the inverse segmentation.
   * **Convert to Data Type: `float32`.** Not 8-bit. v4 argued (correctly) that probability gradients matter for downstream thresholding and repair, then immediately quantised them to 256 levels. Keep float32 — your volumes are small and §10 depends on the gradients.
   * **Renormalize:** leave **off**. Probabilities are already in **[0.0, 1.0]** — v4's claim of a "[1.0, 2.0]" raw range is simply wrong.
   * **Format: `hdf5`**, dataset name `exported_data`.
3. `Batch Processing` tab → add the other volumes → `Process all files`.

---

## 10. Probability map → clean binary mask (Python)

Preprocessing through §9 happens in the Fiji GUI so outputs stay inspectable. From the ilastik export onwards everything is Python, feeding an existing skeletonisation pipeline. `prob_to_mask.py` (§11) implements this whole section; the reasoning is here so the parameters can be tuned rather than trusted.

The handoff contract is: **this section produces a boolean mask and a calibrated EDT in µm. Skeletonisation, graph assembly, and flow solving happen downstream.**

### 10.1 Load and pick the right channel

```python
with h5py.File("CB3-WKY-A_Probabilities.h5") as f:
    arr = np.squeeze(f["exported_data"][...])   # (z, y, x, c)
prob = arr[..., 0]                              # Vessel class
```

If §9.4's channel subregion was set, `c == 1` and this is unambiguous. If not, `c == 2` and **channel 1 is the Background class** — silently thresholding it yields the exact inverse of your segmentation. `prob_to_mask.py` prints the channel count and the chosen index for this reason.

### 10.2 Hysteresis thresholding (replaces v4's single cutoff + morphological repair)

A single global cutoff at 0.5 forces a choice between fragmenting faint capillaries (raise it) and fusing parallel vessels (lower it). Hysteresis avoids the trade: seed on confident voxels, then grow into weak voxels **only where they connect to a seed**.

```python
mask = skimage.filters.apply_hysteresis_threshold(prob, low=0.30, high=0.70)
```

Choose `low` from evidence rather than by default. `prob_to_mask.py --sweep` prints vessel fraction, component count, and the share of vessel voxels in the largest component across a threshold grid. **The right `low` is just above where component count starts climbing steeply and the largest component's share starts falling** — that is the point at which real capillary segments begin to be stranded from the network.

**This replaces v4 §6.3 entirely.** That sequence — `Dilate(1) → Median 3D(2,2,1) → Erode(1)` — applies a majority-vote median to a **binary** mask. On a 2–3-voxel capillary the majority of every 5×5×3 neighbourhood is background, so it *deletes* capillaries rather than bridging gaps. It removes far more than it repairs.

### 10.3 Cavity filling and debris removal

```python
mask = scipy.ndimage.binary_fill_holes(mask)   # 3D — closes cavities only
```

Lectin labels endothelium, so larger vessels appear as rings in cross-section; a 3D fill makes them solid, which is what the EDT needs.

> **Never fill holes slice-by-slice** (`Process > Binary > Fill Holes` in Fiji, or 2D `binary_fill_holes` in a loop). That fills the interior of every in-plane vascular *loop*, silently destroying exactly the anastomotic topology your flow model depends on. A true 3D fill only closes enclosed cavities, which is safe.

Then drop components below ~**50 voxels** (≈ 325 µm³ at 1.866 µm — smaller than the shortest plausible capillary segment, larger than typical debris). If a large fraction of components is removed, `low` is too permissive; go back to §10.2.

### 10.4 Calibrated EDT

```python
edt = scipy.ndimage.distance_transform_edt(
    mask, sampling=(1.8639, 1.8660, 1.8660)   # z, y, x in um
).astype(np.float32)
```

`sampling=` makes the output **physical µm directly** and absorbs the 1.0011 axial:lateral ratio. v4 omitted the µm conversion entirely.

### 10.5 Handing off to skeletonisation

```python
mask = np.load("CB3-WKY-A_mask.npy")     # bool
edt  = np.load("CB3-WKY-A_edt_um.npy")   # float32, um

skeleton = skimage.morphology.skeletonize(mask)   # your pipeline
radii    = edt[skeleton]                          # sample, do NOT multiply
```

Four things to check on the receiving side:

1. **Sample the EDT, don't multiply by it.** v4's "multiply the EDT map with your skeleton" assumes a 0/1 skeleton. Fiji skeletons are 0/255 and `skimage.skeletonize` returns bool — multiplying gives either a 255× inflation or a silently correct result depending on dtype. Boolean indexing is unambiguous.
2. **Trim junction neighbourhoods before averaging radii.** Within ~1 radius of a bifurcation the EDT reports the *junction's* inscribed sphere, not the vessel's, biasing radii upward. Discard skeleton voxels within 2 voxels of any junction.
3. **Use path length, not euclidean length, for resistance.** Euclidean endpoint separation is for tortuosity only.
4. **Do not prune anastomotic loops.** If your pipeline uses Fiji's `Analyze Skeleton`, note that its `Prune cycle method` dropdown resolves **loops**, not spurs (v4 described it as spur removal), and should be set to `none` — vascular loops are real anatomy and removing them corrupts flow topology. Likewise leave `Prune ends` off; it deletes genuine capillary terminals. Remove spurs by length instead. And be aware `Analyze Skeleton` does **not** skeletonise — `Skeletonize (2D/3D)` must run first.

### 10.6 Validation before you simulate

Run these on the graph, per volume, before any haemodynamic result is believed:

| Check | Expectation | A failure means |
|---|---|---|
| Capillary diameter mode | 4–7 µm | Threshold too low (thick) or too high (thin) |
| Murray's law, Σ*r*³ at bifurcations | approximately conserved | Radii biased at junctions — see §10.5.2 |
| Largest connected component | should dominate vessel voxels | `low` too high, network fragmented |
| Vessel volume fraction, across the 3 volumes in a cohort | broadly comparable | Staining or normalisation outlier |
| Endpoint count | modest | Large numbers of dead ends = fragmentation, not anatomy |

Given §0.1, expect the diameter mode to sit near a multiple of 1.87 µm at this resolution. That is quantisation, not biology.

---

## 11. Companion scripts

All three live in this directory and have been verified against your data. **§13 is the operational guide**; this section is the reference.

| Script | Role | Needed if you work in Python? |
|---|---|---|
| **`preprocess_cb.py`** | raw TIFF → ilastik HDF5. Implements §1–8 end-to-end. | **Yes — this is the main entry point** |
| `prob_to_mask.py` | ilastik probabilities → mask + calibrated EDT. Implements §10. | Yes |
| `make_ilastik_input.py` | Fiji-preprocessed TIFFs → ilastik HDF5. | No — superseded by `preprocess_cb.py`. Kept only as a bridge if you ever preprocess a one-off in Fiji |

### `preprocess_cb.py` — raw TIFF → ilastik input

Replaces the entire Fiji GUI route. Full usage, expected output, and timings in **§13**.

```bash
python3 preprocess_cb.py --input raw/ --diagnose                       # look first
python3 preprocess_cb.py --input raw/ --output-dir ilastik_inputs      # then process
```

Three implementation details worth knowing, since they are what make it viable at this scale:

* **Rolling ball runs parallel over z**, one process per slice. Verified bit-identical to the serial result; ~12× wall-clock on 16 workers (36 min → 3 min for six volumes).
* **The ridge filter runs in z-chunks with a 4σ halo.** A monolithic `sato` call peaks at ~17.6 GB on a 100 M-voxel volume, which would not survive the move to 0.933 µm. Chunked, memory scales with `--chunk-z`. Verified identical to monolithic (correlation 1.000000).
* **Coarse scales are computed at half resolution** and upsampled (`--fast-coarse`, on by default). 8× faster, correlation 0.9989 against full resolution — safe because a σ = 8 ridge response varies slowly by construction.

### `make_ilastik_input.py` — Fiji output → ilastik input *(legacy)*

Handles what the Fiji GUI does badly: per-scale normalisation before the cross-scale max (§6.3–6.4) and HDF5 with correct lowercase `axistags` (§8.2). Writes channels one at a time so peak RAM stays near the final file size, which matters at 0.933 µm (§0.3).

```bash
# From Fiji-generated tubeness maps (the GUI path, §6):
python3 make_ilastik_input.py --gray GRAY.tif \
        --vess VESS_FINE.tif VESS_COARSE.tif --out CB3-WKY-A_ilastik.h5

# Skip five manual Tubeness runs and compute vesselness in Python instead:
python3 make_ilastik_input.py --gray GRAY.tif --compute-vesselness \
        --out CB3-WKY-A_ilastik.h5

# Add the TH channel (§1.3), or upsample x2 (§0.1):
python3 make_ilastik_input.py --gray GRAY.tif --vess VESS_FINE.tif VESS_COARSE.tif \
        --extra TH.tif --upsample 2 --out CB3-WKY-A_ilastik.h5
```

### `prob_to_mask.py` — ilastik output → mask + EDT

Implements §10. Stops at the mask/EDT boundary; skeletonisation is left to your existing pipeline.

```bash
# 1. Choose thresholds on evidence, not defaults:
python3 prob_to_mask.py --prob CB3-WKY-A_Probabilities.h5 --sweep

# 2. Produce the mask and calibrated EDT:
python3 prob_to_mask.py --prob CB3-WKY-A_Probabilities.h5 \
        --high 0.70 --low 0.30 --min-size 50 \
        --out-prefix CB3-WKY-A --save-tif

# Halve the radius quantisation step at 1.866 um (8x RAM/time):
python3 prob_to_mask.py --prob ... --refine-radii --out-prefix CB3-WKY-A
```

`--save-tif` writes mask and EDT TIFFs so intermediate results stay inspectable in Fiji, matching how the rest of the pipeline is being run.

---

## 12. Change log versus v4

| v4 location | Problem | v5 resolution |
|---|---|---|
| §1 | "sliding window length *L*" conflated radius with diameter | §0.2 — restated as a radius criterion |
| §1, throughout | Bracketed citations reference no bibliography | Removed; claims justified from your data or from the named tool's documented behaviour |
| Phase 2 | `Histogram Matching` on a hump-shaped z-profile fabricates signal at stack ends | §2 — measure first; default is no correction; `Simple Ratio` if truly monotonic |
| Phase 3 | "Process entire stack checkbox" does not exist | §3 — it is a separate `Process Stack?` prompt |
| Phase 4 | `Median 3D` r=1 destroys 2–4-voxel capillaries | §4 — omit denoising, or `Remove Outliers` |
| Phase 5 | Missing `Use stack histogram`; per-slice normalisation | §5 — mandatory checkbox added |
| Phase 5 | 0.35 % saturation clips vessel cores | §5 — reduced to 0.02 % |
| Phase 5 | Claimed 32-bit normalises to [0, 65535] | §5 — it is [0.0, 1.0] |
| Phase 6.2 | Wrong menu path; incoherent `Use calibration` rationale | §6.2 — corrected path and unit semantics |
| Phase 6.3–6.4 | `Images to Stack` + `Z Project` cannot produce a multiscale map; yields a 2D image | §6.4 — `Image Calculator` with `Max`, stackwise |
| Phase 6.4 | Un-normalised max is dominated by σ=8 | §6.3 — per-scale normalisation before combining |
| Phase 6.5 | 32-bit → 16-bit conversion invites display-range scaling errors | §6.5 — keep float32 |
| Phase 7 | Tiling a 201 MB volume; no overlap; no stitching | §7 — don't tile; if you must, overlap ≥ 32 vx and stitch before skeletonising |
| Phase 7.3 | `Specify...` has no `Slice`/`Stack` fields | §7 — z-range set in `Duplicate` |
| §3 | `convert_tiles_to_hdf5.py` does not exist | §11 — real script provided |
| §3 | `--axes ZCYXS`; `S` is not an ilastik axis | §8.2 — lowercase `zyxc` |
| §4 | Per-channel feature tables are not configurable in ilastik | §9.1 — one feature set for all channels |
| §4 | Mixed 2D/3D features | §9.1 — 3D throughout |
| §5 | "probabilities from [1.0, 2.0]" | §9.4 — probabilities are [0.0, 1.0]; no renormalisation |
| §5 | No instruction to select the vessel class channel on export | §9.4 — set the `c` subregion |
| §5 / §6 | Exports 8-bit while arguing gradients must be preserved | §9.4 — export float32 |
| §6.1 | Single global threshold at 128 | §10.1 — hysteresis thresholding |
| §6.2 | `3D Fill Holes` without warning about 2D variant | §10.2 — 3D only; 2D destroys loop topology |
| §6.3 | Binary `Dilate → Median → Erode` deletes capillaries | §10.1 — replaced by hysteresis |
| §6.4 | `Analyze Skeleton` does not skeletonise; "prune shortest branch" mischaracterised | §10.4 — `Skeletonize (2D/3D)` first; cycle pruning off |
| §6.5 | "Multiply EDT by skeleton" — 255× error; no µm conversion | §10.5 — sample, don't multiply; convert to µm |
| — | No discussion of radius quantisation vs. *r*⁻⁴ resistance | §0.1 — added |
| — | No path from binned iteration data to the unbinned production run | §0.2 — parameter transfer table; labels and classifier do not carry over |
| — | No RAM planning | §0.3 — added for 31 GB / 20 cores |
| — | No cohort strategy | §9.2.1 — one project, all six volumes, weighted toward SHR |
| — | Fiji-only, ~25 manual dialogs per volume × 6 | §13 — `preprocess_cb.py` runs §1–8 in one command, ~15 min unattended |
| `fiji-batch-preprocess.ijm:160` | Per-tile normalisation contradicts the document's own global Phase 5 | §7 — tiling removed; macro retired (see below) |

## 13. The Python pipeline — step by step

Three scripts, run in order. Fiji is not required at any point, though `--save-tif` keeps everything openable in Fiji for inspection.

```
raw TIFF ──[preprocess_cb.py]──▶ *_ilastik.h5 ──[ilastik GUI]──▶ *_Probabilities.h5 ──[prob_to_mask.py]──▶ *_mask.npy
                                                                                                          *_edt_um.npy
                                                                                                              │
                                                                                            your skeletonisation pipeline
```

### 13.0 Prerequisites

```bash
python3 -c "import numpy, scipy, skimage, tifffile, h5py; print('ok')"
```

All five are already present on this machine (`skimage` 0.26, `scipy` 1.17). Ilastik itself is a separate download — the pixel classification GUI is used interactively for labelling, then headless for batch prediction.

### 13.1 Organise the data

The scripts take files, directories, or globs, so no reorganisation is strictly needed. But since you have 3 WKY + 3 SHR that must share one classifier, put them somewhere they can be processed in one command:

```
CB3/
├── raw/                 all 6 raw TIFFs (WKY + SHR)
├── ilastik_inputs/      created by step 1
└── segmentation/        created by step 3
```

Everything below assumes you run from `CB3/`. Adjust paths freely — nothing is hard-coded.

---

### Step 1 — Diagnose before processing

**Always run this first.** It reads the volumes, reports the axial intensity profile, and stops. Seconds, no writes.

```bash
python3 preprocess_cb.py --input raw/ --diagnose
```

**Expected output** (verified on your three WKY volumes):

```
======================================================================
CB3-WKY-CB-A-2x2x2
======================================================================
  [1/6] reading + channel extraction
        axes=ZCYX channels=2 -> vessel volume (435, 456, 507)
  [2/6] z-profile diagnosis
        verdict: hump / tissue-extent dominated (peak at slice 230/435)
        Do NOT apply bleach correction. This shape is the tissue
        block, not photobleaching. Label at several depths in ilastik
        instead.
```

**How to read it:**

| Verdict | Meaning | Action |
|---|---|---|
| `hump / tissue-extent dominated` | Peak intensity is mid-stack — the tissue block, not bleaching | Nothing. Proceed. This is what all three WKY volumes report. |
| `monotonic decay` | Peak at the top, falling with depth — real attenuation | Consider a multiplicative correction. Never histogram matching (§2). |
| `inverse decay (rises with z)` | Peak at the bottom | Check stack orientation before doing anything else |

Also confirm `axes=ZCYX channels=2` and that the vessel volume shape is `(435, Y, X)`. If channels and slices are transposed, fix that first — everything downstream depends on it.

---

### Step 2 — Preprocess

Run one volume first to check timing and output, then the batch.

```bash
# one volume, with inspection TIFFs
python3 preprocess_cb.py --input raw/CB3-WKY-CB-C-2x2x2.tif \
        --output-dir ilastik_inputs --save-tif --workers 8

# then all six
python3 preprocess_cb.py --input raw/ --output-dir ilastik_inputs --workers 8
```

**What it does, in order** — each maps to a section above:

| Stage | Section | Default | Note |
|---|---|---|---|
| 1. Read + extract vessel channel | §1 | `--channel 0` | C1 = lectin. Handles ZCYX/CZYX/ZYX automatically |
| 2. Z-profile diagnosis | §2 | always | Reported, never auto-corrected |
| 3. Rolling-ball background subtraction | §3 | `--rolling-ball 30` | Per-slice, parallel over z |
| 4. Impulse-noise removal | §4 | **off** | `--remove-outliers 0`. Deliberately off — see §4 |
| 5. Global normalisation | §5 | `--saturated 0.02` | Whole-stack histogram, → [0, 1] |
| 6. Multiscale vesselness | §6 | fine `1.0 1.4 2.0`, coarse `4.0 8.0` | Per-scale normalised, then max |
| 7. Write HDF5 + QC | §8 | | `zyxc` axistags, float32 |

**Expected console output** (measured, volume C):

```
  [3/6] rolling-ball background subtraction (radius=30 px = 56.0 um)
        done in 11.9 s
  [4/6] denoising SKIPPED (recommended -- see protocol section 4)
  [5/6] global normalisation (saturated=0.02%, stack histogram)
        anchors [0.0, 8145.2] -> [0, 1]
  [6/6] multiscale vesselness
    fine scales [1.0, 1.4, 2.0] px
      sigma=1.00  ( 1.87 um) done
      ...
        vesselness done in 41.9 s

  -> ilastik_inputs/CB3-WKY-CB-C-2x2x2_ilastik.h5
     shape (435, 315, 255, 3)  float32  axistags zyxc
     channel 0: grayscale
     channel 1: vesselness_fine
     channel 2: vesselness_coarse
     total 75.6 s
```

**Timings and memory, measured on this machine** (31 GB, 20 cores, default settings):

| Volume | Voxels | Time | Peak RAM |
|---|---|---|---|
| C (315×255×435) | 35 M | **88 s** | 5 GB (8 workers) |
| A (507×456×435) | 101 M | **4.6 min** | **12.2 GB** (5 workers) |
| All six | ~435 M | **~20 min unattended** | 12 GB |

#### Memory is auto-managed — this matters

The ridge filter, not the volume, is what exhausts RAM: `skimage`'s `sato` holds roughly **160 bytes of working set per voxel in flight** (Hessian elements and eigenvalue intermediates, several in float64). Peak usage is therefore

```
peak ≈ parent + workers × (chunk_z + 2 × 4σ_max) × Y × X × 160 bytes
```

`--max-memory-gb` (default **12**) inverts that formula and reduces the worker count to fit. You'll see the plan printed:

```
memory plan: budget 12 GB = 1.6 GB parent + 5 x 1.8 GB workers (chunk_z=32)
reduced workers 8 -> 5 to stay in budget; raise --max-memory-gb or lower --chunk-z for speed
```

Measured against the model on volume A — the prediction is accurate to within 2 %:

| Setting | Predicted | Measured peak | Time |
|---|---|---|---|
| `--workers 8 --chunk-z 48` | 21 GB | **19.4 GB** | 264 s |
| default (budget 12 GB → 5 workers) | 12 GB | **12.2 GB** | 280 s |
| `--workers 4 --chunk-z 24` | 8 GB | **9.1 GB** | 373 s |

The default trades **6 % of speed for 7 GB of headroom** versus running 8 workers flat out — worth it, since 19.4 GB of 31 GB leaves nothing for anything else you have open. Raise `--max-memory-gb 20` if the machine is otherwise idle.

### 13.2 What you get

Per volume, in `--output-dir`:

| File | Always? | Contents |
|---|---|---|
| `<base>_ilastik.h5` | yes | **The pipeline input.** `/data`, shape `(z, y, x, c)`, float32, `axistags` `zyxc` |
| `<base>_qc.json` | yes | Every parameter used, volume shape, z-profile verdict, normalisation anchors, elapsed time |
| `<base>_grayscale.tif` | `--save-tif` | Preprocessed intensity channel, µm-calibrated |
| `<base>_vesselness_fine.tif` | `--save-tif` | Capillary-scale ridge response |
| `<base>_vesselness_coarse.tif` | `--save-tif` | Arteriole-scale ridge response |

All channels are in **[0, 1]**. Sanity values from volume C: grayscale mean 0.068, fine 0.024, coarse 0.043, with 0.4–0.7 % of voxels above 0.5. Vesselness channels should be **sparse** — if `frac>0.5` is more than a few percent, the background subtraction is under-doing it.

`<base>_qc.json` is the reproducibility record. Keep it: with six volumes and a shared classifier, "which parameters produced this input" becomes unanswerable within a week otherwise.

### 13.3 Step 3 — Inspect before labelling

Ten minutes here saves hours of labelling against a bad input.

```bash
# open in Fiji if you like, or:
python3 -c "
import h5py, numpy as np
d = h5py.File('ilastik_inputs/CB3-WKY-CB-A-2x2x2_ilastik.h5')['data']
for c in range(d.shape[3]):
    a = d[::4, ..., c]
    print(f'ch{c}: mean={a.mean():.4f}  frac>0.5={100*(a>0.5).mean():.2f}%')
"
```

Check, on a mid-stack slice:

* **Grayscale** — vessels bright, background near zero, no large-scale gradient left. If haze remains, lower `--rolling-ball`; if vessel cores look hollowed, raise it.
* **`vesselness_fine`** — thin capillaries lit up as continuous ridges. This channel carries most of the segmentation signal.
* **`vesselness_coarse`** — only the large feeding vessels. If it lights up capillaries too, your coarse sigmas are too small.
* **Stack ends** (z ≈ 0–20, z ≈ 415–434) — should be mostly empty. Bright noise there means the normalisation anchors were pulled by an outlier.

---

### Step 4 — Ilastik

Unchanged from §9; the inputs are just built differently now.

1. New project → **Pixel Classification**.
2. `Input Data` → add **all six** `_ilastik.h5` files. Ilastik should read the axes as `zyxc` with 3 channels. If it guesses wrong, set them manually in the dialog.
3. `Feature Selection` → the matrix in §9.1. **3D throughout**; the same set applies to all channels — per-channel features are not configurable.
4. `Training` → two classes, labelling discipline in §9.2 and the cohort strategy in §9.2.1. **Label SHR at least as heavily as WKY.**
5. `Prediction Export` → §9.4. The two settings people get wrong: set the **channel subregion to the Vessel class only**, and export **float32**, not 8-bit.
6. `Batch Processing` → all six volumes.

Output: `<base>_ilastik_Probabilities.h5` per volume.

---

### Step 5 — Probability map → mask

```bash
# choose thresholds on evidence
python3 prob_to_mask.py --prob ilastik_inputs/CB3-WKY-CB-A-2x2x2_ilastik_Probabilities.h5 --sweep

# then produce mask + EDT
python3 prob_to_mask.py \
    --prob ilastik_inputs/CB3-WKY-CB-A-2x2x2_ilastik_Probabilities.h5 \
    --high 0.70 --low 0.30 --min-size 50 \
    --out-prefix segmentation/CB3-WKY-CB-A --save-tif
```

Outputs `segmentation/CB3-WKY-CB-A_mask.npy` (bool) and `_edt_um.npy` (float32, **already in µm**). Reading the sweep and interpreting the radius report are covered in §10.2 and §10.5.

Use the **same thresholds for all six volumes** — a per-volume threshold reintroduces exactly the cohort-correlated bias §9.2.1 warns about.

---

### 13.4 Tuning guide

| Symptom | Cause | Fix |
|---|---|---|
| Vessels look eroded / hollow in `_grayscale.tif` | Rolling ball too small | `--rolling-ball 45` |
| Background haze survives | Rolling ball too large | `--rolling-ball 20` |
| Vesselness channels nearly empty | Sigmas too large for the structures | `--sigmas-fine 0.8 1.0 1.4` |
| `vesselness_coarse` lights capillaries | Coarse sigmas too small | `--sigmas-coarse 6 10` |
| Speckle survives into the mask | Genuine impulse noise | `--remove-outliers 500` (check the reported % is well under 1) |
| Machine swapping / sluggish | Budget too high for what else is open | lower `--max-memory-gb` |
| "reduced workers 8 → 1" | Chunks too large to fit the budget | lower `--chunk-z`, or raise `--max-memory-gb` |
| Too slow | Worker count capped by the budget | `--max-memory-gb 20` if the machine is idle; keep `--fast-coarse` on |
| Touching capillaries merge | Classifier problem, not preprocessing | Label the gaps (§9.3); do not collapse the vesselness channels |

**Do not** add `--remove-outliers` by default, and **do not** replace it with a median filter: at 1.866 µm a 3×3×3 median spans 5.6 µm and erases 2–4-voxel capillaries (§4).

### 13.5 Moving to the unbinned 0.933 µm data — read this before you try

Every pixel-valued argument doubles (§0.2):

```bash
python3 preprocess_cb.py --input raw_fullres/ --output-dir ilastik_inputs_fullres \
    --voxel 0.932 0.933 0.933 \
    --rolling-ball 60 \
    --sigmas-fine 2.0 2.8 4.0 --sigmas-coarse 8.0 16.0 \
    --chunk-z 16 --max-memory-gb 26
```

**Be aware this will be tight on 31 GB, and the script will tell you so rather than crashing.** Working through the formula for a 1014 × 912 × 870 volume:

| Term | Value | Why |
|---|---|---|
| Parent (working volume + 2 vesselness results, float32) | **~12.9 GB** | Scales with the volume; unavoidable in the current design |
| Per worker at `--chunk-z 16`, σ_max = 4 (halo 16) | **~7.1 GB** | Scales with chunk × Y × X |
| Workers fitting in a 26 GB budget | **1–2** | (26 − 12.9) / 7.1 |

So expect roughly **1–2 workers and several hours for six volumes**, versus 20 minutes at the binned resolution. That is a real limitation of this script, not a tuning problem: the parent-side 12.9 GB is the binding constraint, and the fix is to back the working volume and the vesselness accumulators with on-disk memmaps rather than RAM. That change is **not implemented** — it hasn't been written or tested, because there is no full-resolution data here to test it against. Treat it as the known next step when you actually make the move.

Interim options if you need full-resolution output sooner:

* Process a **cropped sub-volume** at full resolution — a 512³ region is well within budget and is enough to validate that radii improve as §0.1 predicts.
* Run with `--single-vesselness` to drop one full-volume result array (~3.2 GB of parent).
* Run on a larger-memory machine, where the current script needs no changes at all — just raise `--max-memory-gb`.

And remember that ilastik labels and the trained classifier do **not** transfer (§0.2); you will relabel from scratch.

---

## 14. Status of `fiji-batch-preprocess.ijm`

**Retired — do not run it.** Beyond the per-tile normalisation conflict, it hard-codes the two most damaging v4 steps (`Bleach Correction / Histogram Matching` at line 119, `Median 3D` radius 1 at line 133) and tiles unnecessarily. It also has standalone bugs:

* Line 14 declares `#javascript` while the file is IJ1 macro language throughout.
* Line 48 mixes `endsWith(fileName, ".tif")` with `fileName.endsWith(".tiff")` — two different call styles for the same operation.
* Line 119 passes `recorrection=[Histogram Matching]`; the Bleach Correction plugin's parameter key is `correction=`, so the method argument is silently ignored and the plugin falls back to its default.
* Line 60 "fixes" transposed dimensions by writing `unit=pixel`, which discards the µm calibration that every physical parameter in this protocol depends on.

Since preprocessing is being done interactively in Fiji for now, nothing replaces it. If batch automation is wanted later, the natural split is: Fiji macro for Phases 3–6 (the GUI-native steps), then `make_ilastik_input.py` for assembly — not one monolithic macro.
