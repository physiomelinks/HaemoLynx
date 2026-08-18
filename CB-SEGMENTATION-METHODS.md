# Carotid Body Vessel Segmentation — Methods and Integration Contract

Handover document for the CB haemodynamics pipeline. Describes how the ilastik
project (`.ilp`) was produced, what it expects as input, how to drive it headlessly from
Python, and what its outputs mean.

**Read §2 first if you only need the integration contract.**

Status legend: **[DONE]** established and fixed · **[TBD]** determined at training time,
fill in before relying on it.

---

## 1. What this is for

Segment microvasculature from 3D fluorescence z-stacks of rat carotid bodies, so the
result can be skeletonised into a 1D graph and used for haemodynamic modelling. Two
cohorts are compared: **WKY** (normotensive control) and **SHR** (spontaneously
hypertensive).

```
raw 2-channel TIFF
   └─[preprocess_cb.py]──▶ *_ilastik.h5   (3-channel, float32, zyxc)
                              └─[ilastik headless, cb_vessels.ilp]──▶ *_Probabilities.h5
                                    └─[prob_to_mask.py]──▶ mask.npy + edt_um.npy
                                          └─▶ THIS PIPELINE: skeletonise → graph → flow
```

The segmentation side stops at **binary mask + calibrated Euclidean distance map**.
Skeletonisation, graph assembly and flow solving are this pipeline's responsibility.

---

## 2. Integration contract

### 2.1 The `.ilp` expects a very specific input. This is not negotiable.

The classifier was trained on features computed from a **3-channel, globally normalised,
background-subtracted** volume. Feeding it anything else — a raw TIFF, a differently
normalised volume, or the same channels in a different order — produces confident nonsense
rather than an error.

| Property | Required value |
|---|---|
| Container | HDF5, dataset at internal path **`/data`** |
| Shape | `(z, y, x, c)` |
| dtype | **`float32`** |
| `axistags` attribute | **`zyxc`** (lowercase, vigra JSON) |
| Channels | **exactly 3, in this order** |
| Value range | **[0.0, 1.0]** per channel |

| Channel | Content |
|---|---|
| **0** | `grayscale` — rolling-ball background-subtracted, globally normalised lectin intensity |
| **1** | `vesselness_fine` — multiscale Sato ridge filter, σ = 1.0 / 1.4 / 2.0 px, per-scale normalised then max |
| **2** | `vesselness_coarse` — multiscale Sato ridge filter, σ = 4.0 / 8.0 px, computed at half resolution |

**Channel order is load-bearing.** The Random Forest indexes features by channel position.
Swapping 1 and 2 silently degrades predictions.

### 2.2 Any new volume must be preprocessed by the same script and parameters

Do not reimplement the preprocessing. Call `preprocess_cb.py` with the parameters in §11.1.
If preprocessing ever changes, **all** volumes must be reprocessed and the classifier
retrained — see §9.1.

### 2.3 Voxel size

`1.8660 × 1.8660 × 1.8639 µm` (z, y, x order in metadata is `(1.8639, 1.8660, 1.8660)`).

Recorded in the HDF5 attribute `voxel_size_um_zyx`. Near-isotropic (axial:lateral =
1.0011), so no anisotropy correction is needed anywhere.

> Minor note: SHR volumes have a true z-spacing of 1.8641 µm but were annotated with the
> WKY default 1.8639. The 0.011 % difference is well below any meaningful threshold
> (0.0002 µm on a 2 µm radius) and does not affect processing — `--voxel` is metadata
> only and does not enter the filter maths. Use 1.8639 consistently.

---

## 3. Source data

Six volumes, 2×2×2-binned acquisition, uint16, 2 channels (ZCYX).
Channel 0 = 561 nm = lectin = **vessels** (the one used).
Channel 1 = 640 nm = TH = glomus cells (not used; verified visually as punctate cell
clusters, not tubular).

| Cohort | Vol | Shape (z, y, x) | Voxels |
|---|---|---|---|
| WKY | A | 435 × 456 × 507 | 100.6 M |
| WKY | B | 435 × 357 × 351 | 54.5 M |
| WKY | C | 435 × 315 × 255 | 34.9 M |
| SHR | A | 495 × 459 × 345 | 78.4 M |
| SHR | B | 495 × 483 × 399 | 95.4 M |
| SHR | C | 495 × 495 × 381 | 93.3 M |

Unbinned 0.933 µm versions of all six exist (`*-1x1x1.tif`) but are **not** used — see §10.4.

---

## 4. Stage 1 — Preprocessing **[DONE]**

Implemented in `preprocess_cb.py`. Applied identically to all six volumes.

| Step | Setting | Rationale |
|---|---|---|
| 1. Channel extraction | `--channel 0` | Lectin/vessel channel |
| 2. Axial decay correction | **none** | See §4.1 |
| 3. Rolling-ball background subtraction | radius **30 px** (56 µm), per slice | Larger than the 15 µm radius of the thickest arteriole, so lumens are not eroded |
| 4. Denoising | **none** | See §4.2 |
| 5. Global normalisation | percentile **0.02 %**, whole-stack histogram → [0, 1] | Puts all six volumes on a common intensity scale — this is what makes one shared classifier legitimate |
| 6. Multiscale vesselness | fine σ = 1.0/1.4/2.0 px; coarse σ = 4.0/8.0 px (half-res) | Per-scale normalised **before** the cross-scale max |

### 4.1 Why no bleach correction

All six volumes show a **hump-shaped** axial intensity profile — peak intensity mid-stack,
falling at both ends. That reflects how much tissue each slice intersects, not
photobleaching. Applying histogram matching (as an earlier protocol version did) multiplies
the sparse end slices several-fold and promotes their background noise to vessel-level
intensity. `preprocess_cb.py --diagnose` reports the verdict automatically and will say
`monotonic decay` if a future volume genuinely needs correction.

### 4.2 Why no denoising

Voxels are 1.866 µm; capillaries are 4–7 µm, i.e. **2–4 voxels wide**. A 3×3×3 median spans
5.6 µm — wider than a capillary — and erases the structures being segmented. Any filter
with support comparable to the structure width is destructive here. Ilastik's Gaussian
smoothing features provide denoising reversibly, as features the classifier can weight.

### 4.3 Why the vesselness scales are normalised before combining

Sato/Frangi responses are not γ-normalised across σ. On this data the σ = 8 response is
~3× stronger than σ = 1, so a naive voxelwise max is dominated by the coarsest scale and
all capillary-scale selectivity is lost. Each scale is normalised to [0, 1] first.

Fine and coarse are kept as **separate channels** rather than merged: where two capillaries
run close together, fine shows two ridges while coarse shows one blob. That disagreement is
a learnable discriminative cue. Measured `corr(fine, coarse)` across the six volumes is
**0.545–0.697** — they carry genuinely different information.

---

## 5. Stage 2 — ilastik project **[TBD]**

Workflow: **Pixel Classification**. One `.ilp`, **all six volumes as datasets**.

### 5.1 Classes

| Index | Class | Notes |
|---|---|---|
| **[TBD]** | Vessel | Record the index — needed in §6.3 |
| **[TBD]** | Background | |

> **Record the Vessel class index here before running headless prediction.** ilastik
> exports one probability channel per class in label order. Picking the wrong one silently
> yields the inverse segmentation.

### 5.2 Features **[TBD — record the final set]**

Features apply to **all channels** (ilastik has no per-channel configuration) and are
computed in **3D** (2D-per-slice features would give z-anisotropic predictions and
staircase artifacts in the skeleton).

Physical scale of ilastik's fixed σ grid at 1.866 µm/voxel:

| σ (px) | 0.3 | 0.7 | 1.0 | 1.6 | 3.5 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|---|
| µm | 0.56 | 1.31 | **1.87** | **2.99** | 6.53 | 9.33 | 18.66 |
| | sub-voxel, excluded | wall | capillary radius | capillary half-diameter | arteriole wall | large vessel | tissue context |

Recommended set (23 checkboxes; eigenvalue features return 3 values each in 3D, so ~37
values per channel × 3 channels ≈ **111 features per voxel**):

| Feature | 0.7 | 1.0 | 1.6 | 3.5 |
|---|:-:|:-:|:-:|:-:|
| Gaussian Smoothing | ✓ | ✓ | ✓ | ✓ |
| Laplacian of Gaussian | ✓ | ✓ | ✓ | ✓ |
| Gaussian Gradient Magnitude | ✓ | ✓ | ✓ | ✓ |
| Difference of Gaussians | ✓ | ✓ | ✓ | |
| Structure Tensor Eigenvalues | ✓ | ✓ | ✓ | |
| Hessian of Gaussian Eigenvalues | ✓ | ✓ | ✓ | ✓ |

σ = 0.3 excluded (below one voxel — fits noise). σ = 5.0 and 10.0 excluded (arteriole scale
is already supplied by channel 2).

### 5.3 Labelling strategy

- Labels pooled across all six volumes into one classifier.
- Distributed across depth (each volume's tissue peaks at a different slice — see §11.2).
- Roughly balanced label counts per volume; the RF weights by labelled voxel count.
- SHR labelled at least as heavily as WKY (hypervascular, denser touching-capillary cases).
- Background labelled explicitly in the narrow gaps between parallel capillaries, and over
  the acquisition striping (vertical in WKY-B, horizontal in WKY-C).

---

## 6. Stage 3 — Headless prediction

### 6.1 Command

```bash
run_ilastik.sh --headless \
  --readonly \
  --project=/path/to/cb_vessels.ilp \
  --export_source="Probabilities" \
  --output_format=hdf5 \
  --output_filename_format="/path/to/out/{nickname}_Probabilities.h5" \
  --output_internal_path=exported_data \
  --export_dtype=float32 \
  "/path/to/ilastik_inputs/VOLUME_ilastik.h5/data"
```

Note the `.h5/data` suffix on the input — ilastik needs the internal dataset path.
Verify flag names against `run_ilastik.sh --help` for your installed version.

### 6.2 Python wrapper

```python
import subprocess, pathlib

ILASTIK = "/path/to/ilastik/run_ilastik.sh"
PROJECT = "/path/to/cb_vessels.ilp"

def predict(input_h5: str, out_dir: str) -> str:
    """Run headless pixel classification. Returns the output path."""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ILASTIK, "--headless", "--readonly",
        f"--project={PROJECT}",
        "--export_source=Probabilities",
        "--output_format=hdf5",
        f"--output_filename_format={out_dir}/{{nickname}}_Probabilities.h5",
        "--output_internal_path=exported_data",
        "--export_dtype=float32",
        f"{input_h5}/data",
    ], check=True)
    stem = pathlib.Path(input_h5).stem
    return f"{out_dir}/{stem}_Probabilities.h5"
```

### 6.3 Export all classes, select the channel in Python

**Do not rely on the project's stored subregion settings.** Export every class channel and
slice explicitly — it is more robust in a pipeline and self-documenting:

```python
import h5py, numpy as np

VESSEL_CLASS_INDEX = 0   # [TBD] confirm against §5.1

with h5py.File(prob_path) as f:
    arr = np.squeeze(f["exported_data"][...])   # (z, y, x, n_classes)
assert arr.ndim == 4, f"expected class axis, got {arr.shape}"
prob = arr[..., VESSEL_CLASS_INDEX].astype(np.float32)
assert 0.0 <= prob.min() and prob.max() <= 1.0
```

Sanity check: mean vessel probability should be a few percent, not ~0.5 and not ~0.95.
A mean near `1 - expected` means the wrong class index.

### 6.4 Export dtype

**float32, not 8-bit.** The post-processing in §7 uses hysteresis thresholding, which needs
the probability gradients. Quantising to 256 levels before thresholding discards exactly
the information that separates faint-but-real capillaries from noise.

### 6.5 Resources

Six volumes ≈ 457 M voxels. Feature computation dominates. Predict headless rather than
through the GUI; budget tens of minutes to a few hours depending on the final feature set.
Reference machine: 31 GB RAM, 20 cores. If memory becomes a problem, lower ilastik's thread
count before anything else — concurrent per-block feature stacks are what exhausts RAM, not
the volume itself.

---

## 7. Stage 4 — Probability map → mask + EDT

Implemented in `prob_to_mask.py`.

```bash
python3 prob_to_mask.py --prob VOLUME_Probabilities.h5 --sweep         # pick thresholds
python3 prob_to_mask.py --prob VOLUME_Probabilities.h5 \
        --high 0.70 --low 0.30 --min-size 50 --out-prefix OUT
```

| Step | Setting | Rationale |
|---|---|---|
| 1. Hysteresis threshold | high **0.70**, low **0.30** **[TBD — confirm via `--sweep`]** | A single cutoff forces a choice between fragmenting faint capillaries and fusing adjacent ones. Seeds from `high`, grows into `low` only where connected. |
| 2. 3D cavity fill | `scipy.ndimage.binary_fill_holes` | Lectin stains endothelium, so large vessels appear as rings. **Must be 3D** — a 2D per-slice fill would fill every in-plane vascular loop and destroy topology. |
| 3. Size filter | **50 voxels** (≈ 325 µm³) | Smaller than the shortest plausible capillary segment, larger than typical debris |
| 4. EDT | `distance_transform_edt(mask, sampling=(1.8639, 1.8660, 1.8660))` | `sampling=` makes the output **µm directly** |

Choosing `low`: run `--sweep` and take the value just above where component count starts
climbing steeply and the largest component's share starts falling. That is where real
capillary segments begin to be stranded.

**Use identical thresholds for all six volumes.** Per-volume tuning reintroduces exactly
the cohort-correlated bias the shared classifier exists to prevent.

---

## 8. Stage 5 — Handoff to this pipeline

```python
mask = np.load(f"{prefix}_mask.npy")      # bool,    (z, y, x)
edt  = np.load(f"{prefix}_edt_um.npy")    # float32, (z, y, x), ALREADY IN MICROMETRES
```

### Four things to get right downstream

1. **Sample the EDT at skeleton voxels; do not multiply.**
   `radii = edt[skeleton]`, not `edt * skeleton`. Fiji skeletons are 0/255 and would
   inflate every radius 255×; `skimage.skeletonize` returns bool, which happens to work —
   boolean indexing is unambiguous either way.
2. **Trim junction neighbourhoods before averaging radii.** Within ~1 radius of a
   bifurcation the EDT reports the junction's inscribed sphere, not the vessel's, biasing
   radii upward. Discard skeleton voxels within 2 voxels of any junction node.
3. **Use along-path length for resistance**, not euclidean endpoint separation (that is for
   tortuosity only).
4. **Do not prune anastomotic loops.** Vascular loops are real anatomy; removing them
   corrupts flow topology. If using Fiji's `Analyze Skeleton`, set `Prune cycle method` to
   `none` and leave `Prune ends` off — remove spurs by length instead.

### Validation before trusting any flow result

| Check | Expectation | Failure means |
|---|---|---|
| Capillary diameter mode | 4–7 µm | Threshold too low (thick) or too high (thin) |
| Murray's law, Σr³ at bifurcations | approximately conserved | Junction radius bias — see point 2 |
| Largest connected component | dominates vessel voxels | `low` too high, network fragmented |
| Vessel volume fraction across a cohort | broadly comparable | Staining or normalisation outlier |
| Endpoint count | modest | Fragmentation, not anatomy |

Expect the diameter mode to sit near a multiple of 1.87 µm — that is voxel quantisation,
not biology. See §10.1.

---

## 9. Invariants — violating these invalidates the cohort comparison

### 9.1 One classifier for all six volumes. Never one per cohort.

The classifier is the measuring instrument. Its decision boundary sets where the vessel wall
lies, which sets radius, which enters Poiseuille resistance as **r⁻⁴**. With two separately
trained classifiers, every measured cohort difference is a sum of biology *and* an
unmeasurable difference between the two boundaries, with no way to separate them.

Error asymmetry: a single classifier slightly suboptimal on one cohort adds noise and biases
**toward the null** — conservative and defensible. Two classifiers add systematic bias of
unknown sign directly to the group contrast, and can manufacture or mask an effect.

### 9.2 Identical parameters everywhere

Same preprocessing arguments, same feature set, same hysteresis thresholds, same
`--min-size`, for all volumes in both cohorts. No per-volume or per-cohort tuning.

### 9.3 Required validation

Because the single-classifier design shifts the burden to *demonstrating* fairness:

- Hold out hand-labelled regions in **both** cohorts, excluded from training.
- Report Dice / precision / recall **separately for WKY and SHR**. Comparable accuracy is
  the evidence the instrument is unbiased.
- Run a **threshold sensitivity analysis** — repeat the morphometrics at low = 0.25 / 0.30 /
  0.35. If the cohort effect survives, say so. If its direction flips, it was never real.

---

## 10. Interpretation caveats

### 10.1 Radius-derived quantities are provisional at this resolution

A capillary radius is 1.1–1.9 voxels. Half-voxel boundary error on a nominal 3 µm radius is
**±31 %**, propagating to **~3×** on segment resistance.

| Trustworthy | Provisional |
|---|---|
| Branch and junction counts | Vessel radius, diameter |
| Junction density | Vessel volume fraction |
| Segment length, tortuosity | Surface area |
| Connectivity, loop structure | Flow, resistance, anything from r⁻⁴ |

A half-voxel boundary shift barely moves a centreline but changes radius directly. **Lead
with topology; present radius-derived haemodynamics as provisional** until the unbinned data
is used.

`prob_to_mask.py --refine-radii` upsamples the probability map ×2 before the EDT, halving
the quantisation step (8× memory/time). Verified on WKY-A: ridge-voxel median radius stable
at 1.87 µm while the step drops 1.87 → 0.93 µm. It creates no new information — it only lets
the boundary sit at a sub-voxel position.

### 10.2 Cohort intensity offset — investigated, benign

SHR normalisation anchors (5743–6674) sit entirely below WKY's (8145–11536). Raw channel-0
statistics before normalisation:

| | background (p50) | noise (MAD) | signal (p99.9) | SNR |
|---|---|---|---|---|
| WKY mean | 586 | 586 | 8599 | 15.8 |
| SHR mean | 233 | 255 | 5689 | 28.0 |
| SHR/WKY | 0.40 | 0.43 | 0.66 | 1.77 |

This is a **gain/exposure offset, not a signal-quality deficit**: SHR's background and noise
scale down further than its signal, so its SNR is equal or better. Per-volume normalisation
absorbs this by design. Two caveats: with n = 3 per cohort the means are outlier-dominated,
so this shows no evidence SHR is worse rather than proving it is better; and it is a gain
argument, not proof of equal segmentation accuracy — §9.3 is what settles that.

**Watch WKY-C specifically**: background 955 versus 337–466 for the other WKY volumes, SNR
7.8 — the weakest of all six. If any volume segments poorly, expect it there.

### 10.3 Acquisition striping

Vertical striping in WKY-B, horizontal banding in WKY-C. These survive background
subtraction because they are periodic, not low-frequency. Handled by labelling background
over the striped regions rather than by adding a filter.

### 10.4 The unbinned 0.933 µm data

Exists for all six volumes (`*-1x1x1.tif`). Moving to it would improve radius accuracy more
than anything else available — a capillary becomes 4–7.5 voxels wide and the half-voxel
error drops to ±16 %.

Blockers:
- Every pixel-valued parameter doubles (rolling ball 60 px, σ fine 2.0/2.8/4.0, coarse
  8.0/16.0; ilastik σ set shifts up).
- **ilastik labels and the trained classifier do not transfer** — relabelling from scratch
  is required.
- `preprocess_cb.py` is tight at that resolution on 31 GB (~12.9 GB of parent-side arrays
  leaves room for only 1–2 workers). The fix is memmap-backing the working arrays;
  **not implemented, not tested**.

---

## 11. Reproducibility record

### 11.1 Preprocessing parameters — identical across all six volumes

```json
{
  "script": "preprocess_cb.py",
  "channel": 0,
  "rolling_ball_px": 30.0,
  "remove_outliers": 0,
  "saturated_percent": 0.02,
  "sigmas_fine_px": [1.0, 1.4, 2.0],
  "sigmas_coarse_px": [4.0, 8.0],
  "fast_coarse": true,
  "single_vesselness": false,
  "voxel_um_zyx": [1.8639, 1.866, 1.866],
  "bleach_correction": null,
  "denoising": null,
  "tiling": null,
  "output": "3-channel float32 HDF5, /data, axistags zyxc, range [0,1]"
}
```

### 11.2 Per-volume record

| Volume | Shape (z, y, x) | Norm. anchor (hi) | z-profile peak | z-verdict | Preproc. time |
|---|---|---|---|---|---|
| `C1-CB3-WKY-CB-A-2x2x2_vessels` | 435 × 456 × 507 | 11536.1 | 230 / 435 | hump | 277 s |
| `C1-CB3-WKY-CB-B-2x2x2_vessels` | 435 × 357 × 351 | 10265.3 | 106 / 435 | hump | 134 s |
| `C1-CB3-WKY-CB-C-2x2x2_vessels` | 435 × 315 × 255 | 8145.2 | 189 / 435 | hump | 87 s |
| `CB3-SHR-CB-A-2x2x2` | 495 × 459 × 345 | 6673.8 | 157 / 495 | hump | 183 s |
| `CB3-SHR-CB-B-2x2x2` | 495 × 483 × 399 | 6594.1 | 230 / 495 | hump | 256 s |
| `CB3-SHR-CB-C-2x2x2` | 495 × 495 × 381 | 5742.7 | 164 / 495 | hump | 243 s |

Per-volume `*_qc.json` sidecars in `ilastik_inputs/` carry the full machine-readable record.
The z-profile peaks differ per volume — relevant when choosing where to sample or label.

### 11.3 Scripts

| Script | Role |
|---|---|
| `preprocess_cb.py` | raw TIFF → 3-channel ilastik HDF5. **Must be used for any new volume** (§2.2) |
| `prob_to_mask.py` | ilastik probabilities → mask + calibrated EDT |
| `optimal-filtering-strategies-v5.md` | Full protocol with derivations and rejected alternatives |

### 11.4 Values to fill in once training is complete

- [ ] Vessel class index (§5.1)
- [ ] Final feature set actually used (§5.2)
- [ ] Hysteresis thresholds chosen from `--sweep` (§7)
- [ ] Per-cohort held-out validation scores (§9.3)
- [ ] `.ilp` file path and checksum

---

## 12. Rejected approaches

Recorded so they are not reintroduced. Each was tried or specified in an earlier protocol
version and is **wrong for this data**.

| Approach | Why rejected |
|---|---|
| Bleach correction by histogram matching | Z-profile is a hump, not decay; inflates sparse end slices ~6× and promotes background noise to vessel intensity |
| 3D median filter (radius 1) | 5.6 µm support exceeds a capillary diameter; erases the target structures |
| Lateral tiling for RAM | Unnecessary at this size; causes seam artifacts in Hessian features and fragments network connectivity at the seams |
| Single global probability threshold | Forces a choice between fragmenting faint capillaries and fusing adjacent ones; replaced by hysteresis |
| Binary `dilate → median → erode` gap bridging | Majority vote on a binary mask deletes 2–3 voxel capillaries; removes more than it repairs |
| 8-bit probability export | Discards the gradients hysteresis thresholding depends on |
| Per-cohort classifiers | Confounds biology with classifier-boundary differences (§9.1) |
| Un-normalised max across vesselness scales | Dominated by the coarsest σ; destroys capillary selectivity |
| 2D-per-slice features in ilastik | z-anisotropic predictions, staircase skeleton artifacts |
| 2D per-slice hole filling | Fills every in-plane vascular loop, destroying topology |
