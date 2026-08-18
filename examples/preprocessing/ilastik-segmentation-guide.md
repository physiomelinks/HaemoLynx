# Ilastik Pixel Classification: segmenting the carotid body volumes

> **Scope.** Both channels of the CB3 acquisitions: the lectin/vessel channel, which is already
> segmented, and the TH/glomus cell channel, which is ready to label. One document because the
> two share a project layout, a labelling standard and a downstream consumer, and because the
> differences between them are easier to get right when they sit side by side.
>
> **Companions.** `README.md` for the preprocessing that produces these inputs,
> `optimal-filtering-strategies-v5.md` for the lectin protocol,
> `th-glomus-cell-preprocessing-guide.md` for the TH protocol,
> `../../th_glomus_preprocessing_review.md` for the measurements behind the TH decisions.
>
> **Ilastik version.** 1.4.1rc2, at `~/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh`.

---

## 0. System readiness

Checked 2026-08-18 on this machine.

| resource | state | needed | verdict |
|---|---|---|---|
| RAM | 31 GB total, **19 GB available**, 8 GB swap unused | see below | **ready** |
| disk `/home` | 389 GB, **23 GB free** (94% used) | 1.8 to 7.3 GB for the export | **ready** |
| CPU | 20 cores | more is faster, none is blocking | ready |
| inputs | six `*_TH_ilastik.h5` in `ilastik_inputs/` | verified against the contract | ready |

**You are ready to proceed.**

On memory specifically. Ilastik does not hold the whole feature stack at once; it computes
features lazily in blocks and caches them under a configurable budget. The reassuring evidence
is empirical rather than theoretical: the lectin segmentation already ran to completion on this
same machine over volumes of the same size with **three** input channels, and the TH inputs have
**two**, so the feature work per volume is lighter than a job that has already succeeded here.

Two practical notes:

* There is no `~/.ilastikrc` and no `LAZYFLOW_*` variables set, so ilastik will detect and help
  itself to most of the machine. That is usually what you want. If you would rather cap it,
  create `~/.ilastikrc` containing:

      [lazyflow]
      total_ram_mb = 16000
      threads = 12

* 12 GB of the 31 is currently held by VS Code, Chrome and the Claude apps. Closing the browser
  before a long export is worth more than any ilastik setting.

**Live Update is the expensive operation.** It recomputes features across the visible region of
the whole lane. Toggle it on to check your work and off again while painting, rather than
leaving it on throughout.

---

## 1. What the two channels share

### One project per channel, never mixed

`vessel_segmentation.ilp` and `th_glomus_segmentation.ilp` are separate projects. They are
trained on different channels, different numbers of them, and different label sets. Adding TH
lanes to the vessel project produces confident nonsense rather than an error.

### The project must live beside its data

Both `.ilp` files belong **inside `~/Desktop/LCFM Images/ilastik_inputs/`**. An ilastik project
registers its datasets by path relative to itself, so a project saved elsewhere breaks as soon
as the folder moves, while moving the whole directory together is safe.

### The pooled classifier rule

One project, all six volumes registered as lanes, **every lane carrying labels, at two or more
depths**. This is enforced in code by `ImageLynx.specimens.verify_classifier`, and it exists
because of a real failure: the first trained vessel project had all 454 of its labels on WKY-A
on a single z slice, with the other five lanes registered and empty. A decision boundary learned
from normotensive tissue and applied to hypertensive tissue reintroduces precisely the confound
the study is trying to measure, one level below where anything else in the codebase can see it.

Labelling at a single depth is a milder version of the same problem: each volume's tissue peaks
at a different slice, and the sparse end slices are where background comes closest to signal.

### 3D features only

Never tick "Compute in 2D". Per-slice features give z-anisotropic predictions and staircase
artefacts in the skeleton. `verify_classifier` refuses a project that uses them.

### The input file contract

Both preprocessors write the same shape of file, and both are verified to parse in ilastik's own
vigra:

| | value |
|---|---|
| dataset | `data` at the file root |
| axistags | `zyxc`, channel-last |
| dtype | `float32` |
| voxel size | 1.8639 x 1.866 x 1.866 um, recorded in `voxel_size_um_zyx` |

---

## 2. The lectin / vessel channel

**Status: complete.** `vessel_segmentation.ilp` passes `verify_classifier()`. Probability maps
for all six specimens are in `ilastik_probabilities/`. This section is the reproducible record,
not a task list.

### Inputs

`preprocess_cb.py` output, three channels:

| channel | contents |
|---|---|
| 0 | `grayscale`, rolling-ball subtracted and normalised lectin intensity |
| 1 | `vesselness_fine`, multiscale Sato at sigma 1.0 / 1.4 / 2.0 px |
| 2 | `vesselness_coarse`, multiscale Sato at sigma 4.0 / 8.0 px, half resolution |

### Labels

Two, in this order. **`vessel` is index 0**, which is what `prob_to_mask.py --channel 0`
selects. Reading the wrong channel gives you the inverse segmentation with no error.

1. `vessel`
2. `background`

### The feature matrix actually used

13 features, all 3D. Read directly from the trained project, so this is what the classifier was
built on rather than what was intended:

| feature | 0.3 | 0.7 | 1.0 | 1.6 | 3.5 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|---|
| Gaussian Smoothing | | | X | X | X | | |
| Laplacian of Gaussian | | | X | X | X | | |
| Gaussian Gradient Magnitude | | X | | X | X | | |
| Difference of Gaussians | | | X | X | | | |
| Structure Tensor Eigenvalues | | | | | | | |
| Hessian of Gaussian Eigenvalues | | | X | X | | | |

### Labelling achieved

163,035 labelled voxels across the six lanes, every lane non-empty, 4 to 6 distinct depths each.
That is the benchmark for what "adequately labelled" looks like on this data: roughly 20,000 to
40,000 voxels per lane.

### Re-verifying it

```bash
cd /home/dsas627/PycharmProjects/ImageLynx
venv/bin/python -c "
from ImageLynx.specimens import verify_classifier
import json; print(json.dumps(verify_classifier(), indent=2, default=str))"
```

It raises `ValueError` listing every problem at once if the project drifts, since relabelling is
one trip back to the GUI either way.

---

## 3. The TH / glomus cell channel

**Status: ready to label.** Nothing is trained yet.

### Inputs

`preprocess_th.py` output, two channels:

| channel | contents |
|---|---|
| 0 | `grayscale`, rolling-ball subtracted, tissue-anchored normalised TH intensity |
| 1 | `soma_dog_signed`, signed 3D Difference of Gaussians at sigma 1.0 and 3.0 px |

Channel 1 is **signed on purpose**. It reads strongly positive on the bright cytoplasmic ring,
about **-0.23** in the dark nuclear core and about **0.00** in background. That sign is the only
thing distinguishing "inside the nucleus" from "outside the cell", and an earlier version of the
pipeline clipped it away, mapping 99.8% of cores and 55.8% of background to the same value.

All six were produced in one run with shared anchors `--anchors 708 10578`, so the gain is
identical across the cohort by construction.

### Step 1: new project

Launch ilastik, choose **Pixel Classification**, save as `th_glomus_segmentation.ilp` inside
`ilastik_inputs/`.

### Step 2: add all six lanes

**Input Data**, `Add New...`, `Add separate Image(s)...`, select all six `*_TH_ilastik.h5`. If
prompted for the internal dataset choose `data`. Confirm the axes column reads `zyxc` and each
lane shows 2 channels.

Add all six now. A volume the classifier was never shown cannot be part of a pooled training set.

### Step 3: features

**Feature Selection**, `Select Features...`. A reasonable starting set, scaled to the measured
morphology (ring peak at r = 4.0 px, nucleus 2.4 to 3.2 px across):

| feature | 0.3 | 0.7 | 1.0 | 1.6 | 3.5 | 5.0 |
|---|---|---|---|---|---|---|
| Gaussian Smoothing | | X | X | X | X | X |
| Laplacian of Gaussian | | | X | X | X | |
| Gaussian Gradient Magnitude | | X | X | X | | |
| Difference of Gaussians | | | X | X | | |
| Hessian of Gaussian Eigenvalues | | | X | X | | |

That is a little broader than the vessel set, because cells are blobs at two scales (the ring
and the whole soma) where vessels are ridges at one. Everything in 3D.

### Step 4: four labels, in this order

The exported probability channels follow label order, and the index is what downstream code
selects on, so the order is part of the contract:

| label | name | what to paint | signed DoG reads |
|---|---|---|---|
| 1 | Cytoplasm | thin strokes along the bright rings, away from where cells touch | strongly positive |
| 2 | Nucleus | small dots in the dark cores only | about -0.23 |
| 3 | Boundary | thin lines in the dim gaps between touching somas | near zero, negative |
| 4 | Background | empty space outside the cell clusters | about 0.00 |

Nucleus and Background are **separate classes**, unlike the three-class scheme an earlier draft
of the protocol proposed. Merging them makes the segmented object a hollow cytoplasmic shell and
biases cell volume low by roughly 8%, and a 3D fill will not reliably close a shell two to three
voxels thick. Keeping the nucleus separate also hands you a natural watershed seed, which is far
more robust for counting than seeding from cytoplasm. Whole-cell volume is classes 1 and 2
together.

### Step 5: label every lane, at three depths

Tissue does not sit at the same depth in every stack, so label where the tissue actually is:

| specimen | z extent | tissue-bearing z | suggested depths |
|---|---|---|---|
| WKY-A | 435 | 0 - 430 | 86, 215, 344 |
| WKY-B | 435 | 15 - 360 | 84, 187, 291 |
| WKY-C | 435 | 100 - 430 | 166, 265, 364 |
| SHR-A | 495 | 0 - 320 | 64, 160, 256 |
| SHR-B | 495 | 70 - 450 | 146, 260, 374 |
| SHR-C | 495 | 0 - 345 | 69, 172, 276 |

Aim for the vessel benchmark, roughly 20,000 to 40,000 labelled voxels per lane. Use a 1 or 2 px
brush.

**Avoid the TH-positive structures that are not carotid body.** Sympathetic neurons and nerve
fibres are TH-positive too. In WKY-A the brightest TH structure in the entire stack is a fibrous
body at z = 40 to 140, well away from the parenchyma at z = 200 to 350. Painting it as Cytoplasm
teaches the classifier to find it everywhere. Either avoid those regions or give them a fifth
label of their own.

This is not a hypothetical risk. The first pass of the doughnut measurement during the pipeline
review ran on that slab and concluded there was no doughnut at all, 7.7% of detections showing a
core. Repeating it on genuine parenchyma gave 79.4%. The premise was fine; the region was wrong.

### Step 6: check before exporting

With Live Update on, step through z and confirm that adjacent cells in a dense nest are separated
by a Boundary or Nucleus prediction rather than fused into one blob. That separation is the
entire purpose of the two-channel input, and it is much cheaper to fix with more labels now than
to discover after a six-volume export.

---

## 4. Exporting probabilities

**Prediction Export**, `Choose Export Image Settings...`:

* Source: **Probabilities**
* Format: **hdf5**
* Output: `~/Desktop/LCFM Images/ilastik_probabilities/{nickname}_Probabilities.h5`
* Convert to Data Type: **unsigned 8-bit**, with **Renormalize** ticked

Then `Export All Lanes`.

Sizing across all six volumes, 457 M voxels total:

| export | size |
|---|---|
| 2 classes, float32 (the lectin export, already on disk) | 3.66 GB |
| 4 classes, float32 | 7.31 GB |
| 4 classes, **uint8** | **1.83 GB** |

With 23 GB free, float32 now fits. uint8 is still the better default: it is four times smaller
and four times faster to read, and `prob_to_mask.py` already handles it, detecting `max > 1.5`
and rescaling by 255. Nothing downstream changes.

Headless equivalent:

```bash
DATA=~/Desktop/"LCFM Images"
~/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh --headless \
  --project="$DATA/ilastik_inputs/th_glomus_segmentation.ilp" \
  --export_source="Probabilities" \
  --export_dtype=uint8 \
  --output_format=hdf5 \
  --output_filename_format="$DATA/ilastik_probabilities/{nickname}_Probabilities.h5" \
  "$DATA/ilastik_inputs/"*_TH_ilastik.h5
```

`ImageLynx.io.ilastik.run_ilastik_headless_segmentation` is **not** the right helper here: it
exports `Simple Segmentation`, not `Probabilities`, and `prob_to_mask.py` expects probabilities.

---

## 5. Downstream: probability map to mask

Same script for both channels, different arguments.

```bash
cd examples/preprocessing
PROB=~/Desktop/"LCFM Images"/ilastik_probabilities

# Always sweep first. It prints the fragmentation curve so the threshold is
# chosen on evidence rather than by eye.
python3 prob_to_mask.py --prob "$PROB/<name>_Probabilities.h5" --channel 0 --sweep

# Vessels: channel 0 is 'vessel'. Cavity filling is correct here, since it
# closes hollow arteriole lumens without touching vascular loops.
python3 prob_to_mask.py --prob "$PROB/C1-CB3-WKY-CB-A-2x2x2_vessels_ilastik_Probabilities.h5" \
    --channel 0 --out-prefix CB3-WKY-CB-A

# TH: channel 0 is 'Cytoplasm'. Pass --no-fill-holes, or the 3D fill closes the
# nuclear cores the whole pipeline was built to preserve.
python3 prob_to_mask.py --prob "$PROB/CB3-WKY-CB-A-2x2x2_TH_ilastik_Probabilities.h5" \
    --channel 0 --no-fill-holes --out-prefix CB3-WKY-CB-A-TH
```

Pick the `low` threshold just above where the component count starts climbing steeply and the
largest-component share starts falling. That is where genuine structure begins to be stranded.

For whole-cell TH volume rather than cytoplasm alone, combine channels 0 and 1 (Cytoplasm plus
Nucleus) before thresholding, rather than filling holes in the cytoplasm mask.

---

## 6. Before you trust a segmentation

Run `verify_classifier()`. It checks the label order against the class index the code reads, that
no feature is computed in 2D, that all six specimens are registered as lanes, that every lane
carries labels, and that each has at least two depths.

**It does not yet work for the TH project.** It is written against the vessel label name and
class index, so it will reject a TH project on the first check regardless of how well labelled it
is. Extending it to take the label name and index as arguments is small, and worth doing before
any TH segmentation is used for a measurement. Until then the TH project has no automated guard,
so the pooled-labelling rule has to be held by hand.
