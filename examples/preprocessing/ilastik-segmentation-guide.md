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

`vessel_segmentation.ilp` and `glomus_cell_segmentation.ilp` are separate projects. They are
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

Launch ilastik, choose **Pixel Classification**, save as `glomus_cell_segmentation.ilp` inside
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

### Step 4: two labels, in this order

Exactly parallel to the vessel project. The exported probability channels follow label order and
the index is what downstream code selects on, so the order is part of the contract:

| label | name | what to paint | signed DoG reads |
|---|---|---|---|
| 1 | glomus | the bright rings **and** the dark cores inside them | positive on the ring, about -0.23 in the core |
| 2 | background | empty space outside the clusters, and the gaps between touching cells | about 0.00 |

**Two classes, not four.** An earlier draft of this guide proposed Cytoplasm, Nucleus, Boundary
and Background, carried over from a protocol whose stated goal was watershed segmentation and
cell counting. Every H1 and H2 analysis that consumes this channel asks for a TH-positive volume,
voxel set or cluster boundary: parenchymal volume and length density (H1 1.3), tissue-to-vessel
distance (H1 1.5), flow overlay (H2 2.1), which edges supply the clusters (H2 2.2), metabolic
rate assignment and hypoxic fraction (H2 2.3), depletion within the boundaries (H2 2.4). **None
of them counts cells**, so the classes that exist to split touching somas would do no work while
roughly doubling the labelling effort.

Two decisions inside the two-class scheme do matter, because they are worth 9 to 15% of the
measured TH volume. Make each once and hold it for all six specimens:

* **Nuclear cores are glomus.** Every analysis means "where are the glomus cells", not "where is
  the TH protein". Painting cores as background gives you hollow shells and a volume biased low.
  This is also the one thing a plain intensity threshold gets wrong, and the reason channel 1 is
  kept signed: it reads about -0.23 in a core against about 0.00 in true background, so the
  classifier has a feature that separates them.
* **Intercellular gaps are background.** They are genuinely extracellular.

Measured stake, taken on the six preprocessed volumes: the interior region (cores plus gaps)
is 8.7 to 15.1% of whole-cell volume, averaging 10.9% in WKY against 13.1% in SHR. The direction
is what denser nests would produce, but the per-specimen ranges overlap and n = 3 per group gives
a permutation p floor of 0.10, so **no group difference is claimed**. What the number establishes
is that the decision is worth about a tenth of the headline volume and must therefore be
consistent, not that it needs more classes.

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
body at z = 40 to 140, well away from the parenchyma at z = 200 to 350. Painting it as glomus
teaches the classifier to find it everywhere. Either avoid those regions, or give them a third
label of their own so they are explicitly not glomus.

This is not a hypothetical risk. The first pass of the doughnut measurement during the pipeline
review ran on that slab and concluded there was no doughnut at all, 7.7% of detections showing a
core. Repeating it on genuine parenchyma gave 79.4%. The premise was fine; the region was wrong.

### Step 6: check before exporting

With Live Update on, step through z and confirm two things. That the dark nuclear cores inside
bright rings are predicted **glomus** and not background, which is what the signed DoG channel is
there to make possible. And that the clusters have plausible outer boundaries rather than
bleeding into stroma. Both are much cheaper to fix with more labels now than to discover after a
six-volume export.

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
| 2 classes, float32 | 3.66 GB |
| 2 classes, **uint8** | **0.91 GB** |

With 23 GB free, float32 now fits. uint8 is still the better default: it is four times smaller
and four times faster to read, and `prob_to_mask.py` already handles it, detecting `max > 1.5`
and rescaling by 255. Nothing downstream changes.

Headless equivalent:

```bash
DATA=~/Desktop/"LCFM Images"
~/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh --headless \
  --project="$DATA/ilastik_inputs/glomus_cell_segmentation.ilp" \
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

# TH: channel 0 is 'glomus'. Cores were labelled glomus, so they are already
# inside the mask; --no-fill-holes stops the 3D fill closing genuine gaps
# between adjacent nests as well.
python3 prob_to_mask.py --prob "$PROB/CB3-WKY-CB-A-2x2x2_TH_ilastik_Probabilities.h5" \
    --channel 0 --no-fill-holes --out-prefix CB3-WKY-CB-A-TH
```

Pick the `low` threshold just above where the component count starts climbing steeply and the
largest-component share starts falling. That is where genuine structure begins to be stranded.

Channel 0 is already whole-cell volume, because nuclear cores were labelled glomus rather than
left to a hole-filling step that cannot reliably close a shell two to three voxels thick.

---

## 6. Before you trust a segmentation

Run `verify_classifier()`. It checks the label order against the class index the code reads, that
no feature is computed in 2D, that all six specimens are registered as lanes, that every lane
carries labels, and that each has at least two depths.

It takes a `channel` argument, so it works for either project:

```bash
cd /home/dsas627/PycharmProjects/ImageLynx
venv/bin/python -c "
from ImageLynx.specimens import verify_classifier
import json; print(json.dumps(verify_classifier(channel='th'), indent=2, default=str))"
```

`channel` accepts `'vessel'`, `'th'`, or a `SegmentationChannel`, and defaults to the vessel
channel so existing call sites are unchanged. The two channels are defined in
`ImageLynx.specimens` as `VESSEL_CHANNEL` and `TH_CHANNEL`, each carrying its own project path,
input file naming, target label and index, and input channel names.

Verifying against the wrong channel fails loudly rather than passing by accident: a TH project
checked as `vessel` reports all six specimens unregistered, because the lane match is on the
full input file name.

One asymmetry worth knowing, since it is the reason the channel carries a stem rule rather than
just a suffix. The lectin volumes were preprocessed from separately extracted C1 TIFFs, so their
inputs are named `C1-CB3-WKY-CB-A-2x2x2_vessels_ilastik.h5` for WKY but
`CB3-SHR-CB-A-2x2x2_ilastik.h5` for SHR. The TH volumes were read straight out of the
two-channel acquisition and are all named after it, `CB3-WKY-CB-A-2x2x2_TH_ilastik.h5`. The SHR
lectin stem is therefore a prefix of its own TH input name, so matching on the stem alone would
let the vessel channel claim the TH lanes for the three SHR specimens and not for the three WKY
ones, which is worse than failing outright.
