# TH-Positive Glomus Cell Preprocessing and Segmentation Guide (revision 2)
*An Automated Preprocessing, Multi-Channel Machine Learning, and Shape-Aware Boundary Segmentation Workflow for 3D Confocal Imaging of Carotid Body Receptor Cells*

> **Revision note.** Revision 1 was a Fiji-first protocol with a companion script,
> `process_th_glomus_cells.py`. Every phase below has since been executed against
> `CB3-WKY-CB-{A,B,C}-2x2x2.tif` and several were measured to be harmful. Those are marked
> **CHANGED** with the measurement that forced the change; see
> `th_glomus_preprocessing_review.md` for the full working. The implementation is now
> `preprocess_th.py`, which shares its stages with the lectin pipeline `preprocess_cb.py`
> so that both channels of the same acquisition are treated identically.
> `process_th_glomus_cells.py` is superseded and should not be run.

---

## 1. Physical and Geometric Calibration Parameters

Unlike the continuous, tubular network of the Lectin-labelled microvasculature, the Tyrosine
Hydroxylase (TH) immunofluorescent channel labels **discrete, clustered, rounded-to-ovoid cell
bodies (Type I glomus cell somas)** [45, 165]. These sensory receptors act as the core oxygen
sensors of the carotid body (CB) [55, 107].

Voxel calibration, read from the ImageJ metadata of each file rather than assumed:

$$\Delta x = 1.8660\ \mu\text{m},\quad \Delta y = 1.8660\ \mu\text{m},\quad \Delta z = 1.8639\ \mu\text{m}$$

*   **Glomus Cell Soma:** $8.0$ to $15.0\ \mu\text{m}$ diameter [97], which is $4.3$ to $8.0$
    pixels.
*   **Glomus Cell Nucleus:** $4.4$ to $6.0\ \mu\text{m}$ [97], a non-immunoreactive dark core
    spanning $2.3$ to $3.2$ pixels.
*   **The "Doughnut" Morphology:** TH is cytoplasmic, so each soma appears as a bright ring
    around a dark nuclear core [45, 97]. Preserving this is the single most critical
    requirement.

**Confirmed on this data.** The doughnut is real and resolvable at this voxel size. The mean
radial intensity profile about 2181 detected cores in WKY-A parenchyma rises from the centre to
a peak at $r = 4$ px ($7.46\ \mu\text{m}$), a **core-to-ring contrast of 79.4%** in the raw
data and **82.5%** in the finished output. The design premise holds; revision 1's
implementation was what eroded it.

---

## 2. Phase-by-Phase Protocol

### Phase 1: Channel selection

The acquisitions are single files with axes `ZCYX` and two channels. **Channel 0 is lectin,
channel 1 is TH**, verified by byte-comparing channel 0 against the previously extracted
`C1-*_vessels.tif`.

No manual Fiji split is needed, and `skimage.io.imread` must not be used: it returns a 4D array
and revision 1's script rejected it. `preprocess_th.py` extracts the channel by axis label.

The two channels are one acquisition on an identical grid, so they are **co-registered by
construction**. No registration step is needed for TH-to-vessel distance work, and none should
be introduced.

### Phase 2: Z-axis profile. **CHANGED: do not apply bleach correction.**

Revision 1 applied Histogram Matching to every slice. Measured on WKY-C, that multiplies slice
0 by **26.9x** and raises the TH-positive fraction in the near-empty top of the stack from
0.004% to **1.413%**, a 350-fold increase, fabricating cells out of background noise.

The structural problem is worse. After matching, **every slice has the same p99 and the same
TH-positive fraction (~1.40%)**. That is what histogram matching does by definition, and it
hard-codes TH density to be uniform in depth. Any measurement of TH density, TH volume
fraction or cell count per unit volume is then determined by the correction rather than by the
tissue, and because it normalises to whatever each specimen's middle slice contains, the
residual differs per specimen and can align with the group variable.

The three WKY volumes do not even share a profile shape:

| Volume | verdict | peak slice | p99 first quarter | p99 last quarter |
|---|---|---|---|---|
| WKY-A | monotonic decay | 55 / 435 | 7596 | 4226 |
| WKY-B | hump / tissue-extent dominated | 98 / 435 | 7046 | 2639 |
| WKY-C | hump / tissue-extent dominated | 163 / 435 | 1103 | 5636 |

WKY-C's TH signal *rises five-fold with depth*, which is not photobleaching under any reading.

**Do this instead.** Run `preprocess_th.py --input . --diagnose`. The verdict is recorded in QC
and no correction is applied. Only where the verdict is genuinely `monotonic decay` may you pass
`--z-correct multiplicative`, which applies a single scalar gain per slice, capped to
$[0.25, 4]$ so near-empty slices are left alone rather than amplified. It refuses to run on a
hump unless forced. Label at several depths in Ilastik instead; that is the robust answer.

### Phase 3: Denoising. **CHANGED: no blanket median.**

Revision 1 applied a 3D median with $x = y = z = 1$, a $3\times3\times3$ kernel spanning
$5.6\ \mu\text{m}$ against a nuclear core of only $2.4$ to $3.2$ voxels. Measured cost:
core-to-ring contrast falls from **79.4% to 65.2%** and the core floor rises **62%**. It erodes
the one feature the whole strategy depends on.

`preprocess_cb.py` rejected a blanket median on the lectin channel for the same reason. Use
`--remove-outliers`, which replaces only voxels far above their local median, and check from QC
that under 1% of voxels were touched. The default is off; turn it on only if impulse noise is
visible.

### Phase 4: Background subtraction

Glomus cells aggregate into dense nests that trap scattered light [99, 100]. A rolling-ball
subtraction removes that haze.

Use `--rolling-ball 12` (the default). Note this is a **radius**: 12 px is $22.4\ \mu\text{m}$,
so a $44.8\ \mu\text{m}$ ball, comfortably larger than the $15\ \mu\text{m}$ maximum soma
diameter, so somas ride on top of the ball and survive. Revision 1 compared that radius against
a cell diameter, which is the wrong comparison even though the conclusion happened to be safe.

The implementation is `skimage.restoration.rolling_ball`, the paraboloid ImageJ uses, not a flat
`white_tophat` disk. Both channels now use the same one so the two segmentations stay
comparable.

### Phase 5: Normalisation. **CHANGED: anchor inside tissue, and share anchors across the cohort.**

The carotid body occupies a minority of each field. Measured tissue occupancy:

| Volume | occupancy | whole-volume anchors | within-tissue anchors | ratio |
|---|---|---|---|---|
| WKY-A | 9.0% | 3 - 7543 | 702 - 10321 | 1.37x |
| WKY-B | 12.6% | 11 - 8675 | 715 - 13225 | 1.52x |
| WKY-C | 17.7% | 34 - 7236 | 976 - 9558 | 1.32x |

Occupancy spans **2x across three specimens of the same group**. Because a whole-volume
percentile is set largely by how much black space happened to be in frame, the applied gain
tracks a cropping accident, and if WKY and SHR were framed differently that gain difference
aligns with the group variable.

Two steps:

1. Anchors are taken inside a tissue mask by default, which removes the frame dependence.
2. Run `--diagnose` across the whole cohort first. It prints a cohort-wide anchor pair; pass it
   back as `--anchors LO HI` so the gain is identical for every specimen.

Both the anchors used and the whole-volume anchors that would have been used are written to QC,
so the sensitivity can be measured later by re-running with them perturbed.

### Phase 6: Soma Difference of Gaussians. **CHANGED: keep the sign.**

Revision 1 computed $G(\sigma{=}1) - G(\sigma{=}3)$ and then clipped at zero. **The dark nuclear
core is exactly where that difference is negative.** Measured: the raw DoG is negative at
**99.8%** of cores, and the clip set 99.8% of them to exactly 0, while 55.8% of background was
also exactly 0. The channel therefore mapped "inside the nucleus" and "outside the cell" to the
same number, in a pipeline whose entire purpose is separating them.

The map is now kept signed and scaled symmetrically about zero. In the finished output:

| location | signed DoG |
|---|---|
| nuclear core | **-0.2298** |
| cytoplasmic ring ($r = 4$ px) | +0.0197 |
| background | -0.0000 |

Core and background are now separated by 0.23, where they were separated by exactly 0.

Use `--split-dog` if you prefer two non-negative channels (positive and negative parts) instead
of one signed channel. Sigmas are `--dog-sigmas 1.0 3.0` by default; the measured ring peak sits
at 4.0 px, so that is the scale to tune against.

### Phase 7: Tiling. **CHANGED: do not tile.**

Revision 1 tiled to a 256 grid with `mode='reflect'` padding. Measured padding overhead:

| Volume | tiles | fraction of the tiled volume that is mirrored duplicate |
|---|---|---|
| WKY-A | 4 | 12% |
| WKY-B | 4 | **52%** |
| WKY-C | 2 | 39% |

For WKY-B more than half of what Ilastik would see is a mirror image of real tissue, which is
structurally plausible and will be learnt and labelled as though it were real. Non-overlapping
tiles also split cells at borders, so per-tile counts cannot be summed.

These volumes are 35 to 100 M voxels and process whole in 9 to 24 seconds each, well inside
available memory. If tiling is ever needed for an unbinned dataset, overlap by at least three
times the largest feature sigma, crop the halo on stitch, and pad with a constant, never a
reflection.

---

## 3. Ilastik Training and Segmentation Strategy

Inputs are `<name>_TH_ilastik.h5`, dataset `data`, axes **`zyxc`** with an integer `typeFlags`
axistag that has been verified to parse in ilastik 1.4.1's own vigra. Revision 1 wrote a
`{"key", "type", "description"}` form which raises `KeyError: 'typeFlags'`, after which ilastik
silently falls back to guessing axes from shape. Channel order is
`0: grayscale, 1: soma_dog_signed`.

### Feature selection
*   **Colour/Intensity:** all sigmas on both channels.
*   **Edge (Laplacian / Gradient):** $\sigma = 1.0$ (3D) for membranes and nuclear envelopes,
    $\sigma = 3.5$ for outer boundaries.
*   **Texture (Hessian / Tensor):** $\sigma = 1.6$ (3D) and $\sigma = 3.5$ (2D) for the rounded,
    blob-like boundaries.

### The 2-class labelling strategy. **CHANGED from the 3 and 4 class schemes.**

Revision 1 proposed three classes and an earlier revision of this section proposed four, both
aimed at separating touching somas so that a watershed could count individual cells. Checked
against what the hypotheses actually consume, that is work with no consumer.

Every H1 and H2 analysis that uses this channel asks for a TH-positive **volume, voxel set or
cluster boundary**: parenchymal volume and length density (H1 1.3), tissue-to-vessel distance
(H1 1.5), flow overlay (H2 2.1), which edges supply the clusters (H2 2.2), metabolic rate
assignment and hypoxic fraction (H2 2.3), depletion within the boundaries (H2 2.4). **None of
them counts cells.** So the project is two-class, exactly parallel to the vessel one:

1. **glomus.** The bright cytoplasmic rings **and** the dark nuclear cores inside them.
2. **background.** Empty space outside the clusters, and the dim gaps between touching cells.

Two decisions inside that scheme are worth 9 to 15% of the measured volume, so make each once
and hold it across all six specimens.

**Nuclear cores are glomus.** Every analysis means "where are the glomus cells", not "where is
the TH protein". Calling cores background yields hollow shells, and a 3D fill cannot reliably
close a shell only 2 to 3 voxels thick. This is also precisely what a plain intensity threshold
gets wrong, and the reason channel 2 is kept signed: it reads about $-0.23$ in a core against
about $0.00$ in true background, so the classifier has a feature that tells them apart.

**Intercellular gaps are background.** They are genuinely extracellular.

Measured on the six preprocessed volumes, the interior region (cores plus gaps) is 8.7 to 15.1%
of whole-cell volume, averaging 10.9% in WKY against 13.1% in SHR. The direction is what denser
nests would produce, but the per-specimen ranges overlap and n = 3 per group gives a permutation
p floor of 0.10, so **no group difference is claimed**. The number establishes that the choice is
worth about a tenth of the headline volume and has to be consistent, not that it needs more
classes.

If individual cell counts are ever wanted, for instance to test hyperplasia as a cell-number
claim rather than a volume claim, that is a different segmentation problem and a harder one than
it looks: a whole soma is only 4.3 to 8.0 voxels across at this resolution.

---

## 4. Restricting to the carotid body. **NEW.**

**TH is not specific to glomus cells.** Sympathetic neurons and nerve fibres are also
TH-positive. In WKY-A the brightest TH structure in the entire stack is a large fibrous body
around $z = 40$ to $140$, well away from the CB parenchyma at $z \approx 200$ to $350$.

This is not hypothetical. The first pass of the doughnut measurement for this review ran on that
slab and concluded there was no doughnut at all, with only 7.7% of detections showing a core.
Repeating it on true parenchyma gave 79.4%. The premise was fine; the region was wrong.

Any pipeline that treats "TH-positive" as "glomus cell" will make the same error at scale, and
the automated z-profile verdict for volume A is describing that nerve structure rather than the
CB. **Define a CB region of interest before any quantification**, even a coarse manual one, and
record it alongside the QC.

---

## 5. Running it

```bash
# 1. Diagnose the whole cohort first. Reports z-profile, tissue occupancy and
#    both anchor choices, and prints a cohort-wide anchor pair.
python3 preprocess_th.py --input . --diagnose

# 2. Real run, with the cohort anchors from step 1 so the gain is identical
#    for every specimen.
python3 preprocess_th.py --input . --output-dir ilastik_inputs_th \
        --anchors 715 10321 --save-tif

# 3. Tests, which encode every defect above so it cannot come back.
python3 -m pytest test_preprocess_th.py -v
```

Each volume writes `<name>_TH_qc.json` recording parameters, shape, z-profile verdict, tissue
occupancy, both anchor sets, DoG scale and timing. Revision 1 recorded nothing. Given that the
H2 assessment had to withdraw nine claims once measurements existed, that file is what makes
the same correction possible here.

Downstream, hand the Ilastik probability export to `prob_to_mask.py`, which expects channel-last
`zyxc` and the dataset named `data`, both of which this pipeline now writes.

---

## 6. Step by step in the Ilastik GUI

The six files in `ilastik_inputs/` are ready to label. Verified before writing this section:
dataset `data` at the file root, axistags `zyxc` parsing in ilastik 1.4.1's own vigra on all
six, `float32`, two channels named `grayscale` and `soma_dog_signed`, chunked
`(32, 128, 128, 2)`.

### Before you start: disk and memory

Both are comfortable as of 2026-08-18: about 23 GB free on `/home`, and 19 GB of the 31 GB of
RAM available. The probability export across all six volumes:

| export | size across the six |
|---|---|
| 2 classes, float32 | 3.66 GB |
| 2 classes, **uint8** | **0.91 GB** |

Either fits now, but step 7 exports `uint8`: four times smaller, four times faster to read, and
`prob_to_mask.py` already handles it by detecting `max > 1.5` and rescaling by 255, so nothing
downstream changes.

This was tighter when the inputs were written, at 6.5 GB free, which is why the earlier
four-class float32 figure of 7.31 GB mattered. Two classes removes the constraint entirely.

Closing the browser before a long export is worth more than any ilastik memory setting.

### Step 1: New project, saved in the right place

Launch ilastik, choose **Pixel Classification**, and save the project as
`glomus_cell_segmentation.ilp` **inside `~/Desktop/LCFM Images/ilastik_inputs/`**.

The location is not cosmetic. An ilastik project registers its datasets by path relative to
itself, so a project saved elsewhere breaks the moment the folder moves. `vessel_segmentation.ilp`
already lives there for the same reason.

This is a **separate project from the vessel one**. Do not add TH lanes to
`vessel_segmentation.ilp`: its classifier is trained on different channels and a different
number of them.

### Step 2: Add all six volumes as separate lanes

**Input Data** applet, `Add New...` then `Add separate Image(s)...`, and select all six
`*_TH_ilastik.h5` files at once. If prompted for the internal dataset, choose `data`.

Check the axes column reads `zyxc` and that each lane shows 2 channels. Add all six now,
before labelling. A volume the classifier was never shown cannot be part of a pooled training
set, and `ImageLynx.specimens.verify_classifier` rejects a project with unregistered lanes.

### Step 3: Features, in 3D only

**Feature Selection**, `Select Features...`. Suggested set, following the scales measured on
this data (ring peak at 4.0 px, nucleus 2.4 to 3.2 px):

* **Colour/Intensity**: sigma 0.3, 0.7, 1.0, 1.6, 3.5, 5.0
* **Edge**: sigma 1.0, 1.6, 3.5
* **Texture**: sigma 1.6, 3.5

Leave every feature computing in **3D**. Per-slice 2D features give z-anisotropic predictions
and staircase artefacts downstream, and `verify_classifier` refuses a project that uses them.

### Step 4: Two labels, in this order

Create two labels and **keep this order**, because the exported probability channels follow
label order and the index is what downstream code selects on:

| label | name | what to paint | signed DoG reads |
|---|---|---|---|
| 1 | glomus | the bright rings **and** the dark cores inside them | positive on the ring, about -0.23 in the core |
| 2 | background | empty space outside the clusters, and the gaps between touching cells | about 0.00 |

See section 3 for why this is two classes rather than four, and for the two judgement calls
inside it that are worth about a tenth of the measured volume.

### Step 5: Label every lane, at three depths each

Every one of the six lanes needs labels, at a minimum of two depths and preferably three. This
is the project's existing standard and it exists because of a real failure: the first trained
vessel project had all 454 of its labels on WKY-A, on a single z slice, with the other five
lanes registered and empty. A decision boundary learned from one cohort and applied to the
other reintroduces exactly the confound the study is trying to measure.

Tissue does not sit at the same depth in every stack, so label where there is tissue:

| specimen | z extent | tissue-bearing z | suggested depths |
|---|---|---|---|
| WKY-A | 435 | 0 - 430 | 86, 215, 344 |
| WKY-B | 435 | 15 - 360 | 84, 187, 291 |
| WKY-C | 435 | 100 - 430 | 166, 265, 364 |
| SHR-A | 495 | 0 - 320 | 64, 160, 256 |
| SHR-B | 495 | 70 - 450 | 146, 260, 374 |
| SHR-C | 495 | 0 - 345 | 69, 172, 276 |

**Avoid the TH-positive structures that are not carotid body.** Sympathetic neurons and nerve
fibres label too, and in WKY-A the brightest TH structure in the whole stack is a fibrous body
at z = 40 to 140, well away from the parenchyma at z = 200 to 350. If you paint it as
glomus the classifier will find it everywhere. Either avoid it, or give it a third label of
its own so that it is explicitly not glomus.

Use a small brush, 1 or 2 px. Turn on **Live Update** periodically rather than continuously;
it recomputes features over the whole lane and is slow on these volumes.

### Step 6: Check before exporting

With Live Update on, step through z and confirm that adjacent cells in a dense nest are
separated by a Boundary or Nucleus prediction rather than fused into one blob. That separation
is the entire purpose of the two-channel input, and it is cheaper to fix with more labels now
than to discover after a six-volume export.

### Step 7: Export probabilities as 8-bit

**Prediction Export**, `Choose Export Image Settings...`:

* Source: **Probabilities**
* Convert to Data Type: **unsigned 8-bit**, and tick **Renormalize** so the full range is used
* Format: **hdf5**
* Output path:
  `~/Desktop/LCFM Images/ilastik_probabilities/{nickname}_Probabilities.h5`

Then `Export All Lanes`. This matches the naming the vessel exports already use and keeps the
whole cohort inside 1.83 GB.

Headless equivalent, if you would rather not hold the GUI open:

```bash
~/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh --headless \
  --project="$OUT/glomus_cell_segmentation.ilp" \
  --export_source="Probabilities" \
  --export_dtype=uint8 \
  --output_format=hdf5 \
  --output_filename_format="$DATA/ilastik_probabilities/{nickname}_Probabilities.h5" \
  "$OUT"/*_TH_ilastik.h5
```

Note that `ImageLynx.io.ilastik.run_ilastik_headless_segmentation` is **not** the right helper
here: it exports `Simple Segmentation`, not `Probabilities`, and `prob_to_mask.py` expects
probabilities.

### Step 8: Threshold on evidence, not by eye

```bash
python3 prob_to_mask.py --prob ".../CB3-WKY-CB-A-2x2x2_TH_ilastik_Probabilities.h5" \
                        --channel 0 --sweep
```

`--channel 0` is glomus under the label order in step 4. The sweep prints the fragmentation
curve; pick the operating point just above where the component count starts climbing steeply.

One caveat carried over from the vessel path: `prob_to_mask.py` fills 3D cavities by default,
which for glomus cells would fill the nuclear cores you worked to preserve. Pass
`--no-fill-holes` unless you specifically want whole-cell masks including nuclei, in which
case combining channels 0 and 1 is the more honest route.

### A note on validation

`ImageLynx.specimens.verify_classifier` enforces all of the above: label order, no 2D
features, all six lanes registered, every lane labelled, at least two depths per lane. Pass
`channel="th"` to check the TH project:

```bash
venv/bin/python -c "
from ImageLynx.specimens import verify_classifier
print(verify_classifier(channel='th')['group_label_counts'])"
```

Run it before using a TH segmentation for any measurement. It raises `ValueError` listing every
problem at once, since relabelling is one trip back to the GUI either way.
