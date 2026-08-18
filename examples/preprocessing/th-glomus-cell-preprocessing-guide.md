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

### The 4-class labelling strategy. **CHANGED from 3 classes.**

Revision 1 merged nuclei with external background into a single class. That makes the segmented
object a hollow cytoplasmic shell: nuclear volume fraction is roughly $(5/11.5)^3 \approx 8\%$,
so any cell-volume or TH-volume measurement is biased low by about that much, and a 3D fill will
not reliably close a shell only 2 to 3 voxels thick.

Separate them:

1. **Cytoplasm/Soma.** Sparse thin strokes on the bright cytoplasmic rings. Do not paint near
   where cells touch.
2. **Nucleus.** Small dots inside the dark central cores *only*. The signed DoG reads about
   $-0.23$ here.
3. **Intercellular boundary.** Thin precise lines in the narrow dim gaps between touching somas.
4. **External background.** Empty space outside cell clusters. The signed DoG reads about
   $0.00$ here, which is what separates it from class 2.

Keeping the nucleus as its own class gives you a natural watershed seed, which is far more
robust for counting than seeding from cytoplasm, and lets you reconstruct whole-cell volume as
cytoplasm plus nucleus.

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
