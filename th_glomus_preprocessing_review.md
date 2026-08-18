# TH Glomus Cell Preprocessing: Review Before First Execution

> **Purpose.** Review `process_th_glomus_cells.py` and `th-glomus-cell-preprocessing-guide.md`
> (both dated 2026-08-18, in `~/Desktop/LCFM Images/CB3-WKY/raw_cb_images/`) against the
> established lectin pipeline in the same directory and against the actual CB3 image data,
> before either is run on the cohort.
> **Companions.** `h1_pipeline_capability_assessment.md`, `h2_pipeline_capability_assessment.md`.
> **Reference implementation.** `preprocess_cb.py`, `make_ilastik_input.py`, `prob_to_mask.py`.
> **Test data.** `CB3-WKY-CB-{A,B,C}-2x2x2.tif`, TH channel.
> **Method.** Every numbered finding below was executed against the real volumes, not inferred
> from reading. Where a reading-only prior was overturned by measurement, that is recorded.

## Document status

STATUS — Review complete and all findings applied. Verified on the real cohort.

The TH channel is the Tier 0 blocking item identified in the H2 assessment, and it also blocks
H1 sections 1.3 and 1.5, so this is the right next thing to get right.

### What was applied

`process_th_glomus_cells.py` is superseded by **`preprocess_th.py`**, which imports its shared
stages from `preprocess_cb.py` rather than reimplementing them, so both channels of one
acquisition are treated identically. `th-glomus-cell-preprocessing-guide.md` was revised to
revision 2 to match. **`test_preprocess_th.py`** carries 20 tests, one per finding; all four
attempted mutations (re-clipping the DoG, reverting the axistags, anchoring on the whole volume,
reintroducing reflect padding) were confirmed to fail the suite.

Verified against the real data, not just the tests:

* the shipped `zyxc` axistags parse in ilastik 1.4.1's own vigra
* core-to-ring contrast in the finished output is **82.5%**, up from 79.4% in the raw data,
  because tissue-anchored normalisation stretches the tissue range instead of compressing it
  against the empty background
* the signed DoG reads **-0.2298** at nuclear cores against **-0.0000** in background, a
  separation of 0.23 where the clipped version gave exactly 0
* all three WKY volumes process whole in 9 to 24 s each

### Cohort run: all six specimens

Both groups are processed and consolidated into `~/Desktop/LCFM Images/ilastik_inputs`,
alongside the existing vessel inputs. Shared anchors `--anchors 708 10578`, the median of the
six per-volume tissue anchors, so the applied gain is identical for every specimen.

Channel identity was confirmed twice more, independently of the WKY-C check: WKY-B raw channel 1
is byte-identical to the extracted `C2-CB3-WKY-CB-B-2x2x2_glomus_cells.tif`, and the SHR channels
were identified morphologically (channel 0 tubular network, channel 1 rounded cell nests).
SHR z-spacing is 1.86412 um against WKY's 1.86386, a 0.014% difference that is immaterial.

| Volume | z-profile verdict | tissue occupancy | tissue anchors | core-to-ring contrast |
|---|---|---|---|---|
| WKY-A | monotonic decay | 9.0% | 702 - 10321 | 82.6% |
| WKY-B | hump | 12.6% | 715 - 13225 | 77.3% |
| WKY-C | hump | 17.7% | 976 - 9558 | 71.8% |
| SHR-A | hump | 8.7% | 651 - 11240 | 77.9% |
| SHR-B | hump | 10.6% | 321 - 8156 | 83.6% |
| SHR-C | monotonic decay | 7.9% | 805 - 10836 | 80.5% |

Two things worth reading off this table.

**Four of six volumes are humps**, which is the shape for which revision 1's histogram matching
was most destructive, and the two that are not are the two whose peaks sit at opposite ends
(WKY-A at slice 55, SHR-C at slice 33). There is no single depth correction that suits this
cohort, which is why none is applied.

**The normalisation defect would have been group-differential on this actual cohort.** Tissue
occupancy averages 13.1% in WKY against 9.1% in SHR, so the whole-volume anchor that revision 1
used carries a systematic gap between the groups:

| anchoring | WKY mean high anchor | SHR mean high anchor | systematic gap |
|---|---|---|---|
| whole volume (revision 1) | 7818 | 6666 | **17.3%** |
| within tissue | 11035 | 10077 | 9.5% |
| shared cohort anchor | 10578 | 10578 | **0% by construction** |

With n = 3 per group none of this is a significant difference, and the permutation p floor is
0.10, so no claim is made that WKY and SHR were framed differently. The point is the size: a
17.3% systematic gain difference aligned with the group variable sits inside the 27 to 40% range
of the H1 effects it would contaminate, and it is removed by construction rather than argued
away. The per-volume anchors that would have been used are still in each QC file, so the
sensitivity can be measured directly.

The doughnut survives comparably in both groups, 77.2% mean contrast in WKY against 80.7% in
SHR, so the feature the segmentation depends on is not degraded in one arm.

### The unbinned acquisition exists, but only for SHR

Review point C8 asked whether the 0.933 um data exists for these specimens. It does, as
`CB3-SHR-CB-{A,B,C}-1x1x1.tif`, two channels each. **There is no WKY equivalent**: the only
unbinned WKY file is `C1-CB3-WKY-CB-A-1x1x1_vessels.hdf5`, a vessels-only extraction of a single
specimen, with no TH channel.

So the unbinned data cannot be used for a WKY against SHR comparison, and the binned 2x2x2 stacks
are the only common ground. That settles C8: proceed on the binned data. The unbinned SHR stacks
remain useful for a within-SHR resolution check, for instance to measure how much of the
core-to-ring contrast is lost to binning, but not for anything that crosses the groups.

### Two further bugs, found by executing rather than reading

Neither was visible in the review pass. Both are in `preprocess_cb.py`, so both affected the
lectin pipeline too.

* **`_rb_slice` passed `workers=1` to `skimage.restoration.rolling_ball`.** That keyword was
  renamed to `num_threads` in scikit-image 0.20, and the installed version is 0.24, so the call
  raised `TypeError` immediately. The lectin preprocessing could not have run on this machine as
  it stood. Now resolved by signature inspection so it works either way.
* **`write_h5` hard-coded a 128x128 chunk shape.** h5py rejects a chunk larger than the dataset
  in any dimension, so any volume narrower than 128 crashed. The three CB3 volumes are all wider
  than that, so it never bit, but it made the writer untestable on small inputs. Chunks are now
  clamped to the data shape, which is a no-op at real sizes: both a lectin and a TH output still
  chunk at (32, 128, 128, C). The identical line in `make_ilastik_input.py` was fixed too.

## Headline

The guide's *morphological premise is correct and the data supports it*. Glomus cells do appear
as bright cytoplasmic rings around dark nuclear cores at this resolution: measured core-to-ring
contrast is **79.4%**, with the ring peaking at r = 4 voxels (7.46 um). Building the pipeline
around preserving that doughnut is the right call.

The *implementation does not yet deliver it*. Three defects are blocking, in the sense that
running the script as written produces either unreadable output or fabricated signal. Four more
are high-value. The two most serious both come from the same source: the script re-applies steps
that `preprocess_cb.py` had already tested on this exact data and deliberately rejected.

---

## Blocking

### R1. The Ilastik axis tags do not parse, and the failure is silent

`save_hdf5_with_axis_tags` writes axis entries of the form
`{"key": "z", "type": "space", "description": "depth"}`. Vigra requires an integer `typeFlags`.
Tested against the vigra that ships inside the user's own ilastik 1.4.1rc2:

```
  lectin pipeline (typeFlags)    -> parsed as ['z','y','x','c']  types=['Space','Space','Space','Channels']
  TH script (type: space)        -> FAILED: KeyError: 'typeFlags'
```

Every tile would be written with tags ilastik cannot read, so ilastik falls back to guessing
axes from shape. For a 4D `(Z, C, Y, X)` array that guess is not reliably `zcyx`, and a wrong
guess silently transposes the channel axis into a spatial one. Nothing raises.

Three linked fixes:

* Use the `_TYPEFLAG` construction from `preprocess_cb.write_h5`, which is verified to parse.
* Write **channel-last `zyxc`**, not `zcyx`. `prob_to_mask.load_probability` takes
  `arr[..., channel]` on the strength of the comment "ilastik exports channel-last for zyxc
  input". Feeding it a `zcyx` project silently selects along the wrong axis.
* Name the dataset `data`, not `volume`, to match what the downstream tools open.

Also carry `voxel_size_um_zyx` and `channel_names` across, as `preprocess_cb.write_h5` does.
Those attributes are what keeps the calibration self-describing into the skeletonisation stage.

### R2. Histogram-matching bleach correction fabricates TH signal, and this data contradicts its premise

`preprocess_cb.diagnose_z` exists specifically to decide whether axial decay is real
attenuation or just the extent of the tissue block, and its advice for the latter is
"Do NOT apply bleach correction", with "Never histogram matching" attached to both branches.
`match_z_bleaching` applies it unconditionally.

Running that diagnosis on the TH channel, the three WKY volumes do not even agree with
each other:

| Volume | verdict | peak slice | p99 first quarter | p99 last quarter |
|---|---|---|---|---|
| WKY-A | monotonic decay | 55 / 435 (0.127) | 7596 | 4226 |
| WKY-B | hump / tissue-extent dominated | 98 / 435 (0.226) | 7046 | 2639 |
| WKY-C | hump / tissue-extent dominated | 163 / 435 (0.376) | 1103 | 5636 |

WKY-C's TH signal **rises five-fold with depth**. That is not photobleaching under any reading.

Applying the script's own `match_z_bleaching` to WKY-C and measuring:

| slice | p99 before | p99 after | gain |
|---|---|---|---|
| 0 | 270 | 7263 | **26.9x** |
| 25 | 362 | 7263 | 20.1x |
| 50 | 538 | 7270 | 13.5x |
| 100 | 2864 | 7258 | 2.5x |
| 217 (reference) | 7258 | 7258 | 1.00x |
| 434 | 5455 | 7257 | 1.3x |

Holding a fixed cell threshold taken from the mid-stack tissue core:

| region | TH-positive before | after | change |
|---|---|---|---|
| slices 0-99 (sparse, dim) | 0.004% | 1.413% | **350x** |
| slices 180-259 (tissue core) | 1.000% | 1.402% | 1.4x |
| slices 380-434 (deep) | 0.098% | 1.403% | 14.4x |

Two things are wrong here, and the second is worse than the first.

The obvious one: in slices that contain almost no tissue, background noise is multiplied by up
to 27 and promoted to cell-level intensity, manufacturing a 350-fold increase in TH-positive
voxels out of nothing.

The structural one: after correction **every slice has the same p99 (~7258) and the same
TH-positive fraction (~1.40%)**. That is not a side effect, it is what histogram matching does.
If any H1 or H2 quantity is TH density, TH volume fraction, or glomus cell count per unit
volume, this step *hard-codes the answer to be uniform in z* and erases the real variation. It
then does so relative to whatever the middle slice of each stack happens to contain, which
differs per specimen, so the residual is group-differential. That is the same class of confound
as the fabricated constriction found in the H2 assessment, which suppressed WKY flow 16.9%
against SHR 13.3%.

Recommendation: drop Phase 2 entirely. Run `diagnose_z` per volume and record the verdict in
QC. If a volume genuinely shows monotonic decay, apply a single multiplicative scale factor per
slice, which preserves relative intensities within the slice, and never histogram matching.

### R3. The script cannot read the acquisition files as they exist

`io.imread` on `CB3-WKY-CB-A-2x2x2.tif` returns `ndim=4, shape=(435, 2, 456, 507)`, so the
`ndim != 3` check raises immediately. The files are `ZCYX` with two channels.

Channel identity, verified rather than assumed: channel 0 is byte-identical to the existing
`C1-CB3-WKY-CB-C-2x2x2_vessels.tif`, so **the TH channel is index 1**.

Reuse `preprocess_cb.read_vessel_channel`, which already does axis-aware channel extraction via
`tifffile` series metadata, rather than requiring a manual Fiji split first. `tifffile` is also
the right reader here because it exposes the ImageJ spacing metadata (1.8638551724 um) that
confirms calibration per file instead of trusting a hard-coded constant.

---

## High value

### R4. Clipping the DoG destroys exactly the signal the channel exists to capture

`generate_soma_ring_enhancer` computes `G(1) - G(3)` and then does `np.clip(dog, 0, None)`.
The dark nuclear core is where that difference is *negative*. Measured on the WKY-A CB
parenchyma, at 2181 detected nuclear-core centres:

* raw DoG is negative at **99.8%** of cores, so the core signal is present before the clip
* the clip sets **99.8%** of cores to exactly 0
* **55.8%** of background voxels are also exactly 0
* channel 2 reads **4** at cores against **11789** on the r = 4 ring

So channel 2 maps "inside the nucleus" and "outside the cell entirely" to the same number.
The classifier is handed a feature that cannot separate them, in a pipeline whose entire
purpose is to separate them.

Keep the signed DoG. A feature that swings strongly positive on the ring and strongly negative
in the core is maximally discriminative, and Ilastik handles signed float input. If a
non-negative channel is wanted, supply the positive and negative rectified parts as two
channels rather than discarding one.

### R5. Per-volume normalisation makes the gain depend on how much empty frame was acquired

The carotid body occupies a minority of each field. Measured occupancy, and the resulting
anchor error:

| Volume | tissue occupancy | p99.65 whole volume | p99.65 within tissue | ratio |
|---|---|---|---|---|
| WKY-A | 16.1% | 7545 | 9668 | 1.28x |
| WKY-B | 17.4% | 8676 | 12536 | 1.44x |
| WKY-C | 28.9% | 7237 | 8841 | 1.22x |

Occupancy spans 1.8x across three specimens *from the same group*. Because
`normalize_dynamic_range` takes its anchor from the whole-volume percentile, the applied gain is
set partly by how much black space happened to be in frame, which is a cropping accident. The
anchor error varies 1.22x to 1.44x between specimens on that basis alone.

This is tolerable if every downstream measurement is geometric (cell count, cell volume,
spatial relationship to vessels) and the segmentation is robust to the gain. It is not tolerable
if any measurement is intensity-based, and it is dangerous if WKY and SHR were framed
differently, because then the gain difference aligns with the group variable.

Two changes, both cheap:

* Take anchors from within a tissue mask, or from a shared cohort-wide anchor, not from the
  whole volume.
* Record the anchors in QC and re-run segmentation with them perturbed, to measure the
  sensitivity. This is the same treatment the H2 assessment gave to the calibre threshold, and
  it is the only way to know whether the number matters.

Also, `normalize_dynamic_range` uses `np.min(volume)` for the low anchor while documenting
"0.35% saturation". `robust_normalise` in the lectin pipeline uses a percentile at *both* ends
deliberately, so that a handful of cold or hot voxels cannot compress the range. Make it
symmetric.

### R6. The tiling costs more than it gives, and over half of one volume would be fabricated

These volumes are 35M to 100M voxels. `preprocess_cb.py` states it "deliberately does NOT
denoise by default and does NOT tile", and it processed these same stacks whole. The TH script
reintroduces tiling with `mode='reflect'` padding:

| Volume | tiles | reflect padding | fraction of tiled volume that is mirrored duplicate |
|---|---|---|---|
| WKY-A | 2x2 = 4 | Y+56, X+5 | 12% |
| WKY-B | 2x2 = 4 | Y+155, X+161 | **52%** |
| WKY-C | 2x1 = 2 | Y+197, X+1 | 39% |

For WKY-B, more than half of what Ilastik would see is a mirror image of real tissue. Mirrored
cell nests are structurally plausible and the classifier will happily learn and label them.

Beyond the padding, non-overlapping tiles split cells at the borders, so per-tile counts cannot
simply be summed, and each tile's boundary features are computed against a different extension.

Simplest fix: do not tile. Memory here is a few GB, well inside the 31 GB available. If tiling
is kept for a future unbinned dataset, overlap tiles by at least three times the largest feature
sigma, crop the halo on stitch, and pad with a constant rather than a reflection.

### R7. The 3x3x3 median costs 18% of the doughnut contrast

Measured on the WKY-A CB parenchyma, mean radial profile about 2181 detected cores:

| r (voxels) | r (um) | no median | 3x3x3 median |
|---|---|---|---|
| 0 | 0.00 | 4418 | 7179 |
| 1 | 1.87 | 8513 | 10108 |
| 2 | 3.73 | 16239 | 16048 |
| 4 | 7.46 | **21478** | **20657** |
| 6 | 11.20 | 19369 | 18918 |

Core-to-ring contrast falls from **79.4% to 65.2%**, retaining 82%. The core floor rises by 62%.

This is a cost, not a catastrophe, but the guide itself calls preserving the doughnut "the
single most critical requirement", and the nucleus is only 2.4 to 3.2 voxels across against a
3-voxel kernel. `preprocess_cb.remove_outliers` exists precisely because a 3x3x3 median at this
voxel size was judged too destructive for the lectin channel, and it replaces only voxels far
above their local median. Use that, and confirm from QC that under 1% of voxels are touched.

---

## Things to consider

**C1. TH is not specific to glomus cells, and this already caused a wrong measurement.**
In WKY-A the brightest TH structure in the stack is a large fibrous body around z = 40 to 140,
well away from the CB parenchyma at z = 200 to 350. Sympathetic neurons and nerve fibres are
also TH-positive. My first doughnut measurement ran on that slab and concluded there was no
doughnut at all, with only 7.7% of detections showing a core; repeating it on true CB parenchyma
gave 79.4% contrast. The premise was fine, my region was wrong. Any pipeline that treats
"TH-positive" as "glomus cell" will make the same error at scale. A CB region-of-interest step,
even a coarse manual one, needs to come before quantification, and `diagnose_z` on the TH
channel is measuring that nerve structure rather than the CB in volume A.

**C2. The 3-class scheme yields hollow cells.** Class 2 deliberately merges nuclei with external
background, so the segmented object is a cytoplasmic shell. Nuclear volume fraction is roughly
(5 / 11.5)^3, about 8%, so any cell-volume or TH-volume measurement is biased low by that much
unless the shells are filled. A 3D fill will not close them reliably at 2 to 3 voxels. Prefer
four classes (cytoplasm, nucleus, intercellular boundary, background), which also gives you the
nucleus as a natural watershed seed and makes counting far more robust than seeding from the
cytoplasm.

**C3. `white_tophat` with a flat disk is not the rolling ball.** The status message says
"Rolling Ball radius" but the call is `skimage.morphology.white_tophat` with `disk(12)`;
`preprocess_cb.py` uses `skimage.restoration.rolling_ball`, which is the paraboloid ImageJ
implements. They are different filters with different aggressiveness. Also, radius 12 px is
22.4 um *radius*, so a 44.8 um element; the guide compares that radius against a cell
*diameter* of 15 um, which is the wrong comparison even though the conclusion happens to be
safe. Pick one implementation across the two channels so the two segmentations stay comparable.

**C4. Nothing is recorded.** `preprocess_cb.py` writes `<base>_qc.json` with parameters, shape,
z-profile verdict, normalisation anchors, outlier count and timing. The TH script writes no
provenance at all. Given that the H2 assessment had to withdraw nine claims once measurements
existed, the QC file is what makes that possible. Add it before the first cohort run, not after.

**C5. The channels are co-registered by construction.** TH and lectin are two channels of one
`ZCYX` acquisition, on an identical grid. No registration step is needed for TH-to-vessel
distance work, which is a real advantage for H1 sections 1.3 and 1.5 and worth stating
explicitly so nobody later introduces one.

**C6. Sigma justification.** The guide calls sigma = 3 voxels "the ~5.6 um radius of a glomus
cell". 5.6 um is closer to the soma diameter scale; a cell of 8 to 15 um has a radius of 4 to
7.5 um. The measured ring peak is at r = 4 voxels = 7.46 um. The chosen sigmas are defensible,
but tune them against that measured profile rather than the stated rationale.

**C7. Minor.** `match_z_bleaching` assigns float output from `match_histograms` into an
`np.empty_like` uint16 array, truncating (moot if R2 is applied). `tile_and_export` names its
second argument `ch2_vesselness` in a glomus cell script, and the module docstring says "ZCYXS"
while writing four axes. `np.percentile` runs over the full volume where
`preprocess_cb.percentile_anchors` subsamples to 20M. These are the "names that invert their
meaning" theme from the H2 assessment; cheap to fix now, expensive to trace later.

**C8. Resolution.** The nucleus is 2.4 to 3.2 voxels across in the binned 2x2x2 data. The lectin
protocol refers to an unbinned 0.933 um acquisition. If that exists for these specimens, it
would roughly double every linear measure and make the doughnut far easier to segment. Worth
checking before committing to the binned data, because it changes every pixel-valued parameter
in both scripts.

---

## Suggested order of work

1. R1, R3 and R2 together. Without these the script does not run, or runs and writes files
   Ilastik misreads, or fabricates signal. Reusing `read_vessel_channel`, `write_h5` and
   `diagnose_z` from `preprocess_cb.py` fixes all three and removes the duplication.
2. R6 (stop tiling) and R7 (targeted outlier removal instead of the blanket median). Both are
   deletions, both immediately reduce risk.
3. R4 (signed DoG) and C2 (four classes). These two together are what actually separates
   touching cells, which is the whole point of the design.
4. R5 and C4. Anchor on tissue, write the QC file, then measure the sensitivity of the
   segmentation to the anchor before trusting any cohort comparison.
5. C1. Define the CB region of interest before any quantification.

## Provenance

Claims I made from reading and then overturned by measurement, kept visible:

| Claim | Status |
|---|---|
| The doughnut is not resolvable at 1.866 um; cells are solid blobs | **Withdrawn.** Measured on the wrong slab (a TH-positive nerve structure, not CB parenchyma). On true parenchyma the core-to-ring contrast is 79.4%. |
| Only 7.7% of cells show any interior darkening | **Withdrawn**, same cause. |
| The 3x3x3 median destroys the nuclear core | **Revised.** It costs 18% of the contrast, not all of it. |
