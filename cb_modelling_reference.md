# CB modelling reference

> **What this is.** A working record of the mathematical and physiological modelling in the
> carotid body simulation pipeline: what it models, what was chosen, why, and what the numbers
> were. Written for future-me. Assumes the project is already understood.
>
> **Code this describes:** branch `cb_pipeline_improvements_sweep`, commit `8a2b81c`.
>
> **What would invalidate this document:** a change to the viscosity law, the boundary selection
> rule, the unit conversion constants, the calibre estimator, or which coupling tier is run.

---

## Start here: questions

| If you are asking… | Go to |
|---|---|
| Which viscosity law actually ran? | §3.2 — there are two, and one overwrites the other |
| Would the other viscosity law change my answer? | §4.4 — not for ratios; yes for absolutes |
| Why is vessel diameter measured by EDT and not FWHM? | §2.6 |
| How much of my diameter distribution was measured rather than fabricated? | §2.6 — the guard refuses at any fabrication |
| Is absolute perfusion physiological? | §13.5 — at 60/20 mmHg it runs up to 1.8× fast; it was 20–100× slow only before open item 12 |
| What pressure boundaries did the published H2 numbers use? | §7.8 and §8.1 — 60/20; the pipeline config reads the same pair since open item 10 |
| Which coupling tier produced the oxygen field? | §6.6 — Tier 1; Tier 2 is unreachable |
| What grid resolution was used, and is it converged? | §6.8 — 3 µm, with vessels mapped over their cross-section; within 0.16 mmHg of 2 µm on all six placed-box networks (re-measured 2026-09-30 after open item 38; 4 µm passes the 0.5 mmHg rule by 0.005 mmHg, 3 µm kept) |
| Why is transit time reported as a ratio instead of a number? | §7.6, then §13.3 |
| Which boundary rule is in force, and how much does it move things? | §2.8, then §13.4 |
| Why can I not quote a glomus hypoxic fraction? | §13.6 — the tissue is not diffusion-limited (**needs review**: since the 2026-09-28 re-run TH hypoxia is 0 everywhere, and PO₂ in TH separates the groups) |
| Is calibre a defensible H1 finding? | §13.8 — no, on the stated argument (**needs review**: at 0.95 the gap is half a voxel, without overlap) |
| Can I use the TH channel for a between-group contrast? | §13.9 — qualified, bounded by sensitivity analysis |
| What is turned off in the model, and why? | §10.6 constriction; §6.6 Tier 2; §2.5 bundle collapse |
| What am I allowed to claim? | **§13.10** |
| What was a given parameter set to? | §10 |
| Which section owns a figure I need to change? | "Where each figure is owned", just below |
| What has never been validated? | §12.4 — nothing has |
| What is still unresolved in the code? | The open items table at the end |

---

## §1 — Scope

### 1.1 What is modelled

A **0.027 mm³** sub-volume of rat carotid body — a 160³-voxel box, about 298 µm on a side — in six
specimens, three WKY and three SHR —
with two channels from one acquisition: lectin for the vasculature and TH for the glomus cells.

The chain is five stages, and each inherits the errors of the one before it:

```
3D probability field
   → binary mask                    §2.2–§2.3
   → 1D vascular graph              §2.4–§2.8
   → network flow + rheology        §3, §4
   → 3D tissue gas transport        §5, §6
   → derived physiological numbers  §7
```

### 1.2 What it is for

**H1, morphology.** Does carotid body microvascular structure differ between WKY and SHR?

**H2, perfusion.** Does the perfusion profile differ, and specifically at the glomus cells?

### 1.3 What the model deliberately does not do

- No vessel compliance — walls are rigid
- No cardiac pulsatility — the solve is steady-state
- No autoregulation, and with constriction disabled, no vasomotor tone at all
- No growth or remodelling
- No neural output — the model stops at gas transport and does not represent chemoreception
- No lymphatic drainage or interstitial fluid flow

### 1.4 How to use this document

Every section states what was chosen and why, with the measured evidence that decided it. Sections
end with an **At a glance** line: the choice, the number, the code, and the test.

**Before quoting any number, read §13.10.** Several quantities here are computed correctly and are
still not reportable.
---

## How to read the tables

**Class** is where the number came from:

| Class | Meaning |
|---|---|
| **(i)** | Literature-derived, with a citation |
| **(ii)** | Empirical correlation transferred from another tissue, species or preparation |
| **(iii)** | Chosen, heuristic, or estimated here |

A number with no class is a defect in this document.

**Sensitivity** is what is known about how much the answer moves when the number moves:

- *measured* — swept, and the result is recorded here or in §13
- *assumed* — held fixed, effect not measured
- *unswept* — never varied, effect unknown

Citations are biblatex keys from the project bibliography. `[CITE]` marks a number that needs a
source added before it can be quoted anywhere.

---

## Where each figure is owned

Some figures appear in several sections, because a reader who lands in one section needs the
context there. That is deliberate. What is not deliberate is two copies drifting apart, so
every load-bearing figure has exactly **one owning section**: the place where it is derived,
and the only place to change it. Everywhere else quotes it with a pointer back.

| Figure | Value | Owner | Also appears in |
|---|---|---|---|
| Processing voxel | 1.8639 × 1.866 × 1.866 µm | §10.1 | §2.1, §2.4, §7.2 |
| ROI size | 160³ voxels = 0.0266 mm³, 4–12% of the imaged block | §2.1 | §1.1, §10.1 |
| Imaged block volume | 0.227 mm³ (WKY-C) – 0.653 mm³ (WKY-A) | §10.1 | §2.1 |
| Frozen segmentation threshold | 0.95, band 0.95 / 0.999 as run (0.90 / 0.95 before the 2026-09-28 re-run) | §2.2 | §2.3, §10.2, §13.8 |
| Config hysteresis band | 0.95 / 0.999, read from `cb_settings` (0.90 / 0.95 until 2026-09-28; 0.65 / 0.75 until open item 1) | §2.3 | §10.2 |
| Capillary calibre window | 4.0–7.0 µm | §2.2 | §10.2 |
| Fragmentation tolerance | 1.5 × median endpoint density | §2.2 | §10.2 |
| Stub / reconnection threshold | 5.6 µm = p99 inscribed radius | §2.5 | §2.4, §10.3, §11.1 |
| β₁ | 307 on the reference subvolume | §2.5 | §7.1, §10.3 |
| Junction exclusion | 3.73 µm ≈ one capillary inscribed radius | §2.6 | §10.5, §11.1, §13.3 |
| Median EDT diameter | 6.37 µm (EDT) against 8.20 µm (FWHM), *r* = 0.245, on the method-choice subvolume; the batch at 0.95 gives per-specimen medians 5.87–7.46 µm (mean 6.70) | §2.6 | §3.1, §10.5 |
| Boundary rule and axis | face rule, axis 1, 1-voxel tolerance | §2.8 | §8.1, §8.2, §13.4 |
| Boundary sensitivity | 8.9% (face) against 73.9% (band) | §2.8 | §13.4 |
| Interior terminal share | 82.8–89.4%, mean ≈ 86% | §8.2 | §2.8, §13.4 |
| Pressure boundaries | 60 / 20 mmHg, arteriolar to venular, H2 and pipeline config alike | §8.1 | §7.8, §10.7, §11, §13.5, §14, open item 10 |
| Systemic haematocrit | 0.45 | §4.1 | §4.3, §6.7, §10.7 |
| Perfusion grid pitch | 3 µm, cross-section vessel mapping; converged to 0.5 mmHg (open item 30) | §6.8 | §6.1, §6.5, §10.9, §13.7 |
| Metabolic rate | `BASE_M_MAX` = 0.05 mmol/L/s, H2 and pipeline config alike | §6.4 | §6.5, §13.6, open item 8 |
| Metabolic contrasts | 1×, 2×, 4× | §6.5 | §7.5, §13.6 |
| Hypoxic thresholds | 5, 10, 20 mmHg | §7.5 | §13.6 |
| Tissue-to-vessel distance | median 4.57–6.19 µm, p90 12.1–46.8 µm | §13.7 | §6.8, §7.2 |
| Oxygen diffusion length | 20 µm at PO₂ 10, 35 at 30, 45 at 50 | §13.6 | §6.8 |
| Total inlet flow | 1.33e6–4.00e6 µm³/s (face rule) | §13.5 | §7.6 |
| Flow-weighted velocity | 977–1,812 µm/s against a physiological 200–1,000 | §13.5 | §8.1, §11, §13.10 |
| Calibre error floor | ±47% on an absolute flow quantity | §13.3 | §7.1, §7.6, §11 |
| Within-specimen floor | ±5.9% | §13.3 | §7.3 |
| Pre-threshold filter cost | median-3 destroys 80% of the vessel | §2.3 | §11.1 |

**In code, these live in `src/ImageLynx/cb_settings.py`**, which is the single owner for every
value a driver needs. `specimens.py` owns the voxel size and the specimen list. Anything not
in one of those two is either a `carotid_image_to_model.py` config default — several of which
disagree with what ran, and are listed as open items — or a figure measured rather than set.

---

## §2 — Geometric model, part 1: from voxels to a binary mask

Everything downstream inherits from this section. Resistance goes as $d^{-4}$ (§13.1), so a choice
made here about where the vessel wall sits is amplified fourfold in every flow quantity.

Each subsection ends with **At a glance** — the choice, the number, the code, and the test.

---

### 2.1 ROI placement

**What it does.** Picks where the analysed sub-volume sits inside each specimen's imaged block.

**What was chosen.** Tissue-centroid placement, computed from each specimen's own data, rather
than a centred box.

**Why.** A matched ROI *size* makes the samples the same size; it does not make them the same
anatomy. The carotid body does not sit in the middle of its imaged block, and it does not sit in
the same place in every block. The stacks are not even the same depth — 435 slices for the three
WKY, 495 for the three SHR — so a peak has to be read as a *fraction* of depth, not as a slice
index. Measured from the six QC records:

| Specimen | Stack depth | Peak slice | Depth fraction |
|---|---|---|---|
| WKY-A | 435 | 230 | 0.529 |
| WKY-B | 435 | 106 | 0.244 |
| WKY-C | 435 | 189 | 0.435 |
| SHR-A | 495 | 157 | 0.317 |
| SHR-B | 495 | 230 | 0.465 |
| SHR-C | 495 | 164 | 0.331 |

The peak runs from a quarter of the way in (WKY-B, 0.244) to just past halfway (WKY-A, 0.529). A
centred box therefore lands mid-organ in one specimen and in its sparse margin in another, and the
resulting density difference is a difference in where the box was put. **That scatter is the
argument for per-specimen placement, and it is large.**

**The group-correlated part is real but small.** WKY peaks at a mean depth fraction of 0.402 and
SHR at 0.371 — a gap of 0.031. Set against a within-WKY spread of 0.285 (0.244 to 0.529), the
cohort gap is roughly a tenth of the scatter it sits inside, and with three specimens per group it
is not separable from that scatter. Read it as a reason not to assume centring is neutral, not as
a measured cohort effect. `tests/test_roi_placement.py` asserts only that the gap exceeds 0.02.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Read the specimen's QC record | — | The axial tissue peak was measured once, in preprocessing; recomputing it here could disagree with the recorded value | **On** | `roi_placement.py:113` |
| 2 | z ← `z_profile.peak_slice` | — | The organ sits a quarter of the way into one block (WKY-B, 0.244) and just past halfway into another (WKY-A, 0.529), so a fixed z samples different anatomy in each | **On**; falls back to `shape[0] // 2` and records `z=volume_centre` | `roi_placement.py:118` |
| 3 | Open the Ilastik input HDF5, **channel 0 only**, over the box's own 160 slices | subsample (4, 2, 2) | The vesselness channels are derived from channel 0 and would pull the centroid towards whichever Sato scale happened to dominate. z is clamped first, so only the slices the box will occupy are read (open item 13) | **On** | `roi_placement.py` (`place_roi`) |
| 4 | Maximum-intensity projection along z | — | Collapses the band to one plane so the lateral centre is not weighted by how many slices happen to contain tissue | **On** | `roi_placement.py:85` |
| 5 | Threshold the projection | 99th percentile | The mean of a background-subtracted volume is dominated by near-zero voxels, which drags the centroid back to the geometric middle | **On** | `roi_placement.py:86` |
| 6 | Intensity-weighted centroid of the survivors → y, x | — | Centres the box on signal rather than on the array — the entire reason placement is computed per specimen | **On**; falls back to the volume centre and records why | `roi_placement.py:92` |
| 7 | Rescale the centroid back to full resolution | × 2 in y and x | The centroid was measured on a subsampled block; the crop needs full-resolution indices | **On** | `roi_placement.py:135` |
| 8 | Clamp the centre so the box fits whole | 160³ | A box hanging over an edge would be silently cropped, making that specimen's sample smaller than the rest | **On, but never fires** — all six centres sit inside the legal window, smallest margin 20 voxels (WKY-C, x) | `roi_placement.py:142` |
| 9 | Centre → fractional offsets for `crop_roi` | — | `crop_roi` takes offsets from the volume centre as a fraction of each dimension, not absolute indices | **On** since open item 27: `carotid_image_to_model.py` places the ROI itself and passes `offsets_zyx` to `crop_roi`, which lands exactly on `bounds` | `roi_placement.py`, `carotid_image_to_model.py` (`_apply_roi_placement`) |
| 10 | Crop the probability field to the ROI | 160³ voxels | Everything downstream counts things inside this box; a matched size is what makes those counts comparable across specimens | **On**: the drivers slice `RoiPlacement.bounds`, the network pipeline `crop_roi` with the step 9 offsets — the same box. The pipeline writes `roi_placement.json`, which the drivers check | `cb_h1_batch.py:84`, `carotid_image_to_model.py` |

Steps 1–2 and 3–7 are independent of each other, which is the point of the next paragraph.

**How — two different rules for z and for y/x.** They are not one 3D centroid.

*Axial (z)* comes from the QC record's `peak_slice`. Nothing is recomputed here — the value is
read out of the specimen's preprocessing JSON, and `place_roi` never opens the volume to check it.

It was derived once, in `diagnose_z`, as **stage 2 of 6 of preprocessing** — on the raw extracted
lectin channel, before background subtraction and before any filtering, so the peak reflects the
acquisition rather than anything the pipeline has since done to it. The rule is a **per-slice
brightness profile**, not a measure of tissue area:

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | For each slice $z$, take the 99th percentile of its intensities, giving $s_z$ | p99 | A slice's mean is dominated by background; the p99 tracks the brightest labelled structure in that slice, which is what "where is the organ" has to mean before anything is segmented | **On** | `preprocess_cb.py:163` |
| 2 | Also record the per-slice median as a background trace | p50 | Stored in the QC record as `background_p50_range` so a rising noise floor is visible; it does **not** enter the peak | **On**, diagnostic only | `preprocess_cb.py:164` |
| 3 | Smooth $s_z$ along $z$ with a moving average | $k=\max(3,\lfloor n/20\rfloor)$, edges held | Un-smoothed, the argmax lands on whichever single slice happened to catch the brightest vessel. $k=21$ slices ($\approx$ 39 µm) for the 435-slice WKY stacks and $k=24$ ($\approx$ 45 µm) for the 495-slice SHR ones, so the peak is a regional maximum rather than one lucky plane | **On** | `preprocess_cb.py:167` |
| 4 | Take $\text{peak\_slice}=\arg\max_z \hat{s}_z$ | — | This is the returned value, and the only part of `diagnose_z` that ROI placement uses | **On** | `preprocess_cb.py:168` |
| 5 | Classify the peak's depth fraction into a verdict | $<0.15$, $>0.85$, else | Separates real axial attenuation from a tissue block sitting mid-stack. The second must **not** be bleach-corrected: scaling up the sparse end slices promotes their background noise to vessel intensity | **On**, but **advice to the operator only** — nothing downstream branches on it | `preprocess_cb.py:172` |

So the peak slice is the brightest *region* of the stack, and the verdict string
`"hump / tissue-extent dominated"` names one of three profile shapes — it is not the method by
which the peak was found.

*Lateral (y, x)* comes from `tissue_centroid_yx` on channel 0, subsampled (4, 2, 2), over the 160
slices the box occupies once z is clamped (open item 13; the whole stack before 2026-09-28):

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Read channel 0, subsampled | (4, 2, 2) | A 4× stride in z is free — the next step collapses z anyway — and 2× laterally is 4× less data for a centroid stable to a voxel | **On** | `roi_placement.py:133` |
| 2 | **Maximum-intensity projection** along z, over the box's z-band | — | Collapses the band to one image, so a slice-rich region cannot outvote a bright one. Summing instead would weight by *how many* slices hold tissue, which is the axial question, already answered | **On** | `roi_placement.py:85` |
| 3 | Threshold the projection at the **99th percentile** | p99 | A plain centre of mass over a background-subtracted volume is mostly background, and background is spread evenly, so it drags the answer to the geometric middle — the exact failure this function exists to avoid | **On** | `roi_placement.py:86` |
| 4 | Weighted centre of mass of the survivors | weights = intensity | Intended to let brighter survivors count for more | **On; inert in four of six, 0.31 µm in WKY-A — see below** | `roi_placement.py:92` |
| 5 | Rescale by the lateral stride | × 2, × 2 | The centroid was measured on a subsampled grid. z needs no rescale: the projection discarded it | **On** | `roi_placement.py:135` |

**The intensity weighting mostly does nothing on this data, and it is worth knowing why.**
`preprocess_cb.py` normalises with `--saturated 0.02`, which clips the brightest 0.02% of voxels
to exactly 1.0. The projection in step 2 takes a maximum over the band's 40 subsampled slices, so a
column saturates if *any* voxel in it does — lifting the saturated share from 0.02% of voxels to
**0.63–1.33% of the projection**. Where that share is at least 1% (WKY-B, WKY-C, SHR-B, SHR-C) the
99th percentile lands exactly on **1.0**, the surviving set is precisely the saturated pixels, and
every one carries the same weight: weighted and unweighted centroids agree to **0.00 µm**. SHR-A
(0.99%) cuts at 0.998 and still agrees to 0.00 µm. WKY-A (0.63%) cuts at **0.902**, and there the
weighting moves the centroid by **0.31 µm**.

The consequence is that the answer is mostly set by *where the saturated specks are*, not by a
graded centre of mass. Below the cutoff the behaviour changes: at the 90th percentile the centroid
shifts by 3.7–68.1 µm (largest in SHR-C), against an ROI 298 µm across. So a smaller `--saturated`
in preprocessing would move ROI placement, and the margin is thinner than it was: over the whole
stack (the rule until open item 13) 1.33–1.53% of the projection saturated, so the cutoff sat on 1.0
in all six. That coupling is not obvious from either module.

**What it buys.** The centroid sits 27–127 µm (laterally) from the geometric centre depending on
specimen, so the step is doing real work — a centred box would be measurably elsewhere.

**Why a projection at all, when the peak slice is already known.** The two rules share no
intermediate. `diagnose_z` reduced each slice to a single number, so the axial profile carries
**no lateral information whatever** — it cannot be re-read for a y or x answer. The two also run
in different programs, at different times, on different data: the axial peak was measured inside
`preprocess_cb.py` on the *raw* extracted channel at stage 2 of 6, while the centroid is measured
here on the *finished* normalised output. Nothing is held in memory between them, so the read in
step 1 is not a re-read — it is the only read.

**The subsample saves memory, not time.** The stated case for `(4, 2, 2)` is that it is cheaper,
and for RAM it is: 25 MB against 402 MB for WKY-A's channel 0, at a cost of 6.9 µm in the answer.
But the HDF5 chunks are `(32, 128, 128, 3)` and **gzip-compressed**, so a stride of 4 in z still
lands in every chunk, and a chunk must be decompressed whole — all three channels — to yield any
element of it. Measured on WKY-A's whole stack (the read before open item 13), both orders, cold
and warm: the strided read takes **4.88 s**
and the full-resolution read **4.58 s**. Striding is very slightly *slower*. Treat the subsample
as a memory decision, because that is the only thing it is.

**The projection spans the box's slices, not the whole stack (open item 13, closed).** Until
2026-09-28 step 2 maximised over every slice, but the ROI is only 160 deep, so tissue that would
never be inside the box still voted on where the box went laterally. `place_roi` now clamps z
first and reads only the 160 slices the box occupies. That moved the lateral centres by **8.3 µm
(SHR-C) to 45.1 µm (SHR-B)**, up to 15% of the ROI's 298 µm width; z is unchanged. The cost is the
one noted above: fewer columns saturate over the band, so in WKY-A the cutoff falls to 0.90 and the
intensity weighting takes effect. Using the single peak slice instead would be clearly worse — the
cutoff drops to 0.42–0.59 in every specimen, the surviving set shrinks by a third, and the answer
moves 17–98 µm (measured on the whole-stack rule, before the fix).

**No vesselness is used, deliberately.** Channel 0 is the background-subtracted grayscale. The
vesselness channels exist in the same file — multiscale **Sato**, fine at σ 1.0/1.4/2.0 px and
coarse at σ 4.0/8.0 px — and are *deliberately not read here*, because they are derived from
channel 0 and would weight the centroid towards whichever filter scale happened to dominate.

**No silent fallback.** If neither the QC record nor the preprocessed volume is reachable it falls
back to the volume centre *and records that in `source`*. A silent fallback would reintroduce
precisely the bias the function exists to remove.

**Steps 8 to 10: turning a centre into voxels.** Steps 1–7 produced a centre. These three turn
it into the slices that are actually read, and two of the three do less than the table used to
claim.

*Step 8 — clamping.* `clamp_centre` refuses to let the box hang over an edge:

```python
half = size // 2
if size >= extent:
    clamped.append(extent // 2)
else:
    clamped.append(int(np.clip(centre, half, extent - (size - half))))
```

For a 160-deep box that legal window is `[80, extent - 80]`. **It has never fired on this
dataset.** All six centres sit inside the window on all three axes, and the tightest case is
WKY-C's x centre at 92 against a lower bound of 80 — a margin of 12 voxels, about 22 µm. It is a
guard, not a step that shapes the result. It matters that it is a guard and not a truncation: a
truncated box would be *smaller* than its neighbours, and a matched ROI size is the one thing this
whole section exists to guarantee.

*Step 9 — fractional offsets.* `centre_to_offsets` re-expresses the centre as a signed fraction of
each dimension:

```python
return tuple(
    float((centre - extent / 2.0) / extent)
    for centre, extent in zip(centre_zyx, shape_zyx)
)
```

**Since open item 27 the network pipeline consumes this.** Given `--roi-voxels`,
`carotid_image_to_model.py` calls `place_roi` itself and sets `sub_volume_offset_zyx` to
`offsets_zyx`; `crop_roi` then cuts exactly `bounds` (checked for every legal centre on every
axis of the six shapes, `test_pipeline_roi.py`). It writes `roi_placement.json`, which the batch
and every driver reading a batch output check against `place_roi`. The batch outputs date from the
2026-09-28 re-run, which is on these boxes (open items 27, 15, 17 and 13 together). Placed centres
now, against the array centre:

| Specimen | Array centre | Placed centre | Shift (µm, z/y/x) | Shared volume |
|---|---|---|---|---|
| WKY-A | 217, 228, 253 | 230, 224, 186 | +24, −7, −125 | 52% |
| WKY-B | 217, 178, 175 | 106, 192, 178 | −207, +26, +6 | 27% |
| WKY-C | 217, 157, 127 | 189, 158, 100 | −52, +2, −50 | 68% |
| SHR-A | 247, 229, 172 | 157, 244, 158 | −168, +28, −26 | 36% |
| SHR-B | 247, 241, 199 | 230, 308, 186 | −32, +125, −24 | 48% |
| SHR-C | 247, 247, 190 | 164, 302, 186 | −155, +103, −7 | 31% |

**Before that fix the pipeline never got the offsets**, and the rest of this step is the record of
the outputs made before it (2026-09-26 and earlier; archived in `cb_h1_batch_2026-09-26/`).
`cb_h1_batch.py --stage run` passed `--roi-voxels 160 160 160` but no offsets, so the pipeline
cropped a 160³ box on the **array centre** (`extent // 2`), not on the placed centre. The cached
batch masks confirmed it: IoU with a centred crop 0.92 (WKY-A) and 0.83 (SHR-C), against 0.17 and
0.15 with `placement.bounds`. A fresh WKY-A mask on the placed box now has IoU 0.951 with a
hysteresis mask cut at `bounds`, and 0.174 with the centred one. The pre-fix record, with the
placed centres of the time (whole-stack projection, before open item 13):

| Specimen | Array centre | Placed centre | Shift (µm, z/y/x) | Shared volume |
|---|---|---|---|---|
| WKY-A | 217, 228, 253 | 230, 240, 188 | +24, +22, −121 | 50% |
| WKY-B | 217, 178, 175 | 106, 198, 174 | −207, +37, −2 | 27% |
| WKY-C | 217, 157, 127 | 189, 166, 92 | −52, +17, −65 | 61% |
| SHR-A | 247, 229, 172 | 157, 260, 146 | −168, +58, −49 | 30% |
| SHR-B | 247, 241, 199 | 230, 286, 176 | −32, +84, −43 | 55% |
| SHR-C | 247, 247, 190 | 164, 298, 188 | −155, +95, −4 | 32% |

So before the re-run every network quantity from the batch (`per_edge_morphometry.csv`,
`network_graph.pkl`, the H1 morphometry and figures, and the graph the H2 drivers load) was on the
centred box, while everything that slices with `bounds` was on the placed box. Since the re-run
both are on the placed box.

*Step 10 — the crop the other drivers run.* Every CB driver except the network pipeline slices
the array directly with `RoiPlacement.bounds`:

```python
@property
def bounds(self):
    return tuple(
        slice(c - s // 2, c - s // 2 + s)
        for c, s in zip(self.centre_zyx, self.size_zyx)
    )
```

`sub = volume[placement.bounds]` — `cb_h1_batch.py:84` (threshold stage only), and the same in
`cb_h1_th_metrics.py:75`, `cb_h2_vtk.py:117`, `cb_h2_glomus_perfusion.py:84` and
`cb_h2_hypoxic_fraction.py:92`. Integer arithmetic throughout, so the box is exactly 160 wide and
exactly centred on the requested voxel. The three H2 drivers crop the TH channel this way and lay
it over the batch graph, which since the re-run is cut at the same `bounds`, so the two frames
agree (`cb_h2_vtk.py --verify`: 83–88% of penetrating centreline inside the TH mask, 13–38% in the
transposed control). Before the re-run the graph came from the centred box, the two frames differed
by the shifts in the pre-fix table above, and the glomus mask sat over vessels from another region
(open item 27, closed).

**The two paths now agree.** `crop_roi` used to rebuild the centre from the fraction by truncating
twice — `int()` on the offset, then `int()` again on the start. On an axis of **odd** extent,
`extent / 2.0` ends in `.5`; with the centre above the volume midpoint the residue rounded the
wrong way and the box landed **one voxel low**. On the six, one axis per specimen was affected
(WKY-A in z, the other five in y), and the error was always −1, never +1, so it would not have
averaged out across specimens.

It now rounds once, on the whole centre, and starts the box at `centre − size // 2`, the same
integer rule as `bounds`:

```python
extent = int(orig_shape[ax])
sub_center = int(np.ceil(extent / 2.0 + extent * offsets[i] - 0.5))   # halves round down
start = max(0, sub_center - target_dims[ax] // 2)
```

Halves round down, so a zero offset still centres on `extent // 2`, matching `clamp_centre`. An
offset from `centre_to_offsets` lands exactly on its centre, for every legal centre of a 160-voxel
box on each extent tested, odd and even, 160–521 (`test_preprocessing.py`, `test_roi_placement.py`).

**No published number moved** at the time: the batch run then took that path only with zero
offsets and an even box, which the fix left where it was. The 2026-09-26 re-run (item 26)
reproduced the old masks. Since open item 27 the batch takes the offset path, and this fix is what
makes it land exactly on `bounds`.
What changes is the fractional-offset path in `carotid_image_to_model.py`, which now crops where
its offsets say. It also moves a centred crop one voxel up when the extent is even and the box size
odd, because the old start `int(E/2 − s/2)` rounded down where `bounds` does not. Formerly open
item 14.

**The box is clamped, never truncated.** `clamp_centre` pulls the centre inwards until the box fits
wholly inside the volume. A box hanging over an edge would be silently cropped, making that
specimen's sample smaller than the others — the exact thing a matched size exists to prevent. So in
a small volume the box is no longer centred on the tissue centroid, and that is the intended trade.

**Size.** `DEFAULT_ROI = (160, 160, 160)` voxels in all three drivers — H1 batch and both H2
drivers — which at the processing voxel `(1.8639, 1.866, 1.866)` µm is **298.2 × 298.6 × 298.6 µm,
or 0.0266 mm³**. The imaged blocks it is cut from run 0.227 mm³ (WKY-C) to 0.653 mm³ (WKY-A), so
the ROI is **4–12% of the block** depending on specimen. That spread is why the ROI is matched by
size: raw counts would otherwise track block extent rather than biology.

| | z | y | x |
|---|---|---|---|
| ROI, voxels | 160 | 160 | 160 |
| ROI, µm | 298.2 | 298.6 | 298.6 |
| Subsample for the centroid | 4 | 2 | 2 |

> **At a glance** — tissue centroid, not centre · 160³ voxels = 0.0266 mm³, 4–12% of the block ·
> peak depth fraction ranges 0.244–0.529; cohort means 0.402 vs 0.371 · `roi_placement.py:96`,
> `roi_placement.py:77`, `roi_placement.py:54` · `tests/test_roi_placement.py`

---

### 2.2 Segmentation threshold selection

**What it does.** Chooses the probability threshold that turns the classifier's output into a
binary mask.

**What was chosen.** *Calibre chooses, fragmentation vetoes.* Take the **highest** threshold whose
median mask diameter falls in the capillary window, provided it lies below the fragmentation onset.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Place the ROI and crop the probability volume | 160³ | The threshold has to be chosen on the same sub-volume it will be applied to | **On** — a geometrically-centred box chose differently for 3 of 6 before the 2026-09-28 fixes; with them it chooses the same 0.95 in all six | `cb_h1_batch.py:84` |
| 2 | Cut a mask at each threshold in the grid | `p ≥ t`, a **plain cut** | A cheap monotone family of masks to rank against each other; the sweep is a ranking, not an absolute measurement | **On** — inclusive since open item 17, as the pipeline's hysteresis is; the strict `p > t` before it dropped the level on the threshold | `threshold_selection.py` (`evaluate_threshold`) |
| 3 | EDT → median and p90 diameter **on the skeleton voxels** | `sampling` = voxel size | Calibre is the criterion that selects, and EDT is bounded by the mask so it cannot read a neighbouring vessel. Read on the centreline, as §2.6 does, since open item 15; the median over every foreground voxel is printed as `d_vox`, and the same centreline median on the network's own mask as `d_net` (open item 41) | **On** — the median decides; the p90, `d_vox` and `d_net` are printed and never read | `threshold_selection.py` (`evaluate_threshold`) |
| 4 | Label mask components, largest share, count above floor | 50 voxels | Kept for continuity with `prob_to_mask.py`; component statistics move, but with no knee to read a threshold off | **On** — the total and the share are printed; the above-floor count is neither printed nor read | `threshold_selection.py:200` |
| 5 | Skeletonise the cut mask | raw, **no cleanup** | Endpoint density needs a centreline; there is no other way to count where the network has broken | **On** | `threshold_selection.py:210` |
| 6 | Skeleton length from voxel count | × in-plane pitch | An endpoint count alone scales with network size, so it needs a per-length denominator to compare across thresholds | **On** | `threshold_selection.py:215` |
| 7 | Count degree-1 voxels → endpoint density | per mm of skeleton | A network breaking into beads gains endpoints far faster than it gains components, so this detects fragmentation earlier | **On, but never decisive** — the veto it feeds does not bind on any of the six | `threshold_selection.py:219` |
| 8 | Drop thresholds whose mask is empty | — | At the top of a sweep an empty mask is the expected outcome, not a failure worth reporting as one | **On, but never fires** — foreground is still 10.4–15.0% at 0.99 | `threshold_selection.py:251` |
| 9 | Baseline = **median** endpoint density across the sweep | — | The minimum is a single noisy sample; using it would flag ordinary variation as fragmentation | **On** | `threshold_selection.py:276` |
| 10 | Onset = lowest threshold above 1.5 × baseline | `FRAGMENTATION_TOLERANCE` | Marks where the centreline is demonstrably breaking, so every threshold at or above it can be vetoed | **On, but never binding** — onset is 0.97 or 0.99, always above the window's top (0.95) | `threshold_selection.py:281` |
| 11 | Calibre window = thresholds with median d in range | 4.0–7.0 µm | An external target rather than an internal optimum — a threshold tuned to a property of the data has no independent standard to be wrong against | **On** — but only the 4.0 µm bound can select; see below | `threshold_selection.py:283` |
| 12 | Chosen = **highest** window threshold below onset | — | Calibre falls monotonically with threshold and the risk being traded is over-inclusion, which resistance carries as $r^{-4}$ | **On**, or a refusal | `threshold_selection.py:305` |
| 13 | Repeat 1–12 per specimen; median of six, snapped to grid | 6 specimens | Per-specimen thresholds would absorb exactly the classifier-quality differences H1 is trying to measure | **On** — all six choose 0.95 (2026-09-28; four chose 0.90 and two 0.85 before open items 27, 15, 17 and 13) | `cb_h1_batch.py:106` |
| 14 | Cohort-split check on the per-specimen choices | — | A threshold that splits by group is a confound; reporting it makes that visible rather than hidden | **On**, reported — verdict is *no separation*, because all six chose the same value | `cb_h1_batch.py:101` |

**Steps 1 and 2: what is being cut, and what the cut discards.** These two look like
bookkeeping. They are not.

*Step 1 — the crop.* `read_ilastik_probabilities` returns one class channel of the Ilastik export
as float32 in (z, y, x), and the driver takes the placed box out of it:

```python
volume = read_ilastik_probabilities(
    specimen.probabilities_path, expected_shape_zyx=specimen.shape_zyx)
sub = volume[placement.bounds]          # cb_h1_batch.py:84
```

**Whether the threshold depends on where the box is.** It did: on the selector of the time
(voxel median, strict cut), re-running the whole selection on a geometrically-centred box instead
of the placed one changed the answer for half the cohort — WKY-B 0.90 → 0.93, SHR-B 0.85 → 0.90,
SHR-C 0.85 → 0.90. With the current selector it does not: the centred boxes also choose 0.95 in all
six (windows 0.90–0.95, or 0.93–0.95 in WKY-A and WKY-B; onsets 0.97 or 0.99). Row 1's rationale
still stands, since the network is measured on the placed box, but on this data the choice of box
no longer moves the threshold.

**The ROI is dense by construction, and that has a consequence for the reader's own guard.**
`read_ilastik_probabilities` refuses a channel whose **whole-volume** mean probability exceeds 0.5,
on the reasoning that such a channel must be the background class. Measured:

| Specimen | Whole-volume mean | Mean inside the analysed ROI |
|---|---|---|
| WKY-A | 0.214 | **0.615** |
| WKY-B | 0.219 | **0.531** |
| WKY-C | 0.353 | **0.541** |
| SHR-A | 0.216 | **0.574** |
| SHR-B | 0.251 | **0.553** |
| SHR-C | 0.179 | **0.560** |

Every ROI sits above the limit the guard treats as proof of a swapped channel; every whole volume
sits below it. The guard is checking the right region — a swapped channel is a whole-file property,
and the margin there is a comfortable 0.15–0.32 — but its message describes a vessel channel as *"a
few percent of the volume"*, which is not true of this data anywhere. §2.1 places the ROI on the
densest tissue in the stack, so density inside it is the design, not a warning sign.

**The read is 50× larger than it needs to be.** The export is `(435, 456, 507, 2)` float32,
**uncompressed**, chunked `(50, 51, 50, 1)`. The reader materialises the entire 4D array — 805 MB —
then copies out one 402 MB channel, in order to use a 16 MB box. A direct `h5py` hyperslab of just
the ROI takes **0.01 s against 0.88 s**. Note the contrast with §2.1, where striding saved memory
but no time: that was the *preprocessed* HDF5, gzip-compressed with `(32, 128, 128, 3)` chunks, so
a strided read still had to decompress every chunk whole. This file is a different file with
different properties, and here slicing wins outright. Nothing downstream is wrong; it is simply
paying about a gigabyte and a second per specimen for no return.

*Step 2 — the cut.* One line, and two properties worth knowing:

```python
binary = at_or_above(probabilities, threshold)   # threshold_selection.py (evaluate_threshold)
if not binary.any():
    return None
```

`preprocessing.at_or_above` is `values >= threshold`, with the threshold cast to the array's float
type so a float32 level k/100 compares equal to the threshold k/100.

**Monotone, hence a ranking.** Because the comparison is a single global cut, $p \ge t_2 \subseteq
p \ge t_1$ whenever $t_1 < t_2$ — nested at every step of the grid. That nesting is what licenses
treating the sweep as a *ranking* over thresholds rather than a set of independent measurements,
which is the argument that survives the fact that the pipeline actually builds a different
(hysteresis) mask.

**The field is quantised to hundredths, and every grid threshold lands on a level.** The ROI holds
**101 probability levels** — 0.00, 0.01, … 1.00, each within 4 × 10⁻⁶ of $k/100$. (In WKY-C, SHR-B
and SHR-C a few levels between 0.31 and 0.59 also carry a float32 neighbour one step above the
level, so up to 108 distinct values; none is at a grid threshold, and the inclusive cut keeps them.)
Every value in the sweep grid is also a whole number of hundredths, so each threshold sits exactly
*on* an occupied level, and a whole level's worth of voxels lies on the boundary.

**Until open item 17 the cut was strict, `p > t`, and discarded that whole level.** Two
consequences followed exactly, not approximately:

$$p > t \quad\equiv\quad p \ge t + 0.01 \qquad\text{and}\qquad p > 0.99 \quad\equiv\quad p = 1.0$$

The nominal threshold was one level below the effective one, and **the top of the sweep was not
measuring "probability above 0.99" at all — it was measuring the saturated set.** The tie mass is
small at the bottom of the grid and large at the top (placed ROIs, 2026-09-28):

| | 0.30 | 0.50 | 0.70 | 0.85 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| WKY-A, fraction of ROI exactly at *t* | 0.55% | 0.50% | 0.63% | 0.98% | 1.33% | 2.23% | **5.96%** |
| SHR-C, fraction of ROI exactly at *t* | 0.67% | 0.61% | 0.68% | 0.97% | 1.23% | 1.87% | **4.16%** |

At 0.99 that level is worth **66–67% of the mask the strict cut keeps**. At the frozen 0.95 it is
+8.9% (WKY-A) and +10.2% (SHR-C) of mask volume. The selector, the pipeline's hysteresis (flood and
seed) and `cb_h1_th_metrics.py`'s vessel cut now all use `≥`; the TH cut, the entropy hysteresis
(off) and `prob_to_mask.py` keep `>`.

**And it changed the answer.** Re-running the selection of the time (voxel-median calibre,
whole-stack placement) with `≥` in place of `>`:

| Specimen | with `p > t` | with `p ≥ t` |
|---|---|---|
| WKY-A | 0.90 | **0.93** |
| WKY-B | 0.90 | 0.90 |
| WKY-C | 0.90 | 0.90 |
| SHR-A | 0.90 | 0.90 |
| SHR-B | 0.85 | **0.90** |
| SHR-C | 0.85 | **0.90** |

Three of six moved, and the group asymmetry of the old freeze (formerly open item 16, below), where
SHR-B and SHR-C alone ran below their own calibre floor, did not appear under `≥`. With items 27,
15 and 13 applied as well, all six now choose 0.95 (below). Open item 17, closed.

**The selector does not measure the mask the pipeline builds.** Step 2 is a plain cut (`p ≥ t`) and
step 5 skeletonises it raw. The pipeline instead builds a *hysteresis* mask, closes it, prunes to
the largest component and cleans the skeleton (§2.3, §2.4). So the calibre and fragmentation figures
that choose the threshold are measured on a **thinner, noisier** object than the one that reaches
the haemodynamics. The direction of the discrepancy is knowable — hysteresis and closing both add
voxels, so the real mask is at least as fat and at least as connected — but its size is not
measured. Treating the sweep as a *ranking* over thresholds rather than an absolute calibre
measurement is what makes this acceptable.

**Skeleton length has no diagonal correction.** Step 6 multiplies the skeleton voxel count by the
in-plane pitch alone, so a diagonal step counts as 1 voxel rather than √2 or √3. Length is therefore
underestimated and endpoint density per mm overestimated in absolute terms. It cancels out of the
1.5 × ratio, which is why the criterion survives it, but the `ep/mm` column in the printed table is
not a physical density.

**Why calibre and not component statistics.** The conventional criterion is *the value just above
where component count climbs steeply and the largest component's share starts falling*. Measured on
the six ROIs, both halves of it fail — but not in the way the module docstring claimed, and the
docstring's figures are from a differently-placed sub-volume that predates §2.1. The real numbers,
for WKY-C (placed ROI, `p ≥ t`, 2026-09-28):

| Threshold | Mask components | Largest share | Components > 50 vox | ep/mm |
|---|---|---|---|---|
| 0.30 | 751 | 0.9974 | 6 | 3.12 |
| 0.50 | 566 | 0.9966 | 21 | 5.11 |
| 0.70 | 422 | 0.9966 | 15 | 5.32 |
| 0.90 | 364 | 0.9878 | 14 | 5.44 |
| 0.95 | 541 | 0.9868 | 23 | 6.92 |
| 0.99 | 1,379 | 0.9784 | 20 | 15.96 |

The share **does** fall — from 0.9974 to 0.9784 here, and lower at 0.99 than at 0.30 in all six,
contradicting the docstring's claim that it never does and is *higher* at 0.99 than at 0.70.
Component count is not flat either: it is U-shaped, bottoming at 0.85 (330) and then climbing
4.2-fold. What is true is that neither has a **knee**. Component count changes smoothly — successive
ratios across the grid 0.75, 0.75, 0.87, 0.90, 1.10, 1.10, 1.36, 1.39, 1.83 — so "just above where it
climbs steeply" names no particular threshold. And the share steps down once, at 0.80, then sits
near 0.987 until the network is already breaking at 0.97–0.99. Component statistics do not fail to
move; they move **without a feature sharp enough to read a value from**, or too late. That is a
property of the data's topology — a vascular bed percolates, and a percolating mask stays connected
long after its centreline has begun beading — not of any one classifier.

**What actually selects, measured on all six.** Running `cb_h1_batch.py --stage threshold` over the
placed ROIs (2026-09-28, centreline calibre, `p ≥ t`) gives the frozen 0.95, but the route to it is
narrower than the fourteen-step table suggests:

| Specimen | Calibre window | Onset | Own choice | d at 0.95 |
|---|---|---|---|---|
| WKY-A | 0.93–0.95 | 0.97 | **0.95** | 5.27 µm |
| WKY-B | 0.90–0.95 | 0.99 | **0.95** | 5.27 µm |
| WKY-C | 0.90–0.95 | 0.97 | **0.95** | 5.27 µm |
| SHR-A | 0.90–0.95 | 0.97 | **0.95** | 5.27 µm |
| SHR-B | 0.90–0.95 | 0.99 | **0.95** | 5.27 µm |
| SHR-C | 0.85–0.95 | 0.97 | **0.95** | 5.27 µm |

At 0.97 the centreline median falls to 3.73 µm in every specimen, below the window's 4.0 µm floor,
so the window stops at 0.95 everywhere. (Before the re-run, on the voxel median and strict cut,
four chose 0.90 and two 0.85, with onsets at 0.95 or 0.97.)

**The fragmentation veto never binds.** In every specimen the onset (0.97 or 0.99) sits above the
top of the calibre window (0.95), so no candidate is ever removed by it. Re-running the selection
with the constraint switched off entirely returns the identical six choices. Steps 5–7 and 9–10 —
skeletonisation, length, endpoint density, baseline, onset — are the expensive half of the sweep,
and on this data they change nothing at the 4.0 µm floor. They are a guard that has not yet been
needed, in the same sense as the ROI clamp in §2.1. Keeping them is still right: they are the only
thing standing between the calibre rule and a threshold that meets calibre by shredding the
network, and with a floor at or below 3.73 µm they would bind in four of six (below).

**The median diameter is atomic, and the window admits two values.** The EDT of a binary volume on
this near-cubic grid can only take values $\text{pitch}\times\sqrt{k}$, so a *median* over
skeleton voxels lands on one of a few levels. Across the whole 6 × 10 sweep it visits ten:

| $d$ (µm) | radius / voxel | $\sqrt{k}$ | inside 4.0–7.0? |
|---|---|---|---|
| 3.73 | 1.000 | $\sqrt{1}$ | no — below the floor |
| 5.27 | 1.412 | $\sqrt{2}$ | **yes** |
| 6.46 | 1.731 | $\sqrt{3}$ | **yes** |
| 7.46 | 1.999 | $\sqrt{4}$ | no — above the ceiling |
| 8.34, 9.13, 10.56, 11.19, 11.80, 13.45 | 2.24–3.60 | $\sqrt{5}, \sqrt{6}, \sqrt{8}, \sqrt{9}, \sqrt{10}, \sqrt{13}$ | no |

So "median diameter in the 4–7 µm capillary window" is, on this grid, exactly the statement *the
median centreline voxel's inscribed radius is $\sqrt2$ or $\sqrt3$ voxels*. The selection turns
on a single quantisation step — where the median falls from $\sqrt2$ to $\sqrt1$, between 0.95
and 0.97 in all six — not on a smooth approach to a physiological target. That does not make it
wrong, but it does mean the window's apparent precision is not real: any lower bound from 3.74 to
5.27 µm gives the identical six choices. At 3.73 µm or below, WKY-B and SHR-B move to 0.97 and the
other four are held at 0.95 only by the fragmentation veto; above 5.27 µm four move to 0.90, SHR-C
to 0.85, and WKY-A refuses.

**Only the lower bound can select.** Median calibre falls monotonically with threshold, and step 12
takes the **highest** threshold in the window. The upper bound therefore only ever prunes from the
*low*-threshold end, which the maximum never reads. Sweeping it from 5.5 µm to 25 µm leaves all six
choices unchanged; the only value that changes anything is one low enough to empty the window and
force a refusal. Of the two numbers in `CAPILLARY_DIAMETER_RANGE_UM`, **4.0 selects and 7.0 is
inert** — which matters for the sensitivity analysis the constant's own comment calls for, because
sweeping the width of the window symmetrically tests one live parameter and one dead one.

**The selector's diameter is now the statistic §2.6 reports (open item 15, closed).** Until
2026-09-28 step 3 took the median of `edt[binary]` — every foreground voxel — while §2.6 samples
the EDT **on the centreline**. A volume-weighted median is dragged down by the surface shell, which
is most of a thin mask, and it read 0.63–1.00 × the centreline median on the same masks. The
selector now reads 2 × EDT at the skeleton voxels; the voxel median is still printed, as `d_vox`,
and selects nothing. Side by side on the placed ROIs:

| Threshold | WKY-C `d_vox` | WKY-C centreline | SHR-A `d_vox` | SHR-A centreline |
|---|---|---|---|---|
| 0.70 | 6.46 µm | 9.13 | 6.46 | 8.35 |
| 0.80 | 5.28 | 8.34 | 5.28 | 7.46 |
| 0.90 | 5.27 | 6.46 | 5.27 | 6.46 |
| 0.93 | 3.73 | 5.27 | 3.73 | 5.27 |
| 0.95 | 3.73 | 5.27 | 3.73 | 5.27 |
| 0.97 | 3.73 | 3.73 | 3.73 | 3.73 |

On `d_vox` the selection would pick 0.93 in WKY-A and 0.90 in the other five, so item 15 is what
moved the freeze from 0.90 to 0.95. At the chosen 0.95 the selector reads 5.27 µm, while
§2.6's per-edge medians on the delivered mask run 5.87–7.46 µm by specimen. The estimator is the
same; what is left of the gap is the mask (plain cut here, hysteresis there — noted above) and a
per-voxel against a per-edge median.

**The window is tested on a mask the network never uses, and that is kept, with a check
(open item 41, closed).** Since 2026-09-30 the sweep also prints `d_net`: the same centreline median, on the mask
the pipeline builds (`preprocessing.build_vessel_mask`: caged z pad, hysteresis from the frozen
0.999 seed, hole fill, closing, largest component; a drift test holds it to the pipeline's).
At 0.95 it reads **7.46 µm in all three WKY and 6.46 µm in all three SHR**, against 5.27 µm in
all six on the plain cut. On the network's mask WKY sits one level *above* the 7.0 µm ceiling,
and the calibre window would refuse or move 0.95 in every WKY specimen.

Two replacements were measured, and neither was adopted, because each would choose the
threshold by itself. Both use the network's mask and skeleton, and read the EDT of the same
recipe cut from the field trilinearly supersampled 2× or 3× (a `2n−1`-point grid, so every
native voxel is a supersampled grid point). Median centreline diameter, µm:

| Specimen | Threshold | Plain cut (selects) | Network mask | 2× | 3× |
|---|---|---|---|---|---|
| WKY-A | 0.93 | 5.28 | 8.34 | 6.19 | 6.22 |
| WKY-A | 0.95 | 5.27 | 7.46 | 5.60 | 5.28 |
| WKY-A | 0.97 | 3.73 | 6.46 | 5.60 | 4.65 |
| SHR-C | 0.93 | 5.27 | 7.46 | 5.60 | 5.13 |
| SHR-C | 0.95 | 5.27 | 6.46 | 5.28 | 4.65 |
| SHR-C | 0.97 | 3.73 | 5.27 | 4.17 | 3.73 |

Supersampling does not remove the quantisation, it only makes the steps finer: 5.60 µm is the
half-pitch level $2 \times \tfrac{h}{2}\sqrt{9}$, and WKY-A ties at 0.95 and 0.97 exactly as the
plain cut tied at 0.93 and 0.95. A grouped (interpolated) median does not help, because the level
holding the 50% point carries too much of the mass. And the value does not converge: it falls by
about 1.9 µm from native to 2× and by 0.3 µm from 2× to 3×. The EDT measures to the nearest
*background voxel centre*, so it overstates the radius by about half a grid step, and the
refinement only shrinks that bias. The same bias sits in §2.6's network calibre, about +1.9 µm in
diameter on the native grid. So the estimator moves calibre by about as much as the 4–7 µm window
is wide, and picking one picks the threshold.

So the plain cut keeps selecting, and the sweep re-runs the same rule on the network's own
estimator with the bias taken off, `d_net` less one voxel (`select_on_network_calibre`). It is
printed beside each choice and recorded in `threshold_selection.json` under
`robustness_network_calibre_less_voxel`, never frozen. Measured 2026-09-30
(`rerun_2026-09-29_logs/F_05_threshold.log`): WKY-A 0.95, **WKY-B 0.97**, WKY-C 0.95, SHR-A/B/C
0.95. The median is **0.95**, the same frozen value, and the choices do not separate by group.
WKY-B moves because its `d_net` at 0.97 is 6.46 µm (4.59 corrected) and it is still intact there
(8.49 ep/mm against a 8.75 onset). The frozen threshold therefore survives the network's own
calibre once its known bias is removed. The same bias in §2.6's per-edge diameters, which do
feed resistance, is open item 42.

**Why the highest, not the middle.** Calibre falls monotonically with threshold, and the risk being
traded is over-inclusion: the lower the threshold, the fatter the vessel, and resistance carries
that as $r^{-4}$.

**How fragmentation is measured.** Not by component count but by **endpoint density per mm of
skeleton**. The baseline is the median density across the sweep; the onset is the lowest threshold
whose density exceeds 1.5× that baseline. A network breaking into beads gains endpoints far faster
than it gains components.

**It can refuse.** Two refusals, both reported as segmentation problems rather than resolved by
picking something: no threshold reaches capillary calibre at all, or every threshold at capillary
calibre is at or beyond the fragmentation onset.

**One threshold for all six, and where 0.95 came from.** The selector runs *per specimen*, but no
specimen runs at its own choice. `cb_h1_batch.py --stage threshold` sweeps the grid
`[0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]`, selects per specimen, then takes
the **median of the six selections and snaps it to the nearest grid value**. Measured 2026-09-28:
all six choose 0.95, so the frozen value is **0.95** (`cb_settings.FROZEN_THRESHOLD`). Before open
items 27, 15, 17 and 13, four chose 0.90 and two 0.85, median 0.875, snapped to 0.90. The rationale
is in the driver: per-specimen thresholds would absorb exactly the classifier-quality differences
that H1 is trying to measure, turning a confound into an apparently clean result. The per-specimen
choices are still reported and passed to `assess_cohort_split`, so a threshold that splits by group
is visible rather than hidden.

**The split test passes, and the old freeze asymmetry is gone.** `assess_cohort_split` returns *no
separation*: all six chose the same value, so no specimen runs above or below its own choice.
Foreground fraction at 0.95 overlaps between groups (WKY 0.205–0.274, mean 0.229; SHR 0.201–0.231,
mean 0.215). Under the 0.90 freeze the two specimens run above their own choice were both SHR, at a
median calibre of 3.73 µm — below the 4.0 µm floor the selector enforces — so 0 of 3 WKY and 2 of 3
SHR were analysed at a threshold their own calibre criterion rejected. That asymmetry came from the
strict cut and did not survive `≥` (formerly open item 16, merged into 17, now closed).

**The hysteresis pair follows the frozen value.** `--stage run` passes the frozen threshold as
`--hysteresis-low` only. A low given alone takes low + `HYSTERESIS_HIGH_OFFSET` (0.05) as its seed,
capped at `HYSTERESIS_SEED_CAP` (0.999) (`_apply_hysteresis_overrides`), so a frozen 0.95 gives a
**0.95 / 0.999** pair. On a field quantised to hundredths the inclusive seed `p ≥ 0.999` is exactly
the saturated set `p == 1.0`. `PreprocessingConfig` reads the same pair from
`cb_settings.HYSTERESIS_LOW` / `HYSTERESIS_HIGH`, so a direct run without the flag builds the batch
mask. A seed at or below the flood threshold raises. (The pair was 0.90 / 0.95 until 2026-09-28.)

**The sensitivity runs hold the seed.** `cb_h1_batch.py --stage sensitivity` reruns every specimen
at the frozen value's two grid neighbours (0.93 and 0.97) and passes `--hysteresis-high` set to
`HYSTERESIS_HIGH` beside each `--hysteresis-low`, so only the flood threshold moves. Before
2026-09-30 the neighbours were run by hand with the low bound alone, so 0.93 seeded at 0.98 and
differed from the frozen run in two parameters (re-run notes item 6, open item 41). The sweep holds
the same 0.999 seed for `d_net` at every threshold. Re-run 2026-09-30, 12/12 exit 0: the 0.97
outputs are byte-identical (their seed was already 0.999), and 0.93 moves little. SHR/WKY
group-mean ratios at 0.93 / 0.95 / 0.97 are now β₁ density 1.050 / 1.087 / 1.117, junction
density 1.051 / 1.064 / 1.098, length density 1.013 / 1.022 / 1.030 (0.93 was 1.054, 1.055,
1.015 with the 0.98 seed). The median calibre at 0.93 rose in SHR-B and SHR-C (6.81 → 6.96 µm),
so the threshold shift of §13.2 is now 0.740 µm.

| Constant | Value | Meaning |
|---|---|---|
| `CAPILLARY_DIAMETER_RANGE_UM` | (4.0, 7.0) µm | The calibre window that selects — in practice only the 4.0 bound can change an answer |
| `FRAGMENTATION_TOLERANCE` | 1.5× | Endpoint-density multiple defining onset; the veto it produces never binds on these six |
| `MIN_COMPONENT_VOXELS` | 50 | Floor for `mask_components_above_floor`, which is computed and then neither printed nor read |

> **At a glance** — highest intact threshold in a 4–7 µm centreline-calibre window, `p ≥ t` · all
> six choose 0.95 · fragmentation onset at 1.5× baseline endpoint density · `threshold_selection.py:256`, `threshold_selection.py:163` ·
> `tests/test_threshold_selection.py`

---

### 2.3 Mask formation

**What it does.** Converts the probability field to a binary mask.

**What was chosen.** Hysteresis thresholding with all pre-threshold filtering disabled. A joint
probability-and-entropy variant exists but is **off by default, and cannot run on the CB vessel
data** — turning it on raises. See below.

**Why no pre-threshold filtering.** Every filter whose support is comparable to the structure width
deletes the structure. A 6 µm capillary is 3.2 voxels across at 1.866 µm. A 3×3×3 median spans
5.6 µm; a greyscale opening retains 51% of a 1.6-voxel-radius tube at radius 1 and none at radius 2.
Measured at fixed threshold, varying only this chain: (0, 0) → 199 structures at median radius
1.87 µm; (7, 1) → 71 structures at 2.64 µm; (9, 4) → 3 structures at 5.27 µm. Thin structures are
deleted while fat ones survive.

Speckle removal still happens — but *after* thresholding, as a connected-component size filter,
where it is a topology operation rather than a boundary operation:

| Approach | Foreground | Components | Recall |
|---|---|---|---|
| Truth (clean probability) | 900 | 3 | 100.0% |
| Threshold only | 1,644 | 714 | 100.0% |
| Median 3 → threshold | 181 | 3 | **20.1%** |
| Threshold → 50-voxel size filter | 913 | 3 | **100.0%** |

Same cleanup, 80% of the vessel destroyed, and what survives is thinned.

That 50-voxel row is a **test-fixture demonstration**, not a pipeline stage. In the real pipeline no
size filter is applied to the mask at all — speckle removal happens later, on the *skeleton*
(§2.5), at `min_branch_length = 3` voxels and a 5% component fraction. The row is here to justify
why the pre-threshold filters are off, not to describe a step.

### 2.3.1 The operative path: plain hysteresis

Two thresholds, `low = 0.95` and `high = 0.999` (`cb_settings.HYSTERESIS_LOW` / `HYSTERESIS_HIGH`),
applied as a connectivity rule rather than a cut:

1. **Seed.** Every voxel with $p \ge 0.999$ is a seed — on this field, exactly the saturated voxels, $p = 1.0$.
2. **Grow.** Every voxel with $p \ge 0.95$ is a candidate.
3. **Keep** only the candidates that are face-connected to at least one seed.

A voxel at p = 0.97 is therefore kept or discarded *depending on its neighbours* — kept if it hangs
off a confident core, discarded if it is isolated. This is the whole point: a single global cut at
1.0 severs vessels wherever the classifier dipped, while a single cut at 0.95 admits every
scattered speck. Hysteresis takes the connected interior of the first and the extent of the second.

The comparisons are inclusive since open item 17 (`preprocessing.at_or_above`); the hysteresis is
the module's own, face-connected like skimage's `apply_hysteresis_threshold` and identical to it on
a field with no ties. Before that fix both comparisons were strict, and the field is quantised to
0.01, so a voxel exactly at `low` was not a candidate and one exactly at `high` did not seed.

**Why this matters more here than in a typical image.** Classifier confidence falls at vessel
*walls* — the boundary voxels are genuinely mixed. A hard cut therefore erodes every vessel from
the outside in, and resistance carries that as $d^{-4}$. Hysteresis lets the mask grow out to the wall
provided it started somewhere confident.

**The band is narrow — 0.95 to 0.999.** The growth step is a modest dilation of the saturated core
rather than a long reach, so the mask is close to a plain cut at 0.95 wherever that cut touches a
saturated voxel. Widening the band would recover more wall at the cost of admitting more speckle.
Measured on WKY-A: the hysteresis mask on the placed box has IoU 0.951 with the pipeline's cached
mask; the plain cut at 0.95 keeps 27.4% of the ROI (threshold stage) and the pipeline mask 28.9%
after hole filling.

**Where the values came from.** `low` is the frozen threshold of §2.2: the median of the six
per-specimen selections on the placed ROIs, snapped to the sweep grid. `high` is `low` + 0.05,
capped at 0.999 (`HYSTERESIS_SEED_CAP`). This is the band every H1 run used since 2026-09-28
(0.90 / 0.95 before). The tuner cannot choose it: the preprocessing objective is
`1 − mean probability inside the mask`, which rises monotonically across the whole plausible band,
so its argmin is always the top of the search range rather than a property of the data. The yield
cliff meant to stop it never engages — probability yield is still 0.071 at `low = 0.85`, well above
the 0.05 trigger. (The tuner's upper bounds were raised to 0.95 / 0.99 under open item 1 so the
default stays inside its search range; the CB batch never runs it.)

**History.** Until open item 1 the `PreprocessingConfig` default was 0.65 / 0.75, set by hand from
calibre and connectivity on an earlier reference subvolume. It was never the band H1 ran at. The
measurements behind it:

| `low` | Foreground | r_p90 (µm) | Components | |
|---|---|---|---|---|
| 0.20 | 0.847 | 31.55 | 1 | floods into one blob |
| 0.60 | 0.154 | 4.57 | 61 | |
| **0.65** | 0.118 | **4.17** | 116 | old config default |
| 0.70 | 0.090 | 3.73 | 84 | |
| 0.80 | 0.045 | 3.23 | 367 | breaking into fragments |
| 0.85 | 0.031 | 2.64 | 414 | |

On that subvolume r_p90 of 4.17 µm was the right scale for a ~3 µm capillary radius, and component
count was stable from 0.60 to 0.73 before exploding above 0.80. The per-specimen selector of §2.2,
run on the six placed ROIs with the pooled classifier, chose higher values; those set the band now.

**The full mask-formation order**, as executed:

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | ROI crop of the probability field | 160³ voxels | Bounds the work and fixes the volume every later count is taken from (§2.1) | **On** | `carotid_image_to_model.py:754` |
| 2 | Class-axis detection, then entropy map | `n_classes` | Entropy is independent evidence only at three or more classes; at two it is a folded function of $p$ and would evacuate the vessel walls | **Off** — `enable_shannon_entropy = False`. Turned on at 2 classes, or with no class axis, it raises | `carotid_image_to_model.py:796` |
| 3 | Vessel channel selection | `ilastik_vessel_channel` | The threshold operates on one scalar probability field, not on the class stack | **On** | `carotid_image_to_model.py:789` |
| 4 | Virtual padding in z | 10 voxels, `mode='edge'` | Stops the array boundary caging the mask; `edge` replicates the boundary probability rather than introducing background | **On** (`caged`) | `carotid_image_to_model.py:680` |
| 5 | Median filter | size 0 | Off — a 3×3×3 median spans 5.6 µm against a 3.2-voxel capillary, and costs 80% of the vessel | Off | `carotid_image_to_model.py:684` |
| 6 | Morphological opening | radius 0 | Off — radius 1 retains 51% of a 1.6-voxel-radius tube and radius 2 retains none | Off | `carotid_image_to_model.py:688` |
| 7 | Morphological closing | radius 0 | Off — same objection: any operator whose support matches the structure width deletes the structure | Off | `carotid_image_to_model.py:692` |
| 8 | Probability smoothing | sigma 0.0 | Off — blurring the field moves the wall the threshold lands on, which resistance carries as $d^{-4}$ | Off | `carotid_image_to_model.py:696` |
| 9 | Joint probability–entropy hysteresis | core 0.6 / max 0.95 | Would gate seeding and growth on classifier confidence as well as probability; the guard keeps it out of reach at 2 classes | **Off** — needs a 3-class classifier | `carotid_image_to_model.py:710` |
| 10 | Plain hysteresis threshold, `p ≥` both bounds | 0.95 / 0.999 (`cb_settings`; 0.90 / 0.95 until 2026-09-28; config 0.65 / 0.75 until open item 1) | Confidence legitimately falls at vessel walls, so a single hard cut erodes every vessel from the outside in | **On** | `carotid_image_to_model.py:710` |
| 11 | Hole filling, 3D | — | A lumen voxel the classifier missed would otherwise stay a permanent hole and shrink the EDT inscribed radius through it | **On** | `carotid_image_to_model.py:720` |
| 12 | Un-pad | 10 voxels in z | The pad is scaffolding; leaving it would extend every boundary vessel by 10 slices of replicated probability | **On** | `carotid_image_to_model.py:722` |

Steps 5–8 are the pre-threshold chain, and all four are off. Step 9 is the config default and never
executes. Nothing between the crop and the threshold changes a single probability value on the CB
path — the mask is the threshold, the hole fill, and nothing else.

Note the padding is `mode='edge'`, so the pad replicates the boundary probability rather than
adding background. It exists to stop the boundary caging the mask, and it is stripped before the
mask is returned.

**What the entropy map is.** Per voxel, across the classifier's class probabilities:

$$H = -\sum_{c} p_c \log_2 p_c
\qquad\text{normalised by }\log_2(n_\text{classes})
\;\Longrightarrow\; H \in [0, 1]$$

It measures **how undecided the classifier was at that voxel**, not how likely the voxel is to be
vessel. H = 0 means the classifier put everything on one class; H = 1 means it split evenly across
all of them. Probability answers *which class*; entropy answers *how sure*.

**The two entropy parameters are confidence gates on the two hysteresis tiers.**

| | Probability test | Entropy test | Meaning |
|---|---|---|---|
| **Seed** | $p \ge \text{high}$ | $H \le \texttt{shannon\_core}$ = **0.6** | Where the mask is allowed to start |
| **Candidate** | $p \ge \text{low}$ | $H \le \texttt{shannon\_max}$ = **0.95** | Where it is allowed to grow into |

Seeds are then morphologically reconstructed into the candidate mask by dilation, so a candidate is
kept only if it connects back to a seed.

So `shannon_core` is the **strict** gate and `shannon_max` the **permissive** one — the same
high/low logic as ordinary hysteresis, applied to confidence instead of probability.
$\texttt{shannon\_core} \le \texttt{shannon\_max}$ is enforced, and reversing them raises.

- **`shannon_core = 0.6`** — to *start* a vessel, the classifier must have been fairly decided.
- **`shannon_max = 0.95`** — to *continue* one, almost any confidence will do; this rejects only
  near-total confusion.

The intent is to stop the mask seeding inside ambiguous tissue while still letting it grow through
vessel walls, where a classifier is legitimately less certain.

**It needs three or more classes to mean anything.** With two classes, $H(p)$ is a deterministic
function of p alone, folded about p = 0.5 — so it carries **no information the probability does not
already carry**. $H \le t$ then resolves to $p \le r$ or $p \ge 1 - r$, which carves a band out of the
*middle* of the probability range: the mask becomes non-monotonic in p, keeping low-probability
voxels while discarding higher-probability ones. Every vessel would come out as a core plus a
detached shell, with the wall voxels evacuated — the opposite of the intent.

> **Formerly open item 11 — the entropy path is off, and says so.** The pooled vessel classifier's
> `LabelNames` are `['vessel', 'background']`: **two classes**. The pipeline used to default
> `enable_shannon_entropy = True`, detect the two classes, warn, and run plain hysteresis — so the
> config described a path that never executed, and the only trace was a warning in the run log.
> `shannon_entropy_core` was not even a `PreprocessingConfig` field; it was read with a 0.6 default.
>
> Now `enable_shannon_entropy` defaults to **False**, and turning it on raises if the probability
> field has fewer than three classes or no class axis at all, matching the library function
> `joint_hysteresis_threshold`, which already raised. `shannon_entropy_core` is a config field
> (0.6), and both entropy parameters are read without defaults. CB masks are unchanged: they were
> plain hysteresis before and are now.
>
> The joint path becomes usable if the classifier is retrained with a third class — glomus being
> the obvious candidate, since the TH channel already exists — but it now has to be switched on.

> **At a glance** — plain hysteresis; the joint probability–entropy path is off, and raises if
> switched on at 2 classes · no pre-threshold filtering, median-3 costs 80% recall · `image.py:179`,
> `image.py:261`, `carotid_image_to_model.py:769` · `tests/test_preprocessing.py`,
> `tests/test_new_preprocessing.py`

---

## §2 — Geometric model, part 2: from mask to graph

Sections 2.4 and 2.5 turn the binary mask into a graph of nodes and edges. The subsection
numbers continue from part 1 rather than restarting, because they are cross-referenced from
throughout the document.

---

### 2.4 Skeletonisation and graph construction

**What it does.** Reduces the binary mask to a one-voxel-wide centreline, then converts that
centreline into a graph of nodes and edges carrying physical coordinates and lengths.

**What was chosen.** Mask repair by morphological *closing* (not dilation), largest-component-only
pruning, skeletonisation at native resolution, then skan-based segment extraction with loop
stitching and a 5.6 µm terminal reconnection.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Morphological closing of the mask | radius 1 | Seals segmentation dropouts up to about 2 voxels without moving the vessel surface, which a dilation would | **On** | `carotid_image_to_model.py:839` |
| 2 | Keep only the N largest mask components | N = 1 | Anything not connected to the main tree cannot carry flow between boundaries, and would generate spurious skeleton if kept | **On** | `carotid_image_to_model.py:847` |
| 3 | Skeletonise | downsample 1.0 | The whole haemodynamic model is 1D; native resolution because downsampling fuses capillaries a voxel or two apart | **On**, native resolution | `skeleton.py:21` |
| 4 | Remove small skeleton objects | 3 voxels | A 3-voxel fragment is skeletonisation noise, and each one contributes two spurious endpoints to every density measure | **On** | `skeleton.py:526` |
| 5 | Bundle collapse | density 1.0 | Disabled — it destroyed 208 of 307 loops, and $\beta_1$ is the H1 §1.1 readout | **Not called** — guarded out at density ≥ 1.0 rather than called and short-circuited | `skeleton.py:536` |
| 6 | Component fraction filter | 5% | Catches floating fragments large enough to survive the voxel-count filter but still too small to be vessel | **On** | `skeleton.py:540` |
| 7 | Re-skeletonise | — | Removing objects can leave one-voxel remnants that are no longer proper centrelines | **On** | `skeleton.py:546` |
| 8 | Bridge separated skeleton components | 0 | Disabled — repair belongs on the thick mask, where a closing works; on a 1-voxel skeleton the erosion removes the bridge again | **Disabled** | `skeleton.py:547` |
| 9 | skan path extraction → NetworkX MultiGraph | — | Turns the voxel centreline into nodes and edges; a MultiGraph so two capillaries between the same junctions both survive | **On** | `build.py:22` |
| 10 | Terminal reconnection | 5.6 µm | Dead ends within one p99 vessel radius of each other are segmentation breaks, not real terminations | **On** | `build.py:157` |

**Bridging is a closing, not a dilation.** `bridge_gaps` — a plain distance-transform dilation — was
replaced by `close_binary_mask` at both call sites. This matters because a dilation never erodes
back: every vessel would gain a voxel of radius unconditionally, anything within two voxels would
fuse, and the bias is *not* size-neutral. Cross-sectional area goes as radius squared, so +1 voxel
on a 2-voxel radius is +125% area against +36% on a 6-voxel radius — narrow vessels inflated
hardest. It also thickens the wall that EDT calibre measures against (§2.6), which is why an
EDT/FWHM correlation measured on a dilated mask was contaminated. A closing bridges the same gaps
without permanently expanding boundaries. ⚠ The two are *not* interchangeable on a 1-voxel
skeleton, where the erosion step would remove the bridge again — closing is only correct on the
thick mask.

**There is one closing parameter, and there used to be two.** `closing_radius` and
`bridge_gap_size` both called `close_binary_mask`, one after the other. Closing is idempotent for a
fixed structuring element, so at equal radii the second call could never do anything, and at unequal
ones it was a second, differently sized closing wearing the name of a bridge. `bridge_gap_size` has
been removed from `SkeletonConfig`. Radius is not additive across calls: set `closing_radius` to 2
rather than reaching for a second field, because one call at radius 2 is a different and larger
operation than two at radius 1.

**Padding inside the closing.** `scipy` erodes against a zero border, so without a pad every voxel
touching the array edge is removed — a tube crossing the domain boundary lost its entire first and
last slice. That is not a closing (closing is extensive: X ⊆ close(X)), and it moved the point at
which a vessel terminates one slice inside the domain — which matters directly, because inlets and
outlets are identified by vessels reaching the boundary (§2.8). The array is padded and un-padded
around the operation instead.

**Largest-component-only is a hard cut.** `prune_mask_before = 1` keeps a single connected component
of the *mask*. Anything not connected to the main tree is discarded before skeletonisation — not by
size, but by connectivity. This is separate from, and stricter than, the 3-voxel and 5% filters
applied later to the skeleton.

**What skan produces.** `csr.Skeleton` traces the centreline into *paths*. Each path becomes one
edge; its two endpoints become nodes. A **MultiGraph** is used deliberately, so two distinct vessel
segments running between the same pair of junctions are both kept rather than one overwriting the
other — that would silently delete parallel capillaries and, with them, the loops β₁ counts.

**Loop stitching.** Biconnected components of the voxel graph with ≥ 3 members are flagged as voxel
loops (via `igraph`, O(V+E)). This exists because tiny circular skeletonisation artefacts otherwise
shatter the graph. Edges whose two endpoints both lie in a loop cluster are tagged, and are excluded
from terminal reconnection so the repair cannot fuse a genuine loop shut.

**Terminal reconnection is in microns.** Degree-1 nodes within **5.6 µm** of each other are joined,
closest pair first, each node used at most once, skipping any pair already connected or tagged as a
loop edge. 5.6 µm is the p99 inscribed radius — the same figure that sets the stub-pruning threshold
in §2.5. The reconnected edge is straight: its `voxels` list holds only the two endpoints.

**Edge lengths are summed along the path, not endpoint-to-endpoint.** `length` is the sum of
successive step distances through the physical path, so a tortuous segment is longer than the
straight line between its ends. `weight` is the same value floored at 1e-6 to keep shortest-path
routines finite.

**Coordinates are stored in physical ZYX space**, not index space — node `pos` and edge `voxels` are
both multiplied by the voxel size at build time. Nothing downstream multiplies by spacing again.
Getting this wrong once would scale every length, and therefore every resistance and every transit
time. The spacing is resolved *before* the build for this reason; it was previously detected after,
leaving the graph in voxel units.

**Anisotropy.** The voxel is (1.8639, 1.866, 1.866) µm — axial-to-lateral 1.0011, near enough to
isotropic that a single pitch is exact enough for skeleton length. Diameter measurement does *not*
rely on that: FWHM samples transverse profiles in the physical y–x plane only, with no displacement
along z.

> **At a glance** — closing not dilation, largest component only, native-resolution skeletonisation,
> skan MultiGraph with loop stitching, 5.6 µm terminal reconnection ·
> voxel (1.8639, 1.866, 1.866) µm · `skeleton.py:199`, `skeleton.py:477`, `build.py:22`,
> `_helpers.py:27` · `tests/test_graph.py`, `tests/test_length_measurements.py`

---

### 2.5 Topology conditioning

**What it does.** Removes skeletonisation artefacts and simplifies the graph without changing what
it represents.

**The order of operations.** Everything here runs on the *graph*, after §2.4 has built it.
The skeleton-level filters (small-object removal, bundle collapse, component fraction) belong to
§2.4 steps 5–9 and are listed there; the table below picks up where that one stops.

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Reconnect branches that touched a stitched loop | — | Loop stitching replaces a cluster with one hub node; branches that met the old loop would otherwise be orphaned | **On** | `carotid_image_to_model.py:944` |
| 2 | Merge near-coincident nodes, resolve triangles into Y-junctions | 5.6 µm | Skeletonisation renders one anatomical junction as several nodes a voxel apart, splitting a single bifurcation into a triangle | **On** | `carotid_image_to_model.py:949` |
| 3 | Degree-2 collapse, curvature-preserving | — | A node with exactly two edges is not a junction; keeping it inflates node counts and chops one vessel into several segments | **On** | `carotid_image_to_model.py:960` |
| 4 | Terminal stub pruning, iterated to convergence | 5.6 µm, ≤ 100 passes | Spurs at branch points are artefacts; iterated because removing one stub can expose another behind it | **On**; counts printed unconditionally | `carotid_image_to_model.py:973` |
| 5 | Remove self-loop edges on otherwise isolated nodes | — | An edge from a node to itself with no other connection is geometrically impossible and carries no pressure drop | **On** | `carotid_image_to_model.py:986` |
| 6 | Core dead-end resolution (`eradicate` / `stitch`) | — | Both modes invent topology: `eradicate` deletes real capillaries, `stitch` fabricates connections that were never imaged | **Removed from this path**; the operator remains in `graph/prune.py` for `resistance_network_pipeline.py` | `prune.py:111` |
| 7 | B-spline smoothing of every edge centreline | `smoothing_alpha` 0.75 | The voxel centreline is a staircase; tortuosity read off it would measure the sampling lattice rather than the vessel | **On** | `carotid_image_to_model.py:1014` |
| 8 | Keep largest connected component of the graph | `True` | The operators above can sever pieces the mask held together, and a component with no boundary node makes the Laplacian singular | **On** | `carotid_image_to_model.py:1028` |

Three of these are not in the four-operator table below and are worth naming separately.

**Step 4 iterates.** Pruning is run to convergence, up to 100 passes, not once: removing a stub can
expose a new degree-1 node behind it. This cannot change β₁ at any threshold — only degree-1 nodes
are removed and a degree-1 node lies on no cycle — but it *does* change the length and tortuosity
distributions, because it removes the shortest terminal segments first.

**Step 7 changes every edge length.** B-spline smoothing rewrites the `voxels` polyline, and `length`
is measured along it (§2.4), so tortuosity and therefore resistance both move. It is **frozen and
deliberately not tuned**: it sets the centreline curvature that H1 §1.4 reads tortuosity from, and
no Optuna objective can see tortuosity, so tuning it would optimise against a proxy for the very
thing being measured. It is also why the EDT lattice is broken by the time calibre is read (§13.3).

**Step 7 leaves 15–17% of edges raw, and that does not move H1** (re-run package T, measured
2026-10-07 on the six 0.95 batch networks). These edges were tagged `raw_fallback`, and the pipeline
warning said the spline left the skeleton corridor. It did not. 85% of them are 3-point polylines,
which `bspline_smooth_polyline` returns unchanged (it needs 4 points), and the rest have 4–8 points,
where the spline at this smoothness passes through the points (99th percentile departure 0.0005
voxel). The edge is tagged raw because the result equals the input, not because it failed the
corridor test. Every raw polyline lies on the skeleton. So the spline cannot shorten them: run
without the corridor check, or with the corridor widened to 1.5 or 2 voxels, mean tortuosity moves
by at most 0.0003 per specimen and total length by at most 0.012%. They are short (median 5.3 µm)
and the least tortuous smoothed class, not the most (WKY-A mean 1.12, against 1.19 for `bspline`).
At the hard bound, every such edge set to its chord, mean tortuosity falls by 1.5–1.9% in every
specimen and total length by 0.5–0.7%. The SHR/WKY tortuosity ratio stays 0.991 (median ratio
1.001); the groups then separate by 0.0008 against a within-group spread of 0.018, which
`assess_cohort_split` does not flag. The fallback share is not group-correlated (WKY 15.1–16.8%,
SHR 16.2–16.4%, p 0.7). With the spline run unchecked, §1.3 length within TH moves by under 0.06%
at all three TH cuts, and the density ratio stays 1.18 / 1.21 / 1.26. No re-run is needed.
Since re-run package V these edges are tagged `raw_unchanged`. `raw_fallback` and its warning are
kept for a real corridor failure, where every candidate leaves the skeleton support; none occurred
in the six batch networks. The batch outputs on disk carry the old tag until the batch is next
re-run (package X); every number stays the same.

**Step 8 is the second largest-component cut.** §2.4 step 3 already kept one component of the
*mask*. This keeps one component of the *graph*, because the topology operators above can sever
pieces that the mask held together. Both are on.

**Four operators, and what each is allowed to touch:**

| Operator | Setting | Effect on β₁ |
|---|---|---|
| Terminal stub pruning | 5.6 µm | **None.** Removes only degree-1 nodes, which lie on no cycle. Verified constant at 307 from 0 to 30 µm |
| Degree-2 collapse | on | None — merges chains, preserving cycles |
| Bundle collapse | **disabled** | Would destroy it. See below |
| Component filtering | 5% | Removes disconnected fragments entirely |

**Why bundle collapse is disabled.** The operator deletes dense skeleton regions and replaces each
with one hub node. Density is a uniform filter over the *skeleton*, so a single centreline crossing
the 9³ window contributes 9/729 = 0.0123 and two contribute 0.0247. Its former setting of 0.025 was
therefore "collapse anywhere two capillaries pass within 16.8 µm" — in a capillary bed, the normal
condition rather than a defect.

Measured on the reference subvolume:

| `bundle_density_fraction` | Skeleton voxels | V | E | β₁ |
|---|---|---|---|---|
| 0.025 | 4,788 | 398 | 496 | **99** |
| 0.050 | 6,805 | 1,007 | 1,318 | 312 |
| disabled | 6,789 | 991 | 1,297 | **307** |

It destroyed 208 of 307 fundamental loops — 68% of β₁, which *is* the H1 §1.1 readout — and 29% of
the skeleton with them. It is also group-dependent in the false-negative direction: a denser network
exceeds the threshold in more places, fires more hubs, and loses proportionally more loops, so it
actively suppresses the SHR/WKY difference it is meant to be measuring.

**There is no better operating point, only no operating point.** The density distribution has no
gap — it runs smoothly from 0.02 to 0.06 — so no threshold separates "pathological bundle" from
"capillary bed".

**Why the stub threshold is 5.6 µm.** A skeletonisation spur at a branch point cannot be longer than
the local vessel radius, and the measured inscribed radius is p90 3.73 µm, p99 5.60 µm. Measured
terminal-stub lengths run 3.47–129.83 µm with p25 = 10.94 µm, so a 10 µm cut would sit just below
the lower quartile of *genuine* terminal branches.

> **At a glance** — stubs pruned at 5.6 µm, bundle collapse disabled · β₁ = 307, invariant to stub
> length · `skeleton.py:472`, `graph/prune.py`, `graph/degree2.py` · `tests/test_graph.py`

---

## §2 — Geometric model, part 3: measuring the graph

Sections 2.6 to 2.8 attach the physical quantities the haemodynamics needs — a diameter, a
topological order, and a set of pressure boundaries — to the graph part 2 produced.

---

### 2.6 Calibre assignment

**The single most consequential choice in the pipeline.** Everything in §13.1 follows from it.

**What it does.** Assigns each edge a diameter, from which resistance, lumen volume, surface area
and transit time all follow.

**What was chosen.** EDT — the Euclidean distance transform's inscribed radius — with a
junction-proximity exclusion of 3.73 µm.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Branch orders assigned from the inlets (§2.7) | — | The synthetic fallback law is indexed by branch order, so the labels have to exist even on the path that refuses to use them | **On** | `carotid_image_to_model.py:1076` |
| 2 | Synthetic branch-order diameter table filled by exponential fit | 3 anchor points | Fills every observed order so the fallback dict cannot raise on a lookup; the zero-tolerance guard then forbids its use anyway | **On**, but never consumed under `edt_radius` | `carotid_image_to_model.py:1141` |
| 3 | FWHM ray-casting over the raw field | half-extent 15.0 µm | Off — it is fitted to a probability field that saturates at 1.0 inside a vessel, and its half-extent reaches about 4.7 vessel radii into the neighbours | **Off** — `radius_assignment_mode = "edt_radius"` | `carotid_image_to_model.py:1144` |
| 4 | 3D EDT of the binary mask, in physical units | `sampling` = voxel size | The inscribed radius is bounded by the mask, so unlike FWHM it cannot read across into an adjacent vessel | **On** | `automated.py:1283` |
| 5 | Sample the EDT at every centreline voxel of every edge | — | One reading per edge would land wherever the polyline happened to start; sampling the whole centreline gives a distribution instead | **On** | `automated.py:1305` |
| 6 | Drop samples outside the mask or at radius 0 | — | A centreline point that rounds to outside the mask has no inscribed radius to report | **On** | `automated.py:1316` |
| 7 | Flag which of an edge's two ends are junctions | — | The trim must fire only at ends that actually abut a junction — trimming a free terminal would discard real vessel | **On** | `automated.py:1323` |
| 8 | Trim samples within the exclusion of a junction end | 3.73 µm | Within about one radius of a bifurcation the EDT returns the junction's inscribed sphere rather than the vessel's, biasing calibre upward | **On** | `automated.py:1327` |
| 9 | If nothing survives, keep the untrimmed median and tag it | — | Dropping segments shorter than the exclusion would delete short inter-junction capillaries and bias the distribution towards long vessels | **On** — `untrimmed_too_short`, 35.5% of edges at 0.95 (61% on the 0.90 centred-box outputs) | `automated.py:1338` |
| 10 | Per-edge diameter = 2 × **median** surviving radius | — | Robust against a single local bottleneck or bulge setting the calibre of the whole edge | **On** | `automated.py:1345` |
| 11 | Refuse if any edge fell back to a synthetic diameter | `MAX_SYNTHETIC_FRACTION_EDT = 0.0` | EDT has no legitimate per-edge failure mode on a mask that covers the vessel, so any fallback is a defect rather than an expected shortfall | **On**, raises | `poiseuille.py:13` |
| 12 | Resistance from the assigned diameter | Hagen–Poiseuille | Converts the measured geometry into the one quantity the flow solve consumes | **On** | `poiseuille.py:160` |

**Median, not mean, along the edge.** Step 10 takes the median of the per-voxel diameters so a
single local bottleneck or bulge cannot set the edge's calibre. The full sample list is retained as
`edt_diameter_samples_um`, so the within-edge spread stays recoverable.

**The trim is recorded, not just applied.** Every edge carries `edt_junction_trim` as one of
`trimmed`, `no_junction`, `untrimmed_too_short` or `not_applied`. This is what makes the untrimmed
share in step 9 countable rather than an estimate: 64.2% `trimmed`, 35.5% `untrimmed_too_short`,
0.3% `no_junction` over the 42,211 edges of the 2026-09-28 batch.

**Step 2 runs even though step 11 forbids its output.** The branch-order diameter table is built on
every run, then never read under `edt_radius` — and if it ever were read, the zero-tolerance guard
would raise first. It is live code on a dead path, the same shape the entropy parameters had
before they were switched off (§2.3).

**The alternative.** FWHM: fit a Gaussian plus baseline to the intensity profile across the vessel
and report $2\sqrt{2 \ln 2}\,\sigma$.

**Why EDT, on the evidence.** Both estimators run over the same 1,330 edges:

| Estimator | Coverage | Median | p95 | Max |
|---|---|---|---|---|
| EDT | 1,330 (100.0%) | 6.37 µm | 11.34 | 20.09 |
| FWHM | 1,017 (76.5%) | 8.20 µm | 16.78 | 39.16 |

They correlate weakly — Pearson *r* = 0.245, Spearman ρ = 0.284, median ratio FWHM/EDT = 1.359. The
two genuinely disagree.

FWHM reads 36% larger at the median and its tail is not physical for a bed whose measured inscribed
radius is p99 5.60 µm. Two mechanisms account for it: the pipeline hands FWHM the *probability
field*, which saturates at 1.0 inside a vessel, so the Gaussian is fitted to a plateau; and the
transverse half-extent of 15.0 µm is about 8 voxels, roughly 4.7 vessel radii, so in a bed this
dense the profile runs into neighbouring vessels. EDT is bounded by the mask and can do neither.

**The junction exclusion.** Within about one radius of a bifurcation the EDT returns the *junction's*
inscribed sphere rather than the vessel's, biasing calibre upward. Measured over 3,882 edges,
sweeping the exclusion in half-voxel steps:

| Exclusion (µm) | Voxels | Trimmed | Too short | Moved | Mean (µm) | Mean shift | $r^{-4}$ factor |
|---|---|---|---|---|---|---|---|
| 0.93 | 0.5 | 77.6% | 22.3% | 29.8% | 5.516 | −0.060 | 1.044 |
| 1.87 | 1.0 | 74.9% | 25.0% | 28.9% | 5.509 | −0.067 | 1.049 |
| 2.80 | 1.5 | 50.0% | 49.9% | 25.9% | 5.470 | −0.106 | 1.080 |
| **3.73** | **2.0** | 38.5% | 61.4% | 19.7% | 5.473 | −0.102 | 1.077 |
| 5.60 | 3.0 | 20.0% | 79.9% | 12.1% | 5.516 | −0.060 | 1.044 |

The delivered correction peaks near 1.5 voxels and falls away on both sides — too small removes
nothing, too large leaves most segments untrimmable and carrying the full bias. 3.73 µm is kept
because it is specified externally and corresponds to about one capillary inscribed radius, whereas
2.80 µm would be tuned to this subvolume. The difference between them is **0.3% on resistance**.

**Two corrections this measurement makes.** The population-level effect is ~8% on resistance, not
the 3.2× a synthetic single-edge fixture shows. And the bias is *not* concentrated on short
segments: median segment length is 7.2 µm — 3.9 voxels — so every segment is short relative to the
exclusion, and the shift is if anything larger on longer ones (−0.181 µm in the shortest quartile
rising to −0.298 µm in the longest), because a junction's inscribed sphere scales with the vessel it
belongs to. 176 of 765 moved edges got *wider*, where the junction sat on the narrow side of a
calibre step.

**Fabricated calibre is refused, not warned about.** `MAX_SYNTHETIC_FRACTION_EDT = 0.0`. EDT has no
legitimate per-edge failure mode on a mask that covers the vessel — 100% measured provenance was
observed across 34,900 edges, and across all 42,211 of the 2026-09-28 re-run — so any fallback is a defect rather than an expected shortfall. FWHM
is exempt by default, because Gaussian fitting genuinely fails on individual edges of a probability
field; the fraction is still reported.

**The branch-order fallback law.** Where a diameter must be synthesised it comes from an exponential
function of branch order. **Murray's law is not used** [`murray_physiological_1926`]. Under the
default EDT mode this path cannot silently activate at all — the guard above refuses first.

> **At a glance** — EDT inscribed radius, 3.73 µm junction exclusion, zero tolerance for synthetic
> calibre · EDT 100%/6.37 µm vs FWHM 76.5%/8.20 µm, *r* = 0.245 · `poiseuille.py:160`,
> `poiseuille.py:16`, `automated.py:1238`, `automated.py:971` · `tests/test_edt_diameter.py`,
> `tests/test_haemodynamics_automated_fwhm.py`, `tests/test_silent_fallback_guards.py`

---

### 2.7 Branch-order assignment

**What it does.** Labels each edge with its topological distance from the inlet set.

**How.** BFS from the starting nodes gives every node a hop distance. An edge takes
`min(dist(u), dist(v)) + 1`, formatted `B01`, `B02`, ….

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | BFS from the starting node set → per-node hop distance | — | Hop distance from the inlets is the only ordering available without assuming a calibre hierarchy | **On** | `branch_order.py:104` |
| 2 | Skip edges outside `included_edges` / inside `excluded_edges` | both empty | Lets a caller label a sub-network; both sets are empty here, so every edge is considered | **On**, but vacuous | `branch_order.py:122` |
| 3 | Edges with both ends unreachable → `unreachable_edges` | — | An edge no inlet can reach has no defined order; listing it keeps the gap countable instead of inventing a value | **On**; skipped, not defaulted | `branch_order.py:131` |
| 4 | Edge order = `min(dist(u), dist(v)) + 1` | — | An edge sits one hop beyond whichever of its two ends the blood reaches first | **On** | `branch_order.py:135` |
| 5 | Format as `B01`, `B02`, … and write to the edge | zero-padded to 2 | Zero padding keeps the labels sorting correctly as strings, which the diameter dictionary relies on | **On** | `branch_order.py:136` |
| 6 | Count edges per order, report the unique set | — | Reports how deep the network goes, and is what the exponential fill in §2.6 is sized against | **On**, printed | `carotid_image_to_model.py:1077` |

`assign_hierarchical_branch_orders` (`branch_order.py:153`) is a separate, richer labelling that the
CB path does not call.

**What the label is, and is not.** It is a **hop count from an inlet**. It is not a Strahler order,
not a Horton order, and not a calibre class. `B01` means "one edge from a pressure inlet", so what
it denotes depends entirely on the boundary selection of §2.8 — change the inlets and every label
moves.

**Edges that cannot be labelled.** An edge unreachable from every starting node is recorded in
`unreachable_edges` and skipped rather than given a default order.

**Where the labels are consumed.** The synthetic diameter law (§2.6) and the frozen constriction
geometry (§3.3). Neither is active on the default CB path, so branch order is currently
descriptive rather than load-bearing.

> **At a glance** — BFS hop count from inlets, `min(u,v)+1` · labels `B01…`, unreachable edges
> skipped not defaulted · `branch_order.py:95`, `branch_order.py:153` ·
> `tests/test_branch_order_hierarchy.py`

---

### 2.8 Boundary terminal node selection

**What it does.** Decides which degree-1 nodes receive arterial pressure and which receive venous.

**What was chosen.** The face-crossing rule on **axis 1**, with a tolerance of one voxel, in the
main pipeline and in every H2 driver. `cb_settings.BOUNDARY_AXIS` and
`BOUNDARY_FACE_TOLERANCE_VOXELS` own both values; `GraphConfig.boundary_axis` and
`face_tolerance_voxels` read them, and `test_cb_settings.py` keeps them equal.

> **Open item 2, closed.** Until 2026-09-27 the main pipeline ran the band rule
> (`select_boundary_terminal_nodes`, axis 0, `edge_percent` / `end_percent` = 25 / 25) while the
> H2 drivers ran the face rule on axis 1. On the six batch graphs the band rule put pressure on
> 216–399 terminals per specimen; the face rule put it on 16–37 (WKY-A 126 + 119 → 18 + 9), and on
> the 2026-09-28 graphs puts it on 27–44 (WKY-A 23 + 21). The
> pipeline now calls `select_boundary_terminal_nodes_by_face` with the `GraphConfig` values, so on
> the same graph it picks exactly the nodes the H2 drivers pick.
>
> **No published number moved.** β₁, calibre, length and tortuosity are fixed before boundaries are
> chosen. The pipeline outputs that depend on the inlets are the `branch_order` column of
> `per_edge_morphometry.csv` (§2.7), the pipeline's own flows and rheology, `model_results.md` and
> the Tier 3 `*_perfusion.vti`. No H1 or H2 script reads any of them. The batch outputs of the
> 2026-09-28 re-run (items 27, 13, 15 and 17) carry face-rule values.
>
> **A guard came with it.** The face rule measures terminals against the faces of the shape it is
> given, so `_check_graph_fits_frame` raises if a node lies more than one voxel outside that shape.
> It stopped the `--use-cache-dir` path on an `.h5` input, which passed a `(1, 1, 1)` placeholder,
> instead of solving on invented boundaries. That path now takes the cached mask's shape (open
> item 28), and a second check raises when the shape passed differs from the mask's, which catches
> a shape too large as well as too small.

**The order of operations — the face rule, as the pipeline and the H2 drivers run it.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Collect degree-1 nodes that carry a `pos` | — | Only a dead end can be a vessel entering or leaving the region; an interior junction is already connected on both sides | **On** | `boundaries.py:169` |
| 2 | Scale the axis extent from voxels into microns | `voxel_size[axis]` | `image_shape` is in voxels while node `pos` is in microns — comparing them unscaled shrinks the apparent volume and drags interior dead ends into range | **On** | `boundaries.py:166` |
| 3 | Low-face terminals within tolerance → inlets | 1 voxel | A vessel supplying this region has to cross one of its faces; a dead end mid-volume cannot be an inlet whatever its coordinate | **On** | `boundaries.py:174` |
| 4 | High-face terminals within tolerance → outlets | 1 voxel | The opposite face is where that blood has to leave | **On** | `boundaries.py:178` |
| 5 | A terminal within tolerance of both faces → low face wins | — | A terminal within tolerance of both faces would mean a region one voxel thick; the ambiguity is resolved rather than silently doubled into both sets | **On** | `boundaries.py:176` |
| 6 | Raise if either face carries no terminal | — | An unsolvable region should stay unsolvable rather than be solved with invented boundaries | **On**, no fallback | `boundaries.py:181` |
| 7 | Non-face terminals: nothing under `caged` | `caged` | An interior dead end is a mask defect, not a vessel, so it earns no pressure | **On** | `boundaries.py:191` |

**And the band rule, which the CB path no longer runs.** It remains in the library for the nerve
pipeline (through `select_boundary_nodes_by_method`) and as the comparison rule in
`cb_h2_absolute_perfusion.py`. The table describes it as the CB pipeline ran it until open item 2.

| # | Step | Setting | Why | On the CB path until item 2 | Where |
|---|---|---|---|---|---|
| 1 | Collect degree-1 nodes that carry a `pos` | — | Only a dead end can be a vessel entering or leaving the region; an interior junction is already connected on both sides | **On** | `boundaries.py:53` |
| 2 | Scale the axis extent from voxels into microns | `voxel_size[axis]` | `image_shape` is in voxels while node `pos` is in microns — comparing them unscaled shrinks the apparent volume and drags interior dead ends into the band | **On** | `boundaries.py:48` |
| 3 | Terminals in the lowest `edge_percent` of the axis → inlets | 25% | A positional proxy for "near the arterial end", with no anatomical anchor behind the width | **On** | `boundaries.py:55` |
| 4 | Terminals in the highest `end_percent` of the axis → outlets | 25% | The same proxy at the other end of the axis | **On** | `boundaries.py:56` |
| 5 | If either set is empty, **raise** | — | An unsolvable region should stay unsolvable rather than be solved with boundaries picked from nodes that are not vessel terminations | **On** — the fallback to the extreme decile of *all* nodes now runs only when a caller passes `allow_extreme_fallback=True`. The CB path never does; the nerve pipeline's `select_boundary_nodes_by_method` does | `boundaries.py:60` |
| 6 | Route the remainder by permeability mode | `caged` | Decides whether the remaining interior dead ends drain, resist, or simply stop | **On** — non-band terminals are not boundaries | `boundaries.py:83` |
| 7 | Drop any node that landed in both sets | — | A node cannot carry two different Dirichlet pressures at once | **On**, inlets win | `boundaries.py:99` |
| 8 | Sort inlets ascending, outlets descending, by axis coordinate | — | Makes `resistance_node_pair` deterministic across runs, since it takes the first entry of each list | **On** | `boundaries.py:101` |

Step 5 used to be a fallback, and it was live on the H1 path until the band rule was made to
raise: a graph with no terminal in the band became a solved one by promoting the extreme decile of
*all* nodes — interior spurs included — to pressure boundaries. None of the six H1 runs took it:
each `pipeline.log` reports inlet and outlet counts drawn from terminals (37–226 per side), not the
equal-sized decile the fallback returns. It survives only as an explicit opt-in for the nerve
pipeline.

**Why not a positional band.** About **86% of degree-1 nodes in these graphs are interior** — nowhere
near a region face. They are skeletonisation spurs and segmentation breaks, not vessels entering the
volume. A band rule therefore assigns arterial pressure to mask defects, and the fraction it catches
depends on a band width with no anatomical meaning.

A vessel supplying this region has to cross one of its faces. A dead end in the middle of the volume
cannot be a pressure inlet whatever its coordinate.

**Measured**, varying each rule's own free parameter over its plausible range and taking the spread
of the shunt ratio per specimen:

| Rule and parameter range | Ratio spread |
|---|---|
| Band, axis 1, width 10/25/40% | 73.9% |
| **Face, axis 1, tolerance 1/2/4 voxels** | **8.9%** |

Re-measured on the 2026-09-28 networks (`cb_h2_boundary_selection.py`; 75.8% and 13.3% on the
centred boxes before), and unchanged when the script moved from the ParaView export to the batch
MultiGraph (re-run package O, 2026-10-02). An 8.3-fold reduction, and it comes from the *parameter*, not the axis. The band width has no
principled value, so its whole plausible range is live. The face tolerance is anchored to the voxel
size — one voxel means "on the face" — and the other values exist only to show the answer does not
depend on it.

> **Comparing at a fixed second parameter is misleading, and initially pointed the other way.**
> Axis spread alone is 34.2% for the band rule against 33.6% for the face rule (28.4% and 31.5%
> on the centred boxes), which flatters the band rule by holding the parameter that damages it at its
> default. Both parameters have to move: total spread 113.5% (band) against 40.0% (face).
> These multi-axis figures are from the batch graph (package O). The ParaView-frame run gave 34.9%
> and 113.7% / 44.9%. It read graph-order (z, y, x) node coordinates against the mask's (x, y, z)
> bounds and voxel sizes, so on axes 0 and 2 the face cut used the other axis's extent and spacing.
> Node coordinates sit on voxel planes, so that dropped the plane one voxel in from the axis-0
> outlet face and the axis-2 inlet face (WKY-A, axis 2: 16 inlets against 31). Axis 1 (y) is the
> same in both frames, so the pinned-axis table above did not move.

**Why axis 1.** Not anatomy — availability. On the centred-box graphs it was the only axis solvable
in all six specimens: axis 0 had no outlet terminal in SHR-A, and axis 2 no inlet terminal in
SHR-C. That was a selection criterion, and a property of those graphs rather than a general rule.
**On the 2026-09-28 graphs all three axes are solvable in all six**, so availability no longer
singles out axis 1; it stays because it is pinned in `cb_settings.BOUNDARY_AXIS` and every H2
number uses it. *Needs review:* whether the axis choice should be re-argued or its spread across
axes reported.

**The mask's virtual padding is still on axis 0.** Under `caged`, `_apply_preprocessing_filters`
pads 10 slices on axis 0 only (edge mode) before filtering and strips them after, so vessel ends on
the axis-0 faces are protected from edge effects and those on the axis-1 faces are not. It was set
up for the band rule's axis. Moving it would change the masks and so every H1 number, and the H2
drivers already run the face rule on axis 1 on these same graphs, so it is left as it is.

**It raises rather than falling back.** If a face carries no terminals the rule refuses. A band
fallback would drop to the extreme 10% of *all* nodes, converting an unsolvable region into a solved
one with invented boundaries.

**One tie-break.** A terminal within tolerance of both faces would mean a region one voxel thick.
The low face wins; the ambiguity is not silently doubled into both sets.

**Permeability modes.** Under `caged` (default) non-face terminals are simply not boundaries.
`universal_sink` adds them all as outlets; `robin_resistance` tags them for a distal resistance.

> **At a glance** — face-crossing rule, axis 1, 1-voxel tolerance, raises on an empty face ·
> 86% of degree-1 nodes are interior; ratio spread 8.9% vs 73.9% · `boundaries.py:106`,
> `boundaries.py:208` · `tests/test_boundary_faces.py`

---

## §3 — Haemodynamic model: 1D network flow

### 3.1 Segment resistance

Each edge is a rigid cylinder obeying Hagen–Poiseuille:

$$R = \frac{128\,\mu L}{\pi d^{4}}$$

with $L$ the centreline length in µm, $d$ the assigned diameter in µm, and $\mu$ an apparent
viscosity in cP.

**The $d^{-4}$ term governs everything downstream.** A fractional calibre error becomes roughly four
times itself in resistance. §13.1 gives the measured size of that; it is not restated here.

### 3.2 Two viscosity models exist, and which one is in force depends on the path

This is the easiest thing in the pipeline to get wrong.

**The initial assignment uses a power law.** `PoiseuilleModel.set_poiseuille_resistances` computes

$$\mu = \frac{1}{d^{1.647}}$$

which is not a viscosity in cP and is not Pries–Secomb. It is an empirical stand-in that produces
the right ordering of resistances but not their physical magnitudes.

**The rheology solver overwrites it.** On entry, `solve_coupled_flow_and_hematocrit` recomputes both
viscosity and resistance for every edge from Pries–Secomb at the systemic haematocrit, discarding
whatever `set_poiseuille_resistances` assigned:

$$\begin{aligned}
\mu_\text{app} &= \operatorname{pries\_secomb}(d,\; H = 0.45) \\[2pt]
R &= \frac{128\,\mu_\text{app} L}{\pi d^{4}}
\end{aligned}$$

**The order of operations — which viscosity is in force, when.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | `set_poiseuille_resistances` writes R from the power law | $\mu = 1/d^{1.647}$ | Gives every edge a resistance with the right ordering cheaply, so the graph is solvable before any rheology runs | **On** | `poiseuille.py:160` |
| 2 | Rheology solver **overwrites** R from Pries–Secomb at systemic Hct | H = 0.45, µ_plasma 1.2 cP, in vivo law | The power-law µ is not a viscosity in cP; physical magnitudes have to come from a real relation | **On**, before the loop | `rheology.py:236` |
| 3 | Each Picard pass recomputes µ_app from the edge's current Hct | in vivo Pries–Secomb | Apparent viscosity depends on the local haematocrit, which the previous pass has just changed | **On** | `rheology.py:390` |
| 4 | R **recomputed** from Hagen–Poiseuille at the new µ_app | $R = 128\,\mu_\text{app} L / \pi d^{4}$ | The same expression as step 2, so initialisation and update agree. Until `7ea1b36` this step rescaled a stored base by $\mu_\text{app} / \mu_\text{old}$ — see closed item 12 below | **On** | `rheology.py:408` |
| 5 | ~~`original_resistance` captured once~~ | — | Removed in `7ea1b36` with the rescale it served | **Removed** | — |

> **Item 12 — closed by `7ea1b36`; numbers re-derived 2026-09-27.** Until that commit step 4
> double-applied viscosity, inflating every resistance by roughly 200–540×. The record below is kept
> because it explains why the H2 flow numbers before 2026-09-27 differ from the current ones.
>
> The rescale was correct *if* `original_resistance` holds the power-law resistance, because
> $R_\text{old} \times \mu_\text{app}/\mu_\text{old}$ then telescopes to $128\,\mu_\text{app} L / (\pi d^{4})$. It does not. Step 2 overwrites
> `data["resistance"]` with the **Pries–Secomb** value before the loop starts, and step 5 captured
> `original_resistance` from that overwritten value on the first pass. The power-law µ_old therefore
> divided a resistance that no longer contained it, and the surviving factor was
> $\mu_\text{PS}(d, 0.45)\cdot d^{1.647}$:
>
> | d (µm) | $\mu_\text{PS}(d, 0.45)$ cP | $\mu_\text{old} = 1/d^{1.647}$ | Inflation factor |
> |---|---|---|---|
> | 3 | 37.97 | 0.1638 | **232×** |
> | 4 | 21.24 | 0.1020 | **208×** |
> | 5 | 15.14 | 0.0706 | **215×** |
> | 6 | 12.01 | 0.0523 | **230×** |
> | 8 | 8.77 | 0.0326 | **269×** |
> | 12 | 5.95 | 0.0167 | **357×** |
> | 20 | 3.88 | 0.0072 | **539×** |
>
> **It did not cancel from ratios.** The factor is a function of diameter, not a constant, so it
> varied 1.3× across the capillary band (3–8 µm) and 2.3× across the full measured range. Wider
> vessels were penalised hardest, which redistributed flow away from them. This is unlike the uniform
> pressure and viscosity scalings of §4.4 and §8.1, which genuinely do cancel.
>
> **Iteration 0 was clean; every later iteration was not.** The first pressure solve ran on the
> step-2 resistances, which were correct. The corruption entered at the end of iteration 0, so the
> returned graph carried the inflated values.
>
> **The explanation for §13.5 held.** Absolute perfusion there was 20–100× low, and reaching
> 500 µm/s seemed to need about 3,257 mmHg against the 40 mmHg used. This block proposed the inflated
> resistance as the cause. Re-derived (`examples/cb_h2_absolute_perfusion.py`), the flow-weighted
> velocity rose 240–343× per specimen, to 979–3,319 µm/s, and a 500 µm/s velocity needed 6–20
> mmHg (centred boxes; 977–1,812 µm/s and 11–20 mmHg since the 2026-09-28 re-run, §13.5). Run with the pre-fix rheology code patched in, the same script gives back the old table to
> the digit, so the graphs did not change and the rheology code accounts for all of it. §13.5 has the
> new table.
>
> **Ratios moved in value, not in pattern.** β₁, calibre, length and tortuosity are fixed before any
> resistance is computed (§2.4–§2.6), so H1 is untouched. The H2 §2.1, §2.2 and §2.4 ratios were
> computed on the inflated field (`cb_h2_glomus_perfusion.json`, 2026-08-19; kept as
> `..._2026-08-19_pre_item12.json`) and were re-run. Per specimen they moved by up to 0.09 (WKY-C
> haematocrit ratio 1.023 → 0.930; SHR-C shunt index 1.044 → 1.124). Cohort means moved by at most
> 0.04. The SHR/WKY ratios went 1.11 → 1.10 (shunt index), 1.21 → 1.22 (median flow), 0.93 → 0.93
> (haematocrit) and 0.68 → 0.72 (transit). Flow ratio and transit ratio still separate the cohorts
> with no overlap; shunt index and haematocrit ratio still overlap. Part of each move comes from the
> later #98 fixes (`d9bafbc` feeding-vessel phase separation, `be28179` under-relaxation, `7f05a78`
> rescale once), which the re-run also picked up; `7ea1b36` alone moved the ratios by a similar
> amount. H2 §2.3 had already been re-run after all four (2026-09-26), so it is unchanged.

**So the power law survives only if the rheology solve is not run.** When it is run, it is an
initial condition that is replaced rather than blended — which is what §11 row 12 means by
"replaced by the per-pass Poiseuille recompute".

> **Item 9 — closed by `f92a96c`.** The rheology solver used to fall back to **5.0 µm** silently at
> initialisation and at every update, so a cached graph carrying no calibre solved at a uniform
> 5 µm and reported nothing. It now calls `_require_diameters` before initialising and raises
> unless every edge has a positive `assigned_diameter_um` or `fwhm_diameter_um`, matching
> `map_vessels_to_grid`. `default_diameter_um` opts in to a stated calibre at every step. The H2
> drivers still load diameters from `per_edge_morphometry.csv` first, which is now required rather
> than a workaround.

### 3.3 Variable-diameter segments · **Frozen**

Where a segment's diameter varies along its length, resistance is integrated rather than evaluated
once:

$$R = \int_{0}^{L} \frac{128\,\mu\bigl(d(x)\bigr)}{\pi\, d(x)^{4}}\, dx$$

by trapezoidal quadrature over 1,000 sample points, with $\mu(d) = 1/d^{1.647}$ as above.

Two geometries are implemented. **Sphincter**: one constriction at the vessel origin — ramp down
over the first quarter of the constriction length, hold at *d₂* through the middle half, ramp back
up over the last quarter. **Periodic**: the same shape repeated at a fixed spacing.

Both are disabled on the CB path (§10.6), so this integral is not evaluated in any current run.

### 3.4 Network solve

Mass conservation at every internal node is Kirchhoff's current law. With conductance
*C_uv* = 1/*R_uv*, the system is the weighted graph Laplacian **L** = **D** − **C**:

$$\begin{aligned}
\mathbf{L}\,\mathbf{p} &= 0 && \text{at every unconstrained node} \\[2pt]
p &= p_\text{in} && \text{at inlet terminals} \\[2pt]
p &= p_\text{out} && \text{at outlet terminals}
\end{aligned}$$

Solved by partitioning into known and unknown nodes and taking the Schur complement:

$$\mathbf{L}_{uu}\,\mathbf{p}_u = -\,\mathbf{L}_{uk}\,\mathbf{p}_k$$

**The order of operations.** On the CB path this runs inside the rheology loop (§4.3) rather than
standalone; `solve_flow_from_conductance_matrix` (`resistance.py:201`) is the equivalent entry point
for graph-only callers and adds VTK export. The CB driver also calls it once after the rheology
solve, only to write `*_vessels_flow.vtp`: it divides by the file's `resistance` array, so the
driver writes the solved resistances to the file first (before `aecc53d` it divided by the
power-law values, and after it by NaN).

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Build the sparse conductance matrix, C_uv = 1/R_uv | — | The Laplacian is assembled from conductances, and inverting each resistance once is cheaper than doing it per matrix entry | **On** | `resistance.py:46` |
| 2 | Laplacian **L** = **D** − **C** | — | Kirchhoff's current law at every internal node is exactly $\mathbf{L}\mathbf{p} = 0$ | **On** | `resistance.py:138` |
| 3 | Map inlet node ids → p_in, outlet node ids → p_out | 60/20 mmHg as run (§8.1) | Without a pressure fixed somewhere the system is singular — only differences would be determined, not levels | **On** | `rheology.py:219` |
| 4 | Partition indices into known and unknown | — | Boundary pressures are already known, so those rows are removed rather than solved for | **On** | `rheology.py:227` |
| 5 | Schur complement: $\mathbf{L}_{uu}\,\mathbf{p}_u = -\,\mathbf{L}_{uk}\,\mathbf{p}_k$ | — | Reduces the system to the unknown block, carrying the known pressures across to the right-hand side | **On** | `rheology.py:234` |
| 6 | Direct sparse factorisation | `spsolve`, below 50,000 unknowns | Exact to machine precision, and cheap at the ~10³ nodes these graphs carry | **On** — CB graphs are ~10³ nodes | `resistance.py:152` |
| 7 | Exclude components carrying no pressure boundary, then **raise** if the solve is still singular | — | Nothing drives such a component, so it has no defined pressure; least squares used to answer it with a well-formed field that meant nothing | **On** — excluded nodes get NaN and a warning naming the count | `resistance.py:288` |
| 8 | Above 50,000: CG with an **ILU** preconditioner | `drop_tol` 1e-4, `fill_factor` 10, `rtol` 1e-8, `maxiter` 1000 | Off — a direct factorisation of a 50,000-node system would exhaust memory, so the large case needs an iterative path | **Off** — threshold never reached | `resistance.py:161` |
| 9 | Per edge, $Q = (p_u - p_v) / R$, signed | — | Ohm's law for the hydraulic analogue: once pressures are known, flow follows without another solve | **On** | `rheology.py:252` |
| 10 | Direct the edge from high pressure to low | — | Haematocrit transport and transit time both need to know which way blood actually moves along each edge | **On**, builds the DAG | `rheology.py:261` |

**The iterative branch uses ILU, not Jacobi.** The Jacobi preconditioner described in §9.3 belongs
to the *tissue diffusion* solve (`perfusion.py`), which is a different matrix. The network branch at
step 8 is incomplete-LU, and it never executes on these graphs in any case.

**Step 7 is a silent fallback.** A singular Laplacian — which is what a disconnected component with
no boundary node produces — is caught and answered with a least-squares solution rather than raised.
The graph-level largest-component prune (§2.5 step 8) is what keeps this from firing, so the two are
coupled: turning that prune off would route this path into a silent approximation.

Edge flow then follows directly from $Q = (p_u - p_v) / R$, signed, and the edge is directed from
high pressure to low.

**Solver dispatch.** Direct factorisation below 50,000 nodes, iterative above. The CB graphs sit
far below that, so the flow solve is direct and exact to machine precision.

### 3.5 Effective two-point resistance

Available from the Laplacian pseudo-inverse, giving the resistance between any node pair as if the
rest of the network were a passive medium. Implemented and exported; not part of the H1 or H2
readouts.

### 3.6 Wall shear stress

Derived per edge from flow and calibre and written to `wall_shear_stress_pa`. Exported to the
reporting layer and the VTK artefacts. Not part of the H1 or H2 readouts, and no assumption in §11
constrains it — treat it as diagnostic rather than reportable.

### 3.7 Units of flow

**The flow solve does not work in one unit system.** $R = 128\,\mu L / (\pi d^{4})$ is evaluated with
pressure in mmHg, viscosity in cP and lengths in µm, so its *Q* carries mmHg·µm⁴/(cP·µm) and is
**not a volumetric flow rate**.

Rewriting *R* wholly in SI multiplies it by $(10^{-3}\,\mathrm{Pa\,s/cP}) \cdot (10^{-6}\,\mathrm{m/\mu m}) / (10^{-6}\,\mathrm{m/\mu m})^{4}$ = 10¹⁵.
So

$$\begin{aligned}
Q_\text{SI}\;[\mathrm{m^3/s}]
&= \frac{\Delta P_\text{mmHg} \times 133.322387415}{R_\text{pipeline} \times 10^{15}} \\[4pt]
&= Q_\text{pipeline} \times 133.322387415 \times 10^{-15}
\end{aligned}$$

and multiplying by 10¹⁸ µm³/m³ leaves

$$\texttt{POISEUILLE\_FLOW\_TO\_UM3\_PER\_S} = 133.322387415 \times 10^{3}$$

**Where it is applied.** In `map_vessels_to_grid`, at the boundary between the 1D solve and the 3D
tissue — the one place the two unit systems have to agree. Passing `flow_to_um3_per_s=1.0` leaves
flow in solver units deliberately, for comparison against output produced in those units.

**The factor depends on the caller's pressure unit.** The derivation above assumes pressure in
mmHg, which is what the H2 drivers pass. `carotid_image_to_model.py` passes pressure in mPa
(`HaemodynamicsConfig.input_p_bc`/`output_p_bc`); with viscosity in cP (mPa·s) and lengths in µm its
*Q* is already in µm³/s. The pipeline therefore passes `flow_to_um3_per_s` =
`POISEUILLE_FLOW_TO_UM3_PER_S` / (mPa per mmHg) = 1.0 (`_perfusion_flow_to_um3_per_s`) to
`map_vessels_to_grid` and the Tier 2/3 solvers, and records it on the `.vti` as
`perfusion_flow_to_um3_per_s`. Until open item 31 it took the default, so its perfusion saw flow
1.33 × 10⁵× too high.

---

## §4 — Blood rheology

### 4.1 Apparent viscosity and the Fåhræus–Lindqvist effect

Relative apparent viscosity at a discharge haematocrit of 0.45, as a function of diameter in µm:

$$\begin{aligned}
\text{in vivo:}\quad  \mu_{45} &= 6.0\,e^{-0.085 d} + 3.2 - 2.44\,e^{-0.06\, d^{0.645}} \\[2pt]
\text{in vitro:}\quad \mu_{45} &= 220\,e^{-1.3 d}   + 3.2 - 2.44\,e^{-0.06\, d^{0.645}}
\end{aligned}$$

They differ **only in the first term**, and that term is the whole disagreement. At $d = 8$ µm and
*H* = 0.45 the two give apparent viscosities differing by roughly **3.4×**, and resistance is linear
in viscosity, so this is not a refinement.

`in_vivo` is the default and is fitted to microvessels in living tissue, where the endothelial
surface layer narrows the effective lumen and raises resistance [`pries_resistance_1994`].
`in_vitro` is fitted to blood in glass tubes [`pries_blood_1992`].

**The wall-layer correction follows the law rather than being applied unconditionally.** A glass
tube has no endothelial surface layer, so correcting for one there would be a departure from both
laws rather than a refinement of either.

### 4.2 Phase separation at bifurcations

At a diverging bifurcation, red cells do not split in the same proportion as plasma: they
disproportionately favour the branch with higher flow fraction and larger diameter
[`pries_red_1989`].

With *f_Q1* the fraction of bulk flow entering branch 1, *d₁* and *d₂* the daughter diameters and
*D_F* the diameter of the feeding vessel (see "Feeding diameter" below), all in µm:

$$\begin{aligned}
A &= -13.29 \cdot \frac{(d_1^{2}/d_2^{2}) - 1}{(d_1^{2}/d_2^{2}) + 1}
     \cdot \frac{1 - H_\text{in}}{D_F} \\[4pt]
B &= 1 + 6.98 \cdot \frac{1 - H_\text{in}}{D_F} \\[4pt]
x_0 &= 0.964 \cdot \frac{1 - H_\text{in}}{D_F} \\[6pt]
\operatorname{logit}(f_{E1}) &= A + B \cdot
     \ln\!\left(\frac{f_{Q1} - x_0}{1 - f_{Q1} - x_0}\right),
     \qquad x_0 < f_{Q1} < 1 - x_0
\end{aligned}$$

with *f_E1* = 0 if *f_Q1* ≤ *x₀* and *f_E1* = 1 if *f_Q1* ≥ 1 − *x₀*. Here *f_E1* is the
fraction of the **erythrocyte** flux entering branch 1, and *x₀* is the skimming threshold: a
branch drawing less than *x₀* of the flow gets no red cells. It is not a fixed 5%; it grows as
the feeding vessel narrows (0.053 at *D_F* = 10 µm, 0.13 at 4 µm, both at *H_in* = 0.45). The log
term equals logit[(*f_Q1* − *x₀*)/(1 − 2*x₀*)], the form Pries et al. print. All three
parameters scale with *D_F*, not a daughter's diameter, so the result does not depend on which
daughter is called branch 1.

The constants are as printed in Rasmussen, Secomb & Pries [`rasmussen_modeling_2018`], who
attribute them to Pries, Reglin & Secomb (2003); other sources give the same form as Pries &
Secomb (2005). Which paper first printed them is unconfirmed. The same block is E16–E18 in
`post_radius_assignment_equations.md`; the code is `calculate_phase_separation_hematocrit` in
`rheology.py`.

**If *x₀* ≥ 0.5 the relation is undefined** (1 − 2*x₀* ≤ 0 leaves no flow range for the curve).
That needs *D_F* below about 1.06 µm at *H_in* = 0.45; there both daughters take *H_in*.

**Erythrocyte mass is conserved exactly**: *f_E2* = 1 − *f_E1*. Outlet haematocrits follow as
*H_out* = *H_in* · *f_E* / *f_Q*, then clamped to [0, 0.95].

**Degenerate cases are handled explicitly** rather than by the logit: if either branch takes less
than 10⁻⁶ of the flow, all red cells follow the other.

**Junctions with three or more daughters split one-vs-rest** (§11 row 14; open item 37). Each
daughter *i* takes *f_Ei* from the relation above with *f_Qi*, its own diameter and, as the other
branch, the flow-weighted mean diameter of the remaining daughters; the *f_Ei* are then scaled to
sum to 1, so red-cell flux is conserved. With two daughters this is exactly the Y-split relation,
it does not depend on how the daughters are ordered, and it is continuous as a daughter's flow
falls to zero (it drops below *x₀* and takes no red cells). Until item 37 these junctions mixed
proportionally, and the rule switched on and off whenever a small daughter reversed, which kept
the coupled loop (§4.3) from converging. The extension is ours, not Pries's; the relation was
fitted to Y-splits only.

**Feeding diameter.** *D_F* is the diameter of the one inflowing edge, or the flow-weighted mean
of several. A junction with no inflowing edge has no measured parent, and *D_F* is then the Murray
parent of its daughters, (Σ *d_i*³)^(1/3), counted in `rheology_murray_parent_bifurcations`. The
stand-in was the larger daughter, logged at INFO only, until item 37; it fired 15–29 times per
pass, always at an interior node fed only by rounding-level flow. With stagnant edges left out of
the transport (§4.3) it no longer fires on any of the six networks.

### 4.3 The coupled flow–haematocrit–viscosity solve

Flow depends on resistance, resistance on viscosity, viscosity on haematocrit, and haematocrit on
flow. The loop closes it by Picard iteration:

1. **Initialise** — every edge at systemic haematocrit 0.45, viscosity from Pries–Secomb,
   resistance from Poiseuille.
2. **Solve** the Laplacian system for nodal pressure (§3.4).
3. **Direct** every flowing edge high-pressure to low-pressure, producing a DAG. Stagnant edges
   (|Q| at most `STAGNANT_FLOW_FRACTION` = 10⁻¹⁰ of the largest, the Tier 3 cut) are left out.
4. **Traverse** the DAG from inlets to outlets, applying §4.2 at every diverging junction to
   assign child haematocrits.
5. **Update** haematocrit a fifth of the way towards the new distribution (relaxation 0.2), then
   viscosity and resistance from it.
6. **Repeat** until the relative flow change and the haematocrit residual both fall below
   tolerance, a flow cycle stops the sort, or the iteration cap is reached.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Every edge set to systemic haematocrit | H = 0.45 | The loop needs a starting haematocrit, and systemic is the only value known independently of the network | **On**, once | `rheology.py:235` |
| 2 | µ from Pries–Secomb, R from Hagen–Poiseuille | in vivo law | Gives the first pressure solve a physically scaled resistance rather than the power-law stand-in | **On**, once | `rheology.py:236` |
| 3 | Diameter read `assigned_diameter_um` → `fwhm_diameter_um`, else **raise** before initialising | `default_diameter_um` opt-in, used at every step | Resistance goes as $d^{-4}$, so a substituted calibre produces a fabricated flow field rather than an approximate one | **On** — item 9 closed by `f92a96c`; a zero or negative diameter is refused too | `rheology.py:229` |
| 4 | Solve the Laplacian for nodal pressure (§3.4) | — | Flow cannot be known until pressures are, and pressures change as resistances do | **On**, every iteration | `rheology.py:279` |
| 5 | Per-edge signed flow; direct high → low into a DAG, **leaving out stagnant edges** | \|Q\| ≤ 10⁻¹⁰ × max \|Q\| | Phase separation is defined on a directed tree, so the flow directions have to be resolved first. A stagnant edge's direction is rounding noise; 60–120 per specimen flipped every pass and switched their junctions' rule (open item 37) | **On**, every iteration; 745–895 edges per specimen, all in dead ends. They keep their haematocrit (systemic) | `solve_coupled_flow_and_hematocrit` |
| 6 | Convergence test: max \|ΔQ\| / max \|Q\| **and** the haematocrit residual max \|H_skim − H\| on flowing edges | 10⁻⁶ and 10⁻⁴ (`cb_settings`) | Stops the loop once further passes would not move the answer. Both are scale-free, so the pipeline (mPa) and the H2 drivers (mmHg) apply the same test; the old absolute 10⁻⁴ on flow did not | **On**, from iteration 1; a pass that converges skips steps 7–15 | `solve_coupled_flow_and_hematocrit` |
| 7 | Topological sort of the DAG | — | Haematocrit has to be propagated downstream in order, parent before child | **On**; a cycle breaks the loop with a warning | `rheology.py:320` |
| 8 | Force systemic haematocrit at every inlet | H = 0.45 | The inlets are the one place where haematocrit is prescribed rather than inherited | **On** | `rheology.py:332` |
| 9 | Node haematocrit = flow-weighted mix of inflows | — | A node fed by several vessels carries the flow-weighted mixture, not any one parent's value | **On** | `rheology.py:338` |
| 10 | Degree-2 pass-through: child inherits the mix | — | With one outlet there is nothing to separate, so the child simply inherits | **On** | `rheology.py:346` |
| 11 | Bifurcation: phase separation (§4.2) | — | Red cells do not divide in proportion to plasma at a bifurcation — this is the whole Fåhræus effect the model exists to capture | **On** | `rheology.py:354` |
| 12 | Trifurcation or higher: **one-vs-rest phase separation** (§4.2) | — | The Pries–Secomb relation is defined for a Y-split only. Proportional mixing, used until open item 37, switched on and off as small daughters reversed and kept the loop from converging | **On** | `calculate_multiway_phase_separation_hematocrit` |
| 12a | Under-relax haematocrit | relaxation 0.2 (`cb_settings`) | Plain Picard (1.0) flips a 12 → 8/4 µm Y between two states every pass; 0.5 still wandered on the CB networks, and 0.3 and 0.4 left SHR-C on a periodic cycle | **On** | `solve_coupled_flow_and_hematocrit` |
| 13 | Recompute µ_app from the new haematocrit | in vivo law | Closes the loop: the new haematocrit changes viscosity, which changes resistance, which changes flow | **On** | `rheology.py:390` |
| 14 | Recompute R from Hagen–Poiseuille at the new µ_app | $R = 128\mu_\text{app}L/\pi d^4$ | The same expression as step 2, so initialisation and update agree; replaced the $\mu_\text{app}/\mu_\text{old}$ rescale in `7ea1b36` | **On** | `rheology.py:408` |
| 15 | Wall shear stress from µ_app and abs(Q) | 32µQ/(πd³), mPa → Pa | Shear stress is a per-edge diagnostic that depends on both the new viscosity and the current flow | **On** | `rheology.py:415` |
| 16 | Repeat from step 4 | ≤ 1000 iterations (`cb_settings`) | Repeats until converged or capped, because the system is non-linear and one pass is not a solution. All six converge in 173–457 passes | **On** | `solve_coupled_flow_and_hematocrit` |
| 17 | Record why the loop stopped | `converged`, `flow_cycle` or `max_iterations` | A cycle or cap exit otherwise returns a graph indistinguishable from a converged one | **On**; `rheology_status(G)` returns the reason, pass count, absolute and relative flow change, haematocrit residual, stagnant-edge and Murray-parent counts, and the settings. The pipeline warns on anything but `converged` and saves them in the printed stats and the vessels VTK `field_data`; every H2 driver writes them into its JSON as `"rheology"` and prints a warning | `rheology_status`, `report_unconverged`, `carotid_image_to_model.py`, `cb_h2_*.py` |

**What the numbered summary above does not say.**

**The convergence test is relative.** Until open item 37 step 6 compared `max |Q_new − Q_old|`
against an absolute 10⁻⁴ in the flow units of §3.7, so the pipeline (mPa) and the H2 drivers
(mmHg) ran different tests on the same network. The relative flow test and the haematocrit
residual are unit-free: on all six the two unit scales stop on the same pass and agree on every
edge's haematocrit to 10⁻¹⁰.

**Junctions above degree 3 skim one-vs-rest** (§4.2), an extension of a relation fitted to
Y-splits. After the degree-2 collapse of §2.5 those junctions are the unresolved multi-way
crossings.

**The check runs before the update, so the loop always does at least two passes.** Step 6 is
evaluated after step 5 of iteration 1, against iteration 0's flows. When it passes, steps 7–15 of
that pass are skipped, so the returned resistances, viscosities and wall shear stress are those of
the previous pass.

**Convergence (open item 37).** Undamped, a single 12 → 8/4 µm Y flips its 4 µm branch between
two haematocrit states every pass; relaxation settles it. The CB networks did not converge at all
before item 37: every solve stopped on the 15-pass cap, per-edge haematocrit still moving by up to
0.4 between passes, and relaxation alone (0.5 down to 0.1, 150 passes) only lowered the floor.
Two discrete switches drove it: stagnant dead-end edges whose rounding-level direction flipped,
and junctions switching between skimming and proportional mixing as small daughters reversed.
With both removed (steps 5 and 12) and relaxation 0.2, all six converge: WKY-A 330, WKY-B 457,
WKY-C 404, SHR-A 217, SHR-B 173, SHR-C 184 passes, 30–65 s each. At 0.3 and 0.4 SHR-C settles on
a periodic cycle instead, so 0.2 is the largest step that converges all six.

Limits: 1000 iterations, relative flow tolerance 10⁻⁶, haematocrit tolerance 10⁻⁴, relaxation 0.2,
all in `cb_settings` and shared by the pipeline and every H2 driver.

### 4.4 What the viscosity law does and does not move

The two laws differ by 3.4× in apparent viscosity at capillary calibre. Measured, that difference
**moves no within-specimen ratio**.

Resistance is linear in viscosity, so a change applied to every edge scales the whole network's
resistance and cancels out of any ratio taken within one specimen. This is the same cancellation
§13.2 measures for correlated calibre error, arriving by a different route, and it is a second
independent reason the reportable quantities are ratios.

It does **not** cancel from absolute flow, which is one of the reasons §13.5 exists.

---

## §5 — Blood gas chemistry

All four relations are hard-coded in `perfusion.py`; none is configurable (§10.8).

### 5.1 Oxygen content

Total blood oxygen content in mmol/L, dissolved plus haemoglobin-bound:

$$\begin{aligned}
C_{\mathrm{O_2}} &= \alpha_{\mathrm{O_2}}\,P_{\mathrm{O_2}}
   \;+\; H \cdot c_{\mathrm{Hb,max}} \cdot S(P_{\mathrm{O_2}}) \\[6pt]
S(P_{\mathrm{O_2}}) &= \frac{P_{\mathrm{O_2}}^{\,n}}
   {P_{\mathrm{O_2}}^{\,n} + P_{50}^{\,n}}, \qquad n = 2.7 \\[4pt]
\alpha_{\mathrm{O_2}} &= 1.34 \times 10^{-3}\ \mathrm{mmol\,L^{-1}\,mmHg^{-1}} \\[4pt]
c_{\mathrm{Hb,max}} &= \frac{0.446 \times 20.4}{0.45} \approx 20.22\ \mathrm{mmol/L}
\end{aligned}$$

The Hill exponent $n = 2.7$ is [`hill_possible_1910`]; $c_{\mathrm{Hb,max}}$ is scaled to pure red cell.

Returns zero for $P_{\mathrm{O_2}} \le 0$ rather than extrapolating.

### 5.2 The Bohr effect

P₅₀ is not fixed. It shifts with pH and PCO₂ on the Kelman/Severinghaus empirical form
[`severinghaus_simple_1979`, `kelman_digital_1966`]:

$$\log_{10} P_{50} = \log_{10}(26.0) - 0.4\,(\mathrm{pH} - 7.4)
+ 0.06 \log_{10}\!\left(\frac{P_{\mathrm{CO_2}}}{40}\right)$$

Baseline 26.0 mmHg at pH 7.4 and PCO₂ 40 mmHg. Acidosis or hypercapnia raises P₅₀ — the curve
shifts right and haemoglobin releases oxygen more readily.

### 5.3 Carbon dioxide content and the Haldane effect

McHardy's whole-blood CO₂ dissociation curve, converted from vol% to mmol/L:

$$\begin{aligned}
C_{\mathrm{CO_2}} &= \frac{1}{2.226}\Bigl[\,11.02\,P_{\mathrm{CO_2}}^{\,0.396}
  \;-\; (15 - \mathrm{Hb})\cdot 0.015\,P_{\mathrm{CO_2}}
  \;+\; (95 - 100\,S_{\mathrm{O_2}})\cdot 0.064\cdot\frac{\mathrm{Hb}}{15} \Bigr] \\[4pt]
\mathrm{Hb} &= 15 \cdot \frac{H}{0.45}\ \mathrm{g/dL}
\end{aligned}$$

The bracket is content in vol% (mL CO₂ per 100 mL blood). One mmol of CO₂ is 22.26 mL STPD, hence
the 2.226. The first term is the curve at McHardy's reference state (Hb 15 g/dL, S 95%). The second
is his haemoglobin correction, which is how haematocrit enters: Hb is taken from *H* with the same
15 g/dL at *H* = 0.45 that $c_{\mathrm{Hb,max}}$ assumes (§5.1). The third is his saturation term,
the Haldane effect: deoxygenated blood carries more CO₂. The curve is total content, so dissolved
CO₂ is already in it and is not added separately. Returns zero for P_CO₂ ≤ 0.

**The saturation term is scaled by Hb/15 — a modification of McHardy.** He fitted blood at normal
Hb, where the factor is 1, so at *H* 0.45 the curve is his. The Haldane effect acts through
haemoglobin, but his term carries no Hb, so unscaled it gave plasma a full Haldane effect. That
mattered once Tier 3 passed plasma-skimmed blood through junctions correctly (open item 24): a
daughter at *H* 0.01 losing O₂ gained CO₂ capacity, its PCO₂ fell, and its tissue settled at
39.95 mmHg, below arterial, while producing CO₂. With the scaling, content stays affine in *H*,
which the junction mixing relies on.

At P_CO₂ 40, *H* 0.45 and S 95% this gives 21.3 mmol/L, with a slope of 0.21 mmol/L/mmHg. Full
desaturation adds 2.8 mmol/L, against ≈2.5 mmol/L from the measured human Haldane factor, 0.28 vol%
CO₂ per vol% HbO₂ at Hb 15 g/dL [`loeppky_quantitative_1983`].

**Source.** [`mchardy_relationship_1967`]. The constants were checked against McHardy's equation as
quoted by [`mallat_ratio_2021`], not against the 1967 paper, which is not online:

$$\Delta C_{\mathrm{CO_2}} = 11.02\left[P_v^{\,0.396} - P_a^{\,0.396}\right]
- (15 - \mathrm{Hb})\cdot 0.015\,(P_v - P_a) - (95 - S_{a\mathrm{O_2}})\cdot 0.064$$

with content in vol%, Hb in g/dL and saturation in %. The code comment used to cite Spencer (1979)
[`spencer_computational_1979`]; it was checked by abstract only (whole blood, Hb 15 g/dL, Bohr and
Haldane terms included) and could not be matched.

**Low-haematocrit corner.** At *H* near 0 the Hb term, −0.225·P_CO₂ vol%, makes content peak near
P_CO₂ 135 mmHg and fall slightly above it. Below that the curve rises with P_CO₂ for every *H*
(tested on 0–120 mmHg for *H* 0–0.8), which covers every tissue P_CO₂. Tier 3's inversions widen
their bracket upwards from 150 mmHg and raise if the content is never reached, so a plasma-skimmed
edge asked for more CO₂ than its curve holds fails loudly (open item 24).

> **Changed under open item 19.** The code used to compute
> α_CO₂·P_CO₂ + *H*·[11.02·P_CO₂^0.396 + (0.15 − 0.05·S)·P_CO₂]. That labelled McHardy's vol% as
> mmol/L, multiplied the whole-blood curve by *H*, and added dissolved CO₂ a second time. At *H* 0.45
> the factor *H* almost equals 1/2.226, so the size was close by coincidence: 24.4 mmol/L at
> P_CO₂ 40 (+14%), slope 0.28 (+38%). The Haldane term (0.15 − 0.05·S)·P_CO₂ had no source and
> added 0.90 mmol/L on full desaturation, about ⅓ of the measured effect. Tier 3 only; no H2 number
> used it.

> **One inconsistency, deliberate and small.** The saturation *S_O₂* used in the Haldane term is
> evaluated at a **fixed** P₅₀ = 26 mmHg, not the Bohr-shifted value from §5.2. In acidotic tissue
> CO₂ carriage is therefore underestimated, by at most 1.5% (0.4 mmol/L) over pH 7.2–7.4 and
> P_CO₂ 40–60 mmHg (§11 row 19).

### 5.4 Tissue pH

Henderson–Hasselbalch, with a constant bicarbonate buffer:

$$\mathrm{pH} = 6.1 + \log_{10}\!\left(
\frac{[\mathrm{HCO_3^-}]}{\alpha_{\mathrm{CO_2}}\,P_{\mathrm{CO_2}}}\right),
\qquad [\mathrm{HCO_3^-}] = 24\ \mathrm{mmol/L}$$

No renal compensation, so the pH response to PCO₂ is fixed by this relation alone. Accepts scalars
or arrays, so it can be applied per grid cell.

### 5.5 Species mismatch

Every constant above is human. The tissue is rat. The direction of the resulting bias is not
established (§11 row 17).

---

## §6 — Tissue transport

### 6.1 The perfusion grid

A regular Cartesian grid over the region, `dims = (nz, ny, nx)` at spacing `res = (rz, ry, rx)`,
with linear index **z-fastest**: $\text{idx} = z + y\,n_z + x\,n_z n_y$.

**The grid spans the segmented volume, not the graph's extent.** Padding to the segmentation is
what lets tissue with no vessel in it be represented at all; tissue beyond the segmentation is not
represented (§11 row 25).

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Read every node's `pos`; raise if absent | — | The grid has to cover the vasculature it will exchange oxygen with, and node positions are the only record of that extent | **On** | `perfusion.py:118` |
| 2 | Flip the resolution triple from (x, y, x) order into (z, y, x) | 4 µm isotropic | The caller passes resolution in (x, y, z) while everything internal is (z, y, x); flipping once here avoids a silent axis swap later | **On** | `perfusion.py:126` |
| 3 | Bounding box of the node cloud, padded by half a cell each way | — | Half a cell of padding guarantees no node sits exactly on a face, where its cell index would be ambiguous | **On** | `perfusion.py:129` |
| 4 | **Union** with `bounds_zyx` if supplied — never replacement | segmentation extent | Lets a caller represent tissue beyond the vasculature; a union rather than a replacement so a bound can never shrink the grid below the vessels it must contain | **Only under `--pad-grid`** | `perfusion.py:143` |
| 5 | $\text{dims} = \lceil (\max - \min) / \text{res} \rceil$, cell volume = ∏ res | — | Fixes the cell count and the volume each metabolic sink is multiplied by | **On** | `perfusion.py:149` |

**Step 3 is why the default grid is smaller than the tissue.** The box is fitted to the *graph*, so
a specimen whose vessels stop short of the region edge gets a grid that stops there too, and the
glomus tissue in the gap leaves the analysis silently — the returned fractions stay valid and simply
describe less tissue than was passed in. Step 4 is the opt-in fix, and it is off by default: padding
to the segmentation represents that tissue at the cost of solving cells with no vessel in them
(§11 row 25). Losing more than 1% of the mask warns.

**Step 4 is a union, not an override.** A supplied bound can only ever grow the grid. A caller
cannot accidentally shrink it below the vasculature it has to contain.

> **An indexing trap worth knowing.** The stencil assembly reshapes to `(nx, ny, nz)` because a
> C-order reshape makes the *last* axis fastest, matching the z-fastest linear index. Reading
> `dims` as `(nx, ny, nz)` and reshaping the same way gives an arithmetically correct matrix under
> wrong axis names — two reversals that cancel. If the conductances ever look mislabelled, they
> may be; check the arithmetic, not the names.

### 6.2 Vessel-to-grid mapping

Each edge's centreline voxels are point-sampled. For every cell an edge passes through, the mapping
accumulates that edge's length and lateral surface area ($2\pi r\,\Delta L$) within the cell.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Scan every edge for a usable diameter; **raise** if any lacks one | no default supplied | Diameter feeds surface area and therefore every transvascular flux, so it is not substituted silently | **On** | `perfusion.py:230` |
| 2 | Convert `flow_abs` out of solver units into µm³/s | `POISEUILLE_FLOW_TO_UM3_PER_S` (H2 drivers, pressures in mmHg); 1.0 in the pipeline (pressures in mPa, open item 31) | Edge flow carries mmHg·µm³/cP (at mmHg pressures) while the metabolic sink is mmol/L/s·µm³; coupling them unconverted asked the tissue to consume 2.2 × 10⁴ times the oxygen delivered | **On** | `perfusion.py:248` |
| 3 | Per-voxel length share = `length / (n_voxels − 1)` | — | Spreads an edge's length evenly along its polyline so each sampled point carries a known share | **On** | `perfusion.py:264` |
| 4 | Point-sample each centreline voxel to a linear cell index | — | Maps the 1D network onto the 3D grid — the only place the two representations meet | **On** | `perfusion.py:268` |
| 5 | Accumulate length and lateral surface area $2\pi r\,\Delta L$ per (cell, edge) | — | A cell needs both length (for flow share) and surface area (for exchange) from every edge that crosses it | **On** | `perfusion.py:289` |
| 6 | Voxels outside the grid are dropped, not clipped or wrapped | — | A wrapped index would deposit distal tissue into cell 0 and a clipped one would pile it onto the boundary; both errors are invisible in the output | **On** | `perfusion.py:270` |
| 7 | Total each edge's accumulated length across all cells it entered | — | Normalising against the accumulated length rather than the edge's own `length` is what makes the shares sum to exactly one | **On** | `perfusion.py:307` |
| 8 | `length_fraction` = cell length ÷ that total, so shares sum to 1 | — | Without it an edge crossing five cells injects five times its own oxygen, and the total source grows with grid refinement rather than converging | **On** | `perfusion.py:311` |

**Step 2 is the unit join, and it was once absent.** Edge `flow_abs` carries mmHg·µm³/cP because
$R = 128\,\mu L/(\pi d^{4})$ is evaluated with pressure in mmHg, viscosity in cP and length in µm (§3.7).
The metabolic sink downstream is mmol/L/s × µm³. Coupling them unconverted asked the tissue to
consume **2.2 × 10⁴** times the oxygen the blood delivered, and the steady-state PO₂ was correctly
zero everywhere. The Tier 2 and Tier 3 marches read edge flow themselves, and until open item 20 they
read it raw; they now convert it with the same factor (`flow_to_um3_per_s`) and raise if it disagrees
with the per-cell flows this step wrote. The factor assumes pressures in mmHg; the pipeline's are in
mPa, so its flow is already in µm³/s and it passes 1.0 (§3.7, open item 31).

**Step 8 normalises against the accumulated length, not the edge's `length` attribute.** Point
sampling counts the endpoints of each sub-segment, so the two differ; normalising against the
accumulation is what makes the shares sum to exactly one rather than approximately one.

**Step 6 drops rather than clamps, deliberately.** A wrapped index would deposit distal tissue into
cell 0 and a clipped one would pile it onto the boundary cells. Both errors are invisible in the
output; dropping is visible as missing tissue.

**Flow is then shared by length, not repeated.** Each cell's entry carries a `length_fraction`
normalised against the edge's *accumulated* length, so the shares sum to exactly one:

$$\begin{aligned}
q_\text{total}[\text{cell}] &= \sum_\text{edges} Q_\text{edge}
   \cdot \texttt{length\_fraction} \\[4pt]
s_\text{incoming}[\text{cell}] &= \sum_\text{edges} Q_\text{edge}
   \cdot \texttt{length\_fraction}
   \cdot C_{\mathrm{O_2}}\bigl(P_{\mathrm{O_2,art}},\, H_\text{edge}\bigr)
\end{aligned}$$

Without this an edge crossing five cells injects five times its own oxygen, and the total source
grows with grid refinement rather than converging.

**Point sampling remains an approximation** to line–plane intersection, so *where* a vessel deposits
carries discretisation error even though the total is conserved (§11 row 24).

**Which cells a vessel occupies is a switch, `vessel_mapping`** (open item 30).

- `"centreline"`, the default and the pipeline's mapping: only the cells the centreline crosses,
  whatever the vessel's diameter. Below a vessel diameter a finer grid draws a thinner vessel, a
  line source in the limit, and Tier 1 PO₂ falls about 1.5–1.9 mmHg per halving of the pitch (§6.8).
- `"cross_section"`, used by the H2 drivers: each centreline point is swept over a disc of the
  vessel's radius, perpendicular to the local tangent, and the edge's length and wall area are
  shared among cells by the part of the disc each holds. The disc is sampled at the midpoints of a
  square lattice an eighth of the finer of the grid pitch and the vessel diameter; midpoints,
  because a lattice through the centre puts a ring of samples on the rim, and at a coarse pitch
  those carried 31% of a 9 µm vessel into the neighbouring cells of a 9 µm grid. Where the whole
  disc lies in the centreline's cell the two mappings agree. Steps 6–8 apply unchanged, so the
  shares still sum to one per edge. The tangent is the central difference along the polyline;
  where that vanishes because the centreline turns straight back (one edge in the six specimens,
  a one-voxel spur on WKY-A), the adjoining segment is used and a warning logged, and a point with
  no direction on either side raises.

Every cell a vessel occupies still carries the tissue's full metabolic rate: lumen volume is not
taken out of the sink (§11 row 24).

**It raises on a missing diameter.** Diameter feeds surface area and therefore every transvascular
flux, so it is not substituted silently. Passing `default_diameter_um` is available and is a
deliberate choice to model unmeasured vessels at a stated calibre. The rheology solver now does the
same (§3.2, item 9).

### 6.3 The transport operator — diffusion plus per-cell exchange

**The name "ADR" is loose, and the distinction matters.** The assembled matrix is **pure diffusion**.
There is no advective transport *between* grid cells. Blood delivers oxygen into a cell and carries
it away from the same cell; it does not carry oxygen from one cell to the next.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Convert the diffusion coefficient m²/s → µm²/s, and multiply by α_O₂ | × 10¹², × 1.34 × 10⁻³ mmol/L/mmHg | `sigma_diff` is stored in SI while the whole grid is in microns. α turns a PO₂ difference into an amount of O₂, so every conductance is in µm³/s × mmol/L/mmHg and diffusion is in the same unit as the source, washout and sink (open item 22) | **On** | `build_adr_matrix` |
| 2 | Face conductances D_z, D_y, D_x from σα, face area and normal spacing | — | Fick's law across a cell face: flux scales with the face area and inversely with the distance between cell centres | **On** | `build_adr_matrix` |
| 3 | Per perfused cell, `q_total` = Σ flow × `length_fraction` | — | Blood entering a cell is the only oxygen source the tissue has | **On** | `perfusion.py:388` |
| 4 | Per perfused cell, `s_incoming` = Σ flow-share × C_O₂(PO₂_art, H_edge) | PO₂_art = `po2_arterial_mmHg`, 100 mmHg | Converts that flow into an oxygen delivery rate using the arterial content at the edge's own haematocrit. An edge with no `hematocrit` is refused by `map_vessels_to_grid` rather than given 0.45 | **On** | `perfusion.py:398` |
| 5 | Reshape the index array `(nx, ny, nz)` so the last axis is z | C-order | The linear index is z-fastest, and a C-order reshape makes the last axis fastest — so the shape has to be reversed to match | **On** | `perfusion.py:406` |
| 6 | Off-diagonals for the six face neighbours, both directions | −D per face | A cell exchanges with each face neighbour symmetrically, which is what makes the matrix symmetric and CG-solvable | **On** | `perfusion.py:409` |
| 7 | Diagonal accumulates every conductance the cell participates in | — | Conservation: whatever leaves a cell across its faces must appear on its own diagonal | **On** | `perfusion.py:437` |
| 8 | Add a tiny sink to the diagonal | 10⁻¹² | Pure diffusion under Neumann boundaries has a null space, so a constant offset would otherwise be unconstrained | **On** | `perfusion.py:442` |
| 9 | Assemble COO → CSR | — | COO is cheap to build incrementally; CSR is what the solver needs | **On** | `perfusion.py:447` |

**No advection appears anywhere in this matrix.** Steps 3 and 4 build *vectors*, not matrix
entries: blood delivers oxygen into a cell and carries it away from the same cell, and nothing
transports it from one cell to the next. Step 6 is the entire off-diagonal structure and it is
diffusion only.

**The Neumann boundary is implicit, not imposed.** Step 6 only writes entries for pairs that exist,
so a cell on the domain face simply has fewer neighbours and its diagonal accumulates less. No
boundary row is ever written (§11 row 23).

Diffusive conductance across each cell face, in µm³/s:

$$D_z = \frac{\sigma\,\alpha\, r_y r_x}{r_z}, \qquad
D_y = \frac{\sigma\,\alpha\, r_z r_x}{r_y}, \qquad
D_x = \frac{\sigma\,\alpha\, r_z r_y}{r_x}$$

with $\sigma$ in $\mathrm{\mu m^2/s} = \sigma_\text{config} \times 10^{12}$ and
$\alpha = 1.34 \times 10^{-3}$ mmol/L/mmHg (`ALPHA_O2_MMOL_PER_L_MMHG`, the same α as the blood's
dissolved O₂). **Until open item 22 α was missing here**: diffusion was then in mmHg·µm³/s and was
added to fluxes in mmol/L·µm³/s, which made it ≈1/α ≈ 750× too strong and its diffusion length
≈27× the √(Dα PO₂/M) of §13.6.

assembled as a standard **seven-point stencil** — each cell coupled to its six face neighbours,
with the diagonal accumulating the conductances it participates in. No flux is written at the
domain faces, which is a **Neumann (zero-flux) boundary** by construction (§11 row 23).

**Regularisation.** Pure diffusion under Neumann boundaries has a null space — the rows sum to zero
— so a constant offset is unconstrained. A tiny sink of 10⁻¹² is added to the diagonal at assembly,
and a further 10⁻⁶ before solving.

**Preconditioner.** Jacobi (diagonal), which conjugate gradient requires to be symmetric positive
definite. Non-positive diagonals are detected and reported rather than silently inverted.

### 6.4 Metabolic consumption

$$M(P_{\mathrm{O_2}}) = M_\text{max}\left(1 - e^{-k\,P_{\mathrm{O_2}}}\right),
\qquad k = \texttt{k\_reduce} = 0.1\ \text{per mmol}$$

applied per cell as $M(P_{\mathrm{O_2}}) \cdot V_\text{cell}$.

**This is not Michaelis–Menten**, which is the literature standard. The forms agree in shape — both
saturate — but differ most at low PO₂, which is exactly the regime the hypoxic fraction is read
from (§11 row 21).

### 6.5 Heterogeneous metabolism from the TH segmentation

`M_max` may be a scalar **or a per-cell array**, and the solver applies it elementwise.

The glomus mask is joined to the grid by **volume fraction per cell**, not by sampling the mask at
the cell centre. At 3 µm against 1.866 µm voxels there are about four mask voxels to a cell,
and at the former 10 µm resolution about 154 — so a cell is rarely wholly tissue or wholly stroma,
and a centre sample would discard almost all of the mask and make the answer depend on where cell
centres happened to fall.

The fraction is the **exact overlap volume** (open item 38): each voxel adds to every cell the
volume it shares with it, computed per axis, so the fractions sum to the in-grid mask volume and
none exceeds 1. `mask_fraction_per_cell` raises if either fails. Until item 38 it counted voxel
centres per cell and divided by the mean count, which at 3 µm put a solid glomus cell anywhere
from 0.24 to 1.92, and a clip at 1 dropped about 17% of the TH volume in every specimen
(WKY-A 17.4% of the grid against 20.87% of the voxels).

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Threshold the TH probability field into a glomus mask | `TH_THRESHOLD` | The metabolic contrast is defined against glomus tissue, so the TH channel has to become a binary region first | **On** | `cb_h2_hypoxic_fraction.py:111` |
| 2 | Build the perfusion grid from the graph (§6.1) | 3 µm | The metabolic field has to live on the same cells the transport operator solves on | **On** | `cb_h2_hypoxic_fraction.py:116` |
| 3 | Per-cell **volume fraction** of the mask, by exact voxel–cell overlap, not a centre sample or a centre count | — | At 3 µm against 1.866 µm voxels a cell holds about four mask voxels, so most glomus cells are mixed and a centre sample would discard nearly all of the mask; counting centres aliased and lost ~17% of the volume (open item 38) | **On** | `cb_h2_hypoxic_fraction.py:124` |
| 4 | Warn if more than 1% of the mask volume fell outside the grid; raise if the in-grid volume is not conserved | 1% | The grid is fitted to the graph, so a specimen whose vessels stop short loses tissue silently — worth hearing about. The conservation check is what would have caught open item 38 | **On** | `tissue_regions.py:104` |
| 5 | Mean TH fraction f̄ over all cells | — | The normalisation in the next step needs to know how much of the volume is glomus | **On** | `cb_h2_hypoxic_fraction.py:121` |
| 6 | Stromal rate = `BASE_M_MAX / (1 + f̄(c − 1))` | `BASE_M_MAX` = 0.05 | Holds the volume-weighted mean rate at `BASE_M_MAX` for every contrast, so runs differ in distribution rather than in total consumption | **On** — open item 8 | `cb_h2_hypoxic_fraction.py:122` |
| 7 | Per-cell `M_max` blended between $\text{stroma}\cdot c$ and `stroma` | c ∈ {1, 2, 4} | Puts the glomus rate at $c$ times the stromal one while preserving that mean | **On** | `cb_h2_hypoxic_fraction.py:123` |
| 8 | Solver applies the array elementwise | — | A per-cell rate is the only way heterogeneous metabolism can enter a single linear system | **On** | `perfusion.py:501` |
| 9 | Readouts weighted by TH occupancy, not by cell count | — | A cell that is 40% glomus should contribute 40% of its volume to the glomus readout, not be classified wholly one way | **On** | `cb_h2_hypoxic_fraction.py:130` |

**Step 6 is what makes the contrast sweep interpretable.** Dividing by `1 + f̄(c − 1)` holds the
volume-weighted mean rate at `BASE_M_MAX` for every c, so a run at c = 4 differs from one at c = 1
in *distribution* only. Without it, raising the contrast would also raise total consumption and the
hypoxic fraction would move for two reasons at once.

**Step 9 matters as much as step 7.** A cell that is 40% glomus contributes 40% of its volume to the
glomus readout and 60% to the stromal one. Taking a hard per-cell classification instead would put
whole mixed cells on one side or the other, and at 3 µm against 1.866 µm voxels — about four mask
voxels per cell — most cells that hold any glomus tissue are mixed: 13–36% of cells are mixed
against 4–15% wholly glomus across the six (by exact overlap; 10–34% against 2–7% under the
centre count before open item 38).

Rates are then blended:

$$\begin{aligned}
\text{stroma} &= \frac{\texttt{BASE\_M\_MAX}}{1 + \bar{f}\,(c - 1)} \\[4pt]
M_\text{max}[\text{cell}] &= \operatorname{blend}\!\Bigl(f_\text{TH}[\text{cell}],\;
   \text{tissue rate} = \text{stroma}\cdot c,\;
   \text{stroma rate} = \text{stroma}\Bigr)
\end{aligned}$$

where $c$ is the glomus:stroma contrast and $\bar{f}$ the mean TH fraction. **The volume-weighted mean is
held at `BASE_M_MAX` across contrasts**, so runs at different *c* are comparable rather than simply
scaled versions of each other.

**The contrast is a swept parameter, not a measurement** (§11 row 22). The driver defaults to
`c ∈ {1.0, 2.0, 4.0}`.

### 6.6 The three coupling tiers

| Tier | Solver | Couples | Status |
|---|---|---|---|
| 1 | `solve_perfusion_steady_state` | O₂ only; blood as a well-mixed source and sink per cell | **Active** — this is what §2.3 runs |
| 2 | `solve_coupled_1d3d_perfusion` | O₂ across an endothelial permeability barrier | **Implemented, unreachable** — the dispatch is `if multi_species … elif barrier …`, and multi-species is also on |
| 3 | `solve_multi_species_perfusion` | O₂, CO₂ and pH, linked by the respiratory quotient | Reachable. All three tiers read their step cap and tolerance from config (open item 6) |

> **Tier 3 history.** Its march along each vessel used edge flow in the flow solver's units, not
> µm³/s, so blood gave up its gas in the first cell whatever the flow (open item 20, fixed). With the
> units corrected, the explicit per-cell step overshot at the measured wall permeability; each cell's
> exchange is now implicit, so blood can at most come to equilibrium with its tissue (open item 21).
> The Picard loop was then slow at capillary flow and could stop early (open item 23); it now
> linearises through the blood's actual response, solves each update exactly and is accelerated, and
> stops on the nonlinear residual (§6.7). A fresh WKY-A run converged in 3 iterations (band rule, at
> the double-converted flow of open item 31; the 11 first quoted came from the cache path, open item 28). At the old, unsourced
> permeability (10⁻⁴ cm/s) the tissue was anoxic. Blood crosses each node as pressures, not as content per litre, so a
> plasma-skimmed daughter is not handed more gas than its haematocrit can hold (open item 24).
> Tier 2 had the same unit error (also fixed). Tier 1 left solubility out of its diffusion, and
> Tier 2 out of its diffusion and wall flux, until open item 22; all three tiers now carry α.

**Tier 1's washout is at each cell's own haematocrit** (`cell_discharge_hematocrit`: flow- and
length-share-weighted over the cell's vessels), the same haematocrit its source is built with. O₂
content is affine in H, so this equals the washout summed vessel by vessel. Until open item 29 the
washout used `systemic_hematocrit` (0.45) while the source used each edge's haematocrit. A cell fed
above 0.45 then received more O₂ at arterial PO₂ than it could wash out below hundreds of mmHg
(dissolved O₂ carried the excess); one fed below 0.45 was drained. On WKY-A 27% of perfused cells
are above 0.45 and the surplus was 55× the whole tissue's demand. While diffusion was 750× too
strong (item 22) it averaged these sources and sinks into a near-uniform field; with α in, the
field went hyperoxic (median 186 mmHg).

### 6.7 Numerical stabilisation of the Picard loop

**Tier 1 is solved by Newton's method with a line search** (open item 22). The balance is

$$F(P) = \mathbf{A}P - s_\text{incoming} + q_\text{total}\,C_{\mathrm{O_2}}(P) + M(P)\,V_\text{cell} = 0$$

and each step solves $(\mathbf{A} + \operatorname{diag}(q\,C'(P) + M'(P)\,V))\,\Delta P = -F(P)$ by
Jacobi-preconditioned CG at `rtol` 10⁻¹⁰, then halves the step until $\lVert F \rVert$ falls. It stops
when the largest cell imbalance, divided by its Jacobian diagonal (so in mmHg), is below
`picard_tolerance` (10⁻⁵) of the largest PO₂: the same test as Tier 3 (`_relative_residual`). The cap
and tolerance come from the config (`PerfusionConfig`, and `cb_settings.PerfusionSettings` for H2),
both at 50 and 10⁻⁵; they were hard-coded at those values until open item 6.

**What 10⁻⁵ leaves** (open item 6, WKY-A, §2.3 setup at contrasts 1, 2, 4). Newton takes the
residual from above 10⁻⁴ to ≈5 × 10⁻⁶ in one step, so 10⁻⁴ and 10⁻⁵ stop at the same step (9) and
return the same field, which matches the published §2.3 numbers exactly. Against a 10⁻⁸ solve (10–11
steps) the returned field is up to 0.07–0.10 mmHg off in a single cell, medians (all, TH, stroma) at
most 0.009 mmHg, hypoxic fractions at most 0.021 percentage points (all cells) and 0.014 (TH). A
residual of 10⁻⁵ is a local mmHg correction per cell; diffusion spreads it, so the field error is
larger than the residual suggests. It is well below the grid error (§6.8, §13.7).

Until open item 22 Tier 1 was a Picard loop with a **pseudo-washout**: $q_\text{total}\,\gamma$ on the
diagonal and $q_\text{total}\,\gamma\,P$ on the right, γ = 0.5, stopping on the relative change between
iterates. γ stands in for $C'(P)$, which is near α at both flat ends of the O₂ curve and ≈0.2 at its
steepest. While diffusion was 750× too strong it tied each cell to its neighbours and hid the gap;
with α in the diffusion the loop hit its 50-iteration cap on WKY-A. Plain Newton fails too: at a
slope of α a step jumps by 10³ mmHg, and it swung between ≈18 and ≈1 100 mmHg. With the line
search WKY-A converges in 10–21 steps (≈10–30 s at 4 µm).

**Tier 3 no longer uses a pseudo-washout** (open item 23). It put the full wall conductance
P·A·α on the diagonal (γ = 1). At low flow the blood follows the tissue, so the true slope of the
wall flux in tissue pressure is only k·C′/(C′ + k/*q*), k = P·A·α, and each iteration moved the
tissue that fraction of the way: ≈2 700 iterations on a two-cell chain at 10² µm³/s, and no
progress at all in a plasma-skimmed vessel, whose C′ is tiny. Tier 3 now puts that slope
(`_blood_response_conductance`, summed over a cell's vessels) and the metabolic slope
M_max·k_reduce·e^(−k_reduce·P)·V on the diagonal, and the same terms times the current field on
the right (E43, E51–E53): a Newton step in each cell's own pressure. Each update is solved
exactly by sparse LU (symmetric ordering, refactorised each iteration, ≈1 s per species on
WKY-A's 30 k cells), not by the warm-started CG at `rtol` 1e-5, which returned the last field once
a step was below it. Anderson acceleration (depth 5, `_AndersonMixer`) sits on top, and its
history is dropped whenever the residual rises; unguarded, it diverged on the stiff two-cell chain.

**Tier 3 stops on the residual, not the step** (E54). Each cell's imbalance of the tissue balance
is divided by the same diagonal (diffusion plus blood response plus metabolic slope), giving the
pressure correction it needs; the loop stops when the largest, relative to the largest pressure, is
below `picard_tolerance` for both gases. The step between iterates says little when the loop is
slow: at 10⁻⁴ it stopped the chain 1.7 mmHg short. The check is made on the field that is returned.
Iterations to 10⁻⁴, the default until open item 6: 9–19 on the test networks (Y network,
plasma-skimmed branch, merging inlets, chain), 3 on a fresh WKY-A run (the 11 first quoted came from
the cache path's invented boundaries, open item 28); the returned fields sit within 0.023 mmHg
of a 10⁻¹² solve. The default is now 10⁻⁵, the value Tier 1 uses (open item 6). On a fresh WKY-A
pipeline run 10⁻⁵ takes 4 iterations (residual 2.3 × 10⁻⁶) against 3 at 10⁻⁴, and moves tissue PO₂
by up to 0.29 mmHg (mean 0.03; minimum 78.15 → 78.44, mean 96.79 → 96.82), PCO₂ by up to 0.003 mmHg.
So 10⁻⁴ left more than the 0.02 mmHg the test networks suggested. Those WKY-A runs were at
`M_max` 0.005; at 0.05, the default since open item 8, WKY-A takes 7 iterations (residual
6.0 × 10⁻⁷) and the Y network at 10³ µm³/s 30 (was 14), still within 2 × 10⁻⁴ mmHg of a
10⁻¹² solve. Every WKY-A pipeline figure above ran with edge flow ≈1.33 × 10⁵× too high (open
item 31); the test-network figures do not. At the corrected flow (median edge flow 8.0 × 10⁴ µm³/s)
WKY-A takes 22 iterations (residual 6.9 × 10⁻⁶). All of these WKY-A runs used the band-rule
boundaries. With the pipeline on the face rule (open item 2), median edge flow falls to
9.2 × 10³ µm³/s and WKY-A needs 55 iterations; the old cap of 50 stopped it at a residual of
5.9 × 10⁻⁵, within 0.02 mmHg per cell of a 10⁻⁷ solve. On the six specimens it takes 40–80
iterations (WKY-C 80), so the cap is now 200 (open item 32). On the 2026-09-28 placed-box networks
it converges on all six, e.g. WKY-A in 50 iterations and SHR-A in 53 (SHR-A only after open item 36).
The `.vti` records whether it converged, the iterations and the final residual.

**PO₂ is clamped to ≥ 0** at each iteration. Negative values are non-physical and drive Picard
oscillation.

The loop warns rather than raising if it hits its iteration cap without reaching tolerance.

**The order of operations — the Tier 1 steady-state Newton solve.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Initial guess: PO₂ = 0 mmHg everywhere | — | Starting from zero is the safe end; the line search keeps each step from overshooting | **On** | `solve_perfusion_steady_state` |
| 2 | Add a diagonal regulariser to A | 10⁻⁶ | Pure diffusion under Neumann boundaries is singular. Still ≪ the face conductances (≈8 at 4 µm) | **On**, once | `solve_perfusion_steady_state` |
| 3 | Metabolic sink $M_\text{max}\bigl(1 - e^{-k P_{\mathrm{O_2}}}\bigr)$ and its slope | k = 0.1 | The sink saturates with PO₂, so it is re-evaluated from the current iterate | **On** | `balance` |
| 4 | Washout $q_\text{total} \cdot C_{\mathrm{O_2}}(P_{\mathrm{O_2}}, H_\text{cell})$ and its slope $q\,C'$ per perfused cell | $H_\text{cell}$ = `cell_hematocrit`, the cell's flow-weighted haematocrit (required; raises if missing or outside [0, 1]) | Blood leaves each cell at the local tissue PO₂, on the same content curve its source used (open item 29). The slope is a central difference of `calculate_blood_oxygen_content` (`_content_and_slope`) | **On** — §11 row 27 | `balance` |
| 5 | Residual $F$ and the scaled stop test | `picard_tolerance` 10⁻⁵ | Imbalance ÷ Jacobian diagonal is the mmHg correction each cell needs; relative to max PO₂. Leaves ≤ 0.10 mmHg against a 10⁻⁸ solve on WKY-A | **On** | `_relative_residual` |
| 6 | Newton direction: CG on $\mathbf{A} + \operatorname{diag}(q C' + M' V)$, Jacobi preconditioner | `rtol` 10⁻¹⁰, `maxiter` 20 000 | Exact enough that the direction, not the inner solve, limits convergence. A shortfall warns | **On** | `solve_perfusion_steady_state` |
| 7 | Backtracking line search, clamp PO₂ ≥ 0 | halve until ‖F‖ falls (Armijo 10⁻⁴), down to 10⁻⁴ | Newton on the sigmoid overshoots where C′ ≈ α | **On** | `solve_perfusion_steady_state` |
| 8 | Hitting `max_iter` warns and returns the last iterate; `return_info` reports it | `picard_max_iterations` 200 | A solve that ran out of iterations is a different object from a converged one | **On** | `solve_perfusion_steady_state` |

**Step 4 is a Python loop over the perfused cells**, about 3% of the grid on WKY-A (13 k of 439 k),
so it no longer dominates the runtime; the CG solves do.

### 6.8 Grid resolution

**3 µm, with vessels mapped over their cross-section** (`cb_settings.GRID_UM`; open item 30). It
was 4 µm until item 30, when that grid turned out not to be converged under the centreline mapping.

**Re-measured 2026-10-01, all six, in the full re-run** on the placed boxes, after open items 37
(converged rheology), 38 (exact TH overlap) and 40 (centred TH lookup)
(`cb_h2_hypoxic_fraction_xsec_grid{10,6,4,3,2}.json`, contrast 1, unpadded; logs in
`rerun_2026-10-01_logs/`). The 3 µm row equals the main `cb_h2_hypoxic_fraction.json` run in every
field. Median PO₂ over all cells / PO₂ in TH (mmHg) / TH share of the grid (%):

| Specimen | 10 µm | 6 µm | 4 µm | **3 µm** | 2 µm | TH % of ROI voxels |
|---|---|---|---|---|---|---|
| WKY-A | 95.68 / 95.61 / 18.6 | 95.43 / 95.25 / 19.4 | 95.32 / 95.09 / 19.7 | 95.50 / 95.18 / 20.5 | 95.40 / 95.09 / 20.5 | 20.87 |
| WKY-B | 92.67 / 92.38 / 29.7 | 92.40 / 91.95 / 30.9 | 92.28 / 91.75 / 31.5 | 92.52 / 91.93 / 32.8 | 92.41 / 91.81 / 32.8 | 33.28 |
| WKY-C | 92.44 / 93.23 / 21.0 | 92.35 / 92.74 / 21.8 | 92.29 / 92.50 / 22.2 | 92.69 / 92.62 / 23.1 | 92.58 / 92.49 / 23.1 | 23.51 |
| SHR-A | 92.53 / 91.18 / 20.0 | 92.41 / 90.76 / 20.8 | 92.34 / 90.59 / 21.2 | 92.62 / 90.77 / 22.0 | 92.52 / 90.66 / 22.0 | 22.38 |
| SHR-B | 91.20 / 91.54 / 13.3 | 90.99 / 91.15 / 13.8 | 90.87 / 90.96 / 14.1 | 91.16 / 91.09 / 14.7 | 91.04 / 91.00 / 14.7 | 14.92 |
| SHR-C | 89.77 / 88.81 / 8.7 | 89.59 / 88.50 / 9.0 | 89.52 / 88.38 / 9.2 | 90.01 / 88.54 / 9.6 | 89.86 / 88.48 / 9.6 | 9.70 |

Against the 2026-09-30 sweep (after item 38, before item 40) the all-cell medians and TH shares
are unchanged, and PO₂ in TH moves by at most 0.09 mmHg (the half-voxel TH lookup of item 40).
Largest step per refinement, over both measures and all six: 10 → 6 µm 0.49 (WKY-C in TH; was
0.48), 6 → 4 µm 0.235, 4 → 3 µm 0.495 (SHR-C median), 3 → 2 µm 0.155 mmHg. Every step passes, so
4 µm qualifies too, by 0.005 mmHg; **3 µm is kept** because 4 µm sits on the line and the 3 → 2 µm
step is a third of it. The TH share now rises towards the voxel truth as h falls, the remaining gap being the grid's
overhang past the ROI, where the old centre count had it scattered (17.4–18.7% on WKY-A) and
lowest at 3 µm. Open item 38 moved the all-cell median by at most 0.01 mmHg and PO₂ in TH by at most 0.04 at any grid (the stromal median by up to 0.77, because glomus volume the clip had dropped is no longer counted as stroma), so the 4 → 3 µm step
(unchanged at +0.18 to +0.49 median) is **not** an artefact of the TH aliasing: 3 µm still has the
highest all-cell median of 4, 3 and 2 µm in every specimen (open item 39). TH hypoxia is 0.00% at
every grid.

**The 3 µm bump is the grid extent, not the solve** (open item 39, closed 2026-10-01). The default
grid is the node bounding box plus half a cell on each side, rounded up to whole cells, so its side
depends on h: 310 / 306 / 304 / 300 / 300 µm at 10 / 6 / 4 / 3 / 2 µm in all six specimens,
against a ROI of 298.2–298.7 µm. The overhang cells hold no vessel and no TH, and they count in
the all-cell median, so a coarser grid carries more vessel-free cells and reads lower. 3 and 2 µm
share a 300 µm box; 4 µm does not. (`bounds_zyx` cannot pin this: it is a union with the node box
± h/2, which is always wider than the ROI.) A scratch re-run with the grid pinned to the ROI corner
and a 300 µm side at every h (`cb_h2_hypoxic_fraction_xsec_pinned.json`, logs in
`rerun_2026-10-01_itemM_logs/`) gives:

| Specimen | Default 4 / 3 / 2 µm, median | Pinned 4 / 3 / 2 µm, median | Pinned 4 / 3 / 2 µm, PO₂ in TH |
|---|---|---|---|
| WKY-A | 95.32 / 95.50 / 95.40 | 95.58 / 95.49 / 95.40 | 95.26 / 95.17 / 95.09 |
| WKY-B | 92.28 / 92.52 / 92.41 | 92.62 / 92.52 / 92.41 | 92.04 / 91.93 / 91.81 |
| WKY-C | 92.29 / 92.69 / 92.58 | 92.84 / 92.71 / 92.58 | 92.73 / 92.61 / 92.49 |
| SHR-A | 92.34 / 92.62 / 92.52 | 92.71 / 92.62 / 92.52 | 90.90 / 90.78 / 90.66 |
| SHR-B | 90.87 / 91.16 / 91.04 | 91.30 / 91.18 / 91.04 | 91.26 / 91.14 / 91.00 |
| SHR-C | 89.52 / 90.01 / 89.86 | 90.15 / 90.02 / 89.86 | 88.75 / 88.63 / 88.49 |

On the pinned box both measures fall steadily with h in every specimen: 4 → 3 µm −0.09 to −0.14,
3 → 2 µm −0.09 to −0.16 mmHg. Pinning moves the 3 and 2 µm values by at most 0.02 (their box was
already 300 µm) and the 4 µm median by +0.26 to +0.64, and the TH share is the same at every h
(WKY-A 20.54%; the voxel truth 20.87% less the 1.4 µm overhang). So the bump was a dip at 4 µm,
and the 4 → 3 µm step of +0.18 to +0.49 above was mostly extent. The origin alone matters less:
at 3 µm on a fixed 306 µm box, moving it by 1 and 2 µm shifts the median by at most 0.12 and PO₂
in TH by at most 0.28 (SHR-C). The steps do not shrink from 4 → 3 to 3 → 2 µm, so the field is
still converging at about 0.1 mmHg per µm of h, well inside the 0.5 mmHg rule. **3 µm stays**;
nothing in the H2 drivers changed, and no H2 number moves. The coarser rows of the table above
(10 and 6 µm) carry the same extent effect and should not be read as convergence steps.

Cross-section mapping, measured after open items 22 and 29 (contrast 1, unpadded;
`cb_h2_hypoxic_fraction_xsec_sweep.json`). These sweeps ran on the centred-box networks at 0.90 and
were superseded by the 2026-09-30 sweep above; the table is kept for the record, and its PO₂
values are not today's:

| Grid | WKY-C median PO₂ | WKY-C PO₂ in TH | SHR-C median PO₂ | SHR-C PO₂ in TH | SHR-C TH < 10 mmHg | Cells with no vessel (WKY-C) |
|---|---|---|---|---|---|---|
| 10 µm | 92.99 | 88.23 | 84.41 | 74.41 | 3.10% | 50.0% |
| 6 µm | 92.79 | 87.88 | 85.51 | 74.73 | 3.15% | 65.8% |
| 4 µm | 92.69 | 87.78 | 85.39 | 74.95 | 3.06% | 74.4% |
| 3 µm | 92.87 | 88.32 | 86.17 | 75.06 | 3.09% | 77.5% |
| 2 µm | 92.78 | 88.11 | 86.08 | 75.09 | 3.26% | 82.1% |

("PO₂ in TH" is the TH-weighted mean, stored as `po2_median_th`.) The criterion was that median PO₂
and PO₂ in TH move less than 0.5 mmHg per step on both specimens. From 3 to 2 µm every measure
passes (largest −0.21, WKY-C in TH). From 4 to 3 µm two fail (SHR-C median +0.78, WKY-C in TH +0.54),
so 3 µm is the coarsest grid that qualifies. On all six (`..._xsec_grid4.json` against the main
run) the 4 → 3 µm step is +0.09 to +0.79 mmHg in median PO₂ and −0.34 to +0.54 in TH. Across the
sweep the steps change sign with no trend, probably because the grid origin moves with h (later
traced to the grid extent, open item 39); the drift of the centreline mapping is gone. A 2 µm solve costs about
4 minutes per specimen, a 3 µm one about 1.

Centreline mapping, as it was: **historical, on the centred boxes** (`feb3332`; the 10/6/3/2 µm
files are archived as `cb_h2_hypoxic_fraction_grid{10,6,3,2}_2026-09-29_pre_rerun.json`, and the
4 µm row is that commit's main run). The unsuffixed `_grid*.json` files now hold the placed-box
re-run, given below this table.

| Grid | WKY-C median PO₂ | WKY-C PO₂ in TH | SHR-C median PO₂ | SHR-C PO₂ in TH | SHR-C TH < 10 mmHg | Cells with no vessel (WKY-C) |
|---|---|---|---|---|---|---|
| 10 µm | 91.38 | 86.23 | 80.28 | 71.49 | 3.62% | 74.3% |
| 6 µm | 90.46 | 84.95 | 79.95 | 70.78 | 3.86% | 89.6% |
| 4 µm | 89.52 | 83.84 | 78.35 | 70.01 | 3.95% | 95.3% |
| 3 µm | 89.19 | 83.68 | 78.12 | 69.28 | 4.17% | 97.4% |
| 2 µm | 87.90 | 82.10 | 75.85 | 67.84 | 4.63% | 99.0% |

The 3 → 2 µm step was the largest of the four (−1.28 and −2.27 mmHg). A power law P∞ + C hᵖ had no
limit to find; a straight line in ln h fit to 0.15 mmHg rms on WKY-C, at about 1.5 (WKY-C) and 1.9
(SHR-C) mmHg lost per halving of h.

On the placed boxes (`cb_h2_hypoxic_fraction_grid{10,6,3,2}.json`, all six, contrast 1, re-run
2026-10-01; no 4 µm centreline run exists on these boxes) the drift is the same:

| Grid | WKY-C median PO₂ | WKY-C PO₂ in TH | SHR-C median PO₂ | SHR-C PO₂ in TH | SHR-C TH < 10 mmHg | Cells with no vessel (WKY-C) |
|---|---|---|---|---|---|---|
| 10 µm | 90.58 | 91.80 | 87.44 | 87.00 | 0% | 76.2% |
| 6 µm | 89.66 | 90.51 | 86.27 | 85.80 | 0% | 90.2% |
| 3 µm | 88.38 | 89.03 | 84.83 | 84.07 | 0% | 97.6% |
| 2 µm | 86.86 | 87.72 | 83.04 | 82.60 | 0% | 99.1% |

The steps grow as h falls in every specimen, so there is still no limit. A straight line in ln h
gives 1.2–1.8 mmHg of median PO₂ lost per halving across the six (WKY-C 1.55, SHR-C 1.81; rms
≤ 0.27 mmHg) and 1.4–1.9 in TH.

**Why the centreline mapping drifts.** It puts a vessel only in the cells its centreline crosses,
whatever its diameter (§6.2, §11 row 24). Refining the grid shrinks the vessel with the cell, so in
the limit each vessel is a line with a fixed exchange conductance per unit length. Around a line
the field goes as ln r: the cell holding it sits ever closer to the blood, delivers less, and the
tissue falls by about the same amount per log step in h. `test_perfusion_tier1_grid_refinement.py`
shows this on one straight vessel of diameter 9 µm: mapped by centreline, the mean PO₂ is 17.0,
11.5, 8.6, 6.8 mmHg at h = 9, 3, 1, ⅓ µm; mapped by cross-section, 17.0, 23.4, 23.8, 23.2. Median
edge calibre (`assigned_diameter_um`) is 7.5–8.4 µm across the six (p10 3.7–5.3 µm), so at 4 µm the
median vessel was already two cells wide and drawn as one.

**What the switch moved.** At 4 µm, cross-section against centreline: WKY-C median 92.69 against
89.52, SHR-C 85.39 against 78.35; PO₂ in TH 3.6–7.4 mmHg higher across the six. The centreline
mapping under-delivered, by different amounts per specimen, so the error did not cancel between the
cohorts.

Before items 22 and 29, median PO₂ moved 27.34 → 27.92 → 28.21 at 10, 6 and 4 µm — increments
halving and extrapolating to about 28.5. With diffusion 750× too strong, the ln h term was too small
to see.

What refinement cannot fix is §13.6: the gradient the model is trying to resolve is physically
short, because the tissue is not diffusion-limited.

---

## §7 — Derived physiological quantities

The layer between a solved field and a number in a whitepaper. **Check §13.10 before quoting
anything from here** — several of these quantities are computed but not reportable.

---

### 7.1 Network morphometry

Computed from the graph alone, with no physics.

| Quantity | Definition | Reportable? |
|---|---|---|
| Total centreline length | Sum of edge lengths, µm | Yes |
| $\beta_1$ (fundamental loops) | $E - V + C$ | **Yes** — the H1 §1.1 readout |
| Tortuosity index | Path length / straight-line distance, per edge | Yes |
| Curvature | Per edge, from the smoothed centreline | Yes |
| Branching angle | Angle between every neighbour pair at nodes of degree ≥ 3 | Diagnostic |
| Branching points | Count of nodes with degree > 2 | Yes |
| Tree asymmetry | Partition asymmetry index | Diagnostic |
| Fractal dimension | Box counting | Diagnostic |
| Path efficiency | Shortest-path vs Euclidean over node pairs | Diagnostic |
| Betweenness, communities | Graph-theoretic centrality and modularity | Diagnostic |

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Collapse the MultiGraph to a simple graph for the topological measures | — | Most graph-theoretic measures are undefined or ambiguous on a MultiGraph, so a simple view is taken for those | **On** | `stats.py:684` |
| 2 | Basic counts: nodes, edges, total and mean edge length, mean degree | uses `weight` | The counts every other quantity is normalised against | **On** | `stats.py:19` |
| 3 | Tortuosity per edge and its summary | — | H1 §1.4 is a tortuosity claim, and it is computed once here so the summary and the per-edge CSV cannot disagree | **On** | `stats.py:150` |
| 4 | Branching statistics: junction count, branching angles | degree ≥ 3 | Junction density is an H1 readout, and branching angle is the diagnostic that shows whether junctions were resolved sensibly | **On** | `stats.py:178` |
| 5 | Tree asymmetry | — | Distinguishes a balanced bed from a network dominated by one trunk | **On** | `stats.py:226` |
| 6 | Fractal dimension by box counting | — | A scale-invariant descriptor of how densely the network fills the volume | **On** | `stats.py:254` |
| 7 | Vessel density, both definitions | voxel size passed (required with the image dimensions) | Both definitions are reported because neither is the parenchymal density H1 §1.3 asks for, and saying so requires showing both | **On** | `stats.py:362` |
| 8 | Path efficiency | sampled — `max_pairs` capped in `fast` | Exhaustive all-pairs shortest paths are quadratic in node count, so `fast` samples instead | **On** | `stats.py:281` |
| 9 | Communities and betweenness | summaries only in `fast`; communities only up to `max_nodes_exact` = 1500 nodes | Diagnostic centrality and modularity; reduced to summaries because the full objects are large and unused. Every CB network is larger than 1500 nodes, so modularity is never computed on the CB path: the summary reports connected-component counts instead, under component keys (`Connected Component Count`, always 1 here), with `Community Method: connected_components_fallback` | **On** (components only) | `stats.py:449`, `stats.py:476` |
| 10 | Per-edge morphometry table → CSV | 16 fixed columns | With n = 3 per group the per-edge table is the only place with enough data to describe a distribution at all | **On** | `stats.py:59`, `stats.py:138` |
| 11 | Benchmarking suite, including `graph_fundamental_loops` | `run_benchmarking = False` | Off — and with it the only function in the library that computes $E - V + C$, which is why $\beta_1$ is derived post hoc | **Off** | `benchmarking.py:180` |

**`statistics_mode` is `"fast"` and the caller does not override it.** In `fast`, path efficiency is
sampled rather than exhaustive and communities and betweenness are reduced to summaries. The `full`
mode exists and is never selected on the CB path.

> ⚠ **β₁ is not produced by this step at all.** The pipeline never computes it: `run_benchmarking`
> is `False`, so `graph_fundamental_loops` — the one place in the library that evaluates $E - V + C$ —
> does not run. The β₁ figures quoted throughout this document are computed **post hoc** by
> `cb_h1_figures.py:69` from the per-edge CSV, as $\lvert\text{rows}\rvert - \lvert\text{unique nodes}\rvert + 1$, with **C = 1
> asserted** rather than measured. That assertion is sound only because
> `keep_largest_component_only` is `True` (§2.5 step 8), so the graph is a single component by
> construction. If that setting were ever turned off, every β₁ in this document would be wrong by
> the number of components, silently.

> **"Vessel Density in Whole Image" now applies the voxel size (re-run package G).** Until then
> `compute_comprehensive_vessel_statistics` was called without `voxel_size`, so the image volume
> was `prod(image_dimensions)` in voxels and the density was too high by `prod(1.8639, 1.866, 1.866)`
> = 6.49×. The pipeline now passes the spacing the graph was built with, and the function raises if
> `image_dimensions` comes without a voxel size. On WKY-A the image volume is 2.66 × 10⁷ µm³ (was
> 4.10 × 10⁶) and the density 0.00365 µm/µm³ (was 0.0237). These lines are only printed to
> `pipeline.log`; no H1 or H2 number used them, and the batch logs carry the old values until the
> next batch run.

**Tortuosity is derived from the per-edge table rather than recomputed**, so the summary and the
per-edge CSV cannot disagree about what an edge's tortuosity is.

> ⚠ **"Vessel density" means two different things, and neither is parenchymal density.**
> `compute_vessel_density` reports *Density in Node Bounding Box* as total length divided by the
> **bounding box of the node positions**, and *Density in Whole Image* as total length divided by
> the full image volume. The first is a graph-extent density; the second includes everything that
> is not tissue. The first pair was called "Vessel-Occupied Volume" and "Density in Tissue" until
> re-run package G, which read as a tissue volume it is not.
> The parenchymal quantity H1 §1.3 asks for is the one in §7.2, not either of these.

> **At a glance** — graph-only metrics, tortuosity shared with the per-edge table ·
> `stats.py:147`, `stats.py:349`, `stats.py:213` · `tests/test_statistics.py`,
> `tests/test_synthetic_network_statistics.py`

---

### 7.2 Two-channel morphometry

Joins the vessel channel to the TH channel. **Sound only because they are two channels of one
acquisition** — identical grid, co-registered by construction, no registration step (§11 row 28).

**Vessel set: the batch network** (open item 40). Both methods read the vessel side from the
`cb_h1_batch` run (`cb_h1_th_metrics._load_batch`): the graph for length, the cached skeleton for
distance, the cached mask for vessel volume. All three come from the hysteresis band (§1), so §1.3
and §1.5 describe the same vessels as §1.1–1.4. `check_output_roi` refuses a batch output cut at
another box. Until item 40 the driver cut the probability map plainly at the frozen threshold and
skeletonised that itself, a vessel set with 1.6× the skeleton voxels of the network.

**Centreline length within the glomus.** The network's edge polylines (the same edges and lengths
as `per_edge_morphometry.csv`), classified against the TH mask by the sampling of
`edge_tissue_fraction` (§7.3):

$$L_\text{TH} = \sum_\text{edges} \sum_\text{segments} \ell_\text{seg} \cdot
\frac{\#\{\text{sub-step midpoints in TH}\}}{n_\text{seg}},
\qquad n_\text{seg} = \lceil \ell_\text{seg} / (v_\text{min}/2) \rceil$$

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Load the batch graph, skeleton and mask; refuse another box or shape | ROI 160³ | One vessel set for all of H1; a plain-cut mask here measured different vessels | **On** | `cb_h1_th_metrics.py:72` |
| 2 | Split each polyline segment into sub-steps of at most half the finest voxel | 0.93 µm | A mask crossing cannot be stepped over | **On** | `tissue_regions.py:207` |
| 3 | Look up each sub-step midpoint in the voxel **centred** on it (first corner at −v/2) | — | The graph puts voxel k's centre at k·v; a lookup with the corner at 0 was half a voxel off on every axis | **On** | `tissue_regions.py:210` |
| 4 | Length inside = Σ segment length × share of midpoints inside | — | Length-weighted, so unevenly spaced polyline points do not outvote a long run | **On** | `tissue_regions.py:238` |
| 5 | Total = Σ polyline length; raise on an edge with no geometry | — | A missing edge would shorten the network with no sign in the result | **On** | `tissue_regions.py:280` |
| 6 | Raise if the centreline is empty | — | A distance transform against an empty mask returns infinity everywhere, which would propagate as a very large distance rather than as an error | **On**, never returns ∞ | `th_morphometry.py:53` |
| 7 | EDT from every non-centreline voxel of the **pipeline skeleton**, `sampling` = voxel size | µm directly | `sampling` puts the answer in microns directly and carries the 1.0011 axial-to-lateral ratio, so nothing downstream converts again | **On** | `th_morphometry.py:61` |
| 8 | Read the distance at every TH-positive voxel | — | H1 §1.5 asks how far glomus tissue sits from its supply, which is a question about tissue voxels rather than about vessels | **On** | `th_morphometry.py:62` |

**Why not a voxel-pair sum.** Until item 40 the length summed every 26-adjacent skeleton voxel
pair at its physical step length. That fixes the √3 bias of counting voxels, but where three
skeleton voxels touch at a corner it counts all three links though the path uses two: 9% above
skan's path length on the plain-cut skeleton and 28% above on the pipeline skeleton. The polyline
length has neither bias, and it is the length §1.1–1.4 already use.

**Resolution at the TH boundary.** Each crossing is placed to within one sub-step (0.93 µm), so an
edge's inside length is exact to about ±1 µm per crossing.

**Tissue-to-vessel distance.** Euclidean distance transform from every TH-positive voxel to the
nearest **centreline** voxel, with `sampling` set to the voxel size so the result is in µm directly.

> **To the centreline, not the vessel surface.** The two differ by the local radius. On a 3 µm
> capillary the surface is 1.5 µm closer everywhere, and that offset would be absorbed into any
> group difference rather than appearing as one. H1 §1.5 asks for the centreline distance.

**It raises on an empty centreline** rather than returning infinity, which would propagate as a very
large distance instead of as an error.

> **At a glance** — network polyline length, pipeline-skeleton distance, voxel-centred lookup
> (open item 40) · §1.5 median TVD 7.9–9.7 µm, §1.3 density 2,000–3,310 mm·mm⁻³ (TH > 0.5, re-run 2026-09-30) ·
> `th_morphometry.py:96`, `tissue_regions.py:280` · `tests/test_th_morphometry.py`,
> `tests/test_tissue_regions.py`

---

### 7.3 Functional shunting (H2 §2.1)

**The question.** Does steady-state flow bypass the capillaries that penetrate the TH-positive
clusters, running instead through thoroughfare channels?

**Edge classification.** An edge counts as *penetrating* when at least **50%** of its centreline
length lies inside the TH mask, sampled along the **whole polyline**.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Load the graph and attach EDT diameters from the per-edge CSV | — | Calibre lives in the per-edge CSV rather than the graph, and every quantity here is quadratic or quartic in it | **On** | `cb_h2_glomus_perfusion.py:93` |
| 2 | Select boundaries by the **face** rule | axis 1, 1 voxel | The shunt index compares flow against edge count, so which vessels are inlets sets the entire flow field | **On** | `cb_h2_glomus_perfusion.py:94` |
| 3 | Coupled flow / haematocrit / viscosity solve (§4.3) | 60 / 20 mmHg | There is no shunting to measure until flow has been solved on the network | **On** | `cb_h2_glomus_perfusion.py:97` |
| 4 | Threshold the TH probability field | `TH_THRESHOLD` = 0.5 | The question is about vessels relative to the glomus clusters, so the clusters have to be a region first | **On** | `cb_h2_glomus_perfusion.py:88` |
| 5 | Resample each edge polyline at half the finest voxel | 0.93 µm | The stored polylines are unevenly spaced after B-spline smoothing, so counting points would let a densely sampled stretch outvote a long one | **On** | `tissue_regions.py:208` |
| 6 | Sample the mask at sub-step **midpoints**, length-weighted | — | Each sub-step carries the same length, so averaging over midpoints is exactly a length-weighted average along the segment | **On** | `tissue_regions.py:240` |
| 7 | Points outside the mask array count as outside, never clipped | — | Clipping would pile distal centreline onto the mask border and count it as inside | **On** | `tissue_regions.py:212` |
| 8 | Per-edge fraction = inside length ÷ total length | — | Turns a geometric relationship into one number per edge that the classification can threshold | **On** | `tissue_regions.py:244` |
| 9 | Classify: fraction ≥ 0.5 penetrating, below 0.5 bypassing | `PENETRATION` = 0.5 | A capillary penetrating a cluster usually starts and ends in stroma, so an endpoint test would classify exactly the vessels the question is about as extra-glomus | **On** | `cb_h2_glomus_perfusion.py:132` |
| 10 | Flow share of penetrating edges ÷ their edge share | — | Flow share alone tracks how many edges penetrate, which is itself downstream of the parenchymal volume difference H1 §1.3 reports; the ratio removes that | **On** | `cb_h2_glomus_perfusion.py:156` |

**Step 5 is what makes the classification independent of polyline sampling density.** The stored
`voxels` lists are not uniformly spaced after B-spline smoothing (§2.5 step 7), so counting points
would let a densely sampled stretch outvote a long one. Resampling at half the finest voxel also
guarantees a mask crossing cannot be stepped over.

**Step 6 samples midpoints, not endpoints.** Each sub-step contributes its own length, so the
average over sub-steps is exactly a length-weighted average along the segment.

> **Why not an endpoint test.** A capillary penetrating a cluster usually starts and ends in
> stroma. An endpoint test would classify exactly the vessels the question is about as
> extra-glomus.

**The quantity is a shunt index, not a flow share:**

$$\text{shunt index} = \frac{\text{flow share penetrating}}{\text{edge share penetrating}}$$

An index of 1 means flow is indifferent to the clusters. Below 1 means flow is carried
preferentially by the vessels that bypass them — the shunting the method is trying to detect.

**Flow share alone cannot answer it**, because flow share tracks how many edges penetrate, which is
itself downstream of the parenchymal volume difference H1 §1.3 reports. The ratio removes that.

**This is a within-specimen ratio**, so it sits under the ±5.9% floor rather than the ±47% one
(§13.3).

> **At a glance** — 50% length-in-mask classification, flow share over edge share ·
> TH threshold 0.5, ROI 160³, boundary axis 1 · `examples/cb_h2_glomus_perfusion.py` ·
> `tests/test_cb_h2_vtk_export.py`

---

### 7.4 Spatial haematocrit profiling (H2 §2.2)

**The question.** Do the vessels supplying the glomus clusters carry a lower discharge haematocrit
than the rest — a dense capillary bed largely filled with cell-free plasma?

Same edge classification as §7.3. The readout is the median haematocrit of penetrating edges against
that of bypassing edges, taken from the converged rheology solve (§4.3).

**The order of operations.** Steps 1–9 are identical to §7.3 — the same run produces both
readouts. Only the final reduction differs:

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1–9 | As §7.3, through to the penetrating / bypassing split | — | The classification and the flow field are the same ones §7.3 needs, so one run answers both questions | **On** | `cb_h2_glomus_perfusion.py` |
| 10 | Median `hematocrit` of penetrating edges | — | The median resists the long tail that a few near-zero-flow edges put into the haematocrit distribution | **On** | `cb_h2_glomus_perfusion.py:143` |
| 11 | Median `hematocrit` of bypassing edges | — | The comparison group, computed identically so the two are commensurable | **On** | `cb_h2_glomus_perfusion.py:143` |
| 12 | Report the ratio, not either median alone | — | Absolute haematocrit inherits every assumption in §4.2; the ratio is what survives them | **On** | `cb_h2_glomus_perfusion.py:161` |

**Medians, and non-finite values dropped before taking them.** An edge whose haematocrit did not
resolve is excluded rather than counted as zero.

Haematocrit is produced by the phase-separation model, so this quantity inherits every assumption in
§4.2 — in particular that separation occurs at binary bifurcations only.

---

### 7.5 Glomus-specific hypoxic fraction (H2 §2.3)

**The question.** With a higher metabolic rate assigned to TH-positive voxels and a lower one to
stroma, what fraction of the TH-positive volume falls below a hypoxic PO₂?

**How.** Solve the tissue field on the heterogeneous grid (§6.5), then take the fraction of
TH-weighted cell volume below each threshold. Thresholds swept at **5, 10 and 20 mmHg**; metabolic
contrast swept at **1×, 2× and 4×**; grid at 3 µm, vessels mapped over their cross-section.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Load the graph; select boundaries by the face rule | axis 1 | PO₂ depends on where blood enters, so boundary selection sets the entire tissue field | **On** | `cb_h2_hypoxic_fraction.py:107` |
| 2 | Coupled flow / haematocrit solve (§4.3) | 60 / 20 mmHg | The oxygen source term is flow times content, so neither is known until the network is solved | **On** | `cb_h2_hypoxic_fraction.py:109` |
| 3 | Threshold the TH field into a glomus mask | 0.5 | The metabolic contrast is defined against glomus tissue, which has to be a region before it can carry a rate | **On** | `cb_h2_hypoxic_fraction.py:111` |
| 4 | Build the perfusion grid (§6.1) | 3 µm | The tissue field needs a discretisation. 3 µm is the coarsest grid within 0.5 mmHg of the next refinement once vessels are mapped over their cross-section (§6.8, open item 30) | **On** | `cb_h2_hypoxic_fraction.py:116` |
| 5 | Per-cell TH volume fraction, then the blended `M_max` field (§6.5) | c ∈ {1, 2, 4} | A heterogeneous metabolic field is the entire mechanism this method proposes to detect | **On** | `cb_h2_hypoxic_fraction.py:117` |
| 6 | Map vessels to the grid (§6.2), assemble the operator (§6.3) | `cross_section` | Couples the 1D network to the 3D tissue and assembles the operator that will be solved | **On** | `cb_h2_hypoxic_fraction.py:126` |
| 7 | Newton solve for the tissue PO₂ field (§6.7) | 50 steps, tol 1e-5 (`PerfusionSettings`) | Metabolism saturates with PO₂, so the system is non-linear and needs iteration rather than one solve | **On** | `cb_h2_hypoxic_fraction.py:128` |
| 8 | Report the share of cells with no oxygen source at all | — | Padding the grid raises this share by construction, so it is reported rather than left to be inferred from the PO₂ distribution | **On**, diagnostic | `cb_h2_hypoxic_fraction.py:102` |
| 9 | Hypoxic fraction = TH-weighted volume below each threshold | 5, 10, 20 mmHg | Weighting by TH occupancy rather than by cell count is what makes it a fraction of glomus volume rather than of grid | **On** | `cb_h2_hypoxic_fraction.py:145` |

**Step 8 exists because step 4 can produce cells no vessel reaches.** Under `--pad-grid` that share
rises by construction, which is the stated cost of representing tissue beyond the vasculature
(§6.1 step 4). It is reported rather than left for the reader to infer from the PO₂ distribution.

> ⚠ **Read §13.6 before using this.** The tissue is not diffusion-limited — the oxygen diffusion
> length is 20–45 µm against a median tissue-to-vessel distance of 4.6–6.2 µm. Before open items 22
> and 29, raising the glomus rate to four times stromal moved PO₂ inside the TH volume by
> **0.01 mmHg**, and this block concluded that the mechanism could not operate on this geometry.
>
> **Re-run 2026-09-30 (open items 37, 38, 40; placed boxes at 0.95, converged loop, exact TH
> fraction, centred TH lookup).** PO₂ in TH (TH-weighted mean) is 88.5–95.2 mmHg at uniform
> metabolism and 83.3–93.8 at four times stromal; four times stromal moves it by −1.4 mmHg on WKY-A
> (95.18 → 93.77) and −5.3 on SHR-C (88.54 → 83.25). **TH hypoxia below 5, 10 and 20 mmHg is 0 in
> every specimen at every contrast.** All-cell hypoxia is not: up to 4.5% of cells below 10 mmHg and
> 7.3% below 20 (SHR-A, contrast 1), all of it stroma. PO₂ in TH separates the groups, SHR/WKY
> 0.967 / 0.957 / 0.942 at contrast 1 / 2 / 4, no overlap; it is an absolute PO₂ and sits under the
> absolute-flow floor (§13.3), so the H2 whitepaper records it but does not claim it. The padded grid
> is now identical to the unpadded one: every network has nodes on the first and last voxel plane
> of every axis, so the node box plus half a cell already contains the mask bounds. The H2
> whitepaper now grades §2.3 "Not supported" (returns zero). Before this re-run (unconverged loop,
> clipped TH fraction): PO₂ in TH 88.7–95.2, ratios 0.967 / 0.959 / 0.946.
>
> The output is still meaningful as a curve in the assumed contrast. It is not a number.

---

### 7.6 Transit time and PO₂ depletion (H2 §2.4)

**Per-edge transit time** is lumen volume over volumetric flow:

$$\tau_\text{edge} = \frac{\pi (d/2)^{2} L}{Q}$$

**Quadratic in diameter**, where resistance is quartic — so this carries a different sensitivity to
calibre than the flow solve does, and its own share of the floor in §13.3.

**An edge carrying no flow gets `inf`, not a large number.** Blood that does not move does not
arrive, and a finite stand-in would propagate as a merely slow path.

**The order of operations.**

| # | Step | Setting | Why | On the CB path | Where |
|---|---|---|---|---|---|
| 1 | Scan every edge for a usable diameter; **raise** if any lacks one | — | Transit time is lumen volume over flow, so a fabricated calibre would produce a fabricated transit time quadratically | **On** | `transit.py:38` |
| 2 | Per edge, lumen volume $\pi (d/2)^{2} L$ | — | The blood in an edge is its lumen volume, and that is what has to be displaced for blood to cross it | **On** | `transit.py:52` |
| 3 | $\tau = \text{volume} / \lvert Q \rvert$, or `inf` when Q = 0 | — | Blood that does not move does not arrive; a finite stand-in would propagate as a merely slow path | **On** | `transit.py:53` |
| 4 | Direct each edge by its solved `flow_signed` | — | An edge carrying blood away from a node cannot deliver blood to it, so adjacency alone would report routes no blood takes | **On** | `transit.py:77` |
| 5 | Dijkstra from all inlets at cost 0 | — | Every inlet is an equally valid origin, so the arrival time is the earliest over all of them | **On** | `transit.py:86` |
| 6 | Unreachable nodes carry `inf`, never absent | — | A missing key could be read as zero, which is the opposite of what an unreachable node means | **On** | `transit.py:84` |
| 7 | Score an edge by the **later** of its two ends | — | A penetrating capillary should be scored by how long blood takes to get through it, not to reach its nearer end | **On** | `cb_h2_glomus_perfusion.py:110` |
| 8 | Report the penetrating / bypassing median ratio | — | The magnitude is in arbitrary units (§3.7) and sits under the ±47% calibre floor, so only a ratio computed identically means anything | **On** | `cb_h2_glomus_perfusion.py:166` |

**Step 4 follows flow, not adjacency.** An edge carrying blood *away* from a node cannot deliver
blood *to* it, and ignoring the direction would report a transit time along a route no blood takes.

**Step 5 is Dijkstra rather than a topological pass, deliberately.** The flow directions come from a
numerical solve and can contain a small cycle. A topological sort raises on those; Dijkstra returns
the same answer where the directions are acyclic and terminates where they are not.

**Step 7 takes the later end.** A penetrating capillary is scored by how long blood takes to get
*through* it, not to reach its nearer end.

**Path transit time** is the minimum accumulated τ from any inlet, by Dijkstra over the
flow-directed graph.

> **Reported as a ratio, never an absolute.** Two independent reasons, and they compound. An
> absolute flow quantity sits under the ±47% floor from calibre alone (§13.3). And the pressure,
> viscosity and length units are not reconciled to one system (§3.7), so the magnitude is in
> arbitrary units. Both have the same answer: compare transit time to one set of terminals against
> another, computed identically, and the shared error divides out.

**It raises on a missing diameter** rather than substituting one.

> **At a glance** — $\tau = \pi r^{2} L / Q$, Dijkstra from inlets, `inf` for zero flow, ratios only ·
> `transit.py:28`, `transit.py:57` · `tests/test_transit.py`

---

### 7.7 Cohort-split diagnostics

**Not a physiological quantity — a check on the others.**

Some quantities in this study are supposed to be properties of the *instrument* rather than the
tissue: the segmentation threshold, the foreground fraction at a frozen threshold, the classifier's
mean output probability. If one of those separates cleanly by cohort, part of the measured group
difference is the measuring device, and the biological reading is contaminated in a way nothing
downstream can undo.

**Checking by eye does not work at n = 3.** Complete separation happens by chance with probability
2/C(6,3) = **0.10**, so "all the WKY values are below all the SHR values" is weak evidence on its
own. It is also the exact floor of a two-sided rank test at this n — no arrangement of three against
three can reach a smaller p.

**What the test does.** Computes the exact two-sided permutation p for the difference in means over
every group assignment, and reports the floor alongside it so the p is read against what is
achievable rather than against 0.05.

**The verdict is not "separated" but "concerning":** set only when the groups separate completely
**and** the gap between them exceeds the within-group spread. Separation alone is too weak to act
on; separation with a gap wider than the noise is worth stopping for.

> **At a glance** — exact permutation p with its own floor reported · floor p = 0.10 at n = 3 ·
> `cohort_split.py:56`, `cohort_split.py:41` · `tests/test_cohort_split.py`

---

### 7.8 Pressure boundaries used by these methods

The H2 drivers use **60 mmHg in and 20 mmHg out**, arteriolar to venular, from
`cb_settings.INLET_PRESSURE_MMHG` / `OUTLET_PRESSURE_MMHG`. `HaemodynamicsConfig` declared
100/2 mmHg (MAP to CVP) until open item 10; it now takes its defaults from the same two constants,
and `test_cb_settings.py` keeps the config and both example YAMLs equal to them. Since open item 2
the pipeline also picks the same boundary nodes as the H2 drivers (face rule, axis 1), so the two
now differ only in the rheology and perfusion settings, not in where the pressures act.

The pair is chosen, not measured in the carotid body (§8.1, §10.7). It does not affect the
within-specimen ratios, which are the reportable quantities.

---

## §8 — Boundary and initial conditions

The mechanism for *choosing* boundary nodes is §2.8. This section is about what is *imposed* on them
once chosen, and what happens to everything else.

### 8.1 Pressure boundary conditions

Dirichlet at both ends: a fixed pressure at inlet terminals, a fixed pressure at outlet terminals,
and nothing imposed anywhere else.

| Source | Inlet | Outlet | Gradient | Interpretation |
|---|---|---|---|---|
| **`cb_settings` (H2 drivers and `HaemodynamicsConfig`)** | **60 mmHg** | **20 mmHg** | **40 mmHg** | Arteriolar to venular |
| `HaemodynamicsConfig` before open item 10 | 100 mmHg | 2 mmHg | 98 mmHg | Systemic MAP to central venous pressure |

Every published H2 number used the first row. The config's old pair was 2.45× larger in driving
pressure; closing open item 10 moved the config to the first row, so no published number changed.

**What the old config value assumed.** That the entire arterial-to-venous pressure drop of the
systemic circulation falls across roughly 1 mm of tissue. It does not — most of it falls across the
arterial tree upstream and the venous tree downstream. This overestimates perfusion pressure and
therefore absolute flow (§11 row 15).

**Where 60/20 sits against measurements.** A PubMed search found no microvascular pressure
measured inside the carotid body. In other rat beds, servo-null micropuncture puts ≈50 µm arterioles at 55 mmHg and
≈20 µm arterioles at 30–34 mmHg, with venules of 23–54 µm at 18–26 mmHg, at a systemic pressure of
110 mmHg (stomach [`peti-peterdi_direct_1998`]). In WKY mesentery, arterial microvessels of 15–25 and
25–45 µm read 3.6 and 4.2 kPa (27 and 32 mmHg); SHR read 4.8 and 6.0 kPa (36 and 45 mmHg), while
small venules did not differ between strains [`jin_study_1997`, English abstract only]. In cat
skeletal muscle the smallest venules (8–15 µm) average 24 mmHg [`fronek_microvascular_1975`]. So
20 mmHg is a venular pressure, and 60 mmHg is an upper value for the arterioles feeding a bed of this
size; arterioles of 15–45 µm run nearer 30 mmHg, which would lower the driving pressure further.
SHR arteriolar pressures run higher than WKY; one pair is used for all six specimens on purpose, so
the model carries none of that group difference (§11 row 15).

**At 60/20 mmHg the network runs fast, not slow.** §13.5 measures flow-weighted velocities of
977–1,812 µm/s against a physiological 200–1,000, and a 500 µm/s velocity needs a drop of only
11–20 mmHg (979–3,319 µm/s and 6–20 mmHg on the centred boxes before the 2026-09-28 re-run). That fits the paragraph above: arterioles of this size run nearer 30 mmHg, which would
lower the drive. (Before open item 12 was re-derived this paragraph read 4–10 µm/s and 3,257 mmHg;
those came from the inflated pre-`7ea1b36` resistances.)

**Within-specimen ratios are insensitive to this.** Flow is linear in the pressure difference, so a
uniform change scales every edge's flow and cancels from any ratio taken within one specimen — the
same cancellation §4.4 describes for viscosity.

### 8.2 What happens to terminals that are not boundaries

**About 84% of degree-1 nodes are interior** — nowhere near a region face. Counting terminals within
one voxel of each of the six ROI faces (current batch graphs, `cb_h2_error_propagation.py` S10,
2026-09-30; the census now measures against the ROI extent in the graph frame, as the face rule does,
where it used the exported mask bounds before, which gave 82.8–89.4% on the same graphs; the
centred-box graphs gave 83.5–86.9%):

| Specimen | Terminals | On any face | Interior | Interior share |
|---|---|---|---|---|
| WKY-A | 784 | 148 | 636 | 81.1% |
| WKY-B | 681 | 107 | 574 | 84.3% |
| WKY-C | 715 | 91 | 624 | 87.3% |
| SHR-A | 715 | 132 | 583 | 81.5% |
| SHR-B | 632 | 106 | 526 | 83.2% |
| SHR-C | 714 | 82 | 632 | 88.5% |

On the pinned axis only 27–44 of these are pressure boundaries (8–23 inlets, 13–27 outlets per
specimen); the rest, 93–96% of terminals, are stranded dead ends.

**The crop is not the boundary problem; interior dead ends are.** These are skeletonisation spurs
and segmentation breaks, not vessels severed by the ROI. A real capillary bed has few genuine
interior dead ends, so this is a statement about mask quality rather than about the crop.

Three modes decide what happens to them:

| Mode | Behaviour | Consequence |
|---|---|---|
| **`caged`** (default) | Interior terminals are not boundaries at all | They become no-flow dead ends. Under the band rule roughly **half of all terminals** are stranded this way (45–61% on the 2026-09-28 graphs) |
| `universal_sink` | Every non-inlet terminal becomes an outlet | No stranding, but every mask defect becomes a drain |
| `robin_resistance` | Non-boundary terminals are tagged for a distal resistance | A middle course; multiplier 10.0, unswept |

**Why the inlet:outlet ratio matters.** Under a fixed pressure boundary it directly scales how much
flow the network carries. Measured under the band rule on the centred-box graphs the ratio spanned
**10.7×** across six specimens, from 2.67 to 0.25, with group means 1.87 (WKY) against 0.88 (SHR).
That is the right size and the right direction to become a confound, and it is set by ROI placement
rather than by biology. On the 2026-09-28 graphs the same census gives 1.11–2.13 (1.9×), group
means 1.26 (WKY) against 1.64 (SHR): smaller, and in the other direction.
The face rule (§2.8) is what reduces this; it is the reason boundary selection is the largest single
lever in §13.4.

### 8.3 Domain truncation

The graph is a crop of a larger organ, so vessels genuinely do cross the region faces. The face rule
treats exactly those as pressure boundaries and refuses when a face carries none — it does not
invent them.

**What is not modelled:** any pressure or flow condition representing the vasculature upstream of
the inlet face or downstream of the outlet face. The network is solved as if it were the whole
circuit between those two pressures.

### 8.4 Tissue boundary conditions

**Zero-flux (Neumann) on all six faces, by construction.** The seven-point stencil simply writes no
conductance across the domain faces, so no oxygen enters or leaves there.

**Consequence:** tissue PO₂ is **overestimated** near the domain boundary, because the model gives
that tissue no route to lose oxygen to the tissue beyond the crop (§11 row 23).

**A null space, and how it is handled.** Pure diffusion under Neumann boundaries has rows summing to
zero, so a constant offset in PO₂ is unconstrained and the matrix is singular. Two regularisations
address it: a 10⁻¹² sink added to the diagonal at assembly, and a further 10⁻⁶ before the solve. The
blood and metabolic slopes on the Newton diagonal (§6.7) also contribute diagonal dominance.

**Grid extent.** The grid is padded to span the segmented volume rather than the graph's own extent
(§6.1). Tissue beyond the segmentation is not represented at all — it is outside the domain, not
merely unperfused.

### 8.5 Blood gas inlet conditions

Values carried by blood entering the tissue, all constants (§10.8, §10.9):

| Quantity | Value | Notes |
|---|---|---|
| Arterial PO₂ | 100 mmHg | `po2_arterial_mmHg`, read by all three tiers |
| Arterial PCO₂ | 40 mmHg | |
| Systemic haematocrit | 0.45 | `systemic_hematocrit`; Tier 1 evaluates its washout at this value (§6.6) |
| Tissue bicarbonate | 24 mmol/L | Constant buffer; no renal compensation |

Arterial oxygen **content** is not imposed directly — it is computed from arterial PO₂ and the
edge's own haematocrit through the Hill equation (§5.1), then delivered per cell in proportion to
each edge's length share (§6.2).

### 8.6 Initial conditions

The tissue solves are steady-state, so the initial field is a starting guess for Picard iteration,
not a physical condition. It affects convergence, not the answer.

| Field | Initial value |
|---|---|
| Tissue PO₂ | 0 mmHg everywhere |
| Tissue PCO₂ (Tier 3) | 40 mmHg — arterial baseline |
| Tissue pH (Tier 3) | Henderson–Hasselbalch at the initial PCO₂ (7.401 at HCO₃⁻ 24) |

**PO₂ starting at zero is deliberate.** It approaches the steady state from below, and each iterate
is clamped to ≥ 0, so the sequence cannot enter the non-physical region that drives Picard
oscillation.

The rheology loop starts every edge at systemic haematocrit 0.45 with the corresponding
Pries–Secomb viscosity (§4.3) — likewise a starting guess, replaced on the first pass.

> **At a glance** — Dirichlet pressure at face terminals, everything else caged; Neumann on tissue ·
> 60/20 mmHg in H2 and the pipeline config alike; 86% of terminals interior · `boundaries.py:106`,
> `perfusion.py:349`, `perfusion.py:461` · `tests/test_boundary_faces.py`,
> `tests/test_flow_conservation.py`

---

## §9 — Numerical methods

Settings live in Appendix A. This section is about *why* each choice is what it is.

### 9.1 Spatial discretisation

One scheme, used everywhere in the tissue domain: a **seven-point finite-volume stencil** on a
regular Cartesian grid, with conductance across each face equal to σ × (face area) / (normal
spacing). Second-order accurate in space on a uniform grid.

The 1D network needs no discretisation — the graph *is* the discretisation, and each edge is one
lumped resistor.

### 9.2 Non-linear solution — Picard, not Newton

Three non-linearities, all handled the same way:

| Loop | Non-linearity | Damping |
|---|---|---|
| Rheology (§4.3) | Viscosity depends on haematocrit, which depends on flow | Haematocrit relaxation 0.2; converges in 173–457 passes (open item 37) |
| Tissue Tier 1 (§6.7) | Metabolic sink and venous washout both depend on PO₂ | Newton with a line search (open item 22; was Picard with γ = 0.5) |
| Tissue Tier 3 | The same, coupled across O₂, CO₂ and pH | γ = 1.0 each |

**Why Picard, and where not.** Picard costs one linear solve per iteration and needs no
Jacobian. It is still used by the rheology loop. The tissue tiers now carry the slopes that matter:
Tier 3 a per-cell blood-response linearisation (open item 23), Tier 1 a full Newton step with a
line search (open item 22), both from central differences of the blood-gas functions. Tier 1's
fixed-slope Picard stopped converging once its diffusion carried α.

**The stabilisation is the interesting part.** A slope term on the diagonal, added back on the
right, leaves the steady-state roots unchanged while making the matrix strictly diagonally
dominant. A fixed slope (the old pseudo-washout) converges only while it is close to the true one;
§6.7 describes what replaced it in each tier.

### 9.3 Linear solvers

| System | Solver | Why |
|---|---|---|
| Network Laplacian | Direct below 50,000 nodes, iterative above | CB graphs sit far below the threshold, so the flow solve is exact to machine precision |
| Tissue diffusion, Tier 1 | Conjugate gradient, Jacobi preconditioned, `rtol` 10⁻¹⁰ per Newton step | The matrix is large, sparse, symmetric and positive definite once regularised; 439 k cells at 4 µm |
| Tissue diffusion, Tier 3 | Sparse LU, symmetric ordering | Exact; see §6.7 and open item 23 |

**The preconditioner must be SPD**, which is what conjugate gradient requires. A diagonal
(Jacobi) preconditioner is trivially SPD when the diagonal is positive, and the perfusion matrix's
diagonal is positive by construction. Non-positive diagonals are detected and declined rather than
silently inverted.

### 9.4 Numerical safeguards

| Safeguard | Where | Purpose |
|---|---|---|
| Diagonal regularisation, 10⁻¹² then 10⁻⁶ | ADR assembly and solve | Neumann boundaries leave a null space |
| PO₂ clamped ≥ 0 | Each Picard iterate | Negative PO₂ is non-physical and drives oscillation |
| Haematocrit clamped to [0, 0.95] | Phase separation | Keeps skimming outputs physical |
| Constriction ratio clamped ≥ 0.01 | Config validation | A zero ratio gives infinite resistance and a singular matrix |
| Degenerate bifurcation handled explicitly | Phase separation | Avoids the logit at flow fractions near 0 or 1 |
| Non-convergence warns, never fails silently | Both Picard loops | A truncated solve is reported, not returned as converged |

### 9.5 Quadrature and root finding

Trapezoidal quadrature over 1,000 points for the variable-diameter resistance integral (§3.3,
frozen). Inverting the Hill equation for PO₂ from oxygen content is done by bracketed root finding
in the coupled solvers.

---

## §10 — Parameter reference

### 10.1 Imaging and domain

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `PROCESSING_VOXEL_UM` | (1.8639, 1.866, 1.866) | µm, (z, y, x) | (i) | Acquisition. The single spacing every calculation uses; per-specimen values are kept as provenance only. Slightly anisotropic in z. | fixed |
| `DEFAULT_ROI` | (160, 160, 160) | voxels | (iii) | Analysed sub-volume: 298.2 × 298.6 × 298.6 µm = **0.0266 mm³**. Same value in `cb_h1_batch.py`, `cb_h2_glomus_perfusion.py` and `cb_h2_hypoxic_fraction.py` | matched across specimens by construction |
| Imaged block | 0.227–0.653 | mm³ | (i) | Acquisition. WKY-C smallest, WKY-A largest; the ROI is 4–12% of it | fixed |
| Cohort | 3 WKY, 3 SHR | — | (i) | Study design | — |
| Channels | lectin (vasculature), TH (glomus) | — | (i) | Two channels of one acquisition, so co-registered by construction | — |

### 10.2 Mask formation — `PreprocessingConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `median_filter_size` | 0 | voxels | (iii) | **Disabled.** A 3×3×3 median spans 5.6 µm against a 3.2-voxel capillary. On the three-capillary fixture it cut recall to 20.1% and thinned survivors from r_p90 2.64 µm to 1.87 µm. A post-threshold 50-voxel component filter achieves the same cleanup at 100% recall. | measured |
| `probability_smoothing_sigma` | 0.0 | voxels | (iii) | **Disabled**, same reasoning | measured |
| `morphological_opening_radius` | 0 | voxels | (iii) | **Disabled.** Radius 1 retains 51% of a 1.6-voxel-radius tube; radius 2 retains none | measured |
| `morphological_closing_radius` | 0 | voxels | (iii) | **Disabled**, same reasoning | measured |
| `enable_hysteresis_threshold` | True | — | (iii) | — | — |
| `hysteresis_threshold_low` | 0.95 | probability | (iii) | `cb_settings.HYSTERESIS_LOW` = `FROZEN_THRESHOLD`: the median of the six per-specimen selections, snapped to the sweep grid (§2.2); all six chose 0.95 on 2026-09-28. Inclusive, `p ≥ low` (open item 17). Not tuner-derived — the preprocessing objective's argmin is the top of its search range. Was 0.90 until 2026-09-28, and 0.65 until open item 1, set by hand on an earlier subvolume (§2.3.1) | measured, in sensitivity scope |
| `hysteresis_threshold_high` | 0.999 | probability | (iii) | `cb_settings.HYSTERESIS_HIGH` = min(`HYSTERESIS_SEED_CAP` 0.999, low + `HYSTERESIS_HIGH_OFFSET` 0.05); on this field `p ≥ 0.999` is `p = 1.0`. Was 0.95 until 2026-09-28, and 0.75 until open item 1 | measured, in sensitivity scope |
| `enable_hole_filling` | True | — | (iii) | — | unswept |
| `ilastik_vessel_channel` | 0 | index | (i) | Classifier output layout | — |
| `enable_shannon_entropy` | False | — | (iii) | **Off.** The vessel classifier has 2 classes; turning this on raises (§2.3) | no effect |
| `shannon_entropy_threshold` | 0.95 | normalised entropy | (iii) | Chosen. Max entropy for a *candidate* voxel; permissive gate | no effect on the CB path |
| `shannon_entropy_core` | 0.6 | normalised entropy | (iii) | Chosen. Max entropy for a *seed* voxel; strict gate | no effect on the CB path |

> **Open item 1 (closed).** The config defaults were 0.65 / 0.75 while every H1 run passed the
> frozen 0.90 as `--hysteresis-low` and got a 0.95 seed. The config now reads the band from
> `cb_settings`, and `test_cb_settings.py` keeps them equal. No published number moved. Since the
> 2026-09-28 re-selection the band is 0.95 / 0.999.

### 10.3 Skeletonisation and topology — `SkeletonConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `closing_radius` | 1 | voxels | (iii) | Chosen | unswept |
| `bridge_gap_size` | — | — | — | **Removed** from `SkeletonConfig` (§14.3 row 3). It applied a second radius-1 closing after `closing_radius`, and closing is idempotent | — |
| `min_branch_length` | 3 | voxels | (iii) | Chosen | unswept |
| `max_bridge_distance` | 0 | voxels | (iii) | Disabled | — |
| `component_connectivity` | 3 | — | (iii) | Full 26-connectivity | — |
| `min_component_percent` | 5.0 | % | (iii) | Chosen | unswept |
| `downsample_factor` | 1.0 | — | (iii) | No downsampling | — |
| `sub_volume_percentage` | 0.15 | fraction | (iii) | Superseded in cohort work by `sub_volume_voxels` — a percentage samples a larger absolute box from SHR (89 Mvoxel mean) than WKY (63 Mvoxel), carrying specimen extent into every count | measured |
| `sub_volume_voxels` | None (set per run) | voxels | (iii) | **Required for any cross-specimen comparison**, for the reason above | — |
| `bundle_scan_size` | 9 | voxels (≈16.8 µm) | (iii) | Window for the bundle-density operator | measured |
| `bundle_density_fraction` | 1.0 | fraction | (iii) | **Disabled** (1.0 is unreachable, so the operator short-circuits). At the former 0.025 it destroyed 208 of 307 fundamental loops — 68% of β₁, which *is* the H1 §1.1 readout — and 29% of the skeleton. It is also group-dependent in the false-negative direction: denser networks fire more hubs and lose proportionally more loops, actively suppressing the SHR/WKY difference. No validated operating point exists: the density distribution runs smoothly 0.02–0.06 with no gap | measured, decisive |
| `bundle_max_connections` | 5 | — | (iii) | Inert while the operator is disabled | — |
| `bundle_hub_min_spacing` | 0 | voxels | (iii) | Inert while disabled | — |
| `smoothing_alpha` | 0.75 | — | (iii) | **Frozen, deliberately not tuned.** It sets the centreline curvature that H1 §1.4 reads tortuosity from, and no Optuna objective can see tortuosity — so tuning it would optimise against a proxy for the thing being measured | unswept |
| `core_dead_end_resolution_mode` | "none" | — | (iii) | — | — |
| `core_safe_zone_percent` | 5.0 | % | (iii) | Inert under mode "none" | — |
| `core_stitch_max_distance_um` | 15.0 | µm | (iii) | Inert under mode "none" | — |
| `core_stitch_max_degree` | 4 | — | (iii) | Inert under mode "none" | — |

### 10.4 Graph and boundaries — `GraphConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `keep_largest_component_only` | True | — | (iii) | — | unswept |
| `min_stub_length_um` | 5.6 | µm | (iii) | A skeletonisation spur at a branch point cannot exceed the local vessel radius; measured inscribed radius is p90 3.73 µm, p99 5.60 µm. Cannot affect β₁ (pruning removes only degree-1 nodes, which lie on no cycle) — verified constant at 307 from 0 to 30 µm. Does move the §1.2 and §1.4 per-edge distributions | measured (0.0 / 5.6 / 10.0 / 18.7 µm → 0% / 1.6% / 4.8% / 10.5% of nodes removed) |
| `boundary_permeability_mode` | "caged" | — | (iii) | Chosen. Alternatives `universal_sink`, `robin_resistance` | unswept |
| `robin_distal_resistance_multiplier` | 10.0 | — | (iii) | Inert under "caged" | — |
| `boundary_axis` | 1 | axis index | (iii) | From `cb_settings.BOUNDARY_AXIS`: the only axis with terminals on both faces in all six specimens (§2.8) | measured (§13.4) |
| `face_tolerance_voxels` | 1.0 | voxels | (iii) | From `cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS`: one voxel means "on the face" | measured (1/2/4 voxels, 8.9% ratio spread, §13.4) |

> **Open item 2, closed.** `GraphConfig` used to carry the band-rule parameters (`edge_percent`,
> `end_percent` = 25, `node_edge_axis` = 0) while the H2 drivers ran the face rule on axis 1. The
> band parameters are gone; the two rows above read `cb_settings`, and `test_cb_settings.py` fails
> if they drift. `select_boundary_nodes_by_method` keeps its own band defaults (axis 1, 10 / 10%)
> for the nerve pipeline, which the CB path does not use.

### 10.5 Calibre assignment — `HaemodynamicsConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `radius_assignment_mode` | "edt_radius" | — | (iii) | Chosen on measured evidence over `fwhm_radius`. Both run over the same 1330 edges: EDT 100% coverage, median 6.37 µm, p95 11.34, max 20.09; FWHM 76.5%, median 8.20 µm, p95 16.78, max 39.16. They correlate weakly (Pearson r = 0.245, Spearman ρ = 0.284; median ratio FWHM/EDT = 1.359). FWHM's tail is not physical for a bed whose measured inscribed radius is p99 5.60 µm | measured |
| `constant_radius_um` | 5.0 | µm | (iii) | Used only under `constant_radius` mode | — |
| `edt_junction_proximity_exclusion_um` | 3.73 | µm (2 voxels) | (iii) | Within ~one radius of a bifurcation the EDT returns the junction's inscribed sphere, biasing radius upward. Specified externally as 2 voxels; ≈ one capillary inscribed radius. The swept optimum is nearer 1.5 voxels, but 2.80 µm would be tuned to one subvolume and the difference is 0.3% on resistance | measured (0.93→5.60 µm swept; effect on resistance ~4–8%) |
| `MAX_SYNTHETIC_FRACTION_EDT` | 0.0 | fraction | (iii) | EDT has no legitimate per-edge failure mode on a mask that covers the vessel — 100% measured provenance was observed across 34,900 edges, and all 42,211 of the 2026-09-28 re-run — so any fallback is a defect, not an expected shortfall. FWHM is exempt by default because Gaussian fitting genuinely fails on individual edges | measured |
| `fwhm_sample_spacing_along_edge_um` | 2.0 | µm | (iii) | Chosen | unswept |
| `fwhm_transverse_profile_step_um` | 0.5 | µm | (iii) | Chosen | unswept |
| `fwhm_transverse_half_extent_um` | 15.0 | µm | (iii) | ≈8 voxels, ≈4.7 vessel radii. In a bed this dense the transverse profile runs into neighbouring vessels, which is one of the two mechanisms behind FWHM's inflated tail | measured (indirectly) |

### 10.6 Constriction geometry — `HaemodynamicsConfig` · **Frozen**

Present in the code, disabled for the CB path, and `__post_init__` raises if re-enabled, including from a `--config` YAML (open item 35).

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `constrict_at_pericytes` | False (raises if True) | — | (iii) | Sites are placed by a hard-coded topological rule rather than measured from imaging, and severities come from no model of vasomotor tone. Because the ratio multiplies whatever diameter was measured, the fabrication reaches measured edges too — and resistance goes as $d^{-4}$, so the 0.5 capillary ratio is a 16× local resistance error on a real vessel | measured |
| `constriction_mode` | "sphincter" | — | (iii) | Alternative: "periodic" | — |
| `sphincter_length_um` | 5.0 | µm | (iii) | Chosen | — |
| `intimal_cushion_constriction_ratio` | 0.60 | fraction | (iii) | Chosen; no vasomotor model behind it | — |
| `pre_capillary_constriction_ratio` | 0.50 | fraction | (iii) | Chosen; no vasomotor model behind it | — |
| `pre_capillary_topological_offset` | 1 | branch orders | (iii) | Chosen | — |

### 10.7 Haemodynamics — pressures, viscosity, units

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `input_p_bc` | 7.999 × 10⁶ | mPa (= 60 mmHg, `cb_settings.INLET_PRESSURE_MMHG`) | (iii) | Arteriolar pressure, chosen; not measured in the carotid body. Rat stomach arterioles read 55 mmHg at ≈50 µm and 30–34 mmHg at ≈20 µm [`peti-peterdi_direct_1998`]; WKY mesenteric arterial microvessels 27–32 mmHg at 15–45 µm, SHR 36–45 [`jin_study_1997`, abstract only]. So 60 is an upper value. Was 100 mmHg (MAP; 101 ± 2 anaesthetised WKY [`izuta_cerebral_1995`]) until open item 10 | Flows are linear in p_in − p_out, so within-specimen ratios do not move (§8.1) |
| `output_p_bc` | 2.666 × 10⁶ | mPa (= 20 mmHg, `cb_settings.OUTLET_PRESSURE_MMHG`) | (ii) | Venular pressure: rat stomach venules 18–26 mmHg [`peti-peterdi_direct_1998`]; cat muscle venules of 8–15 µm 24 mmHg [`fronek_microvascular_1975`]; no WKY/SHR difference in small mesenteric venules [`jin_study_1997`]. Was 2 mmHg (CVP; 4 ± 3 in conscious control rats [`willenbrock_effect_1997`]) until open item 10 | As `input_p_bc` |
| `blood_plasma_viscosity_cP` | 1.2 | cP | (i) | Plasma viscosity; normal range 1.10–1.30 mPa·s at 37 °C, human [`kesmarky_plasma_2008`] | assumed |
| Viscosity law | `in_vivo` | — | (ii) | Pries et al. 1994, fitted to microvessels in living tissue, where the endothelial surface layer narrows the effective lumen [`pries_resistance_1994`]. `in_vitro` (Pries et al. 1992, glass tubes) is available and not default [`pries_blood_1992`] | measured — the two differ by ≈3.4× apparent viscosity at D = 8 µm, but a 3–4× change moved no within-specimen ratio (§13) |
| μ₄₅ in vivo | 6.0·e^(−0.085 d) + 3.2 − 2.44·e^(−0.06 d^0.645) | relative | (ii) | [`pries_resistance_1994`] | — |
| μ₄₅ in vitro | 220·e^(−1.3 d) + 3.2 − 2.44·e^(−0.06 d^0.645) | relative | (ii) | [`pries_blood_1992`] | — |
| Phase separation | Pries bifurcation relation | — | (ii) | [`pries_red_1989`], fitted to 65 arteriolar bifurcations in rat mesentery. The constants in use (A, B and *x₀* scaled by the feeding diameter *D_F*, §4.2) are the later parametrisation printed in [`rasmussen_modeling_2018`], attributed there to Pries, Reglin & Secomb (2003) | unswept |
| `PASCALS_PER_MMHG` | 133.322387415 | Pa/mmHg | (i) | Exact by definition of the conventional millimetre of mercury | exact |
| `POISEUILLE_FLOW_TO_UM3_PER_S` | 133.322387415 × 10³ | (µm³/s) per solver unit | (i) | Derived. The solve evaluates *R* = 128 μL/(π d⁴) with pressure in mmHg, viscosity in cP and lengths in µm, so its *Q* carries mmHg·µm⁴/(cP·µm) and is not a volumetric rate. Rewriting *R* in SI multiplies it by 10¹⁵. Only for callers passing mmHg (the H2 drivers); the pipeline passes mPa, so its factor is 1.0 (open item 31) | exact |
| `rheology_max_iterations` | 1000 | iterations | (iii) | `cb_settings.RHEOLOGY_MAX_ITERATIONS`; 2.2× the slowest specimen at 0.95 and 1.4× the slowest sensitivity run (WKY-B at 0.93, 726; open item 37). Was 15, which every solve hit | none: every solve converges |
| `rheology_relaxation` | 0.2 | — | (iii) | `cb_settings.RHEOLOGY_RELAXATION`; the largest step that converges all six (§4.3) | the converged answer does not depend on it |
| `rheology_flow_rtol` | 1 × 10⁻⁶ | relative to max \|Q\| | (iii) | `cb_settings.RHEOLOGY_FLOW_RTOL`. Was an absolute 1 × 10⁻⁴, a different test in mPa and in mmHg | unswept |
| `rheology_hematocrit_atol` | 1 × 10⁻⁴ | haematocrit | (iii) | `cb_settings.RHEOLOGY_HEMATOCRIT_ATOL` | unswept |
| Murray's law | **feeding diameter only** | — | — | The branch-order fallback is exponential, not Murray scaling [`murray_physiological_1926`]. Murray's cube law is used for one thing: *D_F* at a junction with no inflowing edge (§4.2), which does not occur on the six networks | — |

### 10.8 Blood gas chemistry — hard-coded in `perfusion.py`

These are **not** configurable. They live in the function bodies.

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `alpha_o2` | 1.34 × 10⁻³ | mmol/L/mmHg | (i) | O₂ solubility in plasma; 1.37 × 10⁻³ at 37 °C, human [`dash_erratum_2010`] | assumed |
| `hill_n` | 2.7 | — | (i) | Hill coefficient [`hill_possible_1910`]; n = 2.7 fits normal human blood for saturations of 20–98% [`dash_erratum_2010`] | assumed |
| `c_hb_max` | 0.446 × 20.4 / 0.45 ≈ 20.22 | mmol/L | (i) | Haemoglobin O₂ capacity scaled to pure RBC. Derived; compare 4 [Hb]_rbc = 4 × 5.18 = 20.7 mmol/L, human [`dash_erratum_2010`] | assumed |
| Baseline P₅₀ | 26.0 | mmHg | (i) | At pH 7.4, PCO₂ 40 mmHg. **Human** haemoglobin; the source gives about 26.8 mmHg [`dash_erratum_2010`] | assumed |
| Bohr pH coefficient | −0.4 | per pH unit (log₁₀ P₅₀) | (ii) | [`severinghaus_simple_1979`], [`kelman_digital_1966`] | assumed |
| Bohr PCO₂ coefficient | +0.06 | per log₁₀(PCO₂/40) | (ii) | [`severinghaus_simple_1979`], [`kelman_digital_1966`] | assumed |
| `alpha_co2` | 0.03 | mmol/L/mmHg | (i) | CO₂ solubility in plasma; 3.07 × 10⁻² at 37 °C, human [`dash_erratum_2010`] | assumed |
| CO₂ base capacity | 11.02 · PCO₂^0.396 | vol%, ÷ 2.226 to mmol/L | (ii) | McHardy whole-blood CO₂ dissociation curve at Hb 15 g/dL and S 95% [`mchardy_relationship_1967`], human. Total content, so dissolved CO₂ is not added again. Constants checked against the equation as quoted in [`mallat_ratio_2021`], not the 1967 paper. The old code comment's Spencer (1979) [`spencer_computational_1979`] was read by abstract only and could not be matched. Until open item 19 the code read vol% as mmol/L and multiplied by *H*; see §5.3 | assumed |
| CO₂ Hb correction | −(15 − Hb) · 0.015 · PCO₂, Hb = 15·*H*/0.45 | vol% (Hb in g/dL) | (ii) | McHardy's haemoglobin term [`mchardy_relationship_1967`, `mallat_ratio_2021`]. The only way *H* enters the CO₂ curve. Hb from *H* uses the 15 g/dL at *H* 0.45 that `c_Hb,max` assumes. At *H* near 0 content peaks near PCO₂ 135 mmHg (§5.3) | assumed |
| Haldane (saturation) term | (95 − 100·S_O₂) · 0.064 · Hb/15 | vol% | (ii) | McHardy's saturation term [`mchardy_relationship_1967`, `mallat_ratio_2021`], scaled by Hb/15 so plasma has no Haldane effect (a modification; unchanged at *H* 0.45; §5.3). Full desaturation adds 2.8 mmol/L at PCO₂ 40 and *H* 0.45, against ≈2.5 mmol/L implied by the measured human Haldane factor [`loeppky_quantitative_1983`]. Replaced the unsourced (0.15 − 0.05·S_O₂)·PCO₂ term, which gave 0.90 (open item 19). Saturation is evaluated at fixed P₅₀ = 26, not the Bohr-shifted value (§11 row 19) | assumed |
| `pKa` | 6.1 | — | (i) | Henderson–Hasselbalch; 6.10, revised to 6.09 at pH 7.4 and 37.5 °C [`severinghaus_variations_1956`] | assumed |

> ⚠ **Species mismatch.** The haemoglobin parameters above are human. The tissue is rat. Direction
> of the resulting bias is not established.

### 10.9 Tissue transport — `PerfusionConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `do_perfusion_modeling` | True | — | — | — | — |
| `grid_resolution_xyz` | (10, 10, 10) default; **3 µm** for H2 §2.3 | µm | (iii) | 3 µm chosen on convergence with vessels mapped over their cross-section: every measure within 0.21 mmHg from 3 to 2 µm, up to 0.78 from 4 to 3 µm (§6.8). Re-measured on all six after open item 38: within 0.16 from 3 to 2 µm, up to 0.495 from 4 to 3 µm, so 4 µm passes by 0.005 mmHg and 3 µm is kept. It was 4 µm, chosen when median PO₂ ran 27.34 / 27.92 / 28.21 at 10 / 6 / 4 µm; after open items 22 and 29 the centreline mapping (centred boxes) gave 91.38 / 90.46 / 89.52 / 89.19 / 87.90 at 10 / 6 / 4 / 3 / 2 µm and no limit (open item 30) | measured |
| `sigma_diff` | 1.5 × 10⁻⁹ | m²/s | (i) | O₂ diffusivity in tissue (D, not D·α: every tier multiplies it by α_O₂, open item 22). Consistent with K_O₂/α_O₂ ≈ 1.6 × 10⁻⁹ in rat skeletal muscle [`kawashiro_determination_1975`] and (1.04 ± 0.78) × 10⁻⁹ in rat mesentery [`yaegashi_diffusivity_1996`] | assumed |
| `sigma_diff_co2` | 1.6 × 10⁻⁹ | m²/s | (i) | CO₂ diffusivity in tissue: measured K_CO₂/α_CO₂ ≈ 1.6 × 10⁻⁹ in rat skeletal muscle [`kawashiro_determination_1975`]. D is close to O₂'s; CO₂'s ≈20× faster transport comes from its solubility, which the solver multiplies in (`build_diffusion_matrix`). With the code's α values the Krogh ratio K_CO₂/K_O₂ is ≈24 (measured ≈21). Was 3.0 × 10⁻⁸ until open item 18 | assumed |
| `permeability_o2_cm_s` | 9.1 × 10⁻² | cm/s | (ii) | Endothelial O₂ permeability. Measured O₂ mass-transfer coefficient of a cultured human umbilical vein endothelial monolayer, *k* = 1.22 ± 0.45 × 10⁻¹⁰ mol·cm⁻²·s⁻¹·mmHg⁻¹ at 37 °C, n = 8 [`liu_oxygen_1994`], divided by this solver's α_O₂ (1.34 × 10⁻⁹ mol·cm⁻³·mmHg⁻¹), so the wall flux P·A·α·ΔPO₂ equals *k*·A·ΔPO₂. Their assumed 1 µm wall thickness cancels out of *k*/α. With their tissue α of 1.4 × 10⁻⁹ it is 8.7 × 10⁻² cm/s (their *D*_eff/1 µm); the monolayer-plus-media lower bounds give 1.4–2.0 × 10⁻² cm/s. Cultured cells, human and bovine; not rat capillary. Earlier indirect estimates were lower: 4.6 × 10⁻³ (canine heart) and 3 × 10⁻² cm/s (feline heart), as tabulated there. Was 1.0 × 10⁻⁴ with no source until open item 19, which left Tier 3 wall-limited and anoxic. Used by Tier 3 (and the unreachable Tier 2), not by Tier 1 or H2. At this value Tier 3's explicit step overshot at capillary flow (open item 21, now implicit); its Picard loop is still slow there (open item 23) | assumed |
| `permeability_co2_cm_s` | 9.1 × 10⁻² | cm/s | (ii) | Endothelial CO₂ permeability, set equal to `permeability_o2_cm_s`: [`dash_simultaneous_2006`] use one capillary PS for both O₂ and CO₂, and the wall flux is also scaled by α, so CO₂ still crosses ≈22× faster per mmHg. Not measured for CO₂; follows the O₂ value. Was 2.0 × 10⁻³ until open item 18, then 1.0 × 10⁻⁴ until open item 19 | assumed |
| `respiratory_quotient` | 0.82 | — | (i) | CO₂ produced per O₂ consumed; fasting whole-body RQ ≈ 0.80–0.90 depending on diet, human [`miles-chan_fasting_2015`]. Measured 0.85 in rat skeletal muscle [`kawashiro_determination_1975`] | assumed |
| `systemic_hematocrit` | 0.45 | fraction | (i) | Standard haematocrit ≈ 0.45, human [`dash_erratum_2010`]. Also the Tier 1 washout haematocrit (§6.6); `cb_settings.PerfusionSettings` carries the same value | assumed |
| `po2_arterial_mmHg` | 100.0 | mmHg | (i) | Standard arterial PO₂, human [`dash_erratum_2010`]. Read by all three tiers; `cb_settings.PerfusionSettings` carries the same value for the H2 drivers | assumed |
| `pco2_arterial` | 40.0 | mmHg | (i) | Arterial reference 40 ± 2 mmHg, human [`dash_erratum_2010`], [`berend_physiological_2014`]. The same in both example YAMLs; the SHR one said 35 until open item 34 | assumed |
| `hco3_tissue` | 24.0 | mmol/L | (i) | Fixed bicarbonate buffer; no renal compensation. Arterial reference 24 ± 2 mmol/L, human, used here for tissue [`berend_physiological_2014`] | assumed |
| `M_max` | 0.05 | mmol/L/s | (iii) | Maximum metabolic consumption rate. `cb_settings.BASE_M_MAX` for H2 and `PerfusionConfig` for the pipeline; the config said 0.005 until open item 8. 0.05 is the defensible value: it is 0.067 mL O₂ per mL per minute against roughly 0.040 for brain, the right order for a metabolically active organ | unswept in magnitude; the glomus:stroma *ratio* is swept |
| `k_reduce` | 0.1 | per mmol | (iii) | Phenomenological metabolic reduction in hypoxic zones. **Not Michaelis–Menten** — that form is used nowhere in the pipeline, and the two differ most in the low-PO₂ regime, which is exactly where §2.3 reads its answer | unswept |
| `use_endothelial_barrier_model` | True | — | — | **Implemented, unreachable.** The dispatch is `if use_multi_species_model: … elif use_endothelial_barrier_model: …`, and multi-species is also True by default, so the `elif` never fires. Setting this flag alone changes nothing | — |
| `use_multi_species_model` | True | — | — | Selects the O₂/CO₂/pH solver (Tier 3), the pipeline default. **The pipeline's `*_perfusion.vti` is therefore Tier 3, not the H2 field**: every H2 hypoxia number comes from Tier 1 (`solve_perfusion_steady_state`), called directly by `cb_h2_hypoxic_fraction.py` and `cb_h2_vtk.py` with `cb_settings` inputs. The `.vti` field data records the tier, the solver, `M_max`, the inlet/outlet pressures in mmHg, the Picard tolerance, whether the loop converged, its iterations and final residual (Tier 3 only), and a note saying so. Open items T, 23 | — |
| Glomus : stroma metabolic ratio | swept, not fixed | — | (iii) | **Nothing in this study measures it.** §2.3 reports the hypoxic fraction across a range of it rather than at one value | measured by sweep |

---

## §11 — Assumptions, with expected direction of bias

Every row is something the model takes to be true without establishing it here. The point of the
table is the **direction** column: when a result looks wrong, this is where to look for which way
the model would push it.

"Enters at" names the section where the assumption first does work. Where a direction is
*unquantified*, that is stated rather than guessed — an unmeasured bias is not a small one.

### 11.1 Geometry and network construction

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 1 | The imaged sub-volume represents the organ | §2.1 | Any regional gradient in vessel density is sampled rather than averaged. Tissue-centroid ROI placement removes the systematic part; residual unquantified |
| 2 | One segmentation threshold for all six specimens | §2.2 | **Deliberate.** Per-specimen thresholds would absorb the classifier's cohort bias into the mask, where nothing downstream could see it. A shared threshold leaves that bias visible as measurement error instead of converting it into a group difference |
| 3 | Skeletonisation preserves network topology | §2.4–§2.5 | Gap bridging adds a uniform foreground shell — the same wall EDT measures to — so calibre is biased outward |
| 4 | Vessel lumens are circular in cross-section | §3.1 | Non-circular lumens have higher resistance at equal area → resistance **underestimated** |
| 5 | EDT returns the vessel's inscribed radius | §2.6 | Near bifurcations it returns the junction's sphere instead, biasing calibre **upward**; suppressed by the 3.73 µm exclusion, which cannot be applied to segments shorter than it |
| 6 | The branch-order fallback diameter law is exponential; Murray's law is not used | §2.6 | Active on the fallback path only — and under the default EDT mode that path **refuses rather than fabricates**, so the law cannot silently activate |
| 7 | Terminal branches shorter than 5.6 µm are skeletonisation artefacts | §2.5 | Cannot affect β₁ (degree-1 nodes lie on no cycle). Shortens the per-edge length distribution by removing its lower tail |

### 11.2 Haemodynamics and rheology

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 8 | Rigid vessel walls; no compliance | §3.1 | Removes pressure-dependent flow redistribution; resistance is static |
| 9 | Steady state; no cardiac pulsatility | §3.4 | Removes cyclic wall-shear variation; mean flow largely unaffected |
| 10 | No-slip at the vessel wall | §3.1 | Standard; negligible |
| 11 | Plug flow; no radial intraluminal gradient | §6.6 | Transmural driving force slightly **overestimated** |
| 12 | Newtonian fluid at initialisation | §4.3 | Biases initial resistances; replaced by the per-pass Poiseuille recompute (§4.3 step 14) |
| 13 | Rheological correlations transferred from rat mesentery | §4.1–§4.2 | Transferability to carotid body microvasculature **unquantified** |
| 14 | Phase separation at higher-order divisions is a one-vs-rest extension of the Y-split relation | §4.2 | Not validated against data; it reduces to the Pries relation at two daughters. Until open item 37 these divisions mixed proportionally, which **underestimated** haematocrit heterogeneity |
| 15 | One arteriolar-to-venular pressure pair, 60→20 mmHg, is imposed on every specimen's sub-volume | §8 | Probably **overestimates** perfusion pressure: 60 mmHg is an upper value for the arterioles of other rat beds, 15–45 µm arterioles read nearer 30 (§8.1). SHR arteriolar pressures run higher than WKY [`jin_study_1997`], and the shared pair carries none of that difference. The pipeline config declared MAP-to-CVP, 100→2 mmHg, until open item 10; every published H2 number used 60/20 |
| 16 | No vasoregulation of any kind | §3.3 | Constriction is disabled entirely, so there is neither active feedback (myogenic, metabolic, shear-mediated) nor a static constriction geometry. The network is a fixed passive resistor array |

### 11.3 Blood gas chemistry

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 17 | Human haemoglobin parameters applied to rat tissue | §5.1–§5.2 | Species mismatch. Direction **not established** |
| 18 | Constant bicarbonate buffer; no renal compensation | §5.4 | Fixes the pH response to PCO₂ |
| 19 | Haldane saturation evaluated at fixed P₅₀ = 26 mmHg rather than the Bohr-shifted value | §5.3 | CO₂ carriage **underestimated** in acidotic tissue; at most 1.5% (0.4 mmol/L) over pH 7.2–7.4 and PCO₂ 40–60 mmHg |

### 11.4 Tissue transport

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 20 | Uniform tissue diffusivity | §6.3 | A single scalar σ for the whole domain. Smooths the PO₂ field across tissue types. **Note:** metabolic rate is *not* uniform — see row 22 |
| 21 | Metabolism is phenomenological, M(PO₂) = M_max·(1 − e^(−k·PO₂)) | §6.4 | Not Michaelis–Menten, which is the literature standard. The two differ most in the low-PO₂ regime — exactly where §2.3 reads its answer |
| 22 | The glomus-to-stroma metabolic ratio | §6.5 | **Nothing in this study measures it.** Reported across a swept range rather than at one value, so the hypoxic fraction is a curve in this parameter, not a number |
| 23 | Neumann tissue boundary; no exchange beyond the imaged volume | §6.3 | Tissue PO₂ **overestimated** near the domain boundary |
| 24 | Vessel-to-grid mapping is point-sampled along the centreline | §6.2 | An approximation to line–plane intersection, so *where* a vessel deposits carries discretisation error. The *total* is conserved: shares are normalised by accumulated length, so an edge's flow sums to exactly one across the cells it crosses. The centreline mapping ignores the vessel's width, and Tier 1 PO₂ **falls** with every refinement below a vessel diameter; the H2 drivers therefore map each vessel over a disc of its radius, which converges (open item 30). Either way the cells a vessel occupies keep the full tissue metabolic rate: lumen volume is not taken out of the sink, a small overcount of consumption |
| 25 | The grid spans the segmented volume, not the graph's extent | §6.1 | Tissue beyond the graph is represented; tissue beyond the segmentation is not represented at all |
| 26 | No lymphatic drainage or interstitial fluid flow | §6.3 | Omits a minor transport pathway |
| 27 | Tier 1 blood leaves each cell fully equilibrated with its tissue, at the cell's flow-weighted haematocrit | §6.6 | Instant equilibrium per cell (no wall resistance, no march along the vessel). Until open item 29 the washout used `systemic_hematocrit` (0.45), which made a phantom source of every cell fed above it and a sink of every cell fed below. Tier 1 only |

### 11.5 Study design and reporting

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 28 | The two channels are co-registered by construction | §7.2 | Two channels of one acquisition on an identical grid, with no registration step. This is what makes a join between them sound; it would not hold across separate acquisitions |
| 29 | Absolute flow quantities are reported only as within-specimen ratios | §7.6 | Not a modelling assumption but a reporting rule forced by two of them — the ±47% calibre floor and the unreconciled unit magnitude. See §13 |
| 30 | Boundary terminals are selected on axis 1 only | §8 | Chosen as the only axis with terminals on both faces in all six centred-box graphs; on the 2026-09-28 graphs all three axes qualify, so the choice now rests on its being pinned (§2.8, needs review). A specimen whose true inflow is off-axis is served by the wrong terminals |

### 11.6 Four rows that changed against the earlier understanding

Recorded because the old wording is still in circulation elsewhere.

| Row | Was | Is |
|---|---|---|
| 20 / 22 | "Homogeneous tissue; uniform diffusivity **and metabolic rate**" | Metabolic rate is heterogeneous — blended per cell from TH volume fraction and applied elementwise. Only diffusivity is uniform |
| 16 | "Static constriction ratios; no active vasoregulation" | Stronger: constriction is disabled outright, so there is no constriction geometry at all |
| 6 | "Exponential branch-order scaling; bounded by the fallback rate" | Under the default EDT mode the fallback refuses rather than fabricating, so the bound is zero, not small |
| 24 | "Point-sampled vessel-to-grid mapping — discretisation error in deposited exchange area" | Point sampling remains, but the conservation half is fixed: shares now normalise to one per edge |

### 11.7 The three that most constrain what can be claimed

Not the most numerous, the most consequential:

- **Row 22** — the glomus-to-stroma metabolic ratio is assumed. The hypoxic fraction cannot be quoted as a number, only as a function of it.
- **Row 15** — the full arterial-to-venous pressure drop is placed across a 1 mm block. Every absolute flow and perfusion figure inherits this.
- **Row 13** — the rheology is transferred from a different tissue, and the size of that error has never been measured.

---

## §12 — Verification status

**Verification asks whether the equations are solved correctly. Validation asks whether those
equations describe the carotid body.** This section covers the first. There is no second — see
§12.4.

The suite is **547 tests** across 55 files, run under continuous integration.

### 12.1 The six strategies

1. **Analytical closed-form comparison** — solver output against an independently derived exact
   solution.
2. **Conservation and invariant checks** — mass and flux balance asserted directly, with no target
   value needed.
3. **Synthetic phantoms with a prescribed answer** — volumes and graphs built so the correct result
   is known by construction.
4. **Equivalence oracles** — independent code paths checked for mutual agreement.
5. **Graceful degradation** — pathological inputs must fail safely rather than crash, hang, or
   return a plausible wrong number.
6. **Physical bounds** — extreme configurations checked against known limits.

### 12.2 Coverage

| Component | Oracle | Tolerance |
|---|---|---|
| Poiseuille resistance in series | Closed-form series reduction | 10⁻¹⁰ |
| Poiseuille resistance in parallel | Closed-form parallel reduction | 10⁻¹⁰ |
| Variable-diameter resistance (§3.3) | Term-by-term analytic integration | rtol 10⁻³ |
| Wall shear stress | Closed-form recomputation | 10⁻¹⁰ |
| 1D pure diffusion | Closed-form linear gradient | 10⁻¹⁰ |
| Zero-order metabolism | Exact parabolic profile `c₀ − (M/2σ)x(L−x)` | 10⁻¹⁰ |
| Tier 1 slab with solubility (open item 22) | `P₀ − M/(2Dα)·x(L−x)` through `build_adr_matrix` and the Newton solve; the balance and mass balance at the fixed point from their definitions (`test_perfusion_tier1_solubility.py`) | 10⁻³ mmHg |
| Radial point source | Qualitative 1/r decay | Bracketed |
| Krogh cylinder radial diffusion | Analytic radial profile | Bracketed |
| Plasma skimming | Erythrocyte mass conservation | 10⁻⁸ |
| Skimming direction | Inequality (larger branch takes more) | Qualitative |
| Fåhræus–Lindqvist curve | Curve shape, inequality chain | Qualitative |
| In vivo viscosity monotonicity | Rises as vessels narrow | Qualitative |
| Bohr and Haldane shifts | Direction only | Qualitative |
| McHardy CO₂ curve | Content, slope and Haldane size against McHardy and Loeppky (`test_blood_co2_content.py`) | 10⁻² |
| Henderson–Hasselbalch | Closed form at anchor points | 10⁻² |
| Multi-species 0D Fick balance | Coupled Fick + Henderson–Hasselbalch root | 10⁻² PO₂/PCO₂, 10⁻³ pH |
| **Flow unit conversion** | Independent SI computation on a single tube | Derivation, not a fit |
| **Grid-coupling conversion** | End-to-end through `map_vessels_to_grid` | Exact |
| **Length-fraction conservation** | Shares sum to one per edge | Exact |
| **Grid independence of the source** | Source unchanged under refinement, end to end | Exact |
| **Jacobi preconditioner** | Is the inverse diagonal; is SPD | Exact |
| **CG convergence** | On a production-like ill-conditioned system | Converges |
| **Non-positive diagonal** | Declines rather than forming a bad preconditioner | Behavioural |
| **Face boundary rule** | Interior terminal is never a boundary; empty face raises | Behavioural |
| **Two-faced node** | Assigned once, not to both | Exact |
| **Tissue volume fraction** | Occupied volume, not a centre sample | Exact |
| FWHM diameter | Analytical Gaussian phantom | 0.2–0.35 µm |
| EDT junction trimming | Synthetic junction fixture | Behavioural |
| Silent fallback guards | Refuses fabricated calibre | Behavioural |
| Transit time | $\tau = \pi r^{2} L / Q$ by hand; `inf` for zero flow | Exact |
| Cohort split | Exact permutation p against enumeration | Exact |
| Threshold selection | Refuses when calibre unreachable | Behavioural |

Rows in **bold** postdate the earlier coverage table and close the gaps it recorded.

### 12.3 What is verified weakly

Stated as fact, not softened:

- **Directionally only** — the apparent viscosity curve, the skimming output *value* (its mass
  conservation is exact; its magnitude is not checked against a target), and the Bohr shift. The
  CO₂ curve and its Haldane term are checked against McHardy and Loeppky, the Bohr shift is not.
- **Transitively only, through integration tests rather than directly** — the branch-order
  diameter formulae, the default boundary permeability mode, and the
  numerical Hill inversion.
- **Bracketed rather than to a tolerance** — the radial point source and the Krogh cylinder.

### 12.4 The two gaps

**No grid-convergence or order-of-accuracy study exists for any PDE solver.** Every result runs at a
single fixed resolution. §6.8 records that the Tier 1 field does not converge under the centreline
vessel mapping and does, to 0.5 mmHg by 3 µm, under the cross-section one (open item 30): the
mapping, not the stencil, set its behaviour below a vessel diameter. That says nothing about the
stencil's own order. The cheapest closing move is the zero-order metabolism case,
which already has an exact closed-form solution verified to 10⁻¹⁰ at one resolution: run it at 20,
10, 5 and 2.5 µm, plot L² error against spacing on log axes, and fit the slope. A slope near 2 would
demonstrate the expected second-order accuracy of the seven-point stencil.

**No validation exists.** Nothing in this pipeline has been compared against an experimental
measurement of carotid body perfusion or tissue oxygenation. Every claim in §13.10 marked
"supported" is supported *as a verified computation*, not as a validated physiological prediction.

---

## §13 — Error budget and known limits

**Check this section before quoting any number.** It is the one place that says how much the model
can carry. Other sections point here; none of them restate it.

These are properties of the model, not a list of things to fix. Most are bounded by voxel size
against vessel calibre, or by tissue geometry, and no change to the solver touches either.

### 13.1 The governing constraint: resistance goes as $d^{-4}$

Fractional calibre error propagates as $\delta R/R \approx 4\,\delta d/d$. Measured over the pooled 42,211 edges
of the 2026-09-28 batch (threshold 0.95, placed ROIs), taking one voxel (1.866 µm) as the diameter
uncertainty:

| Percentile | Diameter (µm) | $\delta d/d$ | $\delta R/R$ |
|---|---|---|---|
| p5 | 3.728 | 50.1% | **200.2%** |
| p25 | 5.275 | 35.4% | **141.5%** |
| p50 | 6.462 | 28.9% | **115.5%** |
| p75 | 8.343 | 22.4% | **89.5%** |
| p95 | 11.790 | 15.8% | **63.3%** |

- **98.9%** of edges carry more than 50% resistance uncertainty
- **55.4%** carry more than 100%
- The measured p5–p95 calibre spread of 3.16× becomes a **100× spread in resistance**

(On the 34,900 edges of the 0.90 centred-box batch: median 7.904 µm, 95.9% above 50%, 37.2% above
100%, a 3.74× calibre spread and 196× in resistance. The higher threshold gives thinner vessels, so
every edge's relative uncertainty is larger.)

The network's resistance structure is dominated by a quantity measured to roughly a quarter of its
own value. **This is not fixable in the solver** — not by the Picard iteration, the ADR
discretisation, or the rheology.

> **Not a quantisation problem.** The pooled diameters take 588 distinct values with a median gap
> of 0.0028 µm (823 and 0.0023 µm before the re-run) — junction trimming and B-spline smoothing break the raw EDT lattice, so the values
> are numerically dense. The 1.87 µm figure is the scale below which a difference is not
> *physically* resolved, not the spacing of the values. The problem is uncertainty, not
> discretisation.

### 13.2 Independent error averages down; correlated error does not

The segmentation threshold is the dominant correlated term: every edge in a specimen is measured
from one mask at one threshold, so moving it moves every diameter together.

Measured median calibre falls monotonically with threshold in **6 of 6 specimens**. Over the clean
0.93–0.95 interval below the frozen value the mean shift is **0.740 µm, about 0.40 voxel** — a
per-edge $\delta d/d$ of 11.0% and an analytic `δR/R` of 44.2% (`cb_h2_threshold_calibre.py`). The
0.95–0.97 interval moves 1.130 µm, but 0.97 is the fragmentation onset in four of six, which
contaminates it. (0.690 µm, δd/d 10.3%, δR/R 41.2% until 2026-09-30, while the 0.93 run seeded at
0.98 rather than the frozen 0.999; open item 41. Before the 2026-09-28 re-run: 0.922 µm over
0.85–0.90, δd/d 11.7%, δR/R 46.7%.)

Measured by re-solving the networks at that perturbation rather than scaling, since $d^{-4}$ is not
linear (`cb_h2_error_propagation.py`, 2026-09-30, package C of the 2026-09-29 re-run notes). The solve
now reads the batch networks (MultiGraph and `per_edge_morphometry.csv`; re-run at 0.740 µm on
2026-09-30, `rerun_2026-09-29_logs/F_07_error_propagation_0p740.log`) and places pressure with
`select_boundary_terminal_nodes_by_face` on `cb_settings.BOUNDARY_AXIS` at one voxel, the pipeline's
own call. It is still plain Poiseuille with uniform viscosity, not the Pries flow the H2 drivers use,
so it isolates calibre propagation rather than reproducing the H2 flow field:

| Perturbation | Independent | Correlated | Within-specimen ratio |
|---|---|---|---|
| One voxel, 1.866 µm (conservative bound) | 8.8% | 126.9% | 15.5% |
| **Measured threshold shift, 0.740 µm** | 4.2% | **47.0%** | **5.9%** |

(At the old 0.690 µm shift: 3.9 / 43.7 / 5.5%. Before package C the script used its own 25% band
on graph axis 0, 148–248 inlets per specimen instead of the face rule's 8–23: 3.9 / 125.4 / 13.3%
at one voxel and 1.6 / 43.4 / 4.7% at 0.690 µm. Before the 2026-09-28 re-run: 4.1 / 95.3 / 13.2%
at one voxel and 2.2 / 45.3 / 6.3% at 0.922 µm.) The correlated term barely moves with the boundary rule, and the measured 47.0% still
sits close to the 44.2% that $4\,\delta d/d$ predicts, so propagation is near-linear at this scale
even though the underlying law is not. The independent term roughly doubles, since a few face
inlets carry the whole throughput and averaging over them is weaker than over 150–250 band nodes.
**The ratio cancels 88% of the correlated error at both perturbation sizes** (89% under the band
rule), so the cancellation is a property of the ratio rather than of the size or the boundaries
chosen. Per specimen at 0.740 µm: correlated 40.9–52.1%, ratio 4.7–7.5%, the three SHR above the
three WKY (6.4–7.5% against 4.7–4.9%).

### 13.3 The two noise floors

| Quantity | Floor | Against H1's measured group-mean ratios at 0.95 |
|---|---|---|
| Absolute network flow | **±47%** | Cannot resolve them |
| Within-specimen ratio | **±5.9%** | See below — **needs review** |

**This is why §7.6 reports ratios and never absolutes.** It is not caution; it is the difference
between an answerable question and an unanswerable one.

> ⚠ **Needs review after the 2026-09-28 re-run.** This table used to read "against H1's measured
> 27–40% effects … can resolve them, roughly fourfold margin". Those effects were measured at 0.90
> on the centred boxes. At 0.95 on the placed boxes the SHR/WKY group-mean ratios are β₁ density
> 1.087, junction density 1.064, length density 1.022, tortuosity 1.001 (median EDT diameter 0.869),
> and the groups overlap on every density. So the effects are 2–9%, against a within-specimen floor
> of 5.9% (5.5% before the sensitivity runs held the seed, open item 41; 4.7% before the floor moved
> onto the face-rule boundaries, package C): not a fourfold margin, and only β₁ (8.7%) and junction
> density (6.4%) clear it at all, junction density by half a point. (The floor is for flow-derived ratios;
> β₁ and the densities are topological counts, which calibre does not move, §13.10.) The H1
> whitepaper, re-quoted 2026-09-30 (package N), restates the claim on these numbers (§7.2, §12):
> β₁ and junction density "Provisional (weak)", length density "Not supported".

**One residual, in the ratio itself.** The per-specimen shift is uneven — 0.380 µm (WKY-B) to
1.093 µm (SHR-C) — so the correlated error is not identical across specimens and does not cancel
perfectly in a *between-group* comparison. Group means differ: 0.585 µm for WKY against 0.895 µm
for SHR (0.795 while the 0.93 run seeded at 0.98; 1.075 against 0.768 before the 2026-09-28
re-run, the other way round). That is the right shape to
become a confound. With n = 3 it is noted, not established.

### 13.4 Boundary selection is the largest single lever

Larger than calibre error. The face-crossing rule on axis 1 holds residual boundary sensitivity to
**8.9%**, against **73.9%** for the alternative band rule, and cuts total sensitivity from 113.5%
to 40.0% (`cb_h2_boundary_selection.py` on the 2026-09-28 batch graphs, package O; 113.7% / 44.9%
from the ParaView export before, whose axis-0 and axis-2 faces were misplaced (§2.8); 13.3% / 75.8%
and 118.8% / 43.1% on the centred boxes). On the centred boxes axis 1 was the only axis with terminals on both
faces in all six specimens; on the new networks all three axes are (§2.8).

**Measured in the error-propagation frame too** (`cb_h2_error_propagation.py` S20, 2026-09-30, face
rule throughout, plain Poiseuille). With the axis pinned at 1, moving the face tolerance over 1/2/4
voxels spreads the shunt ratio by **3.6%** on average (1.6–6.4% per specimen); moving the axis at one
voxel spreads it by 14.0%. The residual 3.6% is below the 5.9% calibre ratio floor, whereas the 8.9%
above is above it. The two measure different ratios: S20 divides shunt flow by the flow summed over
every edge (as S13 does), and `cb_h2_boundary_selection.py` divides by inlet throughput. So which
term is larger at the pinned axis depends on the ratio's denominator; the axis choice dominates both.

Below the operative floor of §13.3 and below the effects H1 measures — but only because the rule
and its axis are pinned. They are now pinned in one place: `cb_settings` owns both, and the main
pipeline and the H2 drivers read them (open item 2, closed).

### 13.5 Absolute perfusion at 60/20 mmHg is at or above physiological

Measured across all six with the face rule at 60/20 mmHg (`examples/cb_h2_absolute_perfusion.py`,
re-derived 2026-09-27 for open item 12, re-run 2026-09-28 on the placed-box networks and
2026-09-30 on the converged flow–haematocrit loop of open item 37):

| Specimen | Inlets | Total inlet flow (µm³/s) | Flow-weighted velocity | ΔP for 500 µm/s |
|---|---|---|---|---|
| WKY-A | 23 | 4.00e6 | 1,808 µm/s | 11.1 mmHg |
| WKY-B | 8 | 1.51e6 | 1,095 µm/s | 18.3 mmHg |
| WKY-C | 17 | 2.02e6 | 1,156 µm/s | 17.3 mmHg |
| SHR-A | 14 | 1.33e6 | 955 µm/s | 20.9 mmHg |
| SHR-B | 23 | 1.63e6 | 1,103 µm/s | 18.1 mmHg |
| SHR-C | 14 | 1.66e6 | 1,452 µm/s | 13.8 mmHg |

**955–1,808 µm/s against a physiological 200–1,000 µm/s.** Five of six sit above the upper end, by
up to 1.8×. A 500 µm/s flow-weighted velocity needs a drop of 11–21 mmHg, not the 40 used. The
converged loop moved the velocities by −2.2% to −0.1% (SHR-A most). The
solve is linear in the pressure drop (phase separation reads flow fractions, which a uniform scaling
leaves alone), so that is the frozen drop scaled by 500 / v; `tests/test_cb_h2_absolute_perfusion.py`
checks it against a second solve. On the centred boxes at 0.90 the same script gave 979–3,319 µm/s
and 6–20 mmHg.

**This replaces "20–100× below physiological".** The earlier table (4.1–9.7 µm/s, and "about 3,257
mmHg" for 500 µm/s) was computed before `7ea1b36`, while the rheology loop inflated every resistance
208–539× (§3.2, closed item 12). The new script, run with the pre-fix rheology code patched in,
gives back that table to the digit (WKY-A: 18 inlets, 8,924 µm³/s, 6.17 µm/s). So the whole change
is the rheology code: velocity rose 240–343× per specimen, the factor §3.2 predicted. The later
rheology fixes (#98: feeding-vessel phase separation, under-relaxation, rescale once) move it a
further −10% to +2%.

**The boundary rule still sets the throughput.** The band rule (25% of axis 1 at each end) the face
rule replaced carries 2.2–4.3× the inlet flow (4.5e6–1.2e7 µm³/s) and 1.4–2.0× the velocity
(1,651–3,117 µm/s); on the centred boxes it was 4.0–7.6× and 1.6–3.7×. The face rule is kept for
ratio stability (§2.8, §13.4).

**The rheology solve converges.** All 12 solves (face and band rule) stop on the scale-free test,
in 156–457 passes (open item 37); the table records the stop reason per solve. Until 2026-09-30
they stopped on a 15-pass cap with the last pass still changing flow by 0.05–1.7% of the largest
edge flow.

Absolute perfusion is still not a reportable quantity: it sits under the ±47% calibre floor
(§13.3), it moves 2–4× with the boundary rule, and 60/20 mmHg is an upper-end arteriolar/venular
pair, not a measured carotid-body drop (§8.1). But the model is no longer orders of magnitude off.
At 60/20 mmHg it runs somewhat fast, which points at the pressure pair rather than at the network's
resistance.

### 13.6 The tissue is not diffusion-limited

> **Current measurement (re-run 2026-09-30, open items 37, 38, 40).** On the placed-box networks
> at 0.95, with vessels mapped over their cross-section at 3 µm, four times stromal glomus
> metabolism moves PO₂ in TH from 95.18 to 93.77 mmHg on WKY-A and from 88.54 to 83.25 on SHR-C,
> and TH hypoxia is 0 below 5, 10 and 20 mmHg at every contrast. So the mechanism responds (it did
> not before open items 22 and 29, when the measured consequence below was obtained), but not
> enough to make glomus tissue hypoxic. The glomus-to-centreline distance (H1 §1.5) is 7.9–9.7 µm;
> the tissue-to-vessel-surface distance below is 4.6–6.2 µm. The conclusion of this section holds
> at the frozen 60/20 mmHg pair, which runs the network fast (§13.5); a calibrated, lower pressure
> drop has not been tested. §7.5 has the numbers.

**The most consequential limit in this document**, because it constrains the mechanism rather than
the precision.

The oxygen diffusion length is

$$\sqrt{\frac{D\,\alpha\,P_{\mathrm{O_2}}}{M}}
\;=\; 20\ \mathrm{\mu m}\ \text{at}\ P_{\mathrm{O_2}} = 10,
\quad 35\ \mathrm{\mu m}\ \text{at}\ 30,
\quad 45\ \mathrm{\mu m}\ \text{at}\ 50$$

against a **median tissue-to-vessel distance of 4.6–6.2 µm** (5.3–7.9 µm on the WKY centred boxes).
Every tissue point sits at roughly a fifth of its supply radius or less, so the tissue is not diffusion-limited and a local sink cannot produce
a local gradient.

Measured consequence, before open items 22 and 29: raising the glomus metabolic rate to **four
times** the stromal one moved PO₂ within the TH-positive volume by **0.01 mmHg**. Now it moves it by
0.8–5.3 mmHg, and no glomus tissue falls below 20 mmHg.

So the premise of §7.5 — that a higher glomus metabolic rate produces glomus-specific hypoxia —
cannot operate on this geometry at these parameters. **This is a statement about the tissue, not
about the code.** A glomus-specific hypoxic fraction requires the sensors to be diffusion-limited,
and in a bed this dense they are not.

The consumption rate is not the problem: `M_max = 0.05` mmol/L/s is 0.067 mL O₂ per mL per minute,
against roughly 0.040 for brain — the right order for a metabolically active organ.

### 13.7 Grid resolution against the gradient that matters

Measured tissue-to-vessel distance at native resolution — the Euclidean distance from every
non-vessel voxel of the ROI to the nearest vessel voxel of the pipeline mask (2026-09-28 batch,
threshold 0.95):

| Specimen | Foreground | TVD p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| WKY-A | 28.9% | 4.57 µm | 12.08 | 36.80 | 73.27 |
| WKY-B | 21.6% | 5.60 µm | 17.30 | 56.99 | 96.91 |
| WKY-C | 21.4% | 6.19 µm | 34.43 | 74.15 | 110.12 |
| SHR-A | 24.2% | 5.60 µm | 46.83 | 113.10 | 164.52 |
| SHR-B | 21.7% | 5.89 µm | 25.45 | 80.62 | 128.05 |
| SHR-C | 20.1% | 5.90 µm | 31.67 | 81.49 | 123.85 |

(The WKY centred boxes at 0.90 gave p50 5.28–7.92 µm and p90 25.9–53.1 µm.) At the 3 µm grid the
median tissue voxel sits **1.5–2.1 cells** from a vessel. The gradient that decides whether tissue
is hypoxic is therefore spanned by one or two cells for half the tissue — resolved, but barely.
Only the p90 tail, 12.1–46.8 µm, spans a comfortable number of cells.

Refining was tested (§6.8, open item 30). With each vessel mapped to the cells its centreline
crosses, a finer grid drew a thinner vessel and PO₂ kept falling, about 1.5–1.9 mmHg per halving
down to 2 µm. With vessels mapped over their cross-section the field settles: every measure moves
at most 0.155 mmHg from 3 to 2 µm. (Both sweeps were first run on the centred-box networks, where
the 3 → 2 µm step was at most 0.21 mmHg. They were repeated on the placed boxes on 2026-09-30 and
again in the 2026-10-01 re-run; the figure here is from the latest run, §6.8.) The short gradient is still the physical limit. The Newton
stop at 10⁻⁵ adds at most 0.10 mmHg per cell on WKY-A, 0.009 mmHg in the medians (§6.7, open item 6).

### 13.8 Calibre is not a reportable H1 finding

> ⚠ **Needs review after the 2026-09-28 re-run.** The paragraph below predates even the 0.90
> centred-box outputs: those gave WKY 8.35, 8.34, 7.91 µm against SHR 7.46, 7.80, 7.46 µm, a gap of
> 0.62 µm with no overlap (ratio 0.924). At 0.95 on the placed boxes the per-specimen median EDT diameters are WKY 7.46,
> 7.08, 6.96 µm and SHR 6.46, 6.37, 5.87 µm: group means 7.17 and 6.23, a gap of **0.94 µm, half a
> voxel**, and the groups **do not overlap** (smallest WKY 6.96 against largest SHR 6.46). Within-group
> spread is 0.50 µm (WKY) and 0.59 µm (SHR), smaller than the gap. The gap is still below one
> voxel, and the threshold alone moves calibre 0.740 µm (§13.2), with a larger shift in SHR than in
> WKY (§13.3). The H1 whitepaper §8.2 (re-quoted 2026-09-30, package N) re-argues it and keeps
> calibre not reportable: the gap is below one voxel, the groups touch at 0.93 and 0.97, and both
> the group-correlated threshold shift and the EDT bias (open item 42) are larger than the gap.

The between-group calibre gap sat at **one twentieth of the smallest resolvable difference**.
Within-group spread was 0.45 µm (WKY) and 0.34 µm (SHR) — three to four times the gap itself.

A separation that small is a coincidence of where six medians happened to fall. Any claim that SHR
capillaries are narrower was **not supported** on those numbers.

### 13.9 The TH classifier carries a residual cohort skew

The positive class holds **24,935 labels in WKY against 11,673 in SHR — a 2.1× skew**, above the 2×
reporting threshold.

Bounded, not eliminated. The contrast was evaluated against three classifiers spanning that skew
from 22.9× down to 2.1×, and pooled class balance from 1:59 to 1:0.8. **The ratios move by at most
0.01.** So a further reduction would be expected to change little — but the argument is a
sensitivity analysis, not a proof, and it remains the stated bound on any TH-channel contrast.

### 13.10 What the budget permits

| Claim type | Supported? |
|---|---|
| Within-specimen ratios of flow-derived quantities | **Yes** — ±5.9% floor. *Needs review:* the 27–40% H1 effects this was set against are 2–9% at 0.95 (§13.3); H1 whitepaper §7.2 re-quoted on them |
| Absolute flow, velocity or perfusion | **No** — ±47% floor, 2–4× with the boundary rule, and up to 1.8× above physiological at 60/20 mmHg (§13.5) |
| Between-group calibre differences | **No** — the stated reason was a gap of 1/20 of the measurement step. *Needs review:* at 0.95 the gap is half a voxel and the groups do not overlap (§13.8); H1 whitepaper §8.2 keeps it not reportable on other grounds |
| Glomus-specific hypoxic fraction as a number | **No** — the mechanism cannot operate (§13.6); report as a curve in the assumed metabolic contrast. *Needs review:* TH hypoxia is now 0 everywhere, while PO₂ in TH separates the groups (§7.5) |
| Between-group TH-channel contrasts | **Qualified** — bounded by §13.9's sensitivity analysis, not by proof |
| Topological counts (β₁) | **Yes** — unaffected by calibre, and stub pruning cannot move it |

---

## §14 — Provenance and reproducibility

### 14.1 What this document describes

Branch `cb_pipeline_improvements_sweep`, commit `8a2b81c`. Read from the source, not from prior
documentation.

### 14.2 Artefact provenance

**The classifier's identity travels with its output.** Probability maps carry a
`.provenance.json` sidecar recording which classifier produced them, plus a label summary — counts
and boundary placement — so a result can be attributed to a *decision* ("the round before boundary
labelling") rather than to an opaque hash.

Four states, and the distinction between the middle two is the point:

| Status | Meaning |
|---|---|
| `absent` | No artefact |
| `unknown` | Artefact present, no sidecar. **The worse of the two failure states** — nothing can be ruled out about it |
| `stale` | Origin known, and wrong |
| `current` | Origin known, and right |

Treating a missing sidecar as current is precisely the assumption the module exists to refuse. The
sidecar is written after the fact, because prediction is a headless Ilastik invocation this codebase
does not drive — `record_probability_provenance` makes it a deliberate step rather than an
assumption.

**Other quantities carry their own provenance**: diameters carry `diameter_provenance`, centrelines
carry `centreline_smoothing`, radii carry `edt_junction_trim`.

### 14.3 Changes made for tractability, and what they can and cannot affect

These changes were made **after** every H1 and H2 number in this document was produced. None
alters an equation, a parameter value, or a result. They are recorded here because a
repository that has moved on from the one that produced the published figures has to say so.

| # | Change | Can it move a published number? |
|---|---|---|
| 1 | `src/ImageLynx/cb_settings.py` owns the analysis constants; every `cb_*` driver imports them | **No.** Same values, one definition. `tests/test_cb_settings.py` pins them |
| 2 | Missing or degenerate data raises instead of being substituted | **No** for any run that completed. A run that silently used a 5.0 µm diameter, a least-squares pressure field, or the extreme-decile boundary fallback would now fail loudly instead. The boundary fallback was removed later than the other two — `2d98ab8` did not touch `boundaries.py` — and none of the six H1 logs shows it firing |
| 3 | `SkeletonConfig.bridge_gap_size` removed | **No.** It applied a second radius-1 closing, and closing is idempotent, so it could not act |
| 4 | Core dead-end resolution and the constriction branch removed from the CB driver; bundle collapse guarded out | **No.** All three were already inert: mode `"none"`, a validator that raises, and a density threshold no window can reach |

**What change 2 found.** Three integration tests of `resistance_network_pipeline.py` — a
different pipeline, not the carotid body path — had been passing on least-squares answers to
singular systems. Their synthetic fixture skeletonises into 54 components with the largest
holding 53% of voxels, so components with no pressure boundary reached the solver and were
answered with a minimum-norm field over a network nothing was driving. The solve now excludes
those components, reports how many nodes it dropped, and leaves their pressure as NaN.
Effective resistance between two nodes in different components returns infinity, which is the
physical answer, rather than a finite least-squares value.

**No CB result depended on any of those paths**, which is why no figure in this document moves.

### 14.4 Which script produces what

| Script | Produces |
|---|---|
| `preprocessing/preprocess_cb.py` | Ilastik input volumes from raw acquisition |
| `preprocessing/prob_to_mask.py` | Binary mask and EDT from the probability field |
| `carotid_image_to_model.py` | The general image-to-model pipeline: mask → skeleton → graph → flow |
| `cb_h1_batch.py` | The six-specimen H1 cohort run, including threshold selection |
| `cb_h1_th_metrics.py` | H1 §1.3 and §1.5 — glomus volume, length density, tissue-to-vessel distance |
| `cb_h1_figures.py`, `cb_h1_renders.py`, `cb_h1_vtk.py` | H1 figures and ParaView artefacts |
| `cb_h2_boundary_selection.py` | The boundary rule comparison behind §13.4 |
| `cb_h2_threshold_calibre.py` | The correlated-error size behind §13.2 |
| `cb_h2_error_propagation.py` | The independent/correlated/ratio floors behind §13.2–13.3 (face-rule boundaries, plain Poiseuille), the §8.2 terminal census and the S20 spreads in §13.4 |
| `cb_h2_glomus_perfusion.py` | §7.3 shunting, §7.4 haematocrit, §7.6 transit time |
| `cb_h2_hypoxic_fraction.py` | §7.5 hypoxic fraction on the heterogeneous grid |
| `cb_h2_vtk.py` | H2 ParaView artefacts |

### 14.5 Randomness

The pipeline is deterministic except for hyperparameter search. Optuna's TPE sampler takes an
explicit seed, defaulting to a fixed value, and **the seed is written into the tuning provenance
record** rather than only into a log line — a tuned parameter set whose search trajectory cannot be
reproduced is not a reproducible parameter set.

Pericyte constriction draws a random cohort per run (§10.6). It is frozen, so nothing on the CB path
is stochastic.

### 14.6 Reproducing a result

1. Preprocess to Ilastik input with the recorded parameters — identical for all six volumes.
2. Predict headlessly with the classifier named in the sidecar.
3. Threshold to a mask at the frozen 0.95 / 0.999 band (`p ≥` both), the same for all six (§2.2, `cb_settings`).
4. Run the H1 batch to produce graphs and per-edge morphometry.
5. Run the H2 driver for the method in question, at 60/20 mmHg on axis 1 (`cb_settings`).

**One invariant.** One classifier for all six volumes, never one per cohort, and identical
parameters everywhere. Violating either invalidates the cohort comparison, because a per-cohort
difference in the instrument becomes indistinguishable from a difference in the tissue (§7.7).

---

## Appendix A — Solver settings

Purely numerical. Nothing here is a model parameter; changing these should change runtime and
convergence, not the answer. Where a setting *does* move the answer, that is a finding, not a
tuning opportunity.

### A.1 Rheology — coupled flow / haematocrit / viscosity

| Setting | Value | Where | Notes |
|---|---|---|---|
| Picard max iterations | 1000 | `cb_settings.RHEOLOGY_MAX_ITERATIONS` → `HaemodynamicsConfig.rheology_max_iterations` | Was 15 until open item 37 |
| Haematocrit relaxation | 0.2 | `cb_settings.RHEOLOGY_RELAXATION` → `rheology_relaxation` | |
| Flow tolerance | 1 × 10⁻⁶ | `cb_settings.RHEOLOGY_FLOW_RTOL` → `rheology_flow_rtol` | Relative to the largest flow; was an absolute 10⁻⁴ |
| Haematocrit tolerance | 1 × 10⁻⁴ | `cb_settings.RHEOLOGY_HEMATOCRIT_ATOL` → `rheology_hematocrit_atol` | Max \|H_skim − H\| on flowing edges |
| Stagnant cut | 10⁻¹⁰ × max \|Q\| | `resistance.STAGNANT_FLOW_FRACTION` | Shared with Tier 3 |

### A.2 Network flow solve

| Setting | Value | Where | Notes |
|---|---|---|---|
| Direct/iterative dispatch threshold | 50 000 | `resistance.py` `_solve_system_smart` | Nodes. Below this, direct solve |

### A.3 Tissue transport — steady-state ADR (Tier 1)

| Setting | Value | Where | Notes |
|---|---|---|---|
| Newton max iterations | 200 | `picard_max_iterations` (`PerfusionConfig`; `PerfusionSettings` for H2) | Hard-coded at 50 until open item 6; 50 → 200 under open item 32 for Tier 3. Tier 1 stops after 9–11 steps on every §2.3 solve, so no H2 number moved |
| Tolerance | 1 × 10⁻⁵ | `picard_tolerance` (same) | Max cell residual in mmHg / max PO₂ (`_relative_residual`), as Tier 3; was the relative L2 change until open item 22, and hard-coded until open item 6. On WKY-A the field is within 0.10 mmHg per cell, 0.009 mmHg in the medians and 0.021 percentage points in the hypoxic fractions of a 10⁻⁸ solve; 10⁻⁴ returns the same field (§6.7) |
| Line search | halve until ‖F‖ falls, Armijo 10⁻⁴, floor 10⁻⁴ | hard-coded | Replaced the Picard relaxation γ = 0.5 (open item 22) |
| CG relative tolerance | 1 × 10⁻¹⁰ | hard-coded | Per Newton step; was 1 × 10⁻⁶ |
| CG max iterations | 20 000 | hard-coded | Was 1 000 |
| Preconditioner | Jacobi (diagonal) | `_jacobi_preconditioner` | Guarded against non-positive diagonals |
| Singularity handling | diagonal regularisation | `build_adr_matrix` | Pure-diffusion rows sum to zero under Neumann boundaries |

### A.4 Tissue transport — multi-species O₂ / CO₂ / pH (Tier 3)

| Setting | Value | Where | Notes |
|---|---|---|---|
| Diagonal (linearisation) | blood response k·C′/(C′ + k/*q*), plus metabolic slope for O₂ | `_blood_response_conductance` | Replaced the pseudo-washout γ = 1 (full P·A·α), which made the loop crawl at low flow (open item 23, §6.7) |
| Tissue linear solve | sparse LU, `MMD_AT_PLUS_A` ordering, every iteration | `splu` | Exact. Replaced CG at `rtol` 1 × 10⁻⁵, warm-started, `maxiter` 500, which returned the last field once a step was below its tolerance (open item 23) |
| Anderson depth | 5 | `ANDERSON_DEPTH` | History dropped whenever the residual rises |
| Convergence test | max cell residual in mmHg / max \|P\| < `picard_tolerance`, both gases | `_relative_residual` | Was the relative L2 change between iterates |
| Stagnant-flow floor | 10⁻¹⁰ × max \|q\| | `STAGNANT_FLOW_FRACTION` | Edges at or below it carry no blood in the march. The flow solve balances every node to ≈3 × 10⁻¹⁴ of the largest flow (WKY-A), so below this is rounding; in stagnant pockets it left nodes sending blood they never received. Was 10⁻¹² until open item 36: on the six 2026-09-28 networks rounding flows reach 1.7 × 10⁻¹² and real flows start at 2.1 × 10⁻⁸. The rheology loop and Tier 3 take the set from one helper, `stagnant_edges` (`resistance.py`), exact zeros included: 745–895 edges per specimen on the 2026-09-29 batch, of which 605–731 are exactly zero. Tier 3's log once counted only the nonzero ones (136–212), which read as a disagreement between the two; it was not one |

### A.5 Config-level Picard settings

| Setting | Value | Where | Notes |
|---|---|---|---|
| `picard_max_iterations` | 200 | `PerfusionConfig`, `cb_settings.PerfusionSettings` | Read by all three tiers: Newton steps in Tier 1, Picard passes in Tiers 2 and 3. Was 50 until open item 32: at the face-rule flows the pipeline's Tier 3 needs 40–80 passes. `test_cb_settings.py` pins the two to agree and both YAMLs to carry it |
| `picard_tolerance` | 1 × 10⁻⁵ | `PerfusionConfig`, `cb_settings.PerfusionSettings` | Read by all three tiers. Tiers 1 and 3 test the relative residual (open items 22, 23), Tier 2 the relative L2 step. Was 1 × 10⁻⁴ in `PerfusionConfig` while Tier 1 hard-coded 1 × 10⁻⁵ and Tier 2 1 × 10⁻⁴ (open item 6). `test_cb_settings.py` pins the two to agree and both YAMLs to carry it (open item 34), and every tier's `.vti` records it. Tier 3 on WKY-A: 10⁻⁴ → 10⁻⁵ moves PO₂ up to 0.29 mmHg, one more iteration (§6.7) |

Tier 2's inner CG (`rtol` 1 × 10⁻⁶, `maxiter` 1 000) stays hard-coded, as Tier 1's does: it sets the
direction, not the stop. Tier 2 now warns when it hits the cap.

---

## Appendix B — Symbol table

Five symbols are triple- or double-booked across the source material. The resolutions below are
binding for this document.

| Symbol | Meaning | Units |
|---|---|---|
| *d* | Vessel diameter | µm |
| *r* | Vessel radius | µm |
| *L* | Segment centreline length | µm |
| *R* | Hydraulic resistance | mmHg·cP·µm⁻³ (mixed; see §3.7) |
| *G* | Hydraulic conductance, 1/*R* | — |
| **L** | Graph Laplacian | — |
| *p* | Nodal pressure | mmHg |
| *Q* | Volumetric flow | µm³/s after conversion; solver units before |
| *μ* | Apparent viscosity | cP |
| *μ₄₅* | Relative apparent viscosity at *H* = 0.45 | dimensionless |
| *H* | Discharge haematocrit | fraction |
| *f_Q* | Bulk flow fraction into a branch | fraction |
| *f_E* | Erythrocyte flux fraction into a branch | fraction |
| *α* | Asymmetry parameter in the skimming logit | — |
| *β* | Steepness parameter in the skimming logit | — |
| *x₀* | Skimming threshold, 0.964(1 − *H_in*)/*D_F* | fraction |
| *D_F* | Feeding (parent) diameter at a diverging junction | µm |
| *C* | Blood gas content | mmol/L |
| *α_O₂*, *α_CO₂* | Gas solubility in plasma | mmol/L/mmHg |
| *S* | Haemoglobin oxygen saturation | fraction |
| *n_H* | Hill coefficient | 2.7 |
| *P₅₀* | Half-saturation partial pressure | mmHg |
| *σ* | Tissue diffusivity | m²/s in config, µm²/s internally |
| *D_x*, *D_y*, *D_z* | Diffusive conductance across a cell face | µm³/s |
| *M* | Metabolic consumption rate | mmol/L/s |
| *M_max* | Maximum consumption rate | mmol/L/s |
| *k* | Metabolic reduction constant | per mmol |
| *γ* | Picard relaxation / pseudo-washout slope | — |
| *V_cell* | Grid cell volume | µm³ |
| *b* | Branch order (hop count from an inlet) | integer, `B01`… |
| *β₁* | First Betti number, fundamental loop count | integer |
| *c* | Glomus-to-stroma metabolic contrast | multiple |
| *f_TH* | TH mask volume fraction per cell | fraction |
| *τ* | Transit time | solver units; report as a ratio only |

**Deliberately distinguished:** *G* (conductance) from **L** (Laplacian); *α* and *β* (skimming)
from *α_O₂* (solubility); *n_H* (Hill) from *b* (branch order); *L* (length) from **L**
(Laplacian); *C* (gas content) never used for conductance.

---

## Appendix C — Model to code to test

| Model | Code | Test |
|---|---|---|
| ROI placement | `roi_placement.py:96` | `test_roi_placement.py` |
| Threshold selection | `threshold_selection.py:256` | `test_threshold_selection.py` |
| Joint hysteresis mask | `image.py:179` | `test_preprocessing.py`, `test_new_preprocessing.py` |
| Skeletonisation | `skeleton.py:472` | `test_graph.py`, `test_length_measurements.py` |
| Graph construction | `build.py:22` | `test_graph.py` |
| EDT calibre | `automated.py:1238` | `test_edt_diameter.py` |
| FWHM calibre | `automated.py:971` | `test_haemodynamics_automated_fwhm.py`, `test_integration_synthetic_vessel_fwhm.py` |
| Calibre provenance guard | `poiseuille.py:16` | `test_silent_fallback_guards.py` |
| Branch order | `branch_order.py:95` | `test_branch_order_hierarchy.py` |
| Face boundary rule | `boundaries.py:106` | `test_boundary_faces.py` |
| Poiseuille resistance | `poiseuille.py:160` | `test_haemodynamics_analytical.py` |
| Variable-diameter resistance | `poiseuille.py:146` | `test_haemodynamics_analytical.py` |
| Network Laplacian solve | `resistance.py:46`, `resistance.py:138` | `test_haemodynamics_analytical.py` |
| Flow unit conversion | `resistance.py:37` | `test_flow_units.py`, `test_physical_units.py` |
| Pries–Secomb viscosity | `rheology.py:30` | `test_rheology_laws.py` |
| Phase separation | `rheology.py:90` | `test_haemodynamics_analytical.py` |
| Coupled flow–haematocrit | `rheology.py` `solve_coupled_flow_and_hematocrit` | `test_haemodynamics_rheology_integration.py`, `test_rheology_convergence.py` |
| Blood oxygen content | `perfusion.py:13` | `test_haemodynamics_analytical.py` |
| Blood CO₂ content | `perfusion.py:37` | `test_haemodynamics_analytical.py` |
| Tissue pH | `perfusion.py:65` | `test_haemodynamics_analytical.py` |
| Perfusion grid | `perfusion.py:82` | `test_haemodynamics_perfusion.py`, `test_fractional_grid.py` |
| Vessel-to-grid mapping | `perfusion.py:203` | `test_flow_conservation.py` |
| ADR assembly | `perfusion.py:349` | `test_haemodynamics_perfusion.py` |
| Jacobi preconditioner | `perfusion.py:317` | `test_perfusion_preconditioner.py` |
| Tier 1 steady state | `perfusion.py:453` | `test_haemodynamics_analytical.py` |
| Tier 3 multi-species | `perfusion.py:537` | `test_haemodynamics_analytical.py` |
| Heterogeneous metabolism | `tissue_regions.py:40`, `tissue_regions.py:155` | `test_tissue_regions.py` |
| Morphometry | `stats.py:147`, `stats.py:349` | `test_statistics.py`, `test_synthetic_network_statistics.py` |
| Two-channel morphometry | `th_morphometry.py:34`, `th_morphometry.py:78` | `test_th_morphometry.py` |
| Transit time | `transit.py:28`, `transit.py:57` | `test_transit.py` |
| Cohort split | `cohort_split.py:56` | `test_cohort_split.py` |
| Artefact provenance | `artefact_provenance.py` | `test_artefact_provenance.py` |
---

## Open items

| # | Item | Blocks |
|---|---|---|
| ~~1~~ | **Closed.** The hysteresis band differed between `PreprocessingConfig` (0.65 / 0.75) and the frozen 0.90 / 0.95 every H1 run used (`--hysteresis-low 0.90`, seed auto-raised). The config now reads `cb_settings.HYSTERESIS_LOW` / `HYSTERESIS_HIGH`, and `test_cb_settings.py` keeps them equal, so no published number moves. A `--hysteresis-low` given alone now always takes low + 0.05 as its seed (it used to keep the config's high whenever that was still above low), and a seed at or below low raises. The mask step's hidden 0.2 / 0.4 fallbacks are gone. The tuner's upper bounds rose 0.85 / 0.95 → 0.95 / 0.99 so the default stays inside its search range. Fresh WKY-A run with no flag: mask, skeleton, graph and `per_edge_morphometry.csv` identical to the batch output. `test_cb_settings.py`, `test_pipeline_hysteresis_band.py` | — |
| ~~2~~ | **Closed.** Two boundary rules coexisted: the pipeline ran the band rule on axis 0 at 25%, the H2 drivers the face rule on axis 1. The pipeline now calls `select_boundary_terminal_nodes_by_face` with `GraphConfig.boundary_axis` / `face_tolerance_voxels`, both read from `cb_settings`; the band fields are gone from `GraphConfig`, and the band rule stays in the library (raising on an empty band) for the nerve pipeline. Band 216–399 pressure nodes per specimen → face 16–37. Fresh WKY-A (scratchpad; batch outputs untouched): the pipeline's 18 inlets and 9 outlets are exactly the H2 face-rule sets on the same graph; mask, graph and `per_edge_morphometry.csv` identical except `branch_order` (4312 of 4512 edges). Against item 31's band-rule run: median edge flow 8.0e4 → 9.2e3 µm³/s; rheology still stops at 15 iterations (max flow change 1.1e3 µm³/s); Tier 3 now hits its 50-iteration cap unconverged (residual 5.9e-5 against 1e-5; was 22 iterations), with PO₂ mean 70.14 → 49.66, median 85.56 → 58.84 and cells < 10 mmHg 6.30 → 13.21% on that unconverged field (open item 32). A new guard (`_check_graph_fits_frame`) raises when the shape passed for boundary selection does not hold the graph, which stops the item 28 cache path. No published number moves; the batch outputs keep band-rule `branch_order` and flows until the item 27 re-run. `test_pipeline_boundary_rule.py`, `test_cb_settings.py` | §2.8, §7.8, §10.4, §13.4 |
| ~~3~~ | **Closed.** Arterial PO₂ was hard-coded as 100 in the Tier 1 source and Tier 2, and read through `getattr` with a default in Tier 3. All three now read `po2_arterial_mmHg` and raise if it is missing; `cb_settings.PerfusionSettings` gains the field at 100, so no published number moves. Tier 3's other fields (`M_max`, `k_reduce`, RQ, HCO₃⁻, both permeabilities, arterial PCO₂, D_CO₂ and the Picard settings) and the pipeline's tier switch are now also read without a default; the hidden Tier 3 `M_max` fallback was 0.05, 10× `PerfusionConfig`. `test_perfusion_config_values.py` | — |
| ~~4~~ | **Closed.** The Tier 1 washout now reads `systemic_hematocrit` instead of a hard-coded 0.45, and `cb_settings.PerfusionSettings` gains the field at 0.45. The washout was still evaluated at systemic rather than local haematocrit, kept as §11 row 27; open item 29 moved it to each cell's own haematocrit. The 0.45 edge-haematocrit and 100 µm² surface-area defaults in `perfusion.py` are gone too: `map_vessels_to_grid` raises on an edge with no `hematocrit` | — |
| ~~5~~ | **Closed.** `C_arterial` was declared 3× and read 0×; it is removed from `PerfusionConfig`, `cb_settings.PerfusionSettings` and the H2 drivers, and `test_cb_settings.py` fails if it comes back. One analytical test had set it expecting it to control arterial PO₂; it now sets `po2_arterial_mmHg` | — |
| ~~6~~ | **Closed.** Tier 1 hard-coded 50 Newton steps and 1 × 10⁻⁵, Tier 2 50 passes and 1 × 10⁻⁴, while `PerfusionConfig.picard_tolerance` was 1 × 10⁻⁴ and only Tier 3 read it. All three now read `picard_max_iterations` and `picard_tolerance` without a default; `PerfusionConfig` goes to 1 × 10⁻⁵ and `cb_settings.PerfusionSettings` gains both fields at 50 and 1 × 10⁻⁵, so no published number moves (WKY-A §2.3 matches exactly). On WKY-A 10⁻⁴ and 10⁻⁵ return the same field; against 10⁻⁸ it is within 0.10 mmHg per cell and 0.021 percentage points in the hypoxic fractions (§6.7). The pipeline's Tier 3 now stops at 1 × 10⁻⁵: on a fresh WKY-A run (at the double-converted flow of open item 31) 4 iterations instead of 3, tissue PO₂ up to 0.29 mmHg higher (mean 0.03), so 10⁻⁴ had not been as tight as the test networks suggested. Every tier's `.vti` records the tolerance. Inner CG settings stay hard-coded. `test_perfusion_solver_settings.py` | — |
| ~~7~~ | **Closed** (sources only; no code change). The CO₂ curve is McHardy (1967), not Spencer (1979), and returns vol% of whole blood; checked against a secondary quote [`mallat_ratio_2021`], not the 1967 paper. No source was found for the Haldane term. `permeability_o2_cm_s` has a measured counterpart [`liu_oxygen_1994`] whose value was not read, and a derived estimate 10²–10⁴× above the code value. What the search found wrong with the values is open item 19. `sigma_diff_co2` and `permeability_co2_cm_s` stay under item 18 | — |
| ~~8~~ | **Closed.** `M_max` differed 10× between `PerfusionConfig` (0.005) and `cb_settings.BASE_M_MAX` (0.05), and the published §2.3 results used 0.05. `PerfusionConfig` and both example YAMLs now say 0.05, and `test_cb_settings.py` asserts the config equals `BASE_M_MAX`, so no published number moves. Only the pipeline's own Tier 3 `*_perfusion.vti` changes. Fresh WKY-A run (at the double-converted flow of open item 31): converged in 7 iterations (was 4), residual 6.0 × 10⁻⁷; tissue PO₂ mean 96.82 → 75.80 mmHg, median 99.23 → 92.44, minimum 78.44 → 0.26; cells below 5 / 10 / 20 mmHg 0 → 3.00 / 5.08 / 9.63%; PCO₂ maximum 40.74 → 43.43 mmHg, pH minimum 7.393 → 7.365. On the test Y network at 10³ µm³/s Tier 3 now takes 30 iterations (was 14; the test cap is 35). `test_cb_settings.py`, `test_perfusion_config_values.py` | — |
| ~~9~~ | **Closed** by `f92a96c`. The rheology solver substituted a silent 5.0 µm diameter; it now raises, matching `map_vessels_to_grid` and `edge_transit_times`. `2d98ab8` removed the least-squares pressure fallback, but left the rheology solver's initialisation and update on 5.0 µm | §3.2, §3.4, §2.8 |
| ~~10~~ | **Closed.** Pressure boundaries differed between `HaemodynamicsConfig` (100/2 mmHg, MAP to CVP) and `cb_settings` (60/20 mmHg, arteriolar to venular), and every published H2 number used 60/20. The config now takes its defaults from `cb_settings.INLET_PRESSURE_MMHG` / `OUTLET_PRESSURE_MMHG`, both example YAMLs say 60/20 in mPa, and `test_cb_settings.py` keeps all three equal, so no published number moves. §8.1 and §10.7 now place 60/20 against measured rat microvascular pressures [`peti-peterdi_direct_1998`, `jin_study_1997`, `fronek_microvascular_1975`]: 20 is a venular value, 60 an upper arteriolar one. Fresh WKY-A run: flows 0.408× the 100/2 run (40/97.97) on 4 071 of 4 178 flowing edges, diameters and resistances unchanged; haematocrit moved by more than 0.01 on 96 of 4 512 edges, because the rheology loop stops unconverged at 15 iterations in both runs and its path is not exactly scale-free. The Tier 3 `.vti` did **not** change (7 iterations, residual 6.0 × 10⁻⁷, tissue PO₂ mean 75.80, min 0.26 mmHg, as under open item 8): the pipeline's perfusion step over-converts flow, open item 31. `test_cb_settings.py`, `test_perfusion_vti_provenance.py` | — |
| ~~11~~ | **Closed.** The entropy path defaulted to on but could not run at 2 classes, and fell back to plain hysteresis with only a warning. `enable_shannon_entropy` now defaults to False and raises if turned on for a 2-class field or one with no class axis; `shannon_entropy_core` is a real config field. The Optuna preprocessing tuner searches the two entropy thresholds only when entropy is on and an entropy map exists, so it no longer writes untested values for them to `best_preprocessing_params.yaml`. CB masks unchanged | — |
| ~~12~~ | **Closed** by `7ea1b36`; numbers re-derived 2026-09-27. The rheology loop rescaled resistance by $\mu_\text{app} / \mu_\text{old}$ against a base that no longer contained $\mu_\text{old}$, inflating every resistance ~200–540× and diameter-dependently. Re-derived with `cb_h2_glomus_perfusion.py` and the new `cb_h2_absolute_perfusion.py`: flow-weighted velocity 4.1–9.7 → 979–3,319 µm/s (face rule, 60/20 mmHg), so §13.5 now reads 1–3× above physiological, not 20–100× below. H2 §2.1/§2.2/§2.4 ratios moved up to 0.09 per specimen, ≤ 0.04 in cohort means; every overlap/no-overlap pattern held. §3.2 has the detail | §3.2, §13.5; H2 whitepaper §6–§9, §11–§13 |
| ~~13~~ | **Closed** (`a9cc5cd`, re-run 2026-09-28). The lateral ROI centroid projected over the whole stack, not the 160 slices the ROI occupies, so tissue that never enters the box helped place it. `place_roi` now clamps z first and projects only the box's own z-band; lateral centres moved 8.3 µm (SHR-C) to 45.1 µm (SHR-B), z unchanged. New centres (z, y, x): WKY-A (230, 224, 186), WKY-B (106, 192, 178), WKY-C (189, 158, 100), SHR-A (157, 244, 158), SHR-B (230, 308, 186), SHR-C (164, 302, 186). Over the band fewer columns saturate, so in WKY-A the p99 cutoff falls to 0.90 and the intensity weighting moves the centroid 0.31 µm (§2.1). Re-run together with items 27, 15 and 17 | — |
| ~~14~~ | **Closed.** `crop_roi` rounded the centre twice and landed one voxel low on odd axes with the centre above the midpoint. It now rounds once and uses the same `centre − size // 2` rule as `RoiPlacement.bounds`, so the two paths agree exactly. No CB result used that path | — |
| ~~15~~ | **Closed** (`c2a9a36`, re-run 2026-09-28). The threshold selector's “median diameter” was a median over every foreground voxel, while §2.6's calibre is a median over centreline voxels, and the 4–7 µm window is a target for the latter; the voxel median read 0.63–1.00× the centreline one. `evaluate_threshold` now takes the median and p90 of 2 × EDT at the skeleton voxels; the voxel median is printed as `d_vox` and selects nothing. At 0.95 the centreline median is 5.27 µm in all six. On `d_vox` the selection would have picked 0.93 (WKY-A) and 0.90 (the other five), so this item is what moved the freeze to 0.95 (`8f02e7b`, user-approved) | — |
| ~~16~~ | **Merged into 17.** The group asymmetry of the freeze is a consequence of item 17's quantisation, not a separate defect: it disappears under `≥` | — |
| ~~17~~ | **Closed** (`bacff70`, re-run 2026-09-28). The probability field is quantised to hundredths and every sweep threshold lands exactly on a level, so the strict `p > t` discarded a whole level — 0.5% of the ROI at 0.30 rising to 4.2–6.0% at 0.99, where it is two thirds of the mask; `p > 0.99` is exactly `p = 1.0`. The selector, the pipeline's hysteresis (flood and seed) and `cb_h1_th_metrics.py`'s vessel cut now use `p ≥ t` (`preprocessing.at_or_above`, threshold cast to the array's float type); the hysteresis is its own, face-connected like skimage's and identical to it on a field with no ties. The TH cut, the entropy hysteresis (off) and `prob_to_mask.py` keep `>`. **Consequence (formerly item 16):** the 0.90 freeze ran SHR-B and SHR-C above their own choice at 3.73 µm, below the selector's 4.0 µm floor (0 of 3 WKY, 2 of 3 SHR). After the re-run all six choose 0.95, so the asymmetry is gone. With the frozen 0.95 the seed would be 1.00, so `HYSTERESIS_SEED_CAP` (0.999) keeps it below 1: the band is 0.95 / 0.999, seed = `p == 1.0` | — |
| ~~T~~ | **Closed.** The pipeline's perfusion step defaults to Tier 3, so its `*_perfusion.vti` came from a different solver than the H2 hypoxia maps (Tier 1, run by the H2 drivers), and nothing in the file said so. Switching the default could not fix this: turning off multi-species alone lands on Tier 2, and even Tier 1 in the pipeline runs on `PerfusionConfig` inputs (`M_max` 0.005, 100/2 mmHg, band rule) rather than the H2 ones. Tier 3 stays the default; every tier now tags the `.vti` field data with its tier, solver, `M_max`, pressures and a not-H2 note (`_perfusion_provenance`). The Tier 2 switch is now read without a `getattr` default. `test_perfusion_vti_provenance.py` | — |
| ~~18~~ | **Closed.** `sigma_diff_co2` was 3.0 × 10⁻⁸ m²/s and `permeability_co2_cm_s` 2.0 × 10⁻³ cm/s, each 20× the O₂ value; since the multi-species solver also multiplies both by solubility (α_CO₂ ≈ 22× α_O₂), CO₂ moved ≈450× faster than O₂ against a measured Krogh ratio of ≈21 [`kawashiro_determination_1975`]. Now D_CO₂ = 1.6 × 10⁻⁹ m²/s (Kawashiro) and P_CO₂ = P_O₂ = 1.0 × 10⁻⁴ cm/s [`dash_simultaneous_2006`], giving ≈24× and ≈22×. (Both permeabilities moved to 9.1 × 10⁻² cm/s under item 19.) Only Tier 3 reads these, so the pipeline's `*_perfusion.vti` CO₂, pH and (through the Bohr shift) PO₂ fields change; no published H1/H2 number moves. The perfusion keys in `examples/config_*.yaml` sit under `PipelineConfig:` and are ignored by the loader. `test_perfusion_config_values.py` | §10.9, §6 |
| ~~19~~ | **Closed.** The Tier 3 blood-gas relations did not match their sources (found under item 7; §5.3). (a) and (b): the CO₂ curve is now McHardy's as its source defines it, in mmol/L, with his Hb term for *H* and no second dissolved term; his saturation term replaces the unsourced Haldane term, so the Haldane effect is 2.8 mmol/L on full desaturation against ≈2.5 measured [`loeppky_quantitative_1983`]. Content at PCO₂ 40 went from 24.4 to 21.3 mmol/L, slope from 0.28 to 0.21 mmol/L/mmHg. `test_blood_co2_content.py`. (c): `permeability_o2_cm_s` was 1.0 × 10⁻⁴ cm/s with no source and left Tier 3 anoxic; it is now the measured 9.1 × 10⁻² cm/s [`liu_oxygen_1994`], ≈900× higher, and `permeability_co2_cm_s` follows it (item 18). `test_perfusion_config_values.py`. Checking convergence at the new value found items 20 and 21. Tier 1, and so every H2 number, uses none of these; the pipeline's `*_perfusion.vti` does | §5.3, §10.8, §10.9, §6.6 Tier 3 |
| ~~20~~ | **Closed.** The Tier 2 and Tier 3 marches (`c ← c − φ/q`) read edge `flow_abs` raw, in mmHg·µm³/cP, while φ is in mmol/L·µm³/s, so *q* was 1.33 × 10⁵× too small and blood gave up its gas in the first cell of each edge whatever the flow. Both solvers now take `flow_to_um3_per_s` (default `POISEUILLE_FLOW_TO_UM3_PER_S`, as `map_vessels_to_grid`), convert each edge once (`_edge_flows_um3_per_s`), and raise if a flowing edge has no `flow_abs` (it was read with a 0.0 default) or if the per-cell flows were built with a different factor. On a two-cell chain both tiers match a hand calculation of the blood-side drop. On the Y test network Tier 3 mean PO₂ is now 23, 61 and 95 mmHg at 10³, 10⁴ and 10⁵ µm³/s (P = 9.1 × 10⁻² cm/s); it still hits `max_iter` at the lower two (item 21). Tier 2 also overshoots below ≈10² µm³/s. `test_perfusion_march_flow_units.py`. Not in any H1/H2 number | §6.2, §6.6 Tier 2, Tier 3 |
| ~~21~~ | **Closed.** Tier 3's per-cell step took φ at the blood pressure on entry and subtracted φ/*q*; once P·A·α exceeded *q*·dC/dP (P = 9.1 × 10⁻² cm/s, capillary flow, flat top of the O₂ curve) one step went past blood–tissue equilibrium, the Picard loop hit `max_iter` at 10³ and 10⁴ µm³/s, and tissue PCO₂ came out below arterial (down to 1.7 mmHg). Each cell now solves C(P_out) + (P·A·α/*q*)(P_out − P_tissue) = C_in for the outlet pressure (`_implicit_cell_outlet_pressure`, E48–E50): O₂ first at the inlet PCO₂ and the cell's pH, then CO₂ at the new PO₂. The outlet lies between tissue and inlet, *q*·ΔC equals φ, and a failed root find raises instead of keeping the old pressures. Because the flux is now taken at the outlet, which is read at the cell's own pH, even at infinite flow the blood PO₂ seen by the wall carries that cell's Bohr shift (≈ +1 mmHg in the 0D Fick test). On the Y test network (P = 9.1 × 10⁻² cm/s) PCO₂ now stays ≥ 40 mmHg everywhere, and mean PO₂ is 36.6, 77.5 and 94.8 mmHg at 10³, 10⁴ and 10⁵ µm³/s after 50 iterations; the loop still hits `max_iter` at the lower two (item 23). `test_perfusion_implicit_march.py`. Not in any H1/H2 number | §6.6 Tier 3, §6.7 |
| ~~22~~ | **Closed (code).** Tier 1's diffusion term σ·ΔPO₂ (E35) was in mmHg·µm³/s while every other term of its balance is in mmol/L·µm³/s; `sigma_diff` is D (§10.9), so diffusion was ≈1/α ≈ 750× too strong and its diffusion length ≈27× §13.6's. Tier 2's diffusion and wall flux P·S·ΔPO₂ had the same gap. Confirmed on a 1D slab between two fixed ends with a zero-order sink: analytic mid-slab drop M L²/(8Dα) = 4.975 mmHg; the old matrix gave 0.0067, the α version 4.975. All three tiers now use one `ALPHA_O2_MMOL_PER_L_MMHG`. With α in, Tier 1's fixed-slope Picard loop (γ = 0.5) hit `max_iter` on WKY-A and plain Newton oscillated, so Tier 1 is now Newton with a line search and a residual stop (§6.7, Appendix A.3). **Tier 1's fixed point with α alone is hyperoxic** on WKY-A (median 186 mmHg, 78% of cells above arterial) because its washout uses systemic haematocrit while its source uses each edge's: open item 29. The published §2.3 field (~30 mmHg, inert to glomus metabolism) rested on both defects. Re-derived 2026-09-26 after item 29 (H2 whitepaper §10): PO₂ within TH 70–86 mmHg, TH hypoxia below 10 mmHg 0–4%, 4× glomus metabolism moves WKY-A by −2.5 mmHg, and 4 µm is no longer shown grid-converged; §6.8, §13.6 and H2 §10.2 marked for review. `test_perfusion_tier1_solubility.py`, `test_perfusion_march_flow_units.py` (Tier 2) | §6.3, §6.6, §6.7, §13.6, H2 §2.3 |
| ~~23~~ | **Closed.** Tier 3's Picard loop did not reach its fixed point at capillary flow. (a) It was slow: the pseudo-washout put the full wall conductance on the diagonal, so each iteration moved tissue C′/(C′ + P·A·α/*q*) of the way (≈2 700 iterations on a two-cell chain at 10² µm³/s; 378 and 97 on the Y network at 10³ and 10⁴; no progress in a plasma-skimmed branch). (b) It stopped early: a warm-started CG at `rtol` 1e-5 returned the last field once a step was below it, and the relative-change test read zero (0.03–0.25 mmHg short on the chain; 1.7 mmHg at 1e-4). Now the diagonal carries the blood's actual response and the metabolic slope (a Newton step per cell), each update is an exact sparse LU solve, guarded Anderson acceleration sits on top, and the loop stops on the nonlinear residual in mmHg relative to the field (§6.7, E43, E51–E54). Chosen over Anderson alone (75–243 iterations on plasma-skimmed and merging vessels) and the linearisation alone (128 on the Y network). At the default 10⁻⁴: 9–19 iterations on every test network, within 0.023 mmHg of a 10⁻¹² solve; WKY-A on a fresh run (band rule, at the double-converted flow of open item 31, `M_max` 0.005): 3 iterations, minimum tissue PO₂ 78.15 mmHg, mean 96.79. The figures first quoted here (11 iterations, where the old loop hit `max_iter` with minimum tissue PO₂ 59.9 mmHg against 77.4 converged) came from the `--use-cache-dir` path, which chose its boundaries against a placeholder shape (open item 28); the old loop was not re-run on fresh boundaries. The `.vti` records convergence, iterations and residual. Getting WKY-A through Tier 3 at all first needed four other fixes: NaN flows from the flow export, junctions carried as content (item 24), flow direction from a different solve than flow size, and a Haldane effect in plasma. `test_perfusion_tier3_convergence.py`. Not in any H1/H2 number | §6.6, §6.7, Appendix A |
| ~~24~~ | **Closed.** Tier 3 mixed blood-gas *content per litre* at each node and turned it back into pressures for each daughter at PCO₂ 40 and pH 7.4, with the daughter's haematocrit; a failed root find fell back to arterial PO₂ or PCO₂. Phase separation gives daughters a different *H* from the parent, so that content did not describe the daughter's blood: on WKY-A a plasma-skimmed daughter (*H* ≈ 0.01) received arterial CO₂ content at *H* 0.45, more than its curve holds at any PCO₂, and the implicit step (item 21) found no root. The march now carries the blood's state through each node as pressures (E46, E47): one inflow passes its outlet state through unchanged; two or more are mixed by content, flow and red-cell flux and inverted jointly for PO₂ and PCO₂ at the mixture's *H* and flow-weighted pH (`_mixed_blood_state`). Each daughter's content is evaluated at the node state with its own *H*; both curves are affine in *H*, so O₂ and CO₂ are conserved wherever the rheology conserves red-cell and plasma flux. Every fallback in the march now raises: a failed inversion, a non-starting node that sends blood but receives none (it was given arterial blood at `systemic_hematocrit`, which Tier 3 no longer reads), and a cycle in the flow direction (it fell back to node order). `test_perfusion_junction_state.py`. Not in any H1/H2 number | §6.6 Tier 3 |
| ~~25~~ | **Closed.** Tier 3 seeded arterial blood at a literal pH 7.4, while tissue pH is Henderson–Hasselbalch (pKa 6.1, α 0.03, `hco3_tissue`), which gives 7.401 at PCO₂ 40 and HCO₃⁻ 24. Inside each cell the blood's O₂ is read at the tissue pH, so blood changed pH at the first cell with no exchange behind it. Through the Bohr shift and the Haldane term that left tissue below arterial: on WKY-A tissue PCO₂ 7 × 10⁻⁴ mmHg below arterial; with no metabolism on a three-cell chain, tissue PO₂ 0.08 mmHg below arterial (10 mmHg at PCO₂ 45, HCO₃⁻ 20). Arterial pH now comes from the same formula and bicarbonate as the tissue (E46), and the initial tissue pH too (§6.6). WKY-A on a fresh run with the fix in (band rule, at the double-converted flow of open item 31, `M_max` 0.005): mean tissue PO₂ 96.79 mmHg, minimum 78.15. The figures first quoted here (11 iterations, minimum tissue PCO₂ 40.00006 mmHg, mean PO₂ 96.31 → 96.39, minimum 77.4 → 77.44) came from the `--use-cache-dir` path's invented boundaries (open item 28); the before-values were not re-measured. `test_perfusion_arterial_ph.py`. Not in any H1/H2 number | §6.6 Tier 3 |
| ~~27~~ | **Closed** (`b864732`, re-run 2026-09-28 with items 15, 17 and 13). The batch pipeline run cropped the **array centre**, not the placed ROI: `cb_h1_batch.py --stage run` passed `--roi-voxels` but no offsets, so the cached masks matched the centred box (IoU 0.92 WKY-A, 0.83 SHR-C) and not `placement.bounds` (0.17, 0.15), and the boxes shared 27–61% of their volume. H1 morphometry was on centred boxes while the threshold and `cb_h1_th_metrics.py` were on placed ones, and the H2 drivers laid TH cropped at `bounds` over the centred-box graph (frames 85–210 µm apart). `carotid_image_to_model.py` now calls `place_roi` when `--roi-voxels` is given and passes its offsets to `crop_roi`; a placement that fell back to the volume centre raises, and `--roi-centred` is the explicit opt-out. Each run writes `roi_placement.json`, and every driver that reads a batch graph or `per_edge_morphometry.csv` (and the batch after each run) refuses an output whose box differs from `place_roi`'s (`check_output_roi`). Fresh WKY-A mask: IoU 0.951 with hysteresis on the placed box, 0.174 on the centred one. **Re-run results:** all six choose 0.95 (frozen, `8f02e7b`); edges WKY-A 4512 → 7597, WKY-B 3932 → 6140, WKY-C 6699 → 6863, SHR-A 6815 → 8281, SHR-B 8077 → 6954, SHR-C 4865 → 6376. H1 SHR/WKY group-mean ratios: β₁ density 1.087 (was 1.401), junction density 1.064 (1.337), length density 1.022 (1.267), median EDT diameter 0.869 (0.924), tortuosity 1.001; groups overlap on every density. H2 SHR/WKY: shunt index 0.89 (1.10), penetrating flow share 0.78 (0.57, now overlapping), flow ratio 0.91 (1.22, now overlapping), haematocrit 1.02 (0.93), transit 1.03 (0.72, now overlapping); TH hypoxia 0 everywhere and PO₂ in TH now separates the groups (§7.5). `cb_h2_vtk.py --verify` passes on all six. Old outputs: `cb_h1_batch_2026-09-26/` and the `*_2026-09-28_pre_item27` files. Provenance: these runs straddled two commits. Batch SHR-A (re-run alone after its Tier 3 stop), the 0.93 SHR-C run and all six 0.97 runs used `d57a78f` (open item 36, stagnant cutoff 10⁻¹⁰); the other five batch and five 0.93 runs used `8f02e7b` (10⁻¹²). The 2026-09-29 re-run, all on one commit, reproduced every file byte for byte, so no number depends on the split. Not re-run: the grid / cross-section sweeps (§6.8, §13.7) and the pipeline Tier 3 field of the SHR-C sensitivity runs (one edge in the 10⁻¹²–10⁻¹⁰ band of item 36, unused downstream). `tests/test_pipeline_roi.py` | — |
| ~~28~~ | **Closed** (`6c6aecd`). The `--use-cache-dir` path set the image to `np.zeros((1, 1, 1))` for an `.h5` input and to a fixed, uncropped WKY-A raw TIFF for a `.tif` input, and boundary selection and the statistics read that shape. On WKY-A under the band rule it chose 5 inlets / 539 outlets against 126 / 119 on a fresh run, with flows ~30× lower and `branch_order` different on 4487 of 4512 edges (diameters identical). The WKY-A Tier 3 figures first quoted for open items 23 and 25 (11 iterations; mean PO₂ 96.31 → 96.39, minimum 77.4 → 77.44; the old loop's 59.9) came from that path; they are re-quoted from a fresh run (3 iterations, mean 96.79, minimum 78.15), and the before-values were not re-measured. From open item 2 until this fix `_check_graph_fits_frame` stopped the path instead. The cache path now uses the cached mask, which is the frame the graph was built in; `fwhm_radius` is refused there (the cache holds no intensities); and boundary selection raises when the image shape differs from the mask's, which also catches a shape too large, which the fit check misses. WKY-A from the 2026-09-28 batch cache (scratchpad): the same 23 inlets and 21 outlets, `per_edge_morphometry.csv` and `model_results.md` identical, and the Tier 3 field identical (50 iterations, residual 9.7 × 10⁻⁶). `test_cache_path_shape.py`. Not in any H1/H2 number | §2.8, §6.6, §6.7 |
| ~~29~~ | **Closed.** Tier 1 built its source at each edge's haematocrit and its washout at `systemic_hematocrit` (0.45), so a cell fed above 0.45 got more O₂ at arterial PO₂ than it could wash out below hundreds of mmHg, and one fed below was drained. On WKY-A 27% of perfused cells are above 0.45 (per-cell H 0–0.80, median 0.37) and the surplus at arterial PO₂ was 55× the tissue's whole demand (delivery is 850× demand). The 750× diffusion of item 22 averaged these into the published near-uniform ~30 mmHg field, which is why it moved little with metabolism; with α in, the fixed point was hyperoxic (median 186 mmHg, 78% of cells above arterial). The washout now uses each cell's flow-weighted haematocrit (`cell_discharge_hematocrit`), exact because content is affine in H; `solve_perfusion_steady_state` requires it and raises if a perfused cell's is missing. WKY-A: 10 Newton steps, 18 s, max PO₂ 99.99 mmHg. Found 2026-09-26 under item 22. `test_perfusion_tier1_washout_hematocrit.py`. **Moves published numbers** (H2 §2.3, §13.6) | §6.6, §6.7, §11 row 27 |
| ~~30~~ | **Closed.** Tier 1 had no grid-converged limit: with each vessel mapped to the cells its centreline crosses, a finer grid drew it thinner, a line source in the limit, and median PO₂ on WKY-C (centred boxes) ran 91.38, 90.46, 89.52, 89.19, 87.90 at 10, 6, 4, 3, 2 µm (SHR-C 80.28 … 75.85), ≈1.5–1.9 mmHg lost per halving. `map_vessels_to_grid` now takes `vessel_mapping="cross_section"`, which sweeps each centreline point over a disc of the vessel's radius; the H2 drivers use it (the pipeline keeps the centreline default). WKY-C and SHR-C then move at most 0.21 mmHg from 3 to 2 µm but up to 0.78 from 4 to 3 µm, so `cb_settings.GRID_UM` went from 4 to 3 µm. §2.3 re-run: PO₂ in TH 81.0–89.8 mmHg (was 70.0–86.2), TH hypoxia below 10 mmHg 0–3.1% (was 0–3.95%). `test_perfusion_cross_section_mapping.py`, `test_perfusion_tier1_grid_refinement.py` | — |
| ~~31~~ | **Closed.** The pipeline's perfusion step converted flow to µm³/s twice. `carotid_image_to_model.py` passes pressures in mPa, so its Poiseuille flows are already in µm³/s, but it called `map_vessels_to_grid` and the Tier 2/3 solvers with the default `flow_to_um3_per_s` = `POISEUILLE_FLOW_TO_UM3_PER_S` (1.33 × 10⁵), which assumes mmHg, as the H2 drivers pass. Its perfusion saw flow ≈1.33 × 10⁵× too high and its blood hardly desaturated; found under open item 10, when flows fell to 0.408× and the Tier 3 field did not move. The pipeline now derives its factor from its pressure unit (`_perfusion_flow_to_um3_per_s` = `POISEUILLE_FLOW_TO_UM3_PER_S` / mPa per mmHg = 1.0), passes it to all three calls, and records it on every `.vti` as `perfusion_flow_to_um3_per_s` (a `.vti` without the tag predates the fix). Library defaults are unchanged, so the H2 drivers are not affected (§3.7). Fresh WKY-A run: median edge flow 8.0 × 10⁴ µm³/s; Tier 3 converged in 22 iterations (was 7), residual 6.9 × 10⁻⁶; tissue PO₂ mean 75.80 → 70.14 mmHg, median 92.44 → 85.56, minimum 0.26 → 0.20, maximum 99.82; cells below 5 / 10 / 20 mmHg 3.00 / 5.08 / 9.63 → 3.52 / 6.30 / 11.41%; PCO₂ mean 40.97, maximum 43.43 → 43.35 mmHg, pH minimum 7.365 → 7.366. The WKY-A pipeline figures under open items 6, 8, 23 and 25 and in §6.6–6.7 were at the old flow and are marked so, not re-quoted. `test_pipeline_perfusion_flow_units.py` (the factor, a tube solved at the pipeline's pressures against SI Poiseuille, the grid flow, the map/solver mismatch guard, and every pipeline perfusion call passing the factor), `test_perfusion_vti_provenance.py` | The pipeline's `*_perfusion.vti`; no published number |
| ~~32~~ | **Closed.** With the pipeline on the face rule (open item 2), Tier 3 on WKY-A stopped at `picard_max_iterations` = 50 with an O₂ residual of 5.9 × 10⁻⁵ against 10⁻⁵. It does not stall: run on the same inputs with a higher cap it reaches 10⁻⁵ at 55 iterations and 10⁻⁷ at 87. The 50-iteration field is within 0.018 mmHg per cell of the 10⁻⁷ one, and mean PO₂ (49.66 mmHg) and the fraction below 10 mmHg (13.21%) agree to two decimals, so the item 2 figures stand. It is slow because one slow-flow cell swings from pass to pass: the residual rises on alternate passes and the Anderson guard (`_AndersonMixer`) drops its history each time (32 of 87 passes on WKY-A), yet the residual still falls about 10× every 15 passes. Fresh face-rule runs of the other five (scratchpad; batch outputs untouched) converge in 40 (WKY-B), 80 (WKY-C, 27 resets), 42 (SHR-A), 48 (SHR-B) and 52 (SHR-C) iterations. The cap is now 200 in `PerfusionConfig`, `cb_settings.PerfusionSettings` and both YAMLs, 2.5× the slowest. The guard itself is unchanged. Tier 1 stops after 9–11 Newton steps on every §2.3 solve, and the default and padded-grid §2.3 outputs are byte-identical after the change, so no published number moves. `test_cb_settings.py`, `test_perfusion_solver_settings.py` | §6.6, §6.7, A.3, A.5 |
| ~~33~~ | **Closed.** `config_WKY_normotensive.yaml` listed its perfusion settings (`grid_resolution_xyz`, the blood-gas baselines, both diffusivities and permeabilities, `M_max`, `k_reduce`, RQ and both Picard settings) under `PipelineConfig`, which has none of those fields, so `update_dataclass_from_dict` dropped each one with a warning and `--config` ran on the `PerfusionConfig` defaults. The block now sits under `PerfusionConfig`, as in the SHR YAML, with its values unchanged. The dropped values equalled the defaults except `picard_tolerance`, so the one change in behaviour is that a WKY `--config` run now stops Tier 3 at the YAML's 10⁻⁴, as an SHR one already did (open item 34). The older YAML tests searched for keys at any depth and passed despite the misplacement; they now read `PerfusionConfig`. No H1/H2 number moves: the batch passes no `--config`. Found under open item 32 | `test_example_yaml_sections.py` (every section is one the loader reads, every key is a field of its section, loading ignores no key), `test_cb_settings.py`, `test_perfusion_config_values.py` |
| ~~34~~ | **Closed.** Both example YAMLs set `picard_tolerance` 1 × 10⁻⁴ after open item 6 moved `PerfusionConfig` to 1 × 10⁻⁵, so a run with either YAML as `--config` stopped Tier 3 at 10⁻⁴ (the WKY one was dropped until open item 33). The SHR YAML also set `pco2_arterial` 35 mmHg against 40 in `PerfusionConfig` and the WKY YAML, a difference by group. Both YAMLs now say 1 × 10⁻⁵ and 40 mmHg, and their `PerfusionConfig` sections are identical. No H1/H2 number moves: the batch passes no `--config`, and Tier 1 does not read PCO₂. Found under open item 32. The YAMLs' other leftover settings are open item 35 | `test_cb_settings.py` (both Picard settings), `test_perfusion_config_values.py` (PCO₂), `test_example_yaml_sections.py` (the two perfusion sections are equal; every YAML value reaches its section) |
| ~~35~~ | **Closed.** The example YAMLs still carried scenario settings from before the frozen methods. (a) Both set `radius_assignment_mode: constant_radius` with `constant_radius_um` 5, so a `--config` run gave every edge a 10 µm diameter instead of the frozen `edt_radius` measurement. Its provenance is `constant`, which `check_diameter_provenance` does not count as synthetic, so nothing raised. (b) Both set `constrict_at_pericytes: true`, which `HaemodynamicsConfig.__post_init__` forbids. `update_dataclass_from_dict` sets fields with `setattr` after construction, so the check never ran on a `--config` load. Nothing downstream read the flag or the constriction ratios, so it had no effect. (c) The SHR YAML set both constriction ratios to 0.95 against WKY's 1.0, a difference by group (unread, as (b)). (d) The SHR header described "high blood pressure and extreme sympathetic tone", but its pressures were the shared 60/20 mmHg. The WKY YAML also turned on `run_benchmarking` and the SHR one did not. No H1/H2 number is affected: the batch passes no `--config`. Both YAMLs now set `edt_radius` and carry no constriction, `constant_radius_um` or `fwhm_*` keys, and neither sets `run_benchmarking`. Their headers make no group claim, and the two files differ only in their first line. `update_dataclass_from_dict` re-runs `__post_init__` after setting the keys, so a YAML that sets a forbidden value now raises; the old YAMLs raise on `constrict_at_pericytes`. Found under open item 34 | §10.5, §10.6; `test_config.py` (the loader runs the checks), `test_example_yaml_sections.py` (`edt_radius`, no retired keys, the two YAMLs load to the same dict) |
| ~~36~~ | **Closed** (`d57a78f`). The 2026-09-28 SHR-A run (8281 edges) stopped in Tier 3: node 748 heads a dead-end branch whose two edges carry exactly zero flow, and its entry edge carried 1.7 × 10⁻¹² of the largest flow, outward — rounding from the pressure solve, but above the 10⁻¹² stagnant cutoff, so the march saw a node sending blood it never received. Across the six new networks rounding flows reach at most 1.7 × 10⁻¹² and real flows start at 2.1 × 10⁻⁸, with nothing between. `STAGNANT_FLOW_FRACTION` is now 10⁻¹⁰ (59× above the one, 210× below the other); flow at 10⁻⁹ at an unfed node still raises. No edge of the other five batch networks lies in (10⁻¹², 10⁻¹⁰], so their Tier 3 results are unchanged; SHR-A was re-run alone. One SHR-C sensitivity run has an edge in that band; its pipeline Tier 3 field was not re-run and is unused downstream | Appendix A |
| ~~37~~ | **Closed** (`cfee721`; H1 and H2 scripts re-run 2026-09-30, `rerun_2026-09-29_logs/J_*.log`). The coupled flow–haematocrit loop (§4.3) stopped on its 15-pass cap in every specimen, still wandering, so every flow, haematocrit, transit and shear value came from whichever pass it stopped on; `cb_h2_glomus_perfusion.json` did not record it. It was not a two-state swing: on WKY-A total flow was settled (pass 15 vs 100: 0.6% L1) while 511 edges' haematocrit still differed by > 0.01, up to 0.43, and relaxation 0.5 → 0.1 over 150 passes only lowered the floor. Two discrete switches drove it. (a) 745–895 dead-end edges per specimen carry rounding-level flow (≤ 10⁻¹² of the largest, with no edge between 10⁻¹² and 10⁻⁸), and 60–120 of them flipped direction every pass, which made their junctions switch between skimming and proportional mixing and caused every "no inflowing parent" fallback (15–29 per pass, all interior; none at an inlet). They are now left out of the transport at `STAGNANT_FLOW_FRACTION` (10⁻¹⁰, the Tier 3 cut, moved to `resistance.py`). (b) Junctions with three or more outflows mixed proportionally, so a small daughter reversing switched the rule; they now skim one-vs-rest (§4.2), which reduces to the Pries relation at two daughters. With both, relaxation 0.5 still wandered and 0.3/0.4 left SHR-C on a periodic cycle; 0.2 converges all six. The stop is now scale-free (relative flow 10⁻⁶, haematocrit residual 10⁻⁴), and pipeline (mPa) and H2 (mmHg) solves agree on every edge's haematocrit to 10⁻¹⁰. No-parent D_F is now the Murray parent and is counted; it no longer fires. Settings in `cb_settings` (cap 1000, relaxation 0.2); `rheology_status(G)` goes into the pipeline stats and VTK and every H2 JSON. **Pipeline re-run 2026-09-29** (`rerun_2026-09-29_item37_logs/`; old outputs `*_2026-09-29_pre_item37`): all 18 runs converge (0.95: WKY-A 330, WKY-B 457, WKY-C 404, SHR-A 217, SHR-B 173, SHR-C 184 passes; 0.93 and 0.97: 155–726, the slowest WKY-B at 0.93). `per_edge_morphometry.csv` and `roi_placement.json` are byte-identical, so no H1 number moves. Mean edge haematocrit rises (WKY-A 0.352 → 0.365). The Tier 3 field barely moves: mean PO₂ changes by ≤ 0.06 mmHg in all 18 runs (WKY-A 71.34 → 71.35, SHR-C 50.55 → 50.55 at 0.95). **H2 re-run 2026-09-30** (with open items 38 and 40; old outputs `*_2026-09-30_pre_itemJ`): every H2 solve converges (face rule 173–457 passes, band rule 156–388) and records it. Ratios of cohort means, SHR/WKY, unconverged → converged: shunt index 0.889 → 0.870, median flow ratio 0.908 → 0.885, **haematocrit ratio 1.019 → 1.077**, transit ratio 1.026 → 1.026. The haematocrit ratio now separates the cohorts (WKY 0.972–1.026, SHR 1.061–1.072), opposite to the §2.2 prediction; the other three overlap. SHR-C moved most (haematocrit ratio 0.998 → 1.061, flow ratio 0.518 → 0.596). The scratch forecast (0.883 / 0.926 / 1.056 / 1.038) was made before the item 40 TH lookup fix, which moved 4–59 edges per specimen from penetrating to bypassing. Absolute velocity moved −2.2% to −0.1%; §2.3 PO₂ in TH by at most 0.11 mmHg at contrast 1. H2 whitepaper re-quoted. `test_rheology_convergence.py`, `test_cb_settings.py` | §2.1, §2.2, §2.4 and every H2 flow number; §4.2, §4.3, §10.7, A.1 updated |
| ~~38~~ | **Closed** (`870de15`; H2 scripts re-run 2026-09-30). `tissue_regions.mask_fraction_per_cell` counted TH voxel centres per grid cell and divided by the mean count per cell (27 / 6.49 = 4.16 at 3 µm). A 3 µm cell holds 1 or 2 voxel centres per axis, so a solid glomus cell read 0.24, 0.48, 0.96 or up to 1.92, and `np.clip(…, 0, 1)` threw the excess away with no warning; the comment said the clip only caught rounding. About 17% of the TH volume was lost in every specimen (grid TH share / voxel TH share 0.821–0.835; WKY-A 17.4% against 20.87%), and the per-cell fractions aliased into a lattice. It fed the TH-weighted "PO₂ in TH", the per-cell metabolic rate of contrasts 2 and 4 (whose mean-rate normalisation used the aliased f̄) and the `cb_h2_vtk.py` TH field. Not group-correlated. The fraction is now the exact overlap volume of each voxel with each cell (separable per axis), with no clip; the function raises if the in-grid volume is not conserved or a cell exceeds 1, and warns on out-of-grid loss by volume. On the six masks it keeps ≥ 99.97% of the volume at 3 µm (the rest overhangs the grid). **Grid sweep re-run 2026-09-30** (cross-section, contrast 1, all six; §6.8): the all-cell median moved by ≤ 0.01 mmHg and PO₂ in TH by ≤ 0.04 at every grid (the stromal median by up to 0.77, as the TH/stroma split changed), TH hypoxia stays 0.00%, and every refinement step is under 0.5 mmHg; 4 µm now passes by 0.005 mmHg, and 3 µm is kept. Mixed cells 13–36%, wholly glomus 4–15% (§6.5). **Re-run 2026-09-30** with items 37 and 40 (`cb_h2_hypoxic_fraction.py` plain and `--pad-grid`, `cb_h2_vtk.py`): grid TH share now 9.6–32.8% against 9.7–33.3% in the voxels (was 8.1–27.5%); PO₂ in TH moved by at most 0.11 mmHg at contrast 1 and 1.0 at contrast 4 (SHR-C); TH hypoxia 0 throughout; SHR/WKY PO₂ in TH 0.967 / 0.957 / 0.942. The padded run is now identical to the unpadded one (§7.5). `cb_h2_vtk.py --verify`: frames ok in all six. `test_tissue_regions.py` | §6.5, §6.8, §7.5 "PO₂ in TH" and the contrast runs, the H2 VTK TH field |
| ~~39~~ | **Closed** (2026-10-01, docs only; scratch re-run, `cb_h2_hypoxic_fraction_xsec_pinned.json`). The cross-section sweep was not monotone at 3 µm: 3 µm gave the highest all-cell median PO₂ of 4, 3 and 2 µm in every specimen. Not the TH aliasing (open item 38) and mainly not the origin: the **grid extent** moves with h. The default grid is the node box plus half a cell, rounded up to whole cells, so its side is 310 / 306 / 304 / 300 / 300 µm at 10 / 6 / 4 / 3 / 2 µm (ROI 298.2–298.7 µm), and the vessel-free overhang cells pull the all-cell median down on the coarser grids. `bounds_zyx` cannot pin it (a union with the node box ± h/2). Pinned to the ROI corner with a 300 µm side at every h, both measures fall steadily with h in all six (4 → 3 µm −0.09 to −0.14, 3 → 2 µm −0.09 to −0.16 mmHg); the 3 and 2 µm values move by at most 0.02, the 4 µm median by +0.26 to +0.64; the TH share is constant. Moving the origin alone by 1–2 µm at 3 µm shifts the median by at most 0.12 and PO₂ in TH by at most 0.28. 3 µm stays; no H2 number moves. §6.8 has the table. | §6.8 |
| ~~40~~ | **Closed** (`5839aae`; `cb_h1_th_metrics.py` and the H2 scripts re-run 2026-09-30). Three fixes, one cause: H1 did not use one definition of the vessel network. (a) **§1.3 length** (re-run notes item 17). `cb_h1_th_metrics.py` cut the probability map plainly at 0.95 and skeletonised that, where the network uses the hysteresis band plus cleanup (1.6× the skeleton voxels), and `th_morphometry.centreline_length_um` summed every 26-adjacent voxel pair, counting all three links at a corner (9–28% above skan). WKY-A's §1.3 length was 199.4 mm against the network's 97.0 mm. The driver now reads the batch graph, skeleton and mask (`check_output_roi` first), the length is the edge polylines (`tissue_regions.edge_length_inside_um`), §1.5 is measured to the pipeline skeleton, and the pairwise estimator is gone. (b) **Half-voxel lookup.** Graph positions put voxel k's centre at k·v (checked on WKY-A to 10⁻¹⁴), but `edge_tissue_fraction`, `mask_fraction_per_cell` and `mask_bounds_um` put voxel k's *corner* there by default, so every TH lookup was half a voxel (≈ 0.93 µm) off on each axis against the graph and grid. The default first corner is now −v/2 in all three. On the six networks, TH > 0.5, edge TH fractions move by a mean 0.02–0.06 (single short edges by up to 1), and 45–201 more edges per specimen touch TH. (c) **Junction count** (re-run notes item 32). `compute_branching_statistics` counted on `nx.Graph(G)`, which merges parallel edges: 4,475 for WKY-A against 4,631 in the CSV, Figure 3 and `cb_h1_vtk.py`. It now counts on the MultiGraph; the branching angles are unchanged (still chords to neighbour nodes). **Scratch check on the current batch (TH > 0.5, not written to the outputs):** §1.3 total length now equals the CSV sum in all six (WKY-A 96.96 mm); length within TH WKY 13.2 / 20.0 / 15.2 mm, SHR 19.7 / 12.1 / 5.2 mm; density WKY 2,382 / 2,255 / 2,425, SHR 3,308 / 3,045 / 1,999 mm·mm⁻³ (ratio of means 1.18). The half-voxel fix alone lowers length within TH by 1–6%; the rest is the change of vessel set and estimator. **The H1 whitepaper §9A claim that length density separates the cohorts completely no longer holds**: SHR-C now sits below every WKY. §1.5 TVD medians rise (WKY 8.55 / 9.14 / 8.55, SHR 7.92 / 7.92 / 9.70 µm), as the pipeline skeleton is sparser. **Re-run 2026-09-30:** `cb_h1_th_metrics.py` (plain and `--all`) reproduces the scratch check to the digit, at every TH threshold (0.5 / 0.7 / 0.9: density ratio 1.18 / 1.21 / 1.26, SHR-C below every WKY at each; volume ratio 0.61 / 0.60 / 0.60; TVD ratio 0.97). H1 whitepaper §9A re-quoted: §1.3 density and §1.5 TVD no longer separate the cohorts (ledger C14, C16 now Not supported), and the density contrast comes from the smaller SHR TH volume, not more vessel (region length 1.02×, length within TH 0.76×). In H2 the centred lookup changed 4–59 edges per specimen from penetrating to bypassing and raised the `cb_h2_vtk.py` penetrating-inside share from 83–88% to 86–90%. **Left:** the pipeline's `model_results.md` junction count updates at the next batch run. `test_th_morphometry.py`, `test_tissue_regions.py`, `test_statistics.py` | §7.2, §13.7, H1 §1.3 and §1.5, H2 §2.1–§2.3 TH joins |
| ~~41~~ | **Closed** (`b7666dd`, `f18cea0`, `aaff25e`). Threshold and calibre measurement (re-run notes package F, items 3, 6, 31). (a) **Seed.** The sensitivity neighbours were run with `--hysteresis-low` alone, so 0.93 seeded at 0.98 while the frozen run seeds at 0.999; `cb_h1_batch.py --stage sensitivity` now passes `HYSTERESIS_HIGH` beside each low. Re-run 2026-09-30: 0.97 byte-identical, 0.93 ratios β₁ 1.054 → 1.050, junctions 1.055 → 1.051, length 1.015 → 1.013; the 0.93–0.95 calibre shift 0.690 → **0.740 µm**, so the error floors are now independent 4.2%, correlated **47.0%**, ratio **5.9%** (§13.2–13.3). (b) **Which calibre selects.** The selector's plain-cut centreline median is voxel-quantised (0.93 and 0.95 both 5.27 µm) and read on a mask the network does not use (`d_net` 7.46 WKY / 6.46 SHR at 0.95). Every candidate estimator moves calibre by about the window's width (§2.2), so the plain cut is kept and the sweep reports the same rule on `d_net` less one voxel: median 0.95 (WKY-B alone 0.97), no group split. The frozen value stands. `test_threshold_selection.py`, `test_cb_h1_batch_sensitivity.py`, `test_example_main_blocks.py` (the propagation default tracks the measured shift) | §2.2, §13.2–13.3, Figure 3 (read from the sensitivity outputs since `78068aa`, re-run package D) |
| 42 | **Open.** The EDT measures to the nearest *background voxel centre*, so it overstates a radius by about half a grid step and a diameter by about one voxel (§2.2). §2.6's per-edge `edt_diameter_um`, which sets every conductance, is read the same way on the native grid. The size is estimated, not yet measured per edge: on WKY-A at 0.95 the network-mask centreline median falls 7.46 → 5.60 → 5.28 µm at native, 2× and 3× supersampling (§2.2). An offset of one voxel on a 6.7 µm median is about 28% of calibre, which reaches absolute resistance at the fourth power (about 3.7× if all of it is real). Within-specimen ratios cancel a uniform offset but not its calibre dependence, since a fixed offset is a larger share of a thin vessel. Also touches the per-specimen calibre of §13.8. **To do:** measure the bias per edge (supersampled EDT, or a sub-voxel boundary estimate) and decide whether to correct §2.6. A correction re-runs the batch, the sensitivity runs and every H2 driver. | §2.6, §4, §7, §13.2, §13.8 |

**"Pinned" is not "fixed".** Items 1, 2, 8 and 10 are the same defect — a value written down
twice — and all four now have a single owner in `cb_settings.py` plus a test that fails if the
config default drifts further. What is still open is the *decision*: which of the two values is
right. That is a modelling judgement, not a refactor, and changing either one re-dates every
number in §7 and §13. Item 12's absolute flows were re-derived on 2026-09-27 and again in the
2026-09-28 re-run (§13.5).
Items 27, 15, 17 and 13 were fixed together and closed by one full re-run on 2026-09-28: the ROIs
were re-placed, the threshold re-selected (0.95) and every H1 and H2 script re-run, which moved
most numbers in §2, §7 and §13. Several conclusions there are marked **needs review** rather than
rewritten.

---
