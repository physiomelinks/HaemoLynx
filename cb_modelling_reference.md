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
| Why is absolute perfusion so far below physiological? | §13.5 — and pressure is not the cause |
| What pressure boundaries did the published H2 numbers use? | §7.8 and §8.1 — 60/20, not the config's 100/2 |
| Which coupling tier produced the oxygen field? | §6.6 — Tier 1; Tier 2 is unreachable |
| What grid resolution was used, and is it converged? | §6.8 — 4 µm, within ~1% of the limit |
| Why is transit time reported as a ratio instead of a number? | §7.6, then §13.3 |
| Which boundary rule is in force, and how much does it move things? | §2.8, then §13.4 |
| Why can I not quote a glomus hypoxic fraction? | §13.6 — the tissue is not diffusion-limited |
| Is calibre a defensible H1 finding? | §13.8 — no |
| Can I use the TH channel for a between-group contrast? | §13.9 — qualified, bounded by sensitivity analysis |
| What is turned off in the model, and why? | §10.6 constriction; §6.6 Tier 2; §2.5 bundle collapse |
| What am I allowed to claim? | **§13.10** |
| What was a given parameter set to? | §10 |
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

## §2 — Geometric model: from voxels to a vascular graph

Everything downstream inherits from this section. Resistance goes as *d*⁻⁴ (§13.1), so a choice
made here about where the vessel wall sits is amplified fourfold in every flow quantity.

Each subsection ends with **At a glance** — the choice, the number, the code, and the test.

---

### 2.1 ROI placement

**What it does.** Picks where the analysed sub-volume sits inside each specimen's imaged block.

**What was chosen.** Tissue-centroid placement, computed from each specimen's own data, rather
than a centred box.

**Why.** A matched ROI *size* makes the samples the same size; it does not make them the same
anatomy. The carotid body does not sit in the middle of its imaged block, and it does not sit in
the same place in every block — the axial tissue peak ranges from slice 106 of 435 to slice 230 of
435. A centred box therefore lands mid-organ in one specimen and in its sparse margin in another,
and the resulting density difference is a difference in where the box was put.

**Worse, the misplacement is not random with respect to the comparison.** WKY peaks at a mean depth
fraction of 0.40 and SHR at 0.34, so a centred box systematically samples a different part of the
organ in each cohort.

**The order of operations.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Read the specimen's QC record | — | **On** | `roi_placement.py:113` |
| 2 | z ← `z_profile.peak_slice` | — | **On**; falls back to `shape[0] // 2` and records `z=volume_centre` | `roi_placement.py:118` |
| 3 | Open the Ilastik input HDF5, **channel 0 only** | subsample (4, 2, 2) | **On** | `roi_placement.py:133` |
| 4 | Maximum-intensity projection along z | — | **On** | `roi_placement.py:85` |
| 5 | Threshold the projection | 99th percentile | **On** | `roi_placement.py:86` |
| 6 | Intensity-weighted centroid of the survivors → y, x | — | **On**; falls back to the volume centre and records why | `roi_placement.py:92` |
| 7 | Rescale the centroid back to full resolution | × 2 in y and x | **On** | `roi_placement.py:135` |
| 8 | Clamp the centre so the box fits whole | 160³ | **On** | `roi_placement.py:142` |
| 9 | Centre → fractional offsets for `crop_roi` | — | **On** | `roi_placement.py:147` |
| 10 | Crop the probability field to the ROI | 160³ voxels | **On** | `carotid_image_to_model.py:754` |

Steps 1–2 and 3–7 are independent of each other, which is the point of the next paragraph.

**How — two different rules for z and for y/x.** They are not one 3D centroid.

*Axial (z)* comes from the QC record's `peak_slice`, which `preprocess_cb.py` derived from tissue
extent per slice. Nothing is recomputed here; the value is read.

*Lateral (y, x)* comes from `tissue_centroid_yx` on channel 0, subsampled (4, 2, 2): take the
**maximum-intensity projection along z**, threshold at the **99th percentile**, then take the
intensity-weighted centre of mass of the surviving voxels. The percentile threshold is the point —
the mean of a background-subtracted volume is dominated by near-zero voxels, which drags the
centroid back towards the geometric middle and defeats the measurement.

**No vesselness is used, deliberately.** Channel 0 is the background-subtracted grayscale. The
vesselness channels exist in the same file — multiscale **Sato**, fine at σ 1.0/1.4/2.0 px and
coarse at σ 4.0/8.0 px — and are *deliberately not read here*, because they are derived from
channel 0 and would weight the centroid towards whichever filter scale happened to dominate.

**No silent fallback.** If neither the QC record nor the preprocessed volume is reachable it falls
back to the volume centre *and records that in `source`*. A silent fallback would reintroduce
precisely the bias the function exists to remove.

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
> peak slice ranges 106–230 of 435; cohort depth fractions 0.40 vs 0.34 · `roi_placement.py:96`,
> `roi_placement.py:77`, `roi_placement.py:54` · `tests/test_roi_placement.py`

---

### 2.2 Segmentation threshold selection

**What it does.** Chooses the probability threshold that turns the classifier's output into a
binary mask.

**What was chosen.** *Calibre chooses, fragmentation vetoes.* Take the **highest** threshold whose
median mask diameter falls in the capillary window, provided it lies below the fragmentation onset.

**The order of operations.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Place the ROI and crop the probability volume | 160³ | **On** | `cb_h1_batch.py:79` |
| 2 | Cut a mask at each threshold in the grid | `p > t`, a **plain cut** | **On** | `threshold_selection.py:157` |
| 3 | EDT → median and p90 diameter | `sampling` = voxel size | **On**, decisive | `threshold_selection.py:161` |
| 4 | Label mask components, largest share, count above floor | 50 voxels | **On**, reported but not decisive | `threshold_selection.py:166` |
| 5 | Skeletonise the cut mask | raw, **no cleanup** | **On** | `threshold_selection.py:176` |
| 6 | Skeleton length from voxel count | × in-plane pitch | **On** | `threshold_selection.py:181` |
| 7 | Count degree-1 voxels → endpoint density | per mm of skeleton | **On**, decisive | `threshold_selection.py:185` |
| 8 | Drop thresholds whose mask is empty | — | **On** — a legitimate outcome, not an error | `threshold_selection.py:217` |
| 9 | Baseline = **median** endpoint density across the sweep | — | **On** | `threshold_selection.py:241` |
| 10 | Onset = lowest threshold above 1.5 × baseline | `FRAGMENTATION_TOLERANCE` | **On** | `threshold_selection.py:246` |
| 11 | Calibre window = thresholds with median d in range | 4.0–7.0 µm | **On** | `threshold_selection.py:249` |
| 12 | Chosen = **highest** window threshold below onset | — | **On**, or a refusal | `threshold_selection.py:271` |
| 13 | Repeat 1–12 per specimen; median of six, snapped to grid | 6 specimens | **On** | `cb_h1_batch.py:104` |
| 14 | Cohort-split check on the per-specimen choices | — | **On**, reported | `cb_h1_batch.py:99` |

**The selector does not measure the mask the pipeline builds.** Step 2 is a plain cut (`p > t`) and
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

**Why calibre and not component statistics.** The conventional criterion — the value just above
where component count climbs and the largest component's share falls — does not discriminate on
this data. The largest component's share never falls; it is *higher* at 0.99, where the network has
visibly shattered into 7,151 pieces, than at 0.70. Share is counted in voxels, and this network is
one dominant mass at every threshold with fragments too small to move a voxel fraction. Counting
components above a 50-voxel floor is equally flat, wandering between 94 and 139 across the whole
range with no structure. That is a property of the data's topology, not of any one classifier.

**Why the highest, not the middle.** Calibre falls monotonically with threshold, and the risk being
traded is over-inclusion: the lower the threshold, the fatter the vessel, and resistance carries
that as *r*⁻⁴.

**How fragmentation is measured.** Not by component count but by **endpoint density per mm of
skeleton**. The baseline is the median density across the sweep; the onset is the lowest threshold
whose density exceeds 1.5× that baseline. A network breaking into beads gains endpoints far faster
than it gains components.

**It can refuse.** Two refusals, both reported as segmentation problems rather than resolved by
picking something: no threshold reaches capillary calibre at all, or every threshold at capillary
calibre is at or beyond the fragmentation onset.

**One threshold for all six, and where 0.90 came from.** The selector runs *per specimen*, but no
specimen runs at its own choice. `cb_h1_batch.py --stage threshold` sweeps the grid
`[0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]`, selects per specimen, then takes
the **median of the six selections and snaps it to the nearest grid value**. That is the frozen
0.90. The rationale is in the driver: per-specimen thresholds would absorb exactly the
classifier-quality differences that H1 is trying to measure, turning a confound into an apparently
clean result. The per-specimen choices are still reported and passed to `assess_cohort_split`, so a
threshold that splits by group is visible rather than hidden.

**The hysteresis pair follows the frozen value.** `--stage run` passes the frozen threshold as
`--hysteresis-low` only. The pipeline raises the high bound automatically when the low one would
overtake it (`carotid_image_to_model.py:2047`), so a frozen 0.90 gives a 0.90 / 0.95 pair, not the
config's 0.65 / 0.75 with an inverted ordering.

| Constant | Value | Meaning |
|---|---|---|
| `CAPILLARY_DIAMETER_RANGE_UM` | (4.0, 7.0) µm | The calibre window that selects |
| `FRAGMENTATION_TOLERANCE` | 1.5× | Endpoint-density multiple defining onset |
| `MIN_COMPONENT_VOXELS` | 50 | Floor for the component count that is reported but not used to select |

> **At a glance** — highest intact threshold in a 4–7 µm calibre window · fragmentation onset at
> 1.5× baseline endpoint density · `threshold_selection.py:222`, `threshold_selection.py:137` ·
> `tests/test_threshold_selection.py`

---

### 2.3 Mask formation

**What it does.** Converts the probability field to a binary mask.

**What was chosen.** Hysteresis thresholding with all pre-threshold filtering disabled. A joint
probability-and-entropy variant exists and is enabled by default, but **it does not run on the CB
vessel data** — see below.

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

### The operative path: plain hysteresis

Two thresholds, `low = 0.65` and `high = 0.75`, applied as a connectivity rule rather than a cut:

1. **Seed.** Every voxel with `p ≥ 0.75` is a seed.
2. **Grow.** Every voxel with `p ≥ 0.65` is a candidate.
3. **Keep** only the candidates that are connected to at least one seed.

A voxel at p = 0.70 is therefore kept or discarded *depending on its neighbours* — kept if it hangs
off a confident core, discarded if it is isolated. This is the whole point: a single global cut at
0.75 severs vessels wherever the classifier dipped, while a single cut at 0.65 admits every
scattered speck. Hysteresis takes the connected interior of the first and the extent of the second.

**Why this matters more here than in a typical image.** Classifier confidence falls at vessel
*walls* — the boundary voxels are genuinely mixed. A hard cut therefore erodes every vessel from
the outside in, and resistance carries that as *d*⁻⁴. Hysteresis lets the mask grow out to the wall
provided it started somewhere confident.

**The band is narrow — 0.65 to 0.75.** With only 0.10 of separation, the growth step is a modest
dilation of the seed set rather than a long reach, so the mask is closer to a plain cut at 0.75
than the two numbers suggest. Widening the band would recover more wall at the cost of admitting
more speckle.

> ⚠ **Neither number is what H1 ran at.** These are the `PreprocessingConfig` defaults. H1 passed
> the frozen 0.90 as `--hysteresis-low`, which auto-raises `high` to 0.95 (§2.2). So the operative
> band was **0.90 / 0.95**, not 0.65 / 0.75 — a much stricter seed and a much narrower band.

**Where the values came from.** Chosen, not tuned, and the config comment says why the tuner cannot
choose them: the preprocessing objective is `1 − mean probability inside the mask`, which rises
monotonically across the whole plausible band, so its argmin is always the top of the search range
rather than a property of the data. The yield cliff meant to stop it never engages — probability
yield is still 0.071 at `low = 0.85`, well above the 0.05 trigger. The values were set instead from
calibre and connectivity, measured on the reference subvolume:

| `low` | Foreground | r_p90 (µm) | Components | |
|---|---|---|---|---|
| 0.20 | 0.847 | 31.55 | 1 | floods into one blob |
| 0.60 | 0.154 | 4.57 | 61 | |
| **0.65** | 0.118 | **4.17** | 116 | **chosen** |
| 0.70 | 0.090 | 3.73 | 84 | |
| 0.80 | 0.045 | 3.23 | 367 | breaking into fragments |
| 0.85 | 0.031 | 2.64 | 414 | |

r_p90 of 4.17 µm is the right scale for a ~3 µm capillary radius, and component count is stable
from 0.60 to 0.73 before exploding above 0.80. Both criteria agree on 0.60–0.75, and 0.65 sits
inside that range rather than against a search bound.

**The full mask-formation order**, as executed:

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | ROI crop of the probability field | 160³ voxels | **On** | `carotid_image_to_model.py:754` |
| 2 | Class-axis detection, then entropy map | `n_classes` | **Skipped** — 2 classes, warns and leaves `entropy_map = None` | `carotid_image_to_model.py:781` |
| 3 | Vessel channel selection | `ilastik_vessel_channel` | **On** | `carotid_image_to_model.py:789` |
| 4 | Virtual padding in z | 10 voxels, `mode='edge'` | **On** (`caged`) | `carotid_image_to_model.py:680` |
| 5 | Median filter | size 0 | Off | `carotid_image_to_model.py:684` |
| 6 | Morphological opening | radius 0 | Off | `carotid_image_to_model.py:688` |
| 7 | Morphological closing | radius 0 | Off | `carotid_image_to_model.py:692` |
| 8 | Probability smoothing | sigma 0.0 | Off | `carotid_image_to_model.py:696` |
| 9 | Joint probability–entropy hysteresis | core 0.6 / max 0.95 | **Unreachable** — guarded on `entropy_map is not None` | `carotid_image_to_model.py:701` |
| 10 | Plain hysteresis threshold | 0.90 / 0.95 **as run** (config 0.65 / 0.75) | **On** | `carotid_image_to_model.py:710` |
| 11 | Hole filling, 3D | — | **On** | `carotid_image_to_model.py:720` |
| 12 | Un-pad | 10 voxels in z | **On** | `carotid_image_to_model.py:722` |

Steps 5–8 are the pre-threshold chain, and all four are off. Step 9 is the config default and never
executes. Nothing between the crop and the threshold changes a single probability value on the CB
path — the mask is the threshold, the hole fill, and nothing else.

Note the padding is `mode='edge'`, so the pad replicates the boundary probability rather than
adding background. It exists to stop the boundary caging the mask, and it is stripped before the
mask is returned.

**What the entropy map is.** Per voxel, across the classifier's class probabilities:

```
H = −Σ_c p_c · log₂ p_c        normalised by log₂(n_classes) → H ∈ [0, 1]
```

It measures **how undecided the classifier was at that voxel**, not how likely the voxel is to be
vessel. H = 0 means the classifier put everything on one class; H = 1 means it split evenly across
all of them. Probability answers *which class*; entropy answers *how sure*.

**The two entropy parameters are confidence gates on the two hysteresis tiers.**

| | Probability test | Entropy test | Meaning |
|---|---|---|---|
| **Seed** | `p ≥ high` | `H ≤ shannon_core` = **0.6** | Where the mask is allowed to start |
| **Candidate** | `p ≥ low` | `H ≤ shannon_max` = **0.95** | Where it is allowed to grow into |

Seeds are then morphologically reconstructed into the candidate mask by dilation, so a candidate is
kept only if it connects back to a seed.

So `shannon_core` is the **strict** gate and `shannon_max` the **permissive** one — the same
high/low logic as ordinary hysteresis, applied to confidence instead of probability.
`shannon_core ≤ shannon_max` is enforced, and reversing them raises.

- **`shannon_core = 0.6`** — to *start* a vessel, the classifier must have been fairly decided.
- **`shannon_max = 0.95`** — to *continue* one, almost any confidence will do; this rejects only
  near-total confusion.

The intent is to stop the mask seeding inside ambiguous tissue while still letting it grow through
vessel walls, where a classifier is legitimately less certain.

**It needs three or more classes to mean anything.** With two classes, H(p) is a deterministic
function of p alone, folded about p = 0.5 — so it carries **no information the probability does not
already carry**. `H ≤ t` then resolves to `p ≤ r OR p ≥ 1 − r`, which carves a band out of the
*middle* of the probability range: the mask becomes non-monotonic in p, keeping low-probability
voxels while discarding higher-probability ones. Every vessel would come out as a core plus a
detached shell, with the wall voxels evacuated — the opposite of the intent.

> ⚠ **Open item 11 — both entropy parameters are inert on the CB path.** The pooled vessel
> classifier's `LabelNames` are `['vessel', 'background']`: **two classes**. The pipeline detects
> this, warns, leaves `entropy_map` as `None`, and routes to plain `hysteresis_threshold`. So
> `shannon_core` and `shannon_max` never affect any CB mask, and `enable_shannon_entropy = True`
> in the config describes a path that does not execute.
>
> Two consequences worth knowing. The library function `joint_hysteresis_threshold` *raises* on
> `n_classes < 3`, but the pipeline only *warns and falls back* — the safety net is real, but it is
> silent in the run log rather than fatal. And **`shannon_entropy_core` is not a field of
> `PreprocessingConfig` at all** — it is read as `pre_config_dict.get("shannon_entropy_core", 0.6)`
> and exists only in the auto-tuner's search space, so even on a 3-class classifier it could not be
> set from config and would silently sit at 0.6. Same shape as open item 5 (`C_arterial`).
>
> The joint path re-engages by itself if the classifier is retrained with a third class — glomus
> being the obvious candidate, since the TH channel already exists.

> **At a glance** — plain hysteresis in practice; the joint probability–entropy path is dead at
> 2 classes · no pre-threshold filtering, median-3 costs 80% recall · `image.py:179`,
> `image.py:261`, `carotid_image_to_model.py:769` · `tests/test_preprocessing.py`,
> `tests/test_new_preprocessing.py`

---

### 2.4 Skeletonisation and graph construction

**What it does.** Reduces the binary mask to a one-voxel-wide centreline, then converts that
centreline into a graph of nodes and edges carrying physical coordinates and lengths.

**What was chosen.** Mask repair by morphological *closing* (not dilation), largest-component-only
pruning, skeletonisation at native resolution, then skan-based segment extraction with loop
stitching and a 5.6 µm terminal reconnection.

**The order of operations.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Morphological closing of the mask | radius 1 | **On** | `carotid_image_to_model.py:839` |
| 2 | "Gap bridging" of the mask | size 1 | **On, but a no-op** — also a radius-1 closing, and closing is idempotent | `carotid_image_to_model.py:840` |
| 3 | Keep only the N largest mask components | N = 1 | **On** | `carotid_image_to_model.py:847` |
| 4 | Skeletonise | downsample 1.0 | **On**, native resolution | `skeleton.py:21` |
| 5 | Remove small skeleton objects | 3 voxels | **On** | `skeleton.py:526` |
| 6 | Bundle collapse | density 1.0 | **Disabled** — see §2.5 | `skeleton.py:532` |
| 7 | Component fraction filter | 5% | **On** | `skeleton.py:540` |
| 8 | Re-skeletonise | — | **On** | `skeleton.py:546` |
| 9 | Bridge separated skeleton components | 0 | **Disabled** | `skeleton.py:547` |
| 10 | skan path extraction → NetworkX MultiGraph | — | **On** | `build.py:22` |
| 11 | Terminal reconnection | 5.6 µm | **On** | `build.py:157` |

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

**Closing at radius 1 runs twice and acts once.** `closing_radius` and `bridge_gap_size` are both 1
and both call the same function. Closing is idempotent for a fixed structuring element, so step 2
changes nothing. Radius is not additive across calls: one call at radius 2 would be a different and
larger operation, not equivalent to two at radius 1.

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

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Reconnect branches that touched a stitched loop | — | **On** | `carotid_image_to_model.py:944` |
| 2 | Merge near-coincident nodes, resolve triangles into Y-junctions | 5.6 µm | **On** | `carotid_image_to_model.py:949` |
| 3 | Degree-2 collapse, curvature-preserving | — | **On** | `carotid_image_to_model.py:960` |
| 4 | Terminal stub pruning, iterated to convergence | 5.6 µm, ≤ 100 passes | **On**; counts printed unconditionally | `carotid_image_to_model.py:973` |
| 5 | Remove self-loop edges on otherwise isolated nodes | — | **On** | `carotid_image_to_model.py:986` |
| 6 | Core dead-end resolution (`eradicate` / `stitch`) | mode `"none"` | **Disabled** | `carotid_image_to_model.py:991` |
| 7 | B-spline smoothing of every edge centreline | `smoothing_alpha` 0.75 | **On** | `carotid_image_to_model.py:1014` |
| 8 | Keep largest connected component of the graph | `True` | **On** | `carotid_image_to_model.py:1028` |

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

### 2.6 Calibre assignment

**The single most consequential choice in the pipeline.** Everything in §13.1 follows from it.

**What it does.** Assigns each edge a diameter, from which resistance, lumen volume, surface area
and transit time all follow.

**What was chosen.** EDT — the Euclidean distance transform's inscribed radius — with a
junction-proximity exclusion of 3.73 µm.

**The order of operations.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Branch orders assigned from the inlets (§2.7) | — | **On** | `carotid_image_to_model.py:1076` |
| 2 | Synthetic branch-order diameter table filled by exponential fit | 3 anchor points | **On**, but never consumed under `edt_radius` | `carotid_image_to_model.py:1141` |
| 3 | FWHM ray-casting over the raw field | half-extent 15.0 µm | **Off** — `radius_assignment_mode = "edt_radius"` | `carotid_image_to_model.py:1144` |
| 4 | 3D EDT of the binary mask, in physical units | `sampling` = voxel size | **On** | `automated.py:1283` |
| 5 | Sample the EDT at every centreline voxel of every edge | — | **On** | `automated.py:1305` |
| 6 | Drop samples outside the mask or at radius 0 | — | **On** | `automated.py:1316` |
| 7 | Flag which of an edge's two ends are junctions | — | **On** | `automated.py:1323` |
| 8 | Trim samples within the exclusion of a junction end | 3.73 µm | **On** | `automated.py:1327` |
| 9 | If nothing survives, keep the untrimmed median and tag it | — | **On** — `untrimmed_too_short`, 61% of edges | `automated.py:1338` |
| 10 | Per-edge diameter = 2 × **median** surviving radius | — | **On** | `automated.py:1345` |
| 11 | Refuse if any edge fell back to a synthetic diameter | `MAX_SYNTHETIC_FRACTION_EDT = 0.0` | **On**, raises | `poiseuille.py:13` |
| 12 | Resistance from the assigned diameter | Hagen–Poiseuille | **On** | `poiseuille.py:160` |

**Median, not mean, along the edge.** Step 10 takes the median of the per-voxel diameters so a
single local bottleneck or bulge cannot set the edge's calibre. The full sample list is retained as
`edt_diameter_samples_um`, so the within-edge spread stays recoverable.

**The trim is recorded, not just applied.** Every edge carries `edt_junction_trim` as one of
`trimmed`, `no_junction`, `untrimmed_too_short` or `not_applied`. This is what makes the 61%
figure below countable rather than an estimate.

**Step 2 runs even though step 11 forbids its output.** The branch-order diameter table is built on
every run, then never read under `edt_radius` — and if it ever were read, the zero-tolerance guard
would raise first. It is live code on a dead path, the same shape as the entropy parameters
(§2.3, open item 11).

**The alternative.** FWHM: fit a Gaussian plus baseline to the intensity profile across the vessel
and report `2√(2 ln 2)·σ`.

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

| Exclusion (µm) | Voxels | Trimmed | Too short | Moved | Mean (µm) | Mean shift | *r*⁻⁴ factor |
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
observed across 34,900 edges — so any fallback is a defect rather than an expected shortfall. FWHM
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

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | BFS from the starting node set → per-node hop distance | — | **On** | `branch_order.py:104` |
| 2 | Skip edges outside `included_edges` / inside `excluded_edges` | both empty | **On**, but vacuous | `branch_order.py:122` |
| 3 | Edges with both ends unreachable → `unreachable_edges` | — | **On**; skipped, not defaulted | `branch_order.py:131` |
| 4 | Edge order = `min(dist(u), dist(v)) + 1` | — | **On** | `branch_order.py:135` |
| 5 | Format as `B01`, `B02`, … and write to the edge | zero-padded to 2 | **On** | `branch_order.py:136` |
| 6 | Count edges per order, report the unique set | — | **On**, printed | `carotid_image_to_model.py:1077` |

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

**What was chosen.** The face-crossing rule on **axis 1**, with a tolerance of one voxel —
in the H2 drivers. The main pipeline still runs the band rule on axis 0; see the warning below.

> ⚠ **Two different rules are in use, and the main pipeline does not use the face rule.**
>
> | Driver | Function | Rule | Axis | Parameter |
> |---|---|---|---|---|
> | `carotid_image_to_model.py` (**H1**) | `select_boundary_terminal_nodes` | **Band** | **0** | `edge_percent` / `end_percent` = 25 / 25 |
> | `cb_h2_vtk.py`, `cb_h2_hypoxic_fraction.py`, `cb_h2_glomus_perfusion.py` (**H2**) | `select_boundary_terminal_nodes_by_face` | **Face** | **1** | tolerance 1 voxel |
>
> The face rule is the reasoned choice and the evidence below is why. It is **only reached by the
> three H2 drivers**, which load a graph and select boundaries themselves. Every H1 run went through
> the band rule on axis 0 at a 25% band — the rule this section argues against, on the axis that the
> axis analysis did not select. This is the same shape as the pressure split recorded in §8.1: the
> H2 drivers carry the considered settings and the main pipeline carries the config defaults.
>
> **This is open item 2**, and the table above is the concrete form of it. Either the main pipeline
> is switched to the face rule, or every H1 boundary-dependent quantity is reported as
> band-rule-derived. The affected H1 readouts are
> those that depend on inlet identity: branch order (§2.7), transit time, and anything routed
> through `resistance_node_pair`. β₁, calibre and length distributions are unaffected — they are
> fixed before boundaries are chosen.

**The order of operations — the face rule, as the H2 drivers run it.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Collect degree-1 nodes that carry a `pos` | — | **On** | `boundaries.py:45` |
| 2 | Scale the axis extent from voxels into microns | `voxel_size[axis]` | **On** | `boundaries.py:40` |
| 3 | Low-face terminals within tolerance → inlets | 1 voxel | **On** | `boundaries.py:88` |
| 4 | High-face terminals within tolerance → outlets | 1 voxel | **On** | `boundaries.py:88` |
| 5 | A terminal within tolerance of both faces → low face wins | — | **On** | `boundaries.py:88` |
| 6 | Raise if either face carries no terminal | — | **On**, no fallback | `boundaries.py:88` |
| 7 | Non-face terminals: nothing under `caged` | `caged` | **On** | `boundaries.py:95` |

**And the band rule, as the main pipeline runs it.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Collect degree-1 nodes that carry a `pos` | — | **On** | `boundaries.py:45` |
| 2 | Scale the axis extent from voxels into microns | `voxel_size[axis]` | **On** | `boundaries.py:40` |
| 3 | Terminals in the lowest `edge_percent` of the axis → inlets | 25% | **On** | `boundaries.py:47` |
| 4 | Terminals in the highest `end_percent` of the axis → outlets | 25% | **On** | `boundaries.py:48` |
| 5 | If either set is empty, fall back to the extreme 10% of **all** nodes | — | **On** — silent, logged at INFO only | `boundaries.py:52` |
| 6 | Route the remainder by permeability mode | `caged` | **On** — non-band terminals are not boundaries | `boundaries.py:66` |
| 7 | Drop any node that landed in both sets | — | **On**, inlets win | `boundaries.py:81` |
| 8 | Sort inlets ascending, outlets descending, by axis coordinate | — | **On** | `boundaries.py:83` |

⚠ Step 5 of the band rule is the fallback this section warns about below, and it is **live** on the H1 path.
It converts a graph with no terminal in the band into a solved one by promoting the extreme decile
of *all* nodes — interior spurs included — to pressure boundaries. The face rule raises instead.

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
| Band, axis 1, width 10/25/40% | 75.8% |
| **Face, axis 1, tolerance 1/2/4 voxels** | **13.3%** |

A 5.7-fold reduction, and it comes from the *parameter*, not the axis. The band width has no
principled value, so its whole plausible range is live. The face tolerance is anchored to the voxel
size — one voxel means "on the face" — and the other values exist only to show the answer does not
depend on it.

> **Comparing at a fixed second parameter is misleading, and initially pointed the other way.**
> Axis spread alone is 28.4% for the band rule against 31.5% for the face rule, which flatters the
> band rule by holding the parameter that damages it at its default. Both parameters have to move.

**Why axis 1.** Not anatomy — availability. It is the only axis solvable in all six specimens: axis 0
has no outlet terminal in SHR-A, and axis 2 has no inlet terminal in SHR-C. That is a selection
criterion, and a property of these graphs rather than a general rule.

**It raises rather than falling back.** If a face carries no terminals the rule refuses. A band
fallback would drop to the extreme 10% of *all* nodes, converting an unsolvable region into a solved
one with invented boundaries.

**One tie-break.** A terminal within tolerance of both faces would mean a region one voxel thick.
The low face wins; the ambiguity is not silently doubled into both sets.

**Permeability modes.** Under `caged` (default) non-face terminals are simply not boundaries.
`universal_sink` adds them all as outlets; `robin_resistance` tags them for a distal resistance.

> **At a glance** — face-crossing rule, axis 1, 1-voxel tolerance, raises on an empty face ·
> 86% of degree-1 nodes are interior; ratio spread 13.3% vs 75.8% · `boundaries.py:88`,
> `boundaries.py:208` · `tests/test_boundary_faces.py`

---

## §3 — Haemodynamic model: 1D network flow

### 3.1 Segment resistance

Each edge is a rigid cylinder obeying Hagen–Poiseuille:

```
R = 128 · μ · L / (π · d⁴)
```

with *L* the centreline length in µm, *d* the assigned diameter in µm, and *μ* an apparent
viscosity in cP.

**The *d*⁻⁴ term governs everything downstream.** A fractional calibre error becomes roughly four
times itself in resistance. §13.1 gives the measured size of that; it is not restated here.

### 3.2 Two viscosity models exist, and which one is in force depends on the path

This is the easiest thing in the pipeline to get wrong.

**The initial assignment uses a power law.** `PoiseuilleModel.set_poiseuille_resistances` computes

```
μ = 1 / d^1.647
```

which is not a viscosity in cP and is not Pries–Secomb. It is an empirical stand-in that produces
the right ordering of resistances but not their physical magnitudes.

**The rheology solver overwrites it.** On entry, `solve_coupled_flow_and_hematocrit` recomputes both
viscosity and resistance for every edge from Pries–Secomb at the systemic haematocrit, discarding
whatever `set_poiseuille_resistances` assigned:

```
μ_app = pries_secomb(d, H = 0.45)
R     = 128 · μ_app · L / (π · d⁴)
```

**The order of operations — which viscosity is in force, when.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | `set_poiseuille_resistances` writes R from the power law | µ = 1/d^1.647 | **On** | `poiseuille.py:160` |
| 2 | Rheology solver **overwrites** R from Pries–Secomb at systemic Hct | H = 0.45, µ_plasma 1.2 cP, in vivo law | **On**, before the loop | `rheology.py:196` |
| 3 | Each Picard pass recomputes µ_app from the edge's current Hct | in vivo Pries–Secomb | **On** | `rheology.py:349` |
| 4 | R **rescaled** as `original_resistance × µ_app / µ_old` | µ_old = 1/d^1.647 | **On** — see the warning below | `rheology.py:363` |
| 5 | `original_resistance` captured once, on the first pass through step 4 | — | **On**, never updated after | `rheology.py:355` |

> ⚠ **Open item 12 — step 4 double-applies viscosity, inflating every resistance by roughly
> 200–540×.**
>
> The rescale is correct *if* `original_resistance` holds the power-law resistance, because
> `R_old × µ_app/µ_old` then telescopes to `128 µ_app L / (π d⁴)`. It does not. Step 2 overwrites
> `data["resistance"]` with the **Pries–Secomb** value before the loop starts, and step 5 captures
> `original_resistance` from that overwritten value on the first pass. The power-law µ_old therefore
> divides a resistance that no longer contains it, and the surviving factor is
> `µ_PS(d, 0.45) · d^1.647`:
>
> | d (µm) | µ_PS(d, 0.45) cP | µ_old = 1/d^1.647 | Inflation factor |
> |---|---|---|---|
> | 3 | 37.97 | 0.1638 | **232×** |
> | 4 | 21.24 | 0.1020 | **208×** |
> | 5 | 15.14 | 0.0706 | **215×** |
> | 6 | 12.01 | 0.0523 | **230×** |
> | 8 | 8.77 | 0.0326 | **269×** |
> | 12 | 5.95 | 0.0167 | **357×** |
> | 20 | 3.88 | 0.0072 | **539×** |
>
> **It does not cancel from ratios.** The factor is a function of diameter, not a constant, so it
> varies 1.3× across the capillary band (3–8 µm) and 2.3× across the full measured range. Wider
> vessels are penalised hardest, which redistributes flow away from them. This is unlike the uniform
> pressure and viscosity scalings of §4.4 and §8.1, which genuinely do cancel.
>
> **Iteration 0 is clean; every later iteration is not.** The first pressure solve runs on the
> step-2 resistances, which are correct. The corruption enters at the end of iteration 0 and the
> solver converges on the inflated values, so the returned graph carries them.
>
> **A candidate — not a demonstrated — explanation for §13.5.** Absolute perfusion there is 20–100×
> low, and §13.5 computes that reaching 500 µm/s would need about 3,257 mmHg against the 40 mmHg
> used: an implied excess resistance of roughly 81×. That is the same order as the factor above,
> which sits near 230× at the median measured diameter of 6.37 µm. The two are not equal and the
> comparison is loose — velocity is flow-weighted across a diameter distribution — so this is a
> hypothesis to test by re-running with step 4 corrected, not a conclusion. **No H1 or H2 number in
> this document has been re-derived against it.**
>
> **Ratios and topology are unaffected in kind but not in value.** β₁, calibre, length and
> tortuosity are all fixed before any resistance is computed (§2.4–§2.6). Anything downstream of
> the flow solve — shunt ratio, transit time, PO₂ depletion, wall shear stress — is computed on the
> inflated field.

**So the power law survives only if the rheology solve is not run.** When it is run, it is an
initial condition that is replaced rather than blended — which is what §11 row 12 means by
"relaxed by the rescaling step".

> ⚠ **Open item 9 — the rheology solver falls back to 5.0 µm silently.** On initialisation it
> reads `assigned_diameter_um`, then `fwhm_diameter_um`, then defaults to **5.0 µm** without
> raising. A cached graph carrying no calibre therefore solves at a uniform 5 µm for every edge and
> reports nothing. The H2 drivers work around this by loading diameters from
> `per_edge_morphometry.csv` first. Contrast `map_vessels_to_grid`, which raises on a missing
> diameter. The two should behave the same way.

### 3.3 Variable-diameter segments · **Frozen**

Where a segment's diameter varies along its length, resistance is integrated rather than evaluated
once:

```
R = ∫₀ᴸ  128 · μ(d(x)) / (π · d(x)⁴)  dx
```

by trapezoidal quadrature over 1,000 sample points, with `μ(d) = 1/d^1.647` as above.

Two geometries are implemented. **Sphincter**: one constriction at the vessel origin — ramp down
over the first quarter of the constriction length, hold at *d₂* through the middle half, ramp back
up over the last quarter. **Periodic**: the same shape repeated at a fixed spacing.

Both are disabled on the CB path (§10.6), so this integral is not evaluated in any current run.

### 3.4 Network solve

Mass conservation at every internal node is Kirchhoff's current law. With conductance
*C_uv* = 1/*R_uv*, the system is the weighted graph Laplacian **L** = **D** − **C**:

```
L · p = 0     at every unconstrained node
p = p_in      at inlet terminals
p = p_out     at outlet terminals
```

Solved by partitioning into known and unknown nodes and taking the Schur complement:

```
L_uu · p_u = − L_uk · p_k
```

**The order of operations.** On the CB path this runs inside the rheology loop (§4.3) rather than
standalone; `solve_flow_from_conductance_matrix` (`resistance.py:201`) is the equivalent entry point
for graph-only callers and adds VTK export.

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Build the sparse conductance matrix, C_uv = 1/R_uv | — | **On** | `resistance.py:46` |
| 2 | Laplacian **L** = **D** − **C** | — | **On** | `resistance.py:138` |
| 3 | Map inlet node ids → p_in, outlet node ids → p_out | 60/20 mmHg as run (§8.1) | **On** | `rheology.py:219` |
| 4 | Partition indices into known and unknown | — | **On** | `rheology.py:227` |
| 5 | Schur complement: `L_uu · p_u = − L_uk · p_k` | — | **On** | `rheology.py:234` |
| 6 | Direct sparse factorisation | `spsolve`, below 50,000 unknowns | **On** — CB graphs are ~10³ nodes | `resistance.py:152` |
| 7 | On a singular matrix, fall back to least squares | `lsqr` | **On**, and **silent** | `resistance.py:155` |
| 8 | Above 50,000: CG with an **ILU** preconditioner | `drop_tol` 1e-4, `fill_factor` 10, `rtol` 1e-8, `maxiter` 1000 | **Off** — threshold never reached | `resistance.py:161` |
| 9 | Per edge, `Q = (p_u − p_v) / R`, signed | — | **On** | `rheology.py:252` |
| 10 | Direct the edge from high pressure to low | — | **On**, builds the DAG | `rheology.py:261` |

**The iterative branch uses ILU, not Jacobi.** The Jacobi preconditioner described in §9.3 belongs
to the *tissue diffusion* solve (`perfusion.py`), which is a different matrix. The network branch at
step 8 is incomplete-LU, and it never executes on these graphs in any case.

**Step 7 is a silent fallback.** A singular Laplacian — which is what a disconnected component with
no boundary node produces — is caught and answered with a least-squares solution rather than raised.
The graph-level largest-component prune (§2.5 step 8) is what keeps this from firing, so the two are
coupled: turning that prune off would route this path into a silent approximation.

Edge flow then follows directly from `Q = (p_u − p_v) / R`, signed, and the edge is directed from
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

**The flow solve does not work in one unit system.** `R = 128 μ L / (π d⁴)` is evaluated with
pressure in mmHg, viscosity in cP and lengths in µm, so its *Q* carries mmHg·µm⁴/(cP·µm) and is
**not a volumetric flow rate**.

Rewriting *R* wholly in SI multiplies it by `(10⁻³ Pa·s/cP) · (10⁻⁶ m/µm) / (10⁻⁶ m/µm)⁴` = 10¹⁵.
So

```
Q_SI [m³/s] = (ΔP_mmHg · 133.322387415) / (R_pipeline · 10¹⁵)
            = Q_pipeline · 133.322387415 · 10⁻¹⁵
```

and multiplying by 10¹⁸ µm³/m³ leaves

```
POISEUILLE_FLOW_TO_UM3_PER_S = 133.322387415 × 10³
```

**Where it is applied.** In `map_vessels_to_grid`, at the boundary between the 1D solve and the 3D
tissue — the one place the two unit systems have to agree. Passing `flow_to_um3_per_s=1.0` leaves
flow in solver units deliberately, for comparison against output produced in those units.

---

## §4 — Blood rheology

### 4.1 Apparent viscosity and the Fåhræus–Lindqvist effect

Relative apparent viscosity at a discharge haematocrit of 0.45, as a function of diameter in µm:

```
in vivo   μ₄₅ = 6.0 · e^(−0.085 d)  + 3.2 − 2.44 · e^(−0.06 d^0.645)
in vitro  μ₄₅ = 220 · e^(−1.3 d)    + 3.2 − 2.44 · e^(−0.06 d^0.645)
```

They differ **only in the first term**, and that term is the whole disagreement. At *d* = 8 µm and
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

With *f_Q1* the fraction of bulk flow entering branch 1:

```
A = −13.29 · [(d₁²/d₂²) − 1] / [(d₁²/d₂²) + 1] · (1 − H_in) / d₁
B = 1 + 6.98 · (1 − H_in) / d₁

logit(f_E1) = A + B · logit( (f_Q1 − x₀) / (1 − f_Q1 − x₀) )
```

where *f_E1* is the fraction of the **erythrocyte** flux entering branch 1, and *x₀* = 0.05 is the
skimming threshold — red cells effectively fail to enter a branch drawing less than about 5% of the
flow.

**Erythrocyte mass is conserved exactly**: *f_E2* = 1 − *f_E1*. Outlet haematocrits follow as
*H_out* = *H_in* · *f_E* / *f_Q*, then clamped to [0, 0.95].

**Degenerate cases are handled explicitly** rather than by the logit: if either branch takes less
than 10⁻⁶ of the flow, all red cells follow the other.

**Binary bifurcations only** (§11 row 14). A higher-order division mixes proportionally, so
haematocrit heterogeneity is underestimated wherever one occurs.

### 4.3 The coupled flow–haematocrit–viscosity solve

Flow depends on resistance, resistance on viscosity, viscosity on haematocrit, and haematocrit on
flow. The loop closes it by Picard iteration:

1. **Initialise** — every edge at systemic haematocrit 0.45, viscosity from Pries–Secomb,
   resistance from Poiseuille.
2. **Solve** the Laplacian system for nodal pressure (§3.4).
3. **Direct** every edge high-pressure to low-pressure, producing a DAG.
4. **Traverse** the DAG from inlets to outlets, applying §4.2 at every bifurcation to assign child
   haematocrits.
5. **Update** viscosity and resistance from the new haematocrit distribution.
6. **Repeat** until the maximum relative flow change falls below tolerance.

**The order of operations.**

| # | Step | Setting | On the CB path | Where |
|---|---|---|---|---|
| 1 | Every edge set to systemic haematocrit | H = 0.45 | **On**, once | `rheology.py:195` |
| 2 | µ from Pries–Secomb, R from Hagen–Poiseuille | in vivo law | **On**, once | `rheology.py:196` |
| 3 | Diameter read `assigned_diameter_um` → `fwhm_diameter_um` → **5.0 µm** | — | **On** — silent fallback, open item 9 | `rheology.py:191` |
| 4 | Solve the Laplacian for nodal pressure (§3.4) | — | **On**, every iteration | `rheology.py:211` |
| 5 | Per-edge signed flow; direct high → low into a DAG | — | **On**, every iteration | `rheology.py:252` |
| 6 | Convergence test on the max **absolute** flow change | tol 1e-4 | **On**, from iteration 1 | `rheology.py:268` |
| 7 | Topological sort of the DAG | — | **On**; a cycle breaks the loop with a warning | `rheology.py:278` |
| 8 | Force systemic haematocrit at every inlet | H = 0.45 | **On** | `rheology.py:288` |
| 9 | Node haematocrit = flow-weighted mix of inflows | — | **On** | `rheology.py:295` |
| 10 | Degree-2 pass-through: child inherits the mix | — | **On** | `rheology.py:303` |
| 11 | Bifurcation: phase separation (§4.2) | — | **On** | `rheology.py:319` |
| 12 | Trifurcation or higher: **proportional mixing, no skimming** | — | **On** | `rheology.py:333` |
| 13 | Recompute µ_app from the new haematocrit | in vivo law | **On** | `rheology.py:349` |
| 14 | Rescale R by µ_app / µ_old | µ_old = 1/d^1.647 | **On** — ⚠ open item 12, §3.2 | `rheology.py:363` |
| 15 | Wall shear stress from µ_app and \|Q\| | 32µQ/(πd³), mPa → Pa | **On** | `rheology.py:371` |
| 16 | Repeat from step 4 | ≤ 15 iterations | **On** | `rheology.py:207` |

**Three things the numbered summary above does not say.**

**The convergence test is on an absolute flow difference, not a relative one.** Step 6 takes
`max |Q_new − Q_old|` and compares it against 1e-4 — in the flow units of §3.7, not as a fraction.
Whether that is tight or loose therefore depends on the magnitude of the flows themselves, and on
this network's units it is a demanding test rather than a lenient one.

**Junctions above degree 3 get no phase separation.** Step 12 splits haematocrit in proportion to
flow, because the Pries–Secomb phase-separation relation is defined for a Y-split only. Skimming is
therefore absent at every higher-order junction, and after the degree-2 collapse of §2.5 those are
exactly the unresolved multi-way crossings.

**The check runs before the update, so the loop always does at least two passes.** Step 6 is
evaluated at the top of iteration 1 against iteration 0's flows. Since step 14 changes every
resistance by two orders of magnitude between those two passes (open item 12), the iteration-1 test
can never pass, and convergence is reached on the inflated resistances or not at all.

Limits: 15 iterations, tolerance 10⁻⁴.

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

```
C_O₂ = α_O₂ · PO₂  +  H · c_Hb,max · S(PO₂)

S(PO₂) = PO₂ⁿ / (PO₂ⁿ + P₅₀ⁿ)          n = 2.7   [hill_possible_1910]
α_O₂   = 1.34 × 10⁻³ mmol/L/mmHg
c_Hb,max = 0.446 × 20.4 / 0.45 ≈ 20.22 mmol/L     (scaled to pure red cell)
```

Returns zero for PO₂ ≤ 0 rather than extrapolating.

### 5.2 The Bohr effect

P₅₀ is not fixed. It shifts with pH and PCO₂ on the Kelman/Severinghaus empirical form
[`severinghaus_simple_1979`, `kelman_digital_1966`]:

```
log₁₀ P₅₀ = log₁₀(26.0) − 0.4 · (pH − 7.4) + 0.06 · log₁₀( PCO₂ / 40 )
```

Baseline 26.0 mmHg at pH 7.4 and PCO₂ 40 mmHg. Acidosis or hypercapnia raises P₅₀ — the curve
shifts right and haemoglobin releases oxygen more readily.

### 5.3 Carbon dioxide content and the Haldane effect

```
C_CO₂ = α_CO₂ · PCO₂  +  H · [ 11.02 · PCO₂^0.396  +  (0.15 − 0.05 · S_O₂) · PCO₂ ]

α_CO₂ = 0.03 mmol/L/mmHg
```

The first bracketed term is the base carrying capacity; the second is the Haldane shift —
deoxygenated blood carries more CO₂.

> **One inconsistency, deliberate and small.** The saturation *S_O₂* used in the Haldane term is
> evaluated at a **fixed** P₅₀ = 26 mmHg, not the Bohr-shifted value from §5.2. CO₂ carriage is
> therefore underestimated in hypoxic tissue, by under 5% (§11 row 19).

### 5.4 Tissue pH

Henderson–Hasselbalch, with a constant bicarbonate buffer:

```
pH = 6.1 + log₁₀( [HCO₃⁻] / (α_CO₂ · PCO₂) )      [HCO₃⁻] = 24 mmol/L
```

No renal compensation, so the pH response to PCO₂ is fixed by this relation alone. Accepts scalars
or arrays, so it can be applied per grid cell.

### 5.5 Species mismatch

Every constant above is human. The tissue is rat. The direction of the resulting bias is not
established (§11 row 17).

---

## §6 — Tissue transport

### 6.1 The perfusion grid

A regular Cartesian grid over the region, `dims = (nz, ny, nx)` at spacing `res = (rz, ry, rx)`,
with linear index **z-fastest**: `idx = z + y·nz + x·nz·ny`.

**The grid spans the segmented volume, not the graph's extent.** Padding to the segmentation is
what lets tissue with no vessel in it be represented at all; tissue beyond the segmentation is not
represented (§11 row 25).

> **An indexing trap worth knowing.** The stencil assembly reshapes to `(nx, ny, nz)` because a
> C-order reshape makes the *last* axis fastest, matching the z-fastest linear index. Reading
> `dims` as `(nx, ny, nz)` and reshaping the same way gives an arithmetically correct matrix under
> wrong axis names — two reversals that cancel. If the conductances ever look mislabelled, they
> may be; check the arithmetic, not the names.

### 6.2 Vessel-to-grid mapping

Each edge's centreline voxels are point-sampled. For every cell an edge passes through, the mapping
accumulates that edge's length and lateral surface area (2π·r·ΔL) within the cell.

**Flow is then shared by length, not repeated.** Each cell's entry carries a `length_fraction`
normalised against the edge's *accumulated* length, so the shares sum to exactly one:

```
q_total[cell]   = Σ_edges  Q_edge · length_fraction
s_incoming[cell] = Σ_edges  Q_edge · length_fraction · C_O₂(PO₂_art, H_edge)
```

Without this an edge crossing five cells injects five times its own oxygen, and the total source
grows with grid refinement rather than converging.

**Point sampling remains an approximation** to line–plane intersection, so *where* a vessel deposits
carries discretisation error even though the total is conserved (§11 row 24).

**It raises on a missing diameter.** Diameter feeds surface area and therefore every transvascular
flux, so it is not substituted silently. Passing `default_diameter_um` is available and is a
deliberate choice to model unmeasured vessels at a stated calibre. Contrast §3.2's open item 9.

### 6.3 The transport operator — diffusion plus per-cell exchange

**The name "ADR" is loose, and the distinction matters.** The assembled matrix is **pure diffusion**.
There is no advective transport *between* grid cells. Blood delivers oxygen into a cell and carries
it away from the same cell; it does not carry oxygen from one cell to the next.

Diffusive conductance across each cell face, in µm³/s:

```
D_z = σ · (r_y · r_x) / r_z          σ in µm²/s = σ_config × 10¹²
D_y = σ · (r_z · r_x) / r_y
D_x = σ · (r_z · r_y) / r_x
```

assembled as a standard **seven-point stencil** — each cell coupled to its six face neighbours,
with the diagonal accumulating the conductances it participates in. No flux is written at the
domain faces, which is a **Neumann (zero-flux) boundary** by construction (§11 row 23).

**Regularisation.** Pure diffusion under Neumann boundaries has a null space — the rows sum to zero
— so a constant offset is unconstrained. A tiny sink of 10⁻¹² is added to the diagonal at assembly,
and a further 10⁻⁶ before solving.

**Preconditioner.** Jacobi (diagonal), which conjugate gradient requires to be symmetric positive
definite. Non-positive diagonals are detected and reported rather than silently inverted.

### 6.4 Metabolic consumption

```
M(PO₂) = M_max · ( 1 − e^(−k · PO₂) )        k = k_reduce = 0.1 per mmol
```

applied per cell as `M(PO₂) · V_cell`.

**This is not Michaelis–Menten**, which is the literature standard. The forms agree in shape — both
saturate — but differ most at low PO₂, which is exactly the regime the hypoxic fraction is read
from (§11 row 21).

### 6.5 Heterogeneous metabolism from the TH segmentation

`M_max` may be a scalar **or a per-cell array**, and the solver applies it elementwise.

The glomus mask is joined to the grid by **volume fraction per cell**, not by sampling the mask at
the cell centre. At 4 µm against 1.866 µm voxels there are roughly a dozen mask voxels to a cell,
and at the former 10 µm resolution about 154 — so a cell is rarely wholly tissue or wholly stroma,
and a centre sample would discard almost all of the mask and make the answer depend on where cell
centres happened to fall.

Rates are then blended:

```
stroma = BASE_M_MAX / (1 + f̄ · (c − 1))
M_max[cell] = blend( f_TH[cell],  tissue_rate = stroma · c,  stroma_rate = stroma )
```

where *c* is the glomus:stroma contrast and *f̄* the mean TH fraction. **The volume-weighted mean is
held at `BASE_M_MAX` across contrasts**, so runs at different *c* are comparable rather than simply
scaled versions of each other.

**The contrast is a swept parameter, not a measurement** (§11 row 22). The driver defaults to
`c ∈ {1.0, 2.0, 4.0}`.

### 6.6 The three coupling tiers

| Tier | Solver | Couples | Status |
|---|---|---|---|
| 1 | `solve_perfusion_steady_state` | O₂ only; blood as a well-mixed source and sink per cell | **Active** — this is what §2.3 runs |
| 2 | `solve_coupled_1d3d_perfusion` | O₂ across an endothelial permeability barrier | **Implemented, unreachable** — the dispatch is `if multi_species … elif barrier …`, and multi-species is also on |
| 3 | `solve_multi_species_perfusion` | O₂, CO₂ and pH, linked by the respiratory quotient | Reachable; reads Picard settings from config where Tier 1 hard-codes them |

**Tier 1 uses a fixed baseline haematocrit** of 0.45 for the washout, hard-coded rather than read
from the edge (§11 row 27).

### 6.7 Numerical stabilisation of the Picard loop

The non-linear washout acts as a sink on the right-hand side, which is unstable for conjugate
gradient. The loop applies a **pseudo-washout**: add `q_total · γ` to the LHS diagonal *and the
same term* to the RHS.

```
b = s_incoming − s_washout − M(PO₂)·V_cell + (q_total · γ · PO₂)
```

The true steady-state roots are unchanged, but the matrix becomes strictly diagonally dominant and
well conditioned. γ = 0.5 in Tier 1, damping the Picard step and preventing sigmoidal oscillation;
γ = 1.0 in Tier 3.

**PO₂ is clamped to ≥ 0** at each iteration. Negative values are non-physical and drive Picard
oscillation.

The loop warns rather than raising if it hits its iteration cap without reaching tolerance.

### 6.8 Grid resolution

Median PO₂ moves 27.34 → 27.92 → 28.21 at 10, 6 and 4 µm — increments halving each time and
extrapolating to about 28.5. **4 µm is within roughly 1% of that limit at a twenty-seventh of the
cost** of native resolution, and is what §2.3 runs at.

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
| β₁ (fundamental loops) | E − V + components | **Yes** — the H1 §1.1 readout |
| Tortuosity index | Path length / straight-line distance, per edge | Yes |
| Curvature | Per edge, from the smoothed centreline | Yes |
| Branching angle | Angle between every neighbour pair at nodes of degree ≥ 3 | Diagnostic |
| Branching points | Count of nodes with degree > 2 | Yes |
| Tree asymmetry | Partition asymmetry index | Diagnostic |
| Fractal dimension | Box counting | Diagnostic |
| Path efficiency | Shortest-path vs Euclidean over node pairs | Diagnostic |
| Betweenness, communities | Graph-theoretic centrality and modularity | Diagnostic |

**Tortuosity is derived from the per-edge table rather than recomputed**, so the summary and the
per-edge CSV cannot disagree about what an edge's tortuosity is.

> ⚠ **"Vessel density" means two different things, and neither is parenchymal density.**
> `compute_vessel_density` reports *Density in Tissue* as total length divided by the **bounding box
> of the node positions**, and *Density in Whole Image* as total length divided by the full image
> volume. The first is a graph-extent density; the second includes everything that is not tissue.
> The parenchymal quantity H1 §1.3 asks for is the one in §7.2, not either of these.

> **At a glance** — graph-only metrics, tortuosity shared with the per-edge table ·
> `stats.py:147`, `stats.py:349`, `stats.py:213` · `tests/test_statistics.py`,
> `tests/test_synthetic_network_statistics.py`

---

### 7.2 Two-channel morphometry

Joins the vessel channel to the TH channel. **Sound only because they are two channels of one
acquisition** — identical grid, co-registered by construction, no registration step (§11 row 28).

**Centreline length within the glomus.** Length is summed over real steps, not by counting skeleton
voxels:

```
length = Σ_steps  |step ⊙ voxel_um| · count(step)
```

Counting voxels and multiplying by voxel size is the obvious estimator and is wrong by up to √3 — a
diagonal step covers 3.23 µm on this grid where an axial one covers 1.87 µm. On a tortuous network
that is not a small correction, and H1 §1.4 turns on tortuosity, so the two must not disagree about
what length means.

**Steps at the mask boundary count for neither side.** A step is included only when *both*
endpoints lie inside the mask. Assigning a straddling step to the tissue it half touches would
inflate whichever mask is more fragmented.

**Tissue-to-vessel distance.** Euclidean distance transform from every TH-positive voxel to the
nearest **centreline** voxel, with `sampling` set to the voxel size so the result is in µm directly.

> **To the centreline, not the vessel surface.** The two differ by the local radius. On a 3 µm
> capillary the surface is 1.5 µm closer everywhere, and that offset would be absorbed into any
> group difference rather than appearing as one. H1 §1.5 asks for the centreline distance.

**It raises on an empty centreline** rather than returning infinity, which would propagate as a very
large distance instead of as an error.

> **At a glance** — real-step length, centreline distance, boundary steps excluded · median TVD
> 5.3–7.9 µm (§13.7) · `th_morphometry.py:34`, `th_morphometry.py:78` ·
> `tests/test_th_morphometry.py`

---

### 7.3 Functional shunting (H2 §2.1)

**The question.** Does steady-state flow bypass the capillaries that penetrate the TH-positive
clusters, running instead through thoroughfare channels?

**Edge classification.** An edge counts as *penetrating* when at least **50%** of its centreline
length lies inside the TH mask, sampled along the **whole polyline**.

> **Why not an endpoint test.** A capillary penetrating a cluster usually starts and ends in
> stroma. An endpoint test would classify exactly the vessels the question is about as
> extra-glomus.

**The quantity is a shunt index, not a flow share:**

```
shunt index = (flow share penetrating) / (edge share penetrating)
```

An index of 1 means flow is indifferent to the clusters. Below 1 means flow is carried
preferentially by the vessels that bypass them — the shunting the method is trying to detect.

**Flow share alone cannot answer it**, because flow share tracks how many edges penetrate, which is
itself downstream of the parenchymal volume difference H1 §1.3 reports. The ratio removes that.

**This is a within-specimen ratio**, so it sits under the ±6.3% floor rather than the ±45% one
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

Haematocrit is produced by the phase-separation model, so this quantity inherits every assumption in
§4.2 — in particular that separation occurs at binary bifurcations only.

---

### 7.5 Glomus-specific hypoxic fraction (H2 §2.3)

**The question.** With a higher metabolic rate assigned to TH-positive voxels and a lower one to
stroma, what fraction of the TH-positive volume falls below a hypoxic PO₂?

**How.** Solve the tissue field on the heterogeneous grid (§6.5), then take the fraction of
TH-weighted cell volume below each threshold. Thresholds swept at **5, 10 and 20 mmHg**; metabolic
contrast swept at **1×, 2× and 4×**; grid at 4 µm.

> ⚠ **Read §13.6 before using this.** The tissue is not diffusion-limited — the oxygen diffusion
> length is 20–45 µm against a median tissue-to-vessel distance of 5.3–7.9 µm. Raising the glomus
> rate to four times stromal moves PO₂ inside the TH volume by **0.01 mmHg**. The mechanism this
> method is built on cannot operate on this geometry.
>
> The output is still meaningful as a curve in the assumed contrast. It is not a number.

---

### 7.6 Transit time and PO₂ depletion (H2 §2.4)

**Per-edge transit time** is lumen volume over volumetric flow:

```
τ_edge = π · (d/2)² · L / Q
```

**Quadratic in diameter**, where resistance is quartic — so this carries a different sensitivity to
calibre than the flow solve does, and its own share of the floor in §13.3.

**An edge carrying no flow gets `inf`, not a large number.** Blood that does not move does not
arrive, and a finite stand-in would propagate as a merely slow path.

**Path transit time** is the minimum accumulated τ from any inlet, by Dijkstra over the
flow-directed graph.

> **Reported as a ratio, never an absolute.** Two independent reasons, and they compound. An
> absolute flow quantity sits under the ±45% floor from calibre alone (§13.3). And the pressure,
> viscosity and length units are not reconciled to one system (§3.7), so the magnitude is in
> arbitrary units. Both have the same answer: compare transit time to one set of terminals against
> another, computed identically, and the shared error divides out.

**It raises on a missing diameter** rather than substituting one.

> **At a glance** — τ = πr²L/Q, Dijkstra from inlets, `inf` for zero flow, ratios only ·
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

> ⚠ **Open item 10 — the H2 methods do not use the config pressures.** `HaemodynamicsConfig`
> declares 100 mmHg in and 2 mmHg out — MAP to CVP. The H2 drivers use **60 mmHg to 20 mmHg**,
> arteriolar to venular, across the same sub-volume.
>
> The driver value is the more defensible of the two: placing the full systemic gradient across
> roughly 1 mm of tissue is what §11 row 15 flags. But the two disagree by a factor of 2.45 in
> driving pressure, every published H2 number used 60/20, and nothing in the config records that.
>
> Resolve before quoting any absolute flow. It does not affect the within-specimen ratios, which
> are the reportable quantities anyway.

---

## §8 — Boundary and initial conditions

The mechanism for *choosing* boundary nodes is §2.8. This section is about what is *imposed* on them
once chosen, and what happens to everything else.

### 8.1 Pressure boundary conditions

Dirichlet at both ends: a fixed pressure at inlet terminals, a fixed pressure at outlet terminals,
and nothing imposed anywhere else.

| Source | Inlet | Outlet | Gradient | Interpretation |
|---|---|---|---|---|
| `HaemodynamicsConfig` | 100 mmHg | 2 mmHg | 98 mmHg | Systemic MAP to central venous pressure |
| **H2 drivers (what ran)** | **60 mmHg** | **20 mmHg** | **40 mmHg** | Arteriolar to venular |

**These disagree by 2.45× in driving pressure, and every published H2 number used the second row.**
See open item 10.

**What the config value assumes.** That the entire arterial-to-venous pressure drop of the systemic
circulation falls across roughly 1 mm of tissue. It does not — most of it falls across the arterial
tree upstream and the venous tree downstream. This overestimates perfusion pressure and therefore
absolute flow (§11 row 15). The driver's arteriolar-to-venular pair is the more defensible framing
of the same sub-volume.

**Neither choice rescues absolute perfusion.** §13.5 measures flow-weighted velocities of 4–10 µm/s
against a physiological 200–1,000, and reaching 500 µm/s would require about 3,257 mmHg. The
boundary pressure is not what is missing.

**Within-specimen ratios are insensitive to this.** Flow is linear in the pressure difference, so a
uniform change scales every edge's flow and cancels from any ratio taken within one specimen — the
same cancellation §4.4 describes for viscosity.

### 8.2 What happens to terminals that are not boundaries

**About 86% of degree-1 nodes are interior** — nowhere near a region face. Counting terminals within
one voxel of each of the six ROI faces:

| Specimen | Terminals | On any face | Interior | Interior share |
|---|---|---|---|---|
| WKY-A | 544 | 71 | 473 | 86.9% |
| WKY-B | 534 | 88 | 446 | 83.5% |
| WKY-C | 674 | 91 | 583 | 86.5% |
| SHR-A | 545 | 80 | 465 | 85.3% |
| SHR-B | 754 | 104 | 650 | 86.2% |
| SHR-C | 503 | 67 | 436 | 86.7% |

**The crop is not the boundary problem; interior dead ends are.** These are skeletonisation spurs
and segmentation breaks, not vessels severed by the ROI. A real capillary bed has few genuine
interior dead ends, so this is a statement about mask quality rather than about the crop.

Three modes decide what happens to them:

| Mode | Behaviour | Consequence |
|---|---|---|
| **`caged`** (default) | Interior terminals are not boundaries at all | They become no-flow dead ends. Under the band rule roughly **half of all terminals** were stranded this way |
| `universal_sink` | Every non-inlet terminal becomes an outlet | No stranding, but every mask defect becomes a drain |
| `robin_resistance` | Non-boundary terminals are tagged for a distal resistance | A middle course; multiplier 10.0, unswept |

**Why the inlet:outlet ratio matters.** Under a fixed pressure boundary it directly scales how much
flow the network carries. Measured under the band rule the ratio spanned **10.7×** across six
specimens, from 2.67 to 0.25, with group means 1.87 (WKY) against 0.88 (SHR). That is the right size
and the right direction to become a confound, and it is set by ROI placement rather than by biology.
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
pseudo-washout of §6.7 also contributes diagonal dominance.

**Grid extent.** The grid is padded to span the segmented volume rather than the graph's own extent
(§6.1). Tissue beyond the segmentation is not represented at all — it is outside the domain, not
merely unperfused.

### 8.5 Blood gas inlet conditions

Values carried by blood entering the tissue, all constants (§10.8, §10.9):

| Quantity | Value | Notes |
|---|---|---|
| Arterial PO₂ | 100 mmHg | Also hard-coded in two solver bodies — open item 3 |
| Arterial PCO₂ | 40 mmHg | |
| Systemic haematocrit | 0.45 | Tier 1 washout hard-codes the same value — open item 4 |
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
| Tissue pH (Tier 3) | 7.4 |

**PO₂ starting at zero is deliberate.** It approaches the steady state from below, and each iterate
is clamped to ≥ 0, so the sequence cannot enter the non-physical region that drives Picard
oscillation.

The rheology loop starts every edge at systemic haematocrit 0.45 with the corresponding
Pries–Secomb viscosity (§4.3) — likewise a starting guess, replaced on the first pass.

> **At a glance** — Dirichlet pressure at face terminals, everything else caged; Neumann on tissue ·
> config 100/2 mmHg but H2 ran 60/20; 86% of terminals interior · `boundaries.py:88`,
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
| Rheology (§4.3) | Viscosity depends on haematocrit, which depends on flow | None; 15 iterations at 10⁻⁴ |
| Tissue Tier 1 (§6.7) | Metabolic sink and venous washout both depend on PO₂ | γ = 0.5 |
| Tissue Tier 3 | The same, coupled across O₂, CO₂ and pH | γ = 1.0 each |

**Why Picard rather than Newton.** No Jacobian is assembled anywhere. Picard costs one linear solve
per iteration and converges reliably here because the non-linearities are saturating and
monotonic — an exponential approach in the metabolic sink, a sigmoid in the Hill equation. Newton
would converge faster but requires derivatives of the blood-gas chemistry that nothing currently
provides.

**The stabilisation is the interesting part.** The pseudo-washout of §6.7 moves a term from the
right-hand side onto the diagonal and adds it back on the right, leaving the steady-state roots
unchanged while making the matrix strictly diagonally dominant. This converts a system conjugate
gradient handles badly into one it handles well, without changing the answer.

### 9.3 Linear solvers

| System | Solver | Why |
|---|---|---|
| Network Laplacian | Direct below 50,000 nodes, iterative above | CB graphs sit far below the threshold, so the flow solve is exact to machine precision |
| Tissue diffusion | Conjugate gradient, Jacobi preconditioned | The matrix is large, sparse, symmetric and positive definite once regularised |

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
| `hysteresis_threshold_low` | 0.65 | probability | (iii) | **Provisional.** Not tuner-derived — the preprocessing objective rises monotonically across the plausible band, so its argmin is the top of the search range rather than a property of the data. Set instead from calibre (r_p90 4.17 µm, right scale for a ~3 µm capillary) and connectivity (component count stable 0.60–0.73, explodes above 0.80) | measured, in sensitivity scope |
| `hysteresis_threshold_high` | 0.75 | probability | (iii) | Provisional, as above | measured, in sensitivity scope |
| `enable_hole_filling` | True | — | (iii) | — | unswept |
| `ilastik_vessel_channel` | 0 | index | (i) | Classifier output layout | — |
| `enable_shannon_entropy` | True | — | (iii) | **Inert** — the vessel classifier has 2 classes, so the joint path is skipped (§2.3, open item 11) | no effect |
| `shannon_entropy_threshold` | 0.95 | normalised entropy | (iii) | Chosen. Max entropy for a *candidate* voxel; permissive gate | no effect on the CB path |
| `shannon_entropy_core` | 0.6 | normalised entropy | (iii) | Chosen. Max entropy for a *seed* voxel; strict gate. **Not a `PreprocessingConfig` field** — only reachable through the auto-tuner | no effect on the CB path |

> ⚠ **Open item 1 — two different thresholds are in play.** The config defaults above are
> 0.65 / 0.75. The H1 cohort runs instead froze the vessel threshold at **0.90**, which is not
> arbitrary: it is the median of the six per-specimen selections, snapped to the sweep grid
> (§2.2). Passed as `--hysteresis-low`, it auto-raises the high bound to 0.95. So the config
> defaults are *never* the values H1 ran at, and anything reading `PreprocessingConfig` alone
> will describe a segmentation that did not happen. Which value is in force depends entirely on
> which driver ran. Resolve before quoting either.

### 10.3 Skeletonisation and topology — `SkeletonConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `closing_radius` | 1 | voxels | (iii) | Chosen | unswept |
| `bridge_gap_size` | 1 | voxels | (iii) | Chosen. Adds a uniform foreground shell to the mask — the same wall EDT measures distance to, so it feeds directly into calibre | unswept |
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
| `edge_percent` | 25.0 | % | (iii) | Band-rule parameter — see open item 2 | measured (§13) |
| `end_percent` | 25.0 | % | (iii) | Band-rule parameter — see open item 2 | measured (§13) |
| `node_edge_axis` | 0 | axis index | (iii) | See open item 2 — the boundary helper's own default is axis 1 | measured |

> ⚠ **Open item 2 — two boundary rules coexist, and the axis defaults disagree.** `GraphConfig`
> still carries the band-rule parameters (`edge_percent`, `end_percent`, `node_edge_axis = 0`),
> while the H2 drivers select the face-crossing rule on **axis 1** — the only axis with terminals
> on both faces in all six specimens. `select_boundary_nodes_by_method` itself defaults to
> `axis = 1`, `edge_percent = 10.0`, `end_percent = 10.0`, none of which match `GraphConfig`.
> So the rule *and* its parameters depend on the caller. This is the largest single sensitivity
> in the model (§13), so it must be resolved and stated once, not left to the driver.

### 10.5 Calibre assignment — `HaemodynamicsConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `radius_assignment_mode` | "edt_radius" | — | (iii) | Chosen on measured evidence over `fwhm_radius`. Both run over the same 1330 edges: EDT 100% coverage, median 6.37 µm, p95 11.34, max 20.09; FWHM 76.5%, median 8.20 µm, p95 16.78, max 39.16. They correlate weakly (Pearson r = 0.245, Spearman ρ = 0.284; median ratio FWHM/EDT = 1.359). FWHM's tail is not physical for a bed whose measured inscribed radius is p99 5.60 µm | measured |
| `constant_radius_um` | 5.0 | µm | (iii) | Used only under `constant_radius` mode | — |
| `edt_junction_proximity_exclusion_um` | 3.73 | µm (2 voxels) | (iii) | Within ~one radius of a bifurcation the EDT returns the junction's inscribed sphere, biasing radius upward. Specified externally as 2 voxels; ≈ one capillary inscribed radius. The swept optimum is nearer 1.5 voxels, but 2.80 µm would be tuned to one subvolume and the difference is 0.3% on resistance | measured (0.93→5.60 µm swept; effect on resistance ~4–8%) |
| `MAX_SYNTHETIC_FRACTION_EDT` | 0.0 | fraction | (iii) | EDT has no legitimate per-edge failure mode on a mask that covers the vessel — 100% measured provenance was observed across 34,900 edges — so any fallback is a defect, not an expected shortfall. FWHM is exempt by default because Gaussian fitting genuinely fails on individual edges | measured |
| `fwhm_sample_spacing_along_edge_um` | 2.0 | µm | (iii) | Chosen | unswept |
| `fwhm_transverse_profile_step_um` | 0.5 | µm | (iii) | Chosen | unswept |
| `fwhm_transverse_half_extent_um` | 15.0 | µm | (iii) | ≈8 voxels, ≈4.7 vessel radii. In a bed this dense the transverse profile runs into neighbouring vessels, which is one of the two mechanisms behind FWHM's inflated tail | measured (indirectly) |

### 10.6 Constriction geometry — `HaemodynamicsConfig` · **Frozen**

Present in the code, disabled for the CB path, and `__post_init__` raises if re-enabled.

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `constrict_at_pericytes` | False (raises if True) | — | (iii) | Sites are placed by a hard-coded topological rule rather than measured from imaging, and severities come from no model of vasomotor tone. Because the ratio multiplies whatever diameter was measured, the fabrication reaches measured edges too — and resistance goes as *d*⁻⁴, so the 0.5 capillary ratio is a 16× local resistance error on a real vessel | measured |
| `constriction_mode` | "sphincter" | — | (iii) | Alternative: "periodic" | — |
| `sphincter_length_um` | 5.0 | µm | (iii) | Chosen | — |
| `intimal_cushion_constriction_ratio` | 0.60 | fraction | (iii) | Chosen; no vasomotor model behind it | — |
| `pre_capillary_constriction_ratio` | 0.50 | fraction | (iii) | Chosen; no vasomotor model behind it | — |
| `pre_capillary_topological_offset` | 1 | branch orders | (iii) | Chosen | — |

### 10.7 Haemodynamics — pressures, viscosity, units

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `input_p_bc` | 13.332 × 10⁶ | mPa (= 100 mmHg) | (i) | Systemic MAP `[CITE]` | assumed |
| `output_p_bc` | 0.27 × 10⁶ | mPa (= 2 mmHg) | (i) | Central venous pressure `[CITE]` | assumed |
| `blood_plasma_viscosity_cP` | 1.2 | cP | (i) | Plasma viscosity `[CITE]` | assumed |
| Viscosity law | `in_vivo` | — | (ii) | Pries et al. 1994, fitted to microvessels in living tissue, where the endothelial surface layer narrows the effective lumen [`pries_resistance_1994`]. `in_vitro` (Pries et al. 1992, glass tubes) is available and not default [`pries_blood_1992`] | measured — the two differ by ≈3.4× apparent viscosity at D = 8 µm, but a 3–4× change moved no within-specimen ratio (§13) |
| μ₄₅ in vivo | 6.0·e^(−0.085 d) + 3.2 − 2.44·e^(−0.06 d^0.645) | relative | (ii) | [`pries_resistance_1994`] | — |
| μ₄₅ in vitro | 220·e^(−1.3 d) + 3.2 − 2.44·e^(−0.06 d^0.645) | relative | (ii) | [`pries_blood_1992`] | — |
| Phase separation | Pries bifurcation relation | — | (ii) | [`pries_red_1989`], fitted to 65 arteriolar bifurcations in rat mesentery | unswept |
| `PASCALS_PER_MMHG` | 133.322387415 | Pa/mmHg | (i) | Exact by definition of the conventional millimetre of mercury | exact |
| `POISEUILLE_FLOW_TO_UM3_PER_S` | 133.322387415 × 10³ | (µm³/s) per solver unit | (i) | Derived. The solve evaluates *R* = 128 μL/(π d⁴) with pressure in mmHg, viscosity in cP and lengths in µm, so its *Q* carries mmHg·µm⁴/(cP·µm) and is not a volumetric rate. Rewriting *R* in SI multiplies it by 10¹⁵ | exact |
| `rheology_max_iterations` | 15 | iterations | (iii) | See Appendix A | unswept |
| `rheology_tolerance` | 1 × 10⁻⁴ | relative | (iii) | See Appendix A | unswept |
| Murray's law | **not used** | — | — | The branch-order fallback is exponential, not Murray scaling [`murray_physiological_1926`] | — |

### 10.8 Blood gas chemistry — hard-coded in `perfusion.py`

These are **not** configurable. They live in the function bodies.

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `alpha_o2` | 1.34 × 10⁻³ | mmol/L/mmHg | (i) | O₂ solubility in plasma `[CITE]` | assumed |
| `hill_n` | 2.7 | — | (i) | Hill coefficient [`hill_possible_1910`] | assumed |
| `c_hb_max` | 0.446 × 20.4 / 0.45 ≈ 20.22 | mmol/L | (i) | Haemoglobin O₂ capacity scaled to pure RBC `[CITE]` | assumed |
| Baseline P₅₀ | 26.0 | mmHg | (i) | At pH 7.4, PCO₂ 40 mmHg. **Human** haemoglobin `[CITE]` | assumed |
| Bohr pH coefficient | −0.4 | per pH unit (log₁₀ P₅₀) | (ii) | [`severinghaus_simple_1979`], [`kelman_digital_1966`] | assumed |
| Bohr PCO₂ coefficient | +0.06 | per log₁₀(PCO₂/40) | (ii) | [`severinghaus_simple_1979`], [`kelman_digital_1966`] | assumed |
| `alpha_co2` | 0.03 | mmol/L/mmHg | (i) | CO₂ solubility in plasma `[CITE]` | assumed |
| CO₂ base capacity | 11.02 · PCO₂^0.396 | mmol/L | (ii) | Spencer (1979) empirical CO₂ dissociation curve `[CITE — not in bibliography]` | assumed |
| Haldane shift | (0.15 − 0.05·S_O₂) · PCO₂ | mmol/L | (ii) | Same source `[CITE]`. Saturation is evaluated at fixed P₅₀ = 26, not the Bohr-shifted value | assumed |
| `pKa` | 6.1 | — | (i) | Henderson–Hasselbalch `[CITE]` | assumed |

> ⚠ **Species mismatch.** The haemoglobin parameters above are human. The tissue is rat. Direction
> of the resulting bias is not established.

### 10.9 Tissue transport — `PerfusionConfig`

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `do_perfusion_modeling` | True | — | — | — | — |
| `grid_resolution_xyz` | (10, 10, 10) default; **4 µm** for H2 §2.3 | µm | (iii) | 4 µm chosen on convergence: median PO₂ 27.34 / 27.92 / 28.21 at 10 / 6 / 4 µm, increments halving, extrapolating to ≈28.5. 4 µm is within ~1% of that limit at 1/27 the cost of native resolution | measured |
| `sigma_diff` | 1.5 × 10⁻⁹ | m²/s | (i) | O₂ diffusivity in tissue `[CITE]` | assumed |
| `sigma_diff_co2` | 3.0 × 10⁻⁸ | m²/s | (i) | CO₂ diffuses ≈20× faster than O₂ `[CITE]` | assumed |
| `permeability_o2_cm_s` | 1.0 × 10⁻⁴ | cm/s | (ii) | Endothelial O₂ permeability `[CITE]` | assumed |
| `permeability_co2_cm_s` | 2.0 × 10⁻³ | cm/s | (ii) | Endothelial CO₂ permeability `[CITE]` | assumed |
| `respiratory_quotient` | 0.82 | — | (i) | CO₂ produced per O₂ consumed `[CITE]` | assumed |
| `systemic_hematocrit` | 0.45 | fraction | (i) | `[CITE]` | assumed |
| `po2_arterial_mmHg` | 100.0 | mmHg | (i) | `[CITE]` — but see open item 3 | assumed |
| `pco2_arterial` | 40.0 | mmHg | (i) | `[CITE]` | assumed |
| `hco3_tissue` | 24.0 | mmol/L | (i) | Fixed bicarbonate buffer; no renal compensation `[CITE]` | assumed |
| `M_max` | **config 0.005; H2 driver 0.05** | mmol/L/s | (iii) | Maximum metabolic consumption rate. The two disagree by 10× — see open item 8. The driver's 0.05 is the defensible one: it is 0.067 mL O₂ per mL per minute against roughly 0.040 for brain, the right order for a metabolically active organ | unswept in magnitude; the glomus:stroma *ratio* is swept |
| `k_reduce` | 0.1 | per mmol | (iii) | Phenomenological metabolic reduction in hypoxic zones. **Not Michaelis–Menten** — that form is used nowhere in the pipeline, and the two differ most in the low-PO₂ regime, which is exactly where §2.3 reads its answer | unswept |
| `C_arterial` | 0.13 | mmol/L | (iii) | **Dead configuration.** Declared in three places (`PerfusionConfig` and the two H2 driver `PerfConfig` classes) and read nowhere in `src/` or `examples/`. Superseded in practice by the blood-gas path, which computes arterial oxygen content from PO₂ and haematocrit | n/a |
| `use_endothelial_barrier_model` | True | — | — | **Implemented, unreachable.** The dispatch is `if use_multi_species_model: … elif use_endothelial_barrier_model: …`, and multi-species is also True by default, so the `elif` never fires. Setting this flag alone changes nothing | — |
| `use_multi_species_model` | True | — | — | Selects the O₂/CO₂/pH solver | — |
| Glomus : stroma metabolic ratio | swept, not fixed | — | (iii) | **Nothing in this study measures it.** §2.3 reports the hypoxic fraction across a range of it rather than at one value | measured by sweep |

> ⚠ **Open item 3 — arterial PO₂ is set in two places.** `PerfusionConfig.po2_arterial_mmHg`
> exists, but `po2_arterial = 100.0` is also hard-coded inside two solver bodies. If the config
> value is changed, one or both solvers may ignore it. Trace before treating it as a knob.
>
> ⚠ **Open item 4 — baseline haematocrit is duplicated too.** `h_baseline = 0.45` is hard-coded
> in the Tier 1 washout path, duplicating `systemic_hematocrit`. In Tier 1 the washout is
> therefore decoupled from local haematocrit.
>
> ⚠ **Open item 5 — `C_arterial` is dead configuration.** Confirmed: declared three times, read
> nowhere. Either wire it up or delete it; leaving it in a config invites someone to set it and
> expect an effect.

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
| 12 | Newtonian fluid at initialisation | §4.3 | Biases initial resistances; largely relaxed by the resistance-rescaling step |
| 13 | Rheological correlations transferred from rat mesentery | §4.1–§4.2 | Transferability to carotid body microvasculature **unquantified** |
| 14 | Phase separation occurs at binary bifurcations only | §4.2 | Higher-order divisions mix proportionally → haematocrit heterogeneity **underestimated** |
| 15 | A systemic-scale pressure gradient falls across the imaged sub-volume | §8 | The config declares MAP-to-CVP, ~98 mmHg across roughly 1 mm, which **overestimates** perfusion pressure. The H2 drivers instead use 60→20 mmHg, arteriolar to venular, and every published H2 number used that. See open item 10 |
| 16 | No vasoregulation of any kind | §3.3 | Constriction is disabled entirely, so there is neither active feedback (myogenic, metabolic, shear-mediated) nor a static constriction geometry. The network is a fixed passive resistor array |

### 11.3 Blood gas chemistry

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 17 | Human haemoglobin parameters applied to rat tissue | §5.1–§5.2 | Species mismatch. Direction **not established** |
| 18 | Constant bicarbonate buffer; no renal compensation | §5.4 | Fixes the pH response to PCO₂ |
| 19 | Haldane saturation evaluated at fixed P₅₀ = 26 mmHg rather than the Bohr-shifted value | §5.3 | CO₂ carriage **underestimated** in hypoxic tissue; magnitude small (<5%) |

### 11.4 Tissue transport

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 20 | Uniform tissue diffusivity | §6.3 | A single scalar σ for the whole domain. Smooths the PO₂ field across tissue types. **Note:** metabolic rate is *not* uniform — see row 22 |
| 21 | Metabolism is phenomenological, M(PO₂) = M_max·(1 − e^(−k·PO₂)) | §6.4 | Not Michaelis–Menten, which is the literature standard. The two differ most in the low-PO₂ regime — exactly where §2.3 reads its answer |
| 22 | The glomus-to-stroma metabolic ratio | §6.5 | **Nothing in this study measures it.** Reported across a swept range rather than at one value, so the hypoxic fraction is a curve in this parameter, not a number |
| 23 | Neumann tissue boundary; no exchange beyond the imaged volume | §6.3 | Tissue PO₂ **overestimated** near the domain boundary |
| 24 | Vessel-to-grid mapping is point-sampled along the centreline | §6.2 | An approximation to line–plane intersection, so *where* a vessel deposits carries discretisation error. The *total* is conserved: shares are normalised by accumulated length, so an edge's flow sums to exactly one across the cells it crosses |
| 25 | The grid spans the segmented volume, not the graph's extent | §6.1 | Tissue beyond the graph is represented; tissue beyond the segmentation is not represented at all |
| 26 | No lymphatic drainage or interstitial fluid flow | §6.3 | Omits a minor transport pathway |
| 27 | Fixed baseline haematocrit in the Tier 1 washout | §6.6 | `h_baseline = 0.45` is hard-coded, so Tier 1 washout is decoupled from local haematocrit. Tier 1 only |

### 11.5 Study design and reporting

| # | Assumption | Enters at | Expected direction of effect |
|---|---|---|---|
| 28 | The two channels are co-registered by construction | §7.2 | Two channels of one acquisition on an identical grid, with no registration step. This is what makes a join between them sound; it would not hold across separate acquisitions |
| 29 | Absolute flow quantities are reported only as within-specimen ratios | §7.6 | Not a modelling assumption but a reporting rule forced by two of them — the ±45% calibre floor and the unreconciled unit magnitude. See §13 |
| 30 | Boundary terminals are selected on axis 1 only | §8 | The only axis with terminals on both faces in all six specimens. A specimen whose true inflow is off-axis is served by the wrong terminals |

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
| Radial point source | Qualitative 1/r decay | Bracketed |
| Krogh cylinder radial diffusion | Analytic radial profile | Bracketed |
| Plasma skimming | Erythrocyte mass conservation | 10⁻⁸ |
| Skimming direction | Inequality (larger branch takes more) | Qualitative |
| Fåhræus–Lindqvist curve | Curve shape, inequality chain | Qualitative |
| In vivo viscosity monotonicity | Rises as vessels narrow | Qualitative |
| Bohr and Haldane shifts | Direction only | Qualitative |
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
| Transit time | τ = πr²L/Q by hand; `inf` for zero flow | Exact |
| Cohort split | Exact permutation p against enumeration | Exact |
| Threshold selection | Refuses when calibre unreachable | Behavioural |

Rows in **bold** postdate the earlier coverage table and close the gaps it recorded.

### 12.3 What is verified weakly

Stated as fact, not softened:

- **Directionally only** — the apparent viscosity curve, the skimming output *value* (its mass
  conservation is exact; its magnitude is not checked against a target), and the Bohr and Haldane
  shifts.
- **Transitively only, through integration tests rather than directly** — the resistance rescaling
  rule, the branch-order diameter formulae, the default boundary permeability mode, and the
  numerical Hill inversion.
- **Bracketed rather than to a tolerance** — the radial point source and the Krogh cylinder.

### 12.4 The two gaps

**No grid-convergence or order-of-accuracy study exists for any PDE solver.** Every result runs at a
single fixed resolution. §6.8 records that PO₂ converges as the grid refines, which is evidence of
convergence but not a measured *order*. The cheapest closing move is the zero-order metabolism case,
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

### 13.1 The governing constraint: resistance goes as *d*⁻⁴

Fractional calibre error propagates as `δR/R ≈ 4·δd/d`. Measured over the pooled 34,900 edges,
taking one voxel (1.866 µm) as the diameter uncertainty:

| Percentile | Diameter (µm) | δd/d | δR/R |
|---|---|---|---|
| p5 | 3.732 | 50.0% | **200.0%** |
| p25 | 5.868 | 31.8% | **127.2%** |
| p50 | 7.904 | 23.6% | **94.4%** |
| p75 | 10.550 | 17.7% | **70.8%** |
| p95 | 13.963 | 13.4% | **53.5%** |

- **95.9%** of edges carry more than 50% resistance uncertainty
- **37.2%** carry more than 100%
- The measured p5–p95 calibre spread of 3.74× becomes a **196× spread in resistance**

The network's resistance structure is dominated by a quantity measured to roughly a quarter of its
own value. **This is not fixable in the solver** — not by the Picard iteration, the ADR
discretisation, or the rheology.

> **Not a quantisation problem.** The pooled diameters take 823 distinct values with a median gap
> of 0.0023 µm — junction trimming and B-spline smoothing break the raw EDT lattice, so the values
> are numerically dense. The 1.87 µm figure is the scale below which a difference is not
> *physically* resolved, not the spacing of the values. The problem is uncertainty, not
> discretisation.

### 13.2 Independent error averages down; correlated error does not

The segmentation threshold is the dominant correlated term: every edge in a specimen is measured
from one mask at one threshold, so moving it moves every diameter together.

Measured median calibre falls monotonically with threshold in **6 of 6 specimens**. Over the clean
0.85–0.90 interval the mean shift is **0.922 µm, about half a voxel** — a per-edge `δd/d` of 11.7%
and an analytic `δR/R` of 46.7%.

Measured by re-solving the networks at that perturbation rather than scaling, since *d*⁻⁴ is not
linear:

| Perturbation | Independent | Correlated | Within-specimen ratio |
|---|---|---|---|
| One voxel, 1.866 µm (conservative bound) | 4.1% | 95.3% | 13.2% |
| **Measured threshold shift, 0.922 µm** | 2.2% | **45.3%** | **6.3%** |

The measured 45.3% sits close to the 46.7% that `4·δd/d` predicts, so propagation is near-linear at
this scale even though the underlying law is not. **The ratio cancels 86% of the correlated error
at both perturbation sizes**, which makes that cancellation a property of the ratio rather than an
artefact of the size chosen.

### 13.3 The two noise floors

| Quantity | Floor | Against H1's measured 27–40% effects |
|---|---|---|
| Absolute network flow | **±45%** | Cannot resolve them |
| Within-specimen ratio | **±6.3%** | Can resolve them — roughly fourfold margin |

**This is why §7.6 reports ratios and never absolutes.** It is not caution; it is the difference
between an answerable question and an unanswerable one.

**One residual, in the ratio itself.** The per-specimen shift is uneven — 0.441 µm (WKY-C) to
2.117 µm (WKY-A) — so the correlated error is not identical across specimens and does not cancel
perfectly in a *between-group* comparison. Group means differ: 1.075 µm for WKY against 0.768 µm
for SHR. That is the right shape to become a confound. With n = 3 it is noted, not established.

### 13.4 Boundary selection is the largest single lever

Larger than calibre error. The face-crossing rule on axis 1 holds residual boundary sensitivity to
**13.3%**, against **75.8%** for the alternative band rule, and cuts total sensitivity from 118.8%
to 43.1%. Axis 1 is the only axis with terminals on both faces in all six specimens.

Below the operative floor of §13.3 and below the effects H1 measures — but only because the rule
and its axis are pinned. See open item 2 in §10: they are not yet pinned in one place.

### 13.5 Absolute perfusion is 20–100× below physiological

Measured across all six with the face rule at 60/20 mmHg:

| Specimen | Inlets | Total inlet flow (µm³/s) | Flow-weighted velocity |
|---|---|---|---|
| WKY-A | 18 | 8,924 | 6.2 µm/s |
| WKY-B | 10 | 8,699 | 6.4 µm/s |
| WKY-C | 11 | 6,511 | 4.1 µm/s |
| SHR-A | 12 | 16,240 | 9.7 µm/s |
| SHR-B | 12 | 14,870 | 6.3 µm/s |
| SHR-C | 7 | 6,560 | 6.6 µm/s |

**4–10 µm/s against a physiological 200–1,000 µm/s.**

**Boundary pressure is not the cause.** Reaching 500 µm/s would require about **3,257 mmHg**. The
boundary rule accounts for part of the gap — the alternative rule carries five to seven times more
flow and about two and a half times the velocity — but not for its size.

Every absolute perfusion figure inherits this. It is a further reason ratios are the only reportable
form.

### 13.6 The tissue is not diffusion-limited

**The most consequential limit in this document**, because it constrains the mechanism rather than
the precision.

The oxygen diffusion length is

```
sqrt(D · α · PO₂ / M)  =  20 µm at PO₂ 10,  35 µm at 30,  45 µm at 50
```

against a **median tissue-to-vessel distance of 5.3–7.9 µm**. Every tissue point sits at roughly a
fifth of its supply radius, so the tissue is not diffusion-limited and a local sink cannot produce
a local gradient.

Measured consequence: raising the glomus metabolic rate to **four times** the stromal one moves PO₂
within the TH-positive volume by **0.01 mmHg**.

So the premise of §7.5 — that a higher glomus metabolic rate produces glomus-specific hypoxia —
cannot operate on this geometry at these parameters. **This is a statement about the tissue, not
about the code.** A glomus-specific hypoxic fraction requires the sensors to be diffusion-limited,
and in a bed this dense they are not.

The consumption rate is not the problem: `M_max = 0.05` mmol/L/s is 0.067 mL O₂ per mL per minute,
against roughly 0.040 for brain — the right order for a metabolically active organ.

### 13.7 Grid resolution against the gradient that matters

Measured tissue-to-vessel distance at native resolution:

| Specimen | Foreground | TVD p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| WKY-A | 23.5% | 7.92 µm | 53.06 | 113.15 | 172.26 |
| WKY-B | 23.4% | 6.98 µm | 29.01 | 60.95 | 90.58 |
| WKY-C | 29.4% | 5.28 µm | 25.90 | 65.30 | 109.07 |

At the 4 µm grid the median tissue voxel sits **1.3–2.0 cells** from a vessel. The gradient that
decides whether tissue is hypoxic is therefore spanned by one or two cells for half the tissue —
resolved, but barely. Only the p90 tail, 25.9–53.1 µm, spans a comfortable number of cells.

Refining further is cheap in principle and was tested: PO₂ converges (§10.9). The limit is that the
gradient is physically short, not that the solve is inaccurate.

### 13.8 Calibre is not a reportable H1 finding

The between-group calibre gap sits at **one twentieth of the smallest resolvable difference**.
Within-group spread is 0.45 µm (WKY) and 0.34 µm (SHR) — three to four times the gap itself.

A separation that small is a coincidence of where six medians happened to fall. Any claim that SHR
capillaries are narrower is **not supported**.

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
| Within-specimen ratios of flow-derived quantities | **Yes** — ±6.3% floor against 27–40% effects |
| Absolute flow, velocity or perfusion | **No** — ±45% floor, and 20–100× below physiological |
| Between-group calibre differences | **No** — gap is 1/20 of the measurement step |
| Glomus-specific hypoxic fraction as a number | **No** — the mechanism cannot operate (§13.6); report as a curve in the assumed metabolic contrast |
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

### 14.3 Which script produces what

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
| `cb_h2_error_propagation.py` | The independent/correlated/ratio floors behind §13.3 |
| `cb_h2_glomus_perfusion.py` | §7.3 shunting, §7.4 haematocrit, §7.6 transit time |
| `cb_h2_hypoxic_fraction.py` | §7.5 hypoxic fraction on the heterogeneous grid |
| `cb_h2_vtk.py` | H2 ParaView artefacts |

### 14.4 Randomness

The pipeline is deterministic except for hyperparameter search. Optuna's TPE sampler takes an
explicit seed, defaulting to a fixed value, and **the seed is written into the tuning provenance
record** rather than only into a log line — a tuned parameter set whose search trajectory cannot be
reproduced is not a reproducible parameter set.

Pericyte constriction draws a random cohort per run (§10.6). It is frozen, so nothing on the CB path
is stochastic.

### 14.5 Reproducing a result

1. Preprocess to Ilastik input with the recorded parameters — identical for all six volumes.
2. Predict headlessly with the classifier named in the sidecar.
3. Threshold to a mask; the H1 cohort used a single frozen value for all six (open item 1).
4. Run the H1 batch to produce graphs and per-edge morphometry.
5. Run the H2 driver for the method in question, at 60/20 mmHg on axis 1 (open item 10).

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
| Picard max iterations | 15 | `HaemodynamicsConfig.rheology_max_iterations` | |
| Picard tolerance | 1 × 10⁻⁴ | `HaemodynamicsConfig.rheology_tolerance` | Relative |

### A.2 Network flow solve

| Setting | Value | Where | Notes |
|---|---|---|---|
| Direct/iterative dispatch threshold | 50 000 | `resistance.py` `_solve_system_smart` | Nodes. Below this, direct solve |

### A.3 Tissue transport — steady-state ADR (Tier 1)

| Setting | Value | Where | Notes |
|---|---|---|---|
| Picard max iterations | 50 | hard-coded in `solve_perfusion_steady_state` | Duplicates `picard_max_iterations` |
| Picard tolerance | 1 × 10⁻⁵ | hard-coded | **Differs from `PerfusionConfig.picard_tolerance` = 1 × 10⁻⁴** |
| Relaxation γ | 0.5 | hard-coded | Effective linearised slope |
| CG relative tolerance | 1 × 10⁻⁶ | hard-coded | |
| CG max iterations | 1 000 | hard-coded | |
| Preconditioner | Jacobi (diagonal) | `_jacobi_preconditioner` | Guarded against non-positive diagonals |
| Singularity handling | diagonal regularisation | `build_adr_matrix` | Pure-diffusion rows sum to zero under Neumann boundaries |

### A.4 Tissue transport — multi-species O₂ / CO₂ / pH (Tier 3)

| Setting | Value | Where | Notes |
|---|---|---|---|
| Relaxation γ (O₂) | 1.0 | hard-coded | |
| Relaxation γ (CO₂) | 1.0 | hard-coded | |
| CG relative tolerance | 1 × 10⁻⁵ | hard-coded | Looser than Tier 1 |
| CG max iterations | 500 | hard-coded | Half of Tier 1 |

### A.5 Config-level Picard settings

| Setting | Value | Where | Notes |
|---|---|---|---|
| `picard_max_iterations` | 50 | `PerfusionConfig` | Agrees with the hard-coded Tier 1 value |
| `picard_tolerance` | 1 × 10⁻⁴ | `PerfusionConfig` | Does **not** agree with the hard-coded 1 × 10⁻⁵ |

> ⚠ **Open item 6 — the solver settings are inconsistent between config and code.** Tolerances
> and iteration caps are declared in `PerfusionConfig` and then hard-coded again inside the
> solvers, with two values that disagree. Until this is reconciled, treat Appendix A as
> descriptive of the code, not of the config.

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
| *x₀* | Skimming threshold | 0.05 |
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
| Threshold selection | `threshold_selection.py:222` | `test_threshold_selection.py` |
| Joint hysteresis mask | `image.py:179` | `test_preprocessing.py`, `test_new_preprocessing.py` |
| Skeletonisation | `skeleton.py:472` | `test_graph.py`, `test_length_measurements.py` |
| Graph construction | `build.py:22` | `test_graph.py` |
| EDT calibre | `automated.py:1238` | `test_edt_diameter.py` |
| FWHM calibre | `automated.py:971` | `test_haemodynamics_automated_fwhm.py`, `test_integration_synthetic_vessel_fwhm.py` |
| Calibre provenance guard | `poiseuille.py:16` | `test_silent_fallback_guards.py` |
| Branch order | `branch_order.py:95` | `test_branch_order_hierarchy.py` |
| Face boundary rule | `boundaries.py:88` | `test_boundary_faces.py` |
| Poiseuille resistance | `poiseuille.py:160` | `test_haemodynamics_analytical.py` |
| Variable-diameter resistance | `poiseuille.py:146` | `test_haemodynamics_analytical.py` |
| Network Laplacian solve | `resistance.py:46`, `resistance.py:138` | `test_haemodynamics_analytical.py` |
| Flow unit conversion | `resistance.py:37` | `test_flow_units.py`, `test_physical_units.py` |
| Pries–Secomb viscosity | `rheology.py:30` | `test_rheology_laws.py` |
| Phase separation | `rheology.py:90` | `test_haemodynamics_analytical.py` |
| Coupled flow–haematocrit | `rheology.py:164` | `test_haemodynamics_rheology_integration.py` |
| Blood oxygen content | `perfusion.py:13` | `test_haemodynamics_analytical.py` |
| Blood CO₂ content | `perfusion.py:37` | `test_haemodynamics_analytical.py` |
| Tissue pH | `perfusion.py:65` | `test_haemodynamics_analytical.py` |
| Perfusion grid | `perfusion.py:82` | `test_haemodynamics_perfusion.py`, `test_fractional_grid.py` |
| Vessel-to-grid mapping | `perfusion.py:203` | `test_flow_conservation.py` |
| ADR assembly | `perfusion.py:349` | `test_haemodynamics_perfusion.py` |
| Jacobi preconditioner | `perfusion.py:317` | `test_perfusion_preconditioner.py` |
| Tier 1 steady state | `perfusion.py:453` | `test_haemodynamics_analytical.py` |
| Tier 3 multi-species | `perfusion.py:537` | `test_haemodynamics_analytical.py` |
| Heterogeneous metabolism | `tissue_regions.py:28`, `tissue_regions.py:124` | `test_tissue_regions.py` |
| Morphometry | `stats.py:147`, `stats.py:349` | `test_statistics.py`, `test_synthetic_network_statistics.py` |
| Two-channel morphometry | `th_morphometry.py:34`, `th_morphometry.py:78` | `test_th_morphometry.py` |
| Transit time | `transit.py:28`, `transit.py:57` | `test_transit.py` |
| Cohort split | `cohort_split.py:56` | `test_cohort_split.py` |
| Artefact provenance | `artefact_provenance.py` | `test_artefact_provenance.py` |
---

## Open items

| # | Item | Blocks |
|---|---|---|
| 1 | Two segmentation thresholds in play — config 0.65/0.75 vs the frozen 0.90 used for H1 | §2.2, and any quoted calibre |
| 2 | Two boundary rules coexist: H1 runs the band rule on axis 0 at 25%, the H2 drivers run the face rule on axis 1 | §2.8, §8, §13 — the largest sensitivity in the model |
| 3 | Arterial PO₂ set in both config and solver bodies | §5, §6 |
| 4 | Baseline haematocrit duplicated in the Tier 1 washout path | §6.6 |
| 5 | `C_arterial` is dead configuration — declared 3×, read 0× | §6 |
| 6 | Solver tolerances disagree between config and code | Appendix A |
| 7 | 13 parameters still marked `[CITE]`, including every blood-gas solubility and the Spencer CO₂ curve | §10 completeness |
| 8 | `M_max` differs 10× between `PerfusionConfig` (0.005) and the H2 driver's `BASE_M_MAX` (0.05). The published §2.3 results used 0.05 | §6.4, §13.6 |
| 9 | The rheology solver falls back to a silent 5.0 µm diameter; `map_vessels_to_grid` raises on the same condition | §3.2 |
| 10 | Pressure boundaries disagree: config 100/2 mmHg, H2 drivers 60/20 mmHg. Every published H2 number used 60/20 | §7.8, §8, §11 row 15 |
| 11 | Both Shannon-entropy parameters are inert — the vessel classifier has 2 classes, so the joint hysteresis path never runs; `shannon_entropy_core` is not even a config field | §2.3 |
| 12 | The rheology loop rescales resistance by `µ_app / µ_old` against a base that no longer contains `µ_old`, inflating every resistance ~200–540× and diameter-dependently | §3.2, §4.3, and every absolute flow in §7, §13.5 |

---
