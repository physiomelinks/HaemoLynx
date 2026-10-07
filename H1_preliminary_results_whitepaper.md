# Hypothesis 1: Preliminary Results

**Carotid body microvascular morphology, SHR versus WKY**

*ImageLynx / `carotid_image_to_model` · branch `cb_pipeline_improvements_sweep` · Dale Sasis*

> **Re-quoted 2026-09-30** on the current analysis: vessel threshold 0.95 with a 0.999 hysteresis
> seed, regions placed on tissue (`place_roi`), the converged rheology solve, and the batch and
> sensitivity outputs (`examples/outputs/cb_h1_batch/`, `cb_h1_sensitivity/`). §3 and §4 are the
> historical "before" measurements and are unchanged. The earlier version (threshold 0.90,
> SHR/WKY ratios 1.40 / 1.34 / 1.27) is in git history.

---

## 0. Executive summary

This document reports the first end-to-end application of the `carotid_image_to_model` pipeline to Hypothesis 1 (H1): that carotid body (CB) morphology differs between the spontaneously hypertensive rat (SHR) and the normotensive Wistar–Kyoto control (WKY).

**It is a methods-maturation milestone, not a biological finding.** That distinction is load-bearing and is maintained throughout.

A stage-by-stage audit conducted on 2026-07-30 established that the pipeline could not answer H1: not merely imprecisely, but structurally. Two of its optimisation objectives penalised the vascular loop topology that §1.1 proposes as its readout; the Euclidean distance map (EDM) radius estimator that §1.2 names was implemented and never called; and every length in the system was expressed in uncalibrated voxel units. Forty-one changes have since closed those defects, and each closure is quantified rather than asserted.

Applied to all six specimens under one pooled classifier, one frozen parameter set, one segmentation threshold and matched sub-volumes, three topological measures point the same way, by small margins:

| Measure (§1.1) | WKY | SHR | Ratio |
|---|---|---|---|
| β₁ loop density | 6.78 × 10⁴ mm⁻³ | 7.37 × 10⁴ mm⁻³ | **1.087** |
| Junction density | 1.57 × 10⁵ mm⁻³ | 1.67 × 10⁵ mm⁻³ | **1.064** |
| Vessel length density | 3.29 × 10⁶ µm·mm⁻³ | 3.36 × 10⁶ µm·mm⁻³ | **1.022** |

The differences are 2 to 9%, against a within-group spread of 18 to 33%. The groups overlap on every measure (exact p 0.60, 0.60, 0.80), and the length density difference is too small to call a difference at all.

Of four checks on the segmentation, three pass: the selected threshold and the foreground fraction do not split by cohort, and the direction holds at every threshold tested, in all nine group comparisons, with the effect growing as inclusiveness falls. The fourth no longer passes: across the six specimens, all three densities now rise with the foreground fraction (Pearson r +0.56 to +0.77, §6.3). At n = 6 that is not significant, and a denser network would fill more of its box anyway, but the densities can no longer be called independent of the segmentation.

**Three limitations bound what may be concluded.** The groups overlap on every topological measure, and with n = 3 per group the exact two-sided p cannot fall below 0.10. The segmentation classifier is not final: four of six volumes lack perivascular boundary labels. The two TH-dependent sub-methods (§1.3, §1.5) are reported for the full cohort, with the TH classifier's residual 2.1× cohort skew as their stated bound (§9A.5).

**Reproducing the analysis at three thresholds leaves the direction unchanged**, so the direction is not an artefact of the one parameter that most directly controls how much tissue is called vessel. The size of the effect is small at all three.

**Verdict.** All five sub-methods have a working implementation. §1.1 is implemented and measuring what it was specified to measure; it finds SHR slightly denser on average, with overlap. §1.2 is implemented; the medians now separate by cohort, but by less than one voxel and not at the neighbouring thresholds, so it does not support a claim. §1.4 is implemented and shows no difference. §1.3 and §1.5 are implemented and reported for both cohorts on the batch network (§9A); neither separates the cohorts. On the old vessel set their contrast held across three TH classifiers spanning an elevenfold change in cohort skew and a seventyfold change in class balance (§9A.5).

Implementation is not the same as answerability. Four of the five are answerable within the limits of n = 3; §1.2 is not answerable at any labelling effort with the current voxel size.

The TH-dependent measures, re-quoted on the batch network (§9A), point the same way as §1.1's but separate nothing. SHR carotid bodies in this sample carry **less TH-positive parenchyma (0.61×) that is more densely vascularised (1.18×)**, and the cohorts overlap on both: SHR-A has more TH volume than WKY-A, and SHR-C's length density is below every WKY value.

---

## 1. Scope: what H1 asks, and what is currently answerable

H1 and its five sub-methods are defined in `hypothesis_testing_methods.md`; that file's §-numbering is used throughout this document and across the wider documentation set.

| § | Method | Status | Reason |
|---|---|---|---|
| **1.1** | Topological node counting | **Implemented, overlapping** | Graph extraction yields nodes, degree distribution and β₁; SHR 2–9% denser on average, groups overlap (§7.2) |
| **1.2** | EDM geometric profiling | **Implemented, not conclusive** | Estimator runs on 100% of edges; the medians separate by less than one voxel and not at the neighbouring thresholds (§8) |
| **1.3** | Proportional capillary density | **Implemented, overlapping** | Length density 1.18× in SHR; SHR-C sits below every WKY value (§9A.3) |
| **1.4** | Tortuosity | **Implemented, no difference** | SHR/WKY 0.99, groups overlap; no longer tracks segmentation inclusiveness (§9) |
| **1.5** | Tissue-to-vessel distance | **Implemented, no difference** | 0.97×; cohorts overlap, SHR-C furthest of all six (§9A.3) |

§1.3 and §1.5 require a parenchymal landmark to measure against, which the vessel segmentation alone cannot supply. A second two-class Ilastik project segments the TH channel of the same acquisitions, and both methods are implemented in `ImageLynx.statistics.th_morphometry`. The same capability supports all four H2 perfusion methods, which depend on TH-masked analyses.

The TH classifier labels all six volumes at three depths, passes `verify_classifier`, and carries a residual 2.1× cohort skew in its positive class. §9A.5 evaluates the contrast against three classifiers spanning that skew from 22.9× to 2.1× and the pooled class balance from 1:59 to 1:0.8; the ratios move by at most 0.01.

This document therefore reports §1.1 and §1.3 in full, §1.5 and §1.4 with their overlap stated, and §1.2 with an explicit disqualification.

---

## 2. The measurement chain

Every quantity reported here is the output of an eight-stage chain. Each stage carries decisions that affect the result; this section records them and the evidence behind each.

### 2.1 Acquisition

Six carotid bodies, three per cohort, imaged by confocal fluorescence. Each acquisition is a single `ZCYX` file with two channels: channel 0 is lectin, labelling the vascular endothelium, and channel 1 is tyrosine hydroxylase (TH), labelling the type I glomus cells. Channel identity was confirmed by byte-comparison against the separately extracted `C1-*_vessels.tif` and `C2-*_glomus_cells.tif` rather than assumed from the acquisition order.

Both channels enter the ingest path, through separate preprocessing and separate classifiers (§2.2, §2.3). Being two channels of one acquisition, they are co-registered by construction on an identical grid, which is what makes the §9A joins sound without a registration step.

Volumes are 2×2×2-binned, uint16, ZCYX. Physical voxel size, read from each acquisition's own ImageJ metadata rather than assumed:

| | z (µm) | y (µm) | x (µm) |
|---|---|---|---|
| WKY (3 volumes) | 1.86386 | 1.86600 | 1.86600 |
| SHR (3 volumes) | 1.86412 | 1.86600 | 1.86600 |

The z-step differs between cohorts by 0.014%. This is negligible for every result reported here, but it is a *group-correlated acquisition difference* and is disclosed on that basis rather than because it changes anything.

The volumes differ substantially in extent: SHR average 89 Mvoxel against WKY's 63 Mvoxel. Any raw count is therefore larger in SHR before any biology is involved, which is why every quantity in this document is a density and why sampling is size-matched (§5.3).

### 2.2 Preprocessing

Applied identically to all six volumes: channel extraction, rolling-ball background subtraction (radius 30 px ≈ 56 µm, larger than the thickest arteriole so lumens are not eroded), per-volume percentile normalisation, and multiscale Sato vesselness at two scale bands (fine σ = 1.0/1.4/2.0 px; coarse σ = 4.0/8.0 px), each scale normalised before the cross-scale maximum.

No bleach correction and no denoising were applied. The axial intensity profile of all six volumes is hump-shaped, reflecting how much tissue each slice intersects rather than photobleaching, so histogram matching would inflate sparse end slices and promote their background noise to vessel intensity. Denoising is omitted because a 3×3×3 median spans 5.6 µm, wider than a capillary (§4.3).

Output is a three-channel float32 volume (grayscale, vesselness_fine, vesselness_coarse). Channel order is load-bearing: the classifier indexes features by channel position.

### 2.3 Classification

A single Ilastik pixel-classification project (`vessel_segmentation.ilp`) is used for all six volumes. This is the central experimental control. Per-cohort classifiers would confound specimen identity with classifier identity unfixably: a between-group difference in vessel count could then be a difference in the measuring instrument rather than the tissue, with no way to separate them after the fact. The registry refuses a run whose specimens do not share one project.

The error asymmetry justifies the choice. A single classifier slightly suboptimal on one cohort adds noise and biases *toward the null*, which is conservative. Two classifiers add systematic bias of unknown sign directly to the group contrast, and can manufacture or mask an effect.

**Current labelling state**, which bounds every result below:

| Specimen | Labelled voxels | Boundary labels within 9.33 µm of a vessel |
|---|---|---|
| WKY-A | 20,266 | 4.1% |
| WKY-B | 21,515 | **20.5%** |
| WKY-C | 13,504 | 8.5% |
| SHR-A | 42,553 | 2.1% |
| SHR-B | 36,264 | **24.8%** |
| SHR-C | 28,690 | 1.3% |

Only WKY-B and SHR-B have had perivascular background labelled. The consequence is developed in §11.1.

**The TH channel** is segmented by a second, separate project (`glomus_cell_segmentation.ilp`), also two-class (`glomus`, `background`) and also shared across all six volumes. It is separate rather than an extra class in the vessel project because the two are trained on different input channels: the vessel project sees grayscale plus two vesselness bands, the TH project sees grayscale plus a signed Difference-of-Gaussians tuned to the glomus soma scale.

Its labelling is deliberately sparse and boundary-selected rather than extensive:

| Specimen | `glomus` labels | `background` labels | Ratio | Depths | Median distance of `background` from TH tissue |
|---|---|---|---|---|---|
| WKY-A | 11,091 | 6,242 | 1:0.6 | 3 | 4.2 µm |
| WKY-B | 6,653 | 5,784 | 1:0.9 | 3 | 4.2 µm |
| WKY-C | 7,191 | 5,626 | 1:0.8 | 3 | 2.6 µm |
| SHR-A | 6,300 | 5,271 | 1:0.8 | 3 | 2.6 µm |
| SHR-B | 3,221 | 3,268 | 1:1.0 | 3 | 2.6 µm |
| SHR-C | 2,152 | 2,729 | 1:1.3 | 3 | 1.9 µm |

`verify_classifier(channel="th")` passes. 65,528 labelled voxels in total, against the vessel project's 163,035.

The sparsity is the design. Ilastik's random forest weights by labelled voxel count and does not rebalance, so background painted across empty field both dominates the class weighting and teaches nothing: it is trivially separable from tissue. The background labels here sit a median 1.9 to 4.2 µm from TH-positive tissue, with under 4% beyond 30 µm, so nearly all of them fall where the decision boundary actually lies. An earlier labelling of the same project held 1.7 million voxels at a pooled balance of 1:59, with background at a median 62 µm from tissue and 92% of it beyond 20 µm; §9A.5 evaluates the contrast against both.

One check still fires. The positive class is **2.1 times more numerous in WKY** than SHR (24,935 against 11,673), above the 2× reporting threshold. That residual skew is the stated bound on §9A.

### 2.4 Threshold selection

The probability threshold is chosen from vessel calibre, constrained by skeleton fragmentation, not from connected-component statistics.

The component-based criterion in the segmentation handover was tested against the real probability field and returns no answer. On the placed regions the largest component's voxel share stays at 0.95 or above in all six specimens from threshold 0.30 to 0.97, and at 0.99, where WKY-A's skeleton has broken into 1,429 pieces, its mask's largest component still holds 98.8% of the foreground (1.000 at 0.30). Only SHR-B's share drops appreciably, to 0.81, and only at 0.99. This is a property of the data's topology rather than of any one classifier: a vascular bed percolates, and a percolating mask stays connected long after its centreline has begun to bead.

Two measurements do discriminate. **Median centreline diameter** moves monotonically with threshold and has an external target: an expected capillary calibre of 4–7 µm. It falls from 11.2–13.5 µm at threshold 0.30 to 3.73 µm at 0.97 in every specimen. **Skeleton endpoint density** is flat while the network is intact and climbs sharply once it beads: on WKY-A, 2.3–4.9 per mm from 0.30 to 0.95, then 6.8 at 0.97 and 12.4 at 0.99, while skeleton components rise from a minimum of 170 to 1,429. Across the six, the density at 0.99 is 12.4 to 19.3 per mm. Each fragment contributes two endpoints, which is what the mask cannot see. Fragmentation is flagged where the density exceeds 1.5 times the specimen's intact level; the onset is 0.97 in four specimens and 0.99 in WKY-B and SHR-B.

Calibre is therefore the objective and fragmentation the veto. Where the two never agree, the selector returns no threshold and reports which constraint failed, rather than a best-effort value.

### 2.5 Mask and skeleton

Hysteresis thresholding, three-dimensional cavity filling (lectin stains endothelium, so large vessels appear as rings; a 2D per-slice fill would close every in-plane vascular loop and destroy the topology H1 measures), morphological closing, and connected-component filtering. Skeletonisation is 3D.

### 2.6 Graph extraction and morphometry

The skeleton is traced into an undirected multigraph. Stub branches below 5.6 µm (three voxels) are pruned; centrelines are B-spline smoothed with per-edge provenance recorded; per-edge diameters are measured from the 3D EDM, sampled at centreline voxels, with junction neighbourhoods excluded (§4.2).

The primary data product is `per_edge_morphometry.csv`: one row per graph edge, fifteen columns, including both raw radius estimators and four provenance tags. Every measurement carries the record of how it was obtained, so a distribution that mixes measured and fabricated values can be separated after the fact.

### 2.7 What the measurement is made of

> **Figure 4.** `figure4_reconstruction.png`. All six regions, one 14 µm slab each: the segmented volume translucent, the analysed centrelines inside it. **Illustration, not evidence.** Each panel is a 0.0266 mm³ region and three specimens per group cannot support a visual comparison; the figure shows all six precisely so that no pair is chosen for effect, and WKY-C is denser than SHR-C on all three §1.1 measures. A slab is shown because at 20–27% foreground the whole cube is opaque from outside.

> **Figure 5.** `figure5_measured_network.png`. The same network carrying what was measured on it. Left: centrelines coloured by EDT diameter, the estimator §1.2 reports. Right: nodes of degree ≥ 3, the branch points §1.1 counts. Quantities reported as distributions elsewhere in this document are properties of identifiable places in the network.

> **Figure 6.** `figure6_skeleton_detail.png`. Raw skeleton (left) against the analysed centrelines (right), same region. The difference is stub pruning and B-spline smoothing; the voxel staircase visible on the left is what §4.4's calibration and §4.3's operators act on.

### 2.8 Artefact provenance

Each probability map carries a sidecar recording the classifier hash that produced it, its label counts and its boundary-label placement. The pipeline reports each map as `current`, `stale`, `unknown` or `absent`. `unknown` is deliberately distinct from and worse than `stale`: a stale map has a known and wrong origin, whereas an unknown one cannot be ruled out about.

All six maps used in this document report `current` against classifier `49283a27d82e…`.

---

## 3. Why the earlier pipeline could not answer H1

An audit on 2026-07-30 examined stages 1–22 by code inspection *and* direct numerical execution on real data. Five findings mattered.

**3.1 Joint hysteresis carved a band out of the middle of the probability range.** For a two-class output, Shannon entropy is a deterministic, folded function of the vessel probability, so an entropy ceiling resolves to `p ≤ 0.369 OR p ≥ 0.631`. Measured retention: 98% of voxels at p ∈ [0.20, 0.35], **0% at p ∈ [0.40, 0.60]**. Every vessel became a core plus a detached shell with the wall evacuated: 7,627 components, Euler characteristic −18,870.

**3.2 Both optimisation objectives penalised the hypothesis.** The preprocessing objective's Euler-characteristic term and the skeleton objective's fundamental-loops term both drive vascular loop topology toward zero, precisely the quantity §1.1 proposes as its readout. Measured: a parameter set with better Dice (0.688 vs 0.598), better orphaned fraction and 29% more centreline scored *worse* (loss 246.1 vs 164.7), because the loop term was 70–86% of total loss. Both penalties scale with network density, so they suppressed the SHR/WKY difference in the false-negative direction.

**3.3 The EDM estimator §1.2 names was implemented and never called.** `radius_assignment_mode: "edt_radius"` passed validation and silently produced synthetic branch-order diameters. FWHM covered 49.2% of edges against EDT's 99.2%, and on shared edges the two were uncorrelated (Pearson r = 0.079).

**3.4 No TH/glomus channel exists.** §1.3, §1.5 and all four H2 methods unimplementable.

**3.5 Systemic voxel-versus-physical unit confusion.** Node positions and edge voxels were stored in physical units; array indexing, boundary selection and every filter radius in voxels. They agreed only because the TIFF declared no resolution and the reader returned (1, 1, 1).

---

## 4. What changed

Forty-one changes were made. Presenting them as a list would demonstrate activity rather than trustworthiness, so they are grouped by defect class, each with the measurement that demonstrates the change. The full commit-to-finding map is Appendix C.

### 4.1 Objectives that penalised the hypothesis

Both loop-topology penalty terms were removed from the tuning objectives, which were reduced to their fidelity terms. Bundle collapse, a skeletonisation step that replaced dense regions with synthetic paths, was disabled after measurement showed it **destroyed 68% of β₁** (307 loops reduced to 99) on the reference sub-volume. This was the single largest H1 signal loss identified.

### 4.2 Estimators that silently did not run

The EDM estimator is now the default and raises if it measures nothing, rather than falling back to synthetic diameters. Re-measured on the repaired pipeline over 1,330 edges: EDT covered 100.0% with a median diameter of 6.37 µm; FWHM covered 76.5% with a median of 8.20 µm and an unphysical maximum of 39.16 µm. The two remain only weakly correlated (r = +0.245), and EDT is used throughout.

A junction-proximity exclusion was added. Within approximately one radius of a bifurcation the distance transform returns the junction's inscribed sphere rather than the vessel's, biasing radii upward. On a controlled fixture two identical tubes read 6.00 µm and 4.47 µm apart for no reason other than segment length: a 34% error, which Poiseuille resistance carries as 3.2×. On real data the population effect is approximately 8% on resistance, and **61% of segments are too short to trim at two voxels** and therefore retain the bias; those are tagged rather than discarded, because dropping them would bias the distribution toward long vessels.

### 4.3 Signal destroyed before measurement

The entropy criterion is now gated on classifier class count and disengages for two-class output. Median filtering of the probability field was removed: at equal speckle suppression it destroyed **80% of the true vessel** (900 → 181 foreground voxels against a clean-truth 900), where a post-threshold size filter achieved identical component reduction at 100% recall.

Morphological closing was found not to be a closing. A closing is extensive (X ⊆ X•B), and this implementation *removed* foreground, because the underlying library erodes against a zero border. On a boundary-crossing tube the entire first and last slice vanished; on the integration fixture it deleted seven voxels, **every one of them a vessel voxel touching a domain face**. That is 100% of the population determining where a vessel terminates, and therefore which nodes become inlets and outlets.

### 4.4 Units and provenance

Voxel size was calibrated from acquisition metadata, replacing the (1, 1, 1) fallback, with the physical/voxel unit handling corrected in the same change so the benchmark suite could not break silently. Every diameter, centreline and radius now carries a provenance tag; every probability map names the classifier that produced it.

### 4.5 Claims corrected by measurement

Three assertions made during this work were overturned by subsequent measurement and are recorded here because a remediation record that reports only successful fixes is advocacy rather than evidence.

- A proposal to replace the preprocessing objective with soft Dice was retracted before implementation: measurement showed its optimum sits at a flooded mask, because scoring against the probability field reproduces the classifier's over-prediction.
- A claim that centreline-smoothing error concentrates on twisty edges was falsified by the per-edge export, which showed the affected edges to be the shortest and straightest.
- A claim that the doubled morphological closing compounded vessel fusion was wrong: closing is idempotent for a fixed structuring element, so the second call had never had any effect at all.

### 4.6 The capability that did not exist

§3.4 is closed as follows. The TH channel is preprocessed by `examples/preprocessing/preprocess_th.py`, which shares its stages with the lectin preprocessor so both channels of one acquisition are treated identically, and is segmented by a second two-class Ilastik project (§2.3). §1.3 and §1.5 are implemented in `ImageLynx.statistics.th_morphometry` and reported in §9A.

Four measurements made during that work are recorded because each overturned a decision that had already been taken.

- The TH preprocessing protocol specified histogram-matching bleach correction. Measured on WKY-C, it multiplies the top slice by 26.9× and raises the TH-positive fraction in the near-empty top of the stack from 0.004% to 1.413%. Worse, it forces every slice to the same intensity distribution, which hard-codes TH density to be uniform in depth. Removed.
- The protocol specified a four-class labelling scheme (cytoplasm, nucleus, intercellular boundary, background) to support watershed cell counting. Checked against what H1 and H2 actually consume, every one of the six analyses asks for a TH-positive volume, voxel set or cluster boundary and none counts cells. Reduced to two classes.
- The secondary Difference-of-Gaussians channel was clipped at zero. The dark nuclear core is exactly where that difference is negative: 99.8% of cores were being mapped to the same value as background. The map is now signed.
- Normalisation anchors were taken from the whole volume, which is dominated by empty frame. Tissue occupancy spans 7.9% to 17.7% across the six, and the resulting anchor gap between cohorts was 17.3%. Anchors are now taken inside tissue, and one shared pair is used for all six.

---

## 5. Experimental design for this run

### 5.1 One classifier, one parameter set

All six specimens were segmented with the same Ilastik project and processed with an identical, frozen parameter set (Appendix A). No per-specimen or per-cohort tuning was performed at any stage.

### 5.2 One threshold

Each specimen's own optimal threshold was computed, then a single value was frozen for all six. Per-specimen thresholds would absorb classifier differences into what would then appear as a tissue result.

| Specimen | Own optimum | Frozen |
|---|---|---|
| All six | 0.95 | 0.95 |

Every specimen selects 0.95 on its own: the highest threshold below its fragmentation onset whose centreline calibre lies in the 4–7 µm window (5.27–5.28 µm in all six). The same rule applied to the network's own mask, less one voxel for the distance transform's half-step bias, also gives 0.95 in five specimens and 0.97 in WKY-B, median 0.95 (`cb_modelling_reference.md` §2.2, open item 41). Had the choices differed, one threshold could not have been optimal for all six; freezing it would still be the correct cost to pay.

### 5.3 Matched, tissue-placed sub-volumes

Whole volumes are not reachable on the available hardware. Measured scaling on WKY-C: 0.12 Mvoxel took 11 s at 0.77 GB; 4.10 Mvoxel took 348 s at 3.93 GB, approximately one gigabyte per million voxels and superlinear in time. A complete volume is 34.9 Mvoxel, which extrapolates beyond available memory.

Each specimen therefore contributes an identical **160 × 160 × 160 voxel (0.0266 mm³)** region. A *percentage* crop would not be a matched sample: at 45% it yields 3.13 Mvoxel from WKY-C and 8.62 Mvoxel from SHR-B, restoring the extent confound the density normalisation exists to remove.

Placement is computed from each volume's own data rather than centred on the array (`place_roi`): z from the axial tissue peak recorded during preprocessing, y and x from the grayscale centroid over the box's own 160 slices. The tissue peak ranges from slice 106 of 435 (WKY-B) to 230 of 435 (WKY-A), so a centred box would land mid-organ in one specimen and in the sparse margin of another. The misplacement is also group-correlated, with WKY peaking at a mean depth fraction of 0.40 against SHR's 0.37. Every batch output is checked against this placement before it is read (`check_output_roi`); the record is Appendix D.

**This trade is stated rather than hidden.** Placing the box on signal samples the middle of the organ, which is denser than its periphery, so the absolute densities reported here **over-estimate the whole organ**. Applying the same rule to all six keeps the comparison like-for-like, which is what a between-group claim requires; an absolute density quoted from these regions would be wrong.

---

## 6. Instrument validation: is the segmentation producing the difference?

This is the question on which everything else depends. If the segmentation behaves differently on the two cohorts, then a measured group difference is partly the measuring device, and no amount of downstream care recovers the biology.

Four checks were applied. The first three are reported here; the fourth is §6.4.

### 6.1 Does the selected threshold separate by cohort?

Each specimen's independently selected optimum:

| WKY | SHR |
|---|---|
| 0.95, 0.95, 0.95 | 0.95, 0.95, 0.95 |

All six select the same value. On the network-calibre check (§5.2) the choices are 0.95, 0.95, 0.97 in WKY and 0.95 in all three SHR. **No cohort split.**

### 6.2 Does the foreground fraction separate by cohort at the frozen threshold?

| WKY-A | WKY-B | WKY-C | SHR-A | SHR-B | SHR-C |
|---|---|---|---|---|---|
| 0.2739 | 0.2094 | 0.2046 | 0.2307 | 0.2124 | 0.2008 |

WKY spans 0.205–0.274, SHR spans 0.201–0.231: fully overlapping and interleaved (exact p 0.70). **No cohort split.**

This matters more than it appears. Foreground fraction at a fixed threshold is close to what H1 measures. Had it separated, the group contrast would have been partly instrumental regardless of anything downstream.

### 6.3 Does each reported measure track segmentation inclusiveness?

A measure that correlates with how much the classifier includes is measuring the classifier.

Correlation across the six specimens with the foreground fraction at the frozen threshold (§6.2):

| Measure | Pearson r | Spearman ρ | Verdict |
|---|---|---|---|
| **β₁ loop density** | **+0.556** | **+0.771** | **Tracks inclusiveness** |
| **Junction density** | **+0.615** | **+0.771** | **Tracks inclusiveness** |
| **Vessel length density** | **+0.769** | **+0.886** | **Tracks inclusiveness** |
| Median diameter | +0.619 | +0.543 | Related |
| Mean tortuosity | +0.112 | +0.314 | Independent |

**This check has reversed since the 0.90 run**, where the three densities were independent (r −0.20 to −0.23) and tortuosity was confounded (r +0.86). Now all three densities rise with the foreground fraction and tortuosity does not.

Two things limit how much this says. At n = 6 none of the correlations is significant (the largest, r = +0.77, has p ≈ 0.07). And the check cannot tell a confound from biology: a denser network occupies more of its box, so some positive correlation is expected even from a perfect segmentation. It is not one specimen's doing: WKY-A has the highest foreground fraction but not the highest densities, and leaving it out makes the correlation stronger (r +0.90 to +0.97 over the other five). What the check can no longer do is vouch for the densities: it was the evidence that they measured the tissue rather than the classifier, and it no longer gives that evidence. The densities are still reported (§7), with this stated. Tortuosity is no longer withheld as confounded; §9 reports it.

### 6.4 Does the result survive a change of threshold, and does it behave as predicted?

The whole analysis was repeated at the frozen threshold's grid neighbours, 0.93 and 0.97, giving three complete six-specimen runs. Only the flood threshold moves; the hysteresis seed is held at 0.999 in all three (`cb_h1_batch.py --stage sensitivity`).

This tests two distinct things. First, **robustness**: does the direction hold if the one frozen parameter that most directly controls how much tissue is called vessel is moved? Second, a **prediction**. If the masks are over-inclusive and over-inclusion merges adjacent capillaries (merging them more in the denser cohort, because its vessels are closer together), then the measured SHR excess is suppressed, and reducing inclusion should *increase* it. That prediction was stated before the first sweep (0.85 to 0.95 on the earlier regions) was run, and is re-tested here.

**Group ratio (SHR / WKY) by threshold:**

| Measure | 0.93 | 0.95 | 0.97 | Direction |
|---|---|---|---|---|
| β₁ loop density | 1.050 | 1.087 | 1.117 | increases |
| Junction density | 1.051 | 1.064 | 1.098 | increases |
| Vessel length density | 1.013 | 1.022 | 1.030 | increases |

**Robustness.** The SHR group mean exceeds the WKY group mean in all nine comparisons: three measures × three thresholds. The groups overlap at every threshold and no exact p falls below 0.60. The direction of C4–C6 does not depend on the threshold chosen; neither does the overlap.

**The prediction holds, on the interval where it can be tested cleanly.** All three ratios increase monotonically as inclusion falls.

One qualification is necessary. At threshold 0.97, four of the six specimens (WKY-A, WKY-C, SHR-A, SHR-C) are at their individually determined fragmentation onset (§2.4), where a single vessel begins to break into multiple graph edges and loops are created artefactually. The 0.97 column is therefore directionally consistent but contaminated, and should not be read quantitatively. **The clean interval is 0.93 → 0.95**, where all six specimens sit below their fragmentation onset; across it β₁ rises 1.050 → 1.087, junction density 1.051 → 1.064, and length density 1.013 → 1.022.

Two points define a direction, not a trend, and these rises are small: 0.04, 0.01 and 0.01. The prediction is supported rather than established, and it rests on group means over three specimens each.

A competing explanation was considered and is not supported by the data. If raising the threshold simply eroded thin vessels away, and SHR vessels are the narrower of the two (as the published prior in §10 holds, and §8 measures on average), then SHR should lose *more* structure as the threshold rises and the ratio should fall. It rises. The merging interpretation is the one consistent with the observation.

**Consequence for the headline result.** The values reported at the frozen threshold of 0.95 are, on this evidence, lower bounds on the effect this instrument would measure with less inclusive segmentation. That is the direction the incomplete boundary labelling (§2.3, §11.1) is expected to move them when it is finished. On the clean interval the move is a few percent, not the tens of percent the earlier sweep showed.

> **Figure 3.** `figure3_threshold_sensitivity.png`. Group ratio (SHR / WKY) against threshold for the three topological measures, at the frozen 0.95 and its grid neighbours 0.93 and 0.97, all on the placed ROI with the seed held at 0.999. Solid segments span the clean interval where every specimen sits below its fragmentation onset; dashed segments and the shaded band mark where fragmentation contaminates the measurement (onset 0.97 in four of six specimens). The horizontal rule at 1.0 is no difference between cohorts. Every value is read from the batch and sensitivity outputs: β₁ 1.050 / 1.087 / 1.117, junctions 1.051 / 1.064 / 1.098, length 1.013 / 1.022 / 1.030. SHR exceeds WKY in all nine comparisons, but the groups overlap at every threshold (smallest exact p 0.60).

**All four checks are internal, and one no longer passes.** §6.1, §6.2 and §6.4 show that the threshold and the foreground fraction do not differ by cohort and that the direction is not an artefact of the one threshold chosen. §6.3 no longer shows that the reported densities are independent of how inclusive the segmentation is. None of them establishes that the segmentation is accurate in absolute terms; that requires hand-labelled held-out regions scored separately per cohort, which do not yet exist (§11.1).

---

## 7. Results: §1.1 topological node counting

All six specimens completed, and every rheology solve converged. Edge counts range from 6,140 (WKY-B) to 8,281 (SHR-A).

### 7.1 Per-specimen values

| Specimen | Group | Edges | Nodes | β₁ | β₁ mm⁻³ | Junctions mm⁻³ | Length µm·mm⁻³ |
|---|---|---|---|---|---|---|---|
| WKY-A | WKY | 7,597 | 5,589 | 2,009 | 75,574 | 174,208 | 3.65 × 10⁶ |
| WKY-B | WKY | 6,140 | 4,545 | 1,596 | 60,038 | 141,029 | 3.07 × 10⁶ |
| WKY-C | WKY | 6,863 | 5,064 | 1,800 | 67,712 | 155,738 | 3.14 × 10⁶ |
| SHR-A | SHR | 8,281 | 5,979 | 2,303 | 86,634 | 192,491 | 3.80 × 10⁶ |
| SHR-B | SHR | 6,954 | 5,034 | 1,921 | 72,264 | 161,907 | 3.28 × 10⁶ |
| SHR-C | SHR | 6,376 | 4,724 | 1,653 | 62,182 | 146,559 | 2.99 × 10⁶ |

β₁ = E − V + C is the count of independent cycles: the "number of vascular loops" §1.1 names as its measure of pathological network disorganisation. C = 1 by construction, the graph being reduced to its largest connected component. Edges and nodes are counted on the multigraph, so a parallel edge counts; junctions are nodes of degree 3 or more on it; length is the sum of the edge polylines. All three come from `per_edge_morphometry.csv`, as Figure 1 does.

### 7.2 Group comparison

| Measure | WKY mean | SHR mean | Ratio | Hedges' g | g 95% CI | Pairwise ratio range |
|---|---|---|---|---|---|---|
| β₁ loop density | 6.78 × 10⁴ | 7.37 × 10⁴ | 1.087 | 0.46 | [−1.17, +2.09] | 0.82 – 1.44 |
| Junction density | 1.57 × 10⁵ | 1.67 × 10⁵ | 1.064 | 0.39 | [−1.23, +2.02] | 0.84 – 1.36 |
| Vessel length density | 3.29 × 10⁶ | 3.36 × 10⁶ | 1.022 | 0.15 | [−1.45, +1.76] | 0.82 – 1.24 |

**Every confidence interval spans zero, and the differences are small.** SHR is 8.7%, 6.4% and 2.2% above WKY, against a within-group spread (range over mean) of 18–23% in WKY and 24–33% in SHR. Effect sizes are reported because they are more informative than a p-value at this n, not because they are conclusive. The pairwise ratio range is the interpretable quantity: for every measure, at least one WKY specimen exceeds at least one SHR specimen, and WKY-A exceeds both SHR-B and SHR-C on all three.

For scale: the pipeline's within-specimen noise floor on flow-derived ratios is ±5.9% (`cb_modelling_reference.md` §13.3). That floor is not strictly the right yardstick for topological counts, which calibre does not move (§13.10), but it shows the size of the difference. β₁ and junction density sit just above it; length density sits well inside it and is not distinguishable from no difference.

The three measures are internally concordant (length and junction density correlate at r = +0.965, as they must if they are counting the same structure two ways), and they rank in a physically sensible order, with loop density showing the largest separation and length density the smallest.

### 7.3 Node degree distribution

§1.1 asks for the degree distribution, not only the node count.

| Specimen | Nodes | deg 1 | deg 2 | deg 3 | deg 4 | deg ≥5 | Mean | Branch nodes |
|---|---|---|---|---|---|---|---|---|
| WKY-A | 5,589 | 784 | 174 | 4,467 | 159 | 5 | 2.72 | 82.9% |
| WKY-B | 4,545 | 681 | 115 | 3,629 | 118 | 2 | 2.70 | 82.5% |
| WKY-C | 5,064 | 715 | 209 | 3,972 | 163 | 5 | 2.71 | 81.8% |
| SHR-A | 5,979 | 715 | 147 | 4,919 | 194 | 4 | 2.77 | 85.6% |
| SHR-B | 5,034 | 632 | 98 | 4,141 | 158 | 5 | 2.76 | 85.5% |
| SHR-C | 4,724 | 714 | 114 | 3,775 | 120 | 1 | 2.70 | 82.5% |

> **Figure 7.** `figure7_node_degree.png`. The degree distribution, three lines per cohort, near-superimposed. The shape is the same in both groups: a dominant degree-3 population with degree-1 tips an order of magnitude below and degree-5 nodes two orders below that.

The branch-node fraction (degree ≥ 3) no longer separates by cohort: WKY 81.8–82.9%, SHR 82.5–85.6%. SHR-A and SHR-B sit above every WKY specimen, but SHR-C (82.5%) ties WKY-B and sits below WKY-A. On the earlier regions it separated by 0.4 percentage points; that was too weak to carry weight then, and it has not held.

### 7.4 Segment length

Segment length distributions are near-identical between cohorts, overlapping across the whole range: median segment length is 10.1–11.3 µm in WKY and 10.3–10.7 µm in SHR, and mean segment length 12.7 against 12.4 µm. Combined with §7.2 this locates what difference there is: **SHR networks carry slightly more segments, not longer ones** (7,204 against 6,867 edges on average, 1.05×). The small length density excess comes from segment count, which is consistent with the junction and loop densities and inconsistent with elongation of an unchanged network.

The distribution also explains a limitation quantitatively. 30–35% of segments are shorter than 7.46 µm, twice the junction exclusion, and therefore cannot have the junction radius correction applied at all (§11.2).

> **Figure 8.** `figure8_segment_length.png`. Segment length, one line per specimen, with twice the junction exclusion (2 × 3.73 µm) marked. The junction radius correction reaches 62–67% of edges.

> **Figure 1.** `figure1_network_density.png`. Three panels, one per measure, each specimen plotted individually with the group mean as a rule, at threshold 0.95 on the placed ROI. No bars: at n = 3 a bar of group means would conceal the overlap and imply a precision three specimens cannot support. SHR is +8.7% (β₁), +6.4% (junctions) and +2.2% (length) above WKY, the groups overlap in all three, and the exact permutation p is 0.60 / 0.60 / 0.80. The figure reads these from the per-edge tables, as §7.1 does.

---

## 8. Results: §1.2 EDM geometric profiling

### 8.1 Per-specimen distributions

| Specimen | Group | Edges | Median | Mean | p25 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| WKY-A | WKY | 7,597 | 7.46 | 7.80 | 5.28 | 9.14 | 11.79 |
| WKY-B | WKY | 6,140 | 7.08 | 7.01 | 5.27 | 8.34 | 10.55 |
| WKY-C | WKY | 6,863 | 6.96 | 7.05 | 5.27 | 8.34 | 10.56 |
| SHR-A | SHR | 8,281 | 6.46 | 6.70 | 4.50 | 8.34 | 10.55 |
| SHR-B | SHR | 6,954 | 6.37 | 6.48 | 4.50 | 7.90 | 9.45 |
| SHR-C | SHR | 6,376 | 5.87 | 6.48 | 4.50 | 7.90 | 9.77 |

All values in µm, EDT diameter per edge. Mean of the specimen medians: WKY 7.17 µm, SHR 6.23 µm, so SHR is 13% narrower (ratio 0.869). Unlike every other measure, the medians separate completely by cohort (exact p 0.10, the floor at n = 3).

### 8.2 Why this is not reported as a finding

On the earlier regions at threshold 0.90 the medians also separated, by 0.10 µm, one twentieth of a quantisation step, and that alone disqualified the result. The gap is now five times larger, so that argument is weaker. Four others now carry the verdict.

| | |
|---|---|
| WKY minimum | 6.96 µm |
| SHR maximum | 6.46 µm |
| Between-group gap | **0.50 µm** |
| Difference of group means | 0.94 µm |
| One EDM quantisation step | **1.87 µm** |
| Gap as a fraction of one step | **0.27** |

**The gap is still below one voxel.** The distance transform on a discrete grid can only return certain distances; each specimen's diameters take only 219–428 distinct values across 6,000 to 8,000 edges, so the medians are drawn from a coarse set of levels. The within-group spread is 0.50 µm (WKY) and 0.59 µm (SHR), the same size as the gap.

**The separation does not survive the neighbouring thresholds.** At 0.93 the medians are WKY 8.34, 7.46, 7.46 µm against SHR 7.46, 6.96, 6.96 µm: WKY-B and WKY-C tie SHR-A, so the groups touch. At 0.97 they are WKY 6.46, 5.87, 5.28 µm against SHR 5.28, 5.27, 5.27 µm, and WKY-C again ties SHR-A. Every §1.1 direction holds at all three thresholds (§6.4); the calibre separation holds only at the frozen one.

**The threshold moves calibre unevenly by cohort.** One grid step of threshold (0.93 → 0.95) shifts median calibre by 0.740 µm on average, 0.585 µm in WKY against 0.895 µm in SHR (`cb_modelling_reference.md` §13.2, §13.3). A group-correlated shift of that size is larger than the 0.50 µm gap, and it is the right shape to manufacture one. Median diameter also rises with the foreground fraction (r +0.62, §6.3).

**The estimator is biased by about one voxel.** The EDT measures to the nearest background voxel centre, so it overstates each diameter by roughly one voxel (reference open item 42). To first order the bias is common to both cohorts, but it is twice the gap, and it has not been measured per edge.

The absolute values no longer disqualify the measure on their own: four of the six medians lie in the 4–7 µm capillary window (WKY-A and WKY-B are just above it). Taking one voxel off would put all six inside it, SHR-C at its lower edge. So the over-inclusion that §2.3's incomplete labelling predicts is no longer visible in calibre; it may still be present.

> **Figure 2.** `figure2_diameter_distribution.png`. Left: cumulative distribution per specimen with the 1.87 µm quantisation grid drawn, so the discreteness of the measurement is visible rather than smoothed away. Right: the six specimen medians against one measurement step, all fitting inside it. The medians separate completely (SHR < WKY; WKY 6.96–7.46 µm, SHR 5.87–6.46 µm) with a gap of 0.50 µm, 0.27 of one step, and exact p = 0.10, the floor at n = 3. Four of the six medians lie in the 4–7 µm window. The diameters are EDT values without the half-voxel bias correction (reference open item 42).

---

## 9. Results: §1.4 tortuosity

| | WKY | SHR |
|---|---|---|
| Mean tortuosity | 1.153 | 1.142 |

Per specimen (mean over edges of centreline length over chord): WKY 1.152, 1.142, 1.165; SHR 1.142, 1.142, 1.142. The difference is −0.9% (ratio 0.991) and the groups overlap (exact p 0.40). The median edge tortuosity gives the same answer, ratio 1.001. It is −1.5% and −0.4% at the neighbouring thresholds 0.93 and 0.97.

**This measure shows no difference.** It was withheld on the earlier regions because it correlated with segmentation inclusiveness at r = +0.86. On the current batch that correlation is r = +0.11 (Spearman ρ = +0.31, §6.3), so the confound has gone; what is left is no measurable difference between cohorts.

On the earlier regions the two measures that correlated with segmentation (§1.2 calibre, §1.4 tortuosity) were the two disqualified and the three that did not (§7.2) were the three retained, and this section read that as evidence that the §6 validation had discriminating power. That pattern has not held: calibre still correlates, tortuosity no longer does, and the three retained densities now do (§6.3). The validation is no longer evidence of its own discriminating power.

---

## 9A. Results: §1.3 proportional capillary density and §1.5 tissue-to-vessel distance

Both are reported for the full cohort. They are placed together because they share an input, a region and a sensitivity analysis.

### 9A.1 What is being measured

§1.3 asks for the parenchymal volume of the TH-positive glomus clusters, and for centreline length density *within* those clusters rather than within the whole region. §1.5 asks for the distance from every TH-positive voxel to the nearest lectin-positive centreline.

Both are joins between the two channels, which is only sound because they are two channels of a single `ZCYX` acquisition on an identical grid: co-registered by construction, with no registration step to introduce error. Both channels are cropped to the same placed region.

**The vessel side is the batch network** (re-quoted 2026-09-30, `cb_modelling_reference.md` open item 40). Both measures read the graph, skeleton and mask of the `cb_h1_batch` run, which come from the frozen hysteresis band (0.95, seed 0.999), so §1.3 and §1.5 describe the same vessels as §1.1. Until then this section cut the probability map plainly at 0.90 and skeletonised that itself, a different vessel set with about 1.6 times the skeleton voxels.

Two definitional choices are worth stating because the obvious alternative is wrong in each case.

**Length is the network's edge polylines, not a count of voxels or voxel pairs.** Counting skeleton voxels understates a diagonal path by up to √3; summing every 26-adjacent voxel pair fixes that but counts all three links where three voxels meet at a corner, 9 to 28% too much. The polyline length has neither bias, and it is the length §1.1 uses: the total equals the sum over `per_edge_morphometry.csv` in every specimen (WKY-A 96.96 mm). Each edge is split into sub-steps of half a voxel and the share of sub-step midpoints inside the TH mask sets the length inside, so a boundary crossing is placed to within about 1 µm. The lookup is centred on the voxel; until item 40 it was half a voxel off on every axis.

**Distance is to the centreline, not the vessel surface.** The two differ by the local radius. On a capillary that is roughly 1.5 µm everywhere, which would be absorbed into any group difference rather than appearing as one, and §1.2 has already established that this instrument cannot resolve calibre well enough to correct for it.

### 9A.2 Per-specimen values

At TH probability > 0.5, on the batch network, in the same 0.0266 mm³ region as §1.1.

| Specimen | Group | TH volume (mm³) | TH % of region | Centreline in region (mm) | Centreline within TH (mm) | §1.3 length density (mm·mm⁻³) | §1.5 TVD median (µm) |
|---|---|---|---|---|---|---|---|
| WKY-A | WKY | 0.00555 | 20.87% | 96.96 | 13.22 | 2,382.4 | 8.55 |
| WKY-B | WKY | 0.00885 | 33.28% | 81.53 | 19.95 | 2,255.4 | 9.14 |
| WKY-C | WKY | 0.00625 | 23.51% | 83.54 | 15.16 | 2,425.3 | 8.55 |
| SHR-A | SHR | 0.00595 | 22.38% | 101.02 | 19.68 | 3,307.8 | 7.92 |
| SHR-B | SHR | 0.00397 | 14.92% | 87.28 | 12.08 | 3,044.7 | 7.92 |
| SHR-C | SHR | 0.00258 | 9.70% | 79.41 | 5.16 | 1,998.8 | 9.70 |

Tissue-to-vessel distributions, over 0.40 to 1.36 million TH-positive voxels per specimen:

| Specimen | p25 | median | p75 | p90 |
|---|---|---|---|---|
| WKY-A | 5.89 | 8.55 | 11.50 | 14.33 |
| WKY-B | 5.90 | 9.14 | 11.94 | 14.69 |
| WKY-C | 5.90 | 8.55 | 11.50 | 14.57 |
| SHR-A | 5.60 | 7.92 | 11.03 | 14.21 |
| SHR-B | 5.60 | 7.92 | 10.88 | 13.58 |
| SHR-C | 6.72 | 9.70 | 13.71 | 18.85 |

The distances are longer than they were on the plain-cut skeleton (7.69 µm in every WKY specimen), because the network's skeleton is sparser.

### 9A.3 Group comparison

| Measure | WKY | SHR | Ratio | Ranges overlap? |
|---|---|---|---|---|
| §1.3 parenchymal volume | 0.00688 mm³ | 0.00417 mm³ | **0.61** | Yes |
| §1.3 length density | 2,354 mm·mm⁻³ | 2,784 mm·mm⁻³ | **1.18** | Yes |
| §1.5 TVD median | 8.75 µm | 8.51 µm | **0.97** | Yes |

**Nothing separates the cohorts.** On the plain-cut vessel set, length density did (3,208 against 4,081, no overlap). On the network it does not: SHR-A and SHR-B sit well above every WKY specimen (3,308 and 3,045 against at most 2,425), but SHR-C, at 1,999, sits below every one. The mean ratio of 1.18 is two specimens against one.

**Parenchymal volume is lower in SHR on average, with overlap.** SHR-A's TH volume (0.00595 mm³) exceeds WKY-A's (0.00555), so the 0.61 ratio is a difference in group means whose ranges intersect. It barely moved with the change of vessel set, as it should: it does not involve the vessels.

**Tissue-to-vessel distance does not differ.** WKY runs 8.55 to 9.14 µm and SHR 7.92 to 9.70; SHR-A and SHR-B are closer to their vessels than any WKY specimen, SHR-C further than any.

Read together, the direction is still **less TH-positive parenchyma, more densely vascularised** in SHR, but it now rests on SHR-A and SHR-B, with SHR-C on the other side of WKY in both density and distance. SHR-C also has the smallest glomus volume by a factor of 1.5 and the fewest vessels inside it (5.2 mm of centreline), so it is the least precisely measured specimen of the six. The direction is still *not* consistent with the prior §1.3 cites, which is addressed in §10.

### 9A.4 Threshold sensitivity

The TH threshold is frozen at 0.5 (`cb_settings.TH_THRESHOLD`) without a selection exercise of its own, so all three values are reported.

| TH threshold | WKY volume (mm³) | SHR volume (mm³) | Ratio | WKY density | SHR density | Ratio | WKY TVD | SHR TVD | Ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.00688 | 0.00417 | 0.61 | 2,354 | 2,784 | 1.18 | 8.75 | 8.51 | 0.97 |
| 0.7 | 0.00584 | 0.00353 | 0.60 | 2,197 | 2,663 | 1.21 | 8.87 | 8.63 | 0.97 |
| 0.9 | 0.00444 | 0.00266 | 0.60 | 1,978 | 2,494 | 1.26 | 9.07 | 8.82 | 0.97 |

Absolute parenchymal volume falls by a third across the range, so **no absolute level here is a result**. Parenchymal volume and tissue-to-vessel distance hold their ratios within 0.01. Length density strengthens as the threshold rises (1.18 to 1.26), the direction expected if a more conservative TH mask keeps the densest parenchyma, but the overlap does not go away: SHR-C is below every WKY specimen at every threshold.

### 9A.5 Sensitivity to the TH labelling

> **Measured on the old vessel set.** The three-classifier comparison below was run when §9A used
> the plain-cut 0.90 skeleton and the voxel-pair length. It has not been repeated on the batch
> network. It tests whether the TH labelling can produce the contrast, which does not depend on
> the vessel set, but the density ratios it quotes (1.27 to 1.28) are the old ones.

The quantity most likely to manufacture a spurious group difference is the composition of the TH training set. A classifier whose positive examples come predominantly from one cohort may not generalise to the other; one whose labels are overwhelmingly background will under-call the class being measured; and one trained on labels far from any tissue boundary learns little about where that boundary lies. The contrast was therefore evaluated against three TH classifiers spanning all three properties.

| | Classifier A | Classifier B | Classifier C |
|---|---|---|---|
| SHR volumes carrying labels | 1 of 3 | 3 of 3 | 3 of 3 |
| Depths per SHR volume | 1 | 3 | 3 |
| SHR `glomus` labels | 1,016 | 5,577 | 11,673 |
| Cohort skew in the positive class | 22.9× | 4.2× | **2.1×** |
| Pooled `glomus`:`background` | 1:39 | 1:59 | **1:0.8** |
| Total labelled voxels | 969,389 | 1,736,470 | 65,528 |
| Median distance of background labels from TH tissue | not measured | not measured | **1.9 to 4.2 µm** |

Classifier C is not a larger version of the others. Its label set is 96% smaller and deliberately boundary-selected: background labels lie a median 1.9 to 4.2 µm from TH tissue with under 4% beyond 30 µm, against a median 62 µm and 92% beyond 20 µm for the untargeted labelling it replaced. It is a different instrument, not a better-fed one.

All six volumes were predicted from each classifier and §9A recomputed. At TH > 0.5:

| Measure | Classifier A | Classifier B | Classifier C |
|---|---|---|---|
| §1.3 parenchymal volume ratio | 0.60 | 0.60 | 0.60 |
| §1.3 length density ratio | 1.28 | 1.28 | 1.27 |
| §1.5 TVD median ratio | 0.92 | 0.92 | 0.93 |

Absolute levels moved as expected. WKY parenchymal volume runs 0.00769, 0.00778 and 0.00699 mm³ across A, B and C, the last about 10% lower, which is what better-defined negatives should do to a mask. **The ratios did not move.**

**What this establishes.** Three mechanisms by which the labelling could have produced the contrast have each been removed and the contrast is unchanged. Cohort skew fell elevenfold. The pooled class balance swung seventyfold, from a forest calibrated overwhelmingly on background to one with near-equal class weight, so the positive class had full weight to disagree and did not. And the background labels were relocated from empty field to the decision boundary, which is where a wrong boundary would show. The objection that a count-weighted forest could not have moved regardless does not apply to classifier C.

**What it does not establish.** That the segmentation is accurate in absolute terms; §9A.4 shows the absolute levels are threshold-dependent and no claim is made about them. Nor does it substitute for labelled ground truth, which does not exist for either channel (§11.1). The analysis constrains how the *ratio* could arise, not how close the masks are to truth.

### 9A.6 Two cross-checks

**The tissue-to-vessel distance is consistent with the perfusion grid's.** The H2 capability assessment measured a median of 5.3 to 7.9 µm from the perfusion grid on the old boxes, to decide whether a grid cell could resolve the oxygen gradient; `cb_modelling_reference.md` §13.7 measures 4.6 to 6.2 µm from every non-vessel voxel to the network's mask. §1.5 gives 7.9 to 9.7 µm from glomus voxels to the network's centreline. The differences are the expected ones (centreline against surface, glomus tissue against all tissue), and all three put tissue within about 10 µm of a vessel, against an oxygen diffusion length of 20 to 45 µm.

**Length density within TH does not track length per region any more.** On the batch network the total centreline in the region is 1.02 times higher in SHR (WKY 87.3 mm, SHR 89.2 mm), while length per unit TH volume is 1.18 times. SHR actually has *less* centreline inside TH (12.3 mm against 16.1 mm on average), but its TH volume is smaller still (0.61×), so the density rises. The §1.3 contrast is the denominator, not more vessel. On the old vessel set the two agreed at 1.27, which this section used to report as a check on the TH mask.

---

## 10. Interpretation against the published prior

§1.2 of the hypothesis document cites stereological data reporting that SHR capillary network length approximately doubles (10.66 ± 0.60 mm vs 5.36 ± 0.36 mm) while mean capillary cross-sectional area decreases (20.6 ± 1.14 µm² vs 57.80 ± 1.23 µm²): a hypervascular state accommodated by elongation of narrower vessels rather than by vasodilation. The source is Ivanov, Atanasova and Lazarov, *Microvasc Res* 2026;167:104978 (doi:10.1016/j.mvr.2026.104978), checked against its abstract on 2026-10-08. It is an electron-microscopy stereology study, and its controls are normotensive Wistar rats, not WKY.

The present results point the same way on both axes, but **the length effect is too small to count as agreement**:

| | Published prior | Measured here |
|---|---|---|
| Network length | ≈ +99% | +2.2%, groups overlap |
| Vessel calibre | ≈ −40% (from area) | −13%, below one voxel and not stable across thresholds (§8.2) |

A +2% length difference, against a within-group spread of 18–24%, is not a smaller version of a doubling; it is no measurable difference. The calibre direction matches, but §8.2 does not support it as a finding.

Three readings are available and the data cannot presently distinguish them. Over-inclusive segmentation compresses differences: fusing adjacent capillaries reduces apparent branch and loop counts, and does so more in the denser cohort, which would make the measured effects lower bounds (§6.4 supports this weakly, by a few percent). The sampled region may not show it: a 0.0266 mm³ tissue-placed box measures density, not whole-organ length, and an organ can double its capillary length by growing without changing its density (§10.1 makes the same point for parenchyma). Or the prior is not directly comparable: it derives from a different preparation and quantification method (electron-microscopy stereology of the whole organ), and it compares SHR with Wistar rather than WKY controls, so part of its difference may be strain rather than hypertension.

**This comparison is orientation, not corroboration**, and on length it is not even orientation, until the classifier is final and whole-organ quantities can be measured. The cited values are confirmed against their source; their Wistar control remains a difference in design.

### 10.1 The parenchymal prior, which the measurement opposes

§1.3 of the hypothesis document states that CB parenchyma expands up to threefold in SHR through glomus and sustentacular cell hyperplasia. Unlike the §1.2 prior, this figure has no source on record: a PubMed search on 2026-10-08 did not find one. §9A.3 measures TH-positive parenchymal volume at **0.61 times** WKY (group means; the ranges overlap at SHR-A): a contraction, not an expansion, and in the opposite direction to a prior that anticipated a factor of three.

A result that opposes its prior carries a higher burden than one that confirms it, and three things are worth separating.

**The measurement is not of the same thing the prior describes.** The prior concerns whole-organ parenchymal volume. §9A measures TH-positive volume inside a 0.0266 mm³ region placed on tissue signal, which is roughly 1/40 of an organ and is placed by the same rule in every specimen. A larger organ with a lower TH-positive fraction at its centre would produce both the published expansion and the measurement here without contradiction. This document cannot distinguish the two, because it does not measure whole-organ volume (§11.2, region sampling).

**The instrument is not obviously capable of producing it as an artefact.** The mechanism that would manufacture a low SHR parenchymal volume is a TH classifier that under-calls glomus in SHR. §9A.5 removes the three ways that could arise and the ratio does not move.

**On the batch network the density contrast is the parenchyma, not the vessels.** The region carries about the same centreline in both cohorts (1.02×) and SHR carries less of it inside TH, so the higher SHR density within TH comes from the smaller TH volume (§9A.6). A smaller measured SHR parenchyma with unchanged vasculature is exactly the case that could not be told apart from a segmentation offset, so the density adds no evidence of its own.

The defensible statement is that within a matched, tissue-placed sub-volume, TH-positive parenchyma is lower in SHR on average, with overlapping ranges. Whether that scales to the whole organ, and therefore whether it genuinely opposes the published prior, is not answerable from region-sampled data. Item 5 of §13 is the work that would settle it.

---

## 11. Limitations

Limitations are separated by what they constrain. Some bound what may be *claimed*; others bound only the *precision* of a claim that stands. Conflating them is how a preliminary report becomes either overclaiming or uninterpretable hedging.

### 11.1 Limitations that bound the claims

**Statistical power.** n = 3 per group. The exact two-sided permutation p cannot fall below 0.10 for any arrangement of three against three. The topological measures give 0.60, 0.60 and 0.80; only calibre reaches the 0.10 floor, and §8.2 explains why that is not a finding. No claim of statistical significance is made anywhere in this document, and none can be made from this design. *Resolution:* more specimens, or acceptance that H1 is answered descriptively.

**The groups overlap on every topological measure, and the differences are small.** WKY-C exceeds SHR-C, and WKY-A exceeds SHR-B and SHR-C, on all three. Within-group spread is 18–33% against a between-group difference of 2–9%. The reported ratios describe group means over three specimens whose ranges intersect substantially. *Resolution:* as above.

**The densities track segmentation inclusiveness.** Across the six specimens β₁, junction and length density rise with the foreground fraction (r +0.56 to +0.77, §6.3). The foreground fraction does not differ by cohort (§6.2), so this adds noise rather than a known bias to the contrast, but it removes the evidence that the densities are independent of the classifier. *Resolution:* the boundary labelling below, then re-testing the correlation.

**The classifier is not final.** Four of six volumes lack perivascular boundary labels (§2.3). Its effect on calibre is no longer clear: median calibre is 5.9–7.5 µm, four of six inside the expected 4–7 µm, and the EDT overstates it by about one voxel (§8.2), so calibre no longer shows the masks to be over-inclusive. It does not show that they are not. Every number in this document will change when labelling is completed. *Resolution:* approximately a day of labelling plus 90 minutes of computation; the procedure and the acceptance criterion are both defined.

**The TH classifier retains a 2.1× cohort skew.** Its positive class carries 24,935 labels in WKY against 11,673 in SHR (§2.3), above the 2× reporting threshold. §9A.5 constrains what that can be doing: the contrast is unchanged across classifiers spanning 22.9× to 2.1× skew, so a further reduction would be expected to change little. It remains the stated bound on §9A because the argument is a sensitivity analysis rather than a proof. *Resolution:* level the positive class between cohorts. Hours of GUI work; prediction is 6 minutes for all six.

**No absolute accuracy claim is made for either TH quantity.** §9A.4 shows parenchymal volume falling by a third between TH thresholds 0.5 and 0.9 while the volume and distance ratios hold within 0.01 and the density ratio moves from 1.18 to 1.26. The ratios are what this document reports; the absolute levels are threshold-dependent and are not results.

**The TH threshold is not frozen.** The vessel threshold was selected by an explicit exercise and checked for a cohort split (§5.2, §6.1); no equivalent has been run for TH. §9A reports three values instead, and the absolute level moves by a third across them. *Resolution:* the same selection exercise, once the labelling supports it.

**§1.2 is below resolution.** The medians separate by 0.50 µm against a quantisation step of 1.87 µm, the groups touch at both neighbouring thresholds, one threshold step moves SHR calibre 0.31 µm further than WKY, and the estimator carries an unmeasured one-voxel bias (§8.2). No claim about vessel calibre is supported. *Resolution:* measure and correct the EDT bias (reference open item 42); unbinned 1×1×1 acquisition halves the step, at the cost of complete relabelling.

**No per-cohort accuracy validation exists.** The instrument-fairness argument in §6 rests on internal consistency checks, one of which no longer passes (§6.3), not on labelled ground truth. The single-classifier design deliberately shifts the burden onto *demonstrating* fairness, and that demonstration is outstanding. *Resolution:* hand-labelled held-out regions in both cohorts, scored separately.

### 11.2 Limitations that bound the precision

**Radius quantisation.** The distance transform on a discrete grid returns 219–428 distinct diameter values across thousands of edges per specimen. A capillary radius is 1.1–1.9 voxels; half-voxel boundary error is ±31% on radius and approximately 3× on segment resistance.

**Junction correction coverage.** The junction-proximity exclusion reaches 62–67% of edges; the remaining 33–38% are shorter than twice the exclusion distance and retain a known upward radius bias. They are tagged rather than discarded, because dropping them would bias the distribution toward long vessels.

**The calibre window is asserted, not measured.** The 4–7 µm expected capillary range comes from the segmentation handover without citation and is not derived from these specimens. It selects the segmentation threshold, so every downstream quantity inherits it. Sensitivity should be reported across the plausible *width of that window*, not across a band around the chosen threshold.

**Region sampling.** Results describe a 0.0266 mm³ region per specimen, not the whole organ. Placement is on tissue signal, which samples mid-organ where the network is denser than at the periphery, so **absolute densities over-estimate the organ**. The comparison remains like-for-like because the same rule is applied to all six.

**Group-correlated acquisition difference.** The z-step differs by cohort (1.86386 µm WKY, 1.86412 µm SHR) and all six were processed at the WKY value. The discrepancy is 0.014% and changes no result reported here.

---

## 12. Claim ledger

Each claim is graded: **Established** (evidenced and robust to the known limitations), **Provisional** (evidenced but sensitive to a stated limitation), or **Not supported** (measured and disqualified, or unmeasurable).

| # | Claim | Evidence | Rests on | Grade |
|---|---|---|---|---|
| C1 | The pipeline implements §1.1 as specified: node counting, degree distribution and β₁ | §7.1, §7.3 | none | **Established** |
| C2 | The three topological measures are internally concordant | length vs junction r = +0.965; β₁ concordant | none | **Established** |
| C3 | The segmentation is not differentially biased between cohorts on the reported measures | §6.1, §6.2, §6.3 | This classifier; internal checks only. Threshold and foreground fraction do not split by cohort, but the densities now track the foreground fraction (r +0.56 to +0.77) | **Provisional** (was Established) |
| C4 | SHR carotid bodies show higher loop density (β₁ +8.7%) | §7.2 | n = 3; groups overlap (p 0.60); within-group spread 23–33%; tracks inclusiveness (§6.3); classifier not final. Was +40% on the earlier regions | **Provisional (weak)** (was Provisional) |
| C5 | SHR show higher junction density (+6.4%) | §7.2 | as C4; spread 21–28% | **Provisional (weak)** (was Provisional) |
| C6 | SHR show higher vessel length density (+2.2%) | §7.2 | Hedges' g 0.15, p 0.80; smaller than every noise measure in this document | **Not supported** (was Provisional) |
| C7 | SHR show a higher branch-node fraction | §7.3 | SHR-C ties WKY-B and is below WKY-A | **Not supported** (was Provisional, weak) |
| C8 | The direction of C4–C6 is robust to the segmentation threshold | §6.4 | Three thresholds, 9/9 group-mean comparisons; the overlap is robust too | **Established** |
| C8b | The reported effect is a lower bound on what this instrument would measure with less inclusive segmentation | §6.4 | Two clean points (0.93, 0.95); rises of 0.01 to 0.04; n = 3 group means | **Provisional** |
| C9 | The direction is consistent with the published stereological prior | §10 | Length +2% against a doubling is no measurable difference; calibre not supported (C10); prior confirmed against source (§10) but uses Wistar, not WKY, controls | **Not supported** (was Provisional) |
| C10 | SHR capillaries are narrower | §8 | Medians separate (0.50 µm gap, p 0.10), but below one voxel, not at 0.93 or 0.97, with a group-correlated threshold shift larger than the gap and an unmeasured one-voxel EDT bias | **Not supported** |
| C11 | Tortuosity differs between cohorts | §9 | Ratio 0.991, groups overlap; no longer confounded (r +0.11) | **Not supported** |
| C12 | The absolute densities represent the whole organ | §5.3 | Region placed on signal | **Not supported** |
| C13 | §1.3 and §1.5 are implemented and produce stable values | §9A.2, §9A.4 | Nine unit tests against hand arithmetic; four mutations caught | **Established** |
| C14 | Centreline length density within glomus tissue is higher in SHR (+18%) | §9A.3 | Ranges overlap: SHR-C is below every WKY value at every TH threshold; the contrast is the smaller TH volume, not more vessel (§9A.6). Separated without overlap (+27%) on the old plain-cut vessel set | **Not supported** (was Provisional) |
| C15 | TH-positive parenchymal volume is lower in SHR (0.61×) | §9A.3 | n = 3; ranges overlap (SHR-A above WKY-A); stable across TH thresholds; region-sampled, not whole-organ | **Provisional** |
| C16 | Glomus-cell tissue sits closer to the vasculature in SHR (0.97×) | §9A.3 | Ranges overlap; SHR-C is further than any WKY specimen | **Not supported** (was Provisional, weak) |
| C17 | The §9A contrast is not attributable to the TH labelling | §9A.5 | Ratios within 0.01 across three classifiers spanning 22.9× to 2.1× cohort skew and 1:59 to 1:0.8 class balance; measured on the old vessel set, not repeated on the network | **Established** for the parenchymal volume, which does not involve the vessels; not re-measured for the density |
| C18 | The §9A parenchymal result describes the whole organ | §10.1 | Measured in a 0.0266 mm³ tissue-placed region, roughly 1/40 of an organ | **Not supported** |

The document's defensible position is C1, C2, C8, C13 and C17 (Established) plus C3, C8b and C15 (Provisional) and C4 and C5 (Provisional, weak). In plain terms: the method works, and on this sample SHR networks are on average 6–9% denser in loops and junctions than WKY, with overlapping groups. Nothing else should be presented as a result.

---

## 13. Future work and unblocking path

| Priority | Work | Unblocks | Cost |
|---|---|---|---|
| 1 | Complete perivascular boundary labelling on WKY-A, WKY-C, SHR-A, SHR-C | All results; §1.2 and the §6.3 inclusiveness check in particular | Hours of GUI work; 40 min prediction; 45 min re-run |
| 2 | Level the TH positive class between cohorts, from the residual 2.1× | The stated bound on §9A; the confidence of all four H2 methods | Hours of GUI work; 6 min prediction; minutes to re-run §9A |
| 3 | Hand-labelled held-out regions in both cohorts | Per-cohort validation scores | Hours |
| ~~4~~ | ~~Verify the §1.2 stereological prior against its source~~ **Done 2026-10-08:** values confirmed (Ivanov et al. 2026); lengths are mm, controls are Wistar. The §1.3 parenchymal prior is still unsourced | §10 | Literature check |
| 5 | Out-of-core processing, or accept region sampling | Whole-organ densities | Engineering |
| 6 | Unbinned 1×1×1 data | Radius accuracy (±16% rather than ±31%) | Full relabelling; classifier does not transfer |
| 7 | Measure and correct the EDT half-voxel bias per edge (reference open item 42) | §1.2 absolute calibre; the calibre window check in §8.2 | Code and a batch re-run |

Items 1 and 2 are the only ones on the critical path to a defensible H1 result.

---

## Appendix A: Frozen parameter set

| Stage | Parameter | Value |
|---|---|---|
| Calibration | voxel size (z, y, x) | 1.8639, 1.8660, 1.8660 µm |
| Calibration | vessel class index | 0 |
| Calibration | capillary calibre window | 4.0 – 7.0 µm |
| Classification | classifier | `vessel_segmentation.ilp` (`49283a27d82e…`) |
| Preprocessing | median filter | 0 (disabled) |
| Preprocessing | morphological opening / closing | 0 / 0 (disabled) |
| Preprocessing | probability smoothing σ | 0.0 (disabled) |
| Preprocessing | hysteresis low / high | 0.95 / 0.999 (`cb_settings`; sensitivity runs move the low only) |
| Preprocessing | 3D hole filling | enabled |
| Preprocessing | Shannon entropy criterion | disengaged (two-class output) |
| Skeleton | closing radius / bridge gap | 1 / 1 |
| Skeleton | minimum branch length | 3 voxels |
| Skeleton | minimum component | 5.0% |
| Skeleton | bundle collapse | disabled |
| Skeleton | B-spline smoothing α | 0.75 |
| Graph | minimum stub length | 5.6 µm |
| Graph | largest component only | true |
| Morphometry | radius mode | `edt_radius` |
| Morphometry | junction proximity exclusion | 3.73 µm |
| Sampling | region size | 160 × 160 × 160 voxels (0.0266 mm³) |

## Appendix B: Reproduction

Every number in §5 to §9A derives from artefacts under `examples/outputs/cb_h1_batch/` and `examples/outputs/cb_h1_sensitivity/`, produced by the driver `examples/cb_h1_batch.py`.

```
python examples/cb_h1_batch.py --stage placement            # region placement (Appendix D)
python examples/cb_h1_batch.py --stage threshold            # §2.4, §5.2, §6.1, §6.2, threshold_selection.json
python examples/cb_h1_batch.py --stage run --threshold 0.95 # §7, §8, §9
python examples/cb_h1_batch.py --stage sensitivity          # §6.4, §8.2 (0.93 and 0.97, seed held at 0.999)
python examples/cb_h1_th_metrics.py --all --th-threshold 0.5 0.7 0.9  # §9A
python examples/cb_h1_figures.py                            # Figures 1, 2, 3, 7 and 8
```

| Artefact | Supplies |
|---|---|
| `<SPECIMEN>/per_edge_morphometry.csv` | §7.1 to §7.4, §8, §9 |
| `<SPECIMEN>/pipeline.log` | §9 (average tortuosity, which equals the CSV mean) |
| `<SPECIMEN>/roi_placement.json` | Appendix D |
| `threshold_selection.json` | §5.2, §6.1, §6.2, fragmentation onsets (§6.4); the sweep tables in §2.4 are its stage's printed output |
| `cb_h1_sensitivity/t0.93/`, `t0.97/` | §6.4, §8.2, §9 |
| `<SPECIMEN>/*.provenance.json` | §2.8 classifier attribution |
| `figure1_network_density.png` to `figure8_segment_length.png` | Figures 1 to 8 |
| `cb_h1_th_metrics_all.json` | §9A.2 to §9A.4 |

Hedges' g in §7.2 is the bias-corrected standardised mean difference, with the 95% interval g ± 1.96·SE and SE² = (n₁ + n₂)/(n₁n₂) + d²/(2(n₁ + n₂)). The §6.3 correlations are across the six specimens against `foreground_at_frozen`.

Classifier state at the time of the run is recoverable from any specimen's provenance sidecar, and the labelling table in §2.3 from `python examples/carotid_image_to_model.py --list-specimens`.

The reference sub-volume used for the "before" measurements quoted in §3 and §4 is `z 60:110, y 120:280, x 120:280` of WKY-A.

---

## Appendix C: Audit finding to remediation map

Changes on branch `cb_pipeline_improvements_sweep`, all tagged `(#98)`, grouped by the audit finding each closes. Findings 3.1 to 3.3 and 3.5 were closed by forty-one changes between `efcaf5a` and `e72e74c`; finding 3.4 by the six named in its row.

| Audit finding (§3) | Commits | What closed it |
|---|---|---|
| **3.1** Entropy band evacuating vessel walls | `8424ea0`, `b89104c`, `510d597` | Entropy criterion gated on class count; hysteresis search range raised above the band |
| **3.2** Objectives penalising loop topology | `a079048`, `2b6d9ef`, `4bf9f88`, `5b6a507`, `0684082`, `1974cc6` | Loop and Euler terms removed; objectives reduced to fidelity terms; bundle collapse disabled; samplers seeded |
| **3.3** EDM estimator never called | `79baf86`, `89bc841`, `5c1de57`, `162b51e` | Estimator wired in and made the default; raises if it measures nothing; junction exclusion added and measured |
| **3.4** No TH channel | `0d5f471`, `c33bbdc`, `244cef5`, `c5ad164`, `82db392`, `853853c` | Preprocessing and a second two-class classifier for the TH channel; §1.3 and §1.5 implemented and tested. **Partially closed:** the between-group contrast still awaits SHR labelling (§11.1) |
| **3.5** Voxel/physical unit confusion | `2705b38`, `436143f` | Calibrated from acquisition metadata; unit handling corrected in the same change |
| Signal destruction not in the original five | `431c069`, `5be3389`, `462ac53`, `7c9563b`, `e5f4bad` | Dilation replaced by closing; median filter removed; closing stopped eroding the domain boundary |
| Provenance and reproducibility | `610da99`, `89ca8b3`, `471e062`, `06769b4`, `6b924a3`, `8e7baa9`, `ad40171`, `f237e10`, `82cef82`, `7330609`, `e5f4bad` | Per-edge provenance tags; specimen registry; pooled-classifier enforcement; artefact provenance sidecars; label-placement check |
| Threshold selection | `7b7fb04`, `e7514a2`, `6a33611` | Calibre-based selection with fragmentation veto, replacing a criterion that returns no answer |
| Experimental design | `577345b`, `335ce15`, `6b3ac92` | Matched absolute regions; tissue-centred placement; cohort-split test; per-specimen output directories |
| Reporting | `d9d7533`, `89ca8b3`, `e72e74c` | Stub threshold justified in microns; per-edge morphometry export; figures |

Three commits are corrections to claims made earlier in the same sweep (`162b51e`, `737fb6b`, `e5f4bad`), recorded in §4.5.

---

## Appendix D: Region placement record

| Specimen | Volume (z, y, x) | Region centre | Offsets from array centre | Tissue peak z |
|---|---|---|---|---|
| WKY-A | 435, 456, 507 | 230, 224, 186 | +0.029, −0.009, −0.133 | 230 |
| WKY-B | 435, 357, 351 | 106, 192, 178 | −0.256, +0.038, +0.007 | 106 |
| WKY-C | 435, 315, 255 | 189, 158, 100 | −0.066, +0.002, −0.108 | 189 |
| SHR-A | 495, 459, 345 | 157, 244, 158 | −0.183, +0.032, −0.042 | 157 |
| SHR-B | 495, 483, 399 | 230, 308, 186 | −0.035, +0.138, −0.034 | 230 |
| SHR-C | 495, 495, 381 | 164, 302, 186 | −0.169, +0.110, −0.012 | 164 |

From each specimen's `roi_placement.json`: z is the QC peak slice, y and x the grayscale centroid over the box's own slices.
