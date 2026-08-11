# Hypothesis 1 — Preliminary Results

**Carotid body microvascular morphology, SHR versus WKY**

*ImageLynx / `carotid_image_to_model` · branch `cb_pipeline_improvements_sweep` · Dale Sasis*

---

## 0. Executive summary

This document reports the first end-to-end application of the `carotid_image_to_model` pipeline to Hypothesis 1 (H1): that carotid body (CB) morphology differs between the spontaneously hypertensive rat (SHR) and the normotensive Wistar–Kyoto control (WKY).

**It is a methods-maturation milestone, not a biological finding.** That distinction is load-bearing and is maintained throughout.

A stage-by-stage audit conducted on 2026-07-30 established that the pipeline could not answer H1 — not merely imprecisely, but structurally. Two of its optimisation objectives penalised the vascular loop topology that §1.1 proposes as its readout; the Euclidean distance map (EDM) radius estimator that §1.2 names was implemented and never called; and every length in the system was expressed in uncalibrated voxel units. Forty-one changes have since closed those defects, and each closure is quantified rather than asserted.

Applied to all six specimens under one pooled classifier, one frozen parameter set, one segmentation threshold and matched sub-volumes, three independent topological measures point in the same direction:

| Measure (§1.1) | WKY | SHR | Ratio |
|---|---|---|---|
| β₁ loop density | 4.82 × 10⁴ mm⁻³ | 6.75 × 10⁴ mm⁻³ | **1.40** |
| Junction density | 1.10 × 10⁵ mm⁻³ | 1.47 × 10⁵ mm⁻³ | **1.34** |
| Vessel length density | 2.53 × 10⁶ µm·mm⁻³ | 3.21 × 10⁶ µm·mm⁻³ | **1.27** |

Four checks show that the segmentation is not producing this difference. The direction holds at every threshold tested, in all nine specimen-group comparisons, and the measured effect grows as segmentation inclusiveness falls — the behaviour predicted if over-inclusive masks are currently suppressing it.

**Three limitations bound what may be concluded.** The groups overlap on every measure, and with n = 3 per group the exact two-sided p cannot fall below 0.10. The segmentation classifier is not final: four of six volumes lack perivascular boundary labels, and vessel calibre is consequently over-estimated. Two of H1's five sub-methods (§1.3, §1.5) cannot be run at all, for want of a glomus-cell channel.

**Reproducing the analysis at three thresholds leaves the direction unchanged**, so the result is not an artefact of the one parameter that most directly controls how much tissue is called vessel.

**Verdict.** §1.1 is implemented and measuring what it was specified to measure. §1.2 is implemented but below the resolution required to support a claim. §1.4 is implemented but confounded. §1.3 and §1.5 await a capability that does not yet exist.

---

## 1. Scope — what H1 asks, and what is currently answerable

H1 and its five sub-methods are defined in `hypothesis_testing_methods.md`; that file's §-numbering is used throughout this document and across the wider documentation set.

| § | Method | Status | Reason |
|---|---|---|---|
| **1.1** | Topological node counting | **Implemented** | Graph extraction yields nodes, degree distribution and β₁ |
| **1.2** | EDM geometric profiling | **Implemented, not conclusive** | Estimator runs on 100% of edges; calibre resolution insufficient (§8) |
| **1.3** | Proportional capillary density | **Not implementable** | Requires TH-positive glomus volume; no TH channel exists |
| **1.4** | Tortuosity | **Implemented, confounded** | Correlates with segmentation inclusiveness (§9) |
| **1.5** | Tissue-to-vessel distance | **Not implementable** | Requires TH channel |

§1.3 and §1.5 are not shortcomings of this run. The Ilastik output is two-class (vessel, background); there is no parenchymal landmark to measure against. The same gap blocks all four H2 perfusion methods, which depend on TH-masked analyses.

This document therefore reports §1.1 in full, §1.2 with an explicit disqualification, and §1.4 with an explicit confound.

---

## 2. The measurement chain

Every quantity reported here is the output of an eight-stage chain. Each stage carries decisions that affect the result; this section records them and the evidence behind each.

### 2.1 Acquisition

Six carotid bodies, three per cohort, imaged by confocal fluorescence. Lectin labels the vascular endothelium and is the sole channel used; the 640 nm TH channel was verified as punctate glomus-cell clusters rather than tubular structure, and is not part of the ingest path.

Volumes are 2×2×2-binned, uint16, ZCYX. Physical voxel size, read from each acquisition's own ImageJ metadata rather than assumed:

| | z (µm) | y (µm) | x (µm) |
|---|---|---|---|
| WKY (3 volumes) | 1.86386 | 1.86600 | 1.86600 |
| SHR (3 volumes) | 1.86412 | 1.86600 | 1.86600 |

The z-step differs between cohorts by 0.014%. This is negligible for every result reported here, but it is a *group-correlated acquisition difference* and is disclosed on that basis rather than because it changes anything.

The volumes differ substantially in extent — SHR average 89 Mvoxel against WKY's 63 Mvoxel. Any raw count is therefore larger in SHR before any biology is involved, which is why every quantity in this document is a density and why sampling is size-matched (§5.3).

### 2.2 Preprocessing

Applied identically to all six volumes: channel extraction, rolling-ball background subtraction (radius 30 px ≈ 56 µm, larger than the thickest arteriole so lumens are not eroded), per-volume percentile normalisation, and multiscale Sato vesselness at two scale bands (fine σ = 1.0/1.4/2.0 px; coarse σ = 4.0/8.0 px), each scale normalised before the cross-scale maximum.

No bleach correction and no denoising were applied. The axial intensity profile of all six volumes is hump-shaped — reflecting how much tissue each slice intersects, not photobleaching — so histogram matching would inflate sparse end slices and promote their background noise to vessel intensity. Denoising is omitted because a 3×3×3 median spans 5.6 µm, wider than a capillary (§4.3).

Output is a three-channel float32 volume (grayscale, vesselness_fine, vesselness_coarse). Channel order is load-bearing: the classifier indexes features by channel position.

### 2.3 Classification

A single Ilastik pixel-classification project (`vessel_segmentation.ilp`) is used for all six volumes. This is the central experimental control. Per-cohort classifiers would confound specimen identity with classifier identity unfixably: a between-group difference in vessel count could then be a difference in the measuring instrument rather than the tissue, with no way to separate them after the fact. The registry refuses a run whose specimens do not share one project.

The error asymmetry justifies the choice. A single classifier slightly suboptimal on one cohort adds noise and biases *toward the null* — conservative. Two classifiers add systematic bias of unknown sign directly to the group contrast, and can manufacture or mask an effect.

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

### 2.4 Threshold selection

The probability threshold is chosen from vessel calibre, constrained by skeleton fragmentation — not from connected-component statistics.

The component-based criterion in the segmentation handover was tested against the real probability field and returns no answer. The largest component's voxel share never falls; it is *higher* at threshold 0.99, where the network has visibly shattered into 7,151 pieces, than at 0.70. Counting components above a 50-voxel floor is equally flat, wandering between 94 and 139 across the whole range with no structure. This is a property of the data's topology rather than of any one classifier: a vascular bed percolates, and a percolating mask stays connected long after its centreline has begun to bead.

Two measurements do discriminate. **Median inscribed diameter** moves monotonically with threshold and has an external target — an expected capillary calibre of 4–7 µm. **Skeleton endpoint density** is flat while the network is intact and climbs sharply once it beads (2.1–3.2 per mm from threshold 0.30 to 0.97, then 4.8 at 0.99, where skeleton components rise from 172 to 467). Each fragment contributes two endpoints, which is what the mask cannot see.

Calibre is therefore the objective and fragmentation the veto. Where the two never agree, the selector returns no threshold and reports which constraint failed, rather than a best-effort value.

### 2.5 Mask and skeleton

Hysteresis thresholding, three-dimensional cavity filling (lectin stains endothelium, so large vessels appear as rings; a 2D per-slice fill would close every in-plane vascular loop and destroy the topology H1 measures), morphological closing, and connected-component filtering. Skeletonisation is 3D.

### 2.6 Graph extraction and morphometry

The skeleton is traced into an undirected multigraph. Stub branches below 5.6 µm (three voxels) are pruned; centrelines are B-spline smoothed with per-edge provenance recorded; per-edge diameters are measured from the 3D EDM, sampled at centreline voxels, with junction neighbourhoods excluded (§4.2).

The primary data product is `per_edge_morphometry.csv` — one row per graph edge, fifteen columns, including both raw radius estimators and four provenance tags. Every measurement carries the record of how it was obtained, so a distribution that mixes measured and fabricated values can be separated after the fact.

### 2.7 What the measurement is made of

> **Figure 4** — `figure4_reconstruction.png`. All six regions, one 14 µm slab each: the segmented volume translucent, the analysed centrelines inside it. **Illustration, not evidence.** Each panel is a 0.0266 mm³ region and three specimens per group cannot support a visual comparison — the figure shows all six precisely so that no pair is chosen for effect, and WKY-C is visibly denser than SHR-C. A slab is shown because at 26–34% foreground the whole cube is opaque from outside.

> **Figure 5** — `figure5_measured_network.png`. The same network carrying what was measured on it. Left: centrelines coloured by EDT diameter, the estimator §1.2 reports. Right: nodes of degree ≥ 3, the branch points §1.1 counts. Quantities reported as distributions elsewhere in this document are properties of identifiable places in the network.

> **Figure 6** — `figure6_skeleton_detail.png`. Raw skeleton (left) against the analysed centrelines (right), same region. The difference is stub pruning and B-spline smoothing; the voxel staircase visible on the left is what §4.4's calibration and §4.3's operators act on.

### 2.8 Artefact provenance

Each probability map carries a sidecar recording the classifier hash that produced it, its label counts and its boundary-label placement. The pipeline reports each map as `current`, `stale`, `unknown` or `absent`. `unknown` is deliberately distinct from and worse than `stale`: a stale map has a known and wrong origin, whereas an unknown one cannot be ruled out about.

All six maps used in this document report `current` against classifier `49283a27d82e…`.

---

## 3. Why the earlier pipeline could not answer H1

An audit on 2026-07-30 examined stages 1–22 by code inspection *and* direct numerical execution on real data. Five findings mattered.

**3.1 Joint hysteresis carved a band out of the middle of the probability range.** For a two-class output, Shannon entropy is a deterministic, folded function of the vessel probability, so an entropy ceiling resolves to `p ≤ 0.369 OR p ≥ 0.631`. Measured retention: 98% of voxels at p ∈ [0.20, 0.35], **0% at p ∈ [0.40, 0.60]**. Every vessel became a core plus a detached shell with the wall evacuated — 7,627 components, Euler characteristic −18,870.

**3.2 Both optimisation objectives penalised the hypothesis.** The preprocessing objective's Euler-characteristic term and the skeleton objective's fundamental-loops term both drive vascular loop topology toward zero — precisely the quantity §1.1 proposes as its readout. Measured: a parameter set with better Dice (0.688 vs 0.598), better orphaned fraction and 29% more centreline scored *worse* (loss 246.1 vs 164.7), because the loop term was 70–86% of total loss. Both penalties scale with network density, so they suppressed the SHR/WKY difference in the false-negative direction.

**3.3 The EDM estimator §1.2 names was implemented and never called.** `radius_assignment_mode: "edt_radius"` passed validation and silently produced synthetic branch-order diameters. FWHM covered 49.2% of edges against EDT's 99.2%, and on shared edges the two were uncorrelated (Pearson r = 0.079).

**3.4 No TH/glomus channel exists.** §1.3, §1.5 and all four H2 methods unimplementable.

**3.5 Systemic voxel-versus-physical unit confusion.** Node positions and edge voxels were stored in physical units; array indexing, boundary selection and every filter radius in voxels. They agreed only because the TIFF declared no resolution and the reader returned (1, 1, 1).

---

## 4. What changed

Forty-one changes were made. Presenting them as a list would demonstrate activity rather than trustworthiness, so they are grouped by defect class, each with the measurement that demonstrates the change. The full commit-to-finding map is Appendix C.

### 4.1 Objectives that penalised the hypothesis

Both loop-topology penalty terms were removed from the tuning objectives, which were reduced to their fidelity terms. Bundle collapse — a skeletonisation step that replaced dense regions with synthetic paths — was disabled after measurement showed it **destroyed 68% of β₁** (307 loops reduced to 99) on the reference sub-volume. This was the single largest H1 signal loss identified.

### 4.2 Estimators that silently did not run

The EDM estimator is now the default and raises if it measures nothing, rather than falling back to synthetic diameters. Re-measured on the repaired pipeline over 1,330 edges: EDT covered 100.0% with a median diameter of 6.37 µm; FWHM covered 76.5% with a median of 8.20 µm and an unphysical maximum of 39.16 µm. The two remain only weakly correlated (r = +0.245), and EDT is used throughout.

A junction-proximity exclusion was added. Within approximately one radius of a bifurcation the distance transform returns the junction's inscribed sphere rather than the vessel's, biasing radii upward. On a controlled fixture two identical tubes read 6.00 µm and 4.47 µm apart for no reason other than segment length — a 34% error, which Poiseuille resistance carries as 3.2×. On real data the population effect is approximately 8% on resistance, and **61% of segments are too short to trim at two voxels** and therefore retain the bias; those are tagged rather than discarded, because dropping them would bias the distribution toward long vessels.

### 4.3 Signal destroyed before measurement

The entropy criterion is now gated on classifier class count and disengages for two-class output. Median filtering of the probability field was removed: at equal speckle suppression it destroyed **80% of the true vessel** (900 → 181 foreground voxels against a clean-truth 900), where a post-threshold size filter achieved identical component reduction at 100% recall.

Morphological closing was found not to be a closing. A closing is extensive — X ⊆ X•B — and this implementation *removed* foreground, because the underlying library erodes against a zero border. On a boundary-crossing tube the entire first and last slice vanished; on the integration fixture it deleted seven voxels, **every one of them a vessel voxel touching a domain face** — that is, 100% of the population determining where a vessel terminates, and therefore which nodes become inlets and outlets.

### 4.4 Units and provenance

Voxel size was calibrated from acquisition metadata, replacing the (1, 1, 1) fallback, with the physical/voxel unit handling corrected in the same change so the benchmark suite could not break silently. Every diameter, centreline and radius now carries a provenance tag; every probability map names the classifier that produced it.

### 4.5 Claims corrected by measurement

Three assertions made during this work were overturned by subsequent measurement and are recorded here because a remediation record that reports only successful fixes is advocacy rather than evidence.

- A proposal to replace the preprocessing objective with soft Dice was retracted before implementation: measurement showed its optimum sits at a flooded mask, because scoring against the probability field reproduces the classifier's over-prediction.
- A claim that centreline-smoothing error concentrates on twisty edges was falsified by the per-edge export, which showed the affected edges to be the shortest and straightest.
- A claim that the doubled morphological closing compounded vessel fusion was wrong: closing is idempotent for a fixed structuring element, so the second call had never had any effect at all.

---

## 5. Experimental design for this run

### 5.1 One classifier, one parameter set

All six specimens were segmented with the same Ilastik project and processed with an identical, frozen parameter set (Appendix A). No per-specimen or per-cohort tuning was performed at any stage.

### 5.2 One threshold

Each specimen's own optimal threshold was computed, then a single value was frozen for all six. Per-specimen thresholds would absorb classifier differences into what would then appear as a tissue result.

| Specimen | Own optimum | Frozen |
|---|---|---|
| WKY-A, WKY-B, WKY-C, SHR-A | 0.90 | 0.90 |
| SHR-B, SHR-C | 0.85 | 0.90 |

Four of six specimens land inside the 4–7 µm capillary window at the frozen threshold; SHR-B and SHR-C sit just below it, which is why they individually selected 0.85. One threshold cannot be optimal for all six; that is the cost of freezing it, and it is the correct cost to pay.

### 5.3 Matched, tissue-centred sub-volumes

Whole volumes are not reachable on the available hardware. Measured scaling on WKY-C: 0.12 Mvoxel took 11 s at 0.77 GB; 4.10 Mvoxel took 348 s at 3.93 GB — approximately one gigabyte per million voxels, superlinear in time. A complete volume is 34.9 Mvoxel, which extrapolates beyond available memory.

Each specimen therefore contributes an identical **160 × 160 × 160 voxel (0.0266 mm³)** region. A *percentage* crop would not be a matched sample: at 45% it yields 3.13 Mvoxel from WKY-C and 8.62 Mvoxel from SHR-B, restoring the extent confound the density normalisation exists to remove.

Placement is computed from each volume's own data rather than centred on the array: z from the axial tissue peak recorded during preprocessing, y and x from the grayscale centroid. The tissue peak ranges from slice 106 of 435 (WKY-B) to 230 of 435 (WKY-A), so a centred box would land mid-organ in one specimen and in the sparse margin of another — and the misplacement is group-correlated, with WKY peaking at a mean depth fraction of 0.40 against SHR's 0.34.

**This trade is stated rather than hidden.** Centring on signal samples the middle of the organ, which is denser than its periphery, so the absolute densities reported here **over-estimate the whole organ**. Applying the same rule to all six keeps the comparison like-for-like, which is what a between-group claim requires; an absolute density quoted from these regions would be wrong.

---

## 6. Instrument validation — is the segmentation producing the difference?

This is the question on which everything else depends. If the segmentation behaves differently on the two cohorts, then a measured group difference is partly the measuring device, and no amount of downstream care recovers the biology.

Four checks were applied. The first three are reported here; the fourth is §6.4.

### 6.1 Does the selected threshold separate by cohort?

Each specimen's independently selected optimum:

| WKY | SHR |
|---|---|
| 0.90, 0.90, 0.90 | 0.85, 0.85, **0.90** |

SHR-C selects the same value as all three WKY specimens, so the sets interleave rather than separate. **No cohort split.**

### 6.2 Does the foreground fraction separate by cohort at the frozen threshold?

| WKY-A | WKY-B | WKY-C | SHR-A | SHR-B | SHR-C |
|---|---|---|---|---|---|
| 0.3386 | 0.2729 | 0.2613 | 0.2931 | 0.2811 | 0.2598 |

WKY spans 0.261–0.339, SHR spans 0.260–0.293 — fully overlapping and interleaved. **No cohort split.**

This matters more than it appears. Foreground fraction at a fixed threshold is close to what H1 measures. Had it separated, the group contrast would have been partly instrumental regardless of anything downstream.

### 6.3 Does each reported measure track segmentation inclusiveness?

A measure that correlates with how much the classifier includes is measuring the classifier.

| Measure | Pearson r | Spearman ρ | Verdict |
|---|---|---|---|
| β₁ loop density | −0.199 | +0.086 | Independent |
| Junction density | −0.212 | +0.086 | Independent |
| Vessel length density | −0.227 | +0.086 | Independent |
| Median diameter | +0.457 | +0.377 | Weakly related |
| **Mean tortuosity** | **+0.864** | **+0.657** | **Confounded** |

The three measures reported as results are independent of inclusiveness. Tortuosity is not, and is withheld on that basis (§9).

### 6.4 Does the result survive a change of threshold — and does it behave as predicted?

The whole analysis was repeated at thresholds 0.85 and 0.95, giving three complete six-specimen runs.

This tests two distinct things. First, **robustness**: does the direction hold if the one frozen parameter that most directly controls how much tissue is called vessel is moved? Second, a **prediction**. If the masks are over-inclusive (§8.2) and over-inclusion merges adjacent capillaries — merging them more in the denser cohort, because its vessels are closer together — then the measured SHR excess is suppressed, and reducing inclusion should *increase* it. That prediction was stated before the sweep was run.

**Group ratio (SHR / WKY) by threshold:**

| Measure | 0.85 | 0.90 | 0.95 | Direction |
|---|---|---|---|---|
| β₁ loop density | 1.297 | 1.401 | 1.505 | increases |
| Junction density | 1.229 | 1.339 | 1.419 | increases |
| Vessel length density | 1.230 | 1.267 | 1.308 | increases |

**Robustness.** SHR exceeds WKY in all nine specimen-group comparisons — three measures × three thresholds. The groups overlap at every threshold and no exact p falls below 0.20. The direction of C4–C6 does not depend on the threshold chosen.

**The prediction holds, on the interval where it can be tested cleanly.** All three ratios increase monotonically as inclusion falls.

One qualification is necessary. At threshold 0.95, four of the six specimens (WKY-A, WKY-C, SHR-A, SHR-C) are at or beyond their individually determined fragmentation onset, where a single vessel begins to break into multiple graph edges and loops are created artefactually. The 0.95 column is therefore directionally consistent but contaminated, and should not be read quantitatively. **The clean interval is 0.85 → 0.90**, where all six specimens sit below their fragmentation onset; across it β₁ rises 1.297 → 1.401, junction density 1.229 → 1.339, and length density 1.230 → 1.267.

Two points define a direction, not a trend. The prediction is supported rather than established, and it rests on group means over three specimens each.

A competing explanation was considered and is not supported by the data. If raising the threshold simply eroded thin vessels away, and SHR vessels are the narrower of the two (as the published prior in §10 holds), then SHR should lose *more* structure as the threshold rises and the ratio should fall. It rises. The merging interpretation is the one consistent with the observation.

**Consequence for the headline result.** The values reported at the frozen threshold of 0.90 are lower bounds on the effect this instrument would measure with less inclusive segmentation. That is the direction the incomplete boundary labelling (§2.3, §11.1) is expected to move them when it is finished.

> **Figure 3** — `figure3_threshold_sensitivity.png`. Group ratio against threshold for the three topological measures. Solid segments span the clean interval where every specimen sits below its fragmentation onset; dashed segments and the shaded band mark where fragmentation contaminates the measurement. The horizontal rule at 1.0 is no difference between cohorts.

**All four checks are internal.** They demonstrate that the segmentation is not *differentially* biased between cohorts on the quantities reported, and that the result is not an artefact of the one threshold chosen. They do not establish that the segmentation is accurate in absolute terms — that requires hand-labelled held-out regions scored separately per cohort, which do not yet exist (§11.1).

---

## 7. Results — §1.1 topological node counting

All six specimens completed. Edge counts range from 3,932 (WKY-B) to 8,077 (SHR-B).

### 7.1 Per-specimen values

| Specimen | Group | Edges | Nodes | β₁ | β₁ mm⁻³ | Junctions mm⁻³ | Length µm·mm⁻³ |
|---|---|---|---|---|---|---|---|
| WKY-A | WKY | 4,512 | 3,399 | 1,114 | 41,906 | 96,490 | 2.21 × 10⁶ |
| WKY-B | WKY | 3,932 | 2,983 | 950 | 35,737 | 85,204 | 2.07 × 10⁶ |
| WKY-C | WKY | 6,699 | 4,921 | 1,779 | 66,922 | 148,553 | 3.32 × 10⁶ |
| SHR-A | SHR | 6,815 | 4,900 | 1,916 | 72,076 | 152,653 | 3.35 × 10⁶ |
| SHR-B | SHR | 8,077 | 5,894 | 2,184 | 82,157 | 181,205 | 3.89 × 10⁶ |
| SHR-C | SHR | 4,865 | 3,583 | 1,283 | 48,264 | 108,415 | 2.40 × 10⁶ |

β₁ = E − V + C is the count of independent cycles — the "number of vascular loops" §1.1 names as its measure of pathological network disorganisation. C = 1 by construction, the graph being reduced to its largest connected component.

### 7.2 Group comparison

| Measure | WKY mean | SHR mean | Ratio | Hedges' g | g 95% CI | Pairwise ratio range |
|---|---|---|---|---|---|---|
| β₁ loop density | 4.82 × 10⁴ | 6.75 × 10⁴ | 1.40 | 0.91 | [−0.81, +2.63] | 0.72 – 2.30 |
| Junction density | 1.10 × 10⁵ | 1.47 × 10⁵ | 1.34 | 0.85 | [−0.86, +2.55] | 0.73 – 2.13 |
| Vessel length density | 2.53 × 10⁶ | 3.21 × 10⁶ | 1.27 | 0.75 | [−0.93, +2.43] | 0.72 – 1.88 |

**Every confidence interval spans zero.** Effect sizes are reported because they are more informative than a p-value at this n, not because they are conclusive. The pairwise ratio range is the interpretable quantity: for every measure, at least one WKY specimen exceeds at least one SHR specimen.

The three measures are internally concordant — length and junction density correlate at r = +0.998, as they must if they are counting the same structure two ways — and rank in a physically sensible order, with loop density showing the largest separation and length density the smallest.

### 7.3 Node degree distribution

§1.1 asks for the degree distribution, not only the node count.

| Specimen | Nodes | deg 1 | deg 2 | deg 3 | deg 4 | deg ≥5 | Mean | Branch nodes |
|---|---|---|---|---|---|---|---|---|
| WKY-A | 3,399 | 544 | 210 | 2,523 | 119 | 3 | 2.65 | 77.8% |
| WKY-B | 2,983 | 534 | 112 | 2,245 | 89 | 3 | 2.64 | 78.3% |
| WKY-C | 4,921 | 674 | 195 | 3,879 | 168 | 5 | 2.72 | 82.3% |
| SHR-A | 4,900 | 545 | 172 | 3,997 | 180 | 6 | 2.78 | 85.4% |
| SHR-B | 5,894 | 754 | 212 | 4,746 | 172 | 10 | 2.74 | 83.6% |
| SHR-C | 3,583 | 503 | 116 | 2,864 | 97 | 3 | 2.72 | 82.7% |

> **Figure 7** — `figure7_node_degree.png`. The degree distribution, three lines per cohort, near-superimposed. The shape is the same in both groups: a dominant degree-3 population with degree-1 tips an order of magnitude below and degree-5 nodes two orders below that.

The branch-node fraction (degree ≥ 3) separates by cohort — WKY 77.8–82.3%, SHR 82.7–85.4% — but by 0.4 percentage points against a within-group spread of 4.5. Complete separation of three against three arises by chance with probability 0.10. This is consistent in direction with §7.2 and is too weak to carry weight alone.

### 7.4 Segment length

Segment length distributions are near-identical between cohorts, overlapping across the whole range. Combined with §7.2 this locates the difference: **SHR networks carry more segments, not longer ones.** Vessel length density rises because segment count rises, which is consistent with the same reading as the junction and loop densities and inconsistent with elongation of an unchanged network.

The distribution also explains a limitation quantitatively. About a third of segments are shorter than 7.46 µm — twice the junction exclusion — and therefore cannot have the junction radius correction applied at all (§11.2).

> **Figure 8** — `figure8_segment_length.png`. Segment length, one line per specimen, with twice the junction exclusion marked.

> **Figure 1** — `figure1_network_density.png`. Three panels, one per measure, each specimen plotted individually with the group mean as a rule. No bars: at n = 3 a bar of group means would conceal that WKY-C exceeds SHR-C and imply a precision three specimens cannot support.

---

## 8. Results — §1.2 EDM geometric profiling

### 8.1 Per-specimen distributions

| Specimen | Group | Edges | Median | Mean | p25 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| WKY-A | WKY | 4,512 | 8.35 | 9.37 | 7.20 | 11.49 | 13.96 |
| WKY-B | WKY | 3,932 | 8.34 | 8.82 | 6.46 | 10.88 | 13.45 |
| WKY-C | WKY | 6,699 | 7.90 | 8.17 | 5.87 | 9.84 | 11.80 |
| SHR-A | SHR | 6,815 | 7.46 | 7.91 | 5.28 | 9.45 | 11.80 |
| SHR-B | SHR | 8,077 | 7.80 | 8.14 | 5.28 | 10.16 | 12.38 |
| SHR-C | SHR | 4,865 | 7.46 | 7.66 | 5.28 | 9.13 | 11.74 |

All values in µm. WKY median 8.20 µm, SHR 7.58 µm — SHR 8% narrower, and unlike every other measure it separates completely by cohort.

### 8.2 Why this is not reported as a finding

**The separation is smaller than the measurement's resolution.**

| | |
|---|---|
| WKY minimum | 7.90 µm |
| SHR maximum | 7.80 µm |
| Between-group gap | **0.10 µm** |
| One EDM quantisation step | **1.87 µm** |
| Gap as a fraction of one step | **0.054** |

The distance transform on a discrete grid can only return certain distances; each specimen's diameters take 293–484 distinct values across thousands of edges. A separation that sits at one twentieth of the smallest resolvable difference is a coincidence of where six medians happened to fall. The within-group spread is 0.45 µm (WKY) and 0.34 µm (SHR) — three to four times the gap.

The absolute values disqualify the measure independently: median calibre is 7.5–8.4 µm against an expected capillary range of 4–7 µm. The masks are over-inclusive, as the incomplete boundary labelling (§2.3) predicts. Diameters will change when that labelling is completed.

> **Figure 2** — `figure2_diameter_distribution.png`. Left: cumulative distribution per specimen with the 1.87 µm quantisation grid drawn, so the discreteness of the measurement is visible rather than smoothed away. Right: the six group medians against one measurement step, all fitting inside it. The 0.10 µm gap is deliberately not drawn — an arrow for it is illegible at any scale that also shows 1.87 µm, which is the finding rather than a limitation of the figure.

---

## 9. Results — §1.4 tortuosity

| | WKY | SHR |
|---|---|---|
| Mean tortuosity | 1.173 | 1.153 |

The difference is −2% and the groups overlap.

**This measure is withheld, on evidence.** Mean tortuosity correlates with segmentation inclusiveness at **r = +0.86** across the six specimens (Spearman ρ = +0.66). Fatter masks produce more convoluted skeletons, so the quantity being measured is substantially the segmentation's inclusiveness rather than the vessel geometry §1.4 describes. Until the classifier is final, no tortuosity comparison should be presented.

That the two measures which correlate with segmentation (§1.2 calibre, §1.4 tortuosity) are the two disqualified, while the three that do not (§7.2) are the three retained, is itself evidence that the validation in §6 has discriminating power rather than being a formality.

---

## 10. Interpretation against the published prior

§1.2 of the hypothesis document cites stereological data reporting that SHR capillary network length approximately doubles (10.66 m vs 5.36 m) while mean capillary cross-sectional area decreases (20.6 µm² vs 57.8 µm²) — a hypervascular state accommodated by elongation of narrower vessels rather than by vasodilation.

The present results are **directionally consistent on both axes and substantially smaller in magnitude**:

| | Published prior | Measured here |
|---|---|---|
| Network length | ≈ +99% | +27% |
| Vessel calibre | ≈ −40% (from area) | −8%, below resolution |

Two readings are available and the data cannot presently distinguish them. Over-inclusive segmentation compresses differences — fusing adjacent capillaries reduces apparent branch and loop counts, and does so more in the denser cohort — which would make the measured effects lower bounds. Alternatively the prior is not directly comparable: it derives from a different preparation and quantification method, and the hypothesis document itself flags these values as requiring verification against their source before use.

**This comparison should be treated as orientation, not corroboration**, until both the classifier is final and the provenance of the cited values is confirmed.

---

## 11. Limitations

Limitations are separated by what they constrain. Some bound what may be *claimed*; others bound only the *precision* of a claim that stands. Conflating them is how a preliminary report becomes either overclaiming or uninterpretable hedging.

### 11.1 Limitations that bound the claims

**Statistical power.** n = 3 per group. The exact two-sided permutation p cannot fall below 0.10 for any arrangement of three against three; every p reported here is 0.20. No claim of statistical significance is made anywhere in this document, and none can be made from this design. *Resolution:* more specimens, or acceptance that H1 is answered descriptively.

**The groups overlap on every measure.** WKY-C exceeds SHR-C on all three topological measures. Within-group spread is 63–75% against a between-group difference of 27–40%. The reported ratios describe group means over three specimens whose ranges intersect substantially. *Resolution:* as above.

**The classifier is not final.** Four of six volumes lack perivascular boundary labels (§2.3). The consequence is measurable: median calibre is 7.5–8.4 µm against an expected 4–7 µm, so masks are over-inclusive. Every number in this document will change when labelling is completed. *Resolution:* approximately a day of labelling plus 90 minutes of computation; the procedure and the acceptance criterion are both defined.

**§1.3 and §1.5 cannot be run.** No TH channel exists in the ingest path, so proportional capillary density and tissue-to-vessel distance are unimplementable, as are all four H2 methods. *Resolution:* a decision on the TH route (§13, item 2), then labelling.

**§1.2 is below resolution.** The between-group difference in median diameter is 0.10 µm against a quantisation step of 1.87 µm. No claim about vessel calibre is supported. *Resolution:* unbinned 1×1×1 acquisition halves the step, at the cost of complete relabelling.

**§1.4 is confounded.** Tortuosity correlates with segmentation inclusiveness at r = +0.86. *Resolution:* completing the boundary labelling, then re-testing the correlation.

**No per-cohort accuracy validation exists.** The instrument-fairness argument in §6 rests on three internal consistency checks, not on labelled ground truth. The single-classifier design deliberately shifts the burden onto *demonstrating* fairness, and that demonstration is outstanding. *Resolution:* hand-labelled held-out regions in both cohorts, scored separately.

### 11.2 Limitations that bound the precision

**Radius quantisation.** The distance transform on a discrete grid returns 293–484 distinct diameter values across thousands of edges per specimen. A capillary radius is 1.1–1.9 voxels; half-voxel boundary error is ±31% on radius and approximately 3× on segment resistance.

**Junction correction coverage.** The junction-proximity exclusion reaches 63–68% of edges; the remaining 32–37% are shorter than twice the exclusion distance and retain a known upward radius bias. They are tagged rather than discarded, because dropping them would bias the distribution toward long vessels.

**The calibre window is asserted, not measured.** The 4–7 µm expected capillary range comes from the segmentation handover without citation and is not derived from these specimens. It selects the segmentation threshold, so every downstream quantity inherits it. Sensitivity should be reported across the plausible *width of that window*, not across a band around the chosen threshold.

**Region sampling.** Results describe a 0.0266 mm³ region per specimen, not the whole organ. Placement is centred on tissue signal, which samples mid-organ where the network is denser than at the periphery, so **absolute densities over-estimate the organ**. The comparison remains like-for-like because the same rule is applied to all six.

**Group-correlated acquisition difference.** The z-step differs by cohort (1.86386 µm WKY, 1.86412 µm SHR) and all six were processed at the WKY value. The discrepancy is 0.014% and changes no result reported here.

---

## 12. Claim ledger

Each claim is graded: **Established** (evidenced and robust to the known limitations), **Provisional** (evidenced but sensitive to a stated limitation), or **Not supported** (measured and disqualified, or unmeasurable).

| # | Claim | Evidence | Rests on | Grade |
|---|---|---|---|---|
| C1 | The pipeline implements §1.1 as specified — node counting, degree distribution and β₁ | §7.1, §7.3 | — | **Established** |
| C2 | The three topological measures are internally concordant | length vs junction r = +0.998; β₁ concordant | — | **Established** |
| C3 | The segmentation is not differentially biased between cohorts on the reported measures | §6.1, §6.2, §6.3 | This classifier; internal checks only | **Established** |
| C4 | SHR carotid bodies show higher loop density (β₁ +40%) | §7.2 | n = 3; classifier not final | **Provisional** |
| C5 | SHR show higher junction density (+34%) | §7.2 | as C4 | **Provisional** |
| C6 | SHR show higher vessel length density (+27%) | §7.2 | as C4 | **Provisional** |
| C7 | SHR show a higher branch-node fraction | §7.3 | Gap 0.4 pp vs 4.5 pp spread | **Provisional (weak)** |
| C8 | The direction of C4–C6 is robust to the segmentation threshold | §6.4 | Three thresholds, 9/9 comparisons | **Established** |
| C8b | The reported effect is a lower bound on what this instrument would measure with less inclusive segmentation | §6.4 | Two clean points; n = 3 group means | **Provisional** |
| C9 | The direction is consistent with the published stereological prior | §10 | Prior unverified against source | **Provisional** |
| C10 | SHR capillaries are narrower | §8.1 | Gap is 1/20 of the measurement step | **Not supported** |
| C11 | Tortuosity differs between cohorts | §9 | r = +0.86 with inclusiveness | **Not supported** |
| C12 | The absolute densities represent the whole organ | §5.3 | Region centred on signal | **Not supported** |
| C13 | §1.3 and §1.5 are testable | §1 | No TH channel | **Not supported** |

The document's defensible position is C1–C3 and C8 (Established) plus C4–C6 and C8b (Provisional). Nothing else should be presented as a result.

---

## 13. Future work and unblocking path

| Priority | Work | Unblocks | Cost |
|---|---|---|---|
| 1 | Complete perivascular boundary labelling on WKY-A, WKY-C, SHR-A, SHR-C | All results; §1.2 and §1.4 in particular | Hours of GUI work; 40 min prediction; 45 min re-run |
| 2 | Decide the TH channel route (retrain three-class, or separate segmentation) | **§1.3, §1.5, and all four H2 methods** | Decision, then labelling |
| 3 | Hand-labelled held-out regions in both cohorts | Per-cohort validation scores | Hours |
| 4 | Verify the §1.2 stereological prior against its source | §10 | Literature check |
| 5 | Out-of-core processing, or accept region sampling | Whole-organ densities | Engineering |
| 6 | Unbinned 1×1×1 data | Radius accuracy (±16% rather than ±31%) | Full relabelling; classifier does not transfer |

Items 1 and 2 are the only ones on the critical path to a defensible H1 result.

---

## Appendix A — Frozen parameter set

| Stage | Parameter | Value |
|---|---|---|
| Calibration | voxel size (z, y, x) | 1.8639, 1.8660, 1.8660 µm |
| Calibration | vessel class index | 0 |
| Calibration | capillary calibre window | 4.0 – 7.0 µm |
| Classification | classifier | `vessel_segmentation.ilp` (`49283a27d82e…`) |
| Preprocessing | median filter | 0 (disabled) |
| Preprocessing | morphological opening / closing | 0 / 0 (disabled) |
| Preprocessing | probability smoothing σ | 0.0 (disabled) |
| Preprocessing | hysteresis low / high | 0.90 / 0.95 (frozen for this run) |
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

## Appendix B — Reproduction

Every number in this document derives from artefacts under `examples/outputs/cb_h1_batch/`, produced by the three-stage driver `examples/cb_h1_batch.py`.

```
python examples/cb_h1_batch.py --stage placement            # region placement (Appendix D)
python examples/cb_h1_batch.py --stage threshold            # §6.1, §6.2, threshold_selection.json
python examples/cb_h1_batch.py --stage run --threshold 0.90 # §7, §8, §9
python examples/cb_h1_figures.py                            # Figures 1 and 2
```

| Artefact | Supplies |
|---|---|
| `<SPECIMEN>/per_edge_morphometry.csv` | §7.1 (E, V, β₁), §7.3, §8.1 |
| `<SPECIMEN>/pipeline.log` | §7.1 (length, junctions), §9 |
| `threshold_selection.json` | §5.2, §6.1, §6.2 |
| `<SPECIMEN>/*.provenance.json` | §2.7 classifier attribution |
| `figure1_network_density.png`, `figure2_diameter_distribution.png` | Figures 1, 2 |

Classifier state at the time of the run is recoverable from any specimen's provenance sidecar, and the labelling table in §2.3 from `python examples/carotid_image_to_model.py --list-specimens`.

The reference sub-volume used for the "before" measurements quoted in §3 and §4 is `z 60:110, y 120:280, x 120:280` of WKY-A.

---

## Appendix C — Audit finding to remediation map

Forty-one changes on branch `cb_pipeline_improvements_sweep`, all tagged `(#98)`, from `efcaf5a` to `e72e74c`. Grouped by the audit finding each closes.

| Audit finding (§3) | Commits | What closed it |
|---|---|---|
| **3.1** Entropy band evacuating vessel walls | `8424ea0`, `b89104c`, `510d597` | Entropy criterion gated on class count; hysteresis search range raised above the band |
| **3.2** Objectives penalising loop topology | `a079048`, `2b6d9ef`, `4bf9f88`, `5b6a507`, `0684082`, `1974cc6` | Loop and Euler terms removed; objectives reduced to fidelity terms; bundle collapse disabled; samplers seeded |
| **3.3** EDM estimator never called | `79baf86`, `89bc841`, `5c1de57`, `162b51e` | Estimator wired in and made the default; raises if it measures nothing; junction exclusion added and measured |
| **3.4** No TH channel | — | **Not closed.** Requires a segmentation decision (§13) |
| **3.5** Voxel/physical unit confusion | `2705b38`, `436143f` | Calibrated from acquisition metadata; unit handling corrected in the same change |
| Signal destruction not in the original five | `431c069`, `5be3389`, `462ac53`, `7c9563b`, `e5f4bad` | Dilation replaced by closing; median filter removed; closing stopped eroding the domain boundary |
| Provenance and reproducibility | `610da99`, `89ca8b3`, `471e062`, `06769b4`, `6b924a3`, `8e7baa9`, `ad40171`, `f237e10`, `82cef82`, `7330609`, `e5f4bad` | Per-edge provenance tags; specimen registry; pooled-classifier enforcement; artefact provenance sidecars; label-placement check |
| Threshold selection | `7b7fb04`, `e7514a2`, `6a33611` | Calibre-based selection with fragmentation veto, replacing a criterion that returns no answer |
| Experimental design | `577345b`, `335ce15`, `6b3ac92` | Matched absolute regions; tissue-centred placement; cohort-split test; per-specimen output directories |
| Reporting | `d9d7533`, `89ca8b3`, `e72e74c` | Stub threshold justified in microns; per-edge morphometry export; figures |

Three commits are corrections to claims made earlier in the same sweep (`162b51e`, `737fb6b`, `e5f4bad`), recorded in §4.5.

---

## Appendix D — Region placement record

| Specimen | Volume (z, y, x) | Region centre | Offsets from array centre | Tissue peak z |
|---|---|---|---|---|
| WKY-A | 435, 456, 507 | 230, 240, 188 | +0.029, +0.026, −0.129 | 230 |
| WKY-B | 435, 357, 351 | 106, 198, 174 | −0.256, +0.055, −0.004 | 106 |
| WKY-C | 435, 315, 255 | 189, 166, 92 | −0.066, +0.027, −0.139 | 189 |
| SHR-A | 495, 459, 345 | 157, 260, 146 | −0.183, +0.066, −0.077 | 157 |
| SHR-B | 495, 483, 399 | 230, 286, 176 | −0.035, +0.092, −0.059 | 230 |
| SHR-C | 495, 495, 381 | 164, 298, 188 | −0.169, +0.102, −0.007 | 164 |
