# Hypothesis 2: Preliminary Results

*Carotid body perfusion in the spontaneously hypertensive rat against the normotensive
Wistar–Kyoto control, from a 3D vascular network model coupled to a segmented parenchymal mask.*

---

## 0. Executive summary

This document reports the first end-to-end application of the pipeline to Hypothesis 2 (H2): that
carotid body (CB) perfusion differs between the spontaneously hypertensive rat (SHR) and the
normotensive Wistar–Kyoto control (WKY).

**It is a methods-maturation milestone, not a biological finding.** That distinction is
load-bearing and is maintained throughout.

A capability assessment conducted on 2026-08-15 established that all four H2 sub-methods were
blocked, and for one shared reason: there was no glomus-cell channel to use as a spatial
landmark. That channel now exists. Clearing it exposed four further defects, each of which had
been returning a plausible number rather than an error, and each is quantified in §4.

Applied to all six specimens under one classifier per channel, one boundary rule, one frozen
threshold and matched sub-volumes, the four methods give:

| Measure | WKY | SHR | Ratio | Cohorts overlap? |
|---|---|---|---|---|
| §2.4 Transit time to glomus clusters | 1.163 | 0.833 | **0.72** | **No** |
| §2.1 Median flow, penetrating over bypassing | 0.882 | 1.077 | **1.22** | **No** |
| §2.1 Shunt index | 0.894 | 0.987 | 1.10 | Yes |
| §2.2 Haematocrit, penetrating over bypassing | 1.006 | 0.934 | 0.93 | Yes |

> **Re-derived 2026-09-27 (open item 12).** Every flow number here was first computed on a rheology
> loop that inflated each resistance 208–539× by an amount that varied with diameter
> (`cb_modelling_reference.md` §3.2). With the fix (`7ea1b36`) and the later rheology fixes, the
> ratios moved by up to 0.09 per specimen and 0.04 in cohort means; every overlap pattern held.
> Absolute velocity rose about 290×, which reverses §11.1. Old outputs:
> `examples/outputs/cb_h2_glomus_perfusion_2026-08-19_pre_item12.json`.

> **Pending re-run, 2026-09-29 (open item 37).** Every flow, haematocrit and transit number in
> this document comes from a rheology loop that stopped on its 15-pass cap in all six specimens
> without converging (`cb_modelling_reference.md` §4.3). The loop now converges in 173–457 passes
> (`cfee721`), but the H2 scripts have not been re-run on it, so the tables here still show the
> pre-fix values. A scratch comparison on the 2026-09-29 networks gives the expected size of the
> change. Its old values reproduce `cb_h2_glomus_perfusion.json`, which is newer than some tables
> here. As ratios of cohort means (SHR/WKY, old → converged): shunt index 0.889 → 0.883, median flow ratio
> 0.908 → 0.926, haematocrit ratio 1.019 → 1.056, transit ratio 1.026 → 1.038. Median haematocrit
> rises by 0.02–0.05 on both sides; SHR-C moves most (haematocrit ratio 0.998 → 1.051, flow
> ratio 0.518 → 0.595).

**The single clearest negative result is that there is no functional shunting.** §2.1 proposes
that flow in the hypertensive network bypasses the capillaries penetrating the glomus clusters.
The shunt index, flow share divided by edge share, sits at 0.89 in WKY and 0.99 in SHR: flow is
close to indifferent to the clusters in both cohorts. Blood is not being diverted away from the
chemosensors.

**Three limitations bound what may be concluded.** The groups overlap on two of the four measures,
and with n = 3 per group the exact two-sided permutation p cannot fall below 0.10. §2.3's hypoxic
fraction was zero everywhere and its glomus-specific mechanism inert; after open items 22 and 29
(two Tier 1 defects) and 30 (vessels mapped over their cross-section, 3 µm grid) it is 0–3% below
10 mmHg within TH, responds to glomus metabolism and is grid-converged, and its interpretation
**needs review** (§10). And no absolute perfusion quantity is defensible (§11.1): at 60/20 mmHg
the network runs at 1 to 3 times the upper physiological velocity, and its throughput moves 4 to 8
times with the boundary rule.

**Verdict.** §2.1, §2.2 and §2.4 are implemented, posed as within-specimen ratios, and report.
§2.3 is implemented and runs, but returns a quantity its own premise cannot support on this
geometry. Every measure that survives is a ratio, and that is not an accident: four independent
routes, calibre, viscosity, perfusion and the rheology resistance fix, each moved absolute
quantities by large factors. The first three left every ratio unchanged; the fourth, which varied
with diameter, moved ratios by up to 0.09 per specimen and left every cohort pattern intact (§6.1).

---

## 1. Scope: what H2 asks, and what is currently answerable

> **H2:** CB perfusion in a hypertensive SHR CB is different compared to the CB perfusion in a
> normotensive WKY CB.

**A note on numbering.** `§2.1` to `§2.4` always refer to H2's sub-methods as defined in
`hypothesis_testing_methods.md`, never to a section of this document. This document's own
sections are referred to by name where they might be mistaken for one, and section 2 below is
deliberately left unnumbered for that reason.

H2 is defined twice in the repository and the two definitions do not match: four sub-methods in
`hypothesis_testing_methods.md` §2.1–§2.4, five in `modelling_and_hypothesis_testing_documentation.md`
§2, overlapping but with neither containing the other. **This document anchors on
`hypothesis_testing_methods.md`**, matching the H1 whitepaper, and folds the two items unique to
the modelling document into §2.1. The discrepancy should be resolved in the source documents.

| § | Method | Status | Reason |
|---|---|---|---|
| **2.1** | Functional shunting and glomus bypass | **Implemented** | Shunt index near 1 in both cohorts (§7) |
| **2.2** | Spatial haematocrit profiling | **Implemented, overlapping** | Direction as anticipated; ranges intersect (§8) |
| **2.3** | Glomus-specific 3D hypoxic fraction | **Implemented; needs review** | Was inert; after open items 22, 29 non-zero and metabolism-sensitive; grid-converged at 3 µm after open item 30 (§10.3) |
| **2.4** | Oxygen depletion and transit time | **Implemented** | Cohorts separate without overlap (§9) |

All four require the TH-positive glomus mask as a spatial landmark. That mask is the output of a
second two-class Ilastik project over the TH channel of the same acquisitions, described in the
H1 whitepaper §2.3, and its provenance is the H1 §9A analysis rather than anything introduced
here.

---

## 2. The measurement chain

### From two channels to one joined model

Each acquisition is a single `ZCYX` file: channel 0 lectin, channel 1 tyrosine hydroxylase.
Being two channels of one acquisition they are co-registered by construction on an identical
grid, which is what makes every join below sound without a registration step.

| Stage | Output |
|---|---|
| Preprocessing, both channels | `*_ilastik.h5`, `*_TH_ilastik.h5` |
| Two Ilastik projects | vessel and glomus probability maps |
| Region placement | 160³ voxels = 0.0266 mm³, tissue-centred, identical rule for all six |
| Skeletonisation and graph | 4,512 to 8,077 edges per specimen |
| Boundary selection | face-crossing terminals on axis 1 (§5.2) |
| Coupled flow and haematocrit | per-edge flow, discharge haematocrit, viscosity |
| TH join | per-edge tissue fraction, per-cell tissue fraction |
| Perfusion grid | 3 µm ADR solve for §2.3, vessels mapped over their cross-section |

### The two joins

Every H2 method reduces to asking a question about vessels relative to the glomus clusters, so
two primitives carry all four.

**Edges.** `edge_tissue_fraction` gives the fraction of each edge's centreline lying inside the TH
mask. Sampled along the whole polyline rather than at the endpoints, because a capillary
penetrating a cluster typically begins and ends in stroma and an endpoint test would classify
exactly the vessels §2.1 is about as extra-glomus. Weighted by length rather than by vertex: the
stored polylines are not uniformly spaced, and on the test case built for it vertex counting calls
an edge 2% inside where length calls it 90%.

**Grid cells.** `mask_fraction_per_cell` gives the fraction of each perfusion cell occupied by the
mask. Volume fraction rather than a centre sample, because grid cells are larger than the 1.866 µm
voxels: at the earlier 4 µm grid only 0.9% to 4.2% of cells were wholly TH-positive and 21% to 60%
mixed, and a centre sample would decide each of those on one voxel in ten (one in four at 3 µm).

An edge counts as **penetrating** when at least half its length lies inside the TH mask.

---

## 3. Why the pipeline could not answer H2

The assessment recorded these; each is quantified with its remediation in §4.

**3.1 No glomus channel existed.** All four methods require the TH mask as a landmark. The
segmentation was two-class, vessel and background.

**3.2 Boundary conditions were geometric, not anatomical.** Inlets and outlets were whichever
degree-1 nodes fell in a positional band. About 86% of degree-1 nodes are interior
skeletonisation spurs, so most selected inlets were mask defects rather than vessels entering the
region.

**3.3 The viscosity law was a hybrid of the two it chose between.** The base relation was the in
vitro one for glass tubes; the wall-layer correction applied on top belongs to the in vivo law.

**3.4 The perfusion solve had never converged.** Conjugate gradient was preconditioned with an
incomplete-LU factorisation, which is not guaranteed symmetric positive definite.

**3.5 Flow was coupled to tissue in the wrong units**, and each edge's whole flow was recorded
against every cell it crossed.

---

## 4. What changed

Grouped by defect class, each with the measurement that demonstrates it. The full finding-to-commit
map is Appendix C.

### 4.1 The capability that did not exist

The TH channel is preprocessed by `preprocess_th.py` and segmented by a second two-class project.
`edge_tissue_fraction` and `mask_fraction_per_cell` are the joins §2 describes.

### 4.2 Boundary conditions

`select_boundary_terminal_nodes_by_face` admits only terminals within one voxel of a region face.
A vessel supplying the region has to cross one; a dead end in the middle cannot be a pressure
inlet whatever its coordinate. It raises rather than falling back when a face carries no
terminals, where the band rule silently dropped to the extreme 10% of all nodes.

Measured as the spread of the shunt ratio while each rule's own free parameters move:

| Rule | Parameters varied | Ratio spread |
|---|---|---|
| band, axis 1 | width 10/25/40% | 75.8% |
| **face, axis 1** | tolerance 1/2/4 voxels | **13.3%** |

A 5.7-fold reduction, below the ~26% operative floor the assessment had established. Axis 1 is a
selection rather than a preference: it is the only axis with terminals on both faces in all six
specimens.

One measurement pointed the wrong way and is recorded rather than dropped. Varying only the axis,
holding each rule's second parameter at its default, gives 28.4% for the band rule against 31.5%
for the face rule, which reads as the face rule being worse and was briefly believed. That
comparison fixes the parameter that damages the band rule.

### 4.3 Estimators that were not what they were named

The Pries–Secomb viscosity function combined the in vitro base with the in vivo wall correction.
Both laws are now available by name and the correction follows the law. A second error was found
by checking the first correction against the published relation rather than by the tests, which
passed either way: the in vivo wall factor appears **twice**, and applying it once understates
apparent viscosity by 1.26× at 8 µm and 2.2× at 3 µm.

At the study's median calibre the corrected law gives 3.6 to 3.9 times the previous viscosity.
**Every §2.1 and §2.2 ratio moved by at most 0.02.**

### 4.4 A solve that never converged

CG assumes an SPD preconditioner; `spilu` guarantees neither symmetry nor definiteness. Measured
on WKY-C at the production grid with real solved flows:

| Preconditioner | Converged | Relative residual | Time |
|---|---|---|---|
| ILU, as shipped | no | **19.06** | 5.81 s |
| Jacobi, the inverse diagonal | yes | **8.8e-7** | **0.05 s** |

The initial residual is 1 by construction, so 19 is divergence. The whole steady-state solve
became 39 times faster and converged.

### 4.5 Two conservation faults

**Units.** Flow leaves the solve in mmHg·µm³/cP, not µm³/s, while the metabolic sink is in
mmol/L/s times µm³. The sink exceeded the source by 2.24e4. The conversion,
`PASCALS_PER_MMHG × 1e3`, is derived from unit definitions and checked against an independent SI
computation of the same tube to 1e-9 relative.

**Sharing.** Each edge's whole flow was recorded against every cell it crossed, so the total
source was exactly proportional to the mean cells crossed per edge:

| Resolution | Mean cells per edge | Total source | Source per crossing |
|---|---|---|---|
| 10 µm | 2.73 | 8.87e6 | 3.25e6 |
| 4 µm | 4.74 | 1.54e7 | 3.25e6 |
| 3 µm | 5.58 | 1.80e7 | 3.23e6 |

Shared by length, the total is grid-independent to the digit. The solution then converged (median
PO2 27.34, 27.92, 28.21 at 10, 6 and 4 µm); after open items 22 and 29 it ran 91.38, 90.46, 89.52,
89.19, 87.90 down to 2 µm and did not converge, until vessels were mapped over their cross-section
(§10.3).

### 4.6 Claims corrected by measurement

Recorded because a remediation record that reports only successful fixes is advocacy rather than
evidence.

- A claim that the face rule reduced boundary sensitivity was first measured the wrong way, by
  holding each rule's second parameter fixed, and appeared to show the opposite.
- A claim that the boundary pressures implied a capillary velocity ten times too high was
  **withdrawn**: it came from a single straight tube, and the network appeared to run 20 to 100
  times too *slow*. That network figure was itself wrong: it came from the inflated resistances of
  open item 12. Re-derived, the network runs at 979 to 3,319 µm/s, 1 to 3 times *above* the
  physiological range (§11.1). The withdrawn single-tube figure (8,900 µm/s) was too high, but in
  the right direction.
- The assessment reported the Picard loop converging with no warnings. That was the outer loop;
  the inner CG had been failing at every step under a differently worded message.

---

## 5. Experimental design

### 5.1 One classifier per channel

One vessel project and one glomus project, each shared across all six volumes. Per-cohort
classifiers would confound specimen identity with classifier identity unfixably. The registry
refuses a run whose specimens do not share one project, per channel.

### 5.2 One boundary rule, one axis

Face-crossing terminals on axis 1, tolerance one voxel, for all six. Axis 1 because it is the only
axis with terminals on both faces in every specimen. Inlet 60 mmHg, outlet 20 mmHg.

### 5.3 Matched, tissue-centred sub-volumes

160³ voxels, 0.0266 mm³, placed by the same rule as H1: z from each volume's axial tissue peak, y
and x from the grayscale centroid. Centring on signal samples mid-organ, where the network is
denser than at the periphery, so absolute densities over-estimate the organ while the comparison
stays like-for-like.

---

## 6. Instrument validation

### 6.1 Do the reported measures survive the corrections that moved everything else?

The strongest available check, because it was not designed as one. Three separate corrections
each moved an absolute quantity by a large factor:

| Correction | Absolute effect | §2.1 shunt index | §2.2 haematocrit ratio |
|---|---|---|---|
| Viscosity law (§4.3) | 3.6 to 3.9× viscosity | 0.924 → 0.909 | 0.90 → 0.92 |
| Preconditioner (§4.4) | solve 39× faster, residual 19 → 8.8e-7 | unchanged | unchanged |
| Flow units and sharing (§4.5) | source ×1.3e5, then grid-independent | unchanged | unchanged |
| Rheology resistance fix (open item 12) | velocity ×240 to ×343 | 0.904 / 1.002 → 0.894 / 0.987 (WKY / SHR) | 1.025 / 0.953 → 1.006 / 0.934 |

Under the first three, every ratio moved by at most 0.02 while the quantities beneath them moved by
factors of three to five orders. That is the behaviour a within-specimen ratio is supposed to have,
demonstrated rather than assumed.

The fourth is different in kind. The bug inflated resistance by a factor that grew with diameter
(208× at 4 µm, 539× at 20 µm), so it redistributed flow rather than scaling it, and a ratio does not
cancel that. Per specimen the ratios moved by up to 0.09 (WKY-C haematocrit 1.023 → 0.930, SHR-C
shunt index 1.044 → 1.124); cohort means by at most 0.04; the SHR/WKY ratios by at most 0.04
(transit 0.68 → 0.72). Which measures separate the cohorts and which overlap did not change.

### 6.2 Does the boundary choice produce the difference?

The residual spread of the shunt ratio under the face rule, as its tolerance moves over 1, 2 and
4 voxels, is 13.3%. The measured between-group differences are 10% (shunt index), 22% (flow
ratio) and 28% (transit ratio). The two smaller of those are not comfortably clear of the floor
and are reported with that stated.

### 6.3 What is not validated

There is no labelled ground truth for either channel, so no per-cohort segmentation accuracy
score exists. The instrument-fairness argument rests on internal consistency and on the
single-classifier design, exactly as in H1, and that demonstration remains outstanding.

---

## 7. Results: §2.1 functional shunting

### 7.1 Per-specimen values

| Specimen | Group | Penetrating edges | Edge share | Flow share | **Shunt index** | Median flow ratio |
|---|---|---|---|---|---|---|
| WKY-A | WKY | 1,178 | 26.1% | 25.7% | 0.983 | 0.790 |
| WKY-B | WKY | 1,624 | 41.3% | 33.6% | 0.813 | 0.894 |
| WKY-C | WKY | 1,659 | 24.8% | 22.0% | 0.887 | 0.960 |
| SHR-A | SHR | 1,655 | 24.3% | 21.7% | 0.892 | 1.159 |
| SHR-B | SHR | 1,182 | 14.6% | 13.8% | 0.945 | 1.029 |
| SHR-C | SHR | 473 | 9.7% | 10.9% | 1.124 | 1.045 |

### 7.2 The shunt index, and why flow share alone is not it

**Shunt index = flow share ÷ edge share.** A value of 1 means flow is indifferent to the clusters;
below 1 means the bypassing vessels carry disproportionately more, which is the shunting the
method proposes to detect.

| | WKY | SHR | Ratio |
|---|---|---|---|
| Shunt index | 0.894 | 0.987 | 1.10 |

**Both cohorts sit near 1, so there is no evidence of functional shunting in either.** WKY runs
0.81 to 0.98 and SHR 0.89 to 1.12; the ranges overlap. The lowest value, WKY-B at 0.81, means the
penetrating edges there carry about a fifth less than their share, which is modest and lies in the
normotensive cohort, the opposite of what §2.1 predicts.

Flow share alone would have said something else. It runs 27.1% in WKY against 15.5% in SHR, a
0.57 ratio that reads as dramatic diversion in the hypertensive network. But it tracks the edge
share almost exactly, and the edge share is itself downstream of the parenchymal volume
difference H1 §1.3 reports at 0.60. Dividing it out removes an apparent effect that was never
about flow.

### 7.3 Median flow ratio

Flow through penetrating edges over flow through bypassing edges, per specimen: **WKY 0.882
against SHR 1.077, and the cohorts do not overlap** (WKY 0.790 to 0.960, SHR 1.029 to 1.159). For
three against three that is the most extreme arrangement available, giving the design floor of
p = 0.10.

Penetrating capillaries carry about 12% less flow than bypassing ones in WKY and about 8% more in
SHR. The effect is small and the separation is clean; both are reported. The gap between the
groups is narrow: WKY-C at 0.960 and SHR-B at 1.029.

---

## 8. Results: §2.2 spatial haematocrit profiling

Discharge haematocrit is solved by iterating flow against the Pries–Secomb phase-separation model.
The numbers below come from the pre-fix loop, which stopped on its 15-pass cap without converging;
see the open item 37 note at the top.

| Specimen | Hct penetrating | Hct bypassing | Ratio |
|---|---|---|---|
| WKY-A | 0.3916 | 0.3708 | 1.056 |
| WKY-B | 0.3841 | 0.3724 | 1.031 |
| WKY-C | 0.3664 | 0.3939 | 0.930 |
| SHR-A | 0.3653 | 0.3602 | 1.014 |
| SHR-B | 0.3903 | 0.3862 | 1.011 |
| SHR-C | 0.3194 | 0.4109 | 0.777 |

| | WKY | SHR | Ratio |
|---|---|---|---|
| Haematocrit ratio | 1.006 | 0.934 | 0.93 |

**Directionally what the method anticipates, and not supported.** §2.2 proposes RBC starvation in
the glomus microenvironment of the hypertensive network: a dense capillary bed carrying mostly
plasma. SHR does show a lower ratio. But the ranges overlap heavily (WKY 0.930 to 1.056, SHR 0.777
to 1.014) and the group mean rests almost entirely on SHR-C at 0.777. The other two SHR specimens,
at 1.014 and 1.011, sit just above the WKY mean of 1.006, and one WKY specimen (WKY-C, 0.930) now
shows the lower glomus haematocrit the method predicts for SHR.

### 8.1 An open question about the skimming model

Correcting the viscosity law surfaced a property of the phase-separation implementation that is
not settled. On a test bifurcation the in vitro law sends 84% of flow down the wide branch, which
is then also the faster, and it skims red cells as expected. Under the in vivo law the narrow
branch is penalised harder, the split evens to 64/36, and 36% of flow through a quarter of the
area makes the **narrow** branch the faster one; the model then concentrates red cells there,
inverting the classic picture.

The call site pairs each flow with its own diameter correctly, so this is the model keying on
velocity where the Pries phase-separation law is normally posed in fractional blood flow with a
diameter-dependent threshold. At a near-even split the two parameterisations can disagree in
direction. This bears directly on §2.2, whose entire subject is where red cells end up, and it is
recorded rather than fixed.

---

## 9. Results: §2.4 oxygen depletion and transit time

Transit time per edge is lumen volume over flow, accumulated along the **solved flow directions**
rather than along adjacency, since an edge carrying blood away from a node cannot deliver blood
to it. Reported as the ratio of transit time to penetrating edges against bypassing edges.

| Specimen | Penetrating | Bypassing | Ratio |
|---|---|---|---|
| WKY-A | 9.571e4 | 7.412e4 | 1.291 |
| WKY-B | 6.299e4 | 5.843e4 | 1.078 |
| WKY-C | 1.206e5 | 1.076e5 | 1.120 |
| SHR-A | 5.011e4 | 5.962e4 | 0.841 |
| SHR-B | 5.853e4 | 6.586e4 | 0.889 |
| SHR-C | 6.222e4 | 8.077e4 | 0.770 |

| | WKY | SHR | Ratio |
|---|---|---|---|
| Transit ratio | 1.163 | 0.833 | **0.72** |

**The cohorts separate without overlap**, WKY 1.078 to 1.291 against SHR 0.770 to 0.889, giving
p = 0.10, the design floor. This is the largest separation of the four measures.

**The direction is the opposite of what §2.4 anticipates.** It expects sluggish transit to the
sensors in the hypertensive network, producing stagnant hypoxia at the sensor site. Blood reaches
the SHR clusters in about five sixths of the time it takes to reach the surrounding tissue, where
in WKY it takes about a sixth longer.

Whether that means the sensors are well perfused, or merely that a smaller cluster sits closer to
its supply, is not answerable from a ratio and is not claimed. Absolute transit times are in
arbitrary units and are reported only to show the ratio's construction.

---

## 10. Results: §2.3 glomus-specific hypoxic fraction

> ⚠ **Needs review after open items 22 and 29 (re-run 2026-09-26).** Tier 1 left O₂ solubility out of its diffusion (≈750× too strong) and washed blood out at systemic rather than local haematocrit (`cb_modelling_reference.md` open items 22, 29). With both fixed, §2.3 is no longer zero or inert. The interpretation in §10.2 and the conclusions that cite it (summary, §11.1, P10) were written for the old field and have **not** been rewritten; each is marked where it no longer holds. Old outputs: `examples/outputs/cb_h2_hypoxic_fraction_2026-09-26_pre_item22.json`.
>
> **Re-run again for open item 30 (2026-09-26).** The numbers below are at 3 µm with vessels mapped over their cross-section (§10.3), not at 4 µm with vessels mapped by centreline. Outputs before this change: `..._2026-09-26_pre_item30.json`, `..._padded_2026-09-26_pre_item30.json`, `cb_h2_paraview_2026-09-26_pre_item30/`.

### 10.1 What the model returns

Perfusion grid at 3 µm, vessels mapped over their cross-section, metabolic rate assigned per cell
from the TH fraction with the volume-weighted mean held constant so contrasts are comparable.
Uniform metabolism (contrast 1); hypoxic fractions are within the TH volume. Every Tier 1 solve
converged (Newton, residual < 10⁻⁵).

| Specimen | TH volume | PO2 within TH | PO2 in stroma | Hypoxic < 5 | < 10 | < 20 mmHg |
|---|---|---|---|---|---|---|
| WKY-A | 16.9% | 80.97 | 71.10 | 0.01% | 0.59% | 2.99% |
| WKY-B | 28.7% | 82.74 | 79.54 | 0% | 0.03% | 1.54% |
| WKY-C | 19.6% | 88.32 | 80.75 | 0% | 0% | 0% |
| SHR-A | 16.8% | 86.34 | 84.00 | 0% | 0% | 2.31% |
| SHR-B | 14.2% | 89.84 | 87.59 | 0% | 0.06% | 0.81% |
| SHR-C | 7.8% | 75.06 | 69.54 | 0.94% | 3.09% | 6.56% |

"TH volume" is the TH share of the grid's volume, so it moves slightly with the grid (17.7% on
WKY-A at 4 µm).

**Small hypoxic fractions**: below 10 mmHg in four of six specimens, below 20 mmHg in five. PO2
within the glomus clusters is 84.0 mmHg in WKY against 83.8 in SHR, a ratio of 1.00 whose ranges
overlap (81.0–88.3 against 75.1–89.8). Mean TH hypoxic fraction below 10 mmHg is 0.21% in WKY and
1.05% in SHR, and below 20 mmHg 1.51% and 3.23%; with n = 3 per group the ranges overlap at every
threshold. SHR-C carries most of the SHR hypoxia on its own.

Before open item 30 (4 µm, centreline mapping) the same table read 70.0–86.2 mmHg in TH, WKY/SHR
78.1/79.6, and 0–3.95% below 10 mmHg: the centreline mapping under-delivered oxygen, and PO2 in
TH is 3.6–7.9 mmHg higher now. Before items 22 and 29 it read 28–41 mmHg in TH and 0% at every threshold.

### 10.2 Why this is not a result

> ⚠ **Needs review (items 22, 29): the argument of this section was written for the old field
> and no longer matches the re-run.** Raising the glomus metabolic rate from one to four times the
> stromal rate now moves PO2 within TH from 80.97 to 78.95 mmHg on WKY-A (−2.0 mmHg; SHR-C 75.06
> to 71.03, −4.0) and TH hypoxia below 10 mmHg from 0.59% to 1.63% (SHR-C 3.09% to 4.97%), at 3 µm
> with the cross-section mapping (open item 30). The old figure was
> 32.625 to 32.610. The diffusion-length estimate below always included α; the old solver did
> not, so its effective diffusion length was ≈27× longer than the one quoted. The text is kept
> as written until reviewed.

**The glomus-specific mechanism is inert.** Raising the glomus metabolic rate from one to four
times the stromal rate moves PO2 within TH from 32.625 to 32.610 on WKY-A: fifteen thousandths of
a millimetre of mercury for a fourfold change in the parameter the method is built around.

The reason is a property of the tissue. The oxygen diffusion length is

    sqrt(D · alpha · PO2 / M) = 20 µm at PO2 10, 35 µm at 30, 45 µm at 50

against a **median tissue-to-vessel distance of 5.3 to 7.9 µm** (H1 §1.5). Every tissue point sits
at roughly a fifth of its supply radius, so the tissue is not diffusion-limited and a local sink
cannot produce a local gradient. The consumption rate is not at fault: `M_max = 0.05` mmol/L/s is
0.067 mL O2 per mL per minute against roughly 0.040 for brain.

**§2.3 asks for a glomus-specific hypoxic fraction in a bed too densely vascularised to have
one.** That is a statement about the carotid body, not about the implementation, and it is the
most substantive negative result in this document.

**Two specimens are additionally solved on less tissue than they contain.** The perfusion grid
takes its extent from the vascular bounding box, so where vessels stop short of the region edge
the glomus tissue beyond them is not represented: 3.74% of SHR-A's glomus volume and 7.54% of
SHR-C's at 3 µm (4.35% and 7.54% at 4 µm). §2.1, §2.2 and §2.4 are unaffected, being computed against the mask in voxel space
rather than on the grid. For §2.3 it compounds a result already reported as not usable, and it is
recorded as S28.

`--pad-grid` extends the grid to the segmented volume and recovers that tissue. At 3 µm with the
cross-section mapping it moves mean PO2 within TH by −3.54 and −4.70 mmHg on the two specimens
(contrast 2; −3.93 and −4.88 at 4 µm by centreline, −0.77 and −0.66 before items 22 and 29) and
raises TH hypoxia below 10 mmHg from 0.21% to 2.50% (SHR-A) and from 3.67% to 4.70% (SHR-C); it was
zero both ways before items 22 and 29. **Needs review:** the old reading, that the diffusion length
exceeds the unvascularised rim so the recovered cells are supplied by their neighbours, no longer
holds, and S29's choice of the unpadded grid should be revisited. The results in this document are
unpadded (S29). Padded outputs: `examples/outputs/cb_h2_hypoxic_fraction_padded.json`.

### 10.3 Grid convergence (open item 30)

**§2.3 runs at 3 µm with vessels mapped over their cross-section, and is converged there to
0.5 mmHg.** Contrast 1, unpadded, cross-section mapping:

| Grid | WKY-C median PO2 | WKY-C PO2 in TH | SHR-C median PO2 | SHR-C PO2 in TH | SHR-C TH < 10 mmHg |
|---|---|---|---|---|---|
| 10 µm | 92.99 | 88.23 | 84.41 | 74.41 | 3.10% |
| 6 µm | 92.79 | 87.88 | 85.51 | 74.73 | 3.15% |
| 4 µm | 92.69 | 87.78 | 85.39 | 74.95 | 3.06% |
| 3 µm | 92.87 | 88.32 | 86.17 | 75.06 | 3.09% |
| 2 µm | 92.78 | 88.11 | 86.08 | 75.09 | 3.26% |

"PO2 in TH" is the TH-weighted mean. The criterion was that both PO2 measures move less than
0.5 mmHg per step on both specimens. From 3 to 2 µm the largest step is 0.21 mmHg; from 4 to 3 µm
two exceed it (SHR-C median +0.78, WKY-C in TH +0.54), so 3 µm is the coarsest grid that passes.
On all six, the 4 → 3 µm step is +0.09 to +0.79 mmHg in median PO2 and −0.34 to +0.54 in TH.
Across the sweep the steps change sign with no trend.

**Why the grid had to be revisited.** Until open item 30 each vessel was placed only in the cells
its centreline crosses, whatever its width, so a finer grid drew a thinner vessel; in the limit it
is a line source, and the field around a line goes as ln r. Mapped that way, WKY-C median PO2 ran
91.38, 90.46, 89.52, 89.19, 87.90 at 10, 6, 4, 3, 2 µm (SHR-C 80.28, 79.95, 78.35, 78.12, 75.85):
about 1.5 and 1.9 mmHg lost per halving and no limit. On a single straight 9 µm vessel the
centreline mapping falls 17.0, 11.5, 8.6, 6.8 mmHg at 9, 3, 1, ⅓ µm, while the cross-section mapping
gives 17.0, 23.4, 23.8, 23.2 (`tests/test_perfusion_tier1_grid_refinement.py`). Median calibre here
is 7.5–8.4 µm, so at 4 µm most vessels were drawn thinner than they are, and the tissue read as
less oxygenated than the model's own geometry implies: by 3.6–7.4 mmHg in TH at 4 µm, and unequally
across specimens, so the error did not cancel between cohorts.

The cross-section mapping keeps the full tissue metabolic rate in cells that hold lumen, a small
overcount of consumption (reference §11 row 24).

Before items 22 and 29 the centreline sequence was 27.34, 27.92, 28.21, halving and extrapolating
to about 28.5; before the sharing fix of §4.5 it ran 42.0, 46.9, 50.5. Outputs:
`examples/outputs/cb_h2_hypoxic_fraction_xsec_sweep.json` (cross-section, WKY-C and SHR-C, 10–2 µm),
`..._xsec_grid4.json` (cross-section, all six, 4 µm), and for the centreline mapping
`..._grid10.json`, `..._grid6.json`, `..._grid3.json` (all six), `..._grid2.json` (WKY-C, SHR-C).

---

## 11. Limitations

### 11.1 Limitations that bound the claims

**Absolute perfusion is not calibrated, and at 60/20 mmHg it runs fast.** Flow-weighted velocity
is 979 to 3,319 µm/s across the six against a physiological 200 to 1,000; five of six sit above
the upper end. A 500 µm/s velocity would need a drop of 6 to 20 mmHg, not 40. The face boundary
rule carries 4 to 8 times less flow than the band rule it replaced, a cost of §4.2 that its own
validation did not measure. So the absolute scale depends on a pressure pair that is not measured
in the carotid body and on a boundary choice with no anatomical calibration, on top of the ±45%
calibre floor. **No absolute perfusion quantity in this document is defensible.** Every reported
measure is a ratio for this reason.

Until open item 12 was re-derived (2026-09-27) this paragraph said the opposite: 4 to 10 µm/s, 20
to 100 times too slow, needing about 3,257 mmHg. Those figures came from the inflated resistances
of the rheology bug. `examples/cb_h2_absolute_perfusion.py` reproduces both the old table (with
the old code) and the new one.

**Statistical power.** n = 3 per group. The exact two-sided permutation p cannot fall below 0.10
for any arrangement of three against three. No claim of statistical significance is made.

**Two of four measures overlap.** The shunt index and the haematocrit ratio both have intersecting
ranges, and the haematocrit group mean rests on one specimen of three.

**§2.3 is not supported by its own premise** (§10.2), independently of anything measured here.
*Needs review after open items 22 and 29: that premise was measured on a solver with 750× too much
diffusion; see the note at the head of §10.*

**The skimming model's parameterisation is unsettled** (§8.1), and it bears directly on §2.2.

**No per-cohort accuracy validation exists** for either segmentation channel.

### 11.2 Limitations that bound the precision

**Boundary sensitivity.** The residual spread of a ratio under the face rule is 13.3%, against
measured differences of 10%, 22% and 28%.

**Calibre quantisation.** Inherited from H1 §1.2: the distance transform returns a coarse
diameter distribution, and resistance goes as the inverse fourth power of diameter.

**Region sampling.** 0.0266 mm³ per specimen, roughly a fortieth of a cubic millimetre, centred on
tissue signal. Absolute densities over-estimate the organ; the comparison is like-for-like.

**The TH classifier retains a 2.1× cohort skew** in its positive class, the residual bound
recorded in the H1 whitepaper §2.3.

---

## 12. Claim ledger

**Established** (evidenced and robust to the known limitations), **Provisional** (evidenced but
sensitive to a stated limitation), **Not supported** (measured and disqualified, or unmeasurable).

| # | Claim | Evidence | Rests on | Grade |
|---|---|---|---|---|
| P1 | All four H2 methods are implemented and run on all six specimens | §7–§10 | none | **Established** |
| P2 | The reported ratios are insensitive to uniform corrections that moved absolute quantities by three to five orders; a diameter-dependent one (open item 12) moved them by up to 0.09 per specimen and left every cohort pattern intact | §6.1 | Four independent corrections | **Established** |
| P3 | There is no functional shunting in either cohort | §7.2 | Shunt index 0.89 and 0.99 | **Established** |
| P4 | Flow share alone would have reported shunting that is an artefact of edge share | §7.2 | 0.57 ratio removed by normalisation | **Established** |
| P5 | Transit time to the glomus clusters is shorter in SHR relative to surrounding tissue | §9 | No overlap; p = 0.10 floor | **Provisional** |
| P6 | Penetrating capillaries carry relatively more flow in SHR | §7.3 | No overlap; effect is 12% against 8% | **Provisional** |
| P7 | The direction of P5 opposes the stagnant-hypoxia prediction §2.4 makes | §9 | as P5 | **Provisional** |
| P8 | Haematocrit in glomus-penetrating vessels is lower in SHR | §8 | Ranges overlap; rests on SHR-C | **Not supported** |
| P9 | The shunt index differs between cohorts | §7.2 | Ranges overlap; 10% against a 13.3% floor | **Not supported** |
| P10 | A glomus-specific hypoxic fraction is measurable on this geometry | §10.2 | Was: diffusion length 20–45 µm vs TVD 5–8 µm. Re-run after open items 22, 29, 30: 0–3% below 10 mmHg within TH, moves with metabolism; grid-converged at 3 µm | **Needs review** (was Not supported) |
| P11 | Any absolute perfusion quantity reported here is physiological | §11.1 | Velocity 1 to 3× above the physiological range at 60/20 mmHg; throughput 4 to 8× with the boundary rule (was: 20 to 100× low, before open item 12) | **Not supported** |

The defensible position is P1–P4 (Established) plus P5–P7 (Provisional). Nothing else should be
presented as a result.

---

## 13. Future work

| Priority | Work | Unblocks | Cost |
|---|---|---|---|
| 1 | Calibrate absolute perfusion: a measured or defensible pressure drop across the region (6 to 20 mmHg gives 500 µm/s) and a boundary rule whose throughput is anatomically grounded | Every absolute quantity; §2.3 | Investigation |
| 2 | Settle the skimming model's parameterisation, fractional flow against velocity | §2.2 | Literature plus a re-run |
| 3 | Level the TH classifier's residual 2.1× cohort skew | The stated bound on both whitepapers | Hours of labelling |
| 4 | Complete perivascular boundary labelling on the vessel channel | Calibre precision, inherited by every resistance | Hours; H1 §13 item 1 |
| 5 | Hand-labelled held-out regions in both cohorts, both channels | Per-cohort validation scores | Hours |
| 6 | More specimens, or acceptance that H2 is answered descriptively | Statistical power | Experimental |

Items 1 and 3 are the only ones on the critical path to a defensible absolute H2 result. Items 2
and 3 bound the two measures that currently overlap.

---

## Appendix A: Frozen parameter set

| Parameter | Value | Source |
|---|---|---|
| Region size | 160³ voxels, 0.0266 mm³ | H1 §5.3 |
| Voxel size | 1.8639 × 1.866 × 1.866 µm | acquisition metadata |
| Vessel probability threshold | 0.90 | H1 threshold selection |
| TH probability threshold | 0.50 | not frozen; see H1 §9A.4 |
| Boundary rule | face-crossing, axis 1, tolerance 1 voxel | §4.2 |
| Inlet / outlet pressure | 60 / 20 mmHg | §5.2, and §11.1 |
| Systemic haematocrit | 0.45 | conventional |
| Viscosity law | Pries–Secomb **in vivo** | §4.3 |
| Plasma viscosity | 1.2 cP | conventional |
| Perfusion grid | 3 µm, cross-section vessel mapping (converged to 0.5 mmHg) | §10.3 |
| Oxygen diffusivity | 1.5e-9 m²/s | conventional |
| M_max | 0.05 mmol/L/s | §10.2 |
| Penetration cutoff | 0.5 of edge length inside TH | the two joins, §2 |

## Appendix B: Reproduction

```
python examples/cb_h2_boundary_selection.py          # §4.2
python examples/cb_h2_glomus_perfusion.py            # §7, §8, §9
python examples/cb_h2_absolute_perfusion.py          # §11.1 absolute perfusion
python examples/cb_h2_hypoxic_fraction.py            # §10
python examples/cb_h2_error_propagation.py           # the noise floor
python examples/cb_h2_vtk.py                         # ParaView artefacts, not a result
```

`examples/cb_h2_paraview_guide.md` covers the exports and which arrays carry which section.

| Artefact | Supplies |
|---|---|
| `cb_h2_glomus_perfusion.json` | §7, §8, §9 |
| `cb_h2_absolute_perfusion.json` | §11.1 |
| `cb_h2_hypoxic_fraction.json` | §10 |
| `cb_h2_paraview/export_summary.json` | the frame check behind the ParaView exports |
| `<SPECIMEN>/per_edge_morphometry.csv` | diameters; the cached graph carries none |
| `ilastik_probabilities/*_TH_ilastik_Probabilities.h5.provenance.json` | TH classifier attribution |

## Appendix C: Finding to remediation map

| Finding | Remediation | Commits |
|---|---|---|
| **3.1** No glomus channel | TH preprocessing, classifier, and the two joins | `244cef5`, `886015e`, `0486f1f` |
| **3.2** Geometric boundaries | Face-crossing selection, S21 | `d32fc85` |
| **3.3** Hybrid viscosity law | Both laws by name, wall factor applied twice, S22 | `bfed0da` |
| **3.4** Solve never converged | Jacobi preconditioner, S24 | `6bba306` |
| **3.5** Wrong units and repeated source | Unit conversion S25, length sharing S26 | `535bcb1`, `25ab93f` |
| Silent fallbacks | Diameter provenance, 5 µm default, dropped edges | `7e273aa` |
| Transit time and axis naming | S23 | `891ea52` |
| Withdrawal of the velocity claim | S27 | `e98d59d` |
| Rheology resistance inflated 208–539× (open item 12) | Poiseuille recompute, then rescale once; flows re-derived, S30 | `7ea1b36`, `7f05a78` |

The assessment behind these is `h2_pipeline_capability_assessment.md`, findings S1 to S30.
