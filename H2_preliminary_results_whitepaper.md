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
| §2.2 Haematocrit, penetrating over bypassing | 0.991 | 1.067 | **1.08** | **No** |
| §2.1 Shunt index | 0.980 | 0.853 | 0.87 | Yes |
| §2.1 Median flow, penetrating over bypassing | 0.963 | 0.852 | 0.89 | Yes |
| §2.4 Transit time to glomus clusters | 0.992 | 1.018 | 1.03 | Yes |

> **Re-derived 2026-09-30 (open items 37, 38 and 40).** Every table in this document now comes
> from the frozen analysis: vessel threshold 0.95, tissue-placed regions (the 2026-09-28 batch),
> a flow–haematocrit loop that converges in all six specimens (173–457 passes, `cfee721`; it used
> to stop on a 15-pass cap), the exact per-cell TH fraction (`870de15`) and TH lookups centred on
> the voxel (`5839aae`). Until this re-run the tables were on the tissue-centred boxes at 0.90
> from 2026-09-27, and the conclusions have been rewritten to match. Two of the old results do not
> survive: transit time and median flow ratio no longer separate the cohorts, and the haematocrit
> ratio, which used to overlap and point the anticipated way, now separates **in the opposite
> direction**. §6.1 records what moved and why. Earlier outputs are kept as
> `examples/outputs/cb_h2_*_2026-09-30_pre_itemJ.json` (placed boxes, unconverged loop) and
> `..._2026-09-28_pre_item27.json` (centred boxes).

**The one separation runs against the method's prediction.** §2.2 proposes that the vessels
supplying the glomus clusters of the hypertensive network carry fewer red cells. In SHR they carry
more: penetrating over bypassing haematocrit is 1.061 to 1.072 in SHR against 0.972 to 1.026 in
WKY, a 7.7% difference against the 5.9% within-specimen floor. It rests on a skimming model whose
parameterisation is not settled (§8.1).

**There is no cohort-level functional shunting.** The shunt index, flow share divided by edge
share, sits at 0.96 to 0.99 in WKY: flow is indifferent to the clusters. SHR spreads from 0.47 to
1.23. One specimen, SHR-C, does shunt (its penetrating edges carry half their share of flow), and
one, SHR-A, does the opposite, so the cohort means (0.98 against 0.85) overlap and no group claim
is made.

**Three limitations bound what may be concluded.** The groups overlap on three of the four
measures, and with n = 3 per group the exact two-sided permutation p cannot fall below 0.10.
§2.3's glomus-specific hypoxic fraction is zero in every specimen at every metabolic contrast
(§10): PO2 within the clusters is 88.5 to 95.2 mmHg, and although it is lower in every SHR
specimen than in every WKY one, the difference (3 to 6%) is an absolute PO2 that sits under the
calibre floor for absolute flow. And no absolute perfusion quantity is defensible (§11.1): at
60/20 mmHg the network runs at 1 to 1.8 times the upper physiological velocity, and its throughput
moves 2 to 4 times with the boundary rule.

**Verdict.** §2.1, §2.2 and §2.4 are implemented, posed as within-specimen ratios, and report; of
the three only the §2.2 ratio separates the cohorts, and it separates the wrong way. §2.3 is
implemented, runs and is grid-converged, but returns zero on this geometry. Every measure that
survives is a ratio, because four independent corrections (calibre, viscosity, perfusion units
and the rheology resistance fix) each moved absolute quantities by large factors. The ratios are
not immune, though: a correction that changes how flow divides at junctions moves them (§6.1), and
the converged loop of open item 37 is such a change.

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
| **2.1** | Functional shunting and glomus bypass | **Implemented, overlapping** | Shunt index near 1 in WKY; SHR spreads 0.47 to 1.23 (§7) |
| **2.2** | Spatial haematocrit profiling | **Implemented** | Cohorts separate without overlap, opposite to the anticipated direction (§8) |
| **2.3** | Glomus-specific 3D hypoxic fraction | **Implemented, returns zero** | 0% of glomus volume below 5, 10 or 20 mmHg in every specimen and contrast; grid-converged at 3 µm (§10) |
| **2.4** | Oxygen depletion and transit time | **Implemented, overlapping** | Ratios near 1 in both cohorts (§9) |

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
| Skeletonisation and graph | 6,140 to 8,281 edges per specimen |
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
| band, axis 1 | width 10/25/40% | 73.9% |
| **face, axis 1** | tolerance 1/2/4 voxels | **8.9%** |

An 8.3-fold reduction (13.3% against 75.8% on the earlier tissue-centred boxes). Axis 1 was
chosen on those boxes as the only axis with terminals on both faces in all six specimens; on the
placed boxes all three axes qualify, and axis 1 is kept as the frozen choice. Which axis is used is
itself a lever: moving it spreads the ratio by 33 to 34% under every rule, more than any rule's
own parameters.

One measurement pointed the wrong way and is recorded rather than dropped. On the centred boxes, varying only the axis,
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
89.19, 87.90 down to 2 µm (centred boxes) and did not converge, until vessels were mapped over their cross-section
(§10.3).

### 4.6 Claims corrected by measurement

Recorded because a remediation record that reports only successful fixes is advocacy rather than
evidence.

- A claim that the face rule reduced boundary sensitivity was first measured the wrong way, by
  holding each rule's second parameter fixed, and appeared to show the opposite.
- A claim that the boundary pressures implied a capillary velocity ten times too high was
  **withdrawn**: it came from a single straight tube, and the network appeared to run 20 to 100
  times too *slow*. That network figure was itself wrong: it came from the inflated resistances of
  open item 12. Re-derived, the network runs at 955 to 1,808 µm/s, up to 1.8 times *above* the
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

Face-crossing terminals on axis 1, tolerance one voxel, for all six (`cb_settings.BOUNDARY_AXIS`).
Axis 1 was chosen when it was the only axis with terminals on both faces in every specimen; on the
placed boxes all three are, and the choice is held fixed. Inlet 60 mmHg, outlet 20 mmHg.

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
| *Change of sample, not a correction:* placed regions at 0.95 (2026-09-28) | different vessels | 0.894 / 0.987 → 0.988 / 0.878 | 1.006 / 0.934 → 1.010 / 1.030 |
| Converged flow–haematocrit loop and centred TH lookup (open items 37, 40) | 15-pass cap → 173–457 passes | 0.988 / 0.878 → 0.980 / 0.853 | 1.010 / 1.030 → 0.991 / 1.067 |

Under the first three, every ratio moved by at most 0.02 while the quantities beneath them moved by
factors of three to five orders. That is the behaviour a within-specimen ratio is supposed to have,
demonstrated rather than assumed.

The fourth is different in kind. The bug inflated resistance by a factor that grew with diameter
(208× at 4 µm, 539× at 20 µm), so it redistributed flow rather than scaling it, and a ratio does not
cancel that. Per specimen the ratios moved by up to 0.09 (WKY-C haematocrit 1.023 → 0.930, SHR-C
shunt index 1.044 → 1.124); cohort means by at most 0.04; the SHR/WKY ratios by at most 0.04
(transit 0.68 → 0.72). Which measures separate the cohorts and which overlap did not change.

The fifth row is not a correction. Moving each region onto tissue and cutting at the frozen 0.95
measures different vessels, so the ratios are entitled to move, and they did: transit went from
separated (SHR/WKY 0.72) to overlapping (1.03), median flow ratio from separated (1.22) to
overlapping (0.91), and the haematocrit ratio from 0.93 to 1.02. The earlier separations belonged
to the centred boxes, which sat 27 to 127 µm off the tissue centroid.

The sixth is a correction of the fourth kind. The old loop stopped on its 15-pass cap with
per-edge haematocrit still moving by up to 0.4 between passes, so every haematocrit came from
whichever pass it stopped on. Converged, the per-specimen haematocrit ratio moved by up to 0.063
(SHR-C 0.998 → 1.061), and the shunt index and flow ratio by up to 0.09 (SHR-A shunt index
1.323 → 1.232). The same re-run centred the TH lookup on the voxel (it was half a voxel off), which
changed 4 to 59 edges per specimen from penetrating to bypassing. **One overlap pattern changed:**
the haematocrit ratio went from overlapping to separated. So a within-specimen ratio survives
scaling, but not a change in how flow and red cells divide at junctions.

### 6.2 Does the boundary choice produce the difference?

The residual spread of the shunt ratio under the face rule, as its tolerance moves over 1, 2 and
4 voxels, is 8.9% on the placed boxes (13.3% on the centred ones). The measured between-group
differences are 13% (shunt index), 12% (flow ratio), 8% (haematocrit ratio) and 3% (transit
ratio). Only the first two clear it, and those are the two whose ranges overlap; the one measure
that separates the cohorts sits just under it. The calibre floor for a within-specimen ratio is
5.9% (`cb_modelling_reference.md` §13.3), which the haematocrit difference clears by two points.

### 6.3 What is not validated

There is no labelled ground truth for either channel, so no per-cohort segmentation accuracy
score exists. The instrument-fairness argument rests on internal consistency and on the
single-classifier design, exactly as in H1, and that demonstration remains outstanding.

---

## 7. Results: §2.1 functional shunting

### 7.1 Per-specimen values

| Specimen | Group | Penetrating edges | Edge share | Flow share | **Shunt index** | Median flow ratio |
|---|---|---|---|---|---|---|
| WKY-A | WKY | 862 | 11.3% | 10.9% | 0.964 | 0.936 |
| WKY-B | WKY | 1,337 | 21.8% | 21.6% | 0.991 | 1.058 |
| WKY-C | WKY | 1,145 | 16.7% | 16.4% | 0.984 | 0.895 |
| SHR-A | SHR | 1,528 | 18.5% | 22.7% | 1.232 | 0.990 |
| SHR-B | SHR | 886 | 12.7% | 10.9% | 0.856 | 0.972 |
| SHR-C | SHR | 355 | 5.6% | 2.6% | 0.470 | 0.596 |

Every flow–haematocrit solve converged (330, 457, 404, 217, 173 and 184 passes);
`cb_h2_glomus_perfusion.json` records the stop reason per specimen.

### 7.2 The shunt index, and why flow share alone is not it

**Shunt index = flow share ÷ edge share.** A value of 1 means flow is indifferent to the clusters;
below 1 means the bypassing vessels carry disproportionately more, which is the shunting the
method proposes to detect.

| | WKY | SHR | Ratio |
|---|---|---|---|
| Shunt index | 0.980 | 0.853 | 0.87 |

**WKY sits at 1, so there is no functional shunting in the normotensive cohort.** WKY runs 0.964
to 0.991: flow is indifferent to the clusters.

**SHR does not behave as a cohort.** SHR-C, at 0.470, is the one specimen that shunts: its
penetrating edges carry half the flow their number would predict. SHR-A, at 1.232, does the
opposite, and SHR-B, at 0.856, sits between. The SHR range (0.470 to 1.232) contains the whole
WKY range, so the lower SHR mean is SHR-C's, and no cohort-level claim follows. SHR-C also has
the fewest penetrating edges by far (355, against 862 to 1,528 elsewhere) and the smallest glomus
volume (H1 §9A), so its index rests on the smallest sample.

Flow share alone would have said something else. It runs 16.3% in WKY against 12.1% in SHR, a
0.74 ratio that reads as diversion in the hypertensive network. But it tracks the edge share
(16.6% against 12.3%, also 0.74), and the edge share is itself downstream of the parenchymal volume
difference H1 §1.3 reports at 0.61. Dividing it out removes an apparent effect that was never
about flow.

### 7.3 Median flow ratio

Flow through penetrating edges over flow through bypassing edges, per specimen: **WKY 0.963
against SHR 0.852, and the cohorts overlap** (WKY 0.895 to 1.058, SHR 0.596 to 0.990). As with
the shunt index, the SHR mean rests on SHR-C (0.596); SHR-A and SHR-B sit inside the WKY range.

On the tissue-centred boxes this ratio separated the cohorts the other way (WKY 0.882, SHR 1.077,
no overlap). That separation did not survive the move to placed regions (§6.1).

---

## 8. Results: §2.2 spatial haematocrit profiling

Discharge haematocrit is solved by iterating flow against the Pries–Secomb phase-separation model,
to convergence in all six specimens (`cb_modelling_reference.md` §4.3, open item 37).

| Specimen | Hct penetrating | Hct bypassing | Ratio |
|---|---|---|---|
| WKY-A | 0.3843 | 0.3954 | 0.972 |
| WKY-B | 0.4010 | 0.4113 | 0.975 |
| WKY-C | 0.4243 | 0.4135 | 1.026 |
| SHR-A | 0.3996 | 0.3727 | 1.072 |
| SHR-B | 0.4196 | 0.3933 | 1.067 |
| SHR-C | 0.4129 | 0.3890 | 1.061 |

| | WKY | SHR | Ratio |
|---|---|---|---|
| Haematocrit ratio | 0.991 | 1.067 | **1.08** |

**The cohorts separate, and in the opposite direction to the one the method anticipates.** §2.2
proposes RBC starvation in the glomus microenvironment of the hypertensive network: a dense
capillary bed carrying mostly plasma. In every SHR specimen the penetrating vessels carry 6 to 7%
*more* red cells than the bypassing ones; in WKY they carry 3% less to 3% more. The ranges do not
overlap (WKY 0.972 to 1.026, SHR 1.061 to 1.072), so p = 0.10, the design floor. Unlike the
shunt index, the SHR values are tight and no one specimen carries the mean.

Three things limit what this may be taken to mean. The difference, 7.7%, clears the 5.9% calibre
floor for a within-specimen ratio but not the 8.9% boundary-tolerance spread (§6.2). It appeared
only when the flow–haematocrit loop was run to convergence: on the unconverged loop the same
networks gave 1.010 against 1.030, overlapping (§6.1). And it is the output of a skimming model
whose parameterisation is not settled (§8.1). The penetrating vessels are also narrower in SHR
(median EDT diameter 5.3 to 6.4 µm against 7.0 to 7.5 µm in WKY, both voxel-quantised and about
one voxel high, open item 42), which is the regime where that model is least certain.

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
| WKY-A | 4.683e4 | 4.619e4 | 1.014 |
| WKY-B | 7.990e4 | 8.618e4 | 0.927 |
| WKY-C | 6.083e4 | 5.872e4 | 1.036 |
| SHR-A | 9.106e4 | 1.259e5 | 0.723 |
| SHR-B | 8.597e4 | 7.926e4 | 1.085 |
| SHR-C | 9.891e4 | 7.940e4 | 1.246 |

| | WKY | SHR | Ratio |
|---|---|---|---|
| Transit ratio | 0.992 | 1.018 | 1.03 |

**Blood reaches the clusters in the same time as the surrounding tissue in both cohorts.** WKY runs
0.927 to 1.036, a spread of about 5% around 1. SHR spreads from 0.723 (SHR-A) to 1.246 (SHR-C)
and contains the whole WKY range, so the cohort means (0.99 against 1.02) do not differ. §2.4
anticipates sluggish transit to the sensors in the hypertensive network; one SHR specimen shows it
(SHR-C, about a quarter longer) and one shows the reverse (SHR-A, about a quarter shorter).

On the tissue-centred boxes this was the largest separation of the four measures (WKY 1.163, SHR
0.833, no overlap). It did not survive the move to placed regions (§6.1). Absolute transit times
are in arbitrary units and are reported only to show the ratio's construction.

---

## 10. Results: §2.3 glomus-specific hypoxic fraction

> **History.** Tier 1 first left O₂ solubility out of its diffusion (≈750× too strong) and washed
> blood out at systemic rather than local haematocrit (open items 22, 29, fixed 2026-09-26); it
> then mapped each vessel only to the cells its centreline crosses, which does not converge with
> the grid (open item 30, fixed 2026-09-26). On the tissue-centred boxes at 0.90 the fixed model
> gave 0–3% of glomus volume below 10 mmHg. The numbers below are on the placed boxes at 0.95 with
> the converged flow–haematocrit loop, the exact per-cell TH fraction (open item 38; it used to
> lose about 17% of the glomus volume) and the centred TH lookup (open item 40), re-run
> 2026-09-30. Outputs before that: `cb_h2_hypoxic_fraction_2026-09-30_pre_itemJ.json`; they differ
> by at most 0.11 mmHg in PO2 within TH at contrast 1 (1.0 mmHg at contrast 4).

### 10.1 What the model returns

Perfusion grid at 3 µm, vessels mapped over their cross-section, metabolic rate assigned per cell
from the TH fraction with the volume-weighted mean held constant so contrasts are comparable.
Uniform metabolism (contrast 1). PO2 within TH and in stroma are volume-weighted means; the
hypoxic fractions are shares of the glomus volume, and, for comparison, of all cells. Every
flow–haematocrit solve converged, and every Tier 1 solve converged (Newton, residual < 10⁻⁵).

| Specimen | TH volume | PO2 within TH | PO2 in stroma | TH hypoxic < 5 / 10 / 20 mmHg | All cells < 10 / 20 mmHg |
|---|---|---|---|---|---|
| WKY-A | 20.5% | 95.18 | 92.41 | 0 / 0 / 0 | 0 / 0 |
| WKY-B | 32.8% | 91.93 | 83.20 | 0 / 0 / 0 | 0.20% / 1.36% |
| WKY-C | 23.2% | 92.62 | 76.44 | 0 / 0 / 0 | 2.50% / 5.33% |
| SHR-A | 22.0% | 90.77 | 76.87 | 0 / 0 / 0 | 4.48% / 7.27% |
| SHR-B | 14.7% | 91.09 | 80.06 | 0 / 0 / 0 | 1.07% / 2.70% |
| SHR-C | 9.6% | 88.54 | 80.56 | 0 / 0 / 0 | 1.63% / 3.43% |

"TH volume" is the TH share of the grid's volume. It now matches the voxel count to within the
grid's overhang past the region (WKY-A 20.5% against 20.9%).

**No glomus tissue is hypoxic.** In every specimen, at every metabolic contrast (1, 2 and 4 times
stromal), none of the glomus volume falls below 5, 10 or 20 mmHg. The
low-PO2 cells that do exist, up to 7.3% of all cells below 20 mmHg in SHR-A, are in stroma. Where
in the stroma they sit (at the grid edge, or in pockets no vessel reaches) has not been checked.

**PO2 within the clusters is lower in every SHR specimen than in every WKY one.** WKY 93.24 mmHg
(91.93 to 95.18) against SHR 90.14 (88.54 to 91.09), a ratio of 0.967; at two and four times
stromal metabolism, 0.957 and 0.942, still without overlap. This is an absolute PO2, not a
within-specimen ratio: it rests on the absolute flow delivered to the tissue, which carries the
±47% calibre floor and the uncalibrated pressure pair of §11.1. A 3 to 6% difference in it is
recorded, not claimed.

On the centred boxes (before open items 38 and 40, at 0.90) the same table read 75.1 to 89.8 mmHg
within TH, 0 to 3.1% below 10 mmHg, and no cohort difference (ratio 1.00).

### 10.2 Why §2.3 returns zero

**The glomus-specific mechanism responds, but not enough to produce hypoxia.** Raising the glomus
metabolic rate from one to four times the stromal rate lowers PO2 within TH by 1.4 mmHg on WKY-A
(95.18 to 93.77) and 5.3 mmHg on SHR-C (88.54 to 83.25), the largest fall of the six. The mechanism
is live, unlike before open items 22 and 29, when the same change moved it by 0.015 mmHg. But the
lowest PO2 within TH at four times stromal, SHR-C's 83.25 mmHg, is still four times the highest
hypoxic threshold.

The reason is a property of the tissue. The oxygen diffusion length is

    sqrt(D · alpha · PO2 / M) = 20 µm at PO2 10, 35 µm at 30, 45 µm at 50

against a **median glomus-to-centreline distance of 7.9 to 9.7 µm** (H1 §1.5). Every glomus cell
sits at two fifths to a half of the shortest of those lengths (the one at 10 mmHg), so the tissue is not
diffusion-limited and a local sink produces only a shallow local gradient. The consumption rate is
not at fault: `M_max = 0.05` mmol/L/s is 0.067 mL O2 per mL per minute against roughly 0.040 for
brain.

**§2.3 asks for a glomus-specific hypoxic fraction in a bed too densely vascularised to have
one**, at these parameters. That is a statement about the carotid body as modelled, not about the
implementation. It holds at a boundary pressure pair that runs the network up to 1.8 times too fast
(§11.1); a lower, calibrated pressure drop would lower every PO2 here, and whether it would bring
glomus tissue below 20 mmHg has not been tested.

**The grid now covers all the tissue.** The perfusion grid takes its extent from the vascular
bounding box padded by half a cell. On the placed regions every network has nodes on the first and
last voxel plane of every axis, so the grid already contains the whole segmented region: no glomus
volume lies outside it in any specimen. `--pad-grid`, which extends the grid to the segmented
volume, therefore builds the identical grid and returns identical numbers
(`cb_h2_hypoxic_fraction_padded.json`). On the tissue-centred boxes 3.74% of SHR-A's glomus volume
and 7.54% of SHR-C's lay outside the grid (S28), and padding moved PO2 within TH by up to −4.7 mmHg;
the choice between the two grids (S29) no longer matters.

### 10.3 Grid convergence (open item 30)

**§2.3 runs at 3 µm with vessels mapped over their cross-section, and is converged there to
0.5 mmHg.** All six specimens, contrast 1, unpadded, cross-section mapping, on the placed boxes with
the converged loop, the exact TH fraction and the centred TH lookup (re-measured in the
2026-10-01 re-run; the 3 µm row equals the main run). Two specimens shown, the
extremes of the cohort:

| Grid | WKY-C median PO2 | WKY-C PO2 in TH | SHR-C median PO2 | SHR-C PO2 in TH | TH < 20 mmHg, any specimen |
|---|---|---|---|---|---|
| 10 µm | 92.44 | 93.23 | 89.77 | 88.81 | 0% |
| 6 µm | 92.35 | 92.74 | 89.59 | 88.50 | 0% |
| 4 µm | 92.29 | 92.50 | 89.52 | 88.38 | 0% |
| 3 µm | 92.69 | 92.62 | 90.01 | 88.54 | 0% |
| 2 µm | 92.58 | 92.49 | 89.86 | 88.48 | 0% |

"PO2 in TH" is the TH-weighted mean. The criterion is that both PO2 measures move less than
0.5 mmHg per step in every specimen. Over all six, the largest step is 0.49 mmHg from 10 to 6 µm,
0.235 from 6 to 4, 0.495 from 4 to 3 and 0.155 from 3 to 2. So 4 µm passes by 0.005 mmHg and 3 µm,
the frozen grid, with a margin. 3 µm gives the highest median of 4, 3 and 2 µm in every specimen,
a bump the TH fraction fix did not remove. It is the grid extent, not the solve (open item 39,
closed): the default grid's side is 304 µm at 4 µm but 300 µm at 3 and 2 µm, and the extra
vessel-free cells pull the 4 µm median down. On a grid pinned to 300 µm at every h, both measures
fall steadily with h (−0.09 to −0.16 mmHg per step). It is within the criterion either way.

**Why the grid had to be revisited.** Until open item 30 each vessel was placed only in the cells
its centreline crosses, whatever its width, so a finer grid drew a thinner vessel; in the limit it
is a line source, and the field around a line goes as ln r. Mapped that way on the old centred boxes, WKY-C median PO2
ran 91.38, 90.46, 89.52, 89.19, 87.90 at 10, 6, 4, 3, 2 µm (SHR-C 80.28, 79.95, 78.35, 78.12, 75.85):
about 1.5 and 1.9 mmHg lost per halving and no limit. Re-run on the placed boxes (2026-10-01) it
runs 90.58, 89.66, 88.38, 86.86 at 10, 6, 3, 2 µm (SHR-C 87.44, 86.27, 84.83, 83.04), 1.2–1.8 mmHg
lost per halving across the six, with the steps still growing. On a single straight 9 µm vessel the
centreline mapping falls 17.0, 11.5, 8.6, 6.8 mmHg at 9, 3, 1, ⅓ µm, while the cross-section mapping
gives 17.0, 23.4, 23.8, 23.2 (`tests/test_perfusion_tier1_grid_refinement.py`). Median calibre here
is 7.5–8.4 µm, so at 4 µm most vessels were drawn thinner than they are, and the tissue read as
less oxygenated than the model's own geometry implies: by 3.6–7.4 mmHg in TH at 4 µm, and unequally
across specimens, so the error did not cancel between cohorts.

The cross-section mapping keeps the full tissue metabolic rate in cells that hold lumen, a small
overcount of consumption (reference §11 row 24).

Before items 22 and 29 the centreline sequence was 27.34, 27.92, 28.21, halving and extrapolating
to about 28.5; before the sharing fix of §4.5 it ran 42.0, 46.9, 50.5. Outputs:
`examples/outputs/cb_h2_hypoxic_fraction_xsec_grid{10,6,4,3,2}.json` (cross-section, all six), and
for the centreline mapping `..._grid{10,6,3,2}.json` (placed boxes, all six; the centred-box
sequence is archived as `..._grid{10,6,3,2}_2026-09-29_pre_rerun.json`).

---

## 11. Limitations

### 11.1 Limitations that bound the claims

**Absolute perfusion is not calibrated, and at 60/20 mmHg it runs fast.** Flow-weighted velocity
is 955 to 1,808 µm/s across the six against a physiological 200 to 1,000; five of six sit above
the upper end (SHR-A, at 955, just inside). A 500 µm/s velocity would need a drop of 11 to 21 mmHg,
not 40. The face boundary rule carries 2.2 to 4.3 times less flow than the band rule it replaced
(4.0 to 7.6 times on the centred boxes), a cost of §4.2 that its own validation did not measure.
Every solve behind these numbers, under both rules, converged (156 to 457 passes). So the absolute scale depends on a pressure pair that is not measured
in the carotid body and on a boundary choice with no anatomical calibration, on top of the ±47%
calibre floor. **No absolute perfusion quantity in this document is defensible.** Every reported
measure is a ratio for this reason.

Until open item 12 was re-derived (2026-09-27) this paragraph said the opposite: 4 to 10 µm/s, 20
to 100 times too slow, needing about 3,257 mmHg. Those figures came from the inflated resistances
of the rheology bug. `examples/cb_h2_absolute_perfusion.py` reproduces both the old table (with
the old code) and the new one.

**Statistical power.** n = 3 per group. The exact two-sided permutation p cannot fall below 0.10
for any arrangement of three against three. No claim of statistical significance is made.

**Three of four measures overlap.** The shunt index, the median flow ratio and the transit ratio
all have intersecting ranges, and on the first two the SHR mean rests on SHR-C. The one that
separates, the haematocrit ratio, does so by 0.035 between the nearest specimens and appeared only
once the flow–haematocrit loop converged (§6.1).

**§2.3 returns zero on this geometry** (§10.2): no glomus tissue falls below 20 mmHg in any
specimen at any metabolic contrast.

**The skimming model's parameterisation is unsettled** (§8.1), and it bears directly on §2.2.

**No per-cohort accuracy validation exists** for either segmentation channel.

### 11.2 Limitations that bound the precision

**Boundary sensitivity.** The residual spread of a ratio under the face rule is 8.9%, against
measured differences of 13% (shunt index), 12% (flow ratio), 8% (haematocrit ratio) and 3%
(transit ratio). The axis itself, held at 1, would spread the ratio by 33 to 34% if it were free.

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
| P2 | The reported ratios are insensitive to uniform corrections that moved absolute quantities by three to five orders | §6.1 | Three independent corrections, each ≤ 0.02 | **Established** |
| P2a | The ratios are *not* insensitive to corrections that change how flow and red cells divide at junctions | §6.1 | Open item 12 moved them by up to 0.09; the converged loop by up to 0.09 and turned the haematocrit ratio from overlapping to separated | **Established** |
| P3 | There is no functional shunting in WKY | §7.2 | Shunt index 0.964 to 0.991 | **Established** |
| P3a | There is no cohort-level functional shunting in SHR | §7.2 | SHR 0.470 to 1.232; SHR-C shunts, SHR-A the reverse | **Provisional** (was P3, Established, for both cohorts) |
| P4 | Flow share alone would have reported shunting that is an artefact of edge share | §7.2 | 0.74 ratio in flow share, 0.74 in edge share | **Established** |
| P5 | Transit time to the glomus clusters is shorter in SHR relative to surrounding tissue | §9 | Ranges overlap; ratio 1.03 (was 0.72, no overlap, on the centred boxes) | **Not supported** (was Provisional) |
| P6 | Penetrating capillaries carry relatively more flow in SHR | §7.3 | Ranges overlap; SHR mean is lower, 0.85 against 0.96 (was 1.08 against 0.88, no overlap) | **Not supported** (was Provisional) |
| P7 | The direction of P5 opposes the stagnant-hypoxia prediction §2.4 makes | §9 | P5 does not hold | **Not supported** (was Provisional) |
| P8 | Haematocrit in glomus-penetrating vessels is lower in SHR, as §2.2 anticipates | §8 | SHR is higher, without overlap | **Not supported** |
| P8a | Haematocrit in glomus-penetrating vessels, relative to bypassing, is *higher* in SHR | §8 | No overlap, p = 0.10 floor; 7.7% against a 5.9% calibre floor and an 8.9% boundary spread; present only on the converged loop; skimming model unsettled (§8.1) | **Provisional** (new) |
| P9 | The shunt index differs between cohorts | §7.2 | Ranges overlap; SHR mean rests on SHR-C | **Not supported** |
| P10 | A glomus-specific hypoxic fraction is measurable on this geometry | §10.2 | 0% of glomus volume below 5, 10 and 20 mmHg in every specimen at every contrast; grid-converged at 3 µm | **Not supported** (was Needs review) |
| P10a | PO2 within the glomus clusters is lower in SHR | §10.1 | No overlap at any contrast (ratio 0.967 to 0.942), but an absolute PO2 under the ±47% floor and an uncalibrated pressure pair | **Not supported as a finding; recorded** (new) |
| P11 | Any absolute perfusion quantity reported here is physiological | §11.1 | Velocity up to 1.8× above the physiological range at 60/20 mmHg; throughput 2 to 4× with the boundary rule (was: 20 to 100× low, before open item 12) | **Not supported** |

The defensible position is P1 to P4 (Established) plus P3a and P8a (Provisional). P8a is the one
cohort difference in H2, and it runs against the hypothesis's anticipated direction. Nothing else
should be presented as a result.

---

## 13. Future work

| Priority | Work | Unblocks | Cost |
|---|---|---|---|
| 1 | Calibrate absolute perfusion: a measured or defensible pressure drop across the region (11 to 21 mmHg gives 500 µm/s) and a boundary rule whose throughput is anatomically grounded | Every absolute quantity; §2.3 | Investigation |
| 2 | Settle the skimming model's parameterisation, fractional flow against velocity | §2.2, and P8a, the one cohort difference | Literature plus a re-run |
| 3 | Level the TH classifier's residual 2.1× cohort skew | The stated bound on both whitepapers | Hours of labelling |
| 4 | Complete perivascular boundary labelling on the vessel channel | Calibre precision, inherited by every resistance | Hours; H1 §13 item 1 |
| 5 | Hand-labelled held-out regions in both cohorts, both channels | Per-cohort validation scores | Hours |
| 6 | More specimens, or acceptance that H2 is answered descriptively | Statistical power | Experimental |

Items 1 and 3 are the only ones on the critical path to a defensible absolute H2 result. Item 2
decides whether P8a, the only measure that separates the cohorts, is a property of the tissue or
of the skimming model.

---

## Appendix A: Frozen parameter set

| Parameter | Value | Source |
|---|---|---|
| Region size | 160³ voxels, 0.0266 mm³, placed on tissue (`place_roi`) | H1 §5.3 |
| Voxel size | 1.8639 × 1.866 × 1.866 µm | acquisition metadata |
| Vessel probability threshold | 0.95, hysteresis seed 0.999 | H1 threshold selection, `cb_settings` |
| TH probability threshold | 0.50 | `cb_settings.TH_THRESHOLD`; see H1 §9A.4 |
| Boundary rule | face-crossing, axis 1, tolerance 1 voxel | §4.2 |
| Inlet / outlet pressure | 60 / 20 mmHg | §5.2, and §11.1 |
| Systemic haematocrit | 0.45 | conventional |
| Viscosity law | Pries–Secomb **in vivo** | §4.3 |
| Plasma viscosity | 1.2 cP | conventional |
| Flow–haematocrit loop | relaxation 0.2, cap 1000 passes, relative flow 10⁻⁶, haematocrit 10⁻⁴ | `cb_settings`, open item 37 |
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
python examples/cb_h2_hypoxic_fraction.py --pad-grid --out examples/outputs/cb_h2_hypoxic_fraction_padded.json   # §10.2 padded grid
python examples/cb_h2_error_propagation.py           # the noise floor
python examples/cb_h2_vtk.py                         # ParaView artefacts, not a result
```

`examples/cb_h2_paraview_guide.md` covers the exports and which arrays carry which section.

| Artefact | Supplies |
|---|---|
| `cb_h2_glomus_perfusion.json` | §7, §8, §9 |
| `cb_h2_absolute_perfusion.json` | §11.1 |
| `cb_h2_hypoxic_fraction.json` | §10 |
| `cb_h2_hypoxic_fraction_padded.json` | §10.2, padded grid |
| `cb_h2_hypoxic_fraction_xsec_grid{10,6,4,3,2}.json` | §10.3 |
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
| Flow–haematocrit loop never converged (open item 37) | Stagnant edges out of the transport, one-vs-rest skimming, scale-free stop, relaxation 0.2 | `cfee721` |
| Per-cell TH fraction lost ~17% of glomus volume (open item 38) | Exact voxel–cell overlap, no clip | `870de15` |
| TH lookups half a voxel off the graph (open item 40) | Voxel-centred lookup | `5839aae` |

The assessment behind these is `h2_pipeline_capability_assessment.md`, findings S1 to S30.
