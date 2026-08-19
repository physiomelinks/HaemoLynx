# H2 Pipeline Capability Assessment: Critical Review Against the Perfusion Hypothesis

> **Purpose:** Persistent record of the stage-by-stage critical assessment of the ImageLynx
> haemodynamics and mass-transport stack, evaluated against Hypothesis 2 (CB perfusion, SHR vs. WKY)
> and its four proposed analysis methods.
>
> **Companion document:** [`hypothesis_testing_methods.md`](hypothesis_testing_methods.md), which
> defines the hypotheses and analysis methods (§2.1–§2.4 references below point there).
>
> **Sibling document:** [`h1_pipeline_capability_assessment.md`](h1_pipeline_capability_assessment.md),
> the equivalent review for the morphology hypothesis. H2 consumes H1 outputs directly, so its
> findings are inherited rather than restated.
>
> **Assessment date:** 2026-08-15 (Phase 1), extended by Phase 2 Part 1 the same day
> **Branch:** `cb_pipeline_improvements_sweep`, Phase 1 at `c3c236c`, Part 1 at `1472a83`
> **Test data:** the six-specimen H1 artefact set in `examples/outputs/cb_h1_paraview/`
> (34,900 edges, frozen threshold 0.90, matched 0.0266 mm³ ROI per specimen)
>
> **Method:** code inspection plus direct numerical execution of the real pipeline functions on the
> real six-specimen artefacts. Figures labelled **measured** are empirical, not inferred. Findings
> labelled **inspected** come from reading the code and are not backed by execution; they are stated
> separately because that distinction is what makes the document auditable.

---

## Document status

**All three phases complete.** This document is being built in phases so that the
headline verdict is available before the full stage-by-stage review lands. Findings are numbered in
the order they were reached, not in the order they are best read, and superseded ones are annotated
rather than rewritten so that the sequence stays auditable. **S2 is retracted; read S14 with it.**

| Phase | Scope | State |
|---|---|---|
| 1 | Survey, call-graph trace, benchmark execution, headline verdict | **complete** |
| 2 | Stage-by-stage review | **complete** (S10–S20) |
| 3 | Per-method verdicts and tiered plan of attack | **complete** |

Unlike the H1 assessment, which had its status-marker convention retrofitted after a remediation
sweep, this document is born with it. Every finding carries a stable identifier (`S1`, `S2`, ...) so
that later phases and commits can reference findings without quoting them.

| marker | meaning |
|---|---|
| `STATUS — FIXED` | a fix has landed; commit named |
| `STATUS — SUPERSEDED` | the finding may stand, but the figures no longer describe the pipeline |
| `STATUS — OUTSTANDING` | unchanged; still true of the current tree |
| `STATUS — DEFERRED` | out of scope by explicit decision; reason given |

---

## Which definition of H2 this document assesses

H2 is defined twice in the repository, and the two definitions do not match.

| Source | Count | Contents |
|---|---|---|
| [`hypothesis_testing_methods.md`](hypothesis_testing_methods.md) §2.1–§2.4 | 4 | shunting, haematocrit skimming, ADR hypoxic fraction, transit time |
| [`modelling_and_hypothesis_testing_documentation.md`](modelling_and_hypothesis_testing_documentation.md) §2 | 5 | perfusion fields, TH-masked metabolic grid, hypoxic fraction, shunt/perfusion ratio, pressure gradients |

They overlap but neither contains the other. Haematocrit and plasma skimming appear only in the
first. Pressure distribution and the explicit shunt/perfusion flow ratio appear only in the second.

**This document anchors on `hypothesis_testing_methods.md` §2.1–§2.4**, because that is the document
the H1 assessment anchored on and consistency between the two reviews matters more than either
choice. The two items unique to the modelling document are folded in as sub-analyses: pressure
gradients under §2.1, and the explicit shunt/perfusion ratio under §2.1 as its quantitative form.

This discrepancy should be resolved in the source documents rather than left to a reader to notice.

---

## Part 0: Survey and headline verdict

### Scope of this phase

What executes, what is exercised, what has never been run, and what the inherited H1 findings do to
the physics. No stage-level review; that is Phase 2.

### Headline verdict

**The physics is in better condition than the data it would consume.**

That is the opposite of the H1 finding, and it inverts the prior this assessment was planned around.
The H1 review found a pipeline whose operators were actively destroying the signal they were meant
to measure. The perfusion stack has the reverse problem: the solvers are validated against
closed-form solutions and pass, but the radii that drive them carry a per-edge uncertainty that the
governing physics amplifies fourfold.

Three findings determine everything else:

1. **The physics core is genuinely validated** (S1). Fifty-two analytic tests pass, covering
   Poiseuille series and parallel resistance, pure diffusion, zero-order metabolism, radial point
   sources, the Krogh cylinder, Fåhræus–Lindqvist viscosity, Pries–Secomb plasma skimming,
   Bohr–Haldane, and Henderson–Hasselbalch. This is a real analytic benchmark suite covering
   precisely the physics H2 requires, and it is in far better shape than the H1 review would predict.

2. ~~**None of it has ever been run on real carotid body data** (S2).~~ **This was wrong and is
   retracted.** The flow solve, the rheology loop and the perfusion solve have all run on all six
   specimens. See the corrected S2 and the new S14, which is what looking properly turned up.

3. **Resistance goes as d⁻⁴, and the diameters are not good enough to survive that** (S6). At the
   median edge, one voxel of diameter uncertainty implies **94% uncertainty in that edge's
   resistance**. For 95.9% of the 34,900 edges the figure exceeds 50%.

Finding S6 is the governing constraint on H2 and it is not a solver defect. It cannot be fixed in
the physics. It is inherited directly from the measurement H1 declined to report as a finding, and
it will bound every H2 result until the underlying calibre measurement improves.

A fourth, found while acting on the constriction decision, belongs with these: **the pipeline was
multiplying a fabricated constriction ratio onto measured diameters** (S9), reaching 12.3% of edges
and inflating their resistance by a median of about 12× (S14). Fixed in `1ee46a1`.

**Consequence.** H2 is not blocked by the perfusion code. It is blocked by the TH channel, which is
a data-availability decision, and it is bounded by calibre precision, which is an imaging and
segmentation problem. Effort spent hardening the solvers before those two are addressed is effort
spent on the part of the chain that is already working.

**Updated after Phase 2, Part 1.** S6's bound has been resolved into something more useful than a
warning. Independent calibre error averages down more than twentyfold across a real network solve
and is negligible. Correlated error, which is what a shared threshold and classifier produce, does
not average down at all (S12). Calibrated against the threshold shift the H1 sweep actually
measured, 0.922 µm (S15), the operative noise floor is:

| Quantity | Calibre floor | Against H1's 27% to 40% effects |
|---|---|---|
| Absolute network flow | ±45% | cannot resolve them |
| Within-specimen ratio | ±6.3% | comfortably, **but see below** |

A within-specimen ratio cancels 86% of the correlated calibre error, at both perturbation sizes
tested (S13). The rule that follows still holds:

> **Express H2 as ratios computed within a specimen, never as absolute flows compared between
> specimens.** §2.1 and §2.3 already satisfy this. §2.4 does not, and needs re-posing.

**Revised again by S20, and this is the number to carry away.** Calibre is not the only error
source, and it is not the largest. The boundary nodes are chosen by an axis and a band width with
no anatomical basis, and varying only that choice moves the same shunt ratio by **25.3%**, four
times the calibre term. The operative floor for a within-specimen ratio is therefore about **26%**,
against H1 effects of 27% to 40%: a margin of roughly 1.1× to 1.6×, not the fourfold this section
claimed before S20 was measured.

**H2's binding constraint is its boundary conditions, not its calibre and not its physics.** That is
the one piece of good news in it, because unlike calibre it is fixable in software.

---

### S1. The analytic benchmark suite is real, and it passes

**Measured.** All five haemodynamics test modules, 52 tests, pass in 3.51 s:

```
venv/bin/python -m pytest tests/test_haemodynamics_analytical.py tests/test_haemodynamics.py \
  tests/test_haemodynamics_perfusion.py tests/test_haemodynamics_automated_fwhm.py \
  tests/test_haemodynamics_rheology_integration.py
52 passed in 3.51s
```

Coverage in `test_haemodynamics_analytical.py` is against closed-form solutions rather than
regression snapshots:

| Part | Validates |
|---|---|
| 1 | Poiseuille resistors in series and in parallel |
| 2 | 1D pure diffusion, zero-order metabolism (parabolic), radial point source decaying as 1/r |
| 3 | RBC flux conservation under phase separation, plasma skimming direction, wall shear stress, resistance integration for constricted geometry, Fåhræus–Lindqvist curve shape |
| 4 | Krogh cylinder radial diffusion |
| 5 | Bohr and Haldane shift equations, Henderson–Hasselbalch bounds, multi-species 0D Fick mass balance |

**Qualification.** 3.51 s for 52 tests means small synthetic grids. Passing an analytic benchmark
establishes that the discretisation converges to the right answer on a problem with a known answer.
It does not establish behaviour at the size, conditioning, or connectivity of a real extracted
network. That is a Phase 2 measurement.

`STATUS — OUTSTANDING` (no action needed; recorded as the baseline)

### S2. RETRACTED. The stack has run on all six specimens

**This finding was wrong.** It is left in place rather than deleted, because Phase 2's priorities
were set by it and the correction is what produced S14.

**What was claimed:** that `examples/cb_h1_batch.py` contains no reference to `haemodynamics`,
`resistance`, `perfusion` or `flow`, and therefore that the six-specimen run stopped at morphometry.

**Why it was wrong.** The grep was accurate and the inference from it was not.
`cb_h1_batch.py` does not import the haemodynamics modules because it shells out to the pipeline,
`subprocess.run(...)` at [cb_h1_batch.py:144](examples/cb_h1_batch.py:144), and
`carotid_image_to_model.py` then runs every phase including haemodynamics. Absence of a symbol in a
driver is not absence of the behaviour it drives.

**Measured, correcting it.** `examples/outputs/cb_h1_batch/<SPEC>/resistance_network_vessels_flow.vtp`
exists for all six specimens and carries populated `resistance`, `pressure_u`, `pressure_v`,
`pressure_drop`, `flow_signed`, `flow_abs`, `hematocrit`, `viscosity` and `wall_shear_stress_pa`.
For WKY-A: 4,512 edges, `flow_abs` non-zero on 92.9%, `viscosity` spanning 1.2 to 8.71, which means
the Fåhræus–Lindqvist loop genuinely ran rather than returning the plasma baseline.
`resistance_network_perfusion.vti` exists alongside it.

Zero-flow edges run 4.6% to 7.1% per specimen, consistent with the stranded terminals of S10.

`STATUS — FIXED` by retraction. The consequence is not that things are better than reported: it is
that **six specimens' worth of flow, rheology and perfusion output already exist, and were produced
with `constrict_at_pericytes = True`**, so every one of those numbers carries the S9 fabrication.
They must be regenerated, not read.

### S3. The endothelial barrier model is unreachable under default configuration

**Inspected.** `PerfusionConfig` sets both flags to `True`
([carotid_image_to_model.py:362](examples/carotid_image_to_model.py:362),
[:366](examples/carotid_image_to_model.py:366)):

```python
use_endothelial_barrier_model: bool = True
use_multi_species_model: bool = True
```

The solver selection tests multi-species first ([:1373](examples/carotid_image_to_model.py:1373)),
so with both defaults in force the multi-species branch always wins and
`solve_coupled_1d3d_perfusion` is never called. The endothelial permeability model is configured on,
advertised in the log line it never prints, and dead.

A reader setting `use_endothelial_barrier_model = True` to enable that model gets no error, no
warning, and a different solver.

`STATUS — OUTSTANDING`

### S4. The ADR matrix is built unconditionally and discarded on two of three paths

**Inspected.** `build_adr_matrix` is called at
[:1370](examples/carotid_image_to_model.py:1370), outside the branch. Its three outputs
`A, q_total, s_incoming` are consumed only in the `else` branch at
[:1400](examples/carotid_image_to_model.py:1400). Under the default configuration the matrix is
assembled and thrown away.

Wasted assembly is the minor half of this. The material question is whether the coupled solvers
rebuild the operator internally on the same discretisation, or on a different one. If they differ,
the three solvers are not three treatments of one problem.

**Answered by S16: they build an identical and correct discretisation.** The wasted assembly under
the default configuration stands.

`STATUS — OUTSTANDING` (the redundant assembly only)

### S5. Radii reaching the flow solve are measured, but the fallback to synthetic is silent per edge

This finding refutes the prior it was written to test. The concern was that the hard-coded
branch-order diameter law at [:1071–:1125](examples/carotid_image_to_model.py:1071), which anchors
on 15 µm arterial, 4 µm capillary and 20 µm venous, would contaminate the resistance calculation.

**Measured.** Across all six specimens and 34,900 edges, diameter provenance is:

| Specimen | Edges | measured_edt | synthetic | constant | fwhm |
|---|---|---|---|---|---|
| WKY-A | 4,512 | 100.0% | 0 | 0 | 0 |
| WKY-B | 3,932 | 100.0% | 0 | 0 | 0 |
| WKY-C | 6,699 | 100.0% | 0 | 0 | 0 |
| SHR-A | 6,815 | 100.0% | 0 | 0 | 0 |
| SHR-B | 8,077 | 100.0% | 0 | 0 | 0 |
| SHR-C | 4,865 | 100.0% | 0 | 0 | 0 |

The default `radius_assignment_mode` is `"edt_radius"`
([:268](examples/carotid_image_to_model.py:268)), and H1's Stage 21 silent-fallback defect has been
fixed: `_raise_if_measurement_mode_measured_nothing`
([poiseuille.py:9](src/ImageLynx/haemodynamics/poiseuille.py:9)) raises when `edt_radius` is selected
and nothing was measured.

**The residual risk is that the guard is whole-graph, while the fallback is per-edge.**
[poiseuille.py:191–211](src/ImageLynx/haemodynamics/poiseuille.py:191): an edge missing
`edt_diameter_um` falls through to `diameter_by_branch_order`, is tagged
`provenance = "synthetic_branch_order"`, and carries on. The guard fires only when *zero* edges
measured. A run in which 90% of edges measure and 10% fall back to a fabricated 15/4/20 µm law
raises nothing and prints nothing, and those 10% carry their fabrication into resistance at the
fourth power.

The provenance counts exist. Nothing currently asserts on them.

**Important scope limit.** The table above is measured on the H1 morphometry export, which sources
from `export_per_edge_morphometry`. It establishes that every edge *carries* a measured
`edt_diameter_um`, and therefore that the fallback would not fire on this data. It does not
establish that the flow solve consumes that same value, because the flow solve has never been run
here (S2). Phase 2 closes that gap.

**Understated, see S9.** This finding concluded that the synthetic law reached only edges lacking an
EDT measurement. That holds for the baseline diameter and fails for the constricted one: the
constriction ratio was applied multiplicatively to measured diameters as well, and the constricted
path was the default. S9 records the mechanism and the fix.

`STATUS — OUTSTANDING` (the per-edge silent fallback on the baseline diameter; the constriction half
is `FIXED` in `1ee46a1`)

### S6. The d⁻⁴ amplification is the governing constraint on H2

Hagen–Poiseuille resistance scales as the inverse fourth power of diameter, so fractional error
propagates as `δR/R ≈ 4·δd/d`.

The H1 whitepaper §8.2 disqualified calibre as a reportable finding. Its stated resolution limit is
**one EDM quantisation step of 1.87 µm**, against measured median calibre of 7.5 to 8.4 µm. H2 rests
on exactly those numbers, raised to the fourth power.

**Measured**, over the pooled 34,900 edges, taking one voxel (1.866 µm) as the diameter uncertainty:

| Percentile | Diameter (µm) | δd/d | **δR/R** |
|---|---|---|---|
| p5 | 3.732 | 50.0% | **200.0%** |
| p25 | 5.868 | 31.8% | **127.2%** |
| p50 | 7.904 | 23.6% | **94.4%** |
| p75 | 10.550 | 17.7% | **70.8%** |
| p95 | 13.963 | 13.4% | **53.5%** |

- **95.9%** of edges carry more than 50% resistance uncertainty.
- **37.2%** of edges carry more than 100%.

For scale, the measured p5-to-p95 calibre spread of 3.74× becomes a **196× spread in resistance**.
The network's resistance structure is dominated by a quantity measured to roughly a quarter of its
own value.

**One prior refuted.** This assessment was planned around a concern that the EDT quantisation lattice
would appear as a coarse comb in resistance space. It does not. **Measured:** the pooled diameters
take 823 distinct values with a median gap of 0.0023 µm, because junction trimming and B-spline
smoothing break the raw lattice. The values are numerically dense. The 1.87 µm figure is the scale
below which a difference is not *physically* resolved, not the spacing of the values, and the
distinction matters: the problem is uncertainty, not discretisation.

**This is not fixable in the solver.** No improvement to the Picard iteration, the ADR
discretisation, or the rheology reduces it. It is bounded by voxel size against vessel calibre, and
it moves only with better imaging resolution, better segmentation, or an estimator with sub-voxel
precision and a characterised error model.

**Consequence for H2's claims.** A between-group perfusion difference must clear this noise floor.
H1's topological measures showed 27% to 40% group differences at p = 0.20 with n = 3. A flow-derived
measure inherits a per-edge resistance uncertainty near 94% at the median. Whether that averages
down across thousands of edges into a usable network-level quantity depends on whether the errors
are independent, and they are not: they are driven by a shared threshold and a shared classifier, so
they are correlated within a specimen and potentially differentially correlated between cohorts.
**Quantifying that is the single highest-value measurement in Phase 2**, and it is exactly the
question H1's §6 instrument-validation section was built to answer for the topological measures.

`STATUS — OUTSTANDING`

### S7. Boundary conditions are geometric, not anatomical

**Inspected.** `select_boundary_terminal_nodes`
([boundaries.py:10](src/ImageLynx/graph/boundaries.py:10)) selects inlets as degree-1 nodes whose
coordinate along one axis falls in the top `edge_percent`, and outlets as those in the bottom
`end_percent`. Defaults ([:203–:206](examples/carotid_image_to_model.py:203)):

```python
edge_percent: float = 25.0
end_percent: float = 25.0
node_edge_axis: int = 0            # z
boundary_permeability_mode: str = "caged"
```

The criterion is purely positional. There is no calibre criterion, no anatomical criterion, and no
flow-direction criterion. On the H1 ROI, which is a **tissue-centred cube placed by signal intensity
rather than by anatomy**, most degree-1 nodes are vessels cut by the crop, not anatomical terminals.

Three consequences follow, and all three are H2-specific because H1 never needed a flow direction:

1. **The flow axis is a configuration default, not a measurement.** Blood is driven along `z`
   because `node_edge_axis = 0`, not because the anatomy says so. There is no reason the true
   arterial supply of a mid-organ ROI should align with the acquisition `z` axis.
2. **`"caged"` mode strands the other four faces.** Terminals on the `x` and `y` faces are neither
   inlets nor outlets, so they become no-flow dead ends. Every vessel the crop severed on those
   faces is modelled as though it terminated there. Blood entering the top 25% of `z` can leave only
   through the bottom 25% of `z`.
3. **The Tier 2 fallback is more arbitrary still.** If either band comes up empty
   ([boundaries.py:52–60](src/ImageLynx/graph/boundaries.py:52)), the selection silently switches to
   the extreme 10% of *all* nodes by axis coordinate, dead end or not.

If ROI orientation relative to the true vascular axis differs systematically between cohorts, this
becomes a between-group confound of exactly the kind H1 §6 was built to detect. **Phase 2 measures
the terminal-node census per face per specimen**, which settles how much of the network is stranded
and whether the fraction differs by group.

**A note on a defect that is not on the live path.** `select_boundary_nodes_by_method`'s
`edge_percent` branch ([boundaries.py:169](src/ImageLynx/graph/boundaries.py:169)) calls
`select_boundary_terminal_nodes` **without** `voxel_size`, which would compare physical node
positions against a voxel-count extent and mis-scale both bands by the voxel size. The live carotid
path passes `voxel_size` explicitly ([:1032](examples/carotid_image_to_model.py:1032)) and is
unaffected. The `resistance_network_pipeline.py` examples do reach the affected branch.

`STATUS — OUTSTANDING`

### S8. Two solvers are missing from the package's public exports

**Inspected.** [`haemodynamics/__init__.py`](src/ImageLynx/haemodynamics/__init__.py) imports all
three solvers but lists only `solve_perfusion_steady_state` in `__all__`.
`solve_coupled_1d3d_perfusion` and `solve_multi_species_perfusion` are absent, so
`from ImageLynx.haemodynamics import *` does not export them. Attribute access still works, which is
why the entry script is unaffected.

**One prior refuted.** The `ImageLynx.hemodynamics` package was expected to be an accidental
US-spelled duplicate. It is not: it is a deliberate one-line re-export shim that raises
`DeprecationWarning`. No action needed.

`STATUS — OUTSTANDING` (S8 only; the shim finding is closed)

### S9. Variable constriction fabricated a constricted calibre on every measured edge

Found after Phase 1 was committed, while acting on the decision to disable the constriction
capability. It is recorded here because it is the strongest instance of the pattern S5 describes,
and because S5 as written understated it.

**Measured**, by the fail-first test accompanying the fix: with constriction enabled, branch order
B01 resolved to `{"d1": 15.0, "d2": 9.0}`, which is the 0.60 intimal-cushion ratio applied to a
synthetic 15 µm anchor.

The mechanism is what makes this serious. Under `edt_radius` the measured diameter becomes `d1`, and
the constricted calibre is then computed as
`d2 = d1 * (d2_dict / d1_dict)`
([poiseuille.py:333–338](src/ImageLynx/haemodynamics/poiseuille.py:333)), with the ratio taken from
the synthetic branch-order dict. **The fabricated ratio multiplied every edge, including fully
measured ones.** S5 concluded that the synthetic law reached only edges lacking an EDT measurement.
That is true of the *baseline* diameter and false of the *constricted* one, and with
`constrict_at_pericytes` defaulting to `True` the constricted path was the live path.

Resistance scales as d⁻⁴, so the 0.5 ratio at the capillary anchor is a 16× local resistance error
applied to a measured vessel. Combined with S6, an H2 flow solve run before this fix would have
carried both a 94% median uncertainty and a systematic fabricated narrowing.

The sites came from a hard-coded topological rule (branch order 1, and the midpoint branch order)
rather than from the imaging, and the severity ratios from fixed constants rather than from any
model of vasomotor tone.

`STATUS — FIXED` in `1ee46a1`. `constrict_at_pericytes` now defaults `False` and raises if set
`True`; the branch-order fallback stores `d2 = d1`. The capability remains live for
`examples/resistance_network_pipeline.py`, which owns it and supplies a measured mask. The pericyte
modules are marked frozen for the CB work and are **out of scope for Phase 2**.

---

## Part 1: The network solve and how calibre error propagates through it

Phase 2, first tranche. Everything here is measured on the six-specimen artefact set using a
Poiseuille conductance network built from the measured edges, with inlet and outlet nodes chosen
exactly as `select_boundary_terminal_nodes` would choose them. No pipeline re-run was needed, so
these results stand independently of S2.

### S10. The crop is not the boundary problem; interior dead ends are

S7 argued that most degree-1 nodes would be vessels severed by the ROI crop. **That is wrong, and
measured to be wrong.** Counting terminals within one voxel of each of the six ROI faces:

| Specimen | Terminals | On any face | Interior | Interior share |
|---|---|---|---|---|
| WKY-A | 544 | 71 | 473 | 86.9% |
| WKY-B | 534 | 88 | 446 | 83.5% |
| WKY-C | 674 | 91 | 583 | 86.5% |
| SHR-A | 545 | 80 | 465 | 85.3% |
| SHR-B | 754 | 104 | 650 | 86.2% |
| SHR-C | 503 | 67 | 436 | 86.7% |

**About 86% of terminals are interior**, nowhere near a crop plane. They are skeletonisation spurs
and segmentation breaks, not severed vessels. A real capillary bed has few genuine interior dead
ends, so this is a statement about mask quality, and it inherits directly from the incomplete
perivascular labelling H1 §2.3 records.

The consequence is worse than the one S7 described. The selection rule assigns arterial pressure to
whichever degree-1 nodes fall in the top 25% band, and since 86% of candidates are interior spurs,
**most selected inlets are arbitrary interior skeleton artefacts** rather than boundary vessels.

**Measured**, under the default `edge_percent = end_percent = 25.0` on graph axis 0:

| Specimen | Inlets | Outlets | Inlet:outlet | Stranded | Stranded share |
|---|---|---|---|---|---|
| WKY-A | 176 | 66 | 2.67 | 302 | 55.5% |
| WKY-B | 150 | 95 | 1.58 | 289 | 54.1% |
| WKY-C | 188 | 138 | 1.36 | 348 | 51.6% |
| SHR-A | 157 | 140 | 1.12 | 248 | 45.5% |
| SHR-B | 224 | 176 | 1.27 | 354 | 46.9% |
| SHR-C | 33 | 132 | 0.25 | 338 | 67.2% |

Two things to take from this. Roughly half of all terminals are stranded as no-flow dead ends under
`"caged"`. And the inlet-to-outlet ratio spans **10.7×** across six specimens, from 2.67 to 0.25,
which under a fixed pressure boundary condition directly scales how much flow the network carries.
Group means are 1.87 (WKY) against 0.88 (SHR); dropping the SHR-C outlier still leaves 1.87 against
1.20. With n = 3 that is not established as a systematic confound, but it is the right size and the
right direction to become one, and it is set by ROI placement rather than by biology.

`STATUS — OUTSTANDING`

### S11. The networks are singly connected, so the solve is well posed

**Measured.** Every specimen's extracted graph is a **single connected component**, and 100% of
edges lie in a component containing at least one inlet and at least one outlet. There are no
orphaned subnetworks carrying zero flow.

This closes the worst case for open ambiguity 3. Whatever else is wrong with the boundary selection,
it does not leave part of the network unsolvable.

`STATUS — OUTSTANDING` (recorded as a positive; no action)

### S12. Independent calibre error averages down; correlated error does not

The decisive Phase 2 result, and the one that determines whether H2 is answerable.

S6 established a per-edge resistance uncertainty near 94% at the median. Whether that matters at the
network level depends entirely on whether the per-edge errors are independent. **Measured**, by
perturbing every edge's diameter by one voxel and re-solving the network:

| Perturbation | Network flow spread |
|---|---|
| **Independent** (random sign per edge, 24 draws, 1 s.d.) | **4.1%** |
| **Correlated** (every edge shifted the same way, half-range) | **95.3%** |

| Specimen | Independent | Correlated |
|---|---|---|
| WKY-A | 5.3% | 80.8% |
| WKY-B | 2.9% | 87.9% |
| WKY-C | 3.1% | 100.1% |
| SHR-A | 2.7% | 97.9% |
| SHR-B | 4.1% | 101.0% |
| SHR-C | 6.4% | 104.1% |

Independent error averages down by a factor of 23, exactly as the law of large numbers over roughly
5,000 edges predicts. Correlated error does not average down **at all**: essentially the full
per-edge uncertainty survives to the network total.

**The error in this pipeline is the correlated kind.** Every edge in a specimen is measured from one
mask, produced by one classifier at one threshold. A threshold shift moves every diameter the same
way, which is precisely the correlated perturbation. H1 §6.4 measured that directly: moving the
threshold from 0.85 to 0.95 moved every topological measure monotonically, in the same direction,
for all six specimens.

**Consequence.** Absolute network flow carries roughly **±95%** uncertainty. H1's topological group
effects were 27% to 40%. A quantity with 95% uncertainty cannot resolve a 40% difference, so
**absolute flow comparisons between cohorts are not answerable at current calibre precision**, and
no improvement to the solver changes that.

`STATUS — OUTSTANDING`

### S13. A within-specimen ratio cancels 86% of the correlated error

This is the constructive half of S12, and it is the finding that keeps H2 alive.

If the correlated component moves numerator and denominator together, a ratio taken inside one
specimen should largely cancel it. **Measured**, using the fraction of total flow carried by the top
decile of edges by calibre as a geometric stand-in for a thoroughfare channel, with the edge set
held fixed at baseline so that flow redistribution is measured rather than reclassification:

| Specimen | Shunt fraction | Correlated half-range |
|---|---|---|
| WKY-A | 0.2766 | 5.6% |
| WKY-B | 0.2884 | 14.1% |
| WKY-C | 0.2310 | 11.1% |
| SHR-A | 0.3199 | 16.8% |
| SHR-B | 0.2622 | 14.3% |
| SHR-C | 0.2776 | 17.3% |

**Mean 13.2%, against 95.3% for absolute flow: the ratio cancels 86% of the correlated error.**

This yields a concrete design rule, and it should govern how H2 is posed:

> **H2 must be expressed as ratios computed within a specimen, never as absolute flows compared
> between specimens.**

H2 §2.1 is already defined that way, as flow bypassing against flow penetrating glomus tissue, and
so is §2.3's hypoxic *fraction*. Both survive this constraint. §2.4's transit time is an absolute
quantity and does not, unless it is re-posed as a ratio against a within-specimen reference.

A 13.2% residual is not small, and it still has to be cleared by any claimed group difference. But
it is in the range where the 27% to 40% effects H1 measured could be resolved, which ±95% is not.

`STATUS — OUTSTANDING`

### S14. The S9 constriction inflated resistance on 12.3% of edges in the shipped flow output

Found by following the S2 retraction into the flow output that turned out to exist, and it puts a
measured number on how much S9 actually cost.

**One false start, recorded because the method matters.** The exported resistance first appeared to
disagree with Hagen–Poiseuille by a factor of ~55, with a ~15× spread between edges. That comparison
was wrong: it used the exported `viscosity` array, which is the **post-rheology** value in cP, where
the resistance was built from `PoiseuilleModel.calculate_viscosity(d)`, a different quantity in
different units. The apparent spread was just the two viscosities not tracking each other. A finding
that a solver disagrees with its own closed form deserves this level of checking before it is
written down as one.

**Measured, correctly.** Against the viscosity actually used, `calculate_integrated_resistance`
reduces to Hagen–Poiseuille exactly. Direct check at `d1 == d2`:

| L (µm) | d (µm) | R model | R closed form | ratio |
|---|---|---|---|---|
| 20 | 8 | 0.00647651 | 0.00647651 | 1.0000 |
| 50 | 4 | 0.81133 | 0.81133 | 1.0000 |
| 10 | 12 | 0.000328038 | 0.000328038 | 1.0000 |
| 100 | 6 | 0.164377 | 0.164377 | 1.0000 |

Across the shipped six-specimen output the median ratio is likewise exactly `1.0000`. **The
integrator is correct**, which corroborates S1's analytic result on real data rather than fixtures.

**What is left is the constriction.** The edges that do not match are the ones the S9 fabrication
narrowed:

| Specimen | Edges | Constricted | Share | Median inflation | Max |
|---|---|---|---|---|---|
| WKY-A | 4,512 | 653 | 14.5% | 12.66× | 36.8× |
| WKY-B | 3,932 | 426 | 10.8% | 8.62× | 36.8× |
| WKY-C | 6,699 | 901 | 13.4% | 12.30× | 36.8× |
| SHR-A | 6,815 | 846 | 12.4% | 11.75× | 36.8× |
| SHR-B | 8,077 | 1,026 | 12.7% | 12.54× | 36.8× |
| SHR-C | 4,865 | 494 | 10.2% | 10.03× | 36.8× |

**12.3% of edges carry a fabricated constriction that inflates their resistance by a median of
about 12× and by up to 36.8×.** The identical 36.8× ceiling across all six specimens is the
signature of a shared hard-coded ratio rather than anything measured, which is exactly S9's
complaint.

**Consequence.** The existing `resistance_network_vessels_flow.vtp` and `_perfusion.vti` for all six
specimens must be **regenerated, not read**. Nothing in the H1 report depends on them. Part 1 is
also unaffected: S12 and S13 build the conductance network from measured calibre and length
directly and never touch the exported resistance.

`STATUS — FIXED` at source by `1ee46a1`; the stale artefacts remain to be regenerated.

### S15. The noise floor, calibrated against the real threshold shift

S12 used one voxel as the perturbation because that is the scale at which a diameter difference
stops being physically resolved. That is the conservative bound, not the operative one. Since the
threshold is the dominant correlated term, the operative perturbation is however far the threshold
actually moves calibre, and the H1 sensitivity sweep already contains the answer.

**Measured**, median calibre per specimen across the three complete runs:

| Specimen | 0.85 | 0.90 | 0.95 | 0.85 → 0.90 | 0.90 → 0.95 |
|---|---|---|---|---|---|
| WKY-A | 10.462 | 8.345 | 7.464 | −2.117 | −0.881 |
| WKY-B | 9.010 | 8.343 | 7.460 | −0.667 | −0.883 |
| WKY-C | 8.345 | 7.905 | 6.462 | −0.441 | −1.443 |
| SHR-A | 8.343 | 7.464 | 6.369 | −0.879 | −1.095 |
| SHR-B | 8.345 | 7.798 | 6.371 | −0.547 | −1.427 |
| SHR-C | 8.343 | 7.464 | 5.868 | −0.879 | −1.596 |

Calibre falls monotonically with threshold in **6 of 6 specimens**. That shared direction is exactly
what makes the error correlated rather than independent, and so what stops it averaging down. Over
the clean 0.85 to 0.90 interval the mean shift is **0.922 µm, about half a voxel**, giving a
per-edge `δd/d` of 11.7% and an analytic `δR/R` of 46.7%.

**Measured** by re-solving the networks at that perturbation rather than scaling S12's numbers,
since d⁻⁴ is not linear:

| Perturbation | Independent | **Correlated** | **Within-specimen ratio** |
|---|---|---|---|
| One voxel, 1.866 µm (conservative bound) | 4.1% | 95.3% | 13.2% |
| **Measured threshold shift, 0.922 µm** | 2.2% | **45.3%** | **6.3%** |

The measured 45.3% sits close to the 46.7% the analytic `4·δd/d` predicts, which confirms the
propagation is near-linear at this scale even though the underlying law is not. The ratio cancels
**86% of the correlated error at both perturbation sizes**, so that cancellation is a property of
the ratio rather than an artefact of the size chosen.

**This is the finding that makes H2 worth attempting.** The operative noise floor is:

| Quantity | Floor | Against H1's 27% to 40% effects |
|---|---|---|
| Absolute network flow | ±45% | cannot resolve them |
| Within-specimen ratio | **±6.3%** | **can resolve them** |

A ratio measure has roughly a fourfold margin over the smallest effect H1 measured. Absolute flow
has none. S13's design rule is therefore not a caution but the difference between an answerable
question and an unanswerable one.

Two qualifications. The per-specimen shift is itself uneven, from 0.441 µm (WKY-C) to 2.117 µm
(WKY-A), so the correlated error is not identical across specimens and does not cancel perfectly in
a between-group comparison. And the group means differ, 1.075 µm for WKY against 0.768 µm for SHR,
which is the right shape to become a confound; with n = 3 it is noted, not established.

`STATUS — OUTSTANDING`

### S16. The three solvers build an identical and correct discretisation, under inverted names

This resolves S4's open question, and resolves it as no defect. It also cost two wrong readings
before it was right, both recorded.

**The three builders agree exactly.** `build_adr_matrix`
([perfusion.py:212](src/ImageLynx/haemodynamics/perfusion.py:212)), the closure inside
`solve_multi_species_perfusion` ([:393](src/ImageLynx/haemodynamics/perfusion.py:393)) and the
inline assembly in `solve_coupled_1d3d_perfusion`
([:618](src/ImageLynx/haemodynamics/perfusion.py:618)) unpack the dimensions the same way, use the
same coefficient formulas and lay out the same seven-point stencil. Comparing their outputs is
therefore meaningful, which is what S4 asked.

**The naming is inverted throughout, and the inversions cancel.** Two of them:

1. `grid.dims` is `(nz, ny, nx)`, as the constructor's own log records, but every builder unpacks it
   as `nx, ny, nz` and then reshapes as `(nz_dim, ny_dim, nx_dim)`, which is the reverse.
2. `D_x` is built from `(res[1] * res[2]) / res[0]`. With `res` in `(z, y, x)` that is
   `(y·x)/z`, the **z** coefficient, not the x one.

Because the reshape's last axis is the grid's z, and `D_x` is the z coefficient, the two errors
compose into the right answer. Nothing about that is guessable from the names.

**Measured**, on a deliberately non-cubic grid `(3, 7, 29)` with anisotropic spacing
`(20, 10, 5)` µm, both chosen so an isotropic or cubic case could not hide a fault:

| Direction | Matrix coupling | Correct value | Ratio |
|---|---|---|---|
| z | 3,750 | 3,750 | 1.0000 |
| y | 15,000 | 15,000 | 1.0000 |
| x | 60,000 | 60,000 | 1.0000 |

The centre cell has exactly six off-diagonal neighbours, each one unit step away. **The
discretisation is correct.**

**Two wrong readings, recorded because the method is the point.** The first was a suspected
index-ordering mismatch between `get_cell_index` (z fastest) and the C-order reshape (x fastest);
they do differ, and the stencil is still right, so the inference was wrong. The second reported a
16× coefficient mismatch, which was an error in the check rather than in the code:
`PerfusionGrid` reverses the `grid_resolution_xyz` tuple on the way in, so the expectation had been
computed with the spacing in the wrong order. Reading alone produced two false positives here and
execution produced the answer, which is why the assessment's method insists on the latter.

**The residual risk is real even though the code is right.** The arithmetic survives only because
two inversions cancel, so any future edit that corrects one name in isolation breaks the solver
silently, and an isotropic grid, which is the default at `(10, 10, 10)`, cannot detect it. This is
now locked by `test_adr_stencil_connects_physical_neighbours_with_correct_anisotropic_weights`,
verified by mutation: swapping `D_x` and `D_z` makes it fail with
`z-neighbour coupling is not the z diffusion coefficient`.

Renaming the axes to match reality would be a genuine readability improvement and is safe to do now
that the test exists. It is deliberately not done here, because it is a change to a solver in a
document whose job is to assess one.

`STATUS — OUTSTANDING` (no numerical defect; the naming remains a hazard, now guarded)

### S17. The constriction disable, verified end to end on real data

`1ee46a1` was verified by unit test at the configuration surface. This checks it where it matters,
by re-running the whole pipeline on WKY-A with the same frozen parameters as the H1 batch (ROI
160³, threshold 0.90) into a separate output directory, so the pre-fix artefacts survive as
evidence.

**Measured**, comparing the same specimen before and after:

| | Edges | Constricted | Max R inflation | Total flow |
|---|---|---|---|---|
| Before, `constrict_at_pericytes = True` | 4,512 | 653 (14.5%) | 36.79× | 1.4233 × 10¹² |
| **After, disabled** | 4,512 | **0 (0.0%)** | **1.00×** | 1.6292 × 10¹² |

Edge count is identical, so the morphometry is unchanged and only the resistance model moved, which
is what the change was supposed to do. Every edge now sits exactly on Hagen–Poiseuille.

**The fabricated constriction was suppressing total network flow by 14.5%.** That is the size of
the error S9 and S14 describe, expressed as the quantity H2 §2.1 would actually report.

**All six regenerated.** Every specimen now has zero constricted edges and a maximum
resistance ratio of exactly 1.000 against the closed form, with edge counts identical to the
originals:

| Specimen | Edges | Constricted | Flow before | Flow after | Change |
|---|---|---|---|---|---|
| WKY-A | 4,512 | 0 | 1.423 × 10¹² | 1.629 × 10¹² | +14.5% |
| WKY-B | 3,932 | 0 | 8.058 × 10¹¹ | 9.613 × 10¹¹ | +19.3% |
| WKY-C | 6,699 | 0 | 6.681 × 10¹¹ | 7.815 × 10¹¹ | +17.0% |
| SHR-A | 6,815 | 0 | 9.725 × 10¹¹ | 1.095 × 10¹² | +12.6% |
| SHR-B | 8,077 | 0 | 1.115 × 10¹² | 1.272 × 10¹² | +14.1% |
| SHR-C | 4,865 | 0 | 4.691 × 10¹¹ | 5.307 × 10¹¹ | +13.1% |

**The fabrication was differentially biased between cohorts**, and that is the part worth keeping.
It suppressed WKY flow by 16.9% on average against SHR by 13.3%, a 3.6 percentage-point gap in a
quantity H2 would compare between groups. A uniform bias would largely cancel in the ratios S13
recommends; a group-differential one does not. This is the failure mode H1 §6 exists to detect, and
it was present in the shipped flow output for all six specimens.

`STATUS — FIXED`, verified on all six.

### S18. Review notes on `resistance.py` and `rheology.py`

Opening of the Part 2 module review. `probability.py` and the non-ADR parts of `perfusion.py` are
not yet covered.

**`resistance.py` is sound in its core.** The Dirichlet partition
([resistance.py:199](src/ImageLynx/haemodynamics/resistance.py:199)) is the standard reduction, the
two-point effective resistance ([:100](src/ImageLynx/haemodynamics/resistance.py:100)) uses the
correct inject-and-ground construction, and parallel edges sum conductance correctly. Three smaller
points:

1. **Edges with missing or non-positive resistance are dropped silently**
   ([:34](src/ImageLynx/haemodynamics/resistance.py:34)), with no count and no warning. A dropped
   edge is an edge carrying no flow, which changes the network's topology without saying so.
   Upstream validation currently makes this unreachable, so it is a latent trap rather than a live
   defect, but it is the same silent-fallback shape H1 kept finding.
2. **`output_nodes` is mutated in place** ([:154](src/ImageLynx/haemodynamics/resistance.py:154))
   when a Robin ghost node exists, which modifies a list the caller owns.
3. **`1.0 / edge_resistance` is unguarded** ([:241](src/ImageLynx/haemodynamics/resistance.py:241)).
   Measured minimum resistance in the shipped output is 3.1 × 10⁻⁶, so it does not fire.

It also **confirms the S10 mechanism in code**: identical Dirichlet pressures are applied at every
selected terminal, so total flow scales with how many terminals were selected. That is exactly why
the 10.7× inlet-to-outlet asymmetry S10 measured matters.

**`rheology.py` carries one question worth settling before any haematocrit result is quoted.**
`calculate_pries_secomb_viscosity` ([rheology.py:41](src/ImageLynx/haemodynamics/rheology.py:41))
computes

```
mu_45 = 220.0 * exp(-1.3 * D) + 3.2 - 2.44 * exp(-0.06 * D**0.645)
```

The `220·e^(−1.3D)` term is the **in vitro** Pries et al. (1992) relation for glass tubes. The
in vivo relation, which the docstring says the function implements, uses `6·e^(−0.085D)` in that
position. The wall-layer correction `(D/(D−1.1))²` applied at
[:51](src/ImageLynx/haemodynamics/rheology.py:51) is from the in vivo law.

At D = 8 µm and H = 0.45 the two give μ₄₅ of about 1.27 against 4.30, roughly a **3.4× difference in
apparent viscosity**, and resistance is linear in viscosity.

**This is flagged for verification against the source, not asserted as an error.** Both relations
are real and published, and which one is appropriate is a modelling judgement rather than a coding
one. Two things make it lower priority than it first looks: the discrepancy is close to a uniform
multiplier, so S13's within-specimen ratios largely cancel it; and §2.2, the method that depends on
haematocrit most directly, is TH-blocked regardless. It should be settled before §2.2 is attempted.

`STATUS — OUTSTANDING`

### S19. The perfusion grid does not resolve the tissue-to-vessel distance

Completes the Part 2 module review, covering `perfusion.py` outside the ADR assembly that S16
handles. `probability.py` is **out of scope**, not merely unreviewed: every function in it serves
the constriction capability, and its only importers are `pericyte_mask` and `pericyte_comparison`,
both frozen by the S9 decision. Nothing on the carotid path imports it.

**Two positives first.** The Picard loop converges: across all regenerated runs there is not one
`hit max_iter` or non-convergence warning, so the non-linear solve is reaching its `1e-4` tolerance
inside 50 iterations on real networks rather than being truncated. And `map_vessels_to_grid`
point-samples each edge's voxels and accumulates length and surface area per cell, which is a
defensible discretisation of the 1D-to-3D coupling.

**The grid is too coarse for §2.3.** `grid_resolution_xyz` defaults to `(10, 10, 10)` µm, giving a
31³ grid of 29,791 cells over the ROI, of which 5,534 (18.6%) contain a vessel. So unperfused tissue
does exist for a gradient to form across, which is the first thing to check. The problem is the
scale of that gradient.

**Measured**, as the Euclidean distance from every background voxel to the nearest vessel voxel at
native resolution, which is the tissue-to-vessel distance H1 §1.5 defines:

| Specimen | Foreground | TVD p50 | p90 | p99 | max | Grid cells across p90 |
|---|---|---|---|---|---|---|
| WKY-A | 23.5% | 7.92 µm | 53.06 | 113.15 | 172.26 | 5.31 |
| WKY-B | 23.4% | 6.98 µm | 29.01 | 60.95 | 90.58 | 2.90 |
| WKY-C | 29.4% | 5.28 µm | 25.90 | 65.30 | 109.07 | 2.59 |

**The median tissue voxel sits 5.3 to 7.9 µm from a vessel, which is less than one grid cell.** For
half the tissue the oxygen gradient that decides whether it is hypoxic falls entirely inside a
single cell and is not represented at all. Only the p90 tail spans two to five cells, and even there
the resolution is marginal.

A hypoxic fraction computed on this grid would therefore be dominated by the discretisation rather
than by the physiology, which is precisely what §2.3 sets out to measure. **The ADR grid needs to be
several times finer than the median TVD**, which puts it at roughly 1.5 to 2 µm, essentially the
acquisition voxel of 1.866 µm. That is a 160³ grid of 4.1 million cells against the current 29,791,
a factor of 137. A sparse seven-point Laplacian at that size is tractable with the iterative path
`_solve_system_smart` already contains, but it is not free, and it should be measured before §2.3 is
attempted rather than discovered during it.

**Two smaller points.**

1. `map_vessels_to_grid` falls back to a hard-coded **5.0 µm** diameter when
   `assigned_diameter_um` and `fwhm_diameter_um` are both absent or non-positive
   ([perfusion.py:154](src/ImageLynx/haemodynamics/perfusion.py:154)), silently. That diameter sets
   the surface area driving transvascular flux, so a fabricated calibre would enter the oxygen
   budget the same way S5 describes for resistance. Flow defaults to `0.0` on the line above, which
   silently zeroes a source.
2. **The far field is bounded by the crop, not by anatomy.** Maximum TVD is 90 to 172 µm against an
   ROI half-width of 148 µm, so the most hypoxic tissue in each volume sits at a distance comparable
   to the box itself. Whatever the grid resolution, the deepest part of the gradient is a property
   of where the ROI was cut.

`STATUS — OUTSTANDING`

### S20. The boundary choice, not calibre, is the dominant error on a ratio

**This revises S13 and S15 downward, and it is the most consequential finding in Phase 2.**

S13 and S15 established that a within-specimen ratio cancels 86% of the correlated calibre error,
leaving 6.3%, and concluded that ratio measures therefore had a comfortable margin over H1's effect
sizes. That conclusion tested one error source and treated it as the only one. S10 had already shown
that the inlet and outlet nodes are chosen positionally, by an axis and a band width with no
anatomical basis for either, and that was never propagated into the ratio.

**Measured**, taking the same shunt fraction through the same solve while varying only the boundary
choice:

| Specimen | axis0 25% | axis1 25% | axis2 25% | axis0 10% | axis0 40% | Axis | Band | Total |
|---|---|---|---|---|---|---|---|---|
| WKY-A | 0.2766 | 0.2773 | 0.2745 | 0.3156 | 0.3165 | 1.0% | 13.2% | 14.4% |
| WKY-B | 0.2884 | 0.2876 | 0.3262 | 0.2792 | 0.3417 | 12.8% | 20.6% | 20.5% |
| WKY-C | 0.2310 | 0.1870 | 0.1675 | 0.2744 | 0.2844 | 32.6% | 20.3% | 51.1% |
| SHR-A | 0.3199 | 0.3216 | 0.2828 | 0.3157 | 0.3597 | 12.6% | 13.3% | 24.0% |
| SHR-B | 0.2622 | 0.2933 | 0.2444 | 0.3008 | 0.2338 | 18.4% | 25.2% | 25.1% |
| SHR-C | 0.2776 | 0.2968 | 0.2914 | 0.2791 | 0.3271 | 6.7% | 16.8% | 16.8% |

**Mean spread: axis 14.0%, band width 18.2%, combined 25.3%.** Against 6.3% from calibre, the
boundary choice moves the ratio **four times more**. Neither sub-term dominates, so fixing one alone
buys little. WKY-C moves 51.1%, which is larger than any group difference H1 reported.

**The revised noise floor**, combining the two independent sources in quadrature:

| Source | Effect on a within-specimen ratio |
|---|---|
| Correlated calibre error (S13, S15) | 6.3% |
| Boundary axis choice | 14.0% |
| Boundary band width | 18.2% |
| **Combined boundary arbitrariness** | **25.3%** |
| **Operative floor** | **≈ 26%** |

Against H1's 27% to 40% effects that is a margin of roughly **1.1× to 1.6×**, not the fourfold S15
claimed. A ratio measure is no longer comfortably above the floor; it is marginally above it.

**Why this is still better news than the calibre bound.** The calibre floor needs better imaging or
a sub-voxel estimator, neither of which is near. The boundary floor is a modelling choice and is
addressable in software today, three ways in increasing order of merit:

1. **Report the ensemble.** Solve across the boundary choices and report the ratio with its spread.
   This does not reduce the uncertainty but stops it being invisible, and it is available now.
2. **Fix the band width on a principle** rather than a default, which addresses the larger of the
   two sub-terms.
3. **Determine the vascular axis anatomically** rather than taking the acquisition axis, which is
   the only route that genuinely removes the 14.0% rather than quantifying it.

**Consequence for §2.1.** It remains the best-placed method, but its readiness claim changes: it is
not merely waiting on TH data. Its boundary conditions need to be settled first, or every number it
produces carries a ±25% band set by an arbitrary choice.

`STATUS — OUTSTANDING` (the highest-priority H2 item that is not TH-gated)

### A suspected frame transpose, checked and refuted

Worth recording because it would have invalidated every figure in the H1 report. In `_nodes.vtp` the
anisotropic axis appears at coordinate index 0, while `_mask.vti` declares its anisotropic spacing at
index 2, which suggested geometry and mask were written in transposed frames.

**Measured:** sampling the mask at every raw skeleton point, which must by construction be
foreground, gives **100.0%** foreground as stored against 24.4% under the transposed reading. The
frames agree, and the ParaView README's claim that the files overlay without a transform is correct.

---

### S21. The boundary sensitivity is a property of the rule, and a better rule exists

S20 left the boundary choice as the dominant error and T0.2 as the highest-value item not gated
on new data. It is now settled, by building the alternative and measuring it rather than by
argument.

**The rule.** A vessel supplying this region has to cross one of its faces. A dead end in the
middle of the volume cannot be a pressure inlet whatever its coordinate, and S10 measured 86% of
degree-1 nodes to be exactly that. `select_boundary_terminal_nodes_by_face` therefore admits only
terminals within a tolerance of a region face, and raises rather than falling back when a face
carries none.

**Measured**, as the spread of the shunt ratio per specimen while each rule's free parameters
move over their plausible range:

| Rule | Parameters varied | Ratio spread | Failed solves |
|---|---|---|---|
| band | axis 0/1/2 x width 10/25/40% | **118.8%** | 0 of 54 |
| face | axis 0/1/2 x tolerance 1/2/4 voxels | **43.1%** | 6 of 54 |
| band, axis fixed at 1 | width 10/25/40% | **75.8%** | 0 of 18 |
| face, axis fixed at 1 | tolerance 1/2/4 voxels | **13.3%** | 0 of 18 |

**A 5.7-fold reduction at fixed axis**, and it comes from the parameter rather than the axis. The
band width has no principled value, so its whole range is live. The face tolerance is anchored to
the voxel size: one voxel means "on the face", and the other values exist only to show the answer
does not depend on it.

**Axis 1 is a selection, not a preference.** It is the only axis with terminals on both faces in
all six specimens. Axis 0 has no outlet terminal in SHR-A; axis 2 has no inlet terminal in SHR-C.
That is a property of these graphs rather than a general rule, and it is why the face rule raises
on an empty face instead of inventing boundaries.

**A comparison that pointed the wrong way first.** Holding each rule's second parameter at its
default and varying only the axis gives 28.4% for the band rule against 31.5% for the face rule,
which reads as the face rule being worse. That comparison flatters the band rule by fixing the
parameter that damages it. Both parameters have to move, and when they do the ordering reverses
by a factor of nearly three. This is recorded because the first measurement was taken that way
and was briefly believed.

**What this does not do.** It does not make the boundary anatomical. There is still no anatomical
inlet inside a mid-organ cube, and the axis choice is still a choice. It reduces the residual to
13.3%, which is below the ~26% operative floor S20 reported and below the 27 to 40% effects H1
measures, so a within-specimen ratio is no longer boundary-dominated.

Reproduced by `examples/cb_h2_boundary_selection.py`.

`STATUS — T0.2 RESOLVED.`

### S22. The viscosity law was a hybrid of two, and correcting it exposed a third question

T1.2 asked whether `calculate_pries_secomb_viscosity` should use the in vitro or in vivo
Pries-Secomb relation. The answer is in vivo, since H2 models perfusion of living tissue where
the endothelial surface layer is present. But the function was not using either.

**It was a hybrid.** The `mu_45` base was the in vitro relation, `220·e^(−1.3D)`, and the
wall-layer correction `(D/(D−1.1))²` applied on top of it is from the in vivo law. That
combination is not a version of either relation.

**And the in vivo wall factor appears twice, not once.** The published form is

    mu_rel = (1 + (mu_45 − 1) · f(H, C) · W) · W,   W = (D / (D − 1.1))²

with the factor scaling the haematocrit term inside the bracket and applied again outside.
Applying it once understates apparent viscosity by 1.26× at 8 µm and 2.2× at 3 µm, which is the
calibre range every vessel in this study occupies. This was found by checking the first
correction against the published relation rather than by the tests, which passed either way.

**Measured effect at the study's median calibre**, against the hybrid it replaces:

| D (µm) | hybrid | in vitro | in vivo | in vivo / hybrid |
|---|---|---|---|---|
| 7.46 | 2.070 | 1.505 | 8.02 | 3.9× |
| 8.35 | 2.033 | 1.532 | 7.30 | 3.6× |

Resistance is linear in viscosity, so absolute flows fall by about an order of magnitude once
the haematocrit coupling settles. **The §2.1 and §2.2 conclusions are unchanged**: the shunt
index moves from WKY 0.924 / SHR 1.016 to WKY 0.909 / SHR 1.056, and the haematocrit ratio from
0.90× to 0.92×, both still overlapping and both still pointing the same way. This is S13's
lesson again in a second variable: absolute quantities move by multiples, within-specimen
ratios do not.

**The third question, which is open.** Under the in vitro law this pipeline's test bifurcation
sends 84% of flow down the wide branch, which is then also the faster of the two, and it skims
red cells as expected. Under the in vivo law the narrow branch is penalised harder, the split
evens to 64/36, and 36% of flow through a quarter of the area makes the **narrow** branch the
faster one. The skimming model then concentrates red cells there, inverting the classic
picture.

The call site pairs each flow with its own diameter correctly, so this is the model's own
behaviour: it keys on velocity, where the Pries phase-separation law is normally posed in
fractional blood flow with a diameter-dependent threshold. At a near-even split the two
parameterisations can disagree in direction. That matters directly for §2.2, whose whole
subject is where red cells end up. It is recorded rather than fixed, because it is a separate
question from the viscosity law and changing the skimming model is not what T1.2 asked.

`STATUS — T1.2 RESOLVED; a new open question on the skimming model recorded above.`

### S23. Transit time posed as a ratio separates the cohorts; the axis rename changed no number

**T1.3.** §2.4 asks for the transit time from the arterial inlet to the distal ends of the
capillaries inside the TH boundaries. As an absolute that sits under the ±45% calibre floor of
S15, and S22 has just moved apparent viscosity by a factor of three or four without shifting any
ratio. The pipeline's pressure, viscosity and length units are also not reconciled to one system,
so the magnitude is in arbitrary units. Posed as the ratio of transit time to TH-penetrating
edges against transit time to bypassing edges, all of that divides out.

Transit time per edge is lumen volume over flow, accumulated along the **solved flow
directions** rather than along adjacency: an edge carrying blood away from a node cannot deliver
blood to it. Dijkstra rather than a topological pass, because flow directions come from a
numerical solve and can contain a small cycle.

| Specimen | penetrating | bypassing | ratio |
|---|---|---|---|
| WKY-A | 2.381e7 | 1.889e7 | 1.261 |
| WKY-B | 1.715e7 | 1.547e7 | 1.109 |
| WKY-C | 2.811e7 | 2.516e7 | 1.117 |
| SHR-A | 1.245e7 | 1.518e7 | 0.820 |
| SHR-B | 1.463e7 | 1.674e7 | 0.874 |
| SHR-C | 1.382e7 | 2.017e7 | 0.685 |

**The cohorts separate without overlap**, WKY 1.109 to 1.261 against SHR 0.685 to 0.874, giving
the design floor of p = 0.10. The direction is the opposite of what §2.4 anticipates. It expects
sluggish transit to the sensors in SHR; blood reaches the SHR clusters in about two thirds of the
time it takes to reach the surrounding tissue, where in WKY it takes about a fifth longer.

That is consistent with §2.1, which finds no bypass, and with §1.3, which finds a smaller and
more densely vascularised parenchyma. Whether it means the sensors are well perfused or merely
that a smaller cluster is closer to its supply is not answerable from a ratio, and no claim
beyond the measurement is made.

**T2.4.** The perfusion grid axes are renamed at all three assembly sites. S16 established that
the naming was inverted twice and that the inversions cancelled: `grid.dims` is `(nz, ny, nx)`
but was unpacked as `nx, ny, nz`, and `D_x` was built from `(res[1]·res[2])/res[0]`, which with
`res` in `(z, y, x)` is the z coefficient. The rename fixes the unpacking, the coefficient names,
the index-array shape and which coefficient each stencil direction uses, all together, because
fixing any one alone would change the arithmetic.

Verified rather than asserted: the assembled matrix, its indices, the flow and source vectors
were hashed on a deliberately non-cubic anisotropic grid `(3, 7, 29)` at `(20, 10, 5)` µm before
and after. **Identical SHA256.** One further reversed unpack was found in
`solve_multi_species_perfusion` that the snapshot could not see, since it exercises only
`build_adr_matrix`; it is fixed too.

`STATUS — T1.3 and T2.4 RESOLVED.`

### S24. The perfusion solve never converged, and the grid was not what was blocking §2.3

T1.1 asked for the perfusion grid to be refined from 10 µm to roughly 1.5 to 2 µm, a factor of
137 in cells, and for the iterative solver to be benchmarked at that size before committing. The
benchmark was run and found two things ahead of the grid.

**The conjugate gradient solve has never converged.** `solve_perfusion_steady_state` preconditions
CG with an incomplete-LU factorisation. CG assumes its preconditioner is symmetric positive
definite; `spilu` is a general-purpose approximation and guarantees neither, and given one that is
neither, CG does not converge slowly, it diverges.

**Measured** on WKY-C at the production 10 µm grid, with real solved flows:

| Preconditioner | Converged | Relative residual | Time |
|---|---|---|---|
| ILU (`spilu`), as shipped | no, `info=1000` | **19.06** | 5.81 s |
| none | yes | 8.9e-7 | 0.05 s |
| Jacobi, the inverse diagonal | yes | **8.8e-7** | **0.05 s** |

The initial residual is 1 by construction, so 19 is divergence rather than slow progress. The
diagonal of the assembled matrix is a sum of face conductances plus a positive regulariser plus a
non-negative washout term, so it is strictly positive and its inverse is SPD by construction.
Substituting it makes the whole steady-state solve **39 times faster**, 97.7 s to 2.5 s, and
converge.

S19 reported that the Picard loop converges and that no run showed a non-convergence warning.
That is about the **outer** loop. The inner CG emits a differently worded message at every Picard
step, and every run has been emitting it.

**But the field is zero either way, and that is the real block on §2.3.** With the solve fixed,
PO2 comes out at 0 everywhere and Picard now reports hitting its iteration cap. Checking the
balance the equations are being asked to satisfy:

| Quantity | Value |
|---|---|
| Total oxygen source, `sum(s_incoming)` | 66.5 |
| Total metabolic sink, `M_max · V_cell · n_cells` | 1.49e6 |
| **Sink / source** | **2.2e4×** |

The tissue is being asked to consume twenty-two thousand times the oxygen the blood delivers, so
PO2 → 0 everywhere is the correct answer to the system as posed. The cause is the unit
incommensurability T1.3 already had to work around: flow leaves the flow solve in the units of
ΔP/R with mmHg, cP and µm mixed, which is not µm³/s, while the sink is mmol/L/s times µm³. The
two sides of the balance are in different unit systems.

**A hypoxic fraction is therefore not computable at any grid resolution**, and refining the grid
137-fold would have produced a zero field 137 times more finely. §2.3's remaining blocker is the
unit reconciliation, not the discretisation.

**The benchmark answer, for when it matters.** With the preconditioner fixed, on WKY-C:

| Resolution | Cells | Build | Solve | Peak memory |
|---|---|---|---|---|
| 10 µm | 29,791 | 0.31 s | 2.7 s | 400 MB |
| 6 µm | 132,651 | 0.36 s | 19.8 s | 491 MB |
| 4 µm | 438,976 | 0.99 s | 119.9 s | 762 MB |

Time scales at roughly N^1.6, as CG iteration count grows with problem size on a Laplacian.
Extrapolating to native 1.866 µm resolution, 4.1 M cells, gives about **70 minutes per specimen
and 4 to 5 GB**, so around seven hours for the cohort on this machine. That is affordable. These
timings are an upper bound, since Picard currently runs its full 50 iterations against a pinned
zero field and would stop earlier against a real one.

`STATUS — T1.1 BENCHMARKED; the CG preconditioner FIXED; §2.3 now blocked on unit reconciliation.`

## Effect on the four H2 methods

| Method | TH gate | Physics | Noise floor | Also needs |
|---|---|---|---|---|
| §2.1 Functional shunting and glomus bypass | **blocked** | sound (S1, S16, S17) | ≈26%, boundary-dominated (S20) | boundary conditions settled |
| §2.2 Spatial haematocrit profiling | **blocked** | validated (S1 Part 3) | as §2.1 if posed as a ratio | Pries–Secomb question settled (S18) |
| §2.3 Glomus-specific 3D hypoxic fraction | **blocked** | validated (S1 Parts 2, 4); ADR correct (S16); Picard converges (S19) | as §2.1 | **grid 137× finer** (S19) |
| §2.4 Oxygen depletion and transit time | **blocked** | validated (S1 Parts 4, 5) | ±45%, it is an absolute quantity | re-posing as a ratio (S13, S20) |

All four are blocked, and all four for the same reason: there is no TH channel. H1's Stage 1
measured the probability volume as `(435, 2, 456, 507)` with exactly two classes, vessel and
background. Every H2 method requires the glomus mask as a spatial landmark, so none is currently
computable in the form its definition specifies.

**This is a data decision, not a code defect**, and it is the same outstanding decision that blocks
H1 §1.3 and §1.5.

**What is computable today without TH**, and worth having before TH data arrives:

| Available now | Requires |
|---|---|
| Network-wide flow, pressure and WSS distributions | S7 resolved (a defensible inlet/outlet choice) |
| Between-cohort comparison of total network resistance | S6 quantified as a noise floor |
| Haematocrit distribution and skimming behaviour | as above |
| Bulk tissue PO₂ field and bulk hypoxic fraction | uniform `M_max`, so not glomus-specific |
| Shunt/perfusion ratio against a **geometric** proxy for glomus regions | an agreed proxy, clearly labelled as not TH |

The last row is the honest partial: a shunt analysis against a geometric proxy tests the machinery
and produces a real number, but it must not be presented as §2.1, which is defined against the TH
stain.

---

## Open ambiguities

Recorded rather than resolved, because each needs either a measurement in Phase 2 or a decision.

1. ~~**Are the three solvers solving the same discretisation?**~~ **RESOLVED by S16.** Identical,
   and correct, though the axis naming is inverted throughout and now guarded by a test.
2. ~~**Do resistance errors average down?**~~ **RESOLVED by S12.** Independent error does, by 23×.
   Correlated error does not at all, and this pipeline's error is correlated. S13 gives the way
   round it.
3. ~~**How much of each network is stranded by `"caged"` boundaries?**~~ **RESOLVED by S10 and
   S11.** Roughly half of terminals are stranded, but the networks stay singly connected, so the
   solve remains well posed. The live concern moved to the 10.7× inlet-to-outlet asymmetry.
4. **Is `C_arterial = 0.13` mmol/L consistent with `po2_arterial_mmHg = 100.0`?** Two arterial
   boundary specifications coexist in `PerfusionConfig`
   ([:373](examples/carotid_image_to_model.py:373), [:388](examples/carotid_image_to_model.py:388)),
   consumed by different solvers. Whether they describe the same blood is unverified.
5. ~~**Is a 10 µm perfusion grid adequate?**~~ **RESOLVED by S19: no.** The median tissue voxel is
   5.3 to 7.9 µm from a vessel, less than one cell, so for half the tissue the gradient is not
   represented at all. Needs roughly 1.5 to 2 µm, a factor of 137 more cells.
6. ~~**Does H1's threshold sensitivity propagate?**~~ **RESOLVED by S15.** Calibre moves 0.922 µm
   over the clean interval, monotonically in 6 of 6 specimens, which sets the 45.3% floor.

---

## What Phase 2 has measured, and what remains

Done, in Part 1 above:

1. ~~Per-edge resistance error propagation through a real network solve~~ **done, S12 and S13.**
2. ~~Terminal-node census per ROI face, per specimen~~ **done, S10.**

Remaining, in priority order:

3. ~~Regenerate the six-specimen flow and perfusion output~~ **WKY-A done and verified (S17);
   the other five are running.** Constricted edges fall from 653 to 0 and total flow rises 14.5%.
4. ~~Discretisation consistency across the three solvers~~ **done, S16.**
5. ~~Calibre sensitivity across the threshold sweep~~ **done, S15.** 0.922 µm over the clean
   interval, giving the operative noise floor.
6. ~~Stage-by-stage review~~ **done, S18 and S19.** `resistance.py`, `rheology.py` and
   `perfusion.py` are covered. `probability.py`, the pericyte modules and the constriction path are
   **out of scope** by the S9 decision, not merely unreviewed.

7. **Settle the Pries-Secomb in vitro against in vivo question** (S18) before §2.2 is attempted.
   Worth about 3.4× in apparent viscosity.

---

## Phase 3: Per-method verdicts and the plan of attack

### The one-paragraph answer

**H2 is not blocked by its physics.** The solvers pass fifty-two analytic benchmarks against
closed-form solutions (S1), the ADR discretisation is exactly correct including anisotropically
(S16), the Picard loop converges on real networks (S19), and the resistance integrator reduces to
Hagen–Poiseuille to 1.0000 (S14). What blocks H2 is, in order: **no TH channel**, which is a data
decision and stops all four methods dead; **arbitrary boundary conditions**, which put a ±25% band
on every ratio the pipeline can produce (S20); and **calibre precision**, which puts ±45% on any
absolute flow (S15). The first is not a code problem, the second is fixable in software today, and
the third is bounded by imaging and by H1's outstanding segmentation work.

### Consolidated plan of attack

**Tier 0. Blocking. No H2 result is valid until these land.**

| | Item | Blocks | Findings |
|---|---|---|---|
| T0.1 | ~~**Acquire and segment the TH channel.**~~ **Done.** Six TH probability maps from a two-class classifier; H1 §1.3 and §1.5 report on them. | all four methods | S2 (H1 Stage 1) |
| T0.2 | ~~**Settle the boundary conditions.**~~ **Done.** `select_boundary_terminal_nodes_by_face` admits only terminals crossing a region face, cutting the residual ratio spread from 75.8% to 13.3% at fixed axis. | §2.1, §2.4, and any flow-derived quantity | S7, S10, S20, **S21** |

T0.2 is the highest-value item in this document that is **not** gated on new data. Three routes, in
increasing order of merit: report the ensemble across boundary choices so the uncertainty is visible
rather than hidden; fix the band width on a principle, which addresses the larger sub-term; or
determine the vascular axis anatomically, the only route that removes the 14.0% rather than
quantifying it.

**Tier 1. Required before a given method produces a meaningful number.**

| | Item | For | Findings |
|---|---|---|---|
| T1.1 | ~~Refine the perfusion grid, benchmarking the solver first.~~ **Benchmarked.** Native resolution costs about 70 min per specimen and 4 to 5 GB, which is affordable. Deferred: the CG preconditioner was breaking the solve, and with it fixed the field is zero at any resolution for the reason below. | §2.3 | S19, **S24** |
| T1.5 | **New, and now the only block on §2.3.** Reconcile the units. Flow leaves the flow solve in ΔP/R units, not µm³/s, while the metabolic sink is in mmol/L/s times µm³; the sink exceeds the source by 2.2e4×. | §2.3, and the absolute scale of §2.4 | **S24** |
| T1.2 | ~~Settle whether `calculate_pries_secomb_viscosity` should use the in vitro or in vivo relation.~~ **Done.** In vivo, and the function was a hybrid of both with the wall factor applied once instead of twice. §2.1 and §2.2 conclusions unchanged. | §2.2 | S18, **S22** |
| T1.3 | ~~Re-pose transit time as a within-specimen ratio.~~ **Done.** Ratio of transit time to penetrating against bypassing edges, along solved flow directions. Cohorts separate without overlap. | §2.4 | S13, S15, S20, **S23** |
| T1.4 | ~~Regenerate the flow and perfusion artefacts without the fabricated constriction.~~ **Done**, all six. | all | S17 |

**Tier 2. Precision and safety. None blocks a result; each is a trap already sprung once.**

| | Item | Findings |
|---|---|---|
| T2.1 | ~~Assert on `diameter_provenance_counts`.~~ **Done.** `check_diameter_provenance` refuses an `edt_radius` run carrying any synthetic calibre and reports the fabricated share whatever the mode. `fwhm_radius` stays exempt for the reason the older guard gives. | S5 |
| T2.2 | ~~Remove the silent 5.0 µm diameter default in `map_vessels_to_grid`.~~ **Done.** Missing or non-positive calibre now raises, naming how many edges. `default_diameter_um` makes the substitution available but deliberate. | S19 |
| T2.3 | ~~Count and report edges dropped from the conductance matrix.~~ **Done.** Counted separately by cause, returned through an opt-in `report`, and logged as a warning regardless. Return arity unchanged, so the ten existing call sites are untouched. | S18 |
| T2.4 | ~~Rename the perfusion grid axes to match reality.~~ **Done.** All three assembly sites, verified by identical matrix hash on a non-cubic anisotropic grid. | S16, **S23** |
| T2.5 | Improve calibre precision. Gated on H1's outstanding perivascular labelling rather than on anything here. | S6, S15 |

**Tier 3. Experimental design, inherited from H1 and unchanged by anything in this document.**

| | Item |
|---|---|
| T3.1 | n = 3 per group. The exact two-sided permutation p cannot fall below 2/C(6,3) = 0.10, whatever the effect size. |
| T3.2 | Each ROI is 0.0266 mm³, roughly a fortieth of a cubic millimetre, centred on tissue signal and so sampling mid-organ where the network is denser than the whole. |
| T3.3 | Reconcile the two conflicting definitions of H2 in the source documents. |

### Coverage

**Assessed:** `poiseuille.py`, `resistance.py`, `rheology.py`, `perfusion.py`, the boundary
selection in `graph/boundaries.py`, the analytic benchmark suite, and the six-specimen flow,
rheology and perfusion output.

**Out of scope by the S9 decision, not merely unreviewed:** `pericyte_mask.py`,
`pericyte_comparison.py`, `probability.py`, and the constriction path through `PoiseuilleModel`.
All remain live for `examples/resistance_network_pipeline.py`, which owns the capability.

**Not assessed:** the multi-species solver's Bohr–Haldane and Henderson–Hasselbalch coupling beyond
its analytic tests, and `automated.py`, which belongs to the H1 measurement chain rather than to H2.

### Cross-cutting themes

Four patterns recur, and each has bitten more than once.

1. **Silent fallback to a fabricated constant.** Four separate instances: per-edge synthetic
   diameter when EDT is missing (S5), the constriction ratio multiplied onto measured calibre (S9),
   edges dropped from the conductance matrix (S18), and the 5.0 µm default in the perfusion source
   term (S19). In every case the fabricated value is indistinguishable downstream from a measured
   one. This is the same theme the H1 assessment found, in different code.

2. **A positional proxy standing in for anatomy.** The inlet and outlet selection (S7, S10), and the
   ROI placement it inherits. This is now measured as **the single largest error source in H2**
   (S20), larger than the calibre problem that the assessment was planned around.

3. **Names that invert their meaning.** The perfusion grid calls `nz` what is actually `nx`, and
   `D_x` what is actually the z coefficient (S16). The code is correct only because two inversions
   cancel, which no reader would infer from the names.

4. **Reading produced false positives; execution produced the answers.** The S2 retraction, the
   suspected index-ordering bug, the 16× coefficient mismatch, and the suspected frame transpose
   were all wrong, and all four were resolved by running something. Set against S9, S14, S19 and
   S20, which are real and were also found by running something. The method stated in this
   document's header is not a formality.

---

## Provenance

All three phases authored 2026-08-15 on `cb_pipeline_improvements_sweep`. Measured figures are
reproducible from the six-specimen artefact set in `examples/outputs/cb_h1_paraview/`, the
regenerated flow output in `examples/outputs/cb_h2_regen/`, and three scripts:

```bash
venv/bin/python examples/cb_h2_error_propagation.py                      # S10-S13, S20
venv/bin/python examples/cb_h2_error_propagation.py --perturbation-um 0.922   # S15
venv/bin/python examples/cb_h2_threshold_calibre.py                      # S15
venv/bin/python -m pytest tests/test_haemodynamics_analytical.py         # S1
```

**Claims withdrawn or revised by later measurement**, kept visible rather than edited away:

| Claim | Fate |
|---|---|
| The stack has never run on real data (S2) | **retracted**; it had run on all six specimens |
| EDT quantisation forms a coarse comb (S6) | refuted; 823 distinct values, median gap 0.0023 µm |
| `hemodynamics` is an accidental duplicate (S8) | refuted; a deliberate deprecation shim |
| Most terminals are crop-severed vessels (S7) | refuted; about 86% are interior |
| A ratio has a fourfold margin over H1 effects (S15) | **revised to 1.1× to 1.6×** by S20 |
| The exported resistance disagrees with Poiseuille (S14 draft) | withdrawn; compared against the wrong viscosity |
| The ADR index ordering is mismatched (S16 draft) | withdrawn; the inversions cancel |
| The ADR coefficients are swapped by 16× (S16 draft) | withdrawn; error was in the check |
| Geometry and mask are in transposed frames | refuted; 100.0% foreground as stored |
