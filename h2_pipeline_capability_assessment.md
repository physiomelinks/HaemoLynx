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
> **Assessment date:** 2026-08-15
> **Branch:** `cb_pipeline_improvements_sweep` @ `c3c236c`
> **Test data:** the six-specimen H1 artefact set in `examples/outputs/cb_h1_paraview/`
> (34,900 edges, frozen threshold 0.90, matched 0.0266 mm³ ROI per specimen)
>
> **Method:** code inspection plus direct numerical execution of the real pipeline functions on the
> real six-specimen artefacts. Figures labelled **measured** are empirical, not inferred. Findings
> labelled **inspected** come from reading the code and are not backed by execution; they are stated
> separately because that distinction is what makes the document auditable.

---

## Document status

**Phase 1 of 3 complete.** This document is being built in phases so that the headline verdict is
available before the full stage-by-stage review lands.

| Phase | Scope | State |
|---|---|---|
| 1 | Survey, call-graph trace, benchmark execution, headline verdict | **complete** |
| 2 | Stage-by-stage review, Parts 1 to 4 | **Part 1 complete** (S10–S14); Parts 2 to 4 not started |
| 3 | Per-method verdicts and tiered plan of attack | not started |

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

**Consequence.** H2 is not blocked by the perfusion code. It is blocked by the TH channel, which is
a data-availability decision, and it is bounded by calibre precision, which is an imaging and
segmentation problem. Effort spent hardening the solvers before those two are addressed is effort
spent on the part of the chain that is already working.

**Updated after Phase 2, Part 1.** S6's bound has been resolved into something more useful than a
warning. Independent calibre error averages down 23-fold across a real network solve and is
negligible at 4.1%. Correlated error, which is what a shared threshold and classifier produce, does
not average down at all: **absolute network flow carries ±95% uncertainty** (S12). A within-specimen
ratio cancels 86% of that, bringing it to 13.2% (S13). The practical rule that follows governs how
H2 should be posed:

> **Express H2 as ratios computed within a specimen, never as absolute flows compared between
> specimens.** §2.1 and §2.3 already satisfy this. §2.4 does not, and needs re-posing.

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
the three solvers are not three treatments of one problem. Phase 2 measures this.

`STATUS — OUTSTANDING`

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

### A suspected frame transpose, checked and refuted

Worth recording because it would have invalidated every figure in the H1 report. In `_nodes.vtp` the
anisotropic axis appears at coordinate index 0, while `_mask.vti` declares its anisotropic spacing at
index 2, which suggested geometry and mask were written in transposed frames.

**Measured:** sampling the mask at every raw skeleton point, which must by construction be
foreground, gives **100.0%** foreground as stored against 24.4% under the transposed reading. The
frames agree, and the ParaView README's claim that the files overlay without a transform is correct.

---

## Effect on the four H2 methods

| Method | TH gate | Physics gate | Survives S12/S13? | Net |
|---|---|---|---|---|
| §2.1 Functional shunting and glomus bypass | **blocked** | usable, bounded by S6, S10 | **yes**, already a within-specimen ratio | blocked on TH only |
| §2.2 Spatial haematocrit profiling | **blocked** | validated (S1 Part 3), bounded by S6 | **yes** if posed as a distribution or ratio | blocked on TH only |
| §2.3 Glomus-specific 3D hypoxic fraction | **blocked** | validated (S1 Parts 2, 4), bounded by S6 | **yes**, a fraction by definition | blocked on TH only |
| §2.4 Oxygen depletion and transit time | **blocked** | validated (S1 Parts 4, 5), bounded by S6, S10 | **no**, absolute unless re-posed | blocked, and needs redefinition |

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

1. **Are the three solvers solving the same discretisation?** (from S4) If `build_adr_matrix` and
   the coupled solvers' internal assembly differ, comparing their outputs is meaningless.
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
5. **Is a 10 µm perfusion grid adequate?** `grid_resolution_xyz = (10.0, 10.0, 10.0)` against a
   median vessel calibre of 7.9 µm means the grid cell is larger than the vessel. The Krogh-type
   diffusion geometry H2 §2.3 depends on is not resolved at that spacing.
6. **Does H1's threshold sensitivity propagate?** §6.4 measured topology across thresholds 0.85,
   0.90 and 0.95. Calibre was not tracked across that sweep, and through d⁻⁴ a small calibre shift
   is a large resistance shift.

---

## What Phase 2 has measured, and what remains

Done, in Part 1 above:

1. ~~Per-edge resistance error propagation through a real network solve~~ **done, S12 and S13.**
2. ~~Terminal-node census per ROI face, per specimen~~ **done, S10.**

Remaining, in priority order:

3. **Regenerate the six-specimen flow and perfusion output** (S2, S14). The existing files were
   produced with `constrict_at_pericytes = True`, so 12.3% of edges carry an inflated resistance.
   The solve itself is sound, so this is a re-run rather than a repair.
4. **Discretisation consistency across the three solvers** (S4).
5. **Calibre sensitivity across the 0.85 / 0.90 / 0.95 threshold sweep** (open ambiguity 6). Now
   sharper than when it was written: S12 makes the threshold the dominant correlated error term, so
   its effect on calibre sets the size of the noise floor.
6. Stage-by-stage review of `resistance.py`, `rheology.py`, `probability.py` and `perfusion.py`, in
   the H1 assessment's per-stage format. The pericyte and constriction modules are **excluded** by
   the decision recorded in S9.

---

## Provenance

Phase 1 authored 2026-08-15 against `cb_pipeline_improvements_sweep` @ `c3c236c`. Measured figures
are reproducible from the six-specimen artefact set in `examples/outputs/cb_h1_paraview/` and the
test suite invocation quoted in S1. Two priors held by the assessment plan were refuted by
measurement and are recorded as refuted rather than removed: the EDT quantisation comb (S6) and the
duplicate US-spelled package (S8).
