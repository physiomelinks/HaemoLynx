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
| 2 | Stage-by-stage review, Parts 1 to 4 | not started |
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

2. **None of it has ever been run on real carotid body data** (S2). The H1 batch driver contains no
   reference to haemodynamics, resistance, perfusion, or flow. Every figure the stack has ever
   produced came from synthetic test fixtures.

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

### S2. The perfusion stack has never been executed on real specimen data

**Measured.** `examples/cb_h1_batch.py` contains zero matches for `haemodynamics`, `resistance`,
`perfusion`, or `flow`. The six-specimen H1 run stopped at morphometry.

Every number the haemodynamics stack has ever produced came from synthetic fixtures. The first run
on a real extracted network will be the first time the solvers meet a graph with thousands of edges,
real connectivity, terminal reconnection artefacts, and cut boundaries.

This is not a defect. It is a statement about how much is unknown, and it is the reason S1 must not
be read as "the stack is ready".

`STATUS — OUTSTANDING`

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

`STATUS — OUTSTANDING`

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

---

## Effect on the four H2 methods

| Method | TH gate | Physics gate | Net |
|---|---|---|---|
| §2.1 Functional shunting and glomus bypass | **blocked** | usable, bounded by S6, S7 | blocked |
| §2.2 Spatial haematocrit profiling | **blocked** | validated (S1 Part 3), bounded by S6 | blocked |
| §2.3 Glomus-specific 3D hypoxic fraction | **blocked** | validated (S1 Parts 2, 4), bounded by S6 | blocked |
| §2.4 Oxygen depletion and transit time | **blocked** | validated (S1 Parts 4, 5), bounded by S6, S7 | blocked |

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
2. **Do resistance errors average down?** (from S6) The highest-value open question in this
   document. Requires per-edge error propagation through an actual network solve.
3. **How much of each network is stranded by `"caged"` boundaries?** (from S7) Needs the
   terminal-node census per face.
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

## What Phase 2 will measure

In priority order, by the value of the answer rather than the ease of getting it:

1. **Per-edge resistance error propagation through a real network solve** (open ambiguity 2, S6).
   Whether correlated calibre error averages down or survives to the network level decides whether
   H2 is answerable at all, independently of TH.
2. **Terminal-node census per ROI face, per specimen** (S7). Cheap, and it either clears the
   boundary-condition concern or establishes it as a between-group confound.
3. **First execution of the flow solve on real specimen data** (S2). Everything else is prediction
   until this runs.
4. **Discretisation consistency across the three solvers** (S4).
5. **Calibre sensitivity across the 0.85 / 0.90 / 0.95 threshold sweep** (open ambiguity 6).
6. Stage-by-stage review of `resistance.py`, `rheology.py`, `probability.py`, `perfusion.py` and the
   pericyte modules, in the H1 assessment's per-stage format.

---

## Provenance

Phase 1 authored 2026-08-15 against `cb_pipeline_improvements_sweep` @ `c3c236c`. Measured figures
are reproducible from the six-specimen artefact set in `examples/outputs/cb_h1_paraview/` and the
test suite invocation quoted in S1. Two priors held by the assessment plan were refuted by
measurement and are recorded as refuted rather than removed: the EDT quantisation comb (S6) and the
duplicate US-spelled package (S8).
