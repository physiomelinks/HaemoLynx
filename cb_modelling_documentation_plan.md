# Plan: documenting the mathematical and physiological modelling in the CB pipeline

> **Status:** structure plan only. No content written yet. This document says what the
> modelling documentation should contain, in what order, and what each section must resolve.
> Iterate on this file first; write the real document once the shape is agreed.
>
> **Audience — decided 2026-08-24.** A personal working record. One reader: the author,
> mostly six months from now. Not a supervisor document, not a reviewer document, not thesis
> prose. See §3 for what that decision settles.

---

## 0. The problem this has to solve first

There are already seven documents in this repository describing the modelling, and every one of
them predates the `#98` remediation sweep:

| Document | Last touched | Lines |
|---|---|---|
| `examples/cb_image_to_model_modelling_capabilities_summary.md` | 2026-07-22 | 544 |
| `examples/cb_image_to_model_modelling_capabilities_detailed_summary.md` | 2026-07-22 | 1128 |
| `examples/cb_image_to_model_modelling_capabilities_conceptual_summary.md` | 2026-07-22 | 1320 |
| `examples/cb_image_to_model_modelling_capabilities_conceptual_summary_v2.md` | 2026-08-18 | 1500 |
| `examples/modelling_capabilities_supervisor_overview.md` (untracked) | 2026-07-30 | 272 |
| `haemodynamic_modelling_methodology.md` | 2026-07-22 | 158 |
| `modelling_and_hypothesis_testing_documentation.md` | 2026-08-18 | 197 |
| `thesis_methods_section_skeleton.md` | 2026-08-18 | 776 |

The first four share an identical §1–§10 skeleton and differ only in expansion depth. They are a
ladder, not four documents.

The sweep landed 2026-08-19 to 08-20 and changed physics, not presentation:

- the viscosity law was a hybrid of the two relations it chose between (`bfed0da`) — a ~3.4×
  apparent-viscosity difference at D = 8 µm;
- flow left the resistance solve in `mmHg·µm³/cP` and was coupled to a sink in `mmol/L/s`
  unconverted, so the sink exceeded the source by 2.2 × 10⁴ (`535bcb1`);
- each edge's whole flow was recorded against every cell it crossed, so the oxygen source grew
  with grid resolution and the field was not grid-convergent (`25ab93f`);
- the perfusion CG never converged under a non-SPD preconditioner (`6bba306`);
- the boundary rule was replaced — band → face-crossing — cutting boundary sensitivity from
  75.8% to 13.3% (`d32fc85`);
- `constrict_at_pericytes` was disabled and now raises, after fabricating calibre on 12.3% of
  edges (S14);
- three silent fallbacks that returned a number instead of an error were made to raise
  (`7e273aa`).

**So the deliverable is not a new eighth document sitting alongside seven stale ones.** It is one
current document that explicitly supersedes the four-rung ladder, and a decision about what
happens to the ladder. Proposal: keep the ladder in place, add a superseded banner to each of the
four with a pointer, and delete none of them until the new document is complete.

**Second framing decision.** The old ladder documents `carotid_image_to_model.py` — the generic
image-to-model pipeline. The thing worth documenting now is the **carotid-body simulation
pipeline as actually run for H1 and H2**, which is that script *plus* the six H1/H2 driver scripts
in `examples/`, plus `roi_placement`, `threshold_selection`, `tissue_regions`, `transit`,
`cohort_split` and `th_morphometry` — none of which appear in any existing modelling document.
That is the scope gap this document closes.

---

## 1. Cross-cutting conventions the document must adopt up front

These are decisions to make once and apply everywhere; getting them wrong makes the document
unreviewable.

### 1.1 A status label on every capability

Every model in the pipeline sits in one of five states, and mixing them silently is how the
existing documents mislead. Proposed labels, carried as a badge on every subsection heading:

| Label | Meaning | Examples |
|---|---|---|
| **Active** | Runs on the CB path under default configuration | Poiseuille + Laplacian solve, Pries–Secomb in vivo, ADR steady state, EDT calibre |
| **Available, not default** | Implemented and reachable, off by default | FWHM calibre, `constant_radius`, in vitro viscosity law, multi-species solver |
| **Frozen** | Deliberately disabled for CB with a stated reason; live elsewhere | `pericyte_mask`, `probability`, `pericyte_comparison`, `constrict_at_pericytes` (raises) |
| **Implemented, unreachable** | Code exists, no configuration path reaches it | endothelial barrier model under CB defaults (S3); EDT-from-binary-mask measurement (Stage 20) |
| **Superseded** | Replaced during the sweep; documented so old outputs remain readable | band boundary rule, hybrid viscosity law, pre-conversion flow units |

### 1.2 Symbol table, with the collisions already resolved

`thesis_methods_section_skeleton.md` §X.0 has already done this work and found five collisions in
the source ladder: `C` triple-booked (hydraulic conductance / gas content / Pries–Secomb shape),
`A` triple-booked (skimming asymmetry / surface area / system matrix), `D` (diameter / diffusive
conductance), `n` (Hill coefficient / branch order / grid dimension), `L` (segment length /
Laplacian). **Adopt that resolved scheme verbatim.** Do not re-derive it and do not revert to the
ladder's symbols.

### 1.3 Units, stated per equation

The pipeline is not in one consistent unit system and this has already caused one order-of-10⁴
defect. Every equation gets its units annotated inline, and §9 carries the conversion constants
explicitly, including `POISEUILLE_FLOW_TO_UM3_PER_S = 133.322387415 × 10³` and its derivation.

### 1.4 Parameter provenance class, on every number

Three classes, following the thesis skeleton's §X.9 split:
**(i)** literature-derived, with citation; **(ii)** empirical correlation transferred from another
tissue or species (e.g. Pries–Secomb from rat mesentery); **(iii)** chosen, heuristic or estimated.
A number with no class is a defect in the document.

### 1.5 Code references

File and line, in the `path.py:123` form the assessment documents already use, so every claim is
checkable. Equations stay code-free in the body; the reference lives in a margin/footnote column.

---

## 2. Proposed section structure

### §1 — Scope, domain and modelling philosophy

- What is being modelled: a ~1–2 mm³ imaged sub-volume of rat carotid body, two channels
  (lectin vasculature, TH glomus), six specimens, 3 WKY / 3 SHR.
- The model hierarchy in one figure: 3D image → 1D vascular graph → 1D flow + rheology →
  3D tissue transport → derived physiological quantities.
- What the model is *for*: H1 (morphology) and H2 (perfusion), stated as the two questions.
- What the model deliberately does not do: no compliance, no pulsatility, no autoregulation,
  no growth/remodelling, no neural output.
- Freshness contract: "current as of commit X; supersedes the four capability summaries."

### §2 — Geometric model: from voxels to a vascular graph

This section does not exist in any current modelling document, and it must, because every
downstream physical quantity inherits its error and resistance goes as *d*⁻⁴.

- **2.1 ROI placement** (`roi_placement.py`) — why a centred box is a modelling choice and a
  biased one: axial tissue peak ranges slice 106–230 of 435; WKY mean depth fraction 0.40 vs
  SHR 0.34. Tissue-centroid placement as the correction.
- **2.2 Segmentation threshold selection** (`threshold_selection.py`) — the calibre-vs-
  fragmentation criterion, and why the conventional component-count criterion is flat on this
  data's topology.
- **2.3 Mask formation** — joint hysteresis threshold (`image.py:179`), entropy map, morphological
  cleanup, component filtering.
- **2.4 Skeletonisation and graph construction** — `skeleton.py`, `graph/build.py`; gap bridging
  and the mask-inflation artefact it introduces (which contaminated the EDT/FWHM correlation).
- **2.5 Topology conditioning** — pruning, degree-2 collapse, reconnection, optimisation; what
  each does to node/edge counts, and what H1's topological node counting therefore measures.
- **2.6 Calibre assignment — the single most consequential modelling choice.**
  - EDT (inscribed-radius) vs FWHM (Gaussian fit, `2√(2 ln 2)·σ`).
  - Measured comparison: EDT 100% coverage, median 6.37 µm; FWHM 76.5%, median 8.20 µm;
    Pearson *r* = 0.245. Why EDT is the default on the evidence.
  - Junction-proximity exclusion, 3.73 µm, with the swept table.
  - The branch-order exponential fallback law, and Murray's law explicitly *not* used.
  - Provenance guard: `MAX_SYNTHETIC_FRACTION_EDT = 0.0` and why a partly fabricated
    distribution must refuse rather than warn.
- **2.7 Branch-order assignment** (`branch_order.py`) — BFS from boundary nodes; hierarchical
  variant; what "B01" means and where the labels are consumed.
- **2.8 Boundary terminal node selection** (`boundaries.py`) — band rule (superseded) vs
  face-crossing rule (active), the 86%-of-degree-1-nodes-are-spurs finding, and the sensitivity
  numbers that decided it.

### §3 — Haemodynamic model: 1D network flow

- **3.1 Segment resistance** — Poiseuille, *R* = 128 µL/(π d⁴); the *d*⁻⁴ amplification stated
  once, prominently, because it governs the whole error budget.
- **3.2 Variable-diameter segments** — the resistance integral and its quadrature.
- **3.3 Constriction geometry** — sphincter and periodic modes, intimal cushion and pre-capillary
  ratios. **Frozen.** Document the geometry, then document why it is disabled: sites placed by a
  hard-coded topological rule, severities from no vasomotor model, 0.5 ratio = 16× local
  resistance error on a *measured* vessel.
- **3.4 Network solve** — Kirchhoff's current law, conductance matrix, graph Laplacian, Dirichlet
  pressure boundary conditions, direct vs iterative dispatch.
- **3.5 Effective two-point resistance** — the Laplacian pseudo-inverse form; state whether it is
  reported.
- **3.6 Wall shear stress** (`rheology.py:371`) — derived, exported; state whether reported.
- **3.7 Units of flow** — the mixed-unit trap and the conversion constant, with the failure it
  caused.

### §4 — Blood rheology

- **4.1 Apparent viscosity and the Fåhræus–Lindqvist effect** — Pries–Secomb μ₄₅, both relations
  side by side, the 3.4×-at-8-µm disagreement, the endothelial surface layer term and why it
  follows the law rather than being applied unconditionally. In vivo is the active default.
- **4.2 Phase separation / plasma skimming** — the bifurcation split relation, erythrocyte mass
  conservation, binary-bifurcation-only assumption.
- **4.3 Coupled flow–haematocrit–viscosity solution** — Picard iteration, tolerance, iteration
  cap, convergence behaviour, the resistance-rescaling rule.
- **4.4 What the viscosity law does and does not move** — S22: a 3–4× change in apparent viscosity
  did not move any within-specimen ratio. This is the justification for ratio-based reporting and
  belongs here, not buried in the assessment.

### §5 — Blood gas chemistry

- **5.1 Oxygen content** — Hill equation, *n* = 2.7, dissolved + bound, α = 1.34 × 10⁻³ mmol/L/mmHg.
- **5.2 Bohr effect** — the empirical log P₅₀ shift (Kelman/Severinghaus form), baseline P₅₀ = 26 mmHg.
- **5.3 Carbon dioxide content and the Haldane effect.**
- **5.4 Tissue pH** — Henderson–Hasselbalch, fixed bicarbonate buffer, no renal compensation.
- **5.5 Species mismatch** — human haemoglobin parameters on rat tissue, flagged as an assumption
  with a direction.

### §6 — Tissue transport: the 3D advection–diffusion–reaction model

- **6.1 The perfusion grid** (`PerfusionGrid`) — construction, resolution, indexing, and the
  padding-to-segmented-volume fix (`8a2b81c`) plus the prediction it falsified (S29).
- **6.2 Vessel-to-grid mapping** — how edges deposit source terms; the conservation defect where
  an edge's whole flow was recorded against every cell it crossed, and why grid-convergence is the
  test that catches it.
- **6.3 The ADR operator** — seven-point stencil, Neumann boundaries, diagonal regularisation,
  Jacobi preconditioning, CG.
- **6.4 Metabolic consumption** — the phenomenological `M_max`/`k_reduce` form, and how it differs
  from Michaelis–Menten in the low-PO₂ regime (Michaelis–Menten is *not* used anywhere).
- **6.5 Heterogeneous metabolism from the TH segmentation** (`tissue_regions.py`) — volume-fraction
  per cell rather than centre sampling, and why: ~154 mask voxels per cell at 10 µm, tissue-to-
  vessel distance 5.3–7.9 µm sits below one cell width. The glomus:stroma metabolic ratio is a
  swept parameter, not a measurement — say so plainly.
- **6.6 The three coupling tiers** — steady-state ADR; coupled 1D–3D with endothelial permeability
  barrier; multi-species O₂/CO₂/pH with respiratory quotient. State clearly which tier ran for
  which published result, and that the barrier model is unreachable under CB defaults (S3).
- **6.7 Grid resolution** — the convergence evidence: median PO₂ 27.34 / 27.92 / 28.21 at
  10 / 6 / 4 µm, halving increments, extrapolating to ≈28.5; 4 µm within ~1% at 1/27 the cost.

### §7 — Derived physiological quantities (the measurement layer)

The layer that turns solved fields into the numbers H1 and H2 actually quote. Entirely absent from
the existing documents.

- **7.1 Morphometry** (`stats.py`) — vessel density, length density, tortuosity measures, branching
  statistics, tree asymmetry, fractal dimension, path efficiency, betweenness and community
  structure. For each: the definition, and whether it is reported or merely computed.
- **7.2 Two-channel morphometry** (`th_morphometry.py`) — glomus parenchymal volume, centreline
  length density within TH clusters, tissue-to-vessel distance. The co-registration argument (two
  channels of one acquisition, no registration step).
- **7.3 Functional shunting** (H2 §2.1) — the shunt ratio definition, edge classification by
  centreline fraction inside the TH mask sampled along the whole polyline, and why an endpoint test
  would misclassify exactly the vessels of interest.
- **7.4 Spatial haematocrit profiling** (H2 §2.2).
- **7.5 Hypoxic fraction** (H2 §2.3) — threshold definition, restriction to TH-positive volume.
- **7.6 Transit time and PO₂ depletion** (H2 §2.4, `transit.py`) — τ = πr²L/Q, Dijkstra from inlets,
  infinite for zero-flow edges. **Reported as a ratio, never an absolute** — the ±45% calibre floor
  and the unreconciled magnitude are both the reason; state it here where it is used.
- **7.7 Cohort-split diagnostics** (`cohort_split.py`) — the instrument-vs-tissue test, the exact
  permutation *p*, and the n = 3 floor of 0.10 that makes eyeballing separation useless.

### §8 — Boundary and initial conditions

- Pressure BCs: MAP 100 mmHg → CVP 2 mmHg across the imaged sub-volume, and the assumption that
  buys (full organ-scale gradient across ~1 mm — overestimates perfusion pressure).
- Inlet/outlet identification: the face rule, its axis-1 restriction, and why axis 1 is the only
  axis solvable in all six specimens.
- Domain truncation: interior dead ends, not the crop, are the boundary problem (S10).
- Tissue BCs: Neumann, no exchange beyond the imaged volume, and the direction of that bias.
- Blood gas inlet values.

### §9 — Numerical methods

Spatial discretisation; Picard iteration and stabilisation; linear solver dispatch and thresholds;
preconditioning; root finding for the Hill inversion; regularisation constants; physical clamping
and its frequency; unit conversion constants. Keep this section short and push the settings table
to an appendix.

### §10 — Parameter reference

One table, sorted by section, columns: symbol, value, units, provenance class (i/ii/iii), source or
justification, sensitivity (measured / assumed / unswept). Every default in `HaemodynamicsConfig`
and `PerfusionConfig` appears exactly once.

### §11 — Assumptions, with expected direction of bias

Adopt the thesis skeleton's 20-row table wholesale, then extend with the sweep-era additions it
predates: the metabolic contrast assumption (§6.5), the face-rule axis restriction (§8), the
ratio-only reporting convention (§7.6), and grid padding (§6.1).

### §12 — Verification status

- The six verification strategies (analytical closed-form, conservation/invariant, synthetic
  phantoms, equivalence oracles, graceful degradation, physical bounds).
- The coverage table — oracle type and tolerance per component. Already drafted in the thesis
  skeleton §X.10.3; port and extend with `test_flow_conservation.py`, `test_flow_units.py`,
  `test_tissue_regions.py`, `test_transit.py`, `test_boundary_faces.py`,
  `test_perfusion_preconditioner.py`, `test_cohort_split.py`, `test_threshold_selection.py`,
  `test_roi_placement.py`, which postdate it.
- **The gaps, stated as facts:** no grid-convergence / order-of-accuracy study for any PDE solver;
  no validation against experimental CB perfusion or oxygenation measurement; several components
  verified only transitively or only directionally.

### §13 — Error budget and known limits

The section that makes the document trustworthy rather than promotional. Sourced from the H1/H2
capability assessments, restated as model properties rather than as a defect log.

- *d*⁻⁴ amplification: ~94% per-edge resistance uncertainty at the median calibre.
- Independent error averages down across a network solve; correlated error does not. The threshold
  is the dominant correlated term; measured median calibre shift 0.922 µm over the clean interval.
- A within-specimen ratio cancels ~86% of the correlated error — the argument for ratio reporting.
- The ±45% noise floor on any absolute flow quantity.
- Boundary sensitivity: 118.8% → 43.1% → 13.3%, and that it exceeded calibre error.
- Absolute perfusion sits far below physiological, and boundary pressure is not the cause (S27).
- Resolution limit: the grid does not resolve the 5.3–7.9 µm tissue-to-vessel distance.
- Calibre is disqualified as an H1 finding: the between-group gap is ~1/20 of a voxel.
- The SHR TH-channel labelling asymmetry (22.9× more glomus labels in WKY; SHR-B and SHR-C carry
  none), and what it forbids.

### §14 — Provenance and reproducibility

Commit hash the document describes; which script produces which artefact; seeds; the
`artefact_provenance.py` mechanism; environment and runtime; where the six specimens' outputs live.

### Appendices

- **A** — solver settings table (purely numerical, separated from §10 model parameters).
- **B** — full symbol table.
- **C** — map from every model to its source file, line and test.
- **D** — superseded models, with the numbers they produced, so pre-sweep outputs stay readable.

---

## 3. Audience: a personal working record

**Settled 2026-08-24.** The document is written for the author, to record and look up work
already done. That is not a soft constraint. It changes six things.

### 3.1 Write for future-you, not present-you

Present-you knows why EDT beat FWHM and why transit time is reported as a ratio. Six-months-you
will not. So the *reasoning* and the *numbers* behind every choice get written down even though
they feel obvious now. This is the single rule the document is optimised for.

The test for every subsection: **can it answer a question in thirty seconds, six months from
now, without opening the code?** Not "does it read well."

### 3.2 Optimise for lookup, not for narrative

Findability beats flow. In practice:

- tables over prose wherever a table will carry the content;
- every subsection follows the same six-slot shape — *what it does · what was chosen · why ·
  the number · source file and line · the test that covers it*;
- `path.py:123` references everywhere, because the usual next move after reading is to open
  the code;
- a lookup index at the front, mapping likely questions ("why is absolute perfusion low?",
  "which viscosity law ran?") to sections.

### 3.3 Assume the reader knows the project

No introduction to the carotid body. No definition of Poiseuille's law. H1 and H2 get one line
each as a reminder, not a rationale. This is where most of the ladder's length went, and it all
comes out.

### 3.4 Be blunt

No audience to persuade means no hedging. §13 (error budget) and the "gaps" half of §12
(verification) become the most useful parts of the document rather than the risky ones. State
what is broken, what is unvalidated, and what a number cannot support, in those words.

### 3.5 Consequences for the open questions

| Question | Settled as |
|---|---|
| One file or a folder? | **One file.** Ctrl-F across the whole thing is the main access method. |
| Depth? | **One depth.** Full equations, reasoning inline, no conceptual tier. The ladder existed for an outside reader. |
| Length? | Roughly **half** the earlier 60–80 page estimate, once §3.3 padding is removed. |
| The four ladder documents? | **Superseded banner, keep the files.** Pre-sweep outputs still need to be readable against them. |
| Frozen / unreachable capabilities? | Full geometry for the constriction models (live on the `resistance_network_pipeline` path); a paragraph and a pointer for the rest. |
| Citations? | Enough to re-find the source for class (i) and (ii) parameters. No formatted bibliography. |
| Figures? | Two at most — model hierarchy, coupling tiers. Drawn for clarity, not for polish. |

### 3.6 Two things to keep in view

- **Destination.** A personal engineering record is not thesis drafting, so it belongs in the
  repo. If it later becomes source material for methods prose, that derived version moves to
  `~/Desktop/me_bioeng_thesis_drafting/` and this file stays put.
- **It will go stale, exactly as the other seven did.** Carry a commit hash at the top and a
  short "what would invalidate this" list — a change to the viscosity law, the boundary rule,
  the unit constants, the calibre estimator, or the coupling tier in use.

### 3.7 Still open

- Nothing blocking. Writing can start.

## 4. Suggested order of writing

1. §10 parameter table and Appendix A — mechanical, and it forces every provenance question to be
   resolved before prose is committed around a shaky value.
2. §11 assumptions — sets the honest scope everything else must respect.
3. §13 error budget — the constraints that determine how §7 may report anything.
4. §2 — the section that does not exist yet and that everything downstream inherits from.
5. §3 → §6 — the physics core, in order.
6. §7 — the measurement layer.
7. §8, §9, §12, §14.
8. §1 overview, last.
