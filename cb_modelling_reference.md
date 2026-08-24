# CB modelling reference

> **What this is.** A working record of the mathematical and physiological modelling in the
> carotid body simulation pipeline: what it models, what was chosen, why, and what the numbers
> were. Written for future-me. Assumes the project is already understood.
>
> **Code this describes:** branch `cb_pipeline_improvements_sweep`, commit `8a2b81c`.
>
> **What would invalidate this document:** a change to the viscosity law, the boundary selection
> rule, the unit conversion constants, the calibre estimator, or which coupling tier is run.
>
> **Written so far:** §10 (parameter reference) and Appendix A (solver settings). The remaining
> sections are listed at the end in the order they will be written.

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

## §10 — Parameter reference

### 10.1 Imaging and domain

| Parameter | Value | Units | Class | Source / justification | Sensitivity |
|---|---|---|---|---|---|
| `PROCESSING_VOXEL_UM` | (1.8639, 1.866, 1.866) | µm, (z, y, x) | (i) | Acquisition. The single spacing every calculation uses; per-specimen values are kept as provenance only. Slightly anisotropic in z. | fixed |
| Imaged sub-volume | ~1–2 | mm³ | (i) | Acquisition | fixed |
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
| `enable_shannon_entropy` | True | — | (iii) | — | unswept |
| `shannon_entropy_threshold` | 0.95 | — | (iii) | Chosen | unswept |

> ⚠ **Open item 1 — two different thresholds are in play.** The config defaults above are
> 0.65 / 0.75, but the H1 cohort runs froze the vessel threshold at **0.90** (`cb_h1_batch.py
> --threshold 0.90`, and `cb_h1_th_metrics.py` reads "the frozen 0.9 that cb_h1_batch
> selected"). Which value is in force depends entirely on which driver ran. Resolve before
> quoting either.

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
| `M_max` | 0.005 | mmol/L/s | (iii) | Maximum metabolic consumption rate. Chosen | **unswept — and it sets the hypoxic fraction** |
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

## Summary of open items raised by this table

| # | Item | Blocks |
|---|---|---|
| 1 | Two segmentation thresholds in play — config 0.65/0.75 vs the frozen 0.90 used for H1 | §2.2, and any quoted calibre |
| 2 | Two boundary rules coexist with disagreeing axis and percentage defaults | §2.8, §8, §13 — the largest sensitivity in the model |
| 3 | Arterial PO₂ set in both config and solver bodies | §5, §6 |
| 4 | Baseline haematocrit duplicated in the Tier 1 washout path | §6.6 |
| 5 | `C_arterial` is dead configuration — declared 3×, read 0× | §6 |
| 6 | Solver tolerances disagree between config and code | Appendix A |
| 7 | 13 parameters still marked `[CITE]`, including every blood-gas solubility and the Spencer CO₂ curve | §10 completeness |

---

## Still to write

In order:

1. **§11** — assumptions, with expected direction of bias
2. **§13** — error budget
3. **§2** — geometric model, image to graph
4. **§3–§6** — the physics core
5. **§7** — derived physiological quantities
6. **§8, §9, §12, §14**
7. **§1** — scope and overview, last
8. **Front matter** — the question index, once there are sections to point at
9. **Appendices B and C** — symbol table; model → file → test map
