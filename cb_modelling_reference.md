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
> **Written so far:** §10 (parameters), §11 (assumptions), §13 (error budget) and Appendix A
> (solver settings). The remaining sections are listed at the end in the order they will be written.

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
| 15 | The full MAP-to-CVP gradient falls across the imaged sub-volume | §8 | ~98 mmHg across roughly 1 mm. Perfusion pressure and absolute flow **overestimated** |
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
| 8 | `M_max` differs 10× between `PerfusionConfig` (0.005) and the H2 driver's `BASE_M_MAX` (0.05). The published §2.3 results used 0.05 | §6.4, §13.6 |

---

## Still to write

In order:

1. **§2** — geometric model, image to graph
2. **§3–§6** — the physics core
3. **§7** — derived physiological quantities
4. **§8, §9, §12, §14**
5. **§1** — scope and overview, last
6. **Front matter** — the question index, once there are sections to point at
7. **Appendices B and C** — symbol table; model → file → test map
