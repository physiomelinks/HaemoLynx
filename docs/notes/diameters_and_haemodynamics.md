# Diameters and haemodynamics

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## haemodynamics/fwhm_diameter.py (named automated.py until 2026-10-23)

**`haemodynamics/automated.py`** — FWHM vessel-**diameter** measurement from raw TIFF (`measure_edge_diameters_fwhm_from_raw_tiff`, `build_graph_branch_label_volume`; neither is re-exported from `haemolynx.haemodynamics`, so import them from the module). *“automated” is a misnomer; this is diameter estimation.* `haemodynamics/edt_diameter.py` is the cross-check: each edge's diameter from the mask's own cross-section. Between them, `haemodynamics/raw_section.py` (`use_raw_section_fallback`, **off by default**) fits the raw image's whole cross-section as a PSF-blurred lumen on the edges FWHM failed on, so the diameter chain is FWHM → raw section → mask → class median → table (`stamp_edge_diameters`); the class median (`diameter_class_median_min_edges`, default 3) is the median measured diameter of the run's edges with the same `branch_order` label, so a large arteriole the table has no entry for is not modelled at `default_diameter`. Accurate on clean synthetic data, including z-running vessels FWHM cannot read; on a speckled real stack it reads the halo round a vessel, which is why it is opt-in. FWHM itself can fit one bright speck cleanly on such a stack, so an edge needs `fwhm_min_accepted_samples` (default 2) accepted samples, and `haemodynamics/fwhm_decoys.py` (`fwhm_decoy_check`, on by default) re-measures a sample of FWHM's vessels moved into vessel-free tissue: it reports FWHM's false-positive rate and speck-width range on the run's own image and flags widths in that range (`fwhm_in_speck_width_range`). Its counterpart, `haemodynamics/fwhm_planted.py`, asks whether FWHM reads the *right* width: it draws vessels of known width (the real neighbour's mask width ± a quarter, blurred by the image PSF, as bright and as noisy as the image) into a copy of the raw image beside sampled ones, and the "Optimise FWHM settings" search scores every trial on their mean relative error (`fwhm_metrics.PLANTED_ERROR_WEIGHT`), which is what lets it tune the profile model, sampling direction, averaging, spacing, accepted samples and aggregation (its `measurement_model` group, skipped without the run's segmentation). With `fwhm_demote_flagged_edges` (on by default) a flagged width, or one disagreeing with the mask's own past `edt_fwhm_disagreement_warn_ratio`, is set aside (`fwhm_demoted`) and its edge takes the next source; FWHM holds each profile's blur at one image PSF shared with the raw-section fit (`fwhm_fix_blur_to_image_psf`). `haemodynamics/endothelial.py` is the alternative first source, used when `measure_diameters_from_raw_image` is on and the raw image is an endothelial stain (`raw_image_stain`; FWHM takes a plasma column, so one setting chooses between them): the internal diameter inside the wall ring, fitted with the blur and wall thickness estimated from the image's wide vessels; its chain is endothelium → mask → class median → table. Its widths run wall to wall, so the edge carries `diameter_basis="endothelial"` (`poiseuille.EDGE_DIAMETER_BASIS`) and its viscosity takes that form of the law while the rest of the network keeps the run's basis -- `edge_diameter_basis` is read wherever a vessel's viscosity is computed. `haemodynamics/sections.py` holds what the section methods share, including `run_edge_tasks`, which spreads independent per-edge work over worker processes reading one memory-mapped volume.

## haemodynamics/viscosity.py

**`haemodynamics/viscosity.py`** / **`haematocrit_distribution.py`** — the viscosity laws (see
*Viscosity model* above) and, with `haematocrit_model="distributed_iterative"`, a per-edge
discharge haematocrit from Pries & Secomb's phase-separation law (their 2005 A, B and
X0 = 0.964(1 − H_D)/D_F), iterated with the flow solve and under-relaxed each pass as Secomb's
NetFlow does, or a daughter near the skimming threshold oscillates instead of converging.
The law covers one vessel dividing in two; `haematocrit_junction_rule` decides every other
junction: `no_separation` (default, NetFlow's `dishem.cpp`), `sequential_bifurcations`
(`dishem_generalized.cpp`, outflows taken largest flow first) or `split_junctions` (the network
is split, see `apply_network_handling` above). The default, `"fixed"`, uses one `haematocrit`
everywhere. The loop ends with one more flow solve from the resistances its last pass left
(`final_solve=True`), so a vessel's flow is its pressure drop over its reported resistance even
when it stops unconverged; a caller that re-solves itself passes `final_solve=False`.

## haemodynamics/resistance.py

**`haemodynamics/resistance.py`** — the flow solve and the two-point resistance, each with a
setting on the Haemodynamics tab's Advanced button. `haemodynamics_solver` (`dense` default, as
before, or `sparse`) is how *every* flow solve holds the network — baseline, haematocrit passes,
perturbations, sweeps (`build_conductance_matrix_from_graph(solver=)`; the solves take either
matrix): dense is N × N, peaking at 42 N² bytes (16.8 GB at 20,000 nodes, measured
2026-10-15; the help text used to say 3.2 GB), sparse stores the vessels and factorises by
sparse LU, same pressures to rounding. `equivalent_resistance_solver` (`eigendecomposition`
default, as before, peaking at 49 N² bytes, or `sparse`: `calc_two_point_resistance_sparse`,
one solve, `inf` between unconnected nodes) finds the two-point resistance between the first
inlet and the first outlet. Either dense solver the run will use is warned about when its
estimate passes half the free memory (`warn_about_dense_solver_memory`, `dense_solver_memory_bytes`,
`available_memory_bytes`: psutil, else `/proc/meminfo` or `GlobalMemoryStatusEx`), from
`build_network` as soon as the node count is known and again from `solve` before the matrices
are made; preflight has no node count. Those per-N² figures are pinned against traced
allocations in `tests/test_flow_solvers.py`, so a solve that gains an N × N copy fails there.
The solve also reports `Solution.network_resistance`, pressure drop over total inflow through
every inlet (`network_resistance`); both go in the statistics CSV's Haemodynamics section.
Perturbation summaries' `equivalent_resistance` is that network resistance, not the two-point one.

## haemodynamics/perturbations.py

**`haemodynamics/perturbations.py`** — a perturbation is a named, typed settings override
(`PerturbationSpec`) that `pipeline/stages.py`'s `run_perturbations` re-solves a copy of the
solved network for, writing into `{name}_{type}` under `perturbation_output_dir`. Fourteen types
plus `none` (`PERTURBATION_TYPES`): arteriole / pericyte / combined diameter changes, capillary
blocks, and sweeps over dilation, inlet pressure, pericyte spacing or length.
**Capillary blocks by mask** (`capillary_block_mask`, 2026-10-22,
`haemodynamics/capillary_block_mask.py`): the vessels to block come from a mask of blobs
(`capillary_block_mask_path`, placed on the run's grid as the pericyte mask is). Each blob is a
block on the vessels when at least `capillary_block_mask_min_overlap_fraction` (0.5) of its
voxels lie in the run's segmentation (`network.volume.image`, read at the blob's voxels; without
one the graph's lumens stand in, with a warning); each such voxel belongs to the vessel whose
centreline is nearest, within its radius plus a voxel's diagonal, bridges never;
`capillary_block_mask_vessels_per_block` blocks the vessel holding most of the blob
(`largest_overlap`) or every one it lies in. The occlusion, zeroing and comparison are the
capillary block's (`perturbations.is_capillary_block`, `CAPILLARY_BLOCK_TYPES`), one draw;
`{name}_mask_blocks.csv` lists every blob and why one blocks nothing. A batch refuses it (one
mask for every image). Not yet tried on a real mask of blocks.
`SETTINGS_FOR_TYPE` says which settings each type reads; an override outside it is reported as
unused and not applied. `INCOMPARABLE_OVERRIDES` (viscosity law, the raw image's stain, haematocrit,
haematocrit model) are refused, because they would move every resistance and the difference
against the baseline would not be the perturbation's effect. A single re-solve writes
summary/edges/statistics CSVs and plots; a sweep writes its grid CSV and curves, and keeps
per-point flows (`sweep_flows.py`) so the viewer's slider can swap them without N graph copies.

## Open ends (from Coding conventions)

- **Open ends** — a dead end that leaves the image is open in the tissue; `open_end_model`
  chooses its boundary condition: `estimated` (**default** since 2026-10-12, decision D1 of
  PLAN_dead_ends.md; Fry et al. 2012: pressures chosen to keep the network near a target
  pressure and wall shear stress, taken from the closed solve, so an open end may feed the
  network as well as drain it), `closed` (the behaviour before), `terminal_resistance` (drains to
  the outlet pressure through k times its own vessel's resistance) or `outlet`. On E14.5 the
  estimate fed 30% as much blood in through 39 open ends as the designated inlets did, drained
  through 29, and agreed with closed on 86% of flow directions but with the drain-only models on
  only 56-65%; network resistance moved under 8% across all five
  (`outputs/open_end_sensitivity_e14/`). Like the viscosity law it moves every flow, so **flows
  are not comparable across open-end models**; perturbations may not override it and solve with
  the baseline's conditions (kept on the graph, `G.graph["open_end_conditions"]`). See
  `haemodynamics/open_ends.py` below.

## Viscosity model (from Coding conventions)

- **Viscosity model** — selectable, and the choice changes every resistance a run produces, so
  **resistances are not comparable across laws or across diameter bases**. The graph records both
  in `G.graph["viscosity_law"]` / `["diameter_basis"]`. All of it is in
  `haemodynamics/viscosity.py`.
  - The question that decides the answer is **which diameter you measured**, not which law you
    prefer. A diameter basis is `plasma_column` — the channel the fluid occupies, what a plasma
    stain images — or `endothelial` (formerly `anatomical`, which still loads:
    `normalise_diameter_basis`), wall to wall, including the ~1.1 µm endothelial surface layer.
    The run's basis is **what the segmented vessels are**: `segmentation_source` on the Input
    tab, beside the segmented image (`plasma_column` by default; `endothelial` for a mask of the
    wall, `anatomical` an alias), read everywhere through `viscosity.run_diameter_basis` — the
    stage's apply config, a perturbation's model, the sweeps' strategy arguments — and recorded
    as `G.graph["diameter_basis"]`. Every diameter the segmentation gave (the mask's width, the
    class medians, the table) is on it. A diameter read from the raw image carries the basis
    *that* measurement read on its edge (`poiseuille.EDGE_DIAMETER_BASIS`, by
    `MEASUREMENT_DIAMETER_BASIS`): FWHM and the raw section `plasma_column`, the endothelial
    fit `endothelial`, whatever the segmentation is; a class median carries the basis of its
    widths. The retired `diameter_basis` setting reads into `segmentation_source`
    (`replaces`), and a perturbation may not override either (`INCOMPARABLE_OVERRIDES`).
  - `viscosity_law="pries"` (**default**) is Pries et al., fitted 3.3–1978 µm, continuous across
    the whole tree, no placeholder and no warning. Reads the basis: `plasma_column` selects
    the *in vitro* (tube) form, Pries, Neuhaus & Gaehtgens 1992; `endothelial` the 1994 *in
    vivo* form, fitted to rat mesenteric networks against wall-to-wall diameters. The in vivo
    form's `(D/(D−1.1))²` factors stand for the ~1.1 µm endothelial surface layer, and it is far
    above the in vitro form in the smallest vessels (8.6× at 5 µm, 4.4× at 10 µm), so applying it
    to a plasma-column diameter counts the layer the measurement already left out.
  - **The in vitro form was wrong until 2026-10-13**: it used the in vivo law's
    `μ₀.₄₅ = 6·exp(−0.085D)+…` where Pries 1992 has `220·exp(−1.3D)+…` — 3.4× too viscous at
    5 µm, 2.9× at 10 µm, converging above 60 µm. Plasma-column runs before then overstate
    capillary resistance by that much. Every form is now pinned to its published equations
    (Pries & Secomb 2005 eqs 5–11, Secomb & Pries 2013 eqs 3–7) in `tests/test_viscosity_laws.py`.
  - `pries_viscosity_form` (Haemodynamics tab, behind the Advanced button under the law) holds
    every vessel to one form whatever its basis: `in_vivo_1994`, `in_vitro_1992`, or
    `in_vivo_2005`, Pries & Secomb's 2005 model — the in vitro law at the diameter an
    endothelial surface layer leaves free (`D − 2W_ph`), times `(D/D_eff)⁴`, the layer's
    thickness depending on diameter and haematocrit (`viscosity.pries_secomb_2005_layer_um`);
    like 1994 it reads a diameter as wall to wall. `auto` (default) keeps the basis choice above.
    It resolves to a law name (`pries_in_vivo_2005`, …) through `viscosity.run_viscosity_law`,
    which every settings reader uses, so the graph's `viscosity_law` says which form ran; a
    perturbation may not override it. Secomb's own NetFlow still uses the 1994 law, with `D`
    scaled by `(92/MCV)^⅓` for red-cell size and rat plasma at 1.05 cP; HaemoLynx does neither
    (plasma is 1.2 mPa·s).
  - `viscosity_law="capillary_power_law"` is the law used before: `µ(d) = 3.0 mPa·s · (5 µm/d)^1.647`
    up to 7 µm, then a constant 3.5 mPa·s, raising `haemodynamics.PlaceholderViscosityWarning`
    between 7 and 100 µm. Kept for comparison with earlier results. It matches neither Pries
    form — 1.1–1.7× the in vitro law over 3–7 µm, a fifth of the in vivo one — and its slope
    crosses below plasma viscosity at 8.7 µm.
  - `viscosity_law="constant"` is plasma everywhere, for separating geometry effects from
    viscosity ones.
  - Switching from the power law to Pries in vitro **lowers capillary resistance** (0.59× at
    5 µm). `tests/test_viscosity_laws.py` pins the ratios at each diameter.
