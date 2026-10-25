# Plan: the raw image as confirmation in skeletonisation and graph building

Written 2026-10-12 for Harvey to approve, against `Devel_GUI_improvements_HD` at **cdaa645**,
after asking the two sessions whose work this builds on what they are doing (section 1). Every
E14.5 number is from the MCA stack (287 × 512 × 512, 0.98 × 0.98 × 1.0 µm; the dextran channel
C1 of the composite) and from the harness outputs under `outputs/clean_graph_phase1e/`,
`outputs/doubled_run_review/` and the dead-end session's `outputs/dead_end_review/FIT_G4.md`.

**The one-line version.** The segmentation was made from the raw channel, so inside the mask the
raw image mostly repeats what the mask says (G4 proved it: dextran contrast along a chain scores
no better than marking everything one way). It says something new only where the mask is
*silent* (background a bridge, a join or a loop's hole crosses) or *blind* (one binary blob that
is really two lumens with a dark wall between). So the plan puts one small, sampled, memmap-safe
reader of the raw image (`RawEvidence`) under exactly those five decisions, calibrates each with a
marked grid, and leaves everything else mask-only on purpose.

## Status (2026-10-25)

No raw-image rule ships. Every grid the plan built found no raw measure near 90 %, and most did
no better than marking all one way, so each decision stays the mask's. What is left is grid UR2
with Harvey and what follows from it.

| Item | State | Where |
|---|---|---|
| Phase 0: `RawEvidence`, its calibration on the graph, the primitives | **Done** | `87a317e9`, `96b0329a` (2026-10-13); both stains; `stages._record_raw_evidence`; the dead-end wall primitives E1-E3 `bafd3aa2` (`PLAN_dead_ends_raw.md`) |
| Phase 0: harness columns (D6) | **Done** | `8204478d`: the build log names the raw step's seconds and RAM; `scripts/graph_artefacts.py --raw-cost` builds each side without and with the raw image and reports time, peak memory, the raw step and the windows read |
| D6 budgets on E14.5 | **Measured, within budget**: +0.6 % build time (the raw step 1.2 s of 210 s), +1 MB of peak (4,912 -> 4,913 MB); the channel, 144 MiB, is held only while it calibrates, under the build's peak; 0 windows read | `docs/notes/graph_building.md`, *The raw image in graph building* |
| D6 budget on the muscle stack | **Waits on data**: the 16.4 G-voxel segmentation is on this machine, its raw image only as a 1,000 x 1,000 crop at an unknown offset | -- |
| H re-score (Phase 2's free step) | **Done, nothing ships** (2026-10-12) | `outputs/doubled_run_review/FIT_raw.md` (main checkout) |
| Phase 1 (P1, P2): grids M, M2 | **Closed, nothing ships** (2026-10-13). `MaskSupport(raw=)`, `bridge_raw_confirmation`, `bridge_raw_max_gap_um` closed by design. The ahead turn's review ranking: built for the dead-end list `368107db`, taken out `6d335f88` after G3′ | `FIT_M.md`, `FIT_M2.md` |
| Phase 2 (P3): one lumen or two | **Closed, nothing ships**. The `keep_apart` hook exists (`doubled_stretches`, `lumen_loops`, `assemble.lumen_cleanup`) and nothing fills it; `lumen_cleanup(raw=)` closed by design | Phase 2 |
| Phase 3 (P4): grids I, I2 | **Done**: the short-loop rule (D8, `5bd4c5b3`, `a3188f12`), then the hole exception from grids Q, Q2 (`7cc17cff`); the raw hole reading ships nothing | `FIT_I.md`, `FIT_I2.md`, `FIT_Q*.md` |
| Phase 3: the raw reading ranking the Short loops list | **Not built**: it scored no better than all F on I and I2, and a measure that fails its grid stays a measure, not an order (`6d335f88`'s precedent). For Harvey | -- |
| Phase 3: "the Post processing page shows the raw panel" | **Not built**: the plan never said what the panel shows. For Harvey | -- |
| Phase 3 (P5): pinhole plugging | **Closed by the marks**: it needed I2's raw hole threshold, and none passed | Phase 3 |
| Phase 4 (P6): grid UR1 | **Marked and fitted** (Harvey, 2026-10-25): 26 H / 4 V; no measure beats all H over the 30; on the ten recovery draws into (8 H), dextran over its neighbours above 1.05 gets 10/10 | `8204478d`, `c5bc6ad2`, Phase 4 |
| Phase 4: grid UR2 | **Built and sent** (2026-10-25), two rules fixed first: the local dextran above 1.05, and every piece H (recovery drawing nothing). https://claude.ai/artifact/NdhbUg7HEzj9eiLKrrg28R | `c5bc6ad2`, Phase 4 |
| Phase 4: a skip rule in `mask_recovery.py` | **Waits on UR2's marks**; the lead reads dextran, which E14.5's run (claudin-5, D0) does not, so it is Harvey's call even at 90 % | Phase 4 |
| Phase 4: the uncovered-mask review list carrying the measure **[later]** | **Closed by UR1's marks**: no measure ranks V above H over the 30 | Phase 4 |
| Per-depth calibration (Risks) | **Waits**: no rule reads the reader; UR1's key carries the global and the local readings and the depth | Risks |
| Phase 5: docs and the review | **Done** | CLAUDE.md (layout, `lumen_loops`), `docs/notes/graph_building.md`, `RELEASE_REVIEW.md` `graph.6b`, `graph.8b`, `memory.6a` |

---

## 1. What the two sessions are doing (their words, 2026-10-12)

*History: both branches have landed on `Devel_GUI_improvements_HD` -- the raw-image input as
`8dea0eb7` (`raw_image_path`, `raw_image_channel`, `raw_image_stain`, `preprocessing/raw_stain.py`,
the retired settings migrating through `Schema.upgrade`), the clean-graph work as `0300a8a1`
(`graph/doubled_stretches.py`, with the `keep_apart` hook). What follows is what the two sessions
said on 2026-10-12.*

**Deadends handling plan** (branch `claude/raw-image-input`, off `claude/dead-ends`, worktree
`.claude/worktrees/dead-ends`, nothing committed yet):

- One raw image for the whole pipeline, set on the Input tab: new settings `raw_image_path`,
  `raw_image_channel`, `raw_image_stain` (`auto | plasma_column | endothelial`). Retired, with a
  migration (`Setting.migrates`): `ilastik_unsegmented_image_path`, `fwhm_raw_tiff_path/_channel`,
  `endothelial_image_path/_channel`, `use_fwhm_edge_diameters` + `use_endothelial_diameters` (into
  `measure_diameters_from_raw_image`), and `diameter_basis` (into the stain).
- The raw image is read for ilastik, diameters (FWHM or the endothelial fit, by stain), Check
  segmented image, Optimise, and tissue volume (fallback).
- New `preprocessing/raw_stain.py`: `read_raw_stain(raw, mask, voxel_size_zyx) -> StainReading`,
  16 blocks of 40 µm, 0.8 s on E14.5, no RAM beyond the loaded channel; the result will be kept on
  `G.graph["raw_image_stain"]`.
- Loading: `haemodynamics.automated.load_single_channel_tiff_volume`, per use, **float32**
  (memmap when `use_memmap_loading`); no `raw_volume_cache` yet.
- **Not doing**: anything with the raw image in skeletonisation or graph building ("that's
  yours"). Dead-end *classification and repair* with the raw image stays in `PLAN_dead_ends.md`
  (D6, Phase 6) — this plan references it and does not duplicate it.

**Skeleton topology improvements** (branch `claude/clean-graph`, worktree
`.claude/worktrees/clean-graph`, off c65749b; commits 8ea34aa, 44218d3, 0300a8a + one pending):

- `graph/doubled_stretches.py` (new): trim / zip / rejoin every doubled run, inside every
  `consolidate_lumen` round after the loop step; bridges never touched; minimum run 0 (D6 of
  `PLAN_clean_graph.md`). `consolidate_lumen` runs once more after the thick split; a degree-2
  node beside a bridge is never merged; `remove_dangling_bridges`; the census in
  `diagnose_lumen_artefacts`.
- Not touched: `skeleton.py`, collapse, reconnect, `mask_recovery.py`, `prune.py`,
  `dead_ends.py`, `facing_ends.py`, `schema.py`. No new step label, no new setting.
- **Open questions they name for the raw image**: whether two centrelines in one lumen are one
  vessel or two (grid H: 19 D / 11 T, *no mask measure separates them*; "do take it"); the loop
  hole test (Phase 2c, grid I, not started; "yours if you want it"); the `unknown` dead ends
  (grid L; the dead-end session's).
- E14.5 build: 253 s with the clean-up after the split; 421 s with every doubled run resolved.
  Peak memory not measured.

Both will send a completion report here. **Nothing in sections 4–6 starts until both have
landed**; Phase 0 (section 4) touches neither of their files and can start now.

---

## 2. Where the raw image can and cannot help (the assessment)

What was read: `bridge_mask_support.py` (`MaskSupport`, `PackedMask`, `shadowed_samples`),
`reconnect.py` (`MaskBridges`), `skeleton.py` (`bridge_gaps`, `connect_skeleton_components`,
the bundle hub links), `facing_ends.py`, `lumen_loops.py` (`loop_inside_one_lumen`: antipodal
chords through the mask), `mask_recovery.py`, `prune.py`, `dead_end_extension.py`,
`segmentation_cleanup.py`, `segmentation_raw_comparison.py` (Otsu + whole-volume labels),
`fwhm_decoys.py` (vessel-free tissue finder), `haemodynamics/sections.py` (the profile sampler),
`graph/post_processing.py` (`IntensityCostField`: a *windowed* raw cost for A\*, already
memmap-safe), `io/raw_volume_cache.py`, `io/crop.py`, the two peer branches, and the marks.

### 2.1 The evidence that bounds this plan

| grid | question | raw measure tried | score | what it means here |
|---|---|---|---|---|
| G1–G3 (`FIT.md`) | is a dead end real / broken / leaves | `raw_tip_over_chain` | 16/29 | no better than one way |
| **G4** (`FIT_G4.md`, 27 judged) | is a mask-covered chain a vessel | dextran chain contrast | **15/27 = all one way** | "every chain lies in mask made from dextran, so dextran reads 0.2–2.2 on all of them" |
| G4 | same | claudin-5 chain contrast | 20/27 | the best measure, but D5 says one channel only |
| H (`doubled_run_review/FIT.md`) | one vessel drawn twice or two | *none tried* | best mask measure 21/30 | the raw dip between the two centrelines is untried |
| I, L | loops' holes; `unknown` ends | *not built* | — | 2c and 3e of the clean-graph plan |

So **the raw image is not a vessel detector on top of the mask** — on a plasma stain, where the
mask is, the raw is bright by construction. It is informative in two places only:

1. **Background the mask has none in**: the stretch a bridge, a facing join or a recovered
   strand's join crosses; the hole inside a short loop; a tunnel through a blob. Here the mask
   says "nothing", the raw says "lumen" (a dropout, a pinhole) or "dark" (tissue, a wall).
2. **Inside one blob of mask where the mask is binary-blind**: two touching lumens read as one
   blob, but the raw profile across the chord between their centrelines dips at the wall; one
   lumen does not dip.

### 2.2 Incorporation points, ranked

| # | decision today | mask-only rule | what the raw adds | E14.5 size | cost / query | value |
|---|---|---|---|---|---|---|
| **P1** | **Bridges across background** — `MaskSupport.accepts_support` (≤ `bridge_max_background_gap_um` = 2 µm, `bridge_min_mask_fraction`), used by `MaskBridges` (gap / optimise / orphan reconnect), `connect_skeleton_components(mask_support=)`, the bundle hub links, closing | a short dark gap is a dropout | **confirm**: the gap's raw must be lumen-bright, else it is a wall between two vessels and the bridge is refused however short; **extend** (later): a bright gap may be crossed beyond 2 µm | hundreds of candidates per build (most bridges later merge into chains; 39 `recovered` edges survive on the saved graph) | one path of 10–50 samples, one window | high: the one rule every joiner shares |
| **P2** | **Facing dead ends** (`facing_ends`, ≤ 10 µm, one stretch of background) and the parked tip-to-side join | one stretch, turn ≤ 60°, not beside a vessel | the gap must be bright along the chord; with that the gap limit can rise (Kaufhold's threshold relaxation, which `PLAN_dead_ends` Phase 6 asks for) | 284 pairs within 10 µm, 15 joined | one chord | medium; shared with the dead-end plan (section 5) |
| **P3** | **Doubled runs: one lumen or two** — `shadowed_samples` (`bridge_mask_support`), `doubled_stretches` (clean-graph) and `remove_parallel_edges_in_lumen` | chord crosses ≤ 1 µm background → same lumen | the raw dip along the chord: two lumens have a dark wall between them, one does not | 1,045 runs; **30 already marked (grid H)** | one chord per sample, memoised per run as the pair memo is | **highest**: the only untried measure on a grid that no mask measure passed |
| **P4** | **Short loops' holes** — `lumen_loops`, review list "Short loops" | ≥ half the antipodal chords in the mask → one lumen | the loop's in-plane hole is bright (a pinhole in the segmentation → false loop) or dark (tissue → real) | 242 short loops after clean-up; 2,438 on the raw skeleton graph | one in-plane patch of ≤ (loop span)² | high (2c of the clean-graph plan, given to this plan) |
| **P5** | **Pinholes plugged before thinning** — `segmentation_cleanup` | the 1-voxel closing merged 235 components | fill only the tunnels whose interior is bright | 1,277 loops inside one lumen on the raw skeleton graph | per candidate tunnel, its bounding box | medium–high, **only after P4's rule holds** (clean-graph D4) |
| **P6** | **Uncovered mask regions** — `mask_recovery.uncovered_mask_regions` before tracing | volume ≥ `recovery_min_region_volume_um3`, width > 1.5 voxels | region median brightness vs the vessel level: a dim region is halo or a speck, not lumen, and is not traced (fewer of recovery's 172 dead ends) | a few hundred regions | one bounding box | medium; needs grid N — G4 warns dextran inside mask spreads 0.2–2.2 |

### 2.3 Assessed and rejected (stay mask-only)

- **The stub prune ("hair or vessel") and the `unknown` dead ends**: G4 — dextran contrast on a
  mask-covered chain is no measure. Grid L (raw panels for `unknown`) belongs to
  `PLAN_dead_ends` / `PLAN_clean_graph` 3e and is not duplicated here.
- **Skeletonisation itself** (Lee thinning, the thick-vessel EDT ridge): a plasma lumen is flat
  bright, so its intensity ridge is not the centreline; an intensity-weighted thinning is a
  whole-volume operation (RAM and time) for no gain the grids support.
- **Cluster collapse, degree-2 merging, branch orders**: geometry, not evidence.
- **Whole-volume raw operations in the build** (Otsu foreground + labels as Check segmented
  image does: 5 bytes a voxel, 80 GB on the 16.4 G-voxel muscle stack; a whole-volume blur): none.
  Every read below is a window round a question.
- **A second channel** (claudin-5 scored best on G4): D5, one channel only. *Superseded by D0
  (2026-10-13): there is still one channel, and on E14.5 it is claudin-5, read by
  `RawEvidenceEndothelial`.*
- **ilastik probability maps** as a soft mask: a different input, not the raw channel; out of
  scope unless Harvey asks.

---

## 3. Design: `RawEvidence`, one reader, sampled, windowed

New module `preprocessing/raw_evidence.py`, the raw-image sibling of `MaskSupport`, pure and
napari-free, so every rule asks the same object and nothing reads the raw volume by itself.

*As built (`87a317e9`, `96b0329a`; the wall primitives `bafd3aa2`), three things below changed.
The settings adapter is gone: `raw_image_for_build` reads `raw_image_path` / `raw_image_channel` /
`raw_image_stain` only, since the retired names migrate through `Schema.upgrade`. Both stains are
read (`RawEvidencePlasma`, `RawEvidenceEndothelial`, D0 and D2 below), not plasma alone. And the
primitives are `signal_along`, `wall_between` (was `dip_between`) and `hole_is_lumen` (was
`hole_brightness`), each answered in its stain's own terms. No `keep_dtype` loader was needed: the
in-RAM path keeps the file's dtype, the memory-mapped one is float32.*

- **Holds** the one raw channel (`raw_image_path` / `raw_image_channel` once the raw-input branch
  lands; until then an adapter `raw_image_for_build(settings)` that also reads
  `fwhm_raw_tiff_path` / `fwhm_raw_channel`, so the two branches merge without a rename), cut by
  the run's `CropBox` (`io.crop.crop_volume_like_graph`), shape-checked against the mask exactly
  as `load_and_validate_vessel_masks` checks a mask, **in its native dtype** (a uint16 memmap is
  2 bytes a voxel; `load_single_channel_tiff_volume`'s float32 copy is 4 — on the muscle stack
  66 GB). Ask the raw-input session for a `keep_dtype=True` on their loader rather than add a
  second one.
- **Calibrated once per run** (≤ 2 s): the *vessel level* (median raw at ~2,000 sampled mask
  voxels with lumen radius ≥ 1.5 µm, read through `MaskSupport.radius`) and the *background
  level* (median at ~2,000 sampled voxels ≥ 3 µm from the mask, as `fwhm_decoys._clear_of_vessels`
  finds vessel-free tissue). Extends `read_raw_stain`'s block sampling rather than duplicating it:
  one `RawCalibration` dataclass on `G.graph["raw_evidence"]` beside the stain reading, so the
  saved graph says what the rules read. Brightness is `(raw − background) / (vessel − background)`,
  so every threshold below is in those units and transfers across stacks.
- **Reads windows, never the volume**: a query takes the box round its points plus a 1-voxel
  blur margin (`IntensityCostField.__getitem__` is the pattern: crop, blur the crop, use it),
  trilinear samples at the asked points. An LRU of the last few windows only if the harness shows
  repeated reads (P3 asks about neighbouring samples; the pair memo already makes each run a
  single question).
- **Three primitives**, each returning numbers the grids can be keyed on:
  - `signal_along(path_um) -> (fraction_bright, longest_dark_um)`: the raw analogue of
    `BridgeMaskSupport` (P1, P2, P6's joins);
  - `dip_between(a_um, b_um) -> min_brightness_over_end_brightness` along the chord (P3);
  - `hole_brightness(loop_um) -> median brightness of the loop's in-plane hole over its
    vessels'` (P4), in the loop's best-fit plane, the hole being the polygon interior minus a
    lumen radius round each side.
- **Stain**: v1 supports `plasma_column`. With `raw_image_stain == endothelial` (lumen dark,
  wall bright) `RawEvidence` is off with one log line and every rule runs mask-only; decision D2.
  *Superseded: D0 made E14.5's raw image claudin-5, and `RawEvidenceEndothelial` reads a wall
  stain (a vessel is a ring round its centreline, a wall between two lumens a bright line).*
- **Off entirely** when the run has no raw image (the Nerve fixture) or the raw does not align:
  every caller takes `raw: RawEvidence | None` and behaves exactly as today on `None`.

### RAM and time budget (measured by the harness, raw on against off, one commit)

| | E14.5 | muscle stack (16.4 G voxels, memmap) |
|---|---|---|
| extra resident memory | ≤ 50 MB (windows + calibration samples; the memmap pages are the OS's) | ≤ 200 MB |
| calibration | ≤ 2 s | ≤ 10 s (sampled, slab-local reads) |
| all queries in one build | ≤ 5 % of build time (≤ 13 s of 253 s): ~3,000 windows of 10³–10⁴ voxels at well under a millisecond each | ≤ 5 %, with windows read slab-contiguous |
| whole-volume passes over the raw | **0** | **0** |

A query count and the raw read time go in the build log and the harness REPORT as columns, so a
regression is a number, not an impression.

---

## 4. Phases

### Phase 0 — `RawEvidence` and its calibration (no behaviour change; can start now)

- The module, the adapter, the calibration on the graph, the three primitives.
- Tests (`tests/test_raw_evidence.py`): synthetic volumes with known answers — a bright tube
  broken by a dark 3 µm gap (`signal_along` dark run = 3 µm); two parallel tubes with a 1-voxel
  dark wall (`dip_between` < 0.5) against one wide tube (≥ 0.9); a torus with a bright pinhole and
  one with a dark hole (`hole_brightness`); a uint16 memmap read window for window (same answer as
  in-RAM, no whole-volume read: asserted by wrapping the memmap's `__getitem__`); the crop box
  applied; shape mismatch refused; `None` passthrough on every caller.
- Harness columns: raw queries, raw seconds, peak RSS with and without the raw image.
  *Done (2026-10-25, `8204478d`)*: `raw_evidence_from_settings` logs the seconds to load and
  calibrate and the RAM the channel holds (`resident_bytes`); `scripts/graph_artefacts.py
  --raw-cost` builds each side twice, without and with the raw image, and its report tabulates
  build seconds, peak resident memory, the raw step's seconds and RAM, and the windows the
  build's rules read (`report.raw_cost_table`). The evidence is made after the graph and asked
  nothing during the build, so the windows read are 0 until a rule reads it. Measured on E14.5
  with `build_network` itself: see the Status table.

Size S. Files: new module and test only; `stages.build_network` makes the object (one call), and
`lumen_cleanup` carries it (one optional argument).

### Phase 1 — Bridges and facing ends (P1, P2) — after both sessions land

- `MaskSupport(raw=...)`: `accepts_support` also asks `signal_along` over the background
  stretches of the route; `BridgeMaskSupport` gains `raw_fraction_bright` and
  `raw_longest_dark_um`, written on the bridge (`bridge_raw_bright`) beside
  `bridge_background_um`, so the review lists and the grids can read them.
- Setting `bridge_raw_confirmation` (advanced, under "Require mask support"):
  `off` | `confirm` (a dark crossing is refused however short) | `extend` (a bright crossing may
  run to `bridge_raw_max_gap_um`, default 6 µm). `facing_ends` reads the same two.
- **Grid M** (30 bridge and facing candidates, spread over `raw_fraction_bright`, shuffled, mask
  and raw side by side, each in its own plane): marks **V** the vessel continues across the gap /
  **N** two vessels or nothing. Fit the brightness threshold; fresh grid M2.
  **Built 2026-10-13** (`scripts/bridge_review/build_grid_m.py`, key and candidates under
  `outputs/bridge_review/`, page https://claude.ai/artifact/Td1WywPYq56sPP1DHTAF6f): from the
  `clean_graph_phase1_fast2` build, 352 facing pairs within 15 µm, 166 dead ends with a vessel
  ahead and 6 bridges across background; 14 + 12 + 4 shown, spread by value over
  `raw_gap_bright` (the claudin-5 reading on the chord's background samples: a third dark, a
  third middling, a third bright, since 69 % of all candidates read bright). The endothelial
  `vesselness` became the best pair of opposite wall directions, not the brightest single
  point, which any wall within reach satisfied in a capillary bed.
  **Marked 2026-10-13 (Harvey: 24 N, 5 V, one unmarked; `outputs/bridge_review/FIT_M.md`): no
  measure beats all-N (24/29).** The five V gaps span the raw reading's whole range; every
  bridge drawn across background and 12 of 13 facing pairs are N; 4 of 12 dead ends with a
  vessel ahead are V, where only the mask's turn (≤ 48°, 9/12 against 8) is above chance. So
  `confirm` has nothing to confirm and `extend` nothing to extend on this stack: P1 and P2
  ship nothing from grid M. Grid M2 (fresh candidates, same recipe) checks the ahead signal.
  **M2 marked the same day (24 N, 5 V, `FIT_M2.md`): the same result on 30 fresh candidates.**
  No measure beats all-N; the ahead turn scores 10/12 (M's rule unchanged, 10/12 there too;
  pooled 19/24, 79 %), under the bar. **Phase 1 closed, nothing shipped**: the mask rules'
  refusals stand, and the ahead turn ranks the review list only. *That ranking was built for the
  dead-end list by the dead-ends session (`368107db`, 2026-10-25) and taken out again
  (`6d335f88`) once the fresh grid G3′ scored the turn 21/30 against 29/30 for never joining
  (Harvey's marks, 2026-10-25): it stays a measure on each item, not an order.* So
  `MaskSupport(raw=)`, `BridgeMaskSupport`'s raw fields and the settings
  `bridge_raw_confirmation` and `bridge_raw_max_gap_um` were never built: closed by design.
- Default when the raw image is present: `confirm` ships on at ≥ 90 % on M2, else off and the
  verdict only ranks the review lists; `extend` only after its own fresh grid (decision D1).
- Kill criteria on the harness: components and interior dead ends may not rise; build time
  ≤ +5 %.

Size M. Tests: `test_bridge_mask_support.py`, `test_facing_ends.py`, `test_skeleton_bridging.py`.

### Phase 2 — Doubled runs: one lumen or two (P3) — with the clean-graph session

**Outcome (2026-10-12, `scripts/doubled_run_review/rescore_raw.py`, branch `claude/raw-evidence`
at 48ccd51; reading in `outputs/doubled_run_review/FIT_raw.md`): the raw dip does not separate
the marks.** Grid H: best wall measure 17/30 leave-one-out against 19 all-D (mask's best 21/30;
the run's own brightness, not a wall measure, also 21/30);
most T runs show no dip at all, at a one-voxel blur, half a voxel or none. Grid H2: 17/30
against 16 all-V. The wall between two touching capillaries is under a voxel wide and the
image's own blur has removed it. So P3 ships nothing, the `keep_apart` hook stays unfilled, and
the doubled-run question stays with the mask rules. What follows is kept for the record.
*The hook exists (`doubled_stretches.resolve_doubled_stretches(keep_apart=)`,
`lumen_loops.remove_parallel_edges_in_lumen(keep_apart=)`, `assemble.lumen_cleanup(keep_apart=)`)
and nothing fills it; `lumen_cleanup(raw=)` was never added. Closed by design.*

**Phase 0 also found what E14.5's dextran channel is like** (median 3 counts inside the mask,
0 outside): levels are read under a one-voxel blur, the lumen core is capped at 3 µm deep, and
every wall or hole measure is judged against the chord's own ends or the loop's own vessels,
since a wide or shallow vessel is three times the run's level. Those are in the module now.

- **Free first step**: re-score two grids' existing marks with `dip_between` (no new marks
  needed; the keys in `outputs/doubled_run_review/` have every run's samples): **H** (30 runs,
  19 D / 11 T, `marks_H.txt`, `H_key.csv`; best mask measure 21/30) and **H2** (30 dead-end
  chains a wider "beside" test would trim, 16 V / 14 X, `marks_H2/annotations/*.json`,
  `H2_key.csv`; no mask measure beats 16/30 = all V). H2 is the 588 µm of E14.5 doubling that
  only the wider vessel's side sees, which the clean-graph finder leaves to the review list. If
  the dip beats the mask on either, a fresh grid (H3) decides whether it ships.
- The clean-graph session adds `keep_apart: Callable[[DoubledStretch], bool] | None = None` to
  `doubled_stretches.resolve_doubled_stretches` (a run is skipped when it returns True) and the
  same name, taking the pair's two edge keys, to `lumen_loops.remove_parallel_edges_in_lumen`,
  threaded through `assemble.LumenCleanup` / `lumen_cleanup(keep_apart=...)` so `build_network`
  can pass ours (agreed 2026-10-12). This plan supplies the function
  (`RawEvidence.dip_between` over the run's samples, memoised by run like the pair memo). A run
  the raw says is two vessels is left — the 11 T of grid H stop being "the cost". One owner per
  file: they add the hook, we fill it (decision D5).

Size S–M. Tests: `test_doubled_stretches.py` (one synthetic D, one T), `test_lumen_loops.py`.

### Phase 3 — Loops' holes (P4), then pinholes (P5)

- **Grid I** as `PLAN_clean_graph.md` 2c describes it, built here: 30 short loops stratified by
  `hole_brightness`, raw and mask side by side.
  **Built 2026-10-13** (`scripts/loop_review/build_grid_i.py`, key and all 222 candidates under
  `outputs/loop_review/I_*.csv`, page https://claude.ai/artifact/KHmxZmiPkQLXoppdFqQHbo): the
  `clean_graph_phase1_fast2` build's 222 short loops (217 with a hole wide enough to read),
  21-191 µm round, none inside one lumen by the mask rule, 30 spread by value over
  `raw_hole_lumen` (the claudin-5 reading: no wall lining the hole), each in its own plane.
  Marks C (real, keep) / F (false, remove a side).
  **Marked 2026-10-13 (Harvey, mostly on the claudin-5 channel: 27 F, 3 C; `FIT_I.md`): nine
  in ten short loops are false**, and the mask rule found none of them inside one lumen. No
  measure, mask or raw (the hole reading, or the walls crossed on antipodal chords,
  `rescore_chords_i.py`), finds the three real loops (52, 177, 179 µm): all one way gets 27/30
  and nothing beats it leave-one-out. So the raw image does not ship a loop rule either; what
  the grid changes is the question. Decision for Harvey: remove a side of every loop under
  ~170 µm (27/30 right here, one real loop lost), checked first on a fresh grid I2, or keep
  the review list as the mechanism. 2d (pinhole plugging) stays off: 2c found no rule.
  **Grid I2 built 2026-10-13** (Harvey: "build grid I2"; page
  https://claude.ai/artifact/WqzDwLr3zzobASUMCejd1U, `I2_key.csv`): 30 of the 192 loops grid I
  did not show, 23-193 µm round. Grid I had used every loop whose hole read bright, so I2 is the
  18 middling readings and 12 dark ones: the test is of the nine-in-ten base rate, not of the
  raw measure, which found nothing on I.
  **I2 marked the same day (28 F, 2 C; `FIT_I2.md`): the base rate holds.** Pooled 55 of 60
  short loops are false; the five real ones are 46, 52, 172, 177 and 179 µm. No measure finds
  them (all-F is the ceiling on each grid and on the pooled 60), and no length cut beats
  all-F. The rule the marks support is the trivial one, a side of every loop under 200 µm
  removed (92-93 % right on both grids, one real loop in twelve lost), which side being the
  loop breaker's existing choice, not verified by these marks. Decision D8 for Harvey: ship
  it as a graph setting (`remove_short_loops_under_um`, with the Short loops review list
  showing what went), on or off by default.
  **D8 decided 2026-10-13 (Harvey: "ship it on by default")**: `lumen_loops.remove_short_loops`
  runs in every `consolidate_lumen` round after the in-lumen step, `remove_short_loops_under_um`
  = 200 µm (0 off), advanced under "Require mask support"; branch `claude/raw-evidence` at
  5bd4c5b, tests in `test_lumen_loops.py` and `test_consolidate_lumen.py`. Measured on the
  harness against the tip without it (`outputs/short_loops_rule/`) before landing, with the
  kill criteria of Phase 1.
  **Measured 2026-10-13 (E14.5, tip → rule on; a3188f1 adds a guard leaving loops through a
  thick-vessel bridge alone, which changed nothing):** short loops 222 → 1, edges 2,541 →
  1,823, centreline 47.3 → 41.7 mm, thick-vessel bridges 110 → 78, dead ends 646 → 648,
  components 156 → 157, build time no slower, **mask covered 75.4 → 72.5 % (−2.9 points,
  past the half-point criterion written for the stretch step)**. The bridges go before the
  split: a loop inside a fat vessel's lumen loses the strand the split would have bridged.
  Harvey saw the coverage cost and said "commit and push" (2026-10-13): landed on at 200 µm
  (`5bd4c5b3`; the thick-bridge guard `a3188f12`).
  **Grids Q and Q2 (Harvey, 2026-10-15, `7cc17cff`; `outputs/loop_break_review/FIT_Q*.md`)**: 60
  of the rule's breaks on E14.5, each drawn with the side it took. Where a side should go it took
  the right one every time, but loops round a hole over 10 µm were real 29 of 32, and breaking one
  merged it with its neighbour into a longer loop broken in turn. So a loop round a hole wider than
  `short_loop_keep_hole_over_um` (10 µm, the median background its antipodal chords cross,
  `lumen_loops.loop_hole_um`) is left to the Short loops review list: 53/60 leave-one-out against
  33 all one way, fitted on Q (9.2 µm); on Q2 alone 23/30, the same as all one way. A mask
  measure, not the raw image's.
  The raw hole reading ships nothing: on I and I2 it did no better than all F. The fallback this
  phase named ("below 90 % it ranks the Short loops list and the Post processing page shows the
  raw panel") is not built; the Status table says why.
- **P5 only after I2 holds**: `segmentation_cleanup_plug_pinholes` (off by default, clean-graph
  D4): the voxels a 1-voxel closing would add, labelled slab by slab (`memmap_support`), each
  component kept only when its raw median brightness ≥ the fitted threshold; mask components may
  not merge (asserted), measured by the raw skeleton graph's in-lumen loops (1,277) and short
  loops (2,438). **Closed by the marks**: it needed a raw hole threshold fitted on I and held on
  I2, and no raw measure beat all F on either (`FIT_I.md`, `FIT_I2.md`), so there is no threshold
  to plug by. Nothing was built.

Size M. Tests: `test_lumen_loops.py`, `test_segmentation_cleanup.py`.

### Phase 4 — Uncovered regions (P6) — grid UR1

- Grid N: 30 uncovered regions spread by region brightness; marks **V** lumen the graph missed /
  **H** halo, speck or wall. Ships as a skip rule at ≥ 90 % on N2; else the "segmented vessels
  without a centreline" review list carries the brightness and is sorted by it.
  *Renamed **UR1** ("uncovered regions"), its fresh check **UR2**: grids N and N2 are the
  surface-coverage plan's (2026-10-18, lost skeleton), a different question.*

Size S. Tests: `test_mask_recovery.py`.

**UR1 built and sent (2026-10-25, `8204478d`, `scripts/uncovered_review/build_ur1.py`):**
https://claude.ai/artifact/JTxvz6NZFub5UNCtUspLta (marks save to the page's `annotations`
collection: `verdict` V / H / ?, `evidence`, `note`). Key, candidates and calibrations:
`outputs/uncovered_review/UR1_*` in worktree `plan-raw-confirmation-0887be` (git-ignored); the
E14.5 graphs it was built from, and the scripts that made them, in `outputs/raw_confirmation/`
beside them.

- **Data**: E14.5, Harvey's Claudin segmentation (the thick-bridges session's capture of
  2026-10-24, `plan-thick-bridges-e32666/outputs/tb/base/finish_input.pkl`), its graph built with
  this code (1,241 nodes, 1,417 edges), the composite's dextran (channel 0) and claudin-5
  (channel 1). The segmentation came from claudin-5, so dextran is not the mask's own channel here.
- **Pool**: every piece mask recovery is handed -- the graph after
  `reconnect_orphan_and_dangling_nodes`, `uncovered_mask_voxels` in 26-connected pieces of at least
  30 µm³ and wider than a voxel and a half: **1,749**. Recovery draws a centreline into only
  **39** of them (113 edges, and 60 joins to the network). Those 39 are all a skip rule could change; 30 of them
  would have left no fresh grid, and the other 1,710 ask the review list's question. In the
  finished graph a median 78 % of a piece's voxels are still uncovered.
- **Spread**: 10 of the 39 and 20 of the rest, each group split at its median depth and each half
  spread evenly across the dextran brightness over the centrelines' within 25 µm (the plan's
  measure read locally), then shuffled. The 30 run 2-285 µm deep and 31-198,000 µm³.
- **Measures in the key** (none on the page): dextran median, deepest-points median and over the
  neighbours'; claudin-5 median; E2 (`inside_walls`) at the five deepest points, lumen / wall /
  shared; the wall either side of the drawn centreline (`vesselness`); volume, widest radius,
  depth, elongation, distance to the nearest centreline, what recovery drew, the share the
  finished graph leaves uncovered.
- **What the pool already shows**: dextran brightness is mostly depth. The top 30 µm (509 pieces)
  reads a median 1.27 of the run's level (10th-90th percentile 0.59-2.38); every 30-300 µm band
  reads a median 0.31-0.41 (90th percentile ≤ 0.65). A global threshold would be a depth cut (the
  depth risk below); read over the neighbouring centrelines, the medians run 0.42-0.61 in every
  band, hence the local reading. E2 reads "on a wall" at the deepest points of almost every piece
  (median wall share 1.0; no lumen at 1,724 of 1,749), as it did at dead ends, so it is unlikely
  to separate them on this stack.
- **Page**: the piece outlined in yellow in its own plane and across it, in the segmentation,
  dextran and claudin-5; the centreline recovery draws, if any, in cyan; the graph's centrelines
  as grey dots; each card names the piece's centre voxel. One question: V (a vessel of its own the
  graph missed) / H (not a vessel of its own: a halo, a speck, a wall, or the side of a vessel
  already drawn) / ?.
- **After the marks**: fit every key measure against V/H, right / wrong / all one way, with the IDs
  missed, overall and on the ten recovery draws into. A measure at ≥ 90 % is fixed as a rule and
  checked on UR2 (fresh pieces, same recipe) before any skip rule goes into
  `mask_recovery.py`. Failing that, the review list carries the best measure on each item
  **[later]**, sorted by it only if the marks show it ranks V above H (a measure that fails stays
  a measure, not an order, as the dead-end turn did).

**UR1 marked by Harvey and fitted (2026-10-25, `c5bc6ad2`, `scripts/uncovered_review/fit_ur1.py`;
report `outputs/uncovered_review/FIT_UR1.md`, marks in `marks_UR1/`, worktree
`plan-raw-confirmation-0887be`):**

- **The marks**: 26 H, 4 V (UR1-12, 13, 24, 26). "What decided it" is dextran and claudin-5 on
  all 28 with a tick, segmentation shape on 26; no notes. Nine in ten uncovered pieces are not a
  vessel of their own.
- **Every piece (30)**: no measure beats marking all H leave-one-out (26/30). Claudin-5's median
  above 6.7 and dextran's above 2.63 fit 27/30 but fall to 26 and 25 left out.
- **The ten recovery draws into**: 2 V, 8 H, so recovery's centrelines are mostly drawn where
  Harvey sees no vessel of its own. The dextran over the neighbouring centrelines' above 1.05
  separates them 10/10 leave-one-out (all H 8/10); depth ≤ 13 µm 9/10. Two V is too few to trust.
- **The review list (the [later] item) is closed by the marks**: on the population it lists, no
  measure ranks V above H, and 26 of 30 are H, so there is nothing to carry or sort by.
- **The skip rule's lead needs a channel the run does not read.** It is a dextran reading; under
  D0 E14.5's raw image is claudin-5, so `RawEvidence` would hold the wall stain, not the plasma
  one. If UR2 confirms it, a skip rule runs only where the run's raw image is a plasma label, or
  the run reads a second channel (D5 said one). For Harvey.

**UR2 built and sent (2026-10-25, `c5bc6ad2`)**: https://claude.ai/artifact/NdhbUg7HEzj9eiLKrrg28R
(marks in `annotations`), key `outputs/uncovered_review/UR2_key.csv`. Every piece recovery draws
into that UR1 did not show: 29, the same graph and page, a new shuffle. Two rules fixed before the
marks (`fit_ur1.FRESH_RULES["UR2"]`, committed before the page went out): **V when the dextran
over its neighbours is above 1.05** (it calls 3 of the 29 V: UR2-01, 03, 25), and **every piece H**,
which is mask recovery drawing nothing. Either at ≥ 90 % on UR2 goes to Harvey: the first as a skip
rule (with the channel question above), the second as `recover_uncovered_mask_vessels` off by
default, which needs no raw image.

### Phase 5 — Docs and the review

CLAUDE.md (one bullet for `RawEvidence`, one line each under the rules it feeds), the harness
columns, `RELEASE_REVIEW.md` rows for graph accuracy and memory with the evidence, and the
`.haemorun` carrying the calibration.

**Done (2026-10-25)**: `raw_evidence.py` is in CLAUDE.md's Repository layout, and the
`lumen_loops` entry names the grids it failed; it feeds no rule, so there are no per-rule lines.
The module's docstring no longer says rules "will land one at a time": it says none reads it and
why. The harness columns are Phase 0's (`8204478d`). `RELEASE_REVIEW.md` `graph.6b` cites the
fits of H/H2, M, M2, I and I2 and grid UR1, `graph.8b` M's ahead turn on M2 and the short-loop
rule on I2, and `memory.6a` the raw image's cost on E14.5; no score moves. The `.haemorun`
carries the calibration already, as `G.graph["raw_evidence"]` on the saved graph.

Order: 0 → (both completion reports) → H re-score (free) → 1 and grid M → 2 → 3 → 4 → 5.

---

## 5. Boundaries with the other plans

- `PLAN_dead_ends.md` keeps dead-end *kinds*, trace-on, tip-to-side joins and grid L. This plan
  supplies `RawEvidence` for them to call (Phase 6's "threshold relaxation" is P2's `extend`) and
  changes `facing_ends` only through the shared bridge rule.
- `PLAN_clean_graph.md` keeps the rewiring; 2c and 2d move here (Phase 3), and P3 is a hook in
  their module.
- The raw-input branch owns the settings, the loader and the stain; this plan asks it for
  `keep_dtype` and reads its three settings through the adapter.
- Peers' files are edited by targeted replacements only, in a worktree, rebased and re-measured
  before landing (shared checkout).

---

## 6. Decisions for Harvey

| # | question | recommendation |
|---|---|---|
| D0 | **Which channel is E14.5's raw image** (decided 2026-10-13, Harvey: "land the branch and use the endothelial channel"). D5 of `PLAN_clean_graph.md` (one channel, the one the segmentation came from) is overturned for E14.5: the run's raw image is the composite's claudin-5 channel (`raw_image_channel` 1, stain `endothelial`), since the dextran channel cannot resolve a wall between touching capillaries (Phase 2's result) and claudin-5 was the only measure above chance on G4. Phase 0 landed on `Devel_GUI_improvements_HD` at 96b0329 the same day; grid M is built on that channel. | Decided. |
| D1 | What the raw does to bridges by default when the image is present: `confirm` (refuse dark crossings, nothing else changes) or `extend` (allow longer bright gaps). | `confirm` first; `extend` only after its own fresh grid, since it adds vessels. |
| D2 | Runs whose raw channel is an endothelial stain. | `RawEvidence` off in v1 (mask rules as today), logged; a wall-aware reading is a later phase if such runs matter. *Superseded by D0: `RawEvidenceEndothelial` reads a wall stain, and E14.5's grids M, I, I2, DR1 and UR1 were built on it.* |
| D3 | Pinhole plugging changes the segmentation before thinning. | Off by default, only after grid I2 holds (as the clean-graph plan's D4 already says). |
| D4 | Grid order, ~150 marks in all. | H re-score (no marks) → M → I → N; H3, M2, I2, N2 as each passes. *Done in that order; N became UR1 (sent 2026-10-25), and H3 was never needed (H's re-score found nothing to check).* |
| D5 | Who owns P3's code. | The clean-graph session adds the hook in `doubled_stretches`; this plan fills it. |
| D6 | Budgets: +5 % build time, +50 MB resident on E14.5, zero whole-volume raw passes. | Yes; measured by the harness columns of Phase 0 at every phase. *Measured 2026-10-25 at `8204478d`: +0.6 % time, +1 MB of peak, no window read (the Status table); the muscle stack waits on its raw image.* |
| D7 | Start Phase 0 now, before the two completion reports. | Yes: it touches no file either session has open. |

---

## 7. Risks

- **The raw says "bright" across a gap that is two vessels touching end to end** (a wall thinner
  than the PSF). Grid M measures it; `confirm` never adds a bridge the mask rule would not, so
  the risk is only in `extend`.
- **Depth**: dextran contrast falls with depth (Cassot's pattern, `PLAN_dead_ends` 2.1), so one
  global threshold may be wrong deep in the stack. The calibration keeps vessel and background
  levels per z-slab (16 slabs, as `read_raw_stain` samples) and normalises locally; the grids are
  stratified by depth.
  *Not built (2026-10-25): `RawCalibration` keeps one level per run. A per-slab calibration only
  matters to a rule that reads brightness against a threshold, and no rule reads the reader; the
  measures already fitted (M, M2, I, I2, DR1) judge against their own neighbourhood, a chord's
  ends or a loop's vessels. UR1 shows the risk is real on E14.5's dextran (the top 30 µm reads
  about three times the depths), so its key carries both the global and the local reading and
  its depth, and the grid is split by depth. Build the per-slab levels when a measure that needs
  them passes UR1, or another grid.*
- **Memmap page-ins on a disk-backed stack** make windows slow when the box spans many z slices.
  Windows are capped at the question's reach (a chord, a gap, a loop span); the muscle-crop budget
  is measured before any rule ships there.
- **Two branches renaming the same settings**: the adapter reads both names until the raw-input
  branch lands; its migration then owns them.
- **Marks disagree with the measure**: as with every grid, the measure changes, not the marks; a
  rule below 90 % ranks a review list and ships nothing.
