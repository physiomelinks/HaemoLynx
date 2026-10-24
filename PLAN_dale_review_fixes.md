# Plan: fixes from the Dale branch review

Checked on `Devel_GUI_improvements_HD` at 9d247ed, 2026-10-09. All five findings hold. Items 2,
4 and 5 turned out wider than the review said, and item 1 needs a modelling decision before any
code changes.

**Status, 2026-10-24: all five items are done.** Item 3 is in 9c6a15ad, item 2 in b99758ba,
item 5 in b6d9210d and 666cafe8 (the pieces still missing landed on 2026-10-24 in 3bfdbcef,
c990f2ca, 88e97182 and da6ca014), item 4 in b71718b5 and item 1 in 3e25f87b. Item 2's E14.5
before/after was run on 2026-10-24 (see item 2). Nothing waits on Harvey.

**Order:** 3, 2, 5, 4, then 1 once it is decided. Use one commit per item, each with its tests
(see the Testing policy in CLAUDE.md). After each commit, run `pytest -m "not slow and not gui"`
(about 13 min). Run the `gui` subset in the background before pushing. Don't edit `src/` while a
suite is running. Other sessions share this checkout, so commit through a private index.

---

## Item 3: voxel-loop detection (`graph/build.py`): delete it

**Status, 2026-10-09: done, in 9c6a15a (pushed to `Devel_GUI_improvements_HD`).** Both functions now return the graph alone. The
fixture graphs are identical before and after, both after the first build step and at the end.
Regression tests are in `tests/test_graph.py`.

**What it does.** The block is build.py:303–350. For every skeleton voxel it adds a voxel-graph edge
to each voxel at one of the offsets at :306, then `nx.cycle_basis` (:331) lists the graph's cycles.
`loop_vox` is every voxel on a cycle. `loop_edges` (:408) is every skan segment whose two end
voxels are both in `loop_vox`. It has been there unchanged since the first package commit (56bf978,
2026-02-18). Dale's branch tried to make it faster twice (ff1acf5, 64ac7ea) but kept it.

**The bug.**
- `generate_binary_structure(3, 1)` includes the centre, so `(0,0,0)` is one of the offsets, and
  every voxel gets an edge to itself.
- NetworkX reports each self-loop as a cycle, so every skeleton voxel lands in `loop_vox`.
- Every segment starts and ends on a skeleton voxel, so `loop_edges` ends up holding every segment.
- Measured: E14.5 gives 73,726 "loops", 72,431 of them single voxels, one per skeleton voxel.
  `loop_edges` has 20,514 pairs, every one already joined by an edge. The fixtures show the same.

**Why removing it changes nothing.** This holds for the buggy block and for any correct one.
- `loop_edges` is read in only one place: the gap-bridge candidate filters (build.py:425 and :439,
  optimise.py:82 and :100). There it stops two dead ends being bridged when `loop_edges` holds their
  pair.
- Because `loop_edges` is built from segment endpoints, it can only ever hold pairs that a segment
  already joins.
- Between the build and optimise, nothing removes edges. The build's bridging only adds edges, the
  isolated-node cleanup (:502) only removes nodes that have no edges, and
  `reconnect_secondary_loop_edges` removes nothing.
- Every bridge is drawn only after the `G.has_edge(src, tgt)` check (build.py:464,
  optimise.py:125), and that check refuses those same pairs.
- So `loop_edges` can only exclude what `has_edge` already excludes, and that would still be true
  with a correct loop detector.
- `voxel_loops` itself is read by nothing. optimise returns it, and assemble discards it (:549).

**Proven, not assumed.** I ran `build_network` with the block on and off (`max_voxel_graph_size=0`)
and compared node ids, positions, edges, lengths and voxels exactly.

| Input | Skeleton voxels | Graph | Identical |
|---|---|---|---|
| seven-vessel fixture | 112 | 11 / 10 | yes |
| bundled fixture | 185 | 10 / 9 | yes |
| Nerve (test settings) | 1,792 | 40 / 43 | yes |
| Nerve (default settings) | 3,306 | 90 / 101 | yes |
| E14.5, `config.yaml`, saved skeleton | 72,431 | 2,578 / 2,725 | yes, also after the build and optimise steps |

**Cost.**
- The block only runs at or below `max_voxel_graph_size` (100,000 voxels).
- E14.5 (72,431 voxels) and Mason (83,517) are both under that, so every full-size build pays for
  it. That includes every distinct graph trial in Optimise settings (`optimisation/search.py`).
- Measured in isolation: E14.5 27.5 s with the block against 5.4 s without, so about 22 s of 26 s;
  Mason 21.4 s against 2.1 s; Nerve about 1 s.
- Inside the real E14.5 `build_network` the first step took 25.9 s against 4.2 s.
- The two whole `build_network` runs took 774 s and 448 s, but that gap is far bigger than the
  block and comes from run order and machine load. Don't quote it.
- Almost all of the time goes to the per-voxel Python neighbour loop (2.9 of 3.5 s on a
  17,000-voxel lattice), not to `cycle_basis`.

**Why not fix it instead.** A corrected block still couldn't change any graph, for the reason above.
It also wouldn't find vessel loops.
- A 6-connected version finds no cycles at all on any fixture, because a Lee skeleton steps
  diagonally.
- A 26-connected version finds voxel clumps: 55,894 of E14.5's 72,431 voxels lie on such cycles.
- Vessel loops are already handled elsewhere: `lumen_loops`, the Short loops review and
  `statistics/loops.py`.

**Steps**
1. Delete build.py:299–350 (`foreground`, the voxel graph, `cycle_basis`, `loop_vox`),
   `loop_edges` (:380, :408–409 and the two filters) and `max_voxel_graph_size`.
2. Drop `voxel_loops, loop_edges` from `optimise_graph_topology_fixed`, including its two filters
   and its return value.
3. `build_graph_segment_skan_stitched_loops` returns `G` alone, and `optimise_graph_topology_fixed`
   too. Keep both names: the first is in `STEP_LABELS`, so the progress bars and step artefacts
   use it.
4. Update `assemble.py` (:529, :549, the log line "loop detection + segment extraction") and about
   10 test call sites (`G, _, _ = ...`, `test_build_graph_requires_skan`). Both functions are in
   `graph.__all__`, but only internal code and tests call them.
5. Tests:
   - **Regression for the behaviour the filter claimed to give.** An isolated short segment whose
     two ends lie within `reconnect_threshold` of each other gets no second, parallel bridge. Test
     this with fewer than 10 dead ends (the branch with no early `has_edge` check, where only the
     final check protects it) and with more than 10 (the spatial-index branch).
   - A build never calls `nx.cycle_basis` (monkeypatched spy) and never builds a voxel graph.
   - A one-off check before committing: pickle the graphs from the current code first, then build
     them again with the new code and confirm they are identical. Use the fixtures and E14.5, as in
     the table above.

## Item 2: morphology erodes the image faces: pad, operate, crop

**Status, 2026-10-24: done, in b99758ba (2026-10-12), as planned below.**
`memmap_support.at_faces_as_beyond` pads with edge replication and crops back. It wraps
`close_binary_mask` (in RAM, and per block in the low-RAM path) and
`segmentation_cleanup._morphology`. `tests/test_morphology_at_faces.py` has each test listed below.

**E14.5 before/after, 2026-10-24.**
- **Set-up.** Run at c990f2ca with `config.yaml` (skeleton closing radius 1, segmentation clean-up
  off), on the low-RAM path, from segment to build_network. One run used `at_faces_as_beyond` as
  it is; the other replaced it with the identity, which is exactly the code before b99758ba. A
  second "after" run was identical in skeleton and graph, so every difference below comes from
  the change.
- **Skeleton.** 56,886 voxels before, 56,961 after. The 123 voxels gained and the 48 lost all lie
  within 5 voxels of a face (95 and 22 of them within 1).
- **Graph.**
  - 1,229 → 1,228 nodes and 1,407 → 1,406 edges; length 47,681 → 47,813 µm (+0.28%).
  - Dead ends: `leaves_image` 93 → 91 and `broken` 3 → 4; every other kind is unchanged. At the
    x = 0 face, two `leaves_image` ends became a short stop or a join once their face voxel came
    back, and one `island_end` became `leaves_image`.
- **Not only near the faces.** Of the 45 edges found only after, 20 come within 5 µm of a face and
  2 within 10 µm of a changed skeleton voxel. The other 23 are farther from both, up to 205 µm
  from a changed skeleton voxel. The 46 edges found only before split 19, 2 and 25. Two things
  show there:
  - Mask recovery traced one more piece of uncovered mask (47 against 46). Such a piece can run
    far from the face where its coverage changed.
  - Some vessels keep the same two ends but have a different middle: their centrelines are up to
    1.5 µm apart and their lengths 0.01–1 µm apart. Edge 1524–1564, 177 µm from the nearest
    changed skeleton voxel, is 23.93 µm before and 22.87 µm after. The build logs part at the
    first steps after the skeleton (collapse merged 3,713 against 3,714 nodes). Which step makes
    these far choices is not pinned down.

  This is graph building's sensitivity to a small local change, not something the face fix does
  wrong. Files are in `outputs/dale_item2_faces/` (git-ignored): `face_check.py`, `compare.py`,
  `far_changes.py`, `far_edges.py` and `report_before_after.json`.

**What I found, and it is wider than reported.** I used a tube of radius 3 voxels running face to
face along z (580 voxels in total).

| Operation | Default | Voxels lost at the faces |
|---|---|---|
| `close_binary_mask` (skeleton.py:605), radius 1/2/3 | **on**, radius 2 | 58 / 116 / 174: the whole cross-section of the first and last *r* slices |
| same, `use_memmap=True` | | the same: blocks get no halo past the array edge |
| `close_small_gaps` (segmentation_cleanup.py:1074) | off | 1 slice per face at 0.5–1 µm, 2 at 2 µm |
| `remove_surface_whiskers` (opening, :1029). **Not in the review** | off | the tube's end is rounded at each face: 16 voxels lost per face slice against 4 inside at 1 µm, and 24 then 8 at 2 µm |
| `_smooth_vessel_surfaces_morphological` (closing then opening, :1110). **Not in the review** | off | the whole face slice and most of the next: 29, 16 at 1 µm; 29, 29, 24, 8 at 2 µm |

**Cause.** scipy uses `border_value=0`, so erosion reads everything outside the array as background.
Closing's dilation is cut off at the face, so closing stops being extensive (it can remove voxels),
and opening puts an end cap on any vessel the face cuts.

**Fix.** Add one helper next to `map_blockwise` in `preprocessing/memmap_support.py`. Both modules
already import from there. The helper:
1. pads by the operation's reach with `np.pad(..., mode="edge")`,
2. applies the operation,
3. crops back to the original shape.

Edge replication treats a vessel cut by the image boundary as continuing straight past it. That
makes closing extensive again (it can no longer remove a voxel) and stops opening capping vessels
at the face. Dale pads with zeros, which is enough for closing but not for opening.

Apply the helper in `close_binary_mask` (both the in-RAM and blockwise paths) and in
`segmentation_cleanup._morphology`, which covers whisker removal, gap closing and morphological
smoothing. In the blockwise path, apply it per block. A block with a real halo at least `reach`
deep keeps the same core, so the blockwise result stays exact.

**Tests** (`test_skeleton_bridging.py`, `test_segmentation_cleanup.py`, `test_memmap_*.py`)
- A tube through two z faces, plus an oblique tube through x/y faces with anisotropic voxels, loses
  no voxel to any closing, both in RAM and blockwise.
- For the opening paths, the loss in a face slice equals the loss in an interior slice.
- Closing still bridges a one-voxel gap and still leaves two vessels farther apart than the radius
  unmerged, including when both run out through a face.
- Blockwise equals in-RAM on face-touching inputs.

**Expected impact.** Skeleton ends at the faces move outward by up to *r* voxels. That changes the
lengths of vessels leaving the image and the positions of face terminals and boundary nodes. Tests
that pin counts on face-touching fixtures may move. Check that each change is this face effect and
nothing else. Run the E14.5 before/after comparison (recipe in memory); only changes near the faces
are expected.

## Item 5: VTK export carries too little, and treats unsolved vessels as solved

**Status, 2026-10-24: done.** Steps 2 and 3 landed with the dead-end work in b6d9210d
(2026-10-12): the VTK's new arrays and field data, NaN where `graph.flow_is_undefined`, and the
perturbation edges CSV's blanks and `flow_solved` column; 666cafe8 added the units. An audit on
2026-10-24 found these pieces missing, now landed:
- **Step 1, the one list**, in c990f2ca. `graph.SOLVE_DERIVED_EDGE_ATTRIBUTES` (now with
  `discharge_haematocrit`) is what the VTK writes and blanks, what the edges CSV blanks
  (`stages.PERTURBATION_EDGE_ATTRIBUTES`) and what `gui/results.SOLVED_FLOW_COLUMNS` is built from
  (the same set as before, checked name for name). `tests/test_gui_unsolved_vessels.py` pins that
  the three agree. The VTK's `discharge_haematocrit` is now blank where the flows are. In
  da6ca014, a capillary block's `{name}_blocked_vessels.csv` reads the list too
  (`stages._solve_value`, shared with the edges CSV): it had written the solve's 0 as the flow of
  a blocked vessel no inlet and outlet reach.
- **The haematocrit model**, in 88e97182. Nothing recorded it. The solve stage now writes
  `G.graph["haematocrit_model"]`, and the VTK writes it to the field data (`vtk_io.RUN_FIELD_DATA`,
  which the export dictionary's text is built from).
- **Tests.** The edges CSV's blanks are tested in 3bfdbcef (`test_perturbation_outputs.py`, a
  real solve with a blind end and an island). The `discharge_haematocrit` array and the
  `diameter_basis`, `haematocrit` and `haematocrit_model` fields are tested in 88e97182
  (`test_vtk_io.py`, `test_haematocrit_distribution.py`).

Blanking only where the flow is undefined (`no_boundary`), rather than wherever `flow_solved` is
False, is deliberate. A blind end's flow of exactly 0 is the answer (`PLAN_dead_ends.md`, D4).

**What I found**
- `graph_to_vtk` (vtk_io.py:177) writes `edge_u/v/key`, `branch_order`, `resistance`,
  `conductance` and the five flow fields.
- It doesn't write `length`, `diameter_um`, `diameter_source`, `discharge_haematocrit`,
  `flow_solved` or `is_zero_resistance`, and it doesn't write which law and basis produced the
  resistances.
- Vessels with `flow_solved=False` are exported with whatever the solve left on them, which looks
  like ordinary data. The viewer blanks them using `gui/results.py`'s `SOLVED_FLOW_COLUMNS`.
- **Same problem, not in the review:** a perturbation's `<name>_edges.csv` (stages.py:3485)
  writes `flow_signed` and `pressure_drop` for unsolved vessels as well.

**Steps**
1. Move the list of solve-derived edge attributes into `graph/prune.py`, beside `FLOW_SOLVED`.
   Build `SOLVED_FLOW_COLUMNS` from it, so the viewer and every export blank exactly the same set.
2. In `graph_to_vtk`:
   - Add cell arrays `length_um`, `diameter_um`, `diameter_source` (string), `flow_solved` (uint8)
     and `is_zero_resistance`, plus `discharge_haematocrit` when any edge has one.
   - Write NaN into the solve-derived arrays wherever `flow_solved` is False.
   - Add field data from `G.graph`: `viscosity_law`, `diameter_basis`, the haematocrit and
     haematocrit model, and the HaemoLynx version.
3. In the perturbation edges CSV, blank the same columns and add a `flow_solved` column.
4. Tests:
   - `tests/test_vtk_io.py`: read the file back with pyvista. Check every new array's values and
     dtype, that an unsolved edge is NaN while a solved edge is unchanged, and that a graph with no
     diameters or haematocrit still writes. Check the field data.
   - `tests/test_perturbation_outputs.py`: the same checks for the CSV.

## Item 4: ilastik output, with an explicit vessel label and provenance-checked reuse

**Done 2026-10-22 in b71718b5, as planned below, with three additions found against the real ilastik
(1.4.1rc2, Harvey's claudin project):** ilastik is run `--readonly`, so a headless run never
saves the project the sidecar records by its bytes; it is passed `--output_format` from the
suffix, since without it ilastik writes the project's saved format (HDF5 by default) whatever
the name says; and a multipage TIFF comes back `.tiff` whatever extension it was given, so
the runner looks for that name (a `.tif` output was never found before). Each output is
binarised in place to 0/1 rather than by `== label` at every reader, so all of them read it
alike. A failed run that exits 0 now quotes ilastik's last lines. SHA-256 runs at ~300 MB/s
here: about 3.5 s for a 1 GB project, once per process.

**4a. What I found (wrong vessel class)**
- `run_ilastik_headless_segmentation` exports `Simple Segmentation` (ilastik.py:162), which
  writes label indices 1..N.
- A two-label project's output is therefore `{1, 2}`. That lands exactly in the "two non-zero
  labels → minority class" branch (load.py:246–249). The count is taken over the whole volume, so
  wherever vessels make up more than half of it, the background becomes the mask. I can't check
  Dale's 0.54–0.61 figures because they are his data, but the mechanism holds.
- **Also found:** if the classifier predicts one class everywhere, `values.size == 1` → `a > 0`
  (load.py:241), and the entire volume becomes vessel.
- Both failures are silent. The large and small vessel masks are loaded the same way.

**4b. What I found (stale reuse)**
- `ilastik_output_is_up_to_date` compares modification times only, and
  `ilastik_reuse_existing_output` is on by default.
- A swapped `.ilp` whose modification time is older than the output gets reused. Windows copying
  keeps the original modification time, so copying in a colleague's project is the ordinary way to
  hit this.
- The log line says the output is "newer than both" the image and the project. That's true, but
  it reads as "made with this project".

**Steps**
1. Add one setting per classifier: `ilastik_vessel_label`, plus the large and small vessel
   equivalents. Each takes a label name or 1-based index, declared in `pipeline/schema.py` with its
   `requires`.
   - Preflight (`pipeline/checks.py`) reads the project's label names from the `.ilp` and refuses a
     label that isn't one of them, listing the ones that are. The `.ilp` is HDF5; check the dataset
     path, `PixelClassification/LabelNames`, against your own projects.
   - There is no default. An ilastik config without the label fails preflight with that list, which
     is Dale's approach. Existing ilastik configs need one line added.
2. Binarize ilastik outputs as `== label` instead of using the generic guess. An output with no
   voxel of that label raises an error naming the label and the file. The generic loader keeps its
   guess for user-supplied masks, but logs a warning whenever it picks the minority class.
3. Write a provenance sidecar next to each output (`<output>.haemolynx.json`) containing:
   - the SHA-256 of the `.ilp`,
   - the input's path, size and modification time (hashing a multi-GB raw volume on every run
     costs too much),
   - the export source and the vessel label.

   Reuse an output only when its sidecar matches. Otherwise segment again and log the reason as a
   WARNING. An output without a sidecar (made before this change) is segmented again once. Measure
   how long hashing a large `.ilp` takes.
4. Tests (`test_io.py`, `test_segment_stage.py`, `test_load_and_validate_vessel_masks.py`), using
   the existing fake-ilastik approach if there is one:
   - With the label set, an output where vessels are the majority gives the right mask. (This fails
     today.)
   - A one-class output raises an error instead of returning the whole volume.
   - Swapping in a different `.ilp` with an older modification time segments again. (This fails
     today.)
   - The same `.ilp` touched again reuses the output; a missing sidecar segments again.
   - Preflight refuses a label the project doesn't have, tested on a small HDF5 file built like an
     `.ilp`.
   - The tooltip and layout GUI tests cover the new rows automatically.
5. Regenerate the example configs, and update CLAUDE.md's Segmentation section.

## Item 1: viscosity law (decision for Harvey + Finbar)

**Done 2026-10-13 in 3e25f87b (Harvey's decision): option A, plus the newest model as a choice.**
`pries_in_vitro_viscosity` is now Pries 1992 exactly, and `plasma_column` keeps reading it.
Pries & Secomb 2005's in vivo model was added. A new Haemodynamics setting,
`pries_viscosity_form`, holds every vessel to one form: in vivo 1994, in vivo 2005 or in vitro
1992; `auto` keeps the choice by basis. Every form is pinned to its published equations, and the
15% "same physics" test is gone: it only held for the hybrid. `RELEASE_REVIEW.md` `flow.4b`
passes. Option B was not taken.

**What I found**
- `_pries_relative_terms` (viscosity.py:156) uses the in vivo base `6·exp(−0.085D)` for both forms.
- `pries_in_vitro_viscosity` is that base without the `(D/(D−1.1))²` factors, which is neither of
  Pries' published laws. Pries 1992's in vitro law uses `220·exp(−1.3D)`.
- It came in with a79c9e2 (Finbar, 2026-08-11). `pipeline/citations.py` credits it to Pries 1992.
- Dale's bfed0da fixed the mirror-image mix in his code: the in vitro base with the in vivo wall
  factor applied once. He made in vivo the default.

How far the current law is from the true in vitro law, at H = 0.45:

| D (µm) | 3.3 | 4 | 5 | 6 | 8 | 10 | 20 | ≥50 |
|---|---|---|---|---|---|---|---|---|
| current / true in vitro | 1.37 | 2.33 | **3.44** | 3.76 | 3.39 | 2.93 | 1.69 | ≈1.0 |
| in vivo at D+1.1, re-referenced to D / current | 0.86 | 0.87 | 0.88 | 0.88 | 0.89 | 0.90 | 0.93 | 0.98–0.99 |

**Why this is a decision, not a typo.** The module argues that the in vivo law is the in vitro law
quoted against a wider diameter, and that this is why plasma-column diameters get the in vitro law.
`test_the_two_bases_are_the_same_physics_read_two_ways` checks that argument and finds agreement
within 15%. It only agrees because this "in vitro" is the hybrid. Against the true in vitro law, the
same comparison is 2–3.3× off in capillaries.

Numerically, the hybrid is a close stand-in (within 12%) for "the in vivo law at the anatomical
diameter D+1.1, re-referenced to the measured channel D". So the literal fix would swap blood as it
behaves in living vessels for blood as it behaves in glass tubes.

**Options**
- **A. True Pries 1992 in vitro for `plasma_column`.** This is the literal fix.
  - Capillary resistance falls 3.4× at 5 µm, so absolute results move by multiples.
  - The 15% consistency test fails by about 3×, and the module's argument for its default no
    longer holds.
- **B. (Recommended) Exact re-referenced in vivo for `plasma_column`:**
  `µ(D) = µ_vivo(D + 1.1) · (D / (D + 1.1))⁴`.
  - This is a published law with one stated assumption, the one the module already makes: the
    imaged plasma column excludes the 1.1 µm surface layer.
  - Results move by 0.86–0.99× (−12% at 5 µm), and the consistency test becomes exact.
  - The true in vitro law would be added as an explicitly named option for comparison.
- **C. In vivo at the measured diameter (Dale's choice).** This is right only if the imaged diameter
  includes the surface layer. On plasma-column data, the existing tests put it 1.8–6× too high.

**Under any option**
- The function named `pries_in_vitro_viscosity` must be the Pries 1992 in vitro law.
- Update `tests/test_viscosity_laws.py`: the `CAPILLARY_AGREEMENT` ratios, the "agrees to 2% at
  3 µm" claim and the 15% test.
- Update CLAUDE.md's "Viscosity model" bullets ("2.03× at 5 µm" becomes 1.77× under B, 0.59×
  under A), the `citations.py` text and the help text for `viscosity_law` and `diameter_basis`.
- Grep the haematocrit-distribution, constriction and perturbation tests for pinned absolute
  resistances.
- Change `describe_law`'s string, so a saved `.haemorun` made with the hybrid can be told apart.
