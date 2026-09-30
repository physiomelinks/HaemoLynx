# Mason's post-processing work: notes for humans and Claude

The index of everything added for the **"10. Post processing"** tab of the HaemoLynx napari
panel. Keep this file up to date: every change to the tab updates the sections it affects and
adds a line to the change log at the bottom.

**Ground rules**
- **Stay inside tab 10.** Someone else is working on tabs 1–9. New work goes in the files listed
  under *Files*. The spots under *Shared code touched* are the only places outside them that were
  changed. Keep further edits there to the minimum and list any new one here.
- **One commit per addition**, so each feature can be debugged or reverted on its own.
- Commits go on `Devel_GUI_improvements_HD` and are **not pushed** until Mason asks.

---

## What the tab does (as of commit 7)

After a run (at least through **4. Boundaries**), tab 10 works on a copy of the run's network:

1. **Scan network**
   - Lists every junction where **4 or more vessels** meet (4-way and 5-way junctions), busiest
     first, and rings each one in magenta in the viewer.
   - Picks the first junction for you.
2. **Junction list:** click one to zoom to it (a 60 µm box) and list its vessels in a table:
   branchID, length, diameter, branch order and the node at the other end. The chosen junction's
   vessels turn **cyan**. Click rows to select vessels, with ctrl or shift for several; selected
   ones turn **yellow**. Everything else is grey.
3. **Fixes at a junction**
   - **Delete selected vessels.** A junction left joining just two vessels is merged into one
     vessel. A deletion that would leave an inlet, outlet or boundary node with no vessel is
     refused.
   - **Split into bifurcations with a vessel of [15 µm].** The two vessels leaving in the most
     similar direction move to a new node, joined back by a connector vessel of that length.
     The connector takes the mean diameter of the junction's vessels, as an override.
     This repeats until the junction is degree 3, so a 5-way junction takes two splits.
   - **Leave as is:** marks the junction and moves on to the next.
4. **Edit by clicking in the viewer.** This replaces the old bottom-row **Edit** button, which is
   now hidden.
   - **Delete vessel:** then click vessels to remove them, one per click, with the same
     boundary-node protection as the table's Delete.
   - **Add vessel:** then click node A, then node B. The new vessel:
     - is routed through the segmented image (the IMAGE layer) with the windowed A* cost
       field;
     - falls back to a straight line when there's no image, or when the route runs under half
       its length inside the mask;
     - gets the mean `diameter_um` of the vessels already at A and B, stored as a manual
       override, so it survives Regenerate;
     - carries `post_processing_added=True`.

     The status line says which path was used.
   - **Stop editing.**
   - **Delete by branch ID:** type branchIDs as the vessel hover shows them (`12, 40 311`;
     commas, spaces or semicolons), then press the button or Enter. Bad input is reported and
     kept so it can be corrected. Boundary nodes are protected the same way. After a delete, the
     IDs of later vessels move down.
   - Nodes are picked from the graph itself (`nearest_node`, the node nearest the click ray),
     not by napari's point picking, which needs the layer drawn on screen.
5. **Prune disconnected branches** (just above Regenerate): removes every piece of the network
   that no longer has both an inlet and an outlet, e.g. a branch whose only link to the rest was
   deleted (`prune_disconnected_branches`). The report says how many pieces, vessels and boundary
   nodes went.
6. **Regenerate from the edited network:** reruns Diameters → Export on the edited graph, so
   branch orders, resistances, flows and the 3D view catch up. Tab 10's Regenerate
   cuts the rerun's inlet, outlet and boundary lists to the nodes still in the graph
   (`boundaries_following_graph`), so pruned boundary nodes don't stop the solve. It also
   **keeps the FWHM diameters already measured** instead of measuring every vessel again: it
   adds `do_fwhm_measurement` to the rerun's skip list, and the next Run pipeline restores the
   switch. New vessels still get their diameter: **every vessel tab 10 adds (Add vessel, or a
   split's connector) takes the mean diameter of the vessels at its nodes**, as a manual
   override that Regenerate keeps. The old Edit window's Regenerate keeps the strict boundary check and
   re-measures.

   **What Regenerate costs**, measured on the real network: Diameters → Export takes about 10 s.
   It never re-skeletonises or rebuilds the graph when the session has them. The first
   Regenerate after **Load run** takes about 2 minutes more, because a saved run doesn't store the
   loaded volumes (by design, in `gui/run_snapshot.py`), so Skeletonise and Graph reload from
   their cached files once. The tab then clears; scan again
   afterwards.

7. **"What was changed" log** at the bottom of the tab: one timestamped line per action (scan,
   delete, split, leave, click-delete, add, delete by ID, prune, regenerate, and any refusal).
   The same line goes to napari's run log as `Post processing: …`. Vessels are named with
   `describe_vessels` **before** the edit, e.g. `branchID 15 (node 26-136, 26.2 µm, 4 µm)`, so
   the IDs are the ones on screen when it was made. A prune lists every vessel it removed.

Things to know:
- **Regenerate saves the outputs.** Its Export stage writes the VTK files and the stage
  checkpoints to the run's output folder (`vtk_output_prefix`). Edits that haven't been
  regenerated are only in the viewer. To keep the whole session (edits included, once
  regenerated), use **Save run**, and wait for `Wrote run …`.
- **Scan network** reads the network as it is in the viewer now, including edits not yet
  regenerated.
- A vessel's **branchID** is its position in `G.edges(keys=True)`, so it changes after any
  deletion or split.
- The old floating **Edit** window's code is unchanged. Only its button is hidden.

---

## Files

| File | What it holds |
|---|---|
| `src/haemolynx/graph/post_processing.py` | Pure graph rules, no Qt or napari: `edge_keys`, `high_degree_junctions`, `junction_vessels` → `JunctionVessel`, `delete_vessels(protected=...)`, `split_junction(connector_length_um=15)`, `DEFAULT_SPLIT_CONNECTOR_LENGTH_UM`; for Add vessel: `mean_incident_diameter`, `vessel_path_between` (routed or straight, `MIN_ROUTED_INSIDE_FRACTION`), `add_vessel_between`; `prune_disconnected_branches` |
| `src/haemolynx/gui/post_processing.py` | Pure viewer logic: `scan_network` → `NetworkScan`, `vessel_status` / `status_colours` / `STATUS_COLOURS` (grey, cyan, yellow), `junction_marker_layer` (layer `HaemoLynx 4+ junctions`), `junction_table_rows`, `junction_label`, `nearest_node` (click → node), `parse_branch_ids`, `boundaries_following_graph`, `describe_vessels` (log wording), `camera_center_for`, `zoom_for_canvas` |
| `tests/test_graph_post_processing.py` | Graph rules on small hand-built networks |
| `tests/test_gui_post_processing.py` | The pure viewer logic |
| `tests/test_gui_post_processing_widget.py` | The real Qt page with a napari viewer (`gui` marker) |

How `split_junction` picks the pair: the two incident vessels whose directions over their first
10 µm have the largest dot product. The new node goes `connector_length_um` out along their
mean direction. Moved vessels lose the part of their path the connector now covers, and their
`length` is re-measured from `voxels`. The connector is flagged `junction_split_connector=True`
and carries the mean diameter of the junction's vessels as an override (none if none of them
has one). Regenerate assigns its branch order. The junction keeps its node id,
so a boundary role on it survives.

---

## Shared code touched (outside the files above)

| Where | Change |
|---|---|
| `src/haemolynx/gui/_widget.py`: `POST_PROCESSING_TAB`, `JUNCTION_ZOOM_BOX_UM`, `_zoom_viewer_to`, `_post_processing_controls` | **The tab's Qt page.** A self-contained block just above `def settings_widget`; tab-10 work goes here |
| `_widget.py` in `settings_widget`, right after the stage-tab loop | Builds the page and adds it as the last tab (`post_processing = _post_processing_controls(...)`) |
| `_widget.py`, revert-stack loop | Adds one empty "Run from this stage" slot so the stack's pages still line up with the tabs |
| `_widget.py`, `on_regenerate_from_edit` | Body moved into `regenerate_from_graph(graph, from_post_processing=False)`, shared by the old Edit window and tab 10. Tab 10 passes `True`, which cuts the resume's boundary lists to the graph and adds `do_fwhm_measurement` to the rerun's skips |
| `_widget.py`, test hooks | `panel._haemolynx_post_processing` |
| `_widget.py`, bottom "run file" row | `edit_button.visible = False`, one added line: the old Edit button stays in its row (a view-panel test checks the row order) but is hidden, and its code is kept |
| `src/haemolynx/haemodynamics/poiseuille.py`, `stamp_edge_diameters` | A vessel with `diameter_source="override"` now keeps its diameter over the **branch-order table**; a measurement of that vessel (FWHM, endothelial, raw section, EDT) still replaces it, which `test_fresh_fwhm_run_wipes_overrides` pins. **Tell the colleague:** this is haemodynamics code, not tab 10 |
| `tests/test_diameter_assignment.py` | `test_override_beats_the_table_when_nothing_measures_the_vessel` |
| `src/haemolynx/graph/__init__.py` | Imports and `__all__` for the new graph functions |
| `src/haemolynx/gui/chrome_tooltips.py` | `POST_PROCESSING_TOOLTIPS` (hover text for every tab-10 control) |
| `tests/test_gui_widget.py` | The tab-list test expects `POST_PROCESSING_TAB` after the stage tabs |
| `tests/test_gui_tooltips.py` | Includes `POST_PROCESSING_TOOLTIPS` in the non-schema tooltip check |
| `CLAUDE.md` | One-line mentions of the two new modules and of tab 10 not being in `STAGES` |

Tab 10 is deliberately **not** in `pipeline/progress.py`'s `STAGES`. It isn't a pipeline stage,
so it gets no "Run from this stage" button and doesn't touch the progress bars.

---

## Design notes (why it is built this way)

- **Recolour, don't overlay.** Vessels are drawn as 3D tubes by default (`gui/vessel_tubes.py`).
  An extra overlay of thin lines is hidden inside the tubes. So the tab sets `edge_color` on the
  existing `HaemoLynx vessels` layer, and the tubes follow through their own `edge_color` event.
- **Don't retint twice.** When that event is wired (`_haemolynx_follow_tubes`), `recolour` doesn't
  call `_maybe_retint_vessel_tubes` itself. Each retint re-shades the whole mesh, about 0.25 s.
- **Rebuild only after an edit.** Clicking a junction or a table row only recolours. The
  vessels/nodes layers are rebuilt (`ResultLayers.layers_for_graph`) only when an edit changes
  the network.
- **Zoom without a reset.** `zoom_for_canvas` uses `viewer._canvas_size`, because
  `viewer.reset_view()` redraws the scene once more on every click.
- **Protected nodes** come from the checkpoints' boundary roles (inlet, outlet, arteriole and
  venule boundaries), via `checkpoints._carried_boundary_roles()`. They are never merged or
  stranded, because the solve needs all of them.

Timings on the real network (953 vessels), offscreen: scan 0.5 s; junction or row click 0.35 s in
Tubes mode. Lines mode is faster.

---

## Planned next commits (code already written, saved for later)

The full version, before it was split into commits, is saved inside the repo's `.git` folder,
where it can't be committed by accident:
- `.git/mason_tab10_full_post_pull.patch`: the whole diff against `1158437`.
- `.git/mason_tab10_backup/full_post_pull.tgz`: the same files as a tarball.
- The `*_pre_pull*` copies are from before your colleague's 3 commits were pulled.

1. **Off-network vessels (saved code):** `graph.off_network_edges` (biconnected-component test:
   a vessel is on the network iff it lies on a simple inlet → outlet path), a red status in
   `vessel_status`, and "N of M vessels are not on an inlet-to-outlet path" in the scan summary.

Later ideas: remove the red vessels in one go (prune dead ends upwards, optionally dangling
loops); a connectivity map (vessel → junction → vessels, directed by solved flow, as CSV plus a
2D hover map); permanent vessel IDs; undo.

---

## Running the tests

```bash
# pure tests (any env with pytest + the package's deps)
PYTHONPATH=src python -m pytest tests/test_graph_post_processing.py tests/test_gui_post_processing.py

# GUI tests: napari's own Python, plus pytest/pytest-qt from a side folder
PYTHONPATH=<folder with pytest+pytest-qt> QT_QPA_PLATFORM=offscreen \
  ~/napari-venv/bin/python -m pytest -m gui tests/test_gui_post_processing_widget.py
```

**Checking Regenerate end to end** writes into the run's own output folder
(`vtk_output_prefix`, e.g. `/home/sliu205/outputs`), overwriting its checkpoints and `.vtp`
files. Point `vtk_output_prefix` at a scratch folder first, or rerun afterwards.

Known unrelated failures in this environment:
- 3 in `tests/test_gui_view_panel.py`: the scale bar and tubes-after-snap tests fail offscreen
  without a real GL display. They fail without these changes too.
- 13 errors in `tests/test_gui_flow_direction.py` when run outside napari's env: the
  `make_napari_viewer` fixture is missing.

---

## Change log

- **2026-09-30:** Built tab 10 in full: 4+ junctions, off-network vessels, Edit tools moved in.
  After review it was split into one commit per addition, and the full version saved to `.git/`
  (see *Planned next commits*).
- **2026-09-30, commit 1:** Pulled the colleague's 3 new commits on `Devel_GUI_improvements_HD`
  (up to `1158437`), then committed tab 10 with **junction fixing only**: scan for 4- and 5-vessel
  junctions, junction table, Delete / Split (15 µm connector) / Leave, Regenerate. Tests: tab,
  panel, editor and view-panel GUI tests pass apart from the 3 known offscreen failures.
- **2026-09-30:** Commit 1 rebased onto the colleague's README commit `4c392a4` and **pushed** to
  `origin/Devel_GUI_improvements_HD` as `a7ce297`.
- **2026-09-30, commit 2:** Edit by clicking in tab 10:
  - Delete vessel by click.
  - Add vessel by clicking two nodes: routed through the image, straight as the fallback, with
    the mean diameter as an override.
  - Stop editing.
  - The bottom Edit button is hidden.
  - A manual override diameter now beats the branch-order table (`poiseuille.py`).
  - Tests: 3,682 fast tests passed; the GUI tests passed apart from the 3 known offscreen
    view-panel failures.
- **2026-09-30, commit 3:** Delete by branch ID in tab 10's edit box (`parse_branch_ids`), with
  boundary-node protection. Tests: the tab's pure, widget and tooltip tests pass (40).
- **2026-09-30, commit 4:** Prune disconnected branches (`prune_disconnected_branches`), and tab
  10's Regenerate follows pruned boundary nodes (`boundaries_following_graph`).
  - Checked end to end on a copy of `ZStack_Haemolynx.haemorun`: delete branchID 17, which cut a
    9-vessel piece holding outlets 2141 and 2195 off from the inlets, then prune, then
    Regenerate. Every stage from Diameters to Export completed (320 s).
  - Side effect: that check overwrote `/home/sliu205/outputs`' checkpoints and
    `ZStack_Haemolynx_*.vtp` with the test's edited network. Rerun before using them.
  - Tests: the tab's pure, widget, graph, tooltip and editor tests pass (106).
- **2026-09-30, commit 5:** Tab 10's Regenerate keeps the measured FWHM diameters. Diameters
  went from 171 s to 1.5 s on the real run; 747 measured diameters were kept.
- **2026-09-30, commit 6:** A split's connector takes the mean diameter of the junction's vessels
  (override), so every vessel tab 10 adds uses the neighbours' average.
- **2026-09-30, commit 7:** "What was changed" log at the bottom of tab 10, mirrored to napari's
  log. It records every change, with pre-edit branchIDs and every vessel a prune removed.
