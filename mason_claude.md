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
5a. **Network connectivity box** (on **10. Export**, below its settings; the Post processing page
   owns it). It has a dropdown, **All vessels in the viewer** or **Only vessels between an inlet and
   an outlet**. The second uses `inlet_to_outlet_vessels`: dead ends, loops off one node and pieces
   without both an inlet and an outlet are dropped, and the IDs are unchanged.
   **Open 2D connectivity map** draws the CSV exported last, or one you pick
   (`visualization/connectivity_map.py`). It writes `<csv>_map.html` beside the CSV and opens it in
   the browser. The page draws itself: Python only embeds the CSV's text, a config (branch-order
   sort groups from `visualization/_helpers.py`, colours) and `visualization/connectivity_map.js`
   (package data); the layout, both views and the hovering are in that script, which
   `tests/test_connectivity_map.py` runs under Node (skipped without Node).
   - **Load CSV…** at the top of the page (or dropping a CSV on it) draws any other connectivity
     CSV in the same page, no Python needed.
   - **View** dropdown: **Vessels from an inlet** (default): columns count vessels from an inlet;
     inlets green on the left, outlets red on the right, dead ends orange. **By branch order**:
     one column per branch order, left to right Large_Art1.., Art1.., B01.., then Ven and
     Large_Ven counting down to 1 (they count up from the outlets); each vessel a short line in its
     order's column, level with the vessels feeding it; thin grey lines join a vessel to the
     vessels it feeds; vessels with no branch order go in a last `(none)` column.
   - Vessels are coloured by branch order. Inlet and outlet vessels (the arterioles and venules at
     the network's open ends) are drawn bolder than the rest.
   - **Branch order stats** (button in the top bar) opens a table, one row per branch order in
     column order, then **All** (`orderStats`): vessels, mean length, mean diameter, mean vessels
     in and out (counted from the Upstream/Downstream branch IDs, so 0 at an inlet or outlet
     vessel's open end), mean connections = (in + out) / 2, and how many are inlet / outlet
     vessels. Blank lengths/diameters are left out of their means. **Save table CSV** downloads it
     (4 decimals). It follows whichever CSV is loaded.
   - Hovering is the script's own, not plotly's (plotly only answers near a data point, so once
     zoomed in most of a line gave no box): the vessel or node nearest the mouse in pixels shows
     its Branch ID, nodes, length, diameter and notes at any zoom, and the vessel is highlighted.
   **Export connectivity CSV** saves how the network is connected, one row per
   vessel (`graph/connectivity.py`: `connectivity_rows`, `write_connectivity_csv`). Columns:
   Branch ID, From node ID, To node ID, Branch order, Length (um), Diameter (um), Notes, Upstream
   branch IDs, Downstream branch IDs, Edge u, Edge v, Edge key.
   - From → to runs away from the inlets, by path length along the vessels (topology only, no
     solved flow needed). A diverging bifurcation shows as two downstream branch IDs, and a
     converging one as two upstream.
   - Inlet vessels have no from node and outlet vessels no to node, where that end is the
     network's open tip. Notes mark Inlet, Outlet, Dead end, Self-loop, arteriole/venule boundary,
     and "Not connected to an inlet".
   - IDs are the ones the other outputs use, so simulation results can be mapped back: Branch ID
     is the viewer's branchID, which is also the order of the VTK's vessel cells. Node IDs are the
     graph's (VTK `node_id`). Edge u/v/key are the `.pkl` edge key and the VTK's
     `edge_u`/`edge_v`/`edge_key`.
   - It exports the edited network if the tab has scanned one, else the run's graph, and warns if
     edits aren't regenerated yet. The suggested filename is `{stem}_connectivity.csv` beside the
     VTK output.
   - **Load run fixes paths from another machine** (`relocate_run_paths` in `gui/run_snapshot.py`).
     A missing input file is replaced by the file of the same name nearest the `.haemorun`: its
     folder, or up to 3 folders below. An output folder or prefix (`*_dir`, `*_directory`,
     `*_prefix`) that can't be created here moves to `outputs/` beside the run file. The status
     line names what moved.
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
| `src/haemolynx/graph/mask_recovery.py`, `_trace_piece` | **Bug fix, not tab 10.** Drops skeleton components that are closed rings (every voxel two neighbours) before skan: a 3-voxel Lee ring crashed Stage 3 at *recover_uncovered_mask_vessels* with `ValueError: index pointer size 1 should be 2`. **Tell the colleague:** graph-stage code |
| `tests/test_mask_recovery.py` | `test_a_piece_whose_centreline_is_a_tiny_ring_traces_nothing`, `test_a_tiny_ring_beside_a_centreline_goes_and_the_centreline_stays` |
| `_widget.py`: `LAYER_LIST_GRIP_TOOLTIP`, `_let_layer_controls_scroll`, `_resize_layer_list_by_its_title_bar` (+ its call in `settings_widget`) | **Not tab 10.** Dragging napari's layer list title bar resizes it against the layer controls; the controls dock's contents go in a `QScrollArea` and its minimum height back to napari's 50 px, so the drag is no longer stopped at the controls' ~310 px |
| `tests/test_gui_view_panel.py` | `test_dragging_the_layer_list_title_bar_resizes_it`, `test_the_layer_list_drags_up_past_the_layer_controls_own_height` |
| `_widget.py`, Boundaries tab box tools (`_box_tool_widgets`, `_boundary_controls`: `box_colour_of`, `tint_regions`, `set_box_colour`, `on_box_colour`, `on_move_box`) + `chrome_tooltips.py` `box_colour` | **Not tab 10 (tab 4).** "Box colour..." button per role: picks the colour that role's boxes draw in (3D solids and 2D rectangles), display only, session only. Back/Forward in 2D now moves the view's slice to the box (it used to vanish off the slice); in 3D the report gives the new centre, since a flat 3D view cannot show line-of-sight moves |
| `tests/test_gui_boundary_picking_widget.py`, `tests/test_gui_tooltips.py` | `test_in_2d_back_and_forward_take_the_view_with_the_box`, `test_a_roles_boxes_can_be_given_a_colour_of_their_own`; `box_colour` added to the box-tool lists |
| `_widget.py`: `_is_node_hover_layer`, `_node_hover_index`, `_show_node_tooltip`, nodes-first loop in `_branch_hover_viewer_mouse_move`; `branch_hover.py` `format_node_tooltip` | **Not tab 10.** Hovering a node shows "Node ID", vessel count (open end), z/y/x and pressure, before the vessel hover. The network's nodes layer answers even hidden (it starts hidden); hit test by distance like `graph_click.nearest_node_hit`, since napari's picking skips hidden layers |
| `_widget.py` `ACTIONS_FOR_METHOD` (Boundaries tab) | Box tools show for the **volume** method only, and it shows nothing else (Clear dropped); `node_ids` keeps only Pick nodes / Clear node IDs. Note *Use selected node* still switches the role to `node_ids` |
| `tests/test_gui_branch_hover_widget.py`, `tests/test_gui_branch_hover.py` | `test_hovering_a_node_shows_its_id_before_the_vessel_under_it`, `test_a_node_tooltip_leads_with_its_id`; method-controls tests in `test_gui_boundary_picking_widget.py` updated |
| `src/haemolynx/pipeline/stages.py`: `_reset_run_node_list` (used for all ten `settings["*_nodes"][:] = []` in `assign_boundaries`) | **Bug fix, not tab 10 (stage 4).** A run-filled `*_nodes` list holding a bare ID (`large_vessel_outlet_nodes: 3316`) or null crashed Boundaries with `'int' object does not support item assignment`; it is now replaced with an empty list (a real list is still emptied in place) |
| `tests/test_boundary_node_defaults.py` | `test_a_lone_node_id_in_a_run_filled_list_does_not_stop_the_stage`, `test_a_run_filled_list_is_emptied_in_place` |
| `src/haemolynx/visualization/large_vessel_assignment.py` (`MAX_VOLUME_TRACE_POINTS`, `_volume_stride_within`, in `add_binary_mask_volume_trace`); `src/haemolynx/graph/automated_vessel_assignment.py` (both `_add_volume_trace`s now call that helper) | **Bug fix, not tab 10 (stage 4 HTML).** Mask volumes in the 3D HTML views were drawn at full resolution (small-vessel labelling view always; the others at `large_vessel_3d_volume_downsample_stride: 1`): napari was OOM-killed on 62 GB writing `small_vessel_mask_boundary_labelling_3d.html`. Each mask is now cropped and pooled to at most 1M points |
| `tests/test_large_vessel_assignment_visualization.py` | `test_a_whole_image_mask_is_pooled_to_a_volume_a_page_can_hold`, `test_the_small_vessel_labelling_view_pools_its_masks_too` |
| `_widget.py`: `_set_tube_colours`, `_tube_mesh_key` (in `_sync_one_vessel_tubes`), `_note_z_filter_window` + `changed` in `_apply_z_filter`, `_same_volume` (in `_add_or_update`); `results.py` `vessel_mask_volume_layers` | **Speed, not tab 10.** Drawing a stage redid work: a tube recolour went through napari's data path and recomputed every vertex normal (~0.8 s, several times a stage); every stage rebuilt the tube mesh even with the vessels unchanged; the full-range Z-depth filter rewrote every graph layer and rebuilt the tubes; unchanged image/mask volumes were re-sent; masks were float32 copies (2 GB for four). Now: colours go straight to the vispy mesh in 3D, the mesh rebuilds only when its fingerprint changes, the filter skips layers it has not narrowed, equal volumes are not re-set, masks are uint8. Headless, all stages draw in ~16 s, was ~45 s |
| `tests/test_gui_results_widget.py`, `tests/test_gui_results.py` | `test_recolouring_the_tubes_keeps_their_mesh_normals`, `test_a_stage_that_leaves_the_vessels_alone_keeps_the_tube_mesh`, `test_a_full_range_depth_filter_leaves_fresh_layers_alone`, `test_an_unchanged_volume_is_not_sent_again`; mask dtype expectation now uint8 |
| `src/haemolynx/graph/smoothing.py`: `_distance_to_polyline`, `must_pass` on `_is_acceptable` / `_accept` | **Bug fix for Add vessel; graph-stage code.** Optional: the result must also run within the tolerance of each of these points. Only `smooth_traced_path` passes it (every traced point), so graph building's smoothing is unchanged. **Tell the colleague:** graph-stage code |

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
- **2026-10-03:** Cherry-picked the Export tab's connectivity CSV and 2D map (`28284eb` from
  `Mason_topology_map`) onto `Devel_GUI_improvements_HD` as `fb04d55` and pushed it.
- **2026-10-03:** 2D connectivity map: a **By branch order** view behind a dropdown on the page,
  and bolder inlet and outlet vessels in both views (`visualization/connectivity_map.py` only).
  Checked on `ZStack_Haemolynx_connectivity.csv` (1,334 vessels): 23 columns, Art1 → B19 → Ven2 →
  Ven1. Tests: `test_connectivity_map.py` (8, 4 new) and `test_graph_connectivity.py` pass.
- **2026-10-03:** 2D connectivity map: hovering works at any zoom (the page finds the nearest
  vessel itself), and **Load CSV…** / drop a CSV on the page to draw another one. The layout moved
  from Python into `visualization/connectivity_map.js` so the page can do that without Python;
  `connectivity_map_figure` / `connectivity_map_layout` / `branch_order_layout` are gone, replaced
  by `connectivity_map_html`. Checked in headless Chrome on `ZStack_Haemolynx_connectivity.csv`:
  zoomed to 0.1 column wide, a point a quarter along Branch 33 gives its box; a dropped CSV
  redraws. Tests: `test_connectivity_map.py` (17, 13 run the script under Node).
- **2026-10-03:** 2D connectivity map: **Branch order stats** table (per order and All: vessels,
  mean length, diameter, vessels in/out, connections = (in + out) / 2, inlet/outlet vessels), with
  Save table CSV. Tests: `test_connectivity_map.py` (20).
- **2026-10-06:** Fixed Stage 3 crash `index pointer size 1 should be 2` in mask recovery (skan cannot build a skeleton that is only a tiny ring; such rings are now dropped before it, as their self-loop path was dropped anyway). Shared code; see table. Tests: `test_mask_recovery.py` (+2).
- **2026-10-06:** Layer list drag no longer limited by the layer controls' own height (controls now scroll). Shared code; see table. Tests: `test_gui_view_panel.py` (+1).
- **2026-10-06:** Boundaries box tools: Box colour button; 2D Back/Forward keep the box in view; 3D Back/Forward report the box centre. Shared code; see table. Tests: `test_gui_boundary_picking_widget.py` (+2).
- **2026-10-06:** Node hover shows the node ID (works with the nodes layer hidden). Box tools only on the volume method. Shared code; see table.
- **2026-10-06:** Fixed Boundaries crash on a bare ID in a run-filled `*_nodes` setting. Shared code; see table.
- **2026-10-06:** Fixed out-of-memory crash in Boundaries' 3D HTML mask views (masks pooled to at most 1M points). Shared code; see table.
- **2026-10-06:** Faster stage drawing in the viewer (tube recolour without normals, mesh rebuilt only on change, Z filter skips identity, volumes not re-sent, uint8 masks). Shared code; see table.
- **2026-10-09:** Fixed Add vessel cutting off a trace that turns back on itself: a straight-line trace out to a waypoint and back past the start came back as a chord across the turn (15.8 µm instead of 35.5). Its smoothed path must now pass every traced point within the tolerance. Shared code; see table. Tests: `test_graph_post_processing.py` (+2), and `test_add_vessel_from_one_vessel_to_another_forms_a_node_on_each` passes again.
