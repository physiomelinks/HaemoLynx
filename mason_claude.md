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

## What the tab does (as of commit 1)

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
     This repeats until the junction is degree 3, so a 5-way junction takes two splits.
   - **Leave as is:** marks the junction and moves on to the next.
4. **Regenerate from the edited network:** reruns Diameters → Export on the edited graph, so
   branch orders, resistances, flows and the 3D view catch up. The tab then clears; scan again
   afterwards.

Things to know:
- **Scan network** reads the network as it is in the viewer now, including edits not yet
  regenerated.
- A vessel's **branchID** is its position in `G.edges(keys=True)`, so it changes after any
  deletion or split.
- The old floating **Edit** window and its button are unchanged in this commit.

---

## Files

| File | What it holds |
|---|---|
| `src/haemolynx/graph/post_processing.py` | Pure graph rules, no Qt or napari: `edge_keys`, `high_degree_junctions`, `junction_vessels` → `JunctionVessel`, `delete_vessels(protected=...)`, `split_junction(connector_length_um=15)`, `DEFAULT_SPLIT_CONNECTOR_LENGTH_UM` |
| `src/haemolynx/gui/post_processing.py` | Pure viewer logic: `scan_network` → `NetworkScan`, `vessel_status` / `status_colours` / `STATUS_COLOURS` (grey, cyan, yellow), `junction_marker_layer` (layer `HaemoLynx 4+ junctions`), `junction_table_rows`, `junction_label`, `camera_center_for`, `zoom_for_canvas` |
| `tests/test_graph_post_processing.py` | Graph rules on small hand-built networks |
| `tests/test_gui_post_processing.py` | The pure viewer logic |
| `tests/test_gui_post_processing_widget.py` | The real Qt page with a napari viewer (`gui` marker) |

How `split_junction` picks the pair: the two incident vessels whose directions over their first
10 µm have the largest dot product. The new node goes `connector_length_um` out along their
mean direction. Moved vessels lose the part of their path the connector now covers, and their
`length` is re-measured from `voxels`. The connector is flagged `junction_split_connector=True`
and has no diameter or branch order; Regenerate assigns them. The junction keeps its node id,
so a boundary role on it survives.

---

## Shared code touched (outside the files above)

| Where | Change |
|---|---|
| `src/haemolynx/gui/_widget.py`: `POST_PROCESSING_TAB`, `JUNCTION_ZOOM_BOX_UM`, `_zoom_viewer_to`, `_post_processing_controls` | **The tab's Qt page.** A self-contained block just above `def settings_widget`; tab-10 work goes here |
| `_widget.py` in `settings_widget`, right after the stage-tab loop | Builds the page and adds it as the last tab (`post_processing = _post_processing_controls(...)`) |
| `_widget.py`, revert-stack loop | Adds one empty "Run from this stage" slot so the stack's pages still line up with the tabs |
| `_widget.py`, `on_regenerate_from_edit` | Body moved into `regenerate_from_graph(graph)`, shared by the old Edit window and tab 10 |
| `_widget.py`, test hooks | `panel._haemolynx_post_processing` |
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

1. **Off-network vessels:** `graph.off_network_edges` (biconnected-component test: a vessel is on
   the network iff it lies on a simple inlet → outlet path), a red status in `vessel_status`, and
   "N of M vessels are not on an inlet-to-outlet path" in the scan summary.
2. **Edit tools moved into tab 10:**
   - Add vessel, Finish, Delete by click and Stop editing, reusing
     `gui/graph_editor.GraphEditorState` on the same working graph.
   - Click-delete protects boundary nodes.
   - The old Edit button is hidden but kept in its row, because
     `test_gui_view_panel.py::test_the_view_button_sits_left_of_edit_and_has_a_tooltip` checks
     the row order.

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
