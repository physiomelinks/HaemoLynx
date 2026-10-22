# The napari panel

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## gui/run_snapshot.py

**`gui/run_snapshot.py`** / **`gui/stage_checkpoints.py`** — save a finished run to a
`.haemorun` file and load it back; keep per-stage snapshots so a tab can re-run from its stage
(`run_pipeline_stages(start_from=..., resume=...)`) without rebuilding everything before it.
Save run captures the run on the GUI thread and writes it on a worker (`save_run_file(background=True)`;
Save and Load are greyed until it is done). The write is gzip level 1 (`COMPRESS_LEVEL`), into a `.part`
file moved over the name once complete. The volumes the checkpoints and the results share are
pickled once, and the checkpoints' graphs, private copies already, are not copied again. gzip's
default level 9 took 12.6 s on one E14.5-sized binary mask, against 0.44 s at level 1. A whole E14.5
run (1 GB pickled, 2026-10-15) went from 297 s with the panel frozen to 8 s in the background, and
its file from 80 to 104 MB. Level 6 would take 27 s and save 18 MB of that, and level 3 9 s to save 8 MB.

## gui/layout.py

**`gui/layout.py`** — where each row sits on its tab. Most tabs get one box per schema section;
Input, Boundaries, Diameters, Haemodynamics and Export are grouped by hand (`TAB_BOXES`). Every
`advanced` setting goes behind an Advanced button anchored on the innermost ordinary row it
needs (so buttons nest as the settings do: the FWHM decoy check under "Do FWHM measurement",
under "Measure diameters from the raw image"), placed after that row's subtree; a child of an advanced toggle
shares its button. Buttons start closed and show how many hidden settings differ from where the
panel started (`changed_settings`). The panel follows each row's whole prerequisite chain
(`form.prerequisite_chain`), not only its own `requires`, so a row never outlives the toggle
above it. `_widget.py`'s `_AdvancedDisclosure` draws a button; tests open them all with
`panel._haemolynx_expand_advanced()`.

## Statistics tab

**Statistics tab** (`gui/statistics_view.py`, `_widget.py`'s `_StatisticsPanel`) — the
left panel's second tab of four (HaemoLynx view, Statistics, Layer list, Layer controls, as
wide as each other: `LEFT_PANEL_TABS`). `export_results` leaves what it measured on the graph,
`G.graph[pipeline.stages.STATISTICS_REPORT]` = `{"mode", "csv", "sections"}`: one section per
statistic measure that ran (`statistics.compute_vessel_statistics_by_measure`, which
`compute_comprehensive_vessel_statistics` flattens, rows and order unchanged), then the
resistances, branch orders, weighted betweenness and communities, and the tissue volume. On
the graph it goes wherever the graph goes (checkpoints, `.haemorun`); `run_pipeline_stages`
drops it from a resumed network, and an export with nothing to report removes it. The tab draws
a button per section, its rows built and shown only while pressed, and refreshes after every
stage drawn, from a run's returned graph (shown in the viewer or not) and on Clear; a section
open before a rerun stays open. Full mode's per-node rows are cut at `MAX_ROWS`.

## gui/layer_sets.py

**`gui/layer_sets.py`** — what the view panel's "Showing" menu swaps: the baseline network or one
perturbation's (vessels, nodes, flow direction, pericytes, vessel tubes), every other network
hidden, and which kinds of layer were on carried across (`carried_roles`: a kind the network
being left has no layer of -- the baseline's pericytes, usually -- keeps the answer the swap
before carried). A perturbation's `LayerSpec`s carry
`layer_set=<its name>`, stored in the layer's `OURS` tag; the baseline's are `None`. The shown
network is the one `_sync_vessel_tubes` draws as tubes or lines. A sweep perturbation's grid
sliders sit in the view panel's Sweep box (not docks of their own) and show only while that
perturbation is the one shown. Every perturbation type in `PERICYTE_PERTURBATION_TYPES`
(`haemodynamics/perturbations.py`) gets a hidden `HaemoLynx {name} pericytes` Points layer from
the edges' `pericyte_centers_um`, which all three constriction strategies record; a sweep
records each grid point's sites on its `SweepFlowGrid` (`pericyte_sites`, since spacing and
length sweeps move them), and the same sliders swap them in (`_apply_sweep_points`). The view panel's "Colour by" menu (`_VesselColourMenu` in
`_widget.py`) colours the shown network's vessels Vectors layer — tubes take their colours from
it — through `_choose_colour_by`, the same path as the layer controls' own Colour by. Below it,
"Colour range" (`_ShownVesselsColourScale`) is the layer controls' range, Fit buttons and canvas
colour bar for the same network: each network keeps its own range, a range typed or fitted on a
sweep survives its slider (`USER_CONTRAST_LIMITS`), and the canvas bar sits on whichever layer
draws the shown network -- in Tubes mode the tube Surface, given the vessels' colormap and range
(`_sync_canvas_colour_bar`), since napari draws no bar for a hidden layer. Under it,
five "Show" checkboxes (`VIEW_TOGGLES`: flow-direction arrows, nodes, terminal nodes, skeleton,
segmented image) turn one layer each on and off without finding it in the layer list:
`toggle_layer_name` says which -- arrows and nodes are the shown network's, terminal nodes is
the run's boundary-nodes layer (inlets, outlets, arteriole/venule boundary nodes), skeleton its
skeleton and segmented image its `IMAGE` layer -- and
`toggle_states` what each box shows. A box mirrors its layer (the layer's own `visible` event
ticks it back; no layer greys it out), so it never fights the eye icon.

## gui/vessel_tubes.py

**`gui/vessel_tubes.py`** — the vessel tube mesh, display only (the graph and its diameters are
untouched): `vessel_tube_mesh(vectors, features)` is what `_sync_vessel_tubes` calls, reading
`diameter_um`, `edge_index` and `is_zero_resistance` (`gui/results.py`'s `_mark_bridges`).
`tubes_from_vectors` draws one closed, smooth-shaded tube per chain of vessels through degree-2
nodes, and through a junction whose two widest vessels carry each other straight on with every
other vessel there narrower (`_carried_straight_on`; drawn as two, their rounded ends overlapped
in a seam across a thick vessel, Harvey 2026-10-15), each ring still carrying its own vessel's
step so colours and hover split per edge; rounded
ends close it only at terminals and junctions. Mid-vessel a vessel is its own diameter; within
min(L/2, 2r) of each end it eases (smoothstep) to the node's radius: the length-weighted mean of
the two vessels at a degree-2 node, min(own, widest other) at a junction, so a wide vessel's end
never stands out of its narrower daughters as a ball. Drawn diameters are clipped to ½–2× the
length-weighted median of the chain and the vessels carrying it straight on through junctions. A
zero-resistance bridge is not vessel, and is drawn so: `gui/results.py` cuts its rows on the
vessels layer into dashes (`dash_polyline`, `BRIDGE_DASH_UM` / `BRIDGE_GAP_UM`, at least two), so
lines show it dashed, and the tubes leave it out of the chains -- the vessel ends cut square where
it meets the lumen (only the narrower: a thick centreline's end the bridge reaches stays rounded,
`_bridge_starts`), the thick vessel runs on through the junction the bridge made -- drawing each
dash as its own flat-ended piece at `BRIDGE_RADIUS_FRACTION` (½) of the narrowest vessel the
bridge meets; its hover says what it is (`branch_hover.BRIDGE_LINE`). Rings are spaced a third of the local
radius (at least three per vessel) along the centreline smoothed with the ends pinned; bends
tighter than 1.1 r get extra local passes, and where adjacent rings would still cross the ring's
radius is cut (`_FOLD_MARGIN`): voxel staircases, snapped ends and tight bends used to make the
tube pinch, bulge and fold. `tube_mesh(..., quality=)` is driven by the "render quality" slider
on a tubes layer's own controls, which only sets the sides round the tube (`TUBE_QUALITY_SIDES`);
the level is the session's, shared by every tubes layer. `vessel_tube_mesh(..., diameter=)` is
the "Tube diameter" choice, on the view panel and on the same controls (`set_tube_diameter`, also
the session's): `per_vessel` (default) as above, or `uniform`, every vessel `uniform_diameter_um`
across whatever its `diameter_um`, which stays as it was. That width is the µm box shown beside
the choice while it is Uniform (`set_tube_uniform_diameter`; 4 µm to start, 0.5-1000 µm).
