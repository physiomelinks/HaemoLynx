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

## The settings optimisers (Optimise settings, Optimise FWHM settings, Optimise endothelial settings)

History of `PLAN_optimisers.md` (2026-10-23 to 2026-10-24), which brings the optimisers in line
with the pipeline as it now runs and adds the third; the plan's **Measured** section has the E14.5
numbers.

- **Phase 1** (`e1314748`): the pipeline's own argument builders in `pipeline/stages.py`
  (`segmentation_cleanup_arguments`, `skeleton_cleanup_arguments`, `thick_region_settings`,
  `thick_vessel_refinement_arguments`, `centreline_smoothing_arguments`, beside
  `graph_build_arguments`), which `skeletonise` and `build_network` call and the search's four hand
  copies gave way to; the report names the settings a search read but does not tune that are not at
  their defaults. E14.5's skeleton and graph unchanged by it.
- **Phase 2** (`f1184f25`): Optimise settings starts from what a run's skeletonise reads
  (`stages.load_volume_on_run_grid`): the run's grid and whole settings, voxel counts on that grid
  and back, the run's thick-vessel restriction, the plain skeleton through
  `io.load._skeletonize_loaded_volume`. A trial's E14.5 skeleton equals the run's.
- **Phase 3** (`9328c668`): FWHM trials measure through the run's own call
  (`apply.measure_fwhm_diameters`) with the run's config and mask, inside the whole vessel graph,
  on a template cleared of the run's widths.
- **Phase 4** (`e1314748`): `build_network`'s graph work split into `stages.network_context` (the
  radius sampler, the one `MaskSupport`, the thick centreline tree, made once per mask) and
  `stages.finish_network_graph` (everything after graph building: smoothing by the vessel radius,
  the clean-ups, the thick-vessel split, the re-trace, straight steps, the optional removals). Every
  graph sweep scores the finished graph, so a trial's E14.5 graph equals the run's edge for edge.
  The smoothing sweep moved most (it now judges the saved graph with the run's radius tolerance),
  and the search took 1.9 times as long (D5 allowed 2).
- **Phase 5** (panel, low RAM, words): both optimisers grey under `network_source=synthetic`,
  saying why in their hover text (`chrome_tooltips.OPTIMISE_SYNTHETIC_TOOLTIP`) and refusing if
  pressed; a set batch already greyed them (`d4bdfd0c`). The panel greys and gives back only what
  it greyed, so a change of settings mid-run never frees a running optimiser's button. With
  ilastik segmenting, Optimise settings uses its output when that output's record says it is the
  one a run would make now (`stages.current_ilastik_output`), and otherwise refuses saying why; it
  never runs ilastik (a segmentation can take hours, and a run makes it). Before, it refused with
  no input path, and with one it called `segment`, which ran ilastik. Under `use_memmap_loading`
  the worker (`_widget._optimise_settings_for_run`) binarises the mask onto disk and reads the raw
  image onto disk as the stages do (`load_single_channel_tiff_volume(use_memmap=True)`, cut and
  reduced a slab at a time; `io.raw_volume_cache` serves only the segmented image, which
  `load_volume_on_run_grid` already reads through it). `block_reduce_volume` already read a
  memmap a slab at a time; the search's raw-image reduction cast the whole image to float64 first
  (eight bytes a voxel), and now casts a plane of blocks at a time, bit for bit the same.
  "Off" is refused when the mask will not fit in free memory
  (`search.full_resolution_search_refusal`, 128 B a voxel: a full-resolution search's traced
  peak on the dense capillary bed fixture, tiled, was 56, 122 and 267 MB at 0.29, 1.15 and
  2.30 M voxels, growing 76 then 126 B a voxel; measured with tracemalloc, so memory allocated
  outside numpy and Python is not counted). Auto is not capped by it. (`399a48f3`, `924b9910`.)
- **The endothelial optimiser** (D4; `65068bc1`, `42c4be53`, `56be0b79`): **Optimise endothelial
  settings**, the counterpart of the FWHM button for a raw image that stains the wall. It sweeps
  the ring fit's contrast gate first (it decides which rings calibrate), then the blur and wall --
  only where the image's rings will not calibrate them, or the settings already fix them (Harvey,
  2026-10-24): where they calibrate, a fixed value inside the estimate's interval is no different
  from it. A trial is the run's ring fit on the same vessels, each blur's calibration fits made
  once; scored on hollow vessels of known lumen planted with the image's own blur, wall and noise,
  the sample's spread and decoys. Auto times reading and caps the sample under 64 vessels, where
  every trial would start worker processes.
- **B8** (`ca5df387`, `aee7da34`): Optimise FWHM settings' Auto sampled about 10 of E14.5's 1,342
  vessels, too few to plant any vessel of known width, so the accuracy group never ran. Auto now
  samples at least 100, and its time model counts the trials a pass makes (77) and the planted
  vessels and decoys each measures (half a vessel per sampled one): 1,530 s estimated against the
  1,488 s the E14.5 search took. It found FWHM's planted-vessel error about 0.54 there, still open.
- **The scorecard and the report** (2026-10-24, `95941900`): the scorecard marks its graphs' dead ends as the
  run does (`stages.mark_dead_end_kinds`, which `build_network` calls too) and counts each kind,
  with loops inside one lumen beside short loops; a kind is a measure only (`ScoreRow`'s
  `higher_is_better=None`: never better or worse), since which way is better waits on a marked
  grid. A search on a coarser grid says that grid was reduced by block max, where a run's
  downsample on import uses block majority (D2: kept, and said). The FWHM report said it had
  searched a downsampled copy (its pool-to-sample ratio sat in `downsample_factor`); it now says
  how many of the vessels it sampled, as the endothelial report does (2026-10-25, `8eddf2ab`).

## The raw image at the chosen review item

`gui/raw_panel.py` and the Post processing page's "Show the raw image at the chosen item" box
(2026-10-25, Harvey's request closing `PLAN_raw_confirmation.md`, whose Phase 3 named "the raw
panel" without saying what it shows). It draws the chosen short loop or review item as the marked
grids drew theirs, since that is what Harvey's marks were made on: the item in its own plane
(through the middle of its points, along their two longest directions; an item lying in one image
plane drawn as napari draws it, x right and y down) and in the plane across it (deeper lower),
each the segmentation (any voxel within 3 um of the plane) beside the run's raw image (the slab's
mean, blurred 1.5 samples, 2nd-99.8th percentile contrast), the item's vessels cyan, the other
centrelines grey dots, a 5 um bar. Off until ticked: the raw image is read the first time, through
the page's own loader (the Diameters stage's copy in the viewer, else the Input tab's file, kept
for Add vessel too). Each picture reads one window of the volume round its plane, so a
memory-mapped raw image stays on disk. It shows; it decides nothing, and no list is ordered by it.
