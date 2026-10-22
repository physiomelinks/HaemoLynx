# Pipeline stages, the Post processing tab, provenance

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## pipeline/stages.py

**`pipeline/stages.py`** — one function per stage (`segment`, `skeletonise`, `build_network`,
`assign_boundaries`, `assign_diameters`, `build_haemodynamic_model`, `solve`, `post_process`,
`run_perturbations`, `export_results`), each taking settings plus the previous stage's
dataclass, so a caller can run them one at a time and intervene. `run_pipeline_stages` is the
orchestrator that runs all ten in order and returns the graph; its `start_from` / `resume`
arguments let the panel re-run from a chosen stage using a saved run's state instead of
recomputing the earlier ones, and `stop_after` ends a run after a named stage (how the panel
pauses for post-processing). `PipelineResume` carries every boundary list -- the
Large_Art/Large_Ven hand-off nodes included, which a resumed run used to lose -- and the
thick-vessel region. `post_process` measures only the vessels an edit marked
(`haemodynamics.apply.assign_edge_diameters(edges=...)`, FWHM's and EDT's own `edges=`), keeps the
diameter the edit gave as a provisional override, and bridges them with
`graph.insert_thick_vessel_junction_nodes(edges=...)`, which never re-splits an existing bridge;
it re-judges the boundary nodes and the dead ends at the nodes the edits touched
(`_reassign_large_vessel_boundaries`, `_reassign_small_vessel_boundaries`,
`_mark_touched_dead_end_kinds`) and assigns branch orders again over the whole network;
the stage then calls `solve_post_processed` (network handling, `build_haemodynamic_model`,
`solve` on the edited network), whose `Solution.post_processed` is what the viewer draws the
stage from -- only when there were edits. `apply_network_handling(settings, model, boundaries, network)` is
the "Network handling" box on the Haemodynamics tab (`boundary_handling` acts on every vessel no
inlet-to-outlet path runs along, `graph.inlet_to_outlet_vessels`: each component without both an
inlet and an outlet, and every dead-end branch, tree or loop hanging off a perfused one.
`remove_disconnected` deletes them (`graph.remove_vessels_off_inlet_outlet_paths`), logging dead
ends and disconnected components apart, and the boundary-node lists and resistance pair follow
the pruned graph; `leave_unsolved` — the default — keeps them and `solve` marks exactly those
vessels `graph.FLOW_SOLVED=False`, which the viewer draws
light grey, blanks in every flow-based colour column (`gui/results.py`'s
`SOLVED_FLOW_COLUMNS`) and labels Solved/Unsolved on hover; the retired
`remove_disconnected_io_components_after_final_assignment` checkbox still loads, through
`Setting.replaces`; and, with `haematocrit_junction_rule="split_junctions"`,
splitting every 4+ junction into bifurcations on a copy of the graph via
`graph.split_high_degree_junctions`, with connectors as long as the junction's vessels are wide
unless `haematocrit_split_connector_length_um` fixes one), run at the start of the haemodynamics
stage — not at the
end of `assign_boundaries`, where it used to be — so "Run from this stage" on that tab honours
it. A caller running the stages by hand calls it between `assign_diameters` and
`build_haemodynamic_model`, as the examples do. Because `assign_diameters` runs before the
removal, it samples its whole-image calibrations (the FWHM/raw-section PSF, the endothelial
wall, the FWHM decoy check) only from the vessels `remove_disconnected` will keep
(`edges_network_handling_keeps` → `assign_edge_diameters(calibration_edges=)`), and the viewer
redraws the FWHM line layers from the solved network (`gui/results.py`'s `_fwhm_line_layers`).

## pipeline/progress.py

**`pipeline/progress.py`** — `STAGES`, the run's stages in order (the panel draws one tab
per entry and a progress bar counts them — one list, not two). It has eleven entries for ten
stage functions: `solve` has no tab of its own (it shares **6. Haemodynamics**), and
**9. Additional measurements** is a tab with no stage function (its settings are read by
`export_results`). **7. Post processing** (`post_process`, between Haemodynamics and
Perturbations, so the network is edited solved) has no settings: its tab is its own page (`gui/_widget.py`'s
`_post_processing_controls`), with no "Run from this stage". Its Review dropdown
(`gui/post_processing.REVIEW_LISTS`) chooses the one list Scan network makes -- only that
one, so a scan does no more work than it needs -- optionally only within a radius of the
view centre; None, the default, lists, marks and zooms to nothing. Each item a list rings
(and each 4+ junction) is magenta, ring and vessels (the `listed` status), the chosen one
cyan (`marker_colours`); with no edit armed, a click on a ring chooses its item without
moving the camera (`marker_under_click`, `marker_pick_um`). "4+ junctions" shows the
junction list, table and Delete/Leave/Split buttons. The other thirteen are
`graph/network_review.py`'s read-only finders, each returning `ReviewItem`s ranked as the list
shows them: junctions close together (Merge into one junction, `graph.merge_junctions`),
hairpin vessels, dead ends by kind (`graph.classify_dead_ends`, the lumen-artefact report's
dead ends; Trace on from the tip starts Add vessel there; a Show filter lists one kind,
`gui/post_processing.DEAD_END_SHOW`, and shown pieces with no junction get "Remove all N
listed pieces", `remove_listed_pieces`: each deleted through `delete_vessels` with its own
decision and CSV row, skipping one a vessel was drawn onto since the scan and refusing one
holding a boundary node), two vessels in one lumen, facing
dead ends the join left open (`facing_ends.facing_dead_end_pairs`, with why; Join the two
ends), segmented vessels without a centreline (`mask_recovery.uncovered_mask_regions`),
unsolved pieces (`graph.unsolved_pieces`), flow against the branch orders, flow outliers,
boundary-node issues, terminal nodes (`find_terminal_nodes`: the inlets, the outlets, then
dead ends leaving the image widest first, then the other dead ends; Make inlet / Make outlet /
Remove inlet/outlet, `graph.set_terminal_role`: the lists kept on the graph as
`post_processing.BOUNDARY_ROLES`, the graph marked pending, the last inlet or outlet never
removed; `post_process` takes them in place of the run's, `_take_edited_terminal_roles`, and
draws the boundary-nodes layer again), unmeasured diameters and diameter jumps (Set diameter,
`graph.set_vessel_diameter`: an override, the vessel *not* marked edited, so `post_process`
does not measure it again and replace it). The four that read the segmented image run off
the GUI thread, and after an edit only drop what it took away (the list says it may be out
of date); the others are made again. Their thresholds only decide what is listed. Each
decision is a `ReviewDecision`, matched by kind and centre, kept with the run (the
`.haemorun`'s `review_decisions`) and appended to `{stem}_{kind}_review.csv` with the item's
measures, so a rule can be fitted to the marks later; the scan after a regenerate says what
it changed (`network_summary` / `summary_change`). "Short loops" lists the short loops still to review
(`graph.short_loops`, up to `lumen_loops.LOOP_SEARCH_UM` round), shortest first: picking one
zooms to it -- in its own plane in 3D (`gui/post_processing.loop_view`) -- and lists its sides
(`lumen_loops.loop_sides`, cut where vessels or boundary nodes meet it), and Delete this side
(`graph.delete_loop_side`) or Keep decides it. A decision is a `LoopReview`, matched to a loop
by centre and length rather than node ids (deleting a side renumbers the edges beside it),
kept with the run (the `.haemorun`'s `loop_reviews`) so a kept loop is not listed again, and
appended to `{stem}_loop_review.csv` beside the VTK output with `lumen_loops.loop_measures`,
so a rule can be fitted to the marks later. Its
"Prune disconnected and dead-end branches" button (`graph.prune_disconnected_branches`) applies
the same rule as `remove_disconnected`; the boundary lists are trimmed to the pruned graph when
the `post_process` stage runs. Its Regenerate graph runs
the `post_process` stage alone (edits in line, Haemodynamics rerun) while a run is paused there
(`gui/run_state.py`), else goes on to Export; Continue
carries a paused run on. A `.haemorun` saved in the old order (Post processing before
Haemodynamics) is reordered on load (`gui/run_snapshot.py`'s `with_post_process_checkpoint`).
Plus what a run reports through:
`run_pipeline_stages(settings, schema, progress=callback)` hands the callback a `ProgressEvent`
as each stage starts, finishes or fails, and one per topology step inside graph building.
`log_progress` is the ready-made console consumer; the napari panel's bars are the other one.
Nothing here imports a GUI or a progress-bar library — the callback is the whole mechanism.
A finished or failed stage's event carries its wall time (`ProgressEvent.seconds`), which
`log_progress` writes on its line, so the panel's run log shows every stage's time.

## pipeline/provenance.py

**`pipeline/provenance.py`** — every run writes `{stem}_run_manifest.json` into its output
folder (`vtk_output_prefix`'s), with `{stem}_run_settings.yaml` beside it (a config `--config`
loads): the settings after `Schema.upgrade` as the run *started*, the version, the git commit
and dirty paths (null for a wheel, or with no git; read once per process, since a long-lived
panel runs the code it imported), Python, platform and key dependency versions, each input
file's size, mtime and sha256 (whole up to 64 MiB, above that 32 evenly spaced 1 MiB blocks:
`checksum_method` says which), and each stage's start, end, wall time and peak RSS. Peak
memory uses the standard library (psutil is not a dependency): exact for a stage that raised
the process peak, else sampled every 0.2 s (`MEMORY_METHOD`); Windows has no worker-process
figure. `run_pipeline_stages` records through `record_run`, and the two hand-written examples
wrap their stages in it. A failed run writes one too (`status: "failed"`, the error, the
stages finished); a run that fails in its first stage writes none. Writing it never fails a run.

## Post processing stage (from the pipeline overview)

6. **Post processing (hand edits)** — on the *solved* network: `post_process` brings vessels edited by hand in the panel (`graph/post_processing.py`) in line with the rest: lengths, mask tags and the boundary nodes those tags give where the edits touched them (the small-mask arteriole/venule boundary nodes and the Large_Art/Large_Ven hand-off nodes), branch orders over the whole network (a piece an edit cuts off from the inlets loses its orders), the kinds of the dead ends the edits left (judged against the segmentation as the build judged every dead end), diameters by the run's own methods (only the edited vessels are measured), zero-resistance bridges where one opens into a thick vessel; it also reads each vessel the edits drew against the segmentation as graph building reads its bridges (`graph.record_drawn_vessel_support` with `preprocessing.MaskSupport` at the run's bridge limits: `drawn_mask_fraction`, `drawn_background_run_um`, `drawn_beside_vessel` on the edge, `drawn_vessels` in the summary, the panel's log line and the hover's "drawn vessel" line), a diagnostic only -- Harvey draws vessels where the segmentation is missing, so nothing is refused; then `solve_post_processed` reruns the haemodynamics on the edited network (network handling, resistances, solve). A no-op on an unedited network, which is not solved again. With `mid_run_postprocessing` on, a panel run pauses before it (`run_pipeline_stages(stop_after="solve")`) and Continue resumes at it  
