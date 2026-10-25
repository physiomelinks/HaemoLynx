# Batch run and batch statistics

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## Batch run

**Batch run** (2026-10-19, `PLAN_batch_run.md`) — the pipeline once per segmented image in a
folder. `io/batch_matching.py`'s `match_batch(BatchFolders(segmented, raw=, large=, small=),
sizes)`: the images are the segmented folder's image files that are no mask, probability map
or raw image of another file there (`batch_inputs`); the raw image is read by the mask rule
without role words (its identifying words, channel names such as `C1-` left out, all in the
segmented name: `read_raw_name`); masks in the segmented folder by `find_vessel_masks`'s rule
(now with `folders=`/`exclude=`); a mask folder of its own says the size, so each file there
goes to the image sharing most of its name from the start, global unless it names a vessel
type (`mask_owners`: Harvey's `..._Simple Segmentation_intermediate.h5`). A blank folder is
the segmented one. Ties are named, never guessed (`BatchMatch.choose`, `choose_alike`).
`pipeline/batch.py`: `item_settings` (the image's files, every output folder in
`{output}/{segmented stem}/`, no mid-run pause), `check_batch` (refuses node IDs, synthetic,
ilastik, a single pericyte/cell/tissue file in effect; warns on coordinates),
`batch_preflight`. `gui/batch_run.py`'s `run_batch` runs them in turn with a fresh
`ResultLayers` + `StageCheckpoints` each and writes `{stem}/{stem}.haemorun`; a failure is
recorded and the batch goes on, `batch_summary.csv` is rewritten after every image,
`skip_done` skips an image whose `.haemorun` is there. Headless: `python -m
haemolynx.gui.batch_run --config C --segmented D [--raw/--large/--small D] --output D
[--dry-run] [--only STEM ...] [--rerun] [--condition-a P --condition-b P [--paired [--subject-pattern RE] [--paired-aggregate none|median|mean]]] [--exclude-section S ...] [--compare-only]`;
every image passes its checks or nothing runs. Its log lines carry the time
(`configure_console_logging(timestamps=True)`), so each step's time reads off a batch log.
**Conditions and the batch's statistics** (2026-10-19, `statistics/batch_comparison.py`): up
to two conditions, each one or more phrases (comma-separated) a file name holds, ignoring case
(`BatchConditions`: an image in both or neither is left out of the test; two conditions alike
fail the checks). Once the images are done (or cancelled), `run_batch` reads back each image
that ran or was skipped as done -- its `{stem}_statistics.csv` and `{stem}_branch_statistics.csv`
(numbers only: no Metadata, no per-node rows named by node ID, no text, lists or N/A; a branch
order is a section `Branch Order BO3`) -- one value per image, and writes beside the summary
`batch_statistics_by_image.csv` (a column per image, with its condition),
`batch_statistics_summary.csv` (per metric, each condition and All images: N, mean, SD, SEM,
median, quartiles, IQR, min, max, range, CV) and, with two conditions,
`batch_condition_comparison.csv`: each condition's N, median, mean and range, the difference
and ratio of medians (B - A), a two-sided Mann-Whitney U (one image one sample; scipy's exact
test for small samples without ties) with Benjamini-Hochberg q over every metric tested, the
smallest p the test can reach at those N (0.1 at 3 against 3: a note says no difference can
reach 0.05), the rank-biserial r, and Welch's t with Hedges' g beside it; a metric with fewer
than 2 images in a condition or one value everywhere is listed, not tested. Writing them never
fails the batch (`batch_run.write_statistics`). The window's Condition A/B boxes fill a
Condition column as typed, and its Compare statistics button (and `--compare-only`) writes
them from the images already in the output folder, running nothing. Images of one animal are
not independent samples, so the comparison is a choice (the window's Comparison menu,
`--paired`): independent images (above) or **paired by subject**: each image with the image
of the other condition whose name is most alike once the condition phrase is taken out of both
(Harvey, 2026-10-19: pair on the terms given, by similar names, not on one batch's naming).
Only images of one subject pair -- the part of the name before the phrase, separators trimmed,
case ignored (`28_3_d_contra_FITC` is `28_3_d`), or a `--subject-pattern` regular expression's
first group (`^(\d+_\d+)` reads the animal, `28_3`); within a subject one image a side pair,
and more are matched one to one, closest first (`BatchConditions.name_words`: the most words in
common from the start, then anywhere; `E14_WT_1` with `E14_KO_1`), a tie left out, never
guessed. `--paired-aggregate median|mean` (the window's "Several images of a subject") instead
pairs each subject's median or mean over all its images in a condition, metric by metric
(Harvey's AIA batch by animal: 3 pairs, smallest p 0.25). `BatchConditions.pairing` leaves out,
with its reason, every image in neither or both conditions, with no subject the pattern reads,
with no partner, or in a tie. The pairs (with the images in each value) and every image left
out go in `batch_condition_pairs.csv` (the table's Condition cell says the same as you type);
the comparison is then over the pairs: the Wilcoxon signed-rank test (zero
differences dropped) with Benjamini-Hochberg q, W+, the smallest p it can reach (2 / 2^n: 0.5
at 2 pairs, so nothing reaches 0.05 below 6), the matched-pairs rank-biserial r, and the paired
t with Cohen's dz. The summary still covers every image of each condition.
**Sections left out** (2026-10-20): `--exclude-section "Dead Ends"` (repeatable or
comma-separated; `write_batch_statistics(exclude_sections=)`, `run_batch` / `write_statistics`'s
keyword, `BatchSetup.exclude_sections`, the window's "Leave out sections" row) drops those
sections and their subsections (`Strahler Network` takes `Strahler Network > Orders > 1`;
ignoring case, `excluded_section`) from all three CSVs before anything is summarised or
tested, so the Benjamini-Hochberg q is over the metrics left; a name matching no section is
said in the message, and a batch records them in `batch_settings.yaml`
(`# batch_exclude_sections:`). For Harvey's AIA contra/ipsi batch, whose patella dead ends are
segmentation fragments (his P1 marks: none a real blind end): rebuilt from its per-image CSVs
with `exclude_sections="Dead Ends"` it gives `outputs/aia_all8`'s hand-edited CSVs byte for
byte, and without it the originals in `with_dead_ends/`.
In the panel: the Input tab's Batch run button (above the Input row) opens
`_widget.py`'s `_BatchRunWindow` (folders, the matched table with ticks and tie choices,
Check, Use this batch, Leave batch run; double-click a finished row to load its
`.haemorun`). While a batch is set (`batch_mode.setup`, a `gui/batch_run.BatchSetup`, panel
state, not settings) the six path rows it fills show "Batch run" inside the row
(`_show_batch_row`, from `apply_prerequisites`; the rows' values are untouched) and Run
pipeline runs `on_run_batch` -> `_run_batch_in_background`: nothing is drawn.


**Phase 1 of `PLAN_batch_panel.md`** (2026-10-23). A demo batch in a long folder lost a stage
checkpoint pickle at 260 characters; the `.haemorun` was whole, since checkpoints are kept in
memory and the pickles beside it are side files nothing reads back. Measured on Windows 11 with
long paths off: a file path of 260 characters and a folder path of 248 are refused. Each image's
checks now refuse a longer one (`pipeline/batch.py`'s `check_output_path_lengths`, an error by
Harvey's choice), counting each stage's checkpoint, the `.haemorun`'s `.part`, the export
dictionary's run-folder files these settings write (`WRITTEN_WHILE`; figures not counted) and
each perturbation's folder. The window's table says which images a batch with Skip done would
skip (`stems_done_before`) and what the folder's last `batch_summary.csv` said of the rest
(`read_summary_statuses`), and follows a running batch image by image. Conditions with
Statistics off warn (`statistics_warning`). In the panel a set batch greys Check segmented
image, both optimisers and Run from this stage, and Run checks checks every ticked image. Save
run stays on: it saves the run shown. Load run was found to cancel a running batch (it stops the
live run), so it is refused while one runs, and otherwise leaves the batch set. A batch left while
it ran is no longer set again when it ends.

**Phases 2-4 of `PLAN_batch_panel.md`, and the caches** (2026-10-23, Harvey's decisions in the
plan). The window runs a batch itself (Run batch; Run pipeline still runs a set batch) and can
stop it after its image (`run_batch(stop_requested=)`, `BatchSummary.stopped`). The panel's
report and the window's status give `batch_progress_text`: the image running, done and failed
counts, the time so far and an estimate from the mean time of the images done. The table gains
Time, Peak RAM and Vessels, sorts by any heading, and has Tick all/none/failed and, per image,
Open run, Open folder, Show error (the traceback, `BatchResult.error_detail`) and Run again
(`BatchSetup.only`). A Comparison tab lists `batch_condition_comparison.csv` by q
(`comparison_table`), with a button that opens the CSV. Folders are matched as they are typed;
an output folder holding a `batch_settings.yaml` offers its batch (`read_batch_settings`), which
fills the window and loads its settings into the panel. Greyed mask rows name their switches
(`mask_switch_hint`), and a line counts the settings that differ from the defaults
(`settings_changed`). Not done: remembering the window between napari sessions (the panel has
no per-user store yet); Harvey does not want it (2026-10-24).
Found on the way: Check used done-before from when the folders were matched, so after a batch it
said "ready (checked)" of images Skip done would skip; it now reads them again. The summary's
`peak_rss_bytes` was the manifest's process peak, which in one process includes every earlier
image; it is now the largest of the image's own stage peaks. On the six-image demo the working
set did grow, about 35 MB an image (440 to 619 MB): unmeasured on a real batch.
Each image's on-disk caches go once it ends (`clear_image_caches`): its decompressed input in
`memmap_directory`, scratch volumes Windows kept while they were mapped
(`memmap_support.delete_dropped_memmap_files`), its stage checkpoint pickles and its
thick-vessel region cache; its outputs stay. On a two-image low-RAM batch with thick-vessel
skeletonisation (Windows 11), each image had left two memmap-folder files, eight checkpoints
and a region cache; with clearing, the memmap folder is empty and the `.haemorun`, graph and
skeleton remain. A re-run of one image now finds the thick-vessel region again.

**Single runs check their paths too** (2026-10-24). An E14.5 run with its output under
a worktree's `outputs/synthetic_e14p5/template/` ran for 8 minutes and then failed at Export
writing `{stem}_betweenness_communities_resistance.json`, a 266-character path (the E14.5 stem
alone is about 115). `check_output_path_lengths` is now part of every run's `preflight`, not
only a batch image's (`batch_preflight` no longer adds it again). `output_paths` reads a single
run's settings as a config gives them (strings, relative paths from the folder the run starts
in), names the run's files with the stem they will carry (ilastik's output's while ilastik
segments the image), and counts what only a single run writes: ilastik's segmented images and
their `.haemolynx.json` records (main image and vessel masks, named by
`io.vessel_masks.ilastik_vessel_mask_output_path`) and a synthetic network's
`_template_comparison.csv`. The files written by hand in Post processing are still not counted.
The panel's checkpoints and `.haemorun` are counted for a run from the command line too, which
writes neither: by at most 22 characters with Statistics off, none with it on.

**ilastik batches, the other per-image files, and the button** (2026-10-24, `PLAN_batch_run.md`'s
"Later"). With `use_ilastik_segmentation` on, a batch's images are its raw folder's
(`BatchFolders(from_raw=True)`, `pipeline.batch.batch_from_raw`; `batch_inputs(raw=True)` takes every
image file there that is no segmentation, probability map or mask of another). Each image is named
after the segmentation ilastik will make of it, `{raw stem}_segmented`
(`io.ilastik.ilastik_segmented_stem`, now the one naming rule `segment` and `segmented_input_path`
share too), so its folder, `.haemorun`, outputs and statistics all carry that stem, as every batch
image's do. Its segmentation goes in `{folder}/segmentations/` with its provenance record, and
ilastik's reuse is a single run's (`ilastik_reuse_existing_output`). A blank mask folder is then the
raw folder, and masks are read against the raw names. Each image's checks refuse a batch matched one
way while the settings say the other ("match the folders again"). ilastik's vessel masks stay
refused in a batch, now only while their masks are in use: they read one raw file
(`ilastik_unsegmented_*_image_path`) for every image, and which file of each image they should read
is Harvey's to say. While a batch is set the panel greys rows by `BatchSetup.gates`: a row the batch
fills reads as set when a ticked image has that file, so ilastik's switch and the raw image's other
switches are live with the panel's own raw row empty (before, they were greyed in batch mode).
The other files a run reads beside its image (`io.batch_matching.FILE_ROLES`: pericyte mask, cell
mask, the 3D distances' vessel mask, tissue image, mask of blocks) each come from a folder of their
own, given to the image whose name they share the most words with from the start, then the most
words, then the fewest of the image's words they lack (`_Owners`, shared with `mask_owners`:
`E14_cells` is `E14`'s, not `E14_2`'s, where before it tied). A file already matched to the image
(its raw image, a mask) is never one of them, so a tissue folder that is the raw folder gives the
other channel, and choosing a tied raw image takes it out of the tissue image's tie. A role is
matched only while the run, or a perturbation of a type reading it, reads it
(`pipeline.batch.batch_file_roles`, `_readers`); the window shows its row only then. With a folder,
each image's setting and every perturbation override naming it read the image's own file
(`item_settings`); without one, a file the run or a perturbation would read for every image is
refused (it was refused for the run's own settings before, and for any `capillary_block_mask`
entry, whatever it named). A role with nothing to fall back on (pericyte, cell, blocks) and no file
for an image is a problem; the distances' vessel mask and the tissue image fall back to the
segmentation and the raw image, with a note. `batch_summary.csv` gains a column per role, and
`batch_settings.yaml` records the folders given and `# batch_from_raw: yes`. Command line:
`--pericyte`, `--cell`, `--vessel-3d`, `--tissue`, `--capillary-block`; with ilastik on in the
config, `--raw` lists the images and `--segmented` is not read.
The Batch run button greys under a synthetic network and while a run of one image or an optimiser
goes (`batch_run.batch_button_greyed`, following `RunState.on_change`, which every way a run ends
reaches), never while a batch is set or running: its window is where a batch is followed, stopped
and left. Checked by the tests only (`tests/test_batch_*.py`, `test_gui_batch_run_widget.py`; the
ilastik batch end to end with a stand-in for ilastik on the seven-vessel fixture), not on a real
stack or by a tester.

**The window's eight rows, and its hover text** (2026-10-24, `19aea11b`, `PLAN_batch_panel.md`'s loose ends).
Phase 4 said the window kept eight table rows in view after Check, but nothing tested it. Built on
the real display (1920 x 1032 free of the taskbar, ten fixture copies), the window opened at 800 px
with 7 rows in view, and Check's report box, opened to a third, left 4. With the other files' rows
`3d9d071a` added (cell, 3D-distance vessel, tissue) it was 4 and 2. The window now grows, as far as
its screen allows, to keep `VISIBLE_ROWS` rows whole (`_BatchRunWindow._fit_rows`): on opening,
when a folder row or the last batch's button appears, and when the report box opens. The box opens
to a third or `REPORT_LINES` (6) lines, never at the table's cost unless the screen is too short
for both, and then gets `REPORT_MIN_LINES` (3). Measured after it: 8 rows before and after Check,
the window 819 px, 915 px with three other-file rows; with all five shown, 7 rows after Check and
a 3-line box, the window as tall as the screen. Hover text is now checked on the built window:
every control in it, shown or not, including a tie's choice box. Its two tabs had none and now
do. Neither was tried by hand by Harvey or a tester.

**ilastik's vessel masks in a batch** (2026-10-24, Harvey's decision, `PLAN_batch_run.md`). They
were refused, since each classifier read one file for every image. Now every classifier segments
the run's raw image by its own channels (`ilastik_*_channels`; CLAUDE.md, "Segmentation"), so each
image's masks come from its own raw image and go in its own `segmentations/` folder. An image with
no raw image fails its checks (`check_ilastik_vessel_mask_channels`).

**A synthetic template per condition** (2026-10-25, Harvey's B5a, `PLAN_batch_templates.md`):
`run_batch` ends by writing `{condition}.haemobatch` (`batch.haemobatch` without conditions) of the
images that finished or were skipped as done (`write_batch_templates`); an image in neither
condition is in none, a run that cannot be a template is left out and named, and writing them never
fails the batch. What the file is and how it is averaged: `docs/notes/synthetic.md`.
