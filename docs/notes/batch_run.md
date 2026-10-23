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
