# Batch run panel: improvements (plan, 2026-10-23)

> **Status (2026-10-23): Phase 1 implemented, not yet committed; Phases 2-4 are plans.** It
> builds on `PLAN_batch_run.md` (implemented 2026-10-19) and leaves that plan's "Later" list
> (ilastik batches, more per-image inputs, parallel images) for after this one. Harvey chose:
> a path past the limit is an error (decision 1), and conditions with Statistics off are a
> warning (decision 5). Phase 1 changed from this plan where the code showed it was wrong; see
> "Phase 1 as built".

## How this was looked at

At `1cc56273`, the Batch run window was built off-screen (`QT_QPA_PLATFORM=offscreen`) over six
copies of `tests/data/seven_vessel_noisy_3d.tif`, named like Harvey's AIA batch
(`28_3_d_contra_FITC_Simple Segmentation.tif` beside its raw `28_3_d_contra_FITC.tif`, with one
raw tie), and screenshotted at each step. Two of the images were run for real through
`run_batch` (21.6 s). The window was then matched again and screenshotted. The scripts are in the
session scratchpad, not the repo. Nothing was checked on a real stack or by a tester.

## What was found

### Defects (each gets a regression test)

1. **A file past Windows' path limit is not written, and some writers only log it.** The first
   image's `..._checkpoint_build_haemodynamic_model.pkl` was 260 characters long. Windows refused
   it (`FileNotFoundError`), `StageCheckpoints.record` logged the error and carried on, and the
   image was recorded `done`. *Corrected while building Phase 1:* its `.haemorun` does hold that
   stage, since checkpoints are kept in memory and the pickles beside it are side files nothing
   reads back. What the limit costs is any file past it: the manifest's writer also only logs,
   and others fail the image at export, hours in. The second image's path was 256 characters
   and was written. A batch names every file twice,
   `{output}/{stem}/{stem}_...`, so long names like Harvey's (`..._Simple Segmentation`, 38
   characters) reach the limit in a moderately deep output folder. The machine has
   `LongPathsEnabled` unset.
2. **The table says "ready" for images it will skip.** After a batch, pressing Match again shows
   every image as `ready`, although Skip done will skip the two that ran. The statuses from the
   finished run are lost too, and they are never read back from `batch_summary.csv` once napari
   is reopened.
3. **The window does not change while a batch runs.** `on_item` only updates the panel's report
   line. The window's table is filled once, by `show_finished`, at the end.
4. **Batch mode does not grey what acts on one image.** `PLAN_batch_run.md`'s status already
   says this is not done. Run from this stage (`on_run(start_from=...)`) skips the batch check and
   runs the Input tab's own image, while the rows read "Batch run". Check segmented image, Optimise
   settings and Save run do the same.
5. **Conditions with Statistics off do nothing, and nothing says so until the end.** `statistics`
   defaults to off. With conditions typed and Statistics off, Check passes and the batch runs. Only
   the end message says "No image of the batch has a statistics CSV (is Statistics on?)".
6. **Check floods the console.** `batch_preflight` calls `pipeline.preflight`, which prints its
   whole report (`report.print("Preflight")`) once per image.

### Things to check before relying on them

7. **The raw-volume cache keeps every image.** Under `use_memmap_loading`, each image's
   decompressed raw volume stays in `memmap_directory` (`io/raw_volume_cache.py` has no eviction).
   A batch reads each image once, so a batch of large stacks may fill the disk. Measure it on two
   or three real stacks before changing anything.

### Usability, from the screenshots

- When Check fills the report box, the table shrinks to **one visible row** at the window's
  default 900 × 520.
- **The greyed mask rows don't say which switch turns them on.** They read "not used: this run's
  settings use no such masks".
- **Results are hard to find.** A double-click opens a finished image, but the table's hover text
  doesn't say so. A failed image shows only its error's first line; the traceback is only in the
  log. Nothing opens the output folder or the batch statistics.
- **Match is a separate step.** Typing a folder or choosing one with Browse does not match it.
- **One grid holds everything**, with no grouping between the images and the conditions.
- **The batch is forgotten** when napari closes. `batch_settings.yaml` records the folders,
  conditions and sections as comments, but nothing reads them back.
- **No time estimate.** There is no "image 3 of 12, about 2 h left", and no way to stop an
  overnight batch between images: Cancel stops the current image partway.

## Phases

### Phase 1 as built (2026-10-23)

- **Path length: an error.** `pipeline/batch.py`'s `check_output_path_lengths`, in every
  image's `batch_preflight`. Measured here: a file of 260 characters and a folder of 248 are
  refused, so the limits are 259 and 247 (`path_length_limits`, read from `LongPathsEnabled`).
  It counts only files these settings write (`WRITTEN_WHILE`): counting the cell-distance CSVs a
  run with that measurement off never writes refused a batch that would have run. There is no
  "stage not saved" status and no `warnings` column: the premise for them was wrong (above).
- **Done before and the last batch**, **the live table**, **the Statistics warning** and
  **the quiet Check**: as planned below.
- **Batch mode greys** Check segmented image, both optimisers and Run from this stage, and each
  handler refuses if pressed anyway. **Save run is not greyed:** it saves the run shown, never
  the batch's. **Run checks** with a batch set checks every ticked image.
- **Found and fixed on the way:** Load run (a double-click in the table included) cancelled a
  running batch, because loading stops the live run. It is now refused while a batch runs.
  Otherwise Load run leaves the batch set, since the run shown is then one image's. And a batch
  left while it ran was set again when it ended.

Tests: 8 in `test_batch_settings.py`, 6 in `test_batch_run.py`, 8 in
`test_gui_batch_run_widget.py`.

### 1. Defects (1-6), as planned

- **Path length.** `pipeline/batch.py` gains `longest_output_path(settings, match, output_root)`.
  It returns the longest file path the image will write: its checkpoint names
  (`stage_checkpoints`), the `.haemorun`, the manifest, and the perturbation and sweep folders when
  those are on. `check_batch_item` then adds a problem when that path passes 259 characters on
  Windows without long paths. Whether that problem is an error or a warning is decision 1.
  `run_image` also counts what failed to save: `StageCheckpoints.record` failures and a failed
  snapshot. `BatchResult` gains `warnings`, and its status becomes `done (1 stage not saved)`
  rather than `done`. The summary CSV gains a `warnings` column, and its entry in
  `pipeline/export_dictionary.py`.
- **Done before.** `table_rows` takes the output root's state. A `.haemorun` already there reads
  `done before: will be skipped`, or `done before: runs again` with Skip done off. When the output
  folder has a `batch_summary.csv`, its statuses fill the Status column until Check or a run
  replaces them. `BatchSetup` gains `last_statuses`, read by `read_batch_summary(output_root)`.
- **Live table.** `_run_batch_in_background`'s `item_changed` also calls
  `window.show_progress(index, result)`. That sets the row's Status to `running`, then `done`,
  `failed: ...` or `cancelled`, and selects the row that is running.
- **Grey in batch mode.** `apply_prerequisites` greys Run from this stage, Check segmented image,
  Optimise settings and Save run while `batch_mode.setup` is set. Their hover text says why
  ("acts on one image: leave the batch run first"). `on_run(start_from=...)` refuses while a batch
  is set, in case a script calls it.
- **Statistics off.** `check_batch` adds a problem when conditions are set and `statistics` is
  off. Whether it is an error or a warning is decision 5. Compare statistics with no CSVs already
  says so.
- **Quiet Check.** `batch_preflight` gets `pipeline.preflight(..., echo=False)`, a new keyword
  that defaults to today's printing, so the CLI and the examples still print. The window's report
  box already holds the text.

Tests:
- `test_batch_settings.py`: a path over the limit is flagged, and one under it is not. The
  platform and long-path reads are patched, so this runs on Linux CI.
- `test_batch_run.py`: a checkpoint write made to fail gives status `done (1 stage not saved)` and
  a `warnings` cell. `table_rows` shows "done before" both ways. Statuses are read back from a
  summary.
- `test_gui_batch_run_widget.py`: the four controls are greyed in batch mode, and Run from this
  stage refuses. `show_progress` changes the row.
- `test_pipeline_invariants.py` or `test_parsers_checks.py`: `echo=False` prints nothing.

### 2. Following a batch while it runs

- **One line**, in the window and the panel's report: "Image 3 of 12: 28_4_a_contra -- 2 done,
  0 failed, 41 min so far, about 1 h 50 min left". The estimate is the mean wall time of the
  images done in this batch, and is only shown once one has finished. It is worked out by a pure
  `batch_progress_text(summary, now)` in `gui/batch_run.py`.
- **Columns filled in as each image ends:** Time, Peak RAM, Vessels. All three are already in
  `BatchResult`.
- **Stop after this image.** This needs a button and a flag that `run_batch` checks between images
  (`stop_after_current`). The image that is running finishes, gets saved, and the rest are
  recorded `not run`. Cancel keeps its present meaning. Whether to add this is decision 3.

Tests:
- `batch_progress_text` on hand-made summaries: no image finished, some finished, cancelled.
- `run_batch` with `stop_after_current` set during image 1 of 3: image 1 is `done` and the other
  two are `not run`.
- GUI: the line and the columns update from `on_item` events.

### 3. Acting on the results

- **Row actions**, from a right-click menu and from buttons under the table that act on the
  selected row:
  - **Open run**: today's double-click; the hover text will say so.
  - **Open folder**: the image's folder in Explorer, Finder or the file manager, through
    `QDesktopServices.openUrl`.
  - **Show error**: the image's traceback. `BatchResult` keeps it; the summary CSV keeps only the
    first line, as today.
  - **Run again**: the batch limited to that one row, with Skip done off.
- **After the batch:** **Open batch folder** and **Open comparison** (the comparison CSV, with the
  system's program for it). The end message says that failed images are run again by Run pipeline
  while Skip done is on, which is already true: a failed image writes no `.haemorun`.
- **The comparison in the window** (decision 6): a second tab that lists
  `batch_condition_comparison.csv`'s rows sorted by q, with N, medians and p. It only reads the
  CSV.

Tests:
- `BatchResult` keeps the traceback and the summary row still holds one line.
- Run again builds a setup with one match and `skip_done=False`.
- GUI: every action is enabled only on a row it can act on, and each one calls its handler
  (patched).

### 4. Setting up a batch

- **Match on change.** Leaving a folder field, or choosing a folder with Browse, matches the
  folders again. The button becomes "Match again".
- **The batch from a folder.** Choosing an output folder that holds `batch_settings.yaml` offers
  "Use this folder's last batch". That fills the folders, conditions, comparison, subject pattern,
  aggregate and sections from its `# batch_...:` lines, which a new
  `read_batch_settings(path)` parses. Whether it also loads the panel settings (the YAML's body)
  is decision 4.
- **Greyed mask rows** name their switch and its tab, from the schema
  (`graph.vessel_masks_in_use`'s settings) rather than hand-written text.
- **Layout.** Three boxes: *Images* (the folders and Skip done), *Conditions and statistics* (A/B,
  comparison, subject pattern, aggregate, sections), then the table. The report box sits in a
  splitter under the table and starts folded, so Check no longer squeezes the table. The default
  size grows to whatever fits eight rows. Choices are remembered per user through `QSettings`,
  never in the run's settings.
- **Table conveniences:** tick all, tick none and tick failed; sort by any column.
- **Which settings run:** one line, "Settings: the panel's, 14 differ from the defaults [Show]".
  Show lists them.

Tests:
- `read_batch_settings` reads back exactly what `write_batch_settings` writes, including blank
  folders, paired designs and sections.
- GUI: the folder from a `batch_settings.yaml` fills the window. Leaving a field matches again.
  The mask placeholders name a real setting. The box layout keeps eight rows visible after Check.
- `test_gui_tooltips.py`: every new control has hover text.

## Docs

- `CLAUDE.md`'s "Batch run" entry: a sentence at most. History goes in `docs/notes/batch_run.md`.
- `docs/export_dictionary.md` regenerated for the summary's new columns.
- The Input tab's help page (`gui/stage_help.py`) is Harvey's to write. Nothing is added there.
- `RELEASE_REVIEW.md`: no score changes unless a check's evidence changes. `usability.6d` (hover
  text checked by scanning the built panel) is not met by adding entries to `chrome_tooltips.py`.

## Order

Phase 1 first: defects 1 and 4 give wrong or misleading results today. Phases 2 and 3 can then be
built side by side, since they touch different parts of `_BatchRunWindow`. Phase 4 comes last,
because the layout change moves every widget the other phases add. Each phase is one commit or
more, with its tests, and the GUI tests run before each commit (`pytest -m gui
tests/test_gui_batch_run_widget.py tests/test_gui_tooltips.py`).

## Decisions to confirm with Harvey

1. **A path over Windows' limit:** refuse the image (an error, so nothing runs until the output
   folder moves or long paths are turned on), or warn and run with that stage unsaved?
2. **A Run button in the window**, or keep Run pipeline the only start, as `PLAN_batch_run.md`
   chose?
3. **Stop after this image:** add it beside Cancel?
4. **A batch from its folder:** fill the window only, or load the panel settings from
   `batch_settings.yaml` too?
5. **Conditions with Statistics off:** an error, or a warning?
6. **The comparison in the window**, or only a button that opens its CSV?
