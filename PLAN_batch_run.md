# Batch run (plan, 2026-10-18)

> **Status (2026-10-24): complete, bar what waits on Harvey** (the list at the end of this box).
> - **v1, phases 1-5** (2026-10-19), with one change from the plan below: a mask folder given on
>   its own says the mask size, so a file there need not name it
>   (`io/batch_matching.mask_owners`). Harvey's first batch names its small masks
>   `..._Simple Segmentation_intermediate.h5`, which the name rule reads as no mask.
> - **Conditions and the batch's statistics** (2026-10-19): two conditions read from the file
>   names, the statistics side by side and tested between them (`statistics/batch_comparison.py`).
> - **`PLAN_batch_panel.md`** (complete 2026-10-23) replaced "The window" below with its Phase 4
>   layout, and added Run batch, Stop after this image, the row actions and the Comparison tab.
>   Its Phase 1 (`d4bdfd0c`) greys Check segmented image, the optimisers and Run from this stage
>   while a batch is set. **Save run stays on**, by design: it saves the run shown, never the
>   batch's.
> - **The button** (`fa7fd594`, 2026-10-24): greyed under a synthetic network and while a run of
>   one image or an optimiser goes. It is never greyed while a batch is set or running: its window
>   is where a batch is followed, stopped and left.
> - **ilastik batches and more per-image inputs** (`3d9d071a`, 2026-10-24), from "Later": see
>   "Later, as built" below.
> - **"A CLI"** (Later) is `python -m haemolynx.gui.batch_run` (`gui/batch_run_cli.py`), not a
>   flag of the example script.
> - **Harvey's decisions (2026-10-24):** decisions 1-5 kept as built; images stay one at a time;
>   ilastik's vessel masks, in single runs and batches, read chosen channels of the run's raw
>   image, kept separate, each classifier its own (not built yet: "To build" below).
> - **Waits on Harvey:** the Input tab's help text for Batch run (`gui/stage_help.py`, his to
>   write).
>
> Line numbers below are as of 2026-10-18 and have moved since.

A **Batch run** button on the Input tab opens a window where the user gives up to four folders:
segmented images, raw images, large vessel masks, small vessel masks. The panel matches each
segmented image to its raw image and masks by file name, using the rule the panel already uses
to find masks beside an image (`io/vessel_mask_discovery.py`). **Run pipeline** then runs the
pipeline once per segmented image with the panel's other settings. Each image gets its own
folder, named after the image, holding its `.haemorun` and the outputs the settings ask for.

While a batch is set, the per-image path rows on the Input tab (`input_path`, `raw_image_path`,
the four vessel-mask paths) show **"Batch run"** in place of a file name. Run pipeline still
starts the run. Leaving batch mode puts the rows back as they were.

## What already exists (and what it means here)

- **Matching.** `find_vessel_masks(references, sizes)` reads mask names against an image's
  name: whole words, the role words (`large`/`small`, `arteriole`/`venule`/...) that a mask's
  name adds to the image's, the rest a subset of the image's name, and closest-prefix wins with
  ties left unchosen. It only searches the **images' own folders** (`image.parent`), so the
  batch needs a `folders=` argument. The rule stays the same; only where it looks changes.
- **A `.haemorun` needs no viewer.** `ResultLayers` (`gui/results.py`), `StageCheckpoints.record`,
  `capture_run` and `write_run_snapshot` (`gui/run_snapshot.py`) are all pure. `_run_in_background`
  (`gui/_widget.py`) only touches Qt to emit layers to the viewer. So a batch worker can
  build a fresh `ResultLayers` + `StageCheckpoints` per image, pass `on_stage_output=` to
  `run_pipeline_stages`, and write the snapshot without drawing anything.
- **Per-run outputs already follow a folder.** Checkpoint pickles, the run manifest and
  `{stem}_run_settings.yaml` go beside `vtk_output_prefix` (`pipeline/provenance.py`,
  `stage_checkpoints.output_dir_from_prefix`). They are named by `run_stem` (the segmented
  image's stem). So moving `vtk_output_prefix` per image moves almost everything.
- **Output-folder settings** to redirect per image: `vtk_output_prefix`, `base_plot_dir`,
  `perturbation_output_dir`, `sweep_output_dir`, `ilastik_output_dir`.

## The window

> Superseded by `PLAN_batch_panel.md` Phase 4 (2026-10-23): three boxes (Images, Conditions and
> statistics, the table), the report folded under the table, Run batch and Stop after this
> image. The rules below (blank folders, greyed mask rows, the table as the preview) still hold.
> Since 2026-10-24 the window also has a row for each other file the settings read, and greys the
> segmented folder while ilastik segments the raw images.

```
Batch run
  Segmented images   [ folder ............ ] [Browse]   (required)
  Raw images         [ folder ............ ] [Browse]   blank = look in the segmented folder
  Large vessel masks [ folder ............ ] [Browse]   blank = look in the segmented folder
  Small vessel masks [ folder ............ ] [Browse]   blank = look in the segmented folder
  Output folder      [ folder ............ ] [Browse]   default: vtk_output_prefix's folder
  [x] Skip images whose folder already has a .haemorun

  ┌───┬──────────────────────┬────────────┬──────────────┬──────────────┬─────────┐
  │ ✓ │ Segmented            │ Raw        │ Large masks  │ Small masks  │ Status  │
  ├───┼──────────────────────┼────────────┼──────────────┼──────────────┼─────────┤
  │ ✓ │ E14_MCA_Simple Se... │ E14_MCA... │ art + ven    │ global       │ ready   │
  │ ✓ │ E15_MCA_Simple Se... │ —          │ 2 files tie  │ —            │ FAILED  │
  └───┴──────────────────────┴────────────┴──────────────┴──────────────┴─────────┘
  [Match again]  [Check]                          [Leave batch run]  [Use this batch]
```

- **"Segmented doubles as global".** A blank raw, large or small folder means "look in the
  segmented folder". So a user with everything in one folder fills one field. A filled field
  replaces the segmented folder for that role; it does not add to it.
- The large and small mask fields are greyed while their role is not in use
  (`graph.vessel_masks_in_use`), or while ilastik makes those masks. The table says
  "not used" in those columns.
- **The table is the preview.** Every cell is the matched file's name; hovering shows the full
  path. A tie names the tied files. An unticked row is not run. **Check** runs the per-image
  preflight (below) and puts the first error or warning in Status. The full text goes in the
  report box.
- **Use this batch** closes the window and enters batch mode. The button on the Input tab then
  reads "Batch run: 12 images" and reopens the window. **Leave batch run** exits batch mode.

## Matching rules

All in one pure module, `io/batch_matching.py`, built on `vessel_mask_discovery`'s word reading.

1. **Inputs.** Every `.tif/.tiff/.h5` in the segmented folder, minus:
   - files that read as a mask of another file there (`read_mask_name`);
   - probability maps (`_NOT_A_MASK_WORDS`);
   - files that read as the **raw image of another file there**: some other file's name adds
     only unidentifying words to it (`segmented`, `Simple Segmentation`, `mask` ...).
     Example: `E14_MCA.tif` beside `E14_MCA_Simple Segmentation.tiff`.

   Sorted by name.
2. **Raw image.** The raw image matches by the mask rule without role words. A candidate in
   the raw folder is this image's raw image when its identifying words (after dropping
   `raw`/`unsegmented` and the unidentifying words) are a subset of the segmented image's.
   Ranking is the same: common prefix, then words shared, then same format. A tie is left
   unchosen and named. This covers ilastik's `{raw stem}_Simple Segmentation`, and a raw file
   named `E14_MCA_raw.tif`.
3. **Masks.** Masks are `find_vessel_masks([segmented, raw], sizes, folders=[the role's
   folder])`, so the role decision is unchanged: pair, global, a type with no size read as
   large, ties. `FoundVesselMasks.settings()` gives the four path values, with the venule row
   emptied for a global mask, exactly as the panel's own search does.
4. **Collisions.** Two inputs with one stem (e.g. `a.tif` and `a.h5`) would share an output
   folder. Both rows fail with that reason. The same mask file chosen for two images is
   allowed (one global mask for a set of crops), and the table notes it.

`find_vessel_masks` gains `folders: Iterable[Path] | None = None`. `None` keeps today's
behaviour (the references' folders), so `gui/vessel_mask_search.py` is untouched.

## One image's settings

`pipeline/batch.py` (pure) holds this:

- `BatchDirectories(segmented, raw, large, small, output)` and
  `BatchItem(segmented, raw, masks: dict[str, Path|None], problems: tuple[str, ...], selected)`.
- `item_settings(base, item, output_root) -> dict` returns the panel's settings with:
  - `input_path`, `raw_image_path` (None when no raw was matched), and the four mask paths
    replaced;
  - `vtk_output_prefix` = `root/{stem}/{prefix name}`, `base_plot_dir` = `root/{stem}/plots`,
    `perturbation_output_dir` = `root/{stem}/perturbations`, `sweep_output_dir` =
    `root/{stem}/sweeps`, `ilastik_output_dir` = `root/{stem}/segmentations`;
  - `mid_run_postprocessing` off (a batch cannot pause for hand edits; preflight says so as a
    warning).
- **Batch refusals** (`check_batch(base, items)`, added to each item's `preflight` result).
  These are settings that name things in one image and would silently mean something else in
  another:
  - **Errors:**
    - any boundary method `node_ids` (IDs name one build of one graph);
    - `network_source=synthetic`;
    - `use_ilastik_segmentation` (v1 runs over segmented images; see Later);
    - any single-file per-image input that is set: `pericyte_mask_path`, `cell_mask_path`,
      `measurement_3d_vessel_mask_path`, `tissue_raw_tiff_path`. Its shape would only match
      one image, and the run would fail at its end.
  - **Warnings:**
    - boundary `coordinates`/`volumes` methods and `crop_method=coordinates`: physical
      positions, meaningful across images only if they share a frame;
    - a row with no raw image while `measure_diameters_from_raw_image` is on (that run measures
      from the mask);
    - `do_skeletonize`/`do_graph_building` off. The saved skeleton is looked for per image in
      its own folder, so an image with none fails.

  Each selected row's preflight (`pipeline.preflight(item_settings(...))` + `check_batch`) is
  what **Check** shows. **Run pipeline refuses to start while any ticked row fails**, matching
  today's "Checks failed; nothing was run". The user unticks the row or fixes the files.
  Silently skipping one image of an overnight batch is the worse surprise.

## Running it

`gui/batch_run.py`, pure (no Qt), so it can be tested headless and a CLI can call it later:

```python
run_batch(base_settings, schema, items, output_root, *,
          progress=None,        # ProgressEvent callback, as run_pipeline_stages
          on_item=None,         # (index, total, item, status) for the panel's batch line
          check_cancelled=None, # raises RunCancelled, as watched() does today
          skip_done=True) -> BatchSummary
```

For each selected item, in order:

1. If `skip_done` is set and `root/{stem}/{stem}.haemorun` exists, record "skipped (done)" and
   move on. This is how an interrupted overnight batch is resumed.
2. Build a fresh `ResultLayers(settings=s)` and `StageCheckpoints()`, then run
   `run_pipeline_stages(s, schema, progress=..., on_stage_output=record)`, where `record` does
   what `_run_in_background.produced` does minus the viewer emit.
3. `capture_run(...)` → `write_run_snapshot(root/stem/f"{stem}.haemorun")`, with
   `show_results=True` so that **Load run** opens it as if it had been run in the panel.
4. Drop the run's results, checkpoints and graph. Call `io.clear_vessel_mask_cache()` and
   release the raw-volume cache, so image *n*'s volumes are gone before image *n+1* loads.
5. **A failure is caught**, logged with its traceback, recorded, and the batch moves on. The
   manifest's `status: "failed"` is already written by `record_run`. **A cancel** stops the
   current image at its next stage boundary (the existing `RunCancelled` path) and runs no
   more images.

At the end, and after every image (so a crash leaves it current), the batch writes to
`root/batch_summary.csv`: one row per image with its stem, the matched files, status
(`done`/`failed`/`skipped`/`cancelled`), the error, wall time, peak RSS (from the manifest),
and node and vessel counts. It also writes `root/batch_settings.yaml`, the base settings with
the five folders, so the batch can be re-run.

## Panel changes (`gui/_widget.py`)

- **The button.** "Batch run" sits in the Input tab's input box, under the segmented-image
  row. It is greyed under `network_source=synthetic` and while a run is going.
- **Batch mode state.** A small `BatchState` (pure, in `gui/batch_run.py`) holds the folders,
  the items and the ticks. It is panel state, **not schema settings**. A config file describes
  one run, and each image's `{stem}_run_settings.yaml` records the paths that image actually
  used.
- **"Batch run" rows.** Each of the six path rows has its `FileEdit` hidden and a read-only
  label "Batch run" shown in its place. Hover text names the batch's folder for that role. The
  row's value is not touched, so:
  - the mask search (`search_for_vessel_masks`) does not fire;
  - single-run preflight still reads sane values;
  - leaving batch mode needs no restore.

  Prerequisite greying (`apply_prerequisites`) still applies to the label.
- **Run pipeline in batch mode.**
  - It runs the batch checks, refuses with the usual message if any fail, clears the viewer,
    and starts `run_batch` on a `thread_worker`.
  - The progress bars show the current image's stages. The report line reads
    "Image 3 of 12: E14_MCA — Running Diameters (5/11)...". The log window shows every image's
    log in turn, each with a banner line.
  - Cancel works as today.
  - "Show each stage in the viewer" is not drawn during a batch: twelve images' layers would
    pile up and slow the run. It is honoured as "write a `.haemorun`" (always on in a batch).
- **Disabled in batch mode:** "Run from this stage", Check segmented image, the optimisers, and
  Save run. They act on one image. *As built (`d4bdfd0c`): all but Save run, which saves the run
  shown and stays on.*
- **The end.** The window's table shows each row's final status, plus an **Open** button per
  finished row that calls the existing `load_run_file` on its `.haemorun`. The report box shows
  `batch_summary.csv`'s path and the counts.
- **Hover text** for every new control goes in `gui/chrome_tooltips.py`.
  `tests/test_gui_tooltips.py` requires it.
- **Help page.** The Input tab's help entry (`gui/stage_help.py`) is Harvey's to write. Nothing
  is added there.

## Tests

| File | What it pins |
|---|---|
| `tests/test_batch_matching.py` | Inputs exclude masks, probability maps and raw files in one shared folder. Raw matched under ilastik naming and `_raw` naming. Raw tie named, not chosen. Masks from separate folders and from the segmented folder (blank fields). Pair vs global per image. Type-only names read as large. Stem collision fails both rows. One global mask shared by several images. |
| `tests/test_vessel_mask_discovery.py` (extend) | `folders=` searches only those folders; `None` is unchanged. |
| `tests/test_batch_settings.py` | `item_settings` replaces exactly the input and output settings and nothing else. Every batch refusal and warning above, one case each. `mid_run_postprocessing` forced off. |
| `tests/test_batch_run.py` (`integration`, `slow`) | Two small fixtures from `tests/data/` in one folder with masks: two folders each holding a `.haemorun` that `read_run_snapshot` reads back, with checkpoints for every stage and the image's own `input_path`. Outputs (VTK, statistics CSV, manifest) are in the right folder and nowhere else. An image made to fail is recorded as failed and the next still runs. `skip_done` skips a finished one. A cancel mid-image leaves later images unrun and the summary says so. `batch_summary.csv` columns and statuses. |
| `tests/test_gui_batch_run_widget.py` (`gui`) | The button opens the window. The table fills from the folders. Use this batch shows "Batch run" on the six rows with their values untouched. Run pipeline starts `run_batch` (patched) with the ticked items. Leave batch run restores the file editors. Run-from buttons are disabled in batch mode. |

## Phases

1. **Matching.** `folders=` on `find_vessel_masks`, `io/batch_matching.py`, and its tests.
2. **Settings and checks.** `pipeline/batch.py` (`item_settings`, `check_batch`) and its tests.
3. **Runner.** `gui/batch_run.py` (`run_batch`, `BatchSummary`, `BatchState`) and the
   integration test.
4. **Panel.** The button, window, batch-mode rows, Run pipeline dispatch, end-of-batch table,
   tooltips, and GUI tests.
5. **Docs.** `CLAUDE.md`'s layout, key modules and test table. A `RELEASE_REVIEW.md` row only
   if a check's evidence changes. The help page is left to Harvey.

## Later (not in v1)

- **ilastik batches.** With `use_ilastik_segmentation` on, the raw folder drives the list and
  the segmented folder is greyed; the masks are matched against the raw names. *Built
  2026-10-24.*
- **More per-image inputs** in the window: pericyte, cell and tissue masks, each its own
  folder, matched by the same rule. *Built 2026-10-24, with the 3D distances' vessel mask and a
  capillary block's mask of blocks.*
- **A CLI.** `examples/resistance_network_pipeline.py --batch-segmented DIR [--batch-raw DIR
  ...]` over the same `run_batch`, for running a batch on a machine with no display. *Built as
  `python -m haemolynx.gui.batch_run` (`gui/batch_run_cli.py`).*
- **Parallel images.** One image at a time in v1: a run already uses worker pools and most of
  the RAM on a large stack. *Not built: Harvey kept one image at a time (2026-10-24).*

### Later, as built (2026-10-24, `3d9d071a`)

**ilastik batches.** With `use_ilastik_segmentation` on (`pipeline.batch.batch_from_raw`):
- **The images.** The raw folder lists them (`BatchFolders(from_raw=True)`,
  `batch_inputs(raw=True)`): every image file there that is no segmentation, probability map or
  mask of another. The window greys the segmented folder and says why.
- **The name.** Each image is named after the segmentation ilastik will make,
  `{raw stem}_segmented`, so its folder, `.haemorun`, outputs and statistics carry the one stem,
  as every batch image's do. `io.ilastik.ilastik_segmented_stem` and `ilastik_segmented_path` are
  now the one naming rule, shared with `stages.segment` and `segmented_input_path`.
- **Where the segmentation goes.** `{folder}/segmentations/`, whether or not the settings name
  an `ilastik_output_dir`.
- **Reuse.** As in a single run: `ilastik_reuse_existing_output` and the provenance record
  beside each output.
- **Masks.** A blank mask folder is the raw folder, and masks are read against the raw names.
- **The checks.** An image matched one way while the settings say the other is refused: "match
  the folders again".
- **The panel.** While a batch is set, a row it fills reads as set for greying
  (`BatchSetup.gates`), so ilastik's switch and the other raw-image switches are live with the
  panel's own raw image row empty. Before this they were greyed in any batch whose panel had no
  raw image of its own.
- **ilastik's vessel masks** stay refused in a batch, now only while their masks are in use: see
  below.

**The other files read beside each image** (`io.batch_matching.FILE_ROLES`):
- **Which.** Pericyte mask, cell mask, the 3D distances' vessel mask, tissue image and mask of
  blocks, each from a folder of its own (window rows, `--pericyte`/`--cell`/`--vessel-3d`/
  `--tissue`/`--capillary-block`).
- **Matched to an image by name** (`file_owners`, sharing `mask_owners`' ranking): the image
  whose name the file shares the most words with from the start, then the most words.
  - New tie-break: the image with the fewest of its own words the file lacks, so `E14_cells`
    is `E14`'s, not `E14_2`'s. Mask folders gain it too; before, those were ties.
  - A file already matched to the image (its raw image, a mask) is never one of its other files.
    So a tissue folder that is the raw folder gives the other channel, and choosing a tied raw
    image settles the tissue image.
- **Matched only while read.** A role is matched while the run reads it, or a perturbation of a
  type that reads it does (`pipeline.batch.batch_file_roles`). The window shows only those rows,
  and the table gains an Other files column while any has a folder.
- **With a folder,** each image's setting, and every perturbation override naming it, read the
  image's own file (`item_settings`).
- **Without one,** a single file that the run or a perturbation would read for every image is
  refused. Before, any `capillary_block_mask` entry was refused, and a perturbation's own
  pericyte mask was not looked at.
- **No file for an image.** A problem for the pericyte, cell and blocks masks. For the
  distances' vessel mask and the tissue image it is a note: they fall back to the segmentation
  and the raw image.
- **Records.** `batch_summary.csv` gains a column per file; `batch_settings.yaml` records the
  folders and `# batch_from_raw: yes`.

**Tests.**
- `test_batch_matching.py` (9 new), `test_batch_settings.py` (6 new, 1 changed),
  `test_batch_run.py` (6 new; 2 read the Condition column by name now),
  `test_gui_batch_run_widget.py` (2 new, 1 changed).
- `test_batch_run.py` includes a slow test of a real ilastik batch: a stand-in for ilastik labels
  the seven-vessel fixture, and each image is run to the end.
- Nothing was run on a real stack or by a tester.

### Decided by Harvey (2026-10-24)

- **Decisions 1-5** below: kept as built.
- **Several images at once:** no. Images run one at a time. A run already spreads its heavy
  steps over worker pools and, on a large stack, uses most of the RAM.
- **ilastik's vessel masks per image:** they read the image's own raw image file, with a
  chosen channel or a combination of channels of it. Today
  `use_ilastik_large_vessel_segmentation` and `use_ilastik_small_vessel_segmentation` read one
  raw file each (`ilastik_unsegmented_*_image_path`, with no channel setting), so a batch still
  refuses them while their masks are in use. The details:
  - **A combination** is the chosen channels kept separate: ilastik gets a multichannel image
    of them, as the classifier was trained on. No arithmetic on the voxels.
  - **Each classifier its own channels:** large and small, arteriole and venule each name theirs
    (e.g. arteriole from channel 2, venule from channel 3 of the same file).
  - **Single runs too:** vessel-mask ilastik always reads the run's raw image
    (`raw_image_path`) by its channels. The separate `ilastik_unsegmented_*_image_path` files
    become old settings that still load (`Setting.migrates`/`replaces`, as the other raw-image
    paths did).

### To build: ilastik's vessel masks from the raw image's channels (not started)

1. **Settings.** A channel list per classifier (`ilastik_arteriole_channels`, ... four of them),
   each `requires` its classifier's switch and `raw_image_path`. The old
   `ilastik_unsegmented_*_image_path` settings migrate: a file that is the run's raw image
   keeps working, and any other file is reported, since one raw image per run is the rule.
2. **ilastik's input.** ilastik headless reads a whole file, so the chosen channels are written
   to one file first (channels kept separate, the raw image's voxel size and axis order kept).
   It goes in `ilastik_output_dir` beside the masks, named by role and class.
3. **Reuse.** The provenance record names the raw image and its channels, so a change of
   channels segments again.
4. **Outputs and checks.** `ilastik_vessel_mask_output_path` and `output_paths` keep their
   names. Preflight refuses a channel the raw image lacks, and an empty list.
5. **Batch.** The refusal of ilastik's vessel masks goes. Each image's classifiers read its own
   raw image, and the masks go in its `segmentations/` folder.
6. **Panel.** The channel rows sit under each classifier, with hover text. Help text is
   Harvey's.
7. **Tests.** The migration, the channel file (separate channels, metadata kept), provenance
   with channels, preflight, a single run and a batch with the fake ilastik.

### Waits on Harvey

- **The Input tab's help text** for Batch run (`gui/stage_help.py`): his to write.

## Decisions to confirm with Harvey

*Built as proposed (2026-10-19); all five confirmed by Harvey, kept as built (2026-10-24).*

1. **Filled folder replaces, not adds.** A filled raw/large/small folder replaces the
   segmented folder for that role. It does not search both.
2. **Any failing row blocks the whole batch at Run.** It is not skipped.
3. **Output folder default.** The output folder defaults to `vtk_output_prefix`'s folder. Each
   image's folder is `{output}/{segmented stem}`, and the `.haemorun` is `{stem}.haemorun`
   inside it.
4. **Batch folders are not saved.** They are not saved in the config or in a `.haemorun` (only
   in `batch_settings.yaml`).
5. **No viewer during a batch.** The viewer shows nothing during a batch; results are opened
   per image afterwards from the table.
