# Inputs: the raw image

Moved verbatim out of CLAUDE.md on 2026-10-22 (commit before the move: 1a9c2ebc). These are design notes and dated measurements: what was measured, on what, and why a rule or default is what it is. They are history, not the contract, and are not kept up to date line by line; CLAUDE.md and the code are the current state.

## One raw image per run (from Segmentation)

**One raw image per run** (2026-10-13). The Input tab's `raw_image_path`, `raw_image_channel` and
`raw_image_stain` are the only raw image a run reads: ilastik segments it, the Diameters stage
measures from it (`measure_diameters_from_raw_image`), Check segmented image and the optimisers
compare against it, and the tissue volume falls back to it. `raw_image_stain` is `auto`,
`plasma_column` (a lumen label such as dextran) or `endothelial` (the wall round the lumen, such
as claudin-5); auto reads it from the image (`preprocessing/raw_stain.py`'s `read_raw_stain`:
brightness at the mask's edge against the middle of its widest vessels, the one rule: the
raw-evidence calibration in `build_network` decides through it too, and records the answer on
the graph (`G.graph["raw_image_stain"]`, with the reading), which the Diameters stage takes
rather than reading the image again -- it reads the image only when the build could not tell,
cut to the run's crop box, and hands the volume on to the measurement so the file is loaded
once), and the stain picks the diameter method: FWHM for a plasma column, the endothelial ring
fit for a wall. A raw image that is set is read by `build_network` whatever else is on, so
preflight treats a missing one, or a composite without `raw_image_channel`, as an error.
**The raw image is optional** (2026-10-13): with none entered, every setting that reads it is
out of effect whatever its switch says -- `use_ilastik_segmentation` (the segmented image is
read instead; an error only when there is none), `measure_diameters_from_raw_image` (the
diameter chain starts at the mask), the raw evidence -- through `stages.raw_image_is_set`,
`ilastik_segments` and `measures_diameters_from_raw_image`, and preflight warns rather than
fails. Those settings, and `raw_image_channel` / `raw_image_stain`, declare
`requires=("raw_image_path",)`: a `requires` entry may name a path setting (set / unset,
`parsers.schema.PREREQUISITE_GATE_KINDS`), and the panel **greys** a row whose own path gate is
unset rather than hiding it (`gui/form.py`'s `hidden_by`), while the rows nested under it hide
as under any switch not in effect. A `!gate` rule holds while the gate is out of effect, so the
segmented-image row stays when ilastik is ticked with no raw image (`form.unmet_chain`,
`Schema._prerequisite_in_effect`).
The settings it replaced (`ilastik_unsegmented_image_path`, `fwhm_raw_tiff_path`/`_channel`,
`endothelial_image_path`/`_channel`, `use_fwhm_edge_diameters`, `use_endothelial_diameters`,
`diameter_basis`) still load, moved together by `Setting.migrates`/`migrate` (`Schema.upgrade`).

## image_axis_order = auto (2026-10-24)

Added at Harvey's request; `zyx` stays the default. What the headers of the files to hand said
(tifffile 2026.8.23):

| File | Labels |
|---|---|
| `MasonTest.tif` (ImageJ, `slices` set) | `ZYX` |
| Desktop nerve / skeletal-muscle crops (ImageJ composites) | `ZCYX` |
| Skeletal-muscle `.ome.tif` (16.4 G voxels) | `ZYX` |
| E14.5 claudin Probabilities `.h5` (ilastik) | axistags `z c y x` |
| `e14p5_mca.tif`, the test fixtures (written by tifffile) | `QYX`: none |

So every file there reads as `zyx` under auto too, and switching the default would change none of
those runs. What auto can and cannot see:

- A TIFF page is a Y×X plane, so a TIFF's labels can only put z first, unless a tifffile writer
  named its axes (`metadata={"axes": "XYZ"}`). A resliced stack ImageJ still labels `ZYX`.
- tifffile writing an ImageJ stack given no `axes` calls its pages channels (`CYX`, `channels=N`):
  that names no z, so the file is read in the segmented image's order.
- HDF5 is where it matters: ilastik and vigra write `axistags` in any order. Before this, the
  loader read them only to drop size-1 channel and time axes and then applied
  `image_axis_order` whatever they said.

A file with no labels is read in the segmented image's order rather than `zyx`, so masks written
beside a labelled non-`zyx` HDF5 by the same tool line up with it. That is carried as
`"auto:<order>"` (`io.axis_order_for_run`) rather than threaded through every caller, since one
`axis_order` argument reaches several files in the diameter, pericyte, cell-mask and vessel-mask
loaders.
