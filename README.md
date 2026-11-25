# HaemoLynx

Converts raw microscopy images of the microvasculature into network-based representations for network level statistics and computational haemodynamics models for hypothesis testing, experimental design, and more.

## Install

```bash
pip install "HaemoLynx[napari]"    # needs Python 3.11+ (napari's floor, not ours)
```

The `napari` extra brings a Qt binding (PyQt6) with it, so the panel opens on a
fresh environment. If you already run napari with a binding of your own, install
`HaemoLynx[napari-plugin]` instead and keep it.

## Use it in napari

```bash
napari      # Plugins -> HaemoLynx -> Pipeline settings
```

Then: **drag an image in, open the panel, press Run pipeline.** The selected
layer is picked up automatically, so on a segmented TIFF there is nothing else
to set. Each stage's results appear in the viewer as it finishes.

To start from a **raw image** instead, segment it in the panel: set the Input
tab's *Raw image path*, tick *Classify and segment raw image in Napari*, and
press **Train...** on the classifier row. Paint a few strokes of vessel and
background on the raw image, watch the plane in view classify as you paint, and
**Save classifier**. The run then segments the raw image itself before it
skeletonises. A classifier trained in [ilastik](https://www.ilastik.org/) works
too (*Segment raw image with a pretrained Ilastik classifier*), and the training
window can import an ilastik project's painted labels.

To run the same settings over **a folder of images**, press **Batch run** above
the Input row. It matches each image's raw image and masks by file name, runs
them one after another, and saves each image's outputs and `.haemorun` in a
folder of its own, with `batch_summary.csv` and every image's statistics side by
side. Each image's inlets and outlets come from its own large vessel masks.
Given two conditions named in the file names, it tests one against the other.
The same from a terminal:

```bash
python -m haemolynx.gui.batch_run --config my_run.yaml --segmented images/
```

To draw a **synthetic network** matched to a previous run instead of analysing
an image, set the Input tab's *Input* to *Synthetic network from a template run*
and point it at that run's `.haemorun` file (saved with Statistics on), or at a
batch's `.haemobatch` to match a condition's runs averaged. The run starts at the
Boundaries stage and writes `{template}_synthetic_template_comparison.csv`
saying how close each characteristic came. *The template run's network inside a
synthetic one* draws the synthetic network round the run's own instead, joined
where its vessels leave the image; its outputs are named `{template}_surrounded`.

**[tutorials/README.md](tutorials/README.md)** covers the rest — configuring a
run, what each tab and the view panel show, editing the network by hand,
perturbations, batch runs, running from the command line, and changing
settings.

**[docs/export_dictionary.md](docs/export_dictionary.md)** says what every file a
run writes holds, column by column, with units; the statistics rows are in
**[docs/statistics_glossary.md](docs/statistics_glossary.md)**.

## Settings and presets

Every setting is declared once, in
[`src/haemolynx/pipeline/schema.py`](src/haemolynx/pipeline/schema.py), with its
default, limits and help text. The panel's form, the YAML config file and the
command line's `--<setting-name>` flags are all generated from those
declarations, so none of them can name a setting that does not exist.

A **preset** is a named set of those settings to start a run from. The panel
lists its presets in the **Preset** menu under *Load config...*: plain dicts in
[`src/haemolynx/pipeline/presets.py`](src/haemolynx/pipeline/presets.py), each
picking switches and choices only, never a file or a threshold. The example
scripts take the same ones with `--preset NAME`, plus their own from
[`examples/pipeline_presets.py`](examples/pipeline_presets.py), which may name
files too; `--list-presets` lists them all. Every preset is checked against the
schema, so none can set a setting that is gone.

A preset sets only what it names. On the command line each layer overrides the
one before it:

```
schema default  ->  config file (--config)  ->  --preset NAME  ->  --<setting-name> VALUE
```

In the panel a preset is loaded once, like a config: it fills the rows it
names, and a config loaded or a row changed afterwards replaces what it set.

**Your own presets.** *Save these settings as a preset...* at the foot of the
Preset menu keeps the panel's settings on this machine, less what belongs to one
image: its files and folders, channels, labels, voxel size, axis order, crop,
nodes and vessels. Unlike the presets above, a saved one may hold thresholds, and
it is a whole set: loading it puts back every setting it holds as it was saved,
and every other setting it could hold to its default. Saved presets live in a
`presets` folder in your own settings folder, shared by every checkout on the
machine (set `HAEMOLYNX_PRESET_DIR` to keep them elsewhere). A perturbation can
be kept the same way: **Save for later runs** on the Perturbations tab, then pick
it under **+ Add a perturbation** in a later run.

## Developing

Work against a local checkout, with napari running the code you are editing:

```bash
git clone https://github.com/physiomelinks/HaemoLynx.git
cd HaemoLynx
python -m venv .venv                  # 3.11+ for the napari panel
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e ".[dev,napari]"

napari                                # Plugins -> HaemoLynx -> Pipeline settings
```

The install is editable, so whatever branch is checked out is what napari loads
— no reinstall after switching branches.

**[tutorials/dev_README.md](tutorials/dev_README.md)** covers the rest — running
the tests, checking whether a branch changes the numbers, adding a setting, and
releasing.

## Licence

HaemoLynx is released under the [Apache License 2.0](LICENSE).
Copyright 2026 Finbar Argus and Harvey Davis.
