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

To draw a **synthetic network** matched to a previous run instead of analysing
an image, set the Input tab's *Input* to *Synthetic network from a template run*
and point it at that run's `.haemorun` file (saved with Statistics on). The run
starts at the Boundaries stage and writes `{template}_synthetic_template_comparison.csv`
saying how close each characteristic came.

**[tutorials/README.md](tutorials/README.md)** covers the rest — configuring a
run, what the panel shows, colouring results, running from the command line,
and changing settings.

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

## Developing

Work against a local checkout, with napari running the code you are editing:

```bash
git clone https://github.com/physiomelinks/HaemoLynx.git
cd HaemoLynx
python -m venv .venv                  # 3.11+ for the napari panel
source .venv/bin/activate
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
