# HaemoLynx

Converts raw microscopy images of the microvasculature into computational haemodynamics models for hypothesis testing, experimental design, and more.

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

**[tutorials/README.md](tutorials/README.md)** covers the rest — configuring a
run, what the panel shows, colouring results, running from the command line,
and changing settings.

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
