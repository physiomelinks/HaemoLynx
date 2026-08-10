# HaemoLynx

Converts raw microscopy images of the microvasculature into computational haemodynamics models for hypothesis testing, experimental design, and more.

## Run it in napari

The panel is the quickest way to use HaemoLynx: it builds a settings form from
the schema, checks the settings before anything runs, and runs the pipeline in a
background thread while each stage's results appear in the viewer.

```bash
pip install "HaemoLynx[napari]"    # needs Python 3.11+ (napari's floor, not ours)
napari                             # Plugins -> HaemoLynx -> Pipeline settings
```

That extra brings a Qt binding (PyQt6) with it, so the panel opens on a fresh
environment. If you already run napari with a binding of your own, install
`HaemoLynx[napari-plugin]` instead and keep it.

Then: **drag an image in, open the panel, press Run pipeline.** The layer you
have selected is picked up automatically, so on a segmented TIFF there is
nothing else to set.

### The panel: configuring a run

One tab per pipeline stage, in the order they execute -- Input, Skeletonise,
Graph, Boundaries, Diameters, Resistances, Solve, Export -- so a run is
configured the way it runs rather than the way the config file is laid out.

Every row comes from `haemolynx.pipeline.default_schema()`, so a setting
declared there appears with its help text, range and choices, and greys out with
a reason when the setting it depends on is off. There is no second list of
settings to keep in step; a test fails if a setting reaches no tab or more than
one.

**Load config** and **Save config** read and write the same YAML the examples
use, so a run set up here can be repeated from the command line and back.

### Running on the image already open in napari

Open an image, then open the panel: the selected layer is picked up
automatically, and the row at the top lets you choose a different one.

- A layer **read from a TIFF or HDF5** points the run at that file, so the same
  bytes and metadata are used -- no copy is made.
- A layer **built in the viewer** (a threshold, a crop) has no file behind it,
  so its array is written next to the run's outputs and read back. The panel
  says which of the two happened.
- If the layer has a **scale**, it becomes the voxel size. napari scales per
  array axis `(z, y, x)` and the setting is image metadata order `(x, y, z)`,
  so the two are reversed on the way in.

### Watching a run

Each stage puts its work in the viewer as it finishes -- the volume and
skeleton, then the vessel network, the boundary nodes, the pericytes -- so a run
is something to watch rather than a wait for files. Two progress bars show which
stage is running and how far through it is.

Vessels are drawn as a Vectors layer, which is roughly ten times faster to draw
than paths at the size of a real run; the per-vessel numbers ride on a hidden
Points layer at each vessel's midpoint, so hovering still identifies one.

**Show each topology step** additionally redraws the network after each of graph
building's eleven repair steps, which is worth switching on when skeletonisation
is behaving oddly and not otherwise.

A second run updates its own layers in place, so anything you hid stays hidden,
and a layer of your own that happens to share a name is never touched.
**Clear layers** removes everything the plugin added and nothing else.

### Choosing what the colours mean

That is done in napari's **layer controls, on the left**, with the layer
selected -- the panel on the right is for the pipeline only.

- **edge feature:** (vessels) and **node feature:** (nodes and boundary nodes)
  choose the quantity: flow, pressure, branch order, resistance, length,
  diameter, boundary role, and everything else the run produced. A quantity is
  listed from the start and fills in when the stage that computes it runs.
- **colour range:** shows the scale as a colour bar and lets you set its ends.
  **Fit all** spans the smallest and largest value; **Fit 1-99%** ignores the
  extreme 1% at each end, which is the useful one for flow -- a handful of
  vessels carry orders of magnitude more than the rest, and against the full
  range everything else is one colour.
- **Show colour bar in the viewer** draws that scale in the canvas, beside the
  data it describes.

Picking a new quantity fits the range to it automatically. Vessels start
coloured by flow once a run has solved, nodes by pressure.

### Other things worth knowing

The menu also carries **Run a saved config**, which runs a `.yaml` as it stands
without opening the form.

Running the panel does not open plots outside napari: the settings that make
plotly open a web browser mid-run start switched off, and are ordinary rows you
can tick back on.

The panel's own tests build real Qt widgets, so they need napari and a display.
They are marked `gui`, skipped without one, and CI runs them on 3.11 under
xvfb:

```bash
pytest -m gui           # with "HaemoLynx[napari]" installed
```

The library itself never imports napari; the extra is optional, and the panel is
only loaded when you open it.

## Install

Python 3.9 or newer.

```bash
pip install HaemoLynx
```

> Not on PyPI yet — until the first release, install from a checkout as below.
> Everything else on this page already works that way.

```bash
git clone https://github.com/physiomelinks/HaemoLynx.git
cd HaemoLynx
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To work on HaemoLynx itself, take the development extras as well:

```bash
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies. Add the
`notebook` extra (`pip install "HaemoLynx[notebook]"`, or
`pip install -e ".[dev,notebook]"` from a checkout) to get the Jupyter kernel
for the tutorial notebook.

## Start here: the tutorial notebook

**[tutorials/pipeline_tutorial.ipynb](tutorials/pipeline_tutorial.ipynb)** runs
every stage of the pipeline one cell at a time, with a plot after each, and
explains what each stage is for. It is the fastest way to see what HaemoLynx
does.

Open it and run all cells — that is the whole setup:

```bash
pip install "HaemoLynx[notebook]"
jupyter notebook tutorials/pipeline_tutorial.ipynb
```

The first cell installs HaemoLynx if the kernel does not already have it, and
the notebook builds its own small vessel volume when no segmented image is to
hand, so it needs no data download and no clone. From a checkout it picks up
the cropped nerve mask in `tests/data/` and the pipeline's own
`examples/resistance_pipeline_config.yaml` instead.

## Running the examples

Every example is driven by a YAML config file. Run one with no arguments and it
uses that file as it stands:

```bash
python examples/simple_network_haemodynamics.py     # a hand-built 8-vessel network
python examples/resistance_network_pipeline.py      # the full image-to-model pipeline
python examples/brain_network_pipeline.py           # the pipeline, then a pericyte dilation sweep
```

| Example | Config file | What it does |
|---|---|---|
| `simple_network_haemodynamics.py` | `examples/simple_network_config.yaml` | Builds a small network in code, solves flow through it, writes VTK. No image needed — the quickest way to see the haemodynamics API. |
| `resistance_network_pipeline.py` | `examples/resistance_pipeline_config.yaml` | Segmentation → skeletonisation → graph → boundaries → diameters → solve → export, from a real image. |
| `brain_network_pipeline.py` | `examples/brain_pipeline_config.yaml` | The same pipeline, then sweeps pericyte dilation against inlet pressure and plots the curves. |

### The pipeline, one stage at a time

`resistance_network_pipeline.py` is deliberately short: the stages live in
`haemolynx.pipeline` and the example just calls them in order, so you can read a
run's shape without opening the library.

```python
inputs     = segment(settings)                                       # ilastik, or pass a mask through
volume     = skeletonise(settings, inputs)                           # load, resolve voxel size, skeletonise
network    = build_network(settings, volume, SCHEMA)                 # skeleton + vessel masks -> graph
boundaries = assign_boundaries(settings, network)                    # inlets, outlets, vessel boundaries
diameters  = assign_diameters(settings, network, boundaries, SCHEMA) # branch orders, diameter per edge
model      = build_haemodynamic_model(settings, diameters)           # resistance and conductance per edge
solution   = solve(settings, model, boundaries)                      # pressures, flows, equivalent resistance
export_results(settings, network, model, solution)                   # VTK, statistics, plots
```

Call them individually if you want to intervene mid-run — that is exactly what
`brain_network_pipeline.py` does before running its sweep.

## Changing settings

**Edit the config file.** It is the source of truth for a run, and it documents
itself: every setting arrives with its meaning, unit, allowed values and any
prerequisite as a comment.

```yaml
# ------------------------------------------------------------------------
# Boundary assignment
# ------------------------------------------------------------------------
boundary_assignment:
  # Choose the method for selecting input boundary nodes  [one of: coordinates, all_degree_1, volume, edge_percent, degree_1_from_starting]
  starting_node_selection_method: edge_percent

# ------------------------------------------------------------------------
# Solver and output
# ------------------------------------------------------------------------
solver_and_output:
  # Apply this pressure boundary condition at the inlet nodes  [Pa; range 0.0..]
  input_p_bc: 4500.0
```

**Or override one value for a single run.** Every setting has a command-line
flag of the same name, generated from the schema:

```bash
python examples/resistance_network_pipeline.py --input-path /data/my_mask.tiff
python examples/simple_network_haemodynamics.py --inlet-pressure-pa 8000
```

**Or point at a different config file entirely:**

```bash
python examples/resistance_network_pipeline.py --config my_experiment.yaml
```

### Useful flags

Available on every example:

```bash
--list-settings              # print every setting and its value for this run, then exit
--save-config my_run.yaml    # write the settings this run would use, then exit
--config FILE                # run from a different config file
--<setting-name> VALUE       # override one setting
```

The pipeline examples add:

```bash
--list-presets               # named override sets: quick_debug, publication, statistics_only, ...
--preset quick_debug         # apply one on top of the config file
--check-only                 # run the preflight checks and exit without running anything
```

`--check-only` is worth using before a long run: it validates paths, toggles and
their dependencies, and exits non-zero if anything would fail partway through.

```
=== Preflight Checklist ===
[OK] Main input image: tests/data/seven_vessel_noisy_3d.tif
[OK] Input axis order: zyx (canonical; no transpose)
[OK] Statistics mode: fast
Preflight passed.
```

### Bad settings are caught before anything runs

Values are checked against the schema when the config loads, so a typo or an
out-of-range number fails immediately rather than halfway through a long run:

```
2 configuration problems:
  - Unknown setting 'inptu_path'. Did you mean: input_path?
  - Setting 'small_vessel_mask_min_overlap_fraction' is 5.0, above its maximum 1.0.
```

### Adding a setting

Settings are declared once, as a schema, and that declaration generates the
config file, the command-line flags and the validation. The pipeline's own
settings live in the package, in `haemolynx.pipeline.schema`; an example that
adds settings of its own declares those beside it (`examples/*_schema.py`) on
top of the pipeline's. After editing a schema, regenerate the config files:

```bash
python examples/regenerate_configs.py
```

This keeps the values already in your config files and adds any new settings
with their documentation.

## Without a repository checkout

The pipeline is configured entirely from its schema, and the schema ships with
the package — so `pip install imagelynx` is enough to write yourself a config
file and run from it, with no copy of this repository involved:

```python
from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline import write_default_config

write_default_config("my_config.yaml")   # every setting, commented, at its default
# edit my_config.yaml, then:
settings = resolve_settings(schema=default_schema(), config_path="my_config.yaml")
graph = run_pipeline_stages(settings, default_schema())
```

`default_schema().describe()` is plain JSON — the same declaration a GUI can
render a settings form from.

Path defaults in the generated config are relative, so they resolve against the
directory you run in (`images/`, `outputs/`, `plots/`). The examples in this
repository pin their own paths under `examples/` in their config files.

## Tutorial

[The tutorial notebook](tutorials/pipeline_tutorial.ipynb) walks through the
same pipeline stage by stage with plots at each step (see
[Start here](#start-here-the-tutorial-notebook) for how to open it). Edit the
notebook, not the generated `pipeline_tutorial.py`; regenerate that with:

```bash
pytest tests/integration/test_pipeline_tutorial.py
```

## Allowable input mask formats

`tif`, `h5`

## Testing

From the repository root:

```bash
pytest -s               # everything
pytest -m "not slow"    # skip the slow integration tests
```

### Does this branch change the numbers?

`scripts/compare_branches.py` runs the resistance network pipeline twice on the
same dataset with the same settings — once on your checkout, once on a
reference ref in a temporary `git worktree` — and reports every way the two
runs differ:

```bash
python scripts/compare_branches.py                       # against main
python scripts/compare_branches.py --ref devel           # against another ref
python scripts/compare_branches.py --image /data/x.tif   # another dataset
python scripts/compare_branches.py --setting min_stub_length=5.0
```

It writes `COMPARISON.md` and an `index.html` — every plot from both runs, side
by side — into `comparison_outputs/` (gitignored). The report covers graph
metrics, **the first graph-building stage whose output diverges** (which is
what localises a regression), edge attributes, the statistics CSVs, the VTK
exports, and runtime.

This is **not** part of the test suite and never runs in CI: it takes roughly
15 minutes per side and defaults to `examples/images/Nerve_capillaries.tif`, a
328 MB image that is not in the repository. To check the tool itself works —
about a minute, on the committed test fixture:

```bash
python scripts/compare_branches.py --self-check --smoke
```

That compares `HEAD` against `HEAD` and exits non-zero if anything differs.
The reporting logic has fast unit tests of its own in
`tests/test_branch_comparison.py`, which do run in CI.

A few things to know before trusting a report:

* If either side fails, the tool says which one and why, and prints no tables —
  a partial comparison is never presented as a complete one.
* Branches take their settings differently (older entry points have a hundred
  or so keyword arguments and read some settings from module constants). Each
  side is inspected and adapted; a setting that defines the comparison and
  cannot be applied stops that side rather than silently running a different
  configuration, and a setting a branch simply does not have is listed in the
  report as a caveat.
* The boundary boxes are in physical (z, y, x) **micrometres**, not voxel
  indices.

## Licence

HaemoLynx is released under the [Apache License 2.0](LICENSE).
Copyright 2026 Finbar Argus and Harvey Davis.
