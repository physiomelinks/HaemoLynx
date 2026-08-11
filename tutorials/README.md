# Getting started with HaemoLynx

Installing and opening the panel is in the [top-level README](../README.md).
This is everything after that.

- [The notebook](#the-notebook) — the fastest way to see what the pipeline does
- [The napari panel](#the-napari-panel)
- [Running from the command line](#running-from-the-command-line)
- [Changing settings](#changing-settings)
- [Without a repository checkout](#without-a-repository-checkout)

Input masks may be `tif` or `h5`.

## The notebook

**[pipeline_tutorial.ipynb](pipeline_tutorial.ipynb)** runs every stage one cell
at a time, with a plot after each and an explanation of what the stage is for.

```bash
pip install "HaemoLynx[notebook]"
jupyter notebook tutorials/pipeline_tutorial.ipynb
```

It builds its own small vessel volume when no segmented image is to hand, so it
needs no data download and no clone. From a checkout it picks up the cropped
nerve mask in `tests/data/` instead.

Edit the notebook, never the generated `pipeline_tutorial.py`; regenerate that
with `pytest tests/integration/test_pipeline_tutorial.py`.

## The napari panel

One tab per pipeline stage, in the order they execute — Input, Skeletonise,
Graph, Boundaries, Diameters, Haemodynamics, Solve, Export. Every row comes
from the settings schema, so it arrives with its help text, unit, range and
choices, and greys out with a reason when the setting it depends on is off.

**Load config** and **Save config** read and write the same YAML the command
line uses, so a run set up here repeats there and back.

Loading a config only reads the file — a config naming an image that is not on
this machine still loads, because the image a run works on is the layer open in
napari. With a layer already open, loading a config keeps that layer as the
input and ignores the file's `input_path`. Paths are checked when a run starts,
by **Run checks** and by the run itself.

### Setting boundary conditions by pointing at them

On the **Boundaries** tab, **Show these boundary conditions** draws what the
config describes, before anything runs:

- **HaemoLynx BC coordinates** — a ring per coordinate, coloured by role.
- **HaemoLynx BC regions** — a rectangle per volume box, coloured the same way.

Both are editable, and **both are the settings**: a coordinate you drag is a
coordinate the run will use. Napari's own tools do the work — **Pick coordinates
in the viewer** puts the points layer into add mode, then click to place, drag to
move, select and press Delete to remove. Pick the **Role** first; it decides
which of the four settings a new point lands in, and **Assign selected to this
role** moves ones you got wrong.

**Draw a region** does the same for a volume box. Draw a rectangle on a slice and
it becomes a box, centred on that slice, as deep as the **Region depth** slider
says — which defaults to the whole stack, because a boundary band usually is.
Select a region and move the slider to resize it. Regions can only be *drawn* in
the 2D view: napari does not allow editing a Shapes layer in 3D. Switch back to
3D afterwards to see the box you made.

**Snap selected to nearest terminal** moves each coordinate onto the vessel end
the run would choose, and says how far it moved — a large move means the click
missed. It needs a graph, so run at least *3. Graph* first. Coordinates you pick
without it are still correct: a run snaps every one of them to its nearest
terminal anyway.

Two things the panel will tell you rather than fix behind your back. A role only
reads its coordinates when its `*_selection_method` says `coordinates` (and its
regions when it says `volume`), so picking against a role set to `edge_percent`
reports *"Not used"* rather than silently rewriting the method you chose. And an
entry it cannot read — a two-number coordinate from a hand-edited config — is
reported and skipped, not raised.

### Running on the image already open

Open an image, then open the panel: the selected layer is picked up
automatically, and the row at the top lets you choose a different one.

- A layer **read from a TIFF or HDF5** points the run at that file — no copy.
- A layer **built in the viewer** (a threshold, a crop) has no file behind it,
  so its array is written next to the run's outputs and read back.
- If the layer has a **scale**, it becomes the voxel size. If it has none and
  its file says what a voxel is, the layer is given that scale — napari's
  readers ignore a TIFF's resolution tags, and without it an anisotropic stack
  is drawn squashed along z and the vessels do not lie on the vessels.

napari scales per array axis `(z, y, x)`; the voxel-size setting is image
metadata order `(x, y, z)`. The two are reversed on the way in.

### Watching a run

Each stage puts its work in the viewer as it finishes — the volume and
skeleton, then the vessel network, the boundary nodes, the pericytes — with two
progress bars showing which stage is running and how far through it is.

Vessels are drawn as a Vectors layer, roughly ten times faster to draw than
paths at the size of a real run; the per-vessel numbers ride on a hidden Points
layer at each vessel's midpoint, so hovering still identifies one.

**Show each topology step** additionally redraws the network after each of graph
building's eleven repair steps — worth switching on when skeletonisation is
behaving oddly, and not otherwise.

A second run updates its own layers in place, so anything you hid stays hidden,
and a layer of your own that happens to share a name is never touched.
**Clear layers** removes everything the plugin added and nothing else.

### Choosing what the colours mean

In napari's **layer controls on the left**, with the layer selected — the panel
on the right is for the pipeline only.

- **edge feature:** (vessels) and **node feature:** (nodes) choose the quantity:
  flow, pressure, branch order, resistance, length, diameter, and everything
  else the run produced. A quantity is listed from the start and fills in when
  the stage that computes it runs.
- **colour range:** shows the scale and lets you set its ends. **Fit all** spans
  the smallest and largest value; **Fit 1-99%** ignores the extreme 1% at each
  end, which is the useful one for flow — a handful of vessels carry orders of
  magnitude more than the rest, and against the full range everything else is
  one colour.
- **Show colour bar in the viewer** draws that scale in the canvas.

Picking a new quantity fits the range to it automatically. Vessels start
coloured by flow once a run has solved, nodes by pressure.

Running the panel does not open plots outside napari: the settings that make
plotly open a browser start switched off, and are ordinary rows you can tick
back on.

## Running from the command line

Every example is driven by a YAML config file, and runs from it with no
arguments:

| Example | Config | What it does |
|---|---|---|
| `simple_network_haemodynamics.py` | `simple_network_config.yaml` | Builds an 8-vessel network in code, solves flow, writes VTK. No image needed — the quickest look at the haemodynamics API. |
| `resistance_network_pipeline.py` | `resistance_pipeline_config.yaml` | The full pipeline: segmentation → skeletonisation → graph → boundaries → diameters → solve → export. |
| `brain_network_pipeline.py` | `brain_pipeline_config.yaml` | The same pipeline, then a pericyte dilation sweep against inlet pressure. |

```bash
python examples/resistance_network_pipeline.py
```

`resistance_network_pipeline.py` is deliberately short — the stages live in
`haemolynx.pipeline` and the example calls them in order, so a run's shape is
readable without opening the library:

```python
inputs     = segment(settings)                                       # ilastik, or pass a mask through
volume     = skeletonise(settings, inputs)                           # load, resolve voxel size, skeletonise
network    = build_network(settings, volume, SCHEMA)                 # skeleton + masks -> graph
boundaries = assign_boundaries(settings, network)                    # inlets, outlets, vessel boundaries
diameters  = assign_diameters(settings, network, boundaries, SCHEMA) # branch orders, diameter per edge
model      = build_haemodynamic_model(settings, diameters)           # resistance and conductance per edge
solution   = solve(settings, model, boundaries)                      # pressures, flows, equivalent resistance
export_results(settings, network, model, solution)                   # VTK, statistics, plots
```

Call them individually to intervene mid-run — which is what
`brain_network_pipeline.py` does before its sweep.

## Changing settings

**Edit the config file.** It is the source of truth for a run and documents
itself: every setting arrives with its meaning, unit, allowed values and any
prerequisite as a comment.

```yaml
boundary_assignment:
  # Choose the method for selecting input boundary nodes  [one of: coordinates, all_degree_1, volume, edge_percent, degree_1_from_starting]
  starting_node_selection_method: edge_percent

solver_and_output:
  # Apply this pressure boundary condition at the inlet nodes  [Pa; range 0.0..]
  input_p_bc: 4500.0
```

**Or override one value for a single run** — every setting has a command-line
flag of the same name, generated from the schema:

```bash
python examples/resistance_network_pipeline.py --input-path /data/my_mask.tiff
python examples/resistance_network_pipeline.py --config my_experiment.yaml
```

Flags available on every example:

```
--list-settings              # print every setting and its value, then exit
--save-config my_run.yaml    # write the settings this run would use, then exit
--config FILE                # run from a different config file
--<setting-name> VALUE       # override one setting
```

and on the pipeline examples:

```
--list-presets               # named override sets: quick_debug, publication, ...
--preset quick_debug         # apply one on top of the config file
--check-only                 # run the preflight checks and exit
```

`--check-only` is worth using before a long run: it validates paths, toggles and
their dependencies, and exits non-zero if anything would fail partway through.

Bad settings are caught when the config loads, not halfway through a run:

```
2 configuration problems:
  - Unknown setting 'inptu_path'. Did you mean: input_path?
  - Setting 'small_vessel_mask_min_overlap_fraction' is 5.0, above its maximum 1.0.
```

## Without a repository checkout

The schema ships with the package, so `pip install HaemoLynx` is enough to write
a config and run from it with no copy of this repository:

```python
from haemolynx.pipeline import (
    default_schema, resolve_settings, run_pipeline_stages, write_default_config,
)

write_default_config("my_config.yaml")   # every setting, commented, at its default
# edit my_config.yaml, then:
settings = resolve_settings(schema=default_schema(), config_path="my_config.yaml")
graph = run_pipeline_stages(settings, default_schema())
```

`default_schema().describe()` is plain JSON — the same declaration the panel
builds its form from.

Path defaults in a generated config are relative, so they resolve against the
directory you run in (`images/`, `outputs/`, `plots/`). The examples in this
repository pin their own paths under `examples/`.
