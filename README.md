# ImageLynx

Converts raw microscopy images of the microvasculature into computational haemodynamics models for hypothesis testing, experimental design, and more.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies. Add the
`notebook` extra (`pip install -e ".[dev,notebook]"`) to get the Jupyter kernel
needed to run `tutorials/pipeline_tutorial.ipynb` interactively.

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
`ImageLynx.pipeline` and the example just calls them in order, so you can read a
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
  starting_node_selection_method: coordinates

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

Settings are declared once, in the schema beside each example
(`examples/*_schema.py`), and that declaration generates the config file, the
command-line flags and the validation. After editing a schema, regenerate the
config files:

```bash
python examples/regenerate_configs.py
```

This keeps the values already in your config files and adds any new settings
with their documentation.

## Tutorial

`tutorials/pipeline_tutorial.ipynb` walks through the same pipeline stage by
stage with plots at each step. Edit the notebook, not the generated
`pipeline_tutorial.py`; regenerate that with:

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
