# Developing HaemoLynx

The editable install is in the [top-level README](../README.md). This is
everything after that.

- [Dependencies](#dependencies)
- [Testing](#testing)
- [Does this branch change the numbers?](#does-this-branch-change-the-numbers)
- [Adding a setting](#adding-a-setting)
- [Releasing](#releasing)

## Dependencies

`pyproject.toml` is the single source of truth — there is no
`requirements.txt`. The extras:

| Extra | For |
|---|---|
| `dev` | pytest and the test tooling |
| `napari` | the panel, plus a Qt binding (PyQt6) |
| `napari-plugin` | the panel, for an environment that already has a binding |
| `notebook` | the Jupyter kernel for the tutorial notebook |

Python 3.9 or newer for the library; **3.11+ for the napari panel**, which is
napari's floor rather than ours.

## Testing

```bash
pytest                                    # everything
pytest -m "not slow"                      # skip the slow integration tests
QT_QPA_PLATFORM=offscreen pytest -m gui   # the panel; needs napari and pytest-qt
```

The panel's tests build real Qt widgets, so they need napari, a Qt binding and a
display. They are marked `gui`, skipped without one, and CI runs them on 3.11
under xvfb.

**`pytest-qt` must never go in the `dev` extra.** It aborts an entire pytest run
when no Qt binding is installed, so it belongs only in an environment that has
one.

**Import test helpers by bare name** — `from foo import ...`, not
`from tests.foo import ...`. The latter resolves only under `python -m pytest`
from the repository root; CI runs bare `pytest` and it fails there and only
there. `tests/test_repo_conventions.py` guards this.

## Does this branch change the numbers?

`scripts/compare_branches.py` runs the resistance network pipeline twice on the
same dataset — once on your checkout, once on a reference ref in a temporary
`git worktree` — and reports every way the two runs differ:

```bash
python scripts/compare_branches.py                       # against main
python scripts/compare_branches.py --ref devel           # against another ref
python scripts/compare_branches.py --image /data/x.tif   # another dataset
python scripts/compare_branches.py --setting min_stub_length=5.0
```

It writes `COMPARISON.md` and an `index.html` — every plot from both runs, side
by side — into `comparison_outputs/` (gitignored). The report covers graph
metrics, **the first graph-building stage whose output diverges** (which is what
localises a regression), edge attributes, the statistics CSVs, the VTK exports,
and runtime.

The experiment is the config file's, not the tool's: both sides run what the
checkout's config says, so comparing two branches compares what those branches
actually do. The tool pins only what the comparison needs mechanically — the
statistics it reads, the display settings that would otherwise open a browser
mid-run, and each side's own output paths.

This is **not** part of the test suite and never runs in CI: it takes roughly 15
minutes per side on `examples/images/Nerve_capillaries.tif`, a 328 MB image that
is not in the repository. To check the tool itself works — about a minute, on
the committed fixture:

```bash
python scripts/compare_branches.py --self-check --smoke
```

That compares `HEAD` against `HEAD` and exits non-zero if anything differs. The
reporting logic has fast unit tests in `tests/test_branch_comparison.py`, which
do run in CI.

Before trusting a report:

- If either side fails, the tool says which and why, and prints no tables — a
  partial comparison is never presented as a complete one.
- Older entry points take settings differently and read some from module
  constants. Each side is inspected and adapted; a setting that defines the
  comparison and cannot be applied stops that side rather than silently running
  a different configuration, and one a branch simply does not have is listed as
  a caveat.
- Boundary boxes are in physical (z, y, x) **micrometres**, not voxel indices.

## Adding a setting

Settings are declared once, as a schema, and that declaration generates the
config file, the command-line flags, the panel's form and the validation. The
pipeline's own live in `haemolynx.pipeline.schema`; an example that adds
settings of its own declares those beside it (`examples/*_schema.py`) on top.

Give it a `unit` if it has one. This pipeline measures some things in voxels and
some in microns, 10 is a reasonable value for either, and picking the wrong one
is silent. `tests/test_setting_units.py` fails on a new length-like setting that
declares neither a unit nor a reason it has none.

After editing a schema, regenerate the config files:

```bash
python examples/regenerate_configs.py
```

It keeps the values already in your config files and adds new settings with
their documentation — so **do not run it while anything else is writing one**
(a napari session pressing *Save config*, say), or you will commit whatever that
wrote.

## Releasing

`.github/workflows/release.yml` builds, checks and publishes. Two ways in,
deliberately different in how easy they are:

| | How | Where | Repeatable |
|---|---|---|---|
| Rehearsal | Actions → Release → **Run workflow** | TestPyPI | yes |
| Real | push a tag `vX.Y.Z` | PyPI | **no** |

Both build the wheel and sdist, run `twine check`, then install what was built
into an empty environment on 3.9 and 3.12 and use it from a directory that is
not the repository — so a packaging mistake is caught before the upload rather
than by whoever installs it next. The suite runs against `src/` and cannot see
one. A tagged run additionally refuses to publish if the tag and the version in
`pyproject.toml` disagree.

Uploading uses **Trusted Publishing**: GitHub proves who it is with a
short-lived token, so no API token is stored anywhere. It is configured once per
index, under *Manage project → Publishing*, with owner `physiomelinks`,
repository `HaemoLynx`, workflow `release.yml`, and environment `testpypi` or
`pypi` to match. Without it the upload fails with `invalid-publisher`.

**A version number is spent once.** Neither index lets a version be re-uploaded,
even after deleting it, so bump `version` in `pyproject.toml` for each attempt.
The rehearsal job passes `skip-existing` so repeating it is harmless; the PyPI
job deliberately does not.

To cut a release: bump `version` in `pyproject.toml`, tag, push.

## A few things that will bite you

**Do not stack PRs.** Every PR goes to `main`. Three stacked PRs were once
merged within 30 seconds, and because the base branch had already gone to main,
two of them landed on dead ends — eleven commits silently missed main while all
three PRs showed "merged". After any merge, check with
`git merge-base --is-ancestor <branch> origin/main`, not the PR's badge.

**Never commit generated output.** `outputs/`, `plots/`, `tests/plots/`,
`comparison_outputs/` are gitignored. Do not `git add -A` at the repository
root.

**Edit the tutorial notebook, not `pipeline_tutorial.py`.** That file is
generated; regenerate it with
`pytest tests/integration/test_pipeline_tutorial.py`.
