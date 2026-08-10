"""A real run of a real image, and the flow you can actually get at afterwards.

Everything else about the results layers is checked with a hand-built graph,
which is fast and precise but shares the assumption it is testing: that the
pipeline produces what the fixture pretends it does. This one runs the pipeline
on `seven_vessel_noisy_3d.tif`, puts every stage into a real napari viewer as
the panel does, and then asks the question a user asks -- can I see the flow?

"Accessible" is four separate things, and flow has failed each of them at some
point in this feature's life:

    it is on the layer                      -- the value reached the viewer
    it is in `features.columns`             -- napari's own "edge feature:"
                                               dropdown, on the left, reads this
                                               list once when the controls are
                                               built and never again
    colouring by it does not raise          -- `KeyError: nan` came from a
                                               cycle-mode layer meeting the NaN
                                               that sparse flows are full of
    the colours actually vary               -- a column can be present, listed
                                               and mappable and still paint the
                                               whole network one colour
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from haemolynx.gui._widget import _apply_layers, _colour_layer  # noqa: E402
from haemolynx.gui.results import NODES, VESSELS, ResultLayers  # noqa: E402
from haemolynx.pipeline import (  # noqa: E402
    default_schema,
    resolve_settings,
    run_pipeline_stages,
)

pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"

#: The columns the solve is responsible for. Flow is what this file is about;
#: the others ride along because they arrive at the same moment and by the same
#: route, so if one of them is missing the cause is the same.
SOLVE_COLUMNS = ("flow_abs", "flow_signed", "pressure_drop", "pressure_u", "pressure_v")


@pytest.fixture(scope="module")
def run_in_a_viewer(tmp_path_factory):
    """The fixture put through the pipeline, drawn stage by stage, as the panel does."""
    tmp = tmp_path_factory.mktemp("flow_in_viewer")
    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": FIXTURE,
            "vtk_output_prefix": tmp / "run",
            "plot_dir": tmp / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)

    viewer = napari.Viewer(show=False)
    results = ResultLayers()
    stages: list[str] = []
    failures: list[tuple[str, Exception]] = []

    def produced(stage: str, output) -> None:
        stages.append(stage)
        try:
            _apply_layers(viewer, results.stage_finished(stage, output))
        except Exception as exc:  # noqa: BLE001 - reported, not raised, as in the panel
            failures.append((stage, exc))

    graph = run_pipeline_stages(settings, schema, on_stage_output=produced)
    yield viewer, results, graph, stages, failures
    viewer.close()


def test_the_run_reaches_the_solve(run_in_a_viewer):
    """Without this the rest would pass vacuously on a run that never solved."""
    _viewer, _results, _graph, stages, failures = run_in_a_viewer
    assert failures == [], failures
    assert "solve" in stages, stages


def test_absolute_flow_is_on_the_vessels_layer(run_in_a_viewer):
    viewer, _results, graph, _stages, _failures = run_in_a_viewer
    layer = viewer.layers[VESSELS]

    assert "flow_abs" in layer.features
    drawn = np.asarray(layer.features["flow_abs"], dtype=float)
    assert len(drawn) == len(layer.data)

    finite = drawn[np.isfinite(drawn)]
    assert len(finite) > 0, "the solve wrote no flow at all"
    assert (finite >= 0).all(), "an absolute flow cannot be negative"
    assert finite.max() > 0, "every flow is zero, so nothing is actually flowing"


def test_absolute_flow_is_in_the_list_napari_shows_on_the_left(run_in_a_viewer):
    """`features.columns` is what the "edge feature:" dropdown is built from.

    napari fills that combo box in the layer controls' constructor and connects
    to nothing that fires when features change, so a column that only turns up
    later never appears there. Being in this list is the whole difference
    between flow being reachable and not.
    """
    viewer, _results, _graph, _stages, _failures = run_in_a_viewer
    columns = list(viewer.layers[VESSELS].features.columns)

    for name in SOLVE_COLUMNS:
        assert name in columns, f"{name} missing from {columns}"
    assert "pressure" in list(viewer.layers[NODES].features.columns)


def test_colouring_the_vessels_by_absolute_flow_works(run_in_a_viewer):
    """The end of the road: ask for it the way the panel does, and look."""
    viewer, _results, _graph, _stages, _failures = run_in_a_viewer
    layer = viewer.layers[VESSELS]

    # Through a categorical colouring first, which is what a real run does at
    # the diameters stage and what used to leave the layer unable to take a
    # column containing NaN.
    _colour_layer(layer, "branch_order", "categorical",
                  (("BO1", (1.0, 0.0, 0.0, 1.0)),))
    _colour_layer(layer, "flow_abs", "continuous")

    colours = np.asarray(layer.edge_color)
    assert len(colours) == len(layer.data)
    assert np.isfinite(colours).all(), "a colour came back NaN"
    assert len(np.unique(colours, axis=0)) > 1, (
        "every vessel is the same colour, so the flow is not being shown"
    )


def test_the_panel_offers_absolute_flow_once_the_run_has_finished(run_in_a_viewer):
    from haemolynx.gui._widget import _colour_choices

    viewer, _results, _graph, _stages, _failures = run_in_a_viewer
    assert "flow_abs" in _colour_choices(viewer, VESSELS)
    assert "pressure" in _colour_choices(viewer, NODES)


def test_the_drawn_flow_is_the_graphs_flow(run_in_a_viewer):
    """Present and colourable is not the same as right.

    Each vessel is drawn as several segments, so its value is repeated across
    them; check the set of values on the layer against the set on the graph.
    """
    viewer, _results, graph, _stages, _failures = run_in_a_viewer
    drawn = np.asarray(viewer.layers[VESSELS].features["flow_abs"], dtype=float)

    edges = (graph.edges(keys=True, data=True) if graph.is_multigraph()
             else ((u, v, 0, d) for u, v, d in graph.edges(data=True)))
    from_graph = np.array(
        [float(d["flow_abs"]) for *_ids, d in edges if d.get("flow_abs") is not None]
    )

    assert len(from_graph) > 0
    assert np.allclose(
        np.unique(drawn[np.isfinite(drawn)]), np.unique(from_graph), rtol=1e-12
    )
