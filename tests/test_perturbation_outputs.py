"""Perturbation disk outputs and napari layer gating."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics import (  # noqa: E402
    PERTURBATION_TYPES,
    SWEEP_PERTURBATION_TYPES,
    is_sweep_perturbation,
)
from haemolynx.gui.results import ResultLayers, perturbation_layer_names  # noqa: E402
from haemolynx.pipeline import PerturbationResult  # noqa: E402
from haemolynx.visualization.perturbation_plots import (  # noqa: E402
    SWEEP_AXIS_BY_TYPE,
    export_sweep_perturbation_plots,
    wants_napari_flow_layer,
)

# Reuse the stage test fixtures' network builder via a thin local copy so this
# file stays focused on outputs, not on haemodynamic numbers.
from test_perturbation_stage import (  # noqa: E402
    ARTERIOLE_DILATION,
    DILATION_SWEEP,
    SPACING_SWEEP,
    _run,
)


def test_every_declared_sweep_type_is_classified_as_a_sweep():
    declared_sweeps = {name for name in PERTURBATION_TYPES if "sweep" in name}
    assert declared_sweeps == set(SWEEP_PERTURBATION_TYPES)
    for name in declared_sweeps:
        assert is_sweep_perturbation(name)
        assert not wants_napari_flow_layer(name)
    for name in PERTURBATION_TYPES:
        if name not in declared_sweeps:
            assert not is_sweep_perturbation(name)
            if name != "none":
                assert wants_napari_flow_layer(name)


def test_every_sweep_type_has_axis_labelling():
    for name in SWEEP_PERTURBATION_TYPES:
        assert name in SWEEP_AXIS_BY_TYPE, name


def test_a_dilation_sweep_writes_alice_style_plots_and_keeps_no_graph(tmp_path):
    run = _run(tmp_path, [DILATION_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is None
    written = {path.name for path in result.output_dir.iterdir()}
    assert "pericyte_dilation_sweep.csv" in written
    assert "resistance_vs_pericyte_dilation.png" in written
    assert "flow_vs_pericyte_dilation.png" in written
    assert f"{DILATION_SWEEP['name']}_summary.csv" in written


def test_a_spacing_sweep_writes_axis_corrected_plots(tmp_path):
    run = _run(tmp_path, [SPACING_SWEEP])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is None
    written = {path.name for path in result.output_dir.iterdir()}
    assert "pericyte_spacing_sweep.csv" in written
    assert "resistance_vs_pericyte_spacing.png" in written
    assert "flow_vs_pericyte_spacing.png" in written


def test_a_non_sweep_writes_pipeline_like_artifacts_and_keeps_its_graph(tmp_path):
    run = _run(tmp_path, [ARTERIOLE_DILATION])
    result = run.results[0]
    assert result.error is None, result.error
    assert result.graph is not None
    written = {path.name for path in result.output_dir.iterdir()}
    assert f"{ARTERIOLE_DILATION['name']}_summary.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_edges.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_statistics.csv" in written
    assert f"{ARTERIOLE_DILATION['name']}_branch_statistics.csv" in written
    assert "node_degree_distribution.png" in written
    assert (
        "edges_and_nodes_overlay.png" in written
        or "edges_and_nodes_overlay_3d.html" in written
    )
    assert not list(result.output_dir.glob("*.vtp"))


def test_results_builder_skips_sweep_graphs_even_if_one_were_present():
    """Defence in depth: type, not only a cleared graph, gates the layer."""
    from test_gui_results import a_perturbation_run, solved_graph

    sweep = PerturbationResult(
        name="dilate_grid",
        type="pericyte_dilation_sweep",
        graph=solved_graph(),
    )
    single = PerturbationResult(
        name="art_dilate_20",
        type="arteriole_diameter_change",
        graph=solved_graph(),
    )
    group = ResultLayers().stage_finished(
        "run_perturbations", a_perturbation_run(sweep, single)
    )
    names = [spec.name for spec in group.layers]
    vessels, nodes = perturbation_layer_names("art_dilate_20")
    assert vessels in names and nodes in names
    assert perturbation_layer_names("dilate_grid")[0] not in names


def test_export_sweep_plots_accept_raw_rows(tmp_path):
    rows = [
        {
            "dilation_percent": 0,
            "inlet_pressure_pa": 5000,
            "equivalent_resistance": 1.0,
            "total_inlet_flow": 2.0,
        },
        {
            "dilation_percent": 10,
            "inlet_pressure_pa": 5000,
            "equivalent_resistance": 0.5,
            "total_inlet_flow": 4.0,
        },
    ]
    paths = export_sweep_perturbation_plots(
        "arteriole_diameter_sweep", rows, tmp_path
    )
    assert len(paths) == 2
    assert (tmp_path / "resistance_vs_arteriole_dilation.png").is_file()
    assert (tmp_path / "flow_vs_arteriole_dilation.png").is_file()
