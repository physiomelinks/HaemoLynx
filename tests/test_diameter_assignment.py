"""Diameters exist without resistance, with a recorded source, and can be kept.

``assign_diameters`` stamps modelled diameters; ``build_haemodynamic_model``
writes Poiseuille resistance afterwards. A resume with ``do_fwhm_measurement``
off keeps measured and override values instead of remeasuring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics.apply import (  # noqa: E402
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
    assign_edge_diameters,
)
from haemolynx.haemodynamics.poiseuille import (  # noqa: E402
    DIAMETER_SOURCE_MEASURED,
    DIAMETER_SOURCE_OVERRIDE,
    DIAMETER_SOURCE_TABLE,
    set_edge_diameter_override,
)
from haemolynx.pipeline import (  # noqa: E402
    BoundaryNodes,
    default_schema,
    resolve_settings,
)
from haemolynx.pipeline.stages import (  # noqa: E402
    SkeletonisedVolume,
    VesselNetwork,
    assign_diameters,
    build_haemodynamic_model,
)

SCHEMA = default_schema()

EDGE_LENGTH_UM = 400.0
DIAMETERS = {"Art1": 20.0, "B01": 6.0, "Ven1": 20.0}
BRANCH_ORDERS = ("Art1", "B01", "B01", "Ven1")


def _network() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for node in range(len(BRANCH_ORDERS) + 1):
        graph.add_node(node, pos=np.asarray([0.0, 0.0, node * EDGE_LENGTH_UM]))
    for node, branch_order in enumerate(BRANCH_ORDERS):
        graph.add_edge(
            node,
            node + 1,
            key=0,
            branch_order=branch_order,
            length=EDGE_LENGTH_UM,
            voxels=[
                [0.0, 0.0, node * EDGE_LENGTH_UM],
                [0.0, 0.0, (node + 1) * EDGE_LENGTH_UM],
            ],
        )
    return graph


def _vessel_network(tmp_path: Path, graph: nx.MultiGraph | None = None) -> VesselNetwork:
    volume = SkeletonisedVolume(
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        skeleton=np.zeros((2, 2, 2), dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
    )
    return VesselNetwork(graph=graph if graph is not None else _network(), volume=volume)


def _boundaries() -> BoundaryNodes:
    last = len(BRANCH_ORDERS)
    return BoundaryNodes(
        inlet_nodes=[0],
        outlet_nodes=[last],
        arteriole_boundary_nodes=[0],
        venule_boundary_nodes=[last],
        resistance_node_pair=(0, last),
    )


def _settings(tmp_path: Path, **extra) -> dict:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": plot_dir,
            "run_haemodynamics": True,
            "inlet_nodes": [0],
            "outlet_nodes": [len(BRANCH_ORDERS)],
            "arteriole_boundary_nodes": [0],
            "venule_boundary_nodes": [len(BRANCH_ORDERS)],
            "use_fwhm_edge_diameters": False,
            "viscosity_law": "constant",
            "strict_branch_order_assignment": False,
            "automated_vessel_assignment": False,
            "use_small_vessel_masks_for_boundary_assignment": False,
        }
    )
    values.update(extra)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _config(**fwhm) -> HaemodynamicsApplyConfig:
    return HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": False, **fwhm},
    )


def _sources(graph: nx.MultiGraph) -> dict[tuple, str]:
    return {
        (u, v, key): data["diameter_source"]
        for u, v, key, data in graph.edges(keys=True, data=True)
    }


def _has_resistance(graph: nx.MultiGraph) -> bool:
    return any(
        "resistance" in data or "conductance" in data
        for _, _, data in graph.edges(data=True)
    )


def test_assign_diameters_skips_vessel_type_plotly_when_ide_plots_are_off(
    tmp_path, monkeypatch
):
    """Napari turns IDE plots off; Diameters must not dump every voxel to HTML."""

    def boom(*_args, **_kwargs):
        raise AssertionError("vessel-type Plotly HTML must not run with IDE plots off")

    monkeypatch.setattr(
        "haemolynx.pipeline.stages.visualization.visualize_3d_plotly_vessel_types",
        boom,
    )
    assign_diameters(
        _settings(tmp_path, show_plots_in_ide=False, interactive_plots=False),
        _vessel_network(tmp_path),
        _boundaries(),
        SCHEMA,
    )


def test_assign_diameters_writes_vessel_type_plotly_when_ide_plots_are_on(
    tmp_path, monkeypatch
):
    called = []

    def fake_plotly(*_args, **_kwargs):
        called.append(True)

    monkeypatch.setattr(
        "haemolynx.pipeline.stages.visualization.visualize_3d_plotly_vessel_types",
        fake_plotly,
    )
    assign_diameters(
        _settings(tmp_path, show_plots_in_ide=True, interactive_plots=False),
        _vessel_network(tmp_path),
        _boundaries(),
        SCHEMA,
    )
    assert called == [True]


def test_assign_diameters_does_not_write_resistance(tmp_path):
    model = assign_diameters(
        _settings(tmp_path),
        _vessel_network(tmp_path),
        _boundaries(),
        SCHEMA,
    )
    assert not _has_resistance(model.graph)
    for _u, _v, _key, data in model.graph.edges(keys=True, data=True):
        assert data["diameter_source"] == DIAMETER_SOURCE_TABLE
        assert float(data["diameter_um"]) > 0
    assert "resistances" not in model.results


def test_build_haemodynamic_model_writes_resistance_from_stamped_diameters(tmp_path):
    diameters = assign_diameters(
        _settings(tmp_path),
        _vessel_network(tmp_path),
        _boundaries(),
        SCHEMA,
    )
    model = build_haemodynamic_model(_settings(tmp_path / "model"), diameters, SCHEMA)
    for _u, _v, _key, data in model.graph.edges(keys=True, data=True):
        assert "resistance" in data
        assert "conductance" in data
        assert data["conductance"] == pytest.approx(1.0 / data["resistance"])
    assert "poiseuille" in model.results.get("resistances", {})


def test_library_apply_still_writes_diameters_and_resistances():
    graph, summary = apply_poiseuille_haemodynamics(
        _network(),
        diameter_by_branch_order=dict(DIAMETERS),
    )
    assert _has_resistance(graph)
    assert set(_sources(graph).values()) == {DIAMETER_SOURCE_TABLE}
    assert "poiseuille" in summary.get("resistances", {})


def test_set_edge_diameter_override_survives_resume_even_when_fwhm_is_present():
    graph = _network()
    data = graph[1][2][0]
    data["fwhm_diameter_um"] = 4.0
    set_edge_diameter_override(data, 9.0)

    stamped, summary, raw = assign_edge_diameters(
        graph,
        _config(use_fwhm_edge_diameters=True, do_fwhm_measurement=False),
    )
    assert raw is None
    assert not _has_resistance(stamped)
    assert stamped[1][2][0]["diameter_source"] == DIAMETER_SOURCE_OVERRIDE
    assert stamped[1][2][0]["diameter_um"] == pytest.approx(9.0)
    assert summary["diameters"]["override"] == 1
    assert summary["diameters"]["table"] == 3


def test_resume_keeps_measured_and_override_and_does_not_remeasure(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("FWHM must not be remeasured when the toggle is off")

    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply._measure_fwhm_diameters", boom
    )

    graph = _network()
    measured = graph[0][1][0]
    measured["fwhm_diameter_um"] = 4.0
    measured["diameter_um"] = 4.0
    measured["diameter_source"] = DIAMETER_SOURCE_MEASURED
    measured["resistance"] = 99.0
    measured["conductance"] = 0.01
    set_edge_diameter_override(graph[1][2][0], 9.0)
    graph[1][2][0]["resistance"] = 99.0

    stamped, summary, _raw = assign_edge_diameters(
        graph,
        _config(use_fwhm_edge_diameters=True, do_fwhm_measurement=False),
    )
    assert not _has_resistance(stamped)
    assert summary["fwhm"]["skipped"] is True
    assert stamped[0][1][0]["diameter_source"] == DIAMETER_SOURCE_MEASURED
    assert stamped[0][1][0]["diameter_um"] == pytest.approx(4.0)
    assert stamped[1][2][0]["diameter_source"] == DIAMETER_SOURCE_OVERRIDE
    assert stamped[1][2][0]["diameter_um"] == pytest.approx(9.0)
    assert stamped[2][3][0]["diameter_source"] == DIAMETER_SOURCE_TABLE
    assert stamped[2][3][0]["diameter_um"] == pytest.approx(6.0)


def test_fresh_fwhm_run_wipes_overrides(monkeypatch):
    def fake_measure(G, _config, raw_volume=None):
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            data["fwhm_diameter_um"] = 3.0
        return {"edges_measured": G.number_of_edges(), "edges_skipped": []}

    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply._measure_fwhm_diameters", fake_measure
    )
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_fwhm_raw_volume",
        lambda _config: np.zeros((2, 2, 2), dtype=np.float32),
    )

    graph = _network()
    set_edge_diameter_override(graph[0][1][0], 9.0)
    stamped, summary, raw = assign_edge_diameters(
        graph,
        _config(use_fwhm_edge_diameters=True, do_fwhm_measurement=True),
    )
    assert raw is not None
    assert not _has_resistance(stamped)
    assert summary["diameters"]["measured"] == stamped.number_of_edges()
    assert summary["diameters"]["override"] == 0
    assert set(_sources(stamped).values()) == {DIAMETER_SOURCE_MEASURED}
    assert stamped[0][1][0]["diameter_um"] == pytest.approx(3.0)


def test_set_edge_diameter_override_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        set_edge_diameter_override({}, 0.0)
    with pytest.raises(ValueError, match="positive"):
        set_edge_diameter_override({}, float("nan"))
