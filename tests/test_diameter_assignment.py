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
    DIAMETER_SOURCE_CLASS_MEDIAN,
    DIAMETER_SOURCE_EDT,
    DIAMETER_SOURCE_MEASURED,
    DIAMETER_SOURCE_OVERRIDE,
    DIAMETER_SOURCE_TABLE,
    PoiseuilleModel,
    build_diameter_by_branch_order,
    flag_fwhm_edt_disagreement,
    set_edge_diameter_override,
    stamp_edge_diameters,
    table_diameter_for_order,
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
    """Produce IDE plots off; Diameters must not dump every voxel to HTML."""

    def boom(*_args, **_kwargs):
        raise AssertionError("vessel-type Plotly HTML must not run with IDE plots off")

    monkeypatch.setattr(
        "haemolynx.pipeline.stages.visualization.visualize_3d_plotly_vessel_types",
        boom,
    )
    assign_diameters(
        _settings(tmp_path, visualize_results=False),
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
        _settings(tmp_path, visualize_results=True),
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


def test_override_beats_the_table_when_nothing_measures_the_vessel():
    """A table-only run (FWHM off) keeps a hand-set diameter, e.g. a vessel
    drawn in the post-processing tab; its neighbours still take the table."""
    graph = _network()
    set_edge_diameter_override(graph[1][2][0], 9.0)
    stamped, summary, _raw = assign_edge_diameters(graph, _config())
    assert stamped[1][2][0]["diameter_source"] == DIAMETER_SOURCE_OVERRIDE
    assert stamped[1][2][0]["diameter_um"] == pytest.approx(9.0)
    assert summary["diameters"]["override"] == 1
    assert summary["diameters"]["table"] == stamped.number_of_edges() - 1
    others = {k: v for k, v in _sources(stamped).items() if k[:2] != (1, 2)}
    assert set(others.values()) == {DIAMETER_SOURCE_TABLE}


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
    def fake_measure(G, _config, raw_volume=None, **_kwargs):
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


def test_edt_measurement_runs_before_fwhm_so_it_can_seed_the_diameter_guess(monkeypatch):
    """Regression: EDT measurement used to run after FWHM in
    assign_edge_diameters, so edt_diameter_um was never actually on the
    graph in time for FWHM's own diameter_guess_edge_attribute to read it
    -- confirms EDT now runs first, and that its result is genuinely
    visible to FWHM's own measurement call by then, not just that the
    call order changed with no effect on what FWHM actually sees.
    """
    call_order: list[str] = []
    seen_edt_values: list[float | None] = []

    def fake_measure_edt(G, _config, mask_volume=None):
        call_order.append("edt")
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            data["edt_diameter_um"] = 7.0
        return {"edges_measured": G.number_of_edges(), "edges_skipped": []}

    def fake_measure_fwhm(G, _config, raw_volume=None, **_kwargs):
        call_order.append("fwhm")
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            seen_edt_values.append(data.get("edt_diameter_um"))
            data["fwhm_diameter_um"] = 3.0
        return {"edges_measured": G.number_of_edges(), "edges_skipped": []}

    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply._measure_edt_diameters", fake_measure_edt
    )
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply._measure_fwhm_diameters", fake_measure_fwhm
    )
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_fwhm_raw_volume",
        lambda _config: np.zeros((2, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_edt_mask_volume",
        lambda _config: np.zeros((2, 2, 2), dtype=bool),
    )

    graph = _network()
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True},
        edt={"use_edt_diameter_crosscheck": True},
    )
    assign_edge_diameters(graph, config)

    assert call_order == ["edt", "fwhm"]
    assert seen_edt_values == [7.0] * graph.number_of_edges()


def test_a_label_encoded_segmentation_is_binarised_before_either_measurement_reads_it(
    monkeypatch,
):
    """Regression: the stage hands over the loaded segmentation as it is. An
    ilastik "Simple Segmentation" is labels 1 (vessel) and 2 (background),
    and read as bool both are "vessel" -- EDT widths meaningless, and FWHM's
    neighbour stop seeing one vessel everywhere. Both must get the mask the
    skeleton was made from: the minority label."""
    seen: dict[str, np.ndarray] = {}

    def fake_measure_edt(G, _config, mask_volume=None):
        seen["edt"] = mask_volume
        return {"edges_measured": 0, "edges_skipped": []}

    def fake_measure_fwhm(G, _config, raw_volume=None, vessel_mask=None, **_kwargs):
        seen["fwhm"] = vessel_mask
        return {"edges_measured": 0, "edges_skipped": []}

    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_edt_diameters", fake_measure_edt)
    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_fwhm_diameters", fake_measure_fwhm)
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_fwhm_raw_volume",
        lambda _config: np.zeros((4, 4, 4), dtype=np.float32),
    )
    labels = np.full((4, 4, 4), 2, dtype=np.uint8)
    labels[1:3, 1:3, :] = 1  # the vessel: the minority label
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True},
        edt={"use_edt_diameter_crosscheck": True},
    )

    assign_edge_diameters(_network(), config, mask_volume=labels)

    for name in ("edt", "fwhm"):
        assert seen[name].dtype == bool, name
        np.testing.assert_array_equal(seen[name], labels == 1)


def test_set_edge_diameter_override_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        set_edge_diameter_override({}, 0.0)
    with pytest.raises(ValueError, match="positive"):
        set_edge_diameter_override({}, float("nan"))


# --- EDT fallback / disagreement flag (stamp_edge_diameters, flag_fwhm_edt_disagreement) --


def _table_only_network() -> nx.MultiGraph:
    """One edge with only a branch order and an ``edt_diameter_um`` -- no
    ``fwhm_diameter_um`` at all, as if FWHM measurement failed for it."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.asarray([0.0, 0.0, EDGE_LENGTH_UM]))
    graph.add_edge(
        0, 1, key=0, branch_order="B01", length=EDGE_LENGTH_UM, edt_diameter_um=5.0
    )
    return graph


def test_stamp_edge_diameters_use_edt_fallback_false_keeps_table_default():
    graph = _table_only_network()
    counts = stamp_edge_diameters(graph, DIAMETERS, use_edt_fallback=False)
    assert counts["edt_mask"] == 0
    assert counts["table"] == 1
    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_TABLE
    assert graph[0][1][0]["diameter_um"] == pytest.approx(DIAMETERS["B01"])


def test_stamp_edge_diameters_use_edt_fallback_true_prefers_edt_over_table():
    graph = _table_only_network()
    counts = stamp_edge_diameters(graph, DIAMETERS, use_edt_fallback=True)
    assert counts["edt_mask"] == 1
    assert counts["table"] == 0
    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_EDT
    assert graph[0][1][0]["diameter_um"] == pytest.approx(5.0)


_TABLE_TO_51 = build_diameter_by_branch_order(
    all_diams_const=False,
    max_branch_order=51,
    default_diameter=4.0,
    manual_capillary_diameter_by_branch_order={"B01": 6.2},
    manual_large_arteriole_diameter_by_branch_order={"Large_Art1": 30.0},
)


@pytest.mark.parametrize(
    ("order", "expected"),
    [("B51", 4.0), ("B58", 4.0), ("Art60", 6.2), ("Large_Art60", 4.0), ("B01", 6.2)],
)
def test_an_order_past_the_table_takes_its_last_entry_of_the_same_kind(order, expected):
    """The table stops at max_branch_order (51); a real network ran to B59.
    Past the end, every order would take the same default the table's last
    entries do, so the last one of that kind stands in."""
    assert table_diameter_for_order(_TABLE_TO_51, order) == pytest.approx(expected)


@pytest.mark.parametrize("order", ["B03", "Foo7", "unassigned", None])
def test_a_gap_or_an_unknown_label_still_has_no_table_entry(order):
    table = {"B01": 5.0, "B02": 5.0, "B04": 5.0}
    assert table_diameter_for_order(table, order) is None


def test_a_vessel_past_the_table_with_no_width_of_its_own_gets_the_table_diameter():
    """Regression: a B58 vessel that neither FWHM nor the mask could measure
    (here, one the mask left as too thin to resolve) was left unset, and the
    run then failed with "No fallback baseline diameter for branch_order
    'B58'"."""
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, branch_order="B58", length=EDGE_LENGTH_UM,
                   edt_unresolved_diameter_um=1.4)

    counts = stamp_edge_diameters(graph, _TABLE_TO_51, use_edt_fallback=True)
    PoiseuilleModel(40.0, 100.0).set_poiseuille_resistances(graph, _TABLE_TO_51)

    data = graph[0][1][0]
    assert counts["table"] == 1 and counts["unset"] == 0
    assert data["diameter_source"] == DIAMETER_SOURCE_TABLE
    assert data["diameter_um"] == pytest.approx(4.0)
    assert data["resistance"] > 0


def test_stamp_edge_diameters_fwhm_still_wins_over_edt_fallback():
    graph = _table_only_network()
    graph[0][1][0]["fwhm_diameter_um"] = 7.0
    stamp_edge_diameters(graph, DIAMETERS, use_edt_fallback=True)
    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_MEASURED
    assert graph[0][1][0]["diameter_um"] == pytest.approx(7.0)


def test_stamp_edge_diameters_keeps_an_edt_sourced_diameter_on_resume():
    """An edge previously resolved via the EDT fallback survives a
    ``keep_existing`` resume -- the same protection ``measured``/``override``
    already have."""
    graph = _table_only_network()
    stamp_edge_diameters(graph, DIAMETERS, use_edt_fallback=True)
    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_EDT

    counts = stamp_edge_diameters(graph, DIAMETERS, keep_existing=True)
    assert counts["edt_mask"] == 1
    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_EDT
    assert graph[0][1][0]["diameter_um"] == pytest.approx(5.0)


def test_fwhm_edt_disagreement_flag_fires_past_the_ratio():
    graph = _network()
    graph[0][1][0]["fwhm_diameter_um"] = 10.0
    graph[0][1][0]["edt_diameter_um"] = 5.0  # 2x disagreement

    flagged = flag_fwhm_edt_disagreement(graph, warn_ratio=1.5)

    assert flagged == 1
    assert graph[0][1][0]["fwhm_edt_disagreement_ratio"] == pytest.approx(2.0)
    assert graph[0][1][0]["fwhm_low_confidence_vs_edt"] is True


def test_fwhm_edt_disagreement_flag_stays_off_for_close_agreement():
    graph = _network()
    graph[0][1][0]["fwhm_diameter_um"] = 5.0
    graph[0][1][0]["edt_diameter_um"] = 5.2

    flagged = flag_fwhm_edt_disagreement(graph, warn_ratio=1.5)

    assert flagged == 0
    assert graph[0][1][0]["fwhm_low_confidence_vs_edt"] is False
    assert graph[0][1][0]["fwhm_edt_disagreement_ratio"] == pytest.approx(5.2 / 5.0)


def test_fwhm_edt_disagreement_flag_skips_edges_missing_either_value():
    graph = _network()
    graph[0][1][0]["fwhm_diameter_um"] = 5.0  # no edt_diameter_um at all

    flagged = flag_fwhm_edt_disagreement(graph, warn_ratio=1.5)

    assert flagged == 0
    assert "fwhm_edt_disagreement_ratio" not in graph[0][1][0]
    assert "fwhm_low_confidence_vs_edt" not in graph[0][1][0]


# --- the EDT mask defaults to the run's own segmented input --------------------------


def _run_settings(**values):
    from haemolynx.pipeline import default_schema

    settings = {setting.name: setting.default for setting in default_schema()}
    settings["diameter_by_branch_order"] = dict(DIAMETERS)
    settings.update(values)
    return settings


def test_the_segmented_input_is_the_input_path_without_ilastik(tmp_path):
    from haemolynx.pipeline.stages import segmented_input_path

    settings = _run_settings(input_path=tmp_path / "mask.tif", use_ilastik_segmentation=False)
    assert segmented_input_path(settings) == tmp_path / "mask.tif"


def test_with_ilastik_the_segmented_input_is_ilastiks_output_not_its_input(tmp_path):
    """The Input tab's input_path and ilastik's raw-image field are different
    files; with ilastik on, the segmented input is what ilastik writes -- even
    while input_path still holds an unrelated file from before."""
    from haemolynx.pipeline.stages import segmented_input_path

    settings = _run_settings(
        use_ilastik_segmentation=True,
        input_path=tmp_path / "an_old_mask.tif",
        ilastik_unsegmented_image_path=tmp_path / "raw" / "stack.tif",
        ilastik_output_dir=tmp_path / "ilastik",
        ilastik_output_suffix=".tiff",
    )
    expected = tmp_path / "ilastik" / "stack_segmented.tiff"
    assert segmented_input_path(settings) == expected

    # Once `segment` has run it replaces input_path with that same output.
    settings["input_path"] = expected
    assert segmented_input_path(settings) == expected


def test_an_unset_edt_mask_path_points_at_the_segmented_input(tmp_path):
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import _haemodynamics_apply_config

    schema = default_schema()
    unset = _haemodynamics_apply_config(
        _run_settings(input_path=tmp_path / "mask.tif", edt_mask_path=None),
        schema, voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    assert unset.edt_setting("edt_mask_path") == tmp_path / "mask.tif"

    chosen = _haemodynamics_apply_config(
        _run_settings(input_path=tmp_path / "mask.tif", edt_mask_path=tmp_path / "other.tif"),
        schema, voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    assert chosen.edt_setting("edt_mask_path") == tmp_path / "other.tif"


def test_without_a_mask_in_memory_both_measurements_read_the_one_from_the_path(monkeypatch):
    """A resumed run has no segmentation in memory: it is read once from
    edt_mask_path (the segmented input unless set) and binarised, for the EDT
    widths and for FWHM's neighbouring-vessel stops alike."""
    seen: dict[str, np.ndarray] = {}
    labels = np.full((4, 4, 4), 2, dtype=np.uint8)
    labels[1:3, 1:3, :] = 1
    loads = []

    def fake_load(_config):
        loads.append(1)
        return labels

    def fake_edt(G, _config, mask_volume=None):
        seen["edt"] = mask_volume
        return {}

    def fake_fwhm(G, _config, raw_volume=None, vessel_mask=None, **_kwargs):
        seen["fwhm"] = vessel_mask
        return {}

    monkeypatch.setattr("haemolynx.haemodynamics.apply.load_edt_mask_volume", fake_load)
    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_edt_diameters", fake_edt)
    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_fwhm_raw_volume",
        lambda _config: np.zeros((4, 4, 4), dtype=np.float32),
    )
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True},
        edt={"use_edt_diameter_crosscheck": True},
    )

    assign_edge_diameters(_network(), config)

    assert loads == [1]
    for name in ("edt", "fwhm"):
        np.testing.assert_array_equal(seen[name], labels == 1)


def test_the_edt_method_setting_reaches_the_measurement(monkeypatch):
    """edt_diameter_method picks how the mask's width is read; the default
    is the cross-section, and the old inscribed radius stays selectable."""
    from haemolynx.haemodynamics import apply
    from haemolynx.pipeline import default_schema

    seen = []
    monkeypatch.setattr(
        apply.edt_diameter, "measure_edge_diameters_from_binary_mask",
        lambda G, **kwargs: seen.append(kwargs["method"]) or {"edges_measured": 0, "edges_skipped": []},
    )
    for edt in ({"use_edt_diameter_crosscheck": True},
                {"use_edt_diameter_crosscheck": True, "edt_diameter_method": "inscribed_radius"}):
        config = HaemodynamicsApplyConfig(
            diameters={"diameter_by_branch_order": dict(DIAMETERS)}, edt=edt
        )
        apply._measure_edt_diameters(_network(), config, mask_volume=np.zeros((2, 2, 2), bool))

    assert seen == ["cross_section", "inscribed_radius"]
    assert default_schema()["edt_diameter_method"].default == "cross_section"


def test_the_resolvable_diameter_floor_setting_reaches_the_measurement(monkeypatch):
    """edt_min_resolvable_diameter_um sets the narrowest reading taken as a
    width; the schema's default is the measurement's own."""
    from haemolynx.haemodynamics import apply, edt_diameter
    from haemolynx.pipeline import default_schema

    seen = []
    monkeypatch.setattr(
        apply.edt_diameter, "measure_edge_diameters_from_binary_mask",
        lambda G, **kwargs: seen.append(kwargs["min_resolvable_diameter_um"])
        or {"edges_measured": 0, "edges_skipped": []},
    )
    for edt in ({"use_edt_diameter_crosscheck": True},
                {"use_edt_diameter_crosscheck": True, "edt_min_resolvable_diameter_um": 0.0}):
        config = HaemodynamicsApplyConfig(
            diameters={"diameter_by_branch_order": dict(DIAMETERS)}, edt=edt
        )
        apply._measure_edt_diameters(_network(), config, mask_volume=np.zeros((2, 2, 2), bool))

    default = default_schema()["edt_min_resolvable_diameter_um"].default
    assert seen == [edt_diameter.MIN_RESOLVABLE_DIAMETER_UM, 0.0]
    assert default == edt_diameter.MIN_RESOLVABLE_DIAMETER_UM


# --- the raw cross-section fallback, between FWHM and the mask ----------------------


def test_stamping_goes_fwhm_then_raw_section_then_the_mask_then_the_table():
    from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_RAW_SECTION

    graph = nx.MultiGraph()
    rows = [
        {"fwhm_diameter_um": 4.0, "raw_section_diameter_um": 5.0, "edt_diameter_um": 6.0},
        {"raw_section_diameter_um": 5.0, "edt_diameter_um": 6.0},
        {"edt_diameter_um": 6.0},
        {},
    ]
    for index, attrs in enumerate(rows):
        graph.add_edge(index, index + 10, key=0, branch_order="B01", length=EDGE_LENGTH_UM, **attrs)

    counts = stamp_edge_diameters(
        graph, DIAMETERS, use_edt_fallback=True, use_raw_section_fallback=True
    )

    sources = [graph[i][i + 10][0]["diameter_source"] for i in range(4)]
    widths = [graph[i][i + 10][0]["diameter_um"] for i in range(4)]
    assert sources == [
        DIAMETER_SOURCE_MEASURED, DIAMETER_SOURCE_RAW_SECTION, DIAMETER_SOURCE_EDT,
        DIAMETER_SOURCE_TABLE,
    ]
    assert widths == [4.0, 5.0, 6.0, DIAMETERS["B01"]]
    assert counts["raw_section"] == 1


def test_a_raw_section_width_is_ignored_while_the_fallback_is_off():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, branch_order="B01", length=EDGE_LENGTH_UM,
                   raw_section_diameter_um=5.0, edt_diameter_um=6.0)

    stamp_edge_diameters(graph, DIAMETERS, use_edt_fallback=True)

    assert graph[0][1][0]["diameter_source"] == DIAMETER_SOURCE_EDT


def test_keeping_existing_diameters_keeps_a_raw_section_one():
    from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_RAW_SECTION

    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, branch_order="B01", length=EDGE_LENGTH_UM,
                   diameter_um=5.0, diameter_source=DIAMETER_SOURCE_RAW_SECTION)

    counts = stamp_edge_diameters(graph, DIAMETERS, keep_existing=True)

    assert graph[0][1][0]["diameter_um"] == 5.0
    assert counts["raw_section"] == 1


def _fallback_run(monkeypatch, fwhm_settings):
    """assign_edge_diameters with FWHM measuring only the first edge, and the
    raw-section measurement recording what it was asked to do."""
    from haemolynx.haemodynamics import apply

    calls = []

    def fake_fwhm(G, _config, raw_volume=None, vessel_mask=None, **_kwargs):
        first = next(iter(G.edges(keys=True)))
        G.edges[first]["fwhm_diameter_um"] = 4.0
        return {}

    def fake_sections(G, **kwargs):
        calls.append(kwargs)
        for edge in kwargs["edges"]:
            G.edges[edge]["raw_section_diameter_um"] = 5.5
        return {"edges_measured": len(kwargs["edges"]), "edges_skipped": []}

    monkeypatch.setattr(apply, "_measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr(
        apply.raw_section, "measure_edge_diameters_from_raw_sections", fake_sections
    )
    monkeypatch.setattr(
        apply, "load_fwhm_raw_volume", lambda _config: np.zeros((4, 4, 4), dtype=np.float32)
    )
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True, **fwhm_settings},
    )
    graph = _network()
    graph, summary, _raw = assign_edge_diameters(graph, config)
    return graph, summary, calls


def test_the_raw_section_fallback_is_off_unless_asked_for(monkeypatch):
    graph, summary, calls = _fallback_run(monkeypatch, {})

    assert calls == []
    assert "raw_section" not in summary
    assert DIAMETER_SOURCE_TABLE in {d["diameter_source"] for _u, _v, d in graph.edges(data=True)}


def test_with_the_fallback_on_only_the_vessels_fwhm_failed_on_are_read(monkeypatch):
    from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_RAW_SECTION

    graph, summary, calls = _fallback_run(
        monkeypatch,
        {
            "use_raw_section_fallback": True,
            "raw_section_min_lumen_contrast": 4.5,
            "raw_section_psf_sigma_xy_um": 0.4,
            "raw_section_psf_sigma_z_um": 1.3,
            "fwhm_sample_spacing_along_edge_um": 3.0,
            "fwhm_longitudinal_average_um": 6.0,
        },
    )

    (call,) = calls
    first = next(iter(graph.edges(keys=True)))
    assert first not in call["edges"] and len(call["edges"]) == graph.number_of_edges() - 1
    assert call["psf_sigma_zyx"] == (1.3, 0.4, 0.4)
    assert call["min_lumen_contrast"] == 4.5
    assert call["sample_spacing_along_edge_um"] == 3.0
    assert call["average_along_vessel_um"] == 6.0
    sources = [d["diameter_source"] for _u, _v, d in graph.edges(data=True)]
    assert sources.count(DIAMETER_SOURCE_MEASURED) == 1
    assert sources.count(DIAMETER_SOURCE_RAW_SECTION) == graph.number_of_edges() - 1


def test_one_psf_width_alone_means_both_are_estimated(monkeypatch):
    _graph, _summary, calls = _fallback_run(
        monkeypatch, {"use_raw_section_fallback": True, "raw_section_psf_sigma_xy_um": 0.4}
    )

    assert calls[0]["psf_sigma_zyx"] is None


def test_the_raw_section_settings_are_off_by_default_and_follow_fwhm():
    schema = default_schema()

    assert schema["use_raw_section_fallback"].default is False
    assert set(schema["use_raw_section_fallback"].requires) == {
        "use_fwhm_edge_diameters", "do_fwhm_measurement",
    }
    for name in ("raw_section_psf_sigma_xy_um", "raw_section_psf_sigma_z_um"):
        assert schema[name].default is None


# --- one PSF for the raw image's widths; flagged FWHM widths step aside ---------


def _fake_measurements(monkeypatch, tmp_path, *, fwhm_um=3.0, edt_um=None, flag_specks=False):
    """Replace the heavy measurements with ones that record what they were
    handed and write fixed widths; returns the record."""
    import haemolynx.haemodynamics.apply as apply_module

    seen: dict = {"fwhm_psf": [], "raw_psf": [], "raw_edges": []}

    def fake_fwhm(G, **kwargs):
        seen["fwhm_psf"].append(kwargs.get("profile_psf_sigma_zyx"))
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            data["fwhm_diameter_um"] = fwhm_um
            if flag_specks:
                data["fwhm_in_speck_width_range"] = True
        return {"edges_measured": G.number_of_edges(), "edges_skipped": []}

    def fake_raw(G, **kwargs):
        seen["raw_psf"].append(kwargs.get("psf_sigma_zyx"))
        seen["raw_edges"].append(list(kwargs.get("edges") or []))
        for edge in kwargs.get("edges") or []:
            G.edges[edge]["raw_section_diameter_um"] = 5.5
        return {"edges_measured": len(kwargs.get("edges") or []), "edges_skipped": []}

    def fake_edt(G, _config, mask_volume=None):
        for _u, _v, _key, data in G.edges(keys=True, data=True):
            data["edt_diameter_um"] = edt_um
        return {"edges_measured": G.number_of_edges(), "edges_skipped": []}

    monkeypatch.setattr(apply_module.automated, "measure_edge_diameters_fwhm_from_raw_tiff", fake_fwhm)
    monkeypatch.setattr(apply_module.raw_section, "measure_edge_diameters_from_raw_sections", fake_raw)
    monkeypatch.setattr(apply_module, "_measure_edt_diameters", fake_edt)
    monkeypatch.setattr(apply_module, "load_fwhm_raw_volume", lambda _c: np.zeros((2, 2, 2), np.float32))
    monkeypatch.setattr(apply_module, "load_edt_mask_volume", lambda _c: np.zeros((2, 2, 2), bool))
    raw_path = tmp_path / "raw.tif"
    raw_path.write_bytes(b"x")
    return seen, raw_path


def _fwhm_config(raw_path, *, edt=False, **fwhm):
    return HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={
            "use_fwhm_edge_diameters": True,
            "do_fwhm_measurement": True,
            "fwhm_raw_tiff_path": raw_path,
            "fwhm_decoy_check": False,
            "use_raw_section_fallback": True,
            **fwhm,
        },
        edt={"use_edt_diameter_crosscheck": edt, "edt_diameter_prefer_over_table_on_fwhm_failure": True},
    )


def test_fwhm_and_the_raw_section_fallback_share_one_image_psf(monkeypatch, tmp_path):
    """FWHM fitted each profile's blur afresh while the raw-section fit, on
    the same image, held one PSF: the two methods disagreed about what the
    image's blur is."""
    seen, raw_path = _fake_measurements(monkeypatch, tmp_path, fwhm_um=None)
    graph = _network()

    _graph, summary, _raw = assign_edge_diameters(
        graph,
        _fwhm_config(raw_path, raw_section_psf_sigma_xy_um=0.4, raw_section_psf_sigma_z_um=1.2),
    )

    assert seen["fwhm_psf"] == [(1.2, 0.4, 0.4)]
    assert seen["raw_psf"] == [(1.2, 0.4, 0.4)]
    assert graph.graph["fwhm_psf_sigma_zyx"] == (1.2, 0.4, 0.4)
    assert summary["fwhm_psf"]["source"] == "settings"


def test_image_calibrations_sample_only_the_calibration_edges(monkeypatch, tmp_path):
    """What the diameters stage passes when the haemodynamics will remove a
    branch without an inlet and an outlet: the blur, the decoys and the
    fallback's own blur come from the vessels kept, while every vessel is
    still measured."""
    import haemolynx.haemodynamics.apply as apply_module

    _seen, raw_path = _fake_measurements(monkeypatch, tmp_path, fwhm_um=3.0)
    sampled = {}

    def fake_psf(_G, _raw, _voxel, **kwargs):
        sampled["psf"] = kwargs.get("edges")
        return None, {"reason": "too few wide vessels"}

    def fake_decoys(_G, _measure, **kwargs):
        sampled["decoys"] = kwargs.get("edges")
        return {"skipped": True, "reason": "not under test"}

    def fake_raw(_G, **kwargs):
        sampled["raw_section"] = kwargs.get("calibration_edges")
        return {"edges_measured": 0, "edges_skipped": []}

    monkeypatch.setattr(apply_module.raw_section, "estimate_psf_sigma", fake_psf)
    monkeypatch.setattr(apply_module.fwhm_decoys, "fwhm_decoy_check", fake_decoys)
    monkeypatch.setattr(apply_module.raw_section, "measure_edge_diameters_from_raw_sections", fake_raw)
    graph = _network()
    kept = [(0, 1, 0), (1, 2, 0)]

    assign_edge_diameters(
        graph, _fwhm_config(raw_path, fwhm_decoy_check=True, fwhm_demote_flagged_edges=False),
        calibration_edges=iter(kept),
    )

    assert sampled == {"psf": kept, "decoys": kept, "raw_section": kept}
    assert all(d["fwhm_diameter_um"] == 3.0 for _u, _v, d in graph.edges(data=True))


def test_fwhm_fits_each_profiles_blur_when_the_shared_psf_is_switched_off(monkeypatch, tmp_path):
    seen, raw_path = _fake_measurements(monkeypatch, tmp_path)

    assign_edge_diameters(
        _network(),
        _fwhm_config(
            raw_path,
            fwhm_fix_blur_to_image_psf=False,
            raw_section_psf_sigma_xy_um=0.4,
            raw_section_psf_sigma_z_um=1.2,
        ),
    )

    assert seen["fwhm_psf"] == [None]


def test_a_width_disagreeing_with_the_mask_goes_to_the_raw_section_fit(monkeypatch, tmp_path):
    """Regression: the FWHM/EDT disagreement flag was written after the
    diameters were chosen, so a width 3x the mask's own was modelled
    regardless; now the edge is re-measured from its raw section first."""
    seen, raw_path = _fake_measurements(monkeypatch, tmp_path, fwhm_um=3.0, edt_um=9.0)
    graph = _network()

    assign_edge_diameters(graph, _fwhm_config(raw_path, edt=True))

    assert len(seen["raw_edges"][0]) == graph.number_of_edges()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        assert data["fwhm_demoted"] == "edt_disagreement"
        assert data["diameter_source"] == "raw_section"
        assert data["diameter_um"] == pytest.approx(5.5)
        assert data["fwhm_diameter_um"] == pytest.approx(3.0)  # kept for review


def test_flagged_widths_are_modelled_as_before_when_demotion_is_off(monkeypatch, tmp_path):
    seen, raw_path = _fake_measurements(monkeypatch, tmp_path, fwhm_um=3.0, edt_um=9.0)
    graph = _network()

    assign_edge_diameters(graph, _fwhm_config(raw_path, edt=True, fwhm_demote_flagged_edges=False))

    assert seen["raw_edges"][0] == []
    assert set(_sources(graph).values()) == {DIAMETER_SOURCE_MEASURED}


def test_a_speck_width_steps_aside_for_the_next_source():
    from haemolynx.haemodynamics.poiseuille import mark_fwhm_demotions

    graph = _network()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        data["fwhm_diameter_um"] = 1.5
        data["edt_diameter_um"] = 6.0
        data["fwhm_in_speck_width_range"] = True

    counts = mark_fwhm_demotions(graph)
    stamp_edge_diameters(graph, dict(DIAMETERS), use_edt_fallback=True)

    assert counts == {"speck_width": graph.number_of_edges(), "edt_disagreement": 0}
    assert set(_sources(graph).values()) == {DIAMETER_SOURCE_EDT}


# --- class median: an unmeasured vessel takes its label's measured median -----


def _labelled_network(widths_by_label: dict[str, list]) -> nx.MultiGraph:
    """A chain of edges, one per width; ``None`` is an edge nothing measured."""
    graph = nx.MultiGraph()
    node = 0
    graph.add_node(node, pos=np.asarray([0.0, 0.0, 0.0]))
    for label, widths in widths_by_label.items():
        for width in widths:
            graph.add_node(node + 1, pos=np.asarray([0.0, 0.0, 10.0 * (node + 1)]))
            attrs = {"branch_order": label, "length": 10.0}
            if width is not None:
                attrs["fwhm_diameter_um"] = float(width)
            graph.add_edge(node, node + 1, key=0, **attrs)
            node += 1
    return graph


def _unmeasured(graph: nx.MultiGraph) -> list[dict]:
    return [data for _u, _v, data in graph.edges(data=True) if "fwhm_diameter_um" not in data]


_LARGE_TABLE = build_diameter_by_branch_order(
    all_diams_const=False, max_branch_order=3, default_diameter=4.0
)


def test_an_unmeasured_large_arteriole_takes_its_labels_median_not_the_default():
    """Regression: a large arteriole nothing measured fell back to the table,
    which has no entry for it, so it was modelled at default_diameter (4 um)
    beside measured neighbours of ~30 um -- thousands of times their
    resistance."""
    graph = _labelled_network({"Large_Art1": [28.0, 30.0, 34.0, None]})

    counts = stamp_edge_diameters(graph, _LARGE_TABLE, class_median_min_edges=3)

    (edge,) = _unmeasured(graph)
    assert edge["diameter_source"] == DIAMETER_SOURCE_CLASS_MEDIAN
    assert edge["diameter_um"] == pytest.approx(30.0)
    assert counts["class_median"] == 1 and counts["table"] == 0


def test_too_few_measured_edges_leave_the_table():
    graph = _labelled_network({"Large_Art1": [28.0, 30.0, None]})

    stamp_edge_diameters(graph, _LARGE_TABLE, class_median_min_edges=3)

    (edge,) = _unmeasured(graph)
    assert edge["diameter_source"] == DIAMETER_SOURCE_TABLE


def test_zero_goes_straight_to_the_table():
    graph = _labelled_network({"Large_Art1": [28.0, 30.0, 34.0, None]})

    stamp_edge_diameters(graph, _LARGE_TABLE, class_median_min_edges=0)

    (edge,) = _unmeasured(graph)
    assert edge["diameter_source"] == DIAMETER_SOURCE_TABLE
    assert edge["diameter_um"] == pytest.approx(4.0)


def test_the_median_is_per_label_and_only_over_measurements():
    graph = _labelled_network(
        {"B01": [5.0, 5.0, 5.0], "Large_Art1": [None, None, None, None]}
    )
    # A hand-set width is not this run's measurement of its label.
    set_edge_diameter_override(_unmeasured(graph)[0], 40.0)

    stamp_edge_diameters(graph, _LARGE_TABLE, class_median_min_edges=3)

    large = [d for d in _unmeasured(graph) if d["diameter_source"] != DIAMETER_SOURCE_OVERRIDE]
    assert {d["diameter_source"] for d in large} == {DIAMETER_SOURCE_TABLE}


def test_a_median_of_endothelial_widths_keeps_their_wall_to_wall_basis():
    graph = _labelled_network({"Large_Art1": [None, None, None, None]})
    edges = [data for _u, _v, data in graph.edges(data=True)]
    for data, width in zip(edges[:3], (30.0, 32.0, 34.0)):
        data["endothelial_diameter_um"] = width

    stamp_edge_diameters(graph, _LARGE_TABLE, use_endothelial=True, class_median_min_edges=3)

    assert edges[3]["diameter_source"] == DIAMETER_SOURCE_CLASS_MEDIAN
    assert edges[3]["diameter_um"] == pytest.approx(32.0)
    assert edges[3]["diameter_basis"] == "anatomical"


def test_a_run_uses_the_class_median_by_default():
    graph = _labelled_network({"Large_Art1": [28.0, 30.0, 34.0, None]})
    for _u, _v, data in graph.edges(data=True):
        if "fwhm_diameter_um" in data:
            data["diameter_um"] = data["fwhm_diameter_um"]
            data["diameter_source"] = DIAMETER_SOURCE_MEASURED

    stamped, summary, _raw = assign_edge_diameters(
        graph,
        HaemodynamicsApplyConfig(
            diameters={"diameter_by_branch_order": dict(_LARGE_TABLE)},
            fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": False},
        ),
    )

    (edge,) = _unmeasured(stamped)
    assert edge["diameter_source"] == DIAMETER_SOURCE_CLASS_MEDIAN
    assert summary["diameters"]["class_median"] == 1


# --- measuring only the vessels an edit touched (the post_process stage) -------


def test_fresh_edges_go_down_the_whole_chain_while_the_rest_keep_theirs():
    """keep_existing keeps an override; a vessel just measured again does not
    keep its provisional one when a measurement beats it."""
    graph = _network()
    for u, v, k, data in graph.edges(keys=True, data=True):
        set_edge_diameter_override(data, 11.0)
        data["fwhm_diameter_um"] = 3.0
    fresh = [(1, 0, 0)]  # either way round

    stamp_edge_diameters(graph, dict(DIAMETERS), keep_existing=True, fresh_edges=fresh)

    assert graph.edges[0, 1, 0]["diameter_source"] == DIAMETER_SOURCE_MEASURED
    assert graph.edges[0, 1, 0]["diameter_um"] == pytest.approx(3.0)
    for u, v in ((1, 2), (2, 3), (3, 4)):
        assert graph.edges[u, v, 0]["diameter_source"] == DIAMETER_SOURCE_OVERRIDE
        assert graph.edges[u, v, 0]["diameter_um"] == pytest.approx(11.0)


def test_a_fresh_edge_no_method_measured_keeps_its_provisional_diameter_over_the_table():
    graph = _network()
    set_edge_diameter_override(graph.edges[1, 2, 0], 11.0)

    stamp_edge_diameters(graph, dict(DIAMETERS), keep_existing=True, fresh_edges=[(1, 2, 0)])

    assert graph.edges[1, 2, 0]["diameter_source"] == DIAMETER_SOURCE_OVERRIDE
    assert graph.edges[1, 2, 0]["diameter_um"] == pytest.approx(11.0)


def _measurement_spies(monkeypatch):
    seen: dict = {"fwhm": [], "psf": 0, "decoy": 0, "raw_loaded": 0}

    def fake_fwhm(G, _config, raw_volume=None, **kwargs):
        seen["fwhm"].append((kwargs.get("edges"), kwargs.get("psf_sigma_zyx")))
        for u, v, k in kwargs.get("edges") or []:
            G.edges[u, v, k]["fwhm_diameter_um"] = 5.5
        return {"edges_measured": len(kwargs.get("edges") or []), "edges_skipped": []}

    def fake_psf(*_a, **_k):
        seen["psf"] += 1
        return (1.0, 0.5, 0.5), {"source": "image"}

    def fake_decoy(*_a, **_k):
        seen["decoy"] += 1
        return {}

    def fake_raw(_config):
        seen["raw_loaded"] += 1
        return np.zeros((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr("haemolynx.haemodynamics.apply._measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr("haemolynx.haemodynamics.apply.image_psf_for_fwhm", fake_psf)
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.fwhm_decoys.fwhm_decoy_check", fake_decoy
    )
    monkeypatch.setattr("haemolynx.haemodynamics.apply.load_fwhm_raw_volume", fake_raw)
    monkeypatch.setattr(
        "haemolynx.haemodynamics.apply.load_edt_mask_volume",
        lambda _config: np.zeros((2, 2, 2), dtype=bool),
    )
    return seen


def test_a_subset_is_measured_by_fwhm_even_with_do_fwhm_measurement_off(monkeypatch):
    """A resumed run turns do_fwhm_measurement off to keep what it measured;
    a vessel drawn since has nothing to keep."""
    seen = _measurement_spies(monkeypatch)
    graph = _network()
    graph.graph["fwhm_psf_sigma_zyx"] = (2.0, 1.0, 1.0)
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["fwhm_diameter_um"] = 9.0
        data["diameter_source"] = DIAMETER_SOURCE_MEASURED
        data["diameter_um"] = 9.0
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={
            "use_fwhm_edge_diameters": True,
            "do_fwhm_measurement": False,
            "fwhm_decoy_check": True,
        },
    )

    _graph, summary, _raw = assign_edge_diameters(graph, config, edges=[(2, 1, 0)])

    assert seen["fwhm"] == [([(2, 1, 0)], (2.0, 1.0, 1.0))], "the recorded PSF, the edge asked"
    assert seen["psf"] == 0, "the run's recorded PSF is not estimated again"
    assert seen["decoy"] == 0 and summary["fwhm_decoy_check"]["skipped"]
    assert graph.edges[1, 2, 0]["diameter_um"] == pytest.approx(5.5)
    assert graph.edges[0, 1, 0]["diameter_um"] == pytest.approx(9.0)


def test_an_empty_subset_measures_nothing_and_still_restamps_the_table(monkeypatch):
    """Only a deletion was made: no image is read, but a table diameter follows
    the branch order just assigned."""
    seen = _measurement_spies(monkeypatch)
    graph = _network()
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["diameter_source"] = DIAMETER_SOURCE_TABLE
        data["diameter_um"] = 99.0  # stale: its order's table value is not this
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": dict(DIAMETERS)},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": False},
        edt={"use_edt_diameter_crosscheck": True},
    )

    assign_edge_diameters(graph, config, edges=[])

    assert seen["fwhm"] == [] and seen["raw_loaded"] == 0
    for (u, v), order in zip(((0, 1), (1, 2), (2, 3), (3, 4)), BRANCH_ORDERS):
        assert graph.edges[u, v, 0]["diameter_um"] == pytest.approx(DIAMETERS[order])
