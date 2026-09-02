"""Scaling every capillary by one factor, on a synthetic Art/B/Ven network.

Passive whole-branch dilation: capillaries move together, arterioles and
venules stay put, and no focal constriction attributes appear.
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

from haemolynx.haemodynamics.capillary import (  # noqa: E402
    is_capillary_branch_order,
    percent_change_to_scale,
    scale_capillary_diameters,
)
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel  # noqa: E402

DIAMETERS = {"Art1": 7.0, "B01": 5.0, "Ven1": 8.0}


def _network() -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node_id, z in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        G.add_node(node_id, pos=np.asarray((0.0, 0.0, z), dtype=float))
    for u, v, branch_order in (
        (0, 1, "Art1"),
        (1, 2, "B01"),
        (2, 3, "B01"),
        (3, 4, "Ven1"),
    ):
        G.add_edge(
            u,
            v,
            key=0,
            length=10.0,
            branch_order=branch_order,
            fwhm_diameter_um=DIAMETERS[branch_order],
        )
    return G


def _model(viscosity_law: str = "pries") -> PoiseuilleModel:
    return PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
        viscosity_law=viscosity_law,
    )


def _resistances(graph: nx.MultiGraph) -> dict[str, float]:
    return {
        data["branch_order"]: data["resistance"]
        for _u, _v, data in graph.edges(data=True)
    }


def _baseline(viscosity_law: str = "pries") -> dict[str, float]:
    graph, _results = _model(viscosity_law).set_poiseuille_resistances(
        _network(), dict(DIAMETERS), prefer_edge_fwhm_diameter=True
    )
    return _resistances(graph)


@pytest.mark.parametrize(
    "branch_order,expected",
    [("B01", True), ("B12", True), ("Art1", False), ("Ven1", False), (None, False)],
)
def test_only_the_capillary_labels_are_capillaries(branch_order, expected):
    assert is_capillary_branch_order(branch_order) is expected


def test_nothing_but_the_capillaries_moves():
    baseline = _baseline()

    scaled, _table, _summary = scale_capillary_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model()
    )

    after = _resistances(scaled)
    assert after["Art1"] == baseline["Art1"]
    assert after["Ven1"] == baseline["Ven1"]
    assert after["B01"] < baseline["B01"]


def test_every_capillary_edge_of_a_branch_order_moves_together():
    scaled, _table, summary = scale_capillary_diameters(
        _network(), dict(DIAMETERS), percent_change_to_scale(20), model=_model()
    )

    cap_diameters = [
        data["fwhm_diameter_um"]
        for _u, _v, data in scaled.edges(data=True)
        if data["branch_order"] == "B01"
    ]
    assert len(cap_diameters) == 2
    assert cap_diameters[0] == pytest.approx(5.0 * 1.2)
    assert cap_diameters[1] == pytest.approx(5.0 * 1.2)
    assert summary["capillary_edges"] == 2
    for _u, _v, data in scaled.edges(data=True):
        if data["branch_order"] != "B01":
            assert data["fwhm_diameter_um"] == DIAMETERS[data["branch_order"]]


@pytest.mark.parametrize(
    "percent,expected_scale",
    [(20, 1.2), (-20, 0.8)],
)
def test_capillary_percent_constricts_or_dilates(percent, expected_scale):
    """Negative percent narrows; positive widens — same control either way."""
    scaled, table, _summary = scale_capillary_diameters(
        _network(),
        dict(DIAMETERS),
        percent_change_to_scale(percent),
        model=_model(),
    )
    assert table["B01"] == pytest.approx(5.0 * expected_scale)
    assert table["Art1"] == 7.0
    for _u, _v, data in scaled.edges(data=True):
        if data["branch_order"] == "B01":
            assert data["fwhm_diameter_um"] == pytest.approx(5.0 * expected_scale)


def test_scaling_does_not_place_focal_constriction_attributes():
    scaled, _table, _summary = scale_capillary_diameters(
        _network(), dict(DIAMETERS), percent_change_to_scale(20), model=_model()
    )
    for _u, _v, _key, data in scaled.edges(keys=True, data=True):
        assert "pericyte_centers_um" not in data
        assert "pericyte_count_assigned" not in data
        assert "constriction_sites" not in data


def test_the_returned_table_scales_only_the_capillary_entries():
    _scaled, table, _summary = scale_capillary_diameters(
        _network(), dict(DIAMETERS), 1.5, model=_model()
    )

    assert table == {"Art1": 7.0, "B01": 5.0 * 1.5, "Ven1": 8.0}


def test_the_baseline_graph_is_left_alone():
    graph = _network()
    before = {
        (u, v, key): dict(data) for u, v, key, data in graph.edges(keys=True, data=True)
    }

    scale_capillary_diameters(graph, dict(DIAMETERS), 1.2, model=_model())

    after = {
        (u, v, key): dict(data) for u, v, key, data in graph.edges(keys=True, data=True)
    }
    assert after == before


def test_it_is_reachable_from_the_subpackage():
    import haemolynx.haemodynamics as haemodynamics

    assert haemodynamics.scale_capillary_diameters is scale_capillary_diameters
    assert haemodynamics.is_capillary_branch_order is is_capillary_branch_order
    assert "scale_capillary_diameters" in haemodynamics.__all__
    assert "run_capillary_dilation_pressure_sweep" in haemodynamics.__all__


def test_capillary_dilation_sweep_scales_only_capillaries(tmp_path: Path):
    """Passive whole-capillary percent sweep leaves arterioles untouched."""
    from haemolynx.haemodynamics.capillary import (
        run_capillary_dilation_pressure_sweep,
    )

    G = _network()
    baseline_arts = [
        float(data["fwhm_diameter_um"])
        for _u, _v, data in G.edges(data=True)
        if str(data["branch_order"]).startswith("Art")
    ]
    settings = {
        "diameter_by_branch_order": dict(DIAMETERS),
        "outlet_p_bc": 1000.0,
        "inlet_p_bc": 4500.0,
        "use_fwhm_edge_diameters": True,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "capillary_dilation_min_percent": 0,
        "capillary_dilation_max_percent": 20,
        "capillary_dilation_step_percent": 20,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
    }
    sweep = run_capillary_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=[0],
        outlet_nodes=[4],
        output_dir=tmp_path,
        sweep_dilation=True,
        sweep_pressure=False,
    )
    assert Path(sweep["csv_path"]).name == "capillary_dilation_sweep.csv"
    assert len(sweep["results"]) == 2
    after_arts = [
        float(data["fwhm_diameter_um"])
        for _u, _v, data in G.edges(data=True)
        if str(data["branch_order"]).startswith("Art")
    ]
    assert after_arts == baseline_arts
    by_percent = {int(r["dilation_percent"]): r for r in sweep["results"]}
    assert by_percent[20]["equivalent_resistance"] < by_percent[0]["equivalent_resistance"]
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        assert "pericyte_centers_um" not in data
        assert "constriction_sites" not in data


def test_pressure_and_capillary_sweep_varies_both_axes(tmp_path: Path):
    from haemolynx.haemodynamics.capillary import (
        run_capillary_dilation_pressure_sweep,
    )

    G = _network()
    settings = {
        "diameter_by_branch_order": dict(DIAMETERS),
        "outlet_p_bc": 1000.0,
        "use_fwhm_edge_diameters": True,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "capillary_dilation_min_percent": 0,
        "capillary_dilation_max_percent": 10,
        "capillary_dilation_step_percent": 10,
        "inlet_pressure_min_pa": 4500,
        "inlet_pressure_max_pa": 5000,
        "inlet_pressure_step_pa": 500,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
    }
    sweep = run_capillary_dilation_pressure_sweep(
        G,
        settings,
        inlet_nodes=[0],
        outlet_nodes=[4],
        output_dir=tmp_path,
        sweep_dilation=True,
        sweep_pressure=True,
    )
    assert Path(sweep["csv_path"]).name == "capillary_dilation_pressure_sweep.csv"
    assert len(sweep["results"]) == 4
    assert {int(r["dilation_percent"]) for r in sweep["results"]} == {0, 10}
    assert {int(r["inlet_pressure_pa"]) for r in sweep["results"]} == {4500, 5000}
