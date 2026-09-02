"""Scaling every arteriole by one factor, on a synthetic Art/B/Ven network.

Two things have to be true for this to answer the question it claims to. The
arterioles have to actually move -- including on a run whose diameters were
measured per edge, where `prefer_edge_fwhm_diameter` makes the per-edge value
win over the table -- and nothing else may move with them, or the difference in
the answer is not the dilation.
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

from haemolynx.haemodynamics.arteriole import (  # noqa: E402
    is_arteriole_branch_order,
    percent_change_to_scale,
    scale_arteriole_diameters,
)
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel  # noqa: E402

#: The baseline table. `build_diameter_by_branch_order` writes `Art1`/`Ven1`
#: unpadded and capillaries as `B01`, so an arteriole is recognised by prefix.
DIAMETERS = {"Art1": 7.0, "B01": 5.0, "Ven1": 8.0}


def _network() -> nx.MultiGraph:
    """An arteriole, two capillaries and a venule in a line, each 10 um."""
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


# --- the label convention ----------------------------------------------------


@pytest.mark.parametrize(
    "branch_order,expected",
    [("Art1", True), ("Art12", True), ("B01", False), ("Ven1", False), (None, False)],
)
def test_only_the_arteriole_labels_are_arterioles(branch_order, expected):
    assert is_arteriole_branch_order(branch_order) is expected


# --- percent -> scale -------------------------------------------------------


@pytest.mark.parametrize(
    "percent,scale",
    [(0, 1.0), (10, 1.1), (-20, 0.8), (100, 2.0), (-50, 0.5)],
)
def test_percent_change_converts_to_a_scale_factor(percent, scale):
    assert percent_change_to_scale(percent) == pytest.approx(scale)


@pytest.mark.parametrize("percent", [-100, -150, -100.0])
def test_a_percent_that_would_zero_or_invert_diameter_is_refused(percent):
    with pytest.raises(ValueError, match="not > 0"):
        percent_change_to_scale(percent)


def test_a_twenty_percent_dilation_matches_a_scale_of_one_point_two():
    """The user-facing percent and the internal factor agree on the graph."""
    by_percent, table_p, _ = scale_arteriole_diameters(
        _network(),
        dict(DIAMETERS),
        percent_change_to_scale(20),
        model=_model("constant"),
    )
    by_scale, table_s, _ = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model("constant")
    )
    assert _resistances(by_percent) == _resistances(by_scale)
    assert table_p == table_s


# --- what scaling does ------------------------------------------------------


def test_a_scale_of_one_reproduces_the_baseline_exactly():
    """Not approximately: the no-op arm of a comparison must be the baseline.

    A scale of 1.0 is how a run gets a control that shares every other setting,
    so any difference at all here would be attributed to the perturbation.
    """
    scaled, table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.0, model=_model()
    )

    assert _resistances(scaled) == _baseline()
    assert table == DIAMETERS


def test_dilating_the_arterioles_lowers_their_resistance_by_the_fourth_power():
    """Poiseuille is d^-4, but only at a fixed viscosity.

    Checked under `constant`, because the pries law makes viscosity a function
    of diameter -- so under the default law the ratio is close to 1.2^-4 and
    deliberately not equal to it.
    """
    baseline = _baseline("constant")

    scaled, _table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model("constant")
    )

    after = _resistances(scaled)
    assert after["Art1"] == pytest.approx(baseline["Art1"] * 1.2 ** -4, rel=1e-12)
    assert after["Art1"] < baseline["Art1"]


def test_the_default_viscosity_law_does_not_give_a_pure_fourth_power():
    """Worth pinning: the apparent viscosity moves with the diameter too.

    Under `pries` a dilated arteriole carries thinner blood as well as being a
    wider tube, so its resistance falls by rather more than d^-4 -- 7% more at
    this diameter. Quoting 1.2^-4 for a run under the default law would be
    wrong, so the real number is pinned here rather than a tolerance wide
    enough to cover it.
    """
    baseline = _baseline("pries")

    scaled, _table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model("pries")
    )

    ratio = _resistances(scaled)["Art1"] / baseline["Art1"]
    assert ratio == pytest.approx(0.448202, rel=1e-5)
    assert ratio < 1.2 ** -4
    assert ratio / 1.2 ** -4 == pytest.approx(0.9293, rel=1e-3)


def test_nothing_but_the_arterioles_moves():
    """Bit-identical, not close: a changed capillary is a changed answer."""
    baseline = _baseline()

    scaled, _table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model()
    )

    after = _resistances(scaled)
    assert after["B01"] == baseline["B01"]
    assert after["Ven1"] == baseline["Ven1"]


def test_every_edge_of_an_arteriole_branch_order_moves_together():
    """Whole-branch scaling: two Art1 edges dilate by the same percent."""
    graph = _network()
    # A second arteriole segment on the same branch order.
    graph.add_node(5, pos=np.asarray((0.0, 0.0, -10.0), dtype=float))
    graph.add_edge(
        5,
        0,
        key=0,
        length=10.0,
        branch_order="Art1",
        fwhm_diameter_um=DIAMETERS["Art1"],
    )

    scaled, _table, summary = scale_arteriole_diameters(
        graph, dict(DIAMETERS), percent_change_to_scale(20), model=_model()
    )

    art_diameters = [
        data["fwhm_diameter_um"]
        for _u, _v, data in scaled.edges(data=True)
        if data["branch_order"] == "Art1"
    ]
    assert len(art_diameters) == 2
    assert art_diameters[0] == pytest.approx(7.0 * 1.2)
    assert art_diameters[1] == pytest.approx(7.0 * 1.2)
    assert summary["arteriole_edges"] == 2
    # Capillaries and venules keep their measured diameters.
    for _u, _v, data in scaled.edges(data=True):
        if data["branch_order"] != "Art1":
            assert data["fwhm_diameter_um"] == DIAMETERS[data["branch_order"]]


def test_scaling_does_not_place_focal_constriction_attributes():
    """Arteriole dilation is whole-branch; it is not a pericyte constriction."""
    scaled, _table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), percent_change_to_scale(20), model=_model()
    )
    for _u, _v, _key, data in scaled.edges(keys=True, data=True):
        assert "pericyte_centers_um" not in data
        assert "pericyte_count_assigned" not in data
        assert "constriction_sites" not in data


def test_the_returned_table_scales_only_the_arteriole_entries():
    _scaled, table, _summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.5, model=_model()
    )

    assert table == {"Art1": 7.0 * 1.5, "B01": 5.0, "Ven1": 8.0}


@pytest.mark.parametrize(
    "percent,expected_scale",
    [(20, 1.2), (-20, 0.8)],
)
def test_arteriole_percent_constricts_or_dilates(percent, expected_scale):
    """Negative percent narrows; positive widens — same control either way."""
    scaled, table, _summary = scale_arteriole_diameters(
        _network(),
        dict(DIAMETERS),
        percent_change_to_scale(percent),
        model=_model(),
    )
    assert table["Art1"] == pytest.approx(7.0 * expected_scale)
    assert table["B01"] == 5.0
    for _u, _v, data in scaled.edges(data=True):
        if data["branch_order"] == "Art1":
            assert data["fwhm_diameter_um"] == pytest.approx(7.0 * expected_scale)


def test_the_measured_per_edge_diameter_is_scaled_too():
    """`prefer_edge_fwhm_diameter` makes it win, so the table alone is not enough.

    Without this the perturbation would do nothing at all on exactly the runs
    that measured their diameters from the image, and say nothing about it.
    """
    scaled, _table, summary = scale_arteriole_diameters(
        _network(), dict(DIAMETERS), 1.2, model=_model()
    )

    measured = {
        data["branch_order"]: data["fwhm_diameter_um"]
        for _u, _v, data in scaled.edges(data=True)
    }
    assert measured["Art1"] == pytest.approx(7.0 * 1.2)
    assert measured["B01"] == 5.0
    assert measured["Ven1"] == 8.0
    assert summary["edges_with_measured_diameter_scaled"] == 1
    assert summary["arteriole_edges"] == 1
    assert summary["branch_orders_scaled"] == ("Art1",)
    assert summary["resistances"]["edges_set"] == 4


def test_an_unmeasured_arteriole_still_scales_through_the_table():
    """A run with no FWHM measurement has only the branch-order table."""
    graph = _network()
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        del data["fwhm_diameter_um"]
    baseline, _results = _model().set_poiseuille_resistances(
        graph.copy(), dict(DIAMETERS), prefer_edge_fwhm_diameter=True
    )

    scaled, _table, summary = scale_arteriole_diameters(
        graph, dict(DIAMETERS), 1.2, model=_model()
    )

    assert summary["edges_with_measured_diameter_scaled"] == 0
    assert _resistances(scaled)["Art1"] < _resistances(baseline)["Art1"]


def test_the_baseline_graph_is_left_alone():
    """Every perturbation runs from the same baseline, so none may edit it."""
    graph = _network()
    before = {
        (u, v, key): dict(data) for u, v, key, data in graph.edges(keys=True, data=True)
    }

    scale_arteriole_diameters(graph, dict(DIAMETERS), 1.2, model=_model())

    after = {
        (u, v, key): dict(data) for u, v, key, data in graph.edges(keys=True, data=True)
    }
    assert after == before
    assert not any("resistance" in data for data in after.values())


def test_the_baseline_table_is_left_alone():
    table = dict(DIAMETERS)

    scale_arteriole_diameters(_network(), table, 1.2, model=_model())

    assert table == DIAMETERS


@pytest.mark.parametrize("scale", [0.0, -1.0, -0.5])
def test_a_scale_that_is_not_positive_is_refused(scale):
    """A zero or negative diameter is not a narrower vessel, it is nonsense."""
    with pytest.raises(ValueError, match="scale must be > 0"):
        scale_arteriole_diameters(_network(), dict(DIAMETERS), scale, model=_model())


def test_it_is_reachable_from_the_subpackage():
    import haemolynx.haemodynamics as haemodynamics

    assert haemodynamics.scale_arteriole_diameters is scale_arteriole_diameters
    assert haemodynamics.percent_change_to_scale is percent_change_to_scale
    assert "scale_arteriole_diameters" in haemodynamics.__all__
    assert "percent_change_to_scale" in haemodynamics.__all__
