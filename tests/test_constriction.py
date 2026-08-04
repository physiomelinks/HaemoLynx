"""Tests for the one constriction model shared by the pericyte strategies.

`_resolve_d1_d2_for_edge`, the diameter profile, the resistance integral and the
per-edge loop existed twice, once in `haemodynamics/pericyte_mask.py` and once
in `haemodynamics/probability.py`, differing only in error wording. They are now
one implementation in `haemodynamics/constriction.py`; these tests cover it
directly and pin the property that motivated the merge — that the two strategies
produce the *same* resistance for the same constriction sites.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics import constriction
from ImageLynx.haemodynamics.constriction import (
    apply_constriction_sites,
    diameter_at_position,
    integrated_resistance,
    resolve_edge_diameters,
)
from ImageLynx.haemodynamics.constriction_strategy import (
    set_resistances_for_constriction_strategy,
    uniform_constriction_factors,
)
from ImageLynx.haemodynamics.pericyte_mask import MaskConstrictionSites
from ImageLynx.haemodynamics.probability import (
    set_poiseuille_resistances_with_probabilistic_periodic_constrictions,
)


class _FixedSites:
    """A ConstrictionSites that returns the same centers for every edge."""

    def __init__(self, centers: list[float], summary_fields: dict | None = None):
        self.centers = centers
        self.summary_fields = summary_fields or {}
        self.calls: list[tuple] = []

    def centers_for_edge(self, u, v, key, edge_data, *, length):
        self.calls.append((u, v, key, length))
        return list(self.centers)

    def summary(self):
        return dict(self.summary_fields)


def _one_edge_graph(length: float = 100.0, **edge_attrs) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    graph.add_node(1, pos=np.asarray([0.0, 0.0, length]))
    graph.add_edge(0, 1, key=0, length=length, branch_order="B01", **edge_attrs)
    return graph


# --- resolve_edge_diameters -------------------------------------------------


def test_scalar_diameter_is_constricted_by_the_branch_order_factor():
    d1, d2, used_fwhm = resolve_edge_diameters(
        edge_data={},
        branch_order="B01",
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=False,
    )
    assert (d1, d2) == (5.0, 4.0)
    assert used_fwhm is False


def test_explicit_d1_d2_pair_is_used_as_given():
    d1, d2, used_fwhm = resolve_edge_diameters(
        edge_data={},
        branch_order="B01",
        diameter_by_branch_order={"B01": {"d1": 6.0, "d2": 3.0}},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=False,
    )
    assert (d1, d2) == (6.0, 3.0)
    assert used_fwhm is False


def test_measured_edge_diameter_supersedes_the_branch_order_table():
    d1, d2, used_fwhm = resolve_edge_diameters(
        edge_data={"fwhm_diameter_um": 4.0},
        branch_order="B01",
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.5},
        prefer_edge_fwhm_baseline=True,
    )
    assert (d1, d2) == (4.0, 2.0)
    assert used_fwhm is True


def test_unmeasured_edge_falls_back_to_the_table_diameter():
    d1, d2, used_fwhm = resolve_edge_diameters(
        edge_data={"fwhm_diameter_um": 0.0},
        branch_order="B01",
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.5},
        prefer_edge_fwhm_baseline=True,
    )
    assert (d1, d2) == (5.0, 2.5)
    assert used_fwhm is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "diameter_by_branch_order": {},
                "constriction_factor_by_branch_order": {"B01": 0.8},
                "prefer_edge_fwhm_baseline": False,
            },
            "No diameter mapping",
        ),
        (
            {
                "diameter_by_branch_order": {"B01": 5.0},
                "constriction_factor_by_branch_order": None,
                "prefer_edge_fwhm_baseline": False,
            },
            "No constriction factor",
        ),
        (
            {
                "diameter_by_branch_order": {"B01": {"d1": 5.0}},
                "constriction_factor_by_branch_order": {"B01": 0.8},
                "prefer_edge_fwhm_baseline": False,
            },
            "Invalid diameter mapping",
        ),
        (
            {
                "diameter_by_branch_order": {},
                "constriction_factor_by_branch_order": {"B01": 0.8},
                "prefer_edge_fwhm_baseline": True,
            },
            "No fallback baseline diameter",
        ),
        (
            {
                "diameter_by_branch_order": {"B01": {"d1": 5.0, "d2": 4.0}},
                "constriction_factor_by_branch_order": {"B01": 0.8},
                "prefer_edge_fwhm_baseline": True,
            },
            "numeric fallback baseline diameter",
        ),
    ],
)
def test_unusable_diameter_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_edge_diameters(edge_data={}, branch_order="B01", **kwargs)


# --- the diameter profile ---------------------------------------------------


def test_diameter_is_unconstricted_without_sites():
    assert diameter_at_position(50.0, 5.0, 3.0, [], 40.0) == 5.0


def test_diameter_is_fully_constricted_on_the_plateau():
    assert diameter_at_position(50.0, 5.0, 3.0, [50.0], 40.0) == 3.0
    assert diameter_at_position(59.9, 5.0, 3.0, [50.0], 40.0) == pytest.approx(3.0)


def test_diameter_ramps_linearly_back_over_the_next_quarter():
    # Half-way through the 10 um ramp of a 40 um window: half-way back to d1.
    assert diameter_at_position(65.0, 5.0, 3.0, [50.0], 40.0) == pytest.approx(4.0)


def test_diameter_is_unconstricted_outside_the_window():
    assert diameter_at_position(71.0, 5.0, 3.0, [50.0], 40.0) == 5.0


def test_overlapping_sites_take_the_narrowest_diameter():
    # 63 um is on the far ramp of the site at 50 but on the plateau of the one
    # at 70, so the second site wins.
    single = diameter_at_position(63.0, 5.0, 3.0, [50.0], 40.0)
    overlapping = diameter_at_position(63.0, 5.0, 3.0, [50.0, 70.0], 40.0)
    assert single == pytest.approx(3.6)
    assert overlapping == pytest.approx(3.0)


# --- the resistance integral ------------------------------------------------


def test_unconstricted_resistance_matches_the_closed_form():
    length, diameter = 100.0, 5.0
    viscosity = 1.0 / diameter ** constriction.VISCOSITY_DIAMETER_EXPONENT
    expected = (128.0 * viscosity * length) / (np.pi * diameter ** 4)
    computed = integrated_resistance(
        length=length,
        d1=diameter,
        d2=diameter,
        constriction_centers=[],
        constriction_length=40.0,
        num_points=1001,
    )
    assert computed == pytest.approx(expected, rel=1e-9)


def test_constricting_an_edge_raises_its_resistance():
    kwargs = dict(
        length=100.0,
        d1=5.0,
        constriction_length=40.0,
        num_points=1001,
    )
    unconstricted = integrated_resistance(
        d2=5.0, constriction_centers=[50.0], **kwargs
    )
    constricted = integrated_resistance(
        d2=3.0, constriction_centers=[50.0], **kwargs
    )
    assert constricted > unconstricted


def test_a_zero_length_edge_has_infinite_resistance():
    assert integrated_resistance(
        length=0.0,
        d1=5.0,
        d2=3.0,
        constriction_centers=[],
        constriction_length=40.0,
        num_points=101,
    ) == float("inf")


# --- applying sites to a graph ----------------------------------------------


def test_applying_sites_writes_resistance_conductance_and_the_sites_used():
    graph = _one_edge_graph()
    sites = _FixedSites([20.0, 60.0])

    graph, results = apply_constriction_sites(
        graph,
        sites,
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=False,
        constriction_length=40.0,
        num_integration_points=201,
    )

    edge = graph[0][1][0]
    assert edge["resistance"] > 0
    assert edge["conductance"] == pytest.approx(1.0 / edge["resistance"])
    assert edge["pericyte_count_assigned"] == 2
    assert edge["pericyte_centers_um"] == [20.0, 60.0]
    assert results["edges_set"] == 1
    assert results["used_fwhm_baseline"] == 0
    assert sites.calls == [(0, 1, 0, 100.0)]


def test_the_site_summary_is_merged_into_the_results():
    graph, results = apply_constriction_sites(
        _one_edge_graph(),
        _FixedSites([], {"pericyte_count": 7}),
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=False,
        constriction_length=40.0,
        num_integration_points=101,
    )
    assert results["pericyte_count"] == 7
    assert results["edges_set"] == 1


def test_measured_baseline_diameters_are_counted():
    graph = _one_edge_graph(fwhm_diameter_um=4.0)
    _graph, results = apply_constriction_sites(
        graph,
        _FixedSites([]),
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=True,
        constriction_length=40.0,
        num_integration_points=101,
    )
    assert results["used_fwhm_baseline"] == 1


@pytest.mark.parametrize(
    "attribute, message",
    [("branch_order", "missing required 'branch_order'"), ("length", "invalid length")],
)
def test_an_edge_without_the_required_attributes_is_rejected(attribute, message):
    graph = _one_edge_graph()
    del graph[0][1][0][attribute]
    with pytest.raises(ValueError, match=message):
        apply_constriction_sites(
            graph,
            _FixedSites([]),
            diameter_by_branch_order={"B01": 5.0},
            constriction_factor_by_branch_order={"B01": 0.8},
            prefer_edge_fwhm_baseline=False,
            constriction_length=40.0,
            num_integration_points=101,
        )


# --- the strategies agree ---------------------------------------------------


def test_mask_and_periodic_strategies_give_the_same_resistance_for_the_same_site():
    """The point of the merge: the strategies choose sites, not physics.

    A periodic run with one site at 20 um and a mask run whose pericyte landed
    at 20 um must produce the same number, because only one model exists.
    """
    periodic_graph, _results = set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
        _one_edge_graph(),
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        constriction_length=40.0,
        constriction_spacing=1000.0,  # long enough that only the first site fits
        constriction_probability=1.0,
    )
    assert periodic_graph[0][1][0]["pericyte_centers_um"] == [20.0]

    mask_graph, _results = apply_constriction_sites(
        _one_edge_graph(),
        MaskConstrictionSites(
            assigned_centers_by_edge={(0, 1, 0): [20.0]},
            summary_fields={},
        ),
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        prefer_edge_fwhm_baseline=False,
        constriction_length=40.0,
        num_integration_points=1000,
    )

    assert mask_graph[0][1][0]["resistance"] == periodic_graph[0][1][0]["resistance"]


def test_both_strategy_modules_use_the_one_shared_application():
    from ImageLynx.haemodynamics import pericyte_mask, probability

    assert pericyte_mask.apply_constriction_sites is apply_constriction_sites
    assert probability.apply_constriction_sites is apply_constriction_sites


# --- choosing a strategy ----------------------------------------------------


def test_uniform_constriction_factors_covers_every_branch_order():
    assert uniform_constriction_factors({"B01": 5.0, "Art1": 9.0}, 0.8) == {
        "B01": 0.8,
        "Art1": 0.8,
    }


@pytest.mark.parametrize(
    "flags, expected_strategy",
    [
        ({}, "constrictions"),
        ({"prefer_edge_fwhm_baseline": True}, "constrictions"),
        ({"use_probabilistic_constriction": True}, "probabilistic"),
    ],
)
def test_the_settings_select_the_strategy(flags, expected_strategy):
    settings = {
        "use_pericyte_mask_constriction": False,
        "use_probabilistic_constriction": False,
        "prefer_edge_fwhm_baseline": False,
        **flags,
    }
    graph = _one_edge_graph(fwhm_diameter_um=4.0)
    _graph, strategy, results = set_resistances_for_constriction_strategy(
        graph,
        diameter_by_branch_order={"B01": 5.0},
        constriction_factor_by_branch_order={"B01": 0.8},
        constriction_length=40.0,
        constriction_spacing=100.0,
        **settings,
    )
    assert strategy == expected_strategy
    assert results["edges_set"] == 1


def test_the_mask_strategy_needs_a_mask():
    with pytest.raises(ValueError, match="pericyte_mask_path must be set"):
        set_resistances_for_constriction_strategy(
            _one_edge_graph(),
            diameter_by_branch_order={"B01": 5.0},
            constriction_factor_by_branch_order={"B01": 0.8},
            use_pericyte_mask_constriction=True,
            use_probabilistic_constriction=False,
            prefer_edge_fwhm_baseline=False,
            constriction_length=40.0,
            constriction_spacing=100.0,
            pericyte_mask_path=None,
        )
