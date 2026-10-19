"""Flux heterogeneity and the flowing share of the network
(``statistics/flux_heterogeneity.py``), on graphs with hand-computed answers."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import statistics_arguments
from haemolynx.statistics import (
    CONNECTED_NETWORK_MEASURES,
    NETWORK_ANALYSIS_MEASURES,
    STATISTIC_MEASURES,
    compute_flux_heterogeneity,
    compute_vessel_statistics_by_measure,
)
from haemolynx.statistics.flux_heterogeneity import (
    CARRYING_FLOOR_FRACTION,
    VELOCITY_EDGE_ATTRIBUTE,
    carrying_floor,
    is_carrying,
)

NL_PER_MIN = 1e12 * 60.0  # m^3/s -> nL/min


def _vessel(G, u, v, flow, diameter=10.0, length=100.0, branch_order="BO2", **extra):
    attributes = dict(
        flow_abs=float(flow),
        length=float(length),
        branch_order=branch_order,
        flow_state="solved",
        flow_solved=True,
    )
    if diameter is not None:
        attributes["diameter_um"] = float(diameter)
    attributes.update(extra)
    return G.add_edge(u, v, **attributes)


def _velocity_um_s(flow, diameter_um):
    return flow / (math.pi * (diameter_um * 0.5e-6) ** 2) * 1e6


def _volume_um3(diameter_um, length_um):
    return math.pi * (diameter_um / 2.0) ** 2 * length_um


def _cv(values):
    return float(np.std(values, ddof=1) / np.mean(values))


def _bed():
    """An arteriole feeding three capillaries side by side, of different
    flows and widths, draining into a venule."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 6e-12, diameter=20.0, length=200.0, branch_order="Art1")
    _vessel(G, 1, 2, 1e-12, diameter=5.0)
    _vessel(G, 1, 2, 2e-12, diameter=6.0)
    _vessel(G, 1, 2, 3e-12, diameter=8.0)
    _vessel(G, 2, 3, 6e-12, diameter=20.0, length=200.0, branch_order="Ven1")
    return G


CAPILLARY_FLOWS = [1e-12, 2e-12, 3e-12]
CAPILLARY_DIAMETERS = [5.0, 6.0, 8.0]


def test_the_capillary_and_vessel_cvs_are_the_sample_spread_over_the_carrying_vessels():
    G = _bed()
    result = compute_flux_heterogeneity(G)

    assert result["Vessels"] == 5
    assert result["Vessels Carrying Flow"] == 5
    assert result["Capillaries Carrying Flow"] == 3
    assert result["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert result["Capillary Flow Mean (nL/min)"] == pytest.approx(2e-12 * NL_PER_MIN)
    assert result["Capillary Flow SD (nL/min)"] == pytest.approx(1e-12 * NL_PER_MIN)
    velocities = [_velocity_um_s(q, d) for q, d in zip(CAPILLARY_FLOWS, CAPILLARY_DIAMETERS)]
    assert result["Capillary Velocity CV"] == pytest.approx(_cv(velocities))
    assert result["Capillary Velocity Mean (mm/s)"] == pytest.approx(np.mean(velocities) * 1e-3)
    all_flows = [6e-12, *CAPILLARY_FLOWS, 6e-12]
    assert result["Vessel Flow CV"] == pytest.approx(_cv(all_flows))
    all_velocities = [_velocity_um_s(6e-12, 20.0), *velocities, _velocity_um_s(6e-12, 20.0)]
    assert result["Vessel Velocity CV"] == pytest.approx(_cv(all_velocities))
    assert result["Carrying Flow Floor (m³/s)"] == pytest.approx(CARRYING_FLOOR_FRACTION * 6e-12)
    assert "sample SD" in result["Flux Heterogeneity Assumptions"]


def test_the_flowing_volume_is_a_cylinder_per_carrying_vessel():
    G = _bed()
    result = compute_flux_heterogeneity(G)

    capillary_volume = sum(_volume_um3(d, 100.0) for d in CAPILLARY_DIAMETERS)
    assert result["Flowing Capillary Volume (micron³)"] == pytest.approx(capillary_volume)
    assert result["Flowing Capillary Volume Fraction"] == pytest.approx(1.0)
    assert result["Flowing Vessel Volume (micron³)"] == pytest.approx(
        capillary_volume + 2 * _volume_um3(20.0, 200.0)
    )
    assert result["Flowing Vessel Volume Fraction"] == pytest.approx(1.0)
    assert result["Flowing Capillary Length (microns)"] == pytest.approx(300.0)
    assert result["Flowing Vessel Length (microns)"] == pytest.approx(700.0)
    assert result["Vessels Carrying Flow Fraction"] == pytest.approx(1.0)


def test_each_carrying_vessel_gets_its_mean_velocity_in_um_per_s():
    G = _bed()
    compute_flux_heterogeneity(G)
    for (_u, _v, key, data), flow, diameter in zip(
        [e for e in G.edges(keys=True, data=True) if e[3]["branch_order"] == "BO2"],
        CAPILLARY_FLOWS,
        CAPILLARY_DIAMETERS,
    ):
        assert data[VELOCITY_EDGE_ATTRIBUTE] == pytest.approx(_velocity_um_s(flow, diameter))
    # A 5 um capillary at 1 pL/s: 1e-12 / (pi 2.5e-6^2) = 50.9 mm/s... in um/s.
    assert G[1][2][0][VELOCITY_EDGE_ATTRIBUTE] == pytest.approx(1e-12 / (math.pi * 2.5e-6**2) * 1e6)


def test_vessels_behind_dead_ends_and_in_unreached_pieces_carry_no_flow():
    G = _bed()
    # A blind-ended capillary off the bed: flow exactly 0, which is its answer.
    _vessel(G, 1, 4, 0.0, diameter=5.0, flow_state="blind_end", flow_solved=False)
    # A piece no inlet and outlet reach, with whatever the solve left on it.
    _vessel(G, 7, 8, 2e-12, diameter=5.0, flow_state="no_boundary", flow_solved=False)
    result = compute_flux_heterogeneity(G)

    assert result["Vessels"] == 7
    assert result["Vessels Carrying Flow"] == 5
    assert result["Vessels Carrying Flow Fraction"] == pytest.approx(5 / 7)
    assert result["Capillaries Carrying Flow"] == 3
    assert result["Capillaries Carrying Flow Fraction"] == pytest.approx(3 / 5)
    # The CVs are over the carrying vessels alone: unchanged by the two.
    assert result["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    # The two count in the network's volume, not its flowing volume.
    capillary_volume = sum(_volume_um3(d, 100.0) for d in CAPILLARY_DIAMETERS)
    assert result["Flowing Capillary Volume (micron³)"] == pytest.approx(capillary_volume)
    assert result["Flowing Capillary Volume Fraction"] == pytest.approx(
        capillary_volume / (capillary_volume + 2 * _volume_um3(5.0, 100.0))
    )
    assert result["Flowing Capillary Length Fraction"] == pytest.approx(300.0 / 500.0)
    assert math.isnan(G[1][4][0][VELOCITY_EDGE_ATTRIBUTE])
    assert math.isnan(G[7][8][0][VELOCITY_EDGE_ATTRIBUTE])


def test_a_solved_flow_below_the_floor_is_no_flow():
    """The bridge of a balanced Wheatstone bridge: solved, but balanced to
    rounding, which would otherwise read as a vessel at ~0 flow."""
    G = _bed()
    _vessel(G, 1, 2, 1e-24, diameter=5.0)
    floor = carrying_floor(G)
    assert floor == pytest.approx(CARRYING_FLOOR_FRACTION * 6e-12)
    assert not is_carrying(G[1][2][3], floor)
    assert is_carrying(G[1][2][0], floor)

    result = compute_flux_heterogeneity(G)
    assert result["Capillaries Carrying Flow"] == 3
    assert result["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert math.isnan(G[1][2][3][VELOCITY_EDGE_ATTRIBUTE])


def test_a_graph_solved_before_flow_states_is_judged_by_its_flows():
    G = _bed()
    for *_e, data in G.edges(keys=True, data=True):
        del data["flow_state"]
        del data["flow_solved"]
    result = compute_flux_heterogeneity(G)
    assert result["Vessels Carrying Flow"] == 5
    assert result["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))


def test_without_a_solved_flow_it_reports_why_and_writes_nothing():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=10.0, diameter_um=5.0, branch_order="BO2")
    result = compute_flux_heterogeneity(G)
    assert result == {"Flux Heterogeneity Status": "N/A (needs a solved flow (run haemodynamics))"}
    assert VELOCITY_EDGE_ATTRIBUTE not in G[0][1][0]


def test_with_every_flow_zero_no_vessel_carries_flow():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 0.0)
    _vessel(G, 1, 2, 0.0)
    result = compute_flux_heterogeneity(G)
    assert result["Flux Heterogeneity Status"] == "N/A (no vessel carries flow)"
    assert all(VELOCITY_EDGE_ATTRIBUTE not in d for *_e, d in G.edges(keys=True, data=True))


def test_a_stale_velocity_is_cleared_when_the_measure_cannot_run():
    G = _bed()
    compute_flux_heterogeneity(G)
    for *_e, data in G.edges(keys=True, data=True):
        del data["flow_abs"]
    compute_flux_heterogeneity(G)
    assert all(VELOCITY_EDGE_ATTRIBUTE not in d for *_e, d in G.edges(keys=True, data=True))


def test_without_capillaries_the_capillary_rows_say_so():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 2e-12, branch_order="Art1")
    _vessel(G, 1, 2, 2e-12, branch_order="Ven1")
    result = compute_flux_heterogeneity(G)
    assert result["Capillary Flux Heterogeneity Status"].startswith("N/A")
    assert "Capillary Flow CV" not in result
    assert result["Vessels Carrying Flow"] == 2
    assert result["Vessel Flow CV"] == pytest.approx(0.0)


def test_one_carrying_capillary_has_a_mean_but_no_spread():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, 2e-12, branch_order="Art1")
    _vessel(G, 1, 2, 2e-12)
    _vessel(G, 1, 3, 0.0, flow_state="blind_end", flow_solved=False)
    result = compute_flux_heterogeneity(G)
    assert result["Capillaries Carrying Flow"] == 1
    assert result["Capillary Flow Mean (nL/min)"] == pytest.approx(2e-12 * NL_PER_MIN)
    assert result["Capillary Flow CV"] == "N/A (fewer than two capillaries carry flow)"
    assert result["Capillary Velocity CV"] == "N/A (fewer than two capillaries carry flow)"


def test_a_vessel_without_a_diameter_is_left_out_of_velocities_and_refuses_the_volume():
    G = _bed()
    del G[1][2][1]["diameter_um"]
    result = compute_flux_heterogeneity(G)

    assert result["Capillaries Carrying Flow"] == 3
    assert result["Capillaries Carrying Flow Without Diameter"] == 1
    assert result["Vessels Carrying Flow Without Diameter"] == 1
    velocities = [_velocity_um_s(1e-12, 5.0), _velocity_um_s(3e-12, 8.0)]
    assert result["Capillary Velocity CV"] == pytest.approx(_cv(velocities))
    # Flows need no diameter.
    assert result["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert result["Flowing Capillary Volume (micron³)"] == "N/A (1 of 3 vessels have no diameter)"
    assert result["Flowing Vessel Volume Fraction"] == "N/A (1 of 5 vessels have no diameter)"
    assert math.isnan(G[1][2][1][VELOCITY_EDGE_ATTRIBUTE])


def test_the_rows_by_vessel_type_split_the_same_vessels():
    G = _bed()
    _vessel(G, 1, 4, 0.0, diameter=5.0, flow_state="blind_end", flow_solved=False)
    by_type = compute_flux_heterogeneity(G)["Flux Heterogeneity By Vessel Type"]

    assert set(by_type) == {"arteriole", "capillary", "venule"}
    assert by_type["capillary"]["Vessels"] == 4
    assert by_type["capillary"]["Vessels Carrying Flow"] == 3
    assert by_type["capillary"]["Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert by_type["capillary"]["Flowing Vessel Volume (micron³)"] == pytest.approx(
        sum(_volume_um3(d, 100.0) for d in CAPILLARY_DIAMETERS)
    )
    assert by_type["arteriole"]["Vessels Carrying Flow"] == 1
    assert by_type["arteriole"]["Flow CV"] == "N/A (fewer than two vessels carry flow)"


# --- in the report ---------------------------------------------------------------------


def test_it_is_a_network_analysis_measure_with_its_own_section():
    assert "flux_heterogeneity" in STATISTIC_MEASURES
    assert "flux_heterogeneity" in NETWORK_ANALYSIS_MEASURES
    # It reads the flow, so it already sees only what is connected.
    assert "flux_heterogeneity" not in CONNECTED_NETWORK_MEASURES
    sections = compute_vessel_statistics_by_measure(
        _bed(), enabled_measures=frozenset({"flux_heterogeneity"})
    )
    assert list(sections) == ["flux_heterogeneity"]
    assert sections["flux_heterogeneity"]["Capillary Flow CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))


def test_the_schema_toggle_chooses_it_under_network_analysis():
    schema = default_schema()
    setting = schema["statistics_flux_heterogeneity"]
    assert setting.default is True
    assert setting.advanced
    assert set(setting.requires) == {"statistics", "statistics_network_analysis"}

    settings = schema.defaults()
    assert "flux_heterogeneity" in statistics_arguments(settings)["enabled_measures"]
    settings["statistics_flux_heterogeneity"] = False
    assert "flux_heterogeneity" not in statistics_arguments(settings)["enabled_measures"]
    settings["statistics_flux_heterogeneity"] = True
    settings["statistics_network_analysis"] = False
    assert "flux_heterogeneity" not in statistics_arguments(settings)["enabled_measures"]


def test_the_velocity_is_a_vessels_layer_colour_option_that_moves_with_the_flow():
    from haemolynx.gui.results import (
        OPTIONAL_EDGE_COLUMNS,
        SOLVED_FLOW_COLUMNS,
        SWEEP_FLOW_MAGNITUDE_COLUMNS,
    )

    assert OPTIONAL_EDGE_COLUMNS[VELOCITY_EDGE_ATTRIBUTE] == "export_results"
    assert VELOCITY_EDGE_ATTRIBUTE in SWEEP_FLOW_MAGNITUDE_COLUMNS
    assert VELOCITY_EDGE_ATTRIBUTE in SOLVED_FLOW_COLUMNS
