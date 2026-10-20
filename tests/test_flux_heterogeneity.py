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


# --- the red cells ---------------------------------------------------------------------
#
# Predicted from each vessel's flow and discharge haematocrit by
# haemodynamics/red_cells.py (tested on its own in test_red_cells.py): here,
# that the analysis writes and reports them as it does the velocity.

from haemolynx.haemodynamics.red_cells import (  # noqa: E402
    RED_CELL_EDGE_ATTRIBUTES,
    RED_CELL_FLUX_ATTR,
    RED_CELL_LINEAR_DENSITY_ATTR,
    RED_CELL_VELOCITY_ATTR,
    TUBE_HAEMATOCRIT_ATTR,
    fahraeus_ratio,
)

HAEMATOCRIT = 0.45
CELL_VOLUME_FL = 50.0


def _solved_bed(haematocrit=HAEMATOCRIT):
    G = _bed()
    G.graph["haematocrit"] = haematocrit
    return G


def _flux(flow, h, volume_fl=CELL_VOLUME_FL):
    return flow * h / (volume_fl * 1e-18)


def _rbc_velocity_um_s(flow, d, h):
    return _velocity_um_s(flow, d) / fahraeus_ratio(h, d)


def test_each_carrying_vessel_gets_its_red_cells_from_the_runs_haematocrit():
    G = _solved_bed()
    compute_flux_heterogeneity(G, red_cell_volume_fl=CELL_VOLUME_FL)
    for (_u, _v, _k, data), flow, d in zip(
        [e for e in G.edges(keys=True, data=True) if e[3]["branch_order"] == "BO2"],
        CAPILLARY_FLOWS,
        CAPILLARY_DIAMETERS,
    ):
        assert data[RED_CELL_FLUX_ATTR] == pytest.approx(_flux(flow, HAEMATOCRIT))
        assert data[TUBE_HAEMATOCRIT_ATTR] == pytest.approx(
            HAEMATOCRIT * fahraeus_ratio(HAEMATOCRIT, d)
        )
        assert data[RED_CELL_VELOCITY_ATTR] == pytest.approx(_rbc_velocity_um_s(flow, d, HAEMATOCRIT))
        assert data[RED_CELL_VELOCITY_ATTR] > data[VELOCITY_EDGE_ATTRIBUTE]
        assert data[RED_CELL_LINEAR_DENSITY_ATTR] == pytest.approx(
            data[TUBE_HAEMATOCRIT_ATTR] * math.pi * (d / 2) ** 2 * 1000.0 / CELL_VOLUME_FL
        )
        # Cells per mm times mm per second is cells per second.
        assert data[RED_CELL_LINEAR_DENSITY_ATTR] * data[RED_CELL_VELOCITY_ATTR] * 1e-3 == pytest.approx(
            data[RED_CELL_FLUX_ATTR]
        )
    # The 5 um capillary at 1 pL/s, by hand: 1e-12 x 0.45 / 50e-18 = 9,000 cells/s.
    assert G[1][2][0][RED_CELL_FLUX_ATTR] == pytest.approx(9000.0)


def test_with_one_haematocrit_the_flux_cv_is_the_flow_cv_and_the_red_cell_velocity_cv_is_not():
    result = compute_flux_heterogeneity(_solved_bed(), red_cell_volume_fl=CELL_VOLUME_FL)

    assert result["Red Cell Volume (fL)"] == CELL_VOLUME_FL
    assert result["Haematocrit Source"] == "the run's haematocrit (0.45) on every vessel"
    assert result["Capillaries Carrying Flow Without Haematocrit"] == 0
    assert result["Capillary Red Cell Flux CV"] == pytest.approx(result["Capillary Flow CV"])
    assert result["Capillary Red Cell Flux Mean (cells/s)"] == pytest.approx(
        np.mean([_flux(q, HAEMATOCRIT) for q in CAPILLARY_FLOWS])
    )
    assert result["Capillary Red Cell Flux SD (cells/s)"] == pytest.approx(
        np.std([_flux(q, HAEMATOCRIT) for q in CAPILLARY_FLOWS], ddof=1)
    )
    rbc = [_rbc_velocity_um_s(q, d, HAEMATOCRIT) for q, d in zip(CAPILLARY_FLOWS, CAPILLARY_DIAMETERS)]
    assert result["Capillary Red Cell Velocity CV"] == pytest.approx(_cv(rbc))
    assert result["Capillary Red Cell Velocity CV"] != pytest.approx(result["Capillary Velocity CV"])
    assert result["Capillary Red Cell Velocity Mean (mm/s)"] == pytest.approx(np.mean(rbc) * 1e-3)
    assert result["Capillary Tube Haematocrit Mean"] == pytest.approx(
        np.mean([HAEMATOCRIT * fahraeus_ratio(HAEMATOCRIT, d) for d in CAPILLARY_DIAMETERS])
    )
    assert result["Capillary Tube Haematocrit Mean"] < HAEMATOCRIT
    assert result["Capillary Red Cell Linear Density Mean (cells/mm)"] == pytest.approx(
        np.mean(
            [
                HAEMATOCRIT * fahraeus_ratio(HAEMATOCRIT, d) * math.pi * (d / 2) ** 2 * 1000.0 / CELL_VOLUME_FL
                for d in CAPILLARY_DIAMETERS
            ]
        )
    )
    all_flows = [6e-12, *CAPILLARY_FLOWS, 6e-12]
    assert result["Vessel Red Cell Flux CV"] == pytest.approx(_cv(all_flows))
    assert "Fahraeus" in result["Flux Heterogeneity Assumptions"]
    by_type = result["Flux Heterogeneity By Vessel Type"]
    assert by_type["capillary"]["Red Cell Flux CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert by_type["capillary"]["Red Cell Velocity CV"] == pytest.approx(_cv(rbc))
    assert by_type["arteriole"]["Red Cell Flux CV"] == "N/A (fewer than two vessels carry flow)"


def test_a_vessels_own_discharge_haematocrit_comes_before_the_runs():
    G = _solved_bed()
    G[1][2][0]["discharge_haematocrit"] = 0.2
    result = compute_flux_heterogeneity(G, red_cell_volume_fl=CELL_VOLUME_FL)

    assert G[1][2][0][RED_CELL_FLUX_ATTR] == pytest.approx(_flux(1e-12, 0.2))
    assert G[1][2][0][TUBE_HAEMATOCRIT_ATTR] == pytest.approx(0.2 * fahraeus_ratio(0.2, 5.0))
    assert G[1][2][1][RED_CELL_FLUX_ATTR] == pytest.approx(_flux(2e-12, HAEMATOCRIT))
    fluxes = [_flux(1e-12, 0.2), _flux(2e-12, HAEMATOCRIT), _flux(3e-12, HAEMATOCRIT)]
    assert result["Capillary Red Cell Flux CV"] == pytest.approx(_cv(fluxes))
    assert result["Haematocrit Source"] == (
        "the vessel's own discharge_haematocrit on 1 of 5 carrying vessels, "
        "the run's haematocrit (0.45) on the rest"
    )


def test_the_cell_volume_scales_the_flux_and_spacing_and_nothing_else():
    G = _solved_bed()
    compute_flux_heterogeneity(G, red_cell_volume_fl=CELL_VOLUME_FL)
    before = {name: G[1][2][0][name] for name in RED_CELL_EDGE_ATTRIBUTES}
    result = compute_flux_heterogeneity(G, red_cell_volume_fl=CELL_VOLUME_FL / 2)
    after = {name: G[1][2][0][name] for name in RED_CELL_EDGE_ATTRIBUTES}
    assert after[RED_CELL_FLUX_ATTR] == pytest.approx(2 * before[RED_CELL_FLUX_ATTR])
    assert after[RED_CELL_LINEAR_DENSITY_ATTR] == pytest.approx(2 * before[RED_CELL_LINEAR_DENSITY_ATTR])
    assert after[RED_CELL_VELOCITY_ATTR] == pytest.approx(before[RED_CELL_VELOCITY_ATTR])
    assert after[TUBE_HAEMATOCRIT_ATTR] == pytest.approx(before[TUBE_HAEMATOCRIT_ATTR])
    assert result["Red Cell Volume (fL)"] == CELL_VOLUME_FL / 2
    # The default is the module's.
    from haemolynx.haemodynamics.red_cells import DEFAULT_RED_CELL_VOLUME_FL

    assert compute_flux_heterogeneity(G)["Red Cell Volume (fL)"] == DEFAULT_RED_CELL_VOLUME_FL


@pytest.mark.parametrize("volume", [0.0, -1.0, math.nan])
def test_a_cell_volume_that_is_not_positive_is_refused_before_anything_is_written(volume):
    G = _solved_bed()
    with pytest.raises(ValueError, match="red_cell_volume_fl"):
        compute_flux_heterogeneity(G, red_cell_volume_fl=volume)
    assert all(VELOCITY_EDGE_ATTRIBUTE not in d for *_e, d in G.edges(keys=True, data=True))


def test_without_a_haematocrit_the_red_cell_rows_say_so_and_the_velocity_rows_stand():
    G = _bed()  # no G.graph["haematocrit"], no discharge_haematocrit
    result = compute_flux_heterogeneity(G)

    assert result["Capillary Velocity CV"] == pytest.approx(
        _cv([_velocity_um_s(q, d) for q, d in zip(CAPILLARY_FLOWS, CAPILLARY_DIAMETERS)])
    )
    no_haematocrit = (
        "N/A (no haematocrit: no discharge_haematocrit on the vessels and no "
        "haematocrit on the graph)"
    )
    assert result["Capillary Red Cell Flux CV"] == no_haematocrit
    assert result["Capillary Red Cell Flux Mean (cells/s)"] == no_haematocrit
    assert result["Capillary Red Cell Velocity CV"] == no_haematocrit
    assert result["Capillary Tube Haematocrit Mean"] == no_haematocrit
    assert result["Vessel Red Cell Flux CV"] == no_haematocrit
    assert result["Capillaries Carrying Flow Without Haematocrit"] == 3
    assert result["Vessels Carrying Flow Without Haematocrit"] == 5
    assert result["Haematocrit Source"].startswith("none")
    for *_e, data in G.edges(keys=True, data=True):
        assert all(math.isnan(data[name]) for name in RED_CELL_EDGE_ATTRIBUTES)
        assert not math.isnan(data[VELOCITY_EDGE_ATTRIBUTE])


@pytest.mark.parametrize("bad", [1.0, -0.1, "x", None])
def test_a_run_haematocrit_that_is_no_fraction_counts_as_none(bad):
    G = _solved_bed(bad)
    result = compute_flux_heterogeneity(G)
    assert result["Capillaries Carrying Flow Without Haematocrit"] == 3
    assert math.isnan(G[1][2][0][RED_CELL_FLUX_ATTR])


def test_the_flux_needs_no_diameter_but_the_velocity_tube_haematocrit_and_spacing_do():
    G = _solved_bed()
    del G[1][2][1]["diameter_um"]
    result = compute_flux_heterogeneity(G, red_cell_volume_fl=CELL_VOLUME_FL)

    assert G[1][2][1][RED_CELL_FLUX_ATTR] == pytest.approx(_flux(2e-12, HAEMATOCRIT))
    assert math.isnan(G[1][2][1][RED_CELL_VELOCITY_ATTR])
    assert math.isnan(G[1][2][1][TUBE_HAEMATOCRIT_ATTR])
    assert math.isnan(G[1][2][1][RED_CELL_LINEAR_DENSITY_ATTR])
    assert result["Capillary Red Cell Flux CV"] == pytest.approx(_cv(CAPILLARY_FLOWS))
    assert result["Capillary Red Cell Velocity CV"] == pytest.approx(
        _cv([_rbc_velocity_um_s(1e-12, 5.0, HAEMATOCRIT), _rbc_velocity_um_s(3e-12, 8.0, HAEMATOCRIT)])
    )
    assert result["Capillaries Carrying Flow Without Haematocrit"] == 0


def test_a_vessel_carrying_no_flow_has_no_red_cells_and_a_stale_value_is_cleared():
    G = _solved_bed()
    _vessel(G, 1, 4, 0.0, diameter=5.0, flow_state="blind_end", flow_solved=False)
    compute_flux_heterogeneity(G)
    assert all(math.isnan(G[1][4][0][name]) for name in RED_CELL_EDGE_ATTRIBUTES)
    for *_e, data in G.edges(keys=True, data=True):
        del data["flow_abs"]
    compute_flux_heterogeneity(G)
    assert all(
        name not in d for name in RED_CELL_EDGE_ATTRIBUTES for *_e, d in G.edges(keys=True, data=True)
    )


def test_the_run_passes_its_cell_volume_through_to_the_measure():
    schema = default_schema()
    setting = schema["red_cell_volume_fl"]
    assert setting.default == 50.0
    assert setting.unit == "fL"
    assert setting.minimum == 1.0
    assert not setting.advanced
    assert set(setting.requires) == {"run_haemodynamics"}
    assert setting.section == schema["haematocrit"].section
    assert "cells per second" in setting.help

    settings = schema.defaults()
    settings["red_cell_volume_fl"] = 90
    arguments = statistics_arguments(settings)
    assert arguments["red_cell_volume_fl"] == 90.0
    sections = compute_vessel_statistics_by_measure(
        _solved_bed(), enabled_measures=frozenset({"flux_heterogeneity"}), red_cell_volume_fl=90.0
    )
    rows = sections["flux_heterogeneity"]
    assert rows["Red Cell Volume (fL)"] == 90.0
    assert rows["Capillary Red Cell Flux Mean (cells/s)"] == pytest.approx(
        np.mean([_flux(q, HAEMATOCRIT, 90.0) for q in CAPILLARY_FLOWS])
    )


def test_the_red_cells_are_vessels_layer_colour_options_that_blank_on_unsolved_vessels():
    from haemolynx.gui.results import (
        OPTIONAL_EDGE_COLUMNS,
        SOLVED_FLOW_COLUMNS,
        SWEEP_FLOW_MAGNITUDE_COLUMNS,
    )

    for name in RED_CELL_EDGE_ATTRIBUTES:
        assert OPTIONAL_EDGE_COLUMNS[name] == "export_results"
        assert name in SOLVED_FLOW_COLUMNS
    # A pressure sweep moves the flow, so the flux and the cells' velocity;
    # the tube haematocrit and spacing read the haematocrit and diameter alone.
    assert {RED_CELL_VELOCITY_ATTR, RED_CELL_FLUX_ATTR} <= SWEEP_FLOW_MAGNITUDE_COLUMNS
    assert not {TUBE_HAEMATOCRIT_ATTR, RED_CELL_LINEAR_DENSITY_ATTR} & SWEEP_FLOW_MAGNITUDE_COLUMNS


def test_the_csv_notes_call_the_red_cell_rows_predictions(tmp_path):
    """The CSV's keyword notes used to read "Linear Density" as a vessel
    density and "Red Cell Velocity" as the plasma velocity."""
    import csv

    from haemolynx.statistics import export_statistics_to_csv

    result = compute_flux_heterogeneity(_solved_bed())
    path = export_statistics_to_csv(result, tmp_path / "stats.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        notes = {row[1]: (row[3], row[4]) for row in csv.reader(handle)}
    assert notes["Capillary Red Cell Flux Mean"] == (
        "cells/s",
        "Flow x discharge haematocrit / red-cell volume, over vessels carrying flow: a prediction, not a count.",
    )
    assert notes["Capillary Red Cell Linear Density Mean"][0] == "cells/mm"
    assert "cells per mm" in notes["Capillary Red Cell Linear Density Mean"][1]
    assert "normalized" not in notes["Capillary Red Cell Linear Density Mean"][1]
    assert "Fahraeus" in notes["Capillary Red Cell Velocity CV"][1]
    assert "Fahraeus" in notes["Capillary Tube Haematocrit Mean"][1]
    # The plasma velocity keeps its own note.
    assert notes["Capillary Velocity CV"][1] == "Flow over the lumen's cross-section, over vessels carrying flow."
