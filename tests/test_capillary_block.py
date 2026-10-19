"""The "capillary blocks" perturbation: which vessels it blocks, what blocking
does to their resistance, and how the re-solved network is compared with the
baseline -- on networks small enough to check by hand."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics import (
    PERTURBATION_TYPES,
    block_vessels,
    compare_block_to_baseline,
    resolve_blocked_vessels,
    settings_for_perturbation_type,
    stopped_vessels,
    zero_stopped_vessel_flows,
)
from haemolynx.haemodynamics.capillary_block import (
    BLOCK_EDGE_ATTRIBUTE,
    FLOW_CHANGE_EDGE_ATTRIBUTE,
)
from haemolynx.haemodynamics.poiseuille import set_edge_resistance
from haemolynx.haemodynamics.resistance import flow_abs_log10_value
from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, default_schema, run_perturbations
from haemolynx.pipeline.stages import _solve_network

SCHEMA = default_schema()


def _vessel(G, u, v, branch_order="BO2", resistance=1.0, length=10.0, diameter=5.0):
    G.add_edge(u, v, branch_order=branch_order, length=length, diameter_um=diameter)
    key = max(G[u][v])
    set_edge_resistance(G[u][v][key], resistance)
    return key


def _solve(G, inlets, outlets):
    settings = {"inlet_p_bc": 100.0, "outlet_p_bc": 0.0, "haematocrit_model": "constant"}
    boundaries = BoundaryNodes(inlet_nodes=list(inlets), outlet_nodes=list(outlets))
    return _solve_network(G, settings, boundaries)


# --- choosing the vessels ------------------------------------------------------------


def _many_capillaries(n=200):
    G = nx.MultiGraph()
    for i in range(n):
        _vessel(G, i, i + 1, branch_order="BO2" if i % 2 else "BO3")
    _vessel(G, n, n + 1, branch_order="Art1")
    return G


def test_the_fraction_blocked_is_exact_and_only_from_the_chosen_orders():
    G = _many_capillaries()
    blocked, summary = resolve_blocked_vessels(
        G, selection="branch_order_probability", branch_orders=["BO2"], probability=0.1, seed=1
    )
    assert summary["candidate_vessels"] == 100
    assert len(blocked) == summary["blocked_vessels"] == 10
    assert all(G[u][v][k]["branch_order"] == "BO2" for u, v, k in blocked)


@pytest.mark.parametrize("spelling", ["BO2", "bo2", "B02", "BO 2", 2, "2"])
def test_branch_orders_match_however_they_are_spelt(spelling):
    G = _many_capillaries()
    blocked, _ = resolve_blocked_vessels(
        G, selection="branch_order_probability", branch_orders=[spelling], probability=1.0
    )
    assert len(blocked) == 100


def test_the_same_seed_blocks_the_same_vessels_and_another_seed_does_not():
    G = _many_capillaries()
    args = dict(selection="branch_order_probability", branch_orders=["BO2", "BO3"], probability=0.2)
    first, _ = resolve_blocked_vessels(G, seed=7, **args)
    again, _ = resolve_blocked_vessels(G, seed=7, **args)
    other, _ = resolve_blocked_vessels(G, seed=8, **args)
    assert first == again
    assert first != other
    assert len(first) == len(other) == 40


@pytest.mark.parametrize("probability, expected", [(0.0, 0), (1.0, 100), (0.005, 1), (0.004, 0)])
def test_the_count_rounds_half_up(probability, expected):
    G = _many_capillaries()
    blocked, _ = resolve_blocked_vessels(
        G, selection="branch_order_probability", branch_orders=["BO2"], probability=probability
    )
    assert len(blocked) == expected


def test_vessel_ids_are_the_branch_ids_the_viewer_shows():
    """branchID is the edge's position in G.edges(keys=True) -- the same
    enumeration gui.branch_hover numbers the vessels by."""
    G = _many_capillaries(10)
    edges = list(G.edges(keys=True))
    blocked, summary = resolve_blocked_vessels(G, selection="vessel_ids", vessel_ids=[3, "7", 3])
    assert blocked == [edges[3], edges[7]]
    assert summary["vessel_ids"] == [3, 7]


def test_a_vessel_can_also_be_named_by_its_ends_either_way_round():
    G = nx.MultiGraph()
    _vessel(G, (0, 0), (0, 1))
    _vessel(G, (0, 1), (1, 1))
    blocked, _ = resolve_blocked_vessels(
        G, selection="vessel_ids", vessel_ids=[[[1, 1], [0, 1], 0]]  # as read back from YAML
    )
    assert blocked == [((0, 1), (1, 1), 0)]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(selection="vessel_ids", vessel_ids=[99]), "not in this network"),
        (dict(selection="vessel_ids", vessel_ids=[[5, 7, 0]]), "not in this network"),
        (dict(selection="vessel_ids", vessel_ids=[]), "is empty"),
        (dict(selection="vessel_ids", vessel_ids=["BO2"]), "neither a branchID"),
        (dict(selection="branch_order_probability", branch_orders=[]), "is empty"),
        (dict(selection="branch_order_probability", branch_orders=["BO2"], probability=1.5), "between 0 and 1"),
        (dict(selection="somehow"), "must be one of"),
    ],
)
def test_a_selection_that_would_block_the_wrong_vessels_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_blocked_vessels(_many_capillaries(10), **kwargs)


# --- blocking -------------------------------------------------------------------------


def test_blocking_multiplies_resistance_and_marks_every_vessel():
    G = _many_capillaries(4)
    baseline = G.copy()
    edges = list(G.edges(keys=True))

    block_vessels(G, [edges[1]], 1e6)

    for u, v, k, data in G.edges(keys=True, data=True):
        was = baseline[u][v][k]["resistance"]
        if (u, v, k) == edges[1]:
            assert data[BLOCK_EDGE_ATTRIBUTE] == "blocked"
            assert data["resistance"] == pytest.approx(was * 1e6)
            assert data["conductance"] == pytest.approx(1.0 / (was * 1e6))
        else:
            assert data[BLOCK_EDGE_ATTRIBUTE] == "open"
            assert data["resistance"] == was
    # A shallow copy's edge dicts are its own: the baseline is untouched.
    assert baseline[edges[1][0]][edges[1][1]][edges[1][2]]["resistance"] == 1.0


def test_a_factor_that_does_not_block_is_refused():
    with pytest.raises(ValueError, match="greater than 1"):
        block_vessels(_many_capillaries(2), [], 1.0)


# --- comparing with the baseline ---------------------------------------------------------


def _two_routes():
    """Inlet 0 -> 1 (Art1), then two capillary routes to 3: 1-2-3 and 1-4-3,
    then 3 -> 5 (Ven1), outlet 5. Every vessel resistance 1."""
    G = nx.MultiGraph()
    _vessel(G, 0, 1, "Art1")
    _vessel(G, 1, 2, "BO2")
    _vessel(G, 2, 3, "BO2")
    _vessel(G, 1, 4, "BO2")
    _vessel(G, 4, 3, "BO2")
    _vessel(G, 3, 5, "Ven1")
    return G


def test_blocking_one_route_starves_the_vessel_downstream_of_it():
    baseline = _two_routes()
    _solve(baseline, [0], [5])
    blocked = baseline.copy()
    block_vessels(blocked, [(1, 2, 0)], 1e6)
    _solve(blocked, [0], [5])

    summary, by_order = compare_block_to_baseline(
        baseline, blocked, inlet_nodes=[0], outlet_nodes=[5]
    )

    # Series resistance 1 + 1 + 1 (two parallel routes of 2) = 3 -> 1 + 2 + 1 = 4
    # (to within the 1e-6 the block still lets through).
    assert summary["total_inflow_percent_change"] == pytest.approx(-25.0, abs=1e-3)
    # 2-3 is the one other vessel that lost its flow; the blocked vessel
    # itself is not counted, and the other route carries more, not less.
    assert summary["hypoperfused_vessels"] == 1
    assert summary["hypoperfused_length_um"] == pytest.approx(10.0)
    assert summary["reversed_vessels"] == 0
    assert summary["blocked_vessel_length_um"] == pytest.approx(10.0)
    assert summary["outlet_flow_retained_mean"] == pytest.approx(0.75, abs=1e-5)
    assert summary["outlets_losing_half_their_flow"] == 0
    rows = {row["branch_order"]: row for row in by_order}
    assert rows["BO2"]["vessels"] == 4 and rows["BO2"]["blocked"] == 1
    assert rows["BO2"]["hypoperfused"] == 1
    assert list(rows) == ["Art1", "BO2", "Ven1"]
    assert blocked[1][4][0][FLOW_CHANGE_EDGE_ATTRIBUTE] == pytest.approx(1.5, abs=1e-5)
    assert blocked[2][3][0][FLOW_CHANGE_EDGE_ATTRIBUTE] == pytest.approx(0.0, abs=1e-5)


def test_the_bridge_of_a_wheatstone_bridge_reverses_when_one_arm_is_blocked():
    G = nx.MultiGraph()
    _vessel(G, 0, 1, resistance=0.1)   # low: node 1 sits near inlet pressure
    _vessel(G, 0, 2, resistance=10.0)  # high: node 2 sits near outlet pressure
    _vessel(G, 1, 3, resistance=10.0)
    _vessel(G, 2, 3, resistance=0.1)
    _vessel(G, 1, 2, resistance=1.0)   # the bridge: flows 1 -> 2
    _solve(G, [0], [3])
    blocked = G.copy()
    block_vessels(blocked, [(0, 1, 0)], 1e6)  # now 1 is fed only through the bridge
    _solve(blocked, [0], [3])

    summary, _ = compare_block_to_baseline(G, blocked, inlet_nodes=[0], outlet_nodes=[3])

    assert summary["reversed_vessels"] == 1
    assert np.sign(G[1][2][0]["flow_signed"]) != np.sign(blocked[1][2][0]["flow_signed"])


def test_transit_times_are_compared_when_the_network_has_diameters():
    baseline = _two_routes()
    _solve(baseline, [0], [5])
    blocked = baseline.copy()
    block_vessels(blocked, [(1, 2, 0)], 1e6)
    _solve(blocked, [0], [5])

    summary, _ = compare_block_to_baseline(baseline, blocked, inlet_nodes=[0], outlet_nodes=[5])

    # Every vessel holds the same volume V. Baseline: Q = 100/3 through the
    # arteriole and venule, 50/3 down each capillary route, so the mean is
    # V/Q + 2V/(Q/2) + V/Q = 6V/Q = 0.18 V. Blocked: all 25 units of flow take
    # the open route, 4V/25 = 0.16 V -- the blocked route's residual leak is
    # not perfusion, so its near-infinite transit time must not count.
    assert summary["mean_transit_time_s_blocked"] / summary[
        "mean_transit_time_s_baseline"
    ] == pytest.approx(0.16 / 0.18, rel=1e-4)
    # One perfused route left, so every red cell takes the same time.
    assert summary["transit_time_sd_s_blocked"] <= 1e-3 * summary["mean_transit_time_s_blocked"]
    assert "capillary_transit_time_heterogeneity_s_blocked" in summary
    # A comparison writes nothing onto either network but the flow change.
    assert all("transit_time_s" not in d for *_e, d in baseline.edges(keys=True, data=True))
    assert all("transit_time_s" not in d for *_e, d in blocked.edges(keys=True, data=True))


def test_flux_heterogeneity_and_the_flowing_volume_are_compared_before_and_after():
    baseline = _two_routes()
    _solve(baseline, [0], [5])
    blocked = baseline.copy()
    block_vessels(blocked, [(1, 2, 0)], 1e6)
    _solve(blocked, [0], [5])

    summary, _ = compare_block_to_baseline(baseline, blocked, inlet_nodes=[0], outlet_nodes=[5])

    # Baseline: four capillaries of one width each carry half the inflow, so
    # there is no heterogeneity among them; the arteriole and venule carry
    # twice as much, so there is some over every vessel.
    assert summary["capillaries_carrying_flow_baseline"] == 4
    assert summary["vessels_carrying_flow_baseline"] == 6
    assert summary["capillary_flow_cv_baseline"] == pytest.approx(0.0, abs=1e-9)
    assert summary["capillary_velocity_cv_baseline"] == pytest.approx(0.0, abs=1e-9)
    flows = [25.0, 12.5, 12.5, 12.5, 12.5, 25.0]
    assert summary["vessel_flow_cv_baseline"] == pytest.approx(
        np.std(flows, ddof=1) / np.mean(flows), rel=1e-6
    )
    # Blocked: 1-2 is blocked, and 2-3 is fed only through it, so it carries
    # the block's leak and nothing else -- neither is flowing. The two open
    # capillaries carry the whole inflow, as the arteriole and venule do.
    assert summary["capillaries_carrying_flow_blocked"] == 2
    assert summary["vessels_carrying_flow_blocked"] == 4
    assert summary["capillary_flow_cv_blocked"] == pytest.approx(0.0, abs=1e-9)
    # The arteriole and venule carry the block's leak on top: a CV of ~1e-6.
    assert summary["vessel_flow_cv_blocked"] == pytest.approx(0.0, abs=1e-5)
    volume = math.pi * 2.5**2 * 10.0  # one 5 um wide, 10 um long vessel
    assert summary["flowing_capillary_volume_um3_baseline"] == pytest.approx(4 * volume)
    assert summary["flowing_capillary_volume_um3_blocked"] == pytest.approx(2 * volume)
    assert summary["flowing_vessel_volume_um3_baseline"] == pytest.approx(6 * volume)
    assert summary["flowing_vessel_volume_um3_blocked"] == pytest.approx(4 * volume)
    assert summary["flowing_capillary_length_um_blocked"] == pytest.approx(20.0)
    assert summary["flowing_vessel_length_um_blocked"] == pytest.approx(40.0)
    # A comparison writes nothing onto either network but the flow change.
    for G in (baseline, blocked):
        assert all("vessel_velocity_um_s" not in d for *_e, d in G.edges(keys=True, data=True))
        assert all(d["flow_abs"] > 0 for *_e, d in G.edges(keys=True, data=True))


def test_a_vessel_fed_only_through_a_block_counts_as_stopped_in_the_comparison():
    """Its leak flow is above the carrying floor (a millionth of its baseline
    flow, against a billionth of the largest), so only the rule that it lies
    on no inlet-to-outlet path once the block is taken out stops it."""
    baseline = _two_routes()
    _solve(baseline, [0], [5])
    blocked = baseline.copy()
    block_vessels(blocked, [(1, 2, 0)], 1e6)
    _solve(blocked, [0], [5])
    leak = blocked[2][3][0]["flow_abs"]
    assert 0 < leak < 1e-3 * baseline[2][3][0]["flow_abs"]
    assert leak > 1e-9 * blocked[0][1][0]["flow_abs"]

    stopped = {(u, v, k) for u, v, k, _d in stopped_vessels(blocked, [0], [5])}
    assert stopped == {(1, 2, 0), (2, 3, 0)}
    # Without inlets and outlets only the blocked vessel itself is stopped.
    assert {(u, v, k) for u, v, k, _d in stopped_vessels(blocked, [], [])} == {(1, 2, 0)}

    summary, _ = compare_block_to_baseline(baseline, blocked, inlet_nodes=[0], outlet_nodes=[5])

    # The comparison counts them: one blocked vessel, two stopped.
    assert summary["blocked_vessel_length_um"] == pytest.approx(10.0)
    assert summary["stopped_vessels"] == 2
    assert summary["stopped_vessel_length_um"] == pytest.approx(20.0)


def test_zeroing_the_stopped_vessels_keeps_every_pressure():
    baseline = _two_routes()
    _solve(baseline, [0], [5])
    blocked = baseline.copy()
    block_vessels(blocked, [(1, 2, 0)], 1e6)
    _solve(blocked, [0], [5])
    before = {(u, v, k): dict(d) for u, v, k, d in blocked.edges(keys=True, data=True)}
    assert all(d["flow_abs"] > 0 for d in before.values())

    stopped = zero_stopped_vessel_flows(blocked, [0], [5])

    assert stopped == [(1, 2, 0), (2, 3, 0)]
    for edge in stopped:
        data = blocked.edges[edge]
        assert data["flow_abs"] == 0.0 and data["flow_signed"] == 0.0
        assert data["flow_abs_log10"] == flow_abs_log10_value(0.0)
        # The pressures are the solve's, untouched.
        for name in ("pressure_drop", "pressure_u", "pressure_v"):
            assert data[name] == before[edge][name]
    # The block takes (nearly) the whole 50 Pa between the two routes' ends;
    # the vessel beyond it has the leak's drop, tiny but real.
    assert blocked[1][2][0]["pressure_drop"] == pytest.approx(50.0, abs=1e-3)
    assert 0 < blocked[2][3][0]["pressure_drop"] < 1e-3
    # Every other vessel is exactly as the solve left it.
    for edge, was in before.items():
        if edge not in stopped:
            assert blocked.edges[edge] == was
    # Without inlets and outlets only the blocked vessel itself is zeroed.
    again = baseline.copy()
    block_vessels(again, [(1, 2, 0)], 1e6)
    _solve(again, [0], [5])
    assert zero_stopped_vessel_flows(again, [], []) == [(1, 2, 0)]
    assert again[2][3][0]["flow_abs"] > 0


def test_a_vessel_never_solved_is_left_alone_by_the_zeroing():
    G = _two_routes()
    block_vessels(G, [(1, 2, 0)], 1e6)

    assert zero_stopped_vessel_flows(G, [0], [5]) == [(1, 2, 0), (2, 3, 0)]
    assert all("flow_abs" not in d for *_e, d in G.edges(keys=True, data=True))


# --- the perturbation, end to end ---------------------------------------------------------


def _model():
    G = _two_routes()
    for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        data["voxels"] = [[0.0, 0.0, float(i)], [0.0, 1.0, float(i)]]
    for node in G.nodes:
        G.nodes[node]["pos"] = np.asarray([0.0, 0.0, float(node)])
    return HaemodynamicModel(graph=G)


def _settings(tmp_path: Path, entries, **extra):
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "run_haemodynamics": True,
            "run_perturbations": True,
            "perturbations": entries,
            "diameter_by_branch_order": {"Art1": 10.0, "BO2": 5.0, "Ven1": 10.0},
            "measure_diameters_from_raw_image": False,
            "viscosity_law": "constant",
            "inlet_p_bc": 100.0,
            "outlet_p_bc": 0.0,
        }
    )
    values.update(extra)
    return values


def _boundaries():
    return BoundaryNodes(inlet_nodes=[0], outlet_nodes=[5], resistance_node_pair=(0, 5))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _statistics(path: Path) -> dict[tuple[str, str], str]:
    """A statistics CSV as ``{(section, metric): value}``."""
    return {(row["Section"], row["Metric"]): row["Value"] for row in _read(path)}


def test_capillary_block_is_a_perturbation_type_with_its_own_options():
    assert "capillary_block" in PERTURBATION_TYPES
    options = settings_for_perturbation_type("capillary_block")
    assert "capillary_block_probability" in options
    assert "capillary_block_vessel_ids" in options
    assert all(name in SCHEMA for name in options)


def test_the_perturbation_blocks_listed_vessels_and_writes_the_comparison(tmp_path):
    model = _model()
    before = {(u, v, k): d["resistance"] for u, v, k, d in model.graph.edges(keys=True, data=True)}
    entry = {
        "name": "stall",
        "type": "capillary_block",
        "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [1]},
    }

    run = run_perturbations(_settings(tmp_path, [entry]), model, _boundaries(), SCHEMA)

    result = run.results[0]
    assert result.error is None, result.error
    written = {path.name for path in result.outputs}
    assert {"stall_blocked_vessels.csv", "stall_block_comparison.csv",
            "stall_flow_by_branch_order.csv", "stall_summary.csv"} <= written

    (vessel,) = _read(result.output_dir / "stall_blocked_vessels.csv")
    assert (vessel["branch_id"], vessel["u"], vessel["v"]) == ("1", "1", "2")
    assert float(vessel["blocked_resistance"]) == pytest.approx(1e6 * float(vessel["baseline_resistance"]))

    comparison = {row["metric"]: row["value"] for row in _read(result.output_dir / "stall_block_comparison.csv")}
    assert comparison["selection"] == "vessel_ids"
    assert comparison["blocked_vessels"] == "1"
    assert float(comparison["total_inflow_percent_change"]) < 0
    assert int(comparison["hypoperfused_vessels"]) == 1

    (summary,) = _read(result.output_dir / "stall_summary.csv")
    assert float(summary["equivalent_resistance"]) > float(summary["baseline_equivalent_resistance"])

    # The perturbation's own network carries the colour columns; the baseline
    # the run goes on to export is exactly as it arrived.
    assert result.graph[1][2][0][BLOCK_EDGE_ATTRIBUTE] == "blocked"
    assert math.isfinite(result.graph[1][4][0][FLOW_CHANGE_EDGE_ATTRIBUTE])
    assert {(u, v, k): d["resistance"] for u, v, k, d in model.graph.edges(keys=True, data=True)} == before
    assert all(BLOCK_EDGE_ATTRIBUTE not in d for *_e, d in model.graph.edges(keys=True, data=True))


def test_the_perturbations_own_network_carries_no_flow_on_its_stopped_vessels(tmp_path):
    """Regression: the perturbation's network kept the solve's leak flows (a
    millionth of the vessel's at the default factor, above the carrying
    floor), so its own statistics CSV counted the blocked vessel and the one
    fed only through it as carrying flow, while the comparison -- zeroing
    them on its copies -- did not. Now the network itself carries the zeros,
    and everything read from it agrees with the comparison."""
    entry = {
        "name": "stall",
        "type": "capillary_block",
        "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [1]},
    }

    run = run_perturbations(
        _settings(tmp_path, [entry], statistics=True, inlet_nodes=[0], outlet_nodes=[5]),
        _model(),
        _boundaries(),
        SCHEMA,
    )

    result = run.results[0]
    assert result.error is None, result.error
    G = result.graph
    blocked, starved, open_twin = G[1][2][0], G[2][3][0], G[1][4][0]
    for data in (blocked, starved):
        assert data["flow_abs"] == 0.0 and data["flow_signed"] == 0.0
        assert data["flow_abs_log10"] == flow_abs_log10_value(0.0)
        assert data[FLOW_CHANGE_EDGE_ATTRIBUTE] == 0.0
    # The pressures stay: the block takes (nearly) the whole 50 Pa between
    # the two routes' ends, and the starved vessel has the leak's drop.
    assert blocked["pressure_drop"] == pytest.approx(50.0, abs=1e-3)
    assert 0 < starved["pressure_drop"] < 1e-3
    assert open_twin["flow_abs"] == pytest.approx(25.0, abs=1e-3)
    assert open_twin[FLOW_CHANGE_EDGE_ATTRIBUTE] == pytest.approx(1.5, abs=1e-5)

    # The edges CSV and the blocked-vessels CSV write the zeros, with the drops.
    edges = {(row["u"], row["v"]): row for row in _read(result.output_dir / "stall_edges.csv")}
    assert float(edges[("1", "2")]["flow_signed"]) == 0.0
    assert float(edges[("2", "3")]["flow_signed"]) == 0.0
    assert float(edges[("1", "2")]["pressure_drop"]) == pytest.approx(50.0, abs=1e-3)
    (vessel,) = _read(result.output_dir / "stall_blocked_vessels.csv")
    assert float(vessel["blocked_flow_abs"]) == 0.0
    assert float(vessel["baseline_flow_abs"]) > 0

    # The perturbation's own statistics count four vessels (two capillaries)
    # carrying flow, as the comparison does; so do the vessel classes and
    # the spacing between flowing vessels; its transit time is the
    # comparison's blocked one.
    stats = _statistics(result.output_dir / "stall_statistics.csv")
    comparison = result.summary["comparison"]
    assert comparison["stopped_vessels"] == 2
    assert comparison["stopped_vessel_length_um"] == pytest.approx(20.0)
    assert comparison["vessels_carrying_flow_blocked"] == 4
    assert comparison["capillaries_carrying_flow_blocked"] == 2
    assert stats[("Core Statistics", "Vessels Carrying Flow")] == "4"
    assert stats[("Core Statistics", "Capillaries Carrying Flow")] == "2"
    assert stats[("Core Statistics", "Flowing Capillary Length")] == "20"
    assert stats[("Vessel Classes > Capillaries", "Vessels Carrying Flow")] == "2"
    assert stats[("Vessel Classes > Capillaries", "Vessel Count")] == "4"
    assert stats[("Core Statistics", "Intercapillary Distance Between Flowing Vessels Sample Count")] == "4"
    assert float(stats[("Core Statistics", "Mean Transit Time")]) == pytest.approx(
        comparison["mean_transit_time_s_blocked"], rel=1e-5
    )
    assert float(stats[("Core Statistics", "Capillary Flow CV")]) == pytest.approx(
        comparison["capillary_flow_cv_blocked"], abs=1e-9
    )
    # The baseline the run goes on to export still has every flow it had.
    assert all("flow_abs" not in d for *_e, d in _model().graph.edges(keys=True, data=True))


def test_the_perturbation_blocks_a_fraction_of_a_branch_order(tmp_path):
    entry = {
        "name": "tenth",
        "type": "capillary_block",
        "overrides": {"capillary_block_branch_orders": "BO2", "capillary_block_probability": 0.5},
    }

    run = run_perturbations(_settings(tmp_path, [entry]), _model(), _boundaries(), SCHEMA)

    result = run.results[0]
    assert result.error is None, result.error
    blocked = [d for *_e, d in result.graph.edges(keys=True, data=True) if d[BLOCK_EDGE_ATTRIBUTE] == "blocked"]
    assert len(blocked) == 2  # half of the four BO2 capillaries
    assert all(d["branch_order"] == "BO2" for d in blocked)
    assert result.summary["candidate_vessels"] == 4


def test_a_bad_vessel_id_fails_that_perturbation_and_not_the_others(tmp_path):
    entries = [
        {"name": "bad", "type": "capillary_block",
         "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [99]}},
        {"name": "good", "type": "capillary_block",
         "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [0]}},
    ]

    run = run_perturbations(_settings(tmp_path, entries), _model(), _boundaries(), SCHEMA)

    bad, good = run.results
    assert "not in this network" in bad.error
    assert good.error is None


def test_replicate_seeds_start_from_the_runs_own_seed():
    from haemolynx.haemodynamics import replicate_seeds

    assert replicate_seeds(7, 3) == [7, 8, 9]
    assert replicate_seeds(7, 1) == [7]
    fresh = replicate_seeds(None, 3)
    assert fresh[0] is None and all(isinstance(seed, int) for seed in fresh[1:])


def test_replicate_statistics_are_over_the_numeric_metrics_of_every_draw():
    from haemolynx.haemodynamics import summarise_block_replicates

    draws = [
        {"total_inflow_percent_change": -2.0, "reversed_vessels": 1, "status": "ok", "flag": True},
        {"total_inflow_percent_change": -4.0, "reversed_vessels": 3, "status": "ok", "flag": False},
        {"total_inflow_percent_change": -6.0, "reversed_vessels": math.nan},
    ]

    spread = summarise_block_replicates(draws)

    assert set(spread) == {"total_inflow_percent_change", "reversed_vessels"}
    inflow = spread["total_inflow_percent_change"]
    assert inflow["mean"] == pytest.approx(-4.0)
    assert inflow["sd"] == pytest.approx(2.0)
    assert inflow["low"] == pytest.approx(np.percentile([-2.0, -4.0, -6.0], 2.5))
    assert inflow["high"] == pytest.approx(np.percentile([-2.0, -4.0, -6.0], 97.5))
    assert inflow["draws"] == 3
    assert spread["reversed_vessels"]["draws"] == 2  # the NaN draw gave it no value


def test_replicates_report_the_spread_over_several_draws_of_a_block(tmp_path):
    """Regression: a random block was one draw per seed, so its result was a
    single outcome with no spread; replicates draw it again from the seeds
    after the run's own and report every metric's mean and range."""
    entry = {
        "name": "tenth",
        "type": "capillary_block",
        "overrides": {
            "capillary_block_branch_orders": "BO2",
            "capillary_block_probability": 0.5,
            "capillary_block_replicates": 4,
        },
    }

    run = run_perturbations(
        _settings(tmp_path, [entry], capillary_block_seed=11), _model(), _boundaries(), SCHEMA
    )

    result = run.results[0]
    assert result.error is None, result.error
    draws = _read(result.output_dir / "tenth_block_replicates.csv")
    assert [row["seed"] for row in draws] == ["11", "12", "13", "14"]
    assert all(row["blocked_vessels"] == "2" for row in draws)
    changes = [float(row["total_inflow_percent_change"]) for row in draws]
    comparison = {
        row["metric"]: row["value"]
        for row in _read(result.output_dir / "tenth_block_comparison.csv")
    }
    # The first draw is the block the perturbation's own network carries.
    assert float(comparison["total_inflow_percent_change"]) == pytest.approx(changes[0])
    assert comparison["draws"] == "4"
    assert float(comparison["total_inflow_percent_change_draws_mean"]) == pytest.approx(
        np.mean(changes)
    )
    assert float(comparison["total_inflow_percent_change_draws_low"]) == pytest.approx(
        np.percentile(changes, 2.5)
    )
    assert result.summary["comparison"]["draws"] == 4


def test_every_draw_reaches_the_comparison_with_its_stopped_vessels_at_zero(tmp_path, monkeypatch):
    """The first draw's network and each replicate's are zeroed the same way
    before they are compared with the baseline."""
    import haemolynx.haemodynamics as haemodynamics_package

    original = haemodynamics_package.compare_block_to_baseline
    seen: list[tuple[int, int]] = []

    def recording(baseline, blocked, **kwargs):
        stopped = stopped_vessels(blocked, kwargs["inlet_nodes"], kwargs["outlet_nodes"])
        at_zero = sum(1 for *_e, d in stopped if d["flow_abs"] == 0.0 and d["flow_signed"] == 0.0)
        seen.append((len(stopped), at_zero))
        assert all(d["pressure_drop"] != 0.0 for *_e, d in stopped)
        return original(baseline, blocked, **kwargs)

    monkeypatch.setattr(haemodynamics_package, "compare_block_to_baseline", recording)
    entry = {
        "name": "tenth",
        "type": "capillary_block",
        "overrides": {
            "capillary_block_branch_orders": "BO2",
            "capillary_block_probability": 0.5,
            "capillary_block_replicates": 3,
        },
    }

    run = run_perturbations(
        _settings(tmp_path, [entry], capillary_block_seed=11), _model(), _boundaries(), SCHEMA
    )

    assert run.results[0].error is None, run.results[0].error
    # The first draw, then two replicates; two of the four capillaries are
    # blocked each time, so at least the two of them are stopped.
    assert len(seen) == 3
    assert all(stopped >= 2 and at_zero == stopped for stopped, at_zero in seen)


def test_one_draw_writes_no_replicates(tmp_path):
    entry = {
        "name": "once",
        "type": "capillary_block",
        "overrides": {"capillary_block_branch_orders": "BO2", "capillary_block_probability": 0.5},
    }

    run = run_perturbations(_settings(tmp_path, [entry]), _model(), _boundaries(), SCHEMA)

    result = run.results[0]
    assert result.error is None, result.error
    assert not (result.output_dir / "once_block_replicates.csv").exists()
    assert "draws" not in result.summary["comparison"]


def test_blocks_survive_the_haematocrit_iteration(tmp_path):
    """Distributed haematocrit recomputes every resistance each pass: the
    blocks must be re-applied on top, or they would silently vanish."""
    entry = {
        "name": "stall",
        "type": "capillary_block",
        "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [1]},
    }
    model = _model()

    run = run_perturbations(
        _settings(tmp_path, [entry], haematocrit_model="distributed_iterative"),
        model,
        _boundaries(),
        SCHEMA,
    )

    result = run.results[0]
    assert result.error is None, result.error
    blocked = result.graph[1][2][0]["resistance"]
    open_twin = result.graph[1][4][0]["resistance"]
    assert blocked > 1e5 * open_twin


# --- the panel -----------------------------------------------------------------------------


def test_the_panel_offers_the_type_and_its_rows():
    from haemolynx.gui.perturbation_editing import (
        display_label_for_setting,
        display_name_for_type,
        rows_for_type,
    )

    assert display_name_for_type("capillary_block") == "capillary blocks"
    rows = rows_for_type("capillary_block")
    assert rows[0] == "capillary_block_selection"
    assert display_label_for_setting("capillary_block_vessel_ids") == "Vessel IDs to block (branchID)"


def test_the_perturbation_layer_can_be_coloured_by_block_and_flow_change(tmp_path):
    from haemolynx.gui.results import (
        OPTIONAL_EDGE_COLUMNS,
        TEXT_COLUMNS,
        ResultLayers,
        _colouring,
        perturbation_layer_names,
    )
    from haemolynx.pipeline import PerturbationRun

    entry = {
        "name": "stall",
        "type": "capillary_block",
        "overrides": {"capillary_block_selection": "vessel_ids", "capillary_block_vessel_ids": [1]},
    }
    run = run_perturbations(_settings(tmp_path, [entry]), _model(), _boundaries(), SCHEMA)
    assert OPTIONAL_EDGE_COLUMNS["capillary_block"] == "run_perturbations"
    assert "capillary_block" in TEXT_COLUMNS

    group = ResultLayers().stage_finished(
        "run_perturbations", PerturbationRun(output_dir=tmp_path, results=run.results)
    )
    vessels = next(s for s in group.layers if s.name == perturbation_layer_names("stall")[0])
    assert "blocked" in set(vessels.features["capillary_block"])
    assert _colouring(vessels.features, "capillary_block")["colour_kind"] == "categorical"
    assert _colouring(vessels.features, "flow_change_vs_baseline")["colour_kind"] == "continuous"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"capillary_block_selection": "vessel_ids"}, "lists no vessel IDs"),
        ({"capillary_block_branch_orders": []}, "names no branch orders"),
    ],
)
def test_preflight_reports_a_capillary_block_with_nothing_to_block(overrides, message):
    from haemolynx.haemodynamics import perturbation_problems

    values = {setting.name: setting.default for setting in SCHEMA}
    values["perturbations"] = [{"name": "stall", "type": "capillary_block", "overrides": overrides}]

    problems = perturbation_problems(values, SCHEMA)

    assert any(message in line for line in problems), problems


def test_preflight_is_quiet_about_a_well_formed_capillary_block():
    from haemolynx.haemodynamics import perturbation_problems

    values = {setting.name: setting.default for setting in SCHEMA}
    values["perturbations"] = [{"name": "stall", "type": "capillary_block", "overrides": {}}]

    assert perturbation_problems(values, SCHEMA) == ()


def test_branch_orders_no_vessel_has_are_refused_by_name():
    """Otherwise the perturbation silently blocks nothing and reports the
    baseline as its effect."""
    with pytest.raises(ValueError, match="BO9"):
        resolve_blocked_vessels(
            _many_capillaries(),
            selection="branch_order_probability",
            branch_orders=["BO9"],
            probability=0.5,
        )
