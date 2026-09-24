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
)
from haemolynx.haemodynamics.capillary_block import (
    BLOCK_EDGE_ATTRIBUTE,
    FLOW_CHANGE_EDGE_ATTRIBUTE,
)
from haemolynx.haemodynamics.poiseuille import set_edge_resistance
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

    # Less flow through the same volume: blood takes longer.
    assert summary["mean_transit_time_s_blocked"] > summary["mean_transit_time_s_baseline"]
    assert "capillary_transit_time_heterogeneity_s_blocked" in summary
    # A comparison writes nothing onto either network but the flow change.
    assert all("transit_time_s" not in d for *_e, d in baseline.edges(keys=True, data=True))
    assert all("transit_time_s" not in d for *_e, d in blocked.edges(keys=True, data=True))


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
            "use_fwhm_edge_diameters": False,
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
