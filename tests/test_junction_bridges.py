"""Junction bridges (graph/junction_bridges.py): the stretch of a narrow vessel
inside the wider vessel it opens into, at a junction no thick vessel is part
of, made a zero-resistance bridge as long as the wider vessel's radius.

The fixtures are straight tubes drawn a micron a point, so every length a
test reads is known by hand: the parent runs along x and is 10 um wide, so a
bridge into it is 5 um long."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph
from haemolynx.graph.junction_bridges import (
    JUNCTION_BRIDGE,
    JUNCTION_BRIDGE_END,
    MIN_VESSEL_LEFT_UM,
    insert_junction_bridges,
    rebridge_junctions,
    remove_junction_bridges,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges
from haemolynx.pipeline import apply_network_handling, default_schema, stages
from haemolynx.pipeline.stages import BoundaryNodes, HaemodynamicModel

SCHEMA = default_schema()
PARENT_UM = 10.0
CAPILLARY_UM = 4.0


def _polyline(*corners, step: float = 1.0) -> list[list[float]]:
    points = [np.asarray(corners[0], dtype=float)]
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        points.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, n + 1)[1:])
    return [p.tolist() for p in points]


def _vessel(G, u, v, width, *corners, **attrs):
    path = _polyline(G.nodes[u]["pos"], *corners, G.nodes[v]["pos"])
    return G.add_edge(
        u, v, voxels=path, length=graph.calculate_path_length(path), diameter_um=width,
        branch_order="B01", **attrs,
    )


def _tee(daughter_um: float = CAPILLARY_UM, daughter_length: float = 30.0) -> nx.MultiGraph:
    """Parent 0-1-2 along x, 10 um wide; a daughter 1-3 leaving it along y."""
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 0.0, -20.0), 1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 20.0),
                      3: (0.0, daughter_length, 0.0)}.items():
        G.add_node(node, pos=pos)
    _vessel(G, 0, 1, PARENT_UM)
    _vessel(G, 1, 2, PARENT_UM)
    _vessel(G, 1, 3, daughter_um)
    return G


def _loop() -> nx.MultiGraph:
    """Parent 0-1-2-3 along x from inlet 0 to outlet 3, and a capillary leaving
    it at 1 and coming back at 2 (30 + 40 + 30 = 100 um), narrow at both."""
    G = nx.MultiGraph()
    for node, x in ((0, -20.0), (1, 0.0), (2, 40.0), (3, 60.0)):
        G.add_node(node, pos=(0.0, 0.0, x))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        _vessel(G, u, v, PARENT_UM)
    _vessel(G, 1, 2, CAPILLARY_UM, (0.0, 30.0, 0.0), (0.0, 30.0, 40.0))
    return G


def _bridges(G) -> list[tuple]:
    return [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if d.get(JUNCTION_BRIDGE)]


def _vessel_lengths(G) -> list[float]:
    return sorted(round(d["length"], 6) for *_e, d in G.edges(keys=True, data=True) if not d.get(JUNCTION_BRIDGE))


# --- the rule --------------------------------------------------------------------


def test_a_narrow_vessel_gives_the_wider_vessels_radius_to_a_bridge():
    G = _tee()

    counts = insert_junction_bridges(G)

    assert counts["bridges"] == 1 and counts["junctions"] == 1
    ((u, v, _k, bridge),) = _bridges(G)
    end = v if u == 1 else u
    assert 1 in (u, v)
    assert bridge["length"] == pytest.approx(PARENT_UM / 2)
    assert bridge[IS_ZERO_RESISTANCE] is True
    assert bridge["diameter_um"] == PARENT_UM and bridge["diameter_source"] == "bridge"
    assert "branch_order" not in bridge
    assert G.nodes[end][JUNCTION_BRIDGE_END] is True
    assert np.allclose(G.nodes[end]["pos"], (0.0, PARENT_UM / 2, 0.0))
    (daughter,) = [d for _a, _b, d in G.edges(end, data=True) if not d.get(JUNCTION_BRIDGE)]
    assert daughter["length"] == pytest.approx(30.0 - PARENT_UM / 2)
    assert daughter["diameter_um"] == CAPILLARY_UM and daughter["branch_order"] == "B01"
    # The parent runs through: both its halves stay as they were.
    assert G.edges[0, 1, 0]["length"] == pytest.approx(20.0)
    assert G.edges[1, 2, 0]["length"] == pytest.approx(20.0)
    assert sum(d["length"] for *_e, d in G.edges(keys=True, data=True)) == pytest.approx(70.0)


def test_a_bridge_keeps_its_vessels_mask_tags_and_none_of_its_measurements():
    """In the same tier as the vessel it stands for, so the tier's search
    reaches the vessel through it; measured as nothing, as every bridge."""
    G = _tee()
    G.edges[1, 3, 0].update({graph.SMALL_MASK_VESSEL_TYPE: "arteriole", "fwhm_diameter_um": 4.1,
                             "edt_diameter_um": 4.3, "diameter_source": "measured"})

    insert_junction_bridges(G)

    ((_u, _v, _k, bridge),) = _bridges(G)
    assert bridge[graph.SMALL_MASK_VESSEL_TYPE] == "arteriole"
    assert "fwhm_diameter_um" not in bridge and "edt_diameter_um" not in bridge
    assert bridge["diameter_source"] == "bridge"
    (daughter,) = [d for u, v, d in G.edges(data=True) if 3 in (u, v)]
    assert daughter["fwhm_diameter_um"] == 4.1 and daughter["diameter_source"] == "measured"


@pytest.mark.parametrize(
    "daughter_um, bridged",
    [(PARENT_UM, False), (8.5, False), (8.0, False), (7.9, True), (2.0, True)],
)
def test_only_a_vessel_under_0_8_of_the_widest_is_bridged(daughter_um, bridged):
    """Vessels of one width meeting, and a branch nearly as wide as what it
    leaves, stay whole: no stretch of one lies in a wider one's lumen."""
    G = _tee(daughter_um)

    insert_junction_bridges(G)

    assert len(_bridges(G)) == int(bridged)


def test_a_bifurcation_bridges_both_narrower_daughters_and_not_the_parent():
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 0.0, -20.0), 1: (0.0, 0.0, 0.0), 2: (0.0, 20.0, 20.0),
                      3: (0.0, -20.0, 20.0)}.items():
        G.add_node(node, pos=pos)
    _vessel(G, 0, 1, PARENT_UM)
    _vessel(G, 1, 2, 6.0)
    _vessel(G, 1, 3, 7.0)

    insert_junction_bridges(G)

    bridges = _bridges(G)
    assert len(bridges) == 2
    assert all(d["length"] == pytest.approx(PARENT_UM / 2) and 1 in (u, v) for u, v, _k, d in bridges)
    assert G.edges[0, 1, 0]["length"] == pytest.approx(20.0)


def test_the_bridge_is_cut_from_the_junction_whichever_way_the_path_runs():
    G = _tee()
    G.edges[1, 3, 0]["voxels"] = G.edges[1, 3, 0]["voxels"][::-1]

    insert_junction_bridges(G)

    ((u, v, _k, bridge),) = _bridges(G)
    end = v if u == 1 else u
    assert np.allclose(G.nodes[end]["pos"], (0.0, PARENT_UM / 2, 0.0))
    assert np.allclose(bridge["voxels"][0], (0.0, 0.0, 0.0)) or np.allclose(bridge["voxels"][-1], (0.0, 0.0, 0.0))


def test_a_vessel_narrow_at_both_its_junctions_is_bridged_at_both():
    G = _loop()

    counts = insert_junction_bridges(G)

    assert counts == {"bridges": 2, "junctions": 2}
    assert sorted({n for u, v, _k, _d in _bridges(G) for n in (u, v)} & {1, 2}) == [1, 2]
    assert _vessel_lengths(G) == [20.0, 20.0, 40.0, pytest.approx(100.0 - PARENT_UM)]


def test_a_vessel_its_bridges_would_leave_under_a_micron_stays_whole():
    G = _tee(daughter_length=PARENT_UM / 2 + MIN_VESSEL_LEFT_UM / 2)

    counts = insert_junction_bridges(G)

    assert _bridges(G) == [] and counts["vessels_too_short"] == 1


def test_a_junction_with_an_unmeasured_vessel_is_left_alone():
    G = _tee()
    del G.edges[1, 2, 0]["diameter_um"]

    counts = insert_junction_bridges(G)

    assert _bridges(G) == [] and counts["junctions_unmeasured"] == 1


def test_a_junction_a_thick_vessels_bridge_meets_is_left_alone():
    G = _tee()
    G.add_node(9, pos=(0.0, -8.0, 0.0))
    G.add_edge(1, 9, voxels=[[0.0, 0.0, 0.0], [0.0, -8.0, 0.0]], length=8.0, diameter_um=30.0,
               **{IS_ZERO_RESISTANCE: True})

    counts = insert_junction_bridges(G)

    assert _bridges(G) == [] and counts["junctions_at_bridges"] == 1


@pytest.mark.parametrize("centreline_x, bridged", [(0.0, False), (500.0, True)])
def test_a_junction_inside_a_thick_vessels_lumen_is_left_alone(centreline_x, bridged):
    """Where neither vessel is thick: the thick centreline graph building kept
    on the graph says which junctions lie in a thick vessel."""
    G = _tee()
    points = np.array([[0.0, 0.0, centreline_x + x] for x in np.arange(-20.0, 21.0)])
    graph.dead_ends.record_thick_centreline(G, points, np.full(len(points), 8.0), tolerance_um=1.5)

    counts = insert_junction_bridges(G)

    assert len(_bridges(G)) == int(bridged)
    assert counts.get("junctions_in_thick_vessels", 0) == int(not bridged)


def test_only_the_junctions_asked_for_are_bridged():
    G = _loop()

    insert_junction_bridges(G, nodes=[2])

    assert len(_bridges(G)) == 1
    ((u, v, _k, _d),) = _bridges(G)
    assert 2 in (u, v)


# --- putting them back ---------------------------------------------------------------


def test_removing_puts_every_vessel_back_whole():
    G = _loop()
    before = {(min(u, v), max(u, v)): round(d["length"], 6) for u, v, d in G.edges(data=True) if d["diameter_um"] == CAPILLARY_UM}
    insert_junction_bridges(G)

    junctions = remove_junction_bridges(G)

    assert junctions == [1, 2]
    assert sorted(G.nodes) == [0, 1, 2, 3]
    assert not any(data.get(JUNCTION_BRIDGE_END) for _n, data in G.nodes(data=True))
    assert not graph.has_junction_bridges(G)
    (capillary,) = [(u, v, d) for u, v, d in G.edges(data=True) if d["diameter_um"] == CAPILLARY_UM]
    assert {(min(capillary[:2]), max(capillary[:2])): round(capillary[2]["length"], 6)} == before
    path = np.asarray(capillary[2]["voxels"])
    assert {tuple(path[0]), tuple(path[-1])} == {(0.0, 0.0, 0.0), (0.0, 0.0, 40.0)}


def test_bridging_twice_draws_the_same_bridges():
    G = _loop()
    insert_junction_bridges(G)
    first = sorted((round(d["length"], 9), tuple(map(tuple, d["voxels"]))) for *_e, d in _bridges(G))

    rebridge_junctions(G, list(G.nodes))

    assert sorted((round(d["length"], 9), tuple(map(tuple, d["voxels"]))) for *_e, d in _bridges(G)) == first


def test_rebridging_follows_a_width_measured_again():
    G = _tee()
    insert_junction_bridges(G)
    (end,) = [n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)]
    (daughter,) = [d for _a, _b, d in G.edges(end, data=True) if not d.get(JUNCTION_BRIDGE)]
    daughter["diameter_um"] = 9.0

    rebridge_junctions(G, [end])

    assert _bridges(G) == []
    assert _vessel_lengths(G) == [20.0, 20.0, 30.0]


def test_a_bridge_ending_at_a_boundary_node_stays():
    G = _tee()
    insert_junction_bridges(G)
    (end,) = [n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)]

    assert remove_junction_bridges(G, keep={end}) == []
    assert len(_bridges(G)) == 1


def _bifurcation() -> nx.MultiGraph:
    """Parent 0-1 along x, 10 um wide; daughters 1-2 (6 um) and 1-3 (7 um)."""
    G = nx.MultiGraph()
    for node, pos in {0: (0.0, 0.0, -20.0), 1: (0.0, 0.0, 0.0), 2: (0.0, 20.0, 20.0),
                      3: (0.0, -20.0, 20.0)}.items():
        G.add_node(node, pos=pos)
    _vessel(G, 0, 1, PARENT_UM)
    _vessel(G, 1, 2, 6.0)
    _vessel(G, 1, 3, 7.0)
    return G


def test_a_bridge_kept_at_a_boundary_node_leaves_the_vessels_beside_it_bridged_again():
    G = _bifurcation()
    insert_junction_bridges(G)
    first = sorted((round(d["length"], 9), tuple(map(tuple, d["voxels"]))) for *_e, d in _bridges(G))
    # The end of the bridge into daughter 1-3 has since become a boundary node.
    (kept,) = [n for n, data in G.nodes(data=True)
               if data.get(JUNCTION_BRIDGE_END) and any(b == 3 for _a, b in G.edges(n))]

    rebridge_junctions(G, [1], keep={kept})

    # Daughter 1-2 is bridged again beside the kept bridge, as on the first run.
    assert len(_bridges(G)) == 2
    assert sorted((round(d["length"], 9), tuple(map(tuple, d["voxels"]))) for *_e, d in _bridges(G)) == first


def test_new_node_ids_are_found_once_however_many_bridges_are_drawn(monkeypatch):
    from haemolynx.graph import junction_bridges as module

    calls = []
    original = module.next_node_id
    monkeypatch.setattr(module, "next_node_id", lambda G, reserved: calls.append(1) or original(G, reserved))
    G = _loop()

    insert_junction_bridges(G)

    assert len(calls) == 1
    assert sorted(n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)) == [4, 5]


def test_a_bridge_whose_vessel_an_edit_took_goes_with_it():
    G = _tee()
    insert_junction_bridges(G)
    (end,) = [n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)]
    G.remove_node(3)

    assert remove_junction_bridges(G, nodes=[end]) == [1]
    assert end not in G and _bridges(G) == []


# --- what the rest of the pipeline makes of them --------------------------------------


def test_bridges_get_no_branch_order_and_take_no_step_of_one():
    plain, bridged = _loop(), _loop()
    insert_junction_bridges(bridged)

    graph.assign_branch_orders(plain, [0])
    results = graph.assign_branch_orders(bridged, [0])

    assert results["bridges"] == 2
    assert all("branch_order" not in d for *_e, d in _bridges(bridged))
    for u, v in ((0, 1), (1, 2), (2, 3)):
        assert bridged.edges[u, v, 0]["branch_order"] == plain.edges[u, v, 0]["branch_order"]
    (capillary,) = [d for *_e, d in bridged.edges(keys=True, data=True)
                    if d["diameter_um"] == CAPILLARY_UM and not d.get(JUNCTION_BRIDGE)]
    assert capillary["branch_order"] == plain.edges[1, 2, 1]["branch_order"]


def test_a_tier_steps_at_the_same_junctions_with_junction_bridges():
    """Counting only three or more of a tier's own vessels as a branching
    (tier_branchings): the bridge counts as its vessel at the junction."""
    from haemolynx.graph import branch_order

    plain, bridged = _loop(), _loop()
    insert_junction_bridges(bridged)

    def passing(G):
        return branch_order._tier_passing_nodes(G, list(G.edges(keys=True)), branch_order.TIER_BRANCHINGS)

    assert passing(bridged) & set(plain) == passing(plain)
    assert {n for n, data in bridged.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)} <= passing(bridged)


def test_a_listed_dead_end_is_removed_with_its_junction_bridge():
    """Post processing's Dead ends list: a narrow dead-end branch is one
    chain through its bridge to the junction, removed whole, and the
    junction merged away as without the bridge."""
    from types import SimpleNamespace

    from haemolynx.graph.dead_ends import chain_to_junction
    from haemolynx.gui.post_processing import remove_listed_interior_dead_ends

    G = _tee()
    insert_junction_bridges(G)
    nodes, edges = chain_to_junction(G, 3)
    assert nodes[-1] == 1 and len(edges) == 2, "the chain runs through the bridge to the junction"
    item = SimpleNamespace(kind="dead_ends", nodes=(3, 1), edges=tuple(edges), why="",
                           centre_um=G.nodes[3]["pos"], measures={"kind": "stops_short"})

    done = remove_listed_interior_dead_ends(G, [item])

    assert (done.dead_ends, done.bridged) == (1, 0)
    assert sorted(G.nodes) == [0, 2] and not graph.has_junction_bridges(G)
    (only,) = G.edges(data=True)
    assert only[2]["length"] == pytest.approx(40.0)


def test_bridges_carry_a_negligible_resistance():
    from haemolynx.haemodynamics import apply_poiseuille_haemodynamics
    from haemolynx.haemodynamics.apply import ZERO_RESISTANCE_FRACTION

    G = _loop()
    insert_junction_bridges(G)

    G, summary = apply_poiseuille_haemodynamics(G, diameter_by_branch_order={"B01": CAPILLARY_UM})

    real = [d["resistance"] for *_e, d in G.edges(keys=True, data=True) if not d.get(IS_ZERO_RESISTANCE)]
    bridges = [d["resistance"] for *_e, d in _bridges(G)]
    assert bridges == [pytest.approx(min(real) * ZERO_RESISTANCE_FRACTION)] * 2
    assert summary["resistances"]["thick_vessel_bridges"] == 2


def test_the_statistics_see_the_narrow_vessel_meet_its_junction_shortened():
    G = _tee()
    insert_junction_bridges(G)

    contracted, became = contract_bridges(G)

    assert contracted.degree(1) == 3
    (daughter,) = [d for _u, v, d in contracted.edges(1, data=True) if v == 3]
    assert daughter["length"] == pytest.approx(30.0 - PARENT_UM / 2)


# --- the setting and the stages ------------------------------------------------------


def test_the_switch_is_off_by_default_on_the_diameters_tab():
    from haemolynx.gui.tabs import assign_to_stages

    setting = SCHEMA["junction_bridges"]
    assert setting.default is False
    assert setting.requires == ("run_haemodynamics", "network_source=segmented_image")
    assert assign_to_stages(SCHEMA)["junction_bridges"] == "5. Diameters"


@pytest.mark.parametrize(
    "settings, in_effect",
    [
        ({"junction_bridges": True, "run_haemodynamics": True}, True),
        ({"junction_bridges": False, "run_haemodynamics": True}, False),
        ({"junction_bridges": True, "run_haemodynamics": False}, False),
        ({"junction_bridges": True, "run_haemodynamics": True, "network_source": "synthetic"}, False),
    ],
)
def test_bridges_are_drawn_only_from_widths_an_image_run_measured(settings, in_effect):
    assert stages.junction_bridges_in_effect(settings) is in_effect


def _diameters_stage(tmp_path: Path, monkeypatch, G, **extra):
    """The Diameters stage on *G*, each vessel's width stamped as it was drawn
    (each one's measurement is not what is tested here): returns the model and
    the lengths of the vessels the measurement was handed."""
    from test_diameter_assignment import _settings, _vessel_network

    for _u, _v, data in G.edges(data=True):
        data.setdefault("drawn_width", data.get("diameter_um"))
    measured: list[list[float]] = []

    def fake_assign(G, _config, *, mask_volume=None, calibration_edges=None, raw_volume=None, **_kw):
        measured.append(_vessel_lengths(G))
        for _u, _v, data in G.edges(data=True):
            if not data.get(IS_ZERO_RESISTANCE):
                data["diameter_um"] = data["drawn_width"]
        return G, {"diameters": {"measured": G.number_of_edges()}}, raw_volume

    monkeypatch.setattr(stages, "assign_edge_diameters", fake_assign)
    lists = {"inlet_nodes": [0], "outlet_nodes": [3], "arteriole_boundary_nodes": [0],
             "venule_boundary_nodes": [3]}
    settings = _settings(tmp_path, **lists, **extra)
    boundaries = BoundaryNodes(**{k: list(v) for k, v in lists.items()}, resistance_node_pair=(0, 3), graph=G)
    model = stages.assign_diameters(settings, _vessel_network(tmp_path, G), boundaries, SCHEMA)
    return model, measured


@pytest.mark.parametrize("on", [False, True])
def test_the_diameters_stage_bridges_once_widths_are_measured_only_with_the_switch_on(
    tmp_path, monkeypatch, on
):
    model, measured = _diameters_stage(tmp_path, monkeypatch, _loop(), junction_bridges=on)

    assert len(_bridges(model.graph)) == (2 if on else 0)
    assert measured == [[20.0, 20.0, 40.0, 100.0]], "measured whole, bridged after"
    if on:
        assert model.results["junction_bridges"]["bridges"] == 2
        assert all("branch_order" not in d for *_e, d in _bridges(model.graph))
    else:
        assert "junction_bridges" not in model.results


def test_the_diameters_stage_measures_a_bridged_network_whole_again(tmp_path, monkeypatch):
    """A network the stage bridged, run through it again (a resumed run, or
    the switch since turned off), is measured whole: no width is read on a
    vessel shortened by bridges an earlier run drew."""
    model, _ = _diameters_stage(tmp_path, monkeypatch, _loop(), junction_bridges=True)

    again, measured = _diameters_stage(tmp_path, monkeypatch, model.graph, junction_bridges=False)

    assert measured == [[20.0, 20.0, 40.0, pytest.approx(100.0)]]
    assert _bridges(again.graph) == []
    assert sorted(again.graph.nodes) == [0, 1, 2, 3]


def test_post_processing_judges_the_junctions_an_edit_touched_afresh():
    G = _tee()
    insert_junction_bridges(G)
    (end,) = [n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END)]
    (daughter,) = [d for _a, _b, d in G.edges(end, data=True) if not d.get(JUNCTION_BRIDGE)]
    daughter["diameter_um"] = 9.5  # an edit, measured again as wide as the parent

    off = {"junction_bridges": False, "run_haemodynamics": True}
    on = {"junction_bridges": True, "run_haemodynamics": True}
    assert stages._junction_bridges_after_edit(on, G, [end], set())["bridges"] == 0
    assert _bridges(G) == []
    daughter = next(d for _a, b, d in G.edges(1, data=True) if b == 3)
    daughter["diameter_um"] = CAPILLARY_UM
    # The edited vessel's two ends are what the edit touched.
    assert stages._junction_bridges_after_edit(on, G, [1, 3], set())["bridges"] == 1
    (end,) = [n for n, data in G.nodes(data=True) if data.get(JUNCTION_BRIDGE_END) and n in G]
    assert stages._junction_bridges_after_edit(off, G, [end, 3], set()) is None
    assert _bridges(G) == [], "with the switch off, the bridges there are only put back"


def test_network_handling_puts_a_bridge_back_where_its_junction_is_one_no_longer():
    """A dead-end half of the parent removed leaves the junction with the
    other half and the daughter's bridge: the bridge goes back into the
    daughter, and the two vessels left are joined into one, as without it."""
    G = _tee()
    insert_junction_bridges(G)
    model = HaemodynamicModel(graph=G)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], resistance_node_pair=(0, 3), graph=G)
    settings = {"boundary_handling": "remove_disconnected", "junction_bridges": True, "run_haemodynamics": True}

    apply_network_handling(settings, model, boundaries)

    assert not graph.has_junction_bridges(model.graph)
    assert sorted(model.graph.nodes) == [0, 3]
    (only,) = model.graph.edges(data=True)
    assert only[2]["length"] == pytest.approx(20.0 + 30.0)
    assert graph.has_junction_bridges(G), "the graph handed in is a checkpoint, left as it was"
