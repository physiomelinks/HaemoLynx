"""T2.1 to T2.3: three fallbacks that produced a number instead of an error.

Each substitutes a fabricated value for a missing measurement and carries on. The pattern is
the one this pipeline keeps finding: the output is plausible, nothing raises, and the
substitution is invisible in every downstream number.
"""
import logging

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.perfusion import PerfusionGrid, map_vessels_to_grid
from ImageLynx.haemodynamics.resistance import build_conductance_matrix_from_graph


# --- T2.2: the 5.0 um diameter default in map_vessels_to_grid ------------------------------

def _grid_graph(diameters):
    """A star of short edges, each with a polyline, some with a diameter attribute."""
    G = nx.MultiGraph()
    G.add_node("hub", pos=np.array([20.0, 20.0, 20.0]))
    for i, d in enumerate(diameters):
        name = f"n{i}"
        pos = np.array([20.0 + 5.0 * (i + 1), 20.0, 20.0])
        G.add_node(name, pos=pos)
        attrs = {"length": 5.0, "flow_abs": 1.0, "hematocrit": 0.45,
                 "voxels": [(20.0, 20.0, 20.0), tuple(pos)]}
        if d is not None:
            attrs["assigned_diameter_um"] = d
        G.add_edge("hub", name, key=0, **attrs)
    return G


@pytest.mark.parametrize(
    "diameters, count",
    [
        pytest.param([8.0, None, 6.0], "1 of 3 edges", id="one-missing"),
        pytest.param([8.0, None, None, 6.0], "2 of 4 edges", id="two-missing-are-counted"),
        pytest.param([8.0, 0.0], "1 of 2 edges", id="non-positive"),
    ],
)
def test_an_edge_without_a_diameter_is_refused_not_given_five_microns(diameters, count):
    """The default fed vessel surface area, which drives transvascular flux.

    A fabricated 5 um lumen on an unmeasured edge produces a surface area, a flux and a
    tissue PO2 that are all arithmetically fine and none of which mean anything. The refusal
    says how many edges are affected, and a zero diameter is as unusable as an absent one.
    """
    G = _grid_graph(diameters)
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match=f"{count} have no usable diameter"):
        map_vessels_to_grid(G, grid)


def test_the_fallback_can_be_taken_deliberately_but_must_be_asked_for():
    """Refusing outright would block a legitimate exploratory run; being silent is the defect."""
    G = _grid_graph([8.0, None])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    mapping = map_vessels_to_grid(G, grid, default_diameter_um=5.0)
    unmeasured = [item for vessels in mapping.values() for item in vessels
                  if item["edge"] == ("hub", "n1", 0)]
    area = sum(item["surface_area"] for item in unmeasured)
    length = sum(item["length"] for item in unmeasured)
    assert area / length == pytest.approx(np.pi * 5.0)


# --- The 0.45 haematocrit default in map_vessels_to_grid ------------------------------------

def test_an_edge_without_a_haematocrit_is_refused_not_given_systemic():
    """Haematocrit sets each edge's delivered oxygen content.

    A graph that never went through the rheology solve used to be given 0.45 on every edge
    and deliver oxygen as though phase separation had been computed.
    """
    G = _grid_graph([8.0, 6.0, 10.0])
    del G.edges["hub", "n1", 0]["hematocrit"]
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="1 of 3 edges have no 'hematocrit'"):
        map_vessels_to_grid(G, grid)


def test_the_edge_haematocrit_is_carried_into_the_mapping():
    G = _grid_graph([8.0, 6.0])
    G.edges["hub", "n0", 0]["hematocrit"] = 0.2
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    carried = {item["edge"]: item["hematocrit"]
               for vessels in map_vessels_to_grid(G, grid).values() for item in vessels}
    assert carried[("hub", "n0", 0)] == 0.2
    assert carried[("hub", "n1", 0)] == 0.45


# --- T2.3: edges dropped from the conductance matrix ---------------------------------------

def _resistance_graph(resistances):
    G = nx.MultiGraph()
    for i, r in enumerate(resistances):
        attrs = {} if r is None else {"resistance": r}
        G.add_edge("hub", f"n{i}", key=0, **attrs)
    return G


@pytest.mark.parametrize(
    "resistances, total, dropped, missing, non_positive",
    [
        pytest.param([1.0, None, 2.0, None, 3.0], 5, 2, 2, 0, id="missing"),
        # A zero resistance is a short circuit and a negative one is unphysical. They are
        # told apart from a missing value because they mean different upstream faults:
        # absent means the assignment step never ran on that edge, non-positive means it ran
        # and produced something impossible.
        pytest.param([1.0, 0.0, -2.0, 3.0], 4, 2, 0, 2, id="non-positive-counted-separately"),
        pytest.param([1.0, 2.0, 3.0], 3, 0, 0, 0, id="complete-graph-drops-nothing"),
    ],
)
def test_edges_dropped_for_missing_resistance_are_counted(
        resistances, total, dropped, missing, non_positive):
    report = {}
    build_conductance_matrix_from_graph(_resistance_graph(resistances), report=report)
    assert report["edges_total"] == total
    assert report["edges_dropped"] == dropped
    assert report["dropped_missing"] == missing
    assert report["dropped_non_positive"] == non_positive
    assert report["dropped_fraction"] == pytest.approx(dropped / total)


def test_dropping_edges_logs_a_warning():
    """Most call sites do not pass a report, so the count has to reach someone."""
    G = _resistance_graph([1.0, None, None])
    logger = logging.getLogger("ImageLynx.haemodynamics.resistance")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        build_conductance_matrix_from_graph(G)
    finally:
        logger.removeHandler(handler)
    assert any(r.levelno >= logging.WARNING and "2" in r.getMessage() for r in records)


# --- T2.1: the whole-graph guard against a per-edge fallback -------------------------------

from ImageLynx.haemodynamics.poiseuille import (                      # noqa: E402
    _raise_if_measurement_mode_measured_nothing,
    check_diameter_provenance,
)


def test_the_synthetic_fraction_is_reported_whatever_the_mode():
    for mode in ("edt_radius", "fwhm_radius", "constant_radius"):
        result = check_diameter_provenance(
            {"measured_edt": 3, "synthetic_branch_order": 1},
            radius_assignment_mode=mode, max_synthetic_fraction=1.0)
        assert result["synthetic_fraction"] == pytest.approx(0.25)
        assert result["edges"] == 4


def test_the_tolerance_can_be_relaxed_deliberately():
    counts = {"measured_edt": 95, "synthetic_branch_order": 5}
    assert check_diameter_provenance(
        counts, radius_assignment_mode="edt_radius", max_synthetic_fraction=0.1)["ok"]
    with pytest.raises(ValueError):
        check_diameter_provenance(
            counts, radius_assignment_mode="edt_radius", max_synthetic_fraction=0.01)


def test_the_original_zero_measured_guard_still_holds():
    """The new check supplements it rather than replacing it."""
    with pytest.raises(ValueError, match="edt_diameter_um"):
        _raise_if_measurement_mode_measured_nothing("edt_radius", 100, 0)
    _raise_if_measurement_mode_measured_nothing("fwhm_radius", 100, 0)


# --- The band rule's extreme-decile boundary fallback --------------------------------------

from ImageLynx.graph.boundaries import (                               # noqa: E402
    select_boundary_nodes_by_method,
    select_boundary_terminal_nodes,
)

_BAND_SHAPE = (101, 101, 101)


def _band_graph(terminal_z):
    """A hub in the middle of axis 0 with one degree-1 terminal at each given z."""
    G = nx.Graph()
    G.add_node("hub", pos=np.array([50.0, 50.0, 50.0]))
    for i, z in enumerate(terminal_z):
        G.add_node(f"t{i}", pos=np.array([float(z), 50.0 + i, 50.0]))
        G.add_edge("hub", f"t{i}")
    return G


def _closed_cage():
    """Four nodes on a cycle, none of degree 1: the case the fallback was written for."""
    G = nx.cycle_graph(4)
    for n, z in zip(G.nodes, (0.0, 30.0, 70.0, 100.0)):
        G.nodes[n]["pos"] = np.array([z, 50.0, 50.0])
    return G


@pytest.mark.parametrize(
    "make_graph, band",
    [
        pytest.param(lambda: _band_graph([0.0, 10.0]), "outlet band", id="outlet-empty"),
        pytest.param(lambda: _band_graph([90.0, 100.0]), "inlet band", id="inlet-empty"),
        pytest.param(_closed_cage, "inlet and outlet band", id="no-terminals"),
    ],
)
def test_an_empty_band_is_refused_and_named(make_graph, band):
    """A band with no terminal is refused, not filled by position.

    With both terminals in the inlet band the fallback used to hand back the last tenth of
    all nodes along the axis - here the hub, an interior junction no vessel leaves the volume
    through - as the venous boundary. A closed cage has no terminal at all.
    """
    with pytest.raises(ValueError, match=band):
        select_boundary_terminal_nodes(
            make_graph(), _BAND_SHAPE, edge_percent=25.0, end_percent=25.0, axis=0)


def test_the_nerve_pipeline_entry_point_keeps_the_fallback():
    """select_boundary_nodes_by_method is the nerve pipeline's route; it opts in explicitly."""
    G = _band_graph([0.0, 10.0])
    outputs = select_boundary_nodes_by_method(
        G, _BAND_SHAPE, method="edge_percent", node_role="output",
        edge_percent=25.0, end_percent=25.0, axis=0)
    assert "hub" in outputs
