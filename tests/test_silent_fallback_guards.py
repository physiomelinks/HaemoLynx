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
        attrs = {"length": 5.0, "flow_abs": 1.0,
                 "voxels": [(20.0, 20.0, 20.0), tuple(pos)]}
        if d is not None:
            attrs["assigned_diameter_um"] = d
        G.add_edge("hub", name, key=0, **attrs)
    return G


def test_an_edge_without_a_diameter_is_refused_not_given_five_microns():
    """The default fed vessel surface area, which drives transvascular flux.

    A fabricated 5 um lumen on an unmeasured edge produces a surface area, a flux and a
    tissue PO2 that are all arithmetically fine and none of which mean anything.
    """
    G = _grid_graph([8.0, None, 6.0])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="diameter"):
        map_vessels_to_grid(G, grid)


def test_the_refusal_says_how_many_edges_are_affected():
    G = _grid_graph([8.0, None, None, 6.0])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError) as excinfo:
        map_vessels_to_grid(G, grid)
    assert "2" in str(excinfo.value)


def test_a_non_positive_diameter_is_refused_too():
    G = _grid_graph([8.0, 0.0])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="diameter"):
        map_vessels_to_grid(G, grid)


def test_a_fully_measured_graph_maps_without_complaint():
    G = _grid_graph([8.0, 6.0, 10.0])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    mapping = map_vessels_to_grid(G, grid)
    assert mapping, "a fully measured graph produced no mapping"


def test_the_fallback_can_be_taken_deliberately_but_must_be_asked_for():
    """Refusing outright would block a legitimate exploratory run; being silent is the defect."""
    G = _grid_graph([8.0, None])
    grid = PerfusionGrid(G, (10.0, 10.0, 10.0))
    mapping = map_vessels_to_grid(G, grid, default_diameter_um=5.0)
    assert mapping


# --- T2.3: edges dropped from the conductance matrix ---------------------------------------

def _resistance_graph(resistances):
    G = nx.MultiGraph()
    for i, r in enumerate(resistances):
        attrs = {} if r is None else {"resistance": r}
        G.add_edge("hub", f"n{i}", key=0, **attrs)
    return G


def test_edges_dropped_for_missing_resistance_are_counted():
    G = _resistance_graph([1.0, None, 2.0, None, 3.0])
    report = {}
    build_conductance_matrix_from_graph(G, report=report)
    assert report["edges_total"] == 5
    assert report["edges_dropped"] == 2
    assert report["dropped_missing"] == 2
    assert report["dropped_non_positive"] == 0


def test_edges_dropped_for_non_positive_resistance_are_counted_separately():
    """A zero resistance is a short circuit and a negative one is unphysical.

    They are distinguished from a missing value because they mean different upstream faults:
    absent means the assignment step never ran on that edge, non-positive means it ran and
    produced something impossible.
    """
    G = _resistance_graph([1.0, 0.0, -2.0, 3.0])
    report = {}
    build_conductance_matrix_from_graph(G, report=report)
    assert report["dropped_missing"] == 0
    assert report["dropped_non_positive"] == 2
    assert report["edges_dropped"] == 2


def test_a_complete_graph_reports_nothing_dropped():
    report = {}
    build_conductance_matrix_from_graph(_resistance_graph([1.0, 2.0, 3.0]), report=report)
    assert report["edges_dropped"] == 0
    assert report["dropped_fraction"] == 0.0


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


def test_the_return_value_is_unchanged_so_callers_do_not_break():
    """Ten call sites unpack a two-tuple; the report is opt-in."""
    result = build_conductance_matrix_from_graph(_resistance_graph([1.0, 2.0]))
    assert isinstance(result, tuple) and len(result) == 2


# --- T2.1: the whole-graph guard against a per-edge fallback -------------------------------

from ImageLynx.haemodynamics.poiseuille import (                      # noqa: E402
    _raise_if_measurement_mode_measured_nothing,
    check_diameter_provenance,
)


def test_edt_mode_refuses_a_partial_fallback_not_only_a_total_one():
    """The existing guard fires only at zero measured edges; the fallback is per-edge.

    A run where EDT measured 90% of edges and fabricated the other 10% passed silently, and
    resistance goes as the inverse fourth power of diameter, so a fabricated calibre is not a
    small perturbation on the edges that carry it.
    """
    counts = {"measured_edt": 900, "synthetic_branch_order": 100}
    with pytest.raises(ValueError, match="synthetic"):
        check_diameter_provenance(counts, radius_assignment_mode="edt_radius")


def test_edt_mode_accepts_a_fully_measured_graph():
    counts = {"measured_edt": 1000}
    assert check_diameter_provenance(counts, radius_assignment_mode="edt_radius")["ok"]


def test_fwhm_mode_tolerates_a_partial_fallback_by_design():
    """FWHM legitimately fails on individual edges, so partial measurement is a real outcome."""
    counts = {"measured_fwhm": 500, "synthetic_branch_order": 500}
    result = check_diameter_provenance(counts, radius_assignment_mode="fwhm_radius")
    assert result["ok"]
    assert result["synthetic_fraction"] == pytest.approx(0.5)


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
