"""Flux heterogeneity, and how much of the network carries flow.

Over the vessels *carrying* flow, the coefficient of variation (sample SD over
mean) of the flow and of the mean velocity -- for the capillaries, the spread
of perfusion across the bed that red-cell flux heterogeneity measures, and
for every vessel -- and the perfused share of the network: the vessels
carrying flow, their length and their volume (a cylinder per vessel,
pi (d/2)^2 L), each with its share of all vessels'. The capillary-block
perturbation reports the same rows on the baseline and on the blocked
network, so a stall's downstream consequence reads as a rise in the CV and a
fall in the flowing volume.

A vessel carries flow when the solve reached it (``flow_state`` "solved", or
``flow_solved`` not False on a graph solved before flow states) and its flow
is above :data:`CARRYING_FLOOR_FRACTION` of the network's largest -- the
capillary-block comparison's rule for what carried flow at baseline. Vessels
behind dead ends carry exactly none, and the floor leaves out a vessel the
solve balanced to nothing (the bridge of a balanced Wheatstone bridge);
either would pull the CV up with a flow that is no perfusion.

Writes each carrying vessel's mean velocity over its lumen's cross-section,
:data:`VELOCITY_EDGE_ATTRIBUTE` (um/s), onto the edge, NaN elsewhere; the
report's velocities are in mm/s, as the Vessel Classes section's.

And the red cells (:mod:`haemolynx.haemodynamics.red_cells`): each carrying
vessel's tube haematocrit, red-cell velocity, red-cell flux (cells per
second) and linear density (cells per mm), predicted from its flow, its
discharge haematocrit -- the vessel's own ``discharge_haematocrit`` where the
run distributed it, else the run's one ``G.graph["haematocrit"]`` -- and the
mean cell volume *red_cell_volume_fl*, with the same mean / SD / CV rows as
the flow and velocity. With a fixed haematocrit the red-cell flux is the flow
times a constant, so its CV is the flow's; the red-cell velocity's is not the
velocity's, since the Fahraeus ratio depends on diameter. None of these is a
measurement: they carry every error of the flow, and the flux the haematocrit
model's too.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import networkx as nx
import numpy as np

from haemolynx.haemodynamics.red_cells import (
    DEFAULT_RED_CELL_VOLUME_FL,
    RED_CELL_EDGE_ATTRIBUTES,
    RED_CELL_FLUX_ATTR,
    RED_CELL_LINEAR_DENSITY_ATTR,
    RED_CELL_VELOCITY_ATTR,
    TUBE_HAEMATOCRIT_ATTR,
    red_cell_flux_per_s,
    red_cell_linear_density_per_mm,
    red_cell_velocity_um_s,
    tube_haematocrit,
    vessel_discharge_haematocrit,
)

from ._route_graph import _clear_edge_attributes, _iter_edges, _set_edge, _unavailable
from .bifurcation import branch_order_category
from .vessel_classes import _NL_PER_MIN_PER_M3_PER_S, carries_flow

__all__ = [
    "CARRYING_FLOOR_FRACTION",
    "FLUX_HETEROGENEITY_EDGE_ATTRIBUTES",
    "VELOCITY_EDGE_ATTRIBUTE",
    "carrying_floor",
    "compute_flux_heterogeneity",
    "is_carrying",
]

#: A vessel carries flow only above this fraction of the network's largest
#: flow: what the capillary-block comparison counts as carrying at baseline.
CARRYING_FLOOR_FRACTION = 1e-9

#: Per-vessel mean velocity over the lumen's cross-section, um/s.
VELOCITY_EDGE_ATTRIBUTE = "vessel_velocity_um_s"
#: What this measure writes on the edges: the velocity and the red cells'
#: four (``haemodynamics.red_cells.RED_CELL_EDGE_ATTRIBUTES``).
FLUX_HETEROGENEITY_EDGE_ATTRIBUTES: tuple[str, ...] = (
    VELOCITY_EDGE_ATTRIBUTE,
    *RED_CELL_EDGE_ATTRIBUTES,
)

_PREFIX = "Flux Heterogeneity"
ASSUMPTIONS = (
    "CV is the sample SD over the mean across vessels; velocity is the solved "
    "mean (plasma) flow over the lumen's cross-section, not red-cell speed; "
    "the red-cell rows are predictions from that flow and the discharge "
    "haematocrit -- tube haematocrit by the Fahraeus relation of Pries et al. "
    "1990, red-cell velocity as the mean velocity times discharge over tube "
    "haematocrit, flux as flow times discharge haematocrit over the red-cell "
    "volume -- not measurements"
)

_NO_CARRYING = "N/A (no vessel carries flow)"
_ONE_CARRYING = "N/A (fewer than two vessels carry flow)"
_NO_CAPILLARIES = "N/A (no vessels classified as capillaries)"
_NO_CARRYING_CAPILLARIES = "N/A (no capillary carries flow)"
_ONE_CARRYING_CAPILLARY = "N/A (fewer than two capillaries carry flow)"
_NO_HAEMATOCRIT = (
    "N/A (no haematocrit: no discharge_haematocrit on the vessels and no "
    "haematocrit on the graph)"
)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def carrying_floor(G: nx.Graph) -> float:
    """The flow (m^3/s) a vessel must exceed to carry flow:
    :data:`CARRYING_FLOOR_FRACTION` of the largest finite ``flow_abs`` on
    *G*; 0 when no vessel has one."""
    largest = 0.0
    for _u, _v, _k, data in _iter_edges(G):
        flow = _number(data.get("flow_abs"))
        if flow is not None and abs(flow) > largest:
            largest = abs(flow)
    return CARRYING_FLOOR_FRACTION * largest


def is_carrying(data: dict, floor: float) -> bool:
    """Whether a vessel carries flow: the solve reached it
    (:func:`~haemolynx.statistics.vessel_classes.carries_flow`) and its
    flow is above *floor*."""
    flow = _number(data.get("flow_abs"))
    return flow is not None and abs(flow) > floor and carries_flow(data)


def _cylinder_volume_um3(diameter_um: float, length_um: float) -> float:
    return math.pi * (0.5 * diameter_um) ** 2 * length_um


def _velocity_um_s(flow_m3_s: float, diameter_um: float) -> float:
    """Mean velocity over a lumen *diameter_um* wide carrying *flow_m3_s*."""
    radius_m = diameter_um * 0.5e-6
    return flow_m3_s / (math.pi * radius_m**2) * 1e6


def _spread(values: list, none: str, one: str) -> tuple[Union[float, str], ...]:
    """(mean, sample SD, CV) of *values*, or why not."""
    if not values:
        return none, none, none
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, one, one
    sd = float(np.std(values, ddof=1))
    return mean, sd, sd / mean if mean > 0 else math.nan


def _volume_rows(vessels: list, carrying: list, what: str) -> Dict[str, Any]:
    """The flowing volume of *carrying* and its share of *vessels*' volume --
    N/A when any vessel in the group has no diameter, since a cylinder per
    vessel over only some of them would read the group's volume low."""
    unsized = sum(1 for vessel in vessels if vessel["diameter"] is None)
    if not vessels:
        reason = _NO_CAPILLARIES if what == "Capillary" else _NO_CARRYING
        return {f"Flowing {what} Volume (micron³)": reason, f"Flowing {what} Volume Fraction": reason}
    if unsized:
        reason = f"N/A ({unsized} of {len(vessels)} vessels have no diameter)"
        return {f"Flowing {what} Volume (micron³)": reason, f"Flowing {what} Volume Fraction": reason}
    total = sum(_cylinder_volume_um3(v["diameter"], v["length"]) for v in vessels)
    flowing = sum(_cylinder_volume_um3(v["diameter"], v["length"]) for v in carrying)
    return {
        f"Flowing {what} Volume (micron³)": flowing,
        f"Flowing {what} Volume Fraction": flowing / total if total > 0 else math.nan,
    }


def _group_rows(vessels: list, what: str, none: str, one: str) -> Dict[str, Any]:
    """Every row for one group of vessels (all of them, or the capillaries):
    how many carry flow, their length and volume with the group's shares, and
    the spread of their flows and velocities."""
    carrying = [v for v in vessels if v["carrying"]]
    noun = "Vessels" if what == "Vessel" else "Capillaries"
    length = sum(v["length"] for v in vessels)
    flowing_length = sum(v["length"] for v in carrying)
    rows: Dict[str, Any] = {
        f"{noun} Carrying Flow": len(carrying),
        f"{noun} Carrying Flow Fraction": len(carrying) / len(vessels) if vessels else math.nan,
        f"Flowing {what} Length (microns)": flowing_length,
        f"Flowing {what} Length Fraction": flowing_length / length if length > 0 else math.nan,
    }
    rows.update(_volume_rows(vessels, carrying, what))
    flows = [v["flow"] * _NL_PER_MIN_PER_M3_PER_S for v in carrying]
    mean, sd, cv = _spread(flows, none, one)
    rows[f"{what} Flow Mean (nL/min)"] = mean
    rows[f"{what} Flow SD (nL/min)"] = sd
    rows[f"{what} Flow CV"] = cv
    sized = [v for v in carrying if v["diameter"] is not None]
    velocities = [_velocity_um_s(v["flow"], v["diameter"]) * 1e-3 for v in sized]
    mean, sd, cv = _spread(velocities, none, one)
    rows[f"{noun} Carrying Flow Without Diameter"] = len(carrying) - len(sized)
    rows[f"{what} Velocity Mean (mm/s)"] = mean
    rows[f"{what} Velocity SD (mm/s)"] = sd
    rows[f"{what} Velocity CV"] = cv
    rows.update(_red_cell_rows(carrying, what, noun, none, one))
    return rows


def _mean_or(values: list, none: str) -> Union[float, str]:
    return float(np.mean(values)) if values else none


def _red_cell_rows(carrying: list, what: str, noun: str, none: str, one: str) -> Dict[str, Any]:
    """The red-cell rows for one group: the spread of the flux (which needs
    no diameter, like the flow) and of the red-cell velocity (which needs
    one, like the velocity), and the mean tube haematocrit and spacing, over
    the carrying vessels with a haematocrit."""
    with_h = [v for v in carrying if v["haematocrit"] is not None]
    if carrying and not with_h:
        none = _NO_HAEMATOCRIT
    rows: Dict[str, Any] = {
        f"{noun} Carrying Flow Without Haematocrit": len(carrying) - len(with_h)
    }
    mean, sd, cv = _spread([v["red_cells"][RED_CELL_FLUX_ATTR] for v in with_h], none, one)
    rows[f"{what} Red Cell Flux Mean (cells/s)"] = mean
    rows[f"{what} Red Cell Flux SD (cells/s)"] = sd
    rows[f"{what} Red Cell Flux CV"] = cv
    sized = [v for v in with_h if v["diameter"] is not None]
    mean, sd, cv = _spread(
        [v["red_cells"][RED_CELL_VELOCITY_ATTR] * 1e-3 for v in sized], none, one
    )
    rows[f"{what} Red Cell Velocity Mean (mm/s)"] = mean
    rows[f"{what} Red Cell Velocity SD (mm/s)"] = sd
    rows[f"{what} Red Cell Velocity CV"] = cv
    rows[f"{what} Tube Haematocrit Mean"] = _mean_or(
        [v["red_cells"][TUBE_HAEMATOCRIT_ATTR] for v in sized], none
    )
    rows[f"{what} Red Cell Linear Density Mean (cells/mm)"] = _mean_or(
        [v["red_cells"][RED_CELL_LINEAR_DENSITY_ATTR] for v in sized], none
    )
    return rows


def _red_cell_values(vessel: dict, red_cell_volume_fl: float) -> Dict[str, float]:
    """A vessel's red-cell attributes: NaN unless it carries flow at a
    haematocrit; the flux needs no diameter, the other three do."""
    values = {name: math.nan for name in RED_CELL_EDGE_ATTRIBUTES}
    h = vessel["haematocrit"]
    if not vessel["carrying"] or h is None:
        return values
    values[RED_CELL_FLUX_ATTR] = red_cell_flux_per_s(vessel["flow"], h, red_cell_volume_fl)
    d = vessel["diameter"]
    if d is not None:
        values[TUBE_HAEMATOCRIT_ATTR] = tube_haematocrit(h, d)
        values[RED_CELL_VELOCITY_ATTR] = red_cell_velocity_um_s(
            _velocity_um_s(vessel["flow"], d), h, d
        )
        values[RED_CELL_LINEAR_DENSITY_ATTR] = red_cell_linear_density_per_mm(
            d, h, red_cell_volume_fl
        )
    return values


def _haematocrit_source(vessels: list, run_haematocrit: Optional[float]) -> str:
    """Which haematocrit the carrying vessels' red cells were counted at."""
    carrying = sum(1 for v in vessels if v["carrying"])
    own = sum(1 for v in vessels if v["carrying"] and v["own_haematocrit"])
    if own:
        source = f"the vessel's own discharge_haematocrit on {own} of {carrying} carrying vessels"
        if own < carrying:
            source += (
                f", the run's haematocrit ({run_haematocrit:g}) on the rest"
                if run_haematocrit is not None
                else ", none on the rest"
            )
        return source
    if run_haematocrit is not None:
        return f"the run's haematocrit ({run_haematocrit:g}) on every vessel"
    return "none: no discharge_haematocrit on the vessels and no haematocrit on the graph"


def compute_flux_heterogeneity(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    red_cell_volume_fl: float = DEFAULT_RED_CELL_VOLUME_FL,
) -> Dict[str, Any]:
    """Flux heterogeneity and the flowing share of *G*: see the module.

    Needs a solved flow (``flow_abs`` on the edges); every row that needs a
    diameter says so where one is missing, and the red-cell rows say so
    where no vessel has a haematocrit. Writes
    :data:`FLUX_HETEROGENEITY_EDGE_ATTRIBUTES` onto every edge (NaN on a
    vessel carrying no flow, or with no diameter or haematocrit for what
    needs one) once it has an answer, nothing otherwise. *red_cell_volume_fl*
    is the mean cell volume the flux and spacing count cells by (the
    ``red_cell_volume_fl`` setting).
    """
    if not float(red_cell_volume_fl) > 0.0:
        raise ValueError(f"red_cell_volume_fl must be positive, got {red_cell_volume_fl}")
    _clear_edge_attributes(G, FLUX_HETEROGENEITY_EDGE_ATTRIBUTES)
    edges = list(_iter_edges(G))
    if not any(_number(d.get("flow_abs")) is not None for *_e, d in edges):
        return _unavailable(_PREFIX, "needs a solved flow (run haemodynamics)")
    floor = carrying_floor(G)
    run_haematocrit, _own = vessel_discharge_haematocrit({}, G.graph.get("haematocrit"))
    vessels = []
    for u, v, key, data in edges:
        diameter = _number(data.get("diameter_um"))
        length = _number(data.get("length"))
        flow = _number(data.get("flow_abs"))
        haematocrit, own = vessel_discharge_haematocrit(data, run_haematocrit)
        vessels.append(
            {
                "edge": (u, v, key),
                "carrying": is_carrying(data, floor),
                "flow": abs(flow) if flow is not None else 0.0,
                "diameter": diameter if diameter is not None and diameter > 0 else None,
                "length": max(length, 0.0) if length is not None else 0.0,
                "category": branch_order_category(data.get("branch_order")),
                "haematocrit": haematocrit,
                "own_haematocrit": own,
            }
        )
    if not any(vessel["carrying"] for vessel in vessels):
        return _unavailable(_PREFIX, "no vessel carries flow")

    for vessel in vessels:
        velocity = (
            _velocity_um_s(vessel["flow"], vessel["diameter"])
            if vessel["carrying"] and vessel["diameter"] is not None
            else math.nan
        )
        _set_edge(G, *vessel["edge"], VELOCITY_EDGE_ATTRIBUTE, velocity)
        vessel["red_cells"] = _red_cell_values(vessel, red_cell_volume_fl)
        for name, value in vessel["red_cells"].items():
            _set_edge(G, *vessel["edge"], name, value)

    result: Dict[str, Any] = {
        f"{_PREFIX} Assumptions": ASSUMPTIONS,
        "Carrying Flow Floor (m³/s)": floor,
        "Red Cell Volume (fL)": float(red_cell_volume_fl),
        "Haematocrit Source": _haematocrit_source(vessels, run_haematocrit),
        "Vessels": len(vessels),
    }
    result.update(_group_rows(vessels, "Vessel", _NO_CARRYING, _ONE_CARRYING))
    capillaries = [v for v in vessels if v["category"] == "capillary"]
    if capillaries:
        result.update(
            _group_rows(capillaries, "Capillary", _NO_CARRYING_CAPILLARIES, _ONE_CARRYING_CAPILLARY)
        )
    else:
        result["Capillary Flux Heterogeneity Status"] = _NO_CAPILLARIES

    by_type: dict = defaultdict(list)
    for vessel in vessels:
        by_type[vessel["category"]].append(vessel)
    result[f"{_PREFIX} By Vessel Type"] = {}
    for category, members in by_type.items():
        carrying = [v for v in members if v["carrying"]]
        sized = [v for v in carrying if v["diameter"] is not None]
        _m, _sd, flow_cv = _spread([v["flow"] for v in carrying], _NO_CARRYING, _ONE_CARRYING)
        _m, _sd, velocity_cv = _spread(
            [_velocity_um_s(v["flow"], v["diameter"]) for v in sized], _NO_CARRYING, _ONE_CARRYING
        )
        red = _red_cell_rows(carrying, "Vessel", "Vessels", _NO_CARRYING, _ONE_CARRYING)
        result[f"{_PREFIX} By Vessel Type"][category] = {
            "Vessels": len(members),
            "Vessels Carrying Flow": len(carrying),
            "Flow CV": flow_cv,
            "Velocity CV": velocity_cv,
            "Red Cell Flux CV": red["Vessel Red Cell Flux CV"],
            "Red Cell Velocity CV": red["Vessel Red Cell Velocity CV"],
            **_volume_rows(members, carrying, "Vessel"),
        }
    return result
