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
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import networkx as nx
import numpy as np

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
FLUX_HETEROGENEITY_EDGE_ATTRIBUTES: tuple[str, ...] = (VELOCITY_EDGE_ATTRIBUTE,)

_PREFIX = "Flux Heterogeneity"
ASSUMPTIONS = (
    "CV is the sample SD over the mean across vessels; velocity is the solved "
    "mean (plasma) flow over the lumen's cross-section, not red-cell speed"
)

_NO_CARRYING = "N/A (no vessel carries flow)"
_ONE_CARRYING = "N/A (fewer than two vessels carry flow)"
_NO_CAPILLARIES = "N/A (no vessels classified as capillaries)"
_NO_CARRYING_CAPILLARIES = "N/A (no capillary carries flow)"
_ONE_CARRYING_CAPILLARY = "N/A (fewer than two capillaries carry flow)"


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
    return rows


def compute_flux_heterogeneity(G: Union[nx.Graph, nx.MultiGraph]) -> Dict[str, Any]:
    """Flux heterogeneity and the flowing share of *G*: see the module.

    Needs a solved flow (``flow_abs`` on the edges); every row that needs a
    diameter says so where one is missing. Writes
    :data:`VELOCITY_EDGE_ATTRIBUTE` onto every edge (NaN on a vessel carrying
    no flow or with no diameter) once it has an answer, nothing otherwise.
    """
    _clear_edge_attributes(G, FLUX_HETEROGENEITY_EDGE_ATTRIBUTES)
    edges = list(_iter_edges(G))
    if not any(_number(d.get("flow_abs")) is not None for *_e, d in edges):
        return _unavailable(_PREFIX, "needs a solved flow (run haemodynamics)")
    floor = carrying_floor(G)
    vessels = []
    for u, v, key, data in edges:
        diameter = _number(data.get("diameter_um"))
        length = _number(data.get("length"))
        flow = _number(data.get("flow_abs"))
        vessels.append(
            {
                "edge": (u, v, key),
                "carrying": is_carrying(data, floor),
                "flow": abs(flow) if flow is not None else 0.0,
                "diameter": diameter if diameter is not None and diameter > 0 else None,
                "length": max(length, 0.0) if length is not None else 0.0,
                "category": branch_order_category(data.get("branch_order")),
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

    result: Dict[str, Any] = {
        f"{_PREFIX} Assumptions": ASSUMPTIONS,
        "Carrying Flow Floor (m³/s)": floor,
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
        result[f"{_PREFIX} By Vessel Type"][category] = {
            "Vessels": len(members),
            "Vessels Carrying Flow": len(carrying),
            "Flow CV": flow_cv,
            "Velocity CV": velocity_cv,
            **_volume_rows(members, carrying, "Vessel"),
        }
    return result
