"""The statistics CSV's "Vessel Classes" section: the same measures for each
class of vessel, so the classes can be compared.

A vessel's class is its branch-order label's category
(:func:`~haemolynx.statistics.bifurcation.branch_order_category`):
``Large_Art`` large arterioles, ``Art`` small arterioles (every arteriole,
when the run has no large-vessel tier), ``B`` capillaries, ``Ven`` small
venules and ``Large_Ven`` large venules; with global vessel masks,
``Unknown_Large`` large vessels, ``Unknown_Small`` small vessels and
``Unknown_Cap`` capillaries. Branch orders counted from the inlets alone (no
arteriole/venule boundaries) are all ``B``, so every ordered vessel of such a
run is a capillary.

Each class has the same rows: how many vessels and how long, their share of
the network's length, length, volume and surface per unit volume, diameter,
tortuosity, and, once the flow is solved, the flow, velocity, wall shear
stress and pressure of its vessels carrying flow. The densities divide by the
volume "vessel_density" divides by (the tissue measured from the raw image,
else the box the network's nodes span), so the classes' length densities add
up to the network's.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Union

import networkx as nx
import numpy as np

from haemolynx.graph.branch_order import (
    BRANCH_ORDER_CATEGORY_SEQUENCE,
    UNKNOWN_VESSEL_CLASSES,
    flow_path_key,
)
from haemolynx.graph.prune import FLOW_SOLVED, FLOW_STATE
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

from .bifurcation import (
    GLOBAL_MASK_CATEGORIES,
    UNASSIGNED_CATEGORY,
    VESSEL_CATEGORY_SEQUENCE,
    branch_order_category,
)

#: The section's one key in the report, as "Dead Ends" is the dead ends'.
VESSEL_CLASSES = "Vessel Classes"

#: Each class's heading, by its category.
VESSEL_CLASS_TITLES: Dict[str, str] = {
    "large_arteriole": "Large Arterioles",
    "arteriole": "Small Arterioles",
    "capillary": "Capillaries",
    "venule": "Small Venules",
    "large_venule": "Large Venules",
    "large_vessel": "Large Vessels",
    "small_vessel": "Small Vessels",
    UNASSIGNED_CATEGORY: "Unassigned Vessels",
}

#: How a run labelled its vessels -> the classes it reports, listed whether or
#: not a vessel fell in them, so two runs labelled alike report the same rows.
VESSEL_CLASS_SCHEMES: Dict[str, tuple[str, ...]] = {
    "hierarchical": BRANCH_ORDER_CATEGORY_SEQUENCE,
    "global": GLOBAL_MASK_CATEGORIES + ("capillary",),
    "capillary": ("capillary",),
    "none": (),
}
_SCHEME_TEXT = {
    "hierarchical": "branch orders: large and small arterioles, capillaries, small and large venules",
    "global": "global vessel masks: large vessels, small vessels and capillaries",
    "capillary": (
        "capillary branch orders counted from the inlets alone (no arteriole or "
        "venule boundaries), so every ordered vessel is a capillary"
    ),
    "none": "no branch orders assigned",
}

#: m^3/s, the solve's flow unit, in nL/min.
_NL_PER_MIN_PER_M3_PER_S = 1e12 * 60.0

_NONE_IN_CLASS = "N/A (no vessels in this class)"
_NO_DIAMETER = "N/A (no diameter assigned)"
_NO_POSITIONS = "N/A (no position data)"
_NO_VOLUME = "N/A (no volume to divide by)"
_NO_FLOW = "N/A (no vessel of this class carries flow)"


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list, missing: str) -> Union[float, str]:
    return float(np.mean(values)) if values else missing


def _median(values: list, missing: str) -> Union[float, str]:
    return float(np.median(values)) if values else missing


def _per_volume(amount: float, volume: Optional[float]) -> Union[float, str]:
    if volume is None:
        return _NO_VOLUME
    return amount / volume if volume > 0 else 0.0


def carries_flow(data: dict) -> bool:
    """Whether a vessel's solved flow is a flow through it: on a path between
    an inlet and an outlet (``flow_state`` "solved"; a graph solved before
    flow states, ``flow_solved`` not False) with a flow written. Vessels
    behind dead ends carry exactly none, and would only pull a class's
    velocities towards 0."""
    if _number(data.get("flow_abs")) is None:
        return False
    state = data.get(FLOW_STATE)
    if state is not None:
        return state == "solved"
    return data.get(FLOW_SOLVED) is not False


def vessel_class_scheme(labels: Iterable[Any]) -> str:
    """How a run with these branch-order *labels* classed its vessels: a key
    of :data:`VESSEL_CLASS_SCHEMES`."""
    categories = set()
    for label in labels:
        if label is not None and str(label).strip() in UNKNOWN_VESSEL_CLASSES:
            return "global"
        categories.add(branch_order_category(label))
    if categories & {"large_arteriole", "arteriole", "venule", "large_venule"}:
        return "hierarchical"
    return "capillary" if "capillary" in categories else "none"


def _density_volume(
    G: nx.Graph, node_positions: Optional[dict], tissue_volume_um3: Optional[float]
) -> tuple[Optional[float], str]:
    """What the densities divide by, and where it came from, as
    :func:`~haemolynx.statistics.shape.compute_vessel_density` chooses it."""
    if tissue_volume_um3 is not None:
        return float(tissue_volume_um3), "tissue surface measured from the raw image"
    if node_positions:
        positions = np.array([node_positions[n] for n in G.nodes() if n in node_positions])
        if len(positions) > 0:
            return (
                float(np.prod(positions.max(axis=0) - positions.min(axis=0))),
                "bounding box of the network's nodes",
            )
    return None, "N/A (no position data)"


def _labels_text(labels: set) -> str:
    """The branch-order labels a class holds, in the order blood meets them."""
    ordered = sorted((str(label) for label in labels), key=flow_path_key)
    if not ordered:
        return "none"
    if len(ordered) <= 3:
        return ", ".join(ordered)
    return f"{ordered[0]} to {ordered[-1]} ({len(ordered)} orders)"


def _class_rows(
    vessels: list[dict],
    *,
    network_length: float,
    volume: Optional[float],
    image_volume: Optional[float],
    flow_solved: bool,
) -> Dict[str, Any]:
    """One class's rows from its *vessels* (each a dict of what was read off
    one edge)."""
    lengths = [vessel["length"] for vessel in vessels]
    total = float(sum(lengths))
    rows: Dict[str, Any] = {
        "Branch Orders": _labels_text({v["label"] for v in vessels if v["label"] is not None}),
        "Vessel Count": len(vessels),
        "Total Length (microns)": total,
        "Share of Network Vessel Length (fraction)": (
            total / network_length if network_length > 0 else 0.0
        ),
        "Mean Length (microns)": _mean(lengths, _NONE_IN_CLASS),
        "Median Length (microns)": _median(lengths, _NONE_IN_CLASS),
        "Length Density in Tissue (microns/micron³)": _per_volume(total, volume),
    }
    if image_volume is not None:
        rows["Length Density in Whole Image (microns/micron³)"] = _per_volume(total, image_volume)

    diameters = [v["diameter"] for v in vessels if v["diameter"] is not None]
    missing = _NONE_IN_CLASS if not vessels else _NO_DIAMETER
    rows["Mean Diameter (microns)"] = _mean(diameters, missing)
    rows["Median Diameter (microns)"] = _median(diameters, missing)
    if vessels and len(diameters) == len(vessels):
        volume_um3 = float(sum(math.pi * (v["diameter"] / 2.0) ** 2 * v["length"] for v in vessels))
        surface_um2 = float(sum(math.pi * v["diameter"] * v["length"] for v in vessels))
        rows["Vessel Volume (micron³)"] = volume_um3
        rows["Volume Density in Tissue (micron³/micron³)"] = _per_volume(volume_um3, volume)
        rows["Vessel Surface Area (micron²)"] = surface_um2
        rows["Surface Density in Tissue (micron²/micron³)"] = _per_volume(surface_um2, volume)
    else:
        # A cylinder per vessel needs every vessel's diameter: over only some
        # of them the class's volume would read low.
        unmeasured = (
            missing
            if not diameters
            else f"N/A ({len(vessels) - len(diameters)} of {len(vessels)} vessels have no diameter)"
        )
        for name in (
            "Vessel Volume (micron³)",
            "Volume Density in Tissue (micron³/micron³)",
            "Vessel Surface Area (micron²)",
            "Surface Density in Tissue (micron²/micron³)",
        ):
            rows[name] = unmeasured

    tortuosities = [v["tortuosity"] for v in vessels if v["tortuosity"] is not None]
    missing = _NONE_IN_CLASS if not vessels else _NO_POSITIONS
    rows["Mean Tortuosity Index"] = _mean(tortuosities, missing)
    rows["Median Tortuosity Index"] = _median(tortuosities, missing)

    if flow_solved:
        flowing = [v for v in vessels if v["flow_abs"] is not None]
        flows = [v["flow_abs"] * _NL_PER_MIN_PER_M3_PER_S for v in flowing]
        sized = [v for v in flowing if v["diameter"] is not None]
        # Mean velocity over the lumen's cross-section, mm/s.
        velocities = [
            v["flow_abs"] / (math.pi * (v["diameter"] * 0.5e-6) ** 2) * 1e3 for v in sized
        ]
        # Poiseuille's force balance on a uniform tube, tau = dP d / (4 L):
        # the viscosity is in the pressure drop already.
        shear = [
            abs(v["pressure_drop"]) * v["diameter"] / (4.0 * v["length"])
            for v in sized
            if v["pressure_drop"] is not None and v["length"] > 0
        ]
        pressures = [v["pressure"] for v in flowing if v["pressure"] is not None]
        unsized = _NO_FLOW if not flowing else _NO_DIAMETER
        rows["Vessels Carrying Flow"] = len(flowing)
        rows["Mean Flow (nL/min)"] = _mean(flows, _NO_FLOW)
        rows["Median Flow (nL/min)"] = _median(flows, _NO_FLOW)
        rows["Mean Velocity (mm/s)"] = _mean(velocities, unsized)
        rows["Median Velocity (mm/s)"] = _median(velocities, unsized)
        rows["Mean Wall Shear Stress (Pa)"] = _mean(shear, unsized)
        rows["Median Wall Shear Stress (Pa)"] = _median(shear, unsized)
        rows["Mean Pressure (Pa)"] = _mean(pressures, _NO_FLOW)
    return rows


def _read_vessel(u: Any, v: Any, data: dict, node_positions: Optional[dict]) -> dict:
    length = _number(data.get("length"))
    length = max(length, 0.0) if length is not None else 0.0
    tortuosity = None
    if node_positions and u != v and u in node_positions and v in node_positions:
        straight = float(
            np.linalg.norm(np.asarray(node_positions[u], float) - np.asarray(node_positions[v], float))
        )
        if straight > 0 and length > 0:
            tortuosity = length / straight
    diameter = _number(data.get("diameter_um"))
    flowing = carries_flow(data)
    pressure_u, pressure_v = _number(data.get("pressure_u")), _number(data.get("pressure_v"))
    return {
        "label": data.get("branch_order"),
        "length": length,
        "tortuosity": tortuosity,
        "diameter": diameter if diameter is not None and diameter > 0 else None,
        "flow_abs": _number(data.get("flow_abs")) if flowing else None,
        "pressure_drop": _number(data.get("pressure_drop")) if flowing else None,
        "pressure": (
            (pressure_u + pressure_v) / 2.0
            if flowing and pressure_u is not None and pressure_v is not None
            else None
        ),
    }


def compute_vessel_class_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=None,
    image_dimensions=None,
    *,
    tissue_volume_um3: Optional[float] = None,
) -> Dict[str, Any]:
    """``{"Vessel Classes": {...}}`` for *G*: see the module docstring.

    Inside it, how the vessels were classed, the volume the densities divide
    by, then one heading per class (:data:`VESSEL_CLASS_TITLES`) with its
    rows: every class of the run's way of labelling (:data:`VESSEL_CLASS_SCHEMES`),
    in the order blood meets them, then any other class a vessel is in, then
    the unlabelled vessels. A zero-resistance bridge is no vessel and is left
    out. *voxel_size* and *image_dimensions* (``(z, y, x)``, as for
    "vessel_density") add the length density over the whole image.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edges = (
        ((u, v, d) for u, v, _k, d in G.edges(keys=True, data=True))
        if is_mg
        else G.edges(data=True)
    )
    vessels = [
        _read_vessel(u, v, data, node_positions)
        for u, v, data in edges
        if not data.get(IS_ZERO_RESISTANCE)
    ]
    by_class: Dict[str, list[dict]] = {}
    for vessel in vessels:
        by_class.setdefault(branch_order_category(vessel["label"]), []).append(vessel)

    scheme = vessel_class_scheme(vessel["label"] for vessel in vessels)
    listed = list(VESSEL_CLASS_SCHEMES[scheme])
    listed += [c for c in VESSEL_CATEGORY_SEQUENCE if c in by_class and c not in listed]
    listed += sorted(c for c in by_class if c not in listed)

    volume, source = _density_volume(G, node_positions, tissue_volume_um3)
    image_volume = (
        float(np.prod([d * s for d, s in zip(image_dimensions, voxel_size)]))
        if image_dimensions is not None and voxel_size is not None
        else None
    )
    flow_solved = any(vessel["flow_abs"] is not None for vessel in vessels)
    network_length = float(sum(vessel["length"] for vessel in vessels))

    rows: Dict[str, Any] = {
        "Vessel Classes From": _SCHEME_TEXT[scheme],
        "Vessel Class Density Volume (micron³)": volume if volume is not None else source,
        "Vessel Class Density Volume Source": source,
    }
    if not flow_solved:
        rows["Vessel Class Haemodynamics"] = "N/A (no flow solved)"
    for category in listed:
        rows[VESSEL_CLASS_TITLES.get(category, category)] = _class_rows(
            by_class.get(category, []),
            network_length=network_length,
            volume=volume,
            image_volume=image_volume,
            flow_solved=flow_solved,
        )
    return {VESSEL_CLASSES: rows}


__all__ = [
    "VESSEL_CLASSES",
    "VESSEL_CLASS_SCHEMES",
    "VESSEL_CLASS_TITLES",
    "carries_flow",
    "compute_vessel_class_statistics",
    "vessel_class_scheme",
]
