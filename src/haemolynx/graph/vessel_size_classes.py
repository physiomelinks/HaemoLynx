"""Vessels classed by their measured width, a convention the user states.

``classify_vessels_by_size`` classes each vessel by the width it was measured
at against two thresholds the run sets (no default: a convention is stated
by whoever runs it): ``wide`` at least ``vessel_size_wide_min_diameter_um``,
``mid`` at least ``vessel_size_mid_min_diameter_um``, ``narrow`` below it.
A class says how wide a vessel is and nothing more: no arteriole or venule
side, and no claim that a narrow vessel is a capillary by any published
rule. Marked grids found no rule for the small-vessel/capillary call that
held (SZ1, SZ2, SZR; ``docs/notes/vessel_masks_and_boundaries.md``).

The classes ride on the global vessel masks' own path: a wide vessel is
tagged :data:`~haemolynx.graph.branch_order.GLOBAL_MASK_CLASS` ``"large"``
and a mid one ``"small"``, so ``assign_unknown_vessel_classes`` labels them
Unknown_Large, Unknown_Small and Unknown_Cap and the diameter table, the
class medians and every capillary reader work unchanged. What a person
reads names the class by its range (:func:`size_class_title`), never by the
``Unknown_*`` label.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping

import networkx as nx

from ._helpers import edge_id
from .branch_order import GLOBAL_MASK_CLASS
from .thick_vessel_junctions import IS_ZERO_RESISTANCE

__all__ = [
    "MID",
    "NARROW",
    "SIZE_CLASSES",
    "SIZE_CLASS_BY_CATEGORY",
    "SIZE_CONVENTION",
    "VESSEL_CLASSES_SOURCE",
    "VESSEL_SIZE_CLASS",
    "VESSEL_SIZE_FROM",
    "WIDE",
    "check_size_thresholds",
    "clear_vessel_size_classes",
    "size_class_range",
    "size_class_title",
    "size_convention_of",
    "tag_vessels_by_size",
    "vessel_size_class",
]

WIDE = "wide"
MID = "mid"
NARROW = "narrow"
#: The classes, widest first.
SIZE_CLASSES: tuple[str, ...] = (WIDE, MID, NARROW)

#: Edge attribute: the vessel's size class (:data:`SIZE_CLASSES`); absent on
#: a zero-resistance bridge and on every vessel of a run without the
#: convention.
VESSEL_SIZE_CLASS = "vessel_size_class"
#: Edge attribute: where the class came from -- ``width`` (its own measured
#: width), ``neighbours`` (no width; the vessels at both its ends agree) or
#: ``no_width`` (no width, and they do not: narrow).
VESSEL_SIZE_FROM = "vessel_size_from"

#: ``G.graph`` key of the record of how the vessels were classed; holds
#: ``{"source": SIZE_CONVENTION, "mid_min_um", "wide_min_um", "counts",
#: "basis_counts", "unsized"}`` under the convention.
VESSEL_CLASSES_SOURCE = "vessel_classes_source"
SIZE_CONVENTION = "size_convention"

#: The global masks' tag each size class rides on (narrow: none, a capillary).
_MASK_CLASS = {WIDE: "large", MID: "small"}

#: The statistics' vessel category (``statistics.bifurcation.branch_order_category``)
#: of each size class's ``Unknown_*`` label.
SIZE_CLASS_BY_CATEGORY: dict[str, str] = {
    "large_vessel": WIDE,
    "small_vessel": MID,
    "capillary": NARROW,
}


def check_size_thresholds(mid_min_um: Any, wide_min_um: Any) -> tuple[float, float]:
    """``(mid, wide)`` as floats, or ValueError naming what is wrong: either
    unset, not a finite number, negative, or mid not below wide."""
    names = (
        ("vessel_size_mid_min_diameter_um", mid_min_um),
        ("vessel_size_wide_min_diameter_um", wide_min_um),
    )
    values = []
    for name, value in names:
        if value is None:
            raise ValueError(f"{name} is not set: the size convention needs both thresholds.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number of microns, got {value!r}.") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite width of 0 or more microns, got {value!r}.")
        values.append(number)
    mid, wide = values
    if mid >= wide:
        raise ValueError(
            f"vessel_size_mid_min_diameter_um ({mid:g} um) must be below "
            f"vessel_size_wide_min_diameter_um ({wide:g} um)."
        )
    return mid, wide


def vessel_size_class(width_um: Any, mid_min_um: float, wide_min_um: float) -> str | None:
    """The class of a vessel *width_um* wide: each threshold is the least width
    of its class, so a vessel exactly at one is in the wider class. None for a
    width that is missing, not finite or not positive."""
    try:
        width = float(width_um)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width) or width <= 0:
        return None
    if width >= wide_min_um:
        return WIDE
    return MID if width >= mid_min_um else NARROW


def size_class_range(size_class: str, mid_min_um: float, wide_min_um: float) -> str:
    """*size_class*'s widths in words: ``at least 15 µm``, ``6-15 µm``,
    ``under 6 µm``."""
    if size_class == WIDE:
        return f"at least {wide_min_um:g} µm"
    if size_class == MID:
        return f"{mid_min_um:g}-{wide_min_um:g} µm"
    if size_class == NARROW:
        return f"under {mid_min_um:g} µm"
    raise ValueError(f"size_class must be one of {SIZE_CLASSES}, got {size_class!r}.")


def size_class_title(size_class: str, mid_min_um: float, wide_min_um: float) -> str:
    """What a person reads for *size_class*: ``Vessels 6-15 µm wide``."""
    return f"Vessels {size_class_range(size_class, mid_min_um, wide_min_um)} wide"


def size_convention_of(G: nx.Graph) -> dict[str, Any] | None:
    """*G*'s size-convention record (see :data:`VESSEL_CLASSES_SOURCE`), or
    None when its vessels were not classed by size."""
    record = G.graph.get(VESSEL_CLASSES_SOURCE)
    if isinstance(record, Mapping) and record.get("source") == SIZE_CONVENTION:
        return dict(record)
    return None


def clear_vessel_size_classes(G: nx.Graph, *, keep_mask_class: bool) -> bool:
    """Take a size convention's classes off *G* (a network classed by an
    earlier run, run again without the box): the record and every vessel's
    :data:`VESSEL_SIZE_CLASS` and :data:`VESSEL_SIZE_FROM`, and -- unless
    *keep_mask_class*, the global masks' tags being the run's own -- the
    ``GLOBAL_MASK_CLASS`` tags it wrote. Returns whether there was one."""
    if size_convention_of(G) is None:
        return False
    G.graph.pop(VESSEL_CLASSES_SOURCE, None)
    for _u, _v, data in G.edges(data=True):
        if data.pop(VESSEL_SIZE_CLASS, None) is not None and not keep_mask_class:
            data.pop(GLOBAL_MASK_CLASS, None)
        data.pop(VESSEL_SIZE_FROM, None)
    return True


def _edges(G: nx.Graph) -> list[tuple[Any, Any, tuple[Any, Any, Any], dict]]:
    if G.is_multigraph():
        return [(u, v, edge_id(u, v, key), data) for u, v, key, data in G.edges(keys=True, data=True)]
    return [(u, v, edge_id(u, v, 0), data) for u, v, data in G.edges(data=True)]


def _set_class(data: dict, size_class: str, source: str) -> None:
    data[VESSEL_SIZE_CLASS] = size_class
    data[VESSEL_SIZE_FROM] = source
    mask_class = _MASK_CLASS.get(size_class)
    if mask_class is None:
        data.pop(GLOBAL_MASK_CLASS, None)
    else:
        data[GLOBAL_MASK_CLASS] = mask_class


def tag_vessels_by_size(
    G: nx.Graph,
    mid_min_um: float,
    wide_min_um: float,
    *,
    edges: Iterable[tuple[Any, Any, Any]] | None = None,
    unsized: Iterable[tuple[Any, Any, Any]] = (),
    width_attribute: str = "diameter_um",
    default_basis: str = "plasma_column",
) -> dict[str, Any]:
    """Class each vessel of *G* by its width, and record how.

    A vessel's width is its *width_attribute* -- ``diameter_um`` as the
    Diameters stage stamped it from a measurement, before any class median or
    table fills the rest; the mask's ``edt_diameter_um`` when the run measures
    no diameters -- read in the basis it was measured on, with no conversion.
    A vessel in *unsized* (``(u, v, key)``: the ones still waiting for a
    class median or the table) has none, whatever it carries.

    Each vessel gets :data:`VESSEL_SIZE_CLASS` (:func:`vessel_size_class`),
    :data:`VESSEL_SIZE_FROM`, and the global masks' tag its class rides on:
    ``GLOBAL_MASK_CLASS`` ``"large"`` on a wide vessel, ``"small"`` on a mid
    one, none on a narrow one. A vessel with no width takes the class of the
    vessels at both its ends when every one of them classed by its own width
    is of one class (``neighbours``), else narrow (``no_width``). A
    zero-resistance bridge is never classed.

    *edges* (``(u, v, key)``: the ones a hand edit made) limits the classing
    to those, judged beside the classes the others already carry. The record
    (:data:`VESSEL_CLASSES_SOURCE`) counts the whole network either way:
    vessels per class, the bases the widths were read on (*default_basis*
    for a width with no basis of its own, the run's), and the vessels with
    no width by what became of them. Returns the record.
    """
    from haemolynx.haemodynamics.poiseuille import edge_diameter_basis

    mid_min_um, wide_min_um = check_size_thresholds(mid_min_um, wide_min_um)
    everything = _edges(G)
    wanted = None if edges is None else {edge_id(u, v, key) for u, v, key in edges}
    waiting = {edge_id(u, v, key) for u, v, key in unsized}
    vessels = [
        entry for entry in everything
        if not entry[3].get(IS_ZERO_RESISTANCE) and (wanted is None or entry[2] in wanted)
    ]
    for _u, _v, _edge, data in everything:
        if data.get(IS_ZERO_RESISTANCE):
            data.pop(VESSEL_SIZE_CLASS, None)
            data.pop(VESSEL_SIZE_FROM, None)
            data.pop(GLOBAL_MASK_CLASS, None)

    no_width = []
    for u, v, edge, data in vessels:
        size_class = None if edge in waiting else vessel_size_class(
            data.get(width_attribute), mid_min_um, wide_min_um
        )
        if size_class is None:
            data.pop(VESSEL_SIZE_CLASS, None)
            data[VESSEL_SIZE_FROM] = "no_width"
            no_width.append((u, v, data))
        else:
            _set_class(data, size_class, "width")

    def classes_at(node: Any, data: dict) -> set[str]:
        return {
            other[VESSEL_SIZE_CLASS]
            for _a, _b, other in G.edges(node, data=True)
            if other is not data
            and not other.get(IS_ZERO_RESISTANCE)
            and other.get(VESSEL_SIZE_FROM) == "width"
        }

    for u, v, data in no_width:
        at_u, at_v = classes_at(u, data), classes_at(v, data)
        agreed = at_u | at_v
        if at_u and at_v and len(agreed) == 1:
            _set_class(data, next(iter(agreed)), "neighbours")
        else:
            _set_class(data, NARROW, "no_width")

    counts: Counter[str] = Counter()
    bases: Counter[str] = Counter()
    from_neighbours = 0
    left_narrow = 0
    for _u, _v, _edge, data in everything:
        size_class = data.get(VESSEL_SIZE_CLASS)
        if data.get(IS_ZERO_RESISTANCE) or size_class is None:
            continue
        counts[size_class] += 1
        source = data.get(VESSEL_SIZE_FROM)
        if source == "width":
            # Only a stamped diameter carries the basis of its own measurement;
            # the mask's width is on the run's.
            stamped = data if width_attribute == "diameter_um" else {}
            bases[edge_diameter_basis(stamped, default_basis)] += 1
        elif source == "neighbours":
            from_neighbours += 1
        else:
            left_narrow += 1
    record = {
        "source": SIZE_CONVENTION,
        "mid_min_um": mid_min_um,
        "wide_min_um": wide_min_um,
        "counts": {name: int(counts.get(name, 0)) for name in SIZE_CLASSES},
        "basis_counts": dict(sorted(bases.items())),
        "unsized": {"neighbours": from_neighbours, "no_width": left_narrow},
    }
    G.graph[VESSEL_CLASSES_SOURCE] = record
    return dict(record)
