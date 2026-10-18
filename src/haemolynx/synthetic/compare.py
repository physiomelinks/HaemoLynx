"""How close a synthetic network came to its template, characteristic by
characteristic.

:func:`compare_templates` lines a template up against the same measures of
the synthetic network (a second :class:`~.template.NetworkTemplate`: the
synthetic run's own statistics report read by the same reader, or the
network measured directly), one :class:`ComparisonRow` each, with the
tolerance the generator is held to -- what it hits by construction is
``exact``, what it draws from a distribution a relative or absolute band,
what it cannot control (the capillary branch orders, the network-wide
Murray median, the loop-length percentiles) is reported with no band. The
rows go to ``{stem}_template_comparison.csv`` beside the statistics CSV and
into the run's statistics report, and the tests assert the banded ones.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx

from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE
from haemolynx.statistics.csv_export import _numeric_csv_text
from haemolynx.statistics.vessel_classes import VESSEL_CLASS_TITLES

from .template import (
    FACES,
    ROLES,
    SOURCE_GRAPH,
    NetworkTemplate,
    measure_template,
    template_from_report,
)

#: The section of the statistics report the rows are added to.
COMPARISON_SECTION = "synthetic_comparison"
CSV_COLUMNS = ("Section", "Characteristic", "Template", "Synthetic", "Ratio", "Tolerance", "Within")

#: ``(kind, value)``: ``exact``; ``abs`` within value; ``rel`` within value
#: times the template's.
EXACT = ("exact", 0.0)


@dataclass(frozen=True)
class ComparisonRow:
    section: str
    characteristic: str
    template: Any
    synthetic: Any
    ratio: Optional[float]
    tolerance: str
    within: Optional[bool]


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tolerance_text(tolerance: Optional[tuple]) -> str:
    if tolerance is None:
        return ""
    kind, value = tolerance
    if kind == "exact":
        return "exact"
    if kind == "rel":
        return f"rel {value * 100:g}%"
    return f"abs {value:g}"


def _within(t: Optional[float], s: Optional[float], tolerance: Optional[tuple]) -> Optional[bool]:
    if tolerance is None or t is None or s is None:
        return None
    kind, value = tolerance
    if kind == "exact":
        return math.isclose(t, s, rel_tol=1e-9, abs_tol=1e-9)
    if kind == "abs":
        return abs(s - t) <= value + 1e-9
    return abs(s - t) <= value * abs(t) + 1e-9 if t != 0 else s == 0


def row(section: str, characteristic: str, template: Any, synthetic: Any, tolerance: Optional[tuple] = None) -> ComparisonRow:
    t, s = _number(template), _number(synthetic)
    ratio = None if t is None or s is None or t == 0 else s / t
    return ComparisonRow(
        section=section,
        characteristic=characteristic,
        template=template if template is not None else "N/A (not in the template)",
        synthetic=synthetic if synthetic is not None else "N/A (not measured)",
        ratio=ratio,
        tolerance=_tolerance_text(tolerance),
        within=_within(t, s, tolerance),
    )


def _face_counts(endpoints: Iterable) -> dict:
    counts: dict = {}
    for end in endpoints:
        counts[end.face] = counts.get(end.face, 0) + 1
    return counts


def compare_templates(template: NetworkTemplate, synthetic: NetworkTemplate) -> list[ComparisonRow]:
    """One row per characteristic: see the module."""
    rows: list[ComparisonRow] = []
    T, S = template, synthetic

    for axis, name in enumerate("zyx"):
        rows.append(row("box", f"extent {name} (um)", T.extent_um[axis], S.extent_um[axis], EXACT))
    for axis, name in enumerate("zyx"):
        rows.append(row("box", f"voxel size {name} (um)", T.voxel_size_zyx[axis], S.voxel_size_zyx[axis], EXACT))
    rows.append(row("box", "image volume (um^3)", T.image_volume_um3, S.image_volume_um3, ("rel", 0.02)))

    rows.append(row("endpoints", "inlets", T.inlet_count, S.inlet_count, EXACT))
    rows.append(row("endpoints", "outlets", T.outlet_count, S.outlet_count, EXACT))
    for role, t_ends, s_ends in (("inlets", T.inlets, S.inlets), ("outlets", T.outlets, S.outlets)):
        t_faces, s_faces = _face_counts(t_ends), _face_counts(s_ends)
        for face in list(FACES) + ["interior"]:
            if face in t_faces or face in s_faces:
                rows.append(row("endpoints", f"{role} on {face}", t_faces.get(face, 0), s_faces.get(face, 0), EXACT))
    for role in ROLES[2:]:
        rows.append(row("endpoints", f"{role} nodes", getattr(T, f"{role}_count"), getattr(S, f"{role}_count"), EXACT))

    rows.append(row("network", "total vessel length (um)", T.total_length_um, S.total_length_um, ("rel", 0.10)))
    rows.append(row("network", "vessels", T.vessel_count, S.vessel_count, ("rel", 0.05)))
    rows.append(row("network", "nodes", T.node_count, S.node_count))
    rows.append(row("network", "edges", T.edge_count, S.edge_count))
    rows.append(row("network", "average degree", T.average_degree, S.average_degree, ("abs", 0.15)))
    rows.append(row("network", "branching points", T.branching_points, S.branching_points, ("rel", 0.10)))
    rows.append(row("network", "mean tortuosity", T.mean_tortuosity, S.mean_tortuosity, ("abs", 0.15)))
    rows.append(row("network", "daughter angle (degrees)", T.daughter_angle_deg, S.daughter_angle_deg, ("abs", 10.0)))
    rows.append(row("network", "Murray exponent, median fitted", T.murray_exponent, S.murray_exponent))
    t_iqr = T.murray_iqr or (None, None)
    s_iqr = S.murray_iqr or (None, None)
    rows.append(row("network", "Murray exponent IQR low", t_iqr[0], s_iqr[0]))
    rows.append(row("network", "Murray exponent IQR high", t_iqr[1], s_iqr[1]))
    rows.append(row("network", "intercapillary distance mean (um)", T.intercapillary_distance_um, S.intercapillary_distance_um, ("rel", 0.25)))
    rows.append(row("network", "intercapillary distance median (um)", T.intercapillary_distance_median_um, S.intercapillary_distance_median_um, ("rel", 0.25)))

    for category in BRANCH_ORDER_CATEGORY_SEQUENCE:
        t_cls, s_cls = T.classes.get(category), S.classes.get(category)
        if t_cls is None and s_cls is None:
            continue
        section = f"class {VESSEL_CLASS_TITLES[category]}"
        capillary = category == "capillary"
        get = lambda target, name: None if target is None else getattr(target, name)  # noqa: E731
        rows.append(row(section, "vessels", get(t_cls, "count") or 0, get(s_cls, "count") or 0, ("rel", 0.03) if capillary else EXACT))
        rows.append(row(section, "total length (um)", get(t_cls, "total_length_um") or 0.0, get(s_cls, "total_length_um") or 0.0, ("rel", 0.05)))
        rows.append(row(section, "mean length (um)", get(t_cls, "mean_length_um"), get(s_cls, "mean_length_um"), ("rel", 0.10)))
        rows.append(row(section, "median length (um)", get(t_cls, "median_length_um"), get(s_cls, "median_length_um"), ("rel", 0.15)))
        rows.append(row(section, "mean diameter (um)", get(t_cls, "mean_diameter_um"), get(s_cls, "mean_diameter_um"), ("rel", 0.05) if capillary else ("rel", 0.15)))
        rows.append(row(section, "median diameter (um)", get(t_cls, "median_diameter_um"), get(s_cls, "median_diameter_um")))
        rows.append(row(section, "mean tortuosity", get(t_cls, "mean_tortuosity"), get(s_cls, "mean_tortuosity"), ("abs", 0.15)))
        rows.append(row(section, "median tortuosity", get(t_cls, "median_tortuosity"), get(s_cls, "median_tortuosity")))

    t_orders = {o.label: o for o in T.orders}
    s_orders = {o.label: o for o in S.orders}
    labels = [o.label for o in T.orders] + [o.label for o in S.orders if o.label not in t_orders]
    for label in labels:
        t_o, s_o = t_orders.get(label), s_orders.get(label)
        tree = (t_o or s_o).category != "capillary"
        section = f"order {label}"
        get = lambda target, name: None if target is None else getattr(target, name)  # noqa: E731
        rows.append(row(section, "vessels", get(t_o, "edge_count") if t_o else 0, get(s_o, "edge_count") if s_o else 0, EXACT if tree else None))
        rows.append(row(section, "mean length (um)", get(t_o, "mean_length_um"), get(s_o, "mean_length_um"), ("rel", 0.15) if tree else None))
        # Per-order tortuosity and emergence angles are reported, not banded:
        # an order of two vessels is the luck of two draws, and the emergence
        # angle is read off the first ten microns of the bent polylines, the
        # daughter angle (banded above) off the chords the generator set.
        rows.append(row(section, "mean tortuosity", get(t_o, "mean_tortuosity"), get(s_o, "mean_tortuosity")))
        rows.append(row(section, "mean emergence angle (degrees)", get(t_o, "mean_emergence_angle_deg"), get(s_o, "mean_emergence_angle_deg")))
        rows.append(row(section, "mean diameter (um)", get(t_o, "mean_diameter_um"), get(s_o, "mean_diameter_um")))
        rows.append(row(section, "diameter CV", get(t_o, "diameter_cv"), get(s_o, "diameter_cv")))

    rows.append(row("loops", "independent loops", T.loops.independent_loops, S.loops.independent_loops, ("abs", 2.0)))
    rows.append(row("loops", "independent loops per mm^3", T.loops.independent_loops_per_mm3, S.loops.independent_loops_per_mm3, ("rel", 0.05)))
    for name in ("P10", "P25", "Median", "P75", "P90"):
        rows.append(row("loops", f"loop length {name} (um)", T.loops.length_percentiles_um.get(name), S.loops.length_percentiles_um.get(name)))
    rows.append(row("loops", "vessels on no loop (fraction)", T.loops.vessels_on_no_loop_fraction, S.loops.vessels_on_no_loop_fraction))

    rows.append(row("dead ends", "dead ends", T.dead_ends.total, S.dead_ends.total, ("abs", 2.0)))
    rows.append(row("dead ends", "leaving the image", T.dead_ends.leaves_image, S.dead_ends.leaves_image, ("abs", 2.0)))
    rows.append(row("dead ends", "inside the tissue", T.dead_ends.interior, S.dead_ends.interior, ("abs", 2.0)))
    rows.append(row("dead ends", "artefacts", T.dead_ends.artefacts, S.dead_ends.artefacts))
    rows.append(row("dead ends", "ends of pieces without a junction", T.dead_ends.island_ends, S.dead_ends.island_ends))
    rows.append(row("dead ends", "interior per mm of vessel", T.dead_ends.interior_per_mm, S.dead_ends.interior_per_mm, ("rel", 0.25)))
    rows.append(row("dead ends", "chain length median (um)", T.dead_ends.chain_length_median_um, S.dead_ends.chain_length_median_um, ("rel", 0.25)))
    rows.append(row("dead ends", "chain length P10 (um)", T.dead_ends.chain_length_p10_um, S.dead_ends.chain_length_p10_um))
    rows.append(row("dead ends", "chain length P90 (um)", T.dead_ends.chain_length_p90_um, S.dead_ends.chain_length_p90_um))
    return rows


def compare_run(
    template: NetworkTemplate,
    G: nx.Graph,
    *,
    roles: Mapping[str, Sequence],
    sections: Optional[Mapping[str, Any]] = None,
    image_shape: Optional[Sequence[int]] = None,
    voxel_size_zyx: Optional[Sequence[float]] = None,
) -> list[ComparisonRow]:
    """The rows for the synthetic network *G* (its node *roles*): its own
    statistics report's *sections* read as a template by the template's
    source's reader, or, without them, the network measured directly."""
    if sections is not None and template.provenance.source != SOURCE_GRAPH:
        synthetic = template_from_report(sections, G, boundaries=roles, require=(), source=template.provenance.source)
    else:
        synthetic = measure_template(
            G,
            inlet_nodes=roles.get("inlet", ()),
            outlet_nodes=roles.get("outlet", ()),
            arteriole_boundary_nodes=roles.get("arteriole_boundary", ()),
            venule_boundary_nodes=roles.get("venule_boundary", ()),
            large_arteriole_boundary_nodes=roles.get("large_arteriole_boundary", ()),
            large_venule_boundary_nodes=roles.get("large_venule_boundary", ()),
            image_shape=image_shape,
            voxel_size_zyx=voxel_size_zyx,
            samples=False,
        )
    return compare_templates(template, synthetic)


def comparison_summary(rows: Sequence[ComparisonRow]) -> dict:
    """``{"fitted", "within", "outside": [characteristics]}``."""
    fitted = [r for r in rows if r.within is not None]
    return {
        "rows": len(rows),
        "fitted": len(fitted),
        "within": sum(1 for r in fitted if r.within),
        "outside": [f"{r.section}: {r.characteristic}" for r in fitted if not r.within],
    }


def comparison_section(rows: Sequence[ComparisonRow]) -> dict:
    """The rows as a statistics-report section: ``{"<section>: <characteristic>": "template -> synthetic (ratio)"}``."""
    out: dict = {}
    for r in rows:
        text = f"{_text(r.template)} -> {_text(r.synthetic)}"
        if r.ratio is not None:
            text += f" (x{r.ratio:.3g})"
        if r.within is not None:
            text += " within " + r.tolerance if r.within else " OUTSIDE " + r.tolerance
        out[f"{r.section}: {r.characteristic}"] = text
    summary = comparison_summary(rows)
    out["fitted characteristics within tolerance"] = f"{summary['within']} of {summary['fitted']}"
    return out


def _text(value: Any) -> str:
    number = _number(value)
    return _numeric_csv_text(number) if number is not None else str(value)


def write_template_comparison_csv(rows: Sequence[ComparisonRow], path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([
                r.section,
                r.characteristic,
                _text(r.template),
                _text(r.synthetic),
                "" if r.ratio is None else _numeric_csv_text(r.ratio),
                r.tolerance,
                "" if r.within is None else ("yes" if r.within else "no"),
            ])
    return dest


__all__ = [
    "COMPARISON_SECTION",
    "CSV_COLUMNS",
    "ComparisonRow",
    "compare_run",
    "compare_templates",
    "comparison_section",
    "comparison_summary",
    "row",
    "write_template_comparison_csv",
]
