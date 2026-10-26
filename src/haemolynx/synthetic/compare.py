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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE
from haemolynx.statistics.csv_export import _numeric_csv_text
from haemolynx.statistics.vessel_classes import VESSEL_CLASS_TITLES

from .generate import dead_end_pairs_joined
from .organisation import Organisation, correlation, profile, share_inside
from .template import (
    FACES,
    GRAPH_TEMPLATE_RECORD,
    ROLES,
    SOURCE_GRAPH,
    NetworkTemplate,
    measure_template,
    template_from_report,
)

#: The section of the statistics report the rows are added to.
COMPARISON_SECTION = "synthetic_comparison"
CSV_COLUMNS = (
    "Section", "Characteristic", "Template", "Synthetic", "Ratio", "Tolerance", "Within",
    "Members low", "Members high", "Within members",
)

#: ``(kind, value)``: ``exact``; ``abs`` within value; ``rel`` within value
#: times the template's.
EXACT = ("exact", 0.0)

#: Every band the generator is held to, the one table the rows and the tests
#: read: ``{(kind of row, characteristic): tolerance}``. The kinds are the
#: row's section, or for the per-class and per-order sections ``class`` /
#: ``capillary class`` and ``tree order`` / ``capillary order``; the face and
#: role rows of ``endpoints`` share one key each. A characteristic not listed
#: is reported with no band: it is what the generator cannot control.
TOLERANCES: dict[tuple[str, str], tuple] = {
    ("box", "extent z (um)"): EXACT,
    ("box", "extent y (um)"): EXACT,
    ("box", "extent x (um)"): EXACT,
    ("box", "voxel size z (um)"): EXACT,
    ("box", "voxel size y (um)"): EXACT,
    ("box", "voxel size x (um)"): EXACT,
    ("box", "image volume (um^3)"): ("rel", 0.02),
    ("endpoints", "inlets"): EXACT,
    ("endpoints", "outlets"): EXACT,
    ("endpoints", "inlets or outlets on a face"): EXACT,
    ("endpoints", "boundary nodes of a role"): EXACT,
    ("network", "diameter basis"): EXACT,
    ("network", "total vessel length (um)"): ("rel", 0.10),
    ("network", "vessels"): ("rel", 0.05),
    ("network", "average degree"): ("abs", 0.15),
    ("network", "branching points"): ("rel", 0.10),
    ("network", "mean tortuosity"): ("abs", 0.15),
    ("network", "daughter angle (degrees)"): ("abs", 10.0),
    ("network", "intercapillary distance mean (um)"): ("rel", 0.25),
    ("network", "intercapillary distance median (um)"): ("rel", 0.25),
    ("class", "vessels"): EXACT,
    ("capillary class", "vessels"): ("rel", 0.03),
    ("class", "total length (um)"): ("rel", 0.05),
    ("capillary class", "total length (um)"): ("rel", 0.05),
    ("class", "mean length (um)"): ("rel", 0.10),
    ("capillary class", "mean length (um)"): ("rel", 0.10),
    ("class", "median length (um)"): ("rel", 0.15),
    ("capillary class", "median length (um)"): ("rel", 0.15),
    ("class", "mean diameter (um)"): ("rel", 0.15),
    ("capillary class", "mean diameter (um)"): ("rel", 0.05),
    ("class", "mean tortuosity"): ("abs", 0.15),
    ("capillary class", "mean tortuosity"): ("abs", 0.15),
    # Per-order tortuosity and emergence angles are reported, not banded: an
    # order of two vessels is the luck of two draws, and the emergence angle
    # is read off the first ten microns of the bent polylines, the daughter
    # angle (banded above) off the chords the generator set.
    ("tree order", "vessels"): EXACT,
    ("tree order", "mean length (um)"): ("rel", 0.15),
    ("loops", "independent loops"): ("abs", 2.0),
    ("loops", "independent loops per mm^3"): ("rel", 0.05),
    ("dead ends", "dead ends"): ("abs", 2.0),
    ("dead ends", "leaving the image"): ("abs", 2.0),
    ("dead ends", "inside the tissue"): ("abs", 2.0),
    ("dead ends", "interior per mm of vessel"): ("rel", 0.25),
    ("dead ends", "chain length median (um)"): ("rel", 0.25),
    # Where the vessels lie (Harvey, 2026-10-25: D1b). A correlation's
    # template value is 1 (the template's field against itself).
    ("organisation", "capillary vertices by cell (correlation)"): ("abs", 0.3),
    ("organisation", "capillary length by cell (correlation)"): ("abs", 0.3),
    ("organisation", "capillary length along an axis (correlation)"): ("abs", 0.3),
    ("organisation", "tree length in the template's tree cells (share)"): ("abs", 0.25),
    ("organisation", "capillary length along an axis (share)"): ("abs", 0.05),
    ("organisation", "box within 20 um of a capillary"): ("abs", 0.07),
    ("organisation", "box beyond 50 um of a capillary"): ("abs", 0.05),
    ("organisation", "trunk turn (degrees)"): ("abs", 15.0),
    # Harvey, 2026-10-25: E14.5's capillary columns, read off any template
    # (a run's count is a few dozen at most: a third either way).
    ("organisation", "capillary chords along an axis (share)"): ("abs", 0.05),
    # Banded since grid D1b (Harvey, 2026-10-25): its capillaries had lost
    # their curvature (7-11 against 26 degrees) and only their tortuosity
    # row said so. Not calibrated: E14.5 draws 22-25.
    ("organisation", "capillary curvature (degrees per 10 um)"): ("abs", 6.0),
    ("organisation", "tree curvature (degrees per 10 um)"): ("abs", 6.0),
    # Read at two steps (Harvey, grid D1f: the trees Harvey passed over wound
    # more). Not calibrated.
    ("organisation", "tree curvature, median vessel (degrees per 10 um)"): ("abs", 4.0),
    ("organisation", "tree curvature at 2 um steps, median vessel (degrees per 10 um)"): ("abs", 10.0),
    # The trees' shape (Harvey, grid D1b): their side branches and how well
    # their trunks hold their course. Not calibrated: E14.5 draws its 27
    # leaves exactly and its trunks at 0.55-0.62 against 0.63.
    ("organisation", "tree leaves"): ("abs", 3.0),
    ("organisation", "trunk straightness, root to leaf"): ("abs", 0.08),
    # How far each side's trees reach (Harvey's grid D1c). Not calibrated.
    ("organisation", "tree reach (um)"): ("rel", 0.15),
    ("organisation", "capillary runs along an axis spanning 100 um (per mm^2)"): ("rel", 0.35),
    # How the vessels lie (statistics.layout; Harvey, 2026-10-25, after grid
    # D1d: every drawn network sat outside E14.5 on these). Not calibrated.
    ("layout", "junction smallest angle median (degrees)"): ("abs", 10.0),
    ("layout", "junction smallest angle P25 (degrees)"): ("abs", 12.0),
    ("layout", "junction largest angle median (degrees)"): ("abs", 10.0),
    ("layout", "junctions with two vessels within 45 degrees (share)"): ("abs", 0.05),
    ("layout", "junction angle sum median (degrees)"): ("abs", 10.0),
    ("layout", "box within 20 um of a vessel (share)"): ("abs", 0.07),
    ("layout", "box beyond 50 um of a vessel (share)"): ("abs", 0.05),
    ("layout", "distance to the nearest vessel, median (um)"): ("rel", 0.15),
    ("layout", "distance to the nearest vessel, P90 (um)"): ("rel", 0.15),
    ("layout", "column share of vessel length"): ("abs", 0.05),
    ("layout", "columns per mm^2"): ("rel", 0.25),
    ("layout", "column spacing median (um)"): ("rel", 0.25),
    ("layout", "column spacing Clark-Evans ratio"): ("abs", 0.12),
    ("layout", "column tortuosity median"): ("abs", 0.05),
    ("class", "tortuosity P90"): ("abs", 0.15),
    ("capillary class", "tortuosity P90"): ("abs", 0.10),
}

#: Each layout row's characteristic and :class:`~.template.LayoutTarget` field.
LAYOUT_CHARACTERISTICS: tuple[tuple[str, str], ...] = (
    ("junction smallest angle median (degrees)", "junction_smallest_angle_median_deg"),
    ("junction smallest angle P25 (degrees)", "junction_smallest_angle_p25_deg"),
    ("junction largest angle median (degrees)", "junction_largest_angle_median_deg"),
    ("junctions with two vessels within 45 degrees (share)", "acute_junction_share"),
    ("junction angle sum median (degrees)", "junction_angle_sum_median_deg"),
    ("box within 20 um of a vessel (share)", "box_within_20_share"),
    ("box beyond 50 um of a vessel (share)", "box_beyond_50_share"),
    ("distance to the nearest vessel, median (um)", "distance_median_um"),
    ("distance to the nearest vessel, P90 (um)", "distance_p90_um"),
    ("column share of vessel length", "column_length_share"),
    ("columns per mm^2", "columns_per_mm2"),
    ("column spacing median (um)", "column_spacing_median_um"),
    ("column spacing Clark-Evans ratio", "column_clark_evans"),
    ("column tortuosity median", "column_tortuosity_median"),
)


@dataclass(frozen=True)
class ComparisonRow:
    section: str
    characteristic: str
    template: Any
    synthetic: Any
    ratio: Optional[float]
    tolerance: str
    within: Optional[bool]
    #: A batch template's members' own values, each at the average's box,
    #: lowest and highest, and whether the synthetic value lies between them
    #: (Harvey's B4b: is it another member of the condition?). None for a
    #: one-run template and for a row with nothing to range over.
    members_low: Optional[float] = None
    members_high: Optional[float] = None
    within_members: Optional[bool] = None


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
    within = _within(t, s, tolerance)
    if tolerance == EXACT and isinstance(template, str) and isinstance(synthetic, str):
        within = template == synthetic  # a named value, such as the diameter basis
    return ComparisonRow(
        section=section,
        characteristic=characteristic,
        template=template if template is not None else "N/A (not in the template)",
        synthetic=synthetic if synthetic is not None else "N/A (not measured)",
        ratio=ratio,
        tolerance=_tolerance_text(tolerance),
        within=within,
    )


def banded(kind: str, section: str, characteristic: str, template: Any, synthetic: Any, key: Optional[str] = None) -> ComparisonRow:
    """:func:`row` with its band from :data:`TOLERANCES` under ``(kind,
    key or characteristic)``."""
    return row(section, characteristic, template, synthetic, TOLERANCES.get((kind, key or characteristic)))


def _face_counts(endpoints: Iterable) -> dict:
    counts: dict = {}
    for end in endpoints:
        counts[end.face] = counts.get(end.face, 0) + 1
    return counts


def unbanded(row: ComparisonRow, note: str) -> ComparisonRow:
    """*row* reported, not banded, *note* in its tolerance column saying why."""
    return replace(row, tolerance=note, within=None)


def not_drawn(row: ComparisonRow, why: str) -> ComparisonRow:
    """*row* reported, not banded: the generator was asked not to draw it."""
    return unbanded(row, f"not drawn ({why})")


def not_held(row: ComparisonRow, why: str) -> ComparisonRow:
    """*row* reported, not banded: drawn, but what the generator was asked
    to leave out changes it by design."""
    return unbanded(row, f"not held ({why})")


def compare_templates(
    template: NetworkTemplate, synthetic: NetworkTemplate, *, options: Optional[Mapping[str, Any]] = None
) -> list[ComparisonRow]:
    """One row per characteristic, banded by :data:`TOLERANCES`: see the
    module. *options* are the generation options the synthetic network was
    drawn with (its record's ``options``): what they left out is reported,
    not banded."""
    rows: list[ComparisonRow] = []
    T, S = template, synthetic
    options = options or {}
    closed = options.get("open_ends") == "closed"
    no_interior = options.get("interior_dead_ends") is False

    def add(section: str, characteristic: str, t: Any, s: Any, *, kind: Optional[str] = None, key: Optional[str] = None) -> None:
        rows.append(banded(kind or section, section, characteristic, t, s, key))

    for axis, name in enumerate("zyx"):
        add("box", f"extent {name} (um)", T.extent_um[axis], S.extent_um[axis])
    for axis, name in enumerate("zyx"):
        add("box", f"voxel size {name} (um)", T.voxel_size_zyx[axis], S.voxel_size_zyx[axis])
    add("box", "image volume (um^3)", T.image_volume_um3, S.image_volume_um3)

    add("endpoints", "inlets", T.inlet_count, S.inlet_count)
    add("endpoints", "outlets", T.outlet_count, S.outlet_count)
    for role, t_ends, s_ends in (("inlets", T.inlets, S.inlets), ("outlets", T.outlets, S.outlets)):
        t_faces, s_faces = _face_counts(t_ends), _face_counts(s_ends)
        for face in list(FACES) + ["interior"]:
            if face in t_faces or face in s_faces:
                add("endpoints", f"{role} on {face}", t_faces.get(face, 0), s_faces.get(face, 0), key="inlets or outlets on a face")
    for role in ROLES[2:]:
        add("endpoints", f"{role} nodes", getattr(T, f"{role}_count"), getattr(S, f"{role}_count"), key="boundary nodes of a role")
        if no_interior:
            # A tree leaf the template ended blind joins the bed instead.
            rows[-1] = not_held(rows[-1], "every tree leaf joins the bed: no dead ends inside the tissue")

    add("network", "diameter basis", T.diameter_basis, S.diameter_basis)
    add("network", "total vessel length (um)", T.total_length_um, S.total_length_um)
    add("network", "vessels", T.vessel_count, S.vessel_count)
    add("network", "nodes", T.node_count, S.node_count)
    add("network", "edges", T.edge_count, S.edge_count)
    add("network", "average degree", T.average_degree, S.average_degree)
    add("network", "branching points", T.branching_points, S.branching_points)
    add("network", "mean tortuosity", T.mean_tortuosity, S.mean_tortuosity)
    add("network", "daughter angle (degrees)", T.daughter_angle_deg, S.daughter_angle_deg)
    add("network", "Murray exponent, median fitted", T.murray_exponent, S.murray_exponent)
    t_iqr = T.murray_iqr or (None, None)
    s_iqr = S.murray_iqr or (None, None)
    add("network", "Murray exponent IQR low", t_iqr[0], s_iqr[0])
    add("network", "Murray exponent IQR high", t_iqr[1], s_iqr[1])
    add("network", "intercapillary distance mean (um)", T.intercapillary_distance_um, S.intercapillary_distance_um)
    add("network", "intercapillary distance median (um)", T.intercapillary_distance_median_um, S.intercapillary_distance_median_um)

    get = lambda target, name: None if target is None else getattr(target, name)  # noqa: E731
    for category in BRANCH_ORDER_CATEGORY_SEQUENCE:
        t_cls, s_cls = T.classes.get(category), S.classes.get(category)
        if t_cls is None and s_cls is None:
            continue
        section = f"class {VESSEL_CLASS_TITLES[category]}"
        kind = "capillary class" if category == "capillary" else "class"
        add(section, "vessels", get(t_cls, "count") or 0, get(s_cls, "count") or 0, kind=kind)
        add(section, "total length (um)", get(t_cls, "total_length_um") or 0.0, get(s_cls, "total_length_um") or 0.0, kind=kind)
        for name, attribute in (
            ("mean length (um)", "mean_length_um"),
            ("median length (um)", "median_length_um"),
            ("mean diameter (um)", "mean_diameter_um"),
            ("median diameter (um)", "median_diameter_um"),
            ("mean tortuosity", "mean_tortuosity"),
            ("median tortuosity", "median_tortuosity"),
            ("tortuosity P90", "p90_tortuosity"),
        ):
            add(section, name, get(t_cls, attribute), get(s_cls, attribute), kind=kind)

    t_orders = {o.label: o for o in T.orders}
    s_orders = {o.label: o for o in S.orders}
    labels = [o.label for o in T.orders] + [o.label for o in S.orders if o.label not in t_orders]
    for label in labels:
        t_o, s_o = t_orders.get(label), s_orders.get(label)
        kind = "capillary order" if (t_o or s_o).category == "capillary" else "tree order"
        section = f"order {label}"
        add(section, "vessels", get(t_o, "edge_count") if t_o else 0, get(s_o, "edge_count") if s_o else 0, kind=kind)
        for name, attribute in (
            ("mean length (um)", "mean_length_um"),
            ("mean tortuosity", "mean_tortuosity"),
            ("mean emergence angle (degrees)", "mean_emergence_angle_deg"),
            ("mean diameter (um)", "mean_diameter_um"),
            ("diameter CV", "diameter_cv"),
        ):
            add(section, name, get(t_o, attribute), get(s_o, attribute), kind=kind)

    add("loops", "independent loops", T.loops.independent_loops, S.loops.independent_loops)
    add("loops", "independent loops per mm^3", T.loops.independent_loops_per_mm3, S.loops.independent_loops_per_mm3)
    for name in ("P10", "P25", "Median", "P75", "P90"):
        add("loops", f"loop length {name} (um)", T.loops.length_percentiles_um.get(name), S.loops.length_percentiles_um.get(name))
    add("loops", "vessels on no loop (fraction)", T.loops.vessels_on_no_loop_fraction, S.loops.vessels_on_no_loop_fraction)

    # The generator draws one connected network: a template's artefact ends
    # and ends of pieces without a junction are never drawn, so its total is
    # reported, not banded; the kinds drawn are banded below.
    # A closed network draws none leaving the image either.
    undrawn = (
        bool(T.dead_ends.artefacts or T.dead_ends.island_ends)
        or (closed and bool(T.dead_ends.leaves_image))
        or (no_interior and bool(T.dead_ends.interior))
    )
    add("dead ends", "dead ends", T.dead_ends.total, S.dead_ends.total, key="dead ends, kinds not drawn" if undrawn else None)
    add("dead ends", "leaving the image", T.dead_ends.leaves_image, S.dead_ends.leaves_image)
    if closed:
        rows[-1] = not_drawn(rows[-1], "a closed network")
    interior_from = len(rows)
    add("dead ends", "inside the tissue", T.dead_ends.interior, S.dead_ends.interior)
    add("dead ends", "artefacts", T.dead_ends.artefacts, S.dead_ends.artefacts)
    add("dead ends", "ends of pieces without a junction", T.dead_ends.island_ends, S.dead_ends.island_ends)
    add("dead ends", "interior per mm of vessel", T.dead_ends.interior_per_mm, S.dead_ends.interior_per_mm)
    add("dead ends", "chain length median (um)", T.dead_ends.chain_length_median_um, S.dead_ends.chain_length_median_um)
    add("dead ends", "chain length P10 (um)", T.dead_ends.chain_length_p10_um, S.dead_ends.chain_length_p10_um)
    add("dead ends", "chain length P90 (um)", T.dead_ends.chain_length_p90_um, S.dead_ends.chain_length_p90_um)
    joined = dead_end_pairs_joined(T, options) if options else 0
    if joined:
        _against_joined_pairs(rows, T, joined)
    if no_interior:
        # Drawn without them: the interior rows say so (the artefact and
        # island rows are the template's kinds never drawn, as before).
        for i in range(interior_from, len(rows)):
            if rows[i].characteristic not in ("artefacts", "ends of pieces without a junction"):
                rows[i] = not_drawn(rows[i], "no dead ends inside the tissue")
    rows.extend(organisation_rows(T.organisation, S.organisation, T.extent_um, S.extent_um))
    if T.layout is not None or S.layout is not None:
        for characteristic, field in LAYOUT_CHARACTERISTICS:
            add("layout", characteristic, get(T.layout, field), get(S.layout, field))
    return rows


def _against_joined_pairs(rows: list, T: NetworkTemplate, joined: int) -> None:
    """The rows the dead ends joined in pairs change by design, judged
    against what *joined* pairs make of the template's values (shown in
    their tolerance; the template column keeps the template's own)."""
    cls = T.classes.get("capillary")
    expected: dict = {
        ("network", "vessels"): T.vessel_count - joined if T.vessel_count is not None else None,
        ("network", "edges"): T.edge_count - joined if T.edge_count is not None else None,
        ("network", "nodes"): T.node_count - 2 * joined if T.node_count is not None else None,
        ("loops", "independent loops"): (
            T.loops.independent_loops + joined if T.loops.independent_loops is not None else None
        ),
    }
    if T.edge_count and T.node_count and T.node_count > 2 * joined:
        expected[("network", "average degree")] = 2.0 * (T.edge_count - joined) / (T.node_count - 2 * joined)
    if T.loops.independent_loops and T.loops.independent_loops_per_mm3 is not None:
        expected[("loops", "independent loops per mm^3")] = (
            T.loops.independent_loops_per_mm3 * (T.loops.independent_loops + joined) / T.loops.independent_loops
        )
    if cls is not None and cls.count:
        title = f"class {VESSEL_CLASS_TITLES['capillary']}"
        expected[(title, "vessels")] = cls.count - joined
        if cls.total_length_um and cls.count > joined:
            expected[(title, "mean length (um)")] = cls.total_length_um / (cls.count - joined)
    for i, r in enumerate(rows):
        value = expected.get((r.section, r.characteristic))
        if value is None:
            continue
        kind = "capillary class" if r.section.startswith("class ") else r.section
        tolerance = TOLERANCES.get((kind, r.characteristic))
        synthetic = _number(r.synthetic)
        why = f"{joined} pairs of dead ends joined"
        rows[i] = replace(
            r,
            ratio=None if synthetic is None or value == 0 else synthetic / value,
            within=_within(float(value), synthetic, tolerance),
            tolerance=f"{_tolerance_text(tolerance) or 'reported'} of {value:.4g} ({why})",
        )


#: A layout is scored by correlation only where the template's capillary
#: vertices vary between cells (or planes) more than counting noise would:
#: variance over mean, 1 for vertices strewn at random. An even template has
#: no layout to match, and two noise fields' correlation is noise.
STRUCTURE_DISPERSION = 2.0


def _per_face(count: Optional[int], extent_um: Optional[Sequence[float]], axis: int) -> Optional[float]:
    """*count* runs along *axis* per mm^2 of the box's face across it; the
    count itself without the box."""
    if count is None:
        return None
    if extent_um is None:
        return float(count)
    others = [float(e) for i, e in enumerate(extent_um) if i != axis]
    return float(count) / (others[0] * others[1] / 1.0e6)


def dispersion(counts: Any) -> float:
    """Variance over mean of *counts* (0 for none)."""
    values = np.asarray(counts, dtype=float).ravel()
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.var()) / mean if mean > 0 else 0.0


#: The fewest runs along an axis the template must have for that axis's
#: run row to be scored: fewer is no pattern to match.
MIN_RUNS_SCORED = 5


def organisation_rows(
    T: Optional[Organisation],
    S: Optional[Organisation],
    t_extent_um: Optional[Sequence[float]] = None,
    s_extent_um: Optional[Sequence[float]] = None,
) -> list[ComparisonRow]:
    """Where the vessels lie and which way the capillaries run, template
    against synthetic: per-cell fields as correlations (on one grid: a box
    drawn at another scale has none), shares, runs along each axis (per mm^2
    of the face across it, given the boxes; else counts), coverage, the
    trunks' turn and the vessels' curvature."""
    rows: list[ComparisonRow] = []
    if T is None and S is None:
        return rows

    def add(characteristic: str, t: Any, s: Any, key: Optional[str] = None) -> None:
        rows.append(banded("organisation", "organisation", characteristic, t, s, key))

    same_grid = T is not None and S is not None and tuple(T.shape) == tuple(S.shape)
    even = "not held (the template's capillaries are even there: no layout to match)"
    laid_out = T is not None and dispersion(T.capillary_nodes) >= STRUCTURE_DISPERSION
    for name, field in (("capillary vertices by cell (correlation)", "capillary_nodes"), ("capillary length by cell (correlation)", "capillary_length_um")):
        r = correlation(getattr(T, field), getattr(S, field)) if same_grid else None
        add(name, 1.0 if T is not None else None, r)
        if T is not None and not laid_out:
            rows[-1] = unbanded(rows[-1], even)
    for axis, letter in enumerate("zyx"):
        r = None
        if same_grid:
            r = correlation(profile(T.field("capillary_length_um"), axis), profile(S.field("capillary_length_um"), axis))
        add(f"capillary length along {letter} (correlation)", 1.0 if T is not None else None, r,
            key="capillary length along an axis (correlation)")
        if T is not None and dispersion(profile(T.field("capillary_nodes"), axis)) < STRUCTURE_DISPERSION:
            rows[-1] = unbanded(rows[-1], even)
    t_trees = s_trees = None
    if same_grid:
        template_trees = T.field("arterial_length_um") + T.field("venous_length_um")
        synthetic_trees = S.field("arterial_length_um") + S.field("venous_length_um")
        mask = template_trees > 0
        t_trees, s_trees = share_inside(template_trees, mask), share_inside(synthetic_trees, mask)
    add("tree length in the template's tree cells (share)", t_trees, s_trees)
    get = lambda target, name: None if target is None else getattr(target, name)  # noqa: E731
    for axis, letter in enumerate("zyx"):
        t_share = None if T is None else T.axis_share[axis]
        s_share = None if S is None else S.axis_share[axis]
        add(f"capillary length within 30 degrees of {letter} (share)", t_share, s_share,
            key="capillary length along an axis (share)")
    for axis, letter in enumerate("zyx"):
        t_chord = None if T is None or T.chord_axis_share is None else T.chord_axis_share[axis]
        s_chord = None if S is None or S.chord_axis_share is None else S.chord_axis_share[axis]
        add(f"capillary chords within 30 degrees of {letter} (share)", t_chord, s_chord,
            key="capillary chords along an axis (share)")
    for axis, letter in enumerate("zyx"):
        t_runs = None if T is None or T.long_runs is None else T.long_runs[axis]
        s_runs = None if S is None or S.long_runs is None else S.long_runs[axis]
        name = f"capillary runs along {letter} spanning 100 um (per mm^2)"
        add(name, _per_face(t_runs, t_extent_um, axis), _per_face(s_runs, s_extent_um, axis),
            key="capillary runs along an axis spanning 100 um (per mm^2)")
        if t_runs is not None and t_runs < MIN_RUNS_SCORED:
            rows[-1] = not_held(rows[-1], f"the template has {t_runs} such runs: no pattern to match")
    add("box within 20 um of a capillary", get(T, "within_near"), get(S, "within_near"))
    add("box beyond 50 um of a capillary", get(T, "beyond_far"), get(S, "beyond_far"))
    add("trunk turn (degrees)", get(T, "trunk_turn_deg"), get(S, "trunk_turn_deg"))
    leaves = lambda target: None if target is None or target.tree_leaves_by_order is None else sum(  # noqa: E731
        count for _label, count in target.tree_leaves_by_order
    )
    add("tree leaves", leaves(T), leaves(S))
    add("trunk straightness, root to leaf", get(T, "trunk_straightness"), get(S, "trunk_straightness"))
    t_reach = {} if T is None or T.tree_reach_um is None else {s: (m, f) for s, m, f in T.tree_reach_um}
    s_reach = {} if S is None or S.tree_reach_um is None else {s: (m, f) for s, m, f in S.tree_reach_um}
    for side in sorted(set(t_reach) | set(s_reach)):
        for index, which in ((0, "median"), (1, "furthest")):
            add(f"{side} trees' reach, {which} leaf (um)",
                t_reach[side][index] if side in t_reach else None, s_reach[side][index] if side in s_reach else None,
                key="tree reach (um)")
    add("capillary curvature (degrees per 10 um)", get(T, "capillary_curvature_deg"), get(S, "capillary_curvature_deg"))
    add("tree curvature (degrees per 10 um)", get(T, "tree_curvature_deg"), get(S, "tree_curvature_deg"))
    median_vessel = lambda o: (  # noqa: E731
        None if o is None or not o.tree_curvature_by_vessel_deg else float(np.median(o.tree_curvature_by_vessel_deg))
    )
    add("tree curvature, median vessel (degrees per 10 um)", median_vessel(T), median_vessel(S))
    add(
        "tree curvature at 2 um steps, median vessel (degrees per 10 um)",
        get(T, "tree_fine_curvature_deg"), get(S, "tree_fine_curvature_deg"),
    )
    return rows


def compare_run(
    template: NetworkTemplate,
    G: nx.Graph,
    *,
    roles: Mapping[str, Sequence],
    sections: Optional[Mapping[str, Any]] = None,
    image_shape: Optional[Sequence[int]] = None,
    voxel_size_zyx: Optional[Sequence[float]] = None,
    members: Sequence[NetworkTemplate] = (),
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
    record = G.graph.get(GRAPH_TEMPLATE_RECORD) or {}
    rows = compare_templates(template, synthetic, options=record.get("options"))
    if members:
        rows = with_members_range(rows, [compare_templates(m, synthetic, options=record.get("options")) for m in members])
    return rows


def with_members_range(rows: Sequence[ComparisonRow], member_rows: Sequence[Sequence[ComparisonRow]]) -> list[ComparisonRow]:
    """*rows* with each characteristic's range over a batch template's
    members (their rows against the same synthetic network, B4b): the
    lowest and highest of their template values, and whether the synthetic
    value lies between them. A correlation row ranges over nothing: each
    member's own field is its reference."""
    values: dict[tuple[str, str], list[float]] = {}
    for member in member_rows:
        for r in member:
            number = _number(r.template)
            if number is not None:
                values.setdefault((r.section, r.characteristic), []).append(number)
    out = []
    for r in rows:
        found = values.get((r.section, r.characteristic))
        synthetic = _number(r.synthetic)
        if not found or "(correlation)" in r.characteristic:
            out.append(r)
            continue
        low, high = min(found), max(found)
        within = None if synthetic is None else (low - 1e-9 <= synthetic <= high + 1e-9)
        out.append(replace(r, members_low=low, members_high=high, within_members=within))
    return out


def comparison_summary(rows: Sequence[ComparisonRow]) -> dict:
    """``{"fitted", "within", "outside": [characteristics]}``, and, for a
    batch template, how many rows lie within its members' range."""
    fitted = [r for r in rows if r.within is not None]
    ranged = [r for r in rows if r.within_members is not None]
    return {
        "rows": len(rows),
        "fitted": len(fitted),
        "within": sum(1 for r in fitted if r.within),
        "outside": [f"{r.section}: {r.characteristic}" for r in fitted if not r.within],
        "ranged": len(ranged),
        "within_members": sum(1 for r in ranged if r.within_members),
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
        elif r.tolerance.startswith(("not drawn", "not held")):
            text += " " + r.tolerance
        if r.within_members is not None:
            text += (
                f"; members {_text(r.members_low)} to {_text(r.members_high)}: "
                + ("within" if r.within_members else "OUTSIDE")
            )
        out[f"{r.section}: {r.characteristic}"] = text
    summary = comparison_summary(rows)
    out["fitted characteristics within tolerance"] = f"{summary['within']} of {summary['fitted']}"
    if summary["ranged"]:
        out["characteristics within the members' range"] = f"{summary['within_members']} of {summary['ranged']}"
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
                "" if r.members_low is None else _numeric_csv_text(r.members_low),
                "" if r.members_high is None else _numeric_csv_text(r.members_high),
                "" if r.within_members is None else ("yes" if r.within_members else "no"),
            ])
    return dest


__all__ = [
    "COMPARISON_SECTION",
    "CSV_COLUMNS",
    "ComparisonRow",
    "TOLERANCES",
    "banded",
    "compare_run",
    "compare_templates",
    "comparison_section",
    "not_drawn",
    "not_held",
    "with_members_range",
    "unbanded",
    "organisation_rows",
    "comparison_summary",
    "row",
    "write_template_comparison_csv",
]
