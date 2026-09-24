"""Per-junction bifurcation morphometry and per-branch-order aggregation.

Both live together because they share the same "parent vs. daughter" edge
classification (:func:`_iter_parent_daughter_junctions`,
:func:`_normalize_branch_order_tag`/:func:`_branch_order_sort_key`) --
:func:`compute_branch_order_statistics` itself calls
:func:`compute_emergence_angles_by_branch_order`, so splitting the two
groups into separate modules would need one to import from the other in
both directions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import csv
import re

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

from haemolynx.geometry import cumulative_lengths
from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE
from haemolynx.visualization.geometry import edge_polyline

from .csv_export import _numeric_csv_text


def _normalize_branch_order_tag(tag: Any) -> Optional[str]:
    """Normalize branch-order labels to Large_ArtN / ArtN / BON / VenN / Large_VenN."""
    if tag is None:
        return None
    label = str(tag).strip()
    if not label:
        return None
    m = re.match(
        r"^(large_art|large_ven|art|ven|bo|b)\s*0*(\d+)$", label, flags=re.IGNORECASE
    )
    if not m:
        return label
    prefix = m.group(1).lower()
    n = int(m.group(2))
    if prefix == "large_art":
        return f"Large_Art{n}"
    if prefix == "large_ven":
        return f"Large_Ven{n}"
    if prefix == "art":
        return f"Art{n}"
    if prefix == "ven":
        return f"Ven{n}"
    return f"BO{n}"


#: Sort-group rank per (already-normalized, see _normalize_branch_order_tag)
#: branch-order prefix, derived from graph.branch_order's own
#: BRANCH_ORDER_CATEGORY_SEQUENCE -- the single source of truth for which
#: category sorts before which -- rather than an independent copy of the
#: ranking that could silently drift from it (see that constant's own
#: docstring).
_CATEGORY_BY_PREFIX = {
    "large_art": "large_arteriole",
    "art": "arteriole",
    "bo": "capillary",
    "ven": "venule",
    "large_ven": "large_venule",
}
_BRANCH_ORDER_SORT_GROUPS = {
    prefix: BRANCH_ORDER_CATEGORY_SEQUENCE.index(category)
    for prefix, category in _CATEGORY_BY_PREFIX.items()
}

#: What :func:`branch_order_category` returns for a vessel with no (or an
#: unrecognised) branch-order label.
UNASSIGNED_CATEGORY = "unassigned"


def branch_order_category(tag: Any) -> str:
    """The vessel type a branch-order label belongs to -- one of
    ``BRANCH_ORDER_CATEGORY_SEQUENCE`` (large_arteriole, arteriole,
    capillary, venule, large_venule), or :data:`UNASSIGNED_CATEGORY`."""
    normalized = _normalize_branch_order_tag(tag)
    if normalized is None:
        return UNASSIGNED_CATEGORY
    m = re.match(r"^(large_art|large_ven|art|ven|bo)\s*\d+$", normalized, flags=re.IGNORECASE)
    return _CATEGORY_BY_PREFIX[m.group(1).lower()] if m else UNASSIGNED_CATEGORY


def branch_order_label(tag: Any) -> str:
    """A branch-order label normalised for grouping (``Art2``, ``BO3``, ...),
    or :data:`UNASSIGNED_CATEGORY` when there is none."""
    normalized = _normalize_branch_order_tag(tag)
    return normalized if normalized is not None else UNASSIGNED_CATEGORY


def _branch_order_sort_key(tag: str) -> tuple[int, int, str]:
    """Sort as Large_Art*, Art*, BO*, Ven*, Large_Ven*, then unknown labels."""
    m = re.match(
        r"^(large_art|large_ven|art|ven|bo)\s*(\d+)$", str(tag), flags=re.IGNORECASE
    )
    if not m:
        return (len(_BRANCH_ORDER_SORT_GROUPS), 0, str(tag))
    prefix = m.group(1).lower()
    n = int(m.group(2))
    group = _BRANCH_ORDER_SORT_GROUPS.get(prefix, len(_BRANCH_ORDER_SORT_GROUPS))
    return (group, n, str(tag))


def _incident_edge_items(
    G: Union[nx.Graph, nx.MultiGraph], node: Any
) -> list[tuple[Any, Any, Any, dict]]:
    """Incident edges as ``(u, v, key, data)`` with ``u`` equal to ``node``."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return list(G.edges(node, keys=True, data=True))
    return [(u, v, None, d) for u, v, d in G.edges(node, data=True)]


def _iter_parent_daughter_junctions(
    G: Union[nx.Graph, nx.MultiGraph],
):
    """Junctions with a unique lowest-branch-order-rank parent and >=1 daughter.

    Yields ``(node, parent_item, daughter_items)`` where each item is
    ``(u, v, key, data, tag)``. This is the same junction definition
    :func:`compute_emergence_angles_by_branch_order` uses (the unique
    lowest-rank incident edge is the parent, every other labelled edge a
    daughter; tied ranks, unlabelled-only, or degree < 3 contribute
    nothing) -- shared here so a second and third junction-level statistic
    (Murray's law, daughter-daughter angle) do not each redefine "parent"
    on their own.
    """
    for node in G.nodes():
        if int(G.degree(node)) < 3:
            continue
        labelled: list[tuple[Any, Any, Any, dict, str]] = []
        for u, v, key, data in _incident_edge_items(G, node):
            if u == v:
                continue
            tag = _normalize_branch_order_tag(data.get("branch_order"))
            if not tag:
                continue
            labelled.append((u, v, key, data, tag))
        if len(labelled) < 2:
            continue
        ranks = [_branch_order_sort_key(item[4]) for item in labelled]
        min_rank = min(ranks)
        parent_indices = [i for i, rank in enumerate(ranks) if rank == min_rank]
        if len(parent_indices) != 1:
            continue
        parent_i = parent_indices[0]
        daughter_items = [item for i, item in enumerate(labelled) if i != parent_i]
        yield node, labelled[parent_i], daughter_items


def _point_along_polyline(
    points: np.ndarray, distance_um: float
) -> Optional[np.ndarray]:
    """Interpolate a point this far along a polyline, clamped to its length."""
    lengths = cumulative_lengths(points)
    total = float(lengths[-1])
    if total <= 0.0:
        return None
    target = min(max(float(distance_um), 0.0), total)
    if target <= 0.0:
        for i in range(1, len(points)):
            if float(lengths[i]) > 0.0:
                return np.asarray(points[i], dtype=float)
        return None
    idx = int(np.searchsorted(lengths, target, side="left"))
    idx = min(max(idx, 1), len(points) - 1)
    t0 = float(lengths[idx - 1])
    t1 = float(lengths[idx])
    if t1 <= t0:
        return np.asarray(points[idx], dtype=float)
    frac = (target - t0) / (t1 - t0)
    start = np.asarray(points[idx - 1], dtype=float)
    end = np.asarray(points[idx], dtype=float)
    return start + frac * (end - start)


def _outgoing_unit_tangent(
    G: Union[nx.Graph, nx.MultiGraph],
    node: Any,
    u: Any,
    v: Any,
    data: dict,
    tangent_length_um: float,
) -> Optional[np.ndarray]:
    """Unit vector leaving ``node`` along this edge's local centreline."""
    try:
        points = edge_polyline(G, u, v, data)
    except (TypeError, ValueError):
        return None
    if node == u:
        path = points
    elif node == v:
        path = points[::-1]
    else:
        return None
    dest = _point_along_polyline(path, tangent_length_um)
    if dest is None:
        return None
    vec = np.asarray(dest, dtype=float) - np.asarray(path[0], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def _angle_between_unit_vectors(a: np.ndarray, b: np.ndarray) -> float:
    cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _empty_branch_order_record(tag: str) -> Dict[str, Any]:
    return {
        "Branch Order": tag,
        "Edge Count": 0,
        "Mean Length (microns)": 0.0,
        "Mean Tortuosity Index": "N/A (no position data)",
        "Tortuosity Sample Count": 0,
        "Mean Emergence Angle (degrees)": "N/A (no unique parent junction)",
        "Emergence Angle Sample Count": 0,
        "Mean Pressure Drop (Pa)": "N/A (no flow solved)",
        "Total Pressure Drop (Pa)": "N/A (no flow solved)",
        "Pressure Drop Sample Count": 0,
        "Mean Diameter (microns)": "N/A (no diameter assigned)",
        "Diameter Coefficient of Variation": "N/A (no diameter assigned)",
        "Diameter Sample Count": 0,
        "_diameter_sum_sq": 0.0,
    }


def compute_emergence_angles_by_branch_order(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Angle each daughter leaves its parent, grouped by the daughter's order.

    At a junction the parent is the unique incident edge with the lowest
    branch-order rank (Large_Art* before Art* before BO* before Ven* before
    Large_Ven*, then the numeric index).
    Each other labelled incident edge is a daughter. The emergence angle is
    the deflection of the daughter's outgoing centreline tangent from the
    parent's incoming tangent: 0° continues the parent, 90° leaves at a
    right angle.

    Junctions with no unique lowest-order parent (tied ranks, unlabelled
    edges only, or degree < 3) contribute nothing, so root segments have
    no emergence angle.

    Each junction is evaluated independently, but an edge that is a daughter
    at one end can also be a daughter at its other end -- neither endpoint's
    local minimum, the way a capillary bridging two comparable-order
    neighbourhoods looks, not a strict parent -> child step. That edge's
    emergence still counts once, at whichever of its two junctions is
    reached first, not once per end.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    # An edge with neither endpoint the local minimum rank (a capillary
    # bridging two comparable-order neighbourhoods, not a strict parent ->
    # child step) is a "daughter" at both of its junctions. Each junction is
    # processed independently, so without this, that one edge's emergence
    # would be added twice -- once per end -- to the same branch order's
    # sum/count. An edge id (not just (u, v): parallel edges between the same
    # two nodes are different edges) makes sure each edge contributes at most
    # once, no matter which of its two junctions is visited first.
    seen_daughters: set[tuple[Any, Any]] = set()

    for node in G.nodes():
        if int(G.degree(node)) < 3:
            continue
        labelled: list[tuple[Any, Any, Any, dict, str]] = []
        for u, v, key, data in _incident_edge_items(G, node):
            if u == v:
                continue
            tag = _normalize_branch_order_tag(data.get("branch_order"))
            if not tag:
                continue
            labelled.append((u, v, key, data, tag))
        if len(labelled) < 2:
            continue
        ranks = [_branch_order_sort_key(item[4]) for item in labelled]
        min_rank = min(ranks)
        parent_indices = [i for i, rank in enumerate(ranks) if rank == min_rank]
        if len(parent_indices) != 1:
            continue
        parent_i = parent_indices[0]
        p_u, p_v, _p_key, p_data, _parent_tag = labelled[parent_i]
        parent_out = _outgoing_unit_tangent(
            G, node, p_u, p_v, p_data, tangent_length_um
        )
        if parent_out is None:
            continue
        parent_in = -parent_out
        for i, (u, v, key, data, tag) in enumerate(labelled):
            if i == parent_i:
                continue
            edge_id = (frozenset((u, v)), key)
            if edge_id in seen_daughters:
                continue
            daughter_out = _outgoing_unit_tangent(
                G, node, u, v, data, tangent_length_um
            )
            if daughter_out is None:
                continue
            angle = _angle_between_unit_vectors(parent_in, daughter_out)
            sums[tag] = sums.get(tag, 0.0) + angle
            counts[tag] = counts.get(tag, 0) + 1
            seen_daughters.add(edge_id)

    ordered: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(counts, key=_branch_order_sort_key):
        n = counts[tag]
        ordered[tag] = {
            "Branch Order": tag,
            "Mean Emergence Angle (degrees)": sums[tag] / n,
            "Emergence Angle Sample Count": n,
        }
    return ordered


def _positive_diameter_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if value_f > 0 else None


def compute_murray_law_compliance(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    exponent: float = 3.0,
) -> Dict[str, Any]:
    """How well parent^n ~= sum(daughter^n) holds at each qualifying junction.

    Murray's law (the classic ``exponent=3``) predicts the diameter
    relationship a minimum-total-work vascular network would have at every
    bifurcation; a ratio of 1.0 is ideal, and systematic deviation from it
    is a standard marker of abnormal (e.g. tumour) vasculature. Uses the
    same junction definition as :func:`compute_emergence_angles_by_branch_order`
    (see :func:`_iter_parent_daughter_junctions`): the unique lowest-rank
    incident edge is the parent, every other labelled edge a daughter.

    A junction missing a positive diameter on the parent or on any one of
    its daughters is skipped entirely -- a ratio computed from a partial
    set of daughters is not comparable to one computed from all of them.
    """
    ratios: list[float] = []
    for _node, parent_item, daughter_items in _iter_parent_daughter_junctions(G):
        if not daughter_items:
            continue
        parent_d = _positive_diameter_or_none(parent_item[3].get("diameter_um"))
        if parent_d is None:
            continue
        daughter_sum = 0.0
        for daughter_item in daughter_items:
            daughter_d = _positive_diameter_or_none(daughter_item[3].get("diameter_um"))
            if daughter_d is None:
                daughter_sum = None
                break
            daughter_sum += daughter_d**exponent
        if daughter_sum is None:
            continue
        ratios.append(daughter_sum / (parent_d**exponent))

    if not ratios:
        return {
            "Mean Murray Ratio": "N/A (no junction had diameters on every branch)",
            "Murray Ratio Sample Count": 0,
            "Murray Law Exponent": exponent,
        }
    return {
        "Mean Murray Ratio": float(np.mean(ratios)),
        "Murray Ratio Sample Count": len(ratios),
        "Murray Law Exponent": exponent,
    }


def compute_daughter_daughter_angles(
    G: Union[nx.Graph, nx.MultiGraph],
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Any]:
    """Angle between each pair of sibling daughters at a bifurcation.

    Distinct from :func:`compute_emergence_angles_by_branch_order` (parent
    vs. each daughter): this is the angle between the daughters themselves,
    the other half of classic bifurcation morphometry. Uses the same
    junction definition (see :func:`_iter_parent_daughter_junctions`) so a
    junction only contributes here if it also had a well-defined parent,
    keeping the two angles comparable measurements of the same events.

    A junction with more than two daughters (a trifurcation or wider)
    contributes one angle per pair, not just adjacent ones -- there is no
    inherent ordering among daughters at the same junction.
    """
    angles: list[float] = []
    for node, _parent_item, daughter_items in _iter_parent_daughter_junctions(G):
        if len(daughter_items) < 2:
            continue
        tangents: list[np.ndarray] = []
        for u, v, _key, data, _tag in daughter_items:
            tangent = _outgoing_unit_tangent(G, node, u, v, data, tangent_length_um)
            if tangent is not None:
                tangents.append(tangent)
        for i in range(len(tangents)):
            for j in range(i + 1, len(tangents)):
                angles.append(_angle_between_unit_vectors(tangents[i], tangents[j]))

    if not angles:
        return {
            "Mean Daughter-Daughter Angle (degrees)": (
                "N/A (no bifurcation with two measurable daughters)"
            ),
            "Daughter-Daughter Angle Sample Count": 0,
        }
    return {
        "Mean Daughter-Daughter Angle (degrees)": float(np.mean(angles)),
        "Daughter-Daughter Angle Sample Count": len(angles),
    }


def compute_branch_order_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    *,
    tangent_length_um: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute mean length, tortuosity, and emergence angle per branch order.

    Returns a dictionary keyed by branch-order label with a compact summary.
    """
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)

    by_tag: Dict[str, Dict[str, Any]] = {}
    for item in edge_iter:
        u, v = item[0], item[1]
        data = item[-1]
        normalized_tag = _normalize_branch_order_tag(data.get("branch_order"))
        if not normalized_tag:
            continue

        length = data.get("length", 0.0)
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            continue
        if length_f <= 0:
            continue

        if normalized_tag not in by_tag:
            by_tag[normalized_tag] = _empty_branch_order_record(normalized_tag)
        rec = by_tag[normalized_tag]
        rec["Edge Count"] += 1
        rec["Mean Length (microns)"] += length_f

        if (
            node_positions is not None
            and u in node_positions
            and v in node_positions
        ):
            pos_u = np.array(node_positions[u], dtype=float)
            pos_v = np.array(node_positions[v], dtype=float)
            straight = euclidean(pos_u, pos_v)
            if straight > 0:
                tort = length_f / straight
                if rec["Mean Tortuosity Index"] == "N/A (no position data)":
                    rec["Mean Tortuosity Index"] = 0.0
                rec["Mean Tortuosity Index"] += tort
                rec["Tortuosity Sample Count"] += 1

        # Only present once haemodynamics has run and flow has been solved
        # (haemodynamics.resistance.set_edge_flows). The sign of a single
        # edge's drop is an artefact of its arbitrary (u, v) storage order,
        # not physically meaningful, so this aggregates magnitude -- "how
        # much of the network's total pressure loss happens in this order",
        # the classic answer being mostly small arterioles, not capillaries.
        pressure_drop = data.get("pressure_drop")
        if pressure_drop is not None:
            try:
                drop_abs = abs(float(pressure_drop))
            except (TypeError, ValueError):
                drop_abs = None
            if drop_abs is not None:
                if rec["Mean Pressure Drop (Pa)"] == "N/A (no flow solved)":
                    rec["Mean Pressure Drop (Pa)"] = 0.0
                    rec["Total Pressure Drop (Pa)"] = 0.0
                rec["Mean Pressure Drop (Pa)"] += drop_abs
                rec["Total Pressure Drop (Pa)"] += drop_abs
                rec["Pressure Drop Sample Count"] += 1

        # Only present once haemodynamics has assigned a diameter
        # (haemodynamics.poiseuille), whether measured (FWHM) or from the
        # branch-order table. Caliber distribution by generation is one of
        # the most basic readouts in vascular morphometry, and this data
        # already carries it -- it was just never reported on its own.
        diameter = data.get("diameter_um")
        if diameter is not None:
            try:
                diameter_f = float(diameter)
            except (TypeError, ValueError):
                diameter_f = None
            if diameter_f is not None and diameter_f > 0:
                if rec["Mean Diameter (microns)"] == "N/A (no diameter assigned)":
                    rec["Mean Diameter (microns)"] = 0.0
                rec["Mean Diameter (microns)"] += diameter_f
                rec["_diameter_sum_sq"] += diameter_f * diameter_f
                rec["Diameter Sample Count"] += 1

    for rec in by_tag.values():
        edge_count = int(rec["Edge Count"])
        if edge_count > 0:
            rec["Mean Length (microns)"] = rec["Mean Length (microns)"] / edge_count
        t_samples = int(rec["Tortuosity Sample Count"])
        if t_samples > 0 and isinstance(rec["Mean Tortuosity Index"], (int, float)):
            rec["Mean Tortuosity Index"] = rec["Mean Tortuosity Index"] / t_samples
        elif t_samples == 0:
            rec["Mean Tortuosity Index"] = "N/A (insufficient position data)"
        p_samples = int(rec["Pressure Drop Sample Count"])
        if p_samples > 0 and isinstance(rec["Mean Pressure Drop (Pa)"], (int, float)):
            rec["Mean Pressure Drop (Pa)"] = rec["Mean Pressure Drop (Pa)"] / p_samples
        elif p_samples == 0:
            rec["Mean Pressure Drop (Pa)"] = "N/A (no flow solved)"
            rec["Total Pressure Drop (Pa)"] = "N/A (no flow solved)"

    emergence = compute_emergence_angles_by_branch_order(
        G, tangent_length_um=tangent_length_um
    )
    for tag, emergence_rec in emergence.items():
        if tag not in by_tag:
            by_tag[tag] = _empty_branch_order_record(tag)
        rec = by_tag[tag]
        rec["Mean Emergence Angle (degrees)"] = emergence_rec[
            "Mean Emergence Angle (degrees)"
        ]
        rec["Emergence Angle Sample Count"] = emergence_rec[
            "Emergence Angle Sample Count"
        ]

    # After the emergence merge, not before: a tag with no length>0 edges but
    # a real emergence angle gets its record created for the first time
    # right above, and _diameter_sum_sq (an internal accumulator, never
    # meant to reach the caller) must be dropped from every record, not just
    # the ones the main loop already saw.
    for rec in by_tag.values():
        d_samples = int(rec["Diameter Sample Count"])
        if d_samples > 0 and isinstance(rec["Mean Diameter (microns)"], (int, float)):
            mean_d = rec["Mean Diameter (microns)"] / d_samples
            variance = max(0.0, rec["_diameter_sum_sq"] / d_samples - mean_d * mean_d)
            rec["Mean Diameter (microns)"] = mean_d
            rec["Diameter Coefficient of Variation"] = (
                (variance**0.5) / mean_d if mean_d > 0 else 0.0
            )
        else:
            rec["Mean Diameter (microns)"] = "N/A (no diameter assigned)"
            rec["Diameter Coefficient of Variation"] = "N/A (no diameter assigned)"
        del rec["_diameter_sum_sq"]

    ordered = {
        k: by_tag[k]
        for k in sorted(by_tag.keys(), key=_branch_order_sort_key)
    }
    return ordered


def export_branch_order_statistics_to_csv(
    branch_order_stats: Dict[str, Dict[str, Any]],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export per-branch-order summary statistics to readable CSV."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Branch Order",
                "Edge Count",
                "Mean Length (microns)",
                "Mean Tortuosity Index",
                "Mean Emergence Angle (degrees)",
                "Mean Pressure Drop (Pa)",
                "Total Pressure Drop (Pa)",
                "Mean Diameter (microns)",
                "Diameter Coefficient of Variation",
                "Notes",
            ]
        )
        writer.writerow(
            [
                "# Ordered by vessel class",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Large_Art* first, then Art*, then BO*, then Ven*, then Large_Ven*.",
            ]
        )
        for branch_tag in sorted(branch_order_stats.keys(), key=_branch_order_sort_key):
            rec = branch_order_stats[branch_tag]
            mean_len_s = _numeric_csv_text(rec.get("Mean Length (microns)", 0.0))

            mean_tort = rec.get("Mean Tortuosity Index", "N/A")
            mean_tort_s = _numeric_csv_text(mean_tort)
            notes = [
                "Mean tortuosity is path length / straight distance."
                if isinstance(mean_tort, (int, float, np.integer, np.floating))
                else "Tortuosity unavailable (missing/insufficient node positions)."
            ]

            mean_angle = rec.get(
                "Mean Emergence Angle (degrees)",
                "N/A (no unique parent junction)",
            )
            mean_angle_s = _numeric_csv_text(mean_angle)
            notes.append(
                "Mean emergence angle is the deflection from the unique "
                "lower-order parent (0 degrees = collinear)."
                if isinstance(mean_angle, (int, float, np.integer, np.floating))
                else "Emergence angle unavailable (no unique lower-order "
                "parent junction)."
            )

            mean_drop = rec.get("Mean Pressure Drop (Pa)", "N/A (no flow solved)")
            total_drop = rec.get("Total Pressure Drop (Pa)", "N/A (no flow solved)")
            mean_drop_s = _numeric_csv_text(mean_drop)
            total_drop_s = _numeric_csv_text(total_drop)
            notes.append(
                "Pressure drop is |pressure_u - pressure_v|, only present "
                "once flow has been solved; total is this order's share "
                "of the network's overall pressure loss."
                if isinstance(mean_drop, (int, float, np.integer, np.floating))
                else "Pressure drop unavailable (flow not solved)."
            )

            mean_diam = rec.get("Mean Diameter (microns)", "N/A (no diameter assigned)")
            diam_cv = rec.get(
                "Diameter Coefficient of Variation", "N/A (no diameter assigned)"
            )
            mean_diam_s = _numeric_csv_text(mean_diam)
            diam_cv_s = _numeric_csv_text(diam_cv)
            notes.append(
                "Diameter coefficient of variation is its standard "
                "deviation / mean across this order's edges."
                if isinstance(mean_diam, (int, float, np.integer, np.floating))
                else "Diameter unavailable (no diameter assigned)."
            )
            writer.writerow(
                [
                    branch_tag,
                    int(rec.get("Edge Count", 0)),
                    mean_len_s,
                    mean_tort_s,
                    mean_angle_s,
                    mean_drop_s,
                    total_drop_s,
                    mean_diam_s,
                    diam_cv_s,
                    " ".join(notes),
                ]
            )

    return output_path
