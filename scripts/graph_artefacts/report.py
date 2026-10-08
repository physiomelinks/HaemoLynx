"""Measuring a graph against its mask, and the tables a comparison prints.

Every graph -- each step of each code version -- is measured with this
checkout's ``graph.diagnose_lumen_artefacts``, so two versions are compared
with one ruler.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

#: ``(key, column heading)`` in table order.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("edges", "edges"),
    ("centreline_um", "centreline um"),
    ("pairs", "pairs in one lumen"),
    ("forks", "of them forks"),
    ("doubled_um", "doubled centreline um"),
    ("loops", "short loops"),
    ("loops_in_lumen", "loops inside one lumen"),
    ("interior_dead_ends", "dead ends"),
    ("inside_other_lumen", "tip in another lumen"),
    ("beside_vessel", "beside another centreline"),
    ("off_mask", "mostly off the mask"),
    ("mask_continues", "mask runs on past tip"),
    ("short", "shorter than the radius rule"),
    ("isolated", "isolated edges"),
    ("components", "components"),
    ("mask_covered", "mask covered"),
)


def measure(G: nx.MultiGraph, support, *, stub_radius_multiple: float = 3.0, coverage: bool = False) -> dict[str, Any]:
    """One row: the graph's size and its lumen artefacts; with *coverage*, the
    share of mask voxels some centreline's lumen covers (slower)."""
    from haemolynx.graph import diagnose_lumen_artefacts

    report = diagnose_lumen_artefacts(
        G, support.mask, voxel_size_zyx=support.voxel_size_zyx, mask_support=support,
        stub_radius_multiple=stub_radius_multiple,
    )
    counts = report["dead_end_counts"]
    row: dict[str, Any] = {
        "edges": G.number_of_edges(),
        "centreline_um": report["centreline_um"],
        "pairs": report["duplicate_pair_count"],
        "forks": report["fork_pair_count"],
        "doubled_um": report["duplicated_centreline_um"],
        "loops": report["short_loop_count"],
        "loops_in_lumen": report["loops_inside_one_lumen"],
        "interior_dead_ends": report["interior_dead_ends"],
        "inside_other_lumen": counts["inside_other_lumen"],
        "beside_vessel": counts["beside_vessel"],
        "off_mask": counts["off_mask"],
        "mask_continues": counts["mask_continues"],
        "short": counts["short"],
        "isolated": report["isolated_single_edges"],
        "components": nx.number_connected_components(G) if G.number_of_nodes() else 0,
    }
    if coverage:
        from haemolynx.graph.mask_recovery import uncovered_mask_voxels

        total = int(np.count_nonzero(support.mask))
        row["mask_covered"] = 1.0 - len(uncovered_mask_voxels(G, support)) / total if total else 0.0
    return row


def _cell(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key == "mask_covered":
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:,.0f}"
    return f"{value:,}"


def _markdown(header: list[str], rows: Iterable[list[str]]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def steps_table(rows: Iterable[tuple[str, Mapping[str, Any]]]) -> str:
    """One row per step, ``(label, measure(...))``, one column per measure."""
    rows = list(rows)
    keys = [key for key, _heading in COLUMNS if any(key in row for _label, row in rows)]
    headings = dict(COLUMNS)
    return _markdown(
        ["step"] + [headings[key] for key in keys],
        ([label] + [_cell(key, row.get(key)) for key in keys] for label, row in rows),
    )


def comparison_table(sides: Mapping[str, Mapping[str, Any]]) -> str:
    """One row per measure, one column per side; with two sides, a third
    column with the change from the first to the second."""
    names = list(sides)
    header = ["measure"] + names + (["change"] if len(names) == 2 else [])
    rows = []
    for key, heading in COLUMNS:
        values = [sides[name].get(key) for name in names]
        if all(value is None for value in values):
            continue
        row = [heading] + [_cell(key, value) for value in values]
        if len(names) == 2:
            first, second = values
            if first is None or second is None:
                row.append("")
            elif key == "mask_covered":
                row.append(f"{100.0 * (second - first):+.1f} points")
            else:
                row.append(f"{second - first:+,.0f}")
        rows.append(row)
    return _markdown(header, rows)
