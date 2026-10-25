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
    ("bridge_pairs", "of them with a thick-vessel bridge"),
    ("bridges", "thick-vessel bridges"),
    ("doubled_um", "doubled centreline um"),
    ("doubled_long_runs", "doubled runs >= 5 um"),
    ("doubled_long_um", "um in them"),
    ("doubled_chain_um", "doubled um on dead-end chains"),
    ("doubled_fork_um", "at forks"),
    ("doubled_mid_um", "mid-edge"),
    ("doubled_bridge_um", "with a thick-vessel bridge"),
    ("loops", "short loops"),
    ("loops_doubled_side", "of them with a doubled side"),
    ("loops_through_bridge", "through a thick-vessel bridge"),
    ("loops_triangle", "of 3 edges or fewer"),
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
    from haemolynx.graph import IS_ZERO_RESISTANCE, diagnose_lumen_artefacts

    report = diagnose_lumen_artefacts(
        G, support.mask, voxel_size_zyx=support.voxel_size_zyx, mask_support=support,
        stub_radius_multiple=stub_radius_multiple,
    )
    counts = report["dead_end_counts"]

    def bridge(edge) -> bool:
        return G.has_edge(*edge) and bool(G.edges[edge].get(IS_ZERO_RESISTANCE))

    row: dict[str, Any] = {
        "edges": G.number_of_edges(),
        "centreline_um": report["centreline_um"],
        "pairs": report["duplicate_pair_count"],
        "forks": report["fork_pair_count"],
        # The split's bridges run inside the fat lumen: a pair with one is
        # what Phase 6 of the one-centreline-per-lumen plan is about.
        "bridge_pairs": sum(bridge(a) or bridge(b) for a, b in report["duplicate_pairs"]),
        "bridges": sum(bool(d.get(IS_ZERO_RESISTANCE)) for *_, d in G.edges(data=True)),
        "doubled_um": report["duplicated_centreline_um"],
        "doubled_long_runs": report["doubled_long_runs"]["runs"],
        "doubled_long_um": report["doubled_long_runs"]["um"],
        "doubled_chain_um": report["doubled_runs_by_place"]["dead_end_chain"]["um"],
        "doubled_fork_um": report["doubled_runs_by_place"]["fork"]["um"],
        "doubled_mid_um": report["doubled_runs_by_place"]["mid_edge"]["um"],
        "doubled_bridge_um": report["doubled_runs_by_place"]["bridge"]["um"],
        "loops": report["short_loop_count"],
        "loops_doubled_side": report["short_loops_by_side"]["doubled_side"],
        "loops_through_bridge": report["short_loops_by_side"]["through_bridge"],
        "loops_triangle": report["short_loops_by_side"]["triangle"],
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


def bridge_shapes(G: nx.MultiGraph, thick: np.ndarray, mask: np.ndarray, voxel_size_zyx) -> dict[str, Any]:
    """What shape each thick-vessel bridge of *G* has, by this checkout's
    ``graph.diagnose_thick_vessel_bridges`` (PLAN_thick_vessel_bridges.md,
    Phase 0), against the tree of *thick* found again here."""
    from haemolynx.graph import diagnose_thick_vessel_bridges

    return diagnose_thick_vessel_bridges(G, thick, None, mask, voxel_size_zyx=tuple(voxel_size_zyx))


def bridge_shapes_table(sides: Mapping[str, Mapping[str, Any]]) -> str:
    """One row per bridge shape, one column per side: how many, and their
    median length; then length over the lumen radius, and the bridges drawn
    along their path."""
    from haemolynx.graph.thick_vessel_bridge_shapes import BRIDGE_SHAPES, SHAPE_WORDS

    names = list(sides)
    rows = []
    for shape in BRIDGE_SHAPES:
        cells = []
        for name in names:
            count = sides[name]["counts"].get(shape, 0)
            median = sides[name]["median_length_um"].get(shape)
            cells.append(f"{count:,}" + (f" ({median:.1f} um)" if median is not None else ""))
        rows.append([SHAPE_WORDS[shape]] + cells)
    rows.append(["all bridges"] + [f"{sides[name]['bridge_count']:,}" for name in names])
    ratio_cells = []
    for name in names:
        ratio = sides[name]["length_over_radius"]
        ratio_cells.append("" if ratio["median"] is None else f"{ratio['median']:.2f} ({ratio['p90']:.2f})")
    rows.append(["length / lumen radius, median (90th pct)"] + ratio_cells)
    rows.append(["drawn along their path"] + [f"{sides[name]['drawn_path_count']:,}" for name in names])
    return _markdown(["bridge shape (count, median length)"] + names, rows)


#: The raw image's cost to a build (``build_side.raw_cost``), ``(key, heading,
#: format)`` in table order: the build without the raw image against the same
#: build with it (``PLAN_raw_confirmation.md``, decision D6: +5 % build time,
#: +50 MB resident on E14.5).
RAW_COST_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("seconds", "build s", "{:,.1f}"),
    ("peak_rss_mb", "peak resident MB", "{:,.0f}"),
    ("raw_seconds", "raw image loaded and calibrated, s", "{:,.2f}"),
    ("raw_held_mb", "raw image held in RAM, MB", "{:,.0f}"),
    ("raw_queries", "raw windows the rules read", "{:,}"),
    ("raw_voxels_read", "raw voxels they read", "{:,}"),
    ("raw_read_seconds", "raw reading s", "{:,.2f}"),
)


def raw_cost_row(side: Mapping[str, Any]) -> dict[str, Any]:
    """One build's ``side.json`` as a :data:`RAW_COST_COLUMNS` row; a build
    without the raw image has no raw columns."""
    row: dict[str, Any] = {"seconds": side.get("seconds"), "peak_rss_mb": side.get("peak_rss_mb")}
    raw = side.get("raw")
    if raw:
        row.update(
            raw_seconds=raw.get("seconds"), raw_held_mb=raw.get("held_mb"), raw_queries=raw.get("queries"),
            raw_voxels_read=raw.get("voxels_read"), raw_read_seconds=raw.get("read_seconds"),
        )
    return row


def raw_cost_table(costs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """Per side, ``costs[side] = {"without": side.json, "with": side.json}``:
    one row per :data:`RAW_COST_COLUMNS` measure, a column for each build and
    the change the raw image made (build time as a percentage too)."""
    header = ["measure"]
    for name in costs:
        header += [f"{name} without", f"{name} with", "change"]
    rows = []
    for key, heading, form in RAW_COST_COLUMNS:
        row = [heading]
        for pair in costs.values():
            without, with_raw = raw_cost_row(pair["without"]).get(key), raw_cost_row(pair["with"]).get(key)
            row += ["" if v is None else form.format(v) for v in (without, with_raw)]
            if without is None or with_raw is None:
                row.append("")
            elif key == "seconds" and without:
                row.append(f"{with_raw - without:+,.1f} ({100.0 * (with_raw - without) / without:+.1f}%)")
            else:
                row.append(form.replace("{:", "{:+").format(with_raw - without))
        rows.append(row)
    return _markdown(header, rows)
