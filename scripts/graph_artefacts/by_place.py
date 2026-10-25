"""Where a graph reaches the skeleton it was built from, by place in the stack
(``PLAN_surface_coverage.md``, Phase 0): the surface slab, beside the
large-vessel masks, and the thick-vessel region -- and, for each piece the
graph still loses, why the skeleton re-trace leaves it
(``graph.judge_lost_skeleton``).

"Reached" is the re-trace's own measure: a centreline within
``skeleton_retrace.REACH_UM`` of the skeleton voxel. The surface slab is the
first *surface_depth_um* of the stack (E14.5's pia lies at z = 0); "beside
the large vessels" is within *near_large_um* of the large-vessel masks.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

#: The rows of a table, in order: the surface slab and the rest, each beside
#: the large vessels or not. Without large-vessel masks only the first word.
SURFACE, DEEP = "surface", "deep"
BESIDE_LARGE, OTHER = "beside the large vessels", "other"

#: Why a lost voxel's piece is not drawn back, in table order: the
#: re-trace's ``REFUSED_*`` reasons, grouped, and "would be drawn" (None).
WHY_LOST: tuple[tuple[str, str], ...] = (
    ("short", "pieces under 20 um"),
    ("off_mask", "under half in the mask"),
    ("beside", "beside a kept vessel"),
    ("one_place", "meets the graph once"),
    ("small_hole", "round a hole <= the limit"),
    ("refused", "other refusals"),
    ("drawn", "would be drawn"),
)


def _why(reason: str | None) -> str:
    from haemolynx.graph import skeleton_retrace as sr

    return {
        None: "drawn",
        sr.REFUSED_SMALL: "short",
        sr.REFUSED_OFF_MASK: "off_mask",
        sr.REFUSED_BESIDE: "beside",
        sr.REFUSED_SAME_LUMEN: "beside",
        sr.REFUSED_ONE_END: "one_place",
        sr.REFUSED_NO_LOOP: "one_place",
        sr.REFUSED_SMALL_HOLE: "small_hole",
    }.get(reason, "refused")


def places_of(
    voxels: np.ndarray,
    voxel_size_zyx,
    *,
    surface_depth_um: float = 40.0,
    large_mask: np.ndarray | None = None,
    near_large_um: float = 10.0,
) -> np.ndarray:
    """Each voxel's row label (``"surface, beside the large vessels"``, ...)."""
    from haemolynx.preprocessing.skeleton_consistency import _distance_to_region_um

    spacing = np.asarray(voxel_size_zyx, dtype=float)
    voxels = np.asarray(voxels).reshape(-1, 3)
    depth = np.where(voxels[:, 0] * spacing[0] < float(surface_depth_um), SURFACE, DEEP)
    if large_mask is None:
        return depth.astype(object)
    near = _distance_to_region_um(voxels, large_mask, spacing) <= float(near_large_um)
    side = np.where(near, BESIDE_LARGE, OTHER)
    return np.array([f"{d}, {s}" for d, s in zip(depth, side)], dtype=object)


def row_order(large_masks: bool) -> list[str]:
    if not large_masks:
        return [SURFACE, DEEP]
    return [f"{d}, {s}" for d in (SURFACE, DEEP) for s in (BESIDE_LARGE, OTHER)]


def reach_by_place(
    G,
    skeleton: np.ndarray,
    support,
    *,
    surface_depth_um: float = 40.0,
    large_mask: np.ndarray | None = None,
    near_large_um: float = 10.0,
    thick: np.ndarray | None = None,
    keep_hole_over_um: float | None = None,
    judge: bool = True,
) -> dict[str, Any]:
    """Skeleton voxels reached and lost, per place (:func:`places_of`), the
    lost ones by why their piece stays lost (:data:`WHY_LOST`, read from
    ``judge_lost_skeleton`` against *G* as it stands; *judge* False skips
    it), and in the *thick* region apart. ``{"rows": {place: {...}},
    "order": [...], "pieces": [...]}``; an ``"all"`` row sums the rest."""
    from haemolynx.graph import judge_lost_skeleton
    from haemolynx.graph.lumen_loops import DEFAULT_SHORT_LOOP_KEEP_HOLE_UM
    from haemolynx.graph.skeleton_retrace import lost_skeleton_pieces, piece_length_um

    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxels = np.argwhere(np.asarray(skeleton, dtype=bool))
    hole = DEFAULT_SHORT_LOOP_KEEP_HOLE_UM if keep_hole_over_um is None else float(keep_hole_over_um)
    if judge:
        verdicts = judge_lost_skeleton(G, skeleton, support, keep_hole_over_um=hole)
        pieces = [(v.piece, _why(v.reason), v) for v in verdicts]
    else:
        pieces = [(p, None, None) for p in lost_skeleton_pieces(G, skeleton, spacing)]
    index = {tuple(v): i for i, v in enumerate(voxels)}
    lost = np.zeros(len(voxels), dtype=bool)
    why = np.full(len(voxels), "", dtype=object)
    piece_rows = []
    for piece, reason, verdict in pieces:
        rows = [index[tuple(v)] for v in piece.voxels]
        lost[rows] = True
        why[rows] = reason or ""
        if verdict is not None and reason != "short":
            centre = piece.voxels[len(piece.voxels) // 2]
            piece_rows.append({
                "voxels": len(piece.voxels),
                "length_um": round(piece_length_um(piece, spacing), 2),
                "why": reason,
                "reason": verdict.reason or "",
                "hole_um": None if verdict.hole_um is None else round(float(verdict.hole_um), 2),
                "far_fraction": round(verdict.far_fraction, 3),
                "mask_fraction": round(verdict.mask_fraction, 3),
                "centre_zyx_vox": tuple(int(c) for c in centre),
            })
    place = places_of(voxels, spacing, surface_depth_um=surface_depth_um, large_mask=large_mask,
                      near_large_um=near_large_um)
    in_thick = (np.asarray(thick, dtype=bool)[tuple(voxels.T)] if thick is not None
                else np.zeros(len(voxels), dtype=bool))
    order = row_order(large_mask is not None)
    rows: dict[str, dict[str, Any]] = {}
    for name in [*order, "all"]:
        sel = np.ones(len(voxels), dtype=bool) if name == "all" else place == name
        row: dict[str, Any] = {
            "skeleton_voxels": int(sel.sum()),
            "reached": float(np.mean(~lost[sel])) if sel.any() else None,
            "lost": int((sel & lost).sum()),
            "thick_voxels": int((sel & in_thick).sum()),
            "thick_reached": float(np.mean(~lost[sel & in_thick])) if (sel & in_thick).any() else None,
        }
        if judge:
            row.update({key: int((sel & lost & (why == key)).sum()) for key, _heading in WHY_LOST})
        rows[name] = row
    return {"rows": rows, "order": [*order, "all"], "pieces": piece_rows, "judged": judge}


def _percent(value) -> str:
    return "" if value is None else f"{100.0 * float(value):.1f}%"


def reach_table(report: Mapping[str, Any]) -> str:
    """One row per place: skeleton voxels, reached, lost, the lost by why,
    and the thick region's share."""
    header = ["place", "skeleton voxels", "reached", "lost"]
    if report.get("judged"):
        header += [heading for _key, heading in WHY_LOST]
    header += ["in the thick region", "of them reached"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in report["order"]:
        row = report["rows"][name]
        cells = [name, f"{row['skeleton_voxels']:,}", _percent(row["reached"]), f"{row['lost']:,}"]
        if report.get("judged"):
            cells += [f"{row[key]:,}" for key, _heading in WHY_LOST]
        cells += [f"{row['thick_voxels']:,}", _percent(row["thick_reached"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def reach_comparison(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """The reach of each place, one column per side."""
    names = list(reports)
    order: Iterable[str] = next(iter(reports.values()))["order"] if reports else []
    lines = ["| place | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for place in order:
        lines.append("| " + place + " | " + " | ".join(
            _percent(reports[n]["rows"][place]["reached"]) for n in names) + " |")
    return "\n".join(lines)


def by_place_section(
    graphs: Mapping[str, Any],
    skeleton: np.ndarray,
    support,
    *,
    large: np.ndarray | None,
    thick: np.ndarray | None,
    settings: Mapping[str, Any],
    surface_depth_um: float = 40.0,
    near_large_um: float = 10.0,
) -> tuple[str, list[dict], list[dict]]:
    """``(markdown, lost pieces, dropped pieces)``: the reach of each graph
    of *graphs* (column -> graph) by place, a table per graph of what it
    loses and why, and -- with a *thick* region -- the centreline the thick
    gate drops, measured once on *skeleton*."""
    from haemolynx.preprocessing import (
        diagnose_centreline_dropped_by_thick_gate,
        format_centreline_dropped_by_thick_gate_report,
    )

    keep_hole = float(settings.get("short_loop_keep_hole_over_um") or 0.0) or None
    reports = {
        column: reach_by_place(
            G, skeleton, support, surface_depth_um=surface_depth_um, large_mask=large,
            near_large_um=near_large_um, thick=thick, keep_hole_over_um=keep_hole,
        )
        for column, G in graphs.items()
    }
    where = f"the first {surface_depth_um:g} um of the stack" + (
        f"; beside the large vessels = within {near_large_um:g} um of their masks" if large is not None
        else "; no large-vessel masks")
    lines = ["## Skeleton reached, by place", "",
             f"A skeleton voxel is reached when a centreline comes within 3 um of it. Surface = {where}.", "",
             reach_comparison(reports)]
    pieces = []
    for column, report in reports.items():
        lines += ["", f"### {column}", "", reach_table(report)]
        pieces += [{"side": column, **row} for row in report["pieces"]]
    dropped: list[dict] = []
    if thick is not None:
        shell = float(settings.get("skeleton_thick_vessel_flake_filter_um") or 3.0)
        report = diagnose_centreline_dropped_by_thick_gate(
            skeleton, support.mask, thick, voxel_size_zyx=support.voxel_size_zyx, large_mask=large, shell_um=shell,
        )
        lines += ["", "## Centreline the thick gate drops", "",
                  "Plain Lee of the hole-filled mask against the run's skeleton (pieces of 10 voxels or more, "
                  "more than 3 um from it). Grid P (2026-10-18) called 25 of 30 such pieces the large vessel's "
                  "own lumen or wall.", "",
                  format_centreline_dropped_by_thick_gate_report(report, shell_um=shell), "",
                  dropped_table(report, surface_depth_um=surface_depth_um,
                                voxel_size_zyx=support.voxel_size_zyx, shell_um=shell)]
        dropped = report["pieces"]
    return "\n".join(lines), pieces, dropped


def dropped_table(report: Mapping[str, Any], *, surface_depth_um: float, voxel_size_zyx, shell_um: float) -> str:
    """``diagnose_centreline_dropped_by_thick_gate``'s pieces by where they
    lie, in the surface slab and below it."""
    places = ("thick_region", "shell", "beyond")
    headings = ("inside the thick region", f"within {shell_um:g} um of it", "beyond")
    z_um = float(voxel_size_zyx[0])
    sums = {slab: dict.fromkeys(places, 0.0) | {"large": 0.0, "pieces": 0} for slab in (SURFACE, DEEP)}
    for piece in report.get("pieces", []):
        slab = SURFACE if piece["centre_zyx_vox"][0] * z_um < float(surface_depth_um) else DEEP
        for place in places:
            sums[slab][place] += piece[f"{place}_um"]
        sums[slab]["large"] += piece.get("inside_large_um") or 0.0
        sums[slab]["pieces"] += 1
    header = ["slab", "pieces", *[f"um {h}" for h in headings], "um inside the large masks"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for slab, row in sums.items():
        lines.append("| " + " | ".join(
            [slab, str(row["pieces"]), *[f"{row[p]:,.0f}" for p in places], f"{row['large']:,.0f}"]) + " |")
    return "\n".join(lines)
