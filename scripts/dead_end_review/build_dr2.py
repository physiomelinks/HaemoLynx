#!/usr/bin/env python3
"""Build grid DR2: does a dead end's vessel go on past its tip, and does E4's route follow it?

``PLAN_dead_ends_raw.md``, decision R6 (Harvey, 2026-10-25: build E4 and a
fresh grid to judge it on). On DR1 the walls' straight run past the tip (E3)
said "goes on" at 15/21 leave-one-out; its misses were vessels whose walls
leave the straight line, the vessel turning. E4
(``RawEvidenceEndothelial.route_ahead``) follows the walls round the turn.
DR2 judges it on dead ends no grid has shown, DR1's included:

- the first question is DR1's second: does the vessel go on past the tip
  (goes on / ends here / unsure), answered with the route hidden;
- the second asks whether the traced route, drawn when "Show the traced route"
  is ticked, is right: it follows the vessel and stops where the vessel stops
  (R), it goes astray (A), it stops short of where the vessel goes on (S), or
  unsure.

The rules the first question checks were fixed before the grid was built
(``fit_marks.FRESH_RULES["DR2"]``): E3's run over 11 um (fitted on DR1), and
E4's run over the same cut and E4 reaching the limit or another vessel (not
fitted). The examples are spread across E4's run in thirds and, within each,
evenly across how much further E4 went than E3's straight line, then
shuffled; every measure stays in the key.

Each card shows DR1's three planes. The third, past the tip, is turned
about the heading into the plane the route turns in when it turns 10 deg or
more, so a route leaving the chain's own plane is drawn over the image it
runs through.

Usage: as ``build_dr1.py``, with DR1's key among ``--exclude-keys``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_dr1  # noqa: E402
from build_g4 import PanelDrawer, _load_mask  # noqa: E402
from build_grids import _evenly, shown_tips_um  # noqa: E402

GRID = "DR2"
SEED = 20261026
COUNT = 30
#: A route turning this far or more off the heading turns the third plane with it.
PLANE_TURN_DEG = 10.0


def measure(evidence, G, end, centrelines) -> dict:
    """DR1's measures (E1-E3) and E4's (``e4_*``) of one dead end."""
    values = build_dr1.measure(evidence, G, end, centrelines)
    if end.heading is None:
        return values
    _path, others, other_radii = build_dr1.neighbours(G, end, centrelines)
    route = evidence.route_ahead(np.asarray(end.tip_um, dtype=float), end.heading, float(end.tip_radius_um or 2.0),
                                 others_um=others, others_radius_um=other_radii)
    values.update({
        "e4_run_um": route.run_um, "e4_stop": route.stop, "e4_reach_um": route.reach_um,
        "e4_turn_deg": route.turn_deg, "e4_bend_deg": route.bend_deg, "e4_meets_um": route.meets_um,
        "e4_gap_um": route.gap_um, "e4_gap_stretches": route.gap_stretches,
        "e4_longest_gap_um": route.longest_gap_um,
        "e4_beyond_e3_um": route.run_um - float(values["e3_run_um"]),
        # The third rule fixed before DR2 (fit_marks.FRESH_RULES["DR2"]).
        "e4_goes_on": float(route.stop in ("runs_on", "vessel")),
        "e4_path": ";".join(",".join(f"{v:.2f}" for v in point) for point in route.path_um),
    })
    return values


def route_points(values: dict) -> np.ndarray:
    """E4's route from the key's ``e4_path``: ``(n, 3)`` microns, tip first."""
    text = values.get("e4_path") or ""
    return np.asarray([[float(v) for v in point.split(",")] for point in text.split(";") if point], dtype=float)


def choose(candidates: list, measures: dict, count: int = COUNT) -> list:
    """*count* of *candidates*: thirds of E4's run, and within each evenly
    across how much further E4 went than E3's straight line."""
    usable = [e for e in candidates if measures[e.tip].get("e4_run_um") is not None]
    if len(usable) <= count:
        return list(usable)
    run = np.asarray([measures[e.tip]["e4_run_um"] for e in usable], dtype=float)
    order = np.argsort(run, kind="stable")
    thirds = [[usable[i] for i in part] for part in np.array_split(order, 3)]
    beyond = lambda e: (measures[e.tip]["e4_beyond_e3_um"], measures[e.tip]["e4_run_um"], e.chain_length_um)  # noqa: E731
    chosen = []
    for third, share in zip(thirds, build_dr1._shares(count, 3)):
        chosen += _evenly(third, share, key=beyond)
    return chosen


def third_plane(heading: np.ndarray, e2: np.ndarray, route: np.ndarray, tip: np.ndarray) -> tuple:
    """``(a, b, normal)`` of the plane past the tip: the chain's own plane
    (*heading*, *e2*), or, when the route's chord turns
    :data:`PLANE_TURN_DEG` or more off the heading, the plane holding the
    heading and that chord."""
    if len(route) > 1:
        chord = route[-1] - tip
        across = chord - (chord @ heading) * heading
        length = float(np.linalg.norm(chord))
        if length > 0 and np.degrees(np.arccos(np.clip(chord @ heading / length, -1.0, 1.0))) >= PLANE_TURN_DEG:
            b = across / np.linalg.norm(across)
            return heading, b, np.cross(heading, b)
    return heading, e2, np.cross(heading, e2)


def views(drawer: PanelDrawer, G, end, values) -> list[dict]:
    """DR1's planes with E4's route as a hidden overlay (group ``route``),
    the third turned with the route."""
    from build_grids import _chain_points

    tip = np.asarray(end.tip_um, dtype=float)
    heading = np.asarray(end.heading, dtype=float)
    chain = _chain_points(G, end)
    route = route_points(values)
    lines = [
        dict(points=chain, color="#ff3b30", width=2.8),
        dict(points=build_dr1.ahead_line(end, values), color="#2ad4ff", width=1.6, marker=True),
    ]
    if len(route) > 1:
        lines.append(dict(points=route, color="#ffd60a", width=2.2, marker=True, group="route"))
    radius = float(end.tip_radius_um or 2.0)
    half_thickness = float(np.clip(2 * radius, 2.0, 5.0))
    focus = np.vstack([chain, tip[None], (tip + 8.0 * heading)[None]])
    planes = drawer.views(focus, heading, lines, half_thickness, marker=tip)
    _c, e1, e2, _n = drawer.plane(focus, heading)
    a, b, normal = third_plane(e1, e2, route, tip)
    far = max(float(values["e3_end_um"]), float(np.max(np.linalg.norm(route - tip, axis=1))) if len(route) else 0.0)
    ahead = (route[1:] - tip) if len(route) > 1 else np.empty((0, 3))
    centre = tip + 0.5 * (far - build_dr1.BEHIND_TIP_UM) * a
    if len(ahead):
        centre = centre + 0.5 * float(np.mean(ahead @ b)) * b
    R = float(np.clip(0.5 * (far + build_dr1.BEHIND_TIP_UM) + 4.0, 18.0, 32.0))
    planes.append(drawer.draw_plane("Past the tip", centre, a, b, normal, R, lines, half_thickness, marker=tip))
    return planes


PAGE = dict(
    title="Dead End Review DR2: does the vessel go on past its tip, and does the route follow it?",
    question="Does the vessel go on past the tip?",
    second="Is the traced route right?",
    lede=("One dead end of the E14.5 graph per card (red; the ring marks its tip), in three planes: its own "
          "plane, the one across it, and a third past the tip. The dashed cyan line is the tip's heading. Each "
          "plane shows the segmentation, the dextran channel (plasma) and the claudin-5 channel (the vessel "
          "wall). Tick “Show the traced route” to see, in yellow, the route traced through the walls past "
          "the tip. Grey dots are other centrelines; the orange bar is 5 um. Your marks save as you go."),
    steps=["With the route hidden, does the vessel go on past the tip? Goes on if it carries on beyond the tip, "
           "in either stain, whether or not the segmentation follows it; Ends here if it ends at the tip (or "
           "there is no vessel); Unsure if you cannot tell.",
           "Then show the traced route. Right if it follows the vessel and stops where the vessel stops; Goes "
           "astray if it leaves the vessel or follows something else; Stops short if the vessel goes on past "
           "where the route stops; Unsure if you cannot tell.",
           "Tick what decided it, and add a note if it helps."],
    verdicts=[("Y", "Goes on", "good"), ("E", "Ends here", "bad"), ("?", "Unsure", "unsure")],
    second_verdicts=[("R", "Right", "good"), ("A", "Goes astray", "bad"), ("S", "Stops short", "bad"),
                     ("?", "Unsure", "unsure")],
    evidence=["dextran", "endothelium", "segmentation shape", "context"],
)


def page_data(examples: list) -> dict:
    return {
        "batch": GRID, "title": PAGE["title"], "lede": PAGE["lede"], "question": PAGE["question"],
        "steps": PAGE["steps"], "evidence": PAGE["evidence"],
        "legend": [{"label": "the dead end judged", "color": "#ff3b30"},
                   {"label": "its heading", "color": "#2ad4ff"},
                   {"label": "the traced route (when shown)", "color": "#ffd60a"},
                   {"label": "other centrelines", "color": "#9a9a9a"}, {"label": "5 um", "color": "#ff9f0a"}],
        "verdicts": [{"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["verdicts"]],
        "second": {"question": PAGE["second"], "verdicts": [
            {"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["second_verdicts"]]},
        "toggles": [{"group": "route", "label": "Show the traced route"}],
        "examples": examples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True, help=".npy, or the segmentation .tif(f), binarised as the pipeline does")
    parser.add_argument("--composite", required=True, help="the raw ZCYX composite holding both channels")
    parser.add_argument("--dextran-channel", type=int, default=0)
    parser.add_argument("--endothelial-channel", type=int, default=1)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--psf-zyx", type=float, nargs=3, required=True, help="the rings' blur (sigma, um)")
    parser.add_argument("--wall-um", type=float, required=True, help="the rings' wall")
    parser.add_argument("--exclude-keys", nargs="*", default=[], help="earlier grids' keys with tip_zyx_vox")
    parser.add_argument("--exclude-mid-keys", nargs="*", default=[], help="grids M/M2's keys (facing pairs)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from haemolynx.graph.dead_ends import _Centrelines, classify_dead_ends, image_extent_um
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing import raw_evidence as rx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    mask = _load_mask(args.mask)
    vs = tuple(args.voxel_size_zyx)
    dextran = build_dr1._channel(args.composite, args.dextran_channel, mask.shape)
    claudin = build_dr1._channel(args.composite, args.endothelial_channel, mask.shape)
    calibration = rx.calibrate_raw(claudin, mask, vs, stain=rx.ENDOTHELIAL)
    if calibration is None:
        raise SystemExit("The endothelial channel did not calibrate round the mask")
    evidence = rx.make_raw_evidence(claudin, mask.shape, vs, calibration.with_rings(args.psf_zyx, args.wall_um))
    support = MaskSupport(mask, vs)
    print(evidence.calibration.describe())

    ends = classify_dead_ends(G, extent_um=image_extent_um(mask.shape, vs), support=support)
    candidates = build_dr1.fresh(ends.values(), shown_tips_um(args.exclude_keys, vs),
                                 build_dr1.shown_mids_um(args.exclude_mid_keys, vs))
    centrelines = _Centrelines(G, support)
    measures = {e.tip: measure(evidence, G, e, centrelines) for e in candidates}
    print(f"{len(candidates)} fresh interior dead ends of {len(ends)}")
    chosen = choose(candidates, measures)
    random.Random(SEED).shuffle(chosen)

    drawer = PanelDrawer(G, mask, {"Dextran": dextran, "Claudin-5": claudin}, vs)
    examples, rows = [], []
    for number, end in enumerate(chosen, start=1):
        ident = f"{GRID}-{number:02d}"
        tip_vox = tuple(int(round(c)) for c in np.asarray(end.tip_um) / np.asarray(vs))
        examples.append({"id": ident, "where": "tip at z {}, y {}, x {} (voxels)".format(*tip_vox),
                         "views": views(drawer, G, end, measures[end.tip])})
        rows.append({"id": ident, "grid": GRID, "tip_zyx_vox": tip_vox, **end.measures(), **measures[end.tip]})
    template = (HERE / "template_panels.html").read_text(encoding="utf-8")
    html = template.replace("__TITLE__", f"Dead End Review {GRID}").replace(
        "__EXAMPLES_JSON__", json.dumps(page_data(examples), separators=(",", ":")).replace("</", "<\\/"))
    (out / f"{GRID}_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / f"{GRID}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"{GRID}_candidates.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for values in measures.values() for k in values))
        writer = csv.DictWriter(f, fieldnames=["tip", "kind", "tip_zyx_vox", *fields])
        writer.writeheader()
        for end in candidates:
            writer.writerow({"tip": end.tip, "kind": end.kind,
                             "tip_zyx_vox": tuple(int(round(c)) for c in np.asarray(end.tip_um) / np.asarray(vs)),
                             **measures[end.tip]})
    (out / f"{GRID}_calibration.json").write_text(json.dumps(evidence.calibration.as_dict(), indent=2), encoding="utf-8")
    print(f"{GRID}: {len(examples)} examples -> {out / f'{GRID}_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
