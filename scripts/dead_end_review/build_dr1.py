#!/usr/bin/env python3
"""Build grid DR1: does a dead end's vessel go on past its tip?

``PLAN_dead_ends_raw.md`` Phase C. G4 asked whether a dead end's chain is a
real vessel, with the endothelial channel beside the segmentation; its marks
followed shape in context, and the stain, read as a mean along the chain,
was weak evidence. An endothelial stain's information is *across* the chain:
whether a wall ring runs round it, and whether the walls run on past the
tip, where a vessel the dextran never filled has walls and no plasma. DR1
asks both questions of fresh dead ends spread across the two measures of
that (``preprocessing.raw_evidence``):

- E1, ring support: the share of the tip's five sections (every 2 um over
  its last 10 um) with a wall ring fitted round them
  (``RawEvidenceEndothelial.ring_along``);
- E3, continuation: how far the walls run on along the tip's heading
  (``RawEvidenceEndothelial.walls_ahead``).

Each card shows the segmentation, the dextran and the claudin-5 channels in
three planes: the chain's own and the one across it (G4's), and a third
along the heading past the tip, framed on the straight line E3 reads,
drawn dashed. Two questions: is the red chain a real vessel (G4's V / N /
P / ?), and does the vessel go on past the tip (yes / no / unsure).

Examples are interior dead ends (not leaving the image, not island ends)
with a heading, none within 3 um of a tip an earlier grid showed (``--exclude-keys``),
nor within 8 um of a facing pair grids M and M2 showed (``--exclude-mid-keys``).
They are spread across E3's run in thirds and, within each, across three
levels of E1 (no ring, some, half or more), then shuffled; every measure, E2
included, stays in the key beside the page.

Usage::

    python scripts/dead_end_review/build_dr1.py --graph graph.pkl --mask mask.npy \\
        --composite composite.tif --dextran-channel 0 --endothelial-channel 1 \\
        --voxel-size-zyx 1.0 0.98 0.98 --psf-zyx 2.71 1.08 1.08 --wall-um 1.13 \\
        --exclude-keys G1_key.csv ... --exclude-mid-keys M_key.csv M2_key.csv --out outputs/dead_end_review
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

from build_g4 import PanelDrawer, _load_mask  # noqa: E402
from build_grids import SHOWN_TIP_TOLERANCE_UM, shown_tips_um  # noqa: E402

GRID = "DR1"
SEED = 20261025
COUNT = 30
#: A facing pair's midpoint (grids M, M2) within this of a tip: shown already.
SHOWN_MID_TOLERANCE_UM = 8.0
#: Other centrelines within this of the tip, or this of the chain, are the
#: ones a ring or a lumen ahead may belong to.
OTHERS_FROM_TIP_UM = 45.0
OTHERS_FROM_CHAIN_UM = 15.0
#: The third plane runs this far behind the tip, to show where the chain ends.
BEHIND_TIP_UM = 6.0


def neighbours(G, end, centrelines) -> tuple[np.ndarray, np.ndarray, object]:
    """``(path, others, other_radii)``: the dead end's chain, tip first, and
    the points (and lumen radii) of every other centreline within
    :data:`OTHERS_FROM_TIP_UM` of its tip or :data:`OTHERS_FROM_CHAIN_UM` of
    its chain -- whose rings and lumens the raw readings may meet."""
    from haemolynx.graph._helpers import edge_id, points_within
    from haemolynx.graph.dead_ends import chain_path

    path = chain_path(G, end.chain, step_um=0.5) if end.chain else np.asarray(end.tip_um).reshape(1, 3)
    tip = np.asarray(end.tip_um, dtype=float)
    own = {centrelines.ids[edge_id(*e)] for e in end.chain if edge_id(*e) in centrelines.ids}
    near = np.union1d(
        centrelines.near(tip, OTHERS_FROM_TIP_UM, own),
        points_within(centrelines.tree, path, OTHERS_FROM_CHAIN_UM) if len(path) else np.empty(0, dtype=np.intp),
    ).astype(np.intp)
    near = near[~np.isin(centrelines.owners[near], list(own))] if len(near) else near
    other_radii = centrelines.radii[near] if centrelines.radii is not None else 0.0
    return path, centrelines.points[near], other_radii


def measure(evidence, G, end, centrelines) -> dict:
    """E1-E3 of one dead end: the key's columns (``e1_*``, ``e2_*``, ``e3_*``)."""
    from haemolynx.preprocessing import raw_evidence as rx

    path, others, other_radii = neighbours(G, end, centrelines)
    tip = np.asarray(end.tip_um, dtype=float)
    radius = float(end.tip_radius_um or 2.0)
    ring = evidence.ring_along(path, radius, others_um=others)
    sections = rx._along(path, rx.RING_SECTION_STEP_UM)[1:] if len(path) > 1 else path
    inside = evidence.inside_walls(sections[: int(rx.RING_TIP_REACH_UM / rx.RING_SECTION_STEP_UM)], radius,
                                   others_um=others)
    seen = [r for r in ring.readings if r.status == "ring"]
    values = {
        "e1_fraction": ring.fraction, "e1_tip_fraction": ring.tip_fraction,
        "e1_rings_tried": len(ring.readings), "e1_rings_seen": ring.rings_seen,
        "e1_ring_share": ring.rings_seen / len(ring.readings) if ring.readings else 0.0,
        "e1_rings_own": ring.rings(rx.RING_OWN), "e1_rings_other": ring.rings(rx.RING_OTHER),
        "e1_rings_off_centre": ring.rings(rx.RING_OFF_CENTRE), "e1_own_ring_fraction": ring.own_ring_fraction,
        "e1_ring_diameter_um": float(np.median([r.diameter_um for r in seen])) if seen else None,
        "e1_statuses": ";".join(r.status for r in ring.readings),
        "e2_tip_lumen_fraction": inside.lumen_fraction, "e2_tip_shared_fraction": inside.shared_fraction,
        "e2_tip_wall_fraction": float(inside.wall.mean()) if len(inside.wall) else 0.0,
    }
    if end.heading is not None:
        ahead = evidence.walls_ahead(tip, end.heading, radius, others_um=others, others_radius_um=other_radii)
        values.update({
            "e3_run_um": ahead.run_um, "e3_stop": ahead.stop, "e3_meets_um": ahead.meets_um,
            "e3_face_um": ahead.face_um, "e3_gap_um": ahead.gap_um, "e3_gap_stretches": ahead.gap_stretches,
            "e3_longest_gap_um": ahead.longest_gap_um, "e3_start_um": ahead.start_um,
            "e3_end_um": ahead.start_um + ahead.step_um * len(ahead.profile),
            "e3_profile": ";".join(f"{v:.3f}" for v in ahead.profile),
        })
    return values


def ring_share(values: dict) -> float:
    """E1 as the plan defines it: the share of the tip's sections with a wall
    ring fitted round them (whoever's), 0 with none fitted."""
    tried = int(values.get("e1_rings_tried") or 0)
    return float(values.get("e1_rings_seen") or 0) / tried if tried else 0.0


def ring_level(values: dict) -> int:
    """E1 in three levels: no ring (0), rings at some sections (1), at half
    or more (2). On E14.5 half the interior tips fit none -- a capillary's
    lumen is filled in by the blur along z -- so even picks across E1's order
    would be half zeros, and a threshold could not be fitted above them."""
    share = ring_share(values)
    return 0 if share <= 0.0 else 1 if share < 0.5 else 2


def _shares(count: int, parts: int) -> list[int]:
    return [count // parts + (1 if k < count % parts else 0) for k in range(parts)]


def choose(candidates: list, measures: dict, count: int = COUNT) -> list:
    """*count* of *candidates* spread across both measures: thirds of E3's
    run (how far the walls run on past the tip), and within each third the
    three levels of E1 (:func:`ring_level`) in equal shares, each spread
    evenly across the run. A level short of its share is made up from the
    rest of its third, evenly across E1's share and its vesselness."""
    from build_grids import _evenly

    usable = [e for e in candidates if measures[e.tip].get("e3_run_um") is not None]
    if len(usable) <= count:
        return list(usable)
    run = np.asarray([measures[e.tip]["e3_run_um"] for e in usable], dtype=float)
    order = np.argsort(run, kind="stable")
    thirds = [[usable[i] for i in part] for part in np.array_split(order, 3)]
    by_run = lambda e: (measures[e.tip]["e3_run_um"], e.chain_length_um)  # noqa: E731
    by_ring = lambda e: (ring_share(measures[e.tip]), measures[e.tip]["e1_tip_fraction"], e.chain_length_um)  # noqa: E731
    chosen = []
    for third, share in zip(thirds, _shares(count, 3)):
        picked = []
        for level, part in zip((0, 1, 2), _shares(share, 3)):
            cell = [e for e in third if ring_level(measures[e.tip]) == level]
            picked += _evenly(cell, part, key=by_run)
        rest = [e for e in third if e not in picked]
        picked += _evenly(rest, share - len(picked), key=by_ring)
        chosen += picked
    return chosen


def ahead_line(end, values) -> list:
    """The straight line E3 reads, from its first reading to its last, as a
    dashed line (a None between dashes) for the page."""
    tip, heading = np.asarray(end.tip_um, dtype=float), np.asarray(end.heading, dtype=float)
    start, stop = float(values["e3_start_um"]), float(values["e3_end_um"])
    points: list = []
    for a in np.arange(start, stop, 2.0):
        if points:
            points.append(None)
        points += [tip + a * heading, tip + min(a + 1.2, stop) * heading]
    return points


def views(drawer: PanelDrawer, G, end, values) -> list[dict]:
    """G4's two planes through the chain, and a third in the chain's own
    plane framed on the line past the tip."""
    from build_grids import _chain_points

    tip = np.asarray(end.tip_um, dtype=float)
    heading = np.asarray(end.heading, dtype=float)
    chain = _chain_points(G, end)
    dashed = dict(points=ahead_line(end, values), color="#2ad4ff", width=1.6, marker=True)
    lines = [dict(points=chain, color="#ff3b30", width=2.8), dashed]
    radius = float(end.tip_radius_um or 2.0)
    half_thickness = float(np.clip(2 * radius, 2.0, 5.0))
    focus = np.vstack([chain, tip[None], (tip + 8.0 * heading)[None]])
    planes = drawer.views(focus, heading, lines, half_thickness, marker=tip)
    _c, e1, e2, n = drawer.plane(focus, heading)
    far = float(values["e3_end_um"])
    centre = tip + 0.5 * (far - BEHIND_TIP_UM) * heading
    R = float(np.clip(0.5 * (far + BEHIND_TIP_UM) + 4.0, 18.0, 30.0))
    planes.append(drawer.draw_plane("Past the tip (its own plane)", centre, e1, e2, n, R, lines, half_thickness,
                                    marker=tip))
    return planes


PAGE = dict(
    title="Dead End Review DR1: does the vessel go on past its tip?",
    question="Is the red chain a real vessel?",
    second="Does the vessel go on past the tip?",
    lede=("One dead end of the E14.5 graph per card (red; the ring marks its tip), in three planes: its own "
          "plane and the one across it, as on G4, and a third along its heading past the tip. The dashed cyan "
          "line is the straight line ahead of the tip that is being measured, from one lumen radius past the "
          "tip to 30 um further. Each plane shows the segmentation the graph was built from, the dextran "
          "channel (plasma) and the claudin-5 channel (the vessel wall: a vessel is two bright walls with a "
          "dark lumen between). Grey dots are other centrelines; the orange bar is 5 um. Your marks save as "
          "you go."),
    steps=["Is the red chain a real vessel? V if it follows a real vessel, N if there is no vessel there, "
           "P if it is a vessel but its last stretch is not.",
           "Does the vessel go on past the tip? Yes if it carries on beyond the tip, in either stain, whether "
           "or not the segmentation follows it; No if it ends at the tip (or there is no vessel); Unsure if "
           "you cannot tell.",
           "Tick what decided it, and add a note if it helps (where it goes, if it goes on)."],
    verdicts=[("V", "Vessel", "good"), ("N", "Not a vessel", "bad"), ("P", "Partly", "unsure"), ("?", "Unsure", "unsure")],
    second_verdicts=[("Y", "Goes on", "good"), ("E", "Ends here", "bad"), ("?", "Unsure", "unsure")],
    evidence=["dextran", "endothelium", "segmentation shape", "context"],
)


def page_data(examples: list) -> dict:
    return {
        "batch": GRID, "title": PAGE["title"], "lede": PAGE["lede"], "question": PAGE["question"],
        "steps": PAGE["steps"], "evidence": PAGE["evidence"],
        "legend": [{"label": "the dead end judged", "color": "#ff3b30"},
                   {"label": "the line ahead of its tip", "color": "#2ad4ff"},
                   {"label": "other centrelines", "color": "#9a9a9a"}, {"label": "5 um", "color": "#ff9f0a"}],
        "verdicts": [{"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["verdicts"]],
        "second": {"question": PAGE["second"], "verdicts": [
            {"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["second_verdicts"]]},
        "examples": examples,
    }


def shown_mids_um(key_paths, voxel_size_zyx) -> np.ndarray:
    """The facing pairs' midpoints earlier grids showed (grids M, M2: rows of
    kind ``facing``, ``mid_zyx_vox``), physical microns."""
    mids = []
    for path in key_paths or ():
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = (row.get("mid_zyx_vox") or "").strip("()[] ")
                if text and row.get("kind", "facing") == "facing":
                    mids.append([float(c) for c in text.split(",")])
    return np.asarray(mids, dtype=float) * np.asarray(voxel_size_zyx, dtype=float) if mids else np.empty((0, 3))


def fresh(ends, shown_tips: np.ndarray, shown_mids: np.ndarray) -> list:
    """The interior dead ends with a heading no earlier grid showed."""
    from haemolynx.graph.dead_ends import ISLAND_END, LEAVES_IMAGE

    def far(points, tip, tolerance):
        return not len(points) or float(np.min(np.linalg.norm(points - tip, axis=1))) > tolerance

    return [
        e for e in ends
        if e.kind not in (LEAVES_IMAGE, ISLAND_END) and e.heading is not None
        and far(shown_tips, np.asarray(e.tip_um), SHOWN_TIP_TOLERANCE_UM)
        and far(shown_mids, np.asarray(e.tip_um), SHOWN_MID_TOLERANCE_UM)
    ]


def _channel(path: str, channel: int, shape) -> np.ndarray:
    from haemolynx.preprocessing.raw_evidence import load_raw_channel

    volume = load_raw_channel(path, channel)
    if volume.shape != tuple(shape):
        raise SystemExit(f"{path} channel {channel}: {volume.shape} differs from the mask's {tuple(shape)}")
    return volume


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True, help=".npy, or the segmentation .tif(f), binarised as the pipeline does")
    parser.add_argument("--composite", required=True, help="the raw ZCYX composite holding both channels")
    parser.add_argument("--dextran-channel", type=int, default=0)
    parser.add_argument("--endothelial-channel", type=int, default=1)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--psf-zyx", type=float, nargs=3, help="the rings' blur (sigma, um); fitted when absent")
    parser.add_argument("--wall-um", type=float, help="the rings' wall; fitted with the blur when absent")
    parser.add_argument("--workers", type=int, default=8, help="for fitting the rings' blur and wall")
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
    dextran = _channel(args.composite, args.dextran_channel, mask.shape)
    claudin = _channel(args.composite, args.endothelial_channel, mask.shape)
    calibration = rx.calibrate_raw(claudin, mask, vs, stain=rx.ENDOTHELIAL)
    if calibration is None:
        raise SystemExit("The endothelial channel did not calibrate round the mask")
    evidence = rx.make_raw_evidence(claudin, mask.shape, vs, calibration)
    support = MaskSupport(mask, vs)
    if args.psf_zyx and args.wall_um:
        evidence.calibration = calibration.with_rings(args.psf_zyx, args.wall_um)
    else:
        for _u, _v, data in G.edges(data=True):
            middle = np.asarray(data["voxels"], dtype=float)[len(data["voxels"]) // 2].reshape(1, 3)
            data["guide_diameter_um"] = 2.0 * float(support.radius(middle)[0])
        evidence.calibration = rx.ring_calibration(G, evidence, guide_attribute="guide_diameter_um",
                                                   workers=args.workers)
        if evidence.calibration.ring_psf_sigma_zyx is None:
            raise SystemExit("The image's rings did not calibrate; pass --psf-zyx and --wall-um")
    print(evidence.calibration.describe())

    ends = classify_dead_ends(G, extent_um=image_extent_um(mask.shape, vs), support=support)
    candidates = fresh(ends.values(), shown_tips_um(args.exclude_keys, vs), shown_mids_um(args.exclude_mid_keys, vs))
    centrelines = _Centrelines(G, support)
    measures = {e.tip: measure(evidence, G, e, centrelines) for e in candidates}
    print(f"{len(candidates)} fresh interior dead ends of {len(ends)}")
    rng = random.Random(SEED)
    chosen = choose(candidates, measures)
    rng.shuffle(chosen)

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
