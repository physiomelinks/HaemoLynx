#!/usr/bin/env python3
"""Build the marked grids PLAN_dead_ends.md Phase 2 calibrates dead ends on.

Three review pages of about 30 numbered examples each, from one built graph,
following CLAUDE.md's "Marked example grids":

- **G1, interior dead ends** -- what is it? S a real blind end (a sprout),
  B a vessel the segmentation broke (draw where it goes on), T the graph
  stopped short (the segmented vessel runs on), X not a vessel end (an
  artefact), F it leaves the image, ? unsure. Stratified by depth (the top
  40 um, the surface plexus, against the rest) and spread across the kinds,
  the chain's length in radii and the parent / stub radius ratio.
- **G2, near a face** -- F it leaves the image, I it ends inside it. Tips 2-25
  um from a face, half whose mask runs on to the face, half not, spread by
  distance; the stubs the prune keeps at a face among them.
- **G3, joins** -- J the vessel ahead is the same vessel (join them), N it is
  not. Tips with another vessel straight ahead within 10 um, spread by gap,
  background and how many junctions away the vessel ahead is, and pairs of
  facing dead ends the join refused.

Each example is drawn in its own plane (its heading and the direction its
chain bends in): the segmentation (what the code sees) as a slab through the
plane, and the raw image beside it (the "Raw image" toggle), because "sprout
or broken" is a question about the tissue. The element judged is red, its
heading cyan, a join yellow, other centrelines grey dots, a 5 um scale bar.
IDs are shuffled with a fixed seed; the measures stay off the page, in the
key CSV beside it (with each tip's voxel position, for napari).

Usage::

    python scripts/dead_end_review/build_grids.py --graph final.pkl --mask mask.npy \\
        --raw raw_C1.tif --voxel-size-zyx 1.0 0.98 0.98 --out outputs/dead_end_review
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import pickle
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261012
SURFACE_DEPTH_UM = 40.0


def _png(img: np.ndarray) -> str:
    from PIL import Image

    arr = (np.clip(img, 0.0, 1.0).T[::-1] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _evenly(items: list, count: int, key) -> list:
    """*count* of *items*, spread evenly across *key*'s order."""
    if len(items) <= count:
        return list(items)
    ordered = sorted(items, key=key)
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in dict.fromkeys(picks.tolist())]


class Drawer:
    """Draws one example in its own plane: segmentation and raw slabs, and the
    graph as overlays in the plane's (u, v) microns."""

    def __init__(self, G, mask: np.ndarray, raw: np.ndarray | None, voxel_size):
        from haemolynx.graph._helpers import densify_polyline, edge_sample_points

        self.G, self.mask, self.raw = G, mask, raw
        self.vs = np.asarray(voxel_size, dtype=float)
        points = []
        for u, v, d in G.edges(data=True):
            pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
            points.append(densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=1.0))
        self.centrelines = np.vstack(points) if points else np.empty((0, 3))

    def plane(self, focus: np.ndarray, heading: np.ndarray | None):
        """``(centre, e1, e2, n)``: e1 along *heading* (or the focus points'
        main direction), e2 the way they spread across it."""
        centre = focus.mean(axis=0)
        _, _, vh = np.linalg.svd(focus - centre, full_matrices=False)
        e1 = heading / np.linalg.norm(heading) if heading is not None else vh[0]
        e2 = next((w - (w @ e1) * e1 for w in (vh[1], vh[0], vh[2]) if np.linalg.norm(w - (w @ e1) * e1) > 0.2), None)
        if e2 is None:
            e2 = np.cross(e1, [1.0, 0.0, 0.0])
            if np.linalg.norm(e2) < 1e-6:
                e2 = np.cross(e1, [0.0, 1.0, 0.0])
        e2 = e2 / np.linalg.norm(e2)
        return centre, e1, e2, np.cross(e1, e2)

    def draw(self, focus: np.ndarray, heading, lines: list[dict], half_thickness: float, marker=None) -> dict:
        from scipy import ndimage

        c, e1, e2, n = self.plane(focus, heading)
        uv = lambda p: np.column_stack([(p - c) @ e1, (p - c) @ e2])
        R = float(np.clip(np.abs(uv(focus)).max() + 8.0, 15.0, 40.0))
        g = np.arange(-R, R + 1e-9, 0.25)
        U, V = np.meshgrid(g, g, indexing="ij")
        seg = np.zeros(U.shape, dtype=bool)
        raw = np.zeros(U.shape, dtype=float)
        depths = np.arange(-half_thickness, half_thickness + 1e-9, 0.5)
        for w in depths:
            P = (c + U[..., None] * e1 + V[..., None] * e2 + w * n) / self.vs
            coords = P.reshape(-1, 3).T
            seg |= ndimage.map_coordinates(self.mask, coords, order=0, mode="constant").reshape(U.shape).astype(bool)
            if self.raw is not None:
                raw += ndimage.map_coordinates(self.raw, coords, order=1, mode="constant").reshape(U.shape)
        if self.raw is not None:
            # The mean through the slab, lightly smoothed: the dextran signal
            # is speckled, and a maximum shows the speckle, not the vessel.
            raw = ndimage.gaussian_filter(raw / len(depths), sigma=2.0)
            lo, hi = np.percentile(raw, [2.0, 99.8])
            raw = (raw - lo) / (hi - lo) if hi > lo else raw * 0.0
        rel = self.centrelines - c
        near = (np.abs(rel @ n) <= half_thickness + 2) & (np.abs(rel @ e1) <= R) & (np.abs(rel @ e2) <= R)
        overlays = [dict(kind="dots", group="context", color="#9a9a9a",
                         points=np.round(uv(self.centrelines[near]), 1).tolist())]
        for line in lines:
            overlays.append(dict(kind="line", group="graph", color=line["color"], width=line.get("width", 2.4),
                                 points=np.round(uv(np.asarray(line["points"], dtype=float)), 2).tolist()))
        if marker is not None:
            # A ring round the tip, in the plane, so it reads at a glance.
            (mu, mv), = uv(np.asarray(marker, dtype=float)[None])
            angles = np.linspace(0, 2 * np.pi, 17)
            ring = np.column_stack([mu + 1.5 * np.cos(angles), mv + 1.5 * np.sin(angles)])
            overlays.append(dict(kind="line", group="graph", color="#ff3b30", width=1.6,
                                 points=np.round(ring, 2).tolist()))
        return dict(R=round(R, 3), slab=_png(seg.astype(float)), mid=_png(raw if self.raw is not None else seg * 1.0),
                    overlays=overlays)


def _chain_points(G, end, upto_um: float = 30.0) -> np.ndarray:
    from haemolynx.graph.dead_ends import chain_path

    path = chain_path(G, end.chain, step_um=0.5)
    if len(path) < 2:
        return np.asarray([end.tip_um], dtype=float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return path[arc <= upto_um]


def _example(drawer: Drawer, G, end, *, target=None, partner=None) -> dict:
    tip = np.asarray(end.tip_um, dtype=float)
    chain = _chain_points(G, end)
    heading = np.asarray(end.heading) if end.heading is not None else None
    focus = [chain, tip[None]]
    lines = [dict(points=chain, color="#ff3b30", width=2.8)]
    if heading is not None:
        lines.append(dict(points=[tip, tip + 4.0 * heading], color="#2ad4ff", width=2.0))
        focus.append((tip + 8.0 * heading)[None])
    if target is not None:
        lines.append(dict(points=[tip, target], color="#ffd60a", width=2.0))
        focus.append(np.asarray(target, dtype=float)[None])
    if partner is not None:
        other = _chain_points(G, partner)
        lines.append(dict(points=other, color="#ff3b30", width=2.8))
        lines.append(dict(points=[tip, np.asarray(partner.tip_um)], color="#ffd60a", width=2.0))
        focus.append(other)
    radius = end.tip_radius_um or 2.0
    return drawer.draw(np.vstack(focus), heading, lines, half_thickness=float(np.clip(2 * radius, 2.0, 5.0)),
                       marker=tip)


def _ahead_target(drawer: Drawer, end) -> np.ndarray | None:
    if end.ahead_um is None or end.heading is None:
        return None
    return np.asarray(end.tip_um) + end.ahead_um * np.asarray(end.heading)


def _pages(G, ends, facing_pairs, protected_tips):
    """``{grid: [(end, extra), ...]}`` before shuffling."""
    interior = [e for e in ends.values() if e.kind not in ("leaves_image", "island_end")]
    g1 = []
    shares = {"unknown": 6, "stops_short": 4, "broken": 3, "artefact": 2}
    for surface in (True, False):
        layer = [e for e in interior if (e.tip_um[0] < SURFACE_DEPTH_UM) == surface]
        chosen = []
        for kind, count in shares.items():
            pool = [e for e in layer if e.kind == kind]
            chosen += _evenly(pool, count, key=lambda e: e.chain_length_um / max(e.junction_radius_um or 1.0, 1e-6))
        spare = [e for e in layer if e not in chosen]
        chosen += _evenly(spare, max(0, 15 - len(chosen)),
                          key=lambda e: (e.junction_radius_um or 0) / max(e.tip_radius_um or 1.0, 1e-6))
        g1 += [(e, {}) for e in chosen[:15]]
    near = [e for e in ends.values() if e.kind != "island_end" and e.face_distance_um is not None
            and 2.0 <= e.face_distance_um <= 25.0]
    keep = [e for e in near if e.tip in protected_tips][:8]
    rest = [e for e in near if e not in keep]
    g2 = keep + _evenly([e for e in rest if e.reaches_face], (30 - len(keep)) // 2, key=lambda e: e.face_distance_um)
    g2 += _evenly([e for e in rest if not e.reaches_face], 30 - len(g2), key=lambda e: e.face_distance_um)
    g2 = [(e, {}) for e in g2]
    ahead = [e for e in interior if e.ahead_um is not None]
    g3 = [(e, {"target": True}) for e in _evenly(
        ahead, 24, key=lambda e: (e.ahead_um, e.ahead_background_um or 0.0, e.ahead_junctions_away or 99))]
    for pair in facing_pairs[: 30 - len(g3)]:
        g3.append((ends[pair.a], {"partner": ends[pair.b], "pair": pair}))
    return {"G1": g1, "G2": g2, "G3": g3}


PAGES = {
    "G1": dict(
        title="Dead End Review G1: interior dead ends",
        question="What is this dead end?",
        lede=("The E14.5 segmentation (white = vessel) in the plane of one dead end (red), its heading cyan. "
              "Toggle 'Raw image' to see the dextran signal. Your marks and drawings save as you go."),
        steps=["Look at where the red chain ends, and what the segmentation and the raw image do there.",
               "Mark what it is. For B, draw where the vessel goes on.",
               "Add a note on what decided it, if it helps."],
        verdicts=[("S", "Real blind end", "good"), ("B", "Broken: goes on", "bad"), ("T", "Graph stopped short", "bad"),
                  ("X", "Not a vessel end", "bad"), ("F", "Leaves the image", "good"), ("?", "Unsure", "unsure")],
    ),
    "G2": dict(
        title="Dead End Review G2: near an image face",
        question="Does this vessel leave the image?",
        lede=("A dead end (red) 2-25 um from an image face, in its own plane; heading cyan. "
              "The page edge is not the image edge: judge from the segmentation where the vessel goes."),
        steps=["F if the vessel runs out through the face; I if it ends inside the image.",
               "If it ends inside, say in the note whether it is a real end, a broken vessel or not a vessel."],
        verdicts=[("F", "Leaves the image", "good"), ("I", "Ends inside", "bad"), ("?", "Unsure", "unsure")],
    ),
    "G3": dict(
        title="Dead End Review G3: joins across a gap",
        question="Should these be joined?",
        lede=("A dead end (red) with another vessel straight ahead, or two facing dead ends; the join a rule "
              "would draw is yellow. Toggle 'Raw image' to see whether the vessel goes on through the gap."),
        steps=["J if the vessel ahead is the same vessel, broken by the segmentation: join them.",
               "N if it is another vessel (the dead end is a real end or goes elsewhere). Draw where it goes."],
        verdicts=[("J", "Join: same vessel", "good"), ("N", "Not the same vessel", "bad"), ("?", "Unsure", "unsure")],
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True, help=".npy of the binary segmentation")
    parser.add_argument("--raw", help="the raw image (one channel, z y x)")
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--protected-key", help="a CSV with tip_zyx_vox of the stubs the prune keeps at a face")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import tifffile

    from haemolynx.graph.dead_ends import classify_dead_ends, image_extent_um
    from haemolynx.graph.facing_ends import facing_dead_end_pairs
    from haemolynx.preprocessing import MaskSupport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    mask = np.load(args.mask).astype(bool)
    raw = tifffile.imread(args.raw).astype(np.float32) if args.raw else None
    if raw is not None and raw.shape != mask.shape:
        raise SystemExit(f"raw image {raw.shape} and mask {mask.shape} differ")
    vs = tuple(args.voxel_size_zyx)
    support = MaskSupport(mask, vs)
    ends = classify_dead_ends(G, extent_um=image_extent_um(mask.shape, vs), support=support)
    pairs = [p for p in facing_dead_end_pairs(G, support, search_gap_um=20.0) if p.refused and p.a in ends and p.b in ends]
    protected = set()
    if args.protected_key:
        tips = np.asarray([[float(x) for x in row["tip_zyx_vox"].strip("()").split(",")]
                           for row in csv.DictReader(open(args.protected_key))]) * np.asarray(vs)
        for tip, end in ends.items():
            if len(tips) and np.min(np.linalg.norm(tips - np.asarray(end.tip_um), axis=1)) <= 2.0:
                protected.add(tip)
    drawer = Drawer(G, mask, raw, vs)
    template = (HERE / "template.html").read_text(encoding="utf-8")
    rng = random.Random(SEED)
    for grid, chosen in _pages(G, ends, pairs, protected).items():
        rng.shuffle(chosen)
        examples, rows = [], []
        for number, (end, extra) in enumerate(chosen, start=1):
            ident = f"{grid}-{number:02d}"
            partner = extra.get("partner")
            target = _ahead_target(drawer, end) if extra.get("target") else None
            examples.append({"id": ident, **_example(drawer, G, end, target=target, partner=partner)})
            row = {"id": ident, "grid": grid, "tip_zyx_vox": tuple(int(round(c)) for c in np.asarray(end.tip_um) / np.asarray(vs)),
                   "protected_at_face": end.tip in protected, **end.measures()}
            if partner is not None:
                row.update(partner_tip=partner.tip, pair_gap_um=extra["pair"].gap_um, pair_turn_deg=extra["pair"].turn_deg,
                           pair_refused=extra["pair"].refused)
            rows.append(row)
        page = PAGES[grid]
        data = {
            "batch": grid, "title": page["title"], "lede": page["lede"], "question": page["question"],
            "steps": page["steps"],
            "legend": [{"label": "the dead end judged", "color": "#ff3b30"}, {"label": "its heading", "color": "#2ad4ff"},
                       {"label": "the join", "color": "#ffd60a"}, {"label": "other centrelines", "color": "#9a9a9a"},
                       {"label": "your drawing", "color": "#ffd60a"}],
            "verdicts": [{"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in page["verdicts"]],
            "examples": examples,
        }
        html = (template.replace("__TITLE__", f"Dead End Review {grid}").replace("__SECOND_PANEL__", "Raw image (not the segmentation)")
                .replace("__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
        (out / f"{grid}_review.html").write_text(html, encoding="utf-8")
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(out / f"{grid}_key.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{grid}: {len(examples)} examples -> {out / (grid + '_review.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
