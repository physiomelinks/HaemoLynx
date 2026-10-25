#!/usr/bin/env python3
"""Build grid G4: is a dead end's chain a real vessel?

G1-G3 (``build_grids.py``) asked what a dead end is, from the segmentation
with the dextran image behind a toggle. In 54 of their 87 marks the dead end
was "not a real vessel", and no measure of the mask told those apart: their
chains lie in the mask. So G4 puts the images first:

- every card shows the segmentation (what the code sees, with the graph), the
  dextran channel (the plasma the segmentation was made from) and the
  endothelial channel (claudin-5: a wall where no dextran reached is a vessel
  that did not fill), side by side;
- in two planes at right angles through the chain (its own plane, and the
  one across it), over a wider window;
- one question -- is the red chain a real vessel? -- with V a vessel, N not a
  vessel (remove it), P partly (the last stretch is not), ? unsure; and an
  optional "what decided it" (dextran, endothelium, segmentation shape,
  context) beside the note.

Examples are fresh: interior dead ends (not leaving the image, not island
ends) of the graph given, none within 3 um of a tip G1-G3 showed. They are
spread across the two candidate measures, the dextran and the endothelial
contrast along the chain's last 10 um (``channel_contrast``), then shuffled.
The measures stay off the page, in the key CSV beside it.

Usage::

    python scripts/dead_end_review/build_g4.py --graph graph.pkl --mask segmentation.tiff \\
        --raw C1.tif --endothelial composite.tif --endothelial-channel 1 \\
        --voxel-size-zyx 1.0 0.98 0.98 --exclude-keys G1_key.csv G2_key.csv G3_key.csv \\
        --out outputs/dead_end_review
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

from build_grids import Drawer, _chain_points, _evenly, _png  # noqa: E402

SEED = 20261013
#: How many examples, and how far back from the tip the tip stretch runs.
COUNT = 30
TIP_STRETCH_UM = 10.0
#: The background shell round a path, microns from it, outside the mask
#: grown by MASK_MARGIN_UM.
SHELL_UM = (4.0, 8.0)
MASK_MARGIN_UM = 1.5
EXCLUDE_WITHIN_UM = 3.0


def channel_level(volume: np.ndarray, mask: np.ndarray, path_um: np.ndarray, radius_um: float, voxel_size) -> dict:
    """What *volume* reads along a centreline: ``inside`` (the 90th
    percentile within ``radius_um + 1`` of the path -- a lumen filled with
    dextran or a wall ring of endothelium both reach it) and ``background``
    (the median of the shell :data:`SHELL_UM` from the path, outside the
    mask grown by :data:`MASK_MARGIN_UM`). NaN where there is nothing to read."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    vs = np.asarray(voxel_size, dtype=float)
    path_um = np.asarray(path_um, dtype=float)
    if len(path_um) == 0:
        return {"inside": np.nan, "background": np.nan}
    reach = SHELL_UM[1] + 1.0
    lo = np.maximum(np.floor((path_um.min(axis=0) - reach) / vs).astype(int), 0)
    hi = np.minimum(np.ceil((path_um.max(axis=0) + reach) / vs).astype(int) + 1, np.asarray(volume.shape))
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    crop = np.asarray(volume[box], dtype=float)
    grown = ndimage.binary_dilation(
        np.asarray(mask[box], dtype=bool),
        structure=np.ones([2 * int(np.ceil(MASK_MARGIN_UM / s)) + 1 for s in vs], dtype=bool),
    )
    centres = (np.indices(crop.shape).reshape(3, -1).T + lo) * vs
    distance = cKDTree(path_um).query(centres)[0].reshape(crop.shape)
    inside = crop[distance <= float(radius_um) + 1.0]
    shell = crop[(distance >= SHELL_UM[0]) & (distance <= SHELL_UM[1]) & ~grown]
    return {
        "inside": float(np.percentile(inside, 90)) if inside.size else np.nan,
        "background": float(np.median(shell)) if shell.size else np.nan,
    }


def channel_contrast(level: dict, reference: float) -> float:
    """How bright a path reads above its own background, as a share of a
    typical vessel's: 0 is background, 1 as bright as the *reference*
    (a typical flowing vessel's ``inside`` above its background)."""
    if not np.isfinite(level["inside"]) or not np.isfinite(level["background"]) or reference <= 0:
        return np.nan
    return float((level["inside"] - level["background"]) / reference)


def _reference(G, channel, mask, vs, rng: random.Random, count: int = 200) -> float:
    """A typical vessel's contrast in *channel*: the median, over up to
    *count* vessels carrying flow (or any vessels, on a graph never solved),
    of ``inside - background``."""
    from haemolynx.graph import FLOW_SOLVED
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get(FLOW_SOLVED)] or list(G.edges(data=True))
    rng.shuffle(edges)
    lifts = []
    for u, v, d in edges[:count]:
        pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v)}
        path = densify_polyline(edge_sample_points(u, v, d, pos), max_step_um=0.5)
        radius = float(d.get("diameter_um") or 4.0) / 2.0
        level = channel_level(channel, mask, path, radius, vs)
        if np.isfinite(level["inside"]) and np.isfinite(level["background"]):
            lifts.append(level["inside"] - level["background"])
    return float(np.median(lifts)) if lifts else np.nan


def _parent_path(G, end, upto_um: float = 10.0) -> np.ndarray:
    """The first *upto_um* of the vessels at the junction the chain hangs
    from, other than the chain's own."""
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    junction, last = end.junction, end.chain[-1] if end.chain else None
    points = []
    for a, b, key, d in G.edges(junction, keys=True, data=True):
        if last is not None and {a, b} == {last[0], last[1]} and key == last[2]:
            continue
        pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (a, b)}
        path = densify_polyline(edge_sample_points(a, b, d, pos), max_step_um=0.5)
        if np.linalg.norm(path[0] - np.asarray(G.nodes[junction]["pos"], dtype=float)) > np.linalg.norm(
            path[-1] - np.asarray(G.nodes[junction]["pos"], dtype=float)
        ):
            path = path[::-1]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        points.append(path[arc <= upto_um])
    return np.vstack(points) if points else np.empty((0, 3))


def measure(G, end, mask, channels: dict, references: dict, vs) -> dict:
    """The candidate measures of one dead end, for the key."""
    from haemolynx.graph.dead_ends import chain_path

    path = chain_path(G, end.chain, step_um=0.5)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]) if len(path) else path
    tip_stretch = path[arc <= TIP_STRETCH_UM] if len(path) else path
    radius = float(end.tip_radius_um or 2.0)
    parent = _parent_path(G, end)
    values = {}
    for name, volume in channels.items():
        ref = references[name]
        tip = channel_level(volume, mask, tip_stretch, radius, vs)
        whole = channel_level(volume, mask, path, radius, vs)
        values[f"{name}_tip_contrast"] = channel_contrast(tip, ref)
        values[f"{name}_chain_contrast"] = channel_contrast(whole, ref)
        if len(parent):
            beside = channel_level(volume, mask, parent, float(end.junction_radius_um or radius), vs)
            values[f"{name}_parent_contrast"] = channel_contrast(beside, ref)
    return values


class PanelDrawer(Drawer):
    """Every channel in two planes: the chain's own and the one across it."""

    def __init__(self, G, mask, channels: dict, voxel_size):
        super().__init__(G, mask, None, voxel_size)
        self.channels = channels

    def views(self, focus, heading, lines, half_thickness, marker=None) -> list[dict]:
        c, e1, e2, n = self.plane(focus, heading)
        views = []
        for label, (a, b, normal) in (("Its own plane", (e1, e2, n)), ("Across it", (e1, n, e2))):
            R = float(np.clip(np.abs(np.column_stack([(focus - c) @ a, (focus - c) @ b])).max() + 10.0, 20.0, 45.0))
            views.append(self.draw_plane(label, c, a, b, normal, R, lines, half_thickness, marker))
        return views

    def draw_plane(self, label, c, a, b, normal, R, lines, half_thickness, marker=None) -> dict:
        """One plane through *c* spanned by *a* and *b*, *R* microns each way,
        every channel as the mean through a slab *half_thickness* either side
        along *normal*, with the graph's *lines* and the tip *marker* over it.
        A line marked ``marker`` is drawn on the stains too."""
        from scipy import ndimage

        uv = lambda p: np.column_stack([(p - c) @ a, (p - c) @ b])  # noqa: E731
        g = np.arange(-R, R + 1e-9, 0.3)
        U, V = np.meshgrid(g, g, indexing="ij")
        depths = np.arange(-half_thickness, half_thickness + 1e-9, 0.5)
        seg = np.zeros(U.shape, dtype=bool)
        sums = {name: np.zeros(U.shape) for name in self.channels}
        for w in depths:
            coords = ((c + U[..., None] * a + V[..., None] * b + w * normal) / self.vs).reshape(-1, 3).T
            seg |= ndimage.map_coordinates(self.mask, coords, order=0, mode="constant").reshape(U.shape).astype(bool)
            for name, volume in self.channels.items():
                sums[name] += ndimage.map_coordinates(volume, coords, order=1, mode="constant").reshape(U.shape)
        images = [dict(label="Segmentation", png=_png(seg.astype(float)))]
        for name, total in sums.items():
            # The mean through the slab, lightly smoothed: both stains are
            # speckled, and a maximum shows the speckle, not the vessel.
            img = ndimage.gaussian_filter(total / len(depths), sigma=1.5)
            lo, hi = np.percentile(img, [2.0, 99.8])
            img = (img - lo) / (hi - lo) if hi > lo else img * 0.0
            images.append(dict(label=name, png=_png(img)))
        rel = self.centrelines - c
        near = (np.abs(rel @ normal) <= half_thickness + 2) & (np.abs(rel @ a) <= R) & (np.abs(rel @ b) <= R)
        overlays = [dict(kind="dots", group="context", color="#9a9a9a",
                         points=np.round(uv(self.centrelines[near]), 1).tolist())]
        for line in lines:
            # A None in a line's points lifts the pen (a dashed line).
            runs, run = [], []
            for p in list(line["points"]) + [None]:
                if p is None:
                    if run:
                        runs.append(np.round(uv(np.asarray(run, dtype=float)), 2).tolist())
                    run = []
                else:
                    run.append(p)
            points = [q for k, r in enumerate(runs) for q in ([None] if k else []) + r]
            layer = dict(kind="line", group=line.get("group", "graph"), color=line["color"],
                         width=line.get("width", 2.4), points=points)
            if line.get("marker"):
                layer["marker"] = True
            overlays.append(layer)
        if marker is not None:
            (mu, mv), = uv(np.asarray(marker, dtype=float)[None])
            angles = np.linspace(0, 2 * np.pi, 17)
            ring = np.column_stack([mu + 1.5 * np.cos(angles), mv + 1.5 * np.sin(angles)])
            overlays.append(dict(kind="line", group="graph", color="#ff3b30", width=1.6, marker=True,
                                 points=np.round(ring, 2).tolist()))
        return dict(label=label, R=round(R, 3), images=images, overlays=overlays)


def _views(drawer: PanelDrawer, G, end) -> list[dict]:
    tip = np.asarray(end.tip_um, dtype=float)
    chain = _chain_points(G, end)
    heading = np.asarray(end.heading) if end.heading is not None else None
    focus = [chain, tip[None]]
    lines = [dict(points=chain, color="#ff3b30", width=2.8)]
    if heading is not None:
        lines.append(dict(points=[tip, tip + 4.0 * heading], color="#2ad4ff", width=2.0))
        focus.append((tip + 8.0 * heading)[None])
    radius = end.tip_radius_um or 2.0
    return drawer.views(np.vstack(focus), heading, lines, half_thickness=float(np.clip(2 * radius, 2.0, 5.0)),
                        marker=tip)


def _shown_tips(paths, vs) -> np.ndarray:
    tips = []
    for path in paths or ():
        for row in csv.DictReader(open(path, encoding="utf-8")):
            tips.append([float(x) for x in row["tip_zyx_vox"].strip("()").split(",")])
    return np.asarray(tips, dtype=float) * np.asarray(vs) if tips else np.empty((0, 3))


def choose(candidates: list, measures: dict, count: int = COUNT) -> list:
    """*count* of *candidates*, spread across the endothelial contrast's
    thirds and, within each, evenly across the dextran contrast."""
    usable = [e for e in candidates if np.isfinite(measures[e.tip].get("dextran_tip_contrast", np.nan))]
    endo = np.asarray([measures[e.tip].get("endothelium_tip_contrast", np.nan) for e in usable])
    if not np.isfinite(endo).any():
        return _evenly(usable, count, key=lambda e: measures[e.tip]["dextran_tip_contrast"])
    cuts = np.nanpercentile(endo, [100 / 3, 200 / 3])
    thirds = [[e for e, v in zip(usable, endo) if (np.isnan(v) and k == 0) or (np.isfinite(v) and lo < v <= hi)]
              for k, (lo, hi) in enumerate(((-np.inf, cuts[0]), (cuts[0], cuts[1]), (cuts[1], np.inf)))]
    chosen = []
    for k, third in enumerate(thirds):
        share = count // 3 + (1 if k < count % 3 else 0)
        chosen += _evenly(third, share, key=lambda e: measures[e.tip]["dextran_tip_contrast"])
    return chosen


PAGE = dict(
    title="Dead End Review G4: is the red chain a real vessel?",
    question="Is the red chain a real vessel?",
    lede=("One dead end (red, its heading cyan) of the E14.5 graph per card, in two planes at right angles. Each "
          "plane shows the segmentation the graph was built from, the dextran channel (Texas Red, what was "
          "segmented) and the endothelial channel (claudin-5, the vessel wall). Grey dots are other centrelines; "
          "the orange bar is 5 um. Your marks save as you go."),
    steps=["V if the red chain follows a real vessel, whatever happens at its end.",
           "N if there is no vessel there: the chain is an artefact and should go.",
           "P if it is a vessel but its last stretch is not (say roughly where, in the note).",
           "Tick what decided it, and add a note if it helps."],
    verdicts=[("V", "Vessel", "good"), ("N", "Not a vessel", "bad"), ("P", "Partly", "unsure"), ("?", "Unsure", "unsure")],
    evidence=["dextran", "endothelium", "segmentation shape", "context"],
)


def _load_mask(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(bool)
    import tifffile

    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    return np.asarray(_to_binary_volume_for_skeletonization(tifffile.imread(path)), dtype=bool)


def _load_channel(path: str, channel: int | None, shape) -> np.ndarray:
    import tifffile

    volume = tifffile.imread(path)
    if volume.ndim == 4:
        # A composite: ZCYX, the channel axis the short one.
        axis = int(np.argmin(volume.shape[:2]))
        volume = np.take(volume, int(channel or 0), axis=axis)
    if volume.shape != tuple(shape):
        raise SystemExit(f"{path}: {volume.shape} differs from the mask's {tuple(shape)}")
    return volume.astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mask", required=True, help=".npy, or the segmentation .tif(f), binarised as the pipeline does")
    parser.add_argument("--raw", required=True, help="the dextran channel (z y x)")
    parser.add_argument("--endothelial", required=True, help="the endothelial channel, or a ZCYX composite holding it")
    parser.add_argument("--endothelial-channel", type=int, default=1)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--exclude-keys", nargs="*", help="earlier grids' key CSVs: their tips are not shown again")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from haemolynx.graph.dead_ends import ISLAND_END, LEAVES_IMAGE, classify_dead_ends, image_extent_um
    from haemolynx.preprocessing import MaskSupport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(args.graph, "rb"))
    mask = _load_mask(args.mask)
    vs = tuple(args.voxel_size_zyx)
    channels = {
        "dextran": _load_channel(args.raw, None, mask.shape),
        "endothelium": _load_channel(args.endothelial, args.endothelial_channel, mask.shape),
    }
    ends = classify_dead_ends(G, extent_um=image_extent_um(mask.shape, vs), support=MaskSupport(mask, vs))
    shown = _shown_tips(args.exclude_keys, vs)
    candidates = [
        e for e in ends.values()
        if e.kind not in (LEAVES_IMAGE, ISLAND_END)
        and not (len(shown) and np.min(np.linalg.norm(shown - np.asarray(e.tip_um), axis=1)) <= EXCLUDE_WITHIN_UM)
    ]
    rng = random.Random(SEED)
    references = {name: _reference(G, volume, mask, vs, rng) for name, volume in channels.items()}
    print(f"{len(candidates)} candidate dead ends; typical vessel lift {references}")
    measures = {e.tip: measure(G, e, mask, channels, references, vs) for e in candidates}
    chosen = choose(candidates, measures)
    rng.shuffle(chosen)

    drawer = PanelDrawer(G, mask, {"Dextran": channels["dextran"], "Claudin-5": channels["endothelium"]}, vs)
    examples, rows = [], []
    for number, end in enumerate(chosen, start=1):
        ident = f"G4-{number:02d}"
        examples.append({"id": ident, "views": _views(drawer, G, end)})
        rows.append({"id": ident, "grid": "G4",
                     "tip_zyx_vox": tuple(int(round(c)) for c in np.asarray(end.tip_um) / np.asarray(vs)),
                     **end.measures(), **measures[end.tip]})
    data = {
        "batch": "G4", "title": PAGE["title"], "lede": PAGE["lede"], "question": PAGE["question"],
        "steps": PAGE["steps"], "evidence": PAGE["evidence"],
        "legend": [{"label": "the dead end judged", "color": "#ff3b30"}, {"label": "its heading", "color": "#2ad4ff"},
                   {"label": "other centrelines", "color": "#9a9a9a"}, {"label": "5 um", "color": "#ff9f0a"}],
        "verdicts": [{"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["verdicts"]],
        "examples": examples,
    }
    template = (HERE / "template_panels.html").read_text(encoding="utf-8")
    html = template.replace("__TITLE__", "Dead End Review G4").replace(
        "__EXAMPLES_JSON__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out / "G4_review.html").write_text(html, encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "G4_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "G4_references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    print(f"G4: {len(examples)} examples -> {out / 'G4_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
