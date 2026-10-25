#!/usr/bin/env python3
"""Build grid UR1: is a piece of segmentation no centreline covered a vessel?

``PLAN_raw_confirmation.md`` Phase 4 (P6). Mask recovery
(``graph.mask_recovery.recover_uncovered_mask_vessels``) traces the mask no
centreline's lumen covers -- in pieces of at least
``recovery_min_region_volume_um3`` that are somewhere wider than a voxel and a
half -- along the run's skeleton, and joins what it draws to the network. The
plan's question is whether the raw image can tell which of those pieces are a
lumen the graph missed (V) and which a halo, a speck, a wall or the side of a
vessel already drawn (H), so that recovery skips the H.

The pool is every piece recovery is handed: the graph it is handed
(``reconnect_orphan_and_dangling_nodes``' graph) gives the pieces
(``uncovered_mask_voxels``, 26-connected, recovery's own two filters) and the
graph it returns gives the edges it drew in each (``recovered``, joins apart).
On E14.5 it draws into 39 of 1,749: those are the decisions a skip rule would
change, but 30 of 39 would leave no fresh grid, and a piece it draws nothing
into asks the same question for the review list of segmented vessels without
a centreline (``network_review.find_uncovered_mask``).

Each piece is measured in both channels and by its shape (the key's columns):

- dextran (a plasma label): the median brightness of its voxels and of its
  deepest ones, in the run's units (0 background, 1 lumen; not clipped), and
  that over the brightness of the centrelines beside it -- the brightness
  read locally, since on E14.5 the stack's top 30 um reads several times its
  depths (the plan's depth risk);
- claudin-5 (the wall, the run's raw image): the median brightness of its
  voxels (1 the wall), whether walls enclose its deepest points
  (``RawEvidenceEndothelial.inside_walls``, E2) and how much of the drawn
  centreline runs with a wall either side (``vesselness``);
- shape: volume, widest radius, depth, how elongated, how far from the nearest
  centreline the graph had, and whether the finished graph covers it.

The 30 are :data:`DRAWN_SHARE` of the pieces recovery draws into and the rest
from those it does not, each group split at its median depth and each half
spread evenly across the plan's measure read locally (the dextran over its
neighbours'), then shuffled. Each card shows the piece outlined in its own plane and across it,
in the segmentation, the dextran and the claudin-5 channels, with the
centreline recovery draws through it.

Named UR1 ("uncovered regions"): grid N is the surface-coverage plan's lost
skeleton (2026-10-18). Its fresh check is UR2, built by the same script from
the pieces UR1 did not show (``--grid UR2 --exclude-keys UR1_key.csv``); with
``--drawn-only`` only those recovery draws into, all of them when there are no
more than 30.

Usage::

    python scripts/uncovered_review/build_ur1.py --before before_recovery.pkl \\
        --after after_recovery.pkl --final final.pkl --mask mask.npy \\
        --composite composite.tif --dextran-channel 0 --endothelial-channel 1 \\
        --voxel-size-zyx 1.0 0.98 0.98 --out outputs/uncovered_review

    ... --grid UR2 --exclude-keys outputs/uncovered_review/UR1_key.csv --drawn-only
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
sys.path.insert(0, str(HERE.parent / "dead_end_review"))

from build_g4 import PanelDrawer, _load_mask  # noqa: E402
from build_grids import _evenly  # noqa: E402

GRID = "UR1"
SEED = 20261026
COUNT = 30
#: The deepest points a piece is asked about (E2), at least this far apart (um).
DEEP_POINTS = 5
DEEP_SPACING_UM = 2.0
#: Centreline samples within this of a piece are its neighbours (um).
NEIGHBOUR_UM = 25.0
#: How many of the 30 recovery draws a centreline into.
DRAWN_SHARE = 10
#: The piece's outline and the centreline drawn through it.
OUTLINE_COLOUR = "#ffd60a"
DRAWN_COLOUR = "#2ad4ff"


class Regions:
    """The pieces of uncovered mask recovery is handed, with recovery's own
    two filters, and a lookup from a voxel to its piece."""

    def __init__(self, G, support):
        from haemolynx.graph import mask_recovery as mr

        spacing = np.asarray(support.voxel_size_zyx, dtype=float)
        voxel_volume = float(np.prod(spacing))
        self.shape = tuple(support.mask.shape)
        self.pieces: list[np.ndarray] = []
        self.radii: list[float] = []
        for region in mr._components(mr.uncovered_mask_voxels(G, support)):
            if len(region) * voxel_volume < mr.DEFAULT_MIN_REGION_VOLUME_UM3:
                continue
            radius = float(np.max(support.radius(region * spacing)))
            if radius < mr.MIN_INSCRIBED_RADIUS_VOXELS * float(spacing.min()):
                continue
            self.pieces.append(np.asarray(region, dtype=np.int64))
            self.radii.append(radius)
        flat = [np.ravel_multi_index(tuple(p.T), self.shape) for p in self.pieces]
        owner = [np.full(len(f), k, dtype=np.int32) for k, f in enumerate(flat)]
        order = np.argsort(np.concatenate(flat)) if flat else np.empty(0, dtype=np.int64)
        self._flat = np.concatenate(flat)[order] if flat else np.empty(0, dtype=np.int64)
        self._owner = np.concatenate(owner)[order] if owner else np.empty(0, dtype=np.int32)

    def of_voxels(self, voxels: np.ndarray) -> np.ndarray:
        """The piece each voxel (index triples) lies in, -1 for none."""
        voxels = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
        inside = np.all((voxels >= 0) & (voxels < np.asarray(self.shape)), axis=1)
        out = np.full(len(voxels), -1, dtype=np.int32)
        if not inside.any() or not len(self._flat):
            return out
        flat = np.ravel_multi_index(tuple(voxels[inside].T), self.shape)
        at = np.clip(np.searchsorted(self._flat, flat), 0, len(self._flat) - 1)
        hit = self._flat[at] == flat
        found = np.full(len(flat), -1, dtype=np.int32)
        found[hit] = self._owner[at[hit]]
        out[inside] = found
        return out

    def label_volume(self) -> np.ndarray:
        """Every piece's number + 1 at its voxels, 0 elsewhere (for drawing)."""
        volume = np.zeros(self.shape, dtype=np.uint16)
        for k, piece in enumerate(self.pieces):
            volume[tuple(piece.T)] = k + 1
        return volume


def drawn_in(after, regions: Regions, voxel_size) -> dict[int, dict]:
    """What recovery drew in each piece: its recovered edges (centrelines, and
    the joins apart), each given to the piece most of its voxels lie in."""
    spacing = np.asarray(voxel_size, dtype=float)
    drawn: dict[int, dict] = {}
    for u, v, k, data in after.edges(keys=True, data=True):
        if not data.get("recovered"):
            continue
        points = np.asarray(data.get("voxels", []), dtype=float).reshape(-1, 3)
        if not len(points):
            continue
        owners = regions.of_voxels(np.rint(points / spacing).astype(np.int64))
        owners = owners[owners >= 0]
        if not len(owners):
            continue
        piece = int(np.bincount(owners).argmax())
        entry = drawn.setdefault(piece, {"edges": [], "joins": 0, "drawn_um": 0.0, "paths": []})
        if data.get("reconnected"):
            entry["joins"] += 1
            continue
        entry["edges"].append((u, v, k))
        entry["drawn_um"] += float(data.get("length") or np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        entry["paths"].append(points)
    return drawn


def deepest_points(points_um: np.ndarray, radii: np.ndarray, count: int = DEEP_POINTS) -> np.ndarray:
    """Up to *count* of *points_um*, deepest first, each at least
    :data:`DEEP_SPACING_UM` from those taken before."""
    taken: list[np.ndarray] = []
    for i in np.argsort(-radii, kind="stable"):
        if all(np.linalg.norm(points_um[i] - t) >= DEEP_SPACING_UM for t in taken):
            taken.append(points_um[i])
            if len(taken) == count:
                break
    return np.asarray(taken, dtype=float).reshape(-1, 3)


def unclipped(evidence, points_um: np.ndarray) -> np.ndarray:
    """Brightness in the run's units, 0 at the background and 1 at the
    stain's level, not clipped: a lumen brighter than the run's level reads
    above 1."""
    c = evidence.calibration
    span = c.level - c.background
    values = evidence._values(points_um)
    return (values - c.background) / span if span else values * 0.0


def measure(k, regions: Regions, support, plasma, endothelial, centrelines, drawn, final_uncovered) -> dict:
    """Every measure of piece *k*: the key's columns."""
    from haemolynx.graph._helpers import points_within

    spacing = np.asarray(support.voxel_size_zyx, dtype=float)
    voxels = regions.pieces[k]
    points = voxels * spacing
    rng = np.random.default_rng(k)
    sample = points if len(points) <= 2000 else points[np.sort(rng.choice(len(points), 2000, replace=False))]
    radii = support.radius(sample)
    deep = deepest_points(sample, radii)
    centre = points.mean(axis=0)
    spread = np.linalg.svd(points - centre, compute_uv=False) / np.sqrt(max(len(points), 1))
    nearest, _ = centrelines.tree.query(centre) if centrelines.tree is not None else (np.inf, None)
    near = points_within(centrelines.tree, sample, NEIGHBOUR_UM) if centrelines.tree is not None else []
    neighbour = centrelines.points[np.asarray(near, dtype=np.intp)] if len(near) else np.empty((0, 3))
    row: dict = {
        "region": k,
        "centre_zyx_vox": tuple(int(round(c)) for c in centre / spacing),
        "voxels": int(len(voxels)),
        "volume_um3": float(len(voxels) * np.prod(spacing)),
        "radius_um": float(regions.radii[k]),
        "depth_um": float(centre[0]),
        "elongation": float(spread[0] / spread[1]) if len(spread) > 1 and spread[1] > 0 else None,
        "nearest_centreline_um": float(nearest),
        "drawn_um": float(drawn["drawn_um"]),
        "drawn_edges": len(drawn["edges"]),
        "drawn_joins": int(drawn["joins"]),
        "final_uncovered_fraction": _share_in(np.ravel_multi_index(tuple(voxels.T), regions.shape),
                                              final_uncovered),
    }
    if plasma is not None:
        bright = unclipped(plasma, sample)
        deep_bright = unclipped(plasma, deep)
        row["dextran_median"] = float(np.median(bright))
        row["dextran_deep_median"] = float(np.median(deep_bright))
        if len(neighbour):
            level = float(np.median(unclipped(plasma, neighbour)))
            row["dextran_neighbour_median"] = level
            row["dextran_over_neighbour"] = row["dextran_median"] / level if level > 0 else None
    if endothelial is not None:
        row["claudin_median"] = float(np.median(unclipped(endothelial, sample)))
        inside = endothelial.inside_walls(deep, float(np.median(support.radius(deep))) if len(deep) else 1.0,
                                          others_um=neighbour if len(neighbour) else None)
        row["claudin_deep_lumen_fraction"] = inside.lumen_fraction
        row["claudin_deep_wall_fraction"] = float(inside.wall.mean()) if len(inside.wall) else 0.0
        row["claudin_deep_shared_fraction"] = inside.shared_fraction
        walled = [endothelial.signal_along(path, float(np.median(support.radius(path))))
                  for path in drawn["paths"] if len(path) > 1]
        if walled:
            weights = np.asarray([max(w.samples, 1) for w in walled], dtype=float)
            row["claudin_drawn_walled_fraction"] = float(np.average([w.fraction_bright for w in walled],
                                                                    weights=weights))
    return row


def _share_in(values: np.ndarray, sorted_set: np.ndarray) -> float:
    """The share of *values* found in the sorted array *sorted_set*."""
    if not len(values) or not len(sorted_set):
        return 0.0
    at = np.clip(np.searchsorted(sorted_set, values), 0, len(sorted_set) - 1)
    return float(np.mean(sorted_set[at] == values))


def local_brightness(row: dict) -> float:
    """The plan's measure read locally: the dextran over its neighbours'; a
    piece with no centreline within :data:`NEIGHBOUR_UM` reads 0."""
    value = row.get("dextran_over_neighbour")
    return float(value) if value is not None else 0.0


def _shares(count: int, parts: int) -> list[int]:
    return [count // parts + (1 if k < count % parts else 0) for k in range(parts)]


def choose(rows: list[dict], count: int = COUNT, drawn_share: int = DRAWN_SHARE) -> list[dict]:
    """*count* of *rows*: *drawn_share* of those recovery draws into and the
    rest from the others, each group split at its median depth and each half
    spread evenly across :func:`local_brightness`."""
    usable = [r for r in rows if r.get("dextran_median") is not None]
    drawn = [r for r in usable if r["drawn_um"] > 0]
    others = [r for r in usable if r["drawn_um"] <= 0]
    take = min(drawn_share, len(drawn))
    key = lambda r: (local_brightness(r), r["volume_um3"])  # noqa: E731
    chosen = []
    for group, share in ((drawn, take), (others, count - take)):
        if not group or share <= 0:
            continue
        cut = float(np.median([r["depth_um"] for r in group]))
        halves = ([r for r in group if r["depth_um"] <= cut], [r for r in group if r["depth_um"] > cut])
        picked = []
        for half, part in zip(halves, _shares(share, 2)):
            picked += _evenly(half, part, key=key)
        rest = [r for r in group if all(r is not p for p in picked)]
        picked += _evenly(rest, share - len(picked), key=key)
        chosen += picked
    return chosen


class RegionDrawer(PanelDrawer):
    """G4's planes, with the piece outlined in every channel."""

    def __init__(self, G, mask, channels, voxel_size, labels: np.ndarray):
        super().__init__(G, mask, channels, voxel_size)
        self.labels = labels
        self.region = None

    def draw_plane(self, label, c, a, b, normal, R, lines, half_thickness, marker=None) -> dict:
        from scipy import ndimage
        from skimage.measure import find_contours

        plane = super().draw_plane(label, c, a, b, normal, R, lines, half_thickness, marker)
        g = np.arange(-R, R + 1e-9, 0.3)
        U, V = np.meshgrid(g, g, indexing="ij")
        inside = np.zeros(U.shape, dtype=bool)
        for w in np.arange(-half_thickness, half_thickness + 1e-9, 0.5):
            coords = ((c + U[..., None] * a + V[..., None] * b + w * normal) / self.vs).reshape(-1, 3).T
            inside |= ndimage.map_coordinates(self.labels, coords, order=0, mode="constant").reshape(U.shape) == (
                self.region + 1)
        points: list = []
        for contour in find_contours(np.pad(inside, 1).astype(float), 0.5):
            if points:
                points.append(None)
            points += np.round(np.column_stack([g[0] + 0.3 * (contour[:, 0] - 1),
                                                g[0] + 0.3 * (contour[:, 1] - 1)]), 2).tolist()
        if points:
            plane["overlays"].append(dict(kind="line", group="graph", color=OUTLINE_COLOUR, width=1.6,
                                          marker=True, points=points))
        return plane


def views(drawer: RegionDrawer, k: int, regions: Regions, drawn: dict, voxel_size) -> list[dict]:
    """The piece in its own plane (its two longest directions) and across it."""
    spacing = np.asarray(voxel_size, dtype=float)
    points = regions.pieces[k] * spacing
    if len(points) > 4000:
        points = points[np.random.default_rng(k).choice(len(points), 4000, replace=False)]
    drawer.region = k
    lines = [dict(points=[q for path in drawn["paths"] for q in [*path.tolist(), None]][:-1],
                  color=DRAWN_COLOUR, width=2.4)]
    half_thickness = float(np.clip(regions.radii[k], 2.0, 5.0))
    return drawer.views(points, None, [line for line in lines if line["points"]], half_thickness)


PAGE = dict(
    title="Uncovered Region Review {grid}: is the outlined piece a vessel the graph missed?",
    question="Is the outlined piece a vessel of its own?",
    lede=("One piece of the E14.5 segmentation per card, outlined in yellow: segmented vessel that no centreline "
          "of the graph covered before mask recovery ran. Where mask recovery draws a centreline through it and "
          "joins it to the network, that is drawn in cyan; most pieces it leaves alone. Each card shows the piece in its own plane and across it, in the segmentation "
          "the graph was built from, the dextran channel (plasma) and the claudin-5 channel (the vessel wall: a "
          "vessel is two bright walls with a dark lumen between). Grey dots are the centrelines the graph already "
          "had; the orange bar is 5 um. Your marks save as you go."),
    steps=["V if the outlined piece is a vessel of its own that the graph missed: it should get a centreline.",
           "H if it is not a vessel of its own: a halo, a speck, a wall, or the side of a vessel that already has "
           "a centreline. Recovery should leave it.",
           "? if you cannot tell. Tick what decided it, and add a note if it helps."],
    verdicts=[("V", "Vessel missed", "good"), ("H", "Not a vessel of its own", "bad"), ("?", "Unsure", "unsure")],
    evidence=["dextran", "endothelium", "segmentation shape", "context"],
)


def page_data(examples: list, grid: str = GRID) -> dict:
    return {
        "batch": grid, "title": PAGE["title"].format(grid=grid), "lede": PAGE["lede"], "question": PAGE["question"],
        "steps": PAGE["steps"], "evidence": PAGE["evidence"],
        "legend": [{"label": "the piece judged", "color": OUTLINE_COLOUR},
                   {"label": "the centreline recovery draws, if any", "color": DRAWN_COLOUR},
                   {"label": "centrelines the graph had", "color": "#9a9a9a"}, {"label": "5 um", "color": "#ff9f0a"}],
        "verdicts": [{"value": v, "label": label, "short": v, "tone": tone} for v, label, tone in PAGE["verdicts"]],
        "examples": examples,
    }


def shown_centres(key_paths) -> set[str]:
    """The pieces earlier grids showed, by their centre voxel as the keys
    write it (``centre_zyx_vox``)."""
    shown = set()
    for path in key_paths or ():
        with open(path, encoding="utf-8") as handle:
            shown |= {row["centre_zyx_vox"] for row in csv.DictReader(handle) if row.get("centre_zyx_vox")}
    return shown


def fresh(rows: list[dict], shown: set[str], *, drawn_only: bool = False) -> list[dict]:
    """The rows no earlier grid showed; with *drawn_only*, only the pieces
    recovery draws a centreline into."""
    return [r for r in rows if str(r["centre_zyx_vox"]) not in shown and (not drawn_only or r["drawn_um"] > 0)]


def _channel(path: str, channel: int, shape) -> np.ndarray:
    from haemolynx.preprocessing.raw_evidence import load_raw_channel

    volume = load_raw_channel(path, channel)
    if volume.shape != tuple(shape):
        raise SystemExit(f"{path} channel {channel}: {volume.shape} differs from the mask's {tuple(shape)}")
    return volume


def _evidence(volume, mask, voxel_size, stain):
    from haemolynx.preprocessing import raw_evidence as rx

    calibration = rx.calibrate_raw(volume, mask, voxel_size, stain=stain)
    if calibration is None:
        print(f"The {stain} channel did not calibrate round the mask; its columns are left out")
        return None
    print(calibration.describe())
    return rx.make_raw_evidence(volume, mask.shape, voxel_size, calibration)


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="the graph mask recovery was handed")
    parser.add_argument("--after", required=True, help="the graph it returned")
    parser.add_argument("--final", required=True, help="the finished graph")
    parser.add_argument("--mask", required=True, help=".npy, or the segmentation .tif(f), binarised as the pipeline does")
    parser.add_argument("--composite", required=True, help="the raw ZCYX composite holding both channels")
    parser.add_argument("--dextran-channel", type=int, default=0)
    parser.add_argument("--endothelial-channel", type=int, default=1)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default=GRID, help="the grid's name (UR2 for the fresh check)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-keys", nargs="*", default=[], help="earlier grids' keys: their pieces are left out")
    parser.add_argument("--drawn-only", action="store_true", help="only the pieces recovery draws a centreline into")
    args = parser.parse_args(argv)
    grid = args.grid

    from haemolynx.graph.dead_ends import _Centrelines
    from haemolynx.graph.mask_recovery import uncovered_mask_voxels
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing import raw_evidence as rx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = tuple(args.voxel_size_zyx)
    mask = _load_mask(args.mask)
    support = MaskSupport(mask, vs)
    with open(args.before, "rb") as handle:
        before = pickle.load(handle)
    with open(args.after, "rb") as handle:
        after = pickle.load(handle)
    with open(args.final, "rb") as handle:
        final = pickle.load(handle)
    regions = Regions(before, support)
    drawn = drawn_in(after, regions, vs)
    print(f"{len(regions.pieces)} uncovered pieces handed to recovery; it draws a centreline in {len(drawn)}")
    final_uncovered = np.sort(np.ravel_multi_index(tuple(np.asarray(uncovered_mask_voxels(final, support)).T),
                                                   mask.shape))
    dextran = _channel(args.composite, args.dextran_channel, mask.shape)
    claudin = _channel(args.composite, args.endothelial_channel, mask.shape)
    plasma = _evidence(dextran, mask, vs, rx.PLASMA_COLUMN)
    endothelial = _evidence(claudin, mask, vs, rx.ENDOTHELIAL)
    centrelines = _Centrelines(before, support)
    nothing = {"edges": [], "joins": 0, "drawn_um": 0.0, "paths": []}
    rows = [measure(k, regions, support, plasma, endothelial, centrelines, drawn.get(k, nothing), final_uncovered)
            for k in range(len(regions.pieces))]
    print(f"{sum(r['drawn_um'] > 0 for r in rows)} pieces with a centreline drawn in them")
    pool = fresh(rows, shown_centres(args.exclude_keys), drawn_only=args.drawn_only)
    print(f"{len(pool)} pieces in the pool")
    chosen = choose(pool, drawn_share=COUNT if args.drawn_only else DRAWN_SHARE)
    random.Random(args.seed).shuffle(chosen)

    drawer = RegionDrawer(before, mask, {"Dextran": dextran, "Claudin-5": claudin}, vs, regions.label_volume())
    examples, key = [], []
    for number, row in enumerate(chosen, start=1):
        ident = f"{grid}-{number:02d}"
        k = row["region"]
        examples.append({"id": ident, "where": "centre at z {}, y {}, x {} (voxels)".format(*row["centre_zyx_vox"]),
                         "views": views(drawer, k, regions, drawn.get(k, nothing), vs)})
        key.append({"id": ident, "grid": grid, **row})
    template = (HERE.parent / "dead_end_review" / "template_panels.html").read_text(encoding="utf-8")
    html = template.replace("__TITLE__", f"Uncovered Region Review {grid}").replace(
        "__EXAMPLES_JSON__", json.dumps(page_data(examples, grid), separators=(",", ":")).replace("</", "<\\/"))
    (out / f"{grid}_review.html").write_text(html, encoding="utf-8")
    _write_csv(out / f"{grid}_key.csv", key)
    _write_csv(out / f"{grid}_candidates.csv", rows)
    calibrations = {name: None if e is None else e.calibration.as_dict()
                    for name, e in (("dextran", plasma), ("claudin-5", endothelial))}
    (out / f"{grid}_calibration.json").write_text(json.dumps(calibrations, indent=2), encoding="utf-8")
    print(f"{grid}: {len(examples)} examples -> {out / f'{grid}_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
