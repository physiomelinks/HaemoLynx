"""Re-score grids H and H2 with the raw image (PLAN_raw_confirmation.md, Phase 2).

No mask measure separated Harvey's marks on either grid: H (30 doubled runs,
D = one vessel drawn twice / T = two vessels) best 21/30 against 19 all one
way; H2 (30 dead-end chains beside a wider vessel, X = not a vessel / V = a
vessel) best 16/30 = all one way. This reads the one measure neither grid
tried: ``RawEvidence.wall_between`` -- the dip of the raw profile along the
chord from each sample of the run (or chain) to the centreline it runs
beside. Two lumens have a wall between them; one lumen does not.

Each run is rebuilt from the graph its grid was drawn from, so the samples
are the ones that were marked; each chain's partner is the nearest other
centreline sample within the same-lumen reach. Scored leave-one-out with
``scripts/dead_end_review/fit_marks.loo_threshold``, as every grid is.

    python scripts/doubled_run_review/rescore_raw.py \
        --review outputs/doubled_run_review \
        --graph-h outputs/missed_branches_fix/working_tree/final.pkl \
        --graph-h2 outputs/clean_graph_phase1_all_runs/working_tree/final.pkl \
        --mask outputs/clean_graph_phase1e/mask.npy --raw <C1.tif> \
        --voxel-size-zyx 1.0 0.98 0.98

Writes ``FIT_raw.md``, ``H_raw_measures.csv`` and ``H2_raw_measures.csv``
beside the keys. Run with the raw-evidence branch's ``src`` on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dead_end_review"))

#: A pair with at least this much wall evidence counts as "a wall between".
WALL_SEEN = 0.3


def _edge_tuple(text: str):
    u, v, k = text.split("-")
    return int(u), int(v), int(k)


def _pair_measures(evidence, points: np.ndarray, partners: np.ndarray) -> dict:
    walls = np.asarray([evidence.wall_between(p, q) for p, q in zip(points, partners)], dtype=float)
    own = evidence.brightness(points)
    if not len(walls):
        return dict(wall_median=np.nan, wall_max=np.nan, wall_p75=np.nan, wall_seen=np.nan, own_brightness=np.nan, pairs=0)
    return dict(
        wall_median=float(np.median(walls)),
        wall_max=float(walls.max()),
        wall_p75=float(np.percentile(walls, 75)),
        wall_seen=float(np.mean(walls >= WALL_SEEN)),
        own_brightness=float(np.median(own)),
        pairs=int(len(walls)),
    )


def rescore_h(G, support, evidence, key_rows: list[dict]) -> list[dict]:
    from scipy.spatial import cKDTree

    from build_grid import doubled_runs

    runs, _paths = doubled_runs(G, support)
    by_pair: dict[tuple, list] = {}
    text = lambda eid: "-".join(str(part) for part in eid)  # the key writes an edge as u-v-key
    for run in runs:
        by_pair.setdefault((text(run["own"]), text(run["partner"])), []).append(run)
    out = []
    for row in key_rows:
        mid = np.asarray(json.loads(row["mid_zyx_um"].replace("(", "[").replace(")", "]")), dtype=float)
        candidates = by_pair.get((row["own_edge"], row["partner_edge"]), []) + by_pair.get(
            (row["partner_edge"], row["own_edge"]), []
        )
        if not candidates:
            out.append(dict(id=row["id"], matched=False))
            continue
        run = min(candidates, key=lambda r: float(np.linalg.norm(np.mean(r["points"], axis=0) - mid)))
        partner_tree = cKDTree(run["partner_points"])
        _d, nearest = partner_tree.query(run["points"])
        measures = _pair_measures(evidence, run["points"], run["partner_points"][nearest])
        out.append(dict(id=row["id"], matched=True, **measures))
    return out


def _chain_edges(G, tip, junction):
    """The edges from *tip* through degree-2 nodes to *junction*."""
    edges, node, previous = [], tip, None
    for _ in range(10_000):
        nexts = [(u, v, k) for u, v, k in G.edges(node, keys=True) if (v if u == node else u) != previous]
        if not nexts:
            break
        u, v, k = nexts[0]
        edges.append((u, v, k))
        previous, node = node, (v if u == node else u)
        if node == junction or G.degree(node) != 2:
            break
    return edges


def rescore_h2(G, support, evidence, key_rows: list[dict]) -> list[dict]:
    from scipy.spatial import cKDTree

    from haemolynx.graph._helpers import edge_sample_points
    from haemolynx.preprocessing.bridge_mask_support import _densify, _sample_step

    step = _sample_step(support.voxel_size_zyx)
    node_pos = {n: np.asarray(d["pos"], dtype=float) for n, d in G.nodes(data=True) if "pos" in d}
    all_points, owners = [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        if u not in node_pos or v not in node_pos:
            continue
        dense = _densify(edge_sample_points(u, v, data, node_pos), step)
        all_points.append(dense)
        owners.extend([(u, v, k)] * len(dense))
    all_points = np.vstack(all_points)
    owners = np.asarray(owners, dtype=object)
    tree = cKDTree(all_points)
    out = []
    for row in key_rows:
        tip, junction = int(row["tip"]), int(row["junction"])
        if not (G.has_node(tip) and G.has_node(junction)):
            out.append(dict(id=row["id"], matched=False))
            continue
        chain = _chain_edges(G, tip, junction)
        chain_set = {tuple(e) for e in chain} | {(v, u, k) for u, v, k in chain}
        samples = np.vstack([_densify(edge_sample_points(u, v, G.edges[u, v, k], node_pos), step) for u, v, k in chain])
        radii = support.radius(samples)
        reach = 2.0 * float(radii.max()) + 1.0
        points, partners = [], []
        for sample, radius, hits in zip(samples, radii, tree.query_ball_point(samples, r=reach)):
            hits = [h for h in hits if tuple(owners[h]) not in chain_set]
            if not hits:
                continue
            nearest = min(hits, key=lambda h: np.linalg.norm(all_points[h] - sample))
            if np.linalg.norm(all_points[nearest] - sample) > 2.0 * radius + 1.0 + float(support.radius(all_points[nearest][None])[0]):
                continue
            points.append(sample)
            partners.append(all_points[nearest])
        measures = _pair_measures(evidence, np.asarray(points), np.asarray(partners)) if points else _pair_measures(evidence, np.zeros((0, 3)), np.zeros((0, 3)))
        measures["beside_fraction"] = float(len(points) / max(len(samples), 1))
        out.append(dict(id=row["id"], matched=True, **measures))
    return out


def _report(name: str, rows: list[dict], marks: dict, positive: str, negative: str) -> str:
    from fit_marks import loo_threshold

    judged = [(r, marks[r["id"]]["verdict"]) for r in rows if r.get("matched") and marks.get(r["id"], {}).get("verdict") in (positive, negative)]
    y = np.asarray([v == positive for _r, v in judged], dtype=bool)
    lines = [f"## Grid {name}: {positive} against {negative}, by the raw image", "",
             f"Judged {len(judged)} (matched {sum(r.get('matched', False) for r in rows)}/{len(rows)}); "
             f"all one way gets {max(y.sum(), (~y).sum()) if len(y) else 0}/{len(judged)}.", "",
             "| measure | LOO right | fitted rule | fitted right |", "|---|---|---|---|"]
    measures = [m for m in ("wall_median", "wall_p75", "wall_max", "wall_seen", "own_brightness", "beside_fraction") if judged and m in judged[0][0]]
    scored = []
    for measure in measures:
        values = np.asarray([float(r[measure]) for r, _v in judged], dtype=float)
        result = loo_threshold(values, y)
        if result is not None:
            scored.append((result["loo"], measure, result))
    for loo, measure, result in sorted(scored, key=lambda s: -s[0]):
        rule = f"{positive} when {measure} {'>' if result['above'] else '<='} {result['threshold']:.3g}"
        lines.append(f"| {measure} | {loo}/{result['n']} | {rule} | {result['fit_hits']}/{result['n']} |")
    lines += ["", "| id | mark | wall median | wall p75 | wall max | wall seen | own brightness | pairs |", "|---|---|---|---|---|---|---|---|"]
    for r, v in judged:
        lines.append(
            f"| {r['id']} | {v} | {r['wall_median']:.2f} | {r['wall_p75']:.2f} | {r['wall_max']:.2f} | "
            f"{r['wall_seen']:.2f} | {r['own_brightness']:.2f} | {r['pairs']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--graph-h", required=True)
    parser.add_argument("--graph-h2", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)

    import tifffile

    import importlib.util

    # Two scripts folders each have a fit_marks.py: the dead-end one (first on
    # the path) scores, this folder's reads the H/H2 marks.
    spec = importlib.util.spec_from_file_location("doubled_run_fit_marks", HERE / "fit_marks.py")
    doubled_run_fit_marks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doubled_run_fit_marks)
    read_marks = doubled_run_fit_marks.read_marks
    from haemolynx.preprocessing import MaskSupport
    from haemolynx.preprocessing.raw_evidence import calibrate_raw, make_raw_evidence

    review = Path(args.review)
    mask = np.load(args.mask, mmap_mode="r")
    raw = tifffile.imread(args.raw)
    support = MaskSupport(mask, args.voxel_size_zyx)
    calibration = calibrate_raw(raw, mask, args.voxel_size_zyx)
    if calibration is None:
        raise SystemExit("The raw image could not be calibrated")
    evidence = make_raw_evidence(raw, mask.shape, args.voxel_size_zyx, calibration)
    report = [f"# Raw re-score of grids H and H2 (wall_between on the dextran channel)", "",
              f"Calibration: {calibration.describe()}.", ""]
    for grid, graph_path, positive, negative in (("H", args.graph_h, "D", "T"), ("H2", args.graph_h2, "X", "V")):
        G = pickle.load(open(graph_path, "rb"))
        key_rows = list(csv.DictReader(open(review / f"{grid}_key.csv", encoding="utf-8")))
        rows = rescore_h(G, support, evidence, key_rows) if grid == "H" else rescore_h2(G, support, evidence, key_rows)
        marks = read_marks(review, grid)
        for r in rows:
            r["mark"] = marks.get(r["id"], {}).get("verdict")
        with open(review / f"{grid}_raw_measures.csv", "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report.append(_report(grid, rows, marks, positive, negative))
        report.append(f"Graph: `{graph_path}`. Raw queries {evidence.stats.queries}, voxels {evidence.stats.voxels_read}, {evidence.stats.seconds:.1f} s so far.\n")
    (review / "FIT_raw.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
