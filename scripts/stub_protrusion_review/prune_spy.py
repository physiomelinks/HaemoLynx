"""A read-only spy on ``graph.assemble.prune_vascular_stubs``: before each call,
every dead-end chain of the graph it is handed is measured as the protrusion
test sees it, and two ways it could see it instead; after the call, which tips
went. Each call's input graph is pickled, so a grid can draw the stub as the
prune saw it. The graph the build makes is unchanged. Install it before a run:
``prune_spy.install(graph.assemble, graph.prune, out_dir, image_extent_um)``.

Measures per chain (``records.jsonl``, one line each):
- ``ratio_capped``: today's test (``prune.inside_other_lumens``): the 16 nearest
  centreline samples, the chain's own masked out, max over the chain of the
  least (distance - lumen radius), over the tip's lumen radius. < 1 prunes.
- ``ratio_others``: the same over the 16 nearest samples of *other* centrelines
  (the 2026-10-11 variant that pruned collinear face ends).
- ``deviation_deg``: how far the stub turns from running straight on from one
  of the other vessels at its junction (0 = a collinear continuation).
- ``face_um``: the tip's distance to the nearest image face.
"""
from __future__ import annotations

import functools
import inspect
import json
import pickle
from pathlib import Path

import numpy as np

DIRECTION_UM = 5.0


def _direction(path: np.ndarray) -> np.ndarray | None:
    """Unit direction of *path* (junction first) over its first DIRECTION_UM."""
    if len(path) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    end = path[min(int(np.searchsorted(arc, DIRECTION_UM)), len(path) - 1)]
    vector = end - path[0]
    norm = float(np.linalg.norm(vector))
    return None if norm == 0 else vector / norm


def install(assemble, prune, out: Path, extent_um) -> None:
    from haemolynx.graph._helpers import densify_polyline, edge_sample_points

    out.mkdir(parents=True, exist_ok=True)
    real = assemble.prune_vascular_stubs
    calls = [0]
    records = open(out / "records.jsonl", "w", encoding="utf-8")
    extent = np.asarray(extent_um, dtype=float)

    def edge_path_from(G, node, other, key):
        data = G.edges[node, other, key]
        pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in (node, other)}
        return densify_polyline(edge_sample_points(node, other, data, pos), max_step_um=0.5)

    @functools.wraps(real)
    def spy(G, *args, **kwargs):
        calls[0] += 1
        call = calls[0]
        caller = " < ".join(f.function for f in inspect.stack()[1:4])
        lumen_radius_at = kwargs.get("lumen_radius_at")
        inside_lumen = kwargs.get("inside_lumen")
        radius_at = kwargs.get("radius_at")
        multiple = float(kwargs.get("radius_multiple") or 0.0)
        rows = []
        if lumen_radius_at is not None and inside_lumen is not None and G.number_of_edges():
            index = prune._centreline_index(G, lumen_radius_at)
            tree, radii, owners, ids, points = index
            for tip in [n for n in G.nodes if G.degree(n) == 1]:
                nodes, edges = prune._chain(G, tip)
                junction = nodes[-1]
                if G.degree(junction) <= 1 or any("pos" not in G.nodes[n] for n in nodes):
                    continue
                path = prune._chain_path(G, edges)
                if len(path) < 2:
                    continue
                own = [ids.get((frozenset((a, b)), key), -1) for a, b, key in edges]
                tip_radius = float(np.asarray(lumen_radius_at(path[-1:]), dtype=float).reshape(-1)[0])
                k = min(16, len(owners))
                d, i = tree.query(path, k=k)
                d, i = d.reshape(len(path), k), i.reshape(len(path), k)
                excess = np.where(np.isin(owners[i], own), np.inf, d - radii[i])
                capped = float(np.max(np.min(excess, axis=1)))
                # The 16 nearest of other centrelines: widen the query until
                # every sample has 16 that are not its own (or all there are).
                not_own = ~np.isin(owners, own)
                others = np.inf
                if not_own.any():
                    k2 = min(len(owners), 64)
                    while True:
                        d2, i2 = tree.query(path, k=k2)
                        d2, i2 = d2.reshape(len(path), k2), i2.reshape(len(path), k2)
                        ok = not_own[i2]
                        if ok.sum(axis=1).min() >= min(16, int(not_own.sum())) or k2 >= len(owners):
                            break
                        k2 = min(len(owners), k2 * 4)
                    ex2 = np.where(ok, d2 - radii[i2], np.inf)
                    order = np.argsort(np.where(ok, d2, np.inf), axis=1)[:, :16]
                    ex2 = np.take_along_axis(ex2, order, axis=1)
                    others = float(np.max(np.min(ex2, axis=1)))
                stub_dir = _direction(path)
                deviation = None
                if stub_dir is not None:
                    best = 180.0
                    last = edges[-1]
                    for _j, other, key in G.edges(junction, keys=True):
                        if (frozenset((_j, other)), key) == (frozenset((last[0], last[1])), last[2]):
                            continue
                        if other == junction:
                            continue
                        e_dir = _direction(edge_path_from(G, junction, other, key))
                        if e_dir is None:
                            continue
                        cos = float(np.clip(np.dot(stub_dir, -e_dir), -1.0, 1.0))
                        best = min(best, float(np.degrees(np.arccos(cos))))
                    deviation = best
                tip_pos = path[-1]
                face = float(np.min(np.minimum(tip_pos, extent - tip_pos)))
                length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                parent_r = float(radius_at(path[0])) if radius_at is not None else None
                in_mask = int(np.count_nonzero(inside_lumen(path)))
                rows.append({
                    "call": call, "tip": [float(c) for c in tip_pos], "junction": [float(c) for c in path[0]],
                    "tip_node": repr(tip), "junction_node": repr(junction),
                    "length_um": round(length, 3), "tip_radius_um": round(tip_radius, 3),
                    "parent_radius_um": None if parent_r is None else round(parent_r, 3),
                    "threshold_um": None if parent_r is None or multiple <= 0 else round(multiple * parent_r, 3),
                    "protrusion_capped_um": capped, "protrusion_others_um": others,
                    "ratio_capped": capped / tip_radius if tip_radius > 0 else None,
                    "ratio_others": others / tip_radius if tip_radius > 0 else None,
                    "deviation_deg": deviation, "face_um": round(face, 3),
                    "half_off_mask": 2 * in_mask < len(path),
                    "tip_near_junction": parent_r is not None and parent_r > 0
                    and float(np.linalg.norm(path[-1] - path[0])) < parent_r + 1.0,
                    "junction_degree": int(G.degree(junction)), "chain_edges": len(edges),
                })
            with open(out / f"call_{call:03d}.pkl", "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Which tips went: by node id (the prune works on a copy, ids kept).
        tip_ids = {repr(n): n for n in G.nodes if G.degree(n) == 1}
        result = real(G, *args, **kwargs)
        for row in rows:
            row["removed"] = not result.has_node(tip_ids[row["tip_node"]])
            row["caller"] = caller
            records.write(json.dumps(row) + "\n")
        records.flush()
        return result

    assemble.prune_vascular_stubs = spy
