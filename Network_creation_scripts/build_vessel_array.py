#!/usr/bin/env python3
"""
Build a bond-graph vessel array from a curated network, interactively.

    python build_vessel_array.py <curated.vtp | curated.pkl> [--image im.tif]

Panel on the right, top to bottom:

  SWEEP      join vessels that meet at a node joining only those two -- a real
             vessel that skeletonisation cut in half. Run this first: it changes
             which vessels count as junction-adjacent.

  PERICYTES  set the minimum vessel length that earns a pericyte, and read the
             resulting count, coverage and mean spacing before committing.

  INLETS     taken from the VTP's `boundary` array when present, otherwise
             marked here by selecting terminal vessels.

  BUILD      assigns every BC_type deterministically by walking out from the
             inlets, and fills the table below.

  TABLE      one row per element. Hover a vessel in the viewer and its row is
             selected; its neighbours colour ORANGE upstream, MAGENTA
             downstream, and the vessel itself WHITE. Click a row to find that
             vessel in the viewer.

Hover picking only works in 2D, so use Flatten while inspecting.
"""

import csv
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np


def vid_of(u, v, key):
    a, b = (u, v) if u <= v else (v, u)
    return f"{a}_{b}_{key}"


def load_network(path):
    import networkx as nx
    import pickle

    path = Path(path)
    boundary = {}
    if path.suffix == ".pkl":
        G = pickle.load(open(path, "rb"))
        poly = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            vox = np.asarray(d.get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                vox = np.array([G.nodes[u]["pos"], G.nodes[v]["pos"]], dtype=float)
            poly[(u, v, k)] = vox
        return G, poly, boundary

    import pyvista as pv
    mesh = pv.read(path)
    G = nx.MultiGraph()
    poly = {}
    lens = (np.asarray(mesh.cell_data["length"], dtype=float)
            if "length" in mesh.cell_data else None)
    bnd = ([str(x) for x in mesh.cell_data["boundary"]]
           if "boundary" in mesh.cell_data else None)

    raw = []
    for i, vid in enumerate(str(x) for x in mesh.cell_data["vessel_id"]):
        u_s, v_s, k_s = vid.rsplit("_", 2)
        u, v, k = int(u_s), int(v_s), int(k_s)
        pts = mesh.points[mesh.get_cell(i).point_ids][:, ::-1]   # xyz -> zyx
        raw.append((u, v, k, vid, pts, float(lens[i]) if lens is not None else float("nan")))
        if bnd is not None:
            boundary[vid] = bnd[i]

    # A VTP polyline carries no guarantee that it runs from u to v -- some are
    # stored the other way round, and taking pts[0] as u regardless puts node
    # positions badly wrong. Averaging cannot fix it, because the averages are
    # made from the same mistaken assignment. Instead, solve each node on its
    # own: the true position is the endpoint that every vessel meeting there
    # agrees on, so test both ends of one vessel and keep whichever the others
    # sit closest to.
    ends = {}                       # node -> list of (edge index, endA, endB)
    for i, (u, v, k, vid, pts, L) in enumerate(raw):
        ends.setdefault(u, []).append((i, pts[0], pts[-1]))
        ends.setdefault(v, []).append((i, pts[0], pts[-1]))

    pos = {}
    for n, items in ends.items():
        if len(items) < 2:
            continue                # a free end: settled below, once its partner is
        best, best_cost = None, None
        for cand in (items[0][1], items[0][2]):
            cost = 0.0
            for (_, a, b) in items[1:]:
                cost += min(np.linalg.norm(a - cand), np.linalg.norm(b - cand))
            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost
        pos[n] = np.asarray(best, dtype=float)

    # a vessel with one end pinned gives its free end away
    for i, (u, v, k, vid, pts, L) in enumerate(raw):
        for near, far in ((u, v), (v, u)):
            if near in pos and far not in pos:
                d0 = np.linalg.norm(pts[0] - pos[near])
                d1 = np.linalg.norm(pts[-1] - pos[near])
                pos[far] = np.asarray(pts[-1] if d0 <= d1 else pts[0], dtype=float)
    for i, (u, v, k, vid, pts, L) in enumerate(raw):
        pos.setdefault(u, np.asarray(pts[0], dtype=float))
        pos.setdefault(v, np.asarray(pts[-1], dtype=float))

    # now every vessel can be stored running from u to v
    for i, (u, v, k, vid, pts, L) in enumerate(raw):
        if (np.linalg.norm(pts[0] - pos[u]) + np.linalg.norm(pts[-1] - pos[v])
                > np.linalg.norm(pts[-1] - pos[u]) + np.linalg.norm(pts[0] - pos[v])):
            pts = pts[::-1]
        G.add_edge(u, v, key=k, voxels=pts.tolist(), length=L, source_ids=[vid])
        poly[(u, v, k)] = pts
    for n, p in pos.items():
        if n in G:
            G.nodes[n]["pos"] = p

    return G, poly, boundary


def remap_ids(G, wanted):
    """Follow a set of vessel_ids through merges.

    A merged vessel keeps the ids of its parts in `source_ids`, so a mark made
    before simplification still finds its vessel afterwards.
    """
    out = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        vid = vid_of(u, v, k)
        if vid in wanted:
            out.add(vid)
            continue
        if wanted & set(d.get("source_ids", [])):
            out.add(vid)
    return out


def sweep_degree2(G):
    """Join every pair of vessels meeting at a node that joins only those two.

    Each merged vessel keeps the ids of everything it was made from, so results
    computed against the merged network can still be painted onto the original
    geometry.
    """
    merged = 0
    changed = True
    while changed:
        changed = False
        for n in [x for x, d in G.degree() if d == 2]:
            es = list(G.edges(n, keys=True))
            if len(es) != 2:
                continue
            (a1, b1, k1), (a2, b2, k2) = es
            o1 = b1 if a1 == n else a1
            o2 = b2 if a2 == n else a2
            if o1 == o2:
                continue                      # a closed loop; leave it alone
            d1, d2 = G.edges[a1, b1, k1], G.edges[a2, b2, k2]
            pos = np.asarray(G.nodes[n]["pos"], dtype=float)

            def oriented(d, edge, end_at_n):
                vox = np.asarray(d.get("voxels", []), dtype=float)
                if vox.ndim != 2 or len(vox) < 2:
                    vox = np.array([G.nodes[edge[0]]["pos"],
                                    G.nodes[edge[1]]["pos"]], dtype=float)
                last_is_n = (np.linalg.norm(vox[-1] - pos)
                             <= np.linalg.norm(vox[0] - pos))
                if end_at_n != last_is_n:
                    vox = vox[::-1]
                return vox

            v1 = oriented(d1, (a1, b1), True)
            v2 = oriented(d2, (a2, b2), False)
            gap = float(np.linalg.norm(v1[-1] - v2[0]))
            comb = np.vstack([v1, v2]) if gap > 1e-6 else np.vstack([v1, v2[1:]])
            length = d1.get("length", 0.0) + d2.get("length", 0.0) + gap
            src = list(d1.get("source_ids", [])) + list(d2.get("source_ids", []))

            G.remove_edge(a1, b1, k1)
            G.remove_edge(a2, b2, k2)
            G.remove_node(n)
            key = (max(G[o1][o2].keys()) + 1) if G.has_edge(o1, o2) else 0
            G.add_edge(o1, o2, key=key, voxels=comb.tolist(),
                       length=length, source_ids=src)
            merged += 1
            changed = True
            break
    return merged


def split_one_node(G, n, spacer_length=5.0):
    """Split one over-connected node into a chain of Y junctions.

    The node keeps the two vessels pointing most nearly opposite each other --
    the route running through -- plus a short spacer. The rest move onto a new
    node at the far end of that spacer. Repeats on the new node until neither
    exceeds three vessels.
    """
    made = []
    target = n
    guard = 0
    while target in G and G.degree(target) > 3 and guard < 50:
        guard += 1
        pos = np.asarray(G.nodes[target]["pos"], dtype=float)

        legs = []
        for a, b, k in G.edges(target, keys=True):
            vox = np.asarray(G.edges[a, b, k].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                other = b if a == target else a
                vox = np.array([pos, G.nodes[other]["pos"]], dtype=float)
            near_start = np.linalg.norm(vox[0] - pos) <= np.linalg.norm(vox[-1] - pos)
            far = vox[min(3, len(vox) - 1)] if near_start else vox[max(-4, -len(vox))]
            dvec = far - pos
            nrm = np.linalg.norm(dvec)
            legs.append(((a, b, k), dvec / nrm if nrm else dvec))

        best, keep = 2.0, (legs[0][0], legs[1][0])
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                dot = float(np.dot(legs[i][1], legs[j][1]))
                if dot < best:
                    best, keep = dot, (legs[i][0], legs[j][0])
        moving = [e for e, _ in legs if e not in keep]

        dirs = [v for e, v in legs if e in moving]
        off = np.mean(dirs, axis=0) if dirs else np.array([0.0, 0.0, 1.0])
        nrm = np.linalg.norm(off)
        off = (off / nrm) if nrm else np.array([0.0, 0.0, 1.0])
        new_node = max(G.nodes()) + 1
        new_pos = pos + off * spacer_length
        G.add_node(new_node, pos=new_pos)

        for (a, b, k) in moving:
            d = dict(G.edges[a, b, k])
            other = b if a == target else a
            vox = np.asarray(d.get("voxels", []), dtype=float)
            if vox.ndim == 2 and len(vox) >= 2:
                if np.linalg.norm(vox[0] - pos) <= np.linalg.norm(vox[-1] - pos):
                    vox = np.vstack([new_pos, vox[1:]])
                else:
                    vox = np.vstack([vox[:-1], new_pos])
                d["voxels"] = vox.tolist()
            G.remove_edge(a, b, k)
            key = (max(G[other][new_node].keys()) + 1) if G.has_edge(other, new_node) else 0
            G.add_edge(other, new_node, key=key, **d)

        key = (max(G[target][new_node].keys()) + 1) if G.has_edge(target, new_node) else 0
        G.add_edge(target, new_node, key=key,
                   voxels=[pos.tolist(), new_pos.tolist()],
                   length=float(spacer_length), source_ids=[], spacer=True)
        made.append((target, new_node, len(moving)))
        target = new_node          # the surplus may still be over-connected
    return made


def split_high_degree(G, spacer_length=5.0):
    """Break nodes joining more than three vessels into a chain of Y junctions.

    A VV junction takes three vessels. A node joining four or more is split by
    moving the surplus onto a new node, joined back by a short spacer vessel:
    the node keeps two of its vessels plus the spacer, the new node takes the
    rest plus the spacer, and the process repeats until nothing exceeds three.

    Which two stay is chosen by direction -- the pair pointing most nearly
    opposite each other is treated as the vessel running through, so branches
    peel off it rather than the through-route being arbitrarily broken.
    """
    made = []
    guard = 0
    while True:
        guard += 1
        if guard > 500:
            break
        high = [n for n, d in G.degree() if d > 3]
        if not high:
            break
        made.extend(split_one_node(G, high[0], spacer_length))
        continue
        pos = np.asarray(G.nodes[high[0]]["pos"], dtype=float)

        # direction each vessel leaves this node
        legs = []
        for a, b, k in G.edges(n, keys=True):
            vox = np.asarray(G.edges[a, b, k].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                other = b if a == n else a
                vox = np.array([pos, G.nodes[other]["pos"]], dtype=float)
            near_start = np.linalg.norm(vox[0] - pos) <= np.linalg.norm(vox[-1] - pos)
            far = vox[min(3, len(vox) - 1)] if near_start else vox[max(-4, -len(vox))]
            d = far - pos
            nrm = np.linalg.norm(d)
            legs.append(((a, b, k), d / nrm if nrm else d))

        # the most opposed pair is the through-route
        best, keep = 2.0, (legs[0][0], legs[1][0])
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                dot = float(np.dot(legs[i][1], legs[j][1]))
                if dot < best:
                    best, keep = dot, (legs[i][0], legs[j][0])
        moving = [e for e, _ in legs if e not in keep]

        # the new node sits a spacer's length along the mean of the moved legs
        dirs = [v for e, v in legs if e in moving]
        off = np.mean(dirs, axis=0) if dirs else np.array([0.0, 0.0, 1.0])
        nrm = np.linalg.norm(off)
        off = (off / nrm) if nrm else np.array([0.0, 0.0, 1.0])
        new_node = max(G.nodes()) + 1
        new_pos = pos + off * spacer_length
        G.add_node(new_node, pos=new_pos)

        for (a, b, k) in moving:
            d = dict(G.edges[a, b, k])
            other = b if a == n else a
            vox = np.asarray(d.get("voxels", []), dtype=float)
            if vox.ndim == 2 and len(vox) >= 2:
                if np.linalg.norm(vox[0] - pos) <= np.linalg.norm(vox[-1] - pos):
                    vox = np.vstack([new_pos, vox[1:]])
                else:
                    vox = np.vstack([vox[:-1], new_pos])
                d["voxels"] = vox.tolist()
            G.remove_edge(a, b, k)
            key = (max(G[other][new_node].keys()) + 1) if G.has_edge(other, new_node) else 0
            G.add_edge(other, new_node, key=key, **d)

        key = (max(G[n][new_node].keys()) + 1) if G.has_edge(n, new_node) else 0
        G.add_edge(n, new_node, key=key,
                   voxels=[pos.tolist(), new_pos.tolist()],
                   length=float(spacer_length),
                   source_ids=[], spacer=True)
        made.append((n, new_node, len(moving)))
    return made


def edge_length(G, u, v, k):
    """Stored length, or measured off the polyline when it is missing."""
    d = G.edges[u, v, k]
    L = d.get("length", None)
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = float("nan")
    if not np.isfinite(L) or L <= 0:
        vox = np.asarray(d.get("voxels", []), dtype=float)
        if vox.ndim == 2 and len(vox) >= 2:
            L = float(np.linalg.norm(np.diff(vox, axis=0), axis=1).sum())
        else:
            L = 0.0
        d["length"] = L
    return L


def pericyte_plan(G, min_length, peri_length, exclude=(), force_on=(), force_off=()):
    """Which vessels earn a pericyte, and what that implies for density.

    `exclude` are vessels that never get one -- the inlets, which are boundary
    conditions rather than capillary. `force_on` and `force_off` are the manual
    overrides made in the viewer, which win over the length rule.
    """
    eligible, skipped, total = [], [], 0.0
    exclude, force_on, force_off = set(exclude), set(force_on), set(force_off)
    for u, v, k, d in G.edges(keys=True, data=True):
        vid = vid_of(u, v, k)
        L = edge_length(G, u, v, k)
        total += L
        if vid in force_off or (vid in exclude and vid not in force_on):
            skipped.append((vid, L))
            continue
        if vid in force_on:
            eligible.append((vid, L))
            continue
        (eligible if L >= min_length else skipped).append((vid, L))
    n = len(eligible)
    spacing = (total / n) if n else float("inf")
    shortest_after = min((L - peri_length for _, L in eligible), default=float("nan"))
    return {"eligible": eligible, "skipped": skipped, "n": n,
            "total_length": total, "spacing": spacing,
            "shortest_after": shortest_after,
            "coverage": (100.0 * n / max(n + len(skipped), 1))}


def build_array(G, inlet_vids, plan, peri_length):
    """Assign BC types deterministically by walking outward from the inlets.

    A VV junction offers four ports: H1 and H2 take inputs, H3 and H4 outputs.
    An element records the port it meets -- bare `H3` when it touches one
    junction, `H4L_H1R` when it spans two. Along a chain leaving a junction the
    first element is `pp` and the rest are `vp`, which keeps the alternation
    rule (adjacent ports must differ) satisfied everywhere.
    """
    peri_set = {vid for vid, _ in plan["eligible"]}
    junctions = {n for n, d in G.degree() if d >= 3}
    jname = {n: f"VV_junc{i+1}" for i, n in enumerate(sorted(junctions))}
    ports = {n: {"in": [], "out": []} for n in junctions}

    # orient every vessel by walking out from the inlets
    direction = {}
    seeds = []
    for vid in inlet_vids:
        u, v, k = (int(x) for x in vid.rsplit("_", 2))
        if not G.has_edge(u, v, k):
            continue
        start = u if G.degree(u) == 1 else v
        direction[(min(u, v), max(u, v), k)] = (start, v if start == u else u)
        seeds.append(v if start == u else u)

    # record the order vessels are reached, so names rise with depth
    order = []
    generation = {}
    parent = {}
    for vid in inlet_vids:
        u, v, k = (int(x) for x in vid.rsplit("_", 2))
        e = (min(u, v), max(u, v), k)
        if e in direction:
            order.append(e)
            generation[e] = 0
            parent[e] = ""

    seen = set(seeds)
    q = deque([(n, 0) for n in seeds])
    while q:
        node, depth = q.popleft()
        incoming = [e for e in order
                    if direction.get(e, (None, None))[1] == node]
        for a, b, k in G.edges(node, keys=True):
            e = (min(a, b), max(a, b), k)
            if e in direction:
                continue
            other = b if a == node else a
            direction[e] = (node, other)
            order.append(e)
            generation[e] = depth + 1
            parent[e] = incoming[0] if incoming else None
            if other not in seen:
                seen.add(other)
                q.append((other, depth + 1))

    unreached = [e for e in G.edges(keys=True)
                 if (min(e[0], e[1]), max(e[0], e[1]), e[2]) not in direction]
    if not direction:
        raise ValueError(
            "none of the inlet ids match a vessel in the current network. "
            "They were probably invalidated by Sweep or Split; mark inlets "
            "again after simplifying.")

    # allocate junction ports in traversal order
    for e, (frm, to) in direction.items():
        if frm in junctions:
            ports[frm]["out"].append(e)
        if to in junctions:
            ports[to]["in"].append(e)

    port_of = {}
    for n, pr in ports.items():
        for i, e in enumerate(pr["in"][:2]):
            port_of[(e, "R")] = f"H{i+1}"
        for i, e in enumerate(pr["out"][:2]):
            port_of[(e, "L")] = f"H{i+3}"

    over = [n for n in junctions if G.degree(n) > 3]

    rows = []
    name_of = {}

    # name in the order the network was walked: the inlet first, then numbers
    # rising with distance from it
    inlet_edges = []
    for vid in inlet_vids:
        u, v, k = (int(x) for x in vid.rsplit("_", 2))
        inlet_edges.append((min(u, v), max(u, v), k))

    n_vessel = 0
    for e in order:
        if e in inlet_edges:
            name_of[e] = ("inlet" if len(inlet_edges) == 1
                          else f"inlet{inlet_edges.index(e) + 1}")
        else:
            n_vessel += 1
            name_of[e] = f"V{n_vessel:02d}"
    for e in direction:                      # anything the walk missed
        if e not in name_of:
            n_vessel += 1
            name_of[e] = f"V{n_vessel:02d}"

    def elem_name(e, is_peri):
        return ("P" + name_of[e]) if is_peri else name_of[e]

    for e in (order + [x for x in direction if x not in order]):
        frm, to = direction[e]
        vid = vid_of(e[0], e[1], e[2])
        gen = generation.get(e, "")
        par = name_of.get(parent.get(e)) if parent.get(e) else ""
        d = G.edges[e[0], e[1], e[2]]
        L = edge_length(G, e[0], e[1], e[2])
        has_peri = vid in peri_set

        Lport = port_of.get((e, "L"))
        Rport = port_of.get((e, "R"))
        upstream = jname[frm] if frm in junctions else ""
        downstream = jname[to] if to in junctions else ""

        def suffix(first_in_chain):
            head = "pp" if first_in_chain else "vp"
            if Lport and Rport and not has_peri:
                return f"{head}_{Lport}L_{Rport}R"
            if first_in_chain and Lport:
                return f"{head}_{Lport}"
            if (not first_in_chain) and Rport:
                return f"{head}_{Rport}"
            if first_in_chain and Rport and not Lport:
                return f"{head}_{Rport}"
            return head

        if has_peri:
            pname = elem_name(e, True)
            vname = name_of[e]
            rows.append({"name": pname, "BC_type": suffix(True),
                         "vessel_type": "capillary_H_D_pericyte_BS_RO",
                         "inp_vessels": upstream, "out_vessels": vname,
                         "length": round(peri_length, 2),
                         "generation": gen, "parent": par,
                         "graph_vessel_id": vid,
                         "source_ids": " ".join(d.get("source_ids", [vid]))})
            rows.append({"name": vname, "BC_type": suffix(False),
                         "vessel_type": "capillary_H_D",
                         "inp_vessels": pname, "out_vessels": downstream,
                         "length": round(L - peri_length, 2),
                         "generation": gen, "parent": par,
                         "graph_vessel_id": vid,
                         "source_ids": " ".join(d.get("source_ids", [vid]))})
        else:
            rows.append({"name": name_of[e], "BC_type": suffix(True),
                         "vessel_type": "capillary_H_D",
                         "inp_vessels": upstream, "out_vessels": downstream,
                         "length": round(L, 2),
                         "generation": gen, "parent": par,
                         "graph_vessel_id": vid,
                         "source_ids": " ".join(d.get("source_ids", [vid]))})

    # junction rows
    for n in sorted(junctions):
        ins, outs = [], []
        for e in ports[n]["in"]:
            ins.append(name_of[e])
        for e in ports[n]["out"]:
            vid = vid_of(e[0], e[1], e[2])
            outs.append(elem_name(e, vid in peri_set))
        rows.append({"name": jname[n], "BC_type": "vv_junction",
                     "vessel_type": "capillary_H_D",
                     "inp_vessels": " ".join(ins), "out_vessels": " ".join(outs),
                     "length": "", "generation": "", "parent": "",
                     "graph_vessel_id": "", "source_ids": ""})

    return rows, name_of, direction, unreached, over


def validate(rows):
    """Check the alternation rule and that every reference resolves."""
    by = {r["name"]: r for r in rows}
    problems = []
    for r in rows:
        for nxt in str(r["out_vessels"]).split():
            if nxt not in by:
                problems.append(f"{r['name']} -> unknown '{nxt}'")
                continue
            a, b = r["BC_type"], by[nxt]["BC_type"]
            if a == "vv_junction" or b == "vv_junction":
                continue
            if a.split("_")[0][-1] == b.split("_")[0][0]:
                problems.append(f"{r['name']}({a}) -> {nxt}({b}): both "
                                f"'{a.split('_')[0][-1]}' at the joint")
    seen = {}
    for r in rows:
        if r["BC_type"] == "vv_junction":
            continue
        for tag in ("L", "R"):
            pass
    return problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    image_path = None
    if "--image" in sys.argv:
        image_path = Path(sys.argv[sys.argv.index("--image") + 1])

    import napari
    import networkx as nx
    import pandas as pd
    from magicgui.widgets import (Container, TextEdit, PushButton, FloatSpinBox,
                                  LineEdit, Label, ComboBox)
    from qtpy.QtWidgets import (QTableWidget, QTableWidgetItem, QScrollArea,
                                QWidget, QVBoxLayout, QAbstractItemView,
                                QSplitter, QLabel)
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor

    G, poly, boundary = load_network(src)
    print(f"loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} vessels")
    inlet_ids = {v for v, b in boundary.items() if b == "inlet"}
    if inlet_ids:
        print(f"read {len(inlet_ids)} inlet(s) from the file")

    viewer = napari.Viewer(title=f"vessel array: {src.name}")
    if image_path and image_path.exists():
        try:
            import tifffile
            viewer.add_image(tifffile.imread(image_path), name="image",
                             colormap="gray", opacity=0.35, blending="additive")
        except Exception as exc:
            print(f"image not loaded ({exc})")

    state = {"rows": [], "name_of": {}, "direction": {}, "hover": None,
             "flat": False, "edges": [], "plan": None, "over_nodes": [],
             "highlight": set(), "issue_rows": [], "issue_nodes": {},
             "peri_on": set(), "peri_off": set(), "tree": None, "tree_ids": [],
             # `selected` persists until you pick something else. `hover` is
             # only what the cursor is over right now, and must never disturb
             # the selection -- otherwise rotating the view wipes it.
             "selected": None}

    import copy
    undo_stack = []
    log = []          # every action taken, in order, with what it was given

    def note(action, **detail):
        """Record a step so the whole session can be reproduced or reported."""
        entry = {"step": len(log) + 1, "action": action,
                 "at": datetime.now().strftime("%H:%M:%S")}
        entry.update(detail)
        entry["vessels_after"] = G.number_of_edges()
        entry["nodes_after"] = G.number_of_nodes()
        entry["problem_nodes_after"] = sum(1 for _, d in G.degree() if d > 3)
        log.append(entry)
        return entry

    def snapshot(label):
        """Remember the whole graph plus the inlet marks before a change."""
        undo_stack.append((label, copy.deepcopy(G), set(inlet_ids)))
        if len(undo_stack) > 15:
            undo_stack.pop(0)

    def restore():
        if not undo_stack:
            say("nothing to undo")
            return
        label, snap, marks = undo_stack.pop()
        # G is captured by every closure, so refill it rather than rebind it
        G.clear()
        G.add_nodes_from(snap.nodes(data=True))
        G.add_edges_from((u, v, k, d) for u, v, k, d in snap.edges(keys=True, data=True))
        inlet_ids.clear(); inlet_ids.update(marks)
        note("undo", undid=label)
        state["highlight"] = set()
        # anything computed from the old graph is now wrong
        state["plan"] = None
        state["rows"] = []
        state["direction"] = {}
        table.setRowCount(0)
        rebuild_layer()
        do_flag_high()
        recolour()
        n_bad = sum(1 for _, d in G.degree() if d > 3)
        say(f"undid: {label}\n"
            f"vessels: {G.number_of_edges()}   nodes: {G.number_of_nodes()}\n"
            f"problem nodes: {n_bad}\n"
            f"{len(undo_stack)} undo step(s) left\n\n"
            f"pericyte plan and array cleared\n(recompute before building)")

    def build_pick_tree():
        """Sample points along every vessel so the cursor can find one.

        Called from rebuild_layer, after state["edges"] has been refreshed.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        pts, ids = [], []
        for e in state["edges"]:
            vox = np.asarray(G.edges[e].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                vox = np.array([G.nodes[e[0]]["pos"], G.nodes[e[1]]["pos"]],
                               dtype=float)
            step = max(1, len(vox) // 20)
            sample = vox[::step]
            for p in sample:
                # display coordinates: flattening moves everything onto z = 0
                pts.append([0.0, p[1], p[2]] if state["flat"] else list(p))
                ids.append(vid_of(*e))
        if pts:
            arr = np.array(pts, dtype=float)
            state["tree"] = cKDTree(arr) if cKDTree else None
            state["tree_pts"] = arr
            state["tree_ids"] = ids
        else:
            state["tree"] = None
            state["tree_pts"] = None
            state["tree_ids"] = []

    def rebuild_layer():
        edges = sorted(G.edges(keys=True), key=lambda e: (e[0], e[1], e[2]))
        state["edges"] = edges
        paths, ids = [], []
        for e in edges:
            vox = np.asarray(G.edges[e].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                vox = np.array([G.nodes[e[0]]["pos"], G.nodes[e[1]]["pos"]],
                               dtype=float)
            if len(vox) > 12:
                vox = vox[np.linspace(0, len(vox) - 1, 12).astype(int)]
            if state["flat"]:
                vox = np.column_stack([np.zeros(len(vox)), vox[:, 1], vox[:, 2]])
            paths.append(vox)
            ids.append(vid_of(*e))
        feats = pd.DataFrame({"vessel_id": ids})
        if "vessels" in viewer.layers:
            viewer.layers["vessels"].data = paths
            viewer.layers["vessels"].features = feats
        else:
            viewer.add_shapes(paths, shape_type="path", features=feats,
                              edge_color="cyan", edge_width=1.5, name="vessels")

        # The hover tree samples the geometry, so it is only valid for the
        # geometry just drawn. Rebuilding it here means no caller can change
        # the graph and leave picking pointed at vessels that no longer exist.
        build_pick_tree()

        # a stale hover would name a deleted vessel
        live = set(ids)
        if state.get("hover") not in live:
            state["hover"] = None
        if state.get("selected") not in live:
            state["selected"] = None
        return viewer.layers["vessels"]

    vessels = rebuild_layer()
    build_pick_tree()

    # ---------------- panel ----------------

    status = TextEdit(value="", label="")
    try:
        status.native.setReadOnly(True)
        status.native.setMinimumHeight(150)
        status.native.setMaximumHeight(180)
    except Exception:
        pass

    info = TextEdit(value="hover a vessel, or click a table row", label="")
    try:
        info.native.setReadOnly(True)
        info.native.setMinimumHeight(120)
        info.native.setMaximumHeight(150)
    except Exception:
        pass

    min_len = FloatSpinBox(value=30.0, min=0.0, max=500.0, step=1.0,
                           label="pericyte if length >=")
    peri_len = FloatSpinBox(value=15.0, min=0.0, max=100.0, step=0.5,
                            label="pericyte length")
    node_pick = ComboBox(choices=[], label="problem node")
    spacer_len = FloatSpinBox(value=5.0, min=0.5, max=50.0, step=0.5,
                              label="spacer length")
    export_tag = LineEdit(value="", label="export tag")

    issues = QTableWidget()
    issues.setColumnCount(4)
    issues.setHorizontalHeaderLabels(["node", "deg", "vessel_id", "length"])
    issues.setSelectionBehavior(QAbstractItemView.SelectRows)
    issues.setEditTriggers(QAbstractItemView.NoEditTriggers)
    issues.setMinimumHeight(120)

    table = QTableWidget()
    table.setColumnCount(9)
    table.setHorizontalHeaderLabels(["name", "BC_type", "vessel_type",
                                     "inp_vessels", "out_vessels", "length",
                                     "generation", "parent", "source_ids"])
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setMinimumHeight(260)

    def fill_issues(bad):
        """Load the node chooser; the table then shows one node at a time."""
        labels = [f"node {n}  ({d} vessels)" for n, d in bad]
        state["issue_nodes"] = {lab: n for lab, (n, d) in zip(labels, bad)}
        prev = node_pick.value if list(node_pick.choices) else None

        # swap the list without the combo firing on a value that is about to
        # disappear -- doing it naively raises, and the refresh then aborts
        try:
            with node_pick.changed.blocked():
                node_pick.choices = []
                node_pick.choices = labels
        except Exception:
            node_pick.choices = []
            node_pick.choices = labels

        if labels:
            node_pick.value = prev if prev in labels else labels[0]
            show_node_vessels()
        else:
            issues.setRowCount(0)
            state["issue_rows"] = []
            state["highlight"] = set()
            recolour()

    def show_node_vessels(*_):
        """Fill the table with just the chosen node's vessels."""
        lab = node_pick.value
        n = state.get("issue_nodes", {}).get(lab)
        if n is None or n not in G:
            issues.setRowCount(0)
            state["issue_rows"] = []
            return
        rows = [(n, G.degree(n), vid_of(a, b, k), edge_length(G, a, b, k))
                for a, b, k in G.edges(n, keys=True)]
        rows.sort(key=lambda r: r[3])          # shortest first: usual candidate
        state["issue_rows"] = rows
        issues.setRowCount(len(rows))
        for i, (nn, d, vid, L) in enumerate(rows):
            for j, val in enumerate((str(nn), str(d), vid, f"{L:.1f}")):
                issues.setItem(i, j, QTableWidgetItem(val))
        issues.resizeColumnsToContents()

        # highlight the whole junction and look at it
        state["highlight"] = {r[2] for r in rows}
        recolour()
        pos_ = np.asarray(G.nodes[n]["pos"], dtype=float)
        if state["flat"]:
            pos_ = np.array([0.0, pos_[1], pos_[2]])
        try:
            viewer.camera.center = tuple(float(x) for x in pos_)
            viewer.camera.zoom = max(viewer.camera.zoom, 4.0)
        except Exception:
            pass
        issues.selectRow(0)
        say(f"node {n}: {len(rows)} vessels, shortest first\n\n"
            + "\n".join(f"  {r[2]}   {r[3]:.1f}" for r in rows)
            + "\n\nWHITE = the selected vessel"
            + "\nPURPLE = the others at this node"
            + "\n\npick a row, then Delete this vessel")

    def issue_row():
        r = issues.currentRow()
        rows = state.get("issue_rows", [])
        return rows[r] if 0 <= r < len(rows) else None

    def issue_node():
        row = issue_row()
        return row[0] if row else None

    def on_issue_selected():
        row = issue_row()
        if row is None:
            return
        n, d, vid, L = row
        if n not in G:
            return
        siblings = {vid_of(a, b, k) for a, b, k in G.edges(n, keys=True)}
        state["highlight"] = siblings - {vid}
        state["selected"] = vid
        vessels.selected_data = set()
        recolour(vid)          # the chosen one draws white, its siblings purple
        pos_ = np.asarray(G.nodes[n]["pos"], dtype=float)
        if state["flat"]:
            pos_ = np.array([0.0, pos_[1], pos_[2]])
        try:
            viewer.camera.center = tuple(float(x) for x in pos_)
            viewer.camera.zoom = max(viewer.camera.zoom, 3.0)
        except Exception:
            pass
        say(f"node {n}, {d} vessels\n\n"
            f"WHITE: {vid}\n  length {L:.1f}\n\n"
            f"PURPLE: the other {d - 1}\n\n"
            f"Delete this vessel, or\nSplit the selected node")
        return
    def do_split_one():
        n = issue_node()
        if n is None or n not in G:
            say("select a row in the issues table")
            return
        snapshot(f"split node {n}")
        state["plan"] = None; state["rows"] = []; table.setRowCount(0)
        made = split_one_node(G, n, spacer_len.value)
        kept = remap_ids(G, inlet_ids)
        inlet_ids.clear(); inlet_ids.update(kept)
        rebuild_layer()
        do_flag_high()
        still = [x for x, dd in G.degree() if dd > 3]
        say(f"split node {n}: {len(made)} spacer(s) of {spacer_len.value} um\n"
            f"node {n} now joins {G.degree(n) if n in G else 0} vessels\n"
            f"problem nodes left: {len(still)}")

    def do_delete_selected():
        """Delete the one vessel named by the selected issues row."""
        row = issue_row()
        if row is None:
            # fall back to a viewer selection if there is one
            sel = list(vessels.selected_data)
            ids = list(vessels.features["vessel_id"])
            chosen = {ids[i] for i in sel if i < len(ids)}
            if not chosen:
                say("pick a row in the issues table\n"
                    "(each row is one vessel)")
                return
        else:
            chosen = {row[2]}
        snapshot(f"delete {len(chosen)} vessel(s)")
        state["plan"] = None; state["rows"] = []; table.setRowCount(0)
        for vid in chosen:
            for a, b, k in list(G.edges(keys=True)):
                if vid_of(a, b, k) == vid:
                    G.remove_edge(a, b, k)
        G.remove_nodes_from([x for x in list(G.nodes()) if G.degree(x) == 0])
        kept = remap_ids(G, inlet_ids)
        inlet_ids.clear(); inlet_ids.update(kept)
        rebuild_layer()
        do_flag_high()
        state["highlight"] = set()
        note("delete_vessel", vessel_ids=sorted(chosen),
             at_node=(row[0] if row else None))
        n_bad = sum(1 for _, d in G.degree() if d > 3)
        say(f"deleted {len(chosen)}:\n  " + "\n  ".join(sorted(chosen)[:4])
            + f"\nvessels now: {G.number_of_edges()}"
            + f"\nproblem nodes: {n_bad}")

    def say(text):
        status.value = text
        viewer.status = text.replace("\n", "   ")
        print(text.replace("\n", " | "))

    def popup(vid, row, length=None):
        """A tooltip beside the cursor, so the reading is where the eye is."""
        from qtpy.QtWidgets import QToolTip
        from qtpy.QtGui import QCursor
        has_peri = (state["plan"] is not None
                    and vid in {v for v, _ in state["plan"]["eligible"]})
        if row:
            text = (f"<b>{row.get('name','?')}</b>&nbsp;&nbsp;"
                    f"<i>{row.get('BC_type','')}</i><br>"
                    f"length&nbsp;&nbsp;{row.get('length','')}<br>"
                    f"gen&nbsp;&nbsp;{row.get('generation','')}"
                    f"&nbsp;&nbsp;parent&nbsp;{row.get('parent','') or '-'}<br>"
                    f"in&nbsp;&nbsp;{row.get('inp_vessels','') or '-'}<br>"
                    f"out&nbsp;&nbsp;{row.get('out_vessels','') or '-'}<br>"
                    f"pericyte&nbsp;&nbsp;{'yes' if has_peri else 'no'}<br>"
                    f"<small>{vid}</small>")
        else:
            text = (f"<b>{vid}</b><br>"
                    + (f"length&nbsp;&nbsp;{length:.1f}<br>" if length else "")
                    + f"pericyte&nbsp;&nbsp;{'yes' if has_peri else 'no'}<br>"
                    f"<small>build the array for names</small>")
        try:
            QToolTip.showText(QCursor.pos(), text)
        except Exception:
            pass

    def show_info(row):
        """Everything known about one element, in the info box."""
        info.value = (
            f"{row.get('name', '?')}   [{row.get('BC_type', '')}]\n"
            f"length      {row.get('length', '')}\n"
            f"generation  {row.get('generation', '')}\n"
            f"parent      {row.get('parent', '') or '-'}\n"
            f"in          {row.get('inp_vessels', '') or '-'}\n"
            f"out         {row.get('out_vessels', '') or '-'}\n"
            f"type        {row.get('vessel_type', '')}\n"
            f"geometry    {row.get('graph_vessel_id', '')}"
        )

    def row_for_vessel(vid):
        for r in state["rows"]:
            if r.get("graph_vessel_id") == vid and r.get("BC_type") != "vv_junction":
                return r
        return None

    def do_sweep():
        snapshot("sweep degree-2")
        state["plan"] = None; state["rows"] = []; table.setRowCount(0)
        n = sweep_degree2(G)
        before = len(inlet_ids)
        kept = remap_ids(G, inlet_ids)
        inlet_ids.clear(); inlet_ids.update(kept)
        rebuild_layer()
        note("sweep_degree2", joins_merged=n)
        recolour()
        say(f"swept {n} degree-2 joints\n"
            f"problem nodes: {sum(1 for _, d in G.degree() if d > 3)}\n"
            f"vessels now: {G.number_of_edges()}\n"
            f"nodes now:   {G.number_of_nodes()}\n"
            f"inlets carried over: {len(inlet_ids)} of {before}")

    def do_flag_high():
        """Mark every node joining more than three vessels."""
        bad = [(n, d) for n, d in G.degree() if d > 3]
        if not bad:
            if "over-connected" in viewer.layers:
                viewer.layers.remove("over-connected")
            state["over_nodes"] = []
            fill_issues([])          # clear the chooser too, or it goes stale
            say("no node joins more than 3 vessels\nready to build")
            return
        pos = np.array([G.nodes[n]["pos"] for n, _ in bad], dtype=float)
        if state["flat"]:
            pos = np.column_stack([np.zeros(len(pos)), pos[:, 1], pos[:, 2]])
        sizes = np.array([8 + 3 * (d - 3) for _, d in bad], dtype=float)
        if "over-connected" in viewer.layers:
            lay = viewer.layers["over-connected"]
            lay.data = pos
            lay.size = sizes
            lay.face_color = "red"
        else:
            keep = viewer.layers.selection.active
            lay = viewer.add_points(pos, size=sizes, face_color="red",
                                    border_color="white", name="over-connected",
                                    opacity=0.9)
            if keep is not None:
                viewer.layers.selection.active = keep
        # markers sit at single z values, so without this they vanish on every
        # slice but their own -- which is why they looked misaligned
        try:
            lay.out_of_slice_display = True
        except Exception:
            pass
        state["over_nodes"] = bad
        fill_issues(bad)
        counts = {}
        for _, d in bad:
            counts[d] = counts.get(d, 0) + 1
        detail = "  ".join(f"deg{k}:{v}" for k, v in sorted(counts.items()))
        say(f"{len(bad)} node(s) join >3 vessels\n{detail}\n\n"
            f"shown in red. a VV junction takes 3,\nso split them before building")

    def do_split_high():
        snapshot("split all over-connected")
        state["plan"] = None; state["rows"] = []; table.setRowCount(0)
        made = split_high_degree(G, spacer_len.value)
        kept = remap_ids(G, inlet_ids)
        inlet_ids.clear(); inlet_ids.update(kept)
        rebuild_layer()
        recolour()
        if not made:
            say("nothing to split")
            return
        do_flag_high()
        say(f"split {len(made)} node(s)\n"
            f"added {len(made)} spacer vessel(s) of {spacer_len.value} um\n"
            f"vessels now: {G.number_of_edges()}\n"
            f"max degree now: {max(d for _, d in G.degree())}")

    def draw_pericytes(plan, highlight=None):
        """Mark where each pericyte sits: just inside the vessel's upstream end.

        The one on the currently selected vessel draws red, so it is obvious
        which pericyte a table row refers to.
        """
        pts, cols = [], []
        want = {vid for vid, _ in plan["eligible"]}
        for a, b, k in G.edges(keys=True):
            vid = vid_of(a, b, k)
            if vid not in want:
                continue
            vox = np.asarray(G.edges[a, b, k].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                continue
            # a little way in from the start, where the module would sit
            idx = min(len(vox) - 1, max(1, len(vox) // 8))
            p = vox[idx]
            if state["flat"]:
                p = np.array([0.0, p[1], p[2]])
            pts.append(p)
            cols.append("red" if vid == highlight else "gold")
        pts = np.array(pts) if pts else np.empty((0, 3))
        sizes = np.array([11 if c == "red" else 7 for c in cols], dtype=float)
        if "pericytes" in viewer.layers:
            lay = viewer.layers["pericytes"]
            lay.data = pts
            if len(pts):
                lay.face_color = cols
                lay.size = sizes
        else:
            keep = viewer.layers.selection.active
            lay = viewer.add_points(pts, size=sizes if len(pts) else 7,
                                    face_color=cols if len(pts) else "gold",
                                    border_color="black", name="pericytes",
                                    opacity=0.95)
            if keep is not None:
                viewer.layers.selection.active = keep
            try:
                lay.out_of_slice_display = True
            except Exception:
                pass

    def do_density():
        plan = pericyte_plan(G, min_len.value, peri_len.value,
                             exclude=inlet_ids,
                             force_on=state["peri_on"], force_off=state["peri_off"])
        state["plan"] = plan
        draw_pericytes(plan)
        note("pericyte_plan", min_length=min_len.value, pericyte_length=peri_len.value,
             pericytes=plan["n"], coverage_percent=round(plan["coverage"], 1),
             mean_spacing=round(plan["spacing"], 1))
        bad = sum(1 for _, L in plan["eligible"] if L - peri_len.value <= 0)
        say(f"pericytes:  {plan['n']} of {G.number_of_edges()} vessels\n"
            f"coverage:   {plan['coverage']:.0f}%\n"
            f"mean spacing: {plan['spacing']:.0f} um\n"
            f"shortest vessel after: {plan['shortest_after']:.1f} um\n"
            f"non-positive: {bad}"
            + ("\n\nin vivo spacing is 30-100 um" if plan["spacing"] > 100 else ""))

    def do_mark_inlets():
        sel = list(vessels.selected_data)
        feats = vessels.features
        chosen = {str(feats.iloc[i]["vessel_id"]) for i in sel if i < len(feats)}
        terms = {vid_of(*e) for e in state["edges"]
                 if G.degree(e[0]) == 1 or G.degree(e[1]) == 1}
        good = chosen & terms
        inlet_ids.update(good)
        note("mark_inlets", vessel_ids=sorted(good))
        recolour()
        say(f"marked {len(good)} inlet(s)\ntotal inlets: {len(inlet_ids)}")

    def do_build():
        if not inlet_ids:
            say("no inlets: select terminal vessels\nand mark them first")
            return
        if state["plan"] is None:
            do_density()
        try:
            rows, name_of, direction, unreached, over = build_array(
                G, inlet_ids, state["plan"], peri_len.value)
        except ValueError as exc:
            say("cannot build:\n\n" + str(exc))
            return
        if over:
            do_flag_high()
            say(f"cannot build: {len(over)} node(s) join >3 vessels\n"
                f"(shown in red)\n\nrun Split over-connected nodes first")
            return
        state.update(rows=rows, name_of=name_of, direction=direction)
        fill_table(rows)
        problems = validate(rows)
        msg = (f"built {len(rows)} elements\n"
               f"{sum(1 for r in rows if r['BC_type'] == 'vv_junction')} junctions\n"
               f"{sum(1 for r in rows if 'pericyte' in r['vessel_type'])} pericytes")
        if unreached:
            msg += f"\n{len(unreached)} vessels unreachable from inlets"
        msg += ("\nalternation: OK" if not problems
                else f"\n{len(problems)} RULE VIOLATIONS (see terminal)")
        for p in problems:
            print("  violation:", p)
        say(msg)

    def fill_table(rows):
        table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, key in enumerate(["name", "BC_type", "vessel_type",
                                     "inp_vessels", "out_vessels", "length",
                                     "generation", "parent", "source_ids"]):
                table.setItem(i, j, QTableWidgetItem(str(r.get(key, ""))))
        table.resizeColumnsToContents()

    def _save_log():
        import json as _json
        out = src.with_name(src.stem
                            + f"_steps_{datetime.now().strftime('%H%M%S')}.json")
        with open(out, "w") as fh:
            _json.dump({"source": src.name,
                        "settings": {"pericyte_min_length": min_len.value,
                                     "pericyte_length": peri_len.value,
                                     "spacer_length": spacer_len.value},
                        "inlets": sorted(inlet_ids),
                        "steps": log}, fh, indent=2)
        say(f"wrote {out.name}\n{len(log)} step(s) recorded")

    def current_vessel():
        """The sticky selection, else whatever is under the cursor, else the row."""
        if state.get("selected"):
            return state["selected"]
        if state.get("hover"):
            return state["hover"]
        r = table.currentRow()
        if 0 <= r < len(state["rows"]):
            vid = state["rows"][r].get("graph_vessel_id")
            if vid:
                return vid
        return None

    def _replan(msg):
        plan = pericyte_plan(G, min_len.value, peri_len.value,
                             exclude=inlet_ids,
                             force_on=state["peri_on"], force_off=state["peri_off"])
        state["plan"] = plan
        draw_pericytes(plan, highlight=state.get("selected"))

        # if an array already exists, rebuild it so the table and the CSV match
        # the pericytes now on screen, rather than going quietly stale
        rebuilt = ""
        if state["rows"] and inlet_ids:
            try:
                rows, name_of, direction, unreached, over = build_array(
                    G, inlet_ids, plan, peri_len.value)
                if not over:
                    state.update(rows=rows, name_of=name_of, direction=direction)
                    fill_table(rows)
                    rebuilt = "\narray rebuilt to match"
                else:
                    state["rows"] = []
                    table.setRowCount(0)
            except Exception:
                state["rows"] = []
                table.setRowCount(0)
        say(f"{msg}\n\npericytes: {plan['n']}   "
            f"spacing {plan['spacing']:.0f} um\n"
            f"manual: +{len(state['peri_on'])}  -{len(state['peri_off'])}"
            + (rebuilt if rebuilt else "\n\nrebuild the array to use this"))

    def do_peri_add():
        vid = state.get("hover")
        if not vid:
            say("hover a vessel first\n(Flatten, then move the cursor)")
            return
        if vid in inlet_ids:
            say("inlets do not take a pericyte")
            return
        state["peri_off"].discard(vid)
        state["peri_on"].add(vid)
        note("pericyte_add", vessel_id=vid)
        _replan(f"pericyte added to\n{vid}")

    def do_peri_remove():
        vid = current_vessel()
        if not vid:
            say("point at a vessel first:\\nhover it, or click its table row")
            return
        state["peri_on"].discard(vid)
        state["peri_off"].add(vid)
        note("pericyte_remove", vessel_id=vid)
        _replan(f"pericyte removed from\n{vid}\n(gold dot should be gone)")

    def do_peri_clear_manual():
        n = len(state["peri_on"]) + len(state["peri_off"])
        state["peri_on"].clear(); state["peri_off"].clear()
        note("pericyte_clear_manual", cleared=n)
        _replan(f"cleared {n} manual override(s)")

    def do_export():
        if not state["rows"]:
            say("build the array first")
            return
        stamp = datetime.now().strftime("%H%M%S")
        tag = export_tag.value.strip()
        out = src.with_name(src.stem + f"_vessel_array_{stamp}"
                            + (f"_{tag}" if tag else "") + ".csv")
        cols = ["name", "BC_type", "vessel_type", "inp_vessels", "out_vessels",
                "length", "generation", "parent", "graph_vessel_id", "source_ids"]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in state["rows"]:
                w.writerow({c: r.get(c, "") for c in cols})
        note("export", file=out.name, rows=len(state["rows"]))
        recipe = {
            "source": src.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": {
                "pericyte_min_length": min_len.value,
                "pericyte_length": peri_len.value,
                "spacer_length": spacer_len.value,
            },
            "inlets": sorted(inlet_ids),
            "pericyte_manual_added": sorted(state["peri_on"]),
            "pericyte_manual_removed": sorted(state["peri_off"]),
            "final": {"vessels": G.number_of_edges(), "nodes": G.number_of_nodes(),
                      "elements": len(state["rows"])},
            "steps": log,
        }
        rec_path = Path(str(out)[:-4] + "_steps.json")
        import json as _json
        with open(rec_path, "w") as fh:
            _json.dump(recipe, fh, indent=2)
        say(f"wrote {out.name}\nand {rec_path.name}\n"
            f"{len(state['rows'])} rows, {len(log)} steps recorded")

    def do_flatten():
        state["flat"] = not state["flat"]
        rebuild_layer()
        if state["plan"] is not None:
            draw_pericytes(state["plan"], highlight=state["hover"])
        if state["flat"]:
            try:
                viewer.dims.ndisplay = 2
                viewer.dims.set_current_step(0, 0)
            except Exception:
                pass
        recolour()
        if "over-connected" in viewer.layers:
            do_flag_high()
        say("flattened for hover picking" if state["flat"] else "3D restored")

    # ---------------- linked highlighting ----------------

    def neighbours_of(vid):
        """Vessels sharing a node with this one, split by flow direction."""
        up, down = set(), set()
        e = None
        for cand in state["edges"]:
            if vid_of(*cand) == vid:
                e = cand
                break
        if e is None:
            return up, down
        key = (min(e[0], e[1]), max(e[0], e[1]), e[2])
        d = state["direction"].get(key)
        frm, to = d if d else (e[0], e[1])
        for a, b, k in G.edges(frm, keys=True):
            other = vid_of(a, b, k)
            if other != vid:
                up.add(other)
        for a, b, k in G.edges(to, keys=True):
            other = vid_of(a, b, k)
            if other != vid:
                down.add(other)
        return up, down

    def recolour(hover=None):
        # `hover` names the vessel to draw white. Callers pass the selection;
        # mouse movement does not call this at all.
        hover = hover if hover is not None else state.get("selected")
        ids = list(vessels.features["vessel_id"])
        if not ids:
            return
        peri = {v for v, _ in (state["plan"]["eligible"] if state["plan"] else [])}
        up, down = (neighbours_of(hover) if hover else (set(), set()))
        hl = state.get("highlight", set())
        cols, wids = [], []
        for vid in ids:
            if hover and vid == hover:
                # the row picked in the issues table
                cols.append("white"); wids.append(5.5)
            elif vid in hl:
                # the rest of the vessels at that node
                cols.append("purple"); wids.append(4.0)
            elif vid in up:
                cols.append("orange"); wids.append(3.5)
            elif vid in down:
                cols.append("magenta"); wids.append(3.5)
            elif vid in inlet_ids:
                cols.append("lime"); wids.append(3.0)
            elif vid in peri:
                cols.append("cyan"); wids.append(1.8)
            else:
                cols.append("#2b6f7a"); wids.append(1.2)
        vessels.edge_color = cols
        vessels.edge_width = wids
        if state["plan"] is not None and "pericytes" in viewer.layers:
            draw_pericytes(state["plan"], highlight=hover)

    def select_row_for(vid):
        rows = state["rows"]
        if not rows:
            return
        # a vessel may appear as its own row and as a pericyte row
        for i, r in enumerate(rows):
            if vid in str(r.get("source_ids", "")):
                table.selectRow(i)
                table.scrollToItem(table.item(i, 0))
                return

    # Attached to the VIEWER, not the vessels layer: napari only delivers layer
    # mouse callbacks to the active layer, so adding the over-connected or
    # pericyte points layers silently stopped hovering from working.
    @viewer.mouse_move_callbacks.append
    def on_move(_viewer, event):
        # nearest sampled point wins. napari's own shape picking misses thin
        # paths almost every time, which is why hovering did nothing before.
        vid = None
        pts = state.get("tree_pts")
        if pts is not None and len(pts):
            pos = np.asarray(event.position, dtype=float)
            try:
                view_dir = getattr(event, "view_direction", None)
                if viewer.dims.ndisplay == 3 and view_dir is not None:
                    # 3D: the cursor is a ray through the scene, not a point, so
                    # take each sample's perpendicular distance to that ray
                    d = np.asarray(view_dir, dtype=float)
                    d = d / (np.linalg.norm(d) or 1.0)
                    rel = pts - pos[:3]
                    perp = rel - np.outer(rel @ d, d)
                    dist = np.linalg.norm(perp, axis=1)
                else:
                    # 2D: compare in the plane, and ignore anything that is not
                    # on the slice being displayed -- otherwise the cursor picks
                    # vessels that are not on screen
                    dist = np.linalg.norm(pts[:, -2:] - pos[-2:], axis=1)
                    if not state["flat"]:
                        try:
                            z_now = float(viewer.dims.current_step[0])
                        except Exception:
                            z_now = None
                        if z_now is not None:
                            off_slice = np.abs(pts[:, 0] - z_now) > 2.0
                            dist = np.where(off_slice, np.inf, dist)
                i = int(np.argmin(dist))
                if np.isfinite(dist[i]) and dist[i] < 14.0:
                    vid = state["tree_ids"][i]
            except Exception:
                vid = None
        if vid == state["hover"]:
            return
        state["hover"] = vid
        # deliberately no recolour here: repainting on every mouse move is what
        # made the highlight vanish while rotating
        if vid:
            row = row_for_vessel(vid)
            if row:
                show_info(row)
                popup(vid, row)
            else:
                L = None
                for a, b, k in G.edges(keys=True):
                    if vid_of(a, b, k) == vid:
                        L = edge_length(G, a, b, k)
                        break
                info.value = (f"{vid}\nlength {L:.1f}\n\n"
                              "(build the array for names\nand connectivity)"
                              if L is not None else vid)
                popup(vid, None, length=L)
            viewer.status = vid
        else:
            viewer.status = ""

    @viewer.mouse_drag_callbacks.append
    def on_click(_viewer, event):
        """A click with no drag selects the vessel under the cursor."""
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return
        vid = state.get("hover")
        if vid:
            select_vessel(vid)

    def select_vessel(vid):
        """Make this vessel the sticky selection and paint it."""
        state["selected"] = vid
        recolour(vid)
        row = row_for_vessel(vid)
        if row:
            show_info(row)
            select_row_for(vid)
        else:
            info.value = f"{vid}\n(build the array for details)"

    def on_row_clicked():
        r = table.currentRow()
        if r < 0 or r >= len(state["rows"]):
            return
        row = state["rows"][r]
        # match on the current graph id. source_ids are the ORIGINAL ids from
        # the file, and sweep/split renumber vessels, so matching on those
        # silently found nothing and left the highlight where it was.
        target = row.get("graph_vessel_id", "")
        ids = list(vessels.features["vessel_id"])
        sel = {i for i, v in enumerate(ids) if v == target}
        if not sel:                                  # fall back to originals
            want = set(str(row.get("source_ids", "")).split())
            sel = {i for i, v in enumerate(ids) if v in want}
        vessels.selected_data = set()
        if sel:
            state["selected"] = ids[min(sel)]
            recolour(state["selected"])
            show_info(row)
        else:
            info.value = f"{row.get('name')}: no geometry found"
    table.itemSelectionChanged.connect(on_row_clicked)

    issues.itemSelectionChanged.connect(on_issue_selected)
    node_pick.changed.connect(show_node_vessels)

    def btn(text, fn):
        b = PushButton(text=text)
        b.changed.connect(fn)
        return b

    panel = Container(widgets=[
        status,
        Label(value="--- selected element ---"),
        info,
        Label(value="--- simplify ---"),
        btn("Sweep degree-2 joins", do_sweep),
        Label(value="--- over-connected nodes ---"),
        btn("Find nodes joining >3 (red)", do_flag_high),
        node_pick,
        spacer_len,
        btn("Split ALL over-connected", do_split_high),
        btn("Split the selected node", do_split_one),
        btn("Delete this vessel", do_delete_selected),
        btn("Undo last change", restore),
        Label(value="--- pericytes ---"),
        min_len, peri_len,
        btn("Compute density", do_density),
        btn("Add pericyte to hovered vessel", do_peri_add),
        btn("Remove pericyte from hovered", do_peri_remove),
        btn("Clear manual pericyte edits", do_peri_clear_manual),
        Label(value="--- inlets ---"),
        btn("Mark selected as inlet", do_mark_inlets),
        Label(value="--- build ---"),
        btn("Flatten (needed for hover)", do_flatten),
        btn("Build vessel array", do_build),
        export_tag,
        btn("Export CSV", do_export),
        btn("Save steps log", lambda: _save_log()),
    ], labels=True)

    try:
        for w in panel:
            if isinstance(w, PushButton):
                w.native.setMinimumHeight(30)
            elif isinstance(w, (FloatSpinBox, LineEdit)):
                w.native.setMinimumHeight(26)
    except Exception:
        pass

    holder = QWidget()
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(2, 2, 2, 2)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel.native)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setMinimumHeight(180)
    split = QSplitter()
    split.setOrientation(Qt.Vertical)
    split.addWidget(scroll)
    split.addWidget(issues)
    split.addWidget(table)
    split.setSizes([320, 220, 320])
    lay.addWidget(split)
    holder.setMinimumWidth(430)

    viewer.window.add_dock_widget(holder, name="vessel array", area="right")

    # a colour key along the bottom, so none of this has to be remembered
    def swatch(colour, text):
        return (f'<span style="color:{colour};font-size:15px;">&#9632;</span>'
                f'<span style="color:#ddd;">&nbsp;{text}</span>')

    _ = None
    legend = QLabel(
        "&nbsp;&nbsp;".join([
            swatch("white", "selected vessel"),
            swatch("purple", "others at that node"),
            swatch("orange", "neighbours, one end"),
            swatch("magenta", "neighbours, other end"),
            swatch("lime", "inlet"),
            swatch("cyan", "has a pericyte"),
            swatch("gold", "pericyte site"),
            swatch("red", "selected pericyte"),
            swatch("#2b6f7a", "plain vessel"),
            swatch("red", "node joining &gt;3"),
            swatch("yellow", "crop box"),
        ])
    )
    legend.setTextFormat(Qt.RichText)
    legend.setStyleSheet("background:#1a1a1a; padding:4px;")
    legend.setMaximumHeight(34)
    viewer.window.add_dock_widget(legend, name="colour key", area="bottom")

    recolour()
    say(f"loaded {G.number_of_edges()} vessels\n"
        f"{len(inlet_ids)} inlet(s) from file\n\n"
        "sweep, set the pericyte rule,\nthen build")
    napari.run()


if __name__ == "__main__":
    main()
