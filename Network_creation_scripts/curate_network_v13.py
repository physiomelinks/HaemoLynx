#!/usr/bin/env python3
"""
Curate a HaemoLynx vessel network and designate its boundaries, in napari.

    python curate_network_v7.py <graph.pkl | curated.vtp> [--image image.tif]

Everything lives in one panel on the right:

  CURATE   collapse hairpins, clear over-connected hubs, drop short vessels,
           crop to a box (previewed before you commit), keep the largest
           component. Select vessels with the arrow tool and press Delete to
           remove them by hand. Every operation is undoable.

  BOUNDARY terminal vessels -- those with a free end -- draw in RED. Select
           them and mark inlets, which turn GREEN. Terminals left red are
           outlets. Marks survive further curation.

  EXPORT   writes the graph, a ParaView VTP, an undirected vessel array with
           the boundary designations, and a JSON recipe of every step taken.

Each vessel keeps a stable `vessel_id` of the form `u_v_key`, derived from the
graph rather than from row order. That is the key to join simulation results
back onto the geometry: write a table of vessel_id against your per-timestep
values, and the same id identifies the same tube in every file.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def vid_of(u, v, key):
    a, b = (u, v) if u <= v else (v, u)
    return f"{a}_{b}_{key}"


def load_network(path):
    """Accept the HaemoLynx graph pickle or a previously curated VTP."""
    import networkx as nx

    path = Path(path)
    if path.suffix == ".pkl":
        G = pickle.load(open(path, "rb"))
        polylines = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            vox = np.asarray(d.get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                vox = np.array([G.nodes[u]["pos"], G.nodes[v]["pos"]], dtype=float)
            polylines[(u, v, k)] = vox
        return G, polylines

    import pyvista as pv
    mesh = pv.read(path)
    if "vessel_id" not in mesh.cell_data:
        raise SystemExit("no 'vessel_id' array in that VTP")
    G = nx.MultiGraph()
    polylines = {}
    lengths = (np.asarray(mesh.cell_data["length"], dtype=float)
               if "length" in mesh.cell_data else None)
    for i, vid in enumerate(str(x) for x in mesh.cell_data["vessel_id"]):
        u_s, v_s, k_s = vid.rsplit("_", 2)
        u, v, k = int(u_s), int(v_s), int(k_s)
        pts = mesh.points[mesh.get_cell(i).point_ids][:, ::-1]   # xyz -> zyx
        G.add_edge(u, v, key=k, voxels=pts.tolist(),
                   length=float(lengths[i]) if lengths is not None else float("nan"))
        polylines[(u, v, k)] = pts
        G.nodes[u].setdefault("pos", pts[0])
        G.nodes[v].setdefault("pos", pts[-1])
    return G, polylines


def simplify(path, max_points=12):
    if len(path) <= max_points:
        return path
    return path[np.linspace(0, len(path) - 1, max_points).astype(int)]


def _elen(G, u, v, key):
    return float(G.edges[u, v, key].get("length", 0.0))


def collapse_parallel(G, max_length):
    removed = 0
    for u, v in {(min(a, b), max(a, b)) for a, b in G.edges()}:
        if not G.has_edge(u, v):
            continue
        keys = list(G[u][v].keys())
        if len(keys) < 2:
            continue
        lengths = {k: _elen(G, u, v, k) for k in keys}
        if min(lengths.values()) > max_length:
            continue
        keep = min(lengths, key=lengths.get)
        for k in keys:
            if k != keep:
                G.remove_edge(u, v, k)
                removed += 1
    return removed


def _arc_len(G, arc):
    total = 0.0
    for a, b in zip(arc, arc[1:]):
        if not G.has_edge(a, b):
            return float("inf")
        total += min(_elen(G, a, b, k) for k in G[a][b])
    return total


def collapse_loops(G, max_perimeter):
    import networkx as nx
    removed = 0
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from(G.edges())

    for cycle in nx.cycle_basis(simple):
        if len(cycle) < 3:
            continue
        perim = sum(min(_elen(G, a, b, k) for k in G[a][b])
                    for a, b in zip(cycle, cycle[1:] + cycle[:1]) if G.has_edge(a, b))
        if perim > max_perimeter:
            continue
        anchors = [n for n in cycle if G.degree(n) > 2]

        if len(anchors) == 2:
            ring = list(cycle)
            ia, ib = sorted((ring.index(anchors[0]), ring.index(anchors[1])))
            arc1, arc2 = ring[ia:ib + 1], ring[ib:] + ring[:ia + 1]
            longer = arc1 if _arc_len(G, arc1) > _arc_len(G, arc2) else arc2
            for x, y in zip(longer, longer[1:]):
                if G.has_edge(x, y):
                    for k in list(G[x][y].keys()):
                        G.remove_edge(x, y, k)
                        removed += 1
            for n in longer[1:-1]:
                if n in G and G.degree(n) == 0:
                    G.remove_node(n)
        elif len(anchors) <= 1:
            for x, y in zip(cycle, cycle[1:] + cycle[:1]):
                if G.has_edge(x, y):
                    for k in list(G[x][y].keys()):
                        G.remove_edge(x, y, k)
                        removed += 1
            for n in cycle:
                if n in G and G.degree(n) == 0:
                    G.remove_node(n)
    return removed


def box_edges(lo, hi):
    z0, y0, x0 = lo
    z1, y1, x1 = hi
    c = {"000": (z0, y0, x0), "001": (z0, y0, x1), "010": (z0, y1, x0),
         "011": (z0, y1, x1), "100": (z1, y0, x0), "101": (z1, y0, x1),
         "110": (z1, y1, x0), "111": (z1, y1, x1)}
    pairs = [("000", "001"), ("001", "011"), ("011", "010"), ("010", "000"),
             ("100", "101"), ("101", "111"), ("111", "110"), ("110", "100"),
             ("000", "100"), ("001", "101"), ("011", "111"), ("010", "110")]
    return [np.array([c[a], c[b]], dtype=float) for a, b in pairs]


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
                                  SpinBox, LineEdit, Label)

    G_full, polylines = load_network(src)
    print(f"loaded {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} vessels")

    all_edges = sorted(polylines.keys())
    all_ids = [vid_of(u, v, k) for u, v, k in all_edges]
    paths3d = [simplify(polylines[e]) for e in all_edges]
    # z zeroed: napari only draws a shape where it meets the current slice,
    # so a flattened copy is what makes every vessel selectable in 2D
    paths2d = [np.column_stack([np.zeros(len(p)), p[:, 1], p[:, 2]])
               for p in paths3d]
    flat = {"on": False}
    paths = paths3d
    id_to_edge = {vid_of(u, v, k): (u, v, k) for u, v, k in all_edges}

    inlet_ids = set()          # vessel_ids the user has marked as inlets
    history = []
    recipe = []

    viewer = napari.Viewer(title=f"curate: {src.name}")

    if image_path and image_path.exists():
        try:
            import tifffile
            viewer.add_image(tifffile.imread(image_path), name="image",
                             colormap="gray", opacity=0.4, blending="additive")
        except Exception as exc:
            print(f"image not loaded ({exc})")

    # lists, not fixed arrays: merging appends a brand new vessel
    features_all = pd.DataFrame({"vessel_id": list(all_ids)})
    vessels = viewer.add_shapes(paths, shape_type="path", features=features_all,
                                edge_color="cyan", edge_width=1.2, name="vessels")

    node_ids = list(G_full.nodes())
    all_pos = np.array([G_full.nodes[n]["pos"] for n in node_ids], dtype=float)
    nodes_layer = viewer.add_points(
        all_pos, size=np.clip([d for _, d in G_full.degree()], 3, 20),
        features=pd.DataFrame({"node_id": node_ids,
                               "degree": [d for _, d in G_full.degree()]}),
        face_color="degree", face_colormap="plasma", name="nodes", opacity=0.7)

    lo_all = np.floor(all_pos.min(axis=0))
    hi_all = np.ceil(all_pos.max(axis=0))
    print(f"data extent (z,y,x): {lo_all.tolist()} .. {hi_all.tolist()}")

    show_terminals = {"on": False}

    # ---------------- core state ----------------

    def surviving_graph():
        H = nx.MultiGraph()
        for vid in vessels.features["vessel_id"]:
            u, v, k = id_to_edge[vid]
            if G_full.has_edge(u, v, k):
                H.add_edge(u, v, key=k, **G_full.edges[u, v, k])
        for n in H.nodes():
            H.nodes[n].update(G_full.nodes[n])
        return H

    def terminal_ids(H=None):
        H = H if H is not None else surviving_graph()
        return {vid_of(u, v, k) for u, v, k in H.edges(keys=True)
                if H.degree(u) == 1 or H.degree(v) == 1}

    def recolour(H=None):
        ids = list(vessels.features["vessel_id"])
        if not ids:
            return
        terms = terminal_ids(H) if show_terminals["on"] or inlet_ids else set()
        cols, wids = [], []
        for vid in ids:
            if vid in inlet_ids:
                cols.append("lime"); wids.append(3.0)
            elif vid in terms and show_terminals["on"]:
                cols.append("red"); wids.append(3.0)
            else:
                cols.append("cyan"); wids.append(1.2)
        vessels.edge_color = cols
        vessels.edge_width = wids

    def sync_nodes(H):
        if H.number_of_nodes() == 0:
            nodes_layer.data = np.empty((0, 3))
            return
        ids = list(H.nodes())
        pos = np.array([H.nodes[n]["pos"] for n in ids], dtype=float)
        if flat["on"]:
            pos = pos.copy()
            pos[:, 0] = 0.0
        deg = np.array([d for _, d in H.degree()], dtype=float)
        nodes_layer.data = pos
        nodes_layer.features = pd.DataFrame({"node_id": ids, "degree": deg})
        nodes_layer.size = np.clip(deg, 3, 20)
        nodes_layer.face_color = "degree"

    def show_ids(alive):
        mask = features_all["vessel_id"].isin(alive).values
        active = paths2d if flat["on"] else paths3d
        vessels.data = [p for p, m in zip(active, mask) if m]
        vessels.features = features_all[mask].reset_index(drop=True)
        inlet_ids.intersection_update(alive)      # drop marks on deleted vessels
        H = surviving_graph()
        sync_nodes(H)
        recolour(H)

    def redraw(H):
        show_ids({vid_of(u, v, k) for u, v, k in H.edges(keys=True)})

    def checkpoint(label):
        history.append((label, set(vessels.features["vessel_id"]), set(inlet_ids)))
        if len(history) > 30:
            history.pop(0)

    def record(op, params, before, after):
        recipe.append({"step": len(recipe) + 1, "operation": op,
                       "parameters": params, "vessels_before": before,
                       "vessels_after": after, "removed": before - after})

    status = TextEdit(value="", label="")
    try:
        status.native.setReadOnly(True)
        status.native.setMinimumHeight(130)
        status.native.setMaximumHeight(180)
    except Exception:
        pass

    def report(prefix=""):
        H = surviving_graph()
        comps = nx.number_connected_components(H) if H.number_of_nodes() else 0
        dd = [d for _, d in H.degree()] or [0]
        terms = terminal_ids(H)
        lines = [prefix,
                 f"vessels:    {H.number_of_edges()}",
                 f"nodes:      {H.number_of_nodes()}",
                 f"components: {comps}   max degree: {max(dd)}",
                 f"terminal:   {len(terms)}",
                 f"inlet:      {len(inlet_ids)}   outlet: {len(terms) - len(inlet_ids)}",
                 f"undo: {len(history)}   recipe: {len(recipe)}"]
        text = "\n".join(x for x in lines if x)
        status.value = text
        viewer.status = text.replace("\n", "   ")
        print(text.replace("\n", " | "))

    def n_vessels():
        return len(vessels.data)

    # ---------------- curate ----------------

    hairpin_thresh = FloatSpinBox(value=40.0, min=0.0, max=1000.0, step=5.0,
                                  label="loop perimeter <")
    hub_degree = SpinBox(value=8, min=3, max=50, label="hub degree >")
    short_thresh = FloatSpinBox(value=10.0, min=0.0, max=500.0, step=1.0,
                                label="vessel length <")

    def _spin(val, lbl):
        w = FloatSpinBox(value=float(val), min=-1e6, max=1e6, step=10.0)
        w.label = lbl
        return w

    z_min, z_max = _spin(lo_all[0], "z from"), _spin(hi_all[0], "z to")
    y_min, y_max = _spin(lo_all[1], "y from"), _spin(hi_all[1], "y to")
    x_min, x_max = _spin(lo_all[2], "x from"), _spin(hi_all[2], "x to")

    def crop_bounds():
        return (np.array([z_min.value, y_min.value, x_min.value]),
                np.array([z_max.value, y_max.value, x_max.value]))

    def midpoint(H, u, v, k):
        vox = np.asarray(H.edges[u, v, k].get("voxels", []), dtype=float)
        if len(vox):
            return vox[len(vox) // 2]
        return (np.asarray(H.nodes[u]["pos"]) + np.asarray(H.nodes[v]["pos"])) / 2

    def do_hairpins():
        before = n_vessels(); checkpoint("hairpins")
        H = surviving_graph()
        n = collapse_parallel(H, hairpin_thresh.value)
        n += collapse_loops(H, hairpin_thresh.value)
        H.remove_nodes_from([x for x in list(H.nodes()) if H.degree(x) == 0])
        redraw(H)
        record("collapse_hairpins", {"max_perimeter": hairpin_thresh.value},
               before, n_vessels())
        report(f"removed {n} hairpin vessels")

    def do_hubs():
        before = n_vessels(); checkpoint("hubs")
        H = surviving_graph()
        hubs = [n for n, d in H.degree() if d > hub_degree.value]
        doomed = {(u, v, k) for n in hubs for u, v, k in H.edges(n, keys=True)}
        H.remove_edges_from(doomed)
        H.remove_nodes_from([x for x in list(H.nodes()) if H.degree(x) == 0])
        redraw(H)
        record("remove_hubs", {"max_degree": hub_degree.value}, before, n_vessels())
        report(f"cleared {len(hubs)} hubs, {len(doomed)} vessels")

    def do_short():
        before = n_vessels(); checkpoint("short")
        H = surviving_graph()
        doomed = [(u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
                  if d.get("length", np.inf) < short_thresh.value]
        H.remove_edges_from(doomed)
        H.remove_nodes_from([x for x in list(H.nodes()) if H.degree(x) == 0])
        redraw(H)
        record("drop_short", {"min_length": short_thresh.value}, before, n_vessels())
        report(f"dropped {len(doomed)} short vessels")

    def do_show_box():
        lo, hi = crop_bounds()
        e = box_edges(lo, hi)
        if "crop box" in viewer.layers:
            viewer.layers["crop box"].data = e
        else:
            viewer.add_shapes(e, shape_type="path", name="crop box",
                              edge_color="yellow", edge_width=2.0)
        H = surviving_graph()
        inside = sum(1 for u, v, k in H.edges(keys=True)
                     if not (np.any(midpoint(H, u, v, k) < lo)
                             or np.any(midpoint(H, u, v, k) > hi)))
        status.value = (f"crop box drawn in yellow\ninside:  {inside}\n"
                        f"outside: {H.number_of_edges() - inside} would go\n"
                        f"(switch to 3D to see the box)")

    def do_hide_box():
        if "crop box" in viewer.layers:
            viewer.layers.remove("crop box")

    def do_crop():
        before = n_vessels(); checkpoint("crop")
        lo, hi = crop_bounds()
        H = surviving_graph()
        doomed = [(u, v, k) for u, v, k in H.edges(keys=True)
                  if np.any(midpoint(H, u, v, k) < lo) or np.any(midpoint(H, u, v, k) > hi)]
        H.remove_edges_from(doomed)
        H.remove_nodes_from([x for x in list(H.nodes()) if H.degree(x) == 0])
        redraw(H)
        record("crop", {"z": [z_min.value, z_max.value], "y": [y_min.value, y_max.value],
                        "x": [x_min.value, x_max.value]}, before, n_vessels())
        report(f"cropped away {len(doomed)} vessels")

    def do_crop_reset():
        for w, v in [(z_min, lo_all[0]), (z_max, hi_all[0]), (y_min, lo_all[1]),
                     (y_max, hi_all[1]), (x_min, lo_all[2]), (x_max, hi_all[2])]:
            w.value = float(v)
        do_show_box()

    def do_component():
        before = n_vessels(); checkpoint("component")
        H = surviving_graph()
        if H.number_of_nodes():
            H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
            redraw(H)
        record("keep_largest_component", {}, before, n_vessels())
        report("kept largest component")

    def do_undo():
        if not history:
            status.value = "nothing to undo"
            return
        label, prev_ids, prev_inlets = history.pop()
        inlet_ids.clear(); inlet_ids.update(prev_inlets)
        show_ids(prev_ids)
        if recipe:
            recipe.pop()
        report(f"undid: {label}")

    def do_reset():
        checkpoint("reset")
        show_ids(set(features_all["vessel_id"]))
        recipe.clear()
        report("restored full network")

    # ---------------- boundaries ----------------

    def do_flatten():
        flat["on"] = not flat["on"]
        alive = set(vessels.features["vessel_id"])
        keep_inlets = set(inlet_ids)
        show_ids(alive)
        inlet_ids.update(keep_inlets & alive)
        recolour()

        if flat["on"]:
            # everything is projected onto z = 0, so the viewer has to be in 2D
            # and parked on slice 0 or there is simply nothing on screen
            try:
                viewer.dims.ndisplay = 2
                viewer.dims.set_current_step(0, 0)
            except Exception:
                pass
            viewer.layers.selection.active = vessels
            try:
                vessels.mode = "select"
            except Exception:
                pass
            status.value = ("FLATTENED onto slice 0\n\n"
                            "every vessel is visible and\n"
                            "clickable. the select tool is\n"
                            "already active: click or drag\n"
                            "over the red terminals, then\n"
                            "Mark selected as inlet.\n\n"
                            "click Flatten again to restore")
        else:
            status.value = "restored true 3D geometry"
        print(status.value.replace("\n", " "))

    def do_delete_selected():
        chosen = selected_ids()
        if not chosen:
            status.value = ("nothing selected\n\nswitch to select mode (S),\n"
                            "then click or drag over vessels")
            return
        before = n_vessels()
        checkpoint("delete selected")
        alive = set(vessels.features["vessel_id"]) - set(chosen)
        show_ids(alive)
        record("delete_selected", {"vessel_ids": sorted(chosen)}, before, n_vessels())
        report(f"deleted {len(chosen)} selected")

    def merge_pair(H, vid1, vid2):
        """Join two vessels meeting at a plain degree-2 node into one.

        Returns (new_vessel_id, message) or (None, reason).
        """
        e1, e2 = id_to_edge[vid1], id_to_edge[vid2]
        n1, n2 = {e1[0], e1[1]}, {e2[0], e2[1]}
        shared = n1 & n2
        if len(shared) != 1:
            return None, ("those two vessels do not meet at\n"
                          "exactly one node")
        mid = shared.pop()
        if H.degree(mid) != 2:
            return None, (f"their shared node {mid} has degree "
                          f"{H.degree(mid)}.\nmerging would bypass the other\n"
                          f"{H.degree(mid) - 2} vessel(s) attached to it")
        outer1 = (n1 - {mid}).pop()
        outer2 = (n2 - {mid}).pop()
        if outer1 == outer2:
            return None, "those two vessels form a closed loop"

        pos_mid = np.asarray(H.nodes[mid]["pos"], dtype=float)

        def oriented(edge, end_at_mid):
            vox = np.asarray(H.edges[edge].get("voxels", []), dtype=float)
            if vox.ndim != 2 or len(vox) < 2:
                a, b = edge[0], edge[1]
                vox = np.array([H.nodes[a]["pos"], H.nodes[b]["pos"]], dtype=float)
            d_first = np.linalg.norm(vox[0] - pos_mid)
            d_last = np.linalg.norm(vox[-1] - pos_mid)
            at_mid_is_last = d_last <= d_first
            if end_at_mid and not at_mid_is_last:
                vox = vox[::-1]
            if (not end_at_mid) and at_mid_is_last:
                vox = vox[::-1]
            return vox

        vox1 = oriented(e1, end_at_mid=True)     # runs outer1 -> mid
        vox2 = oriented(e2, end_at_mid=False)    # runs mid -> outer2

        # The two centrelines rarely meet exactly: smoothing is applied per
        # edge and pulls each endpoint slightly off the node. Keep both points
        # when there is a real gap, so the join spans it rather than deleting
        # geometry; drop the duplicate only when they genuinely coincide.
        gap = float(np.linalg.norm(vox1[-1] - vox2[0]))
        combined = (np.vstack([vox1, vox2]) if gap > 1e-6
                    else np.vstack([vox1, vox2[1:]]))

        length = (float(H.edges[e1].get("length", 0.0))
                  + float(H.edges[e2].get("length", 0.0)) + gap)

        # a fresh key so the id cannot collide with an existing vessel
        existing = list(G_full[outer1][outer2].keys()) if G_full.has_edge(outer1, outer2) else []
        new_key = (max(existing) + 1) if existing else 0
        G_full.add_edge(outer1, outer2, key=new_key,
                        voxels=combined.tolist(), length=length,
                        merged_from=[vid1, vid2])
        new_vid = vid_of(outer1, outer2, new_key)

        # register the new vessel with the display
        id_to_edge[new_vid] = (outer1, outer2, new_key)
        simplified = simplify(combined)
        paths3d.append(simplified)
        paths2d.append(np.column_stack([np.zeros(len(simplified)),
                                        simplified[:, 1], simplified[:, 2]]))
        features_all.loc[len(features_all)] = {"vessel_id": new_vid}
        note = f" (joint gap {gap:.1f})" if gap > 1.0 else ""
        return new_vid, f"joined into {new_vid}{note}"

    def do_merge_selected():
        chosen = selected_ids()
        if len(chosen) != 2:
            status.value = (f"select exactly 2 vessels\n"
                            f"({len(chosen)} selected)\n\n"
                            "they must meet at a node that\n"
                            "joins only those two")
            return
        before = n_vessels()
        H = surviving_graph()
        new_vid, msg = merge_pair(H, chosen[0], chosen[1])
        if new_vid is None:
            status.value = "cannot merge:\n\n" + msg
            print("merge refused: " + msg.replace("\n", " "))
            return
        checkpoint("merge")
        alive = (set(vessels.features["vessel_id"]) - set(chosen)) | {new_vid}
        show_ids(alive)
        record("merge_vessels", {"vessel_ids": sorted(chosen), "into": new_vid},
               before, n_vessels())
        report(msg)

    def replay(steps, inlets_from_file):
        """Re-apply a recorded recipe to the pristine network."""
        show_ids(set(all_ids))
        recipe.clear()
        inlet_ids.clear()
        applied, skipped = 0, []

        for step in steps:
            op = step.get("operation")
            par = step.get("parameters", {})
            H = surviving_graph()
            try:
                if op == "collapse_hairpins":
                    hairpin_thresh.value = par["max_perimeter"]
                    do_hairpins()
                elif op == "remove_hubs":
                    hub_degree.value = par["max_degree"]
                    do_hubs()
                elif op == "drop_short":
                    short_thresh.value = par["min_length"]
                    do_short()
                elif op == "crop":
                    z_min.value, z_max.value = par["z"]
                    y_min.value, y_max.value = par["y"]
                    x_min.value, x_max.value = par["x"]
                    do_crop()
                elif op == "keep_largest_component":
                    do_component()
                elif op == "delete_selected":
                    before = n_vessels()
                    checkpoint("replay delete")
                    alive = set(vessels.features["vessel_id"]) - set(par["vessel_ids"])
                    show_ids(alive)
                    record("delete_selected", par, before, n_vessels())
                elif op in ("weld_ends", "attach_branch"):
                    skipped.append(f"step {step.get('step')}: {op} "
                                   "cannot be replayed automatically")
                    continue
                elif op == "merge_vessels":
                    v1, v2 = par["vessel_ids"]
                    before = n_vessels()
                    new_vid, msg = merge_pair(surviving_graph(), v1, v2)
                    if new_vid is None:
                        skipped.append(f"step {step.get('step')}: merge ({msg})")
                        continue
                    checkpoint("replay merge")
                    alive = (set(vessels.features["vessel_id"]) - {v1, v2}) | {new_vid}
                    show_ids(alive)
                    record("merge_vessels", par, before, n_vessels())
                else:
                    skipped.append(f"step {step.get('step')}: unknown op '{op}'")
                    continue
                applied += 1
            except Exception as exc:
                skipped.append(f"step {step.get('step')}: {op} failed ({exc})")

        alive_now = set(vessels.features["vessel_id"])
        inlet_ids.update(set(inlets_from_file) & alive_now)
        missing = len(set(inlets_from_file) - alive_now)
        recolour()

        msg = f"replayed {applied}/{len(steps)} steps"
        if skipped:
            msg += f"\n{len(skipped)} could not be applied:"
            for line in skipped[:3]:
                msg += "\n  " + line
            for line in skipped:
                print("  skipped: " + line)
        if missing:
            msg += f"\n{missing} inlet(s) no longer exist"
        report(msg)

    def do_load_recipe():
        try:
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                None, "Load a recipe", str(src.parent), "JSON (*.json)")
        except Exception:
            path = ""
        if not path:
            status.value = "no file chosen"
            return
        with open(path) as fh:
            data = json.load(fh)
        steps = data.get("steps", [])
        inlets = data.get("inlets", [])
        if data.get("source") and data["source"] != src.name:
            print(f"note: recipe was recorded against {data['source']}, "
                  f"this network is {src.name}")
        print(f"loading {Path(path).name}: {len(steps)} steps, {len(inlets)} inlets")
        replay(steps, inlets)

    # ---------- connecting vessels that are not yet connected ----------

    def free_ends(H, vid):
        """The node(s) of this vessel that no other vessel uses."""
        u, v, k = id_to_edge[vid]
        return [n for n in (u, v) if H.degree(n) == 1]

    def register(vid, u, v, key, voxels, length, **extra):
        """Add a vessel to G_full and to the display arrays."""
        G_full.add_edge(u, v, key=key, voxels=np.asarray(voxels).tolist(),
                        length=float(length), **extra)
        id_to_edge[vid] = (u, v, key)
        simp = simplify(np.asarray(voxels, dtype=float))
        paths3d.append(simp)
        paths2d.append(np.column_stack([np.zeros(len(simp)), simp[:, 1], simp[:, 2]]))
        features_all.loc[len(features_all)] = {"vessel_id": vid}

    def next_key(u, v):
        if G_full.has_edge(u, v):
            return max(G_full[u][v].keys()) + 1
        return 0

    def do_weld_ends():
        """Join two vessels whose free ends sit near each other.

        Their endpoints are pulled onto a single shared node, so the two become
        connected. They stay two vessels; use Merge afterwards to make them one.
        """
        chosen = selected_ids()
        if len(chosen) != 2:
            status.value = f"select exactly 2 vessels\n({len(chosen)} selected)"
            return
        H = surviving_graph()
        v1, v2 = chosen
        ends1, ends2 = free_ends(H, v1), free_ends(H, v2)
        if not ends1 or not ends2:
            status.value = ("both vessels need a free end\n\n"
                            f"{v1}: {len(ends1)} free\n{v2}: {len(ends2)} free")
            return

        # nearest pair of free ends
        best = None
        for a in ends1:
            for b in ends2:
                if a == b:
                    continue
                d = float(np.linalg.norm(np.asarray(H.nodes[a]["pos"], dtype=float)
                                         - np.asarray(H.nodes[b]["pos"], dtype=float)))
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            status.value = "those vessels already share that node"
            return
        dist, node_a, node_b = best

        before = n_vessels()
        checkpoint("weld ends")

        # rebuild vessel 2 with its free end moved onto node_a
        u2, v2n, k2 = id_to_edge[v2]
        vox = np.asarray(G_full.edges[u2, v2n, k2].get("voxels", []), dtype=float)
        pos_a = np.asarray(H.nodes[node_a]["pos"], dtype=float)
        pos_b = np.asarray(H.nodes[node_b]["pos"], dtype=float)
        if np.linalg.norm(vox[0] - pos_b) <= np.linalg.norm(vox[-1] - pos_b):
            vox = np.vstack([pos_a, vox])          # extend the near end to node_a
        else:
            vox = np.vstack([vox, pos_a])
        other = v2n if u2 == node_b else u2
        new_key = next_key(other, node_a)
        new_vid = vid_of(other, node_a, new_key)
        register(new_vid, other, node_a, new_key, vox,
                 G_full.edges[u2, v2n, k2].get("length", 0.0) + dist,
                 welded_from=v2)

        alive = (set(vessels.features["vessel_id"]) - {v2}) | {new_vid}
        show_ids(alive)
        record("weld_ends", {"vessel_ids": sorted(chosen), "gap": round(dist, 2),
                             "into": new_vid}, before, n_vessels())
        report(f"welded across {dist:.1f}\nnow share node {node_a}")

    def do_attach_branch():
        """Make a real Y: attach one vessel's free end onto another vessel.

        The target vessel is split at its closest point, creating a new node,
        and the branch is joined there. That is how a T or Y junction is
        represented -- the through-vessel becomes two edges meeting the branch
        at one node.
        """
        chosen = selected_ids()
        if len(chosen) != 2:
            status.value = (f"select exactly 2 vessels\n({len(chosen)} selected)\n\n"
                            "first the branch (with a free end),\n"
                            "then the vessel to attach it to")
            return
        H = surviving_graph()

        # whichever has a free end is the branch
        branch = target = None
        for a, b in ((chosen[0], chosen[1]), (chosen[1], chosen[0])):
            if free_ends(H, a):
                branch, target = a, b
                break
        if branch is None:
            status.value = "neither vessel has a free end"
            return

        bu, bv, bk = id_to_edge[branch]
        tip = free_ends(H, branch)[0]
        tip_pos = np.asarray(H.nodes[tip]["pos"], dtype=float)

        tu, tv, tk = id_to_edge[target]
        tvox = np.asarray(G_full.edges[tu, tv, tk].get("voxels", []), dtype=float)
        if len(tvox) < 3:
            status.value = "target vessel is too short to split"
            return

        d = np.linalg.norm(tvox - tip_pos, axis=1)
        i = int(np.argmin(d))
        i = min(max(i, 1), len(tvox) - 2)          # never split at an endpoint
        gap = float(d[i])

        before = n_vessels()
        checkpoint("attach branch")

        # a new node where the branch lands
        junction = max(G_full.nodes()) + 1
        G_full.add_node(junction, pos=tvox[i])

        tlen = float(G_full.edges[tu, tv, tk].get("length", 0.0))
        frac = i / (len(tvox) - 1)
        for node_end, seg, seg_len in (
                (tu, tvox[:i + 1], tlen * frac),
                (tv, tvox[i:], tlen * (1 - frac))):
            k = next_key(node_end, junction)
            register(vid_of(node_end, junction, k), node_end, junction, k,
                     seg, seg_len, split_from=target)

        # extend the branch to reach the junction
        bvox = np.asarray(G_full.edges[bu, bv, bk].get("voxels", []), dtype=float)
        if np.linalg.norm(bvox[0] - tip_pos) <= np.linalg.norm(bvox[-1] - tip_pos):
            bvox = np.vstack([tvox[i], bvox])
        else:
            bvox = np.vstack([bvox, tvox[i]])
        other = bv if bu == tip else bu
        k = next_key(other, junction)
        new_branch = vid_of(other, junction, k)
        register(new_branch, other, junction, k, bvox,
                 G_full.edges[bu, bv, bk].get("length", 0.0) + gap,
                 attached_from=branch)

        new_ids = {vid_of(tu, junction, next_key(tu, junction) - 1),
                   vid_of(tv, junction, next_key(tv, junction) - 1),
                   new_branch}
        alive = (set(vessels.features["vessel_id"]) - {branch, target}) | new_ids
        show_ids(alive)
        record("attach_branch", {"branch": branch, "target": target,
                                 "gap": round(gap, 2), "junction_node": junction},
               before, n_vessels())
        report(f"Y junction at node {junction}\ngap bridged {gap:.1f}")

    def do_toggle_terminals():
        show_terminals["on"] = not show_terminals["on"]
        recolour()
        state = "shown in red" if show_terminals["on"] else "hidden"
        report(f"terminals {state}")

    def selected_ids():
        sel = list(vessels.selected_data)
        feats = vessels.features
        return [str(feats.iloc[i]["vessel_id"]) for i in sel if i < len(feats)]

    def do_mark_inlet():
        chosen = selected_ids()
        if not chosen:
            status.value = ("nothing selected\n\nuse the arrow tool, then click\n"
                            "or drag a box over the red vessels")
            return
        terms = terminal_ids()
        good = [v for v in chosen if v in terms]
        inlet_ids.update(good)
        recolour()
        skipped = len(chosen) - len(good)
        report(f"marked {len(good)} inlet" + (f" ({skipped} not terminal)" if skipped else ""))

    def do_unmark():
        for v in selected_ids():
            inlet_ids.discard(v)
        recolour()
        report("unmarked selection")

    def do_select_terminals():
        terms = terminal_ids()
        feats = vessels.features
        vessels.selected_data = {i for i in range(len(feats))
                                 if str(feats.iloc[i]["vessel_id"]) in terms
                                 and str(feats.iloc[i]["vessel_id"]) not in inlet_ids}
        report(f"selected {len(vessels.selected_data)} unmarked terminals")

    def do_clear_inlets():
        inlet_ids.clear()
        recolour()
        report("all inlets cleared")

    # ---------------- export ----------------

    export_tag = LineEdit(value="", label="export tag")

    def vessel_array_rows(H):
        terms = terminal_ids(H)
        rows = []
        for i, (u, v, k) in enumerate(sorted(H.edges(keys=True))):
            vid = vid_of(u, v, k)
            at_a = sorted(vid_of(a, b, kk) for a, b, kk in H.edges(u, keys=True)
                          if vid_of(a, b, kk) != vid)
            at_b = sorted(vid_of(a, b, kk) for a, b, kk in H.edges(v, keys=True)
                          if vid_of(a, b, kk) != vid)
            rows.append({
                "vessel": i + 1, "vessel_id": vid,
                "node_a": u, "node_b": v,
                "vessels_at_node_a": " ".join(at_a),
                "vessels_at_node_b": " ".join(at_b),
                "degree_a": H.degree(u), "degree_b": H.degree(v),
                "is_terminal": vid in terms,
                "boundary": "inlet" if vid in inlet_ids
                            else ("outlet" if vid in terms else ""),
                "length": H.edges[u, v, k].get("length", float("nan")),
            })
        return rows

    def do_export():
        H = surviving_graph()
        stamp = datetime.now().strftime("%H%M%S")
        tag = export_tag.value.strip()
        stem = src.with_name(src.stem + f"_curated_{stamp}" + (f"_{tag}" if tag else ""))
        terms = terminal_ids(H)

        with open(str(stem) + ".pkl", "wb") as fh:
            pickle.dump(H, fh)

        rows = vessel_array_rows(H)
        pd.DataFrame(rows).to_csv(str(stem) + "_vessel_array.csv", index=False)

        with open(str(stem) + "_recipe.json", "w") as fh:
            json.dump({"source": src.name,
                       "created": datetime.now().isoformat(timespec="seconds"),
                       "source_totals": {"nodes": G_full.number_of_nodes(),
                                         "vessels": G_full.number_of_edges()},
                       "final_totals": {"nodes": H.number_of_nodes(),
                                        "vessels": H.number_of_edges()},
                       "inlets": sorted(inlet_ids),
                       "n_terminal": len(terms),
                       "note": ("Manual selections made in the viewer are not "
                                "itemised; the totals include them."),
                       "steps": recipe}, fh, indent=2)

        note = ""
        try:
            import pyvista as pv
            pts, lines, ids, lengths, bnd = [], [], [], [], []
            for u, v, k in sorted(H.edges(keys=True)):
                vox = np.asarray(H.edges[u, v, k].get("voxels", []), dtype=float)
                # note: always the real geometry, even while flattened
                if len(vox) < 2:
                    vox = np.array([H.nodes[u]["pos"], H.nodes[v]["pos"]], dtype=float)
                xyz = vox[:, ::-1]
                start = len(pts)
                pts.extend(xyz.tolist())
                lines.extend([len(xyz)] + list(range(start, start + len(xyz))))
                vid = vid_of(u, v, k)
                ids.append(vid)
                lengths.append(H.edges[u, v, k].get("length", np.nan))
                bnd.append("inlet" if vid in inlet_ids
                           else ("outlet" if vid in terms else "interior"))
            poly = pv.PolyData(np.array(pts), lines=np.array(lines))
            poly.cell_data["vessel_id"] = np.array(ids)
            poly.cell_data["length"] = np.array(lengths, dtype=float)
            poly.cell_data["boundary"] = np.array(bnd)
            poly.save(str(stem) + ".vtp")
        except Exception as exc:
            note = f"\nVTP skipped: {exc}"

        status.value = (f"wrote {Path(str(stem)).name}\n"
                        f"  .pkl  .vtp  _vessel_array.csv  _recipe.json\n"
                        f"{H.number_of_edges()} vessels, {len(inlet_ids)} inlet, "
                        f"{len(terms) - len(inlet_ids)} outlet{note}")
        print(status.value)
        if not inlet_ids:
            status.value += "\n\nWARNING: no inlets marked"

    # ---------------- panel ----------------

    def btn(text, fn):
        b = PushButton(text=text)
        b.changed.connect(fn)
        return b

    for w in (z_min, z_max, y_min, y_max, x_min, x_max):
        w.changed.connect(lambda *_: do_show_box() if "crop box" in viewer.layers else None)

    panel = Container(widgets=[
        btn("Count what's left", lambda: report("current state")), status,
        btn("Undo last operation", do_undo),
        Label(value="--- curate ---"),
        hairpin_thresh, btn("Collapse hairpins / short loops", do_hairpins),
        hub_degree, btn("Remove vessels at hubs", do_hubs),
        short_thresh, btn("Drop short vessels", do_short),
        z_min, z_max, y_min, y_max, x_min, x_max,
        btn("Show crop box", do_show_box), btn("Confirm crop", do_crop),
        btn("Hide crop box", do_hide_box), btn("Reset crop box", do_crop_reset),
        btn("Keep largest component", do_component),
        btn("Restore full network", do_reset),
        btn("Delete selected vessels", do_delete_selected),
        btn("Merge 2 selected vessels", do_merge_selected),
        btn("Weld 2 free ends together", do_weld_ends),
        btn("Attach branch (make Y junction)", do_attach_branch),
        Label(value="--- boundaries ---"),
        btn("Highlight terminals (red)", do_toggle_terminals),
        btn("Flatten to 2D for selection", do_flatten),
        btn("Mark selected as inlet", do_mark_inlet),
        btn("Unmark selected", do_unmark),
        btn("Select unmarked terminals", do_select_terminals),
        btn("Clear all inlets", do_clear_inlets),
        Label(value="--- export ---"),
        export_tag, btn("Export everything", do_export),
        btn("Load recipe and replay", do_load_recipe),
    ], labels=True)

    # Each row needs real vertical room or Qt clips its text. That makes the
    # panel taller than any screen, so it goes inside a real scroll area --
    # a QScrollArea has its own small minimum height and simply scrolls the
    # tall content, instead of forcing the whole window to grow.
    try:
        for w in panel:
            n = w.native
            if isinstance(w, PushButton):
                n.setMinimumHeight(30)
            elif isinstance(w, (FloatSpinBox, SpinBox, LineEdit)):
                n.setMinimumHeight(26)
            elif isinstance(w, Label):
                n.setMinimumHeight(24)
        status.native.setMinimumHeight(150)
        status.native.setMaximumHeight(190)
    except Exception as exc:
        print(f"row sizing skipped: {exc}")

    try:
        from qtpy.QtWidgets import QScrollArea
        from qtpy.QtCore import Qt

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel.native)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setMinimumWidth(285)
        scroll.setMaximumWidth(320)
        scroll.setMinimumHeight(200)          # small, so the window stays free
        dock = viewer.window.add_dock_widget(scroll, name="curate", area="right")
    except Exception as exc:
        print(f"scroll area unavailable ({exc}); using plain dock")
        dock = viewer.window.add_dock_widget(panel, name="curate", area="right")

    report("loaded")
    print("\nCurate first, then Highlight terminals and mark your inlets.")
    print("Terminals left red at export are treated as outlets.\n")
    napari.run()


if __name__ == "__main__":
    main()
