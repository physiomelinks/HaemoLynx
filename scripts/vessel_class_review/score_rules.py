"""Phase 0b of PLAN_vessel_classes_from_segmentation.md: the published
arteriole/venule rules, as published (nothing fitted), against Stack A's real
large masks.

The truth is each vessel's `large_mask_vessel_type`, the tag the run's own
overlap rule gave it from Harvey's ilastik arteriole and Vein masks. Stack A has
few large trees, so everything here screens the rules; it decides nothing.

    python scripts/vessel_class_review/score_rules.py
"""
from __future__ import annotations

import csv
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
SOLVED = OUT / "stack_a_solved.pkl"
CSV_OUT = OUT / "stack_a_vessels.csv"
REPORT = OUT / "phase0b_report.md"

MEASURES = ("diameter_um", "length_over_diameter", "ksd_per_um", "distance_factor",
            "emergence_deg", "murray_exponent", "depth_um", "angle_to_z_deg")


def auc(arterioles, venules) -> float:
    """P(a venule's value > an arteriole's): 0.5 is no separation."""
    a, v = np.asarray(arterioles, float), np.asarray(venules, float)
    a, v = a[np.isfinite(a)], v[np.isfinite(v)]
    if not len(a) or not len(v):
        return float("nan")
    greater = (v[:, None] > a[None, :]).sum() + 0.5 * (v[:, None] == a[None, :]).sum()
    return float(greater / (len(a) * len(v)))


def pries_roy(G, inlets, outlets, threshold) -> dict:
    """Pries-Roy, as Smith et al. 2015 state it: vessels at or above the
    capillary cut-off reached from the inlets downstream through diverging
    nodes are arterioles; from the outlets upstream through converging nodes,
    venules. Directions are the solved flows'."""
    from measures import BRIDGE

    out_edges, in_edges = defaultdict(list), defaultdict(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        q = d.get("flow_signed")
        if q is None or not np.isfinite(q) or q == 0 or u == v:
            continue
        a, b = (u, v) if q > 0 else (v, u)
        out_edges[a].append(((u, v, k), b, d))
        in_edges[b].append(((u, v, k), a, d))

    def big(d):
        w = d.get("diameter_um")
        return d.get(BRIDGE) or (w is not None and np.isfinite(w) and float(w) >= threshold)

    def grow(starts, forward):
        reached, stack, seen = set(), list(starts), set(starts)
        while stack:
            node = stack.pop()
            for key, nxt, d in (out_edges if forward else in_edges)[node]:
                if not big(d):
                    continue
                reached.add(key)
                n_in, n_out = len(in_edges[nxt]), len(out_edges[nxt])
                passes = (n_in == 1 and n_out >= 1) if forward else (n_out == 1 and n_in >= 1)
                if passes and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return reached

    arteriolar, venular = grow(inlets, True), grow(outlets, False)
    labels = {}
    for key in arteriolar:
        labels[key] = "arteriole"
    for key in venular:
        labels[key] = "both" if key in labels else "venule"
    return labels


#: The run's boundary pressures (its run_settings.yaml): the solved graph keeps
#: no inlet/outlet lists, but those nodes are held at exactly these.
INLET_PA, OUTLET_PA = 4500.0, 1000.0


def boundary_nodes(G) -> tuple[list, list]:
    inlets = [n for n, d in G.nodes(data=True) if abs(d.get("pressure", -1) - INLET_PA) < 1e-6]
    outlets = [n for n, d in G.nodes(data=True) if abs(d.get("pressure", -1) - OUTLET_PA) < 1e-6]
    return inlets, outlets


def scored(pairs) -> str:
    """'right/n (all-one-way best/n)' for (truth, predicted) pairs."""
    pairs = [(t, p) for t, p in pairs if t in ("arteriole", "venule") and p in ("arteriole", "venule")]
    if not pairs:
        return "no cases"
    right = sum(t == p for t, p in pairs)
    one_way = max(Counter(t for t, _ in pairs).values())
    return f"{right}/{len(pairs)} (all one way: {one_way}/{len(pairs)})"


def main() -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measures import CAPILLARY_UNDER_UM, measure

    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    vessels = measure(G)
    for vessel in vessels:
        vessel.measures["diameter_um"] = vessel.diameter_um

    lines = ["# Phase 0b: published rules against Stack A's large masks", "",
             f"Solved graph: `{SOLVED.name}` ({G.number_of_edges()} edges). "
             f"Vessels at or above {CAPILLARY_UNDER_UM} um: {len(vessels)}, in "
             f"{len({v.tree for v in vessels})} trees.", ""]
    truth_counts = Counter(v.truth for v in vessels)
    lines.append(f"Truth (large-mask tag) among them: {dict(truth_counts)}.")
    sources = Counter(v.diameter_source for v in vessels)
    lines.append(f"Diameter sources: {dict(sources)}.")
    lines.append("")

    labelled = [v for v in vessels if v.truth in ("arteriole", "venule")]
    a = [v for v in labelled if v.truth == "arteriole"]
    vv = [v for v in labelled if v.truth == "venule"]
    lines += ["## Each measure, arterioles against venules (vessels in a large mask)", "",
              "AUC = P(a venule's value > an arteriole's); 0.5 separates nothing. Vessels of one",
              "tree are not independent, so read this as a screen.", "",
              "| Measure | Arteriole median | Venule median | AUC |", "|---|---|---|---|"]
    for name in MEASURES:
        av = [v.measures[name] for v in a]
        vvv = [v.measures[name] for v in vv]
        lines.append(f"| {name} | {np.nanmedian(av):.3g} | {np.nanmedian(vvv):.3g} | {auc(av, vvv):.2f} |"
                     if av and vvv else f"| {name} | - | - | - |")
    lines.append("")

    # The rules as published, on vessels with a partner running beside them.
    paired = [v for v in labelled if v.measures["pair_partner"] is not None]
    m1 = [(v.truth, "venule" if v.measures["pair_width_ratio"] > 1 else "arteriole") for v in paired]
    m2, ksd = [], []
    for v in paired:
        w = vessels[v.measures["pair_partner"]]
        if np.isfinite(v.measures["length_over_diameter"]) and np.isfinite(w.measures["length_over_diameter"]):
            m2.append((v.truth, "arteriole" if v.measures["length_over_diameter"] > w.measures["length_over_diameter"] else "venule"))
        if np.isfinite(v.measures["ksd_per_um"]) and np.isfinite(w.measures["ksd_per_um"]):
            ksd.append((v.truth, "venule" if v.measures["ksd_per_um"] > w.measures["ksd_per_um"] else "arteriole"))
    opposite = [v for v in paired if vessels[v.measures["pair_partner"]].truth in ("arteriole", "venule")
                and vessels[v.measures["pair_partner"]].truth != v.truth]
    m1_opp = [(v.truth, "venule" if v.measures["pair_width_ratio"] > 1 else "arteriole") for v in opposite]
    lines += ["## The rules as published (nothing fitted)", "",
              f"Vessels in a large mask with a partner beside them: {len(paired)} "
              f"(of {len(labelled)}); their partner in the other mask: {len(opposite)}.", "",
              f"- **M1, the wider of a pair is the venule**: {scored(m1)}; on arteriole-venule pairs only: {scored(m1_opp)}",
              f"- **M2, the higher length over diameter is the arteriole** (within the pair): {scored(m2)}",
              f"- **Ksd, the more tortuous is the venule** (Lorthois 2014 found no difference): {scored(ksd)}", ""]

    inlets, outlets = boundary_nodes(G)
    pr = pries_roy(G, inlets, outlets, CAPILLARY_UNDER_UM)
    pr_pairs = [(v.truth, pr.get(v.key) or pr.get((v.v, v.u, v.key[2]))) for v in labelled]
    reached = sum(p in ("arteriole", "venule") for _, p in pr_pairs)
    lines += [f"- **Pries-Roy** (flow direction from the run's {len(inlets)} inlets and {len(outlets)} outlets,"
              f" which the masks chose): {scored(pr_pairs)}; reached {reached} of {len(labelled)} labelled vessels,"
              f" {sum(p == 'both' for _, p in pr_pairs)} from both sides", ""]

    trees = defaultdict(list)
    for v in vessels:
        trees[v.tree].append(v)
    lines += ["## Trees holding large-mask vessels", "",
              "| Tree | Truth by length | Length um | Vessels | Median d um | Median L/D | Median Ksd 1/um | Mean depth um |",
              "|---|---|---|---|---|---|---|---|"]
    for t, members in sorted(trees.items(), key=lambda kv: -sum(v.length_um for v in kv[1])):
        by = Counter()
        for v in members:
            by[v.truth or "none"] += v.length_um
        if not (by["arteriole"] or by["venule"]):
            continue
        total = sum(v.length_um for v in members)
        share = ", ".join(f"{k} {by[k] / total:.0%}" for k in ("arteriole", "venule", "none") if by[k])
        lines.append(f"| {t} | {share} | {total:.0f} | {len(members)} | "
                     f"{np.median([v.diameter_um for v in members]):.1f} | "
                     f"{np.nanmedian([v.measures['length_over_diameter'] for v in members]):.2f} | "
                     f"{np.nanmedian([v.measures['ksd_per_um'] for v in members]):.3f} | "
                     f"{np.mean([v.measures['depth_um'] for v in members]):.0f} |")
    lines.append("")
    trees_total = len(trees)
    lines.append(f"Trees of vessels at or above {CAPILLARY_UNDER_UM} um: {trees_total}; with at least 2 vessels: "
                 f"{sum(len(m) >= 2 for m in trees.values())}.")

    fields = ["index", "u", "v", "key", "tree", "truth", "diameter_source", "length_um",
              *MEASURES, "pair_partner", "pair_share", "pair_width_ratio", "parent_ends", "daughter_ends",
              "pries_roy", "centre_z_um", "centre_y_um", "centre_x_um"]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, v in enumerate(vessels):
            mid = v.points[len(v.points) // 2]
            row = {"index": i, "u": v.u, "v": v.v, "key": v.key[2], "tree": v.tree, "truth": v.truth or "",
                   "diameter_source": v.diameter_source, "length_um": round(v.length_um, 2),
                   "pries_roy": pr.get(v.key) or "",
                   "centre_z_um": round(float(mid[0]), 1), "centre_y_um": round(float(mid[1]), 1),
                   "centre_x_um": round(float(mid[2]), 1)}
            for name in (*MEASURES, "pair_partner", "pair_share", "pair_width_ratio", "parent_ends", "daughter_ends"):
                value = v.measures.get(name)
                row[name] = "" if value is None or (isinstance(value, float) and math.isnan(value)) else (
                    round(value, 4) if isinstance(value, float) else value)
            w.writerow(row)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {CSV_OUT} and {REPORT}")


if __name__ == "__main__":
    main()
