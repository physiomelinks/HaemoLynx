"""Harvey's small-vessel rule (2026-10-26), against grid SZ1's marks.

"They have to propagate from large vessels, then can descend in z as well as
be on the xy plane, and they should be larger than 6 microns, smaller than 10
microns." So a small vessel is one 6-10 um wide that is reached from a large
vessel through such vessels; its direction does not matter. After grid SZ1 (as
stated: 10/26) Harvey changed the band to 6-15 um the same day. The large vessels
are those the run tagged in Harvey's E14.5 large masks, and a thick-vessel
bridge counts as part of the large vessel it joins.

Two readings of "propagate": `strict` grows only through vessels in the band;
`through_wider` also grows through anything wider (only those in the band are
classed small). Each is scored on the run's final widths and, for the
vessels the raw image measured, on those alone.

    python scripts/vessel_class_review/small_vessel_rule.py
"""
from __future__ import annotations

import csv
import pickle
from collections import Counter, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/vessel_class_review"
SOLVED = OUT / "e14p5_solved.pkl"
GRID = OUT / "SZ1"
REPORT = GRID / "SZ1_rule_result.md"

SMALL_MIN_UM, SMALL_MAX_UM = 6.0, 15.0
BRIDGE = "is_zero_resistance"


def small_vessels(G, *, through_wider: bool, lo: float = SMALL_MIN_UM, hi: float = SMALL_MAX_UM) -> set:
    """Edge ids (frozenset({u, v}), key) the rule classes small."""
    large_nodes = set()
    for u, v, _k, d in G.edges(keys=True, data=True):
        if d.get("large_mask_vessel_type"):
            large_nodes |= {u, v}
    # A bridge onto a large vessel's centreline belongs to it.
    changed = True
    while changed:
        changed = False
        for u, v, _k, d in G.edges(keys=True, data=True):
            if d.get(BRIDGE) and (u in large_nodes) != (v in large_nodes):
                large_nodes |= {u, v}
                changed = True

    def width(d):
        w = d.get("diameter_um")
        return float(w) if w is not None and np.isfinite(w) else float("nan")

    def passable(d):
        w = width(d)
        if d.get(BRIDGE) or d.get("large_mask_vessel_type"):
            return False
        return lo <= w <= hi or (through_wider and w > hi)

    small, seen = set(), set(large_nodes)
    queue = deque(large_nodes)
    while queue:
        node = queue.popleft()
        for u, v, k, d in G.edges(node, keys=True, data=True):
            if not passable(d):
                continue
            if lo <= width(d) <= hi:
                small.add((frozenset((u, v)), k))
            other = v if u == node else u
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return small


def main() -> None:
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    marks = {int(r["id"]): r["mark"] for r in csv.DictReader((GRID / "SZ1_marks.csv").open(encoding="utf-8"))}
    key = {int(r["id"]): r for r in csv.DictReader((GRID / "SZ1_key.csv").open(encoding="utf-8"))}
    decided = sorted(i for i, m in marks.items() if m in ("S", "C"))
    vessels = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if not d.get(BRIDGE) and u != v]
    in_band = sum(SMALL_MIN_UM <= float(d.get("diameter_um") or 0) <= SMALL_MAX_UM for *_, d in vessels)
    lines = ["# Harvey's small-vessel rule against grid SZ1", "",
             "A small vessel: 6-15 um wide, reached from a large vessel (in Harvey's E14.5 large masks) "
             "through such vessels; any direction.", "",
             f"{len(vessels)} vessels; {in_band} are 6-15 um wide. Marked S or C: {len(decided)} "
             f"(all S: {sum(marks[i] == 'S' for i in decided)}/{len(decided)}).", "",
             "| Reading | Widths | Vessels classed small | SZ1 right | Wrong |", "|---|---|---|---|---|"]
    for through_wider in (False, True):
        small = small_vessels(G, through_wider=through_wider)
        reading = "through wider vessels too" if through_wider else f"strict: only through {SMALL_MIN_UM:g}-{SMALL_MAX_UM:g} um"
        for subset in ("all marked", "raw-image widths only"):
            pairs = []
            for i in decided:
                r = key[i]
                if subset != "all marked" and r["diameter_source"] != "endothelial":
                    continue
                u, v, k = (int(r["u"]) if r["u"].lstrip("-").isdigit() else r["u"],
                           int(r["v"]) if r["v"].lstrip("-").isdigit() else r["v"], int(r["key"]))
                called = (frozenset((u, v)), k) in small
                pairs.append((i, marks[i] == "S", called))
            right = sum(s == c for _i, s, c in pairs)
            wrong = ", ".join(f"{i}{'S' if s else 'C'}" for i, s, c in pairs if s != c)
            lines.append(f"| {reading} | {subset} | {len(small)} | {right}/{len(pairs)} | {wrong or '-'} |")
    lines += ["", "Each marked case:", "", "| ID | Mark | Width um | Source | Strict | Through wider |",
              "|---|---|---|---|---|---|"]
    strict, wider = small_vessels(G, through_wider=False), small_vessels(G, through_wider=True)
    for i in sorted(key, key=lambda i: float(key[i]["diameter_um"])):
        r = key[i]
        u, v, k = (int(r["u"]) if r["u"].lstrip("-").isdigit() else r["u"],
                   int(r["v"]) if r["v"].lstrip("-").isdigit() else r["v"], int(r["key"]))
        e = (frozenset((u, v)), k)
        lines.append(f"| {i} | {marks.get(i) or '-'} | {float(r['diameter_um']):.2f} | {r['diameter_source']} | "
                     f"{'small' if e in strict else '-'} | {'small' if e in wider else '-'} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(Counter(len(x) for x in (strict, wider)))


if __name__ == "__main__":
    main()
