"""E14.5 under a size convention: how many vessels (and how much length) each
class holds at a pair of thresholds, how the wide class sits against Harvey's
large masks, and how many separate pieces the wide vessels form. Widths are
the ones the run stamped; bridges are left out, and a bridge joins the wide
pieces it touches.

    python scripts/vessel_class_review/size_convention_counts.py 6 15
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOLVED = REPO / "outputs/vessel_class_review/e14p5_solved.pkl"
BRIDGE = "is_zero_resistance"


def size_class(d: dict, mid_min: float, wide_min: float) -> str:
    w = d.get("diameter_um")
    if w is None or not np.isfinite(w):
        return "unmeasured"
    return "wide" if w >= wide_min else "mid" if w >= mid_min else "narrow"


def main(mid_min: float, wide_min: float) -> None:
    with SOLVED.open("rb") as f:
        G = pickle.load(f)
    vessels = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if not d.get(BRIDGE)]
    count, length = Counter(), Counter()
    for *_, d in vessels:
        name = size_class(d, mid_min, wide_min)
        count[name] += 1
        length[name] += float(d.get("length") or 0.0)
    print(f"{len(vessels)} vessels at mid >= {mid_min:g} um, wide >= {wide_min:g} um")
    for name in ("narrow", "mid", "wide", "unmeasured"):
        if count[name]:
            print(f"  {name}: {count[name]} ({length[name] / 1000:.1f} mm)")
    tagged = [d for *_, d in vessels if d.get("large_mask_vessel_type")]
    wide_tagged = sum(size_class(d, mid_min, wide_min) == "wide" for d in tagged)
    print(f"  wide vessels in the large masks: {wide_tagged} of {count['wide']}; "
          f"mask-tagged vessels not wide: {len(tagged) - wide_tagged} of {len(tagged)}")
    narrow_sources = Counter(d.get("diameter_source") for *_, d in vessels
                             if size_class(d, mid_min, wide_min) == "narrow")
    print(f"  narrow widths by source: {dict(narrow_sources)}")
    H = nx.MultiGraph()
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get(BRIDGE) or size_class(d, mid_min, wide_min) == "wide":
            H.add_edge(u, v, k, bridge=bool(d.get(BRIDGE)))
    pieces = sorted((sum(not d["bridge"] for *_, d in H.subgraph(c).edges(data=True))
                     for c in nx.connected_components(H)), reverse=True)
    pieces = [p for p in pieces if p]
    print(f"  wide pieces: {len(pieces)}, single vessels {sum(p == 1 for p in pieces)}, "
          f"largest {pieces[:5]}")


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
