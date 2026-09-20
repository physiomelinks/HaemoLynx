"""Helper functions for visualization."""
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE

#: Sort-group rank per branch-order prefix, derived from
#: graph.branch_order's own BRANCH_ORDER_CATEGORY_SEQUENCE -- the single
#: source of truth for which category (Large_Art outermost on the arterial
#: side, then Art, B/BO capillary, Ven, Large_Ven outermost on the venous
#: side) sorts before which, so this and statistics.stats's own copy cannot
#: silently drift apart (see that constant's own docstring). Only the
#: prefix spellings a raw (non-normalized) branch-order label can actually
#: have -- "b" and "bo" both -- are this module's own concern.
_BRANCH_ORDER_SORT_GROUPS = {
    prefix: BRANCH_ORDER_CATEGORY_SEQUENCE.index(category)
    for prefix, category in {
        "large_art": "large_arteriole",
        "art": "arteriole",
        "b": "capillary",
        "bo": "capillary",
        "ven": "venule",
        "large_ven": "large_venule",
    }.items()
}


def sort_branch_orders_numerically(orders: List[str]) -> List[str]:
    """Sort branch order strings like B01, Art2, Ven3, Large_Art1 numerically."""
    def key_fn(s):
        m = re.search(r"(\d+)$", s, re.I)
        if not m:
            return (99, 0, s)
        n = int(m.group(1))
        prefix = re.sub(r"\d+$", "", s).lower()
        group = _BRANCH_ORDER_SORT_GROUPS.get(prefix, len(_BRANCH_ORDER_SORT_GROUPS))
        return (group, n, s)
    return sorted(orders, key=key_fn)


def create_color_mapping(
    branch_orders: List[str],
    color_palette: str = "viridis",
    reverse_gradient: bool = False,
    group_above: Optional[int] = None,
) -> Dict[str, tuple]:
    """Map branch order -> (r,g,b,a) color."""
    cmap = plt.get_cmap(color_palette)
    if reverse_gradient:
        cmap = cmap.reversed()
    n = len(branch_orders)
    mapping = {}
    for i, bo in enumerate(branch_orders):
        val = i / (n - 1) if n > 1 else 0.5
        mapping[bo] = cmap(val)
    return mapping


def group_branch_orders_for_legend(
    branch_orders: List[str],
    group_above: Optional[int],
    actual_edge_counts: Dict[str, int],
) -> Tuple[List[str], Dict[str, int]]:
    """Group high branch orders for legend. Returns (legend_orders, legend_counts)."""
    if group_above is None:
        return branch_orders, actual_edge_counts
    legend_orders = []
    legend_counts = {}
    for bo in branch_orders:
        m = re.search(r"(\d+)$", bo, re.I)
        num = int(m.group(1)) if m else 0
        prefix = re.sub(r"\d+$", "", bo).lower()
        if prefix not in {"b", "bo"}:
            legend_orders.append(bo)
            legend_counts[bo] = actual_edge_counts.get(bo, 0)
            continue
        if num <= group_above:
            legend_orders.append(bo)
            legend_counts[bo] = actual_edge_counts.get(bo, 0)
        else:
            label = f"BO{group_above}+"
            if label not in legend_counts:
                legend_orders.append(label)
                legend_counts[label] = 0
            legend_counts[label] += actual_edge_counts.get(bo, 0)
    return legend_orders, legend_counts
