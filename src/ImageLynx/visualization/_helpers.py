"""Helper functions for visualization."""
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def sort_branch_orders_numerically(orders: List[str]) -> List[str]:
    """Sort branch order strings like B01, Art2, Ven3 numerically."""
    def key_fn(s):
        m = re.search(r"(\d+)$", s, re.I)
        if not m:
            return (99, 0, s)
        n = int(m.group(1))
        prefix = re.sub(r"\d+$", "", s).lower()
        group = {"art": 0, "b": 1, "bo": 1, "ven": 2}.get(prefix, 3)
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
