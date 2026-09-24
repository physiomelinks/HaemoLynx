"""Capillary blocks: stall chosen vessels and see what the network does.

A stalled capillary -- plugged by a leukocyte, a platelet aggregate or a
microembolus -- still exists; it just carries (almost) nothing. So a block is
modelled as a near-infinite resistance on the vessel rather than its removal:
the vessel's resistance is multiplied by ``capillary_block_resistance_factor``
(10^6 by default, so it carries about a millionth of what it did). Removing it
instead could leave part of the network with no inlet or outlet, where the
pressure system has no solution at all.

Which vessels are blocked is chosen one of two ways:

* ``branch_order_probability`` -- a fraction of the vessels of the chosen
  branch orders (e.g. 10% of BO2), drawn from a seed so a run repeats. The
  fraction is exact: ``round(probability * N)`` of the ``N`` candidates, not
  each vessel independently, so 10% of 50 vessels is always 5.
* ``vessel_ids`` -- listed vessels, by the ``branchID`` the vessels layer's
  hover tooltip shows (the vessel's position in the graph's edge order), or
  as ``[u, v, key]``.

After the perturbed network is re-solved, :func:`compare_block_to_baseline`
sets its flows against the baseline's: total flow and equivalent resistance,
how many vessels (and how much length) lost most of their flow, how many
reversed, how much each outlet still drains, the transit-time distribution,
and all of it per branch order.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from haemolynx.haemodynamics.poiseuille import set_edge_resistance

__all__ = [
    "BLOCK_SELECTIONS",
    "BLOCK_EDGE_ATTRIBUTE",
    "FLOW_CHANGE_EDGE_ATTRIBUTE",
    "block_vessels",
    "compare_block_to_baseline",
    "resolve_blocked_vessels",
]

#: The two ways of choosing what to block.
BLOCK_SELECTIONS: tuple[str, ...] = ("branch_order_probability", "vessel_ids")

#: Written on every vessel of a blocked network: ``blocked`` or ``open``.
BLOCK_EDGE_ATTRIBUTE = "capillary_block"
#: Written on every vessel of a blocked network: its flow as a fraction of its
#: baseline flow (NaN where the baseline carried none).
FLOW_CHANGE_EDGE_ATTRIBUTE = "flow_change_vs_baseline"


def _edges(G: nx.MultiGraph) -> list[tuple[Any, Any, Any, dict]]:
    return list(G.edges(keys=True, data=True))


def _normalised_order(label: Any) -> str | None:
    """A branch-order label in one spelling (``BO2`` for ``B02``/``bo 2``);
    a bare number means that capillary order."""
    from haemolynx.statistics.bifurcation import _normalize_branch_order_tag

    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        label = f"BO{int(label)}"
    elif isinstance(label, str) and label.strip().isdigit():
        label = f"BO{int(label.strip())}"
    return _normalize_branch_order_tag(label)


def resolve_blocked_vessels(
    G: nx.MultiGraph,
    *,
    selection: str,
    branch_orders: Iterable[Any] = (),
    probability: float = 0.1,
    seed: int | None = None,
    vessel_ids: Iterable[Any] = (),
) -> tuple[list[tuple[Any, Any, Any]], dict[str, Any]]:
    """The vessels to block, as ``(u, v, key)`` in graph edge order, and a
    summary of how they were chosen.

    Raises ``ValueError`` for anything that would silently block the wrong
    vessels: an unknown selection, a vessel id that is not in *G*, or no
    branch orders to choose from.
    """
    edges = _edges(G)
    if selection == "branch_order_probability":
        wanted = {_normalised_order(order) for order in branch_orders}
        wanted.discard(None)
        if not wanted:
            raise ValueError(
                "capillary_block_branch_orders is empty: name the branch orders "
                "to block vessels from, e.g. ['BO2']"
            )
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"capillary_block_probability must be between 0 and 1, got {probability!r}"
            )
        candidates = [
            index
            for index, (_u, _v, _k, data) in enumerate(edges)
            if _normalised_order(data.get("branch_order")) in wanted
        ]
        count = int(math.floor(float(probability) * len(candidates) + 0.5))
        rng = np.random.default_rng(seed)
        chosen = (
            sorted(rng.choice(candidates, size=count, replace=False).tolist())
            if count
            else []
        )
        summary = {
            "selection": selection,
            "branch_orders": sorted(wanted),
            "probability": float(probability),
            "seed": seed,
            "candidate_vessels": len(candidates),
        }
    elif selection == "vessel_ids":
        position = {(u, v, k): index for index, (u, v, k, _d) in enumerate(edges)}
        chosen_set: set[int] = set()
        for vessel_id in vessel_ids:
            chosen_set.add(_edge_index(vessel_id, edges, position))
        if not chosen_set:
            raise ValueError(
                "capillary_block_vessel_ids is empty: list the branchIDs (shown "
                "when hovering a vessel) to block"
            )
        chosen = sorted(chosen_set)
        summary = {"selection": selection, "vessel_ids": chosen}
    else:
        raise ValueError(
            f"capillary_block_selection must be one of {BLOCK_SELECTIONS}, got {selection!r}"
        )
    blocked = [edges[index][:3] for index in chosen]
    summary["blocked_vessels"] = len(blocked)
    return blocked, summary


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def _edge_index(vessel_id: Any, edges: list, position: Mapping) -> int:
    """The edge-order index a user-given vessel id names."""
    if isinstance(vessel_id, bool):
        raise ValueError(f"vessel id {vessel_id!r} is not a branchID")
    if isinstance(vessel_id, (int, np.integer)) or (
        isinstance(vessel_id, str) and vessel_id.strip().isdigit()
    ):
        index = int(vessel_id)
        if not 0 <= index < len(edges):
            raise ValueError(
                f"vessel branchID {index} is not in this network (0 to {len(edges) - 1})"
            )
        return index
    if isinstance(vessel_id, (list, tuple)) and len(vessel_id) == 3:
        # A tuple node id read back from a config file arrives as a list.
        u, v, key = (_hashable(part) for part in vessel_id)
        for candidate in ((u, v, key), (v, u, key)):
            if candidate in position:
                return position[candidate]
        raise ValueError(f"vessel {list(vessel_id)!r} (u, v, key) is not in this network")
    raise ValueError(
        f"vessel id {vessel_id!r} is neither a branchID nor a [u, v, key] triple"
    )


def block_vessels(
    G: nx.MultiGraph, blocked: Sequence[tuple[Any, Any, Any]], factor: float
) -> None:
    """Multiply each blocked vessel's resistance by *factor*, and mark every
    vessel ``blocked``/``open`` in :data:`BLOCK_EDGE_ATTRIBUTE`.

    Rebinds ``resistance``/``conductance`` (never mutates a shared value), as
    a perturbation's shallow graph copy requires.
    """
    if not float(factor) > 1.0:
        raise ValueError(
            f"capillary_block_resistance_factor must be greater than 1, got {factor!r}"
        )
    chosen = set(blocked)
    for u, v, key, data in G.edges(keys=True, data=True):
        is_blocked = (u, v, key) in chosen
        data[BLOCK_EDGE_ATTRIBUTE] = "blocked" if is_blocked else "open"
        if is_blocked:
            set_edge_resistance(data, float(data["resistance"]) * float(factor))


# --- comparison with the baseline ---------------------------------------------------


def _signed_flow(data: dict) -> float:
    value = data.get("flow_signed")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _net_flow_into(G: nx.MultiGraph, node: Any) -> float:
    """Net flow arriving at *node* through its vessels, from node pressures
    and conductances -- ``flow_signed`` runs along each edge's stored
    orientation, which says nothing about which end is *node*."""
    p_node = G.nodes[node].get("pressure")
    if p_node is None or not math.isfinite(float(p_node)):
        return math.nan
    total = 0.0
    for _node, neighbour, _key, data in G.edges(node, keys=True, data=True):
        p_other = G.nodes[neighbour].get("pressure")
        g = data.get("conductance")
        if neighbour == node or p_other is None or g is None:
            continue
        total += float(g) * (float(p_other) - float(p_node))
    return total


def _outlet_flows(G: nx.MultiGraph, outlets: Iterable[Any]) -> dict:
    """Flow draining out of the network at each outlet node."""
    return {outlet: _net_flow_into(G, outlet) for outlet in outlets if outlet in G}


def compare_block_to_baseline(
    baseline: nx.MultiGraph,
    blocked: nx.MultiGraph,
    *,
    inlet_nodes: Iterable[Any],
    outlet_nodes: Iterable[Any],
    hypoperfusion_fraction: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """How the blocked network's flow differs from the baseline's.

    Returns ``(summary, rows_by_branch_order)``. A vessel is *hypoperfused*
    when its flow falls by at least *hypoperfusion_fraction* of its baseline
    flow, and *reversed* when its flow runs the other way (both only for
    vessels that carried flow at baseline, and not counting the blocked
    vessels themselves). Writes :data:`FLOW_CHANGE_EDGE_ATTRIBUTE` onto every
    vessel of *blocked*.
    """
    from haemolynx.statistics.bifurcation import (
        _branch_order_sort_key,
        branch_order_label,
    )
    from haemolynx.statistics.transit_time import compute_transit_times

    inlets = list(inlet_nodes)
    outlets = list(outlet_nodes)
    base_flows = []
    for u, v, key, data in baseline.edges(keys=True, data=True):
        base_flows.append(abs(_signed_flow(data)))
    base_flows = np.asarray(base_flows, dtype=float)
    finite = base_flows[np.isfinite(base_flows)]
    carrying_floor = 1e-9 * float(finite.max()) if finite.size else 0.0

    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "vessels": 0,
            "blocked": 0,
            "baseline_flow_sum": 0.0,
            "blocked_flow_sum": 0.0,
            "hypoperfused": 0,
            "hypoperfused_length_um": 0.0,
            "reversed": 0,
        }
    )
    hypoperfused = reversed_count = carrying = 0
    hypoperfused_length = total_length = 0.0
    for (u, v, key, before), q_before in zip(baseline.edges(keys=True, data=True), base_flows):
        after = blocked[u][v][key]
        q_signed_before = _signed_flow(before)
        q_signed_after = _signed_flow(after)
        q_after = abs(q_signed_after)
        is_blocked = after.get(BLOCK_EDGE_ATTRIBUTE) == "blocked"
        length = float(before.get("length", 0.0) or 0.0)
        row = rows[branch_order_label(before.get("branch_order"))]
        row["vessels"] += 1
        row["blocked"] += int(is_blocked)
        if math.isfinite(q_before):
            row["baseline_flow_sum"] += q_before
        if math.isfinite(q_after):
            row["blocked_flow_sum"] += q_after
        was_carrying = math.isfinite(q_before) and q_before > carrying_floor
        after[FLOW_CHANGE_EDGE_ATTRIBUTE] = (
            q_after / q_before if was_carrying and math.isfinite(q_after) else math.nan
        )
        total_length += length
        if not was_carrying or is_blocked:
            continue
        carrying += 1
        if q_after <= (1.0 - float(hypoperfusion_fraction)) * q_before:
            hypoperfused += 1
            hypoperfused_length += length
            row["hypoperfused"] += 1
            row["hypoperfused_length_um"] += length
        if (
            math.isfinite(q_signed_after)
            and abs(q_signed_after) > carrying_floor
            and np.sign(q_signed_after) != np.sign(q_signed_before)
        ):
            reversed_count += 1
            row["reversed"] += 1

    base_outlets = _outlet_flows(baseline, outlets)
    new_outlets = _outlet_flows(blocked, outlets)
    outlet_floor = 1e-9 * max((abs(q) for q in base_outlets.values() if math.isfinite(q)), default=0.0)
    retained = [
        new_outlets[o] / base_outlets[o]
        for o in base_outlets
        if math.isfinite(base_outlets[o]) and base_outlets[o] > outlet_floor
    ]
    outlet_losing_half = sum(1 for fraction in retained if fraction <= 0.5)

    def total_inflow(G: nx.MultiGraph) -> float:
        return float(sum(-_net_flow_into(G, inlet) for inlet in inlets if inlet in G))

    q_before_total = total_inflow(baseline)
    q_after_total = total_inflow(blocked)
    summary: dict[str, Any] = {
        "total_inflow_baseline": q_before_total,
        "total_inflow_blocked": q_after_total,
        "total_inflow_percent_change": (
            (q_after_total - q_before_total) / q_before_total * 100.0
            if q_before_total
            else math.nan
        ),
        "blocked_vessel_length_um": float(
            sum(
                float(d.get("length", 0.0) or 0.0)
                for _u, _v, _k, d in blocked.edges(keys=True, data=True)
                if d.get(BLOCK_EDGE_ATTRIBUTE) == "blocked"
            )
        ),
        "hypoperfusion_threshold_fraction": float(hypoperfusion_fraction),
        "hypoperfused_vessels": hypoperfused,
        "hypoperfused_vessel_fraction": hypoperfused / carrying if carrying else math.nan,
        "hypoperfused_length_um": hypoperfused_length,
        "hypoperfused_length_fraction": (
            hypoperfused_length / total_length if total_length else math.nan
        ),
        "reversed_vessels": reversed_count,
        "outlets": len(base_outlets),
        "outlet_flow_retained_mean": float(np.mean(retained)) if retained else math.nan,
        "outlet_flow_retained_min": float(np.min(retained)) if retained else math.nan,
        "outlets_losing_half_their_flow": outlet_losing_half,
    }

    # Transit times on copies: compute_transit_times writes per-vessel
    # attributes, and neither graph here should gain them from a comparison.
    for label, G in (("baseline", baseline), ("blocked", blocked)):
        times = compute_transit_times(G.copy(), inlets, outlets)
        for key, name in (
            ("Mean Transit Time (s)", "mean_transit_time_s"),
            ("Transit Time SD (s)", "transit_time_sd_s"),
            ("Capillary Mean Transit Time (s)", "capillary_mean_transit_time_s"),
            ("Capillary Transit Time Heterogeneity CTH (s)", "capillary_transit_time_heterogeneity_s"),
        ):
            if key in times:
                summary[f"{name}_{label}"] = times[key]
        if "Transit Time Status" in times:
            summary[f"transit_time_status_{label}"] = times["Transit Time Status"]

    by_order = []
    for label in sorted(rows, key=_branch_order_sort_key):
        row = rows[label]
        before_sum, after_sum = row["baseline_flow_sum"], row["blocked_flow_sum"]
        by_order.append(
            {
                "branch_order": label,
                "vessels": row["vessels"],
                "blocked": row["blocked"],
                "baseline_mean_flow": before_sum / row["vessels"],
                "blocked_mean_flow": after_sum / row["vessels"],
                "mean_flow_percent_change": (
                    (after_sum - before_sum) / before_sum * 100.0 if before_sum else math.nan
                ),
                "hypoperfused": row["hypoperfused"],
                "hypoperfused_length_um": row["hypoperfused_length_um"],
                "reversed": row["reversed"],
            }
        )
    return summary, by_order
