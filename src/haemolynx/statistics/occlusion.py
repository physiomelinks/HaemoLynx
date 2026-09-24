"""What blocking vessels does to inlet->outlet flow.

* :func:`compute_single_vessel_occlusion_impact` -- for every vessel, the
  flow lost (and the length of other vessels starved of flow) if that one
  vessel alone is blocked, summarised for the whole network, per vessel
  type, and per branch order within each type.
* :func:`compute_occlusion_curves` -- progressive, cumulative removal of
  vessels (at random, busiest first, narrowest first), tracking remaining
  flow and inlet->outlet connectivity.

Both use :class:`~haemolynx.statistics._flow_system.TerminalFlowSystem`
with fixed conductances: they do not model haematocrit or viscosity
redistributing after an occlusion.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from ._flow_system import TerminalFlowSystem
from ._route_graph import (
    _clear_edge_attributes,
    _effective_weighting,
    _iter_edges,
    _positive_finite,
    _resolve_terminals,
    _route_graph,
    _set_edge,
    _terminal_problem,
    _unavailable,
)
from ._sampling import REPRODUCIBILITY_SEED
from .bifurcation import (
    UNASSIGNED_CATEGORY,
    _branch_order_sort_key,
    branch_order_category,
    branch_order_label,
)
from haemolynx.graph.branch_order import BRANCH_ORDER_CATEGORY_SEQUENCE

__all__ = [
    "OCCLUSION_IMPACT_EDGE_ATTRIBUTES",
    "compute_single_vessel_occlusion_impact",
    "compute_occlusion_curves",
]

OCCLUSION_IMPACT_EDGE_ATTRIBUTES: tuple[str, ...] = (
    "occlusion_flow_loss",
    "occlusion_hypoperfused_length_um",
)

#: ``fast`` mode evaluates at most this many vessels exactly -- the ones
#: carrying the most flow power (the vessel's own flow x pressure drop,
#: a lower bound on the flow its occlusion costs); the rest are reported
#: "not evaluated". ``full`` evaluates every flow-carrying vessel. Each is
#: one sparse solve, ~18 ms on the 27k-node / 78k-vessel benchmark lattice:
#: 500 is ~13 s there, every vessel ~25 min.
DEFAULT_FAST_MAX_EVALUATED = 500
#: Vessels whose updates are applied together per batch (one sparse solve
#: per vessel; the batch only vectorises what follows the solves).
DEFAULT_BATCH = 32
DEFAULT_TOP_N = 10
ASSUMPTIONS = (
    "fixed vessel conductances; no haematocrit or viscosity redistribution "
    "after the occlusion"
)


def _edge_data(G: nx.Graph, u: Any, v: Any, key: Any) -> dict:
    return G[u][v][key] if key is not None else G[u][v]


def _vessel_length(data: dict) -> float:
    return float(data["length"]) if _positive_finite(data.get("length")) else 1.0


def _summary(values: np.ndarray, hypo: np.ndarray, ids: list, total: int) -> Dict[str, Any]:
    finite = np.isfinite(values)
    result: Dict[str, Any] = {"Vessel Count": total, "Evaluated Count": int(finite.sum())}
    if not finite.any():
        result["Status"] = "N/A (no evaluated vessel)"
        return result
    v = values[finite]
    h = hypo[finite]
    best = int(np.argmax(np.where(finite, values, -np.inf)))
    result.update(
        {
            "Mean Flow Loss": float(v.mean()),
            "Median Flow Loss": float(np.median(v)),
            "P90 Flow Loss": float(np.percentile(v, 90)),
            "Max Flow Loss": float(v.max()),
            "Mean Hypoperfused Length": float(h.mean()),
            "Max Hypoperfused Length": float(h.max()),
            "Most Critical Vessel": (*ids[best], round(float(values[best]), 4)),
        }
    )
    return result


def compute_single_vessel_occlusion_impact(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
    statistics_mode: str = "fast",
    hypoperfusion_fraction: float = 0.5,
    max_evaluated_fast: int = DEFAULT_FAST_MAX_EVALUATED,
    batch: int = DEFAULT_BATCH,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Where would blocking a single vessel have the most effect?

    For every vessel, two exact consequences of blocking it alone:

    * ``occlusion_flow_loss`` -- the fraction of total inlet->outlet flow
      lost (1.0 for a vessel every route depends on, 0 for a dead-end spur);
    * ``occlusion_hypoperfused_length_um`` -- total length of *other*
      vessels whose flow falls by at least *hypoperfusion_fraction* (a local
      measure: a capillary feeding a small region matters locally even when
      the network-wide loss is tiny).

    Computed exactly with a rank-one (Sherman-Morrison) update of the one
    factorised pressure solve: blocking vessel *e* (conductance g, flow f)
    shifts the free pressures by ``f / (1 - g R_e) * L^-1 b_e``, where
    ``R_e = b_e . L^-1 b_e`` -- one sparse solve per vessel, which also
    gives every other vessel's new flow. A vessel carrying no flow changes
    nothing when blocked, so it is 0 without a solve.

    Summarised (1) for the whole network, (2) by vessel type, and (3) by
    branch order within each vessel type (``branch_order`` labels; vessels
    without one are "unassigned"). Writes both attributes onto every edge;
    in ``fast`` mode beyond *max_evaluated_fast* flow-carrying vessels, the
    lower-power remainder are NaN ("not evaluated").
    """
    _clear_edge_attributes(G, OCCLUSION_IMPACT_EDGE_ATTRIBUTES)
    effective, label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    problem = _terminal_problem(inlets, outlets, _route_graph(G, effective).graph)
    if problem is not None:
        return _unavailable("Occlusion Impact", problem, label, both)
    system = TerminalFlowSystem(G, inlets, outlets, effective)
    if system.total_flow <= 0.0:
        return _unavailable("Occlusion Impact", "no inlet-outlet flow", label, both)

    f = system.flow
    g = system.conductance
    m = f.size
    lengths = np.asarray(
        [_vessel_length(_edge_data(G, u, v, key)) for u, v, key in system.edges]
    )
    carrying = np.abs(f) > 1e-12 * float(np.abs(f).max())
    loss = np.zeros(m)
    hypo = np.zeros(m)
    candidates = np.flatnonzero(carrying)
    method = f"exact ({candidates.size} flow-carrying vessels)"
    if statistics_mode == "fast" and candidates.size > max_evaluated_fast:
        power = f[candidates] ** 2 / g[candidates]
        keep = candidates[np.argsort(-power)[:max_evaluated_fast]]
        skipped = np.setdiff1d(candidates, keep)
        loss[skipped] = np.nan
        hypo[skipped] = np.nan
        method = (
            f"exact for the {max_evaluated_fast} highest-flow-power of "
            f"{candidates.size} flow-carrying vessels; the rest not evaluated"
        )
        candidates = keep

    q0 = system.total_flow
    threshold = (1.0 - float(hypoperfusion_fraction)) * np.abs(f)
    incidence = system.incidence_free
    for start in range(0, candidates.size, batch):
        idx = candidates[start : start + batch]
        rhs = incidence[idx].T.toarray()  # n_free x b
        if system.lu is not None and rhs.size:
            # One column at a time: SuperLU's own multi-right-hand-side solve
            # is ~3.5x slower per column than solving them singly.
            x = np.empty_like(rhs)
            for column in range(rhs.shape[1]):
                x[:, column] = system.lu.solve(np.ascontiguousarray(rhs[:, column]))
            r_eff = np.einsum("ij,ji->i", incidence[idx].toarray(), x)
            dflow = (incidence @ x) * g[:, None]
        else:
            r_eff = np.zeros(idx.size)
            dflow = np.zeros((m, idx.size))
        denom = 1.0 - g[idx] * r_eff
        coef = np.where(np.abs(denom) > 1e-12, f[idx] / np.where(denom == 0, 1, denom), 0.0)
        new = f[:, None] + dflow * coef[None, :]
        new[idx, np.arange(idx.size)] = 0.0
        q_new = system.inflow_sign @ new
        loss[idx] = np.clip(1.0 - q_new / q0, 0.0, 1.0)
        starved = carrying[:, None] & (np.abs(new) <= threshold[:, None])
        starved[idx, np.arange(idx.size)] = False
        hypo[idx] = lengths @ starved

    # --- write attributes and group -----------------------------------------
    position = system.edge_position
    all_loss, all_hypo, ids, categories, orders = [], [], [], [], []
    every_vessel_has_length = True
    for u, v, key, data in _iter_edges(G):
        every_vessel_has_length &= _positive_finite(data.get("length"))
        i = position.get((u, v, key))
        lo = float(loss[i]) if i is not None else 0.0
        hy = float(hypo[i]) if i is not None else 0.0
        _set_edge(G, u, v, key, "occlusion_flow_loss", lo)
        _set_edge(G, u, v, key, "occlusion_hypoperfused_length_um", hy)
        all_loss.append(lo)
        all_hypo.append(hy)
        ids.append((u, v, key))
        categories.append(branch_order_category(data.get("branch_order")))
        orders.append(branch_order_label(data.get("branch_order")))
    all_loss = np.asarray(all_loss, dtype=float)
    all_hypo = np.asarray(all_hypo, dtype=float)

    network = _summary(all_loss, all_hypo, ids, len(ids))
    finite = np.isfinite(all_loss)
    ranked = np.argsort(-np.where(finite, all_loss, -np.inf))[:top_n]
    network["Top Vessels"] = [(*ids[i], round(float(all_loss[i]), 4)) for i in ranked if finite[i]]
    network["Vessels Losing Over 1% Of Flow"] = int(np.sum(all_loss[finite] > 0.01))
    network["Vessels Losing Over 10% Of Flow"] = int(np.sum(all_loss[finite] > 0.10))
    network["Vessels Losing Over 99% Of Flow"] = int(np.sum(all_loss[finite] > 0.99))

    by_type: Dict[str, Any] = {}
    by_order: Dict[str, Any] = {}
    type_rank = {c: i for i, c in enumerate(BRANCH_ORDER_CATEGORY_SEQUENCE + (UNASSIGNED_CATEGORY,))}
    members_by_type: dict = defaultdict(list)
    for i, category in enumerate(categories):
        members_by_type[category].append(i)
    for category in sorted(members_by_type, key=lambda c: type_rank.get(c, len(type_rank))):
        rows = np.asarray(members_by_type[category])
        by_type[category] = _summary(all_loss[rows], all_hypo[rows], [ids[r] for r in rows], rows.size)
        members_by_order: dict = defaultdict(list)
        for r in rows:
            members_by_order[orders[r]].append(r)
        by_order[category] = {
            order: _summary(
                all_loss[np.asarray(o_rows)],
                all_hypo[np.asarray(o_rows)],
                [ids[r] for r in o_rows],
                len(o_rows),
            )
            for order, o_rows in sorted(
                members_by_order.items(), key=lambda kv: _branch_order_sort_key(kv[0])
            )
        }

    result: Dict[str, Any] = {
        "Occlusion Impact Route Weighting": label,
        "Occlusion Impact Method": method,
        "Occlusion Impact Assumptions": ASSUMPTIONS,
        "Occlusion Hypoperfusion Threshold (fraction)": float(hypoperfusion_fraction),
        "Occlusion Hypoperfused Length Unit": "um" if every_vessel_has_length else "vessels",
        "Occlusion Impact (Network)": network,
        "Occlusion Impact By Vessel Type": by_type,
        "Occlusion Impact By Branch Order": by_order,
    }
    if both:
        result["Occlusion Impact Nodes Both Inlet And Outlet (excluded)"] = both
    return result


# --- cumulative occlusion curves -------------------------------------------------

#: Removal steps per curve (each a sparse factorisation of the thinned
#: network, ~1.5 s on the 27k-node benchmark lattice).
FAST_CURVE_STEPS = 5
FULL_CURVE_STEPS = 25

#: ``np.trapezoid`` is NumPy >= 2.0; ``np.trapz`` is the same function before it.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _interpolated_crossing(fractions: list, flows: list, level: float) -> Any:
    """First removal fraction at which *flows* falls to *level*, linearly
    interpolated between the two steps either side of it."""
    for i, flow in enumerate(flows):
        if flow <= level:
            if i == 0:
                return float(fractions[0])
            f0, f1 = fractions[i - 1], fractions[i]
            q0, q1 = flows[i - 1], flow
            return float(f0 + (q0 - level) * (f1 - f0) / (q0 - q1))
    return f"> {fractions[-1]:g} (never reached)"


def _curve(G, inlets, outlets, weighting, order, fractions, q0, intact) -> tuple[list, list]:
    flows, reach = [], []
    m = len(order)
    for fraction in fractions:
        # Round half up: Python's round() rounds 0.5 to 0, so on a small
        # network the first step would silently remove nothing.
        removed = set(order[: int(fraction * m + 0.5)])
        # Nothing removed is the intact network, already solved: each
        # factorisation is most of this function's cost.
        system = intact if not removed else TerminalFlowSystem(
            G, inlets, outlets, weighting, removed=removed
        )
        flows.append(max(0.0, system.total_flow * system.scale / q0))
        reach.append(system.reachable_outlet_fraction(inlets, outlets))
    return flows, reach


def compute_occlusion_curves(
    G: nx.Graph,
    inlet_nodes: Optional[Iterable[Any]],
    outlet_nodes: Optional[Iterable[Any]],
    *,
    weighting: str = "length",
    statistics_mode: str = "fast",
    max_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Remaining flow and connectivity as vessels are blocked cumulatively.

    Three removal orders, each up to *max_fraction* of the vessels:
    ``random`` (seeded, averaged over 3 repeats in ``fast`` / 5 in
    ``full``), ``highest_current_flow_share`` (the vessels carrying the most
    of the inlet->outlet flow first, ranked once on the intact network), and
    ``narrowest_diameter`` (smallest ``diameter_um`` first -- small-vessel
    disease; skipped without diameters). At each step: remaining flow as a
    fraction of the intact network's, and the fraction of outlets still
    connected to an inlet. Summaries: a robustness index (mean remaining
    flow across the removal range, 1 = unaffected) and the removal fraction
    at which flow halves.
    """
    if not (0.0 < max_fraction <= 1.0):
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction!r}")
    effective, label = _effective_weighting(G, weighting)
    inlets, outlets, both = _resolve_terminals(G, inlet_nodes, outlet_nodes)
    problem = _terminal_problem(inlets, outlets, _route_graph(G, effective).graph)
    if problem is not None:
        return _unavailable("Occlusion Curves", problem, label, both)
    base = TerminalFlowSystem(G, inlets, outlets, effective)
    q0 = base.total_flow * base.scale
    if q0 <= 0.0:
        return _unavailable("Occlusion Curves", "no inlet-outlet flow", label, both)

    steps = FAST_CURVE_STEPS if statistics_mode == "fast" else FULL_CURVE_STEPS
    repeats = 3 if statistics_mode == "fast" else 5
    fractions = [round(max_fraction * i / steps, 6) for i in range(steps + 1)]
    vessels = list(base.edges)

    orders: Dict[str, Any] = {}
    shares = base.flow_fraction()
    orders["highest_current_flow_share"] = [vessels[i] for i in np.argsort(-shares, kind="stable")]
    diameters = []
    for u, v, key in vessels:
        diameters.append(_edge_data(G, u, v, key).get("diameter_um"))
    if all(_positive_finite(d) for d in diameters):
        orders["narrowest_diameter"] = [
            vessels[i] for i in np.argsort(np.asarray(diameters, dtype=float), kind="stable")
        ]

    result: Dict[str, Any] = {
        "Occlusion Curves Route Weighting": label,
        "Occlusion Curves Assumptions": ASSUMPTIONS,
    }
    curves: Dict[str, Any] = {}
    random_flows, random_reach = [], []
    for r in range(repeats):
        shuffled = list(vessels)
        random.Random(REPRODUCIBILITY_SEED + r).shuffle(shuffled)
        flows, reach = _curve(G, inlets, outlets, effective, shuffled, fractions, q0, base)
        random_flows.append(flows)
        random_reach.append(reach)
    curves["random"] = (
        np.mean(random_flows, axis=0).tolist(),
        np.mean(random_reach, axis=0).tolist(),
    )
    for name, order in orders.items():
        curves[name] = _curve(G, inlets, outlets, effective, order, fractions, q0, base)

    for name, (flows, reach) in curves.items():
        area = float(_trapezoid(flows, fractions) / max_fraction)
        result[f"Occlusion Curve ({name})"] = {
            "Robustness Index": area,
            "Removal Fraction At 50% Flow": _interpolated_crossing(fractions, flows, 0.5),
            "Curve [fraction removed, flow fraction, outlets reachable]": [
                [f, round(q, 4), round(c, 4)] for f, q, c in zip(fractions, flows, reach)
            ],
        }
    if "narrowest_diameter" not in orders:
        result["Occlusion Curve (narrowest_diameter)"] = "N/A (no diameter_um on every vessel)"
    if both:
        result["Occlusion Curves Nodes Both Inlet And Outlet (excluded)"] = both
    return result
