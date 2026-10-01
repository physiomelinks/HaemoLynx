"""Local discharge haematocrit from Pries & Secomb's bifurcation phase-separation law.

The rest of this package treats haematocrit as one scalar shared by every
edge (see ``viscosity.DEFAULT_HAEMATOCRIT`` and ``PoiseuilleModel.haematocrit``).
Real microvascular networks do not work that way: at every diverging
bifurcation, red blood cells partition disproportionately relative to plasma
(the phase-separation, or "plasma skimming", effect) -- the narrower or
lower-flow daughter gets a smaller share of the haematocrit than of the flow.
This module computes that per-edge, local ``discharge_haematocrit`` and
iterates it together with the flow solve to a fixed point, since the two are
mutually dependent: haematocrit sets viscosity, viscosity sets resistance,
resistance sets the flow split, and the flow split is what the phase-
separation law itself needs.

The law is Pries, Ley, Claassen & Gaehtgens (1989), with the parameters
Pries & Secomb (2005) refitted -- driven by the parent and daughter vessel
diameters and the fractional blood flow at the bifurcation, not by
bifurcation angle.

It is only defined for one parent dividing into two daughters. Every other
junction -- three or more outflows, or two or more inflows meeting two or
more outflows -- follows ``junction_rule`` (the ``haematocrit_junction_rule``
setting), one of :data:`JUNCTION_RULES`:

- ``no_separation``: mix the inflows and give every outflow that
  haematocrit, as Secomb's NetFlow code does at any node with more than
  three segments (``dishem.cpp``).
- ``sequential_bifurcations``: the junction is a chain of bifurcations,
  Secomb's generalised version (``dishem_generalized.cpp``).
- ``split_junctions``: the pipeline has already split every junction of
  four or more vessels into bifurcations
  (:func:`haemolynx.graph.split_high_degree_junctions`), so there is nothing
  left to decide; a junction the split could not reduce is mixed, as with
  ``no_separation``.

Every edge's own resistance/viscosity call already accepts a per-edge
``discharge_haematocrit`` override (see
:meth:`haemolynx.haemodynamics.poiseuille.PoiseuilleModel.calculate_viscosity`
and :func:`haemolynx.haemodynamics.constriction.apply_constriction_sites`),
so this module's only job is to decide, and write, that value onto every
edge -- and to drive the outer solve/distribute/recompute loop.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Callable

import networkx as nx
import numpy as np

from .resistance import (
    build_conductance_matrix_from_graph,
    set_edge_flows,
    solve_flow_from_conductance_matrix,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DISCHARGE_HAEMATOCRIT_ATTR",
    "JUNCTION_RULES",
    "JUNCTION_RULE_NO_SEPARATION",
    "JUNCTION_RULE_SEQUENTIAL_BIFURCATIONS",
    "JUNCTION_RULE_SPLIT_JUNCTIONS",
    "pries_secomb_daughter_haematocrit",
    "distribute_discharge_haematocrit",
    "iterate_flow_and_haematocrit",
]

#: Edge attribute this module writes the local discharge haematocrit to.
DISCHARGE_HAEMATOCRIT_ATTR = "discharge_haematocrit"

#: How a junction other than one parent dividing in two is treated (see this
#: module's docstring); the ``haematocrit_junction_rule`` setting's choices.
JUNCTION_RULE_NO_SEPARATION = "no_separation"
JUNCTION_RULE_SEQUENTIAL_BIFURCATIONS = "sequential_bifurcations"
JUNCTION_RULE_SPLIT_JUNCTIONS = "split_junctions"
JUNCTION_RULES: tuple[str, ...] = (
    JUNCTION_RULE_NO_SEPARATION,
    JUNCTION_RULE_SEQUENTIAL_BIFURCATIONS,
    JUNCTION_RULE_SPLIT_JUNCTIONS,
)

#: Below this |flow_signed|, an edge is treated as carrying no RBC transport
#: (direction is undefined and haematocrit cannot meaningfully propagate
#: through it) rather than as a genuine bifurcation branch.
_FLOW_EPSILON = 1e-30

#: Under-relaxation of each pass's haematocrit update in
#: :func:`iterate_flow_and_haematocrit`, Secomb's NetFlow schedule: the full
#: step, then this factor smaller every :data:`_RELAXATION_DECAY_EVERY` passes.
_RELAXATION_DECAY = 0.8
_RELAXATION_DECAY_EVERY = 5

#: Same upper bound the schema's own ``haematocrit`` setting uses -- keeps a
#: computed value safely inside what viscosity.py's Pries formulas accept
#: (they raise for haematocrit >= 1.0).
_MAX_HAEMATOCRIT = 0.99


def _fqe_pries_secomb(
    *,
    fractional_flow: float,
    parent_diameter_um: float,
    own_diameter_um: float,
    other_diameter_um: float,
    parent_haematocrit: float,
) -> float:
    """Fractional erythrocyte (RBC) flow into this daughter, in [0, 1].

    ``logit(FQE) = A + B * logit((FQB - X0) / (1 - 2*X0))``, with

    ``A = -13.29 * ((D_own/D_other)^2 - 1) / ((D_own/D_other)^2 + 1) * (1 - Hd) / D_parent``
    ``B = 1 + 6.98 * (1 - Hd) / D_parent``
    ``X0 = 0.964 * (1 - Hd) / D_parent``

    X0 is the plasma-skimming threshold below which a daughter gets no red
    cells at all. All three are Pries & Secomb's (2005) refit, diameters in
    um. The 1990 threshold ``X0 = 0.4 / D_parent`` belongs with the 1990 form
    of A and is lower (0.4 against 0.53 at Hd = 0.45), so it let red cells
    into low-flow daughters this law leaves empty.

    Pries, Ley, Claassen & Gaehtgens (1989); Pries & Secomb (2005). Not the
    caller-facing function -- see
    :func:`pries_secomb_daughter_haematocrit` for the actual haematocrit
    this converts to, and :func:`distribute_discharge_haematocrit` for how
    the two daughters of one bifurcation are kept exactly mass-conserving.
    """
    if parent_diameter_um <= 0 or own_diameter_um <= 0 or other_diameter_um <= 0:
        raise ValueError(
            "diameters must be positive, got "
            f"parent={parent_diameter_um}, own={own_diameter_um}, "
            f"other={other_diameter_um}."
        )
    if not 0.0 <= parent_haematocrit < 1.0:
        raise ValueError(
            f"parent_haematocrit is a fraction in [0, 1), got {parent_haematocrit}."
        )
    fqb = min(max(float(fractional_flow), 0.0), 1.0)
    # A very narrow parent can push X0 past 0.5; clamp so the valid
    # (x0, 1-x0) window never inverts.
    x0 = min(0.964 * (1.0 - parent_haematocrit) / float(parent_diameter_um), 0.49)
    if fqb <= x0:
        return 0.0
    if fqb >= 1.0 - x0:
        return 1.0

    diameter_ratio_sq = (float(own_diameter_um) / float(other_diameter_um)) ** 2
    a = (
        -13.29
        * (diameter_ratio_sq - 1.0)
        / (diameter_ratio_sq + 1.0)
        * (1.0 - parent_haematocrit)
        / float(parent_diameter_um)
    )
    b = 1.0 + 6.98 * (1.0 - parent_haematocrit) / float(parent_diameter_um)
    scaled = (fqb - x0) / (1.0 - 2.0 * x0)
    logit_scaled = math.log(scaled / (1.0 - scaled))
    logit_fqe = a + b * logit_scaled
    return 1.0 / (1.0 + math.exp(-logit_fqe))


def pries_secomb_daughter_haematocrit(
    *,
    fractional_flow: float,
    parent_diameter_um: float,
    own_diameter_um: float,
    other_diameter_um: float,
    parent_haematocrit: float,
) -> float:
    """This daughter's own discharge haematocrit at a diverging bifurcation.

    ``fractional_flow`` is this daughter's own share of the parent's blood
    flow (0-1); ``other_diameter_um`` is the *sister* daughter's diameter,
    not the parent's. Converts the phase-separation law's RBC-flow fraction
    to an actual discharge haematocrit by mass balance: a daughter carrying
    fraction ``FQE`` of the parent's red cells and fraction ``fractional_flow``
    of its blood flow has haematocrit ``parent_haematocrit * FQE / fractional_flow``.

    Returns 0.0 for a daughter with no flow (nothing to carry a haematocrit),
    clamped to :data:`_MAX_HAEMATOCRIT` at the top end -- viscosity.py's own
    Pries formulas reject haematocrit >= 1.0.
    """
    if fractional_flow <= 0:
        return 0.0
    fqe = _fqe_pries_secomb(
        fractional_flow=fractional_flow,
        parent_diameter_um=parent_diameter_um,
        own_diameter_um=own_diameter_um,
        other_diameter_um=other_diameter_um,
        parent_haematocrit=parent_haematocrit,
    )
    haematocrit = parent_haematocrit * fqe / float(fractional_flow)
    return min(max(haematocrit, 0.0), _MAX_HAEMATOCRIT)


def _edge_diameter(data: dict[str, Any]) -> float:
    diameter = data.get("diameter_um")
    if diameter is None or float(diameter) <= 0:
        raise ValueError(
            "distribute_discharge_haematocrit needs a positive 'diameter_um' "
            "on every edge -- run assign_diameters before this."
        )
    return float(diameter)


def _divide_as_a_chain_of_bifurcations(
    G: nx.MultiGraph,
    node_in: list[tuple],
    node_out: list[tuple],
    *,
    combined_h: float,
    combined_flow: float,
    haematocrit_attr: str,
) -> None:
    """Secomb's generalised rule (``dishem_generalized.cpp``): one junction as
    ``len(node_out) - 1`` bifurcations in a row.

    The first parent is the widest inflow, carrying the mixed inflow. Each
    step splits the next outflow off what is left, against the outflow after
    it: the law is evaluated with those two diameters, the outflow's share of
    the flow still left, and the haematocrit of what is left. The remainder
    then carries on with that second outflow's diameter as its parent, and
    the last outflow takes every red cell still left. Red cells are exactly
    conserved.

    Secomb takes the outflows in the order his network file lists them, which
    means nothing physically; here they go largest flow first (ties in graph
    order), so the answer does not depend on how the graph was built. With
    two outflows (two or more inflows meeting them) it is the bifurcation law
    once, with the widest inflow as the parent.
    """
    outflows = sorted(node_out, key=lambda ref: -ref[3])
    diameters = [_edge_diameter(G[ref[0]][ref[1]][ref[2]]) for ref in outflows]
    if node_in:
        parent_diameter = max(_edge_diameter(G[ref[0]][ref[1]][ref[2]]) for ref in node_in)
    else:
        # A source dividing at once has no parent segment to measure; the
        # outflows' flow-weighted diameter stands in, as for a bifurcation.
        parent_diameter = sum(d * ref[3] for d, ref in zip(diameters, outflows)) / combined_flow

    remaining_flow = combined_flow
    remaining_rbc = combined_h * combined_flow
    for index in range(len(outflows) - 1):
        u, v, key, flow, _up, _down = outflows[index]
        fqe = _fqe_pries_secomb(
            fractional_flow=flow / remaining_flow,
            parent_diameter_um=parent_diameter,
            own_diameter_um=diameters[index],
            other_diameter_um=diameters[index + 1],
            parent_haematocrit=min(remaining_rbc / remaining_flow, _MAX_HAEMATOCRIT),
        )
        G[u][v][key][haematocrit_attr] = min(fqe * remaining_rbc / flow, _MAX_HAEMATOCRIT)
        remaining_rbc *= 1.0 - fqe
        remaining_flow -= flow
        parent_diameter = diameters[index + 1]
    u, v, key, flow, _up, _down = outflows[-1]
    G[u][v][key][haematocrit_attr] = min(remaining_rbc / flow, _MAX_HAEMATOCRIT)


def distribute_discharge_haematocrit(
    G: nx.MultiGraph,
    *,
    inlet_haematocrit: float,
    haematocrit_attr: str = DISCHARGE_HAEMATOCRIT_ATTR,
    junction_rule: str = JUNCTION_RULE_NO_SEPARATION,
) -> dict[str, int]:
    """One topological-order pass writing ``discharge_haematocrit`` per edge.

    Orients every edge by the sign of its own ``flow_signed`` (written by
    :func:`haemolynx.haemodynamics.resistance.set_edge_flows`) -- positive
    means u->v. Node pressure is single-valued, so a directed cycle in that
    orientation is mathematically impossible (it would require pressure to
    strictly decrease all the way around back to its own starting value);
    the oriented graph is therefore always a DAG, and one topological-order
    (Kahn's algorithm) pass is sufficient and exact -- no relaxation needed.

    Per node, in topological order:

    - No (live) inflow: a true source, or the far end of a dead edge. Its
      outflow edges seed at ``inlet_haematocrit``.
    - One inflow, one outflow: pass-through, unchanged.
    - One inflow, exactly two outflow (the common bifurcation): the Pries-
      Secomb law, exactly mass-conserving between the two daughters (the
      second daughter's RBC fraction is ``1 - FQE`` of the first, not a
      second independent formula evaluation). A source dividing in two
      does the same, with the daughters' flow-weighted diameter standing in
      for the parent it has none of.
    - Two or more inflow, one outflow (a confluence): flow-weighted mass-
      balance mixing.
    - Anything else -- three or more outflow, or two or more inflow meeting
      two or more outflow -- is outside the law, and *junction_rule* decides
      (see this module's docstring): with ``no_separation`` or
      ``split_junctions`` every outflow gets the mixed inflow's haematocrit;
      with ``sequential_bifurcations``, see
      :func:`_divide_as_a_chain_of_bifurcations`.

    An edge with no or negligible ``flow_signed`` (missing, non-finite, or
    below a small epsilon) is excluded from the ordering constraint and
    assigned ``inlet_haematocrit`` directly -- it carries no RBC transport
    to propagate one way or the other.

    Writes ``haematocrit_attr`` onto every edge's own data dict in place.
    Returns a small diagnostic dict: ``dead_edges`` (no usable flow
    direction), ``compound_junctions`` (2+ in, 2+ out),
    ``non_bifurcation_junctions`` (every junction *junction_rule* decided),
    and ``unresolved_edges`` (should always be 0 -- a live edge the
    topological sort never reached, logged as a warning if it happens).
    """
    if junction_rule not in JUNCTION_RULES:
        raise ValueError(
            f"junction_rule must be one of {', '.join(JUNCTION_RULES)}, got {junction_rule!r}."
        )
    inlet_h = float(inlet_haematocrit)
    # Per node: live (non-dead) inflow/outflow edges, as
    # (u, v, key, abs_flow, upstream, downstream) tuples.
    in_edges: dict[Any, list[tuple]] = {}
    out_edges: dict[Any, list[tuple]] = {}
    dead_edges: list[tuple] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        flow = data.get("flow_signed")
        if flow is None or not np.isfinite(flow) or abs(float(flow)) <= _FLOW_EPSILON:
            data[haematocrit_attr] = inlet_h
            dead_edges.append((u, v, key))
            continue
        flow = float(flow)
        upstream, downstream = (u, v) if flow > 0 else (v, u)
        ref = (u, v, key, abs(flow), upstream, downstream)
        out_edges.setdefault(upstream, []).append(ref)
        in_edges.setdefault(downstream, []).append(ref)

    remaining_in = {node: len(in_edges.get(node, [])) for node in G.nodes()}
    queue: deque = deque(node for node, count in remaining_in.items() if count == 0)
    visited: set = set()
    compound_junctions = 0
    non_bifurcation_junctions = 0
    live_nodes = {node for node in G.nodes() if in_edges.get(node) or out_edges.get(node)}

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        node_in = in_edges.get(node, [])
        node_out = out_edges.get(node, [])

        if node_out:
            if not node_in:
                combined_h = inlet_h
                combined_flow = sum(ref[3] for ref in node_out)
            else:
                combined_flow = sum(ref[3] for ref in node_in)
                if combined_flow <= 0:
                    combined_h = inlet_h
                else:
                    combined_h = sum(
                        G[ref[0]][ref[1]][ref[2]].get(haematocrit_attr, inlet_h) * ref[3]
                        for ref in node_in
                    ) / combined_flow
                if len(node_in) > 1 and len(node_out) > 1:
                    compound_junctions += 1
            divides_outside_the_law = len(node_out) > 2 or (
                len(node_out) > 1 and len(node_in) > 1
            )
            if divides_outside_the_law:
                non_bifurcation_junctions += 1

            if len(node_out) == 1:
                u, v, key, _flow, _up, _down = node_out[0]
                G[u][v][key][haematocrit_attr] = combined_h
            elif divides_outside_the_law:
                divided_flow = sum(ref[3] for ref in node_out)
                if junction_rule == JUNCTION_RULE_SEQUENTIAL_BIFURCATIONS and divided_flow > 0:
                    # The outflows' own total, not the inflows': at a node
                    # holding a boundary pressure the two differ.
                    _divide_as_a_chain_of_bifurcations(
                        G,
                        node_in,
                        node_out,
                        combined_h=combined_h,
                        combined_flow=divided_flow,
                        haematocrit_attr=haematocrit_attr,
                    )
                else:
                    # No phase separation (Secomb's dishem.cpp): every outflow
                    # gets the mixed inflow's haematocrit.
                    for u, v, key, _flow, _up, _down in node_out:
                        G[u][v][key][haematocrit_attr] = combined_h
            elif combined_flow > 0:
                (u_a, v_a, key_a, flow_a, _, _), (u_b, v_b, key_b, flow_b, _, _) = node_out
                fqb_a = flow_a / combined_flow
                fqb_b = flow_b / combined_flow
                d_a = _edge_diameter(G[u_a][v_a][key_a])
                d_b = _edge_diameter(G[u_b][v_b][key_b])
                if node_in:
                    parent_diameter = _edge_diameter(G[node_in[0][0]][node_in[0][1]][node_in[0][2]])
                else:
                    # A source bifurcating immediately, with no feeding
                    # parent segment to measure -- fall back to the two
                    # daughters' own flow-weighted diameter as the best
                    # available stand-in for the missing parent.
                    parent_diameter = d_a * fqb_a + d_b * fqb_b
                fqe_a = _fqe_pries_secomb(
                    fractional_flow=fqb_a,
                    parent_diameter_um=parent_diameter,
                    own_diameter_um=d_a,
                    other_diameter_um=d_b,
                    parent_haematocrit=combined_h,
                )
                fqe_b = 1.0 - fqe_a  # exact mass conservation, not a second formula call.
                h_a = min(max(combined_h * fqe_a / fqb_a, 0.0), _MAX_HAEMATOCRIT) if fqb_a > 0 else 0.0
                h_b = min(max(combined_h * fqe_b / fqb_b, 0.0), _MAX_HAEMATOCRIT) if fqb_b > 0 else 0.0
                G[u_a][v_a][key_a][haematocrit_attr] = h_a
                G[u_b][v_b][key_b][haematocrit_attr] = h_b
            else:
                # A degenerate zero-flow bifurcation: nothing to divide.
                for u, v, key, _flow, _up, _down in node_out:
                    G[u][v][key][haematocrit_attr] = combined_h

        for ref in node_out:
            downstream = ref[5]
            remaining_in[downstream] -= 1
            if remaining_in[downstream] == 0:
                queue.append(downstream)

    unresolved = live_nodes - visited
    if unresolved:
        # Should not happen (see the docstring's acyclicity argument) --
        # left with the fallback default rather than raised, so one
        # unexpected node cannot abort an otherwise-good distribution.
        logger.warning(
            "distribute_discharge_haematocrit: %d node(s) never reached by "
            "the topological pass (expected 0); their edges keep whatever "
            "haematocrit they already had.",
            len(unresolved),
        )

    return {
        "dead_edges": len(dead_edges),
        "compound_junctions": compound_junctions,
        "non_bifurcation_junctions": non_bifurcation_junctions,
        "unresolved_edges": len(unresolved),
    }


def iterate_flow_and_haematocrit(
    G: nx.MultiGraph,
    *,
    recompute_resistances: Callable[[], None],
    inlet_haematocrit: float,
    inlet_p_bc: float,
    outlet_p_bc: float,
    inlet_nodes: list,
    outlet_nodes: list,
    max_iterations: int,
    tolerance: float,
    junction_rule: str = JUNCTION_RULE_NO_SEPARATION,
) -> dict[str, Any]:
    """Solve flow and distribute haematocrit together, to a fixed point.

    Each pass: solve flow from *G*'s current resistances -> distribute
    discharge haematocrit over that flow field -> compare against the
    previous pass's values and move each edge part of the way there -> call
    *recompute_resistances* (expected to re-derive every edge's resistance
    from its diameter and its now-updated ``discharge_haematocrit``, e.g.
    :func:`haemolynx.haemodynamics.apply.apply_poiseuille_resistances`) --
    then, if the largest per-edge haematocrit change was within *tolerance*,
    stop; otherwise solve again.

    The step is under-relaxed as Secomb's NetFlow does it (``flow.cpp``): the
    whole change at first, then 0.8 times less every five passes. Phase
    separation feeds back on itself -- a daughter given more red cells gets
    more viscous, draws less flow and so fewer red cells -- and a daughter
    near the skimming threshold swings between too many and too few on
    alternate passes, a swing the full step shrinks only a few percent a pass.
    Relaxing moves where the loop stops, not the fixed point it stops at.
    Convergence is judged on the unrelaxed change, as NetFlow judges it.

    *recompute_resistances* runs on every pass, including the one that
    converges: ``resistance``/``conductance`` on *G* always end up derived
    from the ``discharge_haematocrit`` this function is reporting, not from
    one pass behind it. The unavoidable residual is on the other side of
    that recompute instead -- the *flow* this returns was solved from the
    previous pass's resistance, so it is very slightly stale relative to
    the resistance now on the graph, bounded by the same *tolerance* rather
    than a full haematocrit update.

    *G* must already carry an initial resistance/conductance on every edge
    (the uniform-haematocrit baseline) before the first call -- this
    function only ever updates it, never creates it from nothing.
    *junction_rule* is handed to every
    :func:`distribute_discharge_haematocrit` pass.

    Returns a dict with ``pressure``, ``node_list`` (matching
    :func:`~haemolynx.haemodynamics.resistance.solve_flow_from_conductance_matrix`'s
    own return shape), plus ``converged`` (bool), ``iterations`` (how many
    solves ran), ``max_delta`` (largest unrelaxed per-edge haematocrit change
    on the final pass), and the last :func:`distribute_discharge_haematocrit`
    diagnostic's ``dead_edges``/``compound_junctions``/
    ``non_bifurcation_junctions``/``unresolved_edges``.
    """
    max_iterations = max(1, int(max_iterations))
    tolerance = float(tolerance)
    previous: dict[tuple, float] | None = None
    relaxation = 1.0
    converged = False
    max_delta = float("inf")
    diagnostics: dict[str, int] = {}
    flow: dict[str, Any] = {}

    # G's node/edge set is fixed for the rest of this function -- every pass
    # below only rewrites attribute values (discharge_haematocrit, then
    # resistance/conductance), never adds or removes a node or edge -- so the
    # node ordering and the conductance array itself are built once and
    # reused, instead of paying the O(n^2) zero-fill and the Python-level
    # node_to_idx rebuild on every outer iteration.
    node_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    conductance_matrix = np.zeros((len(node_list), len(node_list)), dtype=float)

    for iteration in range(1, max_iterations + 1):
        conductance, node_list = build_conductance_matrix_from_graph(
            G, node_list=node_list, node_to_idx=node_to_idx, out=conductance_matrix,
        )
        flow = solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            inlet_p_bc=inlet_p_bc,
            outlet_p_bc=outlet_p_bc,
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
            node_to_idx=node_to_idx,
        )
        set_edge_flows(G, node_list, flow["pressure"], node_to_idx=node_to_idx)
        diagnostics = distribute_discharge_haematocrit(
            G, inlet_haematocrit=inlet_haematocrit, junction_rule=junction_rule,
        )

        current = {
            (u, v, key): data.get(DISCHARGE_HAEMATOCRIT_ATTR, inlet_haematocrit)
            for u, v, key, data in G.edges(keys=True, data=True)
        }
        has_converged = False
        if previous is not None:
            if iteration % _RELAXATION_DECAY_EVERY == 0:
                relaxation *= _RELAXATION_DECAY
            max_delta = 0.0
            for edge, distributed in current.items():
                before = previous.get(edge, inlet_haematocrit)
                max_delta = max(max_delta, abs(distributed - before))
                current[edge] = before + relaxation * (distributed - before)
                G.edges[edge][DISCHARGE_HAEMATOCRIT_ATTR] = current[edge]
            logger.info(
                "[haematocrit-distribution] iteration %d/%d: max change %.4g "
                "(tolerance %.4g, relaxation %.3g)",
                iteration, max_iterations, max_delta, tolerance, relaxation,
            )
            has_converged = max_delta <= tolerance
        previous = current

        # Always recompute -- resistance/conductance on G must reflect the
        # discharge_haematocrit just distributed, on the converging (or
        # final) pass as much as any other. See this function's own
        # docstring for what stays very slightly stale instead.
        recompute_resistances()

        if has_converged:
            converged = True
            break

    if not converged and max_iterations > 1:
        logger.warning(
            "[haematocrit-distribution] did not converge within %d "
            "iteration(s); largest remaining per-edge change was %.4g "
            "(tolerance %.4g). Consider raising "
            "haematocrit_distribution_max_iterations.",
            max_iterations, max_delta, tolerance,
        )

    return {
        "pressure": flow["pressure"],
        "node_list": node_list,
        # Node order never changes across iterations (see this function's
        # own reuse comment above), so a caller that rebuilds the
        # conductance matrix again afterwards -- e.g. because resistance
        # changed again after the loop's own last build -- can reuse these
        # instead of paying the node_to_idx rebuild and the (n, n)
        # allocation a second time.
        "node_to_idx": node_to_idx,
        "conductance_matrix": conductance_matrix,
        "converged": converged,
        "iterations": iteration,
        "max_delta": 0.0 if max_delta == float("inf") else max_delta,
        **diagnostics,
    }
