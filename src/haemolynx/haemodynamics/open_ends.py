"""Boundary conditions where vessels leave the image: the flow's open ends.

A vessel the image cut through (a dead end of kind ``leaves_image``,
:mod:`haemolynx.graph.dead_ends`) runs on in the tissue. Left as it is, it is a
closed dead end: no blood leaves through it, and every vessel that only leads
to such ends carries no flow. ``open_end_model`` chooses what happens instead
(PLAN_dead_ends.md, Phase 4):

- ``closed`` -- nothing: no flow through any open end (the default, as before).
- ``terminal_resistance`` -- each open end drains through a resistance to the
  outlet pressure: *k* times the resistance of its own vessel (the chain from
  the open end to its junction), *k* being
  ``open_end_terminal_resistance_multiple``. One parameter; no published use
  on cut capillaries was found (the nearest is Blinder et al. 2013's
  resistance on cut surface arterioles).
- ``outlet`` -- every open end held at the outlet pressure (a "universal
  sink"). Simple, and biased towards draining; kept for comparison.
- ``estimated`` -- Fry, Lee, Smith & Secomb (2012, *Microcirculation*
  19:530-538): the open ends' pressures are unknowns, chosen so the
  network's pressures and wall shear stresses stay closest to target values,
  with flow conserved at every other node and the designated inlets and
  outlets held where they are; flow directions are settled by iterating.
  The targets default to the run's own closed solve (its mean pressure and
  median wall shear stress over the vessels it perfuses).

Open ends are only given a condition in a piece of the network that holds a
designated inlet or outlet: they extend the perfused network, they never
create flow in a piece with no inlet or outlet.

**How every solver takes them.** Whatever the model, the conditions become
conductances from each open end to one of two *ghost* nodes added to the
conductance matrix -- one held at the inlet pressure, one at the outlet
pressure -- plus, for ``outlet``, open ends added to the outlets. The
estimated pressures become, exactly, the conductance that carries each open
end's estimated flow at its estimated pressure, so the baseline solve
reproduces the estimate and a perturbation or a sweep keeps the same
conditions (Dirichlet values would not scale with a sweep's pressure drop).
The ghosts are never added to the graph, so nothing that draws, measures or
exports the network sees them; :func:`augment` appends them to the matrix
and the node list, and every caller trims them off again.

The conditions a run used are kept on the graph
(``G.graph[GRAPH_RECORD]``, :meth:`OpenEndConditions.to_record`), so its
perturbations and sweeps solve with the baseline's, and a saved run says
which model its resistances came from.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

CLOSED = "closed"
TERMINAL_RESISTANCE = "terminal_resistance"
OUTLET = "outlet"
ESTIMATED = "estimated"
OPEN_END_MODELS = (CLOSED, TERMINAL_RESISTANCE, OUTLET, ESTIMATED)

#: Where a run's conditions are kept on its graph.
GRAPH_RECORD = "open_end_conditions"
#: The two ghost nodes :func:`augment` appends.
INLET_GHOST = "__open_end_inlet_level__"
OUTLET_GHOST = "__open_end_outlet_level__"

DEFAULT_TERMINAL_RESISTANCE_MULTIPLE = 10.0
DEFAULT_SHEAR_WEIGHT = 1.0
MAX_SIGN_ITERATIONS = 20
#: A vessel whose wall shear stress is below this share of the target carries
#: almost no flow: its direction is left free (no shear target), and its
#: flipping does not keep the estimate from settling. On E14.5 every vessel
#: still flipping after five passes was one of these.
SETTLED_SHEAR_FRACTION = 0.01


@dataclass
class OpenEndConditions:
    """What each open end is held by. Conductances in m^3/(Pa.s)."""

    model: str = CLOSED
    #: Open end -> conductance to the inlet pressure level.
    to_inlet: dict = field(default_factory=dict)
    #: Open end -> conductance to the outlet pressure level.
    to_outlet: dict = field(default_factory=dict)
    #: Open ends held at the outlet pressure itself (the ``outlet`` model).
    as_outlets: tuple = ()
    #: How the conditions were found (targets, iterations, ...).
    info: dict = field(default_factory=dict)

    @property
    def nodes(self) -> list:
        """Every open end given a condition, in a stable order."""
        return list(dict.fromkeys([*self.to_inlet, *self.to_outlet, *self.as_outlets]))

    def __bool__(self) -> bool:
        return bool(self.to_inlet or self.to_outlet or self.as_outlets)

    def to_record(self) -> dict:
        """A plain dict to keep on the graph (and in a saved run)."""
        return {
            "model": self.model,
            "to_inlet": [[node, float(g)] for node, g in self.to_inlet.items()],
            "to_outlet": [[node, float(g)] for node, g in self.to_outlet.items()],
            "as_outlets": list(self.as_outlets),
            "info": dict(self.info),
        }

    @classmethod
    def from_record(cls, record: Optional[Mapping]) -> "OpenEndConditions":
        if not record:
            return cls()

        def node(value):
            return tuple(value) if isinstance(value, list) else value

        return cls(
            model=str(record.get("model", CLOSED)),
            to_inlet={node(n): float(g) for n, g in record.get("to_inlet", [])},
            to_outlet={node(n): float(g) for n, g in record.get("to_outlet", [])},
            as_outlets=tuple(node(n) for n in record.get("as_outlets", [])),
            info=dict(record.get("info", {})),
        )

    def restricted_to(self, nodes: Iterable[Any]) -> "OpenEndConditions":
        """The same conditions on those of *nodes* that are still there (a
        perturbation that deleted vessels)."""
        keep = set(nodes)
        return OpenEndConditions(
            model=self.model,
            to_inlet={n: g for n, g in self.to_inlet.items() if n in keep},
            to_outlet={n: g for n, g in self.to_outlet.items() if n in keep},
            as_outlets=tuple(n for n in self.as_outlets if n in keep),
            info=self.info,
        )


def conditions_on_graph(G: nx.Graph) -> OpenEndConditions:
    """The conditions a run's solve left on *G* (closed when none), on the
    nodes *G* still has."""
    return OpenEndConditions.from_record(G.graph.get(GRAPH_RECORD)).restricted_to(G.nodes)


def keep_on_graph(G: nx.Graph, conditions: OpenEndConditions) -> None:
    G.graph[GRAPH_RECORD] = conditions.to_record()


# --- which ends, and the cheap models ---------------------------------------------------


def open_end_candidates(G: nx.Graph, inlet_nodes: Sequence[Any], outlet_nodes: Sequence[Any]) -> list[Any]:
    """The dead ends leaving the image (``graph.dead_ends.open_end_nodes``)
    that are not inlets or outlets, in a piece of the network holding a
    designated inlet or outlet: what a model may open."""
    from haemolynx.graph.dead_ends import open_end_nodes

    designated = {n for n in (*inlet_nodes, *outlet_nodes) if n in G}
    if not designated:
        return []
    driven: set = set()
    for piece in nx.connected_components(G):
        if piece & designated:
            driven |= piece
    return [n for n in open_end_nodes(G) if n in driven and n not in designated]


def _chain_resistance(G: nx.MultiGraph, tip: Any) -> Optional[float]:
    from haemolynx.graph.dead_ends import chain_to_junction
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    _nodes, edges = chain_to_junction(G, tip)
    total = 0.0
    for a, b, key in edges:
        data = G.edges[a, b, key] if G.is_multigraph() else G.edges[a, b]
        if data.get(IS_ZERO_RESISTANCE):
            continue
        resistance = data.get("resistance")
        if resistance is not None and np.isfinite(resistance) and resistance > 0:
            total += float(resistance)
    return total if total > 0 else None


def terminal_resistance_conditions(
    G: nx.MultiGraph, ends: Sequence[Any], *, multiple: float = DEFAULT_TERMINAL_RESISTANCE_MULTIPLE
) -> OpenEndConditions:
    """Each open end drains to the outlet pressure through *multiple* times its
    own vessel's resistance; an end whose vessel has no resistance takes the
    median of the others'."""
    if multiple <= 0:
        raise ValueError(f"open_end_terminal_resistance_multiple must be positive, got {multiple}")
    own = {n: _chain_resistance(G, n) for n in ends}
    known = [r for r in own.values() if r is not None]
    fallback = float(np.median(known)) if known else None
    to_outlet = {}
    for n, r in own.items():
        r = r if r is not None else fallback
        if r is not None:
            to_outlet[n] = 1.0 / (float(multiple) * r)
    return OpenEndConditions(
        model=TERMINAL_RESISTANCE, to_outlet=to_outlet,
        info={"multiple": float(multiple), "ends": len(to_outlet),
              "without_own_resistance": sum(1 for r in own.values() if r is None)},
    )


def outlet_conditions(ends: Sequence[Any]) -> OpenEndConditions:
    return OpenEndConditions(model=OUTLET, as_outlets=tuple(ends), info={"ends": len(ends)})


# --- the matrix every solver takes ---------------------------------------------------------


def augment(conductance, node_list: Sequence[Any], conditions: Optional[OpenEndConditions], *, node_to_idx=None):
    """``(conductance, node_list, node_to_idx, extra_inlets, extra_outlets)``
    with *conditions* added: the two ghost nodes appended to the matrix (dense
    or sparse, as given) and the node list, joined to the open ends by their
    conductances, and the inlets / outlets a solver must add -- the inlet
    ghost, the outlet ghost and any open end held at the outlet pressure.
    Without conditions, everything comes back as it was."""
    node_list = list(node_list)
    if node_to_idx is None:
        node_to_idx = {n: i for i, n in enumerate(node_list)}
    if not conditions:
        return conductance, node_list, node_to_idx, [], []
    n = len(node_list)
    i_in, i_out = n, n + 1
    rows, cols, vals = [], [], []
    for links, ghost in ((conditions.to_inlet, i_in), (conditions.to_outlet, i_out)):
        for node, g in links.items():
            idx = node_to_idx.get(node)
            if idx is None or not g > 0:
                continue
            rows += [idx, ghost]
            cols += [ghost, idx]
            vals += [float(g), float(g)]
    extra = sp.coo_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))
    if sp.issparse(conductance):
        base = sp.coo_matrix(conductance)
        grown = sp.coo_matrix((base.data, (base.row, base.col)), shape=(n + 2, n + 2))
        augmented = (grown + extra).tocsr()
    else:
        augmented = np.zeros((n + 2, n + 2), dtype=float)
        augmented[:n, :n] = conductance
        augmented += extra.toarray()
    names = node_list + [INLET_GHOST, OUTLET_GHOST]
    index = dict(node_to_idx)
    index[INLET_GHOST], index[OUTLET_GHOST] = i_in, i_out
    extra_inlets = [INLET_GHOST] if conditions.to_inlet else []
    extra_outlets = ([OUTLET_GHOST] if conditions.to_outlet else []) + [
        node for node in conditions.as_outlets if node in node_to_idx
    ]
    return augmented, names, index, extra_inlets, extra_outlets


def open_end_flows(
    conductance, pressure: np.ndarray, node_to_idx: Mapping[Any, int], conditions: Optional[OpenEndConditions],
    *, inlet_p_bc: float, outlet_p_bc: float,
) -> dict[str, float]:
    """The flow (m^3/s) into the network through open ends and out through
    them, from a solve of the graph's own *conductance* (no ghosts) and its
    *pressure*."""
    from .resistance import boundary_flow

    if not conditions:
        return {"in": 0.0, "out": 0.0}
    into = out = 0.0
    for node, g in conditions.to_inlet.items():
        if node in node_to_idx:
            q = g * (float(inlet_p_bc) - float(pressure[node_to_idx[node]]))
            into, out = (into + q, out) if q > 0 else (into, out - q)
    for node, g in conditions.to_outlet.items():
        if node in node_to_idx:
            q = g * (float(pressure[node_to_idx[node]]) - float(outlet_p_bc))
            into, out = (into - q, out) if q < 0 else (into, out + q)
    held = [node_to_idx[n] for n in conditions.as_outlets if n in node_to_idx]
    for idx in held:
        q = boundary_flow(conductance, pressure, [idx])  # out of the node into the network
        into, out = (into + q, out) if q > 0 else (into, out - q)
    return {"in": float(into), "out": float(out)}


#: :func:`open_end_directions`' answers: blood enters the network through the
#: open end (a source), or leaves through it (a sink).
FEEDS = "in"
DRAINS = "out"


def open_end_directions(G: nx.Graph) -> dict[Any, str]:
    """Each open end the solve gave a condition (``G.graph[GRAPH_RECORD]``)
    and which way its solved flow runs: :data:`FEEDS` (blood enters the
    network through it) or :data:`DRAINS`, read off the node pressures and
    the flows of its vessels. An open end whose flow is at or below the
    statistics' carrying floor (``CARRYING_FLOOR_FRACTION`` of the network's
    largest flow), or a graph not solved, gives none. Routes, transit time,
    territories, current flow and occlusion take these as sources and sinks
    beside the designated inlets and outlets, and the VTK names them."""
    from haemolynx.statistics.flux_heterogeneity import CARRYING_FLOOR_FRACTION

    conditions = OpenEndConditions.from_record(G.graph.get(GRAPH_RECORD))
    if not conditions:
        return {}
    flows = [abs(float(d.get("flow_abs"))) for *_e, d in G.edges(data=True) if d.get("flow_abs") is not None]
    finite = [q for q in flows if np.isfinite(q)]
    if not finite:
        return {}
    floor = CARRYING_FLOOR_FRACTION * max(finite)
    found: dict[Any, str] = {}
    for node in conditions.nodes:
        if node not in G or G.nodes[node].get("pressure") is None:
            continue
        p_node = float(G.nodes[node]["pressure"])
        out_of_node = 0.0
        for _n, other, data in G.edges(node, data=True):
            q = data.get("flow_abs")
            p_other = G.nodes[other].get("pressure")
            if other == node or q is None or p_other is None or not np.isfinite(float(q)):
                continue
            out_of_node += float(np.sign(p_node - float(p_other))) * abs(float(q))
        if out_of_node > floor:
            found[node] = FEEDS
        elif out_of_node < -floor:
            found[node] = DRAINS
    return found


# --- the estimate (Fry et al. 2012) ---------------------------------------------------------


def _edge_terms(G: nx.MultiGraph, node_to_idx: Mapping[Any, int], domain: np.ndarray):
    """``(u_idx, v_idx, shear_per_pa, length)`` of every vessel inside the
    domain that has a diameter, a length and a resistance (bridges left out):
    its wall shear stress is ``shear_per_pa * (p_u - p_v)``, d / (4 L)."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    us, vs, cs, ls = [], [], [], []
    for u, v, data in G.edges(data=True):
        if u == v or data.get(IS_ZERO_RESISTANCE):
            continue
        iu, iv = node_to_idx.get(u), node_to_idx.get(v)
        if iu is None or iv is None or not (domain[iu] and domain[iv]):
            continue
        d, length = data.get("diameter_um"), data.get("length")
        if not d or not length or not data.get("conductance"):
            continue
        us.append(iu), vs.append(iv), cs.append(float(d) / (4.0 * float(length))), ls.append(float(length))
    return np.asarray(us, dtype=int), np.asarray(vs, dtype=int), np.asarray(cs), np.asarray(ls)


def estimate_open_end_conditions(
    G: nx.MultiGraph,
    conductance,
    node_list: Sequence[Any],
    *,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
    inlet_p_bc: float,
    outlet_p_bc: float,
    ends: Sequence[Any],
    pressure_target_pa: Optional[float] = None,
    shear_target_pa: Optional[float] = None,
    shear_weight: float = DEFAULT_SHEAR_WEIGHT,
    max_iterations: int = MAX_SIGN_ITERATIONS,
) -> OpenEndConditions:
    """The ``estimated`` model: the open ends' pressures chosen as Fry et al.
    (2012) choose unknown boundary conditions, then turned into the
    conductances that reproduce them (see the module docstring).

    Minimises, over every pressure not held by a designated inlet or outlet,

        wp * sum_i a_i (p_i - p0)^2  +  wt * sum_e L_e (tau_e - s_e tau0)^2

    with flow conserved at every node but the open ends; ``a_i`` is half the
    length of the vessels at node *i*, ``tau_e = d_e (p_u - p_v) / (4 L_e)``
    the wall shear stress of vessel *e* (diameter and length in the same
    units, so Pa), ``s_e`` its flow direction, settled by iterating. The two
    terms are each divided by their own scale -- ``wp = 1 / (sum a * dp^2)``
    with *dp* the inlet-outlet drop, ``wt = shear_weight / (sum L * tau0^2)``
    -- so *shear_weight* is how much the shear target counts against the
    pressure target. *p0* and *tau0* default to the closed solve's
    length-weighted mean pressure and median wall shear stress over the
    vessels it perfuses.
    """
    from scipy.sparse.linalg import spsolve

    from .resistance import calc_laplacian_from_conductance_matrix, reachable_through_conductances

    node_list = list(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    C = sp.csr_matrix(conductance) if not sp.issparse(conductance) else conductance.tocsr()
    L = sp.csr_matrix(calc_laplacian_from_conductance_matrix(C))
    p_in, p_out = float(inlet_p_bc), float(outlet_p_bc)
    known = {}
    for node in inlet_nodes:
        if node in node_to_idx:
            known[node_to_idx[node]] = p_in
    for node in outlet_nodes:
        if node in node_to_idx:
            known[node_to_idx[node]] = p_out
    K = np.array(sorted(known), dtype=int)
    p_K = np.array([known[i] for i in K], dtype=float)
    open_idx = np.array(sorted({node_to_idx[e] for e in ends if e in node_to_idx} - set(K.tolist())), dtype=int)
    if not len(K) or not len(open_idx):
        return OpenEndConditions(model=ESTIMATED, info={"ends": 0})
    domain = reachable_through_conductances(C, K)
    open_idx = open_idx[domain[open_idx]]
    if not len(open_idx):
        return OpenEndConditions(model=ESTIMATED, info={"ends": 0})
    F = np.flatnonzero(domain & ~np.isin(np.arange(n), K))
    is_open = np.isin(F, open_idx)
    interior = F[~is_open]

    # The closed solve: the targets, and the first flow directions.
    from .resistance import solve_reduced_laplacian_sparse

    closed = np.zeros(n)
    closed[K] = p_K
    if len(F):
        closed[F] = solve_reduced_laplacian_sparse(L, closed, F, K)
    us, vs, cs, ls = _edge_terms(G, node_to_idx, domain)
    tau_closed = cs * (closed[us] - closed[vs])
    flowing = np.abs(tau_closed) > 1e-12 * max(1.0, float(np.max(np.abs(tau_closed))) if len(tau_closed) else 1.0)
    node_weight = np.zeros(n)
    np.add.at(node_weight, us, ls / 2.0)
    np.add.at(node_weight, vs, ls / 2.0)
    if pressure_target_pa is None:
        nodes_flowing = np.unique(np.concatenate([us[flowing], vs[flowing]])) if flowing.any() else F
        weights = node_weight[nodes_flowing]
        pressure_target_pa = (
            float(np.average(closed[nodes_flowing], weights=weights)) if weights.sum() > 0 else 0.5 * (p_in + p_out)
        )
    if shear_target_pa is None:
        if flowing.any():
            order = np.argsort(np.abs(tau_closed[flowing]))
            cum = np.cumsum(ls[flowing][order])
            shear_target_pa = float(np.abs(tau_closed[flowing])[order][int(np.searchsorted(cum, cum[-1] / 2.0))])
        else:
            shear_target_pa = 0.0
    if not shear_target_pa or shear_target_pa <= 0:
        logger.warning("open_end_model=estimated: the closed solve perfuses no vessel to take a shear target "
                       "from, so the open ends stay closed.")
        return OpenEndConditions(model=ESTIMATED, info={"ends": 0, "reason": "no shear target"})

    drop = abs(p_in - p_out) or abs(pressure_target_pa) or 1.0
    wp = 1.0 / (max(node_weight[F].sum(), 1e-12) * drop ** 2)
    wt = float(shear_weight) / (max(ls.sum(), 1e-12) * shear_target_pa ** 2)

    # B maps pressures to shear stresses; split into the free and held columns.
    m = len(us)
    B = sp.csr_matrix(
        (np.concatenate([cs, -cs]), (np.concatenate([np.arange(m), np.arange(m)]), np.concatenate([us, vs]))),
        shape=(m, n),
    )
    B_F, B_K = B[:, F], B[:, K]
    W = sp.diags(ls)
    H = 2.0 * (wp * sp.diags(node_weight[F]) + wt * (B_F.T @ W @ B_F))
    A = L[interior][:, F]
    b = -np.asarray(L[interior][:, K] @ p_K).ravel()
    zero = sp.csr_matrix((A.shape[0], A.shape[0]))
    KKT = sp.bmat([[H, A.T], [A, zero]], format="csc")
    reach = _vessels_on_boundary_paths(G, node_to_idx, set(K.tolist()) | set(open_idx.tolist()), us, vs)
    signs = np.where(reach & flowing, np.sign(tau_closed), 0.0)
    pressure = closed.copy()
    converged = False
    iterations = 0
    flips = np.zeros(len(us), dtype=int)
    starts = np.zeros(len(us), dtype=int)
    for iterations in range(1, int(max_iterations) + 1):
        g = 2.0 * (wp * node_weight[F] * pressure_target_pa
                   + wt * np.asarray(B_F.T @ (W @ (signs * shear_target_pa - B_K @ p_K))).ravel())
        x = spsolve(KKT, np.concatenate([g, b]))
        previous = pressure[F].copy()
        pressure[F] = x[: len(F)]
        tau = cs * (pressure[us] - pressure[vs])
        active = reach & (np.abs(tau) > SETTLED_SHEAR_FRACTION * shear_target_pa)
        new_signs = np.where(active, np.sign(tau), 0.0)
        # Directions must settle where vessels carry flow; a near-still one
        # may flip either way without moving anything else. A vessel the
        # last pass set flowing (one the closed solve left still, behind an
        # open end) takes its shear target from the next. One whose direction
        # keeps flipping, or that keeps stopping and starting, is one the
        # target pushes against the rest of the network: from its second
        # flip or start on, its direction is left free (no shear target), so
        # the iteration always settles.
        flips += (new_signs != signs) & active & (signs != 0)
        starts += (new_signs != 0) & (signs == 0) & active
        new_signs[(flips >= 2) | (starts >= 2)] = 0.0
        changed = int(np.count_nonzero((new_signs != signs) & active))
        moved = float(np.max(np.abs(pressure[F] - previous))) if len(F) else 0.0
        signs = new_signs
        if changed == 0 or (iterations > 1 and moved <= 1e-4 * drop):
            converged = True
            break
    if not converged:
        logger.warning("open_end_model=estimated: flow directions still changing after %d passes; "
                       "using the last.", iterations)

    # Each open end's estimated pressure, as the conductance carrying its
    # estimated flow there: to the inlet level where it feeds the network,
    # to the outlet level where it drains it.
    flows_out_of_end = np.asarray(C[open_idx] @ np.ones(n)).ravel() * pressure[open_idx] - np.asarray(C[open_idx] @ pressure).ravel()
    to_inlet, to_outlet = {}, {}
    clamped = 0
    for k, q in zip(open_idx, flows_out_of_end):
        p_k = float(pressure[k])
        node = node_list[int(k)]
        if q > 0:  # feeds the network
            gap = p_in - p_k
            if gap <= 1e-9 * drop:
                gap, clamped = 1e-9 * drop, clamped + 1
            to_inlet[node] = float(q) / gap
        elif q < 0:  # drains it
            gap = p_k - p_out
            if gap <= 1e-9 * drop:
                gap, clamped = 1e-9 * drop, clamped + 1
            to_outlet[node] = float(-q) / gap
    info = {
        "ends": int(len(open_idx)),
        "feeding": len(to_inlet),
        "draining": len(to_outlet),
        "pressure_target_pa": float(pressure_target_pa),
        "shear_target_pa": float(shear_target_pa),
        "shear_weight": float(shear_weight),
        "iterations": int(iterations),
        "converged": bool(converged),
        "directions_left_free": int(np.count_nonzero((flips >= 2) | (starts >= 2))),
        "clamped": int(clamped),
        "pressure_range_pa": [float(pressure[open_idx].min()), float(pressure[open_idx].max())],
    }
    if clamped:
        logger.warning("open_end_model=estimated: %d open end(s) estimated at or beyond the inlet or "
                       "outlet pressure, held just inside it.", clamped)
    return OpenEndConditions(model=ESTIMATED, to_inlet=to_inlet, to_outlet=to_outlet, info=info)


def _vessels_on_boundary_paths(G, node_to_idx, boundary_idx: set, us, vs) -> np.ndarray:
    """Which of the vessels ``(us, vs)`` some path between two different
    boundary nodes runs along: the rest carry no flow whatever the
    pressures, and get no shear target."""
    index_to_node = {i: n for n, i in node_to_idx.items()}
    boundary = {index_to_node[i] for i in boundary_idx if i in index_to_node}
    V = object()
    H = nx.Graph()
    H.add_edges_from((a, b) for a, b in G.edges() if a != b)
    H.add_edges_from((V, node) for node in boundary if node in G)
    on: set = set()
    for block in nx.biconnected_component_edges(H):
        pairs = {frozenset(e) for e in block}
        if len(pairs) > 1 and any(V in p for p in pairs):
            on |= pairs
    return np.asarray([
        frozenset((index_to_node[int(a)], index_to_node[int(b)])) in on for a, b in zip(us, vs)
    ], dtype=bool)


# --- choosing ---------------------------------------------------------------------------------


def open_end_conditions(
    model: str,
    G: nx.MultiGraph,
    *,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
    inlet_p_bc: float,
    outlet_p_bc: float,
    conductance=None,
    node_list: Optional[Sequence[Any]] = None,
    terminal_resistance_multiple: float = DEFAULT_TERMINAL_RESISTANCE_MULTIPLE,
    pressure_target_pa: Optional[float] = None,
    shear_target_pa: Optional[float] = None,
    shear_weight: float = DEFAULT_SHEAR_WEIGHT,
) -> OpenEndConditions:
    """The conditions *model* puts on *G*'s open ends (:data:`OPEN_END_MODELS`).
    ``estimated`` needs the graph's own *conductance* matrix and *node_list*."""
    if model not in OPEN_END_MODELS:
        raise ValueError(f"open_end_model must be one of {', '.join(OPEN_END_MODELS)}, got {model!r}")
    if model == CLOSED:
        return OpenEndConditions()
    ends = open_end_candidates(G, inlet_nodes, outlet_nodes)
    if not ends:
        return OpenEndConditions(model=model, info={"ends": 0})
    if model == TERMINAL_RESISTANCE:
        return terminal_resistance_conditions(G, ends, multiple=terminal_resistance_multiple)
    if model == OUTLET:
        return outlet_conditions(ends)
    if conductance is None or node_list is None:
        raise ValueError("open_end_model=estimated needs the conductance matrix and its node list")
    return estimate_open_end_conditions(
        G, conductance, node_list,
        inlet_nodes=inlet_nodes, outlet_nodes=outlet_nodes, inlet_p_bc=inlet_p_bc, outlet_p_bc=outlet_p_bc,
        ends=ends, pressure_target_pa=pressure_target_pa, shear_target_pa=shear_target_pa,
        shear_weight=shear_weight,
    )


__all__ = [
    "CLOSED",
    "DRAINS",
    "ESTIMATED",
    "FEEDS",
    "GRAPH_RECORD",
    "INLET_GHOST",
    "OPEN_END_MODELS",
    "OUTLET",
    "OUTLET_GHOST",
    "OpenEndConditions",
    "TERMINAL_RESISTANCE",
    "augment",
    "conditions_on_graph",
    "estimate_open_end_conditions",
    "keep_on_graph",
    "open_end_candidates",
    "open_end_conditions",
    "open_end_directions",
    "open_end_flows",
    "outlet_conditions",
    "terminal_resistance_conditions",
]
