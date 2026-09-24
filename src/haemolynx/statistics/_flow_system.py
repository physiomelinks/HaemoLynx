"""A sparse inlet->outlet flow solve shared by the Laplacian-based analyses.

Inlets are held at pressure 1 and outlets at 0; every other node connected
to a terminal is solved for. Conductances come from the same
``statistics_route_weighting`` model as the route analyses (see
:func:`haemolynx.statistics._route_graph._edge_conductance`). The system is
linear, so every *fraction* derived from it (a vessel's share of the total
flow, the flow lost when a vessel is blocked) is independent of the
absolute pressures -- and with ``resistance`` weighting it reproduces the
solved haemodynamic flow pattern exactly.

Conductances are rescaled so the largest is 1 before factorising: SI
conductances are ~1e-16, small enough to hurt a sparse LU's pivoting.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ._route_graph import _edge_conductance, _iter_edges


class TerminalFlowSystem:
    """One factorised inlet(1) -> outlet(0) pressure solve over *G*.

    Attributes, all indexed over :attr:`edges` (every non-self-loop vessel
    not in *removed*, in :func:`_iter_edges` order):

    * ``edges`` -- ``(u, v, key)`` per vessel;
    * ``conductance`` -- rescaled conductance;
    * ``flow`` -- signed flow ``u -> v`` (rescaled units; divide by
      :attr:`total_flow` for a fraction);
    * ``inflow_sign`` -- +1/-1 where the vessel leaves/enters an inlet, else 0,
      so ``inflow_sign @ flow`` is the total inlet->outlet flow.
    """

    def __init__(
        self,
        G: Any,
        inlets: Iterable[Any],
        outlets: Iterable[Any],
        weighting: str,
        *,
        removed: Optional[set] = None,
    ) -> None:
        inlets = set(inlets)
        outlets = set(outlets)
        removed = removed or set()
        self.nodes = list(G.nodes)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        n = len(self.nodes)

        edges, conductance = [], []
        for u, v, key, data in _iter_edges(G):
            if u == v or (u, v, key) in removed:
                continue
            g = _edge_conductance(data, weighting)
            if not (g > 0.0 and np.isfinite(g)):
                continue
            edges.append((u, v, key))
            conductance.append(g)
        self.edges = edges
        self.edge_position = {edge: i for i, edge in enumerate(edges)}
        g = np.asarray(conductance, dtype=float)
        self.scale = float(g.max()) if g.size else 1.0
        g = g / self.scale if g.size else g
        self.conductance = g

        ui = np.asarray([self.index[u] for u, _v, _k in edges], dtype=int)
        vi = np.asarray([self.index[v] for _u, v, _k in edges], dtype=int)
        self.u_index, self.v_index = ui, vi
        m = len(edges)

        # Only nodes whose component contains a terminal get a pressure.
        adjacency = coo_matrix(
            (np.ones(m), (ui, vi)), shape=(n, n)
        ).tocsr() if m else csr_matrix((n, n))
        _count, labels = connected_components(adjacency, directed=False)
        self.component_labels = labels
        terminal_idx = [self.index[t] for t in inlets | outlets if t in self.index]
        live_labels = set(labels[terminal_idx].tolist()) if terminal_idx else set()
        live = np.isin(labels, list(live_labels)) if live_labels else np.zeros(n, bool)

        is_terminal = np.zeros(n, dtype=bool)
        is_terminal[terminal_idx] = True
        free = np.flatnonzero(live & ~is_terminal)
        self.free = free
        self.free_position = np.full(n, -1, dtype=int)
        self.free_position[free] = np.arange(free.size)

        known = np.zeros(n, dtype=float)
        for t in inlets:
            if t in self.index:
                known[self.index[t]] = 1.0
        pressure = np.full(n, np.nan)
        pressure[is_terminal] = known[is_terminal]

        # Incidence of each vessel on the free nodes only (terminal pressures
        # are fixed, so they drop out of the unknowns).
        fu = self.free_position[ui]
        fv = self.free_position[vi]
        rows = np.concatenate([np.flatnonzero(fu >= 0), np.flatnonzero(fv >= 0)])
        cols = np.concatenate([fu[fu >= 0], fv[fv >= 0]])
        vals = np.concatenate([np.ones(int((fu >= 0).sum())), -np.ones(int((fv >= 0).sum()))])
        self.incidence_free = csr_matrix((vals, (rows, cols)), shape=(m, free.size))

        self.lu = None
        if free.size:
            weighted = self.incidence_free.multiply(g[:, None]).tocsr()
            laplacian = (self.incidence_free.T @ weighted).tocsc()
            # Right-hand side: current forced in by the fixed terminal
            # pressures through vessels with exactly one free end.
            p_known_u = np.where(fu < 0, np.nan_to_num(pressure[ui]), 0.0)
            p_known_v = np.where(fv < 0, np.nan_to_num(pressure[vi]), 0.0)
            rhs = np.zeros(free.size)
            one_free_u = (fu >= 0) & (fv < 0)
            one_free_v = (fv >= 0) & (fu < 0)
            np.add.at(rhs, fu[one_free_u], g[one_free_u] * p_known_v[one_free_u])
            np.add.at(rhs, fv[one_free_v], g[one_free_v] * p_known_u[one_free_v])
            # The grounded Laplacian is symmetric positive definite: a
            # symmetric fill-reducing ordering with diagonal pivots halves the
            # fill of the default COLAMD ordering and factorises ~2.5x faster
            # on a 3D mesh, with the same residual.
            self.lu = splu(
                laplacian,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            pressure[free] = self.lu.solve(rhs)
        self.pressure = pressure

        pu = np.nan_to_num(pressure[ui])
        pv = np.nan_to_num(pressure[vi])
        dead = np.isnan(pressure[ui]) | np.isnan(pressure[vi])
        flow = g * (pu - pv)
        flow[dead] = 0.0
        self.flow = flow

        u_in = np.isin(ui, [self.index[t] for t in inlets if t in self.index])
        v_in = np.isin(vi, [self.index[t] for t in inlets if t in self.index])
        self.inflow_sign = u_in.astype(float) - v_in.astype(float)
        self.total_flow = float(self.inflow_sign @ flow)

    def reachable_outlet_fraction(self, inlets: Iterable[Any], outlets: Iterable[Any]) -> float:
        """Fraction of *outlets* still connected to any inlet."""
        outlets = [o for o in outlets if o in self.index]
        if not outlets:
            return 0.0
        inlet_labels = {
            int(self.component_labels[self.index[i]]) for i in inlets if i in self.index
        }
        reached = sum(
            1 for o in outlets if int(self.component_labels[self.index[o]]) in inlet_labels
        )
        return reached / len(outlets)

    def flow_fraction(self) -> np.ndarray:
        """|flow| / total inlet->outlet flow, per vessel (0 when none flows)."""
        if self.total_flow <= 0.0:
            return np.zeros_like(self.flow)
        return np.abs(self.flow) / self.total_flow

    @property
    def equivalent_resistance(self) -> float:
        """Inlet->outlet resistance in the weighting's own units (1 / flow at
        a unit pressure difference, undoing the conductance rescale)."""
        if self.total_flow <= 0.0:
            return float("inf")
        return 1.0 / (self.total_flow * self.scale)
