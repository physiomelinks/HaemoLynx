"""Network resistance from Laplacian."""
from __future__ import annotations

import logging
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

logger = logging.getLogger(__name__)

#: ``haemodynamics_solver``: how a flow solve holds and solves the network.
#: ``dense`` is an N x N array and a dense solve -- 8 N^2 bytes, 3.2 GB at
#: 20,000 nodes, with several copies alive at once. ``sparse`` stores only the
#: vessels and factorises the same equations; the same pressures to rounding.
FLOW_SOLVER_DENSE = "dense"
FLOW_SOLVER_SPARSE = "sparse"
FLOW_SOLVERS = (FLOW_SOLVER_DENSE, FLOW_SOLVER_SPARSE)

#: ``equivalent_resistance_solver``: how the two-point resistance is found.
#: ``eigendecomposition`` is every eigenpair of the dense Laplacian (cubic in
#: the node count: minutes at 5,000 nodes, hours at 20,000); ``sparse`` is
#: one sparse solve with a unit current in at one node and the other grounded.
EQUIVALENT_RESISTANCE_EIGENDECOMPOSITION = "eigendecomposition"
EQUIVALENT_RESISTANCE_SPARSE = "sparse"
EQUIVALENT_RESISTANCE_SOLVERS = (
    EQUIVALENT_RESISTANCE_EIGENDECOMPOSITION,
    EQUIVALENT_RESISTANCE_SPARSE,
)

_NOT_CONNECTED_MESSAGE = (
    "Inlet and outlet boundary nodes are not connected by "
    "conductance-carrying edges; cannot solve for flow. After a "
    "large-vessel cut or boundary reassignment, check that the inlet "
    "and outlet land on the same conductive part of the network."
)
_ZERO_ROW_MESSAGE = (
    "Reduced Laplacian has a zero row; the boundary nodes do not "
    "constrain every conductive node that the solver would solve for."
)
_SINGULAR_MESSAGE = (
    "Conductance network Laplacian is singular and cannot be solved. "
    "Check that inlet and outlet boundary nodes lie on the same "
    "connected, conductance-carrying part of the network and that "
    "boundary pressures differ."
)
_NON_FINITE_PRESSURE_MESSAGE = (
    "Flow solve produced non-finite nodal pressures; the conductance "
    "network is likely ill-conditioned or disconnected."
)

#: Smallest |flow| passed to log10 so zero-flow edges stay finite, not -inf.
FLOW_ABS_LOG10_FLOOR = 1e-20


def flow_abs_log10_value(flow_abs: float) -> float:
    """``log10(max(|flow|, FLOW_ABS_LOG10_FLOOR))`` for vessel colouring."""
    return math.log10(max(abs(float(flow_abs)), FLOW_ABS_LOG10_FLOOR))


def build_conductance_matrix_from_graph(
    G: nx.Graph,
    conductance_attr: str = "conductance",
    *,
    node_list: list | None = None,
    node_to_idx: dict | None = None,
    out: np.ndarray | None = None,
    solver: str = FLOW_SOLVER_DENSE,
) -> tuple[np.ndarray | sp.csr_matrix, list]:
    """Build symmetric conductance matrix from graph edge conductances.

    Returns:
        A tuple of (conductance_matrix, node_list) where matrix indices map to
        node IDs via node_list order.

    *node_list*/*node_to_idx*/*out* let a repeat caller skip rebuilding the
    node ordering and reallocating the array when only edge conductances
    changed since the last call -- e.g.
    :func:`haemolynx.haemodynamics.haematocrit_distribution.iterate_flow_and_haematocrit`,
    which re-solves the same graph's topology every outer iteration and only
    ever rewrites edge attributes, never adds or removes a node or edge.
    Passing none of them reproduces the original behaviour exactly: a fresh
    ``node_list``/``node_to_idx`` from *G*'s current node order and a newly
    zeroed array. *out* is filled in place and returned as-is, not replaced
    with a new array; its shape must match ``(len(node_list), len(node_list))``
    for whichever *node_list* this call ends up using (explicit or rebuilt).

    Self-loops are left out: both ends sit at one pressure, so they carry no
    flow, and on the diagonal they would break the Laplacian.

    *solver* ``"sparse"`` (``haemodynamics_solver``) returns a
    ``scipy.sparse`` CSR matrix holding only the vessels, the same entries the
    dense array has; every solve in this package takes either. *out* is a
    dense array, so it cannot be combined with it.
    """
    _check_flow_solver(solver)
    if node_list is None:
        node_list = list(G.nodes())
    if node_to_idx is None:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    n = len(node_list)
    if solver == FLOW_SOLVER_SPARSE:
        if out is not None:
            raise ValueError("out is a dense array; the sparse solver builds its own matrix.")
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for u, v, data in G.edges(data=True):
            edge_conductance = data.get(conductance_attr)
            if edge_conductance is None or edge_conductance <= 0 or u == v:
                continue
            i = node_to_idx[u]
            j = node_to_idx[v]
            rows += [i, j]
            cols += [j, i]
            values += [float(edge_conductance)] * 2
        # Converting from COO sums duplicate entries: parallel edges add up.
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        return matrix, node_list
    if out is None:
        conductance = np.zeros((n, n), dtype=float)
    else:
        if out.shape != (n, n):
            raise ValueError(
                f"out must be shape ({n}, {n}) to match node_list, got {out.shape}."
            )
        conductance = out
        conductance.fill(0.0)

    for u, v, data in G.edges(data=True):
        edge_conductance = data.get(conductance_attr)
        # A self-loop has one pressure at both ends, so it carries no flow.
        if edge_conductance is None or edge_conductance <= 0 or u == v:
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        # Sum conductance for parallel edges.
        conductance[i, j] += edge_conductance
        conductance[j, i] += edge_conductance

    return conductance, node_list


def _check_flow_solver(solver: str) -> None:
    if solver not in FLOW_SOLVERS:
        raise ValueError(f"Unknown flow solver {solver!r}; expected one of {FLOW_SOLVERS}.")


def calc_laplacian_from_conductance_matrix(C: np.ndarray) -> np.ndarray:
    """Compute graph Laplacian from conductance matrix. L = diag(sum(C,1)) - C.

    A ``scipy.sparse`` *C* gives a sparse CSR Laplacian, checked the same way.
    """
    if sp.issparse(C):
        C = C.tocsr()
        # np.allclose's own test, |C - C^T| <= atol + rtol |C^T|, on the
        # stored entries (an absent one is zero on both sides).
        excess = abs(C - C.T) - abs(C.T) * 1e-5
        if excess.nnz and excess.max() > 1e-8:
            raise ValueError("Conductance matrix must be symmetric")
        if np.any(C.diagonal() != 0):
            raise ValueError("Conductance matrix diagonal must be zero")
        degree = np.asarray(C.sum(axis=1)).ravel()
        return (sp.diags(degree) - C).tocsr()
    if not np.allclose(C, C.T):
        raise ValueError("Conductance matrix must be symmetric")
    if not np.all(np.diagonal(C) == 0):
        raise ValueError("Conductance matrix diagonal must be zero")
    diag = np.sum(C, axis=1)
    return np.diag(diag) - C


def calc_two_point_from_laplacian_matrix_nodeID(
    L: np.ndarray, G: nx.MultiGraph, node_id1, node_id2
) -> float:
    """Effective resistance between two nodes from Laplacian eigen-decomposition."""
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    try:
        node_idx1 = node_to_idx[node_id1]
        node_idx2 = node_to_idx[node_id2]
    except KeyError as e:
        raise ValueError(f"Node {e} not found in graph")
    eigvals, eigvecs = np.linalg.eigh(L)
    # Null-space cut-off must scale with the matrix: conductances are ~1e-16
    # m^3/(Pa.s) in SI units, so any fixed absolute threshold would discard
    # every mode and silently return zero resistance.
    tolerance = float(np.max(eigvals)) * len(eigvals) * np.finfo(float).eps
    R = 0.0
    for ii in range(1, len(eigvals)):
        if eigvals[ii] > tolerance:
            R += (1 / eigvals[ii]) * (
                eigvecs[node_idx1, ii] - eigvecs[node_idx2, ii]
            ) ** 2
    return R


def calc_two_point_resistance_sparse(
    conductance: np.ndarray | sp.spmatrix,
    node_list: list,
    node_id1,
    node_id2,
    *,
    node_to_idx: dict | None = None,
) -> float:
    """Effective resistance between two nodes by one sparse solve.

    One unit of current in at *node_id1*, *node_id2* grounded: the potential
    *node_id1* rises to is the resistance between them (Pa.s/m^3 for
    conductances in m^3/(Pa.s)). The same number
    :func:`calc_two_point_from_laplacian_matrix_nodeID` gives, to rounding,
    without the eigendecomposition. Two nodes on separate conductive pieces of
    the network have no path for a current, so the answer is ``inf`` (with a
    warning) -- where the eigendecomposition returns a finite number.
    *conductance* may be dense or sparse; *node_list* gives its node order.
    """
    if node_to_idx is None:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    try:
        idx1 = node_to_idx[node_id1]
        idx2 = node_to_idx[node_id2]
    except KeyError as e:
        raise ValueError(f"Node {e} not found in graph")
    if idx1 == idx2:
        return 0.0
    matrix = conductance.tocsr() if sp.issparse(conductance) else sp.csr_matrix(conductance)
    labels = _component_labels(matrix)
    if labels[idx1] != labels[idx2]:
        logger.warning(
            f"Two-point resistance: nodes {node_id1} and {node_id2} are on separate "
            "conductive parts of the network, so no current can pass; it is infinite."
        )
        return float("inf")
    component = np.nonzero(labels == labels[idx1])[0]
    keep = component[component != idx2]
    laplacian = calc_laplacian_from_conductance_matrix(matrix)
    current = np.zeros(len(keep), dtype=float)
    position = int(np.searchsorted(keep, idx1))
    current[position] = 1.0
    potential = _sparse_factor(laplacian[keep][:, keep]).solve(current)
    return float(potential[position])


def boundary_flow(
    conductance: np.ndarray | sp.spmatrix,
    pressure: np.ndarray,
    node_idx,
) -> float:
    """Total flow (m^3/s) out of the nodes at *node_idx* into the rest of the
    network: each node's conductances times its pressure drops, summed. Dense
    or sparse *conductance*."""
    idx = np.unique(np.asarray(node_idx, dtype=int))
    if idx.size == 0:
        return 0.0
    rows = conductance.tocsr()[idx] if sp.issparse(conductance) else conductance[idx]
    degree = np.asarray(rows.sum(axis=1)).ravel()
    neighbours = np.asarray(rows @ pressure).ravel()
    return float(np.sum(degree * pressure[idx] - neighbours))


def network_resistance(
    conductance: np.ndarray | sp.spmatrix,
    node_list: list,
    pressure: np.ndarray,
    *,
    inlet_nodes: list,
    inlet_p_bc: float,
    outlet_p_bc: float,
    node_to_idx: dict | None = None,
) -> float | None:
    """The whole network's resistance from a solved flow: the inlet-to-outlet
    pressure drop over the total flow in through the inlets (Pa.s/m^3).

    Every inlet and every outlet takes part, unlike the two-point resistance
    between one inlet and one outlet with every other boundary left floating.
    ``inf`` when no flow enters, ``None`` when the drop is zero (no flow to
    divide by).
    """
    drop = float(inlet_p_bc) - float(outlet_p_bc)
    if drop == 0.0:
        return None
    if node_to_idx is None:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    inflow = boundary_flow(
        conductance, pressure, [node_to_idx[n] for n in inlet_nodes if n in node_to_idx]
    )
    # Exact zero only: flows are ~1e-14 m^3/s, so any tolerance would swallow them.
    return float("inf") if inflow == 0.0 else drop / inflow


def _sparse_factor(matrix: sp.spmatrix):
    """LU factors of a reduced Laplacian -- symmetric positive definite, so
    ordered and pivoted as one (as :mod:`haemolynx.statistics._flow_system`
    does)."""
    return splu(
        matrix.tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )


def _component_labels(conductance: sp.spmatrix) -> np.ndarray:
    """Each node's conductive component: nodes joined by edges with a
    positive conductance share a label."""
    _count, labels = connected_components(conductance.tocsr() > 0, directed=False)
    return labels


def _has_conductance(conductance: sp.spmatrix) -> np.ndarray:
    return np.diff((conductance.tocsr() > 0).indptr) > 0


def reachable_through_conductances(
    adjacency: np.ndarray, seed_idx: np.ndarray
) -> np.ndarray:
    """Boolean mask of nodes connected to any seed through conductive edges.

    A ``scipy.sparse`` *adjacency* is labelled by connected components."""
    if sp.issparse(adjacency):
        labels = _component_labels(adjacency)
        return np.isin(labels, labels[np.asarray(seed_idx, dtype=int)])
    reached = np.zeros(adjacency.shape[0], dtype=bool)
    reached[seed_idx] = True
    frontier = np.asarray(seed_idx, dtype=int)
    while len(frontier):
        neighbours = np.any(adjacency[frontier], axis=0) & ~reached
        frontier = np.nonzero(neighbours)[0]
        reached[frontier] = True
    return reached


def _conductively_connected(
    adjacency: np.ndarray, source_idx: np.ndarray, target_idx: np.ndarray
) -> bool:
    """Return True if any *target* is reachable from any *source* through conductive edges."""
    reached = reachable_through_conductances(adjacency, source_idx)
    return bool(np.any(reached[np.asarray(target_idx, dtype=int)]))


def reachable_unknown_node_indices(
    conductance: np.ndarray, known_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Non-boundary node indices with a conductive path to a boundary node.

    This is the module's shared policy for a node with no such path: a zero
    row in the reduced Laplacian, which forces ``np.linalg.solve``'s
    ``lstsq`` fallback to degrade *every* node's pressure, not just the
    disconnected ones. Restricting the solve to this reachable set instead
    leaves an unreachable node's pressure at 0 and its edges at zero flow.
    Both solvers in this package (:func:`solve_flow_from_conductance_matrix`
    and :func:`haemolynx.haemodynamics.pericyte_sweep.solve_pressure_and_boundary_flow`)
    apply it, so a disconnected component degrades the same way in either one.

    Returns the restricted ``unknown_idx`` and how many additional nodes
    carry a conductance but were excluded as unreachable, for the caller to
    warn about. *conductance* may be dense or sparse.
    """
    n_nodes = conductance.shape[0]
    if sp.issparse(conductance):
        reached = reachable_through_conductances(conductance, known_idx)
        unknown_mask = np.ones(n_nodes, dtype=bool)
        unknown_mask[known_idx] = False
        unknown_idx = np.nonzero(unknown_mask & reached)[0]
        stranded_conductive = int(
            np.sum(unknown_mask & ~reached & _has_conductance(conductance))
        )
        return unknown_idx, stranded_conductive
    adjacency = conductance > 0
    reached = reachable_through_conductances(adjacency, known_idx)
    unknown_mask = np.ones(n_nodes, dtype=bool)
    unknown_mask[known_idx] = False
    unknown_idx = np.nonzero(unknown_mask & reached)[0]
    stranded_conductive = int(np.sum(unknown_mask & ~reached & adjacency.any(axis=1)))
    return unknown_idx, stranded_conductive


def _warn_components_pinned_to_one_pressure(
    adjacency: np.ndarray, bc_idx_to_p: dict
) -> None:
    """Warn for each conductive component whose boundary nodes all impose the
    same pressure — nothing drives it, so its every flow solves to zero."""
    if sp.issparse(adjacency):
        labels = _component_labels(adjacency)
        conductive = _has_conductance(adjacency)
        pressures_by_label: dict[int, set] = {}
        for idx in sorted(bc_idx_to_p):
            if conductive[idx]:
                pressures_by_label.setdefault(int(labels[idx]), set()).add(bc_idx_to_p[idx])
        for label, pressures in pressures_by_label.items():
            if len(pressures) == 1:
                logger.warning(
                    f"A conductive component of {int(np.sum(labels == label))} node(s) only "
                    f"reaches boundary nodes at {pressures.pop()} Pa; with no "
                    "pressure difference across it, its every flow solves to zero. "
                    "Check that the inlet and outlet boundary nodes land on the "
                    "same connected, conductance-carrying part of the network."
                )
        return
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    for start in sorted(bc_idx_to_p):
        if visited[start] or not adjacency[start].any():
            continue
        component = reachable_through_conductances(adjacency, np.array([start]))
        visited |= component
        pressures = {
            bc_idx_to_p[idx] for idx in np.nonzero(component)[0] if idx in bc_idx_to_p
        }
        if len(pressures) == 1:
            logger.warning(
                f"A conductive component of {int(component.sum())} node(s) only "
                f"reaches boundary nodes at {pressures.pop()} Pa; with no "
                "pressure difference across it, its every flow solves to zero. "
                "Check that the inlet and outlet boundary nodes land on the "
                "same connected, conductance-carrying part of the network."
            )


def solve_flow_from_conductance_matrix(
    conductance: np.ndarray,
    node_list: list,
    *,
    inlet_p_bc: float,
    outlet_p_bc: float,
    inlet_nodes: list,
    outlet_nodes: list,
    node_to_idx: dict | None = None,
) -> dict:
    """Solve nodal pressures from a conductance matrix with Dirichlet BCs.

    Boundary conditions are applied by node ID. Returns the node order and the
    pressure at each node; use :func:`set_edge_flows` to turn those into
    per-edge flows on the graph.

    *node_to_idx*, when given, must be the ``{node_id: idx}`` mapping for
    *node_list* in the same order -- lets a caller that already built it
    (e.g. alongside *node_list* from :func:`build_conductance_matrix_from_graph`)
    skip rebuilding it here. Not validated against *node_list* for cost
    reasons; passing a mismatched mapping is the caller's error.

    A ``scipy.sparse`` *conductance* (``haemodynamics_solver="sparse"``) is
    solved by sparse LU factorisation, with the same checks, warnings and
    errors; the pressures agree with the dense solve to rounding.
    """
    if conductance.ndim != 2 or conductance.shape[0] != conductance.shape[1]:
        raise ValueError("conductance must be a square matrix")
    n_nodes = conductance.shape[0]
    if len(node_list) != n_nodes:
        raise ValueError(
            f"node_list length ({len(node_list)}) must match matrix size ({n_nodes})"
        )
    if not inlet_nodes:
        raise ValueError("inlet_nodes cannot be empty")
    if not outlet_nodes:
        raise ValueError("outlet_nodes cannot be empty")

    if node_to_idx is None:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    missing_in = [n for n in inlet_nodes if n not in node_to_idx]
    missing_out = [n for n in outlet_nodes if n not in node_to_idx]
    if missing_in or missing_out:
        raise ValueError(
            "Boundary-condition nodes missing from node_list. "
            f"missing_inlet={missing_in}, missing_output={missing_out}"
        )

    overlap = set(inlet_nodes).intersection(outlet_nodes)
    if overlap and inlet_p_bc != outlet_p_bc:
        raise ValueError(
            "Overlapping inlet/outlet nodes have conflicting pressures: "
            f"{sorted(overlap)}"
        )

    is_sparse = sp.issparse(conductance)
    if not np.all(np.isfinite(conductance.data if is_sparse else conductance)):
        raise ValueError(
            "Conductance matrix contains non-finite values; cannot solve for flow."
        )

    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    pressure = np.zeros(n_nodes, dtype=float)

    bc_idx_to_p: dict[int, float] = {}
    for node_id in inlet_nodes:
        bc_idx_to_p[node_to_idx[node_id]] = float(inlet_p_bc)
    for node_id in outlet_nodes:
        idx = node_to_idx[node_id]
        if idx in bc_idx_to_p and bc_idx_to_p[idx] != float(outlet_p_bc):
            raise ValueError(
                f"Node {node_id} receives conflicting BC pressures "
                f"{bc_idx_to_p[idx]} and {outlet_p_bc}"
            )
        bc_idx_to_p[idx] = float(outlet_p_bc)

    known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
    for idx in known_idx:
        pressure[idx] = bc_idx_to_p[idx]

    if is_sparse:
        _solve_free_pressures_sparse(
            conductance.tocsr(),
            laplacian,
            pressure,
            known_idx=known_idx,
            inlet_idx=np.array([node_to_idx[n] for n in inlet_nodes], dtype=int),
            outlet_idx=np.array([node_to_idx[n] for n in outlet_nodes], dtype=int),
            bc_idx_to_p=bc_idx_to_p,
            pressure_differs=float(inlet_p_bc) != float(outlet_p_bc),
        )
        return {"node_list": node_list, "pressure": pressure}

    # A node with no conductive path to any boundary node gives the reduced
    # Laplacian a zero row; np.linalg.solve then raises and the lstsq
    # fallback degrades the pressure of *every* node, not just the
    # disconnected ones. Solve only what the boundary conditions can reach:
    # the rest has no driving pressure, so it keeps pressure 0 and its
    # edges carry zero flow.
    adjacency = conductance > 0
    inlet_idx = np.array([node_to_idx[n] for n in inlet_nodes], dtype=int)
    outlet_idx = np.array([node_to_idx[n] for n in outlet_nodes], dtype=int)
    if float(inlet_p_bc) != float(outlet_p_bc) and not _conductively_connected(
        adjacency, inlet_idx, outlet_idx
    ):
        raise ValueError(_NOT_CONNECTED_MESSAGE)
    unknown_idx, stranded_conductive = reachable_unknown_node_indices(
        conductance, known_idx
    )
    if stranded_conductive:
        logger.warning(
            f"{stranded_conductive} node(s) carry conductances but have no "
            "conductive path to any boundary node; their pressure stays 0 "
            "and their edges carry zero flow."
        )
    if float(inlet_p_bc) != float(outlet_p_bc):
        _warn_components_pinned_to_one_pressure(adjacency, bc_idx_to_p)

    # Heuristic dense-solve estimate using cubic complexity.
    n_free = int(len(unknown_idx))
    alpha = 2.5e-9
    t_est = alpha * (max(n_free, 1) ** 3)
    logger.info(
        "[flow-solve] Runtime estimate (heuristic): "
        f"t_est = alpha * n_free^3 = {alpha:.2e} * {n_free}^3 = {t_est:.3f} s "
        f"(n={n_nodes}, n_free={n_free})"
    )

    if n_free > 0:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        p_k = pressure[known_idx]
        rhs = -l_uk @ p_k
        if np.any(np.sum(np.abs(l_uu), axis=1) == 0):
            raise ValueError(_ZERO_ROW_MESSAGE)
        try:
            p_u = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(_SINGULAR_MESSAGE) from exc
        if not np.all(np.isfinite(p_u)):
            raise ValueError(_NON_FINITE_PRESSURE_MESSAGE)
        pressure[unknown_idx] = p_u

    return {"node_list": node_list, "pressure": pressure}


def solve_reduced_laplacian_sparse(
    laplacian: sp.spmatrix,
    pressure: np.ndarray,
    unknown_idx: np.ndarray,
    known_idx: np.ndarray,
) -> np.ndarray:
    """The free nodes' pressures, given the boundary nodes' in *pressure*: the
    reduced system ``L_uu p_u = -L_uk p_k`` by sparse LU, with the dense
    solve's own errors for a zero row, a singular system or a non-finite
    answer."""
    laplacian = laplacian.tocsr()
    l_rows = laplacian[unknown_idx]
    l_uu = l_rows[:, unknown_idx]
    rhs = -np.asarray(l_rows[:, known_idx] @ pressure[known_idx]).ravel()
    if np.any(np.asarray(abs(l_uu).sum(axis=1)).ravel() == 0):
        raise ValueError(_ZERO_ROW_MESSAGE)
    try:
        p_u = _sparse_factor(l_uu).solve(rhs)
    except RuntimeError as exc:  # splu: "Factor is exactly singular"
        raise ValueError(_SINGULAR_MESSAGE) from exc
    if not np.all(np.isfinite(p_u)):
        raise ValueError(_NON_FINITE_PRESSURE_MESSAGE)
    return p_u


def _solve_free_pressures_sparse(
    conductance: sp.csr_matrix,
    laplacian: sp.csr_matrix,
    pressure: np.ndarray,
    *,
    known_idx: np.ndarray,
    inlet_idx: np.ndarray,
    outlet_idx: np.ndarray,
    bc_idx_to_p: dict,
    pressure_differs: bool,
) -> None:
    """:func:`solve_flow_from_conductance_matrix` from its boundary pressures
    on, for a sparse matrix: the same connectivity policy, warnings and
    errors, in the same order, then the reduced solve. Fills *pressure*."""
    if pressure_differs and not _conductively_connected(conductance, inlet_idx, outlet_idx):
        raise ValueError(_NOT_CONNECTED_MESSAGE)
    unknown_idx, stranded_conductive = reachable_unknown_node_indices(conductance, known_idx)
    if stranded_conductive:
        logger.warning(
            f"{stranded_conductive} node(s) carry conductances but have no "
            "conductive path to any boundary node; their pressure stays 0 "
            "and their edges carry zero flow."
        )
    if pressure_differs:
        _warn_components_pinned_to_one_pressure(conductance, bc_idx_to_p)
    logger.info(
        f"[flow-solve] Sparse solve: n={conductance.shape[0]}, "
        f"n_free={len(unknown_idx)}, stored conductances={conductance.nnz}."
    )
    if len(unknown_idx):
        pressure[unknown_idx] = solve_reduced_laplacian_sparse(
            laplacian, pressure, unknown_idx, known_idx
        )


def set_edge_flows(
    G: nx.Graph,
    node_list: list,
    pressure: np.ndarray,
    *,
    node_to_idx: dict | None = None,
) -> dict:
    """Write the flow implied by *pressure* onto every edge of *G*.

    Adds ``pressure_drop`` (Pa), ``flow_signed`` and ``flow_abs`` (m^3/s), so
    the flows travel with the graph and any export writes them out like any
    other edge attribute. Also writes ``pressure`` (Pa) onto every node, so
    :func:`flow_conservation_residuals` can audit the solution later.

    *node_to_idx*, when given, must be the ``{node_id: idx}`` mapping for
    *node_list* in the same order -- see
    :func:`solve_flow_from_conductance_matrix`'s own parameter of the same
    name.
    """
    if node_to_idx is None:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    for node_id, idx in node_to_idx.items():
        if node_id in G:
            G.nodes[node_id]["pressure"] = float(pressure[idx])
    edges_set = 0
    total_abs_flow = 0.0
    for u, v, data in G.edges(data=True):
        conductance = data.get("conductance")
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        if conductance is None or u_idx is None or v_idx is None:
            continue
        drop = float(pressure[u_idx] - pressure[v_idx])
        signed = float(conductance) * drop
        data["pressure_u"] = float(pressure[u_idx])
        data["pressure_v"] = float(pressure[v_idx])
        data["pressure_drop"] = drop
        data["flow_signed"] = signed
        flow_abs = abs(signed)
        data["flow_abs"] = flow_abs
        data["flow_abs_log10"] = flow_abs_log10_value(flow_abs)
        edges_set += 1
        total_abs_flow += abs(signed)
    return {"edges_set": edges_set, "total_abs_flow": total_abs_flow}


def flow_conservation_residuals(G: nx.Graph, *, boundary_nodes=()) -> dict:
    """Net signed outflow (m^3/s) at each node without an imposed pressure.

    Kirchhoff's current law: after :func:`set_edge_flows`, the conductive
    flows into any node that is not a boundary node must sum to zero.
    Returns ``{node: residual}`` for every non-boundary node that has a
    ``pressure`` and at least one conductive incident edge; a residual far
    from zero (relative to the largest flow in the network) means the
    solved pressures do not satisfy the conductances stored on the graph.
    """
    boundary = set(boundary_nodes)
    residuals = {}
    for node, node_data in G.nodes(data=True):
        if node in boundary or "pressure" not in node_data:
            continue
        net_outflow = 0.0
        conductive_edges = 0
        for _, other, edge_data in G.edges(node, data=True):
            edge_conductance = edge_data.get("conductance")
            other_pressure = G.nodes[other].get("pressure")
            if not edge_conductance or other_pressure is None:
                continue
            net_outflow += float(edge_conductance) * (
                node_data["pressure"] - other_pressure
            )
            conductive_edges += 1
        if conductive_edges:
            residuals[node] = net_outflow
    return residuals
