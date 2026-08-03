"""Network resistance from Laplacian."""
from pathlib import Path
import numpy as np
import networkx as nx
import pyvista as pv


def build_conductance_matrix_from_graph(
    G: nx.Graph, conductance_attr: str = "conductance"
) -> tuple[np.ndarray, list]:
    """Build symmetric conductance matrix from graph edge conductances.

    Returns:
        A tuple of (conductance_matrix, node_list) where matrix indices map to
        node IDs via node_list order.
    """
    node_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    conductance = np.zeros((len(node_list), len(node_list)), dtype=float)

    for u, v, data in G.edges(data=True):
        edge_conductance = data.get(conductance_attr)
        if edge_conductance is None or edge_conductance <= 0:
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        # Sum conductance for parallel edges.
        conductance[i, j] += edge_conductance
        conductance[j, i] += edge_conductance

    return conductance, node_list


def calc_laplacian_from_conductance_matrix(C: np.ndarray) -> np.ndarray:
    """Compute graph Laplacian from conductance matrix. L = diag(sum(C,1)) - C."""
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


def solve_flow_from_conductance_matrix(
    conductance: np.ndarray,
    node_list: list,
    *,
    input_p_bc: float,
    output_p_bc: float,
    starting_nodes: list,
    output_nodes: list,
) -> dict:
    """Solve nodal pressures from a conductance matrix with Dirichlet BCs.

    Boundary conditions are applied by node ID. Returns the node order and the
    pressure at each node; use :func:`set_edge_flows` to turn those into
    per-edge flows on the graph.
    """
    if conductance.ndim != 2 or conductance.shape[0] != conductance.shape[1]:
        raise ValueError("conductance must be a square matrix")
    n_nodes = conductance.shape[0]
    if len(node_list) != n_nodes:
        raise ValueError(
            f"node_list length ({len(node_list)}) must match matrix size ({n_nodes})"
        )
    if not starting_nodes:
        raise ValueError("starting_nodes cannot be empty")
    if not output_nodes:
        raise ValueError("output_nodes cannot be empty")

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    missing_in = [n for n in starting_nodes if n not in node_to_idx]
    missing_out = [n for n in output_nodes if n not in node_to_idx]
    if missing_in or missing_out:
        raise ValueError(
            "Boundary-condition nodes missing from node_list. "
            f"missing_starting={missing_in}, missing_output={missing_out}"
        )

    overlap = set(starting_nodes).intersection(output_nodes)
    if overlap and input_p_bc != output_p_bc:
        raise ValueError(
            "Overlapping starting/output nodes have conflicting pressures: "
            f"{sorted(overlap)}"
        )

    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    pressure = np.zeros(n_nodes, dtype=float)

    bc_idx_to_p: dict[int, float] = {}
    for node_id in starting_nodes:
        bc_idx_to_p[node_to_idx[node_id]] = float(input_p_bc)
    for node_id in output_nodes:
        idx = node_to_idx[node_id]
        if idx in bc_idx_to_p and bc_idx_to_p[idx] != float(output_p_bc):
            raise ValueError(
                f"Node {node_id} receives conflicting BC pressures "
                f"{bc_idx_to_p[idx]} and {output_p_bc}"
            )
        bc_idx_to_p[idx] = float(output_p_bc)

    known_idx = np.array(sorted(bc_idx_to_p.keys()), dtype=int)
    for idx in known_idx:
        pressure[idx] = bc_idx_to_p[idx]
    unknown_idx = np.array(
        sorted(set(range(n_nodes)).difference(set(known_idx))), dtype=int
    )

    # Heuristic dense-solve estimate using cubic complexity.
    n_free = int(len(unknown_idx))
    alpha = 2.5e-9
    t_est = alpha * (max(n_free, 1) ** 3)
    print(
        "[flow-solve] Runtime estimate (heuristic): "
        f"t_est = alpha * n_free^3 = {alpha:.2e} * {n_free}^3 = {t_est:.3f} s "
        f"(n={n_nodes}, n_free={n_free})"
    )

    if n_free > 0:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        p_k = pressure[known_idx]
        rhs = -l_uk @ p_k
        try:
            p_u = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError:
            # Fallback for singular/ill-conditioned systems.
            p_u = np.linalg.lstsq(l_uu, rhs, rcond=None)[0]
        pressure[unknown_idx] = p_u

    return {"node_list": node_list, "pressure": pressure}


def set_edge_flows(G: nx.Graph, node_list: list, pressure: np.ndarray) -> dict:
    """Write the flow implied by *pressure* onto every edge of *G*.

    Adds ``pressure_drop`` (Pa), ``flow_signed`` and ``flow_abs`` (m^3/s), so
    the flows travel with the graph and any export writes them out like any
    other edge attribute.
    """
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
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
        data["flow_abs"] = abs(signed)
        edges_set += 1
        total_abs_flow += abs(signed)
    return {"edges_set": edges_set, "total_abs_flow": total_abs_flow}
