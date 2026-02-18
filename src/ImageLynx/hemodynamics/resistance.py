"""Network resistance from Laplacian."""
import numpy as np
import networkx as nx


def build_conductance_matrix_from_graph(
    G: nx.Graph, weight_attr: str = "weight"
) -> tuple[np.ndarray, list]:
    """Build symmetric conductance matrix from graph edge weights.

    Returns:
        A tuple of (conductance_matrix, node_list) where matrix indices map to
        node IDs via node_list order.
    """
    node_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    conductance = np.zeros((len(node_list), len(node_list)), dtype=float)

    for u, v, data in G.edges(data=True):
        edge_weight = data.get(weight_attr)
        if edge_weight is None or edge_weight <= 0:
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        # Sum conductance for parallel edges.
        conductance[i, j] += edge_weight
        conductance[j, i] += edge_weight

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
    R = 0.0
    for ii in range(1, len(eigvals)):
        if eigvals[ii] > 1e-10:
            R += (1 / eigvals[ii]) * (
                eigvecs[node_idx1, ii] - eigvecs[node_idx2, ii]
            ) ** 2
    return R
