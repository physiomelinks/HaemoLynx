"""Network resistance from Laplacian."""
from pathlib import Path
import numpy as np
import networkx as nx
import pyvista as pv


def build_conductance_matrix_from_graph(
    G: nx.Graph, resistance_attr: str = "resistance"
) -> tuple[np.ndarray, list]:
    """Build symmetric conductance matrix from graph edge resistances.

    Returns:
        A tuple of (conductance_matrix, node_list) where matrix indices map to
        node IDs via node_list order.
    """
    node_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    conductance = np.zeros((len(node_list), len(node_list)), dtype=float)

    for u, v, data in G.edges(data=True):
        edge_resistance = data.get(resistance_attr)
        if edge_resistance is None:
            continue
        edge_resistance = float(edge_resistance)
        if (not np.isfinite(edge_resistance)) or edge_resistance <= 0:
            continue
        edge_conductance = 1.0 / edge_resistance
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
    R = 0.0
    for ii in range(1, len(eigvals)):
        if eigvals[ii] > 1e-10:
            R += (1 / eigvals[ii]) * (
                eigvecs[node_idx1, ii] - eigvecs[node_idx2, ii]
            ) ** 2
    return R


def solve_pressure_and_boundary_flow(
    *,
    conductance: np.ndarray,
    node_list: list,
    input_p_bc: float,
    output_p_bc: float,
    starting_nodes: list,
    output_nodes: list,
) -> dict[str, float]:
    """Solve nodal pressures and aggregate source/sink boundary flows.

    This is the lightweight solver path without VTK side effects, suitable for
    pairwise comparison utilities and plotting.
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

    pressure = np.zeros(n_nodes, dtype=float)
    laplacian = calc_laplacian_from_conductance_matrix(conductance)

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
    pressure[known_idx] = np.array([bc_idx_to_p[idx] for idx in known_idx], dtype=float)
    unknown_idx = np.array(
        sorted(set(range(n_nodes)).difference(set(known_idx))),
        dtype=int,
    )
    if unknown_idx.size:
        l_uu = laplacian[np.ix_(unknown_idx, unknown_idx)]
        l_uk = laplacian[np.ix_(unknown_idx, known_idx)]
        rhs = -l_uk @ pressure[known_idx]
        try:
            pressure[unknown_idx] = np.linalg.solve(l_uu, rhs)
        except np.linalg.LinAlgError:
            pressure[unknown_idx] = np.linalg.lstsq(l_uu, rhs, rcond=None)[0]

    total_inlet_flow = 0.0
    for node_id in starting_nodes:
        i = node_to_idx[node_id]
        total_inlet_flow += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))

    total_outlet_flow = 0.0
    for node_id in output_nodes:
        i = node_to_idx[node_id]
        total_outlet_flow += float(np.sum(conductance[i, :] * (pressure[i] - pressure)))

    return {
        "total_inlet_flow": float(total_inlet_flow),
        "total_outlet_flow": float(total_outlet_flow),
    }


def solve_flow_from_conductance_matrix(
    conductance: np.ndarray,
    node_list: list,
    input_p_bc: float,
    output_p_bc: float,
    starting_nodes: list,
    output_nodes: list,
    vtk_export: dict,
) -> tuple[dict, dict]:
    """Solve nodal pressures/edge flows from conductance with Dirichlet BCs.

    Boundary conditions are applied by node IDs (matching node_list values).
    The returned vtk_export is updated with flow arrays on vessel cell_data and
    a new `_flow.vtp` output path.
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

    flow_result = {
        "node_list": node_list,
        "pressure": pressure,
    }

    vessels_path = Path(vtk_export["vessels_path"])
    vessels = pv.read(str(vessels_path))
    edge_u = np.asarray(vessels.cell_data.get("edge_u", []))
    edge_v = np.asarray(vessels.cell_data.get("edge_v", []))
    edge_resistance = np.asarray(vessels.cell_data.get("resistance", []), dtype=float)
    if len(edge_u) != vessels.n_cells or len(edge_v) != vessels.n_cells:
        raise ValueError(
            "VTK vessels file is missing edge_u/edge_v cell arrays needed for flow export."
        )
    if len(edge_resistance) != vessels.n_cells:
        raise ValueError(
            "VTK vessels file is missing resistance cell array needed for flow export."
        )

    edge_p_u = np.full(vessels.n_cells, np.nan, dtype=float)
    edge_p_v = np.full(vessels.n_cells, np.nan, dtype=float)
    for ii in range(vessels.n_cells):
        u = int(edge_u[ii])
        v = int(edge_v[ii])
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        if u_idx is not None:
            edge_p_u[ii] = pressure[u_idx]
        if v_idx is not None:
            edge_p_v[ii] = pressure[v_idx]
    pressure_drop = edge_p_u - edge_p_v
    valid_resistance = np.isfinite(edge_resistance) & (edge_resistance > 0.0)
    edge_conductance = np.full(vessels.n_cells, np.nan, dtype=float)
    edge_conductance[valid_resistance] = 1.0 / edge_resistance[valid_resistance]
    flow_signed = edge_conductance * pressure_drop
    flow_abs = np.abs(flow_signed)

    vessels.cell_data["pressure_u"] = edge_p_u
    vessels.cell_data["pressure_v"] = edge_p_v
    vessels.cell_data["pressure_drop"] = pressure_drop
    vessels.cell_data["flow_signed"] = flow_signed
    vessels.cell_data["flow_abs"] = flow_abs

    flow_path = vessels_path.with_name(f"{vessels_path.stem}_flow.vtp")
    vessels.save(flow_path)

    vtk_export = dict(vtk_export)
    vtk_export["vessels_path"] = str(flow_path)
    vtk_export["vessels_flow_path"] = str(flow_path)
    vtk_export["flow_field_names"] = [
        "pressure_u",
        "pressure_v",
        "pressure_drop",
        "flow_signed",
        "flow_abs",
    ]
    vtk_export["flow_cell_count"] = int(vessels.n_cells)

    flow_result["flow_signed"] = flow_signed
    flow_result["flow_abs"] = flow_abs
    return flow_result, vtk_export
