"""Network resistance from Laplacian."""
import logging
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import networkx as nx
import pyvista as pv

logger = logging.getLogger(__name__)


#: Pascals per mmHg, exact by definition of the conventional millimetre of mercury.
PASCALS_PER_MMHG = 133.322387415

#: Converts this pipeline's Poiseuille flow to um^3/s.
#:
#: The flow solve computes ``Q = dP / R`` with ``R = 128 mu L / (pi d^4)``, taking pressure in
#: mmHg, viscosity in cP and lengths in um. That mixture is not an SI-derived system, so Q
#: carries units of mmHg um^4 / (cP um) and is not a volumetric flow rate.
#:
#: Rewriting R wholly in SI multiplies it by ``(1e-3 Pa s/cP) * (1e-6 m/um) / (1e-6 m/um)^4``,
#: which is 1e15. So
#:
#:     Q_SI [m^3/s] = (dP_mmHg * PASCALS_PER_MMHG) / (R_pipeline * 1e15)
#:                  = Q_pipeline * PASCALS_PER_MMHG * 1e-15
#:
#: and multiplying by 1e18 um^3/m^3 leaves ``Q_pipeline * PASCALS_PER_MMHG * 1e3``.
#:
#: This mattered because the perfusion coupling put the unconverted flow against a metabolic
#: sink in mmol/L/s times um^3. The sink then exceeded the source by 2.2e4 times and the
#: steady-state PO2 was correctly zero everywhere. Converted, the same system implies about a
#: 17% oxygen extraction, which is the right order for perfused tissue.
POISEUILLE_FLOW_TO_UM3_PER_S = PASCALS_PER_MMHG * 1e3


def poiseuille_flow_to_um3_per_s(flow, factor: float = POISEUILLE_FLOW_TO_UM3_PER_S):
    """Convert flow from this pipeline's mixed units to um^3/s.

    ``factor=1.0`` leaves the value untouched, for a caller deliberately working in solver
    units or comparing against output produced before the conversion existed.
    """
    return flow * factor


def build_conductance_matrix_from_graph(
    G: nx.Graph, 
    weight_attr: str = "resistance",
    robin_multiplier: float = 10.0,
    report: dict | None = None,
) -> tuple[sp.csr_matrix, list]:
    """Build symmetric conductance matrix from graph edge resistances.
    
    If nodes in the graph are tagged with `is_robin_boundary=True`, this mathematically
    hallucinates a 'ROBIN_GHOST_NODE' into the matrix, coupling the dead-ends to a virtual
    venous ground to simulate flow bleeding out of severed capillaries.

    Edges whose resistance is absent or non-positive cannot contribute a conductance and are
    dropped. That is unavoidable, but it used to be invisible: the network solved happily on
    whatever remained, and a graph missing a tenth of its edges produced a smaller, entirely
    plausible flow field. The two causes are counted separately because they mean different
    upstream faults. Absent means the assignment step never reached that edge; non-positive
    means it ran and produced a short circuit or an unphysical value.

    Pass ``report`` to receive the counts. A warning is logged regardless, since most call
    sites do not. The return value is unchanged, so existing callers are unaffected.
    """
    node_list = list(G.nodes())
    has_robin = any(G.nodes[n].get("is_robin_boundary", False) for n in G.nodes())
    
    if has_robin:
        node_list.append("ROBIN_GHOST_NODE")
        
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    n_nodes = len(node_list)

    rows, cols, data_vals = [], [], []

    dropped_missing = 0
    dropped_non_positive = 0
    edges_total = 0

    for u, v, data in G.edges(data=True):
        edges_total += 1
        resistance = data.get(weight_attr)
        if resistance is None:
            dropped_missing += 1
            continue
        if resistance <= 0:
            dropped_non_positive += 1
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        # Sum conductance (1/resistance) for parallel edges.
        edge_conductance = 1.0 / resistance
        rows.extend([i, j])
        cols.extend([j, i])
        data_vals.extend([edge_conductance, edge_conductance])
        
    dropped = dropped_missing + dropped_non_positive
    counts = {
        "edges_total": edges_total,
        "edges_dropped": dropped,
        "dropped_missing": dropped_missing,
        "dropped_non_positive": dropped_non_positive,
        "dropped_fraction": (dropped / edges_total) if edges_total else 0.0,
    }
    if report is not None:
        report.update(counts)
    if dropped:
        logger.warning(
            "Conductance matrix dropped %d of %d edges (%.1f%%): %d with no '%s' attribute, "
            "%d with a non-positive value. The network solves on the remainder, so this does "
            "not raise, but every flow below is for a smaller network than the graph.",
            dropped, edges_total, 100.0 * counts["dropped_fraction"],
            dropped_missing, weight_attr, dropped_non_positive,
        )

    if has_robin:
        ghost_idx = node_to_idx["ROBIN_GHOST_NODE"]
        for n in G.nodes():
            if G.nodes[n].get("is_robin_boundary", False):
                # Calculate the average resistance of the vessels connected to this cut
                edges_res = [d.get(weight_attr) for _, _, d in G.edges(n, data=True) if d.get(weight_attr) is not None]
                if edges_res:
                    r_avg = sum(edges_res) / len(edges_res)
                    r_ghost = r_avg * robin_multiplier
                    cond = 1.0 / r_ghost
                    i = node_to_idx[n]
                    rows.extend([i, ghost_idx])
                    cols.extend([ghost_idx, i])
                    data_vals.extend([cond, cond])

    conductance = sp.coo_matrix((data_vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return conductance, node_list


def calc_laplacian_from_conductance_matrix(C: sp.csr_matrix) -> sp.csr_matrix:
    """Compute graph Laplacian from conductance matrix. L = diag(sum(C,1)) - C."""
    diag = np.array(C.sum(axis=1)).flatten()
    L = sp.diags(diag) - C
    return L.tocsr()


def _solve_system_smart(A: sp.csr_matrix, b: np.ndarray, iterative_threshold: int = 50000) -> np.ndarray:
    """Solve Ax=b, directly for small systems and iteratively for large ones.

    **A singular system raises rather than falling back to least squares.** A singular
    network Laplacian means a connected component with no pressure boundary on it, which is
    a graph defect rather than a hard linear-algebra problem. The graph-level
    largest-component prune is what normally prevents it, so the two are coupled: turning
    that prune off used to route silently into a least-squares answer that looked solved.
    """
    n = A.shape[0]
    
    # Direct solver (spsolve) is very fast for small to medium systems
    if n < iterative_threshold:
        try:
            x = splinalg.spsolve(A, b)
        except Exception as exc:
            raise ValueError(
                f"The {n}x{n} system is singular and cannot be solved directly "
                f"({exc}). For a network Laplacian this means a connected component "
                f"carries no pressure boundary."
            ) from exc
        if not np.all(np.isfinite(x)):
            raise ValueError(
                f"The direct solve returned non-finite pressures for a {n}x{n} "
                f"system, so the matrix is singular: a connected component carries "
                f"no pressure boundary. Prune to the largest component, or supply "
                f"boundaries on every component. A least-squares fallback here "
                f"returns a minimum-norm field over a component nothing is driving."
            )
        return x

    # Iterative solver (CG) for massive systems to save RAM
    # Use Incomplete LU factorization as a preconditioner
    print(f"[flow-solve] Using iterative solver (CG) with ILU preconditioning for {n} variables...")
    try:
        # ilu can fail if matrix is singular, so we use a small fill_factor
        ilu = splinalg.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
        M = splinalg.LinearOperator(A.shape, ilu.solve)
        x, info = splinalg.cg(A, b, M=M, rtol=1e-8, maxiter=1000)
    except Exception as exc:
        raise ValueError(
            f"Preconditioning or the iterative solve failed on a {n}x{n} system ({exc}). "
            f"A least-squares fallback here would return a pressure field that looks "
            f"solved and is not, so the failure is reported instead."
        ) from exc
    if info != 0:
        raise ValueError(
            f"The iterative solve did not converge on a {n}x{n} system (info={info}). "
            f"A truncated solve is not a solution; raise the iteration cap or fix the "
            f"conditioning rather than accepting a least-squares answer."
        )
    return x


def calc_two_point_from_laplacian_matrix_nodeID(
    L: sp.csr_matrix, G: nx.MultiGraph, node_id1, node_id2
) -> float:
    """Effective resistance between two nodes, from the graph Laplacian.

    Grounding one node leaves every *other* connected component ungrounded, so the full
    matrix is singular whenever the graph is disconnected - which it routinely is after
    skeletonisation. That used to reach a least-squares fallback and return a finite number
    with no meaning. The solve is now restricted to the component the two nodes share, and
    two nodes in different components raise: there is no resistance between them, and
    infinity is the honest answer rather than whatever least squares produced.
    """
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    try:
        node_idx1 = node_to_idx[node_id1]
        node_idx2 = node_to_idx[node_id2]
    except KeyError as e:
        raise ValueError(f"Node {e} not found in graph")

    n = L.shape[0]
    _, labels = sp.csgraph.connected_components(L, directed=False, return_labels=True)
    if labels[node_idx1] != labels[node_idx2]:
        # Infinity is the physical answer, and an unambiguous one: no current can pass
        # between two nodes in different components. The old least-squares fallback
        # returned a finite number here that read as a real resistance.
        logger.warning(
            "Nodes %r and %r lie in different connected components, so no current can pass "
            "between them. Effective resistance is infinite.", node_id1, node_id2,
        )
        return float("inf")

    # Solve inside the shared component only; everything else is ungrounded and singular.
    member = np.flatnonzero(labels == labels[node_idx1])
    local = {g: i for i, g in enumerate(member.tolist())}
    L_sub = L[member, :][:, member].tolil()
    local1, local2 = local[node_idx1], local[node_idx2]

    b = np.zeros(len(member))
    b[local1] = 1.0
    L_sub[local2, :] = 0
    L_sub[:, local2] = 0
    L_sub[local2, local2] = 1.0

    x = _solve_system_smart(L_sub.tocsr(), b)
    return float(x[local1])


def solve_flow_from_conductance_matrix(
    conductance: sp.csr_matrix,
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
    if len(conductance.shape) != 2 or conductance.shape[0] != conductance.shape[1]:
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
        
    if "ROBIN_GHOST_NODE" in node_list and "ROBIN_GHOST_NODE" not in output_nodes:
        output_nodes.append("ROBIN_GHOST_NODE")

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

    # A connected component carrying no pressure boundary has no defined pressure: nothing
    # drives it, and its block of the Laplacian is singular. This used to reach spsolve,
    # return NaN, and be answered with a least-squares minimum-norm field - a well-formed
    # pressure map over a component nothing is driving, indistinguishable in the output from
    # a real one. Such nodes are now excluded from the solve and reported, and their
    # pressure is left as NaN so nothing downstream can read it as a number.
    n_components, labels = sp.csgraph.connected_components(
        conductance, directed=False, return_labels=True)
    supported = set(labels[known_idx].tolist()) if len(known_idx) else set()
    solvable_mask = np.array([labels[i] in supported for i in unknown_idx], dtype=bool)
    unsupported_idx = unknown_idx[~solvable_mask]
    unknown_idx = unknown_idx[solvable_mask]

    if len(unsupported_idx):
        pressure[unsupported_idx] = np.nan
        logger.warning(
            "%d of %d nodes lie in %d connected component(s) with no pressure boundary and "
            "have no defined pressure. They are excluded from the solve and their pressure "
            "is NaN. Every flow reported below is for the boundary-supported part of the "
            "network only.",
            len(unsupported_idx), n_nodes,
            len(set(labels[unsupported_idx].tolist())),
        )
    if not supported:
        raise ValueError(
            f"No connected component carries a pressure boundary: {n_components} "
            f"component(s), {len(known_idx)} boundary node(s). There is nothing to solve."
        )

    n_free = int(len(unknown_idx))
    print(
        f"[flow-solve] Solving sparse matrix with {n_nodes} nodes, {n_free} degrees of freedom..."
    )

    if n_free > 0:
        l_uu = laplacian[unknown_idx, :][:, unknown_idx]
        l_uk = laplacian[unknown_idx, :][:, known_idx]
        p_k = pressure[known_idx]
        rhs = -l_uk.dot(p_k)

        pressure[unknown_idx] = _solve_system_smart(l_uu, rhs)

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
    
    # flow = conductance * deltaP = (1/resistance) * deltaP
    flow_signed = (1.0 / edge_resistance) * pressure_drop
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


