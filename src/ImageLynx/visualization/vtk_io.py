"""VTK export and rendering for vessel graphs and constriction points."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import networkx as nx


def _as_points(path_like: Any) -> np.ndarray:
    arr = np.asarray(path_like, dtype=float)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    if arr.shape[0] < 2:
        raise ValueError("Polyline needs at least two points")
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    if arr.shape[1] < 3:
        arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
    return arr


def _edge_points(
    u: Any, v: Any, edge_data: Dict[str, Any], graph: nx.Graph
) -> np.ndarray:
    voxels = edge_data.get("voxels")
    if voxels is not None and len(voxels) >= 2:
        return _as_points(voxels)

    u_pos = graph.nodes[u].get("pos")
    v_pos = graph.nodes[v].get("pos")
    if u_pos is None or v_pos is None:
        raise ValueError(f"Edge ({u}, {v}) is missing both voxels and node positions")
    return _as_points([u_pos, v_pos])


def _snap_edge_endpoints_to_nodes(
    points: np.ndarray, u: Any, v: Any, graph: nx.Graph
) -> np.ndarray:
    """Force edge endpoints to match node positions when available."""
    out = points.copy()
    u_pos = graph.nodes[u].get("pos")
    v_pos = graph.nodes[v].get("pos")
    if u_pos is not None:
        out[0] = np.asarray(u_pos, dtype=float)[:3]
    if v_pos is not None:
        out[-1] = np.asarray(v_pos, dtype=float)[:3]
    return out


def _orient_edge_points_to_nodes(
    points: np.ndarray, u: Any, v: Any, graph: nx.Graph
) -> np.ndarray:
    """Orient polyline so first point maps best to u and last to v."""
    u_pos = graph.nodes[u].get("pos")
    v_pos = graph.nodes[v].get("pos")
    if u_pos is None or v_pos is None or len(points) < 2:
        return points

    u_arr = np.asarray(u_pos, dtype=float)[:3]
    v_arr = np.asarray(v_pos, dtype=float)[:3]
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)

    direct_cost = float(np.linalg.norm(start - u_arr) + np.linalg.norm(end - v_arr))
    flipped_cost = float(np.linalg.norm(start - v_arr) + np.linalg.norm(end - u_arr))
    if flipped_cost < direct_cost:
        return points[::-1].copy()
    return points


def _point_key(point: np.ndarray, decimals: int) -> tuple[float, float, float]:
    rounded = np.round(np.asarray(point, dtype=float), decimals=decimals)
    return (float(rounded[0]), float(rounded[1]), float(rounded[2]))


def _compress_consecutive_duplicate_ids(ids: List[int]) -> List[int]:
    if not ids:
        return ids
    out = [ids[0]]
    for idx in ids[1:]:
        if idx != out[-1]:
            out.append(idx)
    return out


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def _interpolate_at_length(points: np.ndarray, cumlen: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
        return points[0]
    if s >= cumlen[-1]:
        return points[-1]
    idx = int(np.searchsorted(cumlen, s, side="right")) - 1
    idx = max(0, min(idx, len(points) - 2))
    l0, l1 = cumlen[idx], cumlen[idx + 1]
    if l1 <= l0:
        return points[idx]
    t = (s - l0) / (l1 - l0)
    return points[idx] + t * (points[idx + 1] - points[idx])


def derive_pericyte_points_from_graph(
    graph: nx.Graph,
    *,
    constriction_spacing: float = 100.0,
    constriction_length: float = 40.0,
) -> Dict[str, Any]:
    """Derive pericyte (constriction center) points along each edge path.

    Pericyte centers are placed at `constriction_length/2 + k*constriction_spacing`.
    """
    if constriction_spacing <= 0 or constriction_length <= 0:
        raise ValueError("constriction_spacing and constriction_length must be > 0")

    points: List[np.ndarray] = []
    edge_u: List[int] = []
    edge_v: List[int] = []
    edge_key: List[int] = []
    branch_order: List[str] = []

    is_multigraph = isinstance(graph, nx.MultiGraph)
    edge_iter: Iterable[Any]
    if is_multigraph:
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    first_center = constriction_length / 2.0
    for u, v, k, data in edge_iter:
        try:
            line_pts = _edge_points(u, v, data, graph)
        except ValueError:
            continue
        cumlen = _cumulative_lengths(line_pts)
        length = float(cumlen[-1])
        if length <= 0:
            continue

        sample_pos = first_center
        while sample_pos <= length:
            pt = _interpolate_at_length(line_pts, cumlen, sample_pos)
            points.append(pt)
            edge_u.append(int(u) if isinstance(u, (int, np.integer)) else -1)
            edge_v.append(int(v) if isinstance(v, (int, np.integer)) else -1)
            edge_key.append(int(k) if isinstance(k, (int, np.integer)) else 0)
            branch_order.append(str(data.get("branch_order", "No_BO")))
            sample_pos += constriction_spacing

    return {
        "points": np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float),
        "edge_u": np.asarray(edge_u, dtype=int),
        "edge_v": np.asarray(edge_v, dtype=int),
        "edge_key": np.asarray(edge_key, dtype=int),
        "branch_order": np.asarray(branch_order, dtype=object),
    }


#: Categorical provenance tags and their level order. ParaView cannot colour by a string
#: array, so each tag is written twice: the string to read, and an integer code to colour by.
#: An unrecognised level codes to -1 rather than silently joining an existing class.
_PROVENANCE_LEVELS = {
    "diameter_provenance": ("measured_edt", "measured_fwhm", "constant",
                            "synthetic_branch_order"),
    "edt_junction_trim": ("trimmed", "untrimmed_too_short", "no_junction", "not_applied"),
    "centreline_smoothing": ("bspline", "bspline_relaxed", "raw_fallback", "raw_too_short"),
}

#: Numeric per-edge columns worth carrying onto the geometry.
_MORPHOMETRY_COLUMNS = ("length_um", "euclidean_um", "tortuosity", "curvature",
                        "edt_diameter_um", "n_centreline_points")


def _attach_morphometry(mesh, graph, cell_keys) -> None:
    """Put the quantities H1 reports onto the geometry they were measured from.

    The exported geometry previously carried an assigned diameter and a branch order and
    nothing else, so the network could not be coloured by tortuosity, by the EDT diameter
    section 1.2 reports, or by which edges retained a radius known to be biased.

    Values come from ``export_per_edge_morphometry`` rather than being recomputed here, so
    the definition of tortuosity in a rendering cannot drift from the definition in the
    exported table.
    """
    import numpy as np

    from ..statistics.stats import export_per_edge_morphometry

    try:
        rows = export_per_edge_morphometry(graph)
    except Exception:
        return
    lookup = {(r["u"], r["v"], r["key"]): r for r in rows}
    ordered = [lookup.get(key) for key in cell_keys]

    for column in _MORPHOMETRY_COLUMNS:
        if column in mesh.cell_data:
            continue
        values = []
        for row in ordered:
            value = row.get(column) if row else None
            values.append(float(value) if value is not None else np.nan)
        mesh.cell_data[column] = np.asarray(values, dtype=float)

    for tag, levels in _PROVENANCE_LEVELS.items():
        text = [str(row.get(tag)) if row and row.get(tag) is not None else "" for row in ordered]
        width = max(len(level) for level in levels)
        mesh.cell_data[tag] = np.asarray(text, dtype=f"<U{width}")
        index = {level: position for position, level in enumerate(levels)}
        mesh.cell_data[f"{tag}_code"] = np.asarray(
            [index.get(value, -1) for value in text], dtype=np.int8)

    mesh.cell_data["reconnected"] = np.asarray(
        [1 if row and row.get("reconnected") else 0 for row in ordered], dtype=np.int8)


def graph_to_vtk(
    graph: nx.Graph,
    output_prefix: str | Path,
    *,
    constriction_spacing: float = 100.0,
    constriction_length: float = 40.0,
    point_merge_decimals: int = 6,
) -> Dict[str, Any]:
    """Export vessels, pericytes, and nodes to VTK PolyData files."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for VTK export. Install with `pip install pyvista`.") from exc

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    vessel_path = output_prefix.with_name(output_prefix.name + "_vessels.vtp")
    pericyte_path = output_prefix.with_name(output_prefix.name + "_pericytes.vtp")
    node_path = output_prefix.with_name(output_prefix.name + "_nodes.vtp")

    all_points: List[np.ndarray] = []
    line_cells: List[int] = []
    point_index: Dict[tuple[float, float, float], int] = {}
    edge_u: List[int] = []
    edge_v: List[int] = []
    edge_key: List[int] = []
    branch_order: List[str] = []
    resistances: List[float] = []
    assigned_diameters: List[float] = []
    fwhm_diameters: List[float] = []

    is_multigraph = isinstance(graph, nx.MultiGraph)
    edge_iter: Iterable[Any]
    if is_multigraph:
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for u, v, k, data in edge_iter:
        try:
            pts = _edge_points(u, v, data, graph)
        except ValueError:
            continue
        pts = _orient_edge_points_to_nodes(pts, u, v, graph)
        pts = _snap_edge_endpoints_to_nodes(pts, u, v, graph)

        edge_ids: List[int] = []
        for p in pts:
            key = _point_key(p, point_merge_decimals)
            idx = point_index.get(key)
            if idx is None:
                idx = len(all_points)
                all_points.append(np.asarray(p, dtype=float))
                point_index[key] = idx
            edge_ids.append(idx)

        edge_ids = _compress_consecutive_duplicate_ids(edge_ids)
        if len(edge_ids) < 2:
            continue

        line_cells.append(len(edge_ids))
        line_cells.extend(edge_ids)
        edge_u.append(int(u) if isinstance(u, (int, np.integer)) else -1)
        edge_v.append(int(v) if isinstance(v, (int, np.integer)) else -1)
        edge_key.append(int(k) if isinstance(k, (int, np.integer)) else 0)
        branch_order.append(str(data.get("branch_order", "No_BO")))
        
        r = data.get("resistance")
        resistances.append(float(r) if r is not None else np.nan)
        
        ad = data.get("assigned_diameter_um")
        assigned_diameters.append(float(ad) if ad is not None else np.nan)

        fd = data.get("fwhm_diameter_um")
        fwhm_diameters.append(float(fd) if fd is not None else np.nan)

    vessel_mesh = pv.PolyData()
    vessel_mesh.points = np.asarray(all_points, dtype=float) if all_points else np.empty((0, 3), dtype=float)
    if line_cells:
        vessel_mesh.lines = np.asarray(line_cells, dtype=np.int64)
        vessel_mesh.cell_data["edge_u"] = np.asarray(edge_u, dtype=np.int64)
        vessel_mesh.cell_data["edge_v"] = np.asarray(edge_v, dtype=np.int64)
        vessel_mesh.cell_data["edge_key"] = np.asarray(edge_key, dtype=np.int64)
        vessel_mesh.cell_data["branch_order"] = np.asarray(branch_order)
        vessel_mesh.cell_data["resistance"] = np.asarray(resistances, dtype=float)
        vessel_mesh.cell_data["assigned_diameter_um"] = np.asarray(assigned_diameters, dtype=float)
        vessel_mesh.cell_data["fwhm_diameter_um"] = np.asarray(fwhm_diameters, dtype=float)
        _attach_morphometry(vessel_mesh, graph, list(zip(edge_u, edge_v, edge_key)))
    vessel_mesh.save(vessel_path)

    pericyte = derive_pericyte_points_from_graph(
        graph,
        constriction_spacing=constriction_spacing,
        constriction_length=constriction_length,
    )
    pericyte_mesh = pv.PolyData(pericyte["points"])
    pericyte_mesh.point_data["edge_u"] = pericyte["edge_u"]
    pericyte_mesh.point_data["edge_v"] = pericyte["edge_v"]
    pericyte_mesh.point_data["edge_key"] = pericyte["edge_key"]
    pericyte_mesh.point_data["branch_order"] = pericyte["branch_order"]
    pericyte_mesh.save(pericyte_path)

    node_ids: List[int] = []
    node_points: List[np.ndarray] = []
    for n, data in graph.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            continue
        node_points.append(np.asarray(pos, dtype=float)[:3])
        node_ids.append(int(n) if isinstance(n, (int, np.integer)) else -1)

    node_mesh = pv.PolyData(
        np.asarray(node_points, dtype=float) if node_points else np.empty((0, 3), dtype=float)
    )
    node_mesh.point_data["node_id"] = np.asarray(node_ids, dtype=np.int64)
    # Section 1.1 counts branch points - nodes joining three or more distinct segments - so
    # degree is the readout, and a node file carrying only an identifier cannot show it.
    degrees = np.asarray([int(graph.degree(n)) for n in node_ids], dtype=np.int32)
    node_mesh.point_data["degree"] = degrees
    node_mesh.point_data["is_branch_node"] = (degrees >= 3).astype(np.int8)
    node_mesh.point_data["is_endpoint"] = (degrees == 1).astype(np.int8)
    node_mesh.save(node_path)

    return {
        "vessels_path": str(vessel_path),
        "pericytes_path": str(pericyte_path),
        "nodes_path": str(node_path),
        "vessel_line_count": len(edge_u),
        "pericyte_count": int(len(pericyte["points"])),
        "node_count": len(node_ids),
    }


def visualize_vtk_network(
    vessels_path: str | Path,
    pericytes_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    *,
    vessel_color: str = "dodgerblue",
    pericyte_color: str = "tomato",
    node_color: str = "yellow",
    vessel_line_width: float = 2.0,
    pericyte_size: float = 10.0,
    node_size: float = 8.0,
    show_nodes: bool = False,
    show: bool = True,
) -> Any:
    """Visualize exported network VTK files using PyVista."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for VTK visualization. Install with `pip install pyvista`.") from exc

    vessels = pv.read(str(vessels_path))
    pericytes = pv.read(str(pericytes_path)) if pericytes_path else None
    nodes = pv.read(str(nodes_path)) if nodes_path else None

    plotter = pv.Plotter()
    plotter.add_mesh(vessels, color=vessel_color, line_width=vessel_line_width)
    if pericytes is not None and pericytes.n_points > 0:
        plotter.add_mesh(
            pericytes,
            color=pericyte_color,
            point_size=pericyte_size,
            render_points_as_spheres=True,
        )
    if show_nodes and nodes is not None and nodes.n_points > 0:
        plotter.add_mesh(
            nodes,
            color=node_color,
            point_size=node_size,
            render_points_as_spheres=True,
        )
    if show:
        plotter.show()
    return plotter


def load_vtp(filepath: str | Path) -> Any:
    """Load a VTK PolyData file (.vtp) using PyVista."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required to load VTK files. "
            "Install with `pip install pyvista`."
        ) from exc
    return pv.read(str(filepath))

def export_perfusion_grid_to_vti(
    grid, 
    concentration_array: np.ndarray, 
    output_path: str | Path,
    array_name: str = "O2_Concentration"
) -> str:
    """
    Export the 3D perfusion grid and its calculated concentration to a VTK ImageData (.vti) file.
    
    Args:
        grid: PerfusionGrid object containing dims, res, and min_xyz
        concentration_array: 1D numpy array of solved concentrations
        output_path: Where to save the .vti file
        array_name: Name of the scalar field in ParaView
        
    Returns:
        str: Path to the saved file
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for VTK export. Install with `pip install pyvista`.") from exc

    output_path = Path(output_path)
    if output_path.suffix != ".vti":
        output_path = output_path.with_suffix(".vti")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PyVista dimensions are nodes (points). Cell dimensions are dims - 1.
    # Since we want our data on the cells (blocks), point dimensions = cell dimensions + 1
    nx_dim, ny_dim, nz_dim = grid.dims
    
    vti = pv.ImageData()
    vti.dimensions = (nx_dim + 1, ny_dim + 1, nz_dim + 1)
    vti.spacing = tuple(grid.res)
    vti.origin = tuple(grid.min_xyz)
    
    # Reshape the 1D array to 3D. 
    # Our linear index was: idx = x + y*nx + z*nx*ny
    # This corresponds to Fortran order 'F' when reshaping.
    volume_data = concentration_array.reshape((nx_dim, ny_dim, nz_dim), order='F')
    
    # PyVista expects cell data to be flattened in Fortran order as well
    vti.cell_data[array_name] = volume_data.flatten(order='F')
    
    vti.save(str(output_path))
    return str(output_path)
