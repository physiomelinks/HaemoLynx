"""VTK export and rendering for vessel graphs and constriction points."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import networkx as nx

from haemolynx.geometry import cumulative_lengths
from haemolynx.visualization.geometry import edge_polyline

#: The unit of every array the VTK files carry, spelled as the statistics
#: glossary spells units ("-" is an identifier, label or text; "flag" is 0 or
#: 1). Each file's field data repeats it as ``<array>_unit``, beside
#: ``points_unit`` and ``points_axis_order`` for the points themselves.
ARRAY_UNITS: Dict[str, str] = {
    "edge_u": "-",
    "edge_v": "-",
    "edge_key": "-",
    "branch_order": "-",
    "resistance": "Pa.s/m^3",
    "conductance": "m^3/(Pa.s)",
    "length_um": "um",
    "diameter_um": "um",
    "diameter_source": "-",
    "is_zero_resistance": "flag",
    "flow_solved": "flag",
    "flow_state": "-",
    "cut_by_image_face": "flag",
    "discharge_haematocrit": "fraction",
    "pressure_u": "Pa",
    "pressure_v": "Pa",
    "pressure_drop": "Pa",
    "flow_signed": "m^3/s",
    "flow_abs": "m^3/s",
    "node_id": "-",
    "dead_end_kind": "-",
    "boundary_role": "-",
    "pressure": "Pa",
}

#: A node's ``boundary_role`` in the nodes file: a designated inlet or
#: outlet, an open end leaving the image that the solve gave a condition and
#: that feeds the network or drains it (``haemodynamics.open_ends.open_end_directions``),
#: one given a condition that carries no flow, or "" for every other node.
BOUNDARY_ROLES = ("inlet", "outlet", "open_end_feeds", "open_end_drains", "open_end", "")

#: The vessels file's field data beyond the units: the HaemoLynx version and
#: what produced its numbers -- the blood model and the boundaries, from the
#: graph a run solved, each written once the graph records it.
RUN_FIELD_DATA = (
    "haemolynx_version",
    "viscosity_law",
    "diameter_basis",
    "haematocrit",
    "haematocrit_model",
    "open_end_model",
)

#: Every point of every file HaemoLynx writes is physical (z, y, x) microns.
POINTS_UNIT = "um"
POINTS_AXIS_ORDER = "z, y, x"


def _write_units(mesh: Any, arrays: Iterable[str]) -> None:
    """Field data naming the unit of each of *arrays* and of the points."""
    mesh.field_data["points_unit"] = np.asarray([POINTS_UNIT])
    mesh.field_data["points_axis_order"] = np.asarray([POINTS_AXIS_ORDER])
    for name in arrays:
        mesh.field_data[f"{name}_unit"] = np.asarray([ARRAY_UNITS[name]])


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
            line_pts = edge_polyline(graph, u, v, data, orient=False, snap=False)
        except ValueError:
            continue
        cumlen = cumulative_lengths(line_pts)
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


def graph_to_vtk(
    graph: nx.Graph,
    output_prefix: str | Path,
    *,
    constriction_spacing: float = 100.0,
    constriction_length: float = 40.0,
    point_merge_decimals: int = 6,
    inlet_nodes: Iterable[Any] = (),
    outlet_nodes: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Export vessels, pericytes, and nodes to VTK PolyData files.

    Each vessel carries its ``length_um``, ``diameter_um``, ``diameter_source``,
    ``is_zero_resistance``, ``flow_solved`` and ``flow_state`` (why it carries
    flow or not, ``graph.FLOW_STATE``) beside its resistance and flows; where
    its flow is no answer at all (``graph.flow_is_undefined``: a piece no
    inlet and outlet reach) every array of the solve's
    (``graph.SOLVE_DERIVED_EDGE_ATTRIBUTES``: flows, pressures, discharge
    haematocrit) is NaN, not the 0 the solve left. Each node carries its
    ``dead_end_kind`` ("" where it is not a dead end) and a NaN pressure
    where the solve could not set one. The run's
    blood model and boundaries (viscosity law, diameter basis, haematocrit,
    haematocrit model, open-end model) and the HaemoLynx version go in the
    vessels file's field data (:data:`RUN_FIELD_DATA`).

    Each vessel also says whether the image faces cut it
    (``cut_by_image_face``: on the chain of a dead end leaving the image, its
    length censored), and each node its ``boundary_role``
    (:data:`BOUNDARY_ROLES`): *inlet_nodes* and *outlet_nodes* are the run's
    designated ones, the open ends come from the solve's record on *graph*.
    """
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
    conductances: List[float] = []
    lengths: List[float] = []
    diameters: List[float] = []
    diameter_sources: List[str] = []
    zero_resistance: List[int] = []
    solved_flags: List[int] = []
    flow_states: List[str] = []
    cut_flags: List[int] = []
    from haemolynx.graph.dead_ends import vessels_cut_by_image_faces
    from haemolynx.graph._helpers import edge_id
    from haemolynx.graph.prune import (
        FLOW_SOLVED,
        FLOW_STATE,
        SOLVE_DERIVED_EDGE_ATTRIBUTES,
        flow_is_undefined,
    )
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    #: The solve's attributes this file carries; absent until a run has solved.
    flow_fields: Dict[str, List[float]] = {
        name: [] for name in SOLVE_DERIVED_EDGE_ATTRIBUTES if name in ARRAY_UNITS
    }
    cut = vessels_cut_by_image_faces(graph)

    is_multigraph = isinstance(graph, nx.MultiGraph)
    edge_iter: Iterable[Any]
    if is_multigraph:
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for u, v, k, data in edge_iter:
        try:
            pts = edge_polyline(graph, u, v, data)
        except ValueError:
            continue

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
        c = data.get("conductance")
        resistances.append(float(r) if r is not None else np.nan)
        conductances.append(float(c) if c is not None else np.nan)
        lengths.append(_number(data.get("length")))
        diameters.append(_number(data.get("diameter_um")))
        diameter_sources.append(str(data.get("diameter_source") or ""))
        zero_resistance.append(int(bool(data.get(IS_ZERO_RESISTANCE))))
        solved_flags.append(int(data.get(FLOW_SOLVED) is True))
        flow_states.append(str(data.get(FLOW_STATE) or ""))
        cut_flags.append(int(edge_id(u, v, k) in cut))
        undefined = flow_is_undefined(data)
        for name, values in flow_fields.items():
            values.append(np.nan if undefined else _number(data.get(name)))

    vessel_mesh = pv.PolyData()
    vessel_mesh.points = np.asarray(all_points, dtype=float) if all_points else np.empty((0, 3), dtype=float)
    if line_cells:
        vessel_mesh.lines = np.asarray(line_cells, dtype=np.int64)
        vessel_mesh.cell_data["edge_u"] = np.asarray(edge_u, dtype=np.int64)
        vessel_mesh.cell_data["edge_v"] = np.asarray(edge_v, dtype=np.int64)
        vessel_mesh.cell_data["edge_key"] = np.asarray(edge_key, dtype=np.int64)
        vessel_mesh.cell_data["branch_order"] = np.asarray(branch_order)
        vessel_mesh.cell_data["resistance"] = np.asarray(resistances, dtype=float)
        vessel_mesh.cell_data["conductance"] = np.asarray(conductances, dtype=float)
        vessel_mesh.cell_data["length_um"] = np.asarray(lengths, dtype=float)
        vessel_mesh.cell_data["diameter_um"] = np.asarray(diameters, dtype=float)
        vessel_mesh.cell_data["diameter_source"] = np.asarray(diameter_sources)
        vessel_mesh.cell_data["is_zero_resistance"] = np.asarray(zero_resistance, dtype=np.uint8)
        vessel_mesh.cell_data["flow_solved"] = np.asarray(solved_flags, dtype=np.uint8)
        vessel_mesh.cell_data["flow_state"] = np.asarray(flow_states)
        vessel_mesh.cell_data["cut_by_image_face"] = np.asarray(cut_flags, dtype=np.uint8)
        for name, values in flow_fields.items():
            array = np.asarray(values, dtype=float)
            if np.any(np.isfinite(array)):
                vessel_mesh.cell_data[name] = array
    for name, value in _run_metadata(graph).items():
        vessel_mesh.field_data[name] = np.asarray([value])
    _write_units(vessel_mesh, vessel_mesh.cell_data.keys())
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
    _write_units(pericyte_mesh, pericyte_mesh.point_data.keys())
    pericyte_mesh.save(pericyte_path)

    from haemolynx.haemodynamics.open_ends import (
        DRAINS,
        FEEDS,
        OpenEndConditions,
        open_end_directions,
    )

    inlets, outlets = set(inlet_nodes), set(outlet_nodes)
    given_a_condition = set(OpenEndConditions.from_record(graph.graph.get("open_end_conditions")).nodes)
    directions = open_end_directions(graph)

    def boundary_role(node: Any) -> str:
        if node in inlets:
            return "inlet"
        if node in outlets:
            return "outlet"
        if node in given_a_condition:
            return {FEEDS: "open_end_feeds", DRAINS: "open_end_drains"}.get(directions.get(node), "open_end")
        return ""

    node_ids: List[int] = []
    node_points: List[np.ndarray] = []
    node_kinds: List[str] = []
    node_roles: List[str] = []
    node_pressures: List[float] = []
    for n, data in graph.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            continue
        node_points.append(np.asarray(pos, dtype=float)[:3])
        node_ids.append(int(n) if isinstance(n, (int, np.integer)) else -1)
        node_kinds.append(str(data.get("dead_end_kind") or "") if graph.degree(n) == 1 else "")
        node_roles.append(boundary_role(n))
        incident = [d for *_e, d in graph.edges(n, data=True)]
        undefined = bool(incident) and all(flow_is_undefined(d) for d in incident)
        node_pressures.append(np.nan if undefined else _number(data.get("pressure")))

    node_mesh = pv.PolyData(
        np.asarray(node_points, dtype=float) if node_points else np.empty((0, 3), dtype=float)
    )
    node_mesh.point_data["node_id"] = np.asarray(node_ids, dtype=np.int64)
    node_mesh.point_data["dead_end_kind"] = np.asarray(node_kinds)
    node_mesh.point_data["boundary_role"] = np.asarray(node_roles)
    if np.any(np.isfinite(node_pressures)):
        node_mesh.point_data["pressure"] = np.asarray(node_pressures, dtype=float)
    _write_units(node_mesh, node_mesh.point_data.keys())
    node_mesh.save(node_path)

    return {
        "vessels_path": str(vessel_path),
        "pericytes_path": str(pericyte_path),
        "nodes_path": str(node_path),
        "vessel_line_count": len(edge_u),
        "pericyte_count": int(len(pericyte["points"])),
        "node_count": len(node_ids),
    }


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _run_metadata(graph: nx.Graph) -> Dict[str, Any]:
    """What produced the numbers in a vessels file: the blood model, the
    boundaries and the version, from the graph a run solved."""
    from haemolynx import __version__

    meta: Dict[str, Any] = {"haemolynx_version": str(__version__)}
    for name in ("viscosity_law", "diameter_basis", "haematocrit", "haematocrit_model"):
        if graph.graph.get(name) is not None:
            meta[name] = graph.graph[name]
    record = graph.graph.get("open_end_conditions")
    if record:
        meta["open_end_model"] = str(record.get("model", "closed"))
    return meta


def surface_mesh_to_vtk(
    vertices: np.ndarray, faces: np.ndarray, output_path: str | Path
) -> Path:
    """Write a triangle mesh -- the tissue surface -- as VTK PolyData.

    *vertices* are written as given: physical ``(z, y, x)`` microns, the order
    :func:`graph_to_vtk` writes node ``pos`` in, so the two overlay.
    """
    import pyvista as pv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=float).reshape(-1, 3), cells)
    _write_units(mesh, ())
    mesh.save(output_path)
    return output_path


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
