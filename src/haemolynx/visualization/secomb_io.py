"""A solved network in Secomb's ``Network.dat`` format, and back.

Tim Secomb's reference codes -- NetFlow (github.com/secomb/NetFlowV2: flow
with the in vivo viscosity law and phase separation) and FlowEstimate
(github.com/secomb/FlowEstimateV1: Fry et al. 2012's estimate of unknown
boundary conditions) -- read a network as one text file: segments (two
nodes, a diameter), nodes (x, y, z) and boundary nodes (a pressure or a
flow). :func:`write_secomb_network` writes a HaemoLynx graph that way, so the
same network can be solved by them; :func:`read_secomb_network` reads what
they write back (``NetworkNew.dat``).

Neither program reads a length: each takes a segment's length as the
straight-line distance between its nodes. So each vessel is written as a chain
of segments along its centreline, bent so the chain is exactly the vessel's
``length`` long (the vessel's diameter on every segment of it), and its
resistance in those programs is the one HaemoLynx gives it. A zero-resistance
bridge (``IS_ZERO_RESISTANCE``) cannot be written as one: a segment as short as
its resistance needs could not reach its far node. It is written as a vessel of
its own diameter and length, and :attr:`SecombExport.bridges_as_vessels`
counts them; solve the graph the same way before comparing.

Units, as Secomb's codes take them: diameters and coordinates in microns,
pressures in mmHg, flows in nl/min. A node's ``pos`` is ``(z, y, x)``; the file
holds ``x y z``.

Boundary nodes must each be on one segment, as both programs require. A
designated inlet or outlet is written as a pressure; every other dead end as a
known flow of zero (closed), so NetFlow and FlowEstimate both read it as
closed -- except those passed as *unknown_boundary_nodes*, which are left out
of the list: FlowEstimate estimates a condition there, and NetFlow, which needs
a complete set, would close them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.geometry import cumulative_lengths, resample_at_step

from .geometry import edge_polyline

__all__ = [
    "BC_FLOW",
    "BC_PRESSURE",
    "NL_PER_MIN_PER_M3_S",
    "PA_PER_MMHG",
    "SecombExport",
    "SecombNetwork",
    "read_secomb_network",
    "write_secomb_network",
]

#: One mmHg in Pa. NetFlow's own flow factor (``facfp`` in ``input.cpp``)
#: takes 1333 dyn/cm^2, 0.017% less; pass ``pa_per_mmhg=133.3`` to write the
#: pressures it will turn back into exactly the same flows.
PA_PER_MMHG = 133.322
#: m^3/s -> nl/min (1 nl = 1e-12 m^3).
NL_PER_MIN_PER_M3_S = 6.0e13
#: Boundary types, the file's ``bctyp`` column.
BC_PRESSURE = 0
BC_FLOW = 2
#: Segment type: NetFlow includes only segments of type 4 or 5.
SEGMENT_TYPE = 5
#: Digits after the point for coordinates and diameters. NetFlow reads them
#: into single precision, which holds about seven significant figures.
_DECIMALS = 6


@dataclass
class SecombExport:
    """What :func:`write_secomb_network` wrote, to match results back."""

    path: Path
    #: Graph node -> the node's name in the file.
    node_names: dict = field(default_factory=dict)
    #: ``(u, v, key)`` -> the names of its segments, in order from u to v; a
    #: flow read back on any of them is the vessel's, positive from u to v.
    edge_segments: dict = field(default_factory=dict)
    #: ``(u, v, key)`` -> the length of its chain as written (um), from the
    #: rounded coordinates, which is what NetFlow measures.
    edge_lengths_um: dict = field(default_factory=dict)
    #: Edges left out, by reason: ``self_loop``, ``no_conductance``.
    skipped: dict = field(default_factory=dict)
    #: Zero-resistance bridges written as vessels (see the module docstring).
    bridges_as_vessels: int = 0
    #: Edges whose ``length`` is shorter than the distance between their
    #: nodes, written straight (so longer than ``length``).
    shorter_than_chord: int = 0
    pressure_nodes: dict = field(default_factory=dict)
    closed_nodes: list = field(default_factory=list)
    unknown_nodes: list = field(default_factory=list)


@dataclass
class SecombNetwork:
    """A ``Network.dat`` (or ``NetworkNew.dat``) as read."""

    #: segment name -> ``(type, from_node, to_node, diameter_um, flow_nl_min, hd)``
    segments: dict
    #: node name -> ``(x, y, z)`` in um
    nodes: dict
    #: node name -> ``(bctyp, pressure_mmHg_or_flow_nl_min, hd)``
    boundary: dict


def _edges(G: nx.Graph):
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, 0, data


def _fit_length(points: np.ndarray, target: Optional[float]) -> tuple[np.ndarray, bool]:
    """*points* bent so the polyline from its first to its last point is
    *target* long, the ends fixed: each point's offset from the straight line
    between the ends (taken at its share of the arc length) is scaled until
    the length is right. A polyline with no offset gets a tent-shaped one.
    Returns the points and whether *target* was shorter than the straight
    line, which (its points laid along it) is then what is returned."""
    a, b = points[0], points[-1]
    chord = b - a
    c = float(np.linalg.norm(chord))
    if target is None or not np.isfinite(target) or target <= 0:
        return points, False
    if len(points) == 2 and target > c * (1.0 + 1e-12):
        points = np.vstack([a, (a + b) / 2.0, b])
    arc = cumulative_lengths(points)
    share = arc / arc[-1] if arc[-1] > 0 else np.linspace(0.0, 1.0, len(points))
    base = a + np.outer(share, chord)
    if target <= c * (1.0 + 1e-12):
        return base, target < c * (1.0 - 1e-9)
    offset = points - base
    if float(np.max(np.linalg.norm(offset, axis=1))) <= 1e-9 * target:
        axis = chord / c if c > 0 else np.array([1.0, 0.0, 0.0])
        normal = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(axis, [0.0, 1.0, 0.0])
        normal /= np.linalg.norm(normal)
        offset = np.outer(np.minimum(share, 1.0 - share), normal)

    def length(scale: float) -> float:
        return float(cumulative_lengths(base + scale * offset)[-1])

    low, high = 0.0, 1.0
    for _ in range(200):
        if length(high) >= target:
            break
        low, high = high, high * 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if length(middle) < target:
            low = middle
        else:
            high = middle
        if high - low <= 1e-15 * max(high, 1.0):
            break
    return base + high * offset, False


def _rounded(points: np.ndarray) -> np.ndarray:
    """The points as the file will hold them, without repeats."""
    points = np.round(np.asarray(points, dtype=float), _DECIMALS)
    keep = [0]
    for i in range(1, len(points) - 1):
        if np.any(points[i] != points[keep[-1]]) and np.any(points[i] != points[-1]):
            keep.append(i)
    keep.append(len(points) - 1)
    return points[keep]


def _xyz(point_zyx: np.ndarray) -> str:
    z, y, x = (float(c) for c in point_zyx)
    return f"{x:.{_DECIMALS}f} {y:.{_DECIMALS}f} {z:.{_DECIMALS}f}"


def write_secomb_network(
    G: nx.Graph,
    path,
    *,
    inlet_nodes: Sequence[Any],
    outlet_nodes: Sequence[Any],
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    boundary_haematocrit: float = 0.45,
    unknown_boundary_nodes: Iterable[Any] = (),
    max_segment_um: Optional[float] = 10.0,
    pa_per_mmhg: float = PA_PER_MMHG,
    order_by_flow: bool = False,
    title: str = "HaemoLynx network",
) -> SecombExport:
    """Write *G* as Secomb's ``Network.dat`` (see the module docstring).

    Every edge with a conductance is written, as a chain of segments at most
    *max_segment_um* long (None: one per centreline point) along its
    ``voxels``, with its ``diameter_um``, the chain bent to its ``length``.
    Inlets and outlets are held at *inlet_pressure_pa* / *outlet_pressure_pa*
    and every inflow there carries *boundary_haematocrit*, which is also each
    segment's starting haematocrit. Each segment's flow column is its edge's
    ``flow_signed`` in nl/min (0 without one): NetFlow ignores it, FlowEstimate
    only counts the vessels whose direction its estimate reverses.

    *order_by_flow* lists the vessels largest ``|flow_signed|`` first. NetFlow's
    ``dishem_generalized`` divides a junction's outflows in the order the file
    lists them; HaemoLynx's ``sequential_bifurcations`` takes them largest flow
    first. Ordered so, the two take them alike wherever this flow's order holds.

    Raises ``ValueError`` when an inlet, outlet or unknown boundary node is not
    on exactly one written segment, or a written edge has no ``diameter_um``.
    """
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    path = Path(path)
    export = SecombExport(path=path)
    written = []
    for u, v, key, data in _edges(G):
        if u == v:
            export.skipped["self_loop"] = export.skipped.get("self_loop", 0) + 1
            continue
        conductance = data.get("conductance")
        if conductance is None or not float(conductance) > 0:
            export.skipped["no_conductance"] = export.skipped.get("no_conductance", 0) + 1
            continue
        diameter = data.get("diameter_um")
        if diameter is None or not float(diameter) > 0:
            raise ValueError(f"Edge ({u}, {v}, {key}) has a conductance but no diameter_um to write.")
        written.append((u, v, key, data))
    if order_by_flow:
        written.sort(key=lambda e: -abs(float(e[3].get("flow_signed") or 0.0)))

    degree: dict = {}
    for u, v, _key, _data in written:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    unknown = list(dict.fromkeys(unknown_boundary_nodes))
    designated = {n: inlet_pressure_pa for n in inlet_nodes}
    designated.update({n: outlet_pressure_pa for n in outlet_nodes})
    both = [n for n in unknown if n in designated]
    if both:
        raise ValueError(f"Nodes both designated and unknown boundary nodes: {both}")
    not_ends = [n for n in (*designated, *unknown) if degree.get(n) != 1]
    if not_ends:
        raise ValueError(
            f"Boundary nodes must each be on one vessel for NetFlow and FlowEstimate; "
            f"these are on {[degree.get(n, 0) for n in not_ends]}: {not_ends}"
        )

    node_lines: list[str] = []
    positions: list[np.ndarray] = []

    def name_of(node) -> int:
        if node not in export.node_names:
            pos = G.nodes[node].get("pos")
            if pos is None:
                raise ValueError(f"Node {node} has no pos to write.")
            point = np.round(np.asarray(pos, dtype=float)[:3], _DECIMALS)
            export.node_names[node] = len(node_lines) + 1
            node_lines.append(f"{len(node_lines) + 1} {_xyz(point)}")
            positions.append(point)
        return export.node_names[node]

    for u, v, _key, _data in written:
        name_of(u), name_of(v)

    segment_lines: list[str] = []
    longest = 0.0
    for u, v, key, data in written:
        if data.get(IS_ZERO_RESISTANCE):
            export.bridges_as_vessels += 1
        points = edge_polyline(G, u, v, data)
        points[0] = positions[export.node_names[u] - 1]
        points[-1] = positions[export.node_names[v] - 1]
        if max_segment_um:
            points = resample_at_step(points, float(max_segment_um))
        target = data.get("length")
        points, short = _fit_length(points, None if target is None else float(target))
        export.shorter_than_chord += int(short)
        points = _rounded(points)
        chain = [export.node_names[u]]
        for point in points[1:-1]:
            node_lines.append(f"{len(node_lines) + 1} {_xyz(point)}")
            positions.append(point)
            chain.append(len(node_lines))
        chain.append(export.node_names[v])
        pieces = np.linalg.norm(np.diff(points, axis=0), axis=1)
        longest = max(longest, float(pieces.max()))
        export.edge_lengths_um[(u, v, key)] = float(pieces.sum())
        flow = float(data.get("flow_signed") or 0.0) * NL_PER_MIN_PER_M3_S
        names = []
        for start, end in zip(chain[:-1], chain[1:]):
            names.append(len(segment_lines) + 1)
            segment_lines.append(
                f"{len(segment_lines) + 1} {SEGMENT_TYPE} {start} {end} {float(data['diameter_um']):.{_DECIMALS}f} "
                f"{flow:.9e} {float(boundary_haematocrit):.6f}"
            )
        export.edge_segments[(u, v, key)] = names

    boundary_lines = []
    for node, pressure_pa in designated.items():
        export.pressure_nodes[node] = float(pressure_pa)
        boundary_lines.append(
            f"{export.node_names[node]} {BC_PRESSURE} {float(pressure_pa) / pa_per_mmhg:.9f} "
            f"{float(boundary_haematocrit):.6f}"
        )
    unknown_set = set(unknown)
    for node, count in degree.items():
        if count == 1 and node not in designated and node not in unknown_set:
            export.closed_nodes.append(node)
            boundary_lines.append(f"{export.node_names[node]} {BC_FLOW} 0.0 {float(boundary_haematocrit):.6f}")
    export.unknown_nodes = unknown

    extent = np.max(np.abs(np.asarray(positions)), axis=0) if positions else np.zeros(3)
    box_x, box_y, box_z = (float(np.ceil(extent[i])) for i in (2, 1, 0))
    points_per = [max(1, int(np.ceil(side / 10.0))) for side in (box_x, box_y, box_z)]
    lines = [
        title.replace("\n", " ")[:150],
        f"{box_x:g} {box_y:g} {box_z:g} box dimensions in microns",
        f"{points_per[0]} {points_per[1]} {points_per[2]} number of tissue points in x,y,z directions",
        "10. outer bound distance",
        f"{max(longest, 1.0):.3f} max. segment length",
        f"{max(degree.values(), default=1)} maximum number of segments per node",
        f"{len(segment_lines)} total number of segments",
        "SegName Type StartNode EndNode Diam Flow[nl/min] Hd",
        *segment_lines,
        f"{len(node_lines)} total number of nodes",
        "Name x y z",
        *node_lines,
        f"{len(boundary_lines)} total number of boundary nodes",
        "Node BcType Press/Flow HD",
        *boundary_lines,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return export


def read_secomb_network(path) -> SecombNetwork:
    """Read a ``Network.dat`` as NetFlow's ``input.cpp`` does: a title, six
    lines of which the last gives the segment count, a header, the segments;
    the node count, a header, the nodes; the boundary count, a header, the
    boundary nodes. Columns past the ones read are ignored."""
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    cursor = 1

    def count_line() -> int:
        nonlocal cursor
        while not lines[cursor].strip():
            cursor += 1
        value = int(lines[cursor].split()[0])
        cursor += 2  # the count, then the header under it
        return value

    cursor = 6
    n_segments = count_line()
    segments = {}
    for line in lines[cursor:cursor + n_segments]:
        parts = line.split()
        segments[int(parts[0])] = (
            int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        )
    cursor += n_segments
    n_nodes = count_line()
    nodes = {}
    for line in lines[cursor:cursor + n_nodes]:
        parts = line.split()
        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    cursor += n_nodes
    boundary = {}
    if cursor < len(lines):
        n_boundary = count_line()
        for line in lines[cursor:cursor + n_boundary]:
            parts = line.split()
            hd = float(parts[3]) if len(parts) > 3 else float("nan")
            boundary[int(parts[0])] = (int(parts[1]), float(parts[2]), hd)
    return SecombNetwork(segments=segments, nodes=nodes, boundary=boundary)
