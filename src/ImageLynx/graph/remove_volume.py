"""Remove graph elements intersecting user-defined volumes."""
from __future__ import annotations

from typing import Sequence

import networkx as nx


VolumeBox = tuple[tuple[float, float, float], tuple[float, float, float]]


def _is_xyz_point(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        float(value[0])
        float(value[1])
        float(value[2])
    except (TypeError, ValueError):
        return False
    return True


def _normalize_volume_boxes(volumes_xyz: Sequence[object]) -> list[VolumeBox]:
    """Normalize user volume config into [((x0,y0,z0),(x1,y1,z1)), ...]."""
    raw = list(volumes_xyz or [])
    if not raw:
        return []

    # Common convenience form for a single box:
    # [(x0,y0,z0), (x1,y1,z1)]
    if len(raw) == 2 and _is_xyz_point(raw[0]) and _is_xyz_point(raw[1]):
        lo = raw[0]
        hi = raw[1]
        return [(
            (float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])),
        )]

    boxes: list[VolumeBox] = []
    for idx, entry in enumerate(raw):
        # Standard form: ((x0,y0,z0), (x1,y1,z1))
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and _is_xyz_point(entry[0])
            and _is_xyz_point(entry[1])
        ):
            lo = entry[0]
            hi = entry[1]
            boxes.append((
                (float(lo[0]), float(lo[1]), float(lo[2])),
                (float(hi[0]), float(hi[1]), float(hi[2])),
            ))
            continue

        # Flat shorthand form: (x0,y0,z0,x1,y1,z1)
        if isinstance(entry, (list, tuple)) and len(entry) == 6:
            boxes.append((
                (float(entry[0]), float(entry[1]), float(entry[2])),
                (float(entry[3]), float(entry[4]), float(entry[5])),
            ))
            continue

        raise ValueError(
            "Invalid GRAPH_ELEMENT_REMOVAL_VOLUMES entry at index "
            f"{idx}: {entry!r}. Expected ((x0,y0,z0),(x1,y1,z1)) or "
            "(x0,y0,z0,x1,y1,z1)."
        )

    return boxes


def _point_in_box(point_xyz: Sequence[float], box: VolumeBox) -> bool:
    lo, hi = box
    x0, y0, z0 = (float(lo[0]), float(lo[1]), float(lo[2]))
    x1, y1, z1 = (float(hi[0]), float(hi[1]), float(hi[2]))
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)
    min_z, max_z = min(z0, z1), max(z0, z1)

    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    return (
        (min_x <= x <= max_x)
        and (min_y <= y <= max_y)
        and (min_z <= z <= max_z)
    )


def _edge_intersects_any_box(edge_data: dict, boxes: Sequence[VolumeBox]) -> bool:
    voxels = edge_data.get("voxels")
    if not voxels:
        return False
    for voxel in voxels:
        if voxel is None or len(voxel) < 3:
            continue
        if any(_point_in_box(voxel, box) for box in boxes):
            return True
    return False


def remove_graph_elements_in_volumes(
    G: nx.MultiGraph,
    volumes_xyz: Sequence[object],
    *,
    remove_nodes: bool = True,
    remove_edges: bool = True,
    remove_isolated_nodes_after: bool = True,
) -> tuple[nx.MultiGraph, dict[str, int]]:
    """Remove nodes/edges that intersect any configured volume box.

    Volumes use the same coordinate convention as node ``pos`` and edge ``voxels``.
    """
    boxes = _normalize_volume_boxes(volumes_xyz)
    stats = {
        "configured_volume_count": int(len(boxes)),
        "removed_nodes": 0,
        "removed_edges": 0,
        "removed_isolated_nodes": 0,
    }
    if not boxes:
        return G, stats

    if remove_edges:
        edges_to_remove: list[tuple[int, int, int]] = []
        for u, v, key, edge_data in G.edges(keys=True, data=True):
            if _edge_intersects_any_box(edge_data, boxes):
                edges_to_remove.append((u, v, key))
        if edges_to_remove:
            G.remove_edges_from(edges_to_remove)
            stats["removed_edges"] = int(len(edges_to_remove))

    if remove_nodes:
        nodes_to_remove: list[int] = []
        for node_id, node_data in G.nodes(data=True):
            pos = node_data.get("pos")
            if pos is None or len(pos) < 3:
                continue
            if any(_point_in_box(pos, box) for box in boxes):
                nodes_to_remove.append(int(node_id))
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            stats["removed_nodes"] = int(len(nodes_to_remove))

    if remove_isolated_nodes_after:
        before_nodes = G.number_of_nodes()
        isolated_nodes = [n for n, degree in G.degree() if int(degree) == 0]
        if isolated_nodes:
            G.remove_nodes_from(isolated_nodes)
        stats["removed_isolated_nodes"] = int(before_nodes - G.number_of_nodes())

    return G, stats
