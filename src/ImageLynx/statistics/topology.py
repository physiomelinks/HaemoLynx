"""Topology-level vessel network statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Real
from typing import Any, Dict, Optional, Union

import networkx as nx


def _parse_edge_direction(
    edge_value: Any, positive_means_u_to_v: bool = True
) -> Optional[bool]:
    """Return edge orientation from per-edge value.

    Returns:
        True if direction is u->v, False if v->u, None if unknown.
    """
    if edge_value is None:
        return None

    if isinstance(edge_value, Real):
        if float(edge_value) == 0.0:
            return None
        is_u_to_v = float(edge_value) > 0.0
        return is_u_to_v if positive_means_u_to_v else (not is_u_to_v)

    text = str(edge_value).strip().lower()
    if text in {"u_to_v", "uv", "forward", "fwd", "+"}:
        return True
    if text in {"v_to_u", "vu", "reverse", "rev", "-"}:
        return False
    return None


def summarize_junction_types(
    G: Union[nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph],
    direction_attr: Optional[str] = None,
    positive_means_u_to_v: bool = True,
) -> Dict[str, Any]:
    """Summarize node junction classes by incoming/outgoing edge counts.

    The four classes follow the image convention:
    - 2-in-2-out
    - 2-in-1-out
    - 1-in-2-out
    - 1-in-1-out

    Direction handling:
    - Directed graphs: use in-degree / out-degree directly.
    - Undirected graphs: provide ``direction_attr`` on edges. For numeric values,
      positive means u->v and negative means v->u (configurable with
      ``positive_means_u_to_v``). String values also accept ``u_to_v``/``v_to_u``
      (plus a few aliases).
    """
    key_by_counts = {
        (2, 2): "Junction Count (2-in-2-out)",
        (2, 1): "Junction Count (2-in-1-out)",
        (1, 2): "Junction Count (1-in-2-out)",
        (1, 1): "Junction Count (1-in-1-out)",
    }

    class_counts: Counter[str] = Counter(
        {
            "Junction Count (2-in-2-out)": 0,
            "Junction Count (2-in-1-out)": 0,
            "Junction Count (1-in-2-out)": 0,
            "Junction Count (1-in-1-out)": 0,
        }
    )
    unclassified = 0
    skipped_for_unknown_direction = 0
    node_direction_counts: dict[Any, list[int]] = defaultdict(lambda: [0, 0])

    is_directed = G.is_directed()
    if not is_directed and direction_attr is None:
        raise ValueError(
            "Undirected graph requires direction_attr to infer in/out edges."
        )

    if is_directed:
        for node in G.nodes():
            in_count = int(G.in_degree(node))
            out_count = int(G.out_degree(node))
            if (in_count + out_count) < 2:
                continue
            key = key_by_counts.get((in_count, out_count))
            if key is not None:
                class_counts[key] += 1
            else:
                unclassified += 1
    else:
        edge_iter = (
            G.edges(keys=True, data=True)
            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
            else G.edges(data=True)
        )
        for edge_item in edge_iter:
            u = edge_item[0]
            v = edge_item[1]
            data = edge_item[-1]
            edge_direction = _parse_edge_direction(
                data.get(direction_attr), positive_means_u_to_v=positive_means_u_to_v
            )
            if edge_direction is None:
                skipped_for_unknown_direction += 1
                continue
            if edge_direction:
                node_direction_counts[u][1] += 1  # out
                node_direction_counts[v][0] += 1  # in
            else:
                node_direction_counts[v][1] += 1  # out
                node_direction_counts[u][0] += 1  # in

        for in_count, out_count in node_direction_counts.values():
            if (in_count + out_count) < 2:
                continue
            key = key_by_counts.get((in_count, out_count))
            if key is not None:
                class_counts[key] += 1
            else:
                unclassified += 1

    total_junctions = int(sum(class_counts.values()) + unclassified)
    return {
        **dict(class_counts),
        "Junction Count (Unclassified In/Out)": int(unclassified),
        "Total Junction Count": total_junctions,
        "Direction Attribute Used": (
            "directed_graph" if is_directed else str(direction_attr)
        ),
        "Skipped Edges (Unknown Direction)": int(skipped_for_unknown_direction),
    }


def annotate_edge_direction_from_signed_attribute(
    G: Union[nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph],
    signed_attr: str = "flow_signed",
    direction_attr: str = "edge_direction",
    positive_means_u_to_v: bool = True,
) -> Dict[str, int]:
    """Annotate per-edge direction labels from a signed edge attribute.

    Labels written:
    - ``u_to_v`` when signed value indicates flow from u to v.
    - ``v_to_u`` when signed value indicates flow from v to u.
    - ``unknown`` when signed value is missing/zero/unparseable.
    """
    assigned = 0
    unknown = 0
    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = G.edges(keys=True, data=True) if is_mg else G.edges(data=True)

    for edge_item in edge_iter:
        u = edge_item[0]
        v = edge_item[1]
        data = edge_item[-1]
        direction = _parse_edge_direction(
            data.get(signed_attr), positive_means_u_to_v=positive_means_u_to_v
        )
        if direction is None:
            data[direction_attr] = "unknown"
            unknown += 1
            continue
        data[direction_attr] = "u_to_v" if direction else "v_to_u"
        assigned += 1

    return {
        "Annotated Edges": int(assigned),
        "Unknown Direction Edges": int(unknown),
    }


def summarize_junction_types_from_signed_flow(
    G: Union[nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph],
    signed_attr: str = "flow_signed",
    direction_attr: str = "edge_direction",
    positive_means_u_to_v: bool = True,
) -> Dict[str, Any]:
    """Annotate direction from signed flow and summarize junction classes."""
    annotation_summary = annotate_edge_direction_from_signed_attribute(
        G,
        signed_attr=signed_attr,
        direction_attr=direction_attr,
        positive_means_u_to_v=positive_means_u_to_v,
    )
    junction_summary = summarize_junction_types(
        G,
        direction_attr=direction_attr,
        positive_means_u_to_v=positive_means_u_to_v,
    )
    return {
        **junction_summary,
        **annotation_summary,
    }
