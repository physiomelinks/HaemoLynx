"""A 2D map of how a network is connected, drawn from its connectivity CSV.

The CSV is the one :func:`haemolynx.graph.write_connectivity_csv` writes (the
"Export connectivity CSV" button on the panel's Export tab), and this module
reads nothing else, so any exported file can be drawn again later. The map is
a schematic, not the vessels' shape: each node sits in the column of how many
vessels it is from an inlet, inlets on the left and outlets on the right,
nodes in a column ordered to keep lines from crossing, and separate pieces of
the network stacked one above the other. Each vessel is a line coloured by its
branch order; hovering it shows its Branch ID, nodes, length, diameter and
notes -- the same IDs as the VTK and ``.pkl`` exports.
"""
from __future__ import annotations

import csv
import html
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

__all__ = [
    "ConnectivityMapLayout",
    "connectivity_map_figure",
    "connectivity_map_layout",
    "read_connectivity_csv",
    "write_connectivity_map",
]

#: Vertical gap, in rows, between separate pieces of the network.
_PIECE_GAP = 2.0


def read_connectivity_csv(path: Path | str) -> list[dict[str, str]]:
    """The rows of a connectivity CSV, as written."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ends(row: dict[str, str]) -> tuple[str, str]:
    """A row's (from, to) node IDs, filling a blank inlet/outlet tip from the edge key."""
    start, end = row.get("From node ID", ""), row.get("To node ID", "")
    u, v = row.get("Edge u", ""), row.get("Edge v", "")
    if not start and not end:
        return u, v
    if not start:
        start = u if v == end else v
    if not end:
        end = v if u == start else u
    return start, end


def _notes(row: dict[str, str]) -> set[str]:
    return {note.strip() for note in row.get("Notes", "").split(";") if note.strip()}


@dataclass(frozen=True)
class ConnectivityMapLayout:
    """Where every node of the map sits, and what kind of node it is."""

    #: node ID -> (x, y): x is vessels from an inlet, y the node's row.
    positions: dict[str, tuple[float, float]]
    inlets: frozenset[str]
    outlets: frozenset[str]
    dead_ends: frozenset[str]
    #: (row, from, to) for every vessel that is not a self-loop.
    vessels: tuple[tuple[dict[str, str], str, str], ...]
    self_loops: tuple[dict[str, str], ...]


def connectivity_map_layout(rows: Sequence[dict[str, str]]) -> ConnectivityMapLayout:
    """Lay the CSV's network out in columns from the inlets."""
    D = nx.MultiDiGraph()
    inlets, outlets, dead_ends = set(), set(), set()
    vessels, self_loops = [], []
    for row in rows:
        start, end = _ends(row)
        notes = _notes(row)
        if start == end:
            self_loops.append(row)
            D.add_node(start)
            continue
        D.add_edge(start, end)
        vessels.append((row, start, end))
        if "Inlet" in notes:
            inlets.add(start)
        if "Outlet" in notes:
            outlets.add(end)
        if "Dead end" in notes:
            dead_ends.add(end)

    positions: dict[str, tuple[float, float]] = {}
    top = 0.0
    pieces = sorted(nx.weakly_connected_components(D), key=lambda c: (-len(c), min(c)))
    for piece in pieces:
        sub = D.subgraph(piece)
        sources = [n for n in sub if n in inlets] or [
            n for n in sub if sub.in_degree(n) == 0
        ] or [min(sub, key=str)]
        level = {n: 0 for n in sources}
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            for nxt in sub.successors(node):
                if nxt not in level:
                    level[nxt] = level[node] + 1
                    queue.append(nxt)
        # Whatever the arrows do not reach (a loop no inlet feeds) is placed
        # from its neighbours, ignoring direction.
        undirected = sub.to_undirected(as_view=True)
        for node in nx.bfs_tree(undirected, sources[0]):
            if node not in level:
                level[node] = min(
                    (level[m] + 1 for m in undirected.neighbors(node) if m in level), default=0
                )
        for node in sub:
            level.setdefault(node, 0)

        columns: dict[int, list[str]] = defaultdict(list)
        for node, x in level.items():
            columns[x].append(node)
        y_of: dict[str, float] = {}
        height = 0
        for x in sorted(columns):
            def barycentre(node: str) -> tuple[float, str]:
                above = [y_of[p] for p in sub.predecessors(node) if p in y_of]
                above = above or [y_of[p] for p in undirected.neighbors(node) if p in y_of]
                return (sum(above) / len(above) if above else 0.0, node)

            ordered = sorted(columns[x], key=barycentre)
            for i, node in enumerate(ordered):
                y_of[node] = float(i) - (len(ordered) - 1) / 2.0
            height = max(height, len(ordered))
        for node, y in y_of.items():
            positions[node] = (float(level[node]), top - y)
        top -= height + _PIECE_GAP
    return ConnectivityMapLayout(
        positions=positions,
        inlets=frozenset(inlets),
        outlets=frozenset(outlets),
        dead_ends=frozenset(dead_ends),
        vessels=tuple(vessels),
        self_loops=tuple(self_loops),
    )


def _hover(row: dict[str, str], start: str, end: str) -> str:
    lines = [
        f"<b>Branch {html.escape(row.get('Branch ID', ''))}</b>",
        f"node {html.escape(start)} → node {html.escape(end)}",
    ]
    for label, key, unit in (
        ("branch order", "Branch order", ""),
        ("length", "Length (um)", " µm"),
        ("diameter", "Diameter (um)", " µm"),
        ("upstream", "Upstream branch IDs", ""),
        ("downstream", "Downstream branch IDs", ""),
        ("notes", "Notes", ""),
    ):
        value = row.get(key, "")
        if value:
            lines.append(f"{label}: {html.escape(value)}{unit}")
    return "<br>".join(lines)


def connectivity_map_figure(rows: Sequence[dict[str, str]], *, title: str = "") -> Any:
    """A plotly figure of :func:`connectivity_map_layout`."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    layout = connectivity_map_layout(rows)
    pos = layout.positions
    fig = go.Figure()

    by_order: dict[str, list[tuple[dict[str, str], str, str]]] = defaultdict(list)
    for item in layout.vessels:
        by_order[item[0].get("Branch order", "") or "(none)"].append(item)
    palette = qualitative.Dark24 + qualitative.Light24
    for i, order in enumerate(sorted(by_order)):
        colour = palette[i % len(palette)]
        xs, ys, mx, my, text = [], [], [], [], []
        for row, start, end in by_order[order]:
            (x0, y0), (x1, y1) = pos[start], pos[end]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            mx.append((x0 + x1) / 2)
            my.append((y0 + y1) / 2)
            text.append(_hover(row, start, end))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line={"color": colour, "width": 2},
            name=f"{order} ({len(by_order[order])})", legendgroup=order, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="markers", marker={"color": colour, "size": 7, "opacity": 0.0},
            hovertext=text, hoverinfo="text", legendgroup=order, showlegend=False,
        ))

    kinds = (
        ("Inlet", layout.inlets, "triangle-right", "#2ca02c", 11),
        ("Outlet", layout.outlets, "square", "#d62728", 10),
        ("Dead end", layout.dead_ends - layout.outlets, "x", "#ff7f0e", 8),
    )
    marked = set().union(*(nodes for _label, nodes, *_rest in kinds))
    junctions = [n for n in pos if n not in marked]
    for label, nodes, symbol, colour, size in (
        ("Node", junctions, "circle", "#7f7f7f", 4), *kinds
    ):
        nodes = sorted(nodes, key=str)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
            marker={"symbol": symbol, "color": colour, "size": size},
            hovertext=[f"{label}: node {html.escape(n)}" for n in nodes], hoverinfo="text",
            name=f"{label}s ({len(nodes)})",
        ))
    if layout.self_loops:
        loops = [r for r in layout.self_loops if _ends(r)[0] in pos]
        fig.add_trace(go.Scatter(
            x=[pos[_ends(r)[0]][0] for r in loops], y=[pos[_ends(r)[0]][1] for r in loops],
            mode="markers", marker={"symbol": "circle-open", "color": "#9467bd", "size": 14},
            hovertext=[_hover(r, *_ends(r)) for r in loops], hoverinfo="text",
            name=f"Self-loops ({len(loops)})",
        ))

    heading = title or "Network connectivity"
    fig.update_layout(
        title=(
            f"{html.escape(heading)} — {len(rows)} vessels, {len(layout.inlets)} inlets, "
            f"{len(layout.outlets)} outlets"
        ),
        xaxis={"title": "vessels from an inlet", "zeroline": False, "showgrid": False},
        yaxis={"visible": False},
        hovermode="closest",
        plot_bgcolor="white",
        legend={"itemsizing": "constant"},
        dragmode="pan",
    )
    return fig


def write_connectivity_map(
    csv_path: Path | str, html_path: Path | str | None = None
) -> Path:
    """Draw the connectivity CSV at *csv_path* to an HTML page and return its path.

    *html_path* defaults to ``<csv name>_map.html`` beside the CSV.
    """
    source = Path(csv_path)
    target = Path(html_path) if html_path is not None else source.with_name(f"{source.stem}_map.html")
    figure = connectivity_map_figure(read_connectivity_csv(source), title=source.stem)
    figure.write_html(str(target), include_plotlyjs="cdn", config={"scrollZoom": True})
    return target
