"""End-to-end synthetic test: large-mask terminals, small-mask boundaries, hierarchical orders.

ImageLynx roles (this test matches the main resistance pipeline):
- **Large** arteriole/venule masks -> `select_terminal_nodes_from_large_vessel_masks`
  (degree-1 **input / output** terminals).
- **Small** arteriole/venule masks -> `infer_boundary_nodes_from_small_vessel_masks`
  (**arteriole / venule boundary** nodes for Art/Ven/Cap hierarchy — not terminal IO).
- **Vessel order** -> `assign_hierarchical_branch_orders` (Art*, Ven*, B* on edges).

Run: ``python tests/test_synthetic_vessel_assignment_pipeline.py`` or ``pytest`` on this file.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.graph import (
    assign_hierarchical_branch_orders,
    infer_boundary_nodes_from_small_vessel_masks,
    select_terminal_nodes_from_large_vessel_masks,
)

DEMO_HTML_PATH = (
    REPO_ROOT / "examples" / "plots" / "synthetic_vessel_assignment_pipeline_3d.html"
)


def _voxel_polyline_samples(
    p0: np.ndarray, p1: np.ndarray, *, count: int = 32
) -> list[tuple[float, float, float]]:
    t = np.linspace(0.0, 1.0, int(count), dtype=float)
    pts = (1.0 - t).reshape(-1, 1) * p0.reshape(1, 3) + t.reshape(-1, 1) * p1.reshape(1, 3)
    uniq = np.unique(np.rint(pts).astype(int), axis=0)
    return [tuple(float(x) for x in row) for row in uniq]


def _edge_id(u: int, v: int, key: int) -> tuple[int, int, int]:
    return (u, v, key) if u <= v else (v, u, key)


def _edge_ids_on_path(G: nx.MultiGraph, path: list[int]) -> list[tuple[int, int, int]]:
    ids: list[tuple[int, int, int]] = []
    for u, v in zip(path[:-1], path[1:]):
        keys = list(G[u][v].keys())
        assert keys, f"No edge between {u} and {v}"
        ids.append(_edge_id(u, v, keys[0]))
    return ids


def build_synthetic_integrated_vessel_model() -> tuple[
    nx.MultiGraph,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, list[int]],
]:
    """Linear chain: large masks sit in low-y voxels, small masks in higher y — no large↔small overlap."""
    nz, ny, n_vox_x = 16, 16, 24
    shape = (nz, ny, n_vox_x)

    large_arteriole_mask = np.zeros(shape, dtype=bool)
    large_venule_mask = np.zeros(shape, dtype=bool)
    # Large IO hulls: y indices 3–5 only (disjoint from small-mask y band 7–11).
    large_arteriole_mask[5:13, 3:6, 0:5] = True
    large_venule_mask[5:13, 3:6, 14:n_vox_x] = True

    small_arteriole_mask = np.zeros(shape, dtype=bool)
    small_venule_mask = np.zeros(shape, dtype=bool)
    # Small boundary slabs: y indices 7–11; x ranges unchanged from prior art/cap/ven layout.
    small_arteriole_mask[6:12, 7:12, 0:6] = True
    small_venule_mask[6:12, 7:12, 11:n_vox_x] = True

    G = nx.MultiGraph()
    # Terminals sit in low-y large-mask band; interior trunk at y=8 inside small-mask band only.
    xs = [1.0, 3.0, 5.0, 8.0, 11.0, 13.0, 16.0]
    G.add_node(0, pos=np.array([8.0, 4.0, xs[0]], dtype=float))
    for i, x in enumerate(xs[1:-1], start=1):
        G.add_node(i, pos=np.array([8.0, 8.0, x], dtype=float))
    G.add_node(6, pos=np.array([8.0, 4.0, xs[-1]], dtype=float))

    # Connector (0→1): unique voxels after np.unique must be mostly inside small art (y≥7), else node 1
    # becomes a false arteriole boundary. Short low-y leg + many distinct in-slab lattice points.
    p0 = np.asarray(G.nodes[0]["pos"], dtype=float)
    p1 = np.asarray(G.nodes[1]["pos"], dtype=float)
    v01_u: list[tuple[float, float, float]] = [
        (8.0, 4.0, 1.0),
        (8.0, 5.0, 1.0),
        (8.0, 6.0, 1.0),
        (8.0, 7.0, 1.0),
        (8.0, 7.0, 2.0),
        (8.0, 8.0, 1.0),
        (8.0, 8.0, 2.0),
        (8.0, 8.0, 3.0),
    ]
    G.add_edge(
        0,
        1,
        voxels=v01_u,
        length=float(np.linalg.norm(p1 - p0)),
        weight=1.0,
    )

    for a, b in zip(range(1, len(xs) - 2), range(2, len(xs) - 1)):
        pa = np.asarray(G.nodes[a]["pos"], dtype=float)
        pb = np.asarray(G.nodes[b]["pos"], dtype=float)
        G.add_edge(
            a,
            b,
            voxels=_voxel_polyline_samples(pa, pb, count=36),
            length=float(np.linalg.norm(pb - pa)),
            weight=1.0,
        )

    # Bent trunk→terminal (5→6): majority in small venule band before dropping to low-y large venule.
    p5 = np.asarray(G.nodes[5]["pos"], dtype=float)
    p6 = np.asarray(G.nodes[6]["pos"], dtype=float)
    p_mid_out = np.array([8.0, 7.0, 15.0], dtype=float)
    v56 = _voxel_polyline_samples(p5, p_mid_out, count=22) + _voxel_polyline_samples(
        p_mid_out, p6, count=14
    )
    seen56: set[tuple[float, float, float]] = set()
    v56_u: list[tuple[float, float, float]] = []
    for t in v56:
        if t not in seen56:
            seen56.add(t)
            v56_u.append(t)
    G.add_edge(
        5,
        6,
        voxels=v56_u,
        length=float(np.linalg.norm(p6 - p5)),
        weight=1.0,
    )

    expected = {
        "starting_nodes": [0],
        "output_nodes": [6],
        "arteriole_boundary_nodes": [2],
        "venule_boundary_nodes": [4],
    }
    assert not np.any(
        (large_arteriole_mask | large_venule_mask)
        & (small_arteriole_mask | small_venule_mask)
    )
    return (
        G,
        large_arteriole_mask,
        large_venule_mask,
        small_arteriole_mask,
        small_venule_mask,
        expected,
    )


def write_integrated_vessel_pipeline_3d_html(
    G: nx.MultiGraph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    input_nodes: list[int],
    output_nodes: list[int],
    arteriole_boundary_nodes: list[int],
    venule_boundary_nodes: list[int],
    voxel_size_zyx: tuple[float, float, float],
    output_html_path: str | Path,
    title: str = "Synthetic pipeline: masks + hierarchical branch orders (3D)",
) -> bool:
    """One rotatable figure: four mask volumes, edges by Art/Ven/B*, role markers."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    pos = nx.get_node_attributes(G, "pos")
    if not pos or large_arteriole_mask.shape != large_venule_mask.shape:
        return False
    if small_arteriole_mask.shape != small_venule_mask.shape:
        return False
    if large_arteriole_mask.shape != small_arteriole_mask.shape:
        return False

    out = Path(output_html_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def add_volume(mask: np.ndarray, *, name: str, color: str, opacity: float) -> None:
        if not np.any(mask):
            return
        zs, ys, xs = voxel_size_zyx
        zz, yy, xx = np.indices(mask.shape, dtype=float)
        fig.add_trace(
            go.Volume(
                x=(xx * float(xs)).ravel(),
                y=(yy * float(ys)).ravel(),
                z=(zz * float(zs)).ravel(),
                value=mask.astype(float).ravel(),
                isomin=0.5,
                isomax=1.0,
                opacity=opacity,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=name,
            )
        )

    fig = go.Figure()
    add_volume(
        large_arteriole_mask,
        name="Large arteriole (terminal IO)",
        color="#66FFAA",
        opacity=0.07,
    )
    add_volume(
        large_venule_mask,
        name="Large venule (terminal IO)",
        color="#FF88CC",
        opacity=0.07,
    )
    add_volume(
        small_arteriole_mask,
        name="Small arteriole (boundaries)",
        color="#00AA55",
        opacity=0.11,
    )
    add_volume(
        small_venule_mask,
        name="Small venule (boundaries)",
        color="#DD2288",
        opacity=0.11,
    )

    segs: dict[str, tuple[list[float | None], list[float | None], list[float | None]]] = {
        "Art": ([], [], []),
        "Ven": ([], [], []),
        "B": ([], [], []),
        "other": ([], [], []),
    }

    def push(kind: str, pu: np.ndarray, pv: np.ndarray) -> None:
        lx, ly, lz = segs[kind]
        lx += [float(pu[2]), float(pv[2]), None]
        ly += [float(pu[1]), float(pv[1]), None]
        lz += [float(pu[0]), float(pv[0]), None]

    for u, v, k, data in G.edges(keys=True, data=True):
        if u not in pos or v not in pos:
            continue
        pu = np.asarray(pos[u], dtype=float)
        pv = np.asarray(pos[v], dtype=float)
        label = str(data.get("branch_order", "") or "")
        if label.startswith("Art"):
            push("Art", pu, pv)
        elif label.startswith("Ven"):
            push("Ven", pu, pv)
        elif label.startswith("B"):
            push("B", pu, pv)
        else:
            push("other", pu, pv)

    style = {
        "Art": ("rgba(214, 39, 40, 0.95)", 5),
        "Ven": ("rgba(31, 119, 180, 0.95)", 5),
        "B": ("rgba(44, 160, 44, 0.9)", 4),
        "other": ("rgba(140, 140, 140, 0.6)", 3),
    }
    for kind, (color, width) in style.items():
        ex, ey, ez = segs[kind]
        if not ex:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=color, width=width),
                name=f"Edges {kind}" if kind != "other" else "Edges (unassigned)",
            )
        )

    def coords(nodes: list[int]) -> tuple[list[float], list[float], list[float]]:
        xs = [float(np.asarray(pos[n], dtype=float)[2]) for n in nodes if n in pos]
        ys = [float(np.asarray(pos[n], dtype=float)[1]) for n in nodes if n in pos]
        zs = [float(np.asarray(pos[n], dtype=float)[0]) for n in nodes if n in pos]
        return xs, ys, zs

    in_s, out_s = set(input_nodes), set(output_nodes)
    ab_s, vb_s = set(arteriole_boundary_nodes), set(venule_boundary_nodes)
    special = in_s | out_s | ab_s | vb_s
    neutral = [n for n in G.nodes if n not in special]
    if neutral:
        nx_, ny_, nz_ = coords(neutral)
        fig.add_trace(
            go.Scatter3d(
                x=nx_,
                y=ny_,
                z=nz_,
                mode="markers",
                marker=dict(size=5, color="#9E9E9E"),
                name="Nodes (interior)",
            )
        )
    if input_nodes:
        ix, iy, iz = coords(input_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=10, color="#00FF7F", line=dict(width=1, color="#004422")),
                name="Input terminals (large mask)",
            )
        )
    if output_nodes:
        ox, oy, oz = coords(output_nodes)
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=10, color="#FF3EA5", line=dict(width=1, color="#440022")),
                name="Output terminals (large mask)",
            )
        )
    if arteriole_boundary_nodes:
        bx, by, bz = coords(list(arteriole_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(
                    size=11,
                    color="#00AA66",
                    symbol="diamond",
                    line=dict(width=1, color="#003311"),
                ),
                name="Arteriole boundary (small mask)",
            )
        )
    if venule_boundary_nodes:
        bx, by, bz = coords(list(venule_boundary_nodes))
        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="markers",
                marker=dict(
                    size=11,
                    color="#FF66AA",
                    symbol="diamond",
                    line=dict(width=1, color="#440022"),
                ),
                name="Venule boundary (small mask)",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            "Large (pale) and small (stronger) mask volumes use disjoint y bands — no voxel overlap. "
            "Edge colours: Art=red, Ven=blue, Capillary=green."
        ),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#666",
        font=dict(size=11),
    )
    fig.write_html(str(out), include_plotlyjs="cdn")
    return True


@pytest.mark.plotting
def test_synthetic_large_small_masks_and_hierarchical_orders(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")

    voxel_size_zyx = (1.0, 1.0, 1.0)
    min_overlap = 0.5

    (
        G,
        large_art,
        large_ven,
        small_art,
        small_ven,
        expected,
    ) = build_synthetic_integrated_vessel_model()

    starting_nodes, output_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        voxel_size_zyx=voxel_size_zyx,
        allow_overlap=False,
    )
    assert starting_nodes == expected["starting_nodes"]
    assert output_nodes == expected["output_nodes"]

    boundary_result = infer_boundary_nodes_from_small_vessel_masks(
        G,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        voxel_size_zyx=voxel_size_zyx,
        minimum_overlap_fraction=min_overlap,
        allow_overlap=False,
    )
    art_b = boundary_result["arteriole_boundary_nodes"]
    ven_b = boundary_result["venule_boundary_nodes"]
    assert art_b == expected["arteriole_boundary_nodes"]
    assert ven_b == expected["venule_boundary_nodes"]

    assign_hierarchical_branch_orders(
        G,
        starting_nodes=starting_nodes,
        output_nodes=output_nodes,
        arteriole_boundary_nodes=art_b,
        venule_boundary_nodes=ven_b,
    )

    s0, s6 = expected["starting_nodes"][0], expected["output_nodes"][0]
    ab, vb = art_b[0], ven_b[0]
    path_in_to_ab = nx.shortest_path(G, s0, ab)
    path_out_to_vb = nx.shortest_path(G, s6, vb)
    path_ab_to_vb = nx.shortest_path(G, ab, vb)

    for eid in _edge_ids_on_path(G, path_in_to_ab):
        u, v, key = eid
        assert G[u][v][key]["branch_order"].startswith("Art")
    for eid in _edge_ids_on_path(G, path_out_to_vb):
        u, v, key = eid
        assert G[u][v][key]["branch_order"].startswith("Ven")
    for eid in _edge_ids_on_path(G, path_ab_to_vb):
        u, v, key = eid
        assert G[u][v][key]["branch_order"].startswith("B")

    html_tmp = tmp_path / "synthetic_vessel_assignment_pipeline_3d.html"
    assert write_integrated_vessel_pipeline_3d_html(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=starting_nodes,
        output_nodes=output_nodes,
        arteriole_boundary_nodes=art_b,
        venule_boundary_nodes=ven_b,
        voxel_size_zyx=voxel_size_zyx,
        output_html_path=html_tmp,
    )
    assert html_tmp.is_file() and html_tmp.stat().st_size > 2000

    DEMO_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert write_integrated_vessel_pipeline_3d_html(
        G,
        large_arteriole_mask=large_art,
        large_venule_mask=large_ven,
        small_arteriole_mask=small_art,
        small_venule_mask=small_ven,
        input_nodes=starting_nodes,
        output_nodes=output_nodes,
        arteriole_boundary_nodes=art_b,
        venule_boundary_nodes=ven_b,
        voxel_size_zyx=voxel_size_zyx,
        output_html_path=DEMO_HTML_PATH,
    )

    try:
        webbrowser.open_new_tab(html_tmp.resolve().as_uri())
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
