"""Integration test: synthetic pericyte mask drives constriction on synthetic graph."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# Ensure ImageLynx package is importable for direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.haemodynamics import (
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
)
from ImageLynx.haemodynamics.pericyte_mask import (
    set_poiseuille_weights_with_pericyte_mask,
)


def _build_synthetic_graph() -> nx.MultiGraph:
    """Create a 2-edge centerline graph in physical units (um)."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.asarray([16.0, 16.0, 4.0], dtype=float))
    graph.add_node(1, pos=np.asarray([16.0, 16.0, 16.0], dtype=float))
    graph.add_node(2, pos=np.asarray([16.0, 16.0, 28.0], dtype=float))

    edge0_pts = [[16.0, 16.0, float(x)] for x in range(4, 17)]
    edge1_pts = [[16.0, 16.0, float(x)] for x in range(16, 29)]
    graph.add_edge(
        0,
        1,
        key=0,
        length=12.0,
        weight=1.0,
        branch_order="B01",
        voxels=edge0_pts,
    )
    graph.add_edge(
        1,
        2,
        key=0,
        length=12.0,
        weight=1.0,
        branch_order="B01",
        voxels=edge1_pts,
    )
    return graph


def _build_synthetic_pericyte_mask(
    shape: tuple[int, int, int] = (32, 32, 32),
    *,
    y_offset_um: float = 0.0,
) -> np.ndarray:
    """Create two synthetic bump-on-a-log pericyte volumes near the vessel."""
    mask = np.zeros(shape, dtype=np.uint8)

    zz, yy, xx = np.indices(shape, dtype=float)

    # Vessel axis in this synthetic graph runs along x with y=z fixed near 16.
    # Pericytes are created as a short "log" hugging one side of the vessel plus
    # a lateral "bump" (somatic bulge), giving a bump-on-a-log morphology.
    def add_bump_on_log(cx: float, cy: float, cz: float) -> None:
        dx = xx - cx
        dy = yy - cy
        dz = zz - cz

        log_half_length = 3.8
        log_radius = 1.8
        side_offset_threshold = 0.0
        log_part = (
            (np.abs(dx) <= log_half_length)
            & ((dy**2 + dz**2) <= (log_radius**2))
            & (dy >= side_offset_threshold)
        )

        bump_part = (
            (dx / 2.6) ** 2
            + ((dy - 2.3) / 2.0) ** 2
            + (dz / 2.3) ** 2
            <= 1.0
        )
        mask[log_part | bump_part] = 1

    y_center = 16.0 + float(y_offset_um)
    add_bump_on_log(cx=10.0, cy=y_center, cz=16.0)
    add_bump_on_log(cx=22.0, cy=y_center, cz=16.0)
    return mask


def _effective_resistance_between(
    graph: nx.MultiGraph,
    node_start: int,
    node_end: int,
) -> float:
    conductance, _ = build_conductance_matrix_from_graph(graph)
    laplacian = calc_laplacian_from_conductance_matrix(conductance)
    return float(calc_two_point_from_laplacian_matrix_nodeID(laplacian, graph, node_start, node_end))


def _interpolate_point_along_polyline(points: np.ndarray, s_um: float) -> np.ndarray:
    """Interpolate 3D point at arc-length position on a polyline."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 1:
        return points[0]
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    s = float(np.clip(s_um, 0.0, float(cum[-1])))
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, points.shape[0] - 2))
    l0, l1 = float(cum[idx]), float(cum[idx + 1])
    if l1 <= l0:
        return points[idx]
    t = (s - l0) / (l1 - l0)
    return points[idx] + t * (points[idx + 1] - points[idx])


def _run_pericyte_mask_integration_case(
    tmp_path: Path,
    *,
    case_name: str,
    y_offset_um: float,
    expect_constriction: bool,
) -> None:
    """Run one synthetic pericyte-mask integration scenario."""
    tifffile = pytest.importorskip("tifffile")
    go = pytest.importorskip("plotly.graph_objects")
    measure = pytest.importorskip("skimage.measure")

    output_dir = REPO_ROOT / "tests" / "outputs" / "synthetic_pericyte_mask_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{case_name}_graph_assignment_3d.html"
    metrics_path = output_dir / f"{case_name}_resistance_summary.txt"
    mask_path = tmp_path / f"{case_name}.tif"

    graph_baseline = _build_synthetic_graph()
    pericyte_mask = _build_synthetic_pericyte_mask(y_offset_um=y_offset_um)
    tifffile.imwrite(str(mask_path), pericyte_mask)

    diameter_by_branch_order = {"B01": 5.0}
    graph_baseline, baseline_results = set_poiseuille_weights_with_pericyte_mask(
        graph_baseline,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order={"B01": 1.0},
        pericyte_mask_path=mask_path,
        prefer_edge_fwhm_baseline=False,
        constriction_length=8.0,
    )
    assert baseline_results["weights_set"] > 0
    resistance_before = _effective_resistance_between(graph_baseline, 0, 2)

    graph_constricted = graph_baseline.copy()
    graph_constricted, constriction_results = set_poiseuille_weights_with_pericyte_mask(
        graph_constricted,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order={"B01": 0.8},
        pericyte_mask_path=mask_path,
        prefer_edge_fwhm_baseline=False,
        constriction_length=8.0,
    )
    resistance_after = _effective_resistance_between(graph_constricted, 0, 2)

    summary = (
        f"case={case_name}\n"
        f"mask_y_offset_um={float(y_offset_um):.3f}\n"
        f"resistance_before={resistance_before:.8f}\n"
        f"resistance_after={resistance_after:.8f}\n"
        f"delta={resistance_after - resistance_before:.8f}\n"
        f"ratio={resistance_after / resistance_before:.8f}\n"
        f"pericytes_detected={constriction_results['pericyte_count']}\n"
        f"edges_with_pericytes={constriction_results['edges_with_pericytes']}\n"
    )
    metrics_path.write_text(summary, encoding="utf-8")
    print(summary)

    # Render pericytes as 3D surfaces, graph centerlines, and assigned points.
    assigned_points: list[np.ndarray] = []
    for u, v, key, edge_data in graph_constricted.edges(keys=True, data=True):
        centers = edge_data.get("pericyte_centers_um", [])
        if not centers:
            continue
        edge_pts = np.asarray(edge_data.get("voxels"), dtype=float)
        for s_um in centers:
            assigned_points.append(_interpolate_point_along_polyline(edge_pts, float(s_um)))
    assigned_array = (
        np.asarray(assigned_points, dtype=float)
        if assigned_points
        else np.empty((0, 3), dtype=float)
    )

    fig = go.Figure()
    verts, faces, _, _ = measure.marching_cubes(pericyte_mask.astype(float), level=0.5)
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 2],
            y=verts[:, 1],
            z=verts[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.35,
            color="seagreen",
            name="Pericyte volumes",
        )
    )
    for u, v, key, edge_data in graph_constricted.edges(keys=True, data=True):
        edge_pts = np.asarray(edge_data.get("voxels"), dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=edge_pts[:, 2],
                y=edge_pts[:, 1],
                z=edge_pts[:, 0],
                mode="lines",
                line={"width": 6, "color": "royalblue"},
                name=f"Edge {u}-{v}-{key}",
                showlegend=False,
            )
        )
    if assigned_array.size:
        fig.add_trace(
            go.Scatter3d(
                x=assigned_array[:, 2],
                y=assigned_array[:, 1],
                z=assigned_array[:, 0],
                mode="markers",
                marker={"size": 6, "color": "crimson"},
                name="Assigned pericyte locations",
            )
        )

    fig.update_layout(
        title=(
            f"Synthetic pericyte mask integration ({case_name}) "
            f"(R_before={resistance_before:.4f}, R_after={resistance_after:.4f})"
        ),
        scene={
            "xaxis_title": "x (voxels)",
            "yaxis_title": "y (voxels)",
            "zaxis_title": "z (voxels)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    assert constriction_results["pericyte_count"] >= 2
    if expect_constriction:
        assert constriction_results["eligible_pericyte_count"] >= 1
        assert constriction_results["edges_with_pericytes"] >= 1
        assert resistance_after > resistance_before
    else:
        assert constriction_results["eligible_pericyte_count"] == 0
        assert constriction_results["edges_with_pericytes"] == 0
        assert np.isclose(resistance_after, resistance_before)
    assert metrics_path.exists() and metrics_path.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.slow
def test_synthetic_pericyte_mask_constriction_integration(tmp_path: Path):
    """Near-edge pericyte volumes: resistance before/after + 3D HTML outputs."""
    _run_pericyte_mask_integration_case(
        tmp_path,
        case_name="synthetic_pericyte_mask_near_edge",
        y_offset_um=0.0,
        expect_constriction=True,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_synthetic_pericyte_mask_constriction_integration_ten_um_away(tmp_path: Path):
    """Pericyte volumes 10 um from vessel edge with same output artifacts."""
    _run_pericyte_mask_integration_case(
        tmp_path,
        case_name="synthetic_pericyte_mask_ten_um_away",
        y_offset_um=10.0,
        expect_constriction=False,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_pericyte_mask_integration_case(
            Path(tmpdir),
            case_name="synthetic_pericyte_mask_near_edge",
            y_offset_um=0.0,
            expect_constriction=True,
        )
        _run_pericyte_mask_integration_case(
            Path(tmpdir),
            case_name="synthetic_pericyte_mask_ten_um_away",
            y_offset_um=10.0,
            expect_constriction=False,
        )
    print("Synthetic pericyte mask integration run completed.")
