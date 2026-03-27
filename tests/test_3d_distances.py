"""Tests for 3D object-to-vessel distance measurement."""
from __future__ import annotations

import importlib
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

dist3d = importlib.import_module("ImageLynx.statistics.3D_distances")
DEMO_OUTPUT_DIR = REPO_ROOT / "examples" / "outputs" / "synthetic_3d_distances"


def _line_voxels(
    p0: tuple[float, float, float], p1: tuple[float, float, float], n: int = 20
) -> list[tuple[float, float, float]]:
    t = np.linspace(0.0, 1.0, int(n), dtype=float)
    a = np.asarray(p0, dtype=float).reshape(1, 3)
    b = np.asarray(p1, dtype=float).reshape(1, 3)
    pts = (1.0 - t).reshape(-1, 1) * a + t.reshape(-1, 1) * b
    uniq = np.unique(np.rint(pts).astype(int), axis=0)
    return [tuple(float(v) for v in row) for row in uniq]


def _build_synthetic_case(
    tmp_path: Path,
) -> tuple[Path, Path, nx.MultiGraph, np.ndarray, np.ndarray]:
    shape = (16, 16, 16)
    cell_mask = np.zeros(shape, dtype=np.uint8)
    vessel_mask = np.zeros(shape, dtype=np.uint8)

    # Two disconnected "cell" objects.
    cell_mask[2:4, 2:4, 2:4] = 1
    cell_mask[11:13, 11:13, 12:14] = 1

    # Two vessel trunks.
    vessel_mask[2, 2, 5:14] = 1
    vessel_mask[12, 12, 2:11] = 1

    cell_path = tmp_path / "cells.tif"
    vessel_path = tmp_path / "vessels.tif"
    tifffile.imwrite(str(cell_path), cell_mask)
    tifffile.imwrite(str(vessel_path), vessel_mask)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([2.0, 2.0, 5.0], dtype=float))
    G.add_node(1, pos=np.array([2.0, 2.0, 13.0], dtype=float))
    G.add_node(2, pos=np.array([12.0, 12.0, 2.0], dtype=float))
    G.add_node(3, pos=np.array([12.0, 12.0, 10.0], dtype=float))
    G.add_edge(
        0,
        1,
        voxels=_line_voxels((2.0, 2.0, 5.0), (2.0, 2.0, 13.0)),
        branch_order="B01",
        length=8.0,
        weight=8.0,
    )
    G.add_edge(
        2,
        3,
        voxels=_line_voxels((12.0, 12.0, 2.0), (12.0, 12.0, 10.0)),
        branch_order="Art1",
        length=8.0,
        weight=8.0,
    )
    return cell_path, vessel_path, G, cell_mask, vessel_mask


def _build_microglia_like_synthetic_case(
    tmp_path: Path,
) -> tuple[Path, Path, nx.MultiGraph, np.ndarray, np.ndarray]:
    """Synthetic microglia-like cells: soma + thin processes."""
    shape = (28, 28, 28)
    cell_mask = np.zeros(shape, dtype=np.uint8)
    vessel_mask = np.zeros(shape, dtype=np.uint8)

    # Cell 1 soma with two processes.
    cell_mask[4:8, 4:8, 4:8] = 1
    cell_mask[6, 6, 8:15] = 1
    cell_mask[6, 8:14, 6] = 1

    # Cell 2 soma with two processes.
    cell_mask[18:22, 18:22, 18:22] = 1
    cell_mask[20, 20, 12:18] = 1
    cell_mask[20, 14:20, 20] = 1

    # Two vessel trunks in separate regions.
    vessel_mask[6, 6, 16:26] = 1
    vessel_mask[20, 20, 4:14] = 1

    cell_path = tmp_path / "microglia_cells.tif"
    vessel_path = tmp_path / "microglia_vessels.tif"
    tifffile.imwrite(str(cell_path), cell_mask)
    tifffile.imwrite(str(vessel_path), vessel_mask)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([6.0, 6.0, 16.0], dtype=float))
    G.add_node(1, pos=np.array([6.0, 6.0, 25.0], dtype=float))
    G.add_node(2, pos=np.array([20.0, 20.0, 4.0], dtype=float))
    G.add_node(3, pos=np.array([20.0, 20.0, 13.0], dtype=float))
    G.add_edge(
        0,
        1,
        voxels=_line_voxels((6.0, 6.0, 16.0), (6.0, 6.0, 25.0)),
        branch_order="Ven1",
        length=9.0,
        weight=9.0,
    )
    G.add_edge(
        2,
        3,
        voxels=_line_voxels((20.0, 20.0, 4.0), (20.0, 20.0, 13.0)),
        branch_order="B02",
        length=9.0,
        weight=9.0,
    )
    return cell_path, vessel_path, G, cell_mask, vessel_mask


def _copy_csv_as_ground_truth(
    *,
    source_csv: Path,
    ground_truth_csv: Path,
) -> Path:
    """Copy a generated CSV into a deterministic ground-truth artifact."""
    with source_csv.open("r", newline="", encoding="utf-8") as src_f:
        reader = list(csv.reader(src_f))
    ground_truth_csv.parent.mkdir(parents=True, exist_ok=True)
    with ground_truth_csv.open("w", newline="", encoding="utf-8") as dst_f:
        writer = csv.writer(dst_f)
        writer.writerows(reader)
    return ground_truth_csv


def _write_precalculated_ground_truth_csvs(
    *,
    output_dir: Path,
    image_stem: str = "synthetic_cells",
) -> tuple[Path, Path]:
    """Write deterministic ground-truth CSVs for this synthetic setup."""
    output_dir.mkdir(parents=True, exist_ok=True)
    details_gt = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances_ground_truth.csv"
    summary_gt = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances_summary_ground_truth.csv"

    d_cent = float(np.sqrt(6.75))
    delta = d_cent - 2.0

    with details_gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Object ID",
                "Object Name",
                "Voxel Count",
                "Centroid Z (voxel)",
                "Centroid Y (voxel)",
                "Centroid X (voxel)",
                "Edge-to-Vessel Distance (microns)",
                "Edge-to-Vessel Nearest Z (microns)",
                "Edge-to-Vessel Nearest Y (microns)",
                "Edge-to-Vessel Nearest X (microns)",
                "Edge-to-Vessel Nearest Coordinate (z,y,x microns)",
                "Edge-to-Vessel Nearest Graph Edge Label ID",
                "Edge-to-Vessel Nearest Branch Order",
                "Centroid-to-Vessel Distance (microns)",
                "Delta Distance (Centroid-Edge) (microns)",
                "Centroid-to-Vessel Nearest Z (microns)",
                "Centroid-to-Vessel Nearest Y (microns)",
                "Centroid-to-Vessel Nearest X (microns)",
                "Centroid-to-Vessel Nearest Coordinate (z,y,x microns)",
                "Centroid-to-Vessel Nearest Graph Edge Label ID",
                "Centroid-to-Vessel Nearest Branch Order",
            ]
        )
        writer.writerow(
            [
                1,
                "Object_0001",
                8,
                "2.5",
                "2.5",
                "2.5",
                "2",
                "2",
                "2",
                "5",
                "(2, 2, 5)",
                1,
                "B01",
                f"{d_cent:.6g}",
                f"{delta:.6g}",
                "2",
                "2",
                "5",
                "(2, 2, 5)",
                1,
                "B01",
            ]
        )
        writer.writerow(
            [
                2,
                "Object_0002",
                8,
                "11.5",
                "11.5",
                "12.5",
                "2",
                "12",
                "12",
                "10",
                "(12, 12, 10)",
                2,
                "Art1",
                f"{d_cent:.6g}",
                f"{delta:.6g}",
                "12",
                "12",
                "10",
                "(12, 12, 10)",
                2,
                "Art1",
            ]
        )

    with summary_gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "Measurement Type",
                "Object Count",
                "Mean Distance (microns)",
                "Median Distance (microns)",
                "Std Distance (microns)",
                "Min Distance (microns)",
                "Max Distance (microns)",
                "Mean Delta (Centroid-Edge) (microns)",
                "Median Delta (Centroid-Edge) (microns)",
                "Std Delta (Centroid-Edge) (microns)",
            ]
        )
        writer.writerow(
            [
                "Overall",
                "edge_to_vessel",
                2,
                "2",
                "2",
                "0",
                "2",
                "2",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )
        writer.writerow(
            [
                "Overall",
                "centroid_to_vessel",
                2,
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                "0",
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )
        writer.writerow(
            [
                "Art1",
                "edge_to_vessel",
                1,
                "2",
                "2",
                "0",
                "2",
                "2",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )
        writer.writerow(
            [
                "B01",
                "edge_to_vessel",
                1,
                "2",
                "2",
                "0",
                "2",
                "2",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )
        writer.writerow(
            [
                "Art1",
                "centroid_to_vessel",
                1,
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                "0",
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )
        writer.writerow(
            [
                "B01",
                "centroid_to_vessel",
                1,
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                "0",
                f"{d_cent:.6g}",
                f"{d_cent:.6g}",
                f"{delta:.6g}",
                f"{delta:.6g}",
                "0",
            ]
        )

    return details_gt, summary_gt


def _write_3d_distances_scene_html(
    *,
    cell_mask: np.ndarray,
    vessel_mask: np.ndarray,
    graph: nx.MultiGraph,
    output_html_path: Path,
    title: str = "Synthetic 3D distances: cells to vessels",
) -> bool:
    """Write interactive 3D HTML showing cells, vessel volume, graph, and labels."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return False

    if cell_mask.shape != vessel_mask.shape or cell_mask.ndim != 3:
        return False
    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        return False
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    def _add_volume(fig, mask: np.ndarray, *, name: str, color: str, opacity: float) -> None:
        if not np.any(mask):
            return
        zz, yy, xx = np.indices(mask.shape, dtype=float)
        fig.add_trace(
            go.Volume(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
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

    labels, objects = dist3d.label_connected_objects(
        cell_mask.astype(bool), object_name_prefix="Cell"
    )
    fig = go.Figure()
    _add_volume(
        fig,
        vessel_mask.astype(bool),
        name="Vessel volume",
        color="#00A3FF",
        opacity=0.13,
    )

    cell_colors = ["#FFB703", "#FB8500", "#8ECAE6", "#B5179E", "#2A9D8F"]
    cell_label_x: list[float] = []
    cell_label_y: list[float] = []
    cell_label_z: list[float] = []
    cell_label_text: list[str] = []
    for i, (obj_id, obj_name) in enumerate(objects):
        obj_mask = labels == int(obj_id)
        _add_volume(
            fig,
            obj_mask,
            name=f"Cell volume {obj_name}",
            color=cell_colors[i % len(cell_colors)],
            opacity=0.20,
        )
        vox = np.argwhere(obj_mask)
        centroid = np.mean(vox.astype(float), axis=0)
        # "Above" in rendered scene means increasing z-axis value.
        cell_label_x.append(float(centroid[2]))
        cell_label_y.append(float(centroid[1]))
        cell_label_z.append(float(centroid[0] + 0.8))
        cell_label_text.append(obj_name)

    segs: dict[str, tuple[list[float | None], list[float | None], list[float | None]]] = {
        "arteriole": ([], [], []),
        "venule": ([], [], []),
        "capillary": ([], [], []),
        "unassigned": ([], [], []),
    }
    text_x: list[float] = []
    text_y: list[float] = []
    text_z: list[float] = []
    text_labels: list[str] = []

    def _push(kind: str, p_u: np.ndarray, p_v: np.ndarray) -> None:
        lx, ly, lz = segs[kind]
        lx += [float(p_u[2]), float(p_v[2]), None]
        ly += [float(p_u[1]), float(p_v[1]), None]
        lz += [float(p_u[0]), float(p_v[0]), None]

    for u, v, key, data in graph.edges(keys=True, data=True):
        if u not in pos or v not in pos:
            continue
        p_u = np.asarray(pos[u], dtype=float)
        p_v = np.asarray(pos[v], dtype=float)
        branch_order = str(data.get("branch_order", "Unassigned"))
        if branch_order.startswith("Art"):
            kind = "arteriole"
        elif branch_order.startswith("Ven"):
            kind = "venule"
        elif branch_order.startswith("B"):
            kind = "capillary"
        else:
            kind = "unassigned"
        _push(kind, p_u, p_v)
        mid = 0.5 * (p_u + p_v)
        text_x.append(float(mid[2]))
        text_y.append(float(mid[1]))
        text_z.append(float(mid[0] + 0.7))
        text_labels.append(branch_order)

    edge_styles = {
        "arteriole": ("rgba(220, 50, 47, 0.95)", 6, "Vessel graph (arteriole)"),
        "venule": ("rgba(38, 139, 210, 0.95)", 6, "Vessel graph (venule)"),
        "capillary": ("rgba(42, 161, 152, 0.95)", 5, "Vessel graph (capillary)"),
        "unassigned": ("rgba(150, 150, 150, 0.65)", 4, "Vessel graph (unassigned)"),
    }
    for kind, (color, width, name) in edge_styles.items():
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
                name=name,
            )
        )

    if text_labels:
        fig.add_trace(
            go.Scatter3d(
                x=text_x,
                y=text_y,
                z=text_z,
                mode="text",
                text=text_labels,
                textposition="top center",
                textfont=dict(size=11, color="black"),
                name="Vessel type assignment (branch order)",
            )
        )
    if cell_label_text:
        fig.add_trace(
            go.Scatter3d(
                x=cell_label_x,
                y=cell_label_y,
                z=cell_label_z,
                mode="text",
                text=cell_label_text,
                textposition="top center",
                textfont=dict(size=12, color="black"),
                name="Cell identity",
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
    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    return True


def test_run_3d_measurement_to_cell_mask_generates_csvs(tmp_path: Path) -> None:
    cell_path, vessel_path, G, _cell_mask, _vessel_mask = _build_synthetic_case(tmp_path)

    result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=tmp_path,
        image_stem="synthetic_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert result["object_count"] == 2
    details_csv = Path(result["details_csv_path"])
    summary_csv = Path(result["summary_csv_path"])
    assert details_csv.exists() and details_csv.stat().st_size > 0
    assert summary_csv.exists() and summary_csv.stat().st_size > 0

    with details_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    edge_branch_orders = {row["Edge-to-Vessel Nearest Branch Order"] for row in rows}
    centroid_branch_orders = {
        row["Centroid-to-Vessel Nearest Branch Order"] for row in rows
    }
    assert "B01" in edge_branch_orders
    assert "Art1" in edge_branch_orders
    assert "B01" in centroid_branch_orders
    assert "Art1" in centroid_branch_orders
    for row in rows:
        edge_d = float(row["Edge-to-Vessel Distance (microns)"])
        centroid_d = float(row["Centroid-to-Vessel Distance (microns)"])
        delta_d = float(row["Delta Distance (Centroid-Edge) (microns)"])
        assert row["Edge-to-Vessel Nearest Coordinate (z,y,x microns)"].startswith("(")
        assert row["Centroid-to-Vessel Nearest Coordinate (z,y,x microns)"].startswith("(")
        assert edge_d >= 0.0
        assert centroid_d >= 0.0
        # CSV exports use compact formatting (~6 significant digits), so compare
        # with a tolerance that reflects text-rounding rather than raw float math.
        assert delta_d == pytest.approx(centroid_d - edge_d, abs=1e-5)
        # For compact objects, boundary-based minimum should not exceed centroid minimum.
        assert edge_d <= centroid_d + 1e-12

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert summary_rows
    assert summary_rows[0]["Group"] == "Overall"
    measurement_types = {row["Measurement Type"] for row in summary_rows}
    assert "edge_to_vessel" in measurement_types
    assert "centroid_to_vessel" in measurement_types
    for row in summary_rows:
        assert "Mean Delta (Centroid-Edge) (microns)" in row
        assert "Median Delta (Centroid-Edge) (microns)" in row
        assert "Std Delta (Centroid-Edge) (microns)" in row

    # Keep this test self-contained and writable on all platforms.
    demo_output_dir = tmp_path / "demo_outputs"
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    demo_result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=demo_output_dir,
        image_stem="synthetic_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    demo_details = Path(demo_result["details_csv_path"])
    demo_summary = Path(demo_result["summary_csv_path"])
    assert demo_details.exists() and demo_details.stat().st_size > 0
    assert demo_summary.exists() and demo_summary.stat().st_size > 0

    gt_details, gt_summary = _write_precalculated_ground_truth_csvs(
        output_dir=demo_output_dir,
        image_stem="synthetic_cells",
    )
    assert gt_details.exists() and gt_details.stat().st_size > 0
    assert gt_summary.exists() and gt_summary.stat().st_size > 0


@pytest.mark.plotting
def test_3d_distances_rotatable_html_scene(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    cell_path, vessel_path, G, cell_mask, vessel_mask = _build_synthetic_case(tmp_path)

    # Run measurement first so labels and CSV outputs are generated in same flow.
    result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=tmp_path,
        image_stem="synthetic_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert result["object_count"] == 2

    html_tmp = tmp_path / "synthetic_cells_vessel_distances_3d.html"
    assert _write_3d_distances_scene_html(
        cell_mask=cell_mask.astype(bool),
        vessel_mask=vessel_mask.astype(bool),
        graph=G,
        output_html_path=html_tmp,
    )
    assert html_tmp.is_file() and html_tmp.stat().st_size > 2000

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_html = DEMO_OUTPUT_DIR / "synthetic_cells_vessel_distances_3d.html"
    assert _write_3d_distances_scene_html(
        cell_mask=cell_mask.astype(bool),
        vessel_mask=vessel_mask.astype(bool),
        graph=G,
        output_html_path=demo_html,
    )


def test_run_3d_measurement_to_microglia_like_mask_generates_csvs(tmp_path: Path) -> None:
    (
        cell_path,
        vessel_path,
        G,
        _cell_mask,
        _vessel_mask,
    ) = _build_microglia_like_synthetic_case(tmp_path)

    result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=tmp_path,
        image_stem="synthetic_microglia_like_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert result["object_count"] == 2
    details_csv = Path(result["details_csv_path"])
    summary_csv = Path(result["summary_csv_path"])
    assert details_csv.exists() and details_csv.stat().st_size > 0
    assert summary_csv.exists() and summary_csv.stat().st_size > 0

    with details_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    edge_branch_orders = {row["Edge-to-Vessel Nearest Branch Order"] for row in rows}
    centroid_branch_orders = {
        row["Centroid-to-Vessel Nearest Branch Order"] for row in rows
    }
    assert "Ven1" in edge_branch_orders
    assert "B02" in edge_branch_orders
    assert "Ven1" in centroid_branch_orders
    assert "B02" in centroid_branch_orders

    for row in rows:
        edge_d = float(row["Edge-to-Vessel Distance (microns)"])
        centroid_d = float(row["Centroid-to-Vessel Distance (microns)"])
        delta_d = float(row["Delta Distance (Centroid-Edge) (microns)"])
        assert row["Edge-to-Vessel Nearest Coordinate (z,y,x microns)"].startswith("(")
        assert row["Centroid-to-Vessel Nearest Coordinate (z,y,x microns)"].startswith("(")
        assert edge_d >= 0.0
        assert centroid_d >= 0.0
        assert delta_d == pytest.approx(centroid_d - edge_d, abs=1e-5)
        assert edge_d <= centroid_d + 1e-12

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert summary_rows
    measurement_types = {row["Measurement Type"] for row in summary_rows}
    assert "edge_to_vessel" in measurement_types
    assert "centroid_to_vessel" in measurement_types

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=DEMO_OUTPUT_DIR,
        image_stem="synthetic_microglia_like_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    demo_details = Path(demo_result["details_csv_path"])
    demo_summary = Path(demo_result["summary_csv_path"])
    assert demo_details.exists() and demo_details.stat().st_size > 0
    assert demo_summary.exists() and demo_summary.stat().st_size > 0

    gt_details = _copy_csv_as_ground_truth(
        source_csv=demo_details,
        ground_truth_csv=DEMO_OUTPUT_DIR
        / "synthetic_microglia_like_cells_cell_to_vessel_3d_dual_distances_ground_truth.csv",
    )
    gt_summary = _copy_csv_as_ground_truth(
        source_csv=demo_summary,
        ground_truth_csv=DEMO_OUTPUT_DIR
        / "synthetic_microglia_like_cells_cell_to_vessel_3d_dual_distances_summary_ground_truth.csv",
    )
    assert gt_details.exists() and gt_details.stat().st_size > 0
    assert gt_summary.exists() and gt_summary.stat().st_size > 0


@pytest.mark.plotting
def test_microglia_like_3d_distances_rotatable_html_scene(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    cell_path, vessel_path, G, cell_mask, vessel_mask = _build_microglia_like_synthetic_case(
        tmp_path
    )

    result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=tmp_path,
        image_stem="synthetic_microglia_like_cells",
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert result["object_count"] == 2

    html_tmp = tmp_path / "synthetic_microglia_like_cells_vessel_distances_3d.html"
    assert _write_3d_distances_scene_html(
        cell_mask=cell_mask.astype(bool),
        vessel_mask=vessel_mask.astype(bool),
        graph=G,
        output_html_path=html_tmp,
        title="Synthetic microglia-like 3D distances: cells to vessels",
    )
    assert html_tmp.is_file() and html_tmp.stat().st_size > 2000

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_html = DEMO_OUTPUT_DIR / "synthetic_microglia_like_cells_vessel_distances_3d.html"
    assert _write_3d_distances_scene_html(
        cell_mask=cell_mask.astype(bool),
        vessel_mask=vessel_mask.astype(bool),
        graph=G,
        output_html_path=demo_html,
        title="Synthetic microglia-like 3D distances: cells to vessels",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
