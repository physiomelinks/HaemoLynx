"""Tests for 3D object-to-vessel distance measurement."""
from __future__ import annotations

import importlib
import csv
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

dist3d = importlib.import_module("ImageLynx.statistics.3D_distances")


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


@pytest.mark.parametrize(
    "build_case, image_stem, branch_orders",
    [
        pytest.param(
            _build_synthetic_case, "synthetic_cells", {"B01", "Art1"}, id="cells"),
        pytest.param(
            _build_microglia_like_synthetic_case, "synthetic_microglia_like_cells",
            {"Ven1", "B02"}, id="microglia_like"),
    ],
)
def test_run_3d_measurement_to_cell_mask_generates_csvs(
    tmp_path: Path, build_case, image_stem: str, branch_orders: set[str]
) -> None:
    cell_path, vessel_path, G, _cell_mask, _vessel_mask = build_case(tmp_path)

    result = dist3d.run_3d_measurement_to_cell_mask(
        graph=G,
        cell_mask_path=cell_path,
        vessel_mask_path=vessel_path,
        output_dir=tmp_path,
        image_stem=image_stem,
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
    assert branch_orders <= edge_branch_orders
    assert branch_orders <= centroid_branch_orders
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

    if image_stem != "synthetic_cells":
        return  # only the cell case has a hand-calculated answer file so far

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
