"""Tests for 3D object-to-vessel distance measurement."""
from __future__ import annotations

import csv
import importlib
import statistics
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


# Hand answers, one per object in label order. voxel_size is 1 um, so voxels are microns.
# Each vessel mask is one voxel thick, so erosion empties it and the whole mask is the wall.
_CELLS_ANSWER = [
    # A 2x2x2 cube centred at (2.5, 2.5, 2.5); its corner (2, 2, 3) is 2 um from the B01
    # trunk's first voxel (2, 2, 5), and the centre is sqrt(0.25 + 0.25 + 6.25) from it.
    dict(voxels=8, centroid=(2.5, 2.5, 2.5),
         edge_d=2.0, edge_at=(2, 2, 5), edge_label=1, edge_order="B01",
         centroid_d=float(np.sqrt(6.75)), centroid_at=(2, 2, 5), centroid_label=1,
         centroid_order="B01"),
    # The second cube is 2 um from the Art1 trunk's last voxel (12, 12, 10), the same way.
    dict(voxels=8, centroid=(11.5, 11.5, 12.5),
         edge_d=2.0, edge_at=(12, 12, 10), edge_label=2, edge_order="Art1",
         centroid_d=float(np.sqrt(6.75)), centroid_at=(12, 12, 10), centroid_label=2,
         centroid_order="Art1"),
]
_MICROGLIA_ANSWER = [
    # Soma 4^3 = 64 voxels at 5.5, processes along x (x 8-14, 7 voxels) and y (y 8-13,
    # 6 voxels). The x process's tip (6, 6, 14) is 2 um from the Ven1 trunk at (6, 6, 16).
    # Centroid (430, 457, 465) / 77, which is (32, 5, 767) / 77 from (6, 6, 16).
    dict(voxels=77, centroid=(430 / 77, 457 / 77, 465 / 77),
         edge_d=2.0, edge_at=(6, 6, 16), edge_label=1, edge_order="Ven1",
         centroid_d=float(np.sqrt(32**2 + 5**2 + 767**2) / 77), centroid_at=(6, 6, 16),
         centroid_label=1, centroid_order="Ven1"),
    # Soma at 19.5; the y process's y 18-19 lie in the soma, so it adds 4 voxels (y 14-17)
    # and the x process 6 (x 12-17): 74. That x process runs over the B02 trunk at
    # (20, 20, 12) and (20, 20, 13), so the edge distance is 0, first hit in raster order
    # (20, 20, 12). Centroid (1448, 1430, 1415) / 74, which is (32, 50, 453) / 74 from the
    # trunk's end (20, 20, 13).
    dict(voxels=74, centroid=(1448 / 74, 1430 / 74, 1415 / 74),
         edge_d=0.0, edge_at=(20, 20, 12), edge_label=2, edge_order="B02",
         centroid_d=float(np.sqrt(32**2 + 50**2 + 453**2) / 74), centroid_at=(20, 20, 13),
         centroid_label=2, centroid_order="B02"),
]


def _close(value):
    """The CSVs print six significant figures."""
    return pytest.approx(value, rel=1e-5, abs=1e-6)


def _coordinate(zyx):
    return "({}, {}, {})".format(*(f"{v:.6g}" for v in zyx))


def _summary_row(distances, deltas):
    """Count, mean, median, population std, min, max; then mean, median, std of the deltas."""
    return [
        len(distances),
        statistics.mean(distances), statistics.median(distances),
        statistics.pstdev(distances), min(distances), max(distances),
        statistics.mean(deltas), statistics.median(deltas), statistics.pstdev(deltas),
    ]


def _expected_summary(answer):
    """Overall rows, then one row per branch order (sorted), edge before centroid."""
    kinds = (("edge_to_vessel", "edge_d", "edge_order"),
             ("centroid_to_vessel", "centroid_d", "centroid_order"))
    rows = {}
    for kind, d_key, _ in kinds:
        rows[("Overall", kind)] = _summary_row(
            [a[d_key] for a in answer], [a["centroid_d"] - a["edge_d"] for a in answer])
    for kind, d_key, order_key in kinds:
        for group in sorted({a[order_key] for a in answer}):
            members = [a for a in answer if a[order_key] == group]
            rows[(group, kind)] = _summary_row(
                [a[d_key] for a in members], [a["centroid_d"] - a["edge_d"] for a in members])
    return rows


@pytest.mark.parametrize(
    "build_case, image_stem, answer",
    [
        pytest.param(_build_synthetic_case, "synthetic_cells", _CELLS_ANSWER, id="cells"),
        pytest.param(
            _build_microglia_like_synthetic_case, "synthetic_microglia_like_cells",
            _MICROGLIA_ANSWER, id="microglia_like"),
    ],
)
def test_3d_measurement_matches_the_hand_calculated_distances(
    tmp_path: Path, build_case, image_stem: str, answer: list[dict]
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
    assert result["object_count"] == len(answer)

    with Path(result["details_csv_path"]).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["Object ID"] for row in rows] == [str(i + 1) for i in range(len(answer))]
    for row, a in zip(rows, answer):
        assert int(row["Voxel Count"]) == a["voxels"]
        for axis, value in zip("ZYX", a["centroid"]):
            assert float(row[f"Centroid {axis} (voxel)"]) == _close(value)
        for prefix, d_key, at, label, order in (
            ("Edge-to-Vessel", "edge_d", "edge_at", "edge_label", "edge_order"),
            ("Centroid-to-Vessel", "centroid_d", "centroid_at", "centroid_label",
             "centroid_order"),
        ):
            assert float(row[f"{prefix} Distance (microns)"]) == _close(a[d_key])
            for axis, value in zip("ZYX", a[at]):
                assert float(row[f"{prefix} Nearest {axis} (microns)"]) == _close(value)
            assert row[f"{prefix} Nearest Coordinate (z,y,x microns)"] == _coordinate(a[at])
            assert int(row[f"{prefix} Nearest Graph Edge Label ID"]) == a[label]
            assert row[f"{prefix} Nearest Branch Order"] == a[order]
        assert float(row["Delta Distance (Centroid-Edge) (microns)"]) == _close(
            a["centroid_d"] - a["edge_d"])

    with Path(result["summary_csv_path"]).open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.reader(f))[1:]
    expected = _expected_summary(answer)
    assert [(row[0], row[1]) for row in summary_rows] == list(expected)
    for row in summary_rows:
        assert int(row[2]) == expected[(row[0], row[1])][0]
        assert [float(v) for v in row[3:]] == [_close(v) for v in expected[(row[0], row[1])][1:]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
