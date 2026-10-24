"""Tests for 3D object-to-vessel distance measurement.

Every distance is a gap between voxelised volumes (each voxel the box it
fills), so a cell touching a vessel is 0 um from it; the cases below are laid
out so each answer can be counted off by hand.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.pipeline import default_schema, stages
from haemolynx.pipeline.checks import check_cell_mask_measurement, preflight
from haemolynx.statistics import three_dim_distances as dist3d

SCHEMA = default_schema()


def _line_voxels(
    p0: tuple[float, float, float], p1: tuple[float, float, float], n: int = 20
) -> list[tuple[float, float, float]]:
    t = np.linspace(0.0, 1.0, int(n), dtype=float)
    a = np.asarray(p0, dtype=float).reshape(1, 3)
    b = np.asarray(p1, dtype=float).reshape(1, 3)
    pts = (1.0 - t).reshape(-1, 1) * a + t.reshape(-1, 1) * b
    uniq = np.unique(np.rint(pts).astype(int), axis=0)
    return [tuple(float(v) for v in row) for row in uniq]


def _write_mask(path: Path, mask: np.ndarray, voxel_size_xyz=None) -> Path:
    """A uint8 TIFF; with *voxel_size_xyz*, an ImageJ one saying so."""
    data = np.asarray(mask, dtype=np.uint8)
    if voxel_size_xyz is None:
        tifffile.imwrite(path, data)
    else:
        x, y, z = voxel_size_xyz
        tifffile.imwrite(
            path, data, imagej=True, resolution=(1.0 / x, 1.0 / y),
            metadata={"spacing": z, "unit": "um"},
        )
    return path


def _tube(shape, centre_zy, radius, *, spacing_zy=(1.0, 1.0)) -> np.ndarray:
    """A vessel along x: every voxel within *radius* um of (z, y) = *centre_zy*."""
    zz, yy = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    disc = ((zz - centre_zy[0]) * spacing_zy[0]) ** 2 + (
        (yy - centre_zy[1]) * spacing_zy[1]
    ) ** 2 <= radius**2
    return np.repeat(disc[:, :, None], shape[2], axis=2)


def _rod_graph(z, y, x0, x1, branch_order, *, spacing_zyx=(1.0, 1.0, 1.0), first_node=0, **data):
    """One edge along x through voxels (z, y, x0..x1), positions in microns."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    points = [np.array([z, y, x], dtype=float) * spacing for x in range(x0, x1 + 1)]
    G = nx.MultiGraph()
    G.add_node(first_node, pos=points[0])
    G.add_node(first_node + 1, pos=points[-1])
    G.add_edge(
        first_node, first_node + 1,
        voxels=[tuple(p) for p in points], branch_order=branch_order, **data,
    )
    return G


def _measure(tmp_path, cells, G, *, vessels=None, vessel_array=None, voxel_size_xyz=(1.0, 1.0, 1.0),
             cell_voxel_size_xyz=None, **kwargs):
    """Records for *cells* against *vessels* (written to disk), *vessel_array*
    (handed over as the pipeline does), or neither (the graph's tubes)."""
    stem = f"case{len(list(tmp_path.glob('*.tif')))}"
    cell_path = _write_mask(tmp_path / f"{stem}_cells.tif", cells, cell_voxel_size_xyz)
    vessel_path = None
    if vessels is not None:
        vessel_path = _write_mask(tmp_path / f"{stem}_vessels.tif", vessels, cell_voxel_size_xyz)
    return dist3d._run_3d_measurement_records(
        graph=G,
        cell_mask_path=cell_path,
        voxel_size_xyz=voxel_size_xyz,
        vessel_mask_path=vessel_path,
        vessel_mask=vessel_array,
        cell_mask_h5_dataset_name=None,
        vessel_mask_h5_dataset_name=None,
        connectivity=3,
        axis_order="zyx",
        voxel_size_override_xyz=kwargs.pop("voxel_size_override_xyz", None),
        voxel_size_policy=kwargs.pop("voxel_size_policy", "auto"),
        use_memmap=False,
        memmap_directory=None,
        disk_backed=[],
    )


# --- the two-cell case, against numbers counted by hand ------------------------


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
    )
    G.add_edge(
        2,
        3,
        voxels=_line_voxels((12.0, 12.0, 2.0), (12.0, 12.0, 10.0)),
        branch_order="Art1",
        length=8.0,
    )
    return cell_path, vessel_path, G, cell_mask, vessel_mask


#: The two cells, counted by hand. Each lies one empty voxel from its trunk
#: along x (gap 1 um); its centroid sits on the trunk's line in z and y, 2.5
#: voxels from the nearest trunk voxel's centre along x (gap 2 um).
_EXPECTED_DETAILS = [
    {
        "Object ID": "1", "Object Name": "Object_0001", "Voxel Count": "8",
        "Centroid Z (voxel)": "2.5", "Centroid Y (voxel)": "2.5", "Centroid X (voxel)": "2.5",
        "Centroid Z (um)": "2.5", "Centroid Y (um)": "2.5", "Centroid X (um)": "2.5",
        "Vessel Contact": "separate", "Vessel Overlap Voxel Count": "0",
        "Edge-to-Vessel Distance (um)": "1",
        "Edge-to-Vessel Nearest Coordinate (z,y,x um)": "(2, 2, 5)",
        "Edge-to-Vessel Nearest Graph Edge Label ID": "1",
        "Edge-to-Vessel Nearest Branch Order": "B01",
        "Centroid-to-Vessel Distance (um)": "2",
        "Delta Distance (Centroid-Edge) (um)": "1",
        "Centroid-to-Vessel Nearest Coordinate (z,y,x um)": "(2, 2, 5)",
        "Centroid-to-Vessel Nearest Graph Edge Label ID": "1",
        "Centroid-to-Vessel Nearest Branch Order": "B01",
    },
    {
        "Object ID": "2", "Object Name": "Object_0002", "Voxel Count": "8",
        "Centroid Z (voxel)": "11.5", "Centroid Y (voxel)": "11.5", "Centroid X (voxel)": "12.5",
        "Centroid Z (um)": "11.5", "Centroid Y (um)": "11.5", "Centroid X (um)": "12.5",
        "Vessel Contact": "separate", "Vessel Overlap Voxel Count": "0",
        "Edge-to-Vessel Distance (um)": "1",
        "Edge-to-Vessel Nearest Coordinate (z,y,x um)": "(12, 12, 10)",
        "Edge-to-Vessel Nearest Graph Edge Label ID": "2",
        "Edge-to-Vessel Nearest Branch Order": "Art1",
        "Centroid-to-Vessel Distance (um)": "2",
        "Delta Distance (Centroid-Edge) (um)": "1",
        "Centroid-to-Vessel Nearest Coordinate (z,y,x um)": "(12, 12, 10)",
        "Centroid-to-Vessel Nearest Graph Edge Label ID": "2",
        "Centroid-to-Vessel Nearest Branch Order": "Art1",
    },
]

_EXPECTED_SUMMARY = [
    ["Overall", "edge_to_vessel", "2", "1", "1", "0", "1", "1", "1", "1", "0"],
    ["Overall", "centroid_to_vessel", "2", "2", "2", "0", "2", "2", "1", "1", "0"],
    ["Art1", "edge_to_vessel", "1", "1", "1", "0", "1", "1", "1", "1", "0"],
    ["B01", "edge_to_vessel", "1", "1", "1", "0", "1", "1", "1", "1", "0"],
    ["Art1", "centroid_to_vessel", "1", "2", "2", "0", "2", "2", "1", "1", "0"],
    ["B01", "centroid_to_vessel", "1", "2", "2", "0", "2", "2", "1", "1", "0"],
]


def test_run_3d_measurement_to_cell_mask_writes_the_hand_counted_csvs(tmp_path: Path) -> None:
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
    with Path(result["details_csv_path"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [{k: row[k] for k in expected} for row, expected in zip(rows, _EXPECTED_DETAILS)] == (
        _EXPECTED_DETAILS
    )
    assert len(rows) == 2
    with Path(result["summary_csv_path"]).open(newline="", encoding="utf-8") as f:
        summary = list(csv.reader(f))
    assert summary[1:] == _EXPECTED_SUMMARY


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
    )
    G.add_edge(
        2,
        3,
        voxels=_line_voxels((20.0, 20.0, 4.0), (20.0, 20.0, 13.0)),
        branch_order="B02",
        length=9.0,
    )
    return cell_path, vessel_path, G, cell_mask, vessel_mask


def test_microglia_like_cells_measure_from_their_processes(tmp_path: Path) -> None:
    """Cell 1's process ends one empty voxel short of its vessel; cell 2's
    process runs two voxels into its vessel (x = 12, 13)."""
    cell_path, vessel_path, G, _cell_mask, _vessel_mask = _build_microglia_like_synthetic_case(
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

    with Path(result["details_csv_path"]).open(newline="", encoding="utf-8") as f:
        rows = {row["Object ID"]: row for row in csv.DictReader(f)}
    assert rows["1"]["Edge-to-Vessel Distance (um)"] == "1"
    assert rows["1"]["Vessel Contact"] == "separate"
    assert rows["1"]["Edge-to-Vessel Nearest Coordinate (z,y,x um)"] == "(6, 6, 16)"
    assert rows["1"]["Edge-to-Vessel Nearest Branch Order"] == "Ven1"
    assert rows["2"]["Edge-to-Vessel Distance (um)"] == "0"
    assert rows["2"]["Vessel Contact"] == "overlapping"
    assert rows["2"]["Vessel Overlap Voxel Count"] == "2"
    assert rows["2"]["Edge-to-Vessel Nearest Branch Order"] == "B02"
    # The soma is far from either vessel; its centroid is no nearer than its process.
    for row in rows.values():
        assert float(row["Centroid-to-Vessel Distance (um)"]) > float(
            row["Edge-to-Vessel Distance (um)"]
        )


# --- the gap and the contact ---------------------------------------------------


def test_a_cell_touching_a_vessel_is_0_um_from_it(tmp_path):
    """Regression: voxel centre to voxel centre, a touching cell read one
    voxel step, and along z the z step."""
    shape = (41, 41, 40)
    vessels = _tube(shape, (20, 20), 8.0)
    G = _rod_graph(20, 20, 0, 39, "Art1")
    cells = np.zeros(shape, dtype=bool)
    cells[20, 29:32, 18:22] = True  # the wall is at y = 28

    (record,) = _measure(tmp_path, cells, G, vessels=vessels)

    assert record.edge_to_vessel_distance_microns == 0.0
    assert record.vessel_contact == "touching"
    assert record.vessel_overlap_voxel_count == 0
    assert (record.edge_to_vessel_nearest_z, record.edge_to_vessel_nearest_y) == (20.0, 28.0)
    # The centroid (y = 30) is 1.5 um past the wall voxel's far face.
    assert record.centroid_to_vessel_distance_microns == pytest.approx(1.5)


def test_the_gap_counts_the_empty_voxels_between_with_anisotropic_spacing(tmp_path):
    """z is 2 um, y and x 0.5 um: two empty slices between are 4 um, and a cell
    on the next slice touches."""
    shape = (20, 20, 20)
    vessels = np.zeros(shape, dtype=bool)
    vessels[5, 10, 2:18] = True
    G = _rod_graph(5, 10, 2, 17, "B01", spacing_zyx=(2.0, 0.5, 0.5))
    voxel_size_xyz = (0.5, 0.5, 2.0)

    above = np.zeros(shape, dtype=bool)
    above[8:10, 10, 9:11] = True
    (record,) = _measure(tmp_path, above, G, vessels=vessels, voxel_size_xyz=voxel_size_xyz)
    assert record.edge_to_vessel_distance_microns == pytest.approx(4.0)
    # Centroid z = 8.5, 3 slices past the vessel voxel's face at z = 5.5.
    assert record.centroid_to_vessel_distance_microns == pytest.approx(6.0)
    assert record.centroid_z_microns == pytest.approx(17.0)

    beside = np.zeros(shape, dtype=bool)
    beside[5, 14:16, 9:11] = True  # y 11, 12, 13 empty
    (record,) = _measure(tmp_path, beside, G, vessels=vessels, voxel_size_xyz=voxel_size_xyz)
    assert record.edge_to_vessel_distance_microns == pytest.approx(1.5)

    on_the_next_slice = np.zeros(shape, dtype=bool)
    on_the_next_slice[6:8, 10, 9:11] = True
    (record,) = _measure(tmp_path, on_the_next_slice, G, vessels=vessels, voxel_size_xyz=voxel_size_xyz)
    assert record.edge_to_vessel_distance_microns == 0.0
    assert record.vessel_contact == "touching"


def test_a_cell_overlapping_or_inside_a_vessel_is_0_um_from_it_and_says_so(tmp_path):
    shape = (41, 41, 40)
    vessels = _tube(shape, (20, 20), 8.0)
    G = _rod_graph(20, 20, 0, 39, "Art1")

    inside = np.zeros(shape, dtype=bool)
    inside[20, 22:26, 18:22] = True
    (record,) = _measure(tmp_path, inside, G, vessels=vessels)
    assert record.edge_to_vessel_distance_microns == 0.0
    assert record.centroid_to_vessel_distance_microns == 0.0
    assert record.vessel_contact == "inside"
    assert record.vessel_overlap_voxel_count == record.voxel_count == 16
    assert record.edge_to_vessel_nearest_branch_order == "Art1"

    straddling = np.zeros(shape, dtype=bool)
    straddling[20, 26:31, 18:22] = True  # y 26-28 in the vessel, 29-30 out
    (record,) = _measure(tmp_path, straddling, G, vessels=vessels)
    assert record.edge_to_vessel_distance_microns == 0.0
    assert record.vessel_contact == "overlapping"
    assert record.vessel_overlap_voxel_count == 12


# --- which vessel --------------------------------------------------------------


def _arteriole_with_a_capillary_against_its_wall():
    """A radius-8 arteriole along x, and a one-voxel capillary running beside
    it just outside its wall, diagonally above (z, y) = (26, 26)."""
    shape = (41, 41, 40)
    capillary = np.zeros(shape, dtype=bool)
    capillary[26, 26, :] = True
    vessels = _tube(shape, (20, 20), 8.0) | capillary
    G = nx.compose(
        _rod_graph(20, 20, 0, 39, "Art1"),
        _rod_graph(26, 26, 0, 39, "B05", first_node=10),
    )
    return shape, vessels, G


def test_a_cell_against_a_wide_vessels_wall_is_named_for_it_not_a_capillary_running_past(
    tmp_path,
):
    """Regression: the wall voxel nearest the cell is 8 um from the arteriole's
    centreline and 6.3 um from the capillary's, so the nearest centreline
    named the capillary."""
    shape, vessels, G = _arteriole_with_a_capillary_against_its_wall()
    cells = np.zeros(shape, dtype=bool)
    cells[20, 29:32, 18:22] = True

    (record,) = _measure(tmp_path, cells, G, vessels=vessels)

    assert (record.edge_to_vessel_nearest_z, record.edge_to_vessel_nearest_y) == (20.0, 28.0)
    assert record.edge_to_vessel_nearest_branch_order == "Art1"
    assert record.centroid_to_vessel_nearest_branch_order == "Art1"


def test_a_cell_by_the_capillary_is_named_for_the_capillary(tmp_path):
    shape, vessels, G = _arteriole_with_a_capillary_against_its_wall()
    cells = np.zeros(shape, dtype=bool)
    cells[28:31, 28:31, 18:22] = True

    (record,) = _measure(tmp_path, cells, G, vessels=vessels)

    assert record.edge_to_vessel_nearest_branch_order == "B05"
    assert record.edge_to_vessel_distance_microns == pytest.approx(np.sqrt(2.0))


def test_a_thick_vessel_bridge_is_never_the_vessel_named(tmp_path):
    """A bridge runs from the wall to the centreline inside the lumen; its end
    on the wall is nearer the wall than any centreline, and it has no branch
    order."""
    shape = (33, 33, 40)
    vessels = _tube(shape, (16, 16), 6.0)
    G = _rod_graph(16, 16, 0, 39, "Art1")
    G.add_node(100, pos=np.array([16.0, 22.0, 20.0]))
    G.add_edge(
        100, 0, voxels=[(16.0, float(y), 20.0) for y in range(22, 15, -1)],
        **{IS_ZERO_RESISTANCE: True},
    )
    cells = np.zeros(shape, dtype=bool)
    cells[16, 23:26, 20] = True  # against the wall where the bridge ends

    (record,) = _measure(tmp_path, cells, G, vessels=vessels)

    assert record.edge_to_vessel_distance_microns == 0.0
    assert (record.edge_to_vessel_nearest_y, record.edge_to_vessel_nearest_x) == (22.0, 20.0)
    assert record.edge_to_vessel_nearest_branch_order == "Art1"


# --- what the vessels are ------------------------------------------------------


def test_without_a_vessel_mask_the_vessels_are_tubes_of_their_own_diameter(tmp_path):
    """Regression: drawn from the graph, the vessels were their centrelines,
    so a cell against a radius-8 vessel's wall read 9 um."""
    shape = (41, 41, 40)
    G = _rod_graph(20, 20, 0, 39, "Art1", diameter_um=16.0)
    cells = np.zeros(shape, dtype=bool)
    cells[20, 29:32, 18:22] = True

    (record,) = _measure(tmp_path, cells, G)

    assert record.edge_to_vessel_distance_microns == 0.0
    assert record.vessel_contact == "touching"
    assert record.edge_to_vessel_nearest_branch_order == "Art1"


def test_a_vessel_without_a_diameter_is_its_centreline_and_the_log_says_so(tmp_path, caplog):
    shape = (41, 41, 40)
    G = _rod_graph(20, 20, 0, 39, "Art1")
    cells = np.zeros(shape, dtype=bool)
    cells[20, 29:32, 18:22] = True

    with caplog.at_level(logging.WARNING, logger=dist3d.__name__):
        (record,) = _measure(tmp_path, cells, G)

    assert record.edge_to_vessel_distance_microns == pytest.approx(8.0)
    assert "1 vessel(s) have no diameter_um" in caplog.text


def test_a_vessel_mask_handed_over_as_an_array_is_measured_as_given(tmp_path):
    """What the pipeline does with the run's segmentation: a 0/255 array, not
    cut again."""
    shape = (41, 41, 40)
    vessels = _tube(shape, (20, 20), 8.0).astype(np.uint8) * 255
    G = _rod_graph(20, 20, 0, 39, "Art1")
    cells = np.zeros(shape, dtype=bool)
    cells[20, 33:36, 18:22] = True  # y 29-32 empty

    (record,) = _measure(tmp_path, cells, G, vessel_array=vessels)

    assert record.edge_to_vessel_distance_microns == pytest.approx(4.0)
    assert record.edge_to_vessel_nearest_branch_order == "Art1"


def test_a_vessel_array_of_another_shape_is_refused(tmp_path):
    cells = np.zeros((10, 10, 10), dtype=bool)
    cells[2:4, 2:4, 2:4] = True
    with pytest.raises(ValueError, match="input image's shape"):
        _measure(
            tmp_path, cells, _rod_graph(5, 5, 0, 9, "B01"),
            vessel_array=np.ones((10, 10, 11), dtype=bool),
        )


# --- voxel sizes ---------------------------------------------------------------


def _anisotropic_case():
    shape = (20, 20, 20)
    vessels = np.zeros(shape, dtype=bool)
    vessels[5, 10, 2:18] = True
    cells = np.zeros(shape, dtype=bool)
    cells[8:10, 10, 9:11] = True
    G = _rod_graph(5, 10, 2, 17, "B01", spacing_zyx=(2.0, 0.5, 0.5))
    return cells, vessels, G


def test_a_cell_mask_without_a_voxel_size_of_its_own_takes_the_runs(tmp_path):
    """Regression: an ilastik export carries no voxel size, read as 1 um, and
    was refused on any other run."""
    cells, vessels, G = _anisotropic_case()

    (record,) = _measure(
        tmp_path, cells, G, vessels=vessels, voxel_size_xyz=(0.5, 0.5, 2.0),
        cell_voxel_size_xyz=None,
    )

    assert record.edge_to_vessel_distance_microns == pytest.approx(4.0)


def test_a_cell_mask_saying_another_voxel_size_is_refused(tmp_path):
    cells, vessels, G = _anisotropic_case()
    with pytest.raises(ValueError, match="cell mask's voxel size"):
        _measure(
            tmp_path, cells, G, vessels=vessels, voxel_size_xyz=(0.5, 0.5, 2.0),
            cell_voxel_size_xyz=(0.8, 0.8, 3.0),
        )


def test_the_runs_voxel_size_override_applies_to_the_masks(tmp_path):
    cells, vessels, G = _anisotropic_case()

    (record,) = _measure(
        tmp_path, cells, G, vessels=vessels, voxel_size_xyz=(0.5, 0.5, 2.0),
        cell_voxel_size_xyz=(0.8, 0.8, 3.0),
        voxel_size_override_xyz=(0.5, 0.5, 2.0), voxel_size_policy="override",
    )

    assert record.edge_to_vessel_distance_microns == pytest.approx(4.0)


# --- the export stage ----------------------------------------------------------


def _stage(tmp_path, **overrides):
    """An export stage over a radius-4 vessel, the run's segmentation, with a
    cell against its wall; the graph has its centreline and no diameters."""
    shape = (21, 21, 30)
    segmentation = _tube(shape, (10, 10), 4.0)
    cells = np.zeros(shape, dtype=bool)
    cells[10, 15:18, 10:14] = True  # the wall is at y = 14
    settings = SCHEMA.defaults()
    settings.update(
        input_path=tmp_path / "seg.tif",
        statistics=False,
        run_haemodynamics=False,
        vtk_export=False,
        visualize_vtk=False,
        visualize_results=False,
        export_citations=False,
        measurement_3d_to_cell_mask=True,
        cell_mask_path=_write_mask(tmp_path / "cells.tif", cells),
    )
    settings.update(overrides)
    volume = stages.SkeletonisedVolume(
        image=segmentation.astype(np.uint8),
        skeleton=np.zeros(shape, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    G = _rod_graph(10, 10, 0, 29, "Art1")
    stages.export_results(
        settings, stages.VesselNetwork(graph=G, volume=volume),
        stages.HaemodynamicModel(graph=G), stages.Solution(),
    )
    with (tmp_path / "seg_cell_to_vessel_3d_dual_distances.csv").open(encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    return row, shape


def test_the_export_stage_measures_to_the_runs_segmentation(tmp_path):
    """Regression: with no vessel mask of its own the stage measured to the
    graph's centreline, 4 um further than the wall."""
    row, _shape = _stage(tmp_path)
    assert row["Edge-to-Vessel Distance (um)"] == "0"
    assert row["Vessel Contact"] == "touching"
    assert row["Edge-to-Vessel Nearest Branch Order"] == "Art1"


def test_a_vessel_mask_of_its_own_takes_the_place_of_the_segmentation(tmp_path):
    thin = _tube((21, 21, 30), (10, 10), 1.0)  # its wall at y = 11
    row, _shape = _stage(
        tmp_path,
        measurement_3d_vessel_mask_path=_write_mask(tmp_path / "thin_vessels.tif", thin),
    )
    assert row["Edge-to-Vessel Distance (um)"] == "3"


# --- preflight -----------------------------------------------------------------


def _preflight_settings(tmp_path, cell_shape=(10, 12, 14), cell_voxel_size_xyz=None, **overrides):
    image = _write_mask(tmp_path / "image.tif", np.zeros((10, 12, 14)), (0.5, 0.5, 2.0))
    cells = _write_mask(tmp_path / "cells.tif", np.zeros(cell_shape), cell_voxel_size_xyz)
    settings = {
        "measurement_3d_to_cell_mask": True,
        "cell_mask_path": cells,
        "input_path": image,
        "image_axis_order": "zyx",
        "voxel_size_override_xyz": None,
        "voxel_size_policy": "auto",
    }
    settings.update(overrides)
    return settings


def test_preflight_passes_a_cell_mask_that_fits_the_input_image(tmp_path):
    for voxel_size in (None, (0.5, 0.5, 2.0)):
        report = check_cell_mask_measurement(
            _preflight_settings(tmp_path, cell_voxel_size_xyz=voxel_size)
        )
        assert report.ok, report.errors
        assert [label for label, _ in report.passed] == ["cell_mask_path"]


def test_preflight_catches_what_would_stop_the_measurement_at_the_end_of_the_run(tmp_path):
    report = check_cell_mask_measurement(_preflight_settings(tmp_path, cell_shape=(10, 12, 15)))
    assert any("(10, 12, 15) voxels" in e for e in report.errors)

    report = check_cell_mask_measurement(
        _preflight_settings(tmp_path, cell_voxel_size_xyz=(0.8, 0.8, 3.0))
    )
    assert any("voxel size" in e and "cell_mask_path" in e for e in report.errors)

    vessels = _write_mask(tmp_path / "vessels.tif", np.zeros((10, 12, 13)))
    report = check_cell_mask_measurement(
        _preflight_settings(tmp_path, measurement_3d_vessel_mask_path=vessels)
    )
    assert any(e.startswith("measurement_3d_vessel_mask_path is (10, 12, 13)") for e in report.errors)


def test_preflight_compares_shapes_as_each_file_is_read_under_an_auto_axis_order(tmp_path):
    """Under image_axis_order="auto" the input image may name its axes x, y, z
    while the cell mask, an ImageJ stack, is z, y, x: the two are compared
    as the run reads them, in (z, y, x), not as stored."""
    image = tmp_path / "image_xyz.tif"
    tifffile.imwrite(image, np.zeros((14, 12, 10), dtype=np.uint8), metadata={"axes": "XYZ"})
    cells = tmp_path / "cells_zyx.tif"
    tifffile.imwrite(
        cells, np.zeros((10, 12, 14), dtype=np.uint8), imagej=True, resolution=(2.0, 2.0),
        metadata={"spacing": 2.0, "unit": "um", "axes": "ZYX"},
    )
    settings = _preflight_settings(
        tmp_path,
        cell_mask_path=cells,
        input_path=image,
        image_axis_order="auto",
        voxel_size_override_xyz=[0.5, 0.5, 2.0],
        voxel_size_policy="override",
    )

    assert check_cell_mask_measurement(settings).ok, check_cell_mask_measurement(settings).errors
    report = check_cell_mask_measurement({**settings, "image_axis_order": "zyx"})
    assert any("(10, 12, 14) voxels where the input image is (14, 12, 10)" in e for e in report.errors)


def _preflight_errors(tmp_path, **overrides):
    settings = {**SCHEMA.defaults(), **_preflight_settings(tmp_path), **overrides}
    return [e for e in preflight(settings, SCHEMA).errors if "mask" in e]


def test_preflight_requires_a_cell_mask_but_not_a_vessel_mask_of_its_own(tmp_path):
    """Regression: must_exist with no placeholder on the optional vessel mask
    made preflight refuse every run measuring to the run's own segmentation."""
    assert _preflight_errors(tmp_path) == []

    errors = _preflight_errors(tmp_path, cell_mask_path=None)
    assert len(errors) == 1 and "'cell_mask_path' is not set" in errors[0]


def test_preflight_reports_a_vessel_mask_that_is_set_but_missing_once(tmp_path):
    errors = _preflight_errors(tmp_path, measurement_3d_vessel_mask_path=tmp_path / "gone.tif")
    assert len(errors) == 1
    assert errors[0].startswith("measurement_3d_vessel_mask_path: ") and "gone.tif" in errors[0]
    assert "clear it to use the run's segmented image" in errors[0]


def test_the_vessel_mask_opens_for_reading_in_the_panel():
    """An input file: the panel's dialogue picks an existing one, not a name to
    save to, which it would without must_exist."""
    from haemolynx.gui.form import field_for

    setting = SCHEMA["measurement_3d_vessel_mask_path"]
    assert setting.must_exist and setting.optional_path
    assert field_for(setting).options["mode"] == "r"


def test_preflight_says_nothing_with_the_measurement_off(tmp_path):
    report = check_cell_mask_measurement(
        _preflight_settings(tmp_path, measurement_3d_to_cell_mask=False, cell_mask_path=None)
    )
    assert report.ok and not report.passed


def test_preflight_runs_the_cell_mask_check(tmp_path):
    settings = {
        **SCHEMA.defaults(),
        **_preflight_settings(tmp_path, cell_voxel_size_xyz=(0.8, 0.8, 3.0)),
    }
    report = preflight(settings, SCHEMA)
    assert any("cell_mask_path" in e and "voxel size" in e for e in report.errors)


# --- settings ------------------------------------------------------------------


def test_the_reference_image_settings_are_retired_and_old_configs_still_load():
    """The reference image only had its shape checked against the cell mask's."""
    assert "measurement_3d_reference_image_path" not in SCHEMA.names
    assert "measurement_3d_reference_h5_dataset_name" not in SCHEMA.names
    resolved = SCHEMA.validate(
        {
            "measurement_3d_reference_image_path": "reference.tif",
            "measurement_3d_reference_h5_dataset_name": "data",
        }
    )
    assert "measurement_3d_reference_image_path" not in resolved
    assert resolved["measurement_3d_vessel_mask_path"] is None


# --- plots ---------------------------------------------------------------------


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


@pytest.mark.plotting
@pytest.mark.parametrize(
    "build", [_build_synthetic_case, _build_microglia_like_synthetic_case], ids=["cells", "microglia"]
)
def test_3d_distances_rotatable_html_scene(tmp_path: Path, build) -> None:
    pytest.importorskip("plotly.graph_objects")
    _cell_path, _vessel_path, G, cell_mask, vessel_mask = build(tmp_path)

    html = tmp_path / "vessel_distances_3d.html"
    assert _write_3d_distances_scene_html(
        cell_mask=cell_mask.astype(bool),
        vessel_mask=vessel_mask.astype(bool),
        graph=G,
        output_html_path=html,
    )
    assert html.is_file() and html.stat().st_size > 2000
