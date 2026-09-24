"""The Low RAM option (``use_memmap_loading``) on the measurement side of a run:
same numbers, no volume-sized array in RAM.

Each test runs the ordinary path and the low-RAM path on the same input and
requires identical output, and where it matters also checks that the low-RAM
path really did keep its volumes on disk and tidied up after itself.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

from haemolynx.haemodynamics.automated import build_graph_branch_label_volume
from haemolynx.statistics import three_dim_distances as dist3d


def _blobs(shape, fraction, sigma, seed):
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random(shape), sigma)
    return field > np.quantile(field, 1.0 - fraction)


def _random_graph(shape, voxel_size_zyx, seed, n_edges=12):
    """Straight edges between random points, some crossing, so junction
    labels (-1) appear where two edges claim the same voxel."""
    rng = np.random.default_rng(seed)
    G = nx.MultiGraph()
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    for k in range(n_edges):
        a = rng.uniform(0, np.array(shape) - 1)
        b = rng.uniform(0, np.array(shape) - 1)
        n = int(np.linalg.norm(b - a) * 2) + 2
        pts = np.linspace(a, b, n) * spacing
        G.add_node(2 * k, pos=pts[0])
        G.add_node(2 * k + 1, pos=pts[-1])
        G.add_edge(2 * k, 2 * k + 1, voxels=pts.tolist(), branch_order=f"B{k:02d}")
    return G


def _write(path: Path, array: np.ndarray) -> Path:
    tifffile.imwrite(path, np.asarray(array, dtype=np.uint8) * 255)
    return path


# --- haemodynamics.automated.build_graph_branch_label_volume ----------------


def test_branch_label_volume_on_disk_matches_the_in_ram_one(tmp_path):
    shape, voxel = (20, 24, 28), (2.0, 0.5, 0.5)
    G_plain = _random_graph(shape, voxel, seed=0)
    G_disk = G_plain.copy()

    plain, plain_keys = build_graph_branch_label_volume(G_plain, shape, voxel)
    disk, disk_keys = build_graph_branch_label_volume(
        G_disk, shape, voxel, use_memmap=True, memmap_directory=tmp_path
    )

    assert isinstance(disk, np.memmap)
    assert (plain == -1).any(), "the fixture must produce junction voxels"
    assert np.array_equal(disk, plain)
    assert disk_keys == plain_keys


def test_branch_label_volume_on_disk_honours_a_non_zero_background(tmp_path):
    shape, voxel = (6, 7, 8), (1.0, 1.0, 1.0)
    G = _random_graph(shape, voxel, seed=1, n_edges=3)

    plain, _ = build_graph_branch_label_volume(G.copy(), shape, voxel, background_label=7)
    disk, _ = build_graph_branch_label_volume(
        G.copy(), shape, voxel, background_label=7, use_memmap=True, memmap_directory=tmp_path
    )

    assert np.array_equal(disk, plain)


# --- statistics.three_dim_distances ------------------------------------------


def _cell_case(tmp_path: Path, seed: int, voxel_zyx=(2.0, 0.6, 0.6)):
    shape = (18, 30, 34)
    cells = _blobs(shape, 0.05, 1.2, seed)
    # Objects touching the volume's faces, where the erosion's border applies.
    cells[0, 0:3, 0:3] = True
    cells[-2:, -3:, -3:] = True
    vessels = _blobs(shape, 0.12, 2.0, seed + 100) & ~cells
    G = _random_graph(shape, voxel_zyx, seed)
    return shape, voxel_zyx, cells, vessels, G


def test_object_distances_are_identical_in_low_ram_mode(tmp_path):
    shape, voxel_zyx, cells, vessels, G = _cell_case(tmp_path, seed=3)
    labels, _ = build_graph_branch_label_volume(G.copy(), shape, voxel_zyx)
    kwargs = dict(
        object_mask=cells,
        vessel_mask=vessels,
        voxel_size_zyx=voxel_zyx,
        graph_edge_label_volume=labels,
        edge_label_to_branch_order=dist3d._edge_label_to_branch_order_map(G),
    )

    plain = dist3d.compute_object_to_vessel_distances(**kwargs)
    low_ram = dist3d.compute_object_to_vessel_distances(
        **kwargs, use_memmap=True, memmap_directory=tmp_path / "mm"
    )

    assert len(plain) > 5
    assert low_ram == plain
    assert not list((tmp_path / "mm").iterdir()), "a temporary volume was left behind"


def test_object_distances_with_a_vessel_mask_too_thin_to_erode(tmp_path):
    """Every vessel voxel is boundary, so the erosion fallback is reached."""
    shape, voxel_zyx, cells, _vessels, _G = _cell_case(tmp_path, seed=4)
    vessels = np.zeros(shape, dtype=bool)
    vessels[5, 5, :] = True

    plain = dist3d.compute_object_to_vessel_distances(
        object_mask=cells, vessel_mask=vessels, voxel_size_zyx=voxel_zyx
    )
    low_ram = dist3d.compute_object_to_vessel_distances(
        object_mask=cells, vessel_mask=vessels, voxel_size_zyx=voxel_zyx, use_memmap=True
    )

    assert low_ram == plain


def _run(tmp_path, name, *, use_memmap, **kwargs):
    out = tmp_path / name
    summary = dist3d.run_3d_measurement_to_cell_mask(
        output_dir=out,
        image_stem="run",
        use_memmap=use_memmap,
        memmap_directory=tmp_path / f"{name}_mm" if use_memmap else None,
        **kwargs,
    )
    if use_memmap:
        leftovers = list((tmp_path / f"{name}_mm").glob("*"))
        assert not leftovers, f"temporary volumes left behind: {leftovers}"
    return (
        summary["object_count"],
        Path(summary["details_csv_path"]).read_text(),
        Path(summary["summary_csv_path"]).read_text(),
    )


def test_cell_mask_measurement_writes_the_same_csvs_in_low_ram_mode(tmp_path):
    shape, voxel_zyx, cells, vessels, G = _cell_case(tmp_path, seed=5, voxel_zyx=(1.0, 1.0, 1.0))
    common = dict(
        graph=G,
        cell_mask_path=_write(tmp_path / "cells.tif", cells),
        vessel_mask_path=_write(tmp_path / "vessels.tif", vessels),
        # The written TIFFs carry no metadata, which the loader reads as 1 µm.
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    plain = _run(tmp_path, "plain", use_memmap=False, **common)
    low_ram = _run(tmp_path, "low", use_memmap=True, **common)

    assert plain[0] > 5
    assert low_ram == plain


def test_cell_mask_measurement_from_graph_labels_is_the_same_in_low_ram_mode(tmp_path):
    """No vessel mask: the vessel volume is the graph's own label volume, with
    its shape read from a reference image."""
    shape, voxel_zyx, cells, _vessels, G = _cell_case(tmp_path, seed=6, voxel_zyx=(1.0, 1.0, 1.0))
    common = dict(
        graph=G,
        cell_mask_path=_write(tmp_path / "cells.tif", cells),
        vessel_reference_image_path=_write(tmp_path / "reference.tif", cells),
        # The written TIFFs carry no metadata, which the loader reads as 1 µm.
        voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    plain = _run(tmp_path, "plain", use_memmap=False, **common)
    low_ram = _run(tmp_path, "low", use_memmap=True, **common)

    assert plain[0] > 5
    assert low_ram == plain


def test_reference_shape_comes_from_the_tiff_header_in_low_ram_mode(tmp_path, monkeypatch):
    path = tmp_path / "reference.tif"
    tifffile.imwrite(path, np.zeros((5, 6, 7), dtype=np.uint8))
    flat = tmp_path / "flat.tif"
    tifffile.imwrite(flat, np.zeros((6, 7), dtype=np.uint8))

    def no_read(*args, **kwargs):
        raise AssertionError("the whole reference image was read")

    monkeypatch.setattr(dist3d.tifffile, "imread", no_read)

    assert dist3d._load_volume_shape_only(path, use_memmap=True) == (5, 6, 7)
    assert dist3d._load_volume_shape_only(flat, use_memmap=True) == (1, 6, 7)
