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
import pytest
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


# --- haemodynamics: FWHM diameters and the EDT cross-check -------------------

from haemolynx.haemodynamics import automated, edt_diameter  # noqa: E402

_FWHM_ARGS = dict(
    sample_spacing_along_edge_um=2.0,
    transverse_profile_step_um=0.25,
    transverse_half_extent_um=8.0,
    diameter_guess_um=4.0,
    store_profile_debug=True,
)


def _tube_case(voxel_zyx=(1.0, 1.0, 1.0), seed=0):
    """Three gently curving tubes of different radii in a noisy uint16 stack,
    plus the matching mask and a graph of their centrelines."""
    shape = (24, 64, 72)
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape, dtype=float)
    raw = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    G = nx.MultiGraph()
    spacing = np.asarray(voxel_zyx, dtype=float)
    for k, (y0, radius) in enumerate([(14.0, 2.0), (32.0, 3.0), (50.0, 4.5)]):
        xs = np.arange(4, 68, dtype=float)
        ys = y0 + 3.0 * np.sin(xs / 12.0 + k)
        zs = 12.0 + 2.0 * np.cos(xs / 15.0)
        centre_y = np.interp(x, xs, ys)
        centre_z = np.interp(x, xs, zs)
        inside = (x >= 4) & (x <= 67)
        r2 = (y - centre_y) ** 2 + (z - centre_z) ** 2
        raw = np.maximum(raw, np.where(inside, 1000.0 * np.exp(-r2 / (2 * (radius / 1.2) ** 2)), 0))
        mask |= inside & (r2 <= radius**2)
        pts = np.column_stack([zs, ys, xs]) * spacing
        G.add_node(2 * k, pos=pts[0])
        G.add_node(2 * k + 1, pos=pts[-1])
        G.add_edge(2 * k, 2 * k + 1, voxels=pts.tolist(), branch_order=f"B{k}")
    raw = (raw + rng.normal(50, 15, shape)).clip(0, 65535).astype(np.uint16)
    return raw, mask, G


def _same(a, b):
    """Deep equality for summaries holding arrays, dicts, lists and NaNs."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def _edge_attributes(G):
    return [dict(data) for _u, _v, _k, data in sorted(G.edges(keys=True, data=True), key=lambda e: e[:3])]


def test_fwhm_diameters_are_identical_in_low_ram_mode(tmp_path):
    raw, _mask, G = _tube_case()
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, raw)
    G_plain, G_low = G.copy(), G.copy()

    plain = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G_plain, raw_tiff_path=path, voxel_size_zyx=(1.0, 1.0, 1.0), **_FWHM_ARGS
    )
    low_ram = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G_low,
        raw_tiff_path=path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        use_memmap=True,
        memmap_directory=tmp_path / "mm",
        **_FWHM_ARGS,
    )

    assert plain["edges_measured"] == 3
    assert _same(low_ram, plain)
    assert _same(_edge_attributes(G_low), _edge_attributes(G_plain))
    assert not list((tmp_path / "mm").glob("haemolynx_*")), "a temporary volume was left behind"


def test_fwhm_raw_volume_on_disk_matches_the_in_ram_load_for_any_axis_order(tmp_path):
    raw, _mask, _G = _tube_case()
    path = tmp_path / "raw_xyz.tif"
    tifffile.imwrite(path, np.transpose(raw, (2, 1, 0)))

    plain = automated.load_single_channel_tiff_volume(path, axis_order="xyz")
    low_ram = automated.load_single_channel_tiff_volume(
        path, axis_order="xyz", use_memmap=True, memmap_directory=tmp_path / "mm"
    )

    assert isinstance(low_ram, np.memmap)
    assert low_ram.dtype == np.float32
    assert np.array_equal(low_ram, plain)
    # Only the returned file remains: the transposed intermediate was released.
    assert [p.name for p in (tmp_path / "mm").glob("haemolynx_*")] == [Path(low_ram.filename).name]


@pytest.mark.parametrize("voxel_zyx", [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5)])
def test_edt_crosscheck_is_identical_in_low_ram_mode(voxel_zyx, monkeypatch):
    _raw, mask, G = _tube_case(voxel_zyx)
    G_plain, G_low = G.copy(), G.copy()
    args = dict(voxel_size_zyx=voxel_zyx, sample_spacing_along_edge_um=1.0)

    plain = edt_diameter.measure_edge_diameters_from_binary_mask(G_plain, binary_mask=mask, **args)

    def no_whole_volume_transform(*a, **k):
        raise AssertionError("the whole-volume radius map was built")

    monkeypatch.setattr(edt_diameter, "inscribed_radius_map", no_whole_volume_transform)
    as_uint8 = mask.astype(np.uint8) * 255  # the loaded image, never cast to bool whole
    low_ram = edt_diameter.measure_edge_diameters_from_binary_mask(
        G_low, binary_mask=as_uint8, use_memmap=True, **args
    )

    assert plain["edges_measured"] == 3
    assert _same(low_ram, plain)
    assert _same(_edge_attributes(G_low), _edge_attributes(G_plain))


def test_edt_crosscheck_with_awkward_spacing_agrees_to_the_last_bit():
    """A spacing like 0.61 is not exactly representable, so where two background
    voxels are exactly equidistant scipy's pick can round one ulp differently."""
    voxel_zyx = (1.7, 0.61, 0.59)
    _raw, mask, G = _tube_case(voxel_zyx)
    G_plain, G_low = G.copy(), G.copy()
    args = dict(voxel_size_zyx=voxel_zyx, sample_spacing_along_edge_um=1.0)

    edt_diameter.measure_edge_diameters_from_binary_mask(G_plain, binary_mask=mask, **args)
    edt_diameter.measure_edge_diameters_from_binary_mask(G_low, binary_mask=mask, use_memmap=True, **args)

    for plain, low in zip(_edge_attributes(G_plain), _edge_attributes(G_low)):
        np.testing.assert_allclose(low["edt_diameter_samples_um"], plain["edt_diameter_samples_um"], rtol=1e-15, atol=0)


def test_edt_crosscheck_on_an_empty_mask_measures_nothing_either_way():
    _raw, mask, G = _tube_case()
    empty = np.zeros_like(mask)
    args = dict(voxel_size_zyx=(1.0, 1.0, 1.0), sample_spacing_along_edge_um=1.0)

    plain = edt_diameter.measure_edge_diameters_from_binary_mask(G.copy(), binary_mask=empty, **args)
    low_ram = edt_diameter.measure_edge_diameters_from_binary_mask(
        G.copy(), binary_mask=empty, use_memmap=True, **args
    )

    assert plain["edges_measured"] == 0
    assert _same(low_ram, plain)


def test_fwhm_in_low_ram_mode_keeps_its_volumes_on_disk(tmp_path, monkeypatch):
    raw, _mask, G = _tube_case()
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, raw)
    seen = {}
    real_labels = automated.build_graph_branch_label_volume
    real_sample = automated._sample_transverse_profile

    def spy_labels(*args, **kwargs):
        labels, keys = real_labels(*args, **kwargs)
        seen["labels"] = isinstance(labels, np.memmap)
        return labels, keys

    def spy_sample(raw_volume, *args, **kwargs):
        seen.setdefault("raw", isinstance(raw_volume, np.memmap))
        return real_sample(raw_volume, *args, **kwargs)

    monkeypatch.setattr(automated, "build_graph_branch_label_volume", spy_labels)
    monkeypatch.setattr(automated, "_sample_transverse_profile", spy_sample)

    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G, raw_tiff_path=path, voxel_size_zyx=(1.0, 1.0, 1.0), use_memmap=True, **_FWHM_ARGS
    )

    assert seen == {"labels": True, "raw": True}


# --- graph.edit: the graph editor's routing cost field ------------------------

from haemolynx.graph import edit as graph_edit  # noqa: E402
from haemolynx.gui.graph_editor import GraphEditorState  # noqa: E402


def _cost_mask(kind: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    mask = gaussian_filter(rng.random((40, 90, 100)), 2) > 0.54
    if kind == "single voxel":
        mask[:] = False
        mask[5, 5, 5] = True
    elif kind == "empty half":
        # Most voxels' nearest vessel lies outside any padded window.
        mask[:, :, :80] = False
    elif kind == "all vessel":
        mask[:] = True
    return mask


@pytest.mark.parametrize("kind", ["vessels", "single voxel", "empty half", "all vessel"])
def test_windowed_cost_field_gives_the_whole_field_for_any_window(kind, monkeypatch):
    mask = _cost_mask(kind)
    full = graph_edit.mask_cost_field(mask)
    windowed = graph_edit.mask_cost_field(mask, use_memmap=True)
    assert isinstance(windowed, graph_edit.WindowedMaskCostField)

    rng = np.random.default_rng(8)
    shape = np.array(mask.shape)
    for _ in range(8):
        a, b = rng.integers(0, shape), rng.integers(0, shape)
        window = tuple(slice(lo, hi + 1) for lo, hi in zip(np.minimum(a, b), np.maximum(a, b)))
        assert np.array_equal(windowed[window], full[window])
        start, end = rng.uniform(0, shape - 1), rng.uniform(0, shape - 1)
        assert np.array_equal(
            graph_edit.astar_path(windowed, start, end), graph_edit.astar_path(full, start, end)
        )


def test_windowed_cost_field_never_transforms_the_whole_volume(monkeypatch):
    mask = _cost_mask("empty half")
    windowed = graph_edit.mask_cost_field(mask, use_memmap=True)
    real = graph_edit.distance_transform_edt

    def windows_only(array, *args, **kwargs):
        assert array.size < mask.size, "the whole volume was transformed"
        return real(array, *args, **kwargs)

    monkeypatch.setattr(graph_edit, "distance_transform_edt", windows_only)

    windowed[0:10, 0:10, 0:10]
    graph_edit.astar_path(windowed, (3.0, 4.0, 5.0), (8.0, 10.0, 12.0))


def test_an_empty_mask_falls_back_to_the_whole_volume_field():
    """No mask voxel at all: scipy's transform is not a distance there, so
    only the whole-volume call reproduces it."""
    empty = np.zeros((6, 7, 8), dtype=bool)

    assert np.array_equal(
        graph_edit.mask_cost_field(empty, use_memmap=True), graph_edit.mask_cost_field(empty)
    )


def test_graph_editor_routes_the_same_branch_in_low_ram_mode():
    mask = _cost_mask("vessels")
    states = []
    for use_memmap in (False, True):
        state = GraphEditorState(graph=nx.MultiGraph(), voxel_size_zyx=(2.0, 0.5, 0.5))
        state.cost_field_from_mask(mask.astype(np.uint8) * 255, use_memmap=use_memmap)
        states.append(state)

    assert isinstance(states[1].cost_field, graph_edit.WindowedMaskCostField)
    paths = [
        graph_edit.astar_path(s.cost_field, (4.0, 10.0, 12.0), (30.0, 70.0, 88.0)) for s in states
    ]
    assert np.array_equal(paths[0], paths[1])


# --- gui: what the viewer is handed -------------------------------------------

from types import SimpleNamespace  # noqa: E402

from haemolynx.gui import results as gui_results  # noqa: E402
from haemolynx.gui import _widget as gui_widget  # noqa: E402
from haemolynx.preprocessing.memmap_support import new_memmap_array  # noqa: E402


def _on_disk(array: np.ndarray, tmp_path: Path) -> np.memmap:
    mapped = np.lib.format.open_memmap(
        tmp_path / f"v{len(list(tmp_path.iterdir()))}.npy",
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
    )
    mapped[...] = array
    return mapped


def _display_volumes():
    rng = np.random.default_rng(9)
    mask = rng.random((9, 10, 11)) < 0.1
    return {
        "0/255": mask.astype(np.uint8) * 255,
        "1/2": np.where(mask, 1, 2).astype(np.uint8),
        "ilastik with 0-border": np.where(mask, 1, np.where(rng.random(mask.shape) < 0.02, 0, 2)).astype(np.uint8),
        "four labels": np.where(mask, 3, np.where(rng.random(mask.shape) < 0.05, 7, 0)).astype(np.uint16),
        "grayscale": rng.integers(0, 4000, mask.shape).astype(np.uint16),
        "float": rng.random(mask.shape).astype(np.float32),
        "blank": np.zeros(mask.shape, dtype=np.uint8),
    }


def test_display_value_checks_read_a_disk_backed_volume_by_slab(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "haemolynx.preprocessing.memmap_support.LOW_MEMORY_BLOCK_VOXELS", 150
    )
    for name, volume in _display_volumes().items():
        mapped = _on_disk(volume, tmp_path)
        assert gui_results.binary_value_range(mapped) == gui_results.binary_value_range(volume), name
        plain = gui_results.binarize_for_display(volume)
        low_ram = gui_results.binarize_for_display(mapped)
        assert (low_ram is None) == (plain is None), name
        if plain is not None:
            assert isinstance(low_ram, np.memmap), name
            assert np.array_equal(low_ram, plain), name


def test_the_skeletonise_image_layer_is_the_same_and_stays_on_disk(tmp_path):
    for name, volume in _display_volumes().items():
        specs = []
        for image in (volume, _on_disk(volume, tmp_path)):
            layers = gui_results.ResultLayers().stage_finished(
                "skeletonise",
                SimpleNamespace(
                    image=image,
                    skeleton=np.zeros(volume.shape, dtype=bool),
                    voxel_size_xyz=(1.0, 1.0, 1.0),
                    voxel_size_zyx=(1.0, 1.0, 1.0),
                ),
            )
            specs.append(next(s for s in layers.layers if s.name == gui_results.IMAGE))
        plain, low_ram = specs
        assert np.array_equal(low_ram.data, plain.data), name
        assert low_ram.contrast_limits == plain.contrast_limits, name
        assert low_ram.options == plain.options, name
        if plain.data is not volume:  # it was binarised for display
            assert isinstance(low_ram.data, np.memmap), name


@pytest.mark.parametrize("window", [(0.0, 100.0), (2.0, 5.0), (5.0, 2.0), (7.5, 8.5)])
def test_clip_volume_to_z_on_disk_matches_in_ram(window, tmp_path):
    volume = np.random.default_rng(10).integers(0, 3, (9, 5, 6)).astype(np.uint8)
    plain = gui_results.clip_volume_to_z(volume, 1.0, *window)
    low_ram = gui_results.clip_volume_to_z(_on_disk(volume, tmp_path), 1.0, *window)

    assert np.array_equal(low_ram, plain)
    assert low_ram.dtype == plain.dtype


def test_debug_label_volumes_are_the_same_and_stay_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "haemolynx.preprocessing.memmap_support.LOW_MEMORY_BLOCK_VOXELS", 150
    )
    rng = np.random.default_rng(11)
    a = rng.random((9, 10, 11)) < 0.3
    b = rng.random((9, 10, 11)) < 0.3
    a_disk, b_disk = _on_disk(a, tmp_path), _on_disk(b, tmp_path)

    for build in (gui_widget._thick_thin_skeleton_labels, gui_widget._segmentation_cleanup_diff_labels):
        low_ram = build(a_disk, b_disk)
        assert isinstance(low_ram, np.memmap)
        assert np.array_equal(low_ram, build(a, b))

    as_uint8 = gui_widget._as_uint8_layer_data(a_disk)
    assert isinstance(as_uint8, np.memmap)
    assert np.array_equal(as_uint8, a.astype(np.uint8))
    as_uint8[...] = 9  # a Labels layer can be painted on without touching the source
    assert np.array_equal(a_disk, a)


def test_a_display_volume_on_disk_is_deleted_once_nothing_maps_it(tmp_path):
    import gc

    a = _on_disk(np.random.default_rng(12).random((4, 5, 6)) < 0.3, tmp_path)
    labels = gui_widget._as_uint8_layer_data(a)
    path = Path(labels.filename)
    view = labels[1:]
    del labels
    gc.collect()
    assert path.exists(), "deleted while a view still maps it"
    del view
    gc.collect()
    assert not path.exists()


# --- preprocessing: segmentation cleanup ---------------------------------------

import functools  # noqa: E402

from haemolynx.preprocessing import memmap_support, pointwise_distance  # noqa: E402
from haemolynx.preprocessing import segmentation_cleanup as cleanup  # noqa: E402


def _small_blocks(monkeypatch, block_voxels=6_000):
    """Force the block-wise paths to split a test-sized volume."""
    monkeypatch.setattr(
        cleanup, "map_blockwise", functools.partial(memmap_support.map_blockwise, block_voxels=block_voxels)
    )
    monkeypatch.setattr(
        pointwise_distance,
        "distance_transform_edt_blockwise",
        functools.partial(
            pointwise_distance.distance_transform_edt_blockwise, block_voxels=block_voxels, halo=3
        ),
    )


def _fragmented_vessels(seed=13, debris=True):
    """Tubes broken into aligned fragments (for reconnect), with surface
    whiskers, pinholes and small debris for the other steps to act on."""
    rng = np.random.default_rng(seed)
    shape = (20, 44, 60)
    z, y, x = np.indices(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    for yc, radius in [(10.0, 2.2), (24.0, 3.0), (36.0, 1.6)]:
        tube = (z - 10) ** 2 + (y - yc) ** 2 <= radius**2
        gaps = (x % 17 >= 13) & (x % 17 <= 14)
        mask |= tube & ~gaps
    if not debris:
        return mask
    mask |= rng.random(shape) < 0.004
    mask &= ~(rng.random(shape) < 0.01)
    return mask


def test_blockwise_edt_splits_and_matches_scipy(monkeypatch):
    from scipy.ndimage import distance_transform_edt

    mask = _fragmented_vessels()
    for sampling in [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5)]:
        out = np.empty(mask.shape)
        pointwise_distance.distance_transform_edt_blockwise(
            mask, out, sampling=sampling, halo=2, block_voxels=3_000
        )
        assert np.array_equal(out, distance_transform_edt(mask, sampling=sampling))


def test_blockwise_edt_refuses_a_volume_with_no_background():
    with pytest.raises(ValueError):
        pointwise_distance.distance_transform_edt_blockwise(
            np.ones((4, 5, 6), dtype=bool), np.empty((4, 5, 6)), halo=1, block_voxels=30
        )


@pytest.mark.parametrize(
    "step, kwargs",
    [
        (cleanup.remove_surface_whiskers, dict(whisker_radius_um=1.0)),
        (cleanup.close_small_gaps, dict(closing_radius_um=1.0)),
        (cleanup.smooth_vessel_surfaces, dict(method="morphological", morphological_radius_um=1.0)),
        (cleanup.smooth_vessel_surfaces, dict(method="gaussian", sigma_um=1.0)),
        (cleanup.fill_enclosed_cavities, dict()),
    ],
)
def test_local_cleanup_steps_are_identical_in_low_ram_mode(step, kwargs, monkeypatch, tmp_path):
    _small_blocks(monkeypatch)
    mask = _fragmented_vessels()
    if step is not cleanup.fill_enclosed_cavities:
        kwargs = dict(kwargs, voxel_size_zyx=(2.0, 0.5, 0.5))

    plain = step(mask, **kwargs)
    low_ram = step(mask, use_memmap=True, memmap_directory=tmp_path, **kwargs)

    assert not np.array_equal(plain, mask), "the step must change something"
    assert isinstance(low_ram, np.memmap)
    assert np.array_equal(low_ram, plain)


@pytest.mark.parametrize("voxel_zyx", [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5)])
def test_reconnect_is_identical_in_low_ram_mode(voxel_zyx, monkeypatch, tmp_path):
    _small_blocks(monkeypatch)
    mask = _fragmented_vessels(debris=False)
    # Long enough that some candidates are rejected as well as accepted.
    args = dict(voxel_size_zyx=voxel_zyx, max_bridge_distance_um=15.0, min_cylindricality=0.3)

    plain, plain_stats = cleanup.reconnect_vessel_like_components(mask, **args)

    def no_whole_volume_edt(array, *a, **k):
        raise AssertionError("the whole volume was distance-transformed")

    monkeypatch.setattr(cleanup, "distance_transform_edt", no_whole_volume_edt)
    low_ram, low_stats = cleanup.reconnect_vessel_like_components(
        mask, use_memmap=True, memmap_directory=tmp_path, **args
    )

    assert plain_stats["accepted_bridges"] > 0
    assert plain_stats["rejected_reasons"]
    assert low_stats == plain_stats
    assert isinstance(low_ram, np.memmap)
    assert np.array_equal(low_ram, plain)
    assert [p.name for p in tmp_path.iterdir()] == [Path(low_ram.filename).name]


def test_whole_cleanup_is_identical_and_tidy_in_low_ram_mode(monkeypatch, tmp_path):
    _small_blocks(monkeypatch)
    image = _fragmented_vessels().astype(np.uint8) * 255
    settings = dict(
        voxel_size_zyx=(2.0, 0.5, 0.5),
        fill_cavities=True,
        remove_whiskers=True,
        whisker_radius_um=0.5,
        close_gaps=True,
        close_gaps_radius_um=0.5,
        reconnect_gaps=True,
        reconnect_max_bridge_distance_um=8.0,
        reconnect_min_cylindricality=0.3,
        smooth_surfaces=True,
        smooth_sigma_um=0.5,
        remove_small_volumes=True,
        remove_small_min_volume_um3=4.0,
    )

    plain, plain_raw = cleanup.clean_segmented_mask_for_skeletonisation(image, **settings)
    low_ram, low_raw = cleanup.clean_segmented_mask_for_skeletonisation(
        image, use_memmap=True, memmap_directory=tmp_path, **settings
    )

    assert np.array_equal(low_ram, plain)
    assert np.array_equal(low_raw, plain_raw)
    kept = {Path(low_ram.filename).name, Path(low_raw.filename).name}
    assert {p.name for p in tmp_path.iterdir()} == kept, "an intermediate volume was left behind"
