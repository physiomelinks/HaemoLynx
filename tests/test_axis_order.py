"""Tests for input axis-order handling and canonical (z, y, x) conversion."""
import numpy as np
import pytest
import tifffile

from haemolynx.io import (
    CANONICAL_AXIS_ORDER,
    VALID_AXIS_ORDERS,
    apply_axis_order,
    axis_order_transpose,
    load_3d_tif_with_voxel_size,
    normalize_axis_order,
    voxel_size_xyz_from_zyx,
    voxel_size_zyx_from_xyz,
)


def _distinct_shape_volume() -> np.ndarray:
    """Volume whose axes have distinct lengths, so transposes are detectable."""
    return np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)


def test_normalize_axis_order_accepts_permutations_and_normalizes_case():
    assert normalize_axis_order("ZYX") == "zyx"
    assert normalize_axis_order(" xyz ") == "xyz"
    for order in VALID_AXIS_ORDERS:
        assert normalize_axis_order(order) == order


@pytest.mark.parametrize("bad", ["zy", "zyxx", "zzz", "abc", "", "z y x", None, 3])
def test_normalize_axis_order_rejects_non_permutations(bad):
    with pytest.raises(ValueError, match="permutation of 'xyz'|must be a string"):
        normalize_axis_order(bad)


def test_axis_order_transpose_maps_named_axis_to_canonical_position():
    # Volume stored (x, y, z): z is axis 2, y is axis 1, x is axis 0.
    assert axis_order_transpose("xyz") == (2, 1, 0)
    assert axis_order_transpose("zyx") == (0, 1, 2)
    # Volume stored (y, z, x): z is axis 1, y is axis 0, x is axis 2.
    assert axis_order_transpose("yzx") == (1, 0, 2)


def test_apply_axis_order_is_identity_for_canonical_order():
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, CANONICAL_AXIS_ORDER)
    assert result.shape == volume.shape
    assert np.array_equal(result, volume)


def test_apply_axis_order_preserves_a_memmap_for_canonical_order(tmp_path):
    """Regression: the canonical no-op used to demote a ``numpy.memmap`` to a
    plain ``ndarray`` (``np.asarray`` strips ndarray subclasses even when it
    returns the same underlying buffer, unlike ``np.asanyarray``) -- harmless
    for the bytes, but it broke ``isinstance(x, np.memmap)`` checks
    downstream that decide whether an array owns a backing file to release
    (see ``use_memmap_loading``)."""
    path = tmp_path / "volume.dat"
    volume = np.memmap(path, dtype=np.uint8, mode="w+", shape=(2, 3, 4))
    volume[:] = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)

    result = apply_axis_order(volume, CANONICAL_AXIS_ORDER)

    assert isinstance(result, np.memmap)
    assert np.array_equal(result, volume)


def test_apply_axis_order_moves_the_named_z_axis_to_axis_zero():
    # Shape (2, 3, 4) stored as (x, y, z) means x=2, y=3, z=4.
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, "xyz")
    assert result.shape == (4, 3, 2)
    # A voxel at file index (x=1, y=2, z=3) must land at canonical (z=3, y=2, x=1).
    assert result[3, 2, 1] == volume[1, 2, 3]


@pytest.mark.parametrize("axis_order", VALID_AXIS_ORDERS)
def test_apply_axis_order_preserves_voxel_values_for_every_permutation(axis_order):
    volume = _distinct_shape_volume()
    result = apply_axis_order(volume, axis_order)
    assert sorted(result.shape) == sorted(volume.shape)
    assert np.array_equal(np.sort(result.ravel()), np.sort(volume.ravel()))
    # The axis named "z" must become axis 0 with its original length.
    assert result.shape[0] == volume.shape[axis_order.index("z")]
    assert result.shape[1] == volume.shape[axis_order.index("y")]
    assert result.shape[2] == volume.shape[axis_order.index("x")]


def test_apply_axis_order_rejects_non_3d_volume_for_non_canonical_order():
    with pytest.raises(ValueError, match="requires a 3D volume"):
        apply_axis_order(np.zeros((4, 4)), "xyz")


def test_voxel_size_conversion_reverses_axis_order():
    assert voxel_size_zyx_from_xyz((0.5, 0.6, 2.0)) == (2.0, 0.6, 0.5)
    assert voxel_size_xyz_from_zyx((2.0, 0.6, 0.5)) == (0.5, 0.6, 2.0)


def test_voxel_size_conversion_round_trips():
    voxel_size_xyz = (0.325, 0.4, 3.0)
    assert voxel_size_xyz_from_zyx(voxel_size_zyx_from_xyz(voxel_size_xyz)) == voxel_size_xyz


@pytest.mark.parametrize("bad", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
def test_voxel_size_conversion_rejects_wrong_length(bad):
    with pytest.raises(ValueError, match="must have length 3"):
        voxel_size_zyx_from_xyz(bad)


def test_tif_loader_applies_axis_order(tmp_path):
    """A file written as (x, y, z) loads transposed to canonical (z, y, x)."""
    volume = _distinct_shape_volume()  # (x=2, y=3, z=4) under axis_order="xyz"
    path = tmp_path / "xyz_volume.tif"
    tifffile.imwrite(str(path), volume)

    canonical, _vx, _vy, _vz, _status = load_3d_tif_with_voxel_size(
        str(path), axis_order="xyz"
    )
    assert canonical.shape == (4, 3, 2)
    assert canonical[3, 2, 1] == volume[1, 2, 3]

    as_stored, _vx, _vy, _vz, _status = load_3d_tif_with_voxel_size(str(path))
    assert as_stored.shape == volume.shape


def test_file_axis_spacing_maps_to_the_physical_axis_each_file_axis_is():
    from haemolynx.io import file_axis_spacing_from_xyz, voxel_size_xyz_from_file_axes

    # (pages, height, width) spacings of a file whose pages step along x.
    assert voxel_size_xyz_from_file_axes((2.0, 0.25, 0.5), "xyz") == (2.0, 0.25, 0.5)
    assert voxel_size_xyz_from_file_axes((2.0, 0.25, 0.5), "zyx") == (0.5, 0.25, 2.0)
    assert voxel_size_xyz_from_file_axes((2.0, 0.25, 0.5), "yzx") == (0.5, 2.0, 0.25)
    for order in VALID_AXIS_ORDERS:
        spacing = file_axis_spacing_from_xyz((0.3, 0.7, 1.9), order)
        assert voxel_size_xyz_from_file_axes(spacing, order) == pytest.approx((0.3, 0.7, 1.9))


def _imagej_stack(path, volume=None, *, page_spacing, height_spacing, width_spacing):
    tifffile.imwrite(
        str(path),
        np.zeros((4, 6, 8), dtype=np.uint8) if volume is None else volume,
        imagej=True,
        resolution=(1.0 / width_spacing, 1.0 / height_spacing),
        metadata={"spacing": page_spacing, "unit": "um"},
    )


def test_a_tiffs_spacings_follow_the_axis_order_its_pages_are_read_with(tmp_path):
    """Regression: the page spacing was always labelled z and the width
    spacing x, so a stack whose pages step along x (image_axis_order="xyz")
    had its z and x spacings swapped for the whole run."""
    from haemolynx.io import read_voxel_size_xyz

    path = tmp_path / "pages_along_x.tif"
    _imagej_stack(path, page_spacing=2.0, height_spacing=0.25, width_spacing=0.5)

    _image, vx, vy, vz, _status = load_3d_tif_with_voxel_size(str(path), axis_order="xyz")
    assert (vx, vy, vz) == pytest.approx((2.0, 0.25, 0.5))
    assert read_voxel_size_xyz(path, "xyz")[0] == pytest.approx((2.0, 0.25, 0.5))

    _image, vx, vy, vz, _status = load_3d_tif_with_voxel_size(str(path))
    assert (vx, vy, vz) == pytest.approx((0.5, 0.25, 2.0))


def _branching_tube_zyx() -> np.ndarray:
    """A thick oblique vessel with one side branch, in a (z, y, x) volume whose
    axes all differ in length, so a wrong transpose cannot go unnoticed."""
    zz, yy, xx = np.indices((14, 22, 32), dtype=float)
    volume = np.zeros(zz.shape, dtype=bool)
    for start, end in (((3, 3, 3), (10, 18, 28)), ((6.5, 10.5, 15.5), (11, 3, 26))):
        for t in np.linspace(0.0, 1.0, 80):
            centre = np.add(start, t * np.subtract(end, start))
            volume |= (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= 2.0
    return volume.astype(np.uint8) * 255


@pytest.mark.parametrize(
    "axis_order", [order for order in VALID_AXIS_ORDERS if order != CANONICAL_AXIS_ORDER]
)
def test_a_stack_saved_in_any_axis_order_builds_the_canonical_graph(tmp_path, axis_order):
    """Regression: tests/integration/test_axis_order_pipeline.py wrote an
    (x, y, z) stack with the (z, y, x) file's tags, which since the loader
    reads tags per file axis declares z and x spacings swapped -- its graph
    came out 15% shorter (123.4 vs 144.7 um), which read as an axis-order
    dependence somewhere in skeletonisation, graph building or smoothing.

    A stack stored in any order, carrying the spacing of each of its own axes,
    must load to the canonical volume and voxel and so give the canonical
    graph and lengths, smoothed on anisotropic voxels.
    """
    from haemolynx.graph import build_graph_from_skeleton, smooth_graph_centrelines
    from haemolynx.io import file_axis_spacing_from_xyz, load_and_skeletonize_3d_tif

    voxel_size_xyz = (0.4, 0.5, 2.0)
    canonical_volume = _branching_tube_zyx()

    def save_and_build(order):
        stored = np.transpose(canonical_volume, [CANONICAL_AXIS_ORDER.index(a) for a in order])
        page, height, width = file_axis_spacing_from_xyz(voxel_size_xyz, order)
        path = tmp_path / f"{order}.tif"
        _imagej_stack(path, stored, page_spacing=page, height_spacing=height,
                      width_spacing=width)
        image, skeleton, vx, vy, vz, _status = load_and_skeletonize_3d_tif(
            str(path), axis_order=order
        )
        voxel_size_zyx = voxel_size_zyx_from_xyz((vx, vy, vz))
        graph = build_graph_from_skeleton(skeleton, voxel_size=voxel_size_zyx, min_stub_length=3.0)
        smooth_graph_centrelines(graph, skeleton, voxel_size_zyx=voxel_size_zyx)
        return image, (vx, vy, vz), graph

    image, voxel_size, graph = save_and_build(axis_order)
    _canonical_image, _canonical_voxel_size, canonical_graph = save_and_build(CANONICAL_AXIS_ORDER)

    assert np.array_equal(image, canonical_volume)
    assert voxel_size == pytest.approx(voxel_size_xyz)
    assert graph.number_of_edges() == canonical_graph.number_of_edges() >= 3

    def total_length(g):
        return sum(float(d["length"]) for _, _, d in g.edges(data=True))

    assert total_length(graph) == pytest.approx(total_length(canonical_graph), rel=1e-9)
    positions = sorted(tuple(d["pos"]) for _, d in graph.nodes(data=True))
    canonical_positions = sorted(tuple(d["pos"]) for _, d in canonical_graph.nodes(data=True))
    np.testing.assert_allclose(positions, canonical_positions, rtol=1e-9, atol=1e-9)


def test_a_missing_tag_is_reported_for_the_physical_axis_it_describes(tmp_path):
    path = tmp_path / "no_spacing.tif"
    tifffile.imwrite(str(path), np.zeros((4, 6, 8), dtype=np.uint8), imagej=True,
                     resolution=(2.0, 4.0))

    *_sizes, status = load_3d_tif_with_voxel_size(str(path), axis_order="xyz")

    assert status["missing_axes"] == ["x"]  # the page spacing, and the pages are x


def test_an_h5_element_size_follows_the_axis_order(tmp_path):
    h5py = pytest.importorskip("h5py")
    from haemolynx.io import load_3d_h5_with_voxel_size

    path = tmp_path / "pages_along_x.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=np.zeros((4, 6, 8), dtype=np.uint8))
        dataset.attrs["element_size_um"] = (2.0, 0.25, 0.5)  # one per dataset axis

    _image, vx, vy, vz, _status = load_3d_h5_with_voxel_size(str(path), axis_order="xyz")
    assert (vx, vy, vz) == pytest.approx((2.0, 0.25, 0.5))
    _image, vx, vy, vz, _status = load_3d_h5_with_voxel_size(str(path))
    assert (vx, vy, vz) == pytest.approx((0.5, 0.25, 2.0))
