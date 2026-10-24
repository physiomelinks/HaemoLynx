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


# --- image_axis_order="auto": each file's own axis labels ---------------------


def _labelled_volume() -> np.ndarray:
    """A canonical (z=5, y=6, x=7) volume of distinct values. Longer than four
    on every axis: tifffile reads a leading axis of 3 or 4 as colour samples."""
    return np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)


def _stored(volume_zyx: np.ndarray, order: str) -> np.ndarray:
    return np.transpose(volume_zyx, [CANONICAL_AXIS_ORDER.index(axis) for axis in order])


def _write_labelled_tiff(path, volume_zyx, order, **kwargs):
    """*volume_zyx* stored in *order*, its axes named so in the file."""
    tifffile.imwrite(str(path), _stored(volume_zyx, order), metadata={"axes": order.upper()}, **kwargs)
    return path


def _write_h5(path, data, *, axistags=None, dimension_labels=None, element_size_um=None):
    h5py = pytest.importorskip("h5py")
    import json

    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=data)
        if axistags is not None:
            dataset.attrs["axistags"] = json.dumps(
                {"axes": [{"key": key, "typeFlags": 2, "resolution": 0, "description": ""}
                          for key in axistags]}
            )
        for axis, label in enumerate(dimension_labels or ()):
            dataset.dims[axis].label = label
        if element_size_um is not None:
            dataset.attrs["element_size_um"] = element_size_um
    return path


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ("ZYX", "zyx"),
        ("ZCYX", "zyx"),
        ("TZCYXS", "zyx"),
        ("XYZ", "xyz"),
        (["y", "c", "z", "x"], "yzx"),
        ("QYX", None),  # tifffile's name for an axis it knows nothing about
        ("IYX", None),
        ("YX", None),
        ("ZZYX", None),
    ],
)
def test_the_labelled_order_is_z_y_and_x_among_the_files_axes(labels, expected):
    from haemolynx.io.axis_order import axis_order_from_labels

    assert axis_order_from_labels(labels) == expected


def test_auto_is_refused_where_an_array_is_transposed_without_a_file():
    with pytest.raises(ValueError, match="resolved against a file"):
        normalize_axis_order("auto")
    with pytest.raises(ValueError, match="resolved against a file"):
        apply_axis_order(_distinct_shape_volume(), "auto")
    with pytest.raises(ValueError, match="resolved against a file"):
        axis_order_transpose("auto")


def test_an_auto_order_carries_the_order_a_file_without_labels_is_read_in():
    from haemolynx.io.axis_order import auto_axis_order

    assert normalize_axis_order(" AUTO ", allow_auto=True) == "auto"
    assert normalize_axis_order("auto:XYZ", allow_auto=True) == "auto:xyz"
    assert normalize_axis_order("auto:zyx", allow_auto=True) == "auto"
    assert auto_axis_order() == "auto" and auto_axis_order("yzx") == "auto:yzx"
    with pytest.raises(ValueError, match="permutation of 'xyz'"):
        normalize_axis_order("auto:abc", allow_auto=True)
    with pytest.raises(ValueError, match="or 'auto'"):
        normalize_axis_order("abc", allow_auto=True)


def test_a_file_is_read_in_its_labels_order_only_under_auto():
    from haemolynx.io.axis_order import resolve_axis_order

    assert resolve_axis_order("yzx", "xyz") == "yzx"  # a chosen order wins
    assert resolve_axis_order("auto", "xyz") == "xyz"
    assert resolve_axis_order("auto", None) == "zyx"
    assert resolve_axis_order("auto:xzy", "yxz") == "yxz"
    assert resolve_axis_order("auto:xzy", None) == "xzy"


def test_each_formats_axis_labels_are_read_from_its_header(tmp_path):
    from haemolynx.io import stated_axis_order

    volume = _labelled_volume()
    assert stated_axis_order(_write_labelled_tiff(tmp_path / "xyz.tif", volume, "xyz")) == "xyz"
    imagej = tmp_path / "imagej.tif"
    tifffile.imwrite(str(imagej), volume, imagej=True, metadata={"axes": "ZYX"})
    assert stated_axis_order(imagej) == "zyx"
    composite = tmp_path / "composite.tif"
    tifffile.imwrite(
        str(composite), np.stack([volume, volume], axis=1), imagej=True, metadata={"axes": "ZCYX"}
    )
    assert stated_axis_order(composite) == "zyx"
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(str(plain), volume)  # tifffile names its pages Q: no z
    assert stated_axis_order(plain) is None
    unnamed = tmp_path / "imagej_unnamed.tif"
    tifffile.imwrite(str(unnamed), volume, imagej=True)  # tifffile's ImageJ guess: channels
    assert stated_axis_order(unnamed) is None

    stored = _stored(volume, "xzy")[..., np.newaxis]
    assert stated_axis_order(_write_h5(tmp_path / "tags.h5", stored, axistags="xzyc")) == "xzy"
    labelled = _write_h5(tmp_path / "dims.h5", _stored(volume, "yxz"), dimension_labels="yxz")
    assert stated_axis_order(labelled) == "yxz"
    assert stated_axis_order(_write_h5(tmp_path / "bare.h5", volume)) is None

    text = tmp_path / "notes.txt"
    text.write_text("zyx")
    for nothing in (text, tmp_path / "gone.tif", None, ""):
        assert stated_axis_order(nothing) is None


def test_a_files_labels_are_read_again_once_it_changes(tmp_path):
    from haemolynx.io import stated_axis_order

    path = _write_labelled_tiff(tmp_path / "stack.tif", _labelled_volume(), "xyz")
    assert stated_axis_order(path) == "xyz"
    _write_labelled_tiff(path, _labelled_volume()[:, :, :6], "zyx")
    assert stated_axis_order(path) == "zyx"


def test_an_imagej_stack_states_z_first_whatever_its_pages_step_along(tmp_path):
    """What auto cannot see: a TIFF page is a Y by X plane, and ImageJ labels a
    stack ZYX whatever its pages are, so a resliced stack needs the order
    chosen by hand."""
    from haemolynx.io import axis_order_for_file

    path = tmp_path / "pages_along_x.tif"
    _imagej_stack(path, _stored(_labelled_volume(), "xyz").astype(np.uint8),
                  page_spacing=2.0, height_spacing=0.25, width_spacing=0.5)

    assert axis_order_for_file("auto", path) == "zyx"
    assert axis_order_for_file("xyz", path) == "xyz"


def test_the_loaders_read_an_auto_file_in_its_labels_order(tmp_path):
    from haemolynx.haemodynamics.fwhm_diameter import load_single_channel_tiff_volume
    from haemolynx.io import h5_volume_shape, load_3d_h5_with_voxel_size, read_voxel_size_xyz

    volume = _labelled_volume()
    # Width spacing 2.0 and height 0.25: the width is the file's Z, the height its Y.
    tiff = _write_labelled_tiff(tmp_path / "xyz.tif", volume, "xyz", resolution=(0.5, 4.0))
    image, vx, vy, vz, _status = load_3d_tif_with_voxel_size(str(tiff), axis_order="auto")
    assert np.array_equal(image, volume)
    assert (vx, vy, vz) == pytest.approx((1.0, 0.25, 2.0))  # nothing gives x
    assert read_voxel_size_xyz(tiff, "auto")[0] == pytest.approx((1.0, 0.25, 2.0))
    assert np.array_equal(load_single_channel_tiff_volume(tiff, axis_order="auto"), volume)
    for use_memmap in (False, True):
        loaded = load_single_channel_tiff_volume(
            tiff, axis_order="auto", use_memmap=use_memmap, memmap_directory=tmp_path
        )
        assert np.array_equal(loaded, volume)

    h5 = _write_h5(
        tmp_path / "xyzc.h5", _stored(volume, "xyz")[..., np.newaxis], axistags="xyzc",
        element_size_um=(0.4, 0.5, 2.0, 1.0),  # one per dataset axis, the channel's too
    )
    assert h5_volume_shape(h5, axis_order="auto") == volume.shape
    image, vx, vy, vz, _status = load_3d_h5_with_voxel_size(str(h5), axis_order="auto")
    assert np.array_equal(image, volume)
    assert (vx, vy, vz) == pytest.approx((0.4, 0.5, 2.0))


def test_a_channel_of_an_auto_composite_is_read_in_its_labels_order(tmp_path):
    from haemolynx.haemodynamics.fwhm_diameter import load_single_channel_tiff_volume

    volume = _labelled_volume()
    composite = tmp_path / "composite.tif"
    tifffile.imwrite(
        str(composite), np.stack([volume, volume + 1], axis=1), imagej=True,
        metadata={"axes": "ZCYX"},
    )

    loaded = load_single_channel_tiff_volume(composite, axis_order="auto", channel=1)
    assert np.array_equal(loaded, volume + 1)


def test_a_run_reads_a_file_without_labels_in_the_segmented_images_order(tmp_path):
    """The segmented image names its axes (an ilastik HDF5 stored x, y, z); a
    mask written beside it without labels is in the same order, and one that
    names its own axes is read in those."""
    from haemolynx.io import axis_order_for_run

    volume = _labelled_volume()
    image = _write_h5(tmp_path / "image.h5", _stored(volume, "xyz"), axistags="xyz")
    run_order = axis_order_for_run({"image_axis_order": "auto", "input_path": image})
    assert run_order == "auto:xyz"

    unlabelled = tmp_path / "mask.tif"
    tifffile.imwrite(str(unlabelled), _stored(volume, "xyz"))
    labelled = _write_labelled_tiff(tmp_path / "labelled.tif", volume, "zyx")
    for mask in (unlabelled, labelled):
        loaded, *_ = load_3d_tif_with_voxel_size(str(mask), axis_order=run_order)
        assert np.array_equal(loaded, volume), mask.name


def test_a_runs_order_is_the_chosen_one_or_auto_with_zyx_for_files_without_labels(tmp_path):
    from haemolynx.io import axis_order_for_run

    plain = tmp_path / "plain.tif"
    tifffile.imwrite(str(plain), _labelled_volume())
    labelled = _write_labelled_tiff(tmp_path / "xyz.tif", _labelled_volume(), "xyz")

    assert axis_order_for_run({"image_axis_order": "xzy", "input_path": labelled}) == "xzy"
    assert axis_order_for_run({"image_axis_order": "auto", "input_path": plain}) == "auto"
    # ilastik has not written the segmented image yet.
    assert axis_order_for_run({"image_axis_order": "auto", "input_path": None}) == "auto"
    assert axis_order_for_run({}) == "zyx"


def test_auto_is_a_choice_of_image_axis_order_and_zyx_stays_the_default():
    from haemolynx.pipeline import default_schema

    setting = default_schema()["image_axis_order"]
    assert "auto" in setting.choices
    assert set(VALID_AXIS_ORDERS) <= set(setting.choices)
    assert setting.default == "zyx"
