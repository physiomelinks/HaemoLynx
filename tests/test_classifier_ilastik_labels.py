"""Labels painted in an ilastik project read as training strokes
(``classifier.ilastik_labels``), on projects laid out as ilastik 1.4 writes
them: label blocks with their place and axes, each image's file, axes and
shape."""
from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from haemolynx.classifier.ilastik_labels import (  # noqa: E402
    default_label_mapping,
    ilastik_images,
    ilastik_label_names,
    read_ilastik_labels,
    strokes_from_ilastik,
)


def _tags(keys: str) -> str:
    return json.dumps({"axes": [{"key": k, "typeFlags": 1 if k == "c" else 2, "resolution": 0, "description": ""}
                                for k in keys]})


def write_project(path, *, names=("Label 1", "Label 2"), images=(("raw.tif", "zcyx", (6, 2, 8, 10)),),
                  blocks=None):
    """An ilastik project: *images* ``(file, axes, shape)``, and *blocks*
    per image, ``(start (z, y, x), labels (z, y, x))``, stored as ilastik
    stores them (axes zyxc, one channel)."""
    with h5py.File(path, "w") as f:
        f["PixelClassification/LabelNames"] = np.array([n.encode() for n in names], dtype=object)
        for lane, (file_name, axes, shape) in enumerate(images):
            raw = f.create_group(f"Input Data/infos/lane{lane:04d}/Raw Data")
            raw["filePath"] = f"images\\{file_name}".encode()  # a relative path, as ilastik keeps one
            raw["axistags"] = _tags(axes).encode()
            raw["shape"] = np.array(shape)
            sets = f.require_group(f"PixelClassification/LabelSets/labels{lane:03d}")
            for number, (start, labels) in enumerate((blocks or {}).get(lane, [])):
                labels = np.asarray(labels, np.uint8)
                spatial = labels.shape
                keys = "zyxc" if len(spatial) == 3 else "yxc"
                data = labels[..., np.newaxis]
                block = sets.create_dataset(f"block{number:04d}", data=data)
                block.attrs["axistags"] = _tags(keys)
                stops = [s + n for s, n in zip(start, spatial)]
                block.attrs["blockSlice"] = (
                    "[" + ",".join(f"{a}:{b}" for a, b in zip(start, stops)) + ",0:1]"
                )
    return path


def _block():
    labels = np.zeros((2, 3, 4), np.uint8)
    labels[0, 1, 2] = 1
    labels[1, 2, 3] = 2
    labels[1, 0, 0] = 2
    return labels


def test_the_painted_voxels_come_back_where_they_were_painted(tmp_path):
    project = write_project(tmp_path / "p.ilp", blocks={0: [((2, 3, 4), _block())]})
    assert ilastik_label_names(project) == ("Label 1", "Label 2")
    (image,) = ilastik_images(project)
    assert image.file_name == "raw.tif" and image.axes == "zcyx" and image.spatial_shape == (6, 8, 10)
    points, labels = read_ilastik_labels(project, tmp_path / "raw.tif", shape_zyx=(6, 8, 10))
    found = {tuple(p): int(v) for p, v in zip(points, labels)}
    assert found == {(2, 4, 6): 1, (3, 5, 7): 2, (3, 3, 4): 2}


def test_an_image_read_in_another_axis_order_gets_its_labels_in_that_order(tmp_path):
    # Stored with x as the page axis: ilastik calls the pages z, HaemoLynx
    # (image_axis_order "xyz") reads them as x.
    project = write_project(tmp_path / "p.ilp", images=(("raw.tif", "zyx", (6, 8, 10)),),
                            blocks={0: [((1, 2, 3), np.ones((1, 1, 1), np.uint8))]})
    points, labels = read_ilastik_labels(project, tmp_path / "raw.tif", shape_zyx=(10, 8, 6), axis_order="xyz")
    assert points.tolist() == [[3, 2, 1]] and labels.tolist() == [1]


def test_a_2d_image_has_its_labels_on_its_one_plane(tmp_path):
    project = write_project(tmp_path / "p.ilp", images=(("raw.tif", "yxc", (8, 10, 1)),),
                            blocks={0: [((4, 5), np.full((1, 2), 2, np.uint8))]})
    points, labels = read_ilastik_labels(project, tmp_path / "raw.tif", shape_zyx=(1, 8, 10))
    assert points.tolist() == [[0, 4, 5], [0, 4, 6]] and labels.tolist() == [2, 2]


def test_another_size_or_no_such_image_is_refused(tmp_path):
    project = write_project(tmp_path / "p.ilp", blocks={0: [((0, 0, 0), _block())]})
    with pytest.raises(ValueError, match="6 x 8 x 10"):
        read_ilastik_labels(project, tmp_path / "raw.tif", shape_zyx=(6, 8, 12))
    two = write_project(tmp_path / "two.ilp", images=(("a.tif", "zyx", (6, 8, 10)), ("b.tif", "zyx", (6, 8, 10))))
    with pytest.raises(ValueError, match="a.tif, b.tif"):
        read_ilastik_labels(two, tmp_path / "c.tif", shape_zyx=(6, 8, 10))
    # The project's one image, under another name but its size, is used.
    points, _ = read_ilastik_labels(project, tmp_path / "renamed.tif", shape_zyx=(6, 8, 10))
    assert len(points) == 3


def test_the_image_is_chosen_by_name_among_several(tmp_path):
    project = write_project(
        tmp_path / "p.ilp",
        images=(("a.tif", "zyx", (6, 8, 10)), ("b.tif", "zyx", (6, 8, 10))),
        blocks={0: [((0, 0, 0), np.ones((1, 1, 1), np.uint8))], 1: [((5, 7, 9), np.full((1, 1, 1), 2, np.uint8))]},
    )
    points, labels = read_ilastik_labels(project, tmp_path / "B.TIF", shape_zyx=(6, 8, 10))
    assert points.tolist() == [[5, 7, 9]] and labels.tolist() == [2]


def test_labels_named_alike_map_by_name():
    labels = ("Background", "Capillary", "Arteriole", "Venule")
    assert default_label_mapping(["background", "Capillaries", "Arteries", "Veins"], labels) == {
        1: "Background", 2: "Capillary", 3: "Arteriole", 4: "Venule",
    }
    assert default_label_mapping(["BG", "Vessels"], ("Background", "Vessel")) == {1: "Background", 2: "Vessel"}
    assert default_label_mapping(["Label 1", "Label 2"], labels) == {1: None, 2: None}


@pytest.mark.parametrize("vessel_label", [2, "2", "Label 2"])
def test_unnamed_labels_follow_the_runs_ilastik_vessel_label(vessel_label):
    assert default_label_mapping(["Label 1", "Label 2"], ("Background", "Vessel"), vessel_label=vessel_label) == {
        1: "Background", 2: "Vessel",
    }
    # Not for a job of more than one vessel label.
    assert default_label_mapping(
        ["Label 1", "Label 2"], ("Background", "Capillary", "Arteriole", "Venule"), vessel_label=vessel_label
    ) == {1: None, 2: None}


def test_strokes_are_the_chosen_labels_and_left_out_ones_go():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    ilastik = np.array([1, 2, 3], np.uint8)
    kept, values = strokes_from_ilastik(points, ilastik, {1: "Vessel", 2: None, 3: "Background"},
                                        ("Background", "Vessel"))
    assert kept.tolist() == [[0, 0, 0], [2, 2, 2]] and values.tolist() == [2, 1]
