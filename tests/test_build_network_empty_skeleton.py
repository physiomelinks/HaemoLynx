"""What ``build_network()`` does when the skeleton has no vessel in it.

A vessel is a path between touching skeleton voxels. On a skeleton with no two
touching -- empty, or thinned to isolated points -- skan raised scipy's
"index pointer size 0 should be 1" from deep inside graph building. The cause
is upstream, in the segmentation or skeletonising, so the stage stops before
graph building and says how many voxels the skeleton and the segmented image
have, and which settings to check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

import haemolynx.graph
from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.stages import SkeletonisedVolume, build_network

SCHEMA = default_schema()


def _settings(tmp_path: Path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "visualize_results": False,
            **overrides,
        }
    )
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _volume(tmp_path: Path, skeleton: np.ndarray, image: np.ndarray, **extra) -> SkeletonisedVolume:
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    return SkeletonisedVolume(
        image=image,
        skeleton=skeleton,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
        **extra,
    )


def _graph_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "input_graph.pkl"


def _isolated_voxels() -> np.ndarray:
    skeleton = np.zeros((12, 12, 12), dtype=bool)
    skeleton[3, 3, 3] = skeleton[8, 8, 8] = True
    return skeleton


def _specks_under(skeleton: np.ndarray) -> np.ndarray:
    """A 3x3x3 cube of segmentation round each skeleton voxel."""
    return ndimage.binary_dilation(skeleton, np.ones((3, 3, 3), bool)).astype(np.uint8)


@pytest.fixture
def no_graph_building(monkeypatch):
    """Fail the test if the stage gets as far as building a graph."""

    def refuse(*args, **kwargs):
        raise AssertionError("graph building ran on a skeleton with no vessel")

    monkeypatch.setattr(haemolynx.graph, "build_graph_from_skeleton", refuse)


def test_an_empty_skeleton_of_an_empty_segmentation_points_at_the_segmentation(
    tmp_path, no_graph_building
):
    empty = np.zeros((12, 12, 12), dtype=bool)
    volume = _volume(tmp_path, empty, np.zeros(empty.shape, dtype=np.uint8))

    with pytest.raises(ValueError, match="The skeleton is empty") as raised:
        build_network(_settings(tmp_path), volume, SCHEMA)

    message = str(raised.value)
    assert "The segmented image (input.tif) has no foreground either" in message
    assert "check the segmentation" in message
    assert "index pointer" not in message
    assert not _graph_path(tmp_path).exists()


def test_a_skeleton_of_isolated_voxels_names_its_voxels_and_the_skeletonise_settings(
    tmp_path, no_graph_building
):
    skeleton = _isolated_voxels()
    volume = _volume(tmp_path, skeleton, _specks_under(skeleton))

    with pytest.raises(ValueError, match="no two of them touch") as raised:
        build_network(_settings(tmp_path), volume, SCHEMA)

    message = str(raised.value)
    assert "The skeleton has 2 voxel(s)" in message
    assert "The segmented image (input.tif) has 54 foreground voxels" in message
    assert "isolated specks thin to isolated points" in message
    assert "skeleton_min_branch_length (3 voxels)" in message
    # Off by default, so it removed nothing, and is not named.
    assert "skeleton_min_component_percent" not in message
    assert not _graph_path(tmp_path).exists()


def test_a_skeleton_component_filter_that_is_on_is_named(tmp_path, no_graph_building):
    skeleton = _isolated_voxels()
    volume = _volume(tmp_path, skeleton, _specks_under(skeleton))

    with pytest.raises(ValueError) as raised:
        build_network(
            _settings(tmp_path, skeleton_min_component_percent=5.0, skeleton_min_branch_length=0),
            volume,
            SCHEMA,
        )

    message = str(raised.value)
    assert "skeleton_min_component_percent (5.0%)" in message
    assert "skeleton_min_branch_length" not in message


def test_segmentation_cleanup_that_removed_every_vessel_is_named(tmp_path, no_graph_building):
    empty = np.zeros((12, 12, 12), dtype=bool)
    volume = _volume(
        tmp_path,
        empty,
        np.zeros(empty.shape, dtype=bool),
        raw_segmented_image=_specks_under(_isolated_voxels()).astype(bool),
    )
    settings = _settings(
        tmp_path,
        segmentation_cleanup=True,
        segmentation_cleanup_remove_small_volumes=True,
        segmentation_cleanup_remove_small_min_volume_um3=100.0,
    )

    with pytest.raises(ValueError, match="The skeleton is empty") as raised:
        build_network(settings, volume, SCHEMA)

    message = str(raised.value)
    assert "Segmentation cleanup removed all 54 foreground voxels" in message
    assert "segmentation_cleanup_remove_small_volumes" in message
    assert "segmentation_cleanup_smooth_surfaces" not in message
    assert "has no foreground either" not in message


def test_a_loaded_skeleton_says_where_it_came_from(tmp_path, no_graph_building):
    skeleton = _isolated_voxels()
    volume = _volume(tmp_path, skeleton, _specks_under(skeleton))

    with pytest.raises(ValueError, match="no two of them touch") as raised:
        build_network(_settings(tmp_path, do_skeletonize=False), volume, SCHEMA)

    message = str(raised.value)
    assert str(tmp_path / "out" / "input_skeleton.npy") in message
    assert "turn do_skeletonize on" in message


def test_a_skeleton_with_a_vessel_in_it_still_builds_its_graph(tmp_path):
    """The check stops only a skeleton with no vessel: one straight vessel,
    beside an isolated voxel, builds."""
    skeleton = np.zeros((40, 12, 12), dtype=bool)
    skeleton[5:35, 6, 6] = True
    skeleton[20, 1, 1] = True
    mask = ndimage.binary_dilation(skeleton, iterations=2).astype(np.uint8)

    network = build_network(_settings(tmp_path), _volume(tmp_path, skeleton, mask), SCHEMA)

    assert network.graph.number_of_edges() == 1
    assert _graph_path(tmp_path).exists()
