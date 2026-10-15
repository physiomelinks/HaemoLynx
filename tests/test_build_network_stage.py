"""What ``build_network()`` does when graph building leaves no vessels.

Every later stage needs a vessel, and each failed on something else when there
was none: the 3D drawing on missing node positions, boundary assignment on
missing terminals, while the 2D drawing went through and saved the empty
graph. So the stage stops there and says which step took the last vessel.

Once it has the network it also warns when the run's dense solvers would not
fit in the free memory.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage

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
            **overrides,
        }
    )
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _cross(size: int, start: int, stop: int) -> np.ndarray:
    """Two straight skeleton lines crossing at the volume's centre: four arms."""
    skeleton = np.zeros((size, size, size), dtype=bool)
    centre = size // 2
    skeleton[start:stop, centre, centre] = True
    skeleton[centre, centre, start:stop] = True
    return skeleton


def _mask_under(skeleton: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(skeleton, iterations=2).astype(np.uint8)


def _volume(tmp_path: Path, skeleton: np.ndarray, image: np.ndarray) -> SkeletonisedVolume:
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    return SkeletonisedVolume(
        image=image,
        skeleton=skeleton,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
    )


def _graph_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "input_graph.pkl"


#: 3 um arms on 1 um voxels, all shorter than the default 5 um
#: cluster_collapse_distance: the four vessels collapse into one point.
SHORT_CROSS = _cross(10, 2, 8)
#: 15 um arms: longer than the collapse distance and the stub prune's reach.
LONG_CROSS = _cross(40, 5, 35)


@pytest.mark.parametrize(
    "render",
    [
        {"final_render_mode": "3d"},
        {"final_render_mode": "2d"},
        {"visualize_results": False},
    ],
    ids=["3d drawing", "2d drawing", "no drawing"],
)
def test_a_skeleton_shorter_than_the_collapse_distance_stops_the_stage_and_says_why(
    tmp_path, render
):
    settings = _settings(tmp_path, **render)
    volume = _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS))

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    message = str(raised.value)
    assert "built 4 vessel(s), and the last went at the collapse_node_clusters step" in message
    assert "cluster_collapse_distance (5.0 um)" in message
    # The image is not the cause here, and the message says so.
    assert "11 of those 11 voxels lie in the image's foreground" in message
    # Stopped before anything was saved or drawn from the empty graph.
    assert not _graph_path(tmp_path).exists()
    assert list((tmp_path / "plots").glob("final_graph*")) == []


def test_a_run_with_a_thick_vessel_region_stops_with_the_same_reason(tmp_path):
    """The thick-vessel split, and the clean-up after it, run on whatever graph
    building left before the check: an empty build passes through them and is
    still stopped with the step that took its last vessel."""
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS))
    volume.thick_vessel_mask = _mask_under(SHORT_CROSS).astype(bool)

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    assert "the last went at the collapse_node_clusters step" in str(raised.value)
    assert not _graph_path(tmp_path).exists()


def test_an_image_that_does_not_cover_the_skeleton_is_named_as_the_cause(tmp_path):
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, LONG_CROSS, np.zeros(LONG_CROSS.shape, dtype=np.uint8))

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    message = str(raised.value)
    assert "the last went at the prune_vascular_stubs step" in message
    assert "0 of those 59 voxels lie in the image's foreground" in message
    assert "must be the segmentation the skeleton was made from" in message


def test_the_same_cross_with_longer_arms_builds_its_four_vessels(tmp_path):
    """The check only fires on no vessels: the cross above, in a mask, with
    arms long enough to keep."""
    settings = _settings(tmp_path)
    network = build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)

    assert network.graph.number_of_edges() == 4
    assert _graph_path(tmp_path).exists()
    assert (tmp_path / "plots" / "final_graph_3d.html").exists()


@pytest.mark.parametrize("run_haemodynamics", [True, False])
def test_the_built_network_warns_when_its_dense_solve_will_not_fit(
    tmp_path, monkeypatch, caplog, run_haemodynamics
):
    """As soon as the node count is known, so a run can be stopped before its
    diameters are measured -- and only when the run will solve."""
    from haemolynx.haemodynamics import resistance

    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 1000)
    settings = _settings(tmp_path, run_haemodynamics=run_haemodynamics, visualize_results=False)
    with caplog.at_level(logging.WARNING, logger="haemolynx.haemodynamics.resistance"):
        network = build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)

    nodes = f"{network.graph.number_of_nodes():,} nodes"
    memory = [r.getMessage() for r in caplog.records if "of memory free now" in r.getMessage()]
    if run_haemodynamics:
        assert len(memory) == 2 and all(nodes in message for message in memory)
    else:
        assert memory == []


def _spy_on_graph_building(monkeypatch) -> dict:
    """The keyword arguments graph building, the lumen clean-up and the
    thick-vessel split are called with, by name, in call order."""
    from haemolynx import graph

    seen: dict = {}

    def spy(name, real):
        def wrapper(*args, **kwargs):
            seen.setdefault(name, []).append(kwargs)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(graph, "build_graph_from_skeleton", spy("build", graph.build_graph_from_skeleton))
    monkeypatch.setattr(graph.assemble, "lumen_cleanup", spy("cleanup", graph.assemble.lumen_cleanup))
    monkeypatch.setattr(
        graph, "insert_thick_vessel_junction_nodes", spy("split", graph.insert_thick_vessel_junction_nodes)
    )
    return seen


def test_a_thick_vessel_regions_centreline_reaches_every_clean_up_and_the_split(tmp_path, monkeypatch):
    """Every clean-up keeps the strand on the thick vessels' centreline tree
    of two in one lumen (judged by the wall alone it broke them, E14.5,
    2026-10-15): graph building's own, the one after smoothing and the one
    after the split are all handed it, and the split the same tree, found
    once."""
    seen = _spy_on_graph_building(monkeypatch)
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS))
    volume.thick_vessel_mask = _mask_under(LONG_CROSS).astype(bool)

    build_network(settings, volume, SCHEMA)

    along = seen["build"][0]["along_thick_centreline"]
    assert along is not None
    assert len(seen["cleanup"]) == 2  # graph building's, then build_network's
    assert all(call["along_thick_centreline"] is along for call in seen["cleanup"])
    arm = np.array([[20.0, 20.0, float(x)] for x in range(8, 33)])
    assert along(arm) > 20.0
    (split,) = seen["split"]
    assert split["centreline"].shape == LONG_CROSS.shape and split["centreline"].any()


def test_a_run_without_a_thick_vessel_region_leaves_the_choice_to_the_wall(tmp_path, monkeypatch):
    seen = _spy_on_graph_building(monkeypatch)
    build_network(_settings(tmp_path), _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert seen["build"][0]["along_thick_centreline"] is None
    assert all(call["along_thick_centreline"] is None for call in seen["cleanup"])
    assert "split" not in seen


def test_an_empty_build_leaves_the_graph_a_run_saved_before(tmp_path):
    (tmp_path / "out").mkdir()
    saved = nx.MultiGraph()
    saved.add_edge(0, 1)
    _graph_path(tmp_path).write_bytes(pickle.dumps(saved))
    before = _graph_path(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="Graph building left no vessels"):
        build_network(
            _settings(tmp_path),
            _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS)),
            SCHEMA,
        )

    assert _graph_path(tmp_path).read_bytes() == before


def test_a_saved_graph_with_no_vessels_is_refused_when_loaded(tmp_path):
    (tmp_path / "out").mkdir()
    _graph_path(tmp_path).write_bytes(pickle.dumps(nx.MultiGraph()))

    with pytest.raises(ValueError, match="has no vessels") as raised:
        build_network(
            _settings(tmp_path, do_graph_building=False),
            _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS)),
            SCHEMA,
        )

    assert str(_graph_path(tmp_path)) in str(raised.value)
    assert "do_graph_building" in str(raised.value)
