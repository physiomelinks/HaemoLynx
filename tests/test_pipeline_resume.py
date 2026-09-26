"""A resumed run is handed what it already has, and leaves the pipeline's files alone.

"Run from this stage" used to write the checkpoint's later-stage graph over
``{stem}_graph.pkl`` (graph building's own output) so that ``build_network``
could load it, and then re-ran ilastik, reloaded and re-checked the image,
skeleton and masks, and re-drew the 3D graph before throwing the loaded graph
away. These tests pin the two stage-level halves of the fix: ``build_network``
takes a resumed graph as it is, and ``segment`` takes a segmented image it
already has. How the run hands them over is in ``test_pipeline_progress.py``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.pipeline import SkeletonisedVolume
from haemolynx.pipeline import stages
from test_hidden_settings_cannot_affect_a_run import (
    SCHEMA,
    _settings_for,
    _tiny_skeleton,
    _write_mask,
)


def _volume(tmp_path):
    skeleton = _tiny_skeleton()
    return SkeletonisedVolume(
        image=np.zeros(skeleton.shape, dtype=np.uint8),
        skeleton=skeleton,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
    )


def test_build_network_takes_a_resumed_graph_as_it_is(tmp_path, monkeypatch):
    settings = _settings_for(tmp_path, {"do_graph_building": True, "visualize_results": True})
    settings["input_path"] = tmp_path / "input.tif"
    graph_pkl = tmp_path / "out" / "input_graph.pkl"
    graph_pkl.write_bytes(b"what graph building made")
    ran = []
    for name in ("build_graph_from_skeleton", "diagnose_skeleton_graph_consistency"):
        monkeypatch.setattr(stages.graph, name, lambda *a, _n=name, **k: ran.append(_n))
    monkeypatch.setattr(
        stages, "_write_run_final_graph_3d_html", lambda *a, **k: ran.append("3d html")
    )
    resumed = nx.MultiGraph()
    resumed.add_edge(0, 1, length=5.0, flow=2.0)

    network = stages.build_network(settings, _volume(tmp_path), SCHEMA, resumed_graph=resumed)

    assert network.graph is resumed
    assert resumed.edges[0, 1, 0]["flow"] == 2.0
    assert resumed.graph["image_voxel_size_zyx"] == (1.0, 1.0, 1.0)
    assert ran == []
    assert graph_pkl.read_bytes() == b"what graph building made"


def test_segment_uses_a_segmented_image_it_already_has_instead_of_ilastik(tmp_path, monkeypatch):
    produced = _write_mask(tmp_path / "raw_segmented.tif")
    settings = _settings_for(
        tmp_path,
        {
            "use_ilastik_segmentation": True,
            "ilastik_unsegmented_image_path": tmp_path / "raw.tif",
            "ilastik_classifier_path": tmp_path / "classifier.ilp",
            "ilastik_output_dir": tmp_path,
        },
        input_path=None,
    )
    ran = []
    monkeypatch.setattr(
        stages.io, "run_ilastik_headless_segmentation", lambda **k: ran.append(k)
    )

    inputs = stages.segment(settings, segmented_path=produced)

    assert ran == []
    assert inputs.image_path == produced
    assert settings["input_path"] == produced
    assert inputs.input_format == "tif"
