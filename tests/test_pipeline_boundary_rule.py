"""The pipeline picks pressure boundaries by the face rule, as the H2 drivers do (open item 2).

Until item 2 the pipeline took every degree-1 node in a 25% band of axis 0 as an inlet or
outlet, while the H2 drivers took only the terminals on the two faces of
``cb_settings.BOUNDARY_AXIS``. Most degree-1 nodes are interior dead ends, so the two runs put
pressure on different nodes of the same graph.
"""
import inspect

import networkx as nx
import numpy as np
import pytest

from ImageLynx import cb_settings
from ImageLynx.graph.boundaries import select_boundary_terminal_nodes_by_face

C = pytest.importorskip("carotid_image_to_model")

SHAPE = (50, 50, 50)


def _y_network():
    """Inlet on the low axis-1 face, outlet on the high face, and one interior dead end.

    The interior dead end ``"spur"`` sits at axis-0 coordinate 5, inside the old 25% band of
    axis 0, so the band rule would have given it arterial pressure.
    """
    G = nx.MultiGraph()
    G.add_node("inlet", pos=np.array([10.0, 0.0, 10.0]))
    G.add_node("junction", pos=np.array([10.0, 20.0, 10.0]))
    G.add_node("outlet", pos=np.array([10.0, 49.0, 10.0]))
    G.add_node("spur", pos=np.array([5.0, 25.0, 30.0]))
    G.add_edge("inlet", "junction", key=0, length=20.0, voxels=[[10, 0, 10], [10, 20, 10]])
    G.add_edge("junction", "outlet", key=0, length=29.0, voxels=[[10, 20, 10], [10, 49, 10]])
    G.add_edge("junction", "spur", key=0, length=21.0, voxels=[[10, 20, 10], [5, 25, 30]])
    return G


def _hemo_config():
    # A graph-only check on a mock image with no mask, so EDT would refuse; the boundary step
    # is what is under test, not diameter estimation.
    return C.HaemodynamicsConfig(
        diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}},
        radius_assignment_mode="fwhm_radius",
    )


def test_graph_config_takes_the_face_rule_from_settings():
    config = C.GraphConfig()
    assert config.boundary_axis == cb_settings.BOUNDARY_AXIS
    assert config.face_tolerance_voxels == cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS
    for band_field in ("edge_percent", "end_percent", "node_edge_axis"):
        assert not hasattr(config, band_field), f"band-rule field {band_field} is back"


def test_pipeline_boundaries_are_the_h2_face_rule_boundaries(monkeypatch):
    monkeypatch.setattr(C, "VOXEL_SIZE_UM", (1.0, 1.0, 1.0))
    G = _y_network()
    expected = select_boundary_terminal_nodes_by_face(
        G.copy(), SHAPE, axis=cb_settings.BOUNDARY_AXIS,
        face_tolerance_voxels=cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS,
        voxel_size=(1.0, 1.0, 1.0))

    start, out, pair = C._setup_boundary_conditions_and_haemodynamics(
        G, np.ones(SHAPE), _hemo_config(), C.GraphConfig(), "mock_path", "numpy")

    assert (start, out) == (["inlet"], ["outlet"]) == (expected[0], expected[1])
    assert pair == ("inlet", "outlet")
    assert "spur" not in start + out


def test_a_placeholder_shape_is_refused(monkeypatch):
    """The .h5 cache path hands over np.zeros((1, 1, 1)) (open item 28)."""
    monkeypatch.setattr(C, "VOXEL_SIZE_UM", (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="does not fit the volume"):
        C._setup_boundary_conditions_and_haemodynamics(
            _y_network(), np.zeros((1, 1, 1)), _hemo_config(), C.GraphConfig(),
            "mock_path", "numpy")


@pytest.mark.parametrize("coord, fits", [(49.0, True), (49.9, True), (50.5, False), (-1.5, False)])
def test_frame_guard_allows_one_voxel_of_slack(coord, fits):
    """Nodes on the high face (index 49) fit; anything more than a voxel outside does not."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([10.0, coord, 10.0]))
    G.add_edge(0, 1)
    if fits:
        C._check_graph_fits_frame(G, SHAPE, (1.0, 1.0, 1.0))
    else:
        with pytest.raises(ValueError, match="axis 1"):
            C._check_graph_fits_frame(G, SHAPE, (1.0, 1.0, 1.0))


def test_frame_guard_scales_by_the_voxel_size():
    """At 1.866 um the 50-voxel volume ends at 91.43 um, not at 49."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([10.0, 91.4, 10.0]))
    G.add_edge(0, 1)
    C._check_graph_fits_frame(G, SHAPE, (1.866, 1.866, 1.866))
    with pytest.raises(ValueError):
        C._check_graph_fits_frame(G, SHAPE, (1.0, 1.0, 1.0))


def test_the_pipeline_calls_the_face_rule_not_the_band_rule():
    source = inspect.getsource(C._setup_boundary_conditions_and_haemodynamics)
    assert "select_boundary_terminal_nodes_by_face(" in source
    assert "select_boundary_terminal_nodes(" not in source
