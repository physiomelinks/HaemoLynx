"""The cache path hands boundary selection the cached mask's shape (open item 28).

``--use-cache-dir`` used to set the image to ``np.zeros((1, 1, 1))`` for an .h5 input and to a
fixed, uncropped raw TIFF for a .tif input. Boundary selection and the statistics read that
image's shape, so the cache path chose its inlets and outlets against faces the graph was never
built in (WKY-A: 5 / 539 against 126 / 119 on a fresh run, band rule).
"""
import pickle
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from ImageLynx import cb_settings
from ImageLynx.graph.boundaries import select_boundary_terminal_nodes_by_face

C = pytest.importorskip("carotid_image_to_model")

SHAPE = (20, 50, 20)


def _y_network():
    """Inlet on the low axis-1 face, outlet on the high face, one interior dead end."""
    G = nx.MultiGraph()
    G.add_node("inlet", pos=np.array([10.0, 0.0, 10.0]))
    G.add_node("junction", pos=np.array([10.0, 20.0, 10.0]))
    G.add_node("outlet", pos=np.array([10.0, 49.0, 10.0]))
    G.add_node("spur", pos=np.array([5.0, 25.0, 15.0]))
    G.add_edge("inlet", "junction", key=0, length=20.0, voxels=[[10, 0, 10], [10, 20, 10]])
    G.add_edge("junction", "outlet", key=0, length=29.0, voxels=[[10, 20, 10], [10, 49, 10]])
    G.add_edge("junction", "spur", key=0, length=11.0, voxels=[[10, 20, 10], [5, 25, 15]])
    return G


def _cache(tmp_path, G=None):
    """A cache dir as a fresh run of ``dummy.<ext>`` would leave it, with a SHAPE mask."""
    config = C.PipelineConfig()
    config.vtk_output_prefix = tmp_path / "dummy_network"
    config.pre_generated_mask_and_skeleton = True
    cache_dir = tmp_path / "dummy_cache"
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "vessel_mask.npy", np.zeros(SHAPE, dtype=np.uint8))
    np.save(cache_dir / "skeleton.npy", np.zeros(SHAPE, dtype=bool))
    with open(cache_dir / "network_graph.pkl", "wb") as f:
        pickle.dump(G if G is not None else nx.MultiGraph(), f)
    return config


def _image_passed_to_boundary_step(tmp_path, input_name):
    config = _cache(tmp_path)
    with patch("carotid_image_to_model._setup_boundary_conditions_and_haemodynamics") as mock_bc, \
         patch("carotid_image_to_model._export_and_solve_haemodynamics"), \
         patch("ImageLynx.io.load_3d_tif", side_effect=AssertionError("raw TIFF loaded")):
        mock_bc.return_value = ([], [], [])
        C.carotid_image_to_model(input_name, pipeline_config=config)
    args, kwargs = mock_bc.call_args
    return args[1], kwargs["binary"]


@pytest.mark.parametrize("input_name", ["dummy.h5", "dummy.tif"])
def test_the_cache_path_passes_the_cached_mask_shape(tmp_path, input_name):
    """No (1, 1, 1) placeholder for .h5, no fixed raw TIFF for .tif."""
    image, binary = _image_passed_to_boundary_step(tmp_path, input_name)
    assert image.shape == SHAPE
    assert binary.shape == SHAPE


def test_the_cache_path_picks_the_boundaries_of_the_mask_frame(tmp_path, monkeypatch):
    """The real boundary step, run on the cache path, picks the fresh-frame inlet and outlet."""
    monkeypatch.setattr(C, "VOXEL_SIZE_UM", (1.0, 1.0, 1.0))
    G = _y_network()
    expected = select_boundary_terminal_nodes_by_face(
        G.copy(), SHAPE, axis=cb_settings.BOUNDARY_AXIS,
        face_tolerance_voxels=cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS,
        voxel_size=(1.0, 1.0, 1.0))
    config = _cache(tmp_path, G)

    picked = {}
    real_select = C.graph.select_boundary_terminal_nodes_by_face

    def spy(*args, **kwargs):
        picked["result"] = real_select(*args, **kwargs)
        return picked["result"]

    class _Stop(Exception):
        pass

    # Stop right after boundary selection: diameters and the solve are not under test.
    monkeypatch.setattr(C.graph, "select_boundary_terminal_nodes_by_face", spy)
    monkeypatch.setattr(C.graph, "assign_branch_orders", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        C.carotid_image_to_model("dummy.h5", pipeline_config=config)

    assert picked["result"][:2] == (["inlet"], ["outlet"]) == tuple(expected[:2])


def test_fwhm_radius_is_refused_on_the_cache_path(tmp_path):
    """FWHM reads intensities; the cache holds only the mask."""
    config = _cache(tmp_path)
    hemo = C.HaemodynamicsConfig(
        diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}},
        radius_assignment_mode="fwhm_radius",
    )
    with pytest.raises(ValueError, match="fwhm_radius"):
        C.carotid_image_to_model("dummy.h5", hemo_config=hemo, pipeline_config=config)


def test_a_shape_larger_than_the_mask_is_refused(monkeypatch):
    """The fit check only catches a shape too small to hold the graph; this one is too big."""
    monkeypatch.setattr(C, "VOXEL_SIZE_UM", (1.0, 1.0, 1.0))
    hemo = C.HaemodynamicsConfig(diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}})
    bigger = (SHAPE[0], SHAPE[1] * 2, SHAPE[2])
    with pytest.raises(ValueError, match="vessel mask the graph was built from"):
        C._setup_boundary_conditions_and_haemodynamics(
            _y_network(), np.ones(bigger), hemo, C.GraphConfig(), "mock_path", "numpy",
            binary=np.zeros(SHAPE, dtype=bool))
