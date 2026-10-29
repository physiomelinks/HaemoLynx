"""The pipeline writes the file and folder names the batch-run reader opens.

The writer used to build each name from pieces of its own (the ``resistance_network`` prefix,
the ``_vessels.vtp`` / ``_nodes.vtp`` / ``_vessel_mask.vti`` suffixes, the ``_cache`` folder
suffix) while ``batch_outputs`` spelled the joined names out, so a change on one side left the
other stale. Each piece now has one home, and these check the writer's names against the
reader's. The names themselves are not pinned here (CODING_STANDARDS.md rule 2 spells each
once); the driver snapshot, which opens every run on disk through them, is what shows they
did not change.
"""
from pathlib import Path

import pytest

pytest.importorskip("pyvista")

import carotid_image_to_model as pipeline  # noqa: E402
from ImageLynx import batch_outputs  # noqa: E402
from ImageLynx.visualization import graph_to_vtk  # noqa: E402


def test_graph_to_vtk_writes_the_names_the_reader_opens(simple_graph, tmp_path):
    out = graph_to_vtk(simple_graph, tmp_path / batch_outputs.VTK_PREFIX,
                       constriction_spacing=1.0, constriction_length=0.4)
    assert Path(out["vessels_path"]) == tmp_path / batch_outputs.VESSELS_VTP_NAME
    assert Path(out["nodes_path"]) == tmp_path / batch_outputs.NODES_VTP_NAME


def test_the_pipeline_writes_the_mask_vti_the_reader_opens(tmp_path):
    prefix = tmp_path / batch_outputs.VTK_PREFIX
    assert pipeline.vessel_mask_vti_path(prefix) == tmp_path / batch_outputs.VESSEL_MASK_VTI_NAME


def test_the_pipeline_cache_folder_is_the_one_the_reader_finds(tmp_path):
    image = Path("anywhere") / "C1-CB3_ilastik_Probabilities.h5"
    cache = pipeline.cache_dir_for(tmp_path / batch_outputs.VTK_PREFIX, image)
    assert cache == tmp_path / (image.stem + batch_outputs.CACHE_SUFFIX)
    cache.mkdir()
    assert list(tmp_path.glob(batch_outputs.CACHE_GLOB)) == [cache]


def test_the_pipeline_prefix_is_the_readers():
    assert pipeline.PipelineConfig().vtk_output_prefix.name == batch_outputs.VTK_PREFIX
