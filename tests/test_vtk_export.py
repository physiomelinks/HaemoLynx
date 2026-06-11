import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
examples_path = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from carotid_image_to_model import PipelineConfig, carotid_image_to_model

def test_mandatory_vtk_generation():
    """Verify that when chunk_fraction < 1.0, 4 VTI files are generated."""
    # We will mock pv.ImageData().save to verify how many times it was called and what paths
    
    # We can't easily mock the entire pipeline end-to-end without a real image,
    # so let's just mock _load_raw_probability_field and io.load_3d_tif.
    
    with patch('carotid_image_to_model._load_raw_probability_field') as mock_load_prob, \
         patch('ImageLynx.io.load_3d_tif') as mock_load_tif, \
         patch('pyvista.ImageData.save') as mock_save, \
         patch('carotid_image_to_model.sys.exit') as mock_exit:
         
         # 10x10x10 mock arrays
         mock_prob = np.zeros((10, 10, 10), dtype=np.float32)
         mock_entropy = np.zeros((10, 10, 10), dtype=np.float32)
         mock_load_prob.return_value = (mock_prob, mock_entropy)
         
         mock_load_tif.side_effect = [np.zeros((10, 10, 10), dtype=np.float32)]
         
         # Mock sys.exit to actually halt execution like a real exit
         mock_exit.side_effect = Exception("sys.exit was called")
         
         config = PipelineConfig(chunk_fraction=0.5, export_grid_preview=True)
         
         # Run orchestrator
         with pytest.raises(Exception, match="sys.exit was called"):
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config)
         
         assert mock_load_tif.call_count == 1
         
         # Should save 4 VTK files: RawAnatomy, RawProbability, GridPreview, ShannonEntropy
         assert mock_save.call_count == 4
         
         calls = [call.args[0].name for call in mock_save.call_args_list]
         assert any("raw_anatomy.vti" in name for name in calls)
         assert any("raw_probability.vti" in name for name in calls)
         assert any("grid_preview.vti" in name for name in calls)
         assert any("shannon_entropy.vti" in name for name in calls)

def test_grid_wireframe_painting():
    """Verify that the grid wireframe array is perfectly hollow."""
    from ImageLynx.graph.tiling import generate_evenly_distributed_bounding_boxes
    
    shape = (20, 20, 20)
    fraction = 0.5
    margin = 0
    
    grid_mask = np.zeros(shape, dtype=np.uint8)
    for bbox in generate_evenly_distributed_bounding_boxes(shape, fraction, margin):
        z1, z2, y1, y2, x1, x2 = bbox['core']
        
        if z1 < shape[0]: grid_mask[z1, y1:y2, x1:x2] = 255
        if z2 - 1 >= 0 and z2 - 1 < shape[0]: grid_mask[z2-1, y1:y2, x1:x2] = 255
        if y1 < shape[1]: grid_mask[z1:z2, y1, x1:x2] = 255
        if y2 - 1 >= 0 and y2 - 1 < shape[1]: grid_mask[z1:z2, y2-1, x1:x2] = 255
        if x1 < shape[2]: grid_mask[z1:z2, y1:y2, x1] = 255
        if x2 - 1 >= 0 and x2 - 1 < shape[2]: grid_mask[z1:z2, y1:y2, x2-1] = 255
        
    # Test that the center of a chunk is hollow (e.g., chunk 1 is 0-10, center is 5)
    assert grid_mask[5, 5, 5] == 0
    
    # Test that the boundary is solid
    assert grid_mask[0, 5, 5] == 255
    assert grid_mask[9, 5, 5] == 255
    assert grid_mask[5, 0, 5] == 255
    assert grid_mask[5, 9, 5] == 255

def test_early_termination_toggle():
    """Verify sys.exit is called based on export_grid_preview toggle."""
    with patch('carotid_image_to_model._load_raw_probability_field') as mock_load_prob, \
         patch('ImageLynx.io.load_3d_tif') as mock_load_tif, \
         patch('pyvista.ImageData.save'), \
         patch('carotid_image_to_model.sys.exit') as mock_exit, \
         patch('ImageLynx.pipeline.map_reduce.map_reduce_pipeline') as mock_mr, \
         patch('carotid_image_to_model._build_and_optimize_graph'):
         
         mock_prob = np.zeros((10, 10, 10), dtype=np.float32)
         mock_load_prob.return_value = (mock_prob, None)
         mock_load_tif.side_effect = [np.zeros((10, 10, 10), dtype=np.float32)]
         
         mock_exit.side_effect = Exception("sys.exit was called")
         
         # Run WITH export_grid_preview
         config_exit = PipelineConfig(chunk_fraction=0.5, export_grid_preview=True)
         with pytest.raises(Exception, match="sys.exit was called"):
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config_exit)
         
         assert mock_exit.call_count == 1
         
         # Reset mocks
         mock_exit.reset_mock()
         
         # Mock map_reduce_pipeline so it doesn't crash on an empty array
         mock_mr.return_value = np.zeros((10, 10, 10), dtype=np.uint8)
         
         # Run WITHOUT export_grid_preview
         config_continue = PipelineConfig(chunk_fraction=0.5, export_grid_preview=False, do_graph_building=False) 
         try:
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config_continue)
         except Exception:
             pass # We only care that it proceeded past the sys.exit block
         
         assert mock_exit.call_count == 0
         # Ensure it proceeded to Phase 2 (map_reduce_pipeline)
         assert mock_mr.call_count == 1