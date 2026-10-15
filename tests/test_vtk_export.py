import pytest
import numpy as np
from unittest.mock import patch

from carotid_image_to_model import PipelineConfig, carotid_image_to_model

def test_mandatory_vtk_generation(tmp_path):
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
         
         config = PipelineConfig(chunk_fraction=0.5, export_grid_preview=True,
                                 vtk_output_prefix=tmp_path / "resistance_network")
         
         # Run orchestrator
         with pytest.raises(Exception, match="sys.exit was called"):
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config)
         
         assert mock_load_tif.call_count == 1
         
         # Should save 5 VTK files: RawAnatomy (Global + Sub), RawProbability, GridPreview, ShannonEntropy
         assert mock_save.call_count == 5
         
         calls = [call.args[0].name for call in mock_save.call_args_list]
         assert any("raw_anatomy_global.vti" in name for name in calls)
         assert any("raw_anatomy_subvolume.vti" in name for name in calls)
         assert any("raw_probability.vti" in name for name in calls)
         assert any("grid_preview.vti" in name for name in calls)
         assert any("shannon_entropy.vti" in name for name in calls)

def test_dual_anatomy_export_alignment_real(tmp_path):
    """Verify global and sub-volume anatomy exports maintain correct array lengths using actual PyVista."""
    with patch('carotid_image_to_model._load_raw_probability_field') as mock_load_prob, \
         patch('ImageLynx.io.load_3d_tif') as mock_load_tif, \
         patch('pyvista.ImageData.save') as mock_save, \
         patch('carotid_image_to_model.sys.exit') as mock_exit:
         
         mock_prob = np.zeros((50, 50, 50), dtype=np.float32)
         mock_load_prob.return_value = (mock_prob, None)
         
         mock_load_tif.side_effect = [np.zeros((100, 100, 100), dtype=np.float32)]
         
         mock_exit.side_effect = Exception("sys.exit was called")
         
         from carotid_image_to_model import SkeletonConfig
         config = PipelineConfig(chunk_fraction=0.5, export_grid_preview=True,
                                 vtk_output_prefix=tmp_path / "resistance_network")
         skel_config = SkeletonConfig(sub_volume_percentage=0.5)
         
         with pytest.raises(Exception, match="sys.exit was called"):
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config, skel_config=skel_config)
             
         # The mock_save is called on the pv.ImageData object (self)
         # We can get the object from the call context
         saved_objects = [call.args[0].name for call in mock_save.call_args_list]
         
         # Note: in Python, methods bound to objects pass `self` implicitly, 
         # but mock_save doesn't capture `self` in args unless we patch it differently.
         # The simplest assertion is that it didn't crash PyVista with "Invalid array shape".
         # Since we used the real pv.ImageData, if the lengths didn't match the dimensions,
         # PyVista would have thrown a ValueError during `point_data` assignment and caught by our try-except.
         # If the try-except caught it, the save wouldn't be called for the subvolume!
         
         assert "resistance_network_raw_anatomy_global.vti" in saved_objects
         assert "resistance_network_raw_anatomy_subvolume.vti" in saved_objects

def test_early_termination_toggle(tmp_path):
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
         config_exit = PipelineConfig(chunk_fraction=0.5, export_grid_preview=True,
                                      vtk_output_prefix=tmp_path / "resistance_network")
         with pytest.raises(Exception, match="sys.exit was called"):
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config_exit)
         
         assert mock_exit.call_count == 1
         
         # Reset mocks
         mock_exit.reset_mock()
         
         # Mock map_reduce_pipeline so it doesn't crash on an empty array
         mock_mr.return_value = np.zeros((10, 10, 10), dtype=np.uint8)
         
         # Run WITHOUT export_grid_preview
         # The cache folder sits beside the output prefix, so keep both inside tmp_path.
         config_continue = PipelineConfig(chunk_fraction=0.5, export_grid_preview=False, do_graph_building=False,
                                          vtk_output_prefix=tmp_path / "resistance_network")
         try:
             carotid_image_to_model(image_path="dummy.tif", pipeline_config=config_continue)
         except Exception:
             pass # We only care that it proceeded past the sys.exit block
         
         assert mock_exit.call_count == 0
         # Ensure it proceeded to Phase 2 and Phase 3 (map_reduce_pipeline called twice)
         assert mock_mr.call_count == 2