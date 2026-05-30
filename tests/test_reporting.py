import pytest
import numpy as np
from pathlib import Path
import matplotlib
from ImageLynx.visualization.reporting import generate_model_results_dashboard

# Force Agg backend for testing
matplotlib.use('Agg')

def test_reporting_data_extraction_and_io(tmp_path):
    """
    Tests that the reporting engine successfully extracts 11 metrics, 
    ignores NaNs, and writes exactly 11 PNGs and 1 MD file.
    """
    # Create fake 1D VTK data with NaNs mixed in
    vtk_export = {
        "cell_data": {
            "assigned_diameter_um": np.array([5.0, 10.0, np.nan, 20.0]),
            "flow_abs": np.array([100.0, 200.0, 300.0, np.nan]),
            "flow_signed": np.array([-100.0, 200.0, 300.0, np.nan]),
            "hematocrit": np.array([0.45, 0.45, np.nan, 0.45]),
            "pressure_drop": np.array([10.0, 20.0, 30.0, 40.0]),
            "resistance": np.array([0.1, 0.2, 0.3, 0.4]),
            "viscosity": np.array([1.2, 1.2, 1.2, 1.2]),
            "wall_shear_stress_pa": np.array([15.0, 15.0, 15.0, 15.0])
        }
    }
    
    # Create fake 3D tissue grid data with NaNs
    perfusion_field = {
        "PO2_mmhg": np.array([[100.0, 50.0], [np.nan, 20.0]]),
        "PCO2_mmhg": np.array([[40.0, 45.0], [np.nan, 50.0]]),
        "pH": np.array([[7.4, 7.35], [np.nan, 7.3]])
    }
    
    # Run the dashboard generator
    generate_model_results_dashboard(vtk_export, perfusion_field, tmp_path)
    
    # Assertions
    md_file = tmp_path / "model_results.md"
    assert md_file.exists()
    
    plots_dir = tmp_path / "plots"
    assert plots_dir.exists()
    assert plots_dir.is_dir()
    
    # There should be exactly 11 pngs
    png_files = list(plots_dir.glob("*.png"))
    assert len(png_files) == 11
    
    expected_metrics = [
        "assigned_diameter_um", "flow_abs", "flow_signed", "hematocrit", 
        "pressure_drop", "resistance", "viscosity", "wall_shear_stress_pa",
        "PO2_mmhg", "PCO2_mmhg", "pH"
    ]
    
    for metric in expected_metrics:
        assert (plots_dir / f"{metric}_dist.png").exists()
        
    # Check Markdown content
    content = md_file.read_text()
    assert "Model Results Dashboard" in content
    assert "1D Hemodynamic Network Metrics" in content
    assert "3D Tissue Perfusion Metrics" in content
    assert "![PO2_mmhg Distribution](plots/PO2_mmhg_dist.png)" in content
    
    # Prove the NaNs were dropped successfully because the Min/Max bounds are correct
    # The max diameter should be 20.0 (ignoring NaN)
    assert "**Max:** `20.0" in content

def test_reporting_graceful_missing_data(tmp_path):
    """
    Tests that the dashboard handles missing keys gracefully (e.g. if the 
    Multi-Species solver was disabled, PCO2 and pH don't exist).
    """
    vtk_export = {
        "cell_data": {
            "assigned_diameter_um": np.array([5.0, 10.0]),
            # Missing flow, hematocrit, etc.
        }
    }
    
    # Only PO2 exists
    perfusion_field = {
        "PO2_mmhg": np.array([100.0, 50.0])
    }
    
    generate_model_results_dashboard(vtk_export, perfusion_field, tmp_path)
    
    md_file = tmp_path / "model_results.md"
    assert md_file.exists()
    content = md_file.read_text()
    
    # Check that it generated the plots that exist
    assert (tmp_path / "plots" / "assigned_diameter_um_dist.png").exists()
    assert (tmp_path / "plots" / "PO2_mmhg_dist.png").exists()
    
    # Check that it elegantly flagged the missing ones
    assert "*Metric not found in pipeline output.*" in content
    assert "*Metric not found in pipeline output (solver may be disabled).*" in content

def test_markdown_and_plot_generation_overwrite(tmp_path):
    """
    Tests that calling the dashboard twice overwrites the previous files
    without crashing or appending infinitely.
    """
    vtk_export = {"cell_data": {"assigned_diameter_um": np.array([5.0, 10.0])}}
    perfusion_field = {"PO2_mmhg": np.array([100.0, 50.0])}
    
    # Run once
    generate_model_results_dashboard(vtk_export, perfusion_field, tmp_path)
    file_size_1 = (tmp_path / "model_results.md").stat().st_size
    
    # Change the data slightly and run again
    vtk_export["cell_data"]["assigned_diameter_um"] = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    generate_model_results_dashboard(vtk_export, perfusion_field, tmp_path)
    file_size_2 = (tmp_path / "model_results.md").stat().st_size
    
    # The file should be overwritten (not appended) so the size shouldn't double
    # It will be slightly different because of the new data, but not 2x larger
    assert file_size_2 < (file_size_1 * 1.5)
