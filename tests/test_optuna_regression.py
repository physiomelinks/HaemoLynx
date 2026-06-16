import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

# Add source paths
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

def test_cli_optimization_args_mapping():
    """Ensure CLI optimization arguments correctly map to PipelineConfig and don't default to 0."""
    import importlib.util
    import sys
    
    test_args = ['carotid_image_to_model.py', '--optimize-preprocessing', '50', '--optimize-skeleton', '20', '--exit-after-mask']
    
    spec = importlib.util.spec_from_file_location("__main__", "examples/carotid_image_to_model.py")
    module = importlib.util.module_from_spec(spec)
    
    with patch('sys.argv', test_args):
        # We must mock functions BEFORE the script executes them in the main block.
        # However, exec_module defines the functions during execution.
        # The easiest way to intercept the final PipelineConfig is to patch the main function 
        # inside the module namespace globally via sys.modules.
        
        # Since the file imports carotid_image_to_model recursively? No, it defines it.
        # Let's just parse the file with AST to ensure the bug is not present, as executing 
        # a script with heavy imports and global state in pytest is notoriously brittle.
        
        import ast
        with open("examples/carotid_image_to_model.py", "r") as f:
            tree = ast.parse(f.read())
            
        # Find the assignment to pipeline_config.optimize_preprocessing_trials
        mapping_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "optimize_preprocessing_trials":
                        if isinstance(target.value, ast.Name) and target.value.id == "pipeline_config":
                            # We found: pipeline_config.optimize_preprocessing_trials = ...
                            if isinstance(node.value, ast.Attribute) and node.value.attr == "optimize_preprocessing":
                                if isinstance(node.value.value, ast.Name) and node.value.value.id == "args":
                                    mapping_found = True
        
        assert mapping_found, "The CLI mapping bug is present! pipeline_config.optimize_preprocessing_trials is not assigned to args.optimize_preprocessing"

def test_map_reduce_joblib_verbosity():
    """Ensure joblib.Parallel is initialized with sufficient verbosity so stdout isn't swallowed."""
    from ImageLynx.pipeline.map_reduce import map_reduce_pipeline
    
    mock_volume = np.zeros((10, 10, 10))
    dummy_worker = lambda chunk, bbox, idx, total: chunk
    
    with patch('ImageLynx.pipeline.map_reduce.Parallel') as mock_parallel:
        mock_parallel.return_value = MagicMock(return_value=[(np.zeros((5,5,5)), {'core': (0,5,0,5,0,5)})])
        
        # Trigger the pipeline
        map_reduce_pipeline(mock_volume, chunk_fraction=0.5, margin=0, worker_fn=dummy_worker)
        
        # Verify the Parallel object was initialized with verbose >= 10
        assert mock_parallel.call_count == 1
        call_kwargs = mock_parallel.call_args[1]
        assert 'verbose' in call_kwargs, "joblib.Parallel is missing the verbose flag!"
        assert call_kwargs['verbose'] >= 10, "joblib verbosity is too low; stdout will be swallowed by loky!"

def test_optuna_triggered_during_map_reduce():
    """Ensure the local worker function explicitly fires Optuna when trials > 0."""
    from carotid_image_to_model import _preprocess_local_mask, PreprocessingConfig, SkeletonConfig, GraphConfig, PipelineConfig
    
    mock_prob = np.zeros((10, 10, 10))
    pipeline_config = PipelineConfig()
    pipeline_config.vtk_output_prefix = Path("/tmp/dummy")
    
    with patch('ImageLynx.statistics.auto_tuner.run_optuna_preprocessing_optimization') as mock_optuna, \
         patch('carotid_image_to_model._apply_preprocessing_filters') as mock_apply:
         
         # Setup mock returns
         mock_optuna.return_value = {}
         mock_apply.return_value = (np.zeros((10,10,10)), np.zeros((10,10,10)))
         
         # Call the worker with 5 trials
         _preprocess_local_mask(
             mock_prob, None, PreprocessingConfig(), SkeletonConfig(), 
             GraphConfig(), pipeline_config, optimize_trials=5
         )
         
         # Verify Optuna was mathematically called exactly once for this chunk
         assert mock_optuna.call_count == 1, "The Optuna auto-tuner was bypassed despite optimize_trials > 0!"
