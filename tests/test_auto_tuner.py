import pytest
from pathlib import Path
import optuna
from optuna.trial import FixedTrial
from ImageLynx.statistics.auto_tuner import SkeletonObjective, run_optuna_skeleton_optimization

def _get_dummy_fixed_trial():
    """Helper to generate a consistent FixedTrial for objective logic testing."""
    return FixedTrial({
        "min_branch_length": 10,
        "max_bridge_distance": 5,
        "min_component_percent": 1.0,
        "bundle_scan_size": 5,
        "bundle_density_fraction": 0.5,
        "bundle_max_connections": 3
    })

def test_objective_perfect_scores_yield_zero_loss():
    """Phase 2: Ensures a perfect pipeline result computes to exactly 0.0 loss."""
    def perfect_pipeline(kwargs):
        return {
            "volumetric": {"dice_coefficient": 1.0},
            "completeness": {"orphaned_volume_fraction": 0.0},
            "topology": {"graph_fundamental_loops": 0}
        }
    
    objective = SkeletonObjective(perfect_pipeline)
    loss = objective(_get_dummy_fixed_trial())
    
    assert loss == 0.0, f"Expected 0.0 loss for perfect scores, got {loss}"

def test_objective_penalizes_over_pruning_and_spiderwebs():
    """Phase 2: Ensures the weighted loss function scales penalties correctly."""
    def terrible_pipeline(kwargs):
        return {
            "volumetric": {"dice_coefficient": 0.5},
            "completeness": {"orphaned_volume_fraction": 0.8},
            "topology": {"graph_fundamental_loops": 50}
        }
    
    objective = SkeletonObjective(terrible_pipeline)
    loss = objective(_get_dummy_fixed_trial())
    
    # Expected Loss = (1 - 0.5)*100 + 0.8*100 + 50*0.1 = 50 + 80 + 5 = 135.0
    assert loss == 135.0, f"Expected 135.0 loss, got {loss}"

def test_objective_prunes_on_pipeline_failure():
    """Phase 2: Ensures pipeline crashes are safely caught and signal a TrialPruned."""
    def crashing_pipeline(kwargs):
        raise ValueError("Simulated pipeline crash (e.g. disconnected graph).")
        
    def none_pipeline(kwargs):
        return None # Simulated empty result
        
    objective_crash = SkeletonObjective(crashing_pipeline)
    objective_none = SkeletonObjective(none_pipeline)
    
    trial = _get_dummy_fixed_trial()
    
    with pytest.raises(optuna.TrialPruned):
        objective_crash(trial)
        
    with pytest.raises(optuna.TrialPruned):
        objective_none(trial)

def test_run_optuna_skeleton_optimization_finds_minimum(tmp_path: Path):
    """
    Phase 3: Tests the full end-to-end Optuna TPE engine. 
    We rig the mock pipeline so that 'min_branch_length = 15' is the absolute global minimum.
    """
    def mock_eval(kwargs):
        # We artificially calculate error based on how far 'min_branch_length' is from 15.
        error = abs(kwargs["min_branch_length"] - 15)
        
        dsc = max(0.0, 1.0 - (error * 0.05))
        orphaned = min(1.0, error * 0.02)
        loops = error
        
        return {
            "volumetric": {"dice_coefficient": dsc},
            "completeness": {"orphaned_volume_fraction": orphaned},
            "topology": {"graph_fundamental_loops": loops}
        }
        
    # Run the real optimizer for a few dozen iterations.
    best_params = run_optuna_skeleton_optimization(mock_eval, n_trials=30, output_dir=tmp_path)
    
    # Assert that the TPE algorithm successfully found the hidden optimum (or very close to it)
    assert 10 <= best_params["min_branch_length"] <= 20, \
        f"Optimizer failed to converge near 15, got {best_params['min_branch_length']}"

def test_optuna_generates_yaml_and_html_plots(tmp_path: Path):
    """Phase 4: Ensures the visual dashboards and YAML exports are written to disk."""
    def simple_eval(kwargs):
        # Add artificial variance based on kwargs so the param_importances model doesn't crash from zero variance
        var = kwargs.get("min_branch_length", 0) * 0.01
        return {
            "volumetric": {"dice_coefficient": 0.9 - var},
            "completeness": {"orphaned_volume_fraction": 0.1 + var},
            "topology": {"graph_fundamental_loops": 5}
        }
        
    # Run a tiny 3-trial optimization to trigger the file exports
    run_optuna_skeleton_optimization(simple_eval, n_trials=3, output_dir=tmp_path)
    
    assert (tmp_path / "best_skeleton_params.yaml").exists(), "YAML config file was not exported."
    assert (tmp_path / "optuna_history.html").exists(), "Optimization history HTML was not exported."
    assert (tmp_path / "optuna_param_importances.html").exists(), "Parameter importance HTML was not exported."