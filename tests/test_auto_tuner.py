import pytest
from pathlib import Path
import optuna
from optuna.trial import FixedTrial
from ImageLynx.statistics.auto_tuner import SkeletonObjective, PreprocessingObjective, run_optuna_skeleton_optimization

def _get_dummy_fixed_trial():
    """Helper to generate a consistent FixedTrial for objective logic testing."""
    return FixedTrial({
        "min_branch_length": 10,
        "max_bridge_distance": 5,
        "min_component_percent": 1.0,
        "bundle_scan_size": 5,
        "bundle_density_fraction": 0.05,
        "bundle_max_connections": 3,
        "bundle_hub_min_spacing": 2,
        "smoothing_alpha": 0.5,
        "prune_by_tortuosity": 2.0
    })

def test_objective_perfect_scores_yield_zero_loss():
    """Phase 2: Ensures a perfect pipeline result computes to exactly 0.0 loss."""
    def perfect_pipeline(kwargs):
        return {
            "volumetric": {"dice_coefficient": 1.0},
            "completeness": {"orphaned_volume_fraction": 0.0},
            "topology": {
                "graph_fundamental_loops": 0,
                "terminal_node_ratio": 0.0,
                "degree3_bifurcation_ratio": 1.0,
                "edge_length_std": 0.0
            }
        }
    
    objective = SkeletonObjective(perfect_pipeline)
    loss = objective(_get_dummy_fixed_trial())
    
    assert loss == 0.0, f"Expected 0.0 loss for perfect scores, got {loss}"

def test_objective_penalizes_over_pruning():
    """Phase 2: Ensures the weighted loss function scales penalties correctly.

    Contract change (#98, Tier 2 item 14): this previously expected 635.0, which included
    graph_fundamental_loops * 0.1 = 5.0 for the 50 loops below. The loop term has been
    removed - it is the first Betti number, i.e. the readout H1 section 1.1 depends on - so
    the expected total is now 630.0 and the loop count no longer contributes at all. The test
    was renamed accordingly: "spiderwebs" is no longer a thing this objective penalises.
    """
    def terrible_pipeline(kwargs):
        return {
            "volumetric": {"dice_coefficient": 0.5},
            "completeness": {"orphaned_volume_fraction": 0.8},
            "topology": {
                "graph_fundamental_loops": 50,
                "terminal_node_ratio": 1.0, # 100% dead ends
                "degree3_bifurcation_ratio": 0.0, # 0% Y-bifurcations
                "edge_length_std": 100.0
            }
        }
    
    objective = SkeletonObjective(terrible_pipeline)
    loss = objective(_get_dummy_fixed_trial())
    
    # Expected Loss = 50 (dice) + 80 (orphaned) + 475 (terminal) + 20 (deg3) + 5 (edge_var)
    assert loss == 630.0, f"Expected 630.0 loss, got {loss}"

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
            "topology": {
                "graph_fundamental_loops": loops,
                "terminal_node_ratio": 0.0,
                "degree3_bifurcation_ratio": 1.0,
                "edge_length_std": 0.0
            }
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
            "topology": {
                "graph_fundamental_loops": 5,
                "terminal_node_ratio": 0.0,
                "degree3_bifurcation_ratio": 1.0,
                "edge_length_std": 0.0
            }
        }
        
    # Run a tiny 3-trial optimization to trigger the file exports
    run_optuna_skeleton_optimization(simple_eval, n_trials=3, output_dir=tmp_path)
    
    assert (tmp_path / "best_skeleton_params.yaml").exists(), "YAML config file was not exported."
    assert (tmp_path / "optuna_history.html").exists(), "Optimization history HTML was not exported."
    assert (tmp_path / "optuna_param_importances.html").exists(), "Parameter importance HTML was not exported."

def _skeleton_bench(dice, orphaned, terminal_ratio, loops, deg3=0.9, edge_std=8.0):
    return {
        "volumetric": {"dice_coefficient": dice},
        "completeness": {"orphaned_volume_fraction": orphaned},
        "topology": {
            "graph_fundamental_loops": loops,
            "terminal_node_ratio": terminal_ratio,
            "degree3_bifurcation_ratio": deg3,
            "edge_length_std": edge_std,
        },
    }


def test_skeleton_objective_loss_ignores_loop_count():
    """beta-1 must not enter the loss at all.

    graph_fundamental_loops is E - V + C, the first Betti number, which is exactly what
    H1 section 1.1 reads out. Any weight on it lets the tuner improve its score by deleting
    real anastomoses, and does so harder on denser networks.
    """
    objective = SkeletonObjective(lambda kwargs: None)

    few_loops = _skeleton_bench(0.6, 0.01, 0.03, loops=100)
    many_loops = _skeleton_bench(0.6, 0.01, 0.03, loops=5000)

    assert objective._calculate_loss(few_loops) == objective._calculate_loss(many_loops)


def test_skeleton_objective_now_prefers_the_higher_fidelity_configuration():
    """The measured case from the #98 assessment that proved the objective was inverted.

    Configuration B beats A on every fidelity metric the suite measures - better Dice, less
    orphaned volume, a better terminal ratio - and was nonetheless rejected, 246.1 against
    164.7, because its higher loop count dominated the loss. B must now win.
    """
    objective = SkeletonObjective(lambda kwargs: None)

    config_a = _skeleton_bench(dice=0.598, orphaned=0.010, terminal_ratio=0.062, loops=1156)
    config_b = _skeleton_bench(dice=0.688, orphaned=0.005, terminal_ratio=0.025, loops=2127)

    assert objective._calculate_loss(config_b) < objective._calculate_loss(config_a)


def test_preprocessing_objective_loss_ignores_euler_characteristic():
    """chi = beta0 - beta1 + beta2, so penalising chi < 1 penalises vascular loops.

    A capillary bed legitimately has a strongly negative Euler characteristic.
    """
    objective = PreprocessingObjective(lambda kwargs: None)

    base = {
        "confidence": 0.8, "probability_yield": 0.3, "crispness": 0.5,
        "fragmentation": 10, "surface_area_ratio": 0.3,
        "mean_uncertainty": 0.1, "high_uncertainty_fraction": 0.01,
    }
    positive_euler = dict(base, euler_characteristic=1)
    very_negative_euler = dict(base, euler_characteristic=-18870)

    assert objective._calculate_loss(positive_euler) == objective._calculate_loss(very_negative_euler)
