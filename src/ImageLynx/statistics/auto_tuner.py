import logging
import yaml
import copy
from pathlib import Path
from typing import Callable, Dict, Any

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)

class SkeletonObjective:
    """Optuna objective function for tuning skeletonization parameters."""
    
    def __init__(self, pipeline_eval_fn: Callable):
        self.pipeline_eval_fn = pipeline_eval_fn

    def __call__(self, trial):
        # 1. Define the Bayesian search space (TPE limits)
        skel_kwargs = {
            "min_branch_length": trial.suggest_int("min_branch_length", 3, 12),
            # "max_bridge_distance": trial.suggest_int("max_bridge_distance", 0, 0),
            "max_bridge_distance": 0,
            "min_component_percent": trial.suggest_float("min_component_percent", 4.0, 6.0),
            "bundle_scan_size": trial.suggest_int("bundle_scan_size", 8, 10),
            "bundle_density_fraction": trial.suggest_float("bundle_density_fraction", 0.01, 0.05),
            # "bundle_max_connections": trial.suggest_int("bundle_max_connections", 2, 4),
            "bundle_max_connections": 5,
            # "bundle_hub_min_spacing": trial.suggest_int("bundle_hub_min_spacing", 0, 10),
            "bundle_hub_min_spacing": 0,
            "smoothing_alpha": trial.suggest_float("smoothing_alpha", 0.1, 2.5),
            "prune_by_tortuosity": trial.suggest_float("prune_by_tortuosity", 1.2, 3.5)
        }
        
        # 2. Evaluate pipeline (builds graph and runs benchmarks)
        try:
            bench_results = self.pipeline_eval_fn(skel_kwargs)
        except Exception as e:
            logger.debug(f"Trial pruned due to pipeline failure (likely disconnected graph): {e}")
            raise optuna.TrialPruned()
            
        if bench_results is None:
            raise optuna.TrialPruned()
            
        # 3. Calculate Loss Function (Minimize towards 0.0)
        vol = bench_results.get("volumetric", {})
        comp = bench_results.get("completeness", {})
        topo = bench_results.get("topology", {})
        
        dsc = vol.get("dice_coefficient", 0.0)
        orphaned = comp.get("orphaned_volume_fraction", 1.0)
        loops = topo.get("graph_fundamental_loops", 100)
        terminal_ratio = topo.get("terminal_node_ratio", 1.0)
        deg3_ratio = topo.get("degree3_bifurcation_ratio", 0.0)
        edge_variance = topo.get("edge_length_std", 100.0)
        
        # Base penalty: Missing volume (DSC difference from 1.0, heavily weighted)
        loss = (1.0 - dsc) * 100.0
        
        # Penalty: Over-pruning (orphaned tissue fraction, heavily weighted to preserve capillary beds)
        loss += orphaned * 100.0
        
        # Penalty: Spiderwebs and messy topology (soft penalty per extra loop)
        loss += loops * 0.1 
        
        # Penalty: Dead-ends. If > 5% of network is dead-ends, heavily penalize fragmentation.
        if terminal_ratio > 0.05:
            loss += (terminal_ratio - 0.05) * 500.0
            
        # Penalty: Unnatural Super-Hubs. Enforce that Y-bifurcations (Deg-3) dominate X-bifurcations (Deg-4+).
        # We want deg3_ratio to be as close to 1.0 as possible.
        loss += (1.0 - deg3_ratio) * 20.0
        
        # Penalty: Edge Length Variance. We want smooth, cohesive lengths, not massive jumps.
        # Downscale it so it doesn't overpower DSC, but provides a steady pull towards uniformity.
        loss += edge_variance * 0.05
        
        return loss

class EarlyStoppingCallback:
    """Optuna callback to stop optimization if the score hasn't improved after N trials."""
    def __init__(self, patience: int = 100):
        self.patience = patience
        self.best_score = None
        self.stagnant_trials = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Check if any trial has actually completed to avoid ValueError in study.best_value
        from optuna.trial import TrialState
        completed_trials = study.get_trials(states=[TrialState.COMPLETE])
        if not completed_trials:
            return

        current_score = study.best_value
        if self.best_score is None:
            self.best_score = current_score
            return
            
        if current_score < self.best_score:
            # We found a new minimum, reset the patience counter
            self.best_score = current_score
            self.stagnant_trials = 0
        else:
            # No improvement
            self.stagnant_trials += 1
            if self.stagnant_trials >= self.patience:
                logger.warning(
                    f"Early stopping triggered: The loss hasn't improved for {self.patience} consecutive trials. "
                    "The optimizer has fully converged."
                )
                study.stop()

def run_optuna_skeleton_optimization(
    pipeline_eval_fn: Callable,
    n_trials: int = 30,
    output_dir: Path = Path("outputs"),
    patience: int = 100
) -> Dict[str, Any]:
    """
    Executes the Bayesian optimization loop to find the best skeletonization parameters.
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Please run: pip install optuna")
        
    logger.info(f"=== Starting Optuna Skeletonization Optimization (Max {n_trials} trials, Patience {patience}) ===")
    
    # Use Tree-structured Parzen Estimator (TPE)
    study = optuna.create_study(direction="minimize")
    objective = SkeletonObjective(pipeline_eval_fn)
    early_stopper = EarlyStoppingCallback(patience=patience)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[early_stopper]) # Sequential to avoid IO collisions
    
    best_params = study.best_params
    logger.info(f"=== Optimization Complete ===")
    logger.info(f"Best Loss: {study.best_value:.4f}")
    logger.info(f"Best Parameters: {best_params}")
    
    # Save the parameters to a YAML file for future reuse
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "best_skeleton_params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"SkeletonConfig": best_params}, f)
    logger.info(f"Saved optimal parameters to: {yaml_path}")
    
    # Generate Visualizations
    try:
        import optuna.visualization as vis
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(str(output_dir / "optuna_history.html"))
        
        fig_params = vis.plot_param_importances(study)
        fig_params.write_html(str(output_dir / "optuna_param_importances.html"))
        
        logger.info(f"Saved interactive optimization plots to {output_dir}/")
    except Exception as e:
        logger.warning(f"Could not generate Optuna plots (is Plotly installed?): {e}")
        
    return best_params

class PreprocessingObjective:
    """Optuna objective function for tuning 3D Voxel Preprocessing parameters."""
    
    def __init__(self, pipeline_eval_fn: Callable):
        self.pipeline_eval_fn = pipeline_eval_fn

    def __call__(self, trial):
        # 1. Define the Bayesian search space (TPE limits)
        pre_kwargs = {
            "hysteresis_threshold_low": trial.suggest_float("hysteresis_threshold_low", 0.0, 0.2),
            "hysteresis_threshold_high": trial.suggest_float("hysteresis_threshold_high", 0.2, 0.4),
            # "median_filter_size": trial.suggest_categorical("median_filter_size", [0, 3, 5, 7, 9]),
            "median_filter_size": 9,
            # "morphological_opening_radius": trial.suggest_int("morphological_opening_radius", 0, 1),
            # "morphological_closing_radius": trial.suggest_int("morphological_closing_radius", 0, 1),
            "morphological_opening_radius": 4,
            "morphological_closing_radius": 0,
            "probability_smoothing_sigma": trial.suggest_float("probability_smoothing_sigma", 0.0, 1.0),
            # "probability_smoothing_sigma": 0,
            "shannon_entropy_threshold": trial.suggest_float("shannon_entropy_threshold", 0.95, 0.99)
        }
        
        # Enforce physical constraints: High threshold must be > Low threshold
        if pre_kwargs["hysteresis_threshold_high"] <= pre_kwargs["hysteresis_threshold_low"]:
            raise optuna.TrialPruned()
        
        # 2. Evaluate pipeline (applies filters and runs benchmarks)
        try:
            bench_results = self.pipeline_eval_fn(pre_kwargs)
        except Exception as e:
            logger.debug(f"Trial pruned due to preprocessing failure: {e}")
            raise optuna.TrialPruned()
            
        if bench_results is None:
            raise optuna.TrialPruned()
            
        # 3. Calculate Loss Function (Minimize towards 0.0)
        confidence = bench_results.get("confidence", 0.0)
        prob_yield = bench_results.get("probability_yield", 0.0)
        crispness = bench_results.get("crispness", 0.0)
        fragmentation = bench_results.get("fragmentation", 1000)
        surface_ratio = bench_results.get("surface_area_ratio", 1.0)
        euler_char = bench_results.get("euler_characteristic", 1)

        # Base penalty: Missing Confidence
        loss = (1.0 - confidence) * 100.0

        # Penalty: Yield. We want to preserve at least some baseline probability mass.
        # If yield drops below 5% of the total probability mass, penalize it massively to stop 1-voxel cheats.
        if prob_yield < 0.05:
            loss += (0.05 - prob_yield) * 10000.0

        # Penalty: Crispness (Inverted and scaled, we want higher gradients at the boundaries)
        # Usually gradient magnitude is < 1.0 depending on data scale. Let's subtract crispness.
        loss += max(0.0, 1.0 - crispness) * 50.0

        # Penalty: Fragmentation / Dust Score
        loss += (fragmentation - 1) * 5.0 

        # Penalty: Compactness (Jagged surfaces). Normal values ~0.2 to 0.4.
        loss += surface_ratio * 10.0

        # Penalty: Swiss Cheese (Negative Euler characteristic)
        if euler_char < 1:
            loss += abs(euler_char - 1) * 10.0

        return loss

def run_optuna_preprocessing_optimization(
    pipeline_eval_fn: Callable,
    n_trials: int = 30,
    output_dir: Path = Path("outputs"),
    patience: int = 100
) -> Dict[str, Any]:
    """
    Executes the Bayesian optimization loop to find the best preprocessing parameters.
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Please run: pip install optuna")
        
    logger.info(f"=== Starting Optuna Preprocessing Optimization (Max {n_trials} trials, Patience {patience}) ===")
    
    study = optuna.create_study(direction="minimize")
    objective = PreprocessingObjective(pipeline_eval_fn)
    early_stopper = EarlyStoppingCallback(patience=patience)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[early_stopper])
    
    best_params = study.best_params
    logger.info(f"=== Preprocessing Optimization Complete ===")
    logger.info(f"Best Loss: {study.best_value:.4f}")
    logger.info(f"Best Parameters: {best_params}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "best_preprocessing_params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"PreprocessingConfig": best_params}, f)
    logger.info(f"Saved optimal preprocessing parameters to: {yaml_path}")
    
    try:
        import optuna.visualization as vis
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(str(output_dir / "optuna_preprocessing_history.html"))
        
        fig_params = vis.plot_param_importances(study)
        fig_params.write_html(str(output_dir / "optuna_preprocessing_param_importances.html"))
    except Exception as e:
        logger.warning(f"Could not generate Optuna plots: {e}")
        
    return best_params