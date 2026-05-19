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
            "min_branch_length": trial.suggest_int("min_branch_length", 0, 50),
            "max_bridge_distance": trial.suggest_int("max_bridge_distance", 0, 20),
            "min_component_percent": trial.suggest_float("min_component_percent", 0.01, 5.0),
            "bundle_scan_size": trial.suggest_int("bundle_scan_size", 3, 15),
            "bundle_density_fraction": trial.suggest_float("bundle_density_fraction", 0.1, 0.9),
            "bundle_max_connections": trial.suggest_int("bundle_max_connections", 2, 8)
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
        
        # Base penalty: Missing volume (DSC difference from 1.0, heavily weighted)
        loss = (1.0 - dsc) * 100.0
        
        # Penalty: Over-pruning (orphaned tissue fraction, heavily weighted to preserve capillary beds)
        loss += orphaned * 100.0
        
        # Penalty: Spiderwebs and messy topology (soft penalty per extra loop)
        loss += loops * 0.1 
        
        return loss

class EarlyStoppingCallback:
    """Optuna callback to stop optimization if the score hasn't improved after N trials."""
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_score = None
        self.stagnant_trials = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
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
    patience: int = 15
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