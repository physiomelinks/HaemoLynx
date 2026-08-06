import logging
import yaml
import copy
from pathlib import Path
from typing import Callable, Dict, Any, Optional

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)

# Default random state for both TPE samplers. Fixed rather than left at Optuna's default of
# None so that a tuning run is reproducible, and so that two runs which differ only in the
# objective's weights are comparable trial-for-trial rather than confounded with sampler
# noise. Pass seed=None to draw an independent search trajectory.
DEFAULT_SAMPLER_SEED = 42

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
            "bundle_max_connections": 3,
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
        return self._calculate_loss(bench_results)

    def _calculate_loss(self, bench_results) -> float:
        """Loss from a benchmark result dict. Separated so it is testable without Optuna.

        Reduced to the two fidelity terms. Decomposition over 12 seeded TPE trials on the WKY
        subvolume z 60:110, y 120:280, x 120:280, at the calibrated voxel size, post-a079048:

            dice             42.2% of total loss   (raw 0.007 - 0.036)
            terminal cliff   37.8%                 (raw ratio 0.188 - 0.370)
            orphaned         19.3%                 (raw 0.371 - 0.821)
            edge_length_std   0.6%                 (raw 21.9 - 35.5 um)
            degree3           0.2%                 (raw 0.960 - 1.000)

        Every surviving weight enters the item 25 sensitivity scope. Neither surviving weight
        was changed; both were already 100.

        CAVEAT on the two survivors: both are currently dominated by the mask rather than by
        the skeleton. The binary is 87% foreground at the default thresholds, so the redilated
        centreline covers 1.5% of a volume that is 87% "vessel" and Dice is pinned near zero
        whatever the skeleton parameters do, while most mask voxels are necessarily further
        than 20 um from any centreline and count as orphaned. These are the right two terms
        and they are the wrong two numbers until the hysteresis thresholds are fixed. No H1
        figure should be read off this objective before then.
        """
        vol = bench_results.get("volumetric", {})
        comp = bench_results.get("completeness", {})

        dsc = vol.get("dice_coefficient", 0.0)
        orphaned = comp.get("orphaned_volume_fraction", 1.0)

        # Base penalty: Missing volume (DSC difference from 1.0, heavily weighted)
        loss = (1.0 - dsc) * 100.0

        # Penalty: Over-pruning (orphaned tissue fraction, heavily weighted to preserve capillary beds)
        loss += orphaned * 100.0

        # NO loop penalty. This used to be `loss += loops * 0.1`, where `loops` is
        # graph_fundamental_loops = E - V + C, i.e. the first Betti number. That is precisely
        # the quantity H1 section 1.1 reads out as vascular loop topology, so the tuner's
        # dominant objective was to minimise the hypothesis' own signal. Measured, it was
        # 70-86% of total loss and single-handedly overturned better Dice, better
        # completeness and a better terminal ratio. It is also group-dependent: a denser
        # network incurs a proportionally larger penalty and gets pruned harder, suppressing
        # the SHR/WKY difference in the false-negative direction.
        #
        # It is removed rather than downweighted because any term keyed on beta-1 can be
        # minimised by deleting real anastomoses. The artefact it was nominally aimed at -
        # spurious 1-voxel skeleton loops - is already handled upstream by the voxel-loop
        # detection and stitching in build_graph_segment_skan_stitched_loops.

        # NO dead-end penalty. This was `if terminal_ratio > 0.05: (terminal_ratio - 0.05) * 500`,
        # measured at 37.8% of total loss and active on every single trial, since the ratio
        # never fell below 0.188. Its only mechanism of improvement is deleting terminal
        # branches, and in this data most terminals are real: capillaries genuinely end, and a
        # subvolume genuinely cuts vessels at its faces. It therefore fought the orphaned term
        # - the one term that exists to stop over-pruning - and outweighed it 5 to 1, giving a
        # net bias toward deleting capillaries. That is a false-negative bias on H1's own
        # signal, the same direction as the beta-1 penalty removed in a079048.
        #
        # What it was nominally guarding, a graph shattered into disconnected pieces, is
        # already handled: GraphConfig.keep_largest_component_only forces a single connected
        # component, measured at 1 on every trial.

        # NO degree-3 penalty. This was `(1 - deg3_ratio) * 20`, measured at 0.2% of loss with
        # deg3_ratio sitting between 0.960 and 1.000 - saturated, so there was nothing for the
        # tuner to trade against. Directionally it is also wrong: it asks that Y-bifurcations
        # dominate X-bifurcations, but an anastomosing capillary bed has genuine degree-4
        # crossings, so the term pays for simplifying real topology.

        # NO edge-length-variance penalty. This was `edge_length_std * 0.05`, measured at 0.6%
        # of loss. A real vascular tree has a wide natural spread of segment lengths, from
        # short capillary segments to long arterioles, so a pull toward uniform edge length is
        # a pull toward a homogenised network. The 0.05 weight is also now um-denominated
        # rather than voxel-denominated after 2705b38 and was never recalibrated.

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

def _tuning_provenance(study, seed: Optional[int], n_trials: int) -> Dict[str, Any]:
    """Describes how a tuned parameter set was produced, to be saved alongside it.

    Written as a sibling of the config block rather than inside it, because the pipeline's
    YAML loader feeds its named block straight into a dataclass and would reject these keys.
    A frozen parameter set is only defensible if the run that produced it can be repeated,
    and that requires the seed to survive in the artefact rather than only in a log line.
    """
    return {
        "sampler": "TPESampler",
        "seed": seed,
        "n_trials_requested": int(n_trials),
        "n_trials_run": len(study.trials),
        "best_loss": float(study.best_value),
    }

def run_optuna_skeleton_optimization(
    pipeline_eval_fn: Callable,
    n_trials: int = 30,
    output_dir: Path = Path("outputs"),
    patience: int = 100,
    seed: Optional[int] = DEFAULT_SAMPLER_SEED
) -> Dict[str, Any]:
    """
    Executes the Bayesian optimization loop to find the best skeletonization parameters.

    seed fixes the TPE sampler's random state, so the trial sequence is reproducible across
    runs. Note that this makes the *search* deterministic, not the whole tuning run: if
    pipeline_eval_fn is itself stochastic the losses will still vary. Pass seed=None for an
    independent replicate.
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Please run: pip install optuna")

    logger.info(f"=== Starting Optuna Skeletonization Optimization (Max {n_trials} trials, Patience {patience}, Seed {seed}) ===")

    # Use Tree-structured Parzen Estimator (TPE), seeded for a reproducible trial sequence.
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
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
        yaml.dump({
            "SkeletonConfig": best_params,
            "TuningProvenance": _tuning_provenance(study, seed, n_trials),
        }, f)
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
            "hysteresis_threshold_low": trial.suggest_float("hysteresis_threshold_low", 0.25, 0.5),
            "hysteresis_threshold_high": trial.suggest_float("hysteresis_threshold_high", 0.5, 0.75),
            # "median_filter_size": trial.suggest_categorical("median_filter_size", [0, 3, 5, 7, 9]),
            "median_filter_size": 9,
            # "morphological_opening_radius": trial.suggest_int("morphological_opening_radius", 0, 1),
            # "morphological_closing_radius": trial.suggest_int("morphological_closing_radius", 0, 1),
            "morphological_opening_radius": 4,
            "morphological_closing_radius": 0,
            "probability_smoothing_sigma": trial.suggest_float("probability_smoothing_sigma", 0.0, 1.0),
            # "probability_smoothing_sigma": 0,
            "shannon_entropy_threshold": trial.suggest_float("shannon_entropy_threshold", 0.85, 0.99),
            "shannon_entropy_core": trial.suggest_float("shannon_entropy_core", 0.4, 0.8)
        }
        
        # Enforce physical constraints: High threshold must be > Low threshold
        if pre_kwargs["hysteresis_threshold_high"] <= pre_kwargs["hysteresis_threshold_low"]:
            raise optuna.TrialPruned()
            
        # Core entropy must be less than max entropy
        if pre_kwargs["shannon_entropy_core"] >= pre_kwargs["shannon_entropy_threshold"]:
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
        return self._calculate_loss(bench_results)

    def _calculate_loss(self, bench_results) -> float:
        """Loss from a benchmark result dict. Separated so it is testable without Optuna.

        Reduced to the two terms that were measured to do anything. Decomposition over 25
        seeded TPE trials on the WKY subvolume z 60:110, y 120:280, x 120:280, post-a079048:

            confidence          89.5% of total loss (up to 98.2%)
            fragmentation        7.4%               (up to 27.9%)
            surface_area_ratio   3.1%               (up to  5.1%)
            crispness            0.0% on every trial
            yield cliff          0.0% on every trial
            mean_uncertainty     0.0% on every trial
            high_uncertainty     0.0% on every trial

        Each removal is justified at its site below. Every surviving weight enters the item
        25 sensitivity scope.

        KNOWN DEGENERACY, left visible rather than papered over: confidence is the mean
        probability inside the mask, so it decreases monotonically as the mask shrinks onto
        the highest-probability voxels, and the only thing opposing it is a cliff that does
        not engage until the mask has discarded 95% of the probability mass. Between yield
        1.00 and 0.05 this objective has no restoring force at all - its argmin is "the
        smallest mask still holding 5% of the mass". That defect was previously hidden behind
        four terms that contributed exactly zero. No weight was changed here, because
        changing one would mean inventing a constant; the fix is a single bounded
        precision/recall term (e.g. soft Dice between mask and probability field), which
        replaces the objective rather than re-weighting it.
        """
        confidence = bench_results.get("confidence", 0.0)
        prob_yield = bench_results.get("probability_yield", 0.0)

        # Base penalty: Missing Confidence. The dominant term, and the only real gradient.
        loss = (1.0 - confidence) * 100.0

        # Penalty: Yield. We want to preserve at least some baseline probability mass.
        # If yield drops below 5% of the total probability mass, penalize it massively to stop 1-voxel cheats.
        if prob_yield < 0.05:
            loss += (0.05 - prob_yield) * 10000.0

        # NO crispness penalty. This was `max(0, 1 - crispness) * 50`, which assumes crispness
        # is bounded by 1. It is a mean Sobel gradient magnitude on the probability field and
        # is not bounded that way: measured 2.84 to 3.23 across the search space, so the term
        # clamped to zero on every trial. It was not doing a small amount of work; it was
        # doing none, and on differently scaled data it would switch on at a weight nobody chose.

        # NO fragmentation penalty. This was `(fragmentation - 1) * 5`, uncapped. Measured
        # component counts were 1 to 4, so it never approached the magnitude the cap was meant
        # to contain - but it is redundant rather than harmless: SkeletonConfig.prune_mask_before
        # already keeps only the single largest component downstream, so the mask's component
        # count is forced to 1 whatever the tuner picks. What the term does add is pressure to
        # merge genuinely separate vessels, which is the same "improve the score by degrading
        # the biology" failure as the beta-1 and chi terms removed in a079048.

        # NO compactness penalty. This was `surface_area_ratio * 10`, where the ratio is
        # surface voxels over volume voxels, i.e. roughly 2/r for a tube of radius r. A
        # correctly resolved capillary bed of thin vessels has a HIGH ratio and a fat merged
        # blob has a LOW one, so the term rewarded exactly the degradation the segmentation
        # has to avoid, and would penalise the pipeline harder the better it got.

        # NO Euler penalty. This used to be `if euler_char < 1: loss += abs(euler_char-1)*10`,
        # measured at 83% of total loss. The intent was to punish "swiss cheese" cavities, but
        # chi = beta0 - beta1 + beta2 and a capillary bed legitimately has large beta1 and so
        # a strongly negative chi. Penalising chi < 1 therefore drives vascular loop topology
        # toward zero - the same H1 section 1.1 readout the skeleton objective was attacking -
        # and does so proportionally harder on denser networks.
        #
        # Cavities are the beta2 term and are already handled directly by fill_holes_3d in the
        # preprocessing chain, so nothing is lost by dropping this. A cavity-specific penalty
        # would have to measure beta2 on its own rather than inferring it from chi.

        # NO uncertainty penalties. These were `mean_uncertainty * 20` and
        # `high_uncertainty_fraction * 1000`. run_all_preprocessing_benchmarks only computes
        # either when entropy_map is not None, and the b89104c gate leaves it None for a
        # 2-class classifier, so both read their 0.0 defaults on every trial. Keeping them was
        # a trap rather than a no-op: retraining with a third class would switch a weight of
        # 1000 back on silently, and it would immediately dominate. If the TH channel arrives,
        # they have to be re-derived and re-weighted deliberately.

        return loss

def run_optuna_preprocessing_optimization(
    pipeline_eval_fn: Callable,
    n_trials: int = 30,
    output_dir: Path = Path("outputs"),
    patience: int = 100,
    seed: Optional[int] = DEFAULT_SAMPLER_SEED
) -> Dict[str, Any]:
    """
    Executes the Bayesian optimization loop to find the best preprocessing parameters.

    seed fixes the TPE sampler's random state, so the trial sequence is reproducible across
    runs. Note that this makes the *search* deterministic, not the whole tuning run: if
    pipeline_eval_fn is itself stochastic the losses will still vary. Pass seed=None for an
    independent replicate.
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Please run: pip install optuna")

    logger.info(f"=== Starting Optuna Preprocessing Optimization (Max {n_trials} trials, Patience {patience}, Seed {seed}) ===")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
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
        yaml.dump({
            "PreprocessingConfig": best_params,
            "TuningProvenance": _tuning_provenance(study, seed, n_trials),
        }, f)
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