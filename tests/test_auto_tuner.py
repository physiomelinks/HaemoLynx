import pytest
import yaml
from pathlib import Path
import optuna
from optuna.trial import FixedTrial
from ImageLynx.statistics.auto_tuner import (
    DEFAULT_SAMPLER_SEED,
    PreprocessingObjective,
    SkeletonObjective,
    run_optuna_preprocessing_optimization,
    run_optuna_skeleton_optimization,
)

def _get_dummy_fixed_trial():
    """Helper to generate a consistent FixedTrial for objective logic testing.

    Values must lie inside the declared distributions. min_component_percent was 1.0 against
    a range of [4.0, 6.0] and bundle_scan_size was 5 against [8, 10], which made Optuna emit
    a UserWarning on every run of three tests. smoothing_alpha and prune_by_tortuosity are
    gone with the search dimensions themselves (#98 item 17).
    """
    return FixedTrial({
        "min_branch_length": 10,
        "max_bridge_distance": 5,
        "min_component_percent": 5.0,
        "bundle_scan_size": 9,
        "bundle_density_fraction": 0.05,
        "bundle_max_connections": 3,
        "bundle_hub_min_spacing": 2,
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

    Second contract change (#98, Tier 2 item 15): 630.0 became 130.0. The 500.0 dead-end
    cliff, the 20.0 degree-3 term and the 0.05 edge-length-variance term are all gone, which
    removes exactly 475.0 + 20.0 + 5.0 from this fixture. The residual 130.0 is the two
    surviving fidelity terms and nothing else. This is a deliberate contract removal, not a
    regression: the expectations this test encoded - that a 100% dead-end network and a 0%
    Y-bifurcation network are penalised - are precisely the expectations that were biasing
    the tuner toward deleting real capillaries.
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
    
    # Expected Loss = 50 (dice) + 80 (orphaned). The terminal, deg3 and edge_var terms that
    # used to add 475 + 20 + 5 have been removed.
    assert loss == 130.0, f"Expected 130.0 loss, got {loss}"

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


# --- Sampler seeding (item 18) -------------------------------------------------------------
#
# Both studies previously ran on Optuna's default unseeded TPESampler, so two runs explored
# different trial sequences. That makes any change to the objective's weights inseparable
# from sampler noise: a different best_params could be the re-weighting or could be the draw.
# These tests pin the trial sequence to the seed so weight changes are attributable.

def _recording_skeleton_eval(seen):
    """Deterministic mock eval that records the parameter vector proposed on each trial."""
    def _eval(kwargs):
        seen.append((kwargs["min_branch_length"], kwargs["bundle_density_fraction"]))
        return _skeleton_bench(
            dice=0.9, orphaned=0.05, terminal_ratio=0.02, loops=5,
            edge_std=float(kwargs["min_branch_length"]),
        )
    return _eval


def _recording_preprocessing_eval(seen):
    """Deterministic mock eval for the preprocessing study, recording each proposal."""
    def _eval(kwargs):
        seen.append((kwargs["hysteresis_threshold_low"], kwargs["shannon_entropy_threshold"]))
        return {
            "confidence": 0.8,
            "probability_yield": 0.3,
            "crispness": 0.5,
            "fragmentation": 10,
            "surface_area_ratio": kwargs["hysteresis_threshold_low"],
            "euler_characteristic": 1,
            "mean_uncertainty": 0.1,
            "high_uncertainty_fraction": 0.01,
        }
    return _eval


def test_skeleton_sampler_repeats_the_same_trial_sequence_for_one_seed(tmp_path: Path):
    first, second = [], []
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(first), n_trials=12, output_dir=tmp_path / "a", seed=7)
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(second), n_trials=12, output_dir=tmp_path / "b", seed=7)

    assert len(first) == 12
    assert first == second, "Same seed produced a different trial sequence."


def test_skeleton_sampler_explores_differently_for_a_different_seed(tmp_path: Path):
    """Guards against the seed being accepted but ignored, which would pass the test above."""
    first, second = [], []
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(first), n_trials=12, output_dir=tmp_path / "a", seed=7)
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(second), n_trials=12, output_dir=tmp_path / "b", seed=8)

    assert first != second, "Two different seeds produced an identical trial sequence."


def test_preprocessing_sampler_repeats_the_same_trial_sequence_for_one_seed(tmp_path: Path):
    first, second = [], []
    run_optuna_preprocessing_optimization(
        _recording_preprocessing_eval(first), n_trials=12, output_dir=tmp_path / "a", seed=7)
    run_optuna_preprocessing_optimization(
        _recording_preprocessing_eval(second), n_trials=12, output_dir=tmp_path / "b", seed=7)

    assert len(first) == 12
    assert first == second, "Same seed produced a different trial sequence."


def test_preprocessing_sampler_explores_differently_for_a_different_seed(tmp_path: Path):
    first, second = [], []
    run_optuna_preprocessing_optimization(
        _recording_preprocessing_eval(first), n_trials=12, output_dir=tmp_path / "a", seed=7)
    run_optuna_preprocessing_optimization(
        _recording_preprocessing_eval(second), n_trials=12, output_dir=tmp_path / "b", seed=8)

    assert first != second, "Two different seeds produced an identical trial sequence."


def test_sampler_is_seeded_by_default(tmp_path: Path):
    """The default must be reproducible, not Optuna's unseeded default."""
    assert DEFAULT_SAMPLER_SEED is not None

    first, second = [], []
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(first), n_trials=12, output_dir=tmp_path / "a")
    run_optuna_skeleton_optimization(
        _recording_skeleton_eval(second), n_trials=12, output_dir=tmp_path / "b")

    assert first == second, "Default run was not reproducible; the sampler is unseeded."


@pytest.mark.parametrize("runner,block", [
    (run_optuna_skeleton_optimization, "SkeletonConfig"),
    (run_optuna_preprocessing_optimization, "PreprocessingConfig"),
])
def test_tuning_provenance_records_the_seed_beside_the_parameters(tmp_path, runner, block):
    """A frozen parameter set is only defensible if the run that produced it can be repeated."""
    evals = {"SkeletonConfig": _recording_skeleton_eval, "PreprocessingConfig": _recording_preprocessing_eval}
    runner(evals[block]([]), n_trials=4, output_dir=tmp_path, seed=7)

    written = list(tmp_path.glob("best_*_params.yaml"))
    assert len(written) == 1
    saved = yaml.safe_load(written[0].read_text())

    # The config block the pipeline loader reads must be unchanged in shape.
    assert block in saved and isinstance(saved[block], dict)

    provenance = saved["TuningProvenance"]
    assert provenance["seed"] == 7
    assert provenance["sampler"] == "TPESampler"
    assert provenance["n_trials_requested"] == 4
    assert provenance["n_trials_run"] == 4


# --- Preprocessing objective reduced to its measurable terms (item 15) ---------------------
#
# Measured over 25 seeded TPE trials on the WKY subvolume z 60:110, y 120:280, x 120:280:
# confidence 89.5% of loss, fragmentation 7.4%, surface_area_ratio 3.1%, and crispness, the
# yield cliff and both uncertainty terms exactly 0.0% on every single trial.

def _preproc_bench(**overrides):
    base = {
        "confidence": 0.8, "probability_yield": 0.3, "crispness": 0.5,
        "fragmentation": 10, "surface_area_ratio": 0.3, "euler_characteristic": 1,
        "mean_uncertainty": 0.1, "high_uncertainty_fraction": 0.01,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("key,low,high", [
    # Never computed post-b89104c: the 2-class gate leaves entropy_map None, so
    # run_all_preprocessing_benchmarks emits neither. A weight of 1000 that switches itself
    # back on when the classifier gains a third class is a trap, not a dormant feature.
    ("mean_uncertainty", 0.0, 1.0),
    ("high_uncertainty_fraction", 0.0, 1.0),
    # Mis-scaled: the term assumed crispness <= 1, but it is a mean Sobel gradient magnitude
    # and measured 2.84-3.23, so it clamped to zero on every trial.
    ("crispness", 0.1, 5.0),
    # Redundant with prune_mask_before, which keeps only the largest component anyway, and
    # otherwise pressure to merge genuinely separate vessels.
    ("fragmentation", 1, 5000),
    # surface/volume is ~2/r: thin resolved capillaries score HIGH and a fat merged blob
    # scores LOW, so the term rewarded exactly the degradation to be avoided.
    ("surface_area_ratio", 0.05, 2.0),
])
def test_preprocessing_objective_loss_ignores_removed_term(key, low, high):
    objective = PreprocessingObjective(lambda kwargs: None)
    assert objective._calculate_loss(_preproc_bench(**{key: low})) == \
           objective._calculate_loss(_preproc_bench(**{key: high}))


def test_preprocessing_objective_is_exactly_confidence_above_the_yield_cliff():
    """The two surviving terms, and nothing else. Contract change from #98 item 15."""
    objective = PreprocessingObjective(lambda kwargs: None)
    # yield 0.3 is far above the 0.05 cliff, so the cliff contributes nothing.
    assert objective._calculate_loss(_preproc_bench(confidence=0.8)) == pytest.approx(20.0)
    assert objective._calculate_loss(_preproc_bench(confidence=0.25)) == pytest.approx(75.0)


def test_preprocessing_objective_yield_cliff_still_fires():
    """The only thing opposing confidence's shrink-the-mask gradient must survive."""
    objective = PreprocessingObjective(lambda kwargs: None)
    above = objective._calculate_loss(_preproc_bench(confidence=0.9, probability_yield=0.06))
    below = objective._calculate_loss(_preproc_bench(confidence=0.9, probability_yield=0.04))
    assert below > above
    assert below - above == pytest.approx(0.01 * 10000.0)


def test_preprocessing_objective_cannot_be_improved_by_merging_vessels():
    """Regression for the whole a079048/item-15 family.

    A mask whose vessels have been fattened and merged scores better on component count,
    Euler characteristic and surface-to-volume ratio all at once. None of those may now move
    the loss, so degrading the biology can no longer buy a better score.
    """
    objective = PreprocessingObjective(lambda kwargs: None)
    resolved = _preproc_bench(fragmentation=40, euler_characteristic=-900, surface_area_ratio=0.8)
    merged_blob = _preproc_bench(fragmentation=1, euler_characteristic=1, surface_area_ratio=0.05)
    assert objective._calculate_loss(resolved) == objective._calculate_loss(merged_blob)


# --- Skeleton objective reduced to its fidelity terms (item 15) ----------------------------
#
# Measured over 12 seeded TPE trials on the WKY subvolume z 60:110, y 120:280, x 120:280 at
# the calibrated voxel size: dice 42.2% of loss, terminal cliff 37.8%, orphaned 19.3%,
# edge_length_std 0.6%, degree3 0.2%.

def test_skeleton_objective_loss_ignores_dead_end_ratio():
    """The dead-end cliff's only mechanism of improvement was deleting terminal branches.

    Measured at 37.8% of loss and active on every trial. It outweighed the orphaned term - the
    one term guarding against over-pruning - by 5 to 1, so the net pressure was to delete
    capillaries. In this data most terminals are real: capillaries end, and a subvolume cuts
    vessels at its faces.
    """
    objective = SkeletonObjective(lambda kwargs: None)
    few_dead_ends = _skeleton_bench(0.6, 0.01, terminal_ratio=0.02, loops=100)
    many_dead_ends = _skeleton_bench(0.6, 0.01, terminal_ratio=0.90, loops=100)

    assert objective._calculate_loss(few_dead_ends) == objective._calculate_loss(many_dead_ends)


def test_skeleton_objective_loss_ignores_bifurcation_degree_mix():
    """deg3_ratio measured 0.960-1.000: saturated, nothing to trade against.

    It also pays for simplifying real topology, since an anastomosing capillary bed has
    genuine degree-4 crossings.
    """
    objective = SkeletonObjective(lambda kwargs: None)
    all_y = _skeleton_bench(0.6, 0.01, 0.03, loops=100, deg3=1.0)
    all_x = _skeleton_bench(0.6, 0.01, 0.03, loops=100, deg3=0.0)

    assert objective._calculate_loss(all_y) == objective._calculate_loss(all_x)


def test_skeleton_objective_loss_ignores_edge_length_variance():
    """A real vascular tree has a wide natural spread of segment lengths.

    The weight was also silently re-denominated from voxels to microns by 2705b38.
    """
    objective = SkeletonObjective(lambda kwargs: None)
    uniform = _skeleton_bench(0.6, 0.01, 0.03, loops=100, edge_std=0.0)
    varied = _skeleton_bench(0.6, 0.01, 0.03, loops=100, edge_std=500.0)

    assert objective._calculate_loss(uniform) == objective._calculate_loss(varied)


def test_skeleton_objective_is_exactly_the_two_fidelity_terms():
    objective = SkeletonObjective(lambda kwargs: None)
    bench = _skeleton_bench(dice=0.75, orphaned=0.10, terminal_ratio=0.40,
                            loops=900, deg3=0.2, edge_std=42.0)
    assert objective._calculate_loss(bench) == pytest.approx(25.0 + 10.0)


def test_skeleton_objective_cannot_be_improved_by_pruning_real_branches():
    """Regression for the whole a079048/item-15 family, on the skeleton side.

    An aggressively pruned graph scores better on dead-end ratio, loop count, bifurcation mix
    and edge-length uniformity all at once, while losing coverage. Only the coverage may now
    move the loss, so the pruned graph must score strictly worse.
    """
    objective = SkeletonObjective(lambda kwargs: None)
    faithful = _skeleton_bench(dice=0.60, orphaned=0.02, terminal_ratio=0.35,
                               loops=1200, deg3=0.90, edge_std=30.0)
    over_pruned = _skeleton_bench(dice=0.40, orphaned=0.30, terminal_ratio=0.02,
                                  loops=5, deg3=1.00, edge_std=1.0)

    assert objective._calculate_loss(faithful) < objective._calculate_loss(over_pruned)


# --- Dead search dimensions removed (item 17) ---------------------------------------------

def _suggested_dimensions(objective):
    """Names the objective actually asks the sampler for, recorded from a real trial."""
    seen = []

    class _Recorder:
        def suggest_int(self, name, *a, **k):
            seen.append(name)
            return a[0]

        def suggest_float(self, name, *a, **k):
            seen.append(name)
            return a[0]

        def suggest_categorical(self, name, choices):
            seen.append(name)
            return choices[0]

    try:
        objective(_Recorder())
    except optuna.TrialPruned:
        pass
    return seen


@pytest.mark.parametrize("dimension", ["smoothing_alpha", "prune_by_tortuosity"])
def test_dead_skeleton_search_dimensions_are_no_longer_suggested(dimension):
    """Both were suggested every trial, saved to the YAML, and read by nothing.

    prune_by_tortuosity has no implementation at all. smoothing_alpha is now wired to
    bspline_smoothness but must stay frozen: it sets the curvature H1 section 1.4 measures
    tortuosity from, and this objective is Dice plus orphaned volume, neither of which can
    see it.
    """
    assert dimension not in _suggested_dimensions(SkeletonObjective(lambda kwargs: None))


def test_skeleton_search_dimensions_are_exactly_the_live_ones():
    live = _suggested_dimensions(SkeletonObjective(lambda kwargs: None))
    assert live == [
        "min_branch_length",
        "min_component_percent",
        "bundle_scan_size",
        "bundle_density_fraction",
    ]


def test_smoothing_alpha_reaches_the_centreline_smoother():
    """It was a dead config field beside a hardcoded 0.75 at the call site."""
    C = pytest.importorskip("carotid_image_to_model")
    import inspect

    source = inspect.getsource(C._build_and_optimize_graph)
    assert "bspline_smoothness=skel_config.smoothing_alpha" in source
    # Behaviour-neutral at the defaults: the field carries the value that was hardcoded.
    assert C.SkeletonConfig().smoothing_alpha == 0.75


def test_prune_by_tortuosity_is_gone_from_the_config():
    """No implementation existed, so a YAML setting it should warn rather than look accepted."""
    C = pytest.importorskip("carotid_image_to_model")
    assert not hasattr(C.SkeletonConfig(), "prune_by_tortuosity")


def test_consolidation_threshold_is_gone_from_the_topology_optimiser():
    """Declared in the signature and consumed nowhere, in this or the legacy implementation."""
    import inspect
    from ImageLynx.graph.optimise import optimise_graph_topology_fixed

    assert "consolidation_threshold" not in inspect.signature(optimise_graph_topology_fixed).parameters
