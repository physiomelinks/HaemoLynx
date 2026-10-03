"""Unit tests for haemolynx.optimisation.report -- pure, synthetic data only."""
from __future__ import annotations

from datetime import datetime

from haemolynx.optimisation.report import build_report_text, config_filename
from haemolynx.optimisation.search import OptimisationResult, TrialRecord


def test_config_filename_format():
    stamp = datetime(2026, 9, 10, 14, 30, 22)
    assert config_filename("C:/data/mouse_cortex.tif", now=stamp) == "20260910_143022_mouse_cortex.yaml"


def test_config_filename_strips_extension_and_directory():
    stamp = datetime(2026, 1, 2, 3, 4, 5)
    assert config_filename("/some/dir/sample.h5", now=stamp) == "20260102_030405_sample.yaml"


def test_config_filename_defaults_to_now():
    name = config_filename("mask.tif")
    assert name.endswith("_mask.yaml")
    assert len(name) == len("20260910_143022_mask.yaml")


def _sample_result() -> OptimisationResult:
    trials = (
        TrialRecord(group="closing_radius", setting="skeleton_closing_radius", value=0, score=-1.0),
        TrialRecord(group="closing_radius", setting="skeleton_closing_radius", value=2, score=-3.0),
        TrialRecord(group="closing_radius", setting="skeleton_closing_radius", value=5, score=1000.0, note="failed: boom"),
        TrialRecord(group="min_branch_length", setting="skeleton_min_branch_length", value=3, score=-0.5),
    )
    settings = {"skeleton_closing_radius": 2, "skeleton_min_branch_length": 3}
    return OptimisationResult(settings=settings, trials=trials)


def test_build_report_text_summarises_each_group():
    text = build_report_text(_sample_result())
    assert "closing_radius: tried 3 candidate(s), 1 failed -> skeleton_closing_radius=2" in text
    assert "min_branch_length: tried 1 candidate(s) -> skeleton_min_branch_length=3" in text


def test_build_report_text_on_empty_result():
    result = OptimisationResult(settings={}, trials=())
    text = build_report_text(result)
    assert text == "HaemoLynx settings, optimised from the segmented input image:"


def test_build_report_text_mentions_downsampling_when_used():
    result = OptimisationResult(settings={}, trials=(), downsample_factor=4)
    text = build_report_text(result)
    assert "4x downsampled" in text


def test_build_report_text_omits_downsampling_note_at_factor_1():
    result = OptimisationResult(settings={}, trials=(), downsample_factor=1)
    text = build_report_text(result)
    assert "downsampled" not in text


def test_build_report_text_lists_skipped_groups():
    result = OptimisationResult(
        settings={}, trials=(), groups_run=("closing_radius", "min_branch_length"),
    )
    text = build_report_text(result)
    skipped_line = next(line for line in text.splitlines() if line.startswith("Not optimised"))
    assert "closing_radius" not in skipped_line
    assert "min_branch_length" not in skipped_line
    assert "thick_vessel_gating" in skipped_line


def test_build_report_text_omits_skipped_note_when_everything_ran():
    result = OptimisationResult(settings={}, trials=())  # default groups_run = every group
    text = build_report_text(result)
    assert "Not optimised" not in text


def test_build_report_text_flags_input_data_guard_rejections():
    """A trial that ran fine (no exception) but scored at the guard-penalty
    level -- see `search._Search._regression_penalty` -- must be reported
    distinctly from an outright failure."""
    trials = (
        TrialRecord(group="bundle_refinement", setting="skeleton_bundle_scan_size", value=5, score=-2.0),
        TrialRecord(
            group="bundle_refinement", setting="skeleton_bundle_scan_size", value=9, score=999.0,
            note="",
        ),
    )
    result = OptimisationResult(
        settings={"skeleton_bundle_scan_size": 5}, trials=trials,
    )
    text = build_report_text(result)
    assert "1 rejected by an input-data guard" in text


def test_build_report_text_does_not_double_count_a_failed_trial_as_guarded():
    """A trial scoring at guard-penalty level *because it raised* is already
    counted as "failed" -- it must not also be counted as guard-rejected."""
    trials = (
        TrialRecord(
            group="closing_radius", setting="skeleton_closing_radius", value=5, score=1000.0,
            note="failed: boom",
        ),
    )
    result = OptimisationResult(settings={"skeleton_closing_radius": 0}, trials=trials)
    text = build_report_text(result)
    assert "1 failed" in text
    assert "rejected by an input-data guard" not in text


def _closing_sweep(*, chosen_value: int) -> tuple[TrialRecord, ...]:
    """One closing-radius sweep starting from 0, with what each candidate measured."""
    measured = {0: 1.0, 2: 3.0}
    return tuple(
        TrialRecord(
            group="closing_radius", setting="skeleton_closing_radius", value=value,
            score=-measured[value], seconds=1.5,
            metrics={"components_merged": measured[value], "guard_penalty": 0.0},
            sweep=0, incumbent=value == 0, chosen=value == chosen_value,
        )
        for value in (0, 2)
    )


def test_the_report_says_what_each_moved_setting_moved_for():
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 2}, trials=_closing_sweep(chosen_value=2),
    )
    text = build_report_text(result)
    assert (
        "    skeleton_closing_radius: 0 -> 2 (score -1 -> -3, components_merged 1 -> 3)"
        in text.splitlines()
    )


def test_the_report_lists_no_change_for_a_sweep_that_kept_its_value():
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 0}, trials=_closing_sweep(chosen_value=0),
    )
    assert "->" not in build_report_text(result).split("skeleton_closing_radius=0")[1]


def test_the_report_names_a_choice_whose_starting_value_was_not_a_candidate():
    trials = (
        TrialRecord(
            group="min_stub_length", setting="min_stub_length", value=4.0, score=-2.0,
            metrics={"nodes_removed": 6.0}, sweep=3, chosen=True,
        ),
    )
    result = OptimisationResult(settings={"min_stub_length": 4.0}, trials=trials)
    assert "    min_stub_length: set to 4.0 (score -2, nodes_removed 6)" in build_report_text(result)


def test_the_report_gives_the_search_time_beside_autos_estimate():
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 2}, trials=_closing_sweep(chosen_value=2),
        seconds=125.0, estimated_seconds=60.0, group_seconds={"closing_radius": 12.0},
    )
    lines = build_report_text(result).splitlines()
    assert lines[1] == "Search took 2 min 05 s (Auto estimated 1 min 00 s)."
    group_line = next(line for line in lines if line.startswith("- closing_radius"))
    assert group_line.endswith("(took 12 s)")


def test_the_report_times_a_group_from_its_trials_without_a_group_time():
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 2}, trials=_closing_sweep(chosen_value=2), seconds=4.0,
    )
    lines = build_report_text(result).splitlines()
    assert lines[1] == "Search took 4.0 s."
    assert next(line for line in lines if line.startswith("- closing_radius")).endswith("(took 3.0 s)")


def test_the_report_names_what_a_joint_sweep_chose():
    """Regression: centreline smoothing tries its method, iterations and
    deviation as one value, under a name no setting has, so its line said
    '-> ' and nothing else."""
    combo = ("taubin", 10, 1.0)
    trials = (
        TrialRecord(
            group="centreline_smoothing", setting="centreline_smoothing", value=combo, score=-0.9,
            sweep=9, incumbent=True, chosen=True,
        ),
    )
    result = OptimisationResult(settings={"centreline_smoothing_method": "taubin"}, trials=trials)
    assert "-> centreline_smoothing=('taubin', 10, 1.0)" in build_report_text(result)


def test_the_report_says_what_auto_measured_and_how_coarse_it_allowed():
    result = OptimisationResult(
        settings={}, trials=(), downsample_factor=2, typical_radius_um=3.671, resolution_cap=2,
    )
    assert (
        "Auto measured a typical vessel radius of 3.67 um, so searched no coarser than 2x, "
        "which keeps such a vessel at least 3 search-grid voxels across."
    ) in build_report_text(result).splitlines()


def test_the_report_lists_settings_finer_than_the_search_grid():
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 2, "skeleton_min_branch_length": 3},
        trials=(),
        downsample_factor=16,
        finer_than_search_grid=("skeleton_closing_radius", "skeleton_min_branch_length"),
    )
    assert (
        "Under one voxel of the 16x search grid, so left as you had them: "
        "skeleton_closing_radius=2, skeleton_min_branch_length=3"
    ) in build_report_text(result).splitlines()


def test_a_voxel_settings_change_on_a_downsampled_search_is_in_search_grid_voxels():
    """The trials hold the search grid's values; the settings above them are
    back in full-resolution voxels, so the change line says which it is."""
    result = OptimisationResult(
        settings={"skeleton_closing_radius": 4},
        trials=_closing_sweep(chosen_value=2),
        downsample_factor=2,
    )
    assert "    skeleton_closing_radius: 0 -> 2 [search-grid voxels] (" in build_report_text(result)


def test_the_report_heading_can_be_given():
    result = OptimisationResult(settings={}, trials=())
    assert build_report_text(result, heading="FWHM settings:") == "FWHM settings:"


def test_format_seconds():
    from haemolynx.optimisation.report import format_seconds

    assert format_seconds(4.24) == "4.2 s"
    assert format_seconds(38.4) == "38 s"
    assert format_seconds(185) == "3 min 05 s"
    assert format_seconds(3720) == "1 h 02 min"
