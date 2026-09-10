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
