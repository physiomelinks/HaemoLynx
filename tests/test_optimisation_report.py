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
