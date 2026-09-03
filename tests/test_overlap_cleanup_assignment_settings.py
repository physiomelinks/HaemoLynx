"""Tests for assignment-time overlap-cleanup / fast-mode schema knobs."""
from __future__ import annotations

from haemolynx.pipeline import default_schema


def test_overlap_cleanup_schema_defaults_and_requires():
    schema = default_schema()

    large_keys = [
        "automated_vessel_assignment_fast_mode",
        "automated_vessel_assignment_enable_overlap_cleanup",
        "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode",
        "automated_vessel_overlap_parallel_workers",
    ]
    for name in large_keys:
        assert schema[name].requires == (
            "use_large_vessel_masks",
            "automated_vessel_assignment",
        ), name

    small_keys = [
        "small_vessel_boundary_assignment_fast_mode",
        "small_vessel_boundary_assignment_enable_overlap_cleanup",
        "small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode",
        "small_vessel_overlap_parallel_workers",
    ]
    for name in small_keys:
        assert schema[name].requires == (
            "use_small_vessel_masks_for_boundary_assignment",
        ), name

    assert schema["automated_vessel_assignment_fast_mode"].default is True
    assert schema["automated_vessel_assignment_enable_overlap_cleanup"].default is True
    assert (
        schema["automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode"].default
        is False
    )
    assert schema["automated_vessel_overlap_parallel_workers"].default == 8
    assert schema["small_vessel_boundary_assignment_fast_mode"].default is True
    assert schema["small_vessel_overlap_parallel_workers"].default == 8
    assert "load time" in schema["exclude_smaller_overlapping_volumes"].help.lower()
