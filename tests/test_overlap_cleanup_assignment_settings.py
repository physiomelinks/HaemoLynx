"""Tests for assignment-time overlap-cleanup / fast-mode schema knobs."""
from __future__ import annotations

from haemolynx.pipeline import default_schema


def test_overlap_cleanup_schema_defaults_and_requires():
    schema = default_schema()

    assert schema["automated_vessel_assignment_enable_overlap_cleanup"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
    )
    assert schema["automated_vessel_assignment_fast_mode"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
        "automated_vessel_assignment_enable_overlap_cleanup",
    )
    assert schema[
        "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode"
    ].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
        "automated_vessel_assignment_enable_overlap_cleanup",
        "!automated_vessel_assignment_fast_mode",
    )
    assert schema["automated_vessel_overlap_parallel_workers"].requires == (
        "use_large_vessel_masks",
        "automated_vessel_assignment",
        "automated_vessel_assignment_enable_overlap_cleanup",
    )

    assert schema[
        "small_vessel_boundary_assignment_enable_overlap_cleanup"
    ].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
    )
    assert schema["small_vessel_boundary_assignment_fast_mode"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
        "small_vessel_boundary_assignment_enable_overlap_cleanup",
    )
    assert schema[
        "small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode"
    ].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
        "small_vessel_boundary_assignment_enable_overlap_cleanup",
        "!small_vessel_boundary_assignment_fast_mode",
    )
    assert schema["small_vessel_overlap_parallel_workers"].requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
        "small_vessel_boundary_assignment_enable_overlap_cleanup",
    )

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
