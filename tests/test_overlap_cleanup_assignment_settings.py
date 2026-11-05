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

    assert schema["automated_vessel_assignment_fast_mode"].default is True
    assert schema["automated_vessel_assignment_enable_overlap_cleanup"].default is True
    assert (
        schema["automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode"].default
        is False
    )
    assert schema["small_vessel_boundary_assignment_fast_mode"].default is True
    assert "load time" in schema["exclude_smaller_overlapping_volumes"].help.lower()


def test_the_worker_counts_are_retired_and_old_configs_still_load():
    """Both counts were accepted and never read: the overlap tie-break and
    the edge classification have always run sequentially."""
    schema = default_schema()
    assert "automated_vessel_overlap_parallel_workers" not in schema.names
    assert "small_vessel_overlap_parallel_workers" not in schema.names
    resolved = schema.validate(
        {"automated_vessel_overlap_parallel_workers": 8, "small_vessel_overlap_parallel_workers": 4}
    )
    assert "automated_vessel_overlap_parallel_workers" not in resolved
    assert resolved["automated_vessel_assignment_enable_overlap_cleanup"] is True
