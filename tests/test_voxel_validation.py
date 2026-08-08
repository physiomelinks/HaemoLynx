"""Tests for io.voxel_validation: voxel-size validation and policy resolution.

``resolve_voxel_size_xyz`` decides which voxel size the whole pipeline runs on,
so a mistake here rescales every length, resistance and mask lookup downstream.
Two failure modes matter most and both are silent:

* **Order.** The value is physical ``(x, y, z)``; anything that reorders or sorts
  it hands ``voxel_size_zyx_from_xyz`` a swapped triple. Every test uses three
  *distinct* spacings so a reordering cannot pass.
* **Precedence.** Which of metadata and override wins is the difference between
  running on the real spacing and running on the file's 1.0-micron fallback.
"""
from __future__ import annotations

import pytest

from haemolynx.io import resolve_voxel_size_xyz, validate_voxel_size_xyz

# Coarse z, fine x — the usual confocal case, and all three differ.
METADATA_XYZ = (0.4, 0.5, 2.0)
OVERRIDE_XYZ = (0.25, 0.75, 3.0)

COMPLETE = {"source": "tiff", "status": "complete"}
MISSING = {"source": "tiff", "status": "missing"}
PARTIAL = {"source": "tiff", "status": "partial", "missing_axes": ["z"]}


# --- validate_voxel_size_xyz -----------------------------------------------


def test_validation_preserves_the_x_y_z_order_it_was_given():
    """The triple must come back in the same order; sorting it would swap x and z."""
    assert validate_voxel_size_xyz(METADATA_XYZ, label="metadata voxel size") == (0.4, 0.5, 2.0)


def test_validation_returns_plain_floats_not_numpy_scalars():
    """Downstream code json-serialises this into the voxel-metadata sidecar."""
    resolved = validate_voxel_size_xyz([1, 2, 3], label="metadata voxel size")
    assert resolved == (1.0, 2.0, 3.0)
    assert all(type(value) is float for value in resolved)


@pytest.mark.parametrize(
    "bad",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (),
        (0.0, 1.0, 2.0),
        (1.0, -2.0, 3.0),
        (1.0, float("nan"), 3.0),
        (1.0, float("inf"), 3.0),
    ],
)
def test_validation_rejects_triples_that_cannot_be_a_voxel_size(bad):
    """Zero, negative and non-finite spacings all divide-by-zero or NaN downstream."""
    with pytest.raises(ValueError, match="3 finite positive values"):
        validate_voxel_size_xyz(bad, label="metadata voxel size")


def test_validation_names_the_offending_setting_in_the_error():
    """The same check guards metadata and the user's override; the label disambiguates."""
    with pytest.raises(ValueError, match="voxel_size_override_xyz"):
        validate_voxel_size_xyz((0.0, 1.0, 1.0), label="voxel_size_override_xyz")


# --- policy validation -----------------------------------------------------


def test_an_unknown_policy_is_rejected_rather_than_silently_treated_as_auto():
    with pytest.raises(ValueError, match="voxel_size_policy must be one of"):
        resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, None, "metadata")


def test_policy_matching_ignores_case_and_surrounding_whitespace():
    """Policies arrive from YAML, where trailing spaces and capitals are easy to type."""
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, None, "  Auto ")
    assert (resolved, source) == (METADATA_XYZ, "metadata")


# --- policy 'override' -----------------------------------------------------


def test_override_policy_without_an_override_value_is_an_error():
    """Falling back to metadata here would silently ignore the user's explicit choice."""
    with pytest.raises(ValueError, match="requires voxel_size_override_xyz"):
        resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, None, "override")


def test_override_policy_beats_complete_metadata():
    """The point of 'override' is to overrule metadata the user does not trust."""
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, OVERRIDE_XYZ, "override")
    assert resolved == OVERRIDE_XYZ
    assert source == "manual_override"


def test_override_policy_still_validates_the_metadata_it_is_overruling():
    """Documents a trap: a broken metadata triple aborts the run even when unused."""
    with pytest.raises(ValueError, match="metadata voxel size"):
        resolve_voxel_size_xyz((0.4, 0.0, 2.0), COMPLETE, OVERRIDE_XYZ, "override")


# --- policy 'metadata_only' ------------------------------------------------


def test_metadata_only_policy_returns_metadata_when_it_is_complete():
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, None, "metadata_only")
    assert resolved == METADATA_XYZ
    assert source == "metadata"


def test_metadata_only_policy_ignores_an_override_that_was_also_supplied():
    """A leftover override in the config must not quietly win over the chosen policy."""
    resolved, source = resolve_voxel_size_xyz(
        METADATA_XYZ, COMPLETE, OVERRIDE_XYZ, "metadata_only"
    )
    assert resolved == METADATA_XYZ
    assert source == "metadata"


@pytest.mark.parametrize("status", [MISSING, PARTIAL, None, {}])
def test_metadata_only_policy_refuses_to_run_on_incomplete_metadata(status):
    """Anything short of 'complete' means some axis defaulted to 1.0 micron."""
    with pytest.raises(ValueError, match="requires complete metadata"):
        resolve_voxel_size_xyz(METADATA_XYZ, status, None, "metadata_only")


# --- policy 'auto' ---------------------------------------------------------


def test_auto_policy_prefers_complete_metadata_over_an_override():
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, COMPLETE, OVERRIDE_XYZ, "auto")
    assert resolved == METADATA_XYZ
    assert source == "metadata"


@pytest.mark.parametrize("status", [MISSING, PARTIAL, None, {}])
def test_auto_policy_falls_back_to_the_override_when_metadata_is_incomplete(status):
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, status, OVERRIDE_XYZ, "auto")
    assert resolved == OVERRIDE_XYZ
    assert source == "manual_override"


def test_auto_policy_reports_a_distinct_source_when_it_falls_back_to_bare_metadata():
    """'metadata_fallback' is the only signal that a spacing may be a 1.0 default."""
    resolved, source = resolve_voxel_size_xyz(METADATA_XYZ, MISSING, None, "auto")
    assert resolved == METADATA_XYZ
    assert source == "metadata_fallback"


def test_metadata_status_matching_is_case_insensitive():
    resolved, source = resolve_voxel_size_xyz(
        METADATA_XYZ, {"status": "COMPLETE"}, OVERRIDE_XYZ, "auto"
    )
    assert source == "metadata"
    assert resolved == METADATA_XYZ


def test_a_resolved_override_keeps_x_y_z_order():
    """Guards the override path against the same reordering the metadata path guards."""
    resolved, _ = resolve_voxel_size_xyz(METADATA_XYZ, MISSING, [0.25, 0.75, 3.0], "auto")
    assert resolved[0] == pytest.approx(0.25)
    assert resolved[2] == pytest.approx(3.0)
