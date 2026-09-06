"""Thick-vessel braid detection: hand-verified geometry, no pipeline involved."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.preprocessing.thick_vessel_braid_guard import (
    component_long_axis,
    detect_braided_thick_vessel_components,
    format_braided_thick_vessel_report,
)


# --- component_long_axis -------------------------------------------------


def test_a_line_along_z_reports_axis_zero():
    mask = np.zeros((10, 8, 8), dtype=bool)
    mask[0:10, 4, 4] = True
    assert component_long_axis(mask) == 0


def test_a_line_along_x_reports_axis_two():
    mask = np.zeros((8, 8, 10), dtype=bool)
    mask[4, 4, 0:10] = True
    assert component_long_axis(mask) == 2


def test_a_single_voxel_defaults_to_axis_zero():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    assert component_long_axis(mask) == 0


# --- detect_braided_thick_vessel_components: the metric itself -----------


def _block(shape=(3, 3, 20)) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def test_a_single_clean_centreline_is_not_flagged():
    """One skeleton voxel per occupied slice along the long axis: braid_factor
    is exactly 1.0, well under the default limit of 2.0."""
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[1, 1, :] = True

    assert detect_braided_thick_vessel_components(mask, skeleton) == []


def test_three_parallel_strands_are_flagged():
    """Three skeleton voxels in every occupied slice: braid_factor=3.0,
    above the default limit of 2.0 -- the medial-sheet shape this guard
    exists for."""
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[0, 0, :] = True
    skeleton[1, 1, :] = True
    skeleton[2, 2, :] = True

    flagged = detect_braided_thick_vessel_components(mask, skeleton)

    assert len(flagged) == 1
    component = flagged[0]
    assert component.braid_factor == pytest.approx(3.0)
    assert component.long_axis == 2
    assert component.occupied_slices == 20
    assert component.voxel_count == mask.sum()


def test_exactly_at_the_limit_is_not_flagged():
    """The gate is strictly-greater-than, matching how BRAID_FACTOR_LIMIT is
    already used in thick_vessels.characterisation_rows (`braid > LIMIT`)."""
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[0, 0, :] = True
    skeleton[1, 1, :] = True  # exactly 2 voxels/slice -> braid_factor == 2.0

    assert detect_braided_thick_vessel_components(mask, skeleton) == []


def test_too_few_occupied_slices_is_skipped_however_braided():
    """A stub with only 3 occupied slices cannot give a stable mean, even if
    every one of those slices is fully braided."""
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[:, :, 0:3] = mask[:, :, 0:3]  # 9 voxels/slice, only 3 slices

    assert detect_braided_thick_vessel_components(mask, skeleton) == []
    # Lowering the bar to match reaches it.
    flagged = detect_braided_thick_vessel_components(mask, skeleton, min_occupied_slices=3)
    assert len(flagged) == 1
    assert flagged[0].braid_factor == pytest.approx(9.0)


def test_only_the_braided_component_is_flagged_not_the_clean_one():
    mask = np.zeros((3, 3, 50), dtype=bool)
    mask[:, :, 0:20] = True
    mask[:, :, 30:50] = True  # two separate components, both far apart
    skeleton = np.zeros_like(mask)
    skeleton[1, 1, 0:20] = True  # clean centreline in the first component
    skeleton[0, 0, 30:50] = True  # braided: 3 strands in the second
    skeleton[1, 1, 30:50] = True
    skeleton[2, 2, 30:50] = True

    flagged = detect_braided_thick_vessel_components(mask, skeleton)

    assert len(flagged) == 1
    assert flagged[0].voxel_count == mask[:, :, 30:50].sum()
    assert flagged[0].bounding_box[2] == (30, 50)


def test_worst_first_when_more_than_one_component_is_flagged():
    mask = np.zeros((3, 3, 60), dtype=bool)
    mask[:, :, 0:20] = True
    mask[:, :, 40:60] = True
    skeleton = np.zeros_like(mask)
    # First component: 2 strands (braid_factor 2.0) is not flagged by itself,
    # so use 3 vs 5 strands to keep both clearly above the limit and ordered.
    skeleton[0, 0, 0:20] = True
    skeleton[1, 1, 0:20] = True
    skeleton[2, 2, 0:20] = True
    for offset in range(3):
        skeleton[0, offset, 40:60] = True
    skeleton[1, 1, 40:60] = True
    skeleton[2, 2, 40:60] = True

    flagged = detect_braided_thick_vessel_components(mask, skeleton)

    assert len(flagged) == 2
    assert flagged[0].braid_factor >= flagged[1].braid_factor
    assert flagged[0].bounding_box[2] == (40, 60), "the 5-strand component is worse"


def test_centroid_is_reported_in_physical_microns():
    mask = np.ones((1, 3, 4), dtype=bool)
    skeleton = np.zeros_like(mask)
    skeleton[:, :, 0:3] = True  # 3 voxels/slice (all of y), 3 occupied slices

    flagged = detect_braided_thick_vessel_components(
        mask, skeleton, voxel_size_zyx=(2.0, 1.0, 0.5), min_occupied_slices=3
    )

    assert len(flagged) == 1
    assert flagged[0].braid_factor == pytest.approx(3.0)
    # Mask centroid in voxels is (0, 1.0, 1.5): y in {0,1,2}, x in {0,1,2,3}.
    assert flagged[0].centroid_um == pytest.approx((0.0, 1.0, 0.75))


def test_a_mismatched_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        detect_braided_thick_vessel_components(
            np.zeros((3, 3, 3), dtype=bool), np.zeros((4, 4, 4), dtype=bool)
        )


def test_an_empty_mask_flags_nothing():
    empty = np.zeros((3, 3, 3), dtype=bool)
    assert detect_braided_thick_vessel_components(empty, empty) == []


def test_a_negative_braid_limit_is_rejected():
    with pytest.raises(ValueError, match="braid_factor_limit"):
        detect_braided_thick_vessel_components(
            np.zeros((3, 3, 3), dtype=bool),
            np.zeros((3, 3, 3), dtype=bool),
            braid_factor_limit=-1.0,
        )


def test_a_braid_limit_of_zero_flags_every_component_instead_of_crashing():
    """The schema's own bound on this setting is inclusive of 0.0 (a
    maximally-sensitive "flag everything" reading), so this diagnostic-only
    function must accept it rather than raise -- see
    test_pipeline_schema_api.py's boundary-value regression for the crash
    this used to cause when the schema and this function disagreed. Even a
    single clean centreline (braid_factor == 1.0, normally well under the
    default limit) is flagged once the limit itself is 0.0."""
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[1, 1, :] = True

    flagged = detect_braided_thick_vessel_components(
        mask, skeleton, braid_factor_limit=0.0
    )

    assert len(flagged) == 1
    assert flagged[0].braid_factor == pytest.approx(1.0)


def test_a_nonpositive_min_occupied_slices_is_rejected():
    with pytest.raises(ValueError, match="min_occupied_slices"):
        detect_braided_thick_vessel_components(
            np.zeros((3, 3, 3), dtype=bool),
            np.zeros((3, 3, 3), dtype=bool),
            min_occupied_slices=0,
        )


# --- format_braided_thick_vessel_report -----------------------------------


def test_report_of_no_components_says_so():
    assert (
        format_braided_thick_vessel_report([])
        == "Thick-vessel braid guard: no braided components flagged."
    )


def test_report_names_every_flagged_component():
    mask = _block()
    skeleton = np.zeros_like(mask)
    skeleton[0, 0, :] = True
    skeleton[1, 1, :] = True
    skeleton[2, 2, :] = True

    report = format_braided_thick_vessel_report(
        detect_braided_thick_vessel_components(mask, skeleton)
    )

    assert "1 component(s) flagged" in report
    assert "component=1" in report
    assert "braid_factor=3.00" in report


# --- against real thickness-gated output: good trunks pass, bad ones flag ---


def test_a_long_round_thick_trunk_is_not_flagged():
    """The case the user reports this feature already gets right: a large,
    roughly tube-shaped thick vessel, much longer than it is wide. Round
    cross-section, isolated (no fused branches to confound the reading)."""
    from test_thick_vessel_skeletonisation import SPACING_ZYX, _disk_tube_along_axis
    from haemolynx.preprocessing import THICK_VESSEL_MIN_RADIUS_UM, skeletonize_thickness_gated

    shape = (40, 40, 120)
    mask = _disk_tube_along_axis(shape, (20, 20, 0), 8.0, 120, axis=2)
    skeleton, thick = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        return_thick_mask=True,
    )

    assert detect_braided_thick_vessel_components(thick, skeleton) == []


def test_a_short_wide_round_trunk_is_also_not_flagged():
    """A poor length-to-width ratio alone is not enough to braid -- a round
    cross-section still resolves to a tree, only a flattened one (below)
    exposes the medial-sheet ambiguity this guard exists for."""
    from test_thick_vessel_skeletonisation import SPACING_ZYX, _disk_tube_along_axis
    from haemolynx.preprocessing import THICK_VESSEL_MIN_RADIUS_UM, skeletonize_thickness_gated

    shape = (40, 40, 30)
    mask = _disk_tube_along_axis(shape, (20, 20, 0), 15.0, 30, axis=2)
    skeleton, thick = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=SPACING_ZYX,
        return_thick_mask=True,
    )

    assert detect_braided_thick_vessel_components(thick, skeleton) == []


def test_a_flattened_ribbon_shaped_thick_vessel_is_flagged():
    """A flattened (elliptical), ribbon-like cross-section genuinely has no
    single well-defined medial line -- Lee's classic sheet ambiguity, which a
    round vessel does not have. No fused branches here either, so this
    isolates the shape itself as the cause, not branch topology."""
    from haemolynx.preprocessing import THICK_VESSEL_MIN_RADIUS_UM, skeletonize_thickness_gated

    shape = (24, 88, 76)
    zz, yy, xx = np.indices(shape)
    minor, major = 8.0, 40.0
    cross = ((zz - 12) / minor) ** 2 + ((yy - 44) / major) ** 2 <= 1.0
    mask = cross & (xx >= 8) & (xx < 68)
    skeleton, thick = skeletonize_thickness_gated(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        return_thick_mask=True,
    )

    flagged = detect_braided_thick_vessel_components(thick, skeleton)

    assert len(flagged) == 1
    assert flagged[0].braid_factor > 2.0
