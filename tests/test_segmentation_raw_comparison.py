"""Cross-check a segmented mask against its own raw (unsegmented) image."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.preprocessing import segmentation_raw_comparison as src


def _cylinder_along_x(
    shape: tuple[int, int, int], *, z: float, y: float, radius: float, x0: int, x1: int
) -> np.ndarray:
    zz, yy, xx = np.indices(shape, dtype=float)
    radial = np.sqrt((zz - float(z)) ** 2 + (yy - float(y)) ** 2)
    return (radial <= float(radius)) & (xx >= int(x0)) & (xx <= int(x1))


def _raw_from_mask(mask: np.ndarray, *, foreground=200.0, background=10.0) -> np.ndarray:
    """A clean synthetic 'raw' image whose Otsu threshold exactly recovers *mask*."""
    return np.where(mask, foreground, background).astype(np.float32)


# --- shape validation --------------------------------------------------------------


def test_mismatched_shapes_raise():
    mask = np.zeros((10, 10, 10), dtype=bool)
    raw = np.zeros((10, 10, 11), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        src.compare_segmentation_to_raw_image(mask, raw)


# --- degenerate raw image ------------------------------------------------------------


def test_flat_raw_image_reports_everything_as_added():
    mask = _cylinder_along_x((20, 20, 20), z=10, y=10, radius=3.0, x0=2, x1=17)
    raw = np.full((20, 20, 20), 5.0, dtype=np.float32)

    result = src.compare_segmentation_to_raw_image(mask, raw)

    assert result.raw_foreground_voxel_count == 0
    assert result.added_voxel_count == int(mask.sum())
    assert result.added_fraction == pytest.approx(1.0)
    assert result.removed_voxel_count == 0


def test_empty_mask_and_flat_raw_reports_nothing_added():
    mask = np.zeros((10, 10, 10), dtype=bool)
    raw = np.full((10, 10, 10), 5.0, dtype=np.float32)

    result = src.compare_segmentation_to_raw_image(mask, raw)

    assert result.added_voxel_count == 0
    assert result.added_fraction == 0.0


# --- perfect agreement --------------------------------------------------------------


def test_perfect_agreement_has_no_added_or_removed_voxels():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(mask)

    result = src.compare_segmentation_to_raw_image(mask, raw)

    assert result.added_voxel_count == 0
    assert result.removed_voxel_count == 0
    assert result.agreement_iou == pytest.approx(1.0)
    assert result.missed_structure_count == 0


# --- added voxels ---------------------------------------------------------------------


def test_added_voxels_detected_for_over_segmentation():
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(body)

    over_segmented = body.copy()
    over_segmented[1:4, 1:4, 1:4] = True  # extra mask foreground with no raw support

    result = src.compare_segmentation_to_raw_image(over_segmented, raw)

    assert result.added_voxel_count == 27
    assert result.added_fraction > 0.0
    assert result.removed_voxel_count == 0


# --- removed voxels ---------------------------------------------------------------------


def test_removed_voxels_detected_for_under_segmentation():
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(body)

    under_segmented = body.copy()
    under_segmented[:, :, 2:6] = False  # mask drops part of the real vessel

    result = src.compare_segmentation_to_raw_image(under_segmented, raw)

    assert result.removed_voxel_count > 0
    assert result.removed_fraction > 0.0
    assert result.added_voxel_count == 0


# --- missed structures ---------------------------------------------------------------------


def test_missed_structure_detected_when_whole_vessel_is_absent_from_mask():
    shape = (30, 30, 30)
    kept = _cylinder_along_x(shape, z=10, y=10, radius=3.0, x0=2, x1=27)
    missed = _cylinder_along_x(shape, z=22, y=22, radius=3.0, x0=2, x1=27)
    raw_foreground = kept | missed
    raw = _raw_from_mask(raw_foreground)

    # The mask only captured the first vessel -- the second is missing entirely.
    result = src.compare_segmentation_to_raw_image(kept, raw)

    assert result.missed_structure_count == 1
    assert result.missed_structure_total_voxels == pytest.approx(int(missed.sum()), abs=5)
    assert result.missed_structure_total_volume_um3 > 0.0


def test_a_frayed_edge_on_a_detected_vessel_is_not_a_missed_structure():
    """Partial overlap (most of a component IS in the mask) must not count as
    a whole missed structure -- that's what the voxel-level removed count is
    for; missed_structure_count is reserved for near-total misses."""
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(body)

    frayed = body.copy()
    frayed[:, :, 2:4] = False  # mask misses a slice at one end, most of the body remains

    result = src.compare_segmentation_to_raw_image(frayed, raw)

    assert result.missed_structure_count == 0
    assert result.removed_voxel_count > 0


def test_missed_structure_min_volume_ignores_small_noise_components():
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    raw_foreground = np.zeros(shape, dtype=bool)
    raw_foreground[1, 1, 1] = True  # a single-voxel speck, well below the volume floor
    raw = _raw_from_mask(raw_foreground)

    result = src.compare_segmentation_to_raw_image(
        mask, raw, voxel_size_zyx=(1.0, 1.0, 1.0), missed_structure_min_volume_um3=5.0,
    )

    assert result.missed_structure_count == 0


def test_missed_structure_min_volume_is_voxel_size_aware():
    """The same absolute voxel count must pass or fail the floor differently
    depending on physical voxel size -- this is the whole point of using a
    physical volume, not a raw voxel count, for the floor."""
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    raw_foreground = np.zeros(shape, dtype=bool)
    raw_foreground[5:8, 5:8, 5:8] = True  # a 27-voxel component
    raw = _raw_from_mask(raw_foreground)

    fine = src.compare_segmentation_to_raw_image(
        mask, raw, voxel_size_zyx=(0.1, 0.1, 0.1), missed_structure_min_volume_um3=20.0,
    )
    coarse = src.compare_segmentation_to_raw_image(
        mask, raw, voxel_size_zyx=(2.0, 2.0, 2.0), missed_structure_min_volume_um3=20.0,
    )

    assert fine.missed_structure_count == 0  # 27 voxels * 0.001um3 = 0.027um3, under the floor
    assert coarse.missed_structure_count == 1  # 27 voxels * 8um3 = 216um3, over the floor


# --- physical units ------------------------------------------------------------------


def test_missed_structure_volume_scales_with_voxel_size():
    shape = (30, 30, 30)
    kept = _cylinder_along_x(shape, z=10, y=10, radius=3.0, x0=2, x1=27)
    missed = _cylinder_along_x(shape, z=22, y=22, radius=3.0, x0=2, x1=27)
    raw = _raw_from_mask(kept | missed)

    fine = src.compare_segmentation_to_raw_image(kept, raw, voxel_size_zyx=(1.0, 1.0, 1.0))
    coarse = src.compare_segmentation_to_raw_image(kept, raw, voxel_size_zyx=(2.0, 2.0, 2.0))

    assert coarse.missed_structure_total_volume_um3 == pytest.approx(
        fine.missed_structure_total_volume_um3 * 8.0
    )


# --- report text -----------------------------------------------------------------------


# --- precomputed raw-image analysis (analyze_raw_image_foreground) -----------------


def test_precomputed_foreground_matches_the_unprecomputed_result():
    """A caller that precomputes `RawImageForeground` and passes it in via
    `precomputed` must get bit-for-bit the same comparison a plain call
    (which derives it internally) would -- the point of `precomputed` is to
    skip redoing Otsu thresholding and connected-components labelling for
    every candidate mask against the same raw image, never to change the
    answer.
    """
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(body)
    over_segmented = body.copy()
    over_segmented[1:4, 1:4, 1:4] = True

    foreground = src.analyze_raw_image_foreground(raw, voxel_size_zyx=(1.0, 1.0, 1.0))
    assert foreground is not None

    direct = src.compare_segmentation_to_raw_image(over_segmented, raw, voxel_size_zyx=(1.0, 1.0, 1.0))
    via_precomputed = src.compare_segmentation_to_raw_image(
        over_segmented, raw, voxel_size_zyx=(1.0, 1.0, 1.0), precomputed=foreground,
    )

    assert via_precomputed == direct


def test_analyze_raw_image_foreground_returns_none_for_a_flat_image():
    """Mirrors compare_segmentation_to_raw_image's own degenerate-case
    handling -- a precomputing caller must be able to tell "nothing to
    analyze" apart from a real (if empty) foreground."""
    raw = np.full((10, 10, 10), 5.0, dtype=np.float32)
    assert src.analyze_raw_image_foreground(raw) is None


def test_precomputed_none_from_a_flat_image_matches_the_direct_degenerate_case():
    mask = _cylinder_along_x((20, 20, 20), z=10, y=10, radius=3.0, x0=2, x1=17)
    raw = np.full((20, 20, 20), 5.0, dtype=np.float32)

    foreground = src.analyze_raw_image_foreground(raw)
    assert foreground is None

    direct = src.compare_segmentation_to_raw_image(mask, raw)
    via_precomputed = src.compare_segmentation_to_raw_image(mask, raw, precomputed=foreground)

    assert via_precomputed == direct


def test_precomputed_foreground_avoids_recomputing_otsu_and_labelling(monkeypatch):
    """Regression: `compare_segmentation_to_raw_image` used to redo Otsu
    thresholding and a full connected-components labelling of the raw
    image on every single call, even across repeated calls against the
    same unchanging raw image -- exactly what a settings search sweeping
    ~20 segmentation-cleanup sub-sweeps, one candidate mask per trial,
    does. Confirms neither is called again once `precomputed` is supplied.
    """
    shape = (20, 20, 20)
    body = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(body)
    foreground = src.analyze_raw_image_foreground(raw)

    def boom(*_args, **_kwargs):
        raise AssertionError("threshold_otsu/label should not be called when precomputed is given")

    monkeypatch.setattr(src, "threshold_otsu", boom)
    monkeypatch.setattr(src, "label", boom)

    result = src.compare_segmentation_to_raw_image(body, raw, precomputed=foreground)
    assert result.agreement_iou == pytest.approx(1.0)


def test_report_text_includes_every_field():
    shape = (20, 20, 20)
    mask = _cylinder_along_x(shape, z=10, y=10, radius=4.0, x0=2, x1=17)
    raw = _raw_from_mask(mask)
    result = src.compare_segmentation_to_raw_image(mask, raw)

    text = src.format_segmentation_raw_comparison_report(result)

    assert "agreement" in text.lower()
    assert "added voxels" in text
    assert "removed voxels" in text
    assert "missed structures" in text
