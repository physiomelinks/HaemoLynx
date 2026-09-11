"""What ``skeletonise()`` does with the thickness-gate toggle.

The default is Lee on the whole mask. Turning the Skeletonise-tab toggle on
routes fat plasma-labelled regions through the EDT-ridge tree instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import segment, skeletonise
from haemolynx.preprocessing import (
    BRAID_FACTOR_LIMIT,
    THICK_VESSEL_MIN_RADIUS_UM,
    braid_factor,
    lee_braid_factor,
    thick_vessel_object_mask,
)
from test_thick_vessel_skeletonisation import plasma_labelled_object

SCHEMA = default_schema()


def settings_for(tmp_path: Path, mask_path: Path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": mask_path,
            "vtk_output_prefix": tmp_path / "run" / "network",
            "plot_dir": tmp_path / "plots",
            "do_skeletonize": True,
            **overrides,
        }
    )
    return values


def _write_mask(tmp_path: Path, mask: np.ndarray) -> Path:
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)
    return path


def _write_2d_mask(tmp_path: Path, mask: np.ndarray) -> Path:
    path = tmp_path / "mask_2d.tif"
    tifffile.imwrite(path, np.asarray(mask, dtype=np.uint8) * 255)
    return path


def test_thickness_gate_defaults_off_and_matches_the_locked_radius():
    assert SCHEMA["use_thick_vessel_skeletonisation"].default is False
    assert SCHEMA["skeleton_thick_vessel_min_radius_um"].default == pytest.approx(
        THICK_VESSEL_MIN_RADIUS_UM
    )
    assert SCHEMA["skeleton_fill_mask_holes_before_thickness"].default is True


def test_skeletonise_toggle_off_leaves_the_fat_sheet_on_lee(tmp_path):
    mask, fat_roi = plasma_labelled_object(8.0)
    assert lee_braid_factor(fat_roi, axis=2) > BRAID_FACTOR_LIMIT
    settings = settings_for(tmp_path, _write_mask(tmp_path, mask))
    volume = skeletonise(settings, segment(settings))
    thick = thick_vessel_object_mask(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=volume.voxel_size_zyx,
    )
    assert braid_factor(volume.skeleton & thick, axis=2) > BRAID_FACTOR_LIMIT


def test_skeletonise_forwards_max_bridge_distance_um_setting(tmp_path, monkeypatch):
    """Stage-wiring regression test: skeleton_thick_vessel_max_bridge_distance_um
    must reach skeletonize_thickness_gated's own max_bridge_distance_um
    parameter -- the lower levels (thick_vessels.py) are already pinned,
    but nothing previously checked the settings-dict key this stage reads
    actually matches the one wired through."""
    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.skeletonize_thickness_gated

    def spy(*args, **kwargs):
        captured["max_bridge_distance_um"] = kwargs.get("max_bridge_distance_um")
        return real(*args, **kwargs)

    monkeypatch.setattr(preprocessing_module, "skeletonize_thickness_gated", spy)

    mask, _fat_roi = plasma_labelled_object(8.0)
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, mask),
        use_thick_vessel_skeletonisation=True,
        skeleton_thick_vessel_max_bridge_distance_um=37.0,
    )
    skeletonise(settings, segment(settings))

    assert captured["max_bridge_distance_um"] == pytest.approx(37.0)


def test_skeletonise_restrict_to_mask_off_never_loads_vessel_masks(tmp_path, monkeypatch):
    """The default ("off") must cost nothing extra: no vessel-mask load at
    all, matching plain thickness-gated skeletonisation's existing cost --
    skeleton_thick_vessel_restrict_to_mask is opt-in."""
    import haemolynx.io as io_module

    calls: list = []
    real = io_module.load_and_validate_vessel_masks

    def spy(*args, **kwargs):
        calls.append(kwargs.get("mask_role"))
        return real(*args, **kwargs)

    monkeypatch.setattr(io_module, "load_and_validate_vessel_masks", spy)

    mask, _fat_roi = plasma_labelled_object(8.0)
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, mask),
        use_thick_vessel_skeletonisation=True,
    )
    skeletonise(settings, segment(settings))

    assert calls == []


def test_skeletonise_restrict_to_mask_unions_the_configured_masks(tmp_path, monkeypatch):
    """skeleton_thick_vessel_restrict_to_mask="both" must load both the
    large and small vessel masks and OR them together before forwarding
    the union to skeletonize_thickness_gated's own restrict_thick_to_mask
    -- proving the settings-dict key this stage reads reaches the right
    place, not just that some restriction is passed."""
    import haemolynx.io as io_module
    import haemolynx.preprocessing as preprocessing_module

    mask, _fat_roi = plasma_labelled_object(8.0)
    shape = mask.shape

    large_art = np.zeros(shape, dtype=bool)
    large_art[0, 0, 0] = True
    large_ven = np.zeros(shape, dtype=bool)
    large_ven[0, 0, 1] = True
    small_art = np.zeros(shape, dtype=bool)
    small_art[0, 0, 2] = True
    small_ven = np.zeros(shape, dtype=bool)
    small_ven[0, 0, 3] = True

    calls: list = []

    def fake_load(**kwargs):
        role = kwargs["mask_role"]
        calls.append(role)
        if role == "large":
            return large_art, large_ven, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        return small_art, small_ven, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)

    monkeypatch.setattr(io_module, "load_and_validate_vessel_masks", fake_load)

    captured = {}
    real = preprocessing_module.skeletonize_thickness_gated

    def spy(*args, **kwargs):
        captured["restrict_thick_to_mask"] = kwargs.get("restrict_thick_to_mask")
        return real(*args, **kwargs)

    monkeypatch.setattr(preprocessing_module, "skeletonize_thickness_gated", spy)

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, mask),
        use_thick_vessel_skeletonisation=True,
        skeleton_thick_vessel_restrict_to_mask="both",
    )
    skeletonise(settings, segment(settings))

    assert set(calls) == {"large", "small"}
    expected = large_art | large_ven | small_art | small_ven
    assert np.array_equal(captured["restrict_thick_to_mask"], expected)


def test_skeletonise_warns_when_a_restriction_mask_is_misaligned(tmp_path, monkeypatch, caplog):
    """A restriction mask whose own voxels mostly fall on background (not
    the real segmented image) must log a warning naming which mask and its
    overlap fraction -- see preprocessing.diagnose_mask_restriction_
    alignment. An aligned mask logs an info line instead, not a warning."""
    import haemolynx.io as io_module

    mask, _fat_roi = plasma_labelled_object(8.0)
    shape = mask.shape

    aligned = np.zeros(shape, dtype=bool)
    inside = tuple(np.argwhere(mask)[0])
    aligned[inside] = True  # a real voxel of the segmented mask itself

    misaligned = np.zeros(shape, dtype=bool)
    misaligned[0, 0, 0] = True  # the fixture's own corner, outside the vessel
    assert not mask[0, 0, 0], "fixture assumption: the corner is background"

    def fake_load(**kwargs):
        return aligned, misaligned, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)

    monkeypatch.setattr(io_module, "load_and_validate_vessel_masks", fake_load)

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, mask),
        use_thick_vessel_skeletonisation=True,
        skeleton_thick_vessel_restrict_to_mask="large",
    )
    with caplog.at_level("INFO", logger="haemolynx.pipeline.stages"):
        skeletonise(settings, segment(settings))

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    infos = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any(
        "large_venule_mask" in m and "Thick-vessel mask restriction" in m
        for m in warnings
    ), warnings
    assert any(
        "large_arteriole_mask" in m and "Thick-vessel mask restriction" in m
        for m in infos
    ), infos
    assert not any("large_arteriole_mask" in m for m in warnings)


def test_skeletonise_toggle_on_collapses_the_fat_sheet_and_keeps_capillaries(tmp_path):
    mask, fat_roi = plasma_labelled_object(8.0)
    lee_braid = lee_braid_factor(fat_roi, axis=2)
    assert lee_braid > BRAID_FACTOR_LIMIT
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, mask),
        use_thick_vessel_skeletonisation=True,
    )
    volume = skeletonise(settings, segment(settings))
    thick = thick_vessel_object_mask(
        mask,
        min_radius_um=THICK_VESSEL_MIN_RADIUS_UM,
        voxel_size_zyx=volume.voxel_size_zyx,
    )
    assert braid_factor(volume.skeleton & thick, axis=2) < lee_braid
    capillaries = mask & ~fat_roi
    assert int((volume.skeleton & capillaries).sum()) > 0


def test_skeletonise_loads_a_genuinely_2d_tiff_as_a_single_slice_volume_and_warns(
    tmp_path, caplog
):
    """A 2D input reaches skeletonise() through the same input_path /
    image_axis_order settings row as any other file -- no separate control
    -- and comes out promoted to a (1, H, W) volume with a logged warning,
    per haemolynx.io.load_2d.

    skeleton_closing_radius is disabled here: its morphological closing
    uses a 3D ball structuring element that erodes away a single-slice
    volume entirely, a pre-existing limitation of that particular
    genuinely-3D-specific operation (exactly the kind the loader's own
    warning already tells a caller to check for), not something this
    loading feature is responsible for fixing. Plain Lee skeletonisation
    is dimension-generic and unaffected -- confirmed here on the actual
    stage path, matching the isolated check already done for
    skeletonize_volume."""
    mask_2d = np.zeros((40, 40), dtype=bool)
    mask_2d[20, 5:35] = True
    mask_2d[5:35, 20] = True
    settings = settings_for(
        tmp_path,
        _write_2d_mask(tmp_path, mask_2d),
        skeleton_closing_radius=0,
    )

    with caplog.at_level("WARNING", logger="haemolynx.io.load_2d"):
        volume = skeletonise(settings, segment(settings))

    assert volume.skeleton.shape[0] == 1
    assert volume.skeleton.shape[1:] == mask_2d.shape
    assert int(volume.skeleton.sum()) > 0
    assert any("2D image" in r.message for r in caplog.records)


# --- segmentation cleanup (Input tab, before skeletonisation) ----------------


def _fragmented_mask() -> np.ndarray:
    """Two collinear tube fragments with a small gap, one tiny disconnected
    speck -- exercises reconnect + remove-small in one fixture."""
    mask = np.zeros((10, 10, 40), dtype=bool)
    mask[4:6, 4:6, 0:15] = True
    mask[4:6, 4:6, 20:35] = True
    mask[0, 0, 0] = True
    return mask


def test_segmentation_cleanup_off_by_default_never_calls_clean_segmented_mask(
    tmp_path, monkeypatch
):
    """Default settings must cost nothing extra: the orchestrator is never
    called at all, matching the codebase's "off is free" convention for
    every other opt-in stage feature."""
    import haemolynx.preprocessing as preprocessing_module

    calls: list = []
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(tmp_path, _write_mask(tmp_path, _fragmented_mask()))
    volume = skeletonise(settings, segment(settings))

    assert calls == []
    assert volume.raw_segmented_image is None


def test_skeletonise_forwards_reconnect_max_bridge_distance_um_setting(
    tmp_path, monkeypatch
):
    """`parameters_of` introspects whatever function currently sits at
    ``preprocessing.clean_segmented_mask_for_skeletonisation`` to decide
    which settings to forward -- a plain ``*args, **kwargs`` spy would
    replace that signature and silently break the forwarding it is meant
    to test, so the spy must keep the real one (functools.wraps)."""
    import functools

    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    @functools.wraps(real)
    def spy(*args, **kwargs):
        captured["reconnect_max_bridge_distance_um"] = kwargs.get(
            "reconnect_max_bridge_distance_um"
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_reconnect_gaps=True,
        segmentation_cleanup_reconnect_max_bridge_distance_um=17.0,
    )
    skeletonise(settings, segment(settings))

    assert captured["reconnect_max_bridge_distance_um"] == pytest.approx(17.0)


def test_skeletonise_forwards_smooth_sigma_um_setting(tmp_path, monkeypatch):
    import functools

    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    @functools.wraps(real)
    def spy(*args, **kwargs):
        captured["smooth_sigma_um"] = kwargs.get("smooth_sigma_um")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_smooth_surfaces=True,
        segmentation_cleanup_smooth_sigma_um=2.5,
    )
    skeletonise(settings, segment(settings))

    assert captured["smooth_sigma_um"] == pytest.approx(2.5)


def test_skeletonise_forwards_remove_small_min_volume_um3_setting(tmp_path, monkeypatch):
    import functools

    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    @functools.wraps(real)
    def spy(*args, **kwargs):
        captured["remove_small_min_volume_um3"] = kwargs.get(
            "remove_small_min_volume_um3"
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_remove_small_volumes=True,
        segmentation_cleanup_remove_small_min_volume_um3=9.0,
    )
    skeletonise(settings, segment(settings))

    assert captured["remove_small_min_volume_um3"] == pytest.approx(9.0)


def test_skeletonise_forwards_whisker_radius_um_setting(tmp_path, monkeypatch):
    import functools

    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    @functools.wraps(real)
    def spy(*args, **kwargs):
        captured["whisker_radius_um"] = kwargs.get("whisker_radius_um")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_remove_whiskers=True,
        segmentation_cleanup_whisker_radius_um=1.3,
    )
    skeletonise(settings, segment(settings))

    assert captured["whisker_radius_um"] == pytest.approx(1.3)


def test_skeletonise_forwards_split_min_marker_separation_um_setting(tmp_path, monkeypatch):
    import functools

    import haemolynx.preprocessing as preprocessing_module

    captured = {}
    real = preprocessing_module.clean_segmented_mask_for_skeletonisation

    @functools.wraps(real)
    def spy(*args, **kwargs):
        captured["split_min_marker_separation_um"] = kwargs.get(
            "split_min_marker_separation_um"
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(
        preprocessing_module, "clean_segmented_mask_for_skeletonisation", spy
    )

    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_split_narrow_necks=True,
        segmentation_cleanup_split_min_marker_separation_um=6.0,
    )
    skeletonise(settings, segment(settings))

    assert captured["split_min_marker_separation_um"] == pytest.approx(6.0)


def test_skeletonise_raw_segmented_image_is_set_when_only_remove_whiskers_is_on(
    tmp_path,
):
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_remove_whiskers=True,
    )
    volume = skeletonise(settings, segment(settings))
    assert volume.raw_segmented_image is not None


def test_skeletonise_raw_segmented_image_is_set_when_only_split_narrow_necks_is_on(
    tmp_path,
):
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_split_narrow_necks=True,
    )
    volume = skeletonise(settings, segment(settings))
    assert volume.raw_segmented_image is not None


def test_skeletonise_raw_segmented_image_is_set_only_when_cleanup_ran(tmp_path):
    settings_off = settings_for(tmp_path, _write_mask(tmp_path, _fragmented_mask()))
    volume_off = skeletonise(settings_off, segment(settings_off))
    assert volume_off.raw_segmented_image is None

    settings_on = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_remove_small_volumes=True,
        segmentation_cleanup_remove_small_min_volume_um3=5.0,
    )
    volume_on = skeletonise(settings_on, segment(settings_on))
    assert volume_on.raw_segmented_image is not None
    assert bool(volume_on.raw_segmented_image[0, 0, 0])  # the speck, pre-cleanup


def test_skeletonise_feeds_the_cleaned_mask_to_skeletonisation(tmp_path):
    """volume.image (what skeletonisation and everything downstream reads)
    must be the corrected mask, not the raw one -- the speck removed by
    cleanup must not still be present in volume.image."""
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _fragmented_mask()),
        segmentation_cleanup_remove_small_volumes=True,
        segmentation_cleanup_remove_small_min_volume_um3=5.0,
    )
    volume = skeletonise(settings, segment(settings))

    assert not bool(np.asarray(volume.image)[0, 0, 0])


def _solid_mask_with_an_internal_cavity() -> np.ndarray:
    """A solid block with a small enclosed void -- an imaging-noise-style
    air gap inside an otherwise solid vessel lumen."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    mask[4:6, 4:6, 4:6] = False  # the cavity itself
    return mask


def test_skeletonise_fill_cavities_fills_the_hollow_lumen_in_volume_image(tmp_path):
    """volume.image (what skeletonisation and everything downstream reads)
    must have the enclosed cavity filled in -- a hollow lumen from imaging
    noise must not survive into the mask fed to skeletonisation, and
    raw_segmented_image must still show it hollow, pre-cleanup."""
    settings = settings_for(
        tmp_path,
        _write_mask(tmp_path, _solid_mask_with_an_internal_cavity()),
        segmentation_cleanup_fill_cavities=True,
    )
    volume = skeletonise(settings, segment(settings))

    assert bool(np.asarray(volume.image)[5, 5, 5])  # the cavity, now filled
    assert volume.raw_segmented_image is not None
    assert not bool(volume.raw_segmented_image[5, 5, 5])  # cavity, pre-cleanup
