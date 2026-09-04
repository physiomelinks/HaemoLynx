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
