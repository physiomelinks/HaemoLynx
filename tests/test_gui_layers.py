"""Turning the layer open in napari into the settings for a run.

Checked against a stand-in for a layer rather than napari itself: the three
attributes that matter are `data`, `scale` and `source.path`, and the decisions
made from them are where the mistakes live -- above all the (z, y, x) to
(x, y, z) turn, which is invisible on isotropic data and wrong on every real
stack.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from haemolynx.gui.layers import (
    export_name_for,
    input_for_layer,
    rejection_reason,
    source_path_of,
    voxel_size_xyz_from_scale,
)


def fake_layer(shape=(4, 5, 6), scale=None, path=None, name="vessels", rgb=False):
    return SimpleNamespace(
        data=np.zeros(shape, dtype=np.uint8),
        scale=scale,
        source=SimpleNamespace(path=path),
        name=name,
        rgb=rgb,
    )


# --- what may be used at all -------------------------------------------------


def test_a_3d_layer_is_accepted():
    assert rejection_reason(fake_layer()) is None


def test_a_2d_layer_is_refused_by_name():
    reason = rejection_reason(fake_layer(shape=(5, 6)))
    assert "3D" in reason and "2D" in reason


def test_an_rgb_layer_is_refused():
    assert "RGB" in rejection_reason(fake_layer(rgb=True))


def test_a_layer_with_no_data_is_refused():
    assert rejection_reason(SimpleNamespace(data=None)) is not None


def test_input_for_layer_raises_the_same_reason():
    with pytest.raises(ValueError, match="3D"):
        input_for_layer(fake_layer(shape=(5, 6)))


# --- voxel size: (z, y, x) scale -> (x, y, z) metadata -----------------------


def test_the_layer_scale_is_reversed_into_image_metadata_order():
    """napari scales per array axis (z, y, x); the setting is (x, y, z)."""
    assert voxel_size_xyz_from_scale((2.0, 0.5, 0.4)) == (0.4, 0.5, 2.0)


def test_an_all_ones_scale_carries_no_information():
    """Leave whatever the file says rather than overriding it with nothing."""
    assert voxel_size_xyz_from_scale((1.0, 1.0, 1.0)) is None


def test_a_missing_or_wrong_length_scale_is_ignored():
    assert voxel_size_xyz_from_scale(None) is None
    assert voxel_size_xyz_from_scale((1.0, 2.0)) is None


def test_an_anisotropic_layer_sets_the_override_and_the_policy():
    result = input_for_layer(fake_layer(scale=(2.0, 0.5, 0.4), path=None), Path("/tmp"))
    assert result.settings["voxel_size_override_xyz"] == [0.4, 0.5, 2.0]
    assert result.settings["voxel_size_policy"] == "override"


def test_an_isotropic_layer_leaves_the_voxel_size_alone():
    result = input_for_layer(fake_layer(scale=(1.0, 1.0, 1.0)), Path("/tmp"))
    assert "voxel_size_override_xyz" not in result.settings
    assert "voxel_size_policy" not in result.settings


# --- where the run reads from ------------------------------------------------


def test_a_layer_read_from_a_tiff_points_the_run_at_that_file(tmp_path):
    existing = tmp_path / "mask.tif"
    existing.write_bytes(b"x")

    result = input_for_layer(fake_layer(path=str(existing)), tmp_path)

    assert result.settings["input_path"] == existing
    assert result.needs_export is False
    assert existing.name in result.note


def test_a_layer_with_no_file_behind_it_must_be_written_out(tmp_path):
    result = input_for_layer(fake_layer(path=None, name="threshold result"), tmp_path)

    assert result.needs_export is True
    assert result.settings["input_path"] == tmp_path / "threshold_result.tif"
    assert "written to" in result.note


def test_a_source_the_loaders_cannot_read_is_treated_as_no_file(tmp_path):
    """A layer read from a .png is not something the pipeline can re-read."""
    other = tmp_path / "picture.png"
    other.write_bytes(b"x")

    assert source_path_of(fake_layer(path=str(other))) is None
    assert input_for_layer(fake_layer(path=str(other)), tmp_path).needs_export is True


def test_a_source_path_that_has_since_gone_is_treated_as_no_file(tmp_path):
    missing = tmp_path / "deleted.tif"
    assert source_path_of(fake_layer(path=str(missing))) is None


def test_an_h5_source_is_read_directly(tmp_path):
    existing = tmp_path / "volume.h5"
    existing.write_bytes(b"x")
    assert source_path_of(fake_layer(path=str(existing))) == existing


# --- naming the exported file ------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("vessels", "vessels.tif"),
        ("threshold result", "threshold_result.tif"),
        ("nerve [1]", "nerve_1.tif"),
        ("", "layer.tif"),
        ("///", "layer.tif"),
    ],
)
def test_the_exported_name_is_derived_from_the_layer_name(name, expected):
    assert export_name_for(fake_layer(name=name)) == expected


def test_the_settings_a_layer_produces_validate_against_the_schema(tmp_path):
    """Whatever a layer sets has to be a legal value for that setting."""
    from haemolynx.pipeline import default_schema

    existing = tmp_path / "mask.tif"
    existing.write_bytes(b"x")
    schema = default_schema()

    result = input_for_layer(fake_layer(scale=(2.0, 0.5, 0.4), path=str(existing)), tmp_path)

    values = {setting.name: setting.default for setting in schema}
    values.update(result.settings)
    validated = schema.validate(values)
    assert validated["voxel_size_policy"] == "override"
    assert Path(validated["input_path"]) == existing
