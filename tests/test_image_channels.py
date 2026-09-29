"""Choosing one channel of a multi-channel image.

A setting such as ``fwhm_raw_channel`` counts from 0; Fiji calls the same
channels C1, C2, ... These tests pin what the panel lists for a file
(:func:`haemolynx.io.tiff_channels` and :func:`haemolynx.gui.form.channel_choices`),
what preflight refuses before a run starts, and that FWHM's raw image is
read from the chosen channel of a composite.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from haemolynx.gui.form import (  # noqa: E402
    CHANNEL_SETTINGS,
    NO_CHANNEL_CHOOSE,
    NO_CHANNEL_SINGLE,
    channel_choices,
    field_for,
    widget_type_for,
)
from haemolynx.io import TiffChannel, tiff_channel_axis, tiff_channels  # noqa: E402
from haemolynx.pipeline import default_schema  # noqa: E402
from haemolynx.pipeline.checks import check_image_channels  # noqa: E402


def _lut(channel: int) -> np.ndarray:
    lut = np.zeros((3, 256), dtype=np.uint8)
    lut[channel] = np.arange(256)
    return lut


def _composite(path, *, channels: int = 2, **metadata) -> np.ndarray:
    """A ZCYX ImageJ composite whose channel c holds the value c + 1 (x 100)."""
    stack = np.stack(
        [np.full((3, 8, 8), 100 * (c + 1), dtype=np.uint16) for c in range(channels)], axis=1
    )
    tifffile.imwrite(str(path), stack, imagej=True, metadata={"axes": "ZCYX", **metadata})
    return stack


# --- tiff_channels ----------------------------------------------------------


def test_a_composite_lists_its_channels_by_lut_colour(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path, LUTs=[_lut(0), _lut(1)])

    channels = tiff_channels(path)

    assert channels == [TiffChannel(0, None, "red"), TiffChannel(1, None, "green")]
    assert [c.label for c in channels] == ["C1 (red)", "C2 (green)"]


def test_slice_labels_name_the_channels(tmp_path):
    path = tmp_path / "labelled.tif"
    # A hyperstack's labels run channel-fastest: one per (z, c) plane.
    _composite(path, Labels=["plasma", "claudin-5"] * 3)

    assert [c.label for c in tiff_channels(path)] == ["C1 (plasma)", "C2 (claudin-5)"]


def test_ome_channel_names_are_read(tmp_path):
    path = tmp_path / "named.ome.tif"
    stack = np.zeros((3, 2, 8, 8), dtype=np.uint16)
    tifffile.imwrite(
        str(path), stack, ome=True,
        metadata={"axes": "ZCYX", "Channel": {"Name": ["plasma", "claudin-5"]}},
    )

    assert [c.name for c in tiff_channels(path)] == ["plasma", "claudin-5"]


def test_a_channel_with_nothing_said_about_it_is_just_its_fiji_name(tmp_path):
    path = tmp_path / "plain.tif"
    _composite(path, channels=3)

    assert [c.label for c in tiff_channels(path)] == ["C1", "C2", "C3"]


def test_a_single_channel_stack_has_no_channels_to_choose(tmp_path):
    path = tmp_path / "single.tif"
    tifffile.imwrite(str(path), np.zeros((3, 8, 8), dtype=np.uint16), imagej=True,
                     metadata={"axes": "ZYX"})

    assert tiff_channels(path) == []


def test_a_z_stack_tifffile_labels_as_channels_is_still_a_z_stack(tmp_path):
    """tifffile saves a plain (z, y, x) array with imagej=True as a
    hyperstack of z *channels* (axes CYX). The loaders read it as the z-stack
    it is, so it must not be offered as 40 channels to choose from."""
    path = tmp_path / "stack.tif"
    tifffile.imwrite(str(path), np.zeros((40, 8, 8), dtype=np.uint16), imagej=True)
    with tifffile.TiffFile(str(path)) as tif:
        assert tif.series[0].axes == "CYX"  # the premise

    assert tiff_channels(path) == []


def test_a_single_plane_colour_composite_does_list_its_channels(tmp_path):
    path = tmp_path / "plane.tif"
    tifffile.imwrite(
        str(path), np.zeros((2, 8, 8), dtype=np.uint16), imagej=True,
        metadata={"axes": "CYX", "mode": "composite", "LUTs": [_lut(0), _lut(1)]},
    )

    assert [c.label for c in tiff_channels(path)] == ["C1 (red)", "C2 (green)"]


@pytest.mark.parametrize(
    "axes, shape, metadata, expected",
    [
        ("ZYX", (5, 8, 8), None, None),
        ("ZCYX", (5, 2, 8, 8), None, 1),
        ("CYX", (5, 8, 8), {"mode": "grayscale"}, None),
        ("CYX", (2, 8, 8), {"mode": "composite"}, 0),
        ("CYX", (2, 8, 8), {"mode": "color"}, 0),
        ("CYX", (2, 8, 8), {"LUTs": [np.zeros((3, 256))]}, 0),
        ("TZCYX", (1, 5, 2, 8, 8), None, 2),
        ("TCYX", (1, 3, 8, 8), None, None),
    ],
)
def test_which_axis_holds_channels(axes, shape, metadata, expected):
    assert tiff_channel_axis(axes, shape, metadata) == expected


@pytest.mark.parametrize("name", ["missing.tif", "not_a_tiff.tif"])
def test_a_missing_or_unreadable_file_lists_nothing_rather_than_raising(tmp_path, name):
    (tmp_path / "not_a_tiff.tif").write_bytes(b"not a tiff at all")

    assert tiff_channels(tmp_path / name) == []
    assert tiff_channels(None) == []
    assert tiff_channels("") == []


# --- channel_choices: the drop-down's entries --------------------------------


def test_choices_for_a_single_channel_file_offer_only_none():
    assert channel_choices([]) == [(NO_CHANNEL_SINGLE, None)]


def test_choices_for_a_composite_ask_for_a_channel_then_list_them():
    channels = [TiffChannel(0, None, "red"), TiffChannel(1, "claudin-5")]

    assert channel_choices(channels) == [
        (NO_CHANNEL_CHOOSE, None),
        ("C1 (red)", 0),
        ("C2 (claudin-5)", 1),
    ]


def test_a_chosen_channel_the_file_lacks_stays_listed_and_marked():
    choices = channel_choices([TiffChannel(0), TiffChannel(1)], current=3)

    assert choices[-1] == ("C4 (not in this file)", 3)
    assert [value for _label, value in choices] == [None, 0, 1, 3]


def test_a_channel_set_before_the_file_is_read_is_listed_by_its_fiji_name():
    assert channel_choices([], current=1) == [(NO_CHANNEL_SINGLE, None), ("C2", 1)]


def test_every_channel_setting_is_a_drop_down_paired_with_its_image_setting():
    schema = default_schema()
    names = {setting.name for setting in schema}
    for channel_name, path_name in CHANNEL_SETTINGS.items():
        assert channel_name in names and path_name in names
        setting = schema[channel_name]
        assert widget_type_for(setting) == "ComboBox"
        field = field_for(setting, 1)
        assert field.options["choices"] == [(NO_CHANNEL_SINGLE, None), ("C2", 1)]


# --- preflight ---------------------------------------------------------------


def _fwhm_settings(path, channel=None, **extra):
    return {
        "use_fwhm_edge_diameters": True,
        "fwhm_raw_tiff_path": str(path),
        "fwhm_raw_channel": channel,
        **extra,
    }


def test_a_composite_with_no_channel_chosen_is_refused_naming_its_channels(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path, LUTs=[_lut(0), _lut(1)])

    report = check_image_channels(_fwhm_settings(path))

    assert len(report.errors) == 1
    assert "C1 (red), C2 (green)" in report.errors[0]
    assert "fwhm_raw_channel" in report.errors[0]


def test_a_channel_the_composite_lacks_is_refused(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path)

    report = check_image_channels(_fwhm_settings(path, channel=2))

    assert len(report.errors) == 1
    assert "C3" in report.errors[0] and "2 channels" in report.errors[0]


def test_a_channel_set_for_a_single_channel_image_is_refused(tmp_path):
    path = tmp_path / "single.tif"
    tifffile.imwrite(str(path), np.zeros((3, 8, 8), dtype=np.uint16), imagej=True)

    report = check_image_channels(_fwhm_settings(path, channel=0))

    assert len(report.errors) == 1
    assert "single-channel" in report.errors[0]


def test_a_valid_channel_passes(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path)

    assert not check_image_channels(_fwhm_settings(path, channel=1)).errors


def test_the_endothelial_image_is_checked_the_same_way(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path)
    settings = {
        "use_endothelial_diameters": True,
        "endothelial_image_path": str(path),
        "endothelial_channel": None,
    }

    report = check_image_channels(settings)

    assert len(report.errors) == 1
    assert "endothelial_channel" in report.errors[0]


def test_an_image_its_measurement_does_not_use_is_not_checked(tmp_path):
    path = tmp_path / "composite.tif"
    _composite(path)

    settings = _fwhm_settings(path, use_fwhm_edge_diameters=False)

    assert not check_image_channels(settings).errors


def test_preflight_runs_the_channel_check(tmp_path):
    from haemolynx.pipeline import preflight

    path = tmp_path / "composite.tif"
    _composite(path)
    schema = default_schema()
    settings = {setting.name: setting.default for setting in schema}
    settings.update(_fwhm_settings(path))

    report = preflight(settings, schema)

    assert any("fwhm_raw_channel" in error for error in report.errors)


# --- FWHM reads the chosen channel -------------------------------------------


def test_channel_0_of_a_z_stack_tifffile_labels_as_channels_is_the_whole_stack(tmp_path):
    """Reading "channel 0" must not take one plane of a mislabelled z-stack."""
    from haemolynx.haemodynamics.automated import load_single_channel_tiff_volume

    path = tmp_path / "stack.tif"
    stack = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    tifffile.imwrite(str(path), stack, imagej=True)

    np.testing.assert_array_equal(load_single_channel_tiff_volume(path, channel=0), stack)


def _apply_config(path, channel):
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig

    return HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"capillary": 5.0}},
        fwhm={
            "use_fwhm_edge_diameters": True,
            "fwhm_raw_tiff_path": str(path),
            "fwhm_raw_channel": channel,
        },
    )


@pytest.mark.parametrize("channel, value", [(0, 100.0), (1, 200.0)])
def test_the_fwhm_raw_volume_is_the_chosen_channel(tmp_path, channel, value):
    from haemolynx.haemodynamics.apply import load_fwhm_raw_volume

    path = tmp_path / "composite.tif"
    _composite(path)

    volume = load_fwhm_raw_volume(_apply_config(path, channel))

    assert volume.shape == (3, 8, 8)
    np.testing.assert_array_equal(np.unique(volume), [value])


def test_the_fwhm_measurement_forwards_its_channel_to_the_loader(tmp_path, monkeypatch):
    """measure_edge_diameters_fwhm_from_raw_tiff reads the channel it is
    given, rather than failing on (or averaging) a composite."""
    import networkx as nx

    from haemolynx.haemodynamics import automated

    path = tmp_path / "composite.tif"
    _composite(path)
    loaded = {}
    original = automated.load_single_channel_tiff_volume

    def spy(*args, **kwargs):
        volume = original(*args, **kwargs)
        loaded["channel"] = kwargs.get("channel")
        loaded["values"] = np.unique(volume)
        return volume

    monkeypatch.setattr(automated, "load_single_channel_tiff_volume", spy)

    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        nx.MultiGraph(),
        raw_tiff_path=path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=1.0,
        transverse_profile_step_um=0.5,
        transverse_half_extent_um=4.0,
        raw_channel=1,
    )

    assert loaded["channel"] == 1
    np.testing.assert_array_equal(loaded["values"], [200.0])
