"""The channel drop-downs, built for real.

``raw_image_channel`` and ``tissue_raw_channel`` are drop-downs of the
channels in the file their image setting names, Fiji-named (C1, C2, ...).
These tests build the panel and check what only a real widget shows: the
list follows the file, and a channel written in before the file is (a config
loads in any order) is kept rather than refused.
"""
from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")
tifffile = pytest.importorskip("tifffile")

from haemolynx.gui._widget import settings_widget  # noqa: E402
from haemolynx.gui.form import NO_CHANNEL_CHOOSE, NO_CHANNEL_SINGLE  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qapp):
    return settings_widget(napari_viewer=None)


def _composite(path):
    lut_red = np.zeros((3, 256), dtype=np.uint8)
    lut_red[0] = np.arange(256)
    lut_green = np.zeros((3, 256), dtype=np.uint8)
    lut_green[1] = np.arange(256)
    tifffile.imwrite(
        str(path), np.zeros((3, 2, 8, 8), dtype=np.uint16), imagej=True,
        metadata={"axes": "ZCYX", "LUTs": [lut_red, lut_green]},
    )
    return path


def _labels(row):
    return [row.native.itemText(i) for i in range(row.native.count())]


@pytest.mark.parametrize(
    "channel_name, path_name",
    [("raw_image_channel", "raw_image_path"), ("tissue_raw_channel", "tissue_raw_tiff_path")],
)
def test_the_drop_down_lists_the_channels_of_its_file(panel, tmp_path, channel_name, path_name):
    rows = panel._haemolynx_rows()
    assert _labels(rows[channel_name]) == [NO_CHANNEL_SINGLE]

    rows[path_name].value = _composite(tmp_path / "composite.tif")

    assert _labels(rows[channel_name]) == [NO_CHANNEL_CHOOSE, "C1 (red)", "C2 (green)"]
    rows[channel_name].value = 1
    assert panel._haemolynx_values()[channel_name] == 1


def test_a_channel_set_before_its_file_is_kept_and_then_named(panel, tmp_path):
    rows = panel._haemolynx_rows()

    rows["raw_image_channel"].value = 1

    assert rows["raw_image_channel"].value == 1
    assert _labels(rows["raw_image_channel"])[-1] == "C2"

    rows["raw_image_path"].value = _composite(tmp_path / "composite.tif")

    assert rows["raw_image_channel"].value == 1
    assert _labels(rows["raw_image_channel"]) == [NO_CHANNEL_CHOOSE, "C1 (red)", "C2 (green)"]


def test_a_channel_the_new_file_lacks_is_kept_and_marked(panel, tmp_path):
    rows = panel._haemolynx_rows()
    rows["raw_image_channel"].value = 4

    rows["raw_image_path"].value = _composite(tmp_path / "composite.tif")

    assert rows["raw_image_channel"].value == 4
    assert _labels(rows["raw_image_channel"])[-1] == "C5 (not in this file)"


def test_loading_a_config_sets_the_channel_whatever_the_order(panel, tmp_path):
    from haemolynx.parsers import dump_config
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        measure_diameters_from_raw_image=True,
        raw_image_stain="endothelial",
        raw_image_path=str(_composite(tmp_path / "composite.tif")),
        raw_image_channel=1,
    )
    config = tmp_path / "channel.yaml"
    dump_config(config, schema, values=values)

    panel._haemolynx_load_config(config)

    assert panel._haemolynx_values()["raw_image_channel"] == 1
    assert _labels(panel._haemolynx_rows()["raw_image_channel"])[-1] == "C2 (green)"


def test_the_input_tabs_raw_data_channel_is_the_settings_own_row(panel):
    """The check and the optimiser read the one raw channel: no mirror to
    fall out of step."""
    assert panel._haemolynx_raw_channel_row is panel._haemolynx_rows()["raw_image_channel"]
    assert panel._haemolynx_raw_channel_row.label == "Raw data channel"
