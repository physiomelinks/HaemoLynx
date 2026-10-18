"""The panel fills the vessel-mask rows from the masks beside the input image
when a mask switch is ticked, and leaves a loaded config's alone."""
from __future__ import annotations

from pathlib import Path

import pytest

napari = pytest.importorskip("napari")
pytest.importorskip("magicgui")

from haemolynx.gui._widget import settings_widget  # noqa: E402

pytestmark = pytest.mark.gui


def _files(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _path(value) -> Path | None:
    return None if value in (None, "") else Path(value)


def test_ticking_the_mask_switches_fills_the_rows_from_the_input_folder(
    make_napari_viewer, tmp_path
):
    make_napari_viewer()
    panel = settings_widget()
    rows = panel._haemolynx_rows()
    image, art, ven, small = _files(
        tmp_path,
        "E14_MCA_segmented.tif",
        "E14_MCA_Large_Arterioles.tif",
        "E14_MCA_large_veins.tif",
        "E14_MCA_small_vessels.tif",
    )
    rows["input_path"].value = image
    rows["automated_vessel_assignment"].value = True
    values = panel._haemolynx_values()
    assert _path(values["large_arteriole_mask_path"]) != art

    rows["use_large_vessel_masks"].value = True
    values = panel._haemolynx_values()
    assert _path(values["large_arteriole_mask_path"]) == art
    assert _path(values["large_venule_mask_path"]) == ven
    assert "E14_MCA_Large_Arterioles.tif" in panel._haemolynx_report()

    rows["use_small_vessel_masks_for_boundary_assignment"].value = True
    values = panel._haemolynx_values()
    assert _path(values["small_arteriole_mask_path"]) == small
    assert _path(values["small_venule_mask_path"]) is None
    assert "global mask of every small vessel" in panel._haemolynx_report()


def test_a_new_input_image_replaces_what_the_panel_filled_but_not_a_chosen_file(
    make_napari_viewer, tmp_path
):
    make_napari_viewer()
    panel = settings_widget()
    rows = panel._haemolynx_rows()
    image_a, art_a, _ven_a = _files(
        tmp_path / "a", "A.tif", "A_large_arteriole.tif", "A_large_venule.tif"
    )
    image_b, global_b = _files(tmp_path / "b", "B.tif", "B_large.tif")
    rows["input_path"].value = image_a
    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    assert _path(panel._haemolynx_values()["large_arteriole_mask_path"]) == art_a

    rows["input_path"].value = image_b
    values = panel._haemolynx_values()
    assert _path(values["large_arteriole_mask_path"]) == global_b
    assert _path(values["large_venule_mask_path"]) is None

    (chosen,) = _files(tmp_path / "chosen", "mine.tif")
    rows["large_arteriole_mask_path"].value = chosen
    rows["input_path"].value = image_a
    assert _path(panel._haemolynx_values()["large_arteriole_mask_path"]) == chosen
    assert "Kept the large vessel masks already chosen" in panel._haemolynx_report()


def test_loading_a_config_keeps_the_mask_paths_it_names(make_napari_viewer, tmp_path):
    """Even paths naming no file, which a search would fill: the config's
    ilastik switch is set after its mask paths, so without the pause the rows
    would come into use, and be searched, once the paths were already in."""
    make_napari_viewer()
    panel = settings_widget()
    rows = panel._haemolynx_rows()
    image, _art, _ven = _files(
        tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_large_venule.tif"
    )
    rows["input_path"].value = image
    rows["automated_vessel_assignment"].value = True
    rows["use_large_vessel_masks"].value = True
    rows["use_ilastik_large_vessel_segmentation"].value = True

    missing_art = tmp_path / "gone" / "art.tif"
    missing_ven = tmp_path / "gone" / "ven.tif"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"input_path: {image.as_posix()}\n"
        "automated_vessel_assignment: true\n"
        "use_large_vessel_masks: true\n"
        "use_ilastik_large_vessel_segmentation: false\n"
        f"large_arteriole_mask_path: {missing_art.as_posix()}\n"
        f"large_venule_mask_path: {missing_ven.as_posix()}\n",
        encoding="utf-8",
    )
    panel._haemolynx_load_config(config)
    assert "Could not load" not in panel._haemolynx_report()
    values = panel._haemolynx_values()
    assert values["use_ilastik_large_vessel_segmentation"] is False
    assert _path(values["large_arteriole_mask_path"]) == missing_art
    assert _path(values["large_venule_mask_path"]) == missing_ven
