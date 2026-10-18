"""When the panel searches for vessel masks, and which rows it fills
(`gui/vessel_mask_search.py`)."""
from __future__ import annotations

from pathlib import Path

from haemolynx.gui.vessel_mask_search import (
    TRIGGER_SETTINGS,
    fill_vessel_mask_rows,
    roles_searched,
    roles_to_search,
)

#: The vessel-mask settings as a fresh panel holds them, with both roles off.
OFF = {
    "automated_vessel_assignment": False,
    "use_large_vessel_masks": False,
    "use_small_vessel_masks_for_boundary_assignment": False,
    "use_ilastik_large_vessel_segmentation": False,
    "use_ilastik_small_vessel_segmentation": False,
    "input_path": None,
    "raw_image_path": None,
    "large_arteriole_mask_path": "images/large_arteriole_mask.tif",
    "large_venule_mask_path": None,
    "small_arteriole_mask_path": "images/small_arteriole_mask.tif",
    "small_venule_mask_path": None,
}
LARGE_ON = {**OFF, "automated_vessel_assignment": True, "use_large_vessel_masks": True}
BOTH_ON = {**LARGE_ON, "use_small_vessel_masks_for_boundary_assignment": True}


def _files(folder: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def test_every_trigger_is_a_setting_the_search_reads():
    assert set(TRIGGER_SETTINGS) == {
        "automated_vessel_assignment",
        "use_large_vessel_masks",
        "use_small_vessel_masks_for_boundary_assignment",
        "use_ilastik_large_vessel_segmentation",
        "use_ilastik_small_vessel_segmentation",
        "input_path",
        "raw_image_path",
    }


def test_a_role_is_searched_only_in_use_and_read_from_files():
    assert roles_searched(OFF) == ()
    assert roles_searched(LARGE_ON) == ("large",)
    assert roles_searched(BOTH_ON) == ("large", "small")
    # A switch left ticked under an unticked automated_vessel_assignment is off.
    assert roles_searched({**BOTH_ON, "automated_vessel_assignment": False}) == ()
    # With ilastik making the masks, the mask rows are not read.
    assert roles_searched({**BOTH_ON, "use_ilastik_large_vessel_segmentation": True}) == ("small",)


def test_a_role_coming_into_use_is_searched_for_and_no_other():
    assert roles_to_search(OFF, LARGE_ON) == ("large",)
    assert roles_to_search(LARGE_ON, BOTH_ON) == ("small",)
    assert roles_to_search(BOTH_ON, BOTH_ON) == ()
    assert roles_to_search(BOTH_ON, LARGE_ON) == ()
    # Ticking the gate brings back a role whose switch was left ticked.
    hidden = {**BOTH_ON, "automated_vessel_assignment": False}
    assert roles_to_search(hidden, BOTH_ON) == ("large", "small")
    # Unticking ilastik makes the role's mask rows the ones read.
    ilastik = {**LARGE_ON, "use_ilastik_large_vessel_segmentation": True}
    assert roles_to_search(ilastik, LARGE_ON) == ("large",)


def test_a_new_input_image_searches_again_for_every_role_in_use(tmp_path):
    before = {**BOTH_ON, "input_path": tmp_path / "a.tif"}
    assert roles_to_search(before, {**before, "input_path": tmp_path / "b.tif"}) == (
        "large",
        "small",
    )
    assert roles_to_search(before, {**before, "raw_image_path": tmp_path / "raw.tif"}) == (
        "large",
        "small",
    )
    assert roles_to_search(before, {**before, "input_path": str(tmp_path / "a.tif")}) == ()
    off = {**OFF, "input_path": tmp_path / "a.tif"}
    assert roles_to_search(off, {**off, "input_path": tmp_path / "b.tif"}) == ()


def test_found_masks_fill_the_rows_that_name_no_file(tmp_path):
    image, art, ven, small = _files(
        tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_large_venule.tif", "E14_small.tif"
    )
    settings = {**BOTH_ON, "input_path": image}
    fill = fill_vessel_mask_rows(settings, ("large", "small"))
    assert fill.values == {
        "large_arteriole_mask_path": art,
        "large_venule_mask_path": ven,
        "small_arteriole_mask_path": small,
        "small_venule_mask_path": None,
    }
    report = "\n".join(fill.lines)
    assert "E14_large_arteriole.tif" in report and "E14_large_venule.tif" in report
    assert "global mask of every small vessel" in report


def test_a_file_the_user_chose_keeps_its_whole_role(tmp_path):
    image, _art, _ven = _files(
        tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_large_venule.tif"
    )
    chosen_folder = tmp_path / "chosen"
    chosen_folder.mkdir()
    (chosen,) = _files(chosen_folder, "my_venules.tif")
    settings = {**LARGE_ON, "input_path": image, "large_venule_mask_path": chosen}
    fill = fill_vessel_mask_rows(settings, ("large",))
    assert fill.values == {}
    assert "Kept the large vessel masks already chosen" in fill.lines[0]


def test_a_row_still_holding_what_the_panel_filled_is_replaced(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _image_a, art_a, ven_a = _files(first, "A.tif", "A_large_arteriole.tif", "A_large_venule.tif")
    image_b, global_b = _files(second, "B.tif", "B_large_vessels.tif")
    filled = {"large_arteriole_mask_path": art_a, "large_venule_mask_path": ven_a}
    settings = {**LARGE_ON, "input_path": image_b, **filled}
    fill = fill_vessel_mask_rows(settings, ("large",), filled)
    assert fill.values == {"large_arteriole_mask_path": global_b, "large_venule_mask_path": None}
    # The same rows, chosen by the user rather than filled, stay.
    assert fill_vessel_mask_rows(settings, ("large",)).values == {}


def test_a_row_already_holding_the_found_file_is_no_conflict(tmp_path):
    image, art, ven = _files(
        tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_large_venule.tif"
    )
    settings = {**LARGE_ON, "input_path": image, "large_arteriole_mask_path": art}
    assert fill_vessel_mask_rows(settings, ("large",)).values == {
        "large_arteriole_mask_path": art,
        "large_venule_mask_path": ven,
    }


def test_a_lone_arteriole_mask_says_the_run_will_read_it_as_global(tmp_path):
    image, art = _files(tmp_path, "E14.tif", "E14_large_arteriole.tif")
    fill = fill_vessel_mask_rows({**LARGE_ON, "input_path": image}, ("large",))
    assert fill.values == {"large_arteriole_mask_path": art}
    assert "reads it as a global mask of every large vessel" in fill.lines[0]


def test_masks_naming_only_a_vessel_type_fill_the_large_rows_and_say_so(tmp_path):
    image, art, ven = _files(
        tmp_path, "E14_seg.tiff", "E14_seg_arteriole.tiff", "E14_seg_Vein.tiff"
    )
    fill = fill_vessel_mask_rows({**BOTH_ON, "input_path": image}, ("large", "small"))
    assert fill.values == {"large_arteriole_mask_path": art, "large_venule_mask_path": ven}
    report = "\n".join(fill.lines)
    assert "read as the large ones" in report
    assert "No small vessel masks found" in report


def test_tied_files_are_named_and_no_row_is_filled(tmp_path):
    image, *_ = _files(
        tmp_path, "E14.tif", "E14_large_venule.tif", "E14_large_venule_mask.tif"
    )
    fill = fill_vessel_mask_rows({**LARGE_ON, "input_path": image}, ("large",))
    assert fill.values == {}
    assert any(
        "E14_large_venule.tif, E14_large_venule_mask.tif" in line and "choose one" in line
        for line in fill.lines
    )


def test_nothing_found_or_no_image_is_reported(tmp_path):
    (image,) = _files(tmp_path, "E14.tif")
    fill = fill_vessel_mask_rows({**LARGE_ON, "input_path": image}, ("large",))
    assert fill.values == {}
    assert fill.lines[0].startswith("No large vessel masks found")

    fill = fill_vessel_mask_rows(LARGE_ON, ("large",))
    assert fill.values == {}
    assert fill.lines == ("No input image to look beside for the large vessel masks.",)

    assert fill_vessel_mask_rows({**LARGE_ON, "input_path": image}, ()).lines == ()


def test_the_raw_image_is_looked_beside_when_the_segmented_image_is_elsewhere(tmp_path):
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    raw, art, ven = _files(raw_folder, "E14.tif", "E14_large_artery.tif", "E14_large_vein.tif")
    settings = {**LARGE_ON, "input_path": tmp_path / "out" / "E14_segmented.tif", "raw_image_path": raw}
    assert fill_vessel_mask_rows(settings, ("large",)).values == {
        "large_arteriole_mask_path": art,
        "large_venule_mask_path": ven,
    }
