"""The pericyte mask rows filled from a mask beside the input image, and kept
naming the same mask (``gui/pericyte_mask_search.py``)."""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.gui.pericyte_mask_search import (
    MEASUREMENT_FIELD,
    find_pericyte_mask,
    perturbation_field,
    perturbation_index,
    update_pericyte_mask_rows,
)

ENTRY = perturbation_field(0)
OTHER_ENTRY = perturbation_field(1)


def _touch(folder: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        paths.append(path)
    return paths


@pytest.fixture
def folder(tmp_path):
    image, mask = _touch(tmp_path, "E14_MCA_segmented.tif", "E14_MCA_pericytes.tif")
    _touch(tmp_path, "E14_MCA_large_arterioles.tif", "notes.txt")
    return tmp_path, image, mask


# --- finding the mask --------------------------------------------------------------


def test_a_mask_naming_pericytes_beside_the_image_is_found(folder):
    _tmp, image, mask = folder
    assert find_pericyte_mask([image, None]).path == mask


@pytest.mark.parametrize("name", ["pericyte_mask.tif", "NG2_Pericytes.tiff", "PERICYTE.h5"])
def test_pericyte_or_pericytes_in_any_case_and_any_mask_format(tmp_path, name):
    (image,) = _touch(tmp_path, "image.tif")
    (mask,) = _touch(tmp_path, name)
    assert find_pericyte_mask([image]).path == mask


def test_a_file_that_is_not_a_mask_or_is_the_image_itself_is_not_found(tmp_path):
    (image,) = _touch(tmp_path, "pericytes_raw.tif")
    _touch(tmp_path, "pericyte_counts.csv")
    found = find_pericyte_mask([image])
    assert found.path is None and found.tied == ()


def test_of_several_the_one_sharing_the_images_name_is_chosen(tmp_path):
    image, own, _other = _touch(tmp_path, "E14_MCA.tif", "E14_MCA_pericytes.tif", "E15_pericytes.tif")
    assert find_pericyte_mask([image]).path == own


def test_a_tie_is_named_and_none_chosen(tmp_path):
    image, first, second = _touch(tmp_path, "E14.tif", "E14_pericytes_a.tif", "E14_pericytes_b.tif")
    found = find_pericyte_mask([image])
    assert found.path is None and set(found.tied) == {first, second}


def test_the_raw_images_folder_is_looked_in_too(tmp_path):
    (image,) = _touch(tmp_path / "seg", "image.tif")
    raw, mask = _touch(tmp_path / "raw", "image_raw.tif", "image_pericytes.tif")
    assert find_pericyte_mask([image, raw]).path == mask


def test_row_keys_name_their_entry():
    assert perturbation_index(perturbation_field(3)) == 3
    assert perturbation_index(MEASUREMENT_FIELD) is None


# --- filling the rows ------------------------------------------------------------------


def test_a_row_coming_into_use_is_filled_with_the_mask_found(folder):
    _tmp, image, mask = folder
    fill = update_pericyte_mask_rows({}, {MEASUREMENT_FIELD: None}, {}, references=[image])
    assert fill.values == {MEASUREMENT_FIELD: mask}
    assert "found beside the input image" in fill.lines[0]


def test_both_rows_coming_into_use_are_filled(folder):
    _tmp, image, mask = folder
    fill = update_pericyte_mask_rows(
        {}, {MEASUREMENT_FIELD: None, ENTRY: None}, {}, references=[image]
    )
    assert fill.values == {MEASUREMENT_FIELD: mask, ENTRY: mask}


def test_a_row_already_in_use_is_not_searched_again(folder):
    _tmp, image, _mask = folder
    fill = update_pericyte_mask_rows({MEASUREMENT_FIELD: None}, {MEASUREMENT_FIELD: None}, {}, references=[image])
    assert fill.values == {} and fill.lines == ()


def test_nothing_found_says_so(tmp_path):
    (image,) = _touch(tmp_path, "image.tif")
    fill = update_pericyte_mask_rows({}, {MEASUREMENT_FIELD: None}, {}, references=[image])
    assert fill.values == {} and "No pericyte mask found" in fill.lines[0]


def test_a_mask_entered_into_one_row_fills_the_other(folder):
    tmp, image, found = folder
    (chosen,) = _touch(tmp, "elsewhere/my_cells.tif")
    before = {MEASUREMENT_FIELD: found, ENTRY: found}
    filled = {MEASUREMENT_FIELD: found, ENTRY: found}
    fill = update_pericyte_mask_rows(before, {MEASUREMENT_FIELD: chosen, ENTRY: found}, filled, references=[image])
    # The user's mask overrides the one the panel found.
    assert fill.values == {ENTRY: chosen}
    assert fill.released == (MEASUREMENT_FIELD,)


def test_a_mask_entered_into_a_perturbation_fills_the_measurement_row(folder):
    tmp, image, _found = folder
    (chosen,) = _touch(tmp, "elsewhere/cells.tif")
    fill = update_pericyte_mask_rows(
        {MEASUREMENT_FIELD: None, ENTRY: None}, {MEASUREMENT_FIELD: None, ENTRY: chosen}, {}, references=[image]
    )
    assert fill.values == {MEASUREMENT_FIELD: chosen}


def test_a_row_holding_the_users_own_mask_is_never_replaced(folder):
    tmp, image, _found = folder
    mine, theirs = _touch(tmp, "elsewhere/mine.tif", "elsewhere/theirs.tif")
    # The perturbation's row holds a mask the user chose (not one the panel filled).
    fill = update_pericyte_mask_rows(
        {MEASUREMENT_FIELD: None, ENTRY: theirs}, {MEASUREMENT_FIELD: mine, ENTRY: theirs}, {},
        references=[image],
    )
    assert fill.values == {}


def test_a_row_coming_into_use_takes_the_mask_the_user_chose_over_the_one_found(folder):
    tmp, image, _found = folder
    (mine,) = _touch(tmp, "elsewhere/mine.tif")
    fill = update_pericyte_mask_rows(
        {MEASUREMENT_FIELD: mine}, {MEASUREMENT_FIELD: mine, ENTRY: None}, {}, references=[image]
    )
    assert fill.values == {ENTRY: mine}
    assert "already chosen" in fill.lines[0]


def test_a_new_input_image_replaces_what_the_panel_filled(tmp_path):
    old_image, old_mask = _touch(tmp_path / "old", "a.tif", "a_pericytes.tif")
    new_image, new_mask = _touch(tmp_path / "new", "b.tif", "b_pericytes.tif")
    rows = {MEASUREMENT_FIELD: old_mask, ENTRY: old_mask}
    fill = update_pericyte_mask_rows(rows, rows, dict(rows), references=[new_image], references_changed=True)
    assert fill.values == {MEASUREMENT_FIELD: new_mask, ENTRY: new_mask}


def test_every_entry_in_use_follows_a_mask_entered_into_one(folder):
    tmp, image, found = folder
    (chosen,) = _touch(tmp, "elsewhere/cells.tif")
    before = {MEASUREMENT_FIELD: found, ENTRY: found, OTHER_ENTRY: found}
    after = {**before, OTHER_ENTRY: chosen}
    fill = update_pericyte_mask_rows(before, after, dict(before), references=[image])
    assert fill.values == {MEASUREMENT_FIELD: chosen, ENTRY: chosen}


def test_a_partly_typed_path_fills_nothing_yet(folder):
    tmp, image, _found = folder
    fill = update_pericyte_mask_rows(
        {MEASUREMENT_FIELD: None, ENTRY: None}, {MEASUREMENT_FIELD: tmp / "elsewhere/ce", ENTRY: None}, {},
        references=[image],
    )
    assert fill.values == {} and fill.released == (MEASUREMENT_FIELD,)
