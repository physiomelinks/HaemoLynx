"""Finding the vessel masks beside the input image by file name
(`io/vessel_mask_discovery.py`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from haemolynx.io import FoundVesselMasks, find_vessel_masks
from haemolynx.io.vessel_mask_discovery import file_name_words, read_mask_name


def _files(folder: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def test_words_split_at_punctuation_spaces_and_capitals():
    assert file_name_words("E14.5_MCA LargeArterioles-mask") == [
        "e", "14", "5", "mca", "large", "arterioles", "mask",
    ]
    assert file_name_words("SMALL_VEINS") == ["small", "veins"]


def test_all_four_masks_are_found_beside_the_segmented_image(tmp_path):
    image, large_art, large_ven, small_art, small_ven = _files(
        tmp_path,
        "E14_MCA_segmented.tif",
        "E14_MCA_large_arterioles.tif",
        "E14_MCA_large_veins.tif",
        "E14_MCA_small_artery_mask.tif",
        "E14_MCA_small_venules.tif",
    )
    found = find_vessel_masks([image])
    assert found["large"] == FoundVesselMasks("large", arteriole=large_art, venule=large_ven)
    assert found["small"] == FoundVesselMasks("small", arteriole=small_art, venule=small_ven)
    assert found["large"].settings() == {
        "large_arteriole_mask_path": large_art,
        "large_venule_mask_path": large_ven,
    }
    assert found["small"].settings() == {
        "small_arteriole_mask_path": small_art,
        "small_venule_mask_path": small_ven,
    }


@pytest.mark.parametrize("word", ["arteriole", "Arterioles", "ARTERY", "arteries"])
def test_every_arteriole_word_names_an_arteriole_mask(tmp_path, word):
    image, mask = _files(tmp_path, "img.tif", f"img_large_{word}.tif")
    assert find_vessel_masks([image], ["large"])["large"].arteriole == mask


@pytest.mark.parametrize("word", ["venule", "Venules", "VEIN", "veins"])
def test_every_venule_word_names_a_venule_mask(tmp_path, word):
    image, mask = _files(tmp_path, "img.tif", f"img_small_{word}.tif")
    assert find_vessel_masks([image], ["small"])["small"].venule == mask


def test_a_size_with_no_vessel_type_is_a_global_mask(tmp_path):
    image, large, small = _files(
        tmp_path, "E14_seg.tif", "E14_large_vessels.tif", "E14_Small.tiff"
    )
    found = find_vessel_masks([image])
    assert found["large"].global_mask == large and found["large"].is_global
    assert found["small"].global_mask == small and found["small"].is_global
    # The venule row is emptied: that is what makes the run read it as global.
    assert found["large"].settings() == {
        "large_arteriole_mask_path": large,
        "large_venule_mask_path": None,
    }


def test_a_size_with_both_vessel_types_is_a_global_mask(tmp_path):
    image, mask = _files(tmp_path, "E14.tif", "E14_large_arteries_and_veins.tif")
    found = find_vessel_masks([image], ["large"])["large"]
    assert found.global_mask == mask and found.arteriole is None and found.venule is None


def test_a_pair_is_read_rather_than_a_global_mask_beside_it(tmp_path):
    image, global_mask, art, ven = _files(
        tmp_path, "E14.tif", "E14_large.tif", "E14_large_arteriole.tif", "E14_large_venule.tif"
    )
    found = find_vessel_masks([image], ["large"])["large"]
    assert found.global_mask == global_mask
    assert found.is_pair and not found.is_global
    assert found.settings() == {"large_arteriole_mask_path": art, "large_venule_mask_path": ven}


def test_a_lone_arteriole_or_venule_mask_fills_only_its_own_row(tmp_path):
    image, art, ven = _files(tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_small_vein.tif")
    found = find_vessel_masks([image])
    assert found["large"].settings() == {"large_arteriole_mask_path": art}
    assert found["small"].settings() == {"small_venule_mask_path": ven}


def test_a_vessel_type_with_no_size_is_a_large_mask_never_a_small_one(tmp_path):
    image, art, ven = _files(tmp_path, "E14.tif", "E14_arteriole.tif", "E14_Vein.tiff")
    found = find_vessel_masks([image])
    assert found["large"] == FoundVesselMasks(
        "large", arteriole=art, venule=ven, size_named=False
    )
    assert found["large"].settings() == {
        "large_arteriole_mask_path": art,
        "large_venule_mask_path": ven,
    }
    assert not found["small"].found_any


def test_both_vessel_types_with_no_size_are_a_global_large_mask(tmp_path):
    image, mask = _files(tmp_path, "E14.tif", "E14_arteries_and_veins.tif")
    found = find_vessel_masks([image], ["large"])["large"]
    assert found.global_mask == mask and not found.size_named


@pytest.mark.parametrize(
    "named",
    ["E14_large_arteriole.tif", "E14_large_venules.tif", "E14_large_vessels.tif"],
)
def test_any_file_saying_large_puts_the_type_only_names_aside(tmp_path, named):
    """As long as no file says 'large' (Harvey, 2026-10-18): one that does,
    of whatever kind, decides the large masks alone."""
    image, _art, _ven, explicit = _files(
        tmp_path, "E14.tif", "E14_arteriole.tif", "E14_vein.tif", named
    )
    found = find_vessel_masks([image], ["large"])["large"]
    assert found.size_named
    assert [path for path in (found.arteriole, found.venule, found.global_mask) if path] == [
        explicit
    ]


def test_a_file_saying_small_leaves_the_type_only_names_large(tmp_path):
    image, art, ven, small = _files(
        tmp_path, "E14.tif", "E14_arteriole.tif", "E14_vein.tif", "E14_small_vessels.tif"
    )
    found = find_vessel_masks([image])
    assert (found["large"].arteriole, found["large"].venule) == (art, ven)
    assert found["small"].global_mask == small and found["small"].size_named


def test_a_name_with_neither_size_nor_type_is_no_mask(tmp_path):
    image, *_ = _files(tmp_path, "E14.tif", "E14_mask.tif", "E14_vessels.tif")
    found = find_vessel_masks([image])
    assert not found["large"].found_any and not found["small"].found_any


def test_both_sizes_in_one_name_is_no_mask(tmp_path):
    image, _ = _files(tmp_path, "E14.tif", "E14_large_small_arteriole.tif")
    found = find_vessel_masks([image])
    assert not found["large"].found_any and not found["small"].found_any


def test_another_images_mask_is_not_taken(tmp_path):
    image, _other = _files(tmp_path, "E14_MCA_segmented.tif", "E15_MCA_large_arteriole.tif")
    assert not find_vessel_masks([image], ["large"])["large"].found_any

    (own,) = _files(tmp_path, "E14_MCA_large_arteriole.tif")
    assert find_vessel_masks([image], ["large"])["large"].arteriole == own


def test_a_mask_named_only_by_its_role_belongs_to_the_folder(tmp_path):
    image, mask = _files(tmp_path, "E14_MCA_segmented.tif", "large_arteriole_mask.tif")
    assert find_vessel_masks([image], ["large"])["large"].arteriole == mask


def test_the_mask_sharing_the_image_name_furthest_is_taken(tmp_path):
    image, _shorter, closer = _files(
        tmp_path,
        "E14_MCA_C1_seg.tif",
        "E14_large_arteriole.tif",
        "E14_MCA_large_arteriole.tif",
    )
    assert find_vessel_masks([image], ["large"])["large"].arteriole == closer


def test_role_words_in_the_image_name_are_not_read_as_the_masks_role(tmp_path):
    """Only what a name adds to the image's says which mask it is."""
    image, _same_words, small_ven, large_art = _files(
        tmp_path,
        "large_field_segmented.tif",
        "large_field_mask.tif",
        "large_field_small_veins.tif",
        "large_field_large_arteriole.tif",
    )
    found = find_vessel_masks([image])
    assert found["small"].venule == small_ven
    assert found["large"].arteriole == large_art
    assert found["large"].global_mask is None


def test_the_input_itself_and_other_file_types_are_never_masks(tmp_path):
    image, *_ = _files(
        tmp_path,
        "E14_large_vessels.tif",
        "E14_large_arteriole_classifier.ilp",
        "E14_large_arteriole.png",
        "E14_large_arteriole_Probabilities.h5",
        "E14_large_arteriole_raw.tif",
    )
    (tmp_path / "E14_large_venule.tif").mkdir()
    found = find_vessel_masks([image], ["large"])["large"]
    assert not found.found_any
    assert found.ambiguous == {}


def test_tied_files_are_left_unchosen_unless_one_shares_the_images_format(tmp_path):
    image, first, second = _files(
        tmp_path, "E14.tif", "E14_large_arteriole.tif", "E14_large_arteriole_mask.tif"
    )
    found = find_vessel_masks([image], ["large"])["large"]
    assert found.arteriole is None
    assert found.ambiguous == {"arteriole": (first, second)}
    assert found.settings() == {}

    tiff, _h5 = _files(tmp_path, "E14_small_vein.tif", "E14_small_vein.h5")
    assert find_vessel_masks([image], ["small"])["small"].venule == tiff


def test_masks_beside_the_raw_image_are_found_too(tmp_path):
    """With ilastik segmenting, the segmented image lives in its output folder."""
    raw_folder = tmp_path / "raw"
    out_folder = tmp_path / "ilastik"
    raw_folder.mkdir()
    out_folder.mkdir()
    raw, art, ven = _files(
        raw_folder, "E14.tif", "E14_large_arteriole_segmented.tif", "E14_large_venule_segmented.tif"
    )
    segmented = out_folder / "E14_segmented.tif"
    found = find_vessel_masks([segmented, raw], ["large"])["large"]
    assert (found.arteriole, found.venule) == (art, ven)


def test_the_users_e14p5_exports_are_read_as_the_large_masks(tmp_path):
    """Harvey's E14.5 masks as named on disk: a vessel type, no size."""
    stem = (
        "Composite_06082026_E14p5_clnd5_25x_940nm_1040_texasred3kkda_"
        "zstack_spot1_1p5z_MCAregion_Simple Segmentation"
    )
    image, art, ven = _files(
        tmp_path, f"{stem}.tiff", f"{stem}_arteriole.tiff", f"{stem}_Vein.tiff"
    )
    found = find_vessel_masks([image])
    assert (found["large"].arteriole, found["large"].venule) == (art, ven)
    assert not found["small"].found_any


def test_the_users_ilastik_export_names_are_read(tmp_path):
    stem = (
        "Composite_06082026_E14p5_clnd5_25x_940nm_1040_texasred3kkda_"
        "zstack_spot1_1p5z_MCAregion_Simple Segmentation"
    )
    image, art, ven = _files(
        tmp_path, f"{stem}.tiff", f"{stem}_large_arteriole.tiff", f"{stem}_Large_Vein.tiff"
    )
    raw = tmp_path / f"C1-{stem.replace('_Simple Segmentation', '')}.tif"
    raw.write_bytes(b"")
    found = find_vessel_masks([image, raw], ["large"])["large"]
    assert (found.arteriole, found.venule) == (art, ven)


def test_unset_references_and_missing_folders_find_nothing(tmp_path):
    found = find_vessel_masks([None, "", tmp_path / "missing" / "image.tif"])
    assert set(found) == {"large", "small"}
    assert not any(result.found_any for result in found.values())


def test_an_unknown_size_is_refused():
    with pytest.raises(ValueError, match="large' or 'small"):
        find_vessel_masks([], ["medium"])


def test_read_mask_name_ranks_by_the_shared_start_then_shared_words_then_format():
    assert read_mask_name("a/E14_MCA_large_vein.tif", "a/E14_MCA_seg.tif") == (
        "large", "venule", (3, 3, 1), True,
    )
    assert read_mask_name("a/MCA_large_vein.h5", "a/E14_MCA_seg.tif") == (
        "large", "venule", (0, 1, 0), True,
    )
    assert read_mask_name("a/E14_MCA_vein.tif", "a/E14_MCA_seg.tif") == (
        "large", "venule", (3, 3, 1), False,
    )
    assert read_mask_name("a/E14_MCA_C2_large_vein.tif", "a/E14_MCA_seg.tif") is None
