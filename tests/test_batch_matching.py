"""Matching a batch run's images to their raw images and masks by file name
(`io/batch_matching.py`)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from haemolynx.io import BatchFolders, batch_inputs, match_batch
from haemolynx.io.batch_matching import (
    RAW_ROLE,
    choose_alike,
    is_raw_of,
    mask_role,
    read_raw_name,
)

ILASTIK = "Simple Segmentation"


def _files(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


# --- which files are the batch's images ------------------------------------


def test_masks_raw_images_and_probability_maps_in_one_folder_are_not_inputs(tmp_path):
    first, second, *_others = _files(
        tmp_path,
        f"E14_MCA_{ILASTIK}.tiff",
        f"E15_MCA_{ILASTIK}.tiff",
        "E14_MCA.tif",
        "C1-E15_MCA.tif",
        f"E14_MCA_{ILASTIK}_arteriole.tiff",
        f"E14_MCA_{ILASTIK}_Vein.tiff",
        "E15_MCA_small_vessels.tif",
        "E14_MCA_Probabilities.h5",
        "notes.txt",
        "E14_MCA_classifier.ilp",
    )
    assert batch_inputs(tmp_path) == (first, second)


def test_a_folder_of_segmented_images_alone_is_every_image(tmp_path):
    images = _files(tmp_path, "E14.tif", "E15.tif", "E16.h5")
    assert batch_inputs(tmp_path) == tuple(images)


def test_a_file_saying_raw_is_never_an_input_unless_it_is_a_segmentation(tmp_path):
    raw, segmented = _files(tmp_path, "E14_raw.tif", f"E14_raw_{ILASTIK}.tiff")
    assert batch_inputs(tmp_path) == (segmented,)
    assert is_raw_of(raw, segmented)
    assert not is_raw_of(segmented, raw)


def test_two_images_that_read_as_each_others_masks_are_both_inputs(tmp_path):
    images = _files(tmp_path, "large_field_seg.tif", "small_field_seg.tif")
    assert batch_inputs(tmp_path) == tuple(images)


def test_a_missing_folder_has_no_inputs(tmp_path):
    assert batch_inputs(tmp_path / "missing") == ()


# --- the raw image ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw_name",
    [
        "E14_MCA.tif",  # ilastik: the segmentation is the raw name with words added
        "E14_MCA_raw.tif",
        "C1-E14_MCA.tif",  # ImageJ's split channel
        "E14_MCA_ch2.tif",
        "E14_MCA_unsegmented.h5",
    ],
)
def test_the_raw_image_is_matched_from_its_own_folder(tmp_path, raw_name):
    _files(tmp_path / "seg", f"E14_MCA_{ILASTIK}.tiff")
    raw, _other = _files(tmp_path / "raw", raw_name, "E15_MCA.tif")
    (match,) = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"))
    assert match.raw == raw
    assert match.ok


def test_a_blank_raw_folder_is_the_segmented_folder(tmp_path):
    segmented, raw = _files(tmp_path, f"E14_{ILASTIK}.tiff", "E14.tif")
    (match,) = match_batch(BatchFolders(tmp_path))
    assert (match.segmented, match.raw) == (segmented, raw)


def test_another_images_raw_image_is_not_taken(tmp_path):
    _files(tmp_path / "seg", "E14_MCA_seg.tif")
    _files(
        tmp_path / "raw", "E15_MCA.tif", "E14_ACA.tif", "E14_MCA_seg.tif", "E14_large_vein.tif"
    )
    (match,) = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"))
    assert match.raw is None
    assert match.ok  # the raw image is optional
    assert match.notes == (f"No raw image found in {tmp_path / 'raw'}.",)


def test_the_raw_image_sharing_the_name_furthest_is_taken(tmp_path):
    _files(tmp_path / "seg", "E14_MCA_C1_seg.tif")
    _shorter, closer, _other_channel = _files(
        tmp_path / "raw", "E14.tif", "E14_MCA_C1.tif", "E14_MCA_C2.tif"
    )
    (match,) = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"))
    assert match.raw == closer


def test_tied_raw_images_are_named_and_left_for_the_user(tmp_path):
    _files(tmp_path / "seg", f"E14_{ILASTIK}.tiff")
    first, second = _files(tmp_path / "raw", "C1-E14.tif", "C2-E14.tif")
    (match,) = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"))
    assert match.raw is None
    assert match.raw_ties == (first, second)
    assert match.tied_roles == (RAW_ROLE,)
    assert not match.ok
    assert "C1-E14.tif, C2-E14.tif" in match.problems[0]

    chosen = match.choose(RAW_ROLE, second)
    assert (chosen.raw, chosen.raw_ties, chosen.ok) == (second, (), True)
    with pytest.raises(ValueError, match="not one of the files tied"):
        match.choose(RAW_ROLE, tmp_path / "raw" / "other.tif")


def test_one_choice_resolves_every_images_tie_alike(tmp_path):
    _files(tmp_path / "seg", f"E14_{ILASTIK}.tiff", f"E15_{ILASTIK}.tiff")
    c1_e14, c2_e14, c1_e15, _c2_e15 = _files(
        tmp_path / "raw", "C1-E14.tif", "C2-E14.tif", "C1-E15.tif", "C2-E15.tif"
    )
    matches = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"))
    chosen = choose_alike(matches, RAW_ROLE, c1_e14, matches[0].raw_ties)
    assert [match.raw for match in chosen] == [c1_e14, c1_e15]
    assert all(match.ok for match in chosen)
    # Choosing the other channel chooses it everywhere.
    chosen = choose_alike(matches, RAW_ROLE, c2_e14, matches[0].raw_ties)
    assert [match.raw.name for match in chosen] == ["C2-E14.tif", "C2-E15.tif"]


def test_read_raw_name_refuses_segmentations_masks_and_other_images():
    segmented = "a/E14_MCA_seg.tif"
    assert read_raw_name("a/E14_MCA_mask.tif", segmented) is None
    assert read_raw_name("a/E14_MCA_large_vein.tif", segmented) is None
    assert read_raw_name("a/E14_MCA_Probabilities.h5", segmented) is None
    assert read_raw_name("a/E14_ACA.tif", segmented) is None
    assert read_raw_name("a/E14_MCA.tif", segmented) == (3, 3, 1, 1)
    assert read_raw_name("a/C2-E14_MCA.h5", segmented) == (3, 3, 0, 0)


# --- the masks --------------------------------------------------------------


def test_masks_in_folders_of_their_own_are_matched_to_each_image(tmp_path):
    e14, _e15 = _files(tmp_path / "seg", "E14_seg.tif", "E15_seg.tif")
    e14_art, e14_ven, e15_art, e15_ven = _files(
        tmp_path / "large",
        "E14_large_arteriole.tif",
        "E14_large_vein.tif",
        "E15_large_arteriole.tif",
        "E15_large_vein.tif",
    )
    e14_small, e15_small = _files(tmp_path / "small", "E14_small.tif", "E15_small_vessels.tif")
    folders = BatchFolders(tmp_path / "seg", large=tmp_path / "large", small=tmp_path / "small")
    first, second = match_batch(folders, ["large", "small"])
    assert first.settings() == {
        "input_path": e14,
        "raw_image_path": None,
        "large_arteriole_mask_path": e14_art,
        "large_venule_mask_path": e14_ven,
        "small_arteriole_mask_path": e14_small,
        "small_venule_mask_path": None,
    }
    assert (second.masks["large"].arteriole, second.masks["large"].venule) == (e15_art, e15_ven)
    assert second.masks["small"].global_mask == e15_small
    assert first.ok and second.ok
    assert any("global mask of every small vessel" in note for note in first.notes)


def test_everything_in_the_segmented_folder(tmp_path):
    """Harvey's E14.5 exports: ilastik names, a vessel type and no size."""
    stem = f"E14p5_MCAregion_{ILASTIK}"
    segmented, raw, art, ven = _files(
        tmp_path,
        f"{stem}.tiff",
        "C1-E14p5_MCAregion.tif",
        f"{stem}_arteriole.tiff",
        f"{stem}_Vein.tiff",
    )
    (match,) = match_batch(BatchFolders(tmp_path), ["large"])
    assert (match.segmented, match.raw) == (segmented, raw)
    assert (match.masks["large"].arteriole, match.masks["large"].venule) == (art, ven)
    assert match.ok
    assert any("read as the large ones" in note for note in match.notes)


def test_masks_beside_the_raw_image_are_found_only_when_its_folder_is_given(tmp_path):
    _files(tmp_path / "seg", "E14_seg.tif")
    _files(tmp_path / "raw", "E14.tif", "E14_large_arteriole.tif", "E14_large_vein.tif")
    (match,) = match_batch(BatchFolders(tmp_path / "seg", raw=tmp_path / "raw"), ["large"])
    assert not match.masks["large"].found_any
    assert match.problems == (f"No large vessel masks found in {tmp_path / 'seg'}.",)
    (match,) = match_batch(
        BatchFolders(tmp_path / "seg", raw=tmp_path / "raw", large=tmp_path / "raw"), ["large"]
    )
    assert match.ok


def test_mask_rows_nothing_was_found_for_are_emptied(tmp_path):
    (image,) = _files(tmp_path, "E14_seg.tif")
    (match,) = match_batch(BatchFolders(tmp_path), ["small"])
    assert match.settings() == {
        "input_path": image,
        "raw_image_path": None,
        "small_arteriole_mask_path": None,
        "small_venule_mask_path": None,
    }
    assert not match.ok


def test_sizes_not_searched_are_left_out_of_the_settings(tmp_path):
    _files(tmp_path, "E14_seg.tif", "E14_large_arteriole.tif")
    (match,) = match_batch(BatchFolders(tmp_path))
    assert match.masks == {}
    assert set(match.settings()) == {"input_path", "raw_image_path"}


def test_a_lone_venule_mask_stops_the_image(tmp_path):
    _files(tmp_path, "E14_seg.tif", "E14_large_vein.tif")
    (match,) = match_batch(BatchFolders(tmp_path), ["large"])
    assert match.problems == (
        "Found a large venule mask, E14_large_vein.tif, but no large arteriole mask "
        "to go with it.",
    )


def test_tied_masks_are_chosen_alike_across_the_batch(tmp_path):
    _files(tmp_path, "E14_seg.tif", "E15_seg.tif", "E14_large_vein.tif", "E15_large_vein.tif")
    e14_a, e14_b, e15_a, _e15_b = _files(
        tmp_path,
        "E14_large_arteriole.tif",
        "E14_large_arteriole_mask.tif",
        "E15_large_arteriole.tif",
        "E15_large_arteriole_mask.tif",
    )
    matches = match_batch(BatchFolders(tmp_path), ["large"])
    role = mask_role("large", "arteriole")
    assert matches[0].tied_roles == (role,)
    assert matches[0].ties(role) == (e14_a, e14_b)
    assert not matches[0].ok
    chosen = choose_alike(matches, role, e14_a, matches[0].ties(role))
    assert [match.masks["large"].arteriole for match in chosen] == [e14_a, e15_a]
    assert all(match.ok for match in chosen)


def test_one_mask_shared_by_several_images(tmp_path):
    first, second, shared = _files(
        tmp_path / "seg", "crop1_seg.tif", "crop2_seg.tif", "large_vessels.tif"
    )
    matches = match_batch(BatchFolders(tmp_path / "seg"), ["large"])
    assert [match.segmented for match in matches] == [first, second]
    assert [match.masks["large"].global_mask for match in matches] == [shared, shared]


def test_the_batchs_other_images_are_never_read_as_masks(tmp_path):
    _files(tmp_path, "large_field_seg.tif", "small_field_seg.tif")
    matches = match_batch(BatchFolders(tmp_path), ["large", "small"])
    assert len(matches) == 2
    assert not any(found.found_any for match in matches for found in match.masks.values())


def test_a_mask_folder_of_its_own_says_the_size(tmp_path):
    """Harvey's first batch (AIA_classified_IZ): ilastik exports whose names
    say 'large' and 'intermediate', not 'small', and segmented images whose
    names carry 'all_vessel', which the masks lack."""
    stems = ["28_3_d_contra_patella_cropped_hdf5", "28_3_d_ipsi_FITC", "29_1_b_contra_FITC"]
    for stem in stems:
        _files(tmp_path / "all", f"{stem}_{ILASTIK}_all_vessel.h5")
        _files(tmp_path / "large", f"{stem}_{ILASTIK}_large.h5")
        _files(tmp_path / "intermediate", f"{stem}_{ILASTIK}_intermediate.h5")
    folders = BatchFolders(
        tmp_path / "all", large=tmp_path / "large", small=tmp_path / "intermediate"
    )
    matches = match_batch(folders, ["large", "small"])
    assert [match.segmented.name.split(f"_{ILASTIK}")[0] for match in matches] == sorted(stems)
    for match in matches:
        stem = match.segmented.name.split(f"_{ILASTIK}")[0]
        assert match.ok, match.problems
        assert match.settings() == {
            "input_path": match.segmented,
            "raw_image_path": None,
            "large_arteriole_mask_path": tmp_path / "large" / f"{stem}_{ILASTIK}_large.h5",
            "large_venule_mask_path": None,
            "small_arteriole_mask_path": tmp_path / "intermediate" / f"{stem}_{ILASTIK}_intermediate.h5",
            "small_venule_mask_path": None,
        }


def test_a_file_in_an_own_folder_goes_to_the_image_sharing_most_of_its_name(tmp_path):
    _files(tmp_path / "seg", "A_1_seg.tif", "A_12_seg.tif")
    a1, a12, _nobody = _files(tmp_path / "small", "A_1_x.tif", "A_12_x.tif", "B_x.tif")
    matches = match_batch(BatchFolders(tmp_path / "seg", small=tmp_path / "small"), ["small"])
    by_stem = {match.stem: match.masks["small"].global_mask for match in matches}
    assert by_stem == {"A_1_seg": a1, "A_12_seg": a12}


def test_an_own_folder_reads_vessel_types_and_ties(tmp_path):
    _files(tmp_path / "seg", "E14_seg.tif", "E15_seg.tif")
    art, ven, first, second = _files(
        tmp_path / "large", "E14_art_arteriole.tif", "E14_vein.tif", "E15_a.tif", "E15_b.tif"
    )
    e14, e15 = match_batch(BatchFolders(tmp_path / "seg", large=tmp_path / "large"), ["large"])
    assert (e14.masks["large"].arteriole, e14.masks["large"].venule) == (art, ven)
    assert e15.masks["large"].ambiguous == {"global": (first, second)}
    assert not e15.ok


def test_a_file_naming_the_other_size_is_not_an_own_folders(tmp_path):
    _files(tmp_path / "seg", "E14_seg.tif")
    _files(tmp_path / "masks", "E14_large.tif")
    (match,) = match_batch(BatchFolders(tmp_path / "seg", small=tmp_path / "masks"), ["small"])
    assert not match.masks["small"].found_any


def test_a_mask_folder_given_as_the_segmented_folder_reads_names(tmp_path):
    """Named as the segmented folder, the folder says nothing: a file there is
    a small mask by its name, and one named 'intermediate' is an image."""
    _files(tmp_path, "E14_seg.tif", "E14_intermediate.tif", "E14_small_vessels.tif")
    matches = match_batch(BatchFolders(tmp_path, small=tmp_path), ["small"])
    assert [match.stem for match in matches] == ["E14_intermediate", "E14_seg"]
    assert matches[1].masks["small"].global_mask == tmp_path / "E14_small_vessels.tif"


# --- the batch as a whole ---------------------------------------------------


def test_images_sharing_a_stem_would_share_a_folder_and_both_stop(tmp_path):
    tiff, h5 = _files(tmp_path, "E14_seg.tif", "E14_seg.h5")
    matches = match_batch(BatchFolders(tmp_path))
    assert [match.segmented for match in matches] == [h5, tiff]
    assert [match.same_stem for match in matches] == [(tiff,), (h5,)]
    assert all("share an output folder" in match.problems[0] for match in matches)


def test_a_segmented_folder_is_required_and_must_exist(tmp_path):
    with pytest.raises(ValueError, match="needs a folder of segmented images"):
        BatchFolders("")
    with pytest.raises(ValueError, match="is not a folder"):
        match_batch(BatchFolders(tmp_path / "missing"))


def test_blank_folders_read_as_the_segmented_folder(tmp_path):
    folders = BatchFolders(tmp_path, raw="", large=None, small=tmp_path / "small")
    assert folders.folder("raw") == tmp_path
    assert folders.folder("large") == tmp_path
    assert folders.folder("small") == tmp_path / "small"
    with pytest.raises(ValueError, match="Unknown batch folder"):
        folders.folder("pericyte")


def test_an_unknown_mask_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="large' or 'small"):
        match_batch(BatchFolders(tmp_path), ["medium"])


def test_matching_a_large_folder_stays_quick(tmp_path):
    """Two hundred images, each with a raw image and four masks, in one
    folder: every pair of names is read, so this guards the pairwise reading."""
    names = []
    for index in range(200):
        stem = f"Animal{index:03d}_region"
        names += [
            f"{stem}_{ILASTIK}.tiff",
            f"C1-{stem}.tif",
            f"{stem}_{ILASTIK}_large_arteriole.tiff",
            f"{stem}_{ILASTIK}_large_vein.tiff",
            f"{stem}_{ILASTIK}_small_arteriole.tiff",
            f"{stem}_{ILASTIK}_small_vein.tiff",
        ]
    _files(tmp_path, *names)
    start = time.perf_counter()
    matches = match_batch(BatchFolders(tmp_path), ["large", "small"])
    elapsed = time.perf_counter() - start
    assert len(matches) == 200
    assert all(match.ok for match in matches)
    assert matches[7].raw.name == "C1-Animal007_region.tif"
    assert elapsed < 10, elapsed
