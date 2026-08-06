"""The specimen registry, and the two confounds it exists to prevent.

Per-specimen classifiers would make specimen identity and classifier identity the same
variable - unrecoverable after the fact - and a single hardcoded voxel size is wrong for one
group, since the z step differs between the WKY and SHR acquisitions.
"""
import pytest

from ImageLynx.specimens import (
    GROUPS,
    LEGACY_WKY_A_CLASSIFIER,
    POOLED_CLASSIFIER,
    SPECIMENS,
    Specimen,
    assert_single_classifier,
    get_specimen,
    segmentation_status,
    specimens_in_group,
)


def test_registry_is_three_specimens_per_group():
    """n = 3 per group, with the specimen as the unit of analysis."""
    assert len(SPECIMENS) == 6
    for group in GROUPS:
        assert len(specimens_in_group(group)) == 3
    assert len({s.specimen_id for s in SPECIMENS}) == 6


def test_every_specimen_shares_the_pooled_classifier():
    """The confound this registry exists to make structurally impossible."""
    assert all(s.classifier == POOLED_CLASSIFIER for s in SPECIMENS)
    assert assert_single_classifier() == POOLED_CLASSIFIER


def test_a_split_classifier_is_refused():
    """A per-specimen classifier confounds specimen with classifier, unfixably.

    It has to fail loudly rather than warn: once the probability maps exist there is no way to
    separate a classifier difference from a tissue difference after the fact.
    """
    mixed = (
        SPECIMENS[0],
        Specimen(SPECIMENS[3].specimen_id, SPECIMENS[3].group, SPECIMENS[3].stem,
                 SPECIMENS[3].voxel_size_um, SPECIMENS[3].shape_zyx,
                 classifier=LEGACY_WKY_A_CLASSIFIER),
    )
    with pytest.raises(ValueError, match="not sharing one classifier"):
        assert_single_classifier(mixed)


def test_the_legacy_wky_only_classifier_is_not_used_by_any_specimen():
    """cb_wky_2x2x2_A.ilp was trained on normotensive tissue only.

    Segmenting SHR with it risks a group-dependent sensitivity difference that would appear as
    a vascular difference. It is retained to identify probability maps that predate the pooled
    classifier, not to be used.
    """
    assert POOLED_CLASSIFIER != LEGACY_WKY_A_CLASSIFIER
    assert all(s.classifier != LEGACY_WKY_A_CLASSIFIER for s in SPECIMENS)


def test_the_z_step_differs_between_groups_and_the_registry_knows_it():
    """A group-correlated acquisition difference: WKY 1.86386 um, SHR 1.86412 um.

    0.014%, so it changes no result - but a single hardcoded voxel size is wrong for one group
    and the difference belongs in the methods section rather than in nobody's notes.
    """
    wky = {s.voxel_size_um for s in specimens_in_group("WKY")}
    shr = {s.voxel_size_um for s in specimens_in_group("SHR")}

    assert len(wky) == 1 and len(shr) == 1, "voxel size should be uniform within a group"
    assert wky != shr, "the registry has lost the between-group z-step difference"

    # y and x are the same in-plane pixel size for every specimen; only z differs.
    (wz, wy, wx), (sz, sy, sx) = wky.pop(), shr.pop()
    assert (wy, wx) == (sy, sx)
    assert wz != sz
    assert abs(sz - wz) / wz < 1e-3, "the z difference is meant to be negligible in magnitude"


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_registry_voxel_size_matches_the_acquisition_metadata(specimen):
    """The constants must be derived from the files, not typed in beside them."""
    if not specimen.acquisition_path.exists():
        pytest.skip(f"acquisition not present: {specimen.acquisition_path}")

    import tifffile

    with tifffile.TiffFile(specimen.acquisition_path) as handle:
        z_step = (handle.imagej_metadata or {}).get("spacing")
        x_res = handle.pages[0].tags.get("XResolution").value
        shape = handle.series[0].shape

    in_plane = x_res[1] / x_res[0]
    assert specimen.voxel_size_um == pytest.approx((z_step, in_plane, in_plane))
    # Acquisitions are ZCYX; the registry records the shape after channel separation.
    assert specimen.shape_zyx == (shape[0], shape[2], shape[3])


def test_derived_paths_follow_the_ilastik_naming_convention():
    """Ilastik names its output after the vesselness stem, so the convention is load-bearing."""
    wky_a = get_specimen("wky-a")

    assert wky_a.vessels_path.name == "C1-CB3-WKY-CB-A-2x2x2_vessels.tif"
    assert wky_a.vesselness_path.name == "C1-CB3-WKY-CB-A-2x2x2_vesselness_map.tif"
    assert wky_a.probabilities_path.name == "C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff"
    assert wky_a.probabilities_path.parent.name == "ilastik_batch_processing_output_images"


def test_unknown_specimen_names_the_valid_options():
    with pytest.raises(KeyError, match="WKY-A"):
        get_specimen("WKY-Z")


def test_shr_specimens_are_larger_so_raw_counts_are_not_comparable():
    """Any H1 quantity that is a count rather than a density is confounded with extent.

    SHR average about 89 Mvoxel against 63 Mvoxel for WKY, so a larger beta-1 in SHR would be
    expected before any biology is involved.
    """
    mean_wky = sum(s.volume_um3 for s in specimens_in_group("WKY")) / 3
    mean_shr = sum(s.volume_um3 for s in specimens_in_group("SHR")) / 3
    assert mean_shr > mean_wky * 1.2


def test_segmentation_status_reports_what_is_still_missing():
    status = segmentation_status()
    assert set(status) == {s.specimen_id for s in SPECIMENS}
    for entry in status.values():
        assert entry["group"] in GROUPS
        assert isinstance(entry["missing_inputs"], list)
        assert isinstance(entry["segmented"], bool)


# --- Pipeline wiring ----------------------------------------------------------------------

def _pipeline():
    return pytest.importorskip("carotid_image_to_model")


def test_pipeline_uses_the_pooled_classifier_not_the_wky_only_one():
    """The pipeline pointed at cb_wky_2x2x2_A.ilp, trained on normotensive tissue alone."""
    assert _pipeline().ILASTIK_PROJECT_PATH == POOLED_CLASSIFIER


def test_pipeline_input_path_is_no_longer_hardcoded_to_one_specimen():
    """It read a literal WKY-A filename, so only one of the six could ever be run."""
    import inspect

    source = inspect.getsource(_pipeline())
    assert "active_specimen.probabilities_path" in source
    assert 'ILASTIK_OUTPUT_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff"' not in source


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_each_specimen_resolves_a_distinct_probability_volume(specimen):
    resolved = get_specimen(specimen.specimen_id).probabilities_path
    assert resolved == specimen.probabilities_path
    others = {s.probabilities_path for s in SPECIMENS if s.specimen_id != specimen.specimen_id}
    assert resolved not in others


def test_selecting_a_specimen_selects_its_own_voxel_size():
    """Running SHR against WKY's z step would silently mis-scale one group."""
    wky = get_specimen("WKY-A").voxel_size_um
    shr = get_specimen("SHR-A").voxel_size_um
    assert wky[0] != shr[0]
    assert wky[1:] == shr[1:]
