"""The specimen registry, and the confounds it exists to prevent.

Per-specimen classifiers would make specimen identity and classifier identity the same
variable - unrecoverable after the fact. Paths derived by naming rule would silently resolve
to the wrong volume, because the artefact names are group-correlated. And a per-specimen
voxel size would put radii and lengths on different scales, because the supplied distance
transform is calibrated in one value for all six.
"""
import json

import pytest

from ImageLynx.specimens import (
    GROUPS,
    ILASTIK_INPUT_CHANNELS,
    ILASTIK_INPUT_DATASET,
    LEGACY_WKY_A_CLASSIFIER,
    POOLED_CLASSIFIER,
    PROBABILITIES_DATASET,
    PROCESSING_VOXEL_UM,
    SPECIMENS,
    WEAKEST_SPECIMEN_ID,
    Specimen,
    assert_single_classifier,
    get_specimen,
    resolve_vessel_class_index,
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
    donor = SPECIMENS[3]
    mixed = (
        SPECIMENS[0],
        Specimen(donor.specimen_id, donor.group, donor.stem, donor.preproc_stem,
                 donor.acquisition_subdir, donor.measured_voxel_um, donor.shape_zyx,
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


# --- Group-correlated naming ---------------------------------------------------------------

def test_artefact_stems_are_recorded_not_derived():
    """No single f-string produces both groups' names, and the difference tracks the group.

    WKY was split to a C1-*_vessels TIFF in Fiji before preprocessing and carries that name;
    SHR was preprocessed from the acquisition directly. A rule that happened to fit one group
    would resolve the other to a path that does not exist - or worse, to one that does.
    """
    wky = get_specimen("WKY-A")
    shr = get_specimen("SHR-A")

    assert wky.preproc_stem == "C1-CB3-WKY-CB-A-2x2x2_vessels"
    assert shr.preproc_stem == "CB3-SHR-CB-A-2x2x2"
    assert wky.preproc_stem != wky.stem
    assert shr.preproc_stem == shr.stem

    # The acquisitions are not at the same depth either.
    assert wky.acquisition_subdir == "CB3-WKY/raw_cb_images"
    assert shr.acquisition_subdir == "CB3-SHR"


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_each_specimen_resolves_its_own_distinct_artefacts(specimen):
    resolved = get_specimen(specimen.specimen_id)
    for attr in ("acquisition_path", "ilastik_input_path", "probabilities_path",
                 "mask_path", "edt_path"):
        mine = getattr(resolved, attr)
        others = {getattr(s, attr) for s in SPECIMENS if s.specimen_id != specimen.specimen_id}
        assert mine not in others, f"{attr} collides with another specimen"


def test_derived_paths_follow_the_toolchain_naming_conventions():
    """Each stage names its output after the previous one, so the stems are load-bearing."""
    wky_a = get_specimen("WKY-A")

    assert wky_a.ilastik_input_path.name == "C1-CB3-WKY-CB-A-2x2x2_vessels_ilastik.h5"
    assert wky_a.qc_path.name == "C1-CB3-WKY-CB-A-2x2x2_vessels_qc.json"
    # Ilastik names the export after the input nickname, i.e. the HDF5 stem.
    assert wky_a.probabilities_path.name.endswith("_ilastik_Probabilities.h5")
    assert wky_a.mask_path.name == "WKY-A_mask.npy"
    assert wky_a.edt_path.name == "WKY-A_edt_um.npy"


def test_unknown_specimen_names_the_valid_options():
    with pytest.raises(KeyError, match="WKY-A"):
        get_specimen("WKY-Z")


# --- Voxel size ----------------------------------------------------------------------------

def test_one_voxel_size_is_used_for_every_computation():
    """Radii and lengths have to be on the same scale.

    The supplied distance transform is already in micrometres, calibrated with the WKY z step
    for all six volumes. Computing SHR skeleton lengths at the SHR z step would leave the two
    on different scales for no gain - the difference is 0.014%.
    """
    assert PROCESSING_VOXEL_UM == (1.8639, 1.866, 1.866)
    volumes = {s.specimen_id: s.voxel_volume_um3 for s in SPECIMENS}
    assert len(set(volumes.values())) == 1, "voxel volume must not vary by specimen"


def test_the_acquisitions_still_differ_and_the_registry_still_knows_it():
    """A group-correlated acquisition difference belongs in the methods section.

    It is deliberately not used for computation, but discarding it entirely would lose the
    fact that the two cohorts were not acquired identically.
    """
    wky = {s.measured_voxel_um for s in specimens_in_group("WKY")}
    shr = {s.measured_voxel_um for s in specimens_in_group("SHR")}

    assert len(wky) == 1 and len(shr) == 1, "voxel size should be uniform within a group"
    assert wky != shr, "the registry has lost the between-group z-step difference"

    (wz, wy, wx), (sz, sy, sx) = wky.pop(), shr.pop()
    assert (wy, wx) == (sy, sx), "only the z step differs"
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
    assert specimen.measured_voxel_um == pytest.approx((z_step, in_plane, in_plane))
    # Acquisitions are ZCYX; the registry records the shape after channel separation.
    assert specimen.shape_zyx == (shape[0], shape[2], shape[3])


def test_at_least_one_acquisition_is_findable():
    """The metadata test skips when a file is absent, so a wholesale path break would be silent.

    The acquisitions have already moved once - the WKY volumes were relocated into a
    raw_cb_images subdirectory - and that turned six passing checks into six skips without
    failing anything.
    """
    found = [s.specimen_id for s in SPECIMENS if s.acquisition_path.exists()]
    if not found:
        pytest.skip("no acquisitions reachable at all; nothing to cross-check against")
    assert len(found) >= 1


def test_shr_specimens_are_larger_so_raw_counts_are_not_comparable():
    """Any H1 quantity that is a count rather than a density is confounded with extent.

    SHR average about 89 Mvoxel against 63 Mvoxel for WKY, so a larger beta-1 in SHR would be
    expected before any biology is involved.
    """
    mean_wky = sum(s.volume_um3 for s in specimens_in_group("WKY")) / 3
    mean_shr = sum(s.volume_um3 for s in specimens_in_group("SHR")) / 3
    assert mean_shr > mean_wky * 1.2


# --- The Ilastik contract ------------------------------------------------------------------

def test_the_vessel_class_index_refuses_to_be_guessed():
    """The wrong class index yields the inverse segmentation and no error.

    Ilastik exports one probability channel per class in label order. Reading the background
    channel gives a mean of 1 - expected and a full set of downstream numbers computed from
    the complement of the vessels. There is no safe default, so the absence of a recorded
    value has to be an error rather than a zero.
    """
    with pytest.raises(ValueError, match="vessel class index is not recorded"):
        resolve_vessel_class_index()

    assert resolve_vessel_class_index(override=1) == 1
    with pytest.raises(ValueError):
        resolve_vessel_class_index(override=-1)


def test_the_three_channel_contract_is_recorded_in_order():
    """Channel order is load-bearing: the forest indexes features by channel position."""
    assert ILASTIK_INPUT_CHANNELS == ("grayscale", "vesselness_fine", "vesselness_coarse")
    assert ILASTIK_INPUT_DATASET == "data"
    assert PROBABILITIES_DATASET == "exported_data"


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_preprocessing_actually_produced_the_contracted_volume(specimen):
    """Cross-check the registry against the QC sidecar preprocess_cb.py wrote.

    This is the check that catches a stem pointing at the wrong specimen's file: the sidecar
    records the shape and the channel order of the volume that was actually written.
    """
    if not specimen.qc_path.exists():
        pytest.skip(f"not preprocessed yet: {specimen.qc_path.name}")

    record = json.loads(specimen.qc_path.read_text())

    assert tuple(record["shape_zyx"]) == specimen.shape_zyx
    assert tuple(record["channel_names"]) == ILASTIK_INPUT_CHANNELS
    assert specimen.stem in record["input"]
    # Every volume must have been preprocessed with the same parameters, or the shared
    # classifier is not measuring the same thing on each.
    assert tuple(record["parameters"]["voxel"]) == PROCESSING_VOXEL_UM
    assert record["parameters"]["rolling_ball"] == 30.0
    assert record["parameters"]["saturated"] == 0.02
    assert record["parameters"]["sigmas_fine"] == [1.0, 1.4, 2.0]
    assert record["parameters"]["sigmas_coarse"] == [4.0, 8.0]
    assert record["parameters"]["remove_outliers"] == 0

    # No bleach correction: the axial profile is a hump - tissue extent, not photobleaching -
    # and histogram matching it inflates the sparse end slices until background noise reaches
    # vessel intensity. preprocess_cb.py decides this per volume, so it has to be checked per
    # volume rather than assumed from the parameters.
    assert record["z_profile"]["verdict"].startswith("hump")


def test_every_specimen_was_preprocessed_identically():
    """Per-volume tuning reintroduces the cohort bias the shared classifier exists to prevent."""
    records = {s.specimen_id: json.loads(s.qc_path.read_text())
               for s in SPECIMENS if s.qc_path.exists()}
    if len(records) < 2:
        pytest.skip("fewer than two QC sidecars present")

    keys = ("rolling_ball", "saturated", "sigmas_fine", "sigmas_coarse",
            "single_vesselness", "fast_coarse", "channel", "voxel")
    signatures = {sid: json.dumps([r["parameters"][k] for k in keys])
                  for sid, r in records.items()}
    assert len(set(signatures.values())) == 1, (
        f"preprocessing parameters differ between specimens: {signatures}"
    )


def test_the_weakest_specimen_is_named_in_advance():
    """WKY-C has the worst SNR of the six; predicting that beats discovering it."""
    assert get_specimen(WEAKEST_SPECIMEN_ID).group == "WKY"


def test_segmentation_status_reports_each_stage_separately():
    status = segmentation_status()
    assert set(status) == {s.specimen_id for s in SPECIMENS}
    for entry in status.values():
        assert entry["group"] in GROUPS
        assert set(entry["stages"]) == {"acquired", "preprocessed", "predicted", "masked"}
        assert isinstance(entry["missing_inputs"], list)
        assert isinstance(entry["ready"], bool)


# --- Pipeline wiring -----------------------------------------------------------------------

def _pipeline():
    return pytest.importorskip("carotid_image_to_model")


def test_pipeline_uses_the_pooled_classifier_not_the_wky_only_one():
    """The pipeline pointed at cb_wky_2x2x2_A.ilp, trained on normotensive tissue alone."""
    assert _pipeline().ILASTIK_PROJECT_PATH == POOLED_CLASSIFIER


def test_pipeline_input_path_is_no_longer_hardcoded_to_one_specimen():
    """It read a literal WKY-A filename, so only one of the six could ever be run."""
    import inspect

    source = inspect.getsource(_pipeline())
    assert "active_specimen" in source
    assert 'ILASTIK_OUTPUT_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map_probs.tiff"' not in source


def test_pipeline_voxel_size_is_the_shared_processing_value():
    """Running one group at the other's z step would silently mis-scale it."""
    assert _pipeline().PipelineConfig().voxel_size_um == PROCESSING_VOXEL_UM
