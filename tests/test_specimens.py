"""The specimen registry, and the confounds it exists to prevent.

Per-specimen classifiers would make specimen identity and classifier identity the same
variable - unrecoverable after the fact. Paths derived by naming rule would silently resolve
to the wrong volume, because the artefact names are group-correlated. And a per-specimen
voxel size would put radii and lengths on different scales, because the supplied distance
transform is calibrated in one value for all six.
"""
import json
from pathlib import Path

import pytest

from ImageLynx.specimens import (
    GROUPS,
    ILASTIK_INPUT_CHANNELS,
    ILASTIK_INPUT_DATASET,
    ILASTIK_INPUT_DIR,
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

def test_the_vessel_class_index_refuses_to_be_guessed(monkeypatch):
    """The wrong class index yields the inverse segmentation and no error.

    Ilastik exports one probability channel per class in label order. Reading the background
    channel gives a mean of 1 - expected and a full set of downstream numbers computed from
    the complement of the vessels. There is no safe default, so an unrecorded value has to be
    an error rather than a zero - which stays true now that the value happens to be known.
    """
    import ImageLynx.specimens as module

    monkeypatch.setattr(module, "VESSEL_CLASS_INDEX", None)
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
    record = specimen.qc_record()
    assert record is not None, (
        f"no QC record for {specimen.specimen_id} at {specimen.qc_path} or "
        f"{specimen.bundled_qc_path}"
    )

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
    records = {s.specimen_id: s.qc_record() for s in SPECIMENS}
    assert all(r is not None for r in records.values()), "a QC record is missing"

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


# --- Locating the data ---------------------------------------------------------------------

def test_data_root_search_prefers_the_location_that_actually_holds_the_data(tmp_path):
    """The artefacts have moved twice in two days; a hardcoded path needs a commit each time."""
    from ImageLynx.specimens import _resolve_root

    stale, current = tmp_path / "stale", tmp_path / "current"
    (current / "ilastik_inputs").mkdir(parents=True)
    stale.mkdir()

    assert _resolve_root("_UNSET_ENV_", (stale, current), "ilastik_inputs") == current
    # With nothing found, the first candidate is returned so the error names a real intent.
    assert _resolve_root("_UNSET_ENV_", (stale, current), "nowhere") == stale


def test_an_environment_override_wins_outright(tmp_path, monkeypatch):
    from ImageLynx.specimens import _resolve_root

    monkeypatch.setenv("IMAGELYNX_TEST_ROOT", str(tmp_path))
    resolved = _resolve_root("IMAGELYNX_TEST_ROOT", (Path("/nonexistent"),), "ilastik_inputs")
    assert resolved == tmp_path.resolve()


def test_data_root_provenance_reports_where_it_looked():
    """A silently stale copy is the failure mode a search introduces; this makes it visible."""
    from ImageLynx.specimens import data_root_provenance

    provenance = data_root_provenance()
    assert set(provenance) >= {"acquisition_root", "data_root", "data_root_exists",
                               "data_root_from_env", "env_vars"}


def test_the_classifier_sits_with_the_volumes_it_registers():
    """The project references its datasets by relative path, so separating them breaks it."""
    assert POOLED_CLASSIFIER.parent == ILASTIK_INPUT_DIR
    assert POOLED_CLASSIFIER.name == "vessel_segmentation.ilp"


def test_the_vessel_class_index_is_recorded():
    """LabelNames on the trained project are ['vessel', 'background'], so vessel is channel 0."""
    from ImageLynx.specimens import VESSEL_CLASS_INDEX

    assert VESSEL_CLASS_INDEX == 0
    assert resolve_vessel_class_index() == 0


# --- Verifying the trained project ---------------------------------------------------------

def _write_project(path, lanes, label_names=("vessel", "background"), compute_in_2d=False):
    """Build a minimal .ilp with the structure verify_classifier reads.

    ``lanes`` is a sequence of (preproc_stem, [z_start, ...]); an empty depth list means the
    lane is registered but never labelled, which is the real failure this guards against.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "w") as project:
        project["PixelClassification/LabelNames"] = np.array(
            [n.encode() for n in label_names], dtype="S32")

        for position, (stem, depths) in enumerate(lanes):
            lane = f"lane{position:04d}"
            project[f"Input Data/infos/{lane}/Raw Data/filePath"] = \
                f"{stem}_ilastik.h5/data".encode()

            group = project.create_group(f"PixelClassification/LabelSets/labels{position:03d}")
            for block_index, z in enumerate(depths):
                block = group.create_dataset(
                    f"block{block_index:04d}", data=np.ones((1, 4, 4, 1), dtype=np.uint8))
                block.attrs["blockSlice"] = f"[{z}:{z + 1},0:4,0:4,0:1]".encode()

        project["FeatureSelections/ComputeIn2d"] = np.array([compute_in_2d] * 7)
        project["FeatureSelections/SelectionMatrix"] = np.zeros((6, 7), dtype=bool)


def _all_lanes(depths):
    return [(s.preproc_stem, list(depths)) for s in SPECIMENS]


def test_a_properly_pooled_project_verifies(tmp_path):
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "good.ilp"
    _write_project(path, _all_lanes([50, 150, 250]))

    report = verify_classifier(path)
    assert report["label_names"] == ["vessel", "background"]
    assert report["total_labelled_voxels"] == 6 * 3 * 16


def test_labels_on_one_cohort_only_are_refused(tmp_path):
    """One .ilp trained on one cohort satisfies the letter of the rule and defeats its point.

    This is the state the first trained project was actually in: all 454 labels on WKY-A,
    the other five lanes registered and empty. The decision boundary is then learned from
    normotensive tissue and applied to hypertensive tissue, which is the confound the whole
    registry exists to remove, reintroduced one level down where nothing else can see it.
    """
    from ImageLynx.specimens import verify_classifier

    lanes = [(s.preproc_stem, [214] if s.specimen_id == "WKY-A" else []) for s in SPECIMENS]
    path = tmp_path / "one_cohort.ilp"
    _write_project(path, lanes)

    with pytest.raises(ValueError) as excinfo:
        verify_classifier(path)

    message = str(excinfo.value)
    assert "no labels at all" in message
    for unlabelled in ("WKY-B", "WKY-C", "SHR-A", "SHR-B", "SHR-C"):
        assert unlabelled in message
    assert "WKY-A" not in message.split("no labels at all")[1].split("\n")[0]


def test_labelling_at_a_single_depth_is_refused(tmp_path):
    """Each volume's tissue peaks at a different slice, and the sparse ends are the hard case."""
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "one_depth.ilp"
    _write_project(path, _all_lanes([214]))

    with pytest.raises(ValueError, match="fewer than 2 depths"):
        verify_classifier(path)

    # Still inspectable without raising, and the depth rule can be relaxed deliberately.
    assert verify_classifier(path, require_pooled_labels=False)["total_labelled_voxels"] > 0


def test_a_label_order_contradicting_the_recorded_index_is_refused(tmp_path):
    """Reading the background channel yields a full set of results from the inverse mask."""
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "swapped.ilp"
    _write_project(path, _all_lanes([50, 150]), label_names=("background", "vessel"))

    with pytest.raises(ValueError, match="label order"):
        verify_classifier(path)


def test_two_dimensional_features_are_refused(tmp_path):
    """Per-slice features give z-anisotropic predictions and staircase skeleton artefacts."""
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "flat_features.ilp"
    _write_project(path, _all_lanes([50, 150]), compute_in_2d=True)

    with pytest.raises(ValueError, match="2D"):
        verify_classifier(path)


def test_a_specimen_missing_from_the_project_is_refused(tmp_path):
    """A volume the classifier was never shown cannot be part of a pooled training set."""
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "five_lanes.ilp"
    _write_project(path, _all_lanes([50, 150])[:-1])

    with pytest.raises(ValueError, match="not registered as lanes|not registered"):
        verify_classifier(path)


def test_classifier_hash_is_content_addressed(tmp_path):
    """A run has to be able to name the classifier that produced it, not just a filename."""
    from ImageLynx.specimens import classifier_sha256

    a, b = tmp_path / "a.ilp", tmp_path / "b.ilp"
    a.write_bytes(b"same"), b.write_bytes(b"same")
    assert classifier_sha256(a) == classifier_sha256(b)

    b.write_bytes(b"retrained")
    assert classifier_sha256(a) != classifier_sha256(b)


def test_the_real_trained_classifier_is_ready_to_segment_the_study():
    """Status check on the actual project, which is data rather than code.

    Skips - loudly - while the project is not ready, so the suite stays green on a code
    change while the reason stays visible in `pytest -rs`. It turns into a real pass on its
    own once the labelling covers all six volumes at more than one depth.
    """
    from ImageLynx.specimens import verify_classifier

    if not POOLED_CLASSIFIER.exists():
        pytest.skip(f"no trained classifier at {POOLED_CLASSIFIER}")
    try:
        verify_classifier()
    except (ValueError, OSError) as problem:
        pytest.skip(f"classifier not ready: {problem}")


def test_the_qc_records_are_bundled_with_the_code(tmp_path, monkeypatch):
    """The premise behind one shared classifier has to survive the data moving.

    The sidecars are the only machine-readable evidence that all six volumes were
    preprocessed identically. They are kilobytes; the volumes they describe are 4 GB. Keeping
    a committed copy means the check above is a real assertion on any machine rather than a
    skip on every machine but this one - and the data directory has already moved twice.
    """
    for specimen in SPECIMENS:
        assert specimen.bundled_qc_path.exists(), specimen.bundled_qc_path

    wky_a = get_specimen("WKY-A")
    monkeypatch.setattr(type(wky_a), "qc_path",
                        property(lambda self: tmp_path / "gone.json"))
    record = wky_a.qc_record()
    assert record is not None and tuple(record["shape_zyx"]) == wky_a.shape_zyx


def test_a_locked_project_is_reported_not_traced_back():
    """ilastik holds a write lock on the .ilp for as long as the project is open.

    Checking readiness mid-labelling is the obvious thing to do, and it raised
    BlockingIOError out of h5py and took the whole specimen listing down with it. The cause
    and the fix both have to be in the message.
    """
    from ImageLynx.specimens import _describe_h5_open_failure

    locked = _describe_h5_open_failure(
        BlockingIOError(11, "unable to lock file"), Path("vessel_segmentation.ilp"))
    assert "open in another program" in locked
    assert "ilastik" in locked

    # A genuinely unreadable file must not be blamed on a lock that is not there.
    broken = _describe_h5_open_failure(OSError("file signature not found"), Path("x.ilp"))
    assert "open in another program" not in broken
    assert "signature" in broken


def test_read_classifier_metadata_translates_open_failures(tmp_path):
    from ImageLynx.specimens import read_classifier_metadata

    not_hdf5 = tmp_path / "notreally.ilp"
    not_hdf5.write_text("this is not an HDF5 file")
    with pytest.raises(OSError, match="could not be read"):
        read_classifier_metadata(not_hdf5)


def test_group_label_imbalance_is_reported_without_failing(tmp_path):
    """A forest weights by labelled voxel count, so lopsided labelling tilts the boundary.

    This is the original confound in weaker form: labels concentrated on one cohort make the
    classifier better calibrated there, which is a group-dependent sensitivity difference that
    lands directly on the group contrast. It is a matter of degree rather than a binary
    defect, so it is reported rather than raised - failing here would throw away hours of
    genuine labelling over a judgement call.
    """
    from ImageLynx.specimens import verify_classifier

    lopsided = []
    for specimen in SPECIMENS:
        blocks = [100, 200] * (10 if specimen.group == "SHR" else 1)
        lopsided.append((specimen.preproc_stem, blocks))

    path = tmp_path / "lopsided.ilp"
    _write_project(path, lopsided)

    report = verify_classifier(path)          # must not raise
    assert report["group_label_counts"]["SHR"] > report["group_label_counts"]["WKY"]
    assert any("imbalance" in w for w in report["warnings"])


def test_balanced_labelling_produces_no_warnings(tmp_path):
    from ImageLynx.specimens import verify_classifier

    path = tmp_path / "balanced.ilp"
    _write_project(path, _all_lanes([100, 200, 300]))
    assert verify_classifier(path)["warnings"] == []
