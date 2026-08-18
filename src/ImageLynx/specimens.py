"""The study's specimen registry.

Six carotid bodies: three normotensive (WKY) and three spontaneously hypertensive (SHR),
n = 3 per group, with the specimen as the unit of analysis.

Three things this exists to make structurally impossible.

**Per-specimen classifiers.** There is one ``POOLED_CLASSIFIER`` and every specimen references
it. Segmenting each specimen with its own Ilastik project would confound specimen identity with
classifier identity perfectly and unfixably - a between-group difference in vessel count could
then be a difference in the classifier rather than in the tissue, with no way to tell after the
fact. ``assert_single_classifier`` refuses a run whose specimens do not share one project.

**Paths guessed from a naming rule.** The artefacts are not consistently named, and the
inconsistency is group-correlated, which is the worst possible kind here: WKY carries a ``C1-``
prefix and a ``_vessels`` infix that SHR does not, and the WKY acquisitions sit one directory
deeper than the SHR ones. No f-string produces both. Each specimen therefore records its stems
and subdirectory explicitly rather than deriving them, so a rename shows up as a missing file
instead of as a silently wrong volume.

**Two voxel sizes in one calculation.** The acquisitions genuinely differ - 1.86386 um in z for
WKY against 1.86412 for SHR - but the upstream preprocessing annotated all six with the WKY
value, and the supplied distance transform is calibrated in those units. Radii would then be in
one scale and skeleton lengths in another. ``PROCESSING_VOXEL_UM`` is the single value every
computation uses; ``Specimen.measured_voxel_um`` keeps the acquisition truth for the methods
section, where a group-correlated acquisition difference belongs. The gap is 0.014% and changes
no result, which is the reason to record it once rather than discover it twice.
"""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]

_ACQUISITION_ENV = "IMAGELYNX_CB_ACQUISITION_ROOT"
_DATA_ENV = "IMAGELYNX_CB_DATA_ROOT"

#: Where the raw acquisitions live. Hundreds of megabytes each, so outside the repository.
_ACQUISITION_CANDIDATES = (
    Path.home() / "Desktop" / "LCFM Images",
)

#: Where the derived artefacts live - the directory *containing* ilastik_inputs. These have
#: already moved twice: preprocess_cb.py wrote them beside the acquisitions, and they were
#: then relocated into the repository. Searching known locations rather than hardcoding one
#: means a move is a printed line rather than a commit, and ``data_root_provenance`` reports
#: which candidate won so the search is visible instead of magic. IMAGELYNX_CB_DATA_ROOT
#: overrides it outright.
_DATA_CANDIDATES = (
    _ROOT,
    Path.home() / "Desktop" / "LCFM Images",
)


def _resolve_root(env_var: str, candidates: Sequence[Path], marker: str) -> Path:
    """First the environment, then the first candidate that actually holds ``marker``."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    for candidate in candidates:
        if (candidate / marker).is_dir():
            return candidate
    return candidates[0]


ACQUISITION_ROOT = _resolve_root(_ACQUISITION_ENV, _ACQUISITION_CANDIDATES, "CB3-WKY")
CB_DATA_ROOT = _resolve_root(_DATA_ENV, _DATA_CANDIDATES, "ilastik_inputs")

#: Committed copies of the preprocessing QC sidecars. Kilobytes, and they travel with the
#: code rather than with the 4 GB of volumes they describe.
_BUNDLED_QC_DIR = Path(__file__).resolve().parent / "data" / "preprocessing_qc"


def data_root_provenance() -> Dict[str, object]:
    """Which roots were chosen and why, so a silently stale copy is visible."""
    return {
        "acquisition_root": str(ACQUISITION_ROOT),
        "acquisition_root_from_env": bool(os.environ.get(_ACQUISITION_ENV)),
        "acquisition_root_exists": ACQUISITION_ROOT.is_dir(),
        "data_root": str(CB_DATA_ROOT),
        "data_root_from_env": bool(os.environ.get(_DATA_ENV)),
        "data_root_exists": CB_DATA_ROOT.is_dir(),
        "env_vars": (_ACQUISITION_ENV, _DATA_ENV),
    }


#: preprocess_cb.py output: the 3-channel volumes Ilastik is trained and predicted on, plus
#: the per-volume _qc.json record of how each was produced. The trained project has to sit in
#: this directory too - it registers its datasets by relative path, so separating them breaks
#: it.
ILASTIK_INPUT_DIR = CB_DATA_ROOT / "ilastik_inputs"

#: Headless prediction output, one *_Probabilities.h5 per volume.
PROBABILITIES_DIR = CB_DATA_ROOT / "ilastik_probabilities"

#: prob_to_mask.py output: the binary mask and the calibrated distance transform this
#: pipeline consumes.
MASK_DIR = CB_DATA_ROOT / "masks"

#: The single pixel-classification project every specimen must be segmented with. Training it
#: is interactive work in the Ilastik GUI and cannot be automated; what is enforced here is
#: that one project is used for all six, with labels drawn from both groups.
#:
#: It must live in ILASTIK_INPUT_DIR: the project registers its six datasets by relative path
#: (``C1-CB3-WKY-CB-A-2x2x2_vessels_ilastik.h5/data``), so moving it away from them breaks it,
#: while moving the whole directory together does not.
POOLED_CLASSIFIER = ILASTIK_INPUT_DIR / "vessel_segmentation.ilp"

#: The project WKY-A was originally segmented with, back when the pipeline consumed a
#: 2-channel vesselness TIFF. Trained on normotensive tissue only, so it is not valid for the
#: study - retained to identify probability maps that predate the pooled classifier rather
#: than to be used.
LEGACY_WKY_A_CLASSIFIER = _ROOT / "examples" / "images" / "cb_wky_2x2x2_A.ilp"

# --- The HDF5 contract preprocess_cb.py and Ilastik agree on -------------------------------
#
# The classifier was trained on features computed from these three channels in this order.
# Feeding it anything else produces confident nonsense rather than an error, so the contract
# is worth naming in code rather than leaving in a handover document.

ILASTIK_INPUT_DATASET = "data"
ILASTIK_INPUT_AXISTAGS = "zyxc"
ILASTIK_INPUT_CHANNELS: Tuple[str, str, str] = (
    "grayscale",          # rolling-ball background-subtracted, normalised lectin intensity
    "vesselness_fine",    # multiscale Sato, sigma 1.0/1.4/2.0 px, per-scale normalised
    "vesselness_coarse",  # multiscale Sato, sigma 4.0/8.0 px, at half resolution
)

#: Ilastik's headless export dataset name.
PROBABILITIES_DATASET = "exported_data"

#: Which class channel of the probability export is vessel. Ilastik exports one channel per
#: class in label order, so the wrong index yields the inverse segmentation silently - a mean
#: probability near 1 - expected rather than an error.
#:
#: Read from the trained project, whose LabelNames are ['vessel', 'background']. Kept as a
#: constant rather than a lookup because resolving it would mean opening a 288 KB HDF5 on
#: every import; verify_classifier checks it against the project's real label order, which is
#: the only thing that can contradict it.
VESSEL_CLASS_INDEX: Optional[int] = 0

#: The label name that identifies the vessel class, used to verify VESSEL_CLASS_INDEX.
VESSEL_LABEL_NAME = "vessel"

#: The voxel size every computation uses. All six volumes were preprocessed with this value
#: and the supplied distance transform is calibrated in it, so radii and lengths stay on one
#: scale. See the module docstring for why it is not each specimen's measured value.
PROCESSING_VOXEL_UM: Tuple[float, float, float] = (1.8639, 1.866, 1.866)


# --- The two pixel-classification channels -------------------------------------------------
#
# Both channels are pixel classification over the same six specimens, so everything that
# differs between them is collected here rather than left as module constants that quietly
# mean "vessel". A shared default between the two produces confident nonsense rather than an
# error, because a classifier trained on different channels still predicts happily.

@dataclass(frozen=True)
class SegmentationChannel:
    """One Ilastik pixel-classification channel: its project, inputs and target class."""

    key: str
    project: Path
    #: How the preprocessor named its outputs. The two differ by more than a suffix: the
    #: lectin volumes were preprocessed from separately extracted C1 TIFFs and carry a
    #: ``C1-..._vessels`` stem for WKY, whereas the TH volumes were read straight out of the
    #: two-channel acquisition and are named after it.
    stem_attr: str
    input_suffix: str
    #: The label whose probability channel downstream code reads, and where it sits in the
    #: project's label order. Ilastik exports one channel per class in label order, so the
    #: wrong index silently yields a different segmentation rather than an error.
    target_label: str
    target_index: int
    input_channels: Tuple[str, ...]
    #: Whether a measured baseline hash exists for this project. Only the vessel channel has
    #: one; warning that a TH project deviates from a baseline that was never measured would
    #: be noise dressed up as provenance.
    has_measured_baseline: bool

    def qc_name(self, specimen: "Specimen") -> str:
        """The preprocessing sidecar that records how this channel's input was produced."""
        return self.input_name(specimen).replace("_ilastik.h5", "_qc.json")

    def input_name(self, specimen: "Specimen") -> str:
        """The HDF5 file name this channel's preprocessing wrote for that specimen.

        Deliberately the whole file name rather than the stem. Lane matching is a substring
        test against the path Ilastik stored, and the SHR stems are prefixes of their own TH
        input names, so matching on the stem alone would let the vessel channel claim the TH
        lanes for the three SHR specimens and not for the three WKY ones.
        """
        return f"{getattr(specimen, self.stem_attr)}{self.input_suffix}"


VESSEL_CHANNEL = SegmentationChannel(
    key="vessel",
    project=POOLED_CLASSIFIER,
    stem_attr="preproc_stem",
    input_suffix="_ilastik.h5",
    target_label=VESSEL_LABEL_NAME,
    target_index=0,
    input_channels=ILASTIK_INPUT_CHANNELS,
    has_measured_baseline=True,
)

#: preprocess_th.py output. Two channels, the second signed: it reads strongly positive on
#: the bright cytoplasmic ring, about -0.23 in the dark nuclear core and about 0.00 in
#: background, and that sign is the only thing separating "inside the nucleus" from "outside
#: the cell". Cytoplasm is first so that reading channel 0 stays the convention both
#: channels share.
TH_INPUT_CHANNELS: Tuple[str, str] = ("grayscale", "soma_dog_signed")

TH_CHANNEL = SegmentationChannel(
    key="th",
    project=ILASTIK_INPUT_DIR / "th_glomus_segmentation.ilp",
    stem_attr="stem",
    input_suffix="_TH_ilastik.h5",
    target_label="Cytoplasm",
    target_index=0,
    input_channels=TH_INPUT_CHANNELS,
    has_measured_baseline=False,
)

SEGMENTATION_CHANNELS: Dict[str, SegmentationChannel] = {
    channel.key: channel for channel in (VESSEL_CHANNEL, TH_CHANNEL)
}


def resolve_channel(channel) -> SegmentationChannel:
    """Accept a channel, its key, or None for the vessel default."""
    if channel is None:
        return VESSEL_CHANNEL
    if isinstance(channel, SegmentationChannel):
        return channel
    try:
        return SEGMENTATION_CHANNELS[str(channel)]
    except KeyError:
        raise ValueError(
            f"Unknown segmentation channel {channel!r}. "
            f"Known channels: {sorted(SEGMENTATION_CHANNELS)}."
        ) from None


@dataclass(frozen=True)
class Specimen:
    """One carotid body, and every path and constant that is specific to it."""

    specimen_id: str
    group: str                      # "WKY" (normotensive) or "SHR" (hypertensive)
    stem: str                       # acquisition stem, e.g. "CB3-WKY-CB-A-2x2x2"
    preproc_stem: str               # what preprocess_cb.py named its outputs after
    acquisition_subdir: str         # the acquisitions are not all at the same depth
    measured_voxel_um: Tuple[float, float, float]   # (z, y, x), read from the acquisition
    shape_zyx: Tuple[int, int, int]                 # after channel separation
    classifier: Path = POOLED_CLASSIFIER

    # --- Stage 0: acquisition ---
    @property
    def acquisition_path(self) -> Path:
        """The raw multi-channel ZCYX acquisition. Channel 0 is lectin/vessels, 1 is TH."""
        return ACQUISITION_ROOT / self.acquisition_subdir / f"{self.stem}.tif"

    # --- Stage 1: preprocess_cb.py ---
    @property
    def ilastik_input_path(self) -> Path:
        """3-channel float32 HDF5 at /data, axistags zyxc, each channel in [0, 1]."""
        return ILASTIK_INPUT_DIR / f"{self.preproc_stem}_ilastik.h5"

    @property
    def bundled_th_qc_path(self) -> Path:
        """The committed record of how this specimen's TH input was preprocessed."""
        return _BUNDLED_QC_DIR / TH_CHANNEL.qc_name(self)

    @property
    def th_input_path(self) -> Path:
        """preprocess_th.py output: 2-channel float32 HDF5 at /data, axistags zyxc.

        Named after the acquisition stem rather than ``preproc_stem``, because the TH channel
        was read straight out of the two-channel acquisition while the lectin channel went
        through a separately extracted C1 TIFF.
        """
        return ILASTIK_INPUT_DIR / TH_CHANNEL.input_name(self)

    @property
    def qc_path(self) -> Path:
        """The machine-readable record of how this volume was preprocessed."""
        return ILASTIK_INPUT_DIR / f"{self.preproc_stem}_qc.json"

    @property
    def bundled_qc_path(self) -> Path:
        """The copy committed to the repository.

        The sidecars are the only evidence that all six volumes were preprocessed
        identically, which is the premise that makes one shared classifier legitimate. They
        are a few kilobytes each, so keeping a copy in the package means that premise stays
        checkable on a machine that has no access to 4 GB of HDF5 - and survives the data
        directory being moved again.
        """
        return _BUNDLED_QC_DIR / f"{self.preproc_stem}_qc.json"

    def qc_record(self) -> Optional[dict]:
        """The preprocessing record, live copy preferred, bundled copy as fallback."""
        import json

        for candidate in (self.qc_path, self.bundled_qc_path):
            if candidate.exists():
                return json.loads(candidate.read_text())
        return None

    # --- Stage 2: headless Ilastik prediction ---
    @property
    def probabilities_path(self) -> Path:
        """Ilastik names the export after the input's nickname, i.e. its HDF5 stem."""
        return PROBABILITIES_DIR / f"{self.preproc_stem}_ilastik_Probabilities.h5"

    # --- Stage 3: prob_to_mask.py ---
    @property
    def mask_path(self) -> Path:
        return MASK_DIR / f"{self.specimen_id}_mask.npy"

    @property
    def edt_path(self) -> Path:
        """Distance to background, already in micrometres - do not rescale it."""
        return MASK_DIR / f"{self.specimen_id}_edt_um.npy"

    @property
    def voxel_volume_um3(self) -> float:
        z, y, x = PROCESSING_VOXEL_UM
        return float(z * y * x)

    @property
    def volume_um3(self) -> float:
        """Physical volume of the acquisition.

        The specimens differ substantially in extent - SHR average about 89 Mvoxel against
        63 Mvoxel for WKY - so any H1 quantity that is a raw count rather than a density is
        larger for SHR for reasons that have nothing to do with hypertension.
        """
        nz, ny, nx = self.shape_zyx
        return float(nz * ny * nx) * self.voxel_volume_um3

    def stage_status(self) -> Dict[str, bool]:
        """Which pipeline stages have produced their artefact for this specimen."""
        return {
            "acquired": self.acquisition_path.exists(),
            "preprocessed": self.ilastik_input_path.exists(),
            "predicted": self.probabilities_path.exists(),
            "masked": self.mask_path.exists() and self.edt_path.exists(),
        }

    def probability_status(self) -> str:
        """absent / unknown / stale / current - which classifier made the probability map.

        ``predicted: yes`` says a file exists, which is a different claim from the file being
        usable. A map made by a classifier that has since been retrained is worse than no map,
        because it looks finished.
        """
        from .artefact_provenance import probability_status

        return probability_status(self)

    def missing_inputs(self) -> list:
        """Which files still have to be produced before this specimen can be modelled."""
        wanted = (self.ilastik_input_path, self.probabilities_path,
                  self.mask_path, self.edt_path)
        return [p for p in wanted if not p.exists()]

    def is_ready(self) -> bool:
        """Whether the mask and distance transform this pipeline consumes both exist."""
        return self.stage_status()["masked"]


#: Voxel sizes and shapes read from each acquisition's own ImageJ metadata, not typed in:
#: `spacing` gives the z step and XResolution = 535905/1000000 gives 1.8660023698230097 um in
#: y and x for all six. test_specimens.py re-derives them whenever the files are reachable.
#: These are provenance only - PROCESSING_VOXEL_UM is what any calculation uses.
_WKY_VOXEL = (1.8638551724137933, 1.8660023698230097, 1.8660023698230097)
_SHR_VOXEL = (1.8641151515151515, 1.8660023698230097, 1.8660023698230097)

# The preprocessing stems are group-correlated: the WKY volumes were split to a C1-*_vessels
# TIFF in Fiji before preprocessing and carry that name, the SHR volumes were preprocessed
# from the acquisition directly. Recorded rather than derived - see the module docstring.
SPECIMENS: Tuple[Specimen, ...] = (
    Specimen("WKY-A", "WKY", "CB3-WKY-CB-A-2x2x2", "C1-CB3-WKY-CB-A-2x2x2_vessels",
             "CB3-WKY/raw_cb_images", _WKY_VOXEL, (435, 456, 507)),
    Specimen("WKY-B", "WKY", "CB3-WKY-CB-B-2x2x2", "C1-CB3-WKY-CB-B-2x2x2_vessels",
             "CB3-WKY/raw_cb_images", _WKY_VOXEL, (435, 357, 351)),
    Specimen("WKY-C", "WKY", "CB3-WKY-CB-C-2x2x2", "C1-CB3-WKY-CB-C-2x2x2_vessels",
             "CB3-WKY/raw_cb_images", _WKY_VOXEL, (435, 315, 255)),
    Specimen("SHR-A", "SHR", "CB3-SHR-CB-A-2x2x2", "CB3-SHR-CB-A-2x2x2",
             "CB3-SHR", _SHR_VOXEL, (495, 459, 345)),
    Specimen("SHR-B", "SHR", "CB3-SHR-CB-B-2x2x2", "CB3-SHR-CB-B-2x2x2",
             "CB3-SHR", _SHR_VOXEL, (495, 483, 399)),
    Specimen("SHR-C", "SHR", "CB3-SHR-CB-C-2x2x2", "CB3-SHR-CB-C-2x2x2",
             "CB3-SHR", _SHR_VOXEL, (495, 495, 381)),
)

GROUPS: Tuple[str, str] = ("WKY", "SHR")

#: The volume the handover flags as the weakest of the six: background 955 against 337-466 for
#: the other WKY volumes, SNR 7.8. Named here so a segmentation failure there is a prediction
#: that was made in advance rather than a discovery made afterwards.
WEAKEST_SPECIMEN_ID = "WKY-C"

#: The trained project every measured number in this module was taken on.
#:
#: NOT an accepted state. The acceptance it originally recorded has been withdrawn: the plan
#: is now to relabel before generating the probability maps the study keeps, because the
#: measurement below showed the imbalance reaching the group contrast rather than staying a
#: background risk. What survives is the record of what this classifier does, which is the
#: baseline a relabelled one has to beat.
#:
#: 2026-08-10: the project verified but was lopsided - SHR 79388 labelled voxels against WKY
#: 20142, a factor of 3.9, and WKY-C thinnest of all at 4488 with vessel:background 2.32
#: where the SHR volumes sit near 0.9. A forest weights by labelled voxel count, so this
#: classifier is better calibrated on SHR. Three volumes were predicted to find out whether
#: that reaches the output. It does:
#:
#:     specimen  group  mean p  p<0.05  p>0.95  uncertain  fg@0.5  r_med@0.5 um
#:     WKY-A     WKY    0.2358   0.589   0.106      0.305   0.224          5.60
#:     WKY-C     WKY    0.3949   0.341   0.184      0.475   0.380          5.60
#:     SHR-C     SHR    0.1948   0.659   0.082      0.259   0.185          5.27
#:
#: SHR is the more confidently predicted cohort, exactly as the label counts predict, and the
#: foreground fraction at a fixed threshold is higher for WKY than for SHR. Foreground
#: fraction at a fixed threshold is not a neutral quantity here - it is close to what H1
#: measures. So the imbalance is not a background risk to be carried into the limitations
#: section; it is a measurable, group-correlated shift sitting on the group contrast.
#:
#: Separately, the whole probability scale is shifted towards vessel: at p > 0.5 the median
#: inscribed radius is 5.3-5.6 um against a capillary radius of roughly 2-3 um, and the
#: handover's hysteresis low of 0.30 would take 45% of WKY-C as foreground. No threshold
#: gives capillary calibre and connectivity at once. That points at background being
#: under-labelled everywhere rather than at either cohort specifically.
#:
#: The hash makes this record self-invalidating, and is_measured_baseline makes that operate
#: rather than merely be asserted: once the project is relabelled the hash moves, every
#: number above describes something that is no longer being run, and verify_classifier says
#: so instead of leaving a stale measurement to be read as current.
MEASURED_BASELINE_CLASSIFIER_SHA256 = (
    "79a5f6ac5a5e3d6f5c56d0656deb7f78ec95c5f06be6013e4f0143331136c17a"
)


def get_specimen(specimen_id: str) -> Specimen:
    """Look up one specimen, case-insensitively."""
    wanted = specimen_id.strip().upper()
    for specimen in SPECIMENS:
        if specimen.specimen_id.upper() == wanted:
            return specimen
    known = ", ".join(s.specimen_id for s in SPECIMENS)
    raise KeyError(f"Unknown specimen {specimen_id!r}. Known specimens: {known}")


def specimens_in_group(group: str) -> Tuple[Specimen, ...]:
    return tuple(s for s in SPECIMENS if s.group.upper() == group.strip().upper())


def assert_single_classifier(specimens: Optional[Sequence[Specimen]] = None) -> Path:
    """Refuse a run whose specimens were not all segmented by the same classifier.

    A per-specimen classifier makes specimen identity and classifier identity the same
    variable, so a between-group difference could be either and nothing downstream can
    separate them. Unlike most defects this one is unrecoverable after the fact, which is why
    it is an assertion rather than a warning.
    """
    specimens = tuple(specimens if specimens is not None else SPECIMENS)
    if not specimens:
        raise ValueError("No specimens supplied.")

    classifiers = {s.classifier for s in specimens}
    if len(classifiers) != 1:
        detail = ", ".join(f"{s.specimen_id}={s.classifier.name}" for s in specimens)
        raise ValueError(
            "Specimens are not sharing one classifier, which would confound specimen "
            f"identity with classifier identity: {detail}"
        )
    return classifiers.pop()


def resolve_vessel_class_index(override: Optional[int] = None) -> int:
    """The probability channel that is vessel, or a refusal to guess.

    Ilastik exports one channel per class in label order. Picking the wrong one does not
    fail: it returns the background probability, whose mean is 1 - expected, and every
    downstream number is computed from the inverse segmentation. There is no safe default,
    so this raises until the trained classifier's label order is recorded.
    """
    index = VESSEL_CLASS_INDEX if override is None else override
    if index is None:
        raise ValueError(
            "The vessel class index is not recorded. Ilastik exports one probability "
            "channel per class in label order, and the wrong index silently yields the "
            "inverse segmentation rather than an error. Set specimens.VESSEL_CLASS_INDEX "
            "from the trained project's label order before predicting."
        )
    if index < 0:
        raise ValueError(f"VESSEL_CLASS_INDEX must be non-negative, got {index}.")
    return int(index)


def segmentation_status() -> Dict[str, Dict[str, object]]:
    """What still has to be produced, per specimen, before the study can run."""
    return {
        s.specimen_id: {
            "group": s.group,
            "stages": s.stage_status(),
            "missing_inputs": [p.name for p in s.missing_inputs()],
            "probability_status": s.probability_status(),
            "ready": s.is_ready(),
        }
        for s in SPECIMENS
    }


# --- Reading and verifying the trained project ---------------------------------------------

def classifier_sha256(path: Optional[Path] = None) -> str:
    """Content hash of the .ilp, so a run can record which classifier produced it.

    Not pinned to an expected value anywhere: retraining is supposed to change it. What
    matters is that the hash travels with the output, so a probability map can be traced to
    the project that made it rather than to a filename that may have been reused.
    """
    path = POOLED_CLASSIFIER if path is None else Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_h5_open_failure(failure: OSError, path: Path) -> str:
    """Turn h5py's open errors into something that says what to do about it.

    The common one by far is a lock: ilastik holds a write lock on the project for as long as
    it is open, and checking readiness while still labelling is the obvious thing to do. That
    surfaced as a bare BlockingIOError from deep inside h5py and took the whole specimen
    listing down with it.
    """
    detail = str(failure)
    locked = isinstance(failure, BlockingIOError) or "lock" in detail.lower()
    if locked:
        return (
            f"{path.name} is open in another program and cannot be read. ilastik holds a "
            f"write lock on a project for as long as it is open - save the project and close "
            f"ilastik, then run this again. ({detail})"
        )
    return f"{path.name} could not be read as an Ilastik project: {detail}"


def is_measured_baseline(path: Optional[Path] = None) -> bool:
    """Whether this is the project the measurements recorded in this module were taken on.

    Returns False for a missing file rather than raising: "we are not running the measured
    baseline" is the true and useful answer either way.
    """
    path = POOLED_CLASSIFIER if path is None else Path(path)
    if not path.exists():
        return False
    return classifier_sha256(path) == MEASURED_BASELINE_CLASSIFIER_SHA256


def read_classifier_metadata(
    path: Optional[Path] = None,
    channel=None,
) -> Dict[str, object]:
    """What the trained Ilastik project actually contains.

    Reads the label names, the registered dataset lanes, how many voxels were labelled on
    each and at which z, and whether the features are 3D. Everything ``verify_classifier``
    decides on, separated out so it can be inspected without raising.
    """
    import h5py
    import numpy as np

    channel = resolve_channel(channel)
    path = channel.project if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained classifier at {path}")

    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    try:
        project = h5py.File(path, "r")
    except OSError as failure:
        raise OSError(_describe_h5_open_failure(failure, path)) from failure

    with project:
        label_names = [_text(n) for n in project["PixelClassification/LabelNames"][()]]

        lanes: List[Dict[str, object]] = []
        infos = project["Input Data/infos"]
        for lane_key in sorted(infos.keys()):
            file_path = None
            for role in infos[lane_key].keys():
                role_group = infos[lane_key][role]
                if "filePath" in role_group:
                    file_path = _text(role_group["filePath"][()])
                    break
            lanes.append({"lane": lane_key, "file_path": file_path or ""})

        label_sets = project["PixelClassification/LabelSets"]
        for position, lane_key in enumerate(sorted(label_sets.keys())):
            counts: Dict[int, int] = {}
            z_slices = set()

            def _visit(_name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.size == 0:
                    return
                block = obj[()]
                for value in np.unique(block):
                    if value:
                        counts[int(value)] = counts.get(int(value), 0) + int((block == value).sum())
                extent = obj.attrs.get("blockSlice")
                if extent is not None:
                    z_slices.add(_text(extent).strip("[]").split(",")[0].strip())

            label_sets[lane_key].visititems(_visit)
            if position < len(lanes):
                lanes[position]["labelled_voxels"] = sum(counts.values())
                lanes[position]["labels_by_value"] = counts
                lanes[position]["z_extents"] = sorted(z_slices)

        features = project["FeatureSelections"]
        compute_in_2d = [bool(v) for v in features["ComputeIn2d"][()]]
        selection = np.asarray(features["SelectionMatrix"][()])

    return {
        "path": str(path),
        "channel": channel.key,
        "label_names": label_names,
        "lanes": lanes,
        "compute_in_2d": compute_in_2d,
        "selected_features": int(selection.sum()),
        "total_labelled_voxels": sum(int(l.get("labelled_voxels", 0)) for l in lanes),
    }


def verify_classifier(
    path: Optional[Path] = None,
    require_pooled_labels: bool = True,
    min_depths_per_lane: int = 2,
    channel=None,
) -> Dict[str, object]:
    """Refuse a classifier that cannot support a between-group comparison.

    One ``.ilp`` used for all six volumes satisfies the letter of the single-classifier rule
    while still being trained on one cohort, if only that cohort's lanes carry labels. The
    decision boundary is then learned from normotensive tissue and applied to hypertensive
    tissue, which is the confound the whole registry exists to remove - reintroduced one
    level down, where nothing else in this codebase can see it.

    That is not hypothetical: the first trained project had all 454 of its labels on WKY-A,
    on a single z slice, with the other five lanes registered and empty.

    Labelling at one depth is a milder version of the same problem. Each volume's tissue
    peaks at a different slice, and the sparse end slices are where background noise comes
    closest to vessel intensity, so a classifier trained only near the peak has never seen
    the case it most needs to get right.

    Returns the metadata report on success. Raises ValueError listing every problem at once,
    since relabelling is one trip back to the GUI either way.
    """
    channel = resolve_channel(channel)
    report = read_classifier_metadata(path, channel)
    problems: List[str] = []

    names = report["label_names"]
    index = channel.target_index
    if index is None or index >= len(names) or names[index] != channel.target_label:
        problems.append(
            f"The {channel.key} channel reads probability channel {index}, expecting "
            f"'{channel.target_label}' there, but the project's label order is {names}; "
            f"reading the wrong channel yields a different segmentation without an error."
        )

    if any(report["compute_in_2d"]):
        problems.append(
            "Some features are computed in 2D. Per-slice features give z-anisotropic "
            "predictions and staircase artefacts in the skeleton."
        )

    lanes = report["lanes"]
    registered = {
        s.specimen_id: any(channel.input_name(s) in str(l["file_path"]) for l in lanes)
        for s in SPECIMENS
    }
    unregistered = sorted(sid for sid, present in registered.items() if not present)
    if unregistered:
        problems.append(
            f"Specimens not registered as lanes in the project: {', '.join(unregistered)}. "
            f"A volume the classifier was never shown cannot be part of a pooled training set."
        )

    if require_pooled_labels:
        empty, shallow = [], []
        for lane in lanes:
            who = next((s.specimen_id for s in SPECIMENS
                        if channel.input_name(s) in str(lane["file_path"])), lane["lane"])
            labelled = int(lane.get("labelled_voxels", 0))
            if labelled == 0:
                empty.append(who)
            elif len(lane.get("z_extents", [])) < min_depths_per_lane:
                shallow.append(f"{who} (1 depth, {labelled} voxels)")

        if empty:
            problems.append(
                f"Lanes with no labels at all: {', '.join(empty)}. The classifier is trained "
                f"on the remainder, so its decision boundary comes from one cohort and the "
                f"between-group difference it measures is partly its own."
            )
        if shallow:
            problems.append(
                f"Lanes labelled at fewer than {min_depths_per_lane} depths: "
                f"{', '.join(shallow)}. Each volume's tissue peaks at a different slice and "
                f"the sparse ends are where background most resembles vessel."
            )

    if problems:
        raise ValueError(
            f"{Path(report['path']).name} is not ready to segment this study:\n  - "
            + "\n  - ".join(problems)
        )

    report["group_label_counts"] = _group_label_counts(lanes, channel)
    report["warnings"] = _label_balance_warnings(
        lanes, report["group_label_counts"], channel)
    report["is_measured_baseline"] = (
        is_measured_baseline(Path(report["path"])) if channel.has_measured_baseline else None
    )
    if channel.has_measured_baseline and not report["is_measured_baseline"]:
        report["warnings"].append(
            "This is not the measured baseline classifier. Every measured number recorded "
            "in ImageLynx.specimens - probability calibration, label imbalance, per-cohort "
            "foreground fraction - was taken on a different project and describes something "
            "that is no longer being run. Re-measure before relying on any of them."
        )
    return report


def _lane_specimen(lane, channel=None) -> Optional["Specimen"]:
    channel = resolve_channel(channel)
    return next(
        (s for s in SPECIMENS if channel.input_name(s) in str(lane["file_path"])), None
    )


def _group_label_counts(lanes, channel=None) -> Dict[str, int]:
    counts = {group: 0 for group in GROUPS}
    for lane in lanes:
        specimen = _lane_specimen(lane, channel)
        if specimen is not None:
            counts[specimen.group] += int(lane.get("labelled_voxels", 0))
    return counts


def _label_balance_warnings(lanes, group_counts, channel=None) -> List[str]:
    """Soft problems: real risks, but matters of degree rather than binary defects.

    Reported rather than raised. A forest weights by labelled voxel count, so lopsided
    labelling tilts the decision boundary towards whichever cohort or volume was labelled
    hardest - the original confound in weaker form. But there is no threshold at which it
    becomes categorically wrong, and failing on one would discard hours of real work over a
    judgement call that belongs to whoever did the labelling.
    """
    channel = resolve_channel(channel)
    warnings: List[str] = []

    labelled = {group: count for group, count in group_counts.items() if count}
    if len(labelled) == len(GROUPS):
        low_group = min(labelled, key=labelled.get)
        high_group = max(labelled, key=labelled.get)
        ratio = labelled[high_group] / labelled[low_group]
        if ratio > 2.0:
            warnings.append(
                f"Group label imbalance {ratio:.1f}x: {high_group} {labelled[high_group]} "
                f"voxels against {low_group} {labelled[low_group]}. The forest weights by "
                f"labelled voxel count, so it is better calibrated on {high_group}, and a "
                f"group-dependent sensitivity difference lands on the group contrast."
            )

    per_lane = [(s, int(l.get("labelled_voxels", 0)))
                for l in lanes for s in [_lane_specimen(l, channel)] if s is not None]
    if per_lane:
        mean_count = sum(c for _, c in per_lane) / len(per_lane)
        thin = [f"{s.specimen_id} ({c})" for s, c in per_lane if c < mean_count / 3]
        if thin:
            warnings.append(
                f"Volumes labelled far below the average of {mean_count:.0f} voxels: "
                f"{', '.join(thin)}."
            )
        weakest = next((s for s, _ in per_lane if s.specimen_id == WEAKEST_SPECIMEN_ID), None)
        if weakest is not None:
            weakest_count = dict((s.specimen_id, c) for s, c in per_lane)[WEAKEST_SPECIMEN_ID]
            if weakest_count < mean_count:
                warnings.append(
                    f"{WEAKEST_SPECIMEN_ID} has the fewest labels relative to the average "
                    f"({weakest_count} against {mean_count:.0f}) and is also the volume with "
                    f"the weakest signal of the six. Effort is going where it is least needed."
                )

    # Ilastik label values are 1-based and follow label order, so the target class is
    # target_index + 1. Comparing it against everything else reduces to the old
    # vessel-against-background ratio when there are only two classes, and stays meaningful
    # for the four-class TH project.
    target_value = channel.target_index + 1
    for lane in lanes:
        specimen = _lane_specimen(lane, channel)
        counts = lane.get("labels_by_value", {})
        target = counts.get(target_value, 0)
        rest = sum(count for value, count in counts.items() if value != target_value)
        if specimen is None or not rest:
            continue
        ratio = target / rest
        if not 0.5 <= ratio <= 2.0:
            warnings.append(
                f"{specimen.specimen_id} {channel.target_label}:rest is {ratio:.2f} "
                f"({target} vs {rest}); the classes are sampled very unevenly there."
            )
    return warnings
