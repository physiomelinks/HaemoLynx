"""The study's specimen registry.

Six carotid bodies: three normotensive (WKY) and three spontaneously hypertensive (SHR),
n = 3 per group, with the specimen as the unit of analysis.

Two things this exists to make structurally impossible.

**Per-specimen classifiers.** There is one ``POOLED_CLASSIFIER`` and every specimen references
it. Segmenting each specimen with its own Ilastik project would confound specimen identity with
classifier identity perfectly and unfixably - a between-group difference in vessel count could
then be a difference in the classifier rather than in the tissue, with no way to tell after the
fact. ``assert_single_classifier`` refuses a run whose specimens do not share one project.

**Silently mismatched acquisition geometry.** The z step differs between the two groups -
1.86386 um for WKY against 1.86412 um for SHR - so a single hardcoded voxel size is wrong for
one group. The difference is 0.014% and changes no result, but it is a group-correlated
acquisition difference and belongs in the methods section rather than in nobody's notes. Each
specimen therefore carries its own measured value, and a test re-derives it from the
acquisition file whenever that file is present, so these constants cannot drift from the data.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
ILASTIK_INPUT_DIR = _ROOT / "examples" / "images" / "ilastik_batch_processing_input_images"
ILASTIK_OUTPUT_DIR = _ROOT / "examples" / "images" / "ilastik_batch_processing_output_images"

#: Raw acquisitions live outside the repository; they are ~300 MB each and gitignored.
ACQUISITION_ROOT = Path.home() / "Desktop" / "LCFM Images"

#: The single pixel-classification project every specimen must be segmented with. Training it
#: is interactive work in the Ilastik GUI and cannot be automated; what is enforced here is
#: that one project is used for all six, with labels drawn from both groups.
POOLED_CLASSIFIER = _ROOT / "examples" / "images" / "cb_pooled_2x2x2.ilp"

#: The project WKY-A was originally segmented with. Trained on normotensive tissue only, so it
#: is not valid for the study - retained to identify probability maps that predate the pooled
#: classifier rather than to be used.
LEGACY_WKY_A_CLASSIFIER = _ROOT / "examples" / "images" / "cb_wky_2x2x2_A.ilp"


@dataclass(frozen=True)
class Specimen:
    """One carotid body, and every path and constant that is specific to it."""

    specimen_id: str
    group: str                      # "WKY" (normotensive) or "SHR" (hypertensive)
    stem: str                       # acquisition stem, e.g. "CB3-WKY-CB-A-2x2x2"
    voxel_size_um: Tuple[float, float, float]   # (z, y, x), measured from the acquisition file
    shape_zyx: Tuple[int, int, int]             # after channel separation
    classifier: Path = POOLED_CLASSIFIER

    @property
    def acquisition_path(self) -> Path:
        """The raw multi-channel ZCYX acquisition. Channel 0 is vessels, channel 1 glomus."""
        return ACQUISITION_ROOT / f"CB3-{self.group}" / f"{self.stem}.tif"

    @property
    def vessels_path(self) -> Path:
        """Channel 1 split out in Fiji: the vessel fluorescence Ilastik is trained on."""
        return ILASTIK_INPUT_DIR / f"C1-{self.stem}_vessels.tif"

    @property
    def vesselness_path(self) -> Path:
        """Frangi vesselness computed on the vessel channel: Ilastik's second feature."""
        return ILASTIK_INPUT_DIR / f"C1-{self.stem}_vesselness_map.tif"

    @property
    def probabilities_path(self) -> Path:
        """What the pipeline actually consumes. Ilastik names it after the vesselness stem."""
        return ILASTIK_OUTPUT_DIR / f"C1-{self.stem}_vesselness_map_probs.tiff"

    @property
    def voxel_volume_um3(self) -> float:
        z, y, x = self.voxel_size_um
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

    def missing_inputs(self) -> list:
        """Which files still have to be produced before this specimen can be segmented."""
        return [p for p in (self.vessels_path, self.vesselness_path) if not p.exists()]

    def is_segmented(self) -> bool:
        return self.probabilities_path.exists()


#: Voxel sizes and shapes read from each acquisition's own ImageJ metadata, not typed in:
#: `spacing` gives the z step and XResolution = 535905/1000000 gives 1.8660023698230097 um in
#: y and x for all six. test_specimens.py re-derives them whenever the files are reachable.
_WKY_VOXEL = (1.8638551724137933, 1.8660023698230097, 1.8660023698230097)
_SHR_VOXEL = (1.8641151515151515, 1.8660023698230097, 1.8660023698230097)

SPECIMENS: Tuple[Specimen, ...] = (
    Specimen("WKY-A", "WKY", "CB3-WKY-CB-A-2x2x2", _WKY_VOXEL, (435, 456, 507)),
    Specimen("WKY-B", "WKY", "CB3-WKY-CB-B-2x2x2", _WKY_VOXEL, (435, 357, 351)),
    Specimen("WKY-C", "WKY", "CB3-WKY-CB-C-2x2x2", _WKY_VOXEL, (435, 315, 255)),
    Specimen("SHR-A", "SHR", "CB3-SHR-CB-A-2x2x2", _SHR_VOXEL, (495, 459, 345)),
    Specimen("SHR-B", "SHR", "CB3-SHR-CB-B-2x2x2", _SHR_VOXEL, (495, 483, 399)),
    Specimen("SHR-C", "SHR", "CB3-SHR-CB-C-2x2x2", _SHR_VOXEL, (495, 495, 381)),
)

GROUPS: Tuple[str, str] = ("WKY", "SHR")


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


def segmentation_status() -> Dict[str, Dict[str, object]]:
    """What still has to be produced, per specimen, before the study can run."""
    return {
        s.specimen_id: {
            "group": s.group,
            "missing_inputs": [p.name for p in s.missing_inputs()],
            "segmented": s.is_segmented(),
        }
        for s in SPECIMENS
    }
