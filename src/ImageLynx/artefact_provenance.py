"""Which classifier produced a given probability map.

Everything else in this pipeline can name where it came from - diameters carry
``diameter_provenance``, centrelines carry ``centreline_smoothing``, radii carry
``edt_junction_trim`` - except the artefact the whole study rests on. The classifier's hash
lives in ``ImageLynx.specimens``; the probability maps it produces carry nothing at all. Six
files with the right names are reported as ``predicted: yes`` whether they were written this
morning or by a project that has since been retrained three times, and this one has been
retrained three times in two days.

The point is not to prevent provisional maps. Regenerating all six is about fifteen minutes,
so the compute was never the problem. The problem is that a placeholder becomes load-bearing
precisely because nothing distinguishes it from the real thing, and the question "were these
made with the final classifier?" stops being answerable at exactly the moment it starts
mattering. With a sidecar, a stale map announces itself and provisional maps become cheap
again.

The sidecar is written after the fact rather than during prediction, because prediction is a
headless Ilastik invocation this codebase does not drive. ``record_probability_provenance``
is what makes it a deliberate step rather than an assumption.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

#: Suffix appended to the artefact's own name, so the sidecar sorts next to it and is
#: obviously subordinate to it.
PROVENANCE_SUFFIX = ".provenance.json"

#: What ``probability_status`` can return.
#:
#: ``unknown`` is deliberately distinct from ``stale`` and is the worse of the two: a stale
#: map is one whose origin is known and wrong, an unknown one is a map nothing can be ruled
#: out about. Treating a missing sidecar as current is the assumption this module exists to
#: refuse.
STATUS_ABSENT = "absent"
STATUS_UNKNOWN = "unknown"
STATUS_STALE = "stale"
STATUS_CURRENT = "current"


def provenance_path_for(artefact_path) -> Path:
    """The sidecar that describes ``artefact_path``."""
    artefact_path = Path(artefact_path)
    return artefact_path.with_name(artefact_path.name + PROVENANCE_SUFFIX)


def _label_summary(classifier_path: Path, channel=None) -> Optional[Dict[str, object]]:
    """Label counts and boundary placement, or None if the project cannot be read.

    A hash says two classifiers differ; it does not say how. Carrying the labelling state
    with the artefact means a result can be attributed to a decision - "this was the round
    before boundary labelling" - rather than to an opaque 64-character string.

    Degrades to None rather than raising: a provenance record that fails to write because the
    optional half of it could not be gathered is worse than a partial one.
    """
    try:
        from .specimens import read_classifier_metadata
        from .statistics.label_placement import analyse_label_placement

        metadata = read_classifier_metadata(classifier_path, channel)
        summary = {
            "total_labelled_voxels": metadata["total_labelled_voxels"],
            "selected_features": metadata["selected_features"],
            "label_names": metadata["label_names"],
        }
        try:
            # Label placement is measured against the vessel band, so it means nothing for
            # another channel. Omitted rather than reported as a number with no referent.
            from .specimens import resolve_channel

            if resolve_channel(channel).key == "vessel":
                summary["boundary_fraction_by_specimen"] = {
                    row.specimen_id: round(row.background_within_band_fraction, 4)
                    for row in analyse_label_placement(classifier_path)
                }
        except Exception:
            pass
        return summary
    except Exception:
        return None


def record_probability_provenance(
    specimen,
    artefact_path=None,
    classifier_path: Optional[Path] = None,
    channel=None,
    notes: Optional[str] = None,
) -> Dict[str, object]:
    """Stamp a probability map with the classifier that produced it.

    Run immediately after a headless prediction. Writing it later is still better than not
    writing it, but only if the classifier has not moved in between - which is the very thing
    the record exists to detect, so later is a gamble.
    """
    from . import specimens as registry

    channel = registry.resolve_channel(channel)
    default_artefact = (specimen.probabilities_path if channel.key == "vessel"
                        else specimen.th_probabilities_path)
    artefact_path = Path(artefact_path or default_artefact)
    # For the vessel channel, resolve through the module attribute rather than the frozen
    # channel record. POOLED_CLASSIFIER is the binding every other consumer reads and the
    # one tests redirect; a Path captured on the dataclass at import time would silently
    # ignore that redirection and report a hash for a different file.
    default_classifier = (registry.POOLED_CLASSIFIER if channel.key == "vessel"
                          else channel.project)
    classifier_path = Path(classifier_path or default_classifier)

    record: Dict[str, object] = {
        "specimen_id": specimen.specimen_id,
        "group": specimen.group,
        "artefact": artefact_path.name,
        "channel": channel.key,
        "shape_zyx": list(specimen.shape_zyx),
        "classifier_name": classifier_path.name,
        "classifier_sha256": registry.classifier_sha256(classifier_path),
        "target_label": channel.target_label,
        "target_class_index": channel.target_index,
        "processing_voxel_um": list(registry.PROCESSING_VOXEL_UM),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labelling": _label_summary(classifier_path, channel),
    }
    if channel.key == "vessel":
        # Kept for the maps already on disk, which were written with this key.
        record["vessel_class_index"] = registry.VESSEL_CLASS_INDEX
    if notes:
        record["notes"] = notes
    provenance_path_for(artefact_path).write_text(json.dumps(record, indent=2))
    return record


def read_provenance(artefact_path) -> Optional[Dict[str, object]]:
    """The sidecar for ``artefact_path``, or None if absent or unreadable.

    An unreadable sidecar is reported the same as a missing one, because a record that
    cannot be parsed establishes nothing.
    """
    path = provenance_path_for(artefact_path)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return record if isinstance(record, dict) else None


def probability_status(specimen, artefact_path=None,
                       classifier_path: Optional[Path] = None) -> str:
    """Whether a probability map was made by the classifier currently in use."""
    from . import specimens as registry

    artefact_path = Path(artefact_path or specimen.probabilities_path)
    if not artefact_path.exists():
        return STATUS_ABSENT

    record = read_provenance(artefact_path)
    if not record or "classifier_sha256" not in record:
        return STATUS_UNKNOWN

    classifier_path = Path(classifier_path or registry.POOLED_CLASSIFIER)
    if not classifier_path.exists():
        return STATUS_UNKNOWN
    current = registry.classifier_sha256(classifier_path)
    return STATUS_CURRENT if record["classifier_sha256"] == current else STATUS_STALE
