"""Which classifier produced a given probability map.

Everything in this pipeline can name where it came from except the artefact the whole study
rests on. The classifier's hash lives in ImageLynx.specimens; the probability maps it produces
carry nothing. Six files with the right names in ilastik_probabilities/ are reported as
``predicted: yes`` whether they were made this morning or by a classifier that has since been
retrained three times - and this project has been retrained three times in two days.

That gap is what makes generating provisional maps risky. Not the compute, which is fifteen
minutes, but that a placeholder becomes load-bearing precisely because nothing distinguishes
it from the real thing.
"""
import json

import pytest

from ImageLynx.artefact_provenance import (
    PROVENANCE_SUFFIX,
    provenance_path_for,
    probability_status,
    read_provenance,
    record_probability_provenance,
)
from ImageLynx.specimens import SPECIMENS, get_specimen


@pytest.fixture
def fake_classifier(tmp_path, monkeypatch):
    """A stand-in .ilp and probability map, with the registry pointed at them."""
    import ImageLynx.artefact_provenance as provenance_module
    import ImageLynx.specimens as specimens

    classifier = tmp_path / "vessel_segmentation.ilp"
    classifier.write_bytes(b"classifier version one")
    monkeypatch.setattr(specimens, "POOLED_CLASSIFIER", classifier)
    monkeypatch.setattr(provenance_module, "_label_summary",
                        lambda _p, _c=None: {"stub": True})

    specimen = get_specimen("WKY-C")
    artefact = tmp_path / "probs.h5"
    artefact.write_bytes(b"not really hdf5")
    return classifier, specimen, artefact


def test_the_sidecar_sits_beside_the_artefact_it_describes(tmp_path):
    artefact = tmp_path / "C1-VOL_ilastik_Probabilities.h5"
    assert provenance_path_for(artefact).name.endswith(PROVENANCE_SUFFIX)
    assert provenance_path_for(artefact).parent == artefact.parent


def test_recording_captures_the_classifier_that_made_it(fake_classifier):
    classifier, specimen, artefact = fake_classifier

    record = record_probability_provenance(specimen, artefact_path=artefact)

    assert record["specimen_id"] == "WKY-C"
    assert record["classifier_name"] == "vessel_segmentation.ilp"
    assert len(record["classifier_sha256"]) == 64
    assert record["recorded_at"]

    on_disk = json.loads(provenance_path_for(artefact).read_text())
    assert on_disk == record


def test_status_is_current_while_the_classifier_is_unchanged(fake_classifier):
    _classifier, specimen, artefact = fake_classifier
    record_probability_provenance(specimen, artefact_path=artefact)
    assert probability_status(specimen, artefact_path=artefact) == "current"


def test_status_goes_stale_when_the_classifier_is_retrained(fake_classifier):
    """The failure this exists to catch: a map that outlived the project that made it."""
    classifier, specimen, artefact = fake_classifier
    record_probability_provenance(specimen, artefact_path=artefact)

    classifier.write_bytes(b"classifier version two, relabelled")
    assert probability_status(specimen, artefact_path=artefact) == "stale"


def test_a_map_with_no_sidecar_is_unknown_not_assumed_current(fake_classifier):
    """Worse than stale: nothing at all is recorded, so nothing can be ruled out."""
    _classifier, specimen, artefact = fake_classifier
    assert probability_status(specimen, artefact_path=artefact) == "unknown"


def test_a_missing_map_is_absent(fake_classifier, tmp_path):
    _classifier, specimen, _artefact = fake_classifier
    assert probability_status(specimen, artefact_path=tmp_path / "nope.h5") == "absent"


def test_reading_a_corrupt_sidecar_does_not_raise(fake_classifier):
    _classifier, specimen, artefact = fake_classifier
    provenance_path_for(artefact).write_text("{ this is not json")

    assert read_provenance(artefact) is None
    assert probability_status(specimen, artefact_path=artefact) == "unknown"


def test_the_record_carries_the_labelling_behind_the_classifier(tmp_path, monkeypatch):
    """A hash says two classifiers differ; it does not say how.

    The label counts and boundary placement travel with the artefact so a result can be
    attributed to a labelling state rather than to an opaque 64-character string.
    """
    import ImageLynx.specimens as specimens

    classifier = tmp_path / "vessel_segmentation.ilp"
    classifier.write_bytes(b"x")
    monkeypatch.setattr(specimens, "POOLED_CLASSIFIER", classifier)

    artefact = tmp_path / "probs.h5"
    artefact.write_bytes(b"y")
    record = record_probability_provenance(get_specimen("SHR-A"), artefact_path=artefact)

    assert "labelling" in record
    # Unreadable classifier must degrade to a recorded absence, not an exception.
    assert record["labelling"] is None or isinstance(record["labelling"], dict)


@pytest.mark.parametrize("specimen", SPECIMENS, ids=lambda s: s.specimen_id)
def test_every_specimen_resolves_a_distinct_sidecar(specimen):
    others = {provenance_path_for(s.probabilities_path)
              for s in SPECIMENS if s.specimen_id != specimen.specimen_id}
    assert provenance_path_for(specimen.probabilities_path) not in others
