"""Segmentation source follows a synthetic template's diameter basis when the
template is picked (``gui/template_basis.py``)."""
from __future__ import annotations

import pytest

from haemolynx.gui.template_basis import TRIGGER_SETTINGS, template_basis_update
from haemolynx.synthetic import clear_template_cache
from synthetic_fixtures import write_fixture_run


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_template_cache()
    yield
    clear_template_cache()


@pytest.fixture
def claudin_run(tmp_path):
    """A template run whose segmentation was of the vessel wall."""
    return write_fixture_run(tmp_path, settings={"segmentation_source": "endothelial"}, name="claudin.haemorun")


def _values(template, **overrides):
    return {
        "network_source": "synthetic",
        "synthetic_template_path": template,
        "segmentation_source": "plasma_column",
        **overrides,
    }


def test_picking_a_template_sets_the_segmentation_source_to_its_basis(claudin_run):
    update = template_basis_update(_values(None), _values(claudin_run))
    assert update.value == "endothelial"
    assert "Segmentation source set to endothelial" in update.line


def test_switching_to_synthetic_with_a_template_picked_sets_it_too(claudin_run):
    before = _values(claudin_run, network_source="segmented_image")
    assert template_basis_update(before, _values(claudin_run)).value == "endothelial"


def test_nothing_changes_unless_a_trigger_did(claudin_run):
    values = _values(claudin_run)
    assert template_basis_update(values, values).value is None
    # Another row changing is not a trigger.
    assert template_basis_update({**values, "synthetic_seed": 1}, {**values, "synthetic_seed": 2}).value is None
    assert set(TRIGGER_SETTINGS) == {"network_source", "synthetic_template_path"}


def test_nothing_changes_for_an_image_run_or_a_matching_row(claudin_run):
    assert template_basis_update(
        _values(None, network_source="segmented_image"), _values(claudin_run, network_source="segmented_image")
    ).value is None
    assert template_basis_update(_values(None), _values(claudin_run, segmentation_source="endothelial")) == type(
        template_basis_update(_values(None), _values(None))
    )()


def test_a_template_that_cannot_be_read_changes_nothing(tmp_path):
    tif = tmp_path / "stack.tif"
    tif.write_bytes(b"\x00")
    renamed = tmp_path / "renamed.haemorun"
    renamed.write_bytes(b"not a run")
    for path in (tif, renamed, tmp_path / "missing.haemorun", ""):
        update = template_basis_update(_values(None), _values(path))
        assert update.value is None and update.line is None, path


def test_a_template_naming_no_basis_changes_nothing(tmp_path):
    # Without segmentation_source in its settings or a basis on its graph, a
    # run's basis is unknown: the row is left as it is.
    path = write_fixture_run(tmp_path, settings={"segmentation_source": None})
    assert template_basis_update(_values(None), _values(path)).value is None
