"""Every key in the example YAMLs lands on the config it is written under.

``--config`` hands each top-level section to ``update_dataclass_from_dict``, which drops a key
the dataclass does not have with only a warning, and a section the loader does not read is
skipped without one. Open item 33: the WKY YAML kept its whole perfusion block under
``PipelineConfig``, so a ``--config`` run used the ``PerfusionConfig`` defaults instead. The
older YAML tests searched for keys at any depth and did not notice.
"""
import dataclasses
import logging
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

import carotid_image_to_model as pipeline

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
YAMLS = ["config_WKY_normotensive.yaml", "config_SHR_hypertensive.yaml"]

#: The sections the loader reads, in the order ``carotid_image_to_model.py`` applies them.
SECTIONS = {
    cls.__name__: cls
    for cls in (
        pipeline.PreprocessingConfig,
        pipeline.SkeletonConfig,
        pipeline.GraphConfig,
        pipeline.HaemodynamicsConfig,
        pipeline.PerfusionConfig,
        pipeline.VisualizationConfig,
        pipeline.PipelineConfig,
    )
}


def _load(name):
    return yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", YAMLS)
def test_every_section_is_one_the_loader_reads(name):
    unknown = sorted(set(_load(name)) - set(SECTIONS))
    assert not unknown, f"{name}: sections the loader never reads: {unknown}"


@pytest.mark.parametrize("name", YAMLS)
def test_every_key_is_a_field_of_its_section(name):
    stray = {}
    for section, keys in _load(name).items():
        fields = {f.name for f in dataclasses.fields(SECTIONS[section])}
        missing = sorted(set(keys or {}) - fields)
        if missing:
            stray[section] = missing
    assert not stray, f"{name}: keys that are not fields of their section: {stray}"


@pytest.mark.parametrize("name", YAMLS)
def test_loading_ignores_no_key(name, caplog):
    loaded = _load(name)
    configs = {section: SECTIONS[section]() for section in loaded}
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        for section, keys in loaded.items():
            pipeline.update_dataclass_from_dict(configs[section], keys)
    ignored = [r.getMessage() for r in caplog.records if "ignored" in r.getMessage()]
    assert not ignored, ignored

    # Every value reaches the section it is written under.
    for section, keys in loaded.items():
        for key, value in (keys or {}).items():
            got = getattr(configs[section], key)
            if isinstance(value, list):
                got = list(got)
            assert got == value, f"{name}: {section}.{key} is {got!r}, YAML says {value!r}"


def test_the_two_yamls_carry_the_same_settings():
    """A setting that differs by group would confound the WKY-vs-SHR comparison. Open item 34:
    the SHR YAML set arterial PCO2 35 mmHg against WKY's 40. Open item 35: SHR set both
    constriction ratios to 0.95 against WKY's 1.0, and only WKY turned on benchmarking."""
    wky, shr = (_load(name) for name in YAMLS)
    assert wky == shr


#: Keys the frozen method never reads: constriction is disabled (reference §10.6) and the
#: constant radius only applies outside edt_radius (§10.5).
RETIRED_HAEMODYNAMICS_KEYS = {
    "constrict_at_pericytes",
    "constriction_mode",
    "sphincter_length_um",
    "intimal_cushion_constriction_ratio",
    "pre_capillary_constriction_ratio",
    "pre_capillary_topological_offset",
    "constant_radius_um",
}


@pytest.mark.parametrize("name", YAMLS)
def test_the_yamls_use_the_frozen_radius_estimator(name):
    """Open item 35: both YAMLs set constant_radius (every edge 10 um across) and turned
    constriction on, so a --config run replaced every measured calibre."""
    section = _load(name)["HaemodynamicsConfig"]
    assert section["radius_assignment_mode"] == pipeline.HaemodynamicsConfig().radius_assignment_mode
    assert section["radius_assignment_mode"] == "edt_radius"
    retired = sorted(RETIRED_HAEMODYNAMICS_KEYS & set(section))
    assert not retired, f"{name}: keys the frozen method never reads: {retired}"
