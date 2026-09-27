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

    # The perfusion block now reaches PerfusionConfig. The YAMLs' 1e-4 is below the config's
    # 1e-5 until open item 34, so it shows the value was applied rather than defaulted.
    assert configs["PerfusionConfig"].picard_tolerance == 1e-4
