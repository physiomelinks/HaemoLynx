"""The one stem a run's outputs are named with (``pipeline/naming.py``)."""
from __future__ import annotations

from pathlib import Path

from haemolynx.pipeline.naming import (
    SURROUNDED_STEM_SUFFIX,
    SYNTHETIC_STEM_SUFFIX,
    run_input_path,
    run_stem,
    synthetic_run_stem,
)


def test_an_image_run_is_named_after_its_image():
    settings = {"input_path": Path("data") / "stack.tif", "network_source": "segmented_image"}
    assert run_stem(settings) == "stack"
    assert run_input_path(settings) == Path("data") / "stack.tif"
    assert run_stem({"input_path": "C:/x/mask.ome.tiff"}) == "mask.ome"


def test_a_synthetic_run_is_named_after_its_template():
    settings = {
        "input_path": Path("data") / "stack.tif",  # left over from an image run: not read
        "network_source": "synthetic",
        "synthetic_template_path": Path("runs") / "e14_5.haemorun",
    }
    assert run_stem(settings) == "e14_5_synthetic"
    assert run_input_path(settings) is None
    assert synthetic_run_stem("a/b/template.haemorun") == "template" + SYNTHETIC_STEM_SUFFIX


def test_the_template_runs_network_inside_a_synthetic_one_is_named_surrounded():
    settings = {
        "input_path": Path("data") / "stack.tif",
        "network_source": "synthetic_surround",
        "synthetic_template_path": Path("runs") / "e14_5.haemorun",
    }
    assert run_stem(settings) == "e14_5_surrounded"
    assert run_input_path(settings) is None
    assert synthetic_run_stem("a/b/template.haemorun", surrounded=True) == "template" + SURROUNDED_STEM_SUFFIX


def test_no_image_and_no_template_give_no_stem():
    assert run_stem({}) is None
    assert run_stem({"input_path": None}) is None
    assert run_stem({"input_path": ""}) is None
    assert run_stem({"network_source": "synthetic"}) is None
    assert run_stem({"network_source": "synthetic", "synthetic_template_path": ""}) is None
    assert run_input_path({}) is None
