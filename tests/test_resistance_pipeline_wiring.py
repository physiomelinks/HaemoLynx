"""The config file reaching the resistance pipeline's stages.

These exercise the settings layer only — config in, resolved settings out — so
they stay fast and do not touch image data.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("yaml")

from ImageLynx.parsers import ConfigError  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
    spec = importlib.util.spec_from_file_location("resistance_network_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source() -> str:
    """The stage runner's source: it moved to the library so both examples share it."""
    return (REPO_ROOT / "src" / "ImageLynx" / "pipeline.py").read_text()


# --- loading ---------------------------------------------------------------


def test_settings_come_from_the_committed_config_file(pipeline):
    settings = pipeline.resolve_settings()
    assert settings["input_path"].name == "brain_microvessels.tiff"
    assert set(pipeline.SCHEMA.names) <= set(settings)


def test_overrides_are_applied_and_validated(pipeline):
    assert pipeline.resolve_settings(overrides={"do_skeletonize": False})[
        "do_skeletonize"
    ] is False

    with pytest.raises(ConfigError, match="Unknown setting 'do_skeletonise'"):
        pipeline.resolve_settings(overrides={"do_skeletonise": False})


def test_an_out_of_range_value_is_refused_before_any_image_is_read(pipeline):
    with pytest.raises(ConfigError, match="small_vessel_mask_min_overlap_fraction"):
        pipeline.resolve_settings(
            overrides={"small_vessel_mask_min_overlap_fraction": 5.0}
        )


def test_an_already_resolved_dict_can_be_passed_back_in(pipeline):
    """Resolving twice must be a no-op, derived entries and all."""
    once = pipeline.resolve_settings()
    twice = pipeline.resolve_settings(once)
    assert twice == once


# --- derived settings ------------------------------------------------------


def test_the_branch_order_tables_are_derived_when_the_config_leaves_them_unset(pipeline):
    settings = pipeline.resolve_settings()
    diameters = settings["diameter_by_branch_order"]
    constrictions = settings["constriction_by_branch_order"]

    assert diameters["B01"] > 0
    assert constrictions["B01"] == 1.0
    assert constrictions["B02"] == 0.8
    assert f"B{settings['max_branch_order']:02d}" in diameters


def test_an_explicit_diameter_table_is_left_alone(pipeline):
    settings = pipeline.resolve_settings(
        overrides={"diameter_by_branch_order": {"B01": 3.0}}
    )
    assert settings["diameter_by_branch_order"] == {"B01": 3.0}


def test_the_plot_directory_is_derived_from_the_base_plot_directory(pipeline):
    settings = pipeline.resolve_settings(overrides={"base_plot_dir": "somewhere/plots"})
    assert settings["plot_dir"] == Path("somewhere/plots/nerve")


def test_an_explicit_plot_directory_wins(pipeline):
    settings = pipeline.resolve_settings(overrides={"plot_dir": "chosen/dir"})
    assert settings["plot_dir"] == Path("chosen/dir")


# --- how the stages read settings ------------------------------------------


def test_the_stage_runner_takes_the_settings_dict_and_a_schema(pipeline):
    """The 127-parameter signature is gone; settings are read by name."""
    assert list(inspect.signature(pipeline.run_pipeline_stages).parameters) == [
        "settings",
        "schema",
    ]


def test_the_module_no_longer_star_imports_its_settings():
    example = (REPO_ROOT / "examples" / "resistance_network_pipeline.py").read_text()
    assert "from resistance_pipeline_settings import *" not in example


def test_settings_once_read_from_module_globals_are_read_from_the_dict(source, pipeline):
    """`vtk_export`, `statistics` and `custom_edges` were module globals, so
    configuring them did nothing at all."""
    for setting in ("vtk_export", "statistics"):
        assert f'settings["{setting}"]' in source
    assert "VTK_export" not in source
    # custom_edges reaches haemodynamics inside the diameters section.
    assert "custom_edges" in pipeline.SCHEMA.section_names("Diameters and pericytes")


@pytest.mark.parametrize(
    "argument_name,setting_name,value",
    [
        ("image_path", "input_path", Path("elsewhere/other.tif")),
        ("axis_order", "image_axis_order", "xyz"),
        ("do_pericyte_constriction", "do_pericyte_construction", True),
    ],
)
def test_overrides_may_name_the_old_argument_or_the_setting(
    pipeline, argument_name, setting_name, value
):
    """Regression: `axis_order` had no entry in the old kwargs bridge, so a
    configured `IMAGE_AXIS_ORDER` was dropped and the import-time default won."""
    by_argument = pipeline.resolve_settings(overrides={argument_name: value})
    by_setting = pipeline.resolve_settings(overrides={setting_name: value})
    assert by_argument[setting_name] == value
    assert by_setting[setting_name] == value


def test_node_lists_stay_mutable(pipeline):
    """The stages fill these in place, so a tuple would raise mid-run."""
    settings = pipeline.resolve_settings()
    for name in ("starting_nodes", "output_nodes"):
        settings[name][:] = [1, 2]
        assert settings[name] == [1, 2]


# --- the public entry point ------------------------------------------------


def test_the_entry_point_takes_the_settings_dict(pipeline, monkeypatch):
    recorded: dict = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda settings, schema: recorded.update(settings)
    )
    pipeline.image_to_model_pipeline(
        pipeline.resolve_settings(overrides={"do_skeletonize": False})
    )
    assert recorded["do_skeletonize"] is False


def test_the_entry_point_runs_the_config_file_with_no_arguments(pipeline, monkeypatch):
    recorded: dict = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda settings, schema: recorded.update(settings)
    )
    pipeline.image_to_model_pipeline()
    assert recorded["input_path"].name == "brain_microvessels.tiff"


def test_the_entry_point_still_accepts_individual_overrides(pipeline, monkeypatch):
    """Callers that name values directly keep working, in either spelling."""
    recorded: dict = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda settings, schema: recorded.update(settings)
    )
    pipeline.image_to_model_pipeline(
        image_path=Path("a.tif"), plot_dir=Path("plots"), do_graph_building=False
    )
    assert recorded["input_path"] == Path("a.tif")
    assert recorded["plot_dir"] == Path("plots")
    assert recorded["do_graph_building"] is False
