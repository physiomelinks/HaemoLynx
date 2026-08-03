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
    return (REPO_ROOT / "src" / "ImageLynx" / "pipeline" / "stages.py").read_text()


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


def test_the_legacy_argument_names_are_gone(pipeline):
    """`image_path` and friends were the old signature's names for settings that
    are called something else. Accepting both kept a translation table alive --
    the construct that let `axis_order` be silently dropped -- so they are gone
    and the setting names are the only spelling."""
    for retired in ("image_path", "axis_order", "do_pericyte_constriction"):
        with pytest.raises(ConfigError, match="Unknown setting"):
            pipeline.resolve_settings(overrides={retired: "x"})


def test_settings_reach_the_run_under_their_own_names(pipeline):
    settings = pipeline.resolve_settings(
        overrides={"input_path": "elsewhere/other.tif", "image_axis_order": "xyz"}
    )
    assert settings["input_path"] == Path("elsewhere/other.tif")
    assert settings["image_axis_order"] == "xyz"


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
    """Callers that name individual settings directly keep working."""
    recorded: dict = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda settings, schema: recorded.update(settings)
    )
    pipeline.image_to_model_pipeline(
        input_path=Path("a.tif"), plot_dir=Path("plots"), do_graph_building=False
    )
    assert recorded["input_path"] == Path("a.tif")
    assert recorded["plot_dir"] == Path("plots")
    assert recorded["do_graph_building"] is False
