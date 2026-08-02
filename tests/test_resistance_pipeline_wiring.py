"""The config file reaching the resistance pipeline's stages.

These exercise the mapping layer only — settings in, stage arguments out — so
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


# --- loading ---------------------------------------------------------------


def test_settings_come_from_the_committed_config_file(pipeline):
    settings = pipeline.resolve_settings()
    assert settings["input_path"].name == "brain_microvessels.tiff"
    assert set(settings) == set(pipeline.SCHEMA.names)


def test_overrides_are_applied_and_validated(pipeline):
    settings = pipeline.resolve_settings(overrides={"do_skeletonize": False})
    assert settings["do_skeletonize"] is False

    with pytest.raises(ConfigError, match="Unknown setting 'do_skeletonise'"):
        pipeline.resolve_settings(overrides={"do_skeletonise": False})


def test_an_out_of_range_value_is_refused_before_any_image_is_read(pipeline):
    with pytest.raises(ConfigError, match="small_vessel_mask_min_overlap_fraction"):
        pipeline.resolve_settings(overrides={"small_vessel_mask_min_overlap_fraction": 5.0})


# --- derived settings ------------------------------------------------------


def test_the_branch_order_tables_are_derived_when_the_config_leaves_them_unset(pipeline):
    settings = pipeline.resolve_settings()
    diameters = settings["diameter_by_branch_order"]
    constrictions = settings["constriction_by_branch_order"]

    assert diameters["B01"] > 0
    assert constrictions["B01"] == 1.0
    assert constrictions["B02"] == 0.8
    # Derived from max_branch_order, so the tables must cover it.
    assert f"B{settings['max_branch_order']:02d}" in diameters


def test_an_explicit_diameter_table_is_left_alone(pipeline):
    settings = pipeline.resolve_settings(
        overrides={"diameter_by_branch_order": {"B01": 3.0}}
    )
    assert settings["diameter_by_branch_order"] == {"B01": 3.0}


# --- settings to stage arguments -------------------------------------------


def test_every_stage_argument_is_supplied(pipeline):
    arguments = pipeline.stage_arguments(pipeline.resolve_settings())
    assert set(arguments) == set(pipeline.STAGE_PARAMETERS)


@pytest.mark.parametrize(
    "setting_name,argument_name",
    [
        ("input_path", "image_path"),
        ("image_axis_order", "axis_order"),
        ("do_pericyte_construction", "do_pericyte_constriction"),
    ],
)
def test_settings_reach_the_argument_they_are_named_differently_from(
    pipeline, setting_name, argument_name
):
    """Regression: `axis_order` had no alias entry, so a configured
    `IMAGE_AXIS_ORDER` was silently dropped and the import-time default won."""
    schema_setting = pipeline.SCHEMA[setting_name]
    value = (
        "xyz"
        if setting_name == "image_axis_order"
        else (Path("elsewhere/other.tif") if schema_setting.kind == "path" else True)
    )
    settings = pipeline.resolve_settings(overrides={setting_name: value})
    assert pipeline.stage_arguments(settings)[argument_name] == value


def test_the_plot_directory_is_derived_from_the_base_plot_directory(pipeline):
    settings = pipeline.resolve_settings(overrides={"base_plot_dir": "somewhere/plots"})
    assert pipeline.stage_arguments(settings)["plot_dir"] == Path("somewhere/plots/nerve")


def test_settings_consumed_by_the_mapping_are_not_passed_on(pipeline):
    """The derived-table inputs configure this layer, not the stages."""
    arguments = pipeline.stage_arguments(pipeline.resolve_settings())
    for name in ("all_diams_const", "max_branch_order", "base_plot_dir"):
        assert name not in arguments


def test_node_lists_stay_mutable(pipeline):
    """The stages fill these in place, so a tuple would raise mid-run."""
    settings = pipeline.resolve_settings()
    for name in ("starting_nodes", "output_nodes"):
        settings[name][:] = [1, 2]
        assert settings[name] == [1, 2]


# --- settings the stages read as module globals ----------------------------


def test_settings_read_as_globals_are_applied(pipeline, monkeypatch):
    """`vtk_export`, `statistics` and `custom_edges` are read from module
    globals inside the stages, so the wiring has to set them or a config
    change to any of the three would be silently lost."""
    recorded = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda **kwargs: recorded.update(kwargs)
    )
    pipeline.image_to_model_pipeline(statistics=False, vtk_export=False)

    assert pipeline.STATISTICS is False
    assert pipeline.VTK_export is False
    assert recorded, "the stages should still have been called"


# --- the public entry point ------------------------------------------------


def test_the_entry_point_takes_the_settings_dict(pipeline, monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda **kwargs: recorded.update(kwargs)
    )
    settings = pipeline.resolve_settings(overrides={"do_skeletonize": False})
    pipeline.image_to_model_pipeline(settings)
    assert recorded["do_skeletonize"] is False


def test_the_entry_point_still_accepts_individual_stage_arguments(pipeline, monkeypatch):
    """Callers that name arguments directly keep working, in either spelling."""
    recorded = {}
    monkeypatch.setattr(
        pipeline, "run_pipeline_stages", lambda **kwargs: recorded.update(kwargs)
    )
    pipeline.image_to_model_pipeline(
        image_path=Path("a.tif"), plot_dir=Path("plots"), do_graph_building=False
    )
    assert recorded["image_path"] == Path("a.tif")
    assert recorded["plot_dir"] == Path("plots")
    assert recorded["do_graph_building"] is False
