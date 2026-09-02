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

from haemolynx.parsers import ConfigError  # noqa: E402


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
    return (REPO_ROOT / "src" / "haemolynx" / "pipeline" / "stages.py").read_text()


# --- loading ---------------------------------------------------------------


def test_settings_come_from_the_committed_config_file(pipeline):
    settings = pipeline.resolve_settings()
    assert settings["input_path"].name == "Nerve_capillaries.tif"
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


#: The stages a run is made of, in order, and what each returns.
STAGES = [
    ("segment", "SegmentedInputs"),
    ("skeletonise", "SkeletonisedVolume"),
    ("build_network", "VesselNetwork"),
    ("assign_boundaries", "BoundaryNodes"),
    ("assign_diameters", "HaemodynamicModel"),
    ("build_haemodynamic_model", "HaemodynamicModel"),
    ("solve", "Solution"),
    ("run_perturbations", "PerturbationRun"),
    ("export_results", "Solution"),
]

#: How each stage's call reads in the example. The last two are called for what
#: they write rather than for a value the script goes on to use.
CALL_MARKERS = {name: f"= {name}(" for name, _ in STAGES}
CALL_MARKERS["run_perturbations"] = "run_perturbations("
CALL_MARKERS["export_results"] = "export_results("


@pytest.mark.parametrize("stage_name,_returns", STAGES)
def test_every_stage_takes_the_settings_dict_first(stage_name, _returns):
    """The 127-parameter signature is gone; each stage reads settings by name."""
    from haemolynx import pipeline as pipeline_package

    stage = getattr(pipeline_package, stage_name)
    assert list(inspect.signature(stage).parameters)[0] == "settings"


def test_the_example_calls_the_stages_in_order():
    """The example is the stages, in order, so a reader sees the shape of a run."""
    example = (REPO_ROOT / "examples" / "resistance_network_pipeline.py").read_text()
    positions = [example.index(CALL_MARKERS[name]) for name, _ in STAGES]
    assert positions == sorted(positions), "stages are called out of order"


def test_the_example_perturbs_the_baseline_before_exporting_it():
    """Order matters here: the export must write the baseline, not a perturbed copy.

    `run_perturbations` comes after `solve`, so each perturbation has solved
    pressures to difference against, and before `export_results`, which writes
    out the one graph every perturbation was started from.
    """
    example = (REPO_ROOT / "examples" / "resistance_network_pipeline.py").read_text()
    assert (
        example.index("= solve(")
        < example.index("run_perturbations(settings")
        < example.index("export_results(settings")
    )


def test_the_constants_module_is_gone():
    """Defaults live in the schema; a second copy could only drift from it."""
    assert not (REPO_ROOT / "examples" / "resistance_pipeline_settings.py").exists()
    assert not (REPO_ROOT / "examples" / "presets.py").exists()


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
    for name in ("inlet_nodes", "outlet_nodes"):
        settings[name][:] = [1, 2]
        assert settings[name] == [1, 2]


# --- the public entry point ------------------------------------------------


class _StopAfterFirstStage(Exception):
    """Raised by the stubbed first stage; the rest of the run is not the point."""


@pytest.fixture
def settings_reaching_the_first_stage(pipeline, monkeypatch):
    """Capture what the entry point hands to `segment`, then stop the run."""
    recorded: dict = {}

    def _capture(settings):
        recorded.update(settings)
        raise _StopAfterFirstStage

    monkeypatch.setattr(pipeline, "segment", _capture)
    return recorded


def test_the_entry_point_takes_the_settings_dict(
    pipeline, settings_reaching_the_first_stage
):
    with pytest.raises(_StopAfterFirstStage):
        pipeline.image_to_model_pipeline(
            pipeline.resolve_settings(overrides={"do_skeletonize": False})
        )
    assert settings_reaching_the_first_stage["do_skeletonize"] is False


def test_the_entry_point_runs_the_config_file_with_no_arguments(
    pipeline, settings_reaching_the_first_stage
):
    with pytest.raises(_StopAfterFirstStage):
        pipeline.image_to_model_pipeline()
    assert settings_reaching_the_first_stage["input_path"].name == "Nerve_capillaries.tif"


def test_the_entry_point_still_accepts_individual_overrides(
    pipeline, settings_reaching_the_first_stage
):
    """Callers that name individual settings directly keep working."""
    with pytest.raises(_StopAfterFirstStage):
        pipeline.image_to_model_pipeline(
            input_path=Path("a.tif"), plot_dir=Path("plots"), do_graph_building=False
        )
    assert settings_reaching_the_first_stage["input_path"] == Path("a.tif")
    assert settings_reaching_the_first_stage["plot_dir"] == Path("plots")
    assert settings_reaching_the_first_stage["do_graph_building"] is False


# --- pre-run checks ---------------------------------------------------------


def test_preflight_is_derived_from_the_schema_not_a_hand_written_list():
    """The example's preflight.py is gone; the checks read the schema."""
    from haemolynx.pipeline import preflight

    assert not (REPO_ROOT / "examples" / "preflight.py").exists()
    assert callable(preflight)


def test_preflight_accepts_a_runnable_configuration(pipeline, tmp_path):
    from haemolynx.pipeline import preflight

    image = tmp_path / "mask.tif"
    image.write_bytes(b"x")
    settings = pipeline.resolve_settings(overrides={"input_path": image})
    assert preflight(settings, pipeline.SCHEMA).ok


def test_preflight_rejects_a_missing_input_image(pipeline, tmp_path):
    from haemolynx.pipeline import preflight

    settings = pipeline.resolve_settings(overrides={"input_path": tmp_path / "absent.tif"})
    report = preflight(settings, pipeline.SCHEMA)
    assert not report.ok
    assert any("input_path" in message for message in report.errors)


def test_preflight_demands_the_masks_a_toggle_turns_on(pipeline, tmp_path):
    """Turning on large-vessel masks makes their paths required."""
    from haemolynx.pipeline import preflight

    image = tmp_path / "mask.tif"
    image.write_bytes(b"x")
    settings = pipeline.resolve_settings(
        overrides={"input_path": image, "use_large_vessel_masks": True}
    )
    report = preflight(settings, pipeline.SCHEMA)
    assert not report.ok
    assert any("large_arteriole_mask_path" in message for message in report.errors)


def test_preflight_requires_a_cached_graph_when_graph_building_is_off(pipeline, tmp_path):
    from haemolynx.pipeline import preflight

    image = tmp_path / "mask.tif"
    image.write_bytes(b"x")
    settings = pipeline.resolve_settings(
        overrides={
            "input_path": image,
            "do_graph_building": False,
            "vtk_output_prefix": tmp_path / "outputs" / "network",
        }
    )
    report = preflight(settings, pipeline.SCHEMA)
    assert any("do_graph_building" in message for message in report.errors)
