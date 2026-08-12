"""The carotid example: it imports, and it runs from its config file.

Both were once untrue -- the module raised ``NameError: PLOT_DIR`` at import
and its ``__main__`` called the entry point with an argument it did not take,
so nothing here could ever have run. These tests exercise the settings layer
only, so they stay fast and touch no image data.
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

EXAMPLE_PATH = REPO_ROOT / "examples" / "carotid_image_to_model.py"


@pytest.fixture(scope="module")
def carotid():
    spec = importlib.util.spec_from_file_location("carotid_image_to_model", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source() -> str:
    return EXAMPLE_PATH.read_text(encoding="utf-8")


# --- settings --------------------------------------------------------------


def test_settings_come_from_the_committed_config_file(carotid):
    settings = carotid.resolve_settings()
    assert settings["input_path"].name == "carotid_mask.tif"
    assert set(carotid.SCHEMA.names) <= set(settings)


def test_the_config_and_the_schema_describe_the_same_settings(carotid):
    from haemolynx.parsers import load_config

    settings = load_config(carotid.CONFIG_PATH, carotid.SCHEMA)
    assert set(settings) == set(carotid.SCHEMA.names)


def test_the_schema_is_the_pipeline_schema_with_this_dataset_s_values(carotid):
    """A carotid-only setting would be a setting the main pipeline cannot read."""
    from haemolynx.pipeline.schema import default_schema

    PIPELINE_SCHEMA = default_schema()

    assert set(carotid.SCHEMA.names) == set(PIPELINE_SCHEMA.names)
    assert carotid.SCHEMA["inlet_p_bc"].default != PIPELINE_SCHEMA["inlet_p_bc"].default


def test_the_boundary_nodes_come_from_the_ends_of_the_image(carotid):
    settings = carotid.resolve_settings()
    assert settings["inlet_node_selection_method"] == "edge_percent"
    assert settings["outlet_node_selection_method"] == "edge_percent"


def test_plots_go_in_their_own_directory(carotid):
    settings = carotid.resolve_settings()
    assert settings["plot_dir"] == Path(settings["base_plot_dir"]) / "carotid"


def test_an_explicit_plot_directory_wins(carotid, tmp_path):
    settings = carotid.resolve_settings(overrides={"plot_dir": tmp_path})
    assert settings["plot_dir"] == tmp_path
    assert carotid.resolve_settings(settings)["plot_dir"] == tmp_path


def test_overrides_are_validated_against_the_schema(carotid):
    assert carotid.resolve_settings(overrides={"do_skeletonize": False})[
        "do_skeletonize"
    ] is False

    with pytest.raises(ConfigError, match="Unknown setting 'do_skeletonise'"):
        carotid.resolve_settings(overrides={"do_skeletonise": False})


# --- entry point -----------------------------------------------------------


def test_the_entry_point_takes_a_settings_dict(carotid):
    """It used to be called with `input_path=`, which was never a parameter."""
    parameters = list(inspect.signature(carotid.main).parameters)
    assert parameters[0] == "settings"


def test_the_entry_point_runs_the_shared_stages_not_a_fork(carotid, source):
    assert "run_pipeline_stages" in source
    for forked_call in (
        "build_graph_segment_skan_stitched_loops",
        "preprocess_skeleton_for_graph",
        "load_and_skeletonize_3d_tif",
        "graph_to_vtk",
    ):
        assert forked_call not in source, f"{forked_call} belongs to haemolynx.pipeline"


def test_the_command_line_is_generated_from_the_schema(source):
    assert "settings_from_command_line(" in source
    assert "argparse" not in source


def test_a_missing_image_is_reported_by_name(carotid, tmp_path):
    """The run reaches the pipeline: it fails on the data, not on the wiring."""
    missing = tmp_path / "no_such_carotid.tif"
    with pytest.raises(FileNotFoundError, match="no_such_carotid.tif"):
        carotid.main(input_path=missing)


def test_the_preflight_check_refuses_a_missing_input(carotid, tmp_path):
    settings = carotid.resolve_settings(overrides={"input_path": tmp_path / "gone.tif"})
    with pytest.raises(SystemExit) as exit_info:
        carotid._preflight_or_exit(settings)
    assert exit_info.value.code == 2
