"""Tests for preset/config/wizard/preflight pipeline tooling."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import presets as preset_tools
import preflight
import resistance_pipeline_settings as pipeline_settings
import wizard


def test_statistics_only_preset_settings_resolve():
    preset_descriptions = pipeline_settings.list_presets()
    assert "statistics_only" in preset_descriptions

    resolved = pipeline_settings.build_settings_for_preset("statistics_only")
    assert resolved["STATISTICS"] is True
    assert resolved["RUN_HAEMODYNAMICS"] is True
    assert resolved["DO_EQUIV_RESISTANCE_CALCULATION"] is False
    assert resolved["USE_FWHM_EDGE_DIAMETERS"] is False


def test_yaml_config_load_and_save_roundtrip(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    image_path = tmp_path / "input_image.tif"
    image_path.write_bytes(b"dummy")

    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                "preset: quick_debug",
                "settings:",
                f"  INPUT_PATH: '{image_path.as_posix()}'",
                "  RUN_HAEMODYNAMICS: true",
                "pipeline:",
                f"  image_path: '{image_path.as_posix()}'",
                "  run_haemodynamics: false",
            ]
        ),
        encoding="utf-8",
    )

    loaded = preset_tools.load_config_yaml(
        config_path=config_path,
        valid_setting_names=pipeline_settings.VALID_SETTING_NAMES,
        available_preset_names=set(pipeline_settings.PRESET_DEFINITIONS.keys()),
        pipeline_param_names={"image_path", "run_haemodynamics"},
    )
    assert loaded["preset_name"] == "quick_debug"
    assert loaded["settings_overrides"]["INPUT_PATH"] == image_path
    assert loaded["settings_overrides"]["RUN_HAEMODYNAMICS"] is True
    assert loaded["pipeline_overrides"]["image_path"] == image_path
    assert loaded["pipeline_overrides"]["run_haemodynamics"] is False

    output_yaml = tmp_path / "effective.yaml"
    saved = preset_tools.save_effective_config_yaml(
        output_path=output_yaml,
        preset_name=loaded["preset_name"],
        settings=loaded["settings_overrides"],
        pipeline_kwargs=loaded["pipeline_overrides"],
    )
    assert saved.exists()
    serialized = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert serialized["preset"] == "quick_debug"
    assert Path(serialized["settings"]["INPUT_PATH"]) == image_path
    assert Path(serialized["pipeline"]["image_path"]) == image_path


def test_interactive_wizard_collects_expected_overrides(monkeypatch: pytest.MonkeyPatch):
    answers = iter(
        [
            "statistics_only",  # preset
            "C:/data/input.tif",  # image path
            "y",  # use large masks
            "n",  # ilastik large
            "y",  # use small masks
            "y",  # ilastik small
            "",  # automated assignment default=True
            "n",  # run haemodynamics
            "y",  # run statistics
            "y",  # run 3d distance
            "C:/data/cell_mask.tif",  # cell mask path
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    result = wizard.run_interactive_setup_wizard(
        default_preset="default",
        available_presets=["default", "statistics_only"],
    )

    assert result["preset_name"] == "statistics_only"
    assert result["pipeline_overrides"]["image_path"] == Path("C:/data/input.tif")
    assert result["settings_overrides"]["USE_LARGE_VESSEL_MASKS"] is True
    assert result["settings_overrides"]["USE_ILASTIK_LARGE_VESSEL_SEGMENTATION"] is False
    assert result["settings_overrides"]["USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT"] is True
    assert result["settings_overrides"]["USE_ILASTIK_SMALL_VESSEL_SEGMENTATION"] is True
    assert result["settings_overrides"]["AUTOMATED_VESSEL_ASSIGNMENT"] is True
    assert result["settings_overrides"]["RUN_HAEMODYNAMICS"] is False
    assert result["settings_overrides"]["STATISTICS"] is True
    assert result["settings_overrides"]["MEASUREMENT_3D_TO_CELL_MASK"] is True
    assert result["settings_overrides"]["CELL_MASK_PATH"] == Path("C:/data/cell_mask.tif")


def test_preflight_reports_pass_and_actionable_failures(tmp_path: Path):
    image_path = tmp_path / "segmented_input.tif"
    image_path.write_bytes(b"dummy")

    base_kwargs = {
        "image_path": image_path,
        "use_ilastik_segmentation": False,
        "final_render_mode": "3d",
        "ide_plot_mode": "final_only",
        "statistics_mode": "fast",
        "run_haemodynamics": False,
        "do_equiv_resistance_calculation": False,
        "use_fwhm_edge_diameters": False,
        "measurement_3d_to_cell_mask": False,
    }

    ok_report = preflight.run_preflight_checklist(base_kwargs)
    assert ok_report["ok"] is True
    assert ok_report["errors"] == []

    failing_kwargs = dict(base_kwargs)
    failing_kwargs["measurement_3d_to_cell_mask"] = True
    failing_kwargs["cell_mask_path"] = None
    failing_report = preflight.run_preflight_checklist(failing_kwargs)
    assert failing_report["ok"] is False
    assert any("3D distance cell mask" in err for err in failing_report["errors"])


def test_preflight_fails_for_missing_ilastik_executable(tmp_path: Path):
    unsegmented = tmp_path / "unsegmented.tif"
    classifier = tmp_path / "classifier.ilp"
    unsegmented.write_bytes(b"dummy")
    classifier.write_bytes(b"dummy")

    kwargs = {
        "image_path": unsegmented,
        "use_ilastik_segmentation": True,
        "ilastik_unsegmented_image_path": unsegmented,
        "ilastik_classifier_path": classifier,
        "ilastik_executable": "definitely_not_a_real_ilastik_binary_12345",
        "final_render_mode": "3d",
        "ide_plot_mode": "final_only",
        "statistics_mode": "fast",
        "run_haemodynamics": False,
        "do_skeletonize": True,
        "do_graph_building": True,
    }
    report = preflight.run_preflight_checklist(kwargs)
    assert report["ok"] is False
    assert any("Ilastik executable" in err for err in report["errors"])


def test_preflight_fails_for_strict_branch_mode_without_mask_automation(tmp_path: Path):
    image_path = tmp_path / "segmented_input.tif"
    image_path.write_bytes(b"dummy")

    kwargs = {
        "image_path": image_path,
        "use_ilastik_segmentation": False,
        "final_render_mode": "3d",
        "ide_plot_mode": "final_only",
        "statistics_mode": "fast",
        "run_haemodynamics": False,
        "do_skeletonize": True,
        "do_graph_building": True,
        "strict_branch_order_assignment": True,
        "automated_vessel_assignment": False,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }
    report = preflight.run_preflight_checklist(kwargs)
    assert report["ok"] is False
    assert any("Strict branch-order assignment" in err for err in report["errors"])


def test_resolve_preset_inheritance_merges_parent_overrides():
    preset_defs = {
        "base": {
            "description": "Base preset",
            "overrides": {"A": 1, "B": 2},
        },
        "child": {
            "extends": "base",
            "description": "Child preset",
            "overrides": {"B": 99, "C": 3},
        },
    }
    resolved = preset_tools.resolve_preset_inheritance(preset_defs)
    assert resolved["child"]["description"] == "Child preset"
    assert resolved["child"]["overrides"] == {"A": 1, "B": 99, "C": 3}


def test_resolve_preset_inheritance_detects_cycle():
    cyclic_defs = {
        "a": {"extends": "b", "overrides": {}},
        "b": {"extends": "a", "overrides": {}},
    }
    with pytest.raises(ValueError, match="Cyclic preset inheritance"):
        preset_tools.resolve_preset_inheritance(cyclic_defs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))


def test_preflight_validates_input_axis_order(tmp_path: Path):
    image_path = tmp_path / "segmented_input.tif"
    image_path.write_bytes(b"dummy")

    base_kwargs = {
        "image_path": image_path,
        "use_ilastik_segmentation": False,
        "final_render_mode": "3d",
        "ide_plot_mode": "final_only",
        "statistics_mode": "fast",
        "run_haemodynamics": False,
        "do_equiv_resistance_calculation": False,
        "use_fwhm_edge_diameters": False,
        "measurement_3d_to_cell_mask": False,
    }

    # Any permutation of xyz is accepted; the default is the canonical order.
    for axis_order in ("zyx", "xyz", "YZX"):
        report = preflight.run_preflight_checklist({**base_kwargs, "axis_order": axis_order})
        assert report["ok"] is True, axis_order
        assert report["errors"] == []

    bad_report = preflight.run_preflight_checklist({**base_kwargs, "axis_order": "zzz"})
    assert bad_report["ok"] is False
    assert any("Input axis order" in err for err in bad_report["errors"])
