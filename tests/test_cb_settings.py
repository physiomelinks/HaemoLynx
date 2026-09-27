"""The analysis settings have one owner, and the known disagreements cannot grow.

``ImageLynx.cb_settings`` exists because four of the open items in
``cb_modelling_reference.md`` were the same defect: a constant written out separately in a
driver had drifted from the ``carotid_image_to_model.py`` config default it was meant to
match, and nothing noticed.

Two things are tested here.

1. No driver defines those constants as literals any more. A shared module only helps if
   nobody reintroduces a local copy.
2. The pipeline config agrees with the settings on each value that used to disagree: the
   hysteresis band (open item 1), the boundary rule (item 2), the metabolic rate (item 8),
   the pressures (item 10) and the perfusion stop settings (item 6). Each was resolved in favour of the settings, because
   every published H1 and H2 number was produced at the settings values.
"""
import ast
from pathlib import Path

import pytest

from ImageLynx import cb_settings

REPO = Path(__file__).resolve().parents[1]
DRIVERS = sorted((REPO / "examples").glob("cb_*.py"))

#: Constant names that must not be assigned a bare literal in any driver.
OWNED = {
    "ROI", "DEFAULT_ROI", "BOUNDARY_AXIS", "TH_THRESHOLD", "GRID_UM",
    "PENETRATION", "BASE_M_MAX", "FROZEN_THRESHOLD", "FROZEN_VESSEL_THRESHOLD",
    "DEFAULT_GRID",
}


def _module_level_assignments(path):
    """{name: value node} for every module-level assignment in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
                elif isinstance(target, ast.Tuple):
                    for el in target.elts:
                        if isinstance(el, ast.Name):
                            out[el.id] = node.value
    return out


@pytest.mark.parametrize("path", DRIVERS, ids=lambda p: p.name)
def test_no_driver_redefines_an_owned_constant_as_a_literal(path):
    """Every owned constant must come from cb_settings, not from a local literal."""
    offenders = []
    for name, value in _module_level_assignments(path).items():
        if name not in OWNED:
            continue
        source = ast.dump(value)
        if "cb_settings" not in source:
            offenders.append(f"{name} = {ast.unparse(value)}")
    assert not offenders, (
        f"{path.name} defines analysis settings locally: {offenders}. "
        f"Import them from ImageLynx.cb_settings instead - a second copy is how open "
        f"items 1, 2, 8 and 10 came about."
    )


def test_pressure_pair_is_the_one_the_published_numbers_used():
    """60/20 mmHg, arteriolar to venular. The pipeline config reads it too (open item 10)."""
    assert cb_settings.INLET_PRESSURE_MMHG == 60.0
    assert cb_settings.OUTLET_PRESSURE_MMHG == 20.0


def test_frozen_threshold_lies_on_the_sweep_grid():
    """The frozen value is a grid point, because it is a snapped median of six choices."""
    assert cb_settings.FROZEN_THRESHOLD in cb_settings.THRESHOLD_GRID


def test_hysteresis_band_matches_what_the_pipeline_auto_raises_to():
    """--stage run passes low only; the pipeline raises high to low + 0.05."""
    assert cb_settings.HYSTERESIS_LOW == cb_settings.FROZEN_THRESHOLD
    assert cb_settings.HYSTERESIS_HIGH == pytest.approx(
        cb_settings.FROZEN_THRESHOLD + cb_settings.HYSTERESIS_HIGH_OFFSET)
    assert cb_settings.HYSTERESIS_HIGH > cb_settings.HYSTERESIS_LOW


def test_roi_volume_is_the_figure_the_document_quotes():
    """160^3 at the processing voxel is 0.0266 mm^3 - section 1.1 and section 2.1."""
    assert cb_settings.ROI_MM3 == pytest.approx(0.0266, abs=5e-5)


def test_metabolic_mean_is_held_across_the_contrast_sweep():
    """A contrast of c scales the split, not the total. Section 6.5."""
    for contrast in cb_settings.METABOLIC_CONTRASTS:
        f_bar = 0.235          # a representative TH volume fraction, section 13.7
        stroma = cb_settings.BASE_M_MAX / (1.0 + f_bar * (contrast - 1.0))
        mean = f_bar * stroma * contrast + (1.0 - f_bar) * stroma
        assert mean == pytest.approx(cb_settings.BASE_M_MAX, rel=1e-9)


def test_pipeline_and_h1_share_the_hysteresis_band():
    """Open item 1: the pipeline config said 0.65/0.75 while every H1 run used 0.90/0.95."""
    from carotid_image_to_model import PreprocessingConfig

    config = PreprocessingConfig()
    assert config.hysteresis_threshold_low == cb_settings.HYSTERESIS_LOW == 0.90
    assert config.hysteresis_threshold_high == cb_settings.HYSTERESIS_HIGH
    assert config.hysteresis_threshold_high == pytest.approx(0.95)


def test_pipeline_and_h2_share_the_perfusion_stop_settings():
    """Open item 6: the pipeline config said 1e-4 while Tier 1 hard-coded 1e-5 for H2.
    Open item 32: the cap went 50 -> 200 in both; the pipeline's Tier 3 needs up to 80 passes."""
    from carotid_image_to_model import PerfusionConfig

    config, settings = PerfusionConfig(), cb_settings.PerfusionSettings()
    assert config.picard_tolerance == settings.picard_tolerance == 1e-5
    assert config.picard_max_iterations == settings.picard_max_iterations == 200


@pytest.mark.parametrize("name", ["config_WKY_normotensive.yaml", "config_SHR_hypertensive.yaml"])
def test_the_example_yamls_carry_the_iteration_cap(name):
    """Open item 32: a YAML still at 50 would override the config and stop Tier 3 short
    (WKY-C needs 80 passes).
    Searched at any depth: the WKY YAML holds its perfusion block under PipelineConfig."""
    yaml = pytest.importorskip("yaml")
    from carotid_image_to_model import PerfusionConfig

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for child in node.values():
                found = find(child, key)
                if found is not None:
                    return found
        return None

    loaded = yaml.safe_load((REPO / "examples" / name).read_text(encoding="utf-8"))
    assert find(loaded, "picard_max_iterations") == PerfusionConfig().picard_max_iterations


#: 1 mmHg = 133.322387415 Pa = 1.33322387415e5 mPa, the unit the pipeline config uses.
_MPA_PER_MMHG = 133.322387415e3


def test_pipeline_and_h2_share_the_pressure_boundaries():
    """Open item 10: the pipeline config said 100/2 mmHg while the H2 drivers ran at 60/20."""
    from carotid_image_to_model import HaemodynamicsConfig

    config = HaemodynamicsConfig()
    assert config.input_p_bc / _MPA_PER_MMHG == pytest.approx(cb_settings.INLET_PRESSURE_MMHG)
    assert config.output_p_bc / _MPA_PER_MMHG == pytest.approx(cb_settings.OUTLET_PRESSURE_MMHG)
    assert (cb_settings.INLET_PRESSURE_MMHG, cb_settings.OUTLET_PRESSURE_MMHG) == (60.0, 20.0)


@pytest.mark.parametrize("name", ["config_WKY_normotensive.yaml", "config_SHR_hypertensive.yaml"])
def test_the_example_yamls_carry_the_pressure_boundaries(name):
    """Both YAMLs restate the pressures in mPa; a stale copy would override the config."""
    yaml = pytest.importorskip("yaml")
    from carotid_image_to_model import HaemodynamicsConfig

    config = HaemodynamicsConfig()
    loaded = yaml.safe_load((REPO / "examples" / name).read_text(encoding="utf-8"))
    hemo = loaded["HaemodynamicsConfig"]
    assert float(hemo["input_p_bc"]) == pytest.approx(config.input_p_bc, rel=1e-6)
    assert float(hemo["output_p_bc"]) == pytest.approx(config.output_p_bc, rel=1e-6)


def test_pipeline_and_h2_share_the_boundary_rule():
    """Open item 2: the pipeline used the band rule on axis 0; H2 the face rule on axis 1."""
    from carotid_image_to_model import GraphConfig

    config = GraphConfig()
    assert config.boundary_axis == cb_settings.BOUNDARY_AXIS == 1
    assert config.face_tolerance_voxels == cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS == 1.0


def test_pipeline_and_h2_share_the_metabolic_rate():
    """Open item 8: the pipeline config said 0.005 while the H2 drivers ran at 0.05."""
    from carotid_image_to_model import PerfusionConfig

    config, settings = PerfusionConfig(), cb_settings.PerfusionSettings()
    assert config.M_max == settings.M_max == cb_settings.BASE_M_MAX == 0.05


def test_dead_arterial_concentration_is_gone_everywhere():
    """Open item 5: ``C_arterial`` was declared three times and read nowhere.

    Leaving it in a config invited someone to set it and expect an effect. One analytical
    test did exactly that, setting it to the arterial PO2 it meant to test, and passed only
    because the value matched the solver's own hidden default.
    """
    assert not hasattr(cb_settings.PerfusionSettings(), "C_arterial")
    offenders = [
        str(path.relative_to(REPO))
        for folder in ("src", "examples", "tests")
        for path in (REPO / folder).rglob("*.py")
        if path.name != Path(__file__).name
        and "C_arterial" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize("name", ["cb_h2_hypoxic_fraction.py", "cb_h2_vtk.py"])
def test_h2_drivers_take_their_perfusion_config_from_settings(name):
    """Both H2 transport drivers use the one settings class, not a local copy of it."""
    value = _module_level_assignments(REPO / "examples" / name)["PerfConfig"]
    assert ast.unparse(value) == "cb_settings.PerfusionSettings"
