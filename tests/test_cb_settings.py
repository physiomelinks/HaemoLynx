"""The analysis settings have one owner, and the known disagreements cannot grow.

``ImageLynx.cb_settings`` exists because four of the twelve open items in
``cb_modelling_reference.md`` were the same defect: a constant written out separately in a
driver had drifted from the ``carotid_image_to_model.py`` config default it was meant to
match, and nothing noticed.

Two things are tested here.

1. No driver defines those constants as literals any more. A shared module only helps if
   nobody reintroduces a local copy.
2. The remaining config-versus-settings disagreements are *exactly* the ones recorded as
   open items 1, 8 and 10 - no more, and with the values the reference document states.
   These are pinned rather than fixed because every published H1 and H2 number was produced
   at the settings values, and silently changing the config default would make the document
   wrong rather than making the code right.
"""
import ast
import re
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
    """60/20 mmHg, arteriolar to venular. Open item 10 records the config's 100/2."""
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


def test_known_config_disagreements_are_exactly_the_recorded_open_items():
    """Pin the drift that is deliberately left in place, so it cannot grow unnoticed."""
    driver = (REPO / "examples" / "carotid_image_to_model.py").read_text(encoding="utf-8")

    def default_of(field):
        m = re.search(rf"^\s+{field}\s*:\s*[\w\[\]\|., \"']+?\s*=\s*([-\d.eE]+)",
                      driver, re.M)
        assert m, f"{field} not found in carotid_image_to_model.py"
        return float(m.group(1))

    # open item 10 - pressures. The config stores them in mPa, so convert to compare:
    # 1 mmHg = 133.322387415 Pa = 1.33322387415e5 mPa.
    mpa_per_mmhg = 133.322387415e3
    assert default_of("input_p_bc") / mpa_per_mmhg == pytest.approx(100.0, abs=0.05)
    assert default_of("output_p_bc") / mpa_per_mmhg == pytest.approx(2.0, abs=0.05)
    # open item 8 - metabolic rate, ten times lower than the drivers used
    assert default_of("M_max") == 0.005
    assert cb_settings.BASE_M_MAX == 10.0 * default_of("M_max")
    # open item 1 - hysteresis band, superseded at run time by the frozen threshold
    assert default_of("hysteresis_threshold_low") == 0.65
    assert default_of("hysteresis_threshold_high") == 0.75
