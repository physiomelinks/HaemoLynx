"""Every setting that carries a physical quantity has to say which one.

A number in a config file is useless without its unit, and the ones that hurt
are the plausible-either-way lengths: this pipeline measures some things in
voxels (anything that indexes the array) and some in microns (anything derived
from node ``pos`` or edge ``length``), and 10 is a reasonable value for either.
Guessing wrong is silent -- the run completes and prunes the wrong vessels.

So each setting below is pinned to the unit its consuming code actually works
in, traced from the value's use site rather than its name, and a convention
test stops the next length-like setting being added without one.
"""
from __future__ import annotations

import pytest

from haemolynx.pipeline import default_schema


SCHEMA = default_schema()

#: The spellings a unit may use. One per quantity -- `um` and `microns` for the
#: same thing reads as two different units to anyone scanning a config file.
ALLOWED_UNITS = {"um", "voxels", "percent", "fraction", "Pa"}

#: Settings whose unit is settled by what the consuming code compares them
#: against. `um` wherever the value meets node `pos` or edge `length`, which
#: are physical microns; `voxels` wherever it meets an array index.
EXPECTED_UNITS = {
    # Compared against edge `length`, which is microns.
    "min_stub_length": "um",
    # Compared against distances between node `pos`, which is microns.
    "graph_reconnect_threshold": "um",
    "final_orphan_reconnect_threshold": "um",
    "cluster_collapse_distance": "um",
    # Measured against a skeleton KD-tree scaled by voxel_size_zyx: microns.
    "centreline_max_deviation": "um",
    # Compared against node `pos` directly -- microns, despite reading like
    # voxel indices, which is the mistake these invite.
    "inlet_node_coordinates": "um",
    "outlet_node_coordinates": "um",
    "arteriole_boundary_node_coordinates": "um",
    "venule_boundary_node_coordinates": "um",
    "inlet_node_volumes": "um",
    "outlet_node_volumes": "um",
    "arteriole_boundary_node_volumes": "um",
    "venule_boundary_node_volumes": "um",
    # Applied to the binary volume, so they index the array: voxels.
    "skeleton_closing_radius": "voxels",
    "skeleton_bridge_gap_size": "voxels",
    "skeleton_min_branch_length": "voxels",
    "skeleton_max_bridge_distance": "voxels",
    # Pressures.
    "inlet_p_bc": "Pa",
    "outlet_p_bc": "Pa",
    # Walked along an edge's centreline in `get_diameter_at_position`, whose
    # position runs over edge `length`: microns.
    "constriction_length_um": "um",
    "constriction_spacing_um": "um",
    "constriction_spacing_min_um": "um",
    "constriction_spacing_max_um": "um",
    "constriction_spacing_step_um": "um",
    "constriction_length_min_um": "um",
    "constriction_length_max_um": "um",
    "constriction_length_step_um": "um",
}

#: Names that look like a measurement but hold a dimensionless number. Listed
#: so the convention test below can tell "no unit because none applies" from
#: "no unit because nobody wrote one".
DIMENSIONLESS = {
    # An array axis index (0=z, 1=y, 2=x), not a coordinate.
    "boundary_axis",
    # A connectivity rank in 1..3, not a distance.
    "skeleton_component_connectivity",
    # Counts and identifiers.
    "centreline_smoothing_iterations",
    "max_branch_order",
    "pericyte_constriction_seed",
    "inlet_nodes",
    "outlet_nodes",
    "arteriole_boundary_nodes",
    "venule_boundary_nodes",
    "custom_edges",
    # Diameter multipliers: 1.0 is unconstricted, 0.8 narrows by a fifth.
    "pericyte_comparison_baseline_value",
    "pericyte_comparison_constricted_value",
    "constriction_by_branch_order",
    "fwhm_min_total_extent_multiplier",
    "fwhm_same_edge_arc_window_multiplier",
    # Label sentinels written into the int32 branch-label volume.
    "fwhm_background_label",
    "fwhm_junction_label",
    # A goodness-of-fit score in 0..1.
    "fwhm_min_fit_r2",
    # Names a file, an axis order, a mode -- not a quantity.
    "image_axis_order",
    "voxel_size_policy",
}

#: Substrings that mark a name as describing a measurement.
MEASUREMENT_WORDS = (
    "length", "distance", "radius", "diameter", "deviation", "spacing",
    "extent", "offset", "threshold", "coordinates", "volumes", "pressure",
    "_um", "microns", "size", "percent", "fraction", "probability", "axis",
    "gap", "separation", "width",
)


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_UNITS.items()))
def test_the_settings_that_carry_a_length_declare_the_right_one(name, expected):
    """Pinned per setting, because the wrong one here fails silently."""
    assert name in SCHEMA, f"{name} is no longer a setting; update this test"
    assert SCHEMA[name].unit == expected, (
        f"{name} should be in {expected}: it is compared against "
        f"{'physical node positions or edge lengths' if expected == 'um' else 'array indices'}"
    )


def test_every_unit_uses_one_spelling():
    """`um` and `microns` for the same quantity read as two different units."""
    used = {setting.unit for setting in SCHEMA if setting.unit}
    unknown = used - ALLOWED_UNITS
    assert not unknown, (
        f"unrecognised unit spellings: {sorted(unknown)}. "
        f"Use one of {sorted(ALLOWED_UNITS)}, or add the new one deliberately"
    )


def test_no_setting_that_measures_something_is_missing_its_unit():
    """The guard: a new length-like setting must declare a unit or be listed
    as dimensionless, so neither can be forgotten by accident."""
    missing = [
        setting.name
        for setting in SCHEMA
        if not setting.unit
        # A switch has no unit however it is named: several read as
        # measurements ("use_fwhm_edge_diameters") but only turn a step on.
        and setting.kind not in ("bool", "choice", "path")
        and setting.name not in DIMENSIONLESS
        and any(word in setting.name for word in MEASUREMENT_WORDS)
    ]
    assert not missing, (
        f"these look like measurements but declare no unit: {sorted(missing)}. "
        "Trace what the value is compared against and give it that unit, or "
        "add it to DIMENSIONLESS with a note saying why none applies"
    )


def test_the_dimensionless_list_has_no_stale_entries():
    """A name listed as dimensionless that has since gained a unit, or gone."""
    for name in sorted(DIMENSIONLESS):
        assert name in SCHEMA, f"{name} is no longer a setting; drop it from DIMENSIONLESS"
        assert not SCHEMA[name].unit, (
            f"{name} now declares unit={SCHEMA[name].unit!r}; drop it from DIMENSIONLESS"
        )


def test_the_unit_reaches_the_generated_config(tmp_path):
    """A unit nobody can see is not documentation."""
    from haemolynx.pipeline import write_default_config

    path = tmp_path / "config.yaml"
    write_default_config(path)
    lines = path.read_text().splitlines()

    seen = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split(":", 1)[0]
        if key in EXPECTED_UNITS:
            seen.add(key)
            preceding = lines[index - 1].strip()
            assert preceding.startswith("#"), f"{key} has no comment above it"
            unit = EXPECTED_UNITS[key]
            assert f"[{unit};" in preceding or f"[{unit}]" in preceding, (
                f"{key}'s comment does not state its unit: {preceding}"
            )

    assert seen == set(EXPECTED_UNITS), (
        f"not every pinned setting was found in the config: "
        f"{sorted(set(EXPECTED_UNITS) - seen)}"
    )
