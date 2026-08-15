"""Tests for default diameter config values used by examples/pipeline."""

DIAMETER_BY_BRANCH_ORDER = {
    "BO1": 6.2,
    "BO2": 4.0,
    "BO3": 5.0,
}

DIAMETER_BY_BRANCH_ORDER_ENHANCED = {
    "BO1": {"d1": 6.2, "d2": 6.2},
    "BO2": {"d1": 4.0, "d2": 3.2},
    "BO3": {"d1": 5.0, "d2": 4.0},
}


def test_diameter_config_keys():
    assert "BO1" in DIAMETER_BY_BRANCH_ORDER
    assert DIAMETER_BY_BRANCH_ORDER["BO1"] == 6.2


def test_enhanced_config_structure():
    assert "BO1" in DIAMETER_BY_BRANCH_ORDER_ENHANCED
    d = DIAMETER_BY_BRANCH_ORDER_ENHANCED["BO1"]
    assert "d1" in d
    assert "d2" in d

def test_haemodynamics_config_bounds_safety():
    """Verify that extreme dataclass inputs are safely clamped or rejected."""
    import pytest
    import sys
    from pathlib import Path
    
    # We need to import HaemodynamicsConfig from the examples script.
    # We add examples to path dynamically for the test if needed.
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import HaemodynamicsConfig
    
    # 1. Reverse Pressures should crash
    with pytest.raises(ValueError, match="must be strictly greater"):
        HaemodynamicsConfig(diameter_by_branch_order={}, input_p_bc=10.0, output_p_bc=100.0)
        
    # 2. Invalid radius mode should crash
    with pytest.raises(ValueError, match="radius_assignment_mode must be 'fwhm_radius', 'edt_radius', or 'constant_radius'"):
        HaemodynamicsConfig(diameter_by_branch_order={}, radius_assignment_mode="invalid_mode")

    # 3. Invalid constriction mode should crash
    with pytest.raises(ValueError, match="must be 'sphincter' or 'periodic'"):
        HaemodynamicsConfig(diameter_by_branch_order={}, constriction_mode="magic")
        
    # 4. Negative length should crash
    with pytest.raises(ValueError, match="cannot be negative"):
        HaemodynamicsConfig(diameter_by_branch_order={}, sphincter_length_um=-10.0)
        
    # 4. Extreme 0.0 diameter pinches should clamp to 0.01
    config = HaemodynamicsConfig(
        diameter_by_branch_order={}, 
        intimal_cushion_constriction_ratio=0.0,
        pre_capillary_constriction_ratio=-5.0
    )
    assert config.intimal_cushion_constriction_ratio == 0.01
    assert config.pre_capillary_constriction_ratio == 0.01


def _load_carotid_config():
    import sys
    from pathlib import Path

    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import HaemodynamicsConfig

    return HaemodynamicsConfig


def test_variable_constriction_is_disabled_and_cannot_be_re_enabled_silently():
    """Constriction is off by default, and turning it on fails loudly.

    With constriction on, the ratio is read from the synthetic branch-order dict and
    multiplied onto whatever diameter was measured, including a real EDT measurement
    ([poiseuille.py:333](src/ImageLynx/haemodynamics/poiseuille.py:333)). The sites come
    from a hard-coded topological rule rather than from the imaging. Resistance goes as
    the inverse fourth power of diameter, so the 0.5 ratio at the capillary anchor is a
    16x local resistance error applied to a measured vessel.

    The capability stays in the tree for the resistance-network pipelines that own it.
    The carotid config must not reach it.
    """
    import pytest

    HaemodynamicsConfig = _load_carotid_config()

    assert HaemodynamicsConfig(diameter_by_branch_order={}).constrict_at_pericytes is False

    with pytest.raises(ValueError, match="constrict_at_pericytes is disabled"):
        HaemodynamicsConfig(diameter_by_branch_order={}, constrict_at_pericytes=True)


def test_branch_order_fallback_carries_no_constriction():
    """Every branch-order entry the fallback generates has d2 equal to d1.

    d2 exists only to describe a constricted calibre. With constriction disabled the two
    must agree, so that a config written now cannot resurrect a fabricated ratio if the
    capability is ever re-enabled.
    """
    import sys
    from pathlib import Path

    import networkx as nx
    import numpy as np

    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import (
        GraphConfig,
        HaemodynamicsConfig,
        _setup_boundary_conditions_and_haemodynamics,
    )

    G = nx.MultiGraph()
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 20.0, 20.0]))
    G.add_node(3, pos=np.array([40.0, 40.0, 40.0]))
    G.add_edge(1, 2, key=0, length=20.0, voxels=[[0, 0, 0], [20, 20, 20]])
    G.add_edge(2, 3, key=0, length=20.0, voxels=[[20, 20, 20], [40, 40, 40]])

    hemo_config = HaemodynamicsConfig(
        diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}},
        radius_assignment_mode="fwhm_radius",
    )
    _setup_boundary_conditions_and_haemodynamics(
        G,
        np.ones((50, 50, 50)),
        hemo_config,
        GraphConfig(edge_percent=25.0, end_percent=25.0),
        "mock_path",
        "numpy",
    )

    generated = {
        label: value
        for label, value in hemo_config.diameter_by_branch_order.items()
        if isinstance(value, dict)
    }
    assert generated, "the fallback populated no branch orders, so nothing was verified"
    for label, value in generated.items():
        assert value["d2"] == value["d1"], f"{label} carries a constriction: {value}"
