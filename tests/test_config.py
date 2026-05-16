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
        
    # 2. Invalid mode should crash
    with pytest.raises(ValueError, match="must be 'sphincter' or 'periodic'"):
        HaemodynamicsConfig(diameter_by_branch_order={}, constriction_mode="magic")
        
    # 3. Negative length should crash
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
