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
