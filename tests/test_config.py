"""Tests for config."""
from ImageLynx.config import (
    DIAMETER_BY_BRANCH_ORDER,
    DIAMETER_BY_BRANCH_ORDER_ENHANCED,
)


def test_diameter_config_keys():
    assert "BO1" in DIAMETER_BY_BRANCH_ORDER
    assert DIAMETER_BY_BRANCH_ORDER["BO1"] == 6.2


def test_enhanced_config_structure():
    assert "BO1" in DIAMETER_BY_BRANCH_ORDER_ENHANCED
    d = DIAMETER_BY_BRANCH_ORDER_ENHANCED["BO1"]
    assert "d1" in d
    assert "d2" in d
