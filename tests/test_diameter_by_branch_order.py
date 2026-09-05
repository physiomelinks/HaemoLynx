"""Unit tests for build_diameter_by_branch_order's Large_Art/Large_Ven keys."""
from __future__ import annotations

from haemolynx.haemodynamics.poiseuille import build_diameter_by_branch_order


def test_all_diams_const_gives_every_large_vessel_key_the_flat_default():
    table = build_diameter_by_branch_order(
        all_diams_const=True, max_branch_order=3, default_diameter=9.0
    )
    for i in range(1, 4):
        assert table[f"Large_Art{i}"] == 9.0
        assert table[f"Large_Ven{i}"] == 9.0


def test_manual_override_applies_to_the_named_large_vessel_key():
    table = build_diameter_by_branch_order(
        all_diams_const=False,
        max_branch_order=3,
        default_diameter=4.0,
        manual_large_arteriole_diameter_by_branch_order={"Large_Art1": 40.0},
        manual_large_venule_diameter_by_branch_order={"Large_Ven1": 50.0},
    )
    assert table["Large_Art1"] == 40.0
    assert table["Large_Ven1"] == 50.0


def test_unlisted_large_vessel_order_falls_back_to_default_diameter_not_capillary_size():
    """The Large_Art/Large_Ven fallback is default_diameter directly, not the
    B01-derived default_small_vessel_diameter the small Art/Ven loop uses --
    a large vessel is by definition bigger than a capillary, so that
    fallback would be actively wrong here."""
    table = build_diameter_by_branch_order(
        all_diams_const=False,
        max_branch_order=3,
        default_diameter=4.0,
        manual_capillary_diameter_by_branch_order={"B01": 6.2},
        manual_large_arteriole_diameter_by_branch_order={"Large_Art1": 40.0},
    )
    assert table["Large_Art2"] == 4.0
    assert table["Large_Ven1"] == 4.0
    # Confirm the capillary-derived fallback the small Art/Ven loop uses is
    # indeed different here, so this test would catch reusing it by mistake.
    assert table["Art1"] == 6.2
    assert table["Large_Art2"] != table["Art1"]


def test_large_vessel_keys_present_up_to_max_branch_order():
    table = build_diameter_by_branch_order(
        all_diams_const=False, max_branch_order=5, default_diameter=4.0
    )
    for i in range(1, 6):
        assert f"Large_Art{i}" in table
        assert f"Large_Ven{i}" in table
    assert "Large_Art6" not in table
    assert "Large_Ven6" not in table
