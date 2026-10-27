"""The one value-comparison rule that batch_compare, reference_run and the VTK comparer share."""
import math

import numpy as np
import pytest

from ImageLynx.value_compare import (close, float_pairs, gap_text, max_scaled_atol, values_equal,
                                     within)


def test_a_float_passes_inside_rtol_times_the_reference_plus_atol_and_fails_outside():
    # The band for a reference of 10 at rtol 1e-3 and atol 0.02 is 0.01 + 0.02 = 0.03.
    assert close(10.0, 10.029, rtol=1e-3, atol=0.02)
    assert not close(10.0, 10.031, rtol=1e-3, atol=0.02)
    assert close(10.0, 9.971, rtol=1e-3, atol=0.02)


def test_rtol_scales_by_the_reference_side_not_the_new_one():
    # The gap is 1 both ways. At rtol 0.095 the band is 1.045 for a reference of 11 and 0.95
    # for a reference of 10, so only the first passes.
    assert close(11.0, 10.0, rtol=0.095)
    assert not close(10.0, 11.0, rtol=0.095)


def test_with_no_tolerance_only_equal_values_pass():
    assert close(1.0, 1.0)
    assert not close(1.0, 1.0 + 1e-15)
    assert close(0.0, -0.0)


def test_nan_equals_nan_and_an_infinity_only_itself():
    assert close(math.nan, math.nan)
    assert not close(math.nan, 1.0) and not close(1.0, math.nan)
    assert close(math.inf, math.inf, rtol=1.0)
    assert not close(math.inf, -math.inf, rtol=1.0)
    assert not close(math.inf, 1e308, rtol=1.0, atol=1e308)


def test_within_works_element_by_element():
    ref = np.array([1.0, 2.0, math.nan, math.inf])
    new = np.array([1.0, 2.5, math.nan, math.inf])
    assert within(ref, new, rtol=0.1).tolist() == [True, False, True, True]


def test_the_absolute_floor_is_rtol_times_the_largest_finite_magnitude():
    assert max_scaled_atol([1.0, -40.0, math.nan, math.inf], 1e-6) == pytest.approx(4e-5)
    assert max_scaled_atol([math.nan], 1e-6) == 0.0
    assert max_scaled_atol([], 1e-6) == 0.0


def test_values_are_equal_only_with_the_same_types_and_structure():
    assert values_equal({"a": [1, (2.0, "x")]}, {"a": [1, (2.0, "x")]})
    assert not values_equal(1, 1.0)
    assert not values_equal(np.float64(1.0), 1.0)
    assert not values_equal([1.0, 2.0], (1.0, 2.0))
    assert not values_equal({"a": 1}, {"a": 1, "b": 2})
    assert not values_equal([1.0], [1.0, 2.0])
    assert not values_equal("x", "y")


def test_floats_inside_a_structure_are_compared_under_the_rule():
    assert values_equal({"pos": np.array([1.0, math.nan])},
                        {"pos": np.array([1.0 + 1e-10, math.nan])}, rtol=1e-9)
    assert not values_equal({"pos": np.array([1.0])}, {"pos": np.array([1.1])}, rtol=1e-9)
    assert values_equal([np.float32(2.0)], [np.float32(2.0)])


def test_arrays_must_share_shape_and_dtype_and_non_float_arrays_are_exact():
    assert not values_equal(np.zeros(3), np.zeros(4))
    assert not values_equal(np.zeros(3), np.zeros(3, dtype=np.float32))
    assert not values_equal(np.array([1, 2]), np.array([1, 3]), rtol=1.0)
    assert values_equal(np.array([1, 2]), np.array([1, 2]))


def test_float_pairs_lists_the_float_leaves_or_none_when_anything_else_differs():
    assert float_pairs({"a": [1, 2.0], "b": "x"}, {"a": [1, 3.0], "b": "x"}) == [(2.0, 3.0)]
    assert float_pairs({"a": [1, 2.0]}, {"a": [2, 2.0]}) is None
    assert float_pairs(np.array([1.0, 2.0]), np.array([1.0, 5.0])) == [(1.0, 1.0), (2.0, 5.0)]


def test_the_gap_text_gives_the_largest_absolute_and_relative_gap_of_finite_pairs():
    assert gap_text([10.0, 20.0, math.nan], [10.5, 20.2, 1.0]) == \
        ", max abs 0.5, max rel 0.05"
    assert gap_text([0.0], [1e-9]) == ", max abs 1e-09, max rel inf"
    assert gap_text([math.nan], [1.0]) == ""
