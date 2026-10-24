"""A graph attribute becomes a number only when it is there and finite (package AB).

The H2 drivers used to read ``data.get("flow_abs", nan)`` and friends, so an edge the solver
never reached would have been written into the tables and the VTK as NaN (or as length 0)
without a word. ``edge_float`` is the graph-side counterpart of ``BatchRun.numeric_column``.
"""
import math

import pytest

from ImageLynx.graph.edge_attributes import edge_float


def test_a_present_finite_value_comes_back_as_a_float():
    assert edge_float({"flow_abs": 3}, "flow_abs", (0, 1, 0)) == 3.0


@pytest.mark.parametrize("data", [{}, {"hematocrit": None}],
                         ids=["missing", "none"])
def test_an_absent_value_raises_naming_the_edge_and_attribute(data):
    with pytest.raises(ValueError, match=r"edge \(4, 7, 1\) has no 'hematocrit'"):
        edge_float(data, "hematocrit", (4, 7, 1))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_value_raises_naming_the_edge(value):
    with pytest.raises(ValueError, match=r"edge \(4, 7, 1\) has a non-finite 'length'"):
        edge_float({"length": value}, "length", (4, 7, 1))


@pytest.mark.parametrize("value", ["wide", True, [3.0]])
def test_a_value_that_is_not_a_number_raises_saying_so(value):
    # float(True) is 1.0, so a flag stored in the wrong attribute would pass as a length.
    with pytest.raises(ValueError, match=r"edge \(4, 7, 1\) has a 'length' that is not a number"):
        edge_float({"length": value}, "length", (4, 7, 1))


def test_zero_is_a_value_not_a_gap():
    # A stagnant edge has flow 0; only absence and non-finite values are refused.
    assert edge_float({"flow_abs": 0.0}, "flow_abs", (0, 1, 0)) == 0.0
