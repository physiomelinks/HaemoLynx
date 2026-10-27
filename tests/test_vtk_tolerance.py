"""driver_snapshots' VTK comparer given a tolerance rule, on tiny VTK files.

Without a rule it stays exact (``tests/test_driver_snapshots.py``). With one, each float array
is compared under its own rtol and ``atol = rtol * max|reference array|``; integer and text
arrays, the cells and an image grid stay exact.
"""
import numpy as np
import pytest

from ImageLynx.batch_compare import MAX_LISTED
from ImageLynx.driver_snapshots import POINTS, vtk_problems

pv = pytest.importorskip("pyvista")

RTOL = {POINTS: 1e-9, "flow": 1e-6}


def _rule(kind, name):
    return RTOL[POINTS] if kind == POINTS else RTOL.get(name, 0.0)


def _polydata(flow=(10.0, 20.0), points=None, tag=(1, 2, 3)):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]) if points is None else points
    mesh = pv.PolyData(points, lines=np.array([2, 0, 1, 2, 1, 2]))
    mesh.cell_data["flow"] = np.array(flow, dtype=np.float64)
    mesh.point_data["tag"] = np.array(tag, dtype=np.int32)
    mesh.field_data["status"] = np.array(["converged"])
    return mesh


def _pair(tmp_path, old, new, suffix=".vtp"):
    paths = tmp_path / f"old{suffix}", tmp_path / f"new{suffix}"
    old.save(paths[0])
    new.save(paths[1])
    return paths


def test_a_float_array_passes_inside_its_band_and_fails_outside(tmp_path):
    # flow 10 against a largest |flow| of 20: band 1e-6 * 10 + 1e-6 * 20 = 3e-5.
    inside = _pair(tmp_path, _polydata(), _polydata(flow=(10.0 + 2.9e-5, 20.0)))
    assert vtk_problems(*inside, rtol_for=_rule) == []
    outside = _pair(tmp_path, _polydata(), _polydata(flow=(10.0 + 3.1e-5, 20.0)))
    assert vtk_problems(*outside, rtol_for=_rule) == [
        "cell array 'flow': 1 of 2 values differ, max abs 3.1e-05, max rel 3.1e-06"]


def test_without_a_rule_the_same_change_is_a_difference(tmp_path):
    inside = _pair(tmp_path, _polydata(), _polydata(flow=(10.0 + 2.9e-5, 20.0)))
    assert vtk_problems(*inside) == ["cell array 'flow' values differ"]


def test_points_follow_their_rule_with_the_floor_scaled_by_the_largest_coordinate(tmp_path):
    # A coordinate of 0 may move by 1e-9 * 2 (the largest coordinate).
    moved = np.array([[0.0, 0, 1.9e-9], [1, 0, 0], [2, 0, 0]])
    assert vtk_problems(*_pair(tmp_path, _polydata(), _polydata(points=moved)),
                        rtol_for=_rule) == []
    moved[0, 2] = 2.1e-9
    assert vtk_problems(*_pair(tmp_path, _polydata(), _polydata(points=moved)),
                        rtol_for=_rule) == ["points: 1 of 9 values differ, max abs 2.1e-09, "
                                            "max rel inf"]


def test_integer_and_text_arrays_stay_exact_under_any_rule(tmp_path):
    changed = _polydata(tag=(1, 2, 4))
    changed.field_data["status"] = np.array(["max_iterations"])
    problems = vtk_problems(*_pair(tmp_path, _polydata(), changed), rtol_for=lambda k, n: 1.0)
    assert problems == ["point array 'tag': 1 of 3 values differ",
                        "field array 'status': 1 of 1 values differ"]


def test_an_array_on_one_side_only_is_a_difference(tmp_path):
    extra = _polydata()
    extra.cell_data["pressure"] = np.zeros(2)
    assert vtk_problems(*_pair(tmp_path, _polydata(), extra), rtol_for=_rule,
                        sides=("REFERENCE", "NEW")) == ["cell array 'pressure' only in NEW"]
    assert vtk_problems(*_pair(tmp_path, extra, _polydata()), rtol_for=_rule,
                        sides=("REFERENCE", "NEW")) == ["cell array 'pressure' only in REFERENCE"]


def test_the_data_kinds_compared_can_be_chosen(tmp_path):
    changed = _polydata()
    changed.field_data["status"] = np.array(["max_iterations"])
    assert vtk_problems(*_pair(tmp_path, _polydata(), changed), rtol_for=_rule,
                        kinds=("point_data", "cell_data")) == []


def test_an_image_grid_stays_exact_and_its_arrays_follow_the_rule(tmp_path):
    def grid(spacing=1.0, po2=(50.0, 100.0)):
        image = pv.ImageData(dimensions=(3, 2, 2), spacing=(spacing, 1.0, 1.0))
        image.cell_data["flow"] = np.array(po2)
        return image

    assert vtk_problems(*_pair(tmp_path, grid(), grid(po2=(50.0 + 1e-4, 100.0)), ".vti"),
                        rtol_for=_rule) == []
    assert vtk_problems(*_pair(tmp_path, grid(), grid(spacing=1.0 + 1e-12), ".vti"),
                        rtol_for=lambda k, n: 1.0) == ["grid spacing differs"]


def test_a_long_list_of_differences_ends_by_counting_the_rest(tmp_path):
    old, new = _polydata(), _polydata()
    for i in range(MAX_LISTED + 2):
        old.cell_data[f"a{i:02d}"] = np.array([1.0, 2.0])
        new.cell_data[f"a{i:02d}"] = np.array([1.0, 3.0])
    problems = vtk_problems(*_pair(tmp_path, old, new))
    assert len(problems) == MAX_LISTED + 1
    assert problems[-1] == "and 2 more"
