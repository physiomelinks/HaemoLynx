"""The human kit's arithmetic: the SUS score, hand-measured diameters against
a run's, and the visual grid's numbering. No mark is made here."""
from __future__ import annotations

import csv

import networkx as nx
import numpy as np
import pytest

from release_review import human_kit, visual_grid


def test_the_sus_score():
    assert human_kit.sus_score([5, 1] * 5) == 100.0
    assert human_kit.sus_score([1, 5] * 5) == 0.0
    assert human_kit.sus_score([3] * 10) == 50.0
    # Brooke's worked form: odd items minus 1, even items from 5, times 2.5.
    assert human_kit.sus_score([4, 2, 4, 1, 4, 2, 5, 1, 4, 2]) == 2.5 * (3 + 3 + 3 + 4 + 3 + 3 + 4 + 4 + 3 + 3)
    with pytest.raises(ValueError):
        human_kit.sus_score([3] * 9)
    with pytest.raises(ValueError):
        human_kit.sus_score([0] + [3] * 9)


def _shapes_csv(path, lines):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "shape-type", "vertex-index", "axis-0", "axis-1", "axis-2"])
        for index, (start, end) in enumerate(lines):
            writer.writerow([index, "line", 0, *start])
            writer.writerow([index, "line", 1, *end])


def _graph():
    """Two straight vessels along x, at y = 10 and y = 40 (pixels), diameters
    5 and 10 um by the model, 5.5 by the mask's method on the first."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.array([5.0, 10.0, 0.0]))
    graph.add_node(1, pos=np.array([5.0, 10.0, 30.0]))
    graph.add_node(2, pos=np.array([5.0, 40.0, 0.0]))
    graph.add_node(3, pos=np.array([5.0, 40.0, 30.0]))
    graph.add_edge(0, 1, voxels=[[5, 10, x] for x in range(31)], diameter_um=5.0, edt_diameter_um=5.5,
                   edt_diameter_samples_um=[5.0, 6.0])
    graph.add_edge(2, 3, voxels=[[5, 40, x] for x in range(31)], diameter_um=10.0, edt_diameter_um=float("nan"))
    return graph


def test_hand_lines_are_matched_to_the_nearest_vessel_and_compared(tmp_path):
    path = tmp_path / "lines.csv"
    _shapes_csv(path, [
        ((5, 7.5, 15), (5, 12.5, 15)),    # 5 um across the first vessel
        ((5, 35.0, 15), (5, 45.0, 15)),   # 10 um across the second
        ((5, 70.0, 15), (5, 72.0, 15)),   # nowhere near a vessel
    ])
    lines = human_kit.read_shapes_csv(path)
    assert [line.index for line in lines] == [0, 1, 2]
    assert human_kit.line_length_um(lines[0], (1.0, 1.0, 1.0)) == pytest.approx(5.0)
    found = human_kit.compare(lines, _graph(), (1.0, 1.0, 1.0))
    assert [each["line"] for each in found["unmatched"]] == [2]
    model = found["methods"]["diameter_um"]
    assert model["vessels"] == 2 and model["within"] == 1.0 and model["bias"] == pytest.approx(0.0)
    mask = found["methods"]["edt_diameter_um"]  # the NaN on the second vessel is not a width
    assert mask["vessels"] == 1 and mask["bias"] == pytest.approx(0.1) and not mask["enough"]


def test_a_shape_that_is_not_a_line_is_refused(tmp_path):
    path = tmp_path / "shapes.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "shape-type", "vertex-index", "axis-0", "axis-1", "axis-2"])
        for vertex in range(4):
            writer.writerow([0, "rectangle", vertex, 0, vertex, vertex])
    with pytest.raises(ValueError, match="not a line"):
        human_kit.read_shapes_csv(path)


def test_the_visual_grid_is_shuffled_numbered_and_keyed(tmp_path):
    from matplotlib import pyplot

    folder = tmp_path / "plots"
    folder.mkdir()
    for index in range(40):
        figure = pyplot.figure(figsize=(1, 1))
        figure.savefig(folder / f"figure_{index:02d}.png")
        pyplot.close(figure)
    pictures = visual_grid.gather([folder])
    assert len(pictures) == 40
    items = visual_grid.numbered(pictures, seed=1)
    assert [number for number, _ in items] == list(range(1, 31))
    assert [path.name for _, path in items] != sorted(path.name for _, path in items)
    assert items == visual_grid.numbered(pictures, seed=1)
    page = visual_grid.write(tmp_path / "grid", items)
    text = page.read_text(encoding="utf-8")
    assert text.count("<figure>") == 30 and "GOOD" in text
    assert "figure_" not in text  # the names stay in the key
    with (tmp_path / "grid" / "marks.csv").open() as handle:
        marks = list(csv.reader(handle))
    assert marks[1] == ["1", "", ""] and len(marks) == 31
