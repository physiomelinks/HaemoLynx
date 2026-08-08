"""What drawing a real run costs, measured rather than assumed.

Vessels are drawn as a Vectors layer rather than as Shapes paths, and the
reason is a number: at the size of a real nerve run a Shapes layer took about
2.4 s to add and render against about 0.2 s for Vectors, and that cost lands as
a frozen viewer in the middle of a run, once for every stage that creates
geometry. At whole-brain scale the same measurement was 39 s against 3 s.

The assertion is a ratio, not a stopwatch: CI renders through llvmpipe on a
shared machine, so an absolute threshold would flake. The ceiling that follows
it is an alarm for something being wrong by an order of magnitude, not a target.
Run with `-s` to see both numbers.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest

napari = pytest.importorskip("napari")

from haemolynx.gui.results import edge_polylines, polylines_to_vectors  # noqa: E402

pytestmark = [pytest.mark.gui, pytest.mark.slow]

#: A real nerve run: 774 nodes, 1,040 vessels, ~33,600 polyline points.
VESSELS = 1_040
POINTS_PER_VESSEL = 32


def a_network_at_scale() -> nx.MultiGraph:
    """A graph the size of a real run, laid out so nothing overlaps."""
    rng = np.random.default_rng(20240917)
    graph = nx.MultiGraph()
    for index in range(VESSELS + 1):
        graph.add_node(index, pos=rng.uniform(0, 500, size=3))
    for index in range(VESSELS):
        start = graph.nodes[index]["pos"]
        end = graph.nodes[index + 1]["pos"]
        path = np.linspace(start, end, POINTS_PER_VESSEL)
        graph.add_edge(index, index + 1, key=0, voxels=path.tolist(),
                       length=float(np.linalg.norm(end - start)), segment_id=index)
    return graph


def _timed_add(viewer, add, *args, **kwargs) -> float:
    """How long it takes to add a layer *and* draw it.

    The add alone is misleading: for paths, building the napari model is cheap
    and the cost is in vispy, which only happens when something asks for a
    frame.
    """
    start = time.perf_counter()
    layer = add(*args, **kwargs)
    viewer.window.screenshot(canvas_only=True, flash=False)
    elapsed = time.perf_counter() - start
    viewer.layers.remove(layer)
    return elapsed


def test_vectors_are_much_cheaper_to_draw_than_paths(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    paths, _identity = edge_polylines(a_network_at_scale())
    vectors, _owner = polylines_to_vectors(paths)

    # The first Shapes layer in a process pays a one-time backend import of
    # several seconds; measuring without this warm-up reports nonsense.
    _timed_add(viewer, viewer.add_shapes, [paths[0]], shape_type="path")
    _timed_add(viewer, viewer.add_vectors, vectors[:10])

    shapes_seconds = _timed_add(viewer, viewer.add_shapes, paths, shape_type="path")
    vectors_seconds = _timed_add(viewer, viewer.add_vectors, vectors)

    with capsys.disabled():
        print(
            f"\n{len(paths)} vessels / {len(vectors)} segments: "
            f"shapes {shapes_seconds:.2f}s, vectors {vectors_seconds:.2f}s "
            f"({shapes_seconds / max(vectors_seconds, 1e-6):.1f}x)"
        )

    assert vectors_seconds < shapes_seconds / 3, (
        f"vectors {vectors_seconds:.2f}s vs shapes {shapes_seconds:.2f}s -- the "
        "renderer choice rests on this gap; if it has closed, reconsider it"
    )
    assert vectors_seconds < 10.0, (
        f"drawing a real run took {vectors_seconds:.2f}s, which a user feels as "
        "the viewer freezing mid-run"
    )
