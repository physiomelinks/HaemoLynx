"""How long a vessel is, checked against curves whose length we know.

A skeleton path steps from voxel to voxel, so a vessel running at an angle
comes back as a 45-degree zigzag that is longer than the vessel it traces.
Resistance is proportional to length, so that is a measurement error, not a
drawing one.

Two tests, and they fail in opposite directions on purpose. The first draws
twenty helices whose arc length is a number we can compute, skeletonises them,
and asks whether the pipeline gets that number back -- too long means the
staircase is still there, too short means the smoothing has eaten real
curvature. The second runs the real nerve fixture and checks both that its
total length has come down and that no point has wandered off the skeleton it
came from.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph as graph_tools
from haemolynx import preprocessing
from haemolynx.graph.smoothing import smooth_graph_centrelines

REPO_ROOT = Path(__file__).resolve().parents[1]
NERVE = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"

#: Volume the synthetic vessels are drawn into, in voxels.
SHAPE = (64, 160, 160)
VESSEL_COUNT = 20


# --- vessels whose length is known -------------------------------------------


def _helix(turns, radius, pitch, centre, phase):
    """A helix, which has curvature everywhere and a closed-form arc length."""

    def curve(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        angle = 2 * np.pi * turns * t + phase
        out = np.zeros((len(t), 3))
        out[:, 0] = pitch * turns * t
        out[:, 1] = radius * np.cos(angle)
        out[:, 2] = radius * np.sin(angle)
        return out + np.asarray(centre, dtype=float)

    return curve


def _sampled(curve, samples=20001):
    """Points along the curve and the arc length reached at each."""
    points = curve(np.linspace(0.0, 1.0, samples))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, np.concatenate(([0.0], np.cumsum(steps)))


def _true_length_between(points, cumulative, start, end):
    """Arc length between the curve points nearest *start* and *end*.

    Skeletonising a tube erodes its ends, so the recovered centreline is a
    little shorter than the curve drawn. That is not a length error, so the
    truth is taken between the ends the skeleton actually found.
    """
    first = int(np.argmin(np.linalg.norm(points - np.asarray(start), axis=1)))
    last = int(np.argmin(np.linalg.norm(points - np.asarray(end), axis=1)))
    return abs(float(cumulative[last] - cumulative[first]))


def _draw_tube(volume, curve, radius=2.0, samples=4000):
    centres = np.round(curve(np.linspace(0.0, 1.0, samples))).astype(int)
    reach = int(np.ceil(radius))
    offsets = np.array(
        [
            (dz, dy, dx)
            for dz in range(-reach, reach + 1)
            for dy in range(-reach, reach + 1)
            for dx in range(-reach, reach + 1)
            if dz * dz + dy * dy + dx * dx <= radius * radius
        ]
    )
    for offset in offsets:
        hit = centres + offset
        inside = np.all((hit >= 0) & (hit < np.asarray(volume.shape)), axis=1)
        hit = hit[inside]
        volume[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    return volume


@pytest.fixture(scope="module")
def known_vessels():
    """Twenty helices, the volume they are drawn in, and the graph recovered."""
    rng = np.random.default_rng(20240917)
    curves = []
    for index in range(VESSEL_COUNT):
        row, column = divmod(index, 5)
        curves.append(
            _helix(
                turns=float(rng.uniform(0.6, 1.4)),
                radius=float(rng.uniform(5.0, 9.0)),
                pitch=float(rng.uniform(34.0, 46.0)),
                centre=(8.0, 18.0 + row * 36.0, 18.0 + column * 30.0),
                phase=float(rng.uniform(0.0, 2 * np.pi)),
            )
        )

    volume = np.zeros(SHAPE, dtype=bool)
    for curve in curves:
        _draw_tube(volume, curve)

    skeleton = preprocessing.skeletonize_volume(volume)
    skeleton = preprocessing.preprocess_skeleton_for_graph(
        skeleton, min_branch_length=3, max_bridge_distance=2,
        component_connectivity=3, min_component_fraction=0.0,
        closing_radius=1, bridge_gap_size=1,
    )
    graph = graph_tools.build_graph_from_skeleton(
        skeleton, voxel_size=(1.0, 1.0, 1.0),
        min_stub_length=3.0, cluster_collapse_distance=5.0,
    )
    return curves, skeleton, graph


def _errors_against_truth(curves, graph):
    """Per-vessel (measured, true) lengths, matched by where each edge sits."""
    sampled = [_sampled(curve) for curve in curves]
    pairs = []
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        path = np.asarray(data["voxels"], dtype=float)
        middle = path[len(path) // 2]
        nearest = min(
            range(len(sampled)),
            key=lambda i: np.linalg.norm(sampled[i][0] - middle, axis=1).min(),
        )
        points, cumulative = sampled[nearest]
        pairs.append(
            (float(data["length"]), _true_length_between(points, cumulative, path[0], path[-1]))
        )
    return pairs


@pytest.mark.slow
def test_the_fixture_recovers_one_edge_per_vessel(known_vessels):
    """If the graph were wrong, a length agreement would mean nothing."""
    curves, _skeleton, graph = known_vessels
    assert graph.number_of_edges() == len(curves)


@pytest.mark.slow
def test_the_raw_skeleton_overstates_length_by_about_seven_percent(known_vessels):
    """The artefact this is all about, stated as a number.

    Not a target to hold, but the measurement that justifies smoothing at all:
    if this ever drops near zero, the skeletonisation has changed and the
    thresholds below deserve revisiting.
    """
    curves, _skeleton, graph = known_vessels
    pairs = _errors_against_truth(curves, graph)
    measured = sum(m for m, _t in pairs)
    truth = sum(t for _m, t in pairs)
    assert 0.04 < measured / truth - 1 < 0.12


@pytest.mark.slow
def test_smoothing_recovers_the_true_length(known_vessels):
    """The whole point: the number the haemodynamics uses should be right.

    Bounded on both sides. Over 2% and the staircase is still in there; under
    -2% and the smoothing is cutting corners that are really in the vessel.
    """
    curves, skeleton, graph = known_vessels
    smoothed = graph.copy()
    smooth_graph_centrelines(smoothed, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0))

    pairs = _errors_against_truth(curves, smoothed)
    measured = sum(m for m, _t in pairs)
    truth = sum(t for _m, t in pairs)
    error = measured / truth - 1

    assert abs(error) < 0.02, f"total length is {100 * error:+.2f}% from the truth"

    per_vessel = np.array([abs(m / t - 1) for m, t in pairs if t > 1.0])
    assert per_vessel.mean() < 0.02
    assert per_vessel.max() < 0.05, (
        f"worst vessel is {100 * per_vessel.max():.2f}% out"
    )


@pytest.mark.slow
def test_smoothing_beats_the_raw_path_on_every_measure(known_vessels):
    curves, skeleton, graph = known_vessels
    smoothed = graph.copy()
    smooth_graph_centrelines(smoothed, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0))

    raw_error = abs(
        sum(m for m, _t in _errors_against_truth(curves, graph))
        / sum(t for _m, t in _errors_against_truth(curves, graph))
        - 1
    )
    smoothed_error = abs(
        sum(m for m, _t in _errors_against_truth(curves, smoothed))
        / sum(t for _m, t in _errors_against_truth(curves, smoothed))
        - 1
    )
    assert smoothed_error < raw_error / 3


# --- the real thing ----------------------------------------------------------


@pytest.fixture(scope="module")
def nerve_graph():
    """The cropped nerve stack, through segmentation and graph building."""
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import build_network, segment, skeletonise

    import tempfile

    if not NERVE.exists():
        pytest.skip(f"missing fixture: {NERVE}")

    output = Path(tempfile.mkdtemp())
    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": NERVE,
            "vtk_output_prefix": output / "run",
            "plot_dir": output / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)
    volume = skeletonise(settings, segment(settings))
    network = build_network(settings, volume, schema)
    return network.graph, volume.skeleton, volume.voxel_size_zyx


def _total_length(graph) -> float:
    return sum(float(data["length"]) for *_ids, data in graph.edges(keys=True, data=True))


def _deviations(graph, skeleton, voxel_size_zyx) -> np.ndarray:
    """How far every centreline point sits from the nearest skeleton voxel."""
    from scipy.spatial import cKDTree

    support = np.argwhere(skeleton).astype(float) * np.asarray(voxel_size_zyx)
    tree = cKDTree(support)
    points = np.concatenate(
        [
            np.asarray(data["voxels"], dtype=float)
            for *_ids, data in graph.edges(keys=True, data=True)
            if data.get("voxels") is not None and len(data["voxels"]) >= 2
        ]
    )
    return tree.query(points)[0]


@pytest.mark.slow
@pytest.mark.integration
def test_the_nerve_total_length_is_smoothed_but_not_oversmoothed(nerve_graph):
    """Bounded both ways, on real data.

    The raw skeleton path totals about 5610 um on this fixture and the vessels
    it traces are around 5170. Above the upper bound the staircase is still
    being measured; below the lower bound the centrelines have been pulled
    straighter than the vessels are.
    """
    graph, _skeleton, _voxel_size = nerve_graph
    total = _total_length(graph)

    assert total < 5_400.0, f"{total:.0f} um: the voxel staircase is still in the length"
    assert total > 4_900.0, f"{total:.0f} um: the centrelines have been oversmoothed"


@pytest.mark.slow
@pytest.mark.integration
def test_the_nerve_centrelines_stay_on_the_skeleton(nerve_graph):
    """Smoothing must not move a vessel off the thing it was traced from."""
    graph, skeleton, voxel_size = nerve_graph
    deviations = _deviations(graph, skeleton, voxel_size)

    assert deviations.mean() < 0.40, (
        f"mean deviation {deviations.mean():.3f} um: the centrelines have drifted"
    )
    assert deviations.max() < 1.60, (
        f"worst point {deviations.max():.3f} um from any skeleton voxel"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_every_vessel_says_what_happened_to_it(nerve_graph):
    """Provenance, so a surprising length can be traced to a decision."""
    graph, _skeleton, _voxel_size = nerve_graph
    outcomes = {
        data.get("centreline_smoothing")
        for *_ids, data in graph.edges(keys=True, data=True)
    }
    assert outcomes <= {"smoothed", "relaxed", "kept_raw", "too_short"}
    assert "smoothed" in outcomes


# --- the smoother itself -----------------------------------------------------


def test_endpoints_never_move():
    """They are the junctions with the neighbouring vessels."""
    path = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0]], dtype=float)
    smoothed = graph_tools.taubin_smooth_polyline(path, iterations=20)
    assert smoothed[0].tolist() == [0.0, 0.0, 0.0]
    assert smoothed[-1].tolist() == [4.0, 0.0, 0.0]


def test_a_straight_line_is_left_alone():
    path = np.linspace([0, 0, 0], [10, 0, 0], 11)
    assert np.allclose(graph_tools.taubin_smooth_polyline(path, iterations=10), path)


def test_smoothing_does_not_shrink_a_curve_away():
    """Plain Laplacian smoothing collapses a curve towards its chord.

    Taubin's negative pass is what stops that, so a semicircle must keep most
    of its length rather than tending towards the straight line between its
    ends.
    """
    angles = np.linspace(0.0, np.pi, 60)
    path = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1) * 10.0
    length = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())  # noqa: E731

    smoothed = graph_tools.taubin_smooth_polyline(path, iterations=50)

    assert length(smoothed) > 0.97 * length(path)


def test_a_two_point_edge_is_left_alone():
    path = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert np.allclose(graph_tools.taubin_smooth_polyline(path), path)


def test_an_unknown_method_says_which_ones_exist():
    with pytest.raises(ValueError, match="taubin"):
        graph_tools.smooth_polyline(np.zeros((5, 3)), method="wobble")


def test_a_graph_with_no_skeleton_is_left_alone():
    """Without the skeleton there is nothing to say a path has drifted."""
    graph = nx.MultiGraph()
    graph.add_node(0, pos=np.zeros(3))
    graph.add_node(1, pos=np.array([4.0, 0.0, 0.0]))
    original = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0]]
    graph.add_edge(0, 1, key=0, voxels=original, length=10.0)

    counts = smooth_graph_centrelines(graph, np.zeros((4, 4, 4), dtype=bool))

    assert counts == {"smoothed": 0, "relaxed": 0, "kept_raw": 0, "too_short": 0}
    assert graph.edges[0, 1, 0]["voxels"] == original


def test_a_path_that_would_leave_the_vessel_is_blended_back():
    """The guard, on a case built to trip it.

    A hairpin whose corner is real: smoothing cuts it, and the corner is the
    only skeleton support there, so the result has to be pulled back towards
    the original rather than accepted.
    """
    corner = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0]], dtype=float)
    skeleton = np.zeros((16, 16, 4), dtype=bool)
    for point in corner:
        skeleton[int(point[0]), int(point[1]), int(point[2])] = True

    graph = nx.MultiGraph()
    graph.add_node(0, pos=corner[0])
    graph.add_node(1, pos=corner[-1])
    graph.add_edge(0, 1, key=0, voxels=corner.tolist(), length=20.0)

    counts = smooth_graph_centrelines(
        graph, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0), max_deviation=0.5
    )

    assert counts["smoothed"] == 0
    assert counts["relaxed"] + counts["kept_raw"] == 1


def test_smoothing_never_makes_a_centreline_longer():
    """Removing a staircase can only shorten a path.

    Taubin's response is slightly above one at low frequencies, so run far
    past the useful range it starts inflating a curve rather than smoothing it
    -- and an inflated curve wiggles *within* the distance tolerance, so the
    skeleton check alone does not see it. Unguarded, a thousand passes measured
    8.2% over the true length: worse than not smoothing at all.
    """
    angles = np.linspace(0.0, np.pi, 60)
    path = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1) * 10.0
    skeleton = np.zeros((24, 24, 4), dtype=bool)
    for point in path:
        skeleton[int(round(point[0])) + 11, int(round(point[1])), 0] = True

    graph = nx.MultiGraph()
    graph.add_node(0, pos=path[0])
    graph.add_node(1, pos=path[-1])
    original_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    graph.add_edge(0, 1, key=0, voxels=path.tolist(), length=original_length)

    smooth_graph_centrelines(
        graph, skeleton, voxel_size_zyx=(1.0, 1.0, 1.0), iterations=1000,
        max_deviation=99.0,   # distance is not what should stop this
    )

    assert graph.edges[0, 1, 0]["length"] <= original_length + 1e-9
