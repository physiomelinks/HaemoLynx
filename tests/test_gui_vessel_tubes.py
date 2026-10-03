"""Vessel tube meshes, without napari."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.gui.results import VESSELS, VESSEL_TUBES
from haemolynx.gui.vessel_tubes import (
    DEFAULT_TUBE_QUALITY,
    TUBE_QUALITY_SIDES,
    TUBE_RADIUS_UM,
    clamp_tube_quality,
    colors_for_tube_vertices,
    tube_mesh,
    tube_radii_um,
    tube_radius_um,
    tubes_from_vectors,
    vessel_tubes_layer_name,
)


def _polyline_vectors(points):
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:] - points[:-1]], axis=1)


def _helix(turns=3, points=200):
    t = np.linspace(0.0, 2.0 * np.pi * turns, points)
    return np.stack([3.0 * np.cos(t), 3.0 * np.sin(t), 0.8 * t], axis=1)


def _staircase(steps=40):
    """A vessel running at 45 degrees through the voxel grid: unit steps along
    x then y, turning a right angle at every step."""
    moves = np.tile([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (steps // 2, 1))
    return np.vstack([np.zeros((1, 3)), np.cumsum(moves, axis=0)])


def _edge_use(faces):
    """How many faces use each undirected edge: 2 everywhere for a closed surface."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return set(counts.tolist())


def _nearest_on_polyline(points, polyline):
    """Each point's distance to *polyline*, and the nearest point on it."""
    polyline = np.asarray(polyline, dtype=float)
    a, ab = polyline[:-1], np.diff(polyline, axis=0)
    t = np.einsum("pij,ij->pi", points[:, None] - a[None], ab) / np.einsum("ij,ij->i", ab, ab)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    distance = np.linalg.norm(points[:, None] - closest, axis=2)
    nearest = distance.argmin(axis=1)
    return distance.min(axis=1), closest[np.arange(len(points)), nearest]


def _ring_vertices(vertices, sides, tubes=1):
    """The tube rings' vertices, ``(rings, sides, 3)``, without the rounded ends."""
    levels = max(2, sides // 4)
    ends = 2 * tubes * ((levels - 1) * sides + 1)
    return vertices[: len(vertices) - ends].reshape(-1, sides, 3)


def _cut(vertices, faces, point, normal):
    """Where the plane through *point* square to *normal* cuts the mesh."""
    side = (vertices - point) @ normal
    cut = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        si, sj = side[faces[:, i]], side[faces[:, j]]
        crosses = (si < 0) != (sj < 0)
        w = si[crosses] / (si[crosses] - sj[crosses])
        vi, vj = vertices[faces[crosses, i]], vertices[faces[crosses, j]]
        cut.append(vi + w[:, None] * (vj - vi))
    return np.concatenate(cut)


def test_empty_vectors_yield_empty_mesh():
    vertices, faces, index = tubes_from_vectors(np.empty((0, 2, 3)))
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert index.shape == (0,)


def test_zero_length_and_non_finite_steps_are_skipped():
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
        ]
    )
    vertices, faces, index = tubes_from_vectors(vectors)
    assert vertices.shape == (0, 3) and faces.shape == (0, 3) and index.shape == (0,)

    points = [[0, 0, 0], [0, 0, 2], [0, 0, 2], [0, 0, 4]]
    vertices, faces, index = tubes_from_vectors(_polyline_vectors(points), sides=8)
    assert 1 not in index.tolist()  # the zero-length step
    assert _edge_use(faces) == {2}


def test_bad_input_raises():
    with pytest.raises(ValueError):
        tubes_from_vectors(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        tubes_from_vectors(_polyline_vectors(_helix(points=5)), sides=2)
    with pytest.raises(ValueError):
        tubes_from_vectors(np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]), radius=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        tubes_from_vectors(np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]), radius=0.0)


def test_a_straight_vessel_is_its_own_radius_everywhere():
    """Each vessel's own diameter, halved, round its whole length and over
    its rounded ends -- along every axis, where a single cross product
    would vanish."""
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            [[20.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
        ]
    )
    radii = np.array([1.0, 6.5, 2.0])
    vertices, faces, index = tubes_from_vectors(vectors, radius=radii, sides=8)
    assert _edge_use(faces) == {2}
    for row, radius in enumerate(radii):
        own = vertices[index == row]
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        distance, _ = _nearest_on_polyline(own, segment)
        np.testing.assert_allclose(distance, radius, atol=1e-9)


def test_an_unknown_radius_falls_back_to_the_default():
    """A missing/zero/negative diameter (not yet assigned) still draws a
    visible tube -- only that vessel falls back, the others keep their own."""
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[20.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    vertices, _faces, index = tubes_from_vectors(vectors, radius=np.array([np.nan, 0.0, 3.0]))
    for row, radius in enumerate([TUBE_RADIUS_UM, TUBE_RADIUS_UM, 3.0]):
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        distance, _ = _nearest_on_polyline(vertices[index == row], segment)
        np.testing.assert_allclose(distance, radius, atol=1e-9)


def test_a_tube_is_one_closed_surface_facing_outward():
    """No bands and no open ends; every face lit from outside the vessel."""
    helix = _helix()
    vertices, faces, _index = tubes_from_vectors(_polyline_vectors(helix), radius=1.0, sides=16)
    assert _edge_use(faces) == {2}
    assert faces.min() >= 0 and faces.max() < len(vertices)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    centroid = (a + b + c) / 3.0
    _distance, foot = _nearest_on_polyline(centroid, helix)
    assert np.all(np.einsum("ij,ij->i", np.cross(b - a, c - a), centroid - foot) > 0.0)


def test_the_tube_runs_from_node_to_node():
    """Its ends are where the vessel's own are, so vessels meeting at a node
    meet there, and each is rounded off one radius beyond its node."""
    helix = _helix(points=60)
    sides = 12
    vertices, _faces, _index = tubes_from_vectors(_polyline_vectors(helix), radius=1.5, sides=sides)
    rings = _ring_vertices(vertices, sides)
    np.testing.assert_allclose(rings[0].mean(axis=0), helix[0], atol=1e-9)
    np.testing.assert_allclose(rings[-1].mean(axis=0), helix[-1], atol=1e-9)
    start_pole, end_pole = vertices[-2], vertices[-1]
    assert np.linalg.norm(start_pole - helix[0]) == pytest.approx(1.5)
    assert np.linalg.norm(end_pole - helix[-1]) == pytest.approx(1.5)


def test_the_rings_follow_the_vessel_not_its_voxel_steps():
    """The same straight vessel sampled at any step draws the same tube."""
    coarse = np.stack([np.zeros(31), np.zeros(31), np.linspace(0.0, 30.0, 31)], axis=1)
    fine = np.stack([np.zeros(121), np.zeros(121), np.linspace(0.0, 30.0, 121)], axis=1)
    from_coarse, faces_coarse, _ = tubes_from_vectors(_polyline_vectors(coarse), radius=3.0)
    from_fine, faces_fine, _ = tubes_from_vectors(_polyline_vectors(fine), radius=3.0)
    np.testing.assert_allclose(from_fine, from_coarse, atol=1e-9)
    np.testing.assert_array_equal(faces_fine, faces_coarse)


@pytest.mark.parametrize("quality", range(len(TUBE_QUALITY_SIDES)))
def test_a_staircase_centreline_is_drawn_one_straight_tube(quality):
    """A real capillary: steps under a micron, the vessel three across, the
    centreline kinking at every voxel. Drawn on that path, the tube pinches,
    bulges and wobbles; drawn on it smoothed over the vessel's own radius,
    every cut across it is the vessel's circle, on one straight line."""
    radius = 3.0
    points = _staircase()
    vertices, faces, _index = tube_mesh(_polyline_vectors(points), radius=radius, quality=quality)
    course = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    sideways = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    upward = np.array([0.0, 0.0, 1.0])
    across_flats = 2.0 * radius * np.cos(np.pi / TUBE_QUALITY_SIDES[quality])
    middles = []
    for along in np.linspace(8.0, 20.0, 25):  # clear of both ends
        cut = _cut(vertices, faces, along * course, course)
        for direction in (sideways, upward):
            width = np.ptp(cut @ direction)
            assert across_flats * 0.99 <= width <= 2.0 * radius * 1.01, (along, width)
        middles.append(0.5 * ((cut @ sideways).max() + (cut @ sideways).min()))
    assert np.ptp(middles) < 0.05  # no wobble from voxel to voxel


def test_a_sharp_bend_is_drawn_as_a_curve():
    """A right-angle kink in the centreline turns the tube round a curve, a
    few degrees ring to ring, and every ring is the vessel's own circle."""
    bend = [[0, 0, 0], [0, 0, 10], [0, 10, 10]]
    sides = 12
    vertices, _faces, _index = tubes_from_vectors(_polyline_vectors(bend), radius=1.0, sides=sides)
    rings = _ring_vertices(vertices, sides)
    centres = rings.mean(axis=1)
    np.testing.assert_allclose(np.linalg.norm(rings - centres[:, None], axis=2), 1.0, atol=1e-9)
    facing = np.cross(rings[:, 0] - centres, rings[:, sides // 4] - centres)
    facing /= np.linalg.norm(facing, axis=1, keepdims=True)
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", facing[1:], facing[:-1]), -1.0, 1.0)))
    assert turn.max() < 25.0
    assert facing[0] @ [0, 0, 1] == pytest.approx(1.0)
    assert facing[-1] @ [0, 1, 0] == pytest.approx(1.0)


def test_two_vessels_meeting_at_a_node_stay_two_tubes():
    """Each one straight to the node and rounded off there, not bent towards
    the other."""
    first, second = [[0, 0, 0], [0, 0, 5], [0, 0, 10]], [[0, 0, 10], [0, 5, 10], [0, 10, 10]]
    vectors = np.concatenate([_polyline_vectors(first), _polyline_vectors(second)])
    vertices, faces, index = tubes_from_vectors(vectors, radius=1.0, sides=8, groups=[7, 7, 9, 9])
    assert _edge_use(faces) == {2}
    assert set(index.tolist()) == {0, 1, 2, 3}
    for rows, line in (([0, 1], first), ([2, 3], second)):
        own = vertices[np.isin(index, rows)]
        distance, _ = _nearest_on_polyline(own, np.asarray(line, dtype=float))
        np.testing.assert_allclose(distance, 1.0, atol=1e-9)


def test_each_vertex_belongs_to_the_step_it_was_drawn_on():
    """Per-step radius and colour: a ring takes them from the step it lies on."""
    points = np.stack([np.zeros(4), np.zeros(4), [0.0, 4.0, 8.0, 12.0]], axis=1)
    sides = 8
    vertices, _faces, index = tubes_from_vectors(
        _polyline_vectors(points), radius=np.array([1.0, 3.0, np.nan]), sides=sides
    )
    ring_count = len(_ring_vertices(vertices, sides)) * sides
    rings, owner = vertices[:ring_count], index[:ring_count]
    radial = np.linalg.norm(rings[:, :2], axis=1)
    for row, radius in enumerate([1.0, 3.0, TUBE_RADIUS_UM]):
        assert np.any(owner == row)
        np.testing.assert_allclose(radial[owner == row], radius)
        z = rings[owner == row, 2]
        assert z.min() >= 4.0 * row - 1e-9 and z.max() <= 4.0 * (row + 1) + 1e-9

    colours = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    repeated = colors_for_tube_vertices(index, colours)
    for row in range(3):
        np.testing.assert_array_equal(repeated[index == row], np.broadcast_to(colours[row], repeated[index == row].shape))


def test_a_tube_does_not_twist():
    """Each ring's own frame differs; the rings are lined up, not left twisted."""
    sides = 16
    vertices, faces, _index = tubes_from_vectors(_polyline_vectors(_helix()), radius=1.0, sides=sides)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    longest = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).max()
    # One ring along and one side round at most -- never across the tube.
    assert longest < 1.0


def test_tube_radius_is_at_least_two_microns():
    assert tube_radius_um(0.6) == pytest.approx(TUBE_RADIUS_UM)
    assert tube_radius_um(3.0) == pytest.approx(3.0)
    assert tube_radius_um(None) == pytest.approx(TUBE_RADIUS_UM)


def test_tube_radii_um_halves_diameters_and_handles_empty_input():
    np.testing.assert_allclose(tube_radii_um(np.array([4.0, 8.0, 3.0])), [2.0, 4.0, 1.5])
    assert tube_radii_um(None) is None
    assert tube_radii_um(np.array([])) is None


def test_tube_layer_name_stays_haemolynx_owned():
    assert vessel_tubes_layer_name(VESSELS) == VESSEL_TUBES
    assert vessel_tubes_layer_name(f"{VESSELS} (HaemoLynx)") == f"{VESSEL_TUBES} (HaemoLynx)"


def test_each_quality_step_is_the_same_tube_rounder():
    assert list(TUBE_QUALITY_SIDES) == sorted(TUBE_QUALITY_SIDES)
    assert len(set(TUBE_QUALITY_SIDES)) == len(TUBE_QUALITY_SIDES)
    vectors = _polyline_vectors(_helix())
    for quality, sides in enumerate(TUBE_QUALITY_SIDES):
        for got, want in zip(tube_mesh(vectors, radius=1.5, quality=quality),
                             tubes_from_vectors(vectors, radius=1.5, sides=sides)):
            np.testing.assert_array_equal(got, want)
    counts = [len(tube_mesh(vectors, quality=q)[0]) for q in range(len(TUBE_QUALITY_SIDES))]
    assert counts == sorted(counts)


def test_quality_is_clamped_to_a_real_level():
    top = len(TUBE_QUALITY_SIDES) - 1
    assert clamp_tube_quality(-3) == 0
    assert clamp_tube_quality(top + 5) == top
    assert clamp_tube_quality("2") == 2
    assert clamp_tube_quality(None) == DEFAULT_TUBE_QUALITY


def test_the_top_quality_is_fast_enough_for_a_large_network():
    import time

    rng = np.random.default_rng(0)
    vectors = _polyline_vectors(np.cumsum(rng.normal(size=(100_001, 3)), axis=0))
    groups = np.repeat(np.arange(100_000 // 20), 20)
    started = time.perf_counter()
    tube_mesh(vectors, quality=len(TUBE_QUALITY_SIDES) - 1, groups=groups)
    assert time.perf_counter() - started < 10.0


@pytest.mark.slow
def test_many_separate_vessels_build_fast():
    """A whole-network-scale rebuild must stay fast enough not to freeze the GUI.

    Toggling the Tubes/Lines radio, or moving the Z-depth slider while tubes
    are on, rebuilds the whole mesh synchronously on the Qt main thread (see
    ``_sync_vessel_tubes`` / ``apply_view_z`` in ``gui/_widget.py``). A
    per-vessel Python loop here would take many seconds at this size.
    """
    import time

    rng = np.random.default_rng(1)
    n_seg = 50_000
    vectors = np.stack([rng.random((n_seg, 3)) * 1000.0, rng.normal(size=(n_seg, 3))], axis=1)

    started = time.perf_counter()
    vertices, faces, index = tube_mesh(vectors)
    elapsed = time.perf_counter() - started

    assert set(np.unique(index).tolist()) == set(range(n_seg))
    assert np.all(np.isfinite(vertices))
    assert np.all(faces >= 0) and np.all(faces < len(vertices))
    assert elapsed < 5.0, f"tube_mesh took {elapsed:.2f}s for {n_seg} vessels"
    for row in rng.choice(n_seg, size=20, replace=False):
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        distance, _ = _nearest_on_polyline(vertices[index == row], segment)
        np.testing.assert_allclose(distance, TUBE_RADIUS_UM, atol=1e-8)
