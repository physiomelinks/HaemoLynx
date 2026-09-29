"""Per-segment vessel tube meshes, without napari."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.gui.results import VESSELS, VESSEL_TUBES
from haemolynx.gui.vessel_tubes import (
    DEFAULT_TUBE_QUALITY,
    DEFAULT_TUBE_SIDES,
    TUBE_QUALITY_SIDES,
    TUBE_RADIUS_UM,
    TUBE_SHADING,
    clamp_tube_quality,
    colors_for_tube_vertices,
    joined_tubes_from_vectors,
    tube_mesh,
    tube_radii_um,
    tube_radius_um,
    tube_shading_for_quality,
    tubes_from_vectors,
    vessel_tubes_layer_name,
)


def _radial_distances(origin, direction, vertices) -> np.ndarray:
    tangent = np.asarray(direction, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    rel = np.asarray(vertices, dtype=float) - np.asarray(origin, dtype=float)
    axial = rel @ tangent
    radial = rel - axial[:, None] * tangent
    return np.linalg.norm(radial, axis=1)


def test_empty_vectors_yield_empty_mesh():
    vertices, faces, index = tubes_from_vectors(np.empty((0, 2, 3)))
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert index.shape == (0,)


def test_zero_length_segments_are_skipped():
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
        ]
    )
    vertices, faces, index = tubes_from_vectors(vectors)
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert index.shape == (0,)


def test_axis_aligned_x_and_z_prisms_have_nonzero_radius():
    radius = 2.0
    sides = 6
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    vertices, faces, index = tubes_from_vectors(
        vectors, radius=radius, sides=sides
    )
    assert vertices.shape == (2 * sides * 2, 3)
    assert faces.shape == (2 * sides * 2, 3)
    assert np.all(faces >= 0)
    assert np.all(faces < len(vertices))
    # Faces are non-degenerate triangles.
    for tri in faces:
        a, b, c = vertices[tri]
        area = np.linalg.norm(np.cross(b - a, c - a))
        assert area > 0.0

    for src in (0, 1):
        owned = vertices[index == src]
        distances = _radial_distances(vectors[src, 0], vectors[src, 1], owned)
        np.testing.assert_allclose(distances, radius, atol=1e-9)
        assert np.min(distances) > 0.0


def test_axis_aligned_y_prism_is_also_nondegenerate():
    vectors = np.array([[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    vertices, faces, index = tubes_from_vectors(vectors, radius=2.0, sides=6)
    distances = _radial_distances(vectors[0, 0], vectors[0, 1], vertices)
    np.testing.assert_allclose(distances, 2.0, atol=1e-9)
    assert faces.shape[0] == DEFAULT_TUBE_SIDES * 2
    assert set(index.tolist()) == {0}


def test_consecutive_polyline_steps_are_disjoint_and_abut():
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    sides = 6
    vertices, _faces, index = tubes_from_vectors(
        vectors, radius=2.0, sides=sides
    )
    first = vertices[index == 0]
    second = vertices[index == 1]
    assert set(np.flatnonzero(index == 0)).isdisjoint(set(np.flatnonzero(index == 1)))
    end_ring = first[sides:]
    start_ring = second[:sides]
    np.testing.assert_allclose(end_ring, start_ring, atol=1e-9)


def test_segment_index_maps_vertices_onto_vector_rows():
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    vertices, _faces, index = tubes_from_vectors(vectors, sides=4)
    assert set(index.tolist()) == {1, 2}
    assert len(vertices) == 2 * 4 * 2
    colours = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    )
    repeated = colors_for_tube_vertices(index, colours)
    np.testing.assert_array_equal(repeated[index == 1], np.broadcast_to(colours[1], repeated[index == 1].shape))
    np.testing.assert_array_equal(repeated[index == 2], np.broadcast_to(colours[2], repeated[index == 2].shape))


def test_tube_radius_is_at_least_two_microns():
    assert tube_radius_um(0.6) == pytest.approx(TUBE_RADIUS_UM)
    assert tube_radius_um(3.0) == pytest.approx(3.0)
    assert tube_radius_um(None) == pytest.approx(TUBE_RADIUS_UM)


def test_per_segment_radius_array_gives_each_segment_its_own_radius():
    """A vessel's own diameter, not one uniform radius for the whole network."""
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    radii = np.array([1.0, 6.5])
    sides = 6
    vertices, _faces, index = tubes_from_vectors(vectors, radius=radii, sides=sides)

    for src, expected in enumerate(radii):
        owned = vertices[index == src]
        distances = _radial_distances(vectors[src, 0], vectors[src, 1], owned)
        np.testing.assert_allclose(distances, expected, atol=1e-9)


def test_per_segment_radius_falls_back_to_default_for_invalid_entries():
    """A missing/zero/negative diameter (not yet assigned) still draws a
    visible tube instead of vanishing or breaking the whole mesh -- only
    that one segment falls back, the valid one keeps its own radius."""
    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    radii = np.array([np.nan, 0.0, 3.0])
    vertices, _faces, index = tubes_from_vectors(vectors, radius=radii, sides=6)

    for src in (0, 1):
        owned = vertices[index == src]
        distances = _radial_distances(vectors[src, 0], vectors[src, 1], owned)
        np.testing.assert_allclose(distances, TUBE_RADIUS_UM, atol=1e-9)
    owned = vertices[index == 2]
    distances = _radial_distances(vectors[2, 0], vectors[2, 1], owned)
    np.testing.assert_allclose(distances, 3.0, atol=1e-9)


def test_per_segment_radius_array_wrong_length_raises():
    vectors = np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    with pytest.raises(ValueError):
        tubes_from_vectors(vectors, radius=np.array([1.0, 2.0]))


def test_tube_radii_um_halves_diameters_and_handles_empty_input():
    np.testing.assert_allclose(
        tube_radii_um(np.array([4.0, 8.0, 3.0])), [2.0, 4.0, 1.5]
    )
    assert tube_radii_um(None) is None
    assert tube_radii_um(np.array([])) is None


def test_tube_layer_name_stays_haemolynx_owned():
    assert vessel_tubes_layer_name(VESSELS) == VESSEL_TUBES
    assert vessel_tubes_layer_name(f"{VESSELS} (HaemoLynx)") == (
        f"{VESSEL_TUBES} (HaemoLynx)"
    )


def test_mesh_size_is_linear_in_segments_and_sides():
    rng = np.random.default_rng(0)
    n_seg = 40
    origins = rng.random((n_seg, 3))
    directions = rng.normal(size=(n_seg, 3))
    vectors = np.stack([origins, directions], axis=1)
    sides = 6
    vertices, faces, index = tubes_from_vectors(vectors, sides=sides)
    assert len(vertices) == n_seg * sides * 2
    assert len(faces) == n_seg * sides * 2
    assert len(index) == len(vertices)
    # Far cheaper than triangulating Shapes paths: a few hundred faces per
    # segment, not a vispy path mesh per vessel.
    assert len(faces) < n_seg * 20


@pytest.mark.slow
def test_large_network_mesh_is_vectorized_and_fast():
    """A whole-network-scale rebuild must stay fast enough not to freeze the GUI.

    Toggling the Tubes/Lines radio, or moving the Z-depth slider while tubes
    are on, rebuilds the whole mesh synchronously on the Qt main thread (see
    ``_sync_vessel_tubes`` / ``apply_view_z`` in ``gui/_widget.py``). A
    per-segment Python loop here used to take ~11s at 100k segments -- long
    enough to read as a freeze or an unresponsive-app crash. Bound generously
    below the old loop's cost so a regression to scalar Python is caught
    without making CI flaky on a slow runner.
    """
    import time

    rng = np.random.default_rng(1)
    n_seg = 50_000
    sides = 6
    origins = rng.random((n_seg, 3)) * 1000.0
    directions = rng.normal(size=(n_seg, 3))
    vectors = np.stack([origins, directions], axis=1)

    start = time.perf_counter()
    vertices, faces, index = tubes_from_vectors(vectors, sides=sides)
    elapsed = time.perf_counter() - start

    assert len(vertices) == n_seg * sides * 2
    assert len(faces) == n_seg * sides * 2
    assert np.all(np.isfinite(vertices))
    assert np.all(faces >= 0) and np.all(faces < len(vertices))
    assert elapsed < 5.0, f"tubes_from_vectors took {elapsed:.2f}s for {n_seg} segments"

    # Every vertex sits exactly `radius` from its segment's centreline --
    # the invariant the scalar version also had to satisfy.
    for src in rng.choice(n_seg, size=20, replace=False):
        owned = vertices[index == src]
        distances = _radial_distances(vectors[src, 0], vectors[src, 1], owned)
        np.testing.assert_allclose(distances, TUBE_RADIUS_UM, atol=1e-8)


# --- render quality: joined, rounder tubes ------------------------------------


def _polyline_vectors(points):
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:] - points[:-1]], axis=1)


def _edge_use(faces):
    """How many faces use each undirected edge: 2 everywhere for a closed surface."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return set(counts.tolist())


def _helix(turns=3, points=200):
    t = np.linspace(0.0, 2.0 * np.pi * turns, points)
    return np.stack([3.0 * np.cos(t), 3.0 * np.sin(t), 0.8 * t], axis=1)


def test_quality_zero_is_the_original_drawing_exactly():
    vectors = _polyline_vectors(_helix())
    expected = tubes_from_vectors(vectors, radius=1.5, sides=TUBE_QUALITY_SIDES[0])
    for got, want in zip(tube_mesh(vectors, radius=1.5, quality=0), expected):
        np.testing.assert_array_equal(got, want)
    assert TUBE_QUALITY_SIDES[0] == DEFAULT_TUBE_SIDES
    assert DEFAULT_TUBE_QUALITY == 0


def test_each_quality_step_is_rounder():
    assert list(TUBE_QUALITY_SIDES) == sorted(TUBE_QUALITY_SIDES)
    assert len(set(TUBE_QUALITY_SIDES)) == len(TUBE_QUALITY_SIDES)
    vectors = _polyline_vectors(_helix())
    counts = [len(tube_mesh(vectors, quality=q)[0]) for q in range(len(TUBE_QUALITY_SIDES))]
    assert counts[1:] == sorted(counts[1:])


def test_a_joined_tube_is_one_closed_surface():
    """No bands: the rings are shared, so every edge borders exactly two faces."""
    vertices, faces, _index = joined_tubes_from_vectors(
        _polyline_vectors(_helix()), radius=1.0, sides=16
    )
    assert _edge_use(faces) == {2}
    assert faces.min() >= 0 and faces.max() < len(vertices)


def test_the_original_prisms_are_separate_open_bands():
    """The premise: level 0 leaves every prism's ends open and unshared."""
    _vertices, faces, _index = tubes_from_vectors(_polyline_vectors(_helix()), sides=6)
    assert 1 in _edge_use(faces)


def test_a_straight_tube_keeps_its_radius():
    line = np.stack([np.zeros(10), np.zeros(10), np.arange(10.0)], axis=1)
    vertices, _faces, _index = joined_tubes_from_vectors(
        _polyline_vectors(line), radius=1.0, sides=8
    )
    rings = vertices[: 8 * 10]
    np.testing.assert_allclose(np.linalg.norm(rings[:, :2], axis=1), 1.0)


def test_a_bend_is_mitred_so_the_tube_keeps_its_width():
    bend = _polyline_vectors([[0, 0, 0], [0, 0, 5], [0, 5, 5]])
    vertices, _faces, _index = joined_tubes_from_vectors(bend, radius=1.0, sides=8)
    joint = np.linalg.norm(vertices[8:16] - [0, 0, 5], axis=1)
    # A 90-degree mitre is a sqrt(2)-stretched ellipse across the bend.
    assert joint.min() == pytest.approx(1.0)
    assert joint.max() == pytest.approx(np.sqrt(2.0))


def test_a_joined_tube_does_not_twist():
    """Each ring's own frame differs; the rings are lined up, not left twisted."""
    helix = _helix()
    vectors = _polyline_vectors(helix)
    sides = 16
    vertices, faces, _index = joined_tubes_from_vectors(vectors, radius=1.0, sides=sides)
    side_faces = faces[: len(vectors) * sides * 2]
    step = np.linalg.norm(vectors[:, 1], axis=1).max()
    across = 2.0 * np.pi * 1.0 / sides
    longest = np.linalg.norm(vertices[side_faces[:, 0]] - vertices[side_faces[:, 2]], axis=1).max()
    # A face runs one step along and at most one side round -- never across the tube.
    assert longest <= np.hypot(step, across) * 1.2


def test_two_vessels_meeting_at_a_node_stay_two_tubes():
    vectors = np.concatenate([
        _polyline_vectors([[0, 0, 0], [0, 0, 5]]),
        _polyline_vectors([[0, 0, 5], [0, 5, 5]]),
    ])
    joined, _faces, _index = joined_tubes_from_vectors(vectors, radius=1.0, sides=8)
    apart, faces, index = joined_tubes_from_vectors(vectors, radius=1.0, sides=8, groups=[7, 9])
    assert len(apart) == len(joined) + 8 + 2  # one more ring and two more caps
    assert _edge_use(faces) == {2}
    assert set(index.tolist()) == {0, 1}


def test_joined_vertices_map_onto_their_own_steps():
    vectors = _polyline_vectors(_helix(points=20))
    vertices, _faces, index = joined_tubes_from_vectors(vectors, radius=1.0, sides=6)
    assert len(index) == len(vertices)
    assert index.min() == 0 and index.max() == len(vectors) - 1
    # Ring k (k < steps) is where step k starts.
    np.testing.assert_array_equal(index[: 6 * len(vectors)], np.repeat(np.arange(len(vectors)), 6))


def test_joined_tubes_skip_zero_length_steps_and_take_per_step_radii():
    points = [[0, 0, 0], [0, 0, 2], [0, 0, 2], [0, 0, 4]]
    vectors = _polyline_vectors(points)
    vertices, faces, index = joined_tubes_from_vectors(
        vectors, radius=np.array([1.0, 5.0, np.nan]), sides=8
    )
    assert 1 not in index.tolist()  # the zero-length step
    assert _edge_use(faces) == {2}
    start_ring = np.linalg.norm(vertices[:8, :2], axis=1)
    np.testing.assert_allclose(start_ring, 1.0)
    end_ring = np.linalg.norm(vertices[16:24, :2], axis=1)
    np.testing.assert_allclose(end_ring, TUBE_RADIUS_UM)  # NaN falls back


def test_empty_and_bad_input_for_joined_tubes():
    vertices, faces, index = joined_tubes_from_vectors(np.empty((0, 2, 3)))
    assert vertices.shape == (0, 3) and faces.shape == (0, 3) and index.shape == (0,)
    with pytest.raises(ValueError):
        joined_tubes_from_vectors(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        joined_tubes_from_vectors(_polyline_vectors(_helix(points=5)), sides=2)


def test_shading_is_flat_for_the_prisms_and_smooth_for_joined_tubes():
    assert tube_shading_for_quality(0) == TUBE_SHADING
    assert {tube_shading_for_quality(q) for q in range(1, len(TUBE_QUALITY_SIDES))} == {"smooth"}


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
