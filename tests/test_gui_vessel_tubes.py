"""Per-segment vessel tube meshes, without napari."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.gui.results import VESSELS, VESSEL_TUBES
from haemolynx.gui.vessel_tubes import (
    DEFAULT_TUBE_SIDES,
    TUBE_RADIUS_UM,
    colors_for_tube_vertices,
    tube_radius_um,
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
