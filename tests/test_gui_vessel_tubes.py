"""Vessel tube meshes, without napari."""
from __future__ import annotations

import numpy as np
import pytest

from haemolynx.graph import IS_ZERO_RESISTANCE
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


def _resampled(start, stop, step=0.5):
    count = int(np.ceil(np.linalg.norm(np.subtract(stop, start)) / step)) + 1
    return np.linspace(np.asarray(start, dtype=float), np.asarray(stop, dtype=float), count)


def _network(paths, radii):
    """Vectors for *paths*, with each row's radius and the vessel it is on."""
    vectors = np.concatenate([_polyline_vectors(path) for path in paths])
    owner = np.repeat(np.arange(len(paths)), [len(path) - 1 for path in paths])
    return vectors, np.asarray(radii, dtype=float)[owner], owner


def _poles(faces, sides):
    """The tips of the rounded ends: the only vertices *sides* faces share,
    for any *sides* over six."""
    return np.flatnonzero(np.bincount(faces.ravel()) == sides)


def _pieces(faces):
    """How many separate surfaces *faces* make."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = int(faces.max()) + 1
    links = coo_matrix(
        (np.ones(2 * len(faces)), (np.repeat(faces[:, 0], 2), faces[:, 1:].ravel())),
        shape=(count, count),
    )
    return connected_components(links, directed=False)[0]


def _ring_frames(rings):
    """Each ring's centre, radius and the way it faces along the tube."""
    centre = rings.mean(axis=1)
    radius = np.linalg.norm(rings - centre[:, None], axis=2).max(axis=1)
    facing = np.cross(rings[:, 0] - centre, rings[:, rings.shape[1] // 4] - centre)
    facing /= np.linalg.norm(facing, axis=1, keepdims=True)
    return centre, radius, facing


def _assert_no_ring_reaches_through_the_next(rings):
    """Each ring wholly ahead of the one before it, and that one wholly behind it."""
    centre, _radius, facing = _ring_frames(rings)
    ahead = np.einsum("isk,ik->is", rings[1:] - centre[:-1, None], facing[:-1])
    behind = np.einsum("isk,ik->is", rings[:-1] - centre[1:, None], facing[1:])
    assert ahead.min() > 0.0 and behind.max() < 0.0


def _assert_the_tube_faces_outward(vertices, faces, rings):
    """Every face between two rings lit from outside the axis joining them."""
    count = rings.shape[0] * rings.shape[1]
    band = faces[(faces < count).all(axis=1)]
    a, b, c = vertices[band[:, 0]], vertices[band[:, 1]], vertices[band[:, 2]]
    centroid = (a + b + c) / 3.0
    centre = rings.mean(axis=1)
    ring = band // rings.shape[1]
    p, q = centre[ring.min(axis=1)], centre[ring.max(axis=1)]
    t = np.clip(
        np.einsum("ij,ij->i", centroid - p, q - p)
        / np.maximum(np.einsum("ij,ij->i", q - p, q - p), 1e-12),
        0.0, 1.0,
    )
    foot = p + t[:, None] * (q - p)
    assert np.all(np.einsum("ij,ij->i", np.cross(b - a, c - a), centroid - foot) > 0.0)


def _cap_volume_share(vertices, faces, sides):
    """The rounded ends' share of all the volume the tubes enclose."""
    levels = max(2, sides // 4)
    caps = len(_poles(faces, sides))
    first_level = vertices[len(vertices) - caps - caps * (levels - 1) * sides : len(vertices) - caps]
    first_level = first_level.reshape(caps, levels - 1, sides, 3)[:, 0]
    centre = first_level.mean(axis=1)
    radius = np.linalg.norm(first_level[:, 0] - centre, axis=1) / np.cos(0.5 * np.pi / levels)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    enclosed = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return float((2.0 / 3.0) * np.pi * (radius ** 3).sum() / enclosed)


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


def test_a_vessel_is_its_own_radius_mid_vessel_and_its_nodes_at_its_ends():
    """Each vessel's own diameter, halved, along its middle; at each end the
    radius of the node there. A vessel ending on nothing is its own radius
    to the end and over its rounded end -- along every axis, where a single
    cross product would vanish."""
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

    # At a junction each vessel ends no wider than the widest of the others.
    node = np.array([0.0, 0.0, 30.0])
    paths = [_resampled([0.0, 0.0, 0.0], node), _resampled(node, [0.0, 30.0, 30.0]),
             _resampled(node, [0.0, -30.0, 30.0])]
    vectors, radii, owner = _network(paths, [4.0, 3.0, 2.0])
    sides = 12
    vertices, faces, index = tubes_from_vectors(vectors, radius=radii, sides=sides, groups=owner)
    assert _edge_use(faces) == {2}
    rings = _ring_vertices(vertices, sides, tubes=3)
    centre, radius, _facing = _ring_frames(rings)
    ring_owner = owner[index[: rings.shape[0] * sides : sides]]
    for vessel, (own, at_node) in enumerate(((4.0, 3.0), (3.0, 3.0), (2.0, 2.0))):
        path = paths[vessel]
        mine = np.flatnonzero(ring_owner == vessel)

        def radius_at(point):
            return radius[mine[np.argmin(np.linalg.norm(centre[mine] - point, axis=1))]]

        far = path[0] if vessel == 0 else path[-1]
        assert radius_at(0.5 * (path[0] + path[-1])) == pytest.approx(own)
        assert radius_at(far) == pytest.approx(own)
        assert radius_at(node) == pytest.approx(at_node)


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


def test_the_tube_runs_from_end_node_to_end_node():
    """A vessel cut in two at a node only the two halves meet is still one
    tube: its ends where the vessel's are, each rounded off one radius
    beyond its node, and no rounded end at the node between."""
    helix = _helix(points=60)
    sides = 12
    groups = np.repeat([0, 1], [30, 29])
    vertices, faces, _index = tubes_from_vectors(
        _polyline_vectors(helix), radius=1.5, sides=sides, groups=groups
    )
    assert _pieces(faces) == 1
    rings = _ring_vertices(vertices, sides)
    np.testing.assert_allclose(rings[0].mean(axis=0), helix[0], atol=1e-9)
    np.testing.assert_allclose(rings[-1].mean(axis=0), helix[-1], atol=1e-9)
    np.testing.assert_array_equal(_poles(faces, sides), [len(vertices) - 2, len(vertices) - 1])
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


def test_two_vessels_meeting_at_a_node_are_one_tube_coloured_apart():
    """Only the two of them at the node: one tube turning through it, with no
    rounded end there, and each vessel's steps still its own colour."""
    first, second = [[0, 0, 0], [0, 0, 5], [0, 0, 10]], [[0, 0, 10], [0, 5, 10], [0, 10, 10]]
    vectors = np.concatenate([_polyline_vectors(first), _polyline_vectors(second)])
    sides = 8
    vertices, faces, index = tubes_from_vectors(vectors, radius=1.0, sides=sides, groups=[7, 7, 9, 9])
    assert _edge_use(faces) == {2}
    assert _pieces(faces) == 1
    poles = vertices[_poles(faces, sides)]
    assert len(poles) == 2
    assert np.linalg.norm(poles - [0.0, 0.0, 10.0], axis=1).min() > 5.0
    rings = _ring_vertices(vertices, sides)
    ring_owner = index[: rings.shape[0] * sides : sides]
    assert set(ring_owner.tolist()) == {0, 1, 2, 3}
    assert np.all(np.diff(ring_owner) >= 0)
    colours = np.eye(4)
    np.testing.assert_array_equal(colors_for_tube_vertices(index, colours), colours[index])


def test_each_vertex_belongs_to_the_step_it_was_drawn_on():
    """A ring takes its colour from the step it lies on, and its radius from
    the vessel: its own mid-vessel, easing to the node's at each end."""
    points = np.stack([np.zeros(4), np.zeros(4), [0.0, 4.0, 8.0, 12.0]], axis=1)
    sides = 8
    vertices, _faces, index = tubes_from_vectors(
        _polyline_vectors(points), radius=np.array([1.0, 3.0, np.nan]), sides=sides
    )
    rings = _ring_vertices(vertices, sides)
    centre, radius, _facing = _ring_frames(rings)
    owner = index[: rings.shape[0] * sides : sides]
    for row in range(3):
        z = centre[owner == row, 2]
        assert z.size and z.min() >= 4.0 * row - 0.2 and z.max() <= 4.0 * (row + 1) + 0.2
    for z, want in ((0.0, 1.0), (4.0, 2.0), (8.0, 2.5), (12.0, TUBE_RADIUS_UM)):
        assert radius[np.argmin(np.abs(centre[:, 2] - z))] == pytest.approx(want)
    assert radius[owner == 1].max() == pytest.approx(3.0, abs=0.05)

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


def test_a_junctions_widest_vessel_narrows_into_its_widest_neighbour():
    """A 12 µm vessel dividing into two of 5 µm. Drawn its own width to the
    node, its rounded end stood out of both daughters as a ball."""
    node = np.zeros(3)
    angle = np.radians(40.0)
    parent = _resampled([-40.0, 0.0, 0.0], node)
    daughters = [
        _resampled(node, [40.0 * np.cos(angle), sign * 40.0 * np.sin(angle), 0.0])
        for sign in (1.0, -1.0)
    ]
    vectors, radii, owner = _network([parent, *daughters], [6.0, 2.5, 2.5])
    sides = 12
    vertices, faces, index = tubes_from_vectors(vectors, radius=radii, sides=sides, groups=owner)
    assert _edge_use(faces) == {2}
    # One rounded end per vessel end: three terminals, three at the junction.
    assert len(_poles(faces, sides)) == 6

    own = vertices[owner[index] == 0]
    assert np.hypot(own[:, 1], own[:, 2]).max() <= 6.0 + 1e-9
    past = own[own[:, 0] > 1e-9]
    assert np.linalg.norm(past, axis=1).max() <= 2.5 + 1e-9
    beyond = own[own[:, 0] > 0.1 * 6.0]
    inside = np.zeros(len(beyond), dtype=bool)
    for daughter in daughters:
        distance, _ = _nearest_on_polyline(beyond, daughter)
        inside |= distance <= 2.5 + 1e-9
    assert inside.all()

    rings = _ring_vertices(vertices, sides, tubes=3)
    centre, radius, _facing = _ring_frames(rings)
    mine = owner[index[: rings.shape[0] * sides : sides]] == 0
    profile = radius[mine][np.argsort(centre[mine, 0])]
    assert profile[0] == pytest.approx(6.0) and profile[-1] == pytest.approx(2.5)
    assert np.all(np.diff(profile) <= 1e-9)


def test_a_vessel_narrowing_at_a_node_tapers_without_overshoot():
    """6 µm into 3 µm through a node only the two meet: one tube whose width
    falls steadily from one to the other, not a step."""
    wide = _resampled([0.0, 0.0, 0.0], [0.0, 0.0, 20.0])
    narrow = _resampled([0.0, 0.0, 20.0], [0.0, 0.0, 40.0])
    vectors, radii, owner = _network([wide, narrow], [3.0, 1.5])
    sides = 12
    vertices, faces, _index = tubes_from_vectors(vectors, radius=radii, sides=sides, groups=owner)
    assert _edge_use(faces) == {2}
    assert _pieces(faces) == 1
    assert len(_poles(faces, sides)) == 2
    centre, radius, _facing = _ring_frames(_ring_vertices(vertices, sides))
    profile = radius[np.argsort(centre[:, 2])]
    assert profile[0] == pytest.approx(3.0) and profile[-1] == pytest.approx(1.5)
    assert profile.min() >= 1.5 - 1e-9 and profile.max() <= 3.0 + 1e-9
    assert np.all(np.diff(profile) <= 1e-9)
    assert np.abs(np.diff(profile)).max() < 0.25


def test_a_chain_of_short_vessels_is_mostly_tube_not_rounded_ends():
    """Ten vessels each half as long as they are wide, end to end: rounded
    off at every node, they were more ball than tube."""
    paths = [_resampled([0.0, 0.0, 3.0 * i], [0.0, 0.0, 3.0 * (i + 1)]) for i in range(10)]
    vectors, radii, owner = _network(paths, [3.0] * 10)
    sides = 12
    vertices, faces, _index = tubes_from_vectors(vectors, radius=radii, sides=sides, groups=owner)
    assert _edge_use(faces) == {2}
    assert len(_poles(faces, sides)) == 2
    assert _cap_volume_share(vertices, faces, sides) < 0.15


def test_a_short_fat_vessel_is_drawn_within_twice_its_neighbours():
    """A 2 µm long vessel measured 14 µm across between two of 6 µm: drawn
    no wider than 12 µm. Only drawn: the radius it was given is not changed."""
    paths = [
        _resampled([0.0, 0.0, 0.0], [0.0, 0.0, 20.0]),
        _resampled([0.0, 0.0, 20.0], [0.0, 0.0, 22.0]),
        _resampled([0.0, 0.0, 22.0], [0.0, 0.0, 42.0]),
    ]
    vectors, radii, owner = _network(paths, [3.0, 7.0, 3.0])
    given = radii.copy()
    vertices, faces, _index = tubes_from_vectors(vectors, radius=radii, sides=12, groups=owner)
    assert _edge_use(faces) == {2}
    assert np.hypot(vertices[:, 0], vertices[:, 1]).max() <= 6.0 + 1e-9
    np.testing.assert_array_equal(radii, given)


def test_a_zero_resistance_bridge_is_drawn_at_the_vessel_it_bridges():
    """The bridge from a 3 µm vessel into a 12 µm one carries the wide
    vessel's lumen diameter; it is drawn at the 3 µm vessel's."""
    thin = _resampled([0.0, 0.0, 0.0], [0.0, 0.0, 15.0])
    bridge = _resampled([0.0, 0.0, 15.0], [0.0, 0.0, 21.0])
    fat = [_resampled([0.0, -30.0, 21.0], [0.0, 0.0, 21.0]),
           _resampled([0.0, 0.0, 21.0], [0.0, 30.0, 21.0])]
    vectors, radii, owner = _network([thin, bridge, *fat], [1.5, 6.0, 6.0, 6.0])
    sides = 12
    vertices, faces, index = tubes_from_vectors(
        vectors, radius=radii, sides=sides, groups=owner, bridges=owner == 1
    )
    assert _edge_use(faces) == {2}
    rings = _ring_vertices(vertices, sides, tubes=3)
    centre, radius, _facing = _ring_frames(rings)
    ring_owner = owner[index[: rings.shape[0] * sides : sides]]
    np.testing.assert_allclose(radius[ring_owner == 1], 1.5)
    np.testing.assert_allclose(radius[ring_owner == 0], 1.5)
    for vessel in (2, 3):
        mine = np.flatnonzero(ring_owner == vessel)
        middle = mine[np.argmin(np.abs(np.abs(centre[mine, 1]) - 15.0))]
        assert radius[middle] == pytest.approx(6.0)

    _v, _f, unflagged = tubes_from_vectors(vectors, radius=radii, sides=sides, groups=owner)
    unflagged_vertices = _v[owner[unflagged] == 1]
    assert np.hypot(unflagged_vertices[:, 0], unflagged_vertices[:, 1]).max() > 1.5 + 0.5


def test_a_bend_tighter_than_the_tube_is_wide_does_not_fold():
    """A hairpin of 4 µm radius on a 12 µm vessel: no ring reaches through
    the next, and the tube still faces outward all the way round."""
    t = np.linspace(0.0, np.pi, 60)
    arc = np.stack([4.0 * np.cos(t), 4.0 * np.sin(t), np.zeros_like(t)], axis=1)
    path = np.vstack([
        _resampled([4.0, -20.0, 0.0], arc[0])[:-1], arc, _resampled(arc[-1], [-4.0, -20.0, 0.0])[1:],
    ])
    sides = 12
    vertices, faces, _index = tubes_from_vectors(_polyline_vectors(path), radius=6.0, sides=sides)
    assert _edge_use(faces) == {2}
    rings = _ring_vertices(vertices, sides)
    _assert_no_ring_reaches_through_the_next(rings)
    _assert_the_tube_faces_outward(vertices, faces, rings)
    np.testing.assert_allclose(rings[0].mean(axis=0), path[0], atol=1e-9)
    np.testing.assert_allclose(rings[-1].mean(axis=0), path[-1], atol=1e-9)


def test_an_end_snapped_off_its_path_does_not_fold():
    """Cluster collapse moved the node 5 µm off the vessel's path and the
    snap put the end there. The end stays on the node, facing the tube's
    own course rather than the jump onto it, and nothing folds."""
    path = _resampled([0.0, 5.0, 0.0], [40.0, 5.0, 0.0])
    path[0] = [0.0, 0.0, 0.0]
    sides = 12
    vertices, faces, _index = tubes_from_vectors(_polyline_vectors(path), radius=6.0, sides=sides)
    assert _edge_use(faces) == {2}
    rings = _ring_vertices(vertices, sides)
    _assert_no_ring_reaches_through_the_next(rings)
    _assert_the_tube_faces_outward(vertices, faces, rings)
    centre, _radius, facing = _ring_frames(rings)
    np.testing.assert_allclose(centre[0], path[0], atol=1e-9)
    two_along = (centre[2] - centre[0]) / np.linalg.norm(centre[2] - centre[0])
    assert facing[0] @ two_along == pytest.approx(1.0)


def test_a_vessels_layer_draws_each_vessel_at_its_own_diameter():
    """The Vectors layer's own ``diameter_um`` column, halved; without one,
    the layer's edge width."""
    from haemolynx.gui.vessel_tubes import vessel_tube_mesh

    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[20.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    def distances(vertices, index, row):
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        return _nearest_on_polyline(vertices[index == row], segment)[0]

    features = {"diameter_um": [4.0, np.nan, 8.5], "edge_index": [0, 1, 2]}
    vertices, _faces, index = vessel_tube_mesh(vectors, features)
    for row, radius in enumerate([2.0, TUBE_RADIUS_UM, 4.25]):
        np.testing.assert_allclose(distances(vertices, index, row), radius, atol=1e-9)
    for missing in ({}, None):
        vertices, _faces, index = vessel_tube_mesh(vectors, missing, edge_width=3.0)
        for row in range(3):
            np.testing.assert_allclose(distances(vertices, index, row), 3.0, atol=1e-9)


def test_a_vessels_layer_is_drawn_as_its_columns_say():
    """A napari features table: diameters, which steps are which vessel and
    which vessels are bridges, at the session's quality."""
    from haemolynx.gui.vessel_tubes import vessel_tube_mesh

    pandas = pytest.importorskip("pandas")
    thin = _resampled([0.0, 0.0, 0.0], [0.0, 0.0, 15.0])
    bridge = _resampled([0.0, 0.0, 15.0], [0.0, 0.0, 21.0])
    fat = _resampled([0.0, -30.0, 21.0], [0.0, 0.0, 21.0])
    vectors, radii, owner = _network([thin, bridge, fat, fat[::-1] * [1, -1, 1]], [1.5, 6.0, 6.0, 6.0])
    flags = owner == 1
    features = pandas.DataFrame(
        {"diameter_um": 2.0 * radii, "edge_index": owner, IS_ZERO_RESISTANCE: flags}
    )
    for quality in (0, len(TUBE_QUALITY_SIDES) - 1):
        got = vessel_tube_mesh(vectors, features, quality=quality)
        want = tube_mesh(vectors, radius=radii, quality=quality, groups=owner, bridges=flags)
        for got_array, want_array in zip(got, want):
            np.testing.assert_array_equal(got_array, want_array)


def test_uniform_draws_every_vessel_at_one_width_whatever_its_diameter():
    """The "Tube diameter" choice: Uniform leaves ``diameter_um`` out and
    draws each vessel as wide as one with no diameter yet; Per vessel, the
    default, is the drawing there was before the choice."""
    from haemolynx.gui.vessel_tubes import (
        TUBE_DIAMETER_PER_VESSEL,
        TUBE_DIAMETER_UNIFORM,
        vessel_tube_mesh,
    )

    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[20.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    def distances(vertices, index, row):
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        return _nearest_on_polyline(vertices[index == row], segment)[0]

    features = {"diameter_um": [4.0, 12.0, 8.5], "edge_index": [0, 1, 2]}
    vertices, _faces, index = vessel_tube_mesh(vectors, features, diameter=TUBE_DIAMETER_UNIFORM)
    for row in range(3):
        np.testing.assert_allclose(distances(vertices, index, row), TUBE_RADIUS_UM, atol=1e-9)
    vertices, _faces, index = vessel_tube_mesh(
        vectors, features, edge_width=3.0, diameter=TUBE_DIAMETER_UNIFORM)
    for row in range(3):
        np.testing.assert_allclose(distances(vertices, index, row), 3.0, atol=1e-9)

    uniform = vessel_tube_mesh(vectors, features, diameter=TUBE_DIAMETER_UNIFORM)
    for got, want in zip(uniform, vessel_tube_mesh(vectors, {"edge_index": [0, 1, 2]})):
        np.testing.assert_array_equal(got, want)
    per_vessel = vessel_tube_mesh(vectors, features, diameter=TUBE_DIAMETER_PER_VESSEL)
    for got, want in zip(per_vessel, vessel_tube_mesh(vectors, features)):
        np.testing.assert_array_equal(got, want)
    vertices, _faces, index = per_vessel
    np.testing.assert_allclose(distances(vertices, index, 1), 6.0, atol=1e-9)
    # The column itself is left as it was.
    assert features["diameter_um"] == [4.0, 12.0, 8.5]


def test_uniform_draws_every_vessel_at_the_width_it_is_given():
    """The µm box beside Uniform: every vessel that wide, whatever its own
    diameter or the layer's edge width; Per vessel takes no notice of it."""
    from haemolynx.gui.vessel_tubes import (
        TUBE_DIAMETER_PER_VESSEL,
        TUBE_DIAMETER_UNIFORM,
        UNIFORM_TUBE_DIAMETER_RANGE_UM,
        vessel_tube_mesh,
    )

    vectors = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[20.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    def distances(vertices, index, row):
        segment = np.stack([vectors[row, 0], vectors[row, 0] + vectors[row, 1]])
        return _nearest_on_polyline(vertices[index == row], segment)[0]

    features = {"diameter_um": [4.0, 12.0, 8.5], "edge_index": [0, 1, 2]}
    for width in (10.0, 1.0):
        vertices, _faces, index = vessel_tube_mesh(
            vectors, features, edge_width=3.0,
            diameter=TUBE_DIAMETER_UNIFORM, uniform_diameter_um=width,
        )
        for row in range(3):
            np.testing.assert_allclose(distances(vertices, index, row), width / 2, atol=1e-9)
    # Held to the box's range.
    vertices, _faces, index = vessel_tube_mesh(
        vectors, features, diameter=TUBE_DIAMETER_UNIFORM, uniform_diameter_um=1e6)
    np.testing.assert_allclose(
        distances(vertices, index, 0), UNIFORM_TUBE_DIAMETER_RANGE_UM[1] / 2, atol=1e-6)

    per_vessel = vessel_tube_mesh(
        vectors, features, diameter=TUBE_DIAMETER_PER_VESSEL, uniform_diameter_um=10.0)
    for got, want in zip(per_vessel, vessel_tube_mesh(vectors, features)):
        np.testing.assert_array_equal(got, want)


def test_a_uniform_width_that_is_not_a_number_is_the_default():
    from haemolynx.gui.vessel_tubes import (
        DEFAULT_UNIFORM_TUBE_DIAMETER_UM,
        UNIFORM_TUBE_DIAMETER_RANGE_UM,
        valid_uniform_tube_diameter_um,
    )

    low, high = UNIFORM_TUBE_DIAMETER_RANGE_UM
    # To start, as wide as a vessel with no diameter yet is drawn.
    assert DEFAULT_UNIFORM_TUBE_DIAMETER_UM == 2 * TUBE_RADIUS_UM
    assert low <= DEFAULT_UNIFORM_TUBE_DIAMETER_UM <= high
    assert valid_uniform_tube_diameter_um(25.0) == 25.0
    assert valid_uniform_tube_diameter_um("25") == 25.0
    assert valid_uniform_tube_diameter_um(0.0) == low
    assert valid_uniform_tube_diameter_um(-3.0) == low
    assert valid_uniform_tube_diameter_um(high * 10) == high
    for odd in (None, "wide", np.nan, np.inf):
        assert valid_uniform_tube_diameter_um(odd) == DEFAULT_UNIFORM_TUBE_DIAMETER_UM


def test_a_diameter_choice_it_does_not_know_is_per_vessel():
    from haemolynx.gui.vessel_tubes import (
        DEFAULT_TUBE_DIAMETER,
        TUBE_DIAMETER_LABELS,
        TUBE_DIAMETER_PER_VESSEL,
        TUBE_DIAMETER_UNIFORM,
        TUBE_DIAMETERS,
        valid_tube_diameter,
    )

    assert DEFAULT_TUBE_DIAMETER == TUBE_DIAMETER_PER_VESSEL
    assert set(TUBE_DIAMETER_LABELS) == set(TUBE_DIAMETERS)
    for choice in TUBE_DIAMETERS:
        assert valid_tube_diameter(choice) == choice
    # A label is not a choice.
    for odd in (None, "", 3, TUBE_DIAMETER_LABELS[TUBE_DIAMETER_UNIFORM]):
        assert valid_tube_diameter(odd) == TUBE_DIAMETER_PER_VESSEL


def test_the_tooltip_gives_the_width_uniform_tubes_start_at():
    from haemolynx.gui.chrome_tooltips import TUBE_DIAMETER_TOOLTIP
    from haemolynx.gui.vessel_tubes import DEFAULT_UNIFORM_TUBE_DIAMETER_UM

    assert f"{DEFAULT_UNIFORM_TUBE_DIAMETER_UM:g} µm to start" in TUBE_DIAMETER_TOOLTIP


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
def test_a_network_of_junctions_builds_fast():
    """A lattice of 3-way to 6-way junctions, about 30,000 vessels: every
    node's radius and every tube's neighbours worked out at once."""
    import time

    n = 22
    grid = np.stack(np.meshgrid(*[np.arange(n) * 12.0] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.arange(n ** 3).reshape(n, n, n)
    pairs = np.concatenate([
        np.stack([index[:-1].ravel(), index[1:].ravel()], axis=1),
        np.stack([index[:, :-1].ravel(), index[:, 1:].ravel()], axis=1),
        np.stack([index[:, :, :-1].ravel(), index[:, :, 1:].ravel()], axis=1),
    ])
    rng = np.random.default_rng(2)
    paths = [_resampled(grid[a], grid[b], step=1.0) for a, b in pairs]
    vectors, radii, owner = _network(paths, rng.uniform(1.0, 4.0, len(paths)))

    started = time.perf_counter()
    vertices, faces, index = tube_mesh(vectors, radius=radii, groups=owner)
    elapsed = time.perf_counter() - started

    assert set(np.unique(owner[index]).tolist()) == set(range(len(paths)))
    assert faces.max() < len(vertices)
    assert elapsed < 10.0, f"tube_mesh took {elapsed:.2f}s for {len(paths)} vessels"


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
