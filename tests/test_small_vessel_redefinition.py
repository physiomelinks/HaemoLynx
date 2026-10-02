"""Tests for graph/small_vessel_redefinition.py.

Tangential redefinition gives a small-vessel piece the class of the large vessel
it runs alongside; the sandwiched pass flips a piece lying in line between two
pieces of the other class. Several of these tests pin failures of the version
this module replaced: its tangency test never rejected anything (a fallback
projected the normal off the vessel's own axis), it measured directions in
voxel indices, it left a split piece split when its majority already matched,
it relabelled a whole tree on one contact, and it compared a sandwiched piece's
ends only with its neighbours' two extreme voxels.
"""
from __future__ import annotations

import logging

import numpy as np
import networkx as nx
import pytest

from haemolynx.graph import redefine_small_masks_from_large_tangential_contact
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import (
    SkeletonisedVolume,
    VesselNetwork,
    assign_boundaries,
)


def _tube(shape, start, end, radius, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Voxels whose centres lie within *radius* microns of a segment (microns, z y x)."""
    centres = np.stack(np.indices(shape), axis=-1) * np.asarray(spacing, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    along = np.clip(((centres - a) @ (b - a)) / float((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(centres - (a + along[..., None] * (b - a)), axis=-1) <= radius


def _redefine(small_arteriole, small_venule, large_arteriole=None, large_venule=None,
              spacing=(1.0, 1.0, 1.0), **options):
    empty = np.zeros_like(small_arteriole, dtype=bool)
    return redefine_small_masks_from_large_tangential_contact(
        small_arteriole_mask=small_arteriole,
        small_venule_mask=small_venule,
        large_arteriole_mask=large_arteriole if large_arteriole is not None else empty,
        large_venule_mask=large_venule if large_venule is not None else empty,
        voxel_size_zyx=spacing,
        **options,
    )


SHAPE = (24, 30, 60)
#: A large arteriole along x; its wall is at y = 15.
LARGE_ARTERIOLE = _tube(SHAPE, (12, 10, 0), (12, 10, 59), 5.0)


def _alongside() -> np.ndarray:
    """A small vessel lying along the large arteriole, 1 um from its wall."""
    return _tube(SHAPE, (12, 18, 5), (12, 18, 25), 2.0)


def _end_on() -> np.ndarray:
    """A small vessel ending against the large arteriole's side, pointing into it."""
    return _tube(SHAPE, (12, 16, 42), (12, 28, 42), 2.0)


# --- tangential contact ---------------------------------------------------------


def test_a_small_vessel_running_alongside_a_large_one_takes_its_class() -> None:
    small_venule = _alongside()
    result = _redefine(np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE)

    assert np.array_equal(result["small_arteriole_mask"], small_venule)
    assert not np.any(result["small_venule_mask"])
    stats = result["stats"]
    assert stats["reassigned_to_arteriole"] == 1
    (change,) = stats["changes"]
    assert change["to"] == "arteriole" and change["arteriole_share_before"] == 0.0
    assert change["tangency_cosine"] < 0.1


def test_a_small_vessel_meeting_a_large_one_end_on_keeps_its_class() -> None:
    """The old tangency test never rejected: a failing contact was re-measured
    against the normal with its along-axis part removed, which always passes."""
    small_venule = _end_on()
    result = _redefine(np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE)

    assert np.array_equal(result["small_venule_mask"], small_venule)
    assert not np.any(result["small_arteriole_mask"])
    assert result["stats"]["not_tangential"] == 1
    assert result["stats"]["reassigned_to_arteriole"] == 0


def test_a_looser_tangency_limit_lets_an_end_on_contact_through() -> None:
    """The setting that had no effect now decides the end-on case."""
    small_venule = _end_on()
    result = _redefine(
        np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE, tangency_cosine_max=1.0
    )

    assert np.array_equal(result["small_arteriole_mask"], small_venule)


def test_directions_are_measured_in_microns_not_voxel_indices() -> None:
    """A small vessel alongside a large one, both tilted 45 degrees in z-x, in a
    stack sampled three times coarser in z.

    In microns the small vessel lies along the large one's wall. Measured in
    voxel indices its direction leans towards x, far enough from the wall
    to fail the tangency limit -- which is what the old code did.
    """
    spacing = (3.0, 1.0, 1.0)
    shape = (24, 30, 64)
    axis = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    normal = np.asarray([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    large_start = np.asarray([6.0, 15.0, 0.0])
    large_arteriole = _tube(shape, large_start, large_start + 60 * np.sqrt(2) * axis, 6.0, spacing)
    small_start = large_start + 10 * np.sqrt(2) * axis + 9.0 * normal
    small_venule = _tube(shape, small_start, small_start + 42 * np.sqrt(2) * axis, 2.5, spacing)
    assert not np.any(small_venule & large_arteriole)

    voxel_axis = np.linalg.eigh(np.cov(np.argwhere(small_venule).T.astype(float)))[1][:, -1]
    assert abs(float(voxel_axis @ normal)) > 0.35  # the voxel-index reading fails

    result = _redefine(
        np.zeros(shape, bool), small_venule, large_arteriole, spacing=spacing
    )

    assert np.array_equal(result["small_arteriole_mask"], small_venule)
    assert result["stats"]["changes"][0]["tangency_cosine"] < 0.35


def test_a_split_piece_whose_majority_already_matches_is_made_one_class() -> None:
    """The old code returned "keep" when the large vessel matched the piece's
    majority, leaving its minority voxels in the other class."""
    piece = _alongside()
    x = np.indices(SHAPE)[2]
    small_arteriole = piece & (x < 19)
    small_venule = piece & (x >= 19)

    result = _redefine(small_arteriole, small_venule, LARGE_ARTERIOLE)

    assert np.array_equal(result["small_arteriole_mask"], piece)
    assert not np.any(result["small_venule_mask"])
    (change,) = result["stats"]["changes"]
    assert 0.6 < change["arteriole_share_before"] < 0.8


def test_a_piece_already_of_that_class_is_counted_not_changed() -> None:
    piece = _alongside()
    result = _redefine(piece, np.zeros(SHAPE, bool), LARGE_ARTERIOLE)

    assert np.array_equal(result["small_arteriole_mask"], piece)
    assert result["stats"]["already_that_class"] == 1
    assert result["stats"]["changes"] == []


def _diverging_vessel(shape):
    """Touches the large vessel over its first ~15 um, then veers away over ~90 um."""
    return _tube(shape, (15, 22, 0), (15, 22, 10), 2.0) | _tube(
        shape, (15, 22, 10), (15, 40, 99), 2.0
    )


def test_one_short_contact_does_not_relabel_a_long_vessel() -> None:
    shape = (30, 60, 100)
    large_arteriole = _tube(shape, (15, 15, 0), (15, 15, 99), 5.0)
    small_venule = _diverging_vessel(shape)

    kept = _redefine(np.zeros(shape, bool), small_venule, large_arteriole)
    relabelled = _redefine(
        np.zeros(shape, bool), small_venule, large_arteriole, min_contact_fraction=0.0
    )

    assert np.array_equal(kept["small_venule_mask"], small_venule)
    assert kept["stats"]["too_little_contact"] == 1
    assert np.array_equal(relabelled["small_arteriole_mask"], small_venule)
    assert relabelled["stats"]["changes"][0]["contact_fraction"] < 0.15


def test_a_piece_alongside_both_classes_equally_is_left_alone() -> None:
    shape = (24, 40, 60)
    large_arteriole = _tube(shape, (12, 11, 0), (12, 11, 59), 5.0)
    large_venule = _tube(shape, (12, 29, 0), (12, 29, 59), 5.0)
    small_venule = _tube(shape, (12, 20, 5), (12, 20, 50), 2.0)

    result = _redefine(np.zeros(shape, bool), small_venule, large_arteriole, large_venule)

    assert np.array_equal(result["small_venule_mask"], small_venule)
    assert result["stats"]["ambiguous"] == 1


def test_a_piece_far_from_every_large_vessel_is_left_alone() -> None:
    small_venule = _tube(SHAPE, (12, 28, 5), (12, 28, 50), 1.5)
    result = _redefine(np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE)

    assert np.array_equal(result["small_venule_mask"], small_venule)
    assert result["stats"]["no_contact"] == 1


def test_the_run_log_names_each_change_and_why_the_rest_were_left(caplog) -> None:
    small_venule = _alongside() | _end_on()
    far = _tube(SHAPE, (22, 28, 5), (22, 28, 50), 1.0)

    with caplog.at_level(logging.INFO, logger="haemolynx.graph.small_vessel_redefinition"):
        result = _redefine(np.zeros(SHAPE, bool), small_venule | far, LARGE_ARTERIOLE)

    stats = result["stats"]
    assert (stats["reassigned_to_arteriole"], stats["not_tangential"], stats["no_contact"]) == (1, 1, 1)
    assert "(0% arteriole) -> arteriole" in caplog.text
    assert "1 meeting one end-on" in caplog.text
    assert "1 touching no large vessel" in caplog.text


def test_disabled_returns_the_masks_unchanged() -> None:
    small_venule = _alongside()
    result = _redefine(
        np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE, enable_redefinition=False
    )
    assert np.array_equal(result["small_venule_mask"], small_venule)
    assert result["stats"]["redefinition_enabled"] is False


def test_worker_threads_give_the_same_answer() -> None:
    small_venule = _alongside() | _end_on()
    serial = _redefine(np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE)
    threaded = _redefine(
        np.zeros(SHAPE, bool), small_venule, LARGE_ARTERIOLE, reassignment_parallel_workers=4
    )
    assert np.array_equal(serial["small_arteriole_mask"], threaded["small_arteriole_mask"])
    assert np.array_equal(serial["small_venule_mask"], threaded["small_venule_mask"])


@pytest.mark.parametrize("name", ["tangency_cosine_max", "min_contact_fraction"])
def test_out_of_range_fractions_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        _redefine(np.zeros(SHAPE, bool), _alongside(), LARGE_ARTERIOLE, **{name: 1.5})


# --- sandwiched pieces ------------------------------------------------------------

LINE = (32, 24, 70)


def _rod(x0: int, x1: int, radius: float = 1.4) -> np.ndarray:
    return _tube(LINE, (12, 12, x0), (12, 12, x1), radius)


def test_no_large_masks_skips_tangential_contact_but_flips_a_sandwiched_piece() -> None:
    small_venule = _rod(3, 9) | _rod(17, 24)
    middle = _rod(10, 16, 1.3)

    result = redefine_small_masks_from_large_tangential_contact(
        small_arteriole_mask=middle,
        small_venule_mask=small_venule,
        large_arteriole_mask=None,
        large_venule_mask=None,
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )

    assert result["stats"]["used_large_mask_tangency"] is False
    assert result["stats"]["sandwiched_flips_to_venule"] == 1
    assert np.array_equal(result["small_venule_mask"], small_venule | middle)
    assert not np.any(result["small_arteriole_mask"])


def test_a_piece_ending_side_on_against_a_vessel_is_not_sandwiched() -> None:
    left = _rod(2, 9)
    middle = _rod(10, 20)
    across = _tube(LINE, (12, 2, 22), (12, 22, 22), 1.4)  # runs along y, past middle's end

    result = _redefine(middle, left | across)

    assert np.array_equal(result["small_arteriole_mask"], middle)
    assert result["stats"]["sandwiched_flips_to_venule"] == 0


def test_a_piece_larger_than_its_two_neighbours_together_is_not_flipped() -> None:
    long_piece = _rod(10, 50)
    specks = _rod(0, 8) | _rod(52, 60)

    result = _redefine(long_piece, specks)

    assert np.array_equal(result["small_arteriole_mask"], long_piece)
    assert result["stats"]["sandwiched_flips_to_venule"] == 0


def test_a_neighbour_that_bends_away_still_counts_where_it_meets_the_piece() -> None:
    """An L-shaped neighbour: judged by its own direction near the joint, not
    by its overall direction or its two extreme voxels, which here are the far
    ends of its legs."""
    shape = (60, 24, 70)

    def rod(start, end):
        return _tube(shape, start, end, 1.4)

    left = rod((12, 12, 5), (12, 12, 19)) | rod((12, 12, 5), (58, 12, 5))
    middle = rod((12, 12, 21), (12, 12, 30))
    right = rod((12, 12, 32), (12, 12, 45))

    result = _redefine(middle, left | right)

    assert np.array_equal(result["small_venule_mask"], left | middle | right)
    assert result["stats"]["sandwiched_flips_to_venule"] == 1


def test_a_chain_of_alternating_pieces_settles_into_consistent_classes() -> None:
    """Largest first: the middle venule joins the arterioles either side, and
    the small arteriole beyond it then has no venule on that side to flip it."""
    big_arteriole = _rod(2, 20)
    middle_venule = _rod(22, 31)
    small_arteriole = _rod(33, 37)
    big_venule = _rod(39, 65)

    result = _redefine(big_arteriole | small_arteriole, middle_venule | big_venule)

    assert np.array_equal(
        result["small_arteriole_mask"], big_arteriole | middle_venule | small_arteriole
    )
    assert np.array_equal(result["small_venule_mask"], big_venule)


def test_the_sandwiched_pass_can_be_switched_off() -> None:
    small_venule = _rod(3, 9) | _rod(17, 24)
    middle = _rod(10, 16, 1.3)

    result = _redefine(middle, small_venule, enable_sandwiched_component_reassignment=False)

    assert np.array_equal(result["small_arteriole_mask"], middle)


# --- settings and the stage ---------------------------------------------------------

TANGENTIAL_GATE = (
    "use_small_vessel_masks_for_boundary_assignment",
    "automated_vessel_assignment",
    "small_vessel_tangential_redefinition_enable",
)


def test_the_new_settings_sit_under_tangential_redefinition() -> None:
    schema = default_schema()
    assert schema["small_vessel_tangential_redefinition_min_contact_fraction"].default == 0.15
    assert schema["small_vessel_tangential_redefinition_min_contact_fraction"].requires == TANGENTIAL_GATE
    assert schema["small_vessel_sandwiched_reassignment_enable"].default is True
    assert schema["small_vessel_sandwiched_reassignment_enable"].requires == TANGENTIAL_GATE
    for name, default in (
        ("small_vessel_sandwiched_max_gap_microns", 12.0),
        ("small_vessel_sandwiched_min_facing_cosine", 0.82),
        ("small_vessel_sandwiched_max_axis_angle_degrees", 45.0),
    ):
        assert schema[name].default == default
        assert schema[name].requires == TANGENTIAL_GATE + ("small_vessel_sandwiched_reassignment_enable",)


def test_assign_boundaries_passes_the_settings_and_shows_the_relabelled_masks(
    tmp_path, monkeypatch
) -> None:
    """The stage hands every setting over, and BoundaryNodes carries the small
    masks boundary labelling read, so the viewer shows the relabelled ones."""
    from haemolynx import graph as graph_module

    shape = (10, 10, 10)
    G = nx.MultiGraph()
    for z in range(shape[0]):
        G.add_node(z, pos=(float(z), 5.0, 5.0))
    for z in range(shape[0] - 1):
        G.add_edge(
            z, z + 1, voxels=[(float(z), 5.0, 5.0), (float(z + 1), 5.0, 5.0)], length=1.0
        )
    small_arteriole = np.zeros(shape, dtype=bool)
    small_arteriole[0:3, 4:7, 4:7] = True
    small_venule = np.zeros(shape, dtype=bool)
    small_venule[7:10, 4:7, 4:7] = True
    relabelled = np.zeros(shape, dtype=bool)
    relabelled[0:5, 4:7, 4:7] = True
    received = {}

    def spy(**kwargs):
        received.update(kwargs)
        return {
            "small_arteriole_mask": relabelled,
            "small_venule_mask": small_venule,
            "stats": {},
        }

    monkeypatch.setattr(graph_module, "redefine_small_masks_from_large_tangential_contact", spy)
    large_arteriole = np.zeros(shape, dtype=bool)
    large_arteriole[0:2, 4:6, 4:6] = True  # touches the z=0 face: the inlet stump
    large_venule = np.zeros(shape, dtype=bool)
    large_venule[8:10, 4:6, 4:6] = True
    settings = default_schema().defaults()
    settings.update(
        {
            "plot_dir": tmp_path,
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "use_thick_vessel_skeletonisation": True,
            "cut_network_at_large_vessel_volumes": False,
            "assign_large_vessel_branch_orders": True,
            "automated_vessel_assignment_fast_mode": False,
            "automated_vessel_assignment_enable_overlap_cleanup": False,
            "automated_vessel_assignment_use_legacy_mode": True,
            "large_vessel_assignment_max_dilation_microns": 0.0,
            "write_fast_mode_preassignment_large_vessel_debug_3d_html": False,
            "boundary_handling": "leave_unsolved",
            "use_small_vessel_masks_for_boundary_assignment": True,
            "small_vessel_tangential_redefinition_enable": True,
            "small_vessel_tangential_redefinition_min_contact_fraction": 0.4,
            "small_vessel_sandwiched_reassignment_enable": False,
            "small_vessel_sandwiched_max_gap_microns": 7.0,
            "small_vessel_sandwiched_min_facing_cosine": 0.5,
            "small_vessel_sandwiched_max_axis_angle_degrees": 30.0,
            "write_small_vessel_boundary_labelling_3d_html": False,
        }
    )
    for role in (
        "inlet_nodes", "outlet_nodes", "arteriole_boundary_nodes", "venule_boundary_nodes",
        "large_arteriole_boundary_nodes", "large_venule_boundary_nodes",
        "arteriole_boundary_node_coordinates", "venule_boundary_node_coordinates",
        "arteriole_boundary_node_volumes", "venule_boundary_node_volumes",
        "inlet_node_coordinates", "outlet_node_coordinates",
    ):
        settings[role] = []
    image = np.zeros(shape, dtype=np.uint8)
    network = VesselNetwork(
        graph=G,
        volume=SkeletonisedVolume(
            image=image,
            skeleton=image.copy(),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            voxel_size_zyx=(1.0, 1.0, 1.0),
            output_dir=tmp_path,
        ),
        large_arteriole_mask=large_arteriole,
        large_venule_mask=large_venule,
        small_arteriole_mask=small_arteriole,
        small_venule_mask=small_venule,
    )

    boundaries = assign_boundaries(settings, network)

    assert received["min_contact_fraction"] == 0.4
    assert received["enable_sandwiched_component_reassignment"] is False
    assert received["sandwiched_max_endpoint_distance_microns"] == 7.0
    assert received["sandwiched_min_facing_cosine"] == 0.5
    assert received["sandwiched_max_axis_angle_degrees"] == 30.0
    assert np.array_equal(boundaries.small_arteriole_mask, relabelled)
    assert np.array_equal(boundaries.small_venule_mask, small_venule)
