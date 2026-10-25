"""Small synthetic masks and graphs, one per kind of dead end, each with its
answer known (PLAN_dead_ends.md, Phase 0).

Voxels are 1 um on every axis, so voxel indices and physical (z, y, x)
microns are the same numbers. Every case is the same 20 x 60 x 80 um volume
with a capillary of radius 2.5 um running along x through (z, y) = (10, 30)
from one x face to the other, a junction on it at x = 40, and one more vessel
hanging off that junction whose free end is the dead end being judged.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from lumen_artefact_fixtures import VOXEL_SIZE, capsule, polyline_graph

SHAPE = (20, 60, 80)
RADIUS = 2.5
JUNCTION = (10.0, 30.0, 40.0)
#: The capillary the dead ends hang off: both its ends leave the image.
TRUNK = ((10.0, 30.0, 0.0), JUNCTION, (10.0, 30.0, 79.0))


@dataclass(frozen=True)
class Case:
    name: str
    mask: np.ndarray
    graph: nx.MultiGraph
    #: The dead end judged, and what it is.
    tip: tuple
    kind: str
    why: str


def _trunk_mask() -> np.ndarray:
    return capsule(np.zeros(SHAPE, dtype=bool), (10, 30, -5), (10, 30, 85), RADIUS)


def _with_branch(branch_path, mask_to=None, *, extra_paths=(), extra_mask=()):
    """The trunk plus one branch from the junction along *branch_path*; the
    branch's mask runs to *mask_to* (its last point when None, no mask when
    False)."""
    mask = _trunk_mask()
    if mask_to is not False:
        end = branch_path[-1] if mask_to is None else mask_to
        capsule(mask, JUNCTION, end, RADIUS)
    for a, b in extra_mask:
        capsule(mask, a, b, RADIUS)
    G = polyline_graph([[TRUNK[0], JUNCTION], [JUNCTION, TRUNK[2]], [JUNCTION, *branch_path], *extra_paths])
    G.graph["image_voxel_size_zyx"] = VOXEL_SIZE
    return mask, G


def _node(point) -> tuple:
    return tuple(float(round(c, 3)) for c in point)


def leaves_through_face() -> Case:
    """A vessel running out through the y face: its tip lies on the face."""
    mask, G = _with_branch([(10, 59, 40)], mask_to=(10, 70, 40))
    return Case("leaves_through_face", mask, G, _node((10, 59, 40)), "leaves_image",
                "the tip touches the y face")


def stops_short_of_face() -> Case:
    """The graph stops 8 um short of the y face, but the tube runs on to it."""
    mask, G = _with_branch([(10, 51, 40)], mask_to=(10, 70, 40))
    return Case("stops_short_of_face", mask, G, _node((10, 51, 40)), "leaves_image",
                "the mask runs on from the tip to the face, within four radii")


def interior_near_face() -> Case:
    """A blind end 8 um from the y face whose tube does not reach it."""
    mask, G = _with_branch([(10, 51, 40)])
    return Case("interior_near_face", mask, G, _node((10, 51, 40)), "unknown",
                "near a face, but the mask stops at the tip: not a vessel the image cut")


def mask_runs_on() -> Case:
    """The graph stops 13 um before the segmented vessel does, in the tissue."""
    mask, G = _with_branch([(10, 18, 40)], mask_to=(10, 5, 40))
    return Case("mask_runs_on", mask, G, _node((10, 18, 40)), "stops_short",
                "the segmented vessel runs on past the tip")


def broken_towards_side() -> Case:
    """A branch ending 9 um short of another vessel's side, across a gap in
    the segmentation: the other vessel is not one of its neighbours."""
    other = [(10, 10, 0), (10, 10, 79)]
    mask, G = _with_branch(
        [(10, 19, 40)], extra_paths=[other], extra_mask=[((10, 10, -5), (10, 10, 85))],
    )
    return Case("broken_towards_side", mask, G, _node((10, 19, 40)), "broken",
                "another vessel lies straight ahead across one stretch of background")


def facing_pair() -> Case:
    """Fixture 5: two dead ends heading straight for each other across 6 um of
    background -- the branch off the junction, and one off a second vessel
    along x at y = 56 -- with the segmentation broken between them."""
    other = [[(10, 56, 0), (10, 56, 40)], [(10, 56, 40), (10, 56, 79)], [(10, 56, 40), (10, 46, 40)]]
    mask, G = _with_branch(
        [(10, 40, 40)], mask_to=(10, 37.5, 40), extra_paths=other,
        extra_mask=[((10, 56, -5), (10, 56, 85)), ((10, 56, 40), (10, 48.5, 40))],
    )
    return Case("facing_pair", mask, G, _node((10, 40, 40)), "broken",
                "another dead end faces it across one 6 um stretch of background")


def short_spur() -> Case:
    """A blind stub 5 um long off a capillary of radius 2.5: under 3 radii."""
    mask, G = _with_branch([(10, 35, 40)])
    return Case("short_spur", mask, G, _node((10, 35, 40)), "artefact",
                "shorter than the stub prune's three radii")


def off_the_mask() -> Case:
    """A 20 um branch with no segmented vessel under it."""
    mask, G = _with_branch([(10, 30, 40), (10, 45, 52)][1:], mask_to=False)
    return Case("off_the_mask", mask, G, _node((10, 45, 52)), "artefact",
                "less than half of it lies in the mask")


def blind_sprout() -> Case:
    """A blind end 4 radii long in the tissue, nothing ahead: a sprout or a
    broken vessel, which only a person (or marks) can say."""
    mask, G = _with_branch([(10, 20, 40)])
    return Case("blind_sprout", mask, G, _node((10, 20, 40)), "unknown",
                "an interior blind end with no evidence either way")


def tip_near_own_junction() -> Case:
    """Fixture 10: a stub whose tip lies 4 um from the junction it hangs from,
    within cluster collapse's 5 um, which merges the tip into its junction
    before any rule judges it."""
    mask, G = _with_branch([(10, 34, 40)])
    return Case("tip_near_own_junction", mask, G, _node((10, 34, 40)), "artefact",
                "4 um long: under the stub prune's three radii")


def island() -> Case:
    """A two-edge piece of its own, away from every face."""
    mask, G = _with_branch(
        [(10, 35, 40)],
        extra_paths=[[(10, 50, 10), (10, 50, 20)], [(10, 50, 20), (10, 50, 30)]],
        extra_mask=[((10, 50, 10), (10, 50, 30))],
    )
    return Case("island", mask, G, _node((10, 50, 10)), "island_end",
                "an end of a piece with no junction")


#: A thick vessel along x at (z, y) = (10, 7), 6 um in radius (its wall at
#: y = 13), both its ends leaving the image, and the tip of
#: :func:`at_a_thick_wall`'s branch, 12 um from its centreline.
THICK_AXIS = ((10.0, 7.0, 0.0), (10.0, 7.0, 79.0))
THICK_RADIUS = 6.0
THICK_WALL_TIP = (10.0, 19.0, 40.0)


def at_a_thick_wall(*, recorded: bool = True) -> Case:
    """An 11 um branch heading straight for a thick vessel and stopping 6 um
    short of its lumen across background: within the wall join's reach, off
    the thick centreline, not joined. Its kind comes from the thick
    centreline the build records on the graph; without the record it is a
    blind end (the thick centreline lies past the 10 um a vessel ahead is
    looked for)."""
    from haemolynx.graph.dead_ends import record_thick_centreline

    mask, G = _with_branch([THICK_WALL_TIP], extra_paths=[list(THICK_AXIS)])
    capsule(mask, (10, 7, -5), (10, 7, 85), THICK_RADIUS)
    if recorded:
        points = np.array([[10.0, 7.0, x] for x in range(80)])
        record_thick_centreline(G, points, np.full(len(points), THICK_RADIUS), tolerance_um=1.5)
    return Case("at_a_thick_wall", mask, G, _node(THICK_WALL_TIP), "thick_wall" if recorded else "unknown",
                "it stops within reach of a thick vessel's lumen, off its centreline")


CASES = (
    leaves_through_face,
    stops_short_of_face,
    interior_near_face,
    mask_runs_on,
    broken_towards_side,
    short_spur,
    off_the_mask,
    blind_sprout,
    island,
    facing_pair,
    tip_near_own_junction,
    at_a_thick_wall,
)
