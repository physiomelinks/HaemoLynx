"""Relabel small-vessel mask pieces from how they meet the large vessels and each other.

:func:`redefine_small_masks_from_large_tangential_contact` runs two passes during
boundary assignment (``small_vessel_tangential_redefinition_enable``):

1. **Tangential contact.** A small-vessel piece that runs alongside a large vessel
   takes that vessel's class, as a whole. The part of the piece within
   ``max_contact_distance_microns`` of the large mask is examined: it has to come
   within ``touch_distance_microns`` of it, and its direction has to lie along the
   large vessel's surface there (``tangency_cosine_max``) rather than point into
   it -- a small vessel meeting a large one end-on is a branch of whichever class
   it is, not a fringe of the large vessel. Enough of the piece has to touch
   (``min_contact_fraction``) that one contact cannot relabel a whole tree.
2. **Sandwiched pieces.** A piece lying in line between two pieces of the other
   class, one beyond each of its ends, takes their class when it is smaller than
   the two together.

A piece in the first pass is a connected component of the two small masks
together, so a piece split between the classes comes out as one class. All
geometry is in microns: a direction measured in voxel indices leans towards the
finely sampled axes of an anisotropic stack (by 15 degrees for a typical vessel
on a 0.59 x 0.59 x 2 um stack), which is enough to turn "alongside" into "end-on".
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, find_objects, label
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

_FULL_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
_FACE_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_FACE_STRUCTURE[1, 1, :] = _FACE_STRUCTURE[1, :, 1] = _FACE_STRUCTURE[:, 1, 1] = True
_CLASSES = ("arteriole", "venule")
_OPPOSITE = {"arteriole": "venule", "venule": "arteriole"}

#: A piece has a direction only when its spread along it is at least this many
#: times its spread across it (variances, so about 1.2 diameters long for a
#: tube). A stub no longer than it is wide points nowhere in particular, and
#: judging "alongside" from it would be a coin toss.
_MIN_ELONGATION = 2.0

#: Changed pieces named one per line in the run log; the rest are counted.
_MAX_LOGGED_CHANGES = 20


def redefine_small_masks_from_large_tangential_contact(
    *,
    small_arteriole_mask: np.ndarray,
    small_venule_mask: np.ndarray,
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    voxel_size_zyx: tuple[float, float, float],
    enable_redefinition: bool = True,
    max_contact_distance_microns: float = 12.0,
    touch_distance_microns: float = 3.0,
    tangency_cosine_max: float = 0.35,
    reassignment_margin: float = 0.10,
    min_contact_fraction: float = 0.15,
    opposite_exclusion_distance_microns: float = 3.0,
    tangency_weight: float = 4.0,
    opposite_penalty_weight: float = 4.0,
    enable_sandwiched_component_reassignment: bool = True,
    sandwiched_max_endpoint_distance_microns: float = 12.0,
    sandwiched_min_facing_cosine: float = 0.82,
    sandwiched_max_axis_angle_degrees: float = 45.0,
    reassignment_parallel_workers: int = 0,
) -> dict[str, Any]:
    """Relabel small-vessel pieces by tangential contact, then sandwiched pieces.

    Returns the relabelled ``small_arteriole_mask`` / ``small_venule_mask`` and
    ``stats``: how many pieces each pass changed, why the others were left
    (``no_contact``, ``not_tangential``, ``too_little_contact``, ``ambiguous``,
    ``already_that_class``), and one record per change (``changes``,
    ``sandwiched_flips``). Pieces neither pass changes keep their voxels as
    they were, overlap included.

    When both classes of large vessel qualify, the lower score wins by at least
    ``reassignment_margin``; a contact's score is its distance (um) plus
    ``tangency_weight`` times its tangency cosine plus
    ``opposite_penalty_weight`` times how far inside
    ``opposite_exclusion_distance_microns`` of the other class it comes.
    """
    art = np.asarray(small_arteriole_mask).astype(bool, copy=False)
    ven = np.asarray(small_venule_mask).astype(bool, copy=False)
    if art.shape != ven.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {art.shape} and {ven.shape}."
        )
    stats: dict[str, Any] = {"redefinition_enabled": bool(enable_redefinition)}
    if not enable_redefinition:
        stats.update(_empty_tangential_stats(0), **_empty_sandwich_stats())
        return {
            "small_arteriole_mask": art.copy(),
            "small_venule_mask": ven.copy(),
            "stats": stats,
        }
    large = {"arteriole": large_arteriole_mask, "venule": large_venule_mask}
    for name, mask in large.items():
        if mask is not None and np.shape(mask) != art.shape:
            raise ValueError(
                f"large_{name}_mask must share the small masks' shape {art.shape}; "
                f"got {np.shape(mask)}."
            )
    for name, value in (
        ("tangency_cosine_max", tangency_cosine_max),
        ("min_contact_fraction", min_contact_fraction),
        ("sandwiched_min_facing_cosine", sandwiched_min_facing_cosine),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    spacing = np.asarray([float(v) for v in voxel_size_zyx], dtype=float)
    start = time.perf_counter()

    have_large = any(mask is not None and np.any(mask) for mask in large.values())
    stats["used_large_mask_tangency"] = bool(have_large)
    if have_large:
        art, ven, tangential_stats = _tangential_pass(
            art,
            ven,
            large,
            spacing,
            _TangentialLimits(
                max_contact=float(max_contact_distance_microns),
                touch=float(touch_distance_microns),
                tangency_cosine_max=float(tangency_cosine_max),
                margin=float(reassignment_margin),
                min_contact_fraction=float(min_contact_fraction),
                opposite_exclusion=float(opposite_exclusion_distance_microns),
                tangency_weight=float(tangency_weight),
                opposite_penalty_weight=float(opposite_penalty_weight),
            ),
            workers=int(reassignment_parallel_workers),
        )
    else:
        art, ven = art.copy(), ven.copy()
        tangential_stats = _empty_tangential_stats(0)
    tangential_done = time.perf_counter()
    stats.update(tangential_stats)

    if enable_sandwiched_component_reassignment:
        art, ven, sandwich_stats = _sandwich_pass(
            art,
            ven,
            spacing,
            max_gap=float(sandwiched_max_endpoint_distance_microns),
            min_facing=float(sandwiched_min_facing_cosine),
            max_angle_degrees=float(sandwiched_max_axis_angle_degrees),
        )
    else:
        sandwich_stats = _empty_sandwich_stats()
    stats.update(sandwich_stats)
    stats["tangential_phase_elapsed_s"] = tangential_done - start
    stats["sandwiched_phase_elapsed_s"] = time.perf_counter() - tangential_done
    stats["total_elapsed_s"] = time.perf_counter() - start
    _log_summary(stats)
    return {"small_arteriole_mask": art, "small_venule_mask": ven, "stats": stats}


# --- tangential contact ------------------------------------------------------


@dataclass(frozen=True)
class _TangentialLimits:
    max_contact: float
    touch: float
    tangency_cosine_max: float
    margin: float
    min_contact_fraction: float
    opposite_exclusion: float
    tangency_weight: float
    opposite_penalty_weight: float


@dataclass
class _ClassContact:
    """The best qualifying contact a piece makes with one class of large vessel."""

    score: float
    contact_fraction: float
    tangency_cosine: float
    distance_microns: float


@dataclass
class _Verdict:
    outcome: str
    target: str | None = None
    contact: _ClassContact | None = None


class _LargeMaskSurface:
    """One large-vessel mask's surface voxels, indexed in microns."""

    def __init__(self, mask: np.ndarray, spacing: np.ndarray, normal_radius: float):
        self.mask = np.asarray(mask).astype(bool, copy=False)
        self.normal_radius = float(normal_radius)
        self.points = _surface_voxels(self.mask) * spacing
        self.tree = cKDTree(self.points)
        self._spacing = spacing

    def distances(
        self, voxels_zyx: np.ndarray, limit: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance to the mask (0 inside) and to its surface, beyond *limit* inf.

        The third array indexes the nearest surface voxel, where it is within
        *limit*.
        """
        to_surface, nearest = self.tree.query(
            voxels_zyx * self._spacing, distance_upper_bound=float(limit)
        )
        inside = self.mask[tuple(voxels_zyx.T)]
        return np.where(inside, 0.0, to_surface), to_surface, nearest

    def normal_at(self, surface_index: int) -> np.ndarray | None:
        """The surface's normal around one of its voxels, sign aside."""
        around = self.tree.query_ball_point(
            self.points[int(surface_index)], r=self.normal_radius
        )
        return _plane_normal(self.points[around])


def _tangential_pass(
    art: np.ndarray,
    ven: np.ndarray,
    large: dict[str, np.ndarray | None],
    spacing: np.ndarray,
    limits: _TangentialLimits,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels, count = label(art | ven, structure=_FULL_STRUCTURE)
    stats = _empty_tangential_stats(int(count))
    if count == 0:
        return art.copy(), ven.copy(), stats
    # A normal needs a patch of surface several voxels across, whatever the
    # coarsest axis; the touch distance is the scale the contact is judged on.
    normal_radius = max(limits.touch, 2.0 * float(spacing.max()))
    surfaces = {
        name: _LargeMaskSurface(mask, spacing, normal_radius)
        for name, mask in large.items()
        if mask is not None and np.any(mask)
    }
    slices = find_objects(labels)

    def judge(component_id: int) -> tuple[int, np.ndarray, _Verdict]:
        box = slices[component_id - 1]
        offset = np.asarray([axis.start for axis in box])
        voxels = np.argwhere(labels[box] == component_id) + offset
        return component_id, voxels, _judge_piece(voxels, surfaces, spacing, limits)

    ids = range(1, int(count) + 1)
    if workers > 1 and count > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(judge, ids))
    else:
        verdicts = [judge(component_id) for component_id in ids]

    out_art = art.copy()
    out_ven = ven.copy()
    for _component_id, voxels, verdict in verdicts:
        if verdict.target is None:
            stats[verdict.outcome] += 1
            continue
        index = tuple(voxels.T)
        to_arteriole = verdict.target == "arteriole"
        was_arteriole = art[index]
        was_venule = ven[index]
        if np.all(was_arteriole == to_arteriole) and np.all(was_venule != to_arteriole):
            stats["already_that_class"] += 1
            continue
        out_art[index] = to_arteriole
        out_ven[index] = not to_arteriole
        stats[f"reassigned_to_{verdict.target}"] += 1
        stats["changes"].append(
            {
                "voxel_count": int(voxels.shape[0]),
                "arteriole_share_before": float(np.mean(was_arteriole)),
                "to": verdict.target,
                "contact_fraction": verdict.contact.contact_fraction,
                "tangency_cosine": verdict.contact.tangency_cosine,
                "distance_microns": verdict.contact.distance_microns,
            }
        )
    stats["changes"].sort(key=lambda change: -change["voxel_count"])
    return out_art, out_ven, stats


def _judge_piece(
    voxels: np.ndarray,
    surfaces: dict[str, _LargeMaskSurface],
    spacing: np.ndarray,
    limits: _TangentialLimits,
) -> _Verdict:
    distances = {
        name: surface.distances(voxels, limits.max_contact)
        for name, surface in surfaces.items()
    }
    if not any(np.any(d[0] <= limits.touch) for d in distances.values()):
        return _Verdict("no_contact")

    contacts: dict[str, _ClassContact] = {}
    for name, surface in surfaces.items():
        opposite = distances.get(_OPPOSITE[name])
        contact = _best_class_contact(
            voxels,
            distances[name],
            None if opposite is None else opposite[0],
            surface,
            spacing,
            limits,
        )
        if contact is not None:
            contacts[name] = contact
    if not contacts:
        return _Verdict("not_tangential")
    qualified = {
        name: contact
        for name, contact in contacts.items()
        if contact.contact_fraction >= limits.min_contact_fraction
    }
    if not qualified:
        return _Verdict("too_little_contact")
    if len(qualified) == 1:
        (name, contact), = qualified.items()
        return _Verdict("reassigned", name, contact)
    arteriole, venule = qualified["arteriole"], qualified["venule"]
    if venule.score - arteriole.score >= limits.margin:
        return _Verdict("reassigned", "arteriole", arteriole)
    if arteriole.score - venule.score >= limits.margin:
        return _Verdict("reassigned", "venule", venule)
    return _Verdict("ambiguous")


def _best_class_contact(
    voxels: np.ndarray,
    distance: tuple[np.ndarray, np.ndarray, np.ndarray],
    opposite_distance: np.ndarray | None,
    surface: _LargeMaskSurface,
    spacing: np.ndarray,
    limits: _TangentialLimits,
) -> _ClassContact | None:
    """The best tangential patch of *voxels* against one large mask, if any.

    A patch is a connected stretch of the piece within the contact distance of
    the mask. Its direction is that stretch's own, so a vessel running along
    the large one and a vessel ending against it are told apart by the part
    that is actually near it.
    """
    to_mask, to_surface, nearest = distance
    near = np.flatnonzero(to_mask <= limits.max_contact)
    if near.size == 0:
        return None
    best: _ClassContact | None = None
    touching_voxels = 0
    for patch in _patches(voxels[near]):
        members = near[patch]
        closest = float(to_mask[members].min())
        if closest > limits.touch:
            continue
        cosine = _patch_tangency(
            voxels[members], to_mask[members], to_surface[members],
            nearest[members], surface, spacing,
        )
        if cosine is None or cosine > limits.tangency_cosine_max:
            continue
        touching_voxels += int(np.count_nonzero(to_mask[members] <= limits.touch))
        opposite = (
            np.inf if opposite_distance is None else float(opposite_distance[members].min())
        )
        score = (
            closest
            + limits.tangency_weight * cosine
            + limits.opposite_penalty_weight * max(0.0, limits.opposite_exclusion - opposite)
        )
        if best is None or score < best.score:
            best = _ClassContact(score, 0.0, cosine, closest)
    if best is None:
        return None
    best.contact_fraction = touching_voxels / float(voxels.shape[0])
    return best


def _patch_tangency(
    patch_voxels: np.ndarray,
    to_mask: np.ndarray,
    to_surface: np.ndarray,
    nearest: np.ndarray,
    surface: _LargeMaskSurface,
    spacing: np.ndarray,
) -> float | None:
    """|cos| between a patch's direction and the large surface's normal where
    the patch comes closest; 0 for a patch deep inside the large mask, None
    when either direction is undefined."""
    axis = _principal_axis(patch_voxels * spacing)
    if axis is None:
        return None
    closest = to_mask == to_mask.min()
    reachable = closest & np.isfinite(to_surface)
    if not np.any(reachable):
        # Every closest voxel is inside the large mask and beyond the contact
        # distance from its wall: the piece lies within the large vessel.
        return 0.0
    candidates = np.flatnonzero(reachable)
    at = candidates[np.argmin(to_surface[candidates])]
    normal = surface.normal_at(int(nearest[at]))
    if normal is None:
        return None
    return float(abs(np.dot(axis, normal)))


def _patches(voxels: np.ndarray) -> list[np.ndarray]:
    """Index arrays of *voxels*' 26-connected groups."""
    low = voxels.min(axis=0)
    local = voxels - low
    grid = np.zeros(tuple(local.max(axis=0) + 1), dtype=bool)
    grid[tuple(local.T)] = True
    labels, count = label(grid, structure=_FULL_STRUCTURE)
    ids = labels[tuple(local.T)]
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(1, count + 2))
    return [order[bounds[k] : bounds[k + 1]] for k in range(count)]


# --- sandwiched pieces -------------------------------------------------------


@dataclass
class _Pieces:
    """Every small-mask piece, both classes, indexed together in microns.

    Pieces ``0 .. art_count - 1`` came from the arteriole mask, the rest from
    the venule mask; ``kind`` is each one's class as the pass stands (0
    arteriole, 1 venule). ``members[p]`` indexes piece ``p``'s rows of
    ``voxels`` / ``points``.
    """

    voxels: np.ndarray
    points: np.ndarray
    owner: np.ndarray
    members: list[np.ndarray]
    sizes: np.ndarray
    kind: np.ndarray
    art_count: int
    tree: cKDTree = field(repr=False)


def _sandwich_pass(
    art: np.ndarray,
    ven: np.ndarray,
    spacing: np.ndarray,
    *,
    max_gap: float,
    min_facing: float,
    max_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stats = _empty_sandwich_stats()
    pieces = _index_pieces(art, ven, spacing)
    if pieces is None:
        return art, ven, stats
    sizes = pieces.sizes
    # Two voxel centres further apart than a voxel diagonal do not touch.
    touching = float(np.linalg.norm(spacing))
    max_angle = np.radians(float(max_angle_degrees))
    # Largest first, each against its neighbours' classes as they stand, so a
    # chain of alternating pieces settles into one class instead of every
    # piece flipping at once.
    for piece in np.argsort(-sizes, kind="stable"):
        points = pieces.points[pieces.members[piece]]
        axis = _principal_axis(points)
        if axis is None:
            continue
        along = points @ axis
        target = 1 - int(pieces.kind[piece])
        neighbours = []
        for outward, extreme in ((-axis, along.min()), (axis, along.max())):
            end = points[np.abs(along - extreme) <= touching].mean(axis=0)
            neighbour = _in_line_neighbour(
                pieces, piece, end, outward, axis, target,
                max_gap=max_gap, min_facing=min_facing,
                max_angle=max_angle, touching=touching,
            )
            if neighbour is None:
                break
            neighbours.append(neighbour)
        if len(neighbours) != 2 or neighbours[0] == neighbours[1]:
            continue
        if sizes[piece] >= sizes[neighbours[0]] + sizes[neighbours[1]]:
            continue
        pieces.kind[piece] = target
        to = _CLASSES[target]
        stats[f"sandwiched_flips_to_{to}"] += 1
        stats["sandwiched_flips"].append(
            {
                "voxel_count": int(sizes[piece]),
                "to": to,
                "neighbour_voxel_counts": [int(sizes[n]) for n in neighbours],
            }
        )

    started_venule = np.arange(sizes.size) >= pieces.art_count
    flipped = np.flatnonzero(pieces.kind != started_venule)
    if flipped.size == 0:
        return art, ven, stats
    out_art = art.copy()
    out_ven = ven.copy()
    for piece in flipped:
        index = tuple(pieces.voxels[pieces.members[piece]].T)
        to_arteriole = bool(pieces.kind[piece] == 0)
        out_art[index] = to_arteriole
        out_ven[index] = not to_arteriole
    return out_art, out_ven, stats


def _index_pieces(art: np.ndarray, ven: np.ndarray, spacing: np.ndarray) -> _Pieces | None:
    art_labels, art_count = label(art, structure=_FULL_STRUCTURE)
    ven_labels, ven_count = label(ven, structure=_FULL_STRUCTURE)
    if art_count == 0 or ven_count == 0:
        return None
    art_voxels = np.argwhere(art_labels)
    ven_voxels = np.argwhere(ven_labels)
    owner = np.concatenate(
        [
            art_labels[tuple(art_voxels.T)] - 1,
            ven_labels[tuple(ven_voxels.T)] - 1 + art_count,
        ]
    )
    voxels = np.concatenate([art_voxels, ven_voxels])
    points = voxels.astype(float) * spacing
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(art_count + ven_count + 1))
    members = [order[bounds[k] : bounds[k + 1]] for k in range(art_count + ven_count)]
    kind = np.concatenate([np.zeros(art_count, dtype=int), np.ones(ven_count, dtype=int)])
    return _Pieces(
        voxels=voxels,
        points=points,
        owner=owner,
        members=members,
        sizes=np.diff(bounds),
        kind=kind,
        art_count=int(art_count),
        tree=cKDTree(points),
    )


def _in_line_neighbour(
    pieces: _Pieces,
    piece: int,
    end: np.ndarray,
    outward: np.ndarray,
    axis: np.ndarray,
    target: int,
    *,
    max_gap: float,
    min_facing: float,
    max_angle: float,
    touching: float,
) -> int | None:
    """The nearest *target*-class piece beyond one end of *piece* and in line with it.

    Beyond: a touching voxel has to lie past the end, not beside it, and one
    across a gap has to be where the piece points (``min_facing``). In line:
    the neighbour's own direction near that voxel is within ``max_angle`` of
    the piece's, so a vessel it merely ends against, side-on, does not count.
    """
    near = np.asarray(pieces.tree.query_ball_point(end, r=max_gap), dtype=int)
    if near.size == 0:
        return None
    owners = pieces.owner[near]
    keep = (owners != piece) & (pieces.kind[owners] == target)
    near, owners = near[keep], owners[keep]
    if near.size == 0:
        return None
    offsets = pieces.points[near] - end
    gaps = np.linalg.norm(offsets, axis=1)
    ahead = offsets @ outward
    with np.errstate(invalid="ignore", divide="ignore"):
        facing = ahead / gaps
    beyond = np.where(gaps <= touching, ahead > 0.0, facing >= min_facing)
    near, owners, gaps = near[beyond], owners[beyond], gaps[beyond]
    tried: set[int] = set()
    for i in np.argsort(gaps, kind="stable"):
        neighbour = int(owners[i])
        if neighbour in tried:
            continue
        tried.add(neighbour)
        contact = pieces.points[near[i]]
        local = np.asarray(pieces.tree.query_ball_point(contact, r=max_gap), dtype=int)
        local = local[pieces.owner[local] == neighbour]
        neighbour_axis = _principal_axis(pieces.points[local])
        if neighbour_axis is None:
            neighbour_axis = _principal_axis(pieces.points[pieces.members[neighbour]])
        if neighbour_axis is None:
            continue
        angle = np.arccos(min(1.0, abs(float(np.dot(axis, neighbour_axis)))))
        if angle <= max_angle:
            return neighbour
    return None


# --- shared geometry -----------------------------------------------------------


def _principal_axis(points_um: np.ndarray) -> np.ndarray | None:
    """Unit direction of greatest spread, or None for a blob with no clear one."""
    if points_um.shape[0] < 3:
        return None
    centred = points_um - points_um.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
    if eigenvalues[-1] <= 0.0 or eigenvalues[-1] < _MIN_ELONGATION * eigenvalues[-2]:
        return None
    return eigenvectors[:, -1]


def _plane_normal(points_um: np.ndarray) -> np.ndarray | None:
    """Direction of least spread of a patch of surface, or None if it is not one."""
    if points_um.shape[0] < 3:
        return None
    centred = points_um - points_um.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
    if eigenvalues[1] <= 0.0:
        return None
    return eigenvectors[:, 0]


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Voxels of *mask* with a face neighbour outside it; volume edges excepted.

    A cut face where a vessel leaves the image is not wall, so it gives no
    normal. The nearest mask voxel to any outside point is always one of these.
    """
    occupied = [np.flatnonzero(np.any(mask, axis=other)) for other in ((1, 2), (0, 2), (0, 1))]
    low = np.asarray([max(0, int(o[0]) - 1) for o in occupied])
    high = np.asarray([min(n, int(o[-1]) + 2) for o, n in zip(occupied, mask.shape)])
    box = tuple(slice(a, b) for a, b in zip(low, high))
    local = mask[box]
    wall = local & ~binary_erosion(local, structure=_FACE_STRUCTURE, border_value=1)
    return (np.argwhere(wall) + low).astype(float)


# --- reporting -------------------------------------------------------------------


def _empty_tangential_stats(component_count: int) -> dict[str, Any]:
    return {
        "component_count": component_count,
        "reassigned_to_arteriole": 0,
        "reassigned_to_venule": 0,
        "no_contact": 0,
        "not_tangential": 0,
        "too_little_contact": 0,
        "ambiguous": 0,
        "already_that_class": 0,
        "changes": [],
    }


def _empty_sandwich_stats() -> dict[str, Any]:
    return {
        "sandwiched_flips_to_arteriole": 0,
        "sandwiched_flips_to_venule": 0,
        "sandwiched_flips": [],
    }


def _log_summary(stats: dict[str, Any]) -> None:
    for change in stats["changes"][:_MAX_LOGGED_CHANGES]:
        logger.info(
            "Small-vessel tangential redefinition: piece of "
            f"{change['voxel_count']} voxels ({change['arteriole_share_before']:.0%} "
            f"arteriole) -> {change['to']} (touches "
            f"{change['contact_fraction']:.0%} of it, tangency cosine "
            f"{change['tangency_cosine']:.2f}, {change['distance_microns']:.1f} um)."
        )
    if len(stats["changes"]) > _MAX_LOGGED_CHANGES:
        logger.info(
            "Small-vessel tangential redefinition: and "
            f"{len(stats['changes']) - _MAX_LOGGED_CHANGES} smaller piece(s)."
        )
    for flip in stats["sandwiched_flips"][:_MAX_LOGGED_CHANGES]:
        logger.info(
            "Small-vessel sandwiched piece: "
            f"{flip['voxel_count']} voxels -> {flip['to']} (between pieces of "
            f"{flip['neighbour_voxel_counts'][0]} and "
            f"{flip['neighbour_voxel_counts'][1]} voxels)."
        )
    if len(stats["sandwiched_flips"]) > _MAX_LOGGED_CHANGES:
        logger.info(
            "Small-vessel sandwiched piece: and "
            f"{len(stats['sandwiched_flips']) - _MAX_LOGGED_CHANGES} smaller piece(s)."
        )
    tangency = (
        f"over {stats['component_count']} piece(s): "
        f"{stats['reassigned_to_arteriole']} to arteriole and "
        f"{stats['reassigned_to_venule']} to venule by tangential contact; left "
        f"{stats['already_that_class']} already that class, "
        f"{stats['no_contact']} touching no large vessel, "
        f"{stats['not_tangential']} meeting one end-on, "
        f"{stats['too_little_contact']} touching too little and "
        f"{stats['ambiguous']} touching both classes too evenly"
        if stats.get("used_large_mask_tangency")
        else "no large-vessel masks, so no tangential contact"
    )
    logger.info(
        f"Small-vessel tangential redefinition {tangency}; flipped "
        f"{stats['sandwiched_flips_to_arteriole']} sandwiched piece(s) to "
        f"arteriole and {stats['sandwiched_flips_to_venule']} to venule "
        f"({stats['total_elapsed_s']:.1f} s)."
    )
