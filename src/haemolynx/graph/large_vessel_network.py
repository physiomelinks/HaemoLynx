"""Inlet/outlet seeding for large vessels kept in the network as Large_Art/Large_Ven.

A large arteriole or venule mask that is meant to stay in the network (see
``cut_at_large_vessel_volumes.py`` for the alternative, where it is cut away)
still needs an inlet/outlet: the point where the real vessel was truncated by
the imaging field of view, not an internal narrowing. Overlap-based terminal
selection (``terminal_node_assignment.py``) only ever looks at the graph's
*existing* degree-1 nodes and has no notion of "this is where the field of
view cut the vessel off" versus "this is a genuine interior arm tip" -- both
read identically as a degree-1 node overlapping the mask. This module answers
that question from the mask's own geometry instead: a component that touches
the image volume's own face was truncated by the field of view; one that
doesn't was not.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import networkx as nx

from ._helpers import points_inside_mask, sort_nodes
from .boundaries import BoundaryCoordinateWarning, _terminal_nodes_and_position_map

logger = logging.getLogger(__name__)

#: Full 26-neighbour connectivity, matching graph/large_vessels.py's convention.
_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

#: A face contact is a stump only while its footprint is at most this many
#: times the area of the disc its own inscribed radius spans (pi r^2, the
#: vessel's cross-section): a vessel cut by the face leaves about one
#: cross-section (an oblique cut up to 75 degrees stays under 4), while one
#: lying *along* the face -- a pial vessel against the top of a stack --
#: leaves a strip many cross-sections long. On E14.5 MCA (0.98 x 0.98 x 1 um)
#: the real stumps at the x faces read 1.6-3.0 and the vessels' flanks on the
#: z=0 face 5.0-27 (`outputs/` run of 2026-10-11; one dataset, one reader --
#: not a marked grid). Those flanks seeded 8 inlets and 11 outlets, each
#: snapped 11-36 um to some capillary tip.
STUMP_MAX_FOOTPRINT_CROSS_SECTIONS = 4.0

#: How deep into the stack (um) one component's footprints on a face are
#: told apart: those it joins within this depth of the face are one contact
#: (a vessel lying along the face touches it in patches its own body joins
#: there -- on E14.5 the pial vessels against z = 0), those it joins only
#: deeper in a contact each (branches of one tree leaving the face apart;
#: Harvey, 2026-10-25). About a large vessel's diameter; not fitted to marks.
STUMP_CONTACT_DEPTH_UM = 12.0

#: How far outside its stump's mask component a vessel's own centreline may
#: run and still be that vessel. The large masks and the network can come
#: from two segmentations: on E14.5 MCA the large masks are the Composite
#: ilastik export and the network the Claudin one, and the three large
#: vessels' centrelines leave their masks 2.0-5.8 um before the face
#: (2026-10-21; one dataset, not a marked grid). Held to the mask at 0 um,
#: the snap passed those ends over for the one terminal inside each mask,
#: anywhere in the image: the inlet 354 um from its stump, both vein stumps
#: on one node ~100 um away.
STUMP_SNAP_OFFSET_UM = 6.0

#: The stretch of a terminal's vessel, from its tip, read for
#: :data:`STUMP_SNAP_MIN_VESSEL_FRACTION`.
STUMP_SNAP_VESSEL_UM = 50.0

#: The share of that stretch within :data:`STUMP_SNAP_OFFSET_UM` of the
#: stump's component for the terminal to be the large vessel's own end, not
#: a capillary ending beside it.
STUMP_SNAP_MIN_VESSEL_FRACTION = 0.5

#: With the run's segmentation, a terminal is the large vessel's own end only
#: while at least this share of that stretch lies inside the stump's
#: component ...
STUMP_SNAP_MIN_INSIDE_FRACTION = 0.5

#: ... and there its segmented radius is at least this many times the
#: component's depth (how far its edge is) at the same points, the median
#: over them: the large vessel's centreline fills its mask, a capillary
#: ending in it does not. Harvey's marks on E14.5 (2026-10-26): the
#: arteriole's and the vein's ends 1.9 and 2.3, a capillary ending in the
#: vein's mask 1.0, another never inside it. Four marks on one stack, not a
#: marked grid; the masks and the network there come from two segmentations.
STUMP_SNAP_MIN_FILL = 1.5

#: How far round a stretch (um) a distance is read: a local distance
#: transform, so a stack of any size costs only the boxes round its ends.
_DEPTH_READ_UM = 25.0


@dataclass(frozen=True, eq=False)
class _Stump:
    """Where one mask component is cut off by one face of the volume: one
    connected piece of its footprint on that face."""

    point: np.ndarray  # the footprint piece's centroid, (z, y, x) um
    axis: int
    face_index: int  # the face plane's voxel index along *axis*
    component: int  # its number in the mask's ForegroundLabels
    radius_um: float  # the footprint piece's inscribed radius
    footprint_um: np.ndarray  # the piece's voxels, (n, 3) (z, y, x) um


def _footprint_shape(
    footprint: np.ndarray, in_plane_spacing: tuple[float, float]
) -> tuple[float, float]:
    """A face footprint's area over the area of the disc its inscribed
    radius spans, and that radius (um); ``(inf, 0)`` for an empty footprint."""
    from scipy.ndimage import distance_transform_edt

    if not footprint.any():
        return float("inf"), 0.0
    # Padded, so the array's own edge counts as outside the footprint.
    inscribed = distance_transform_edt(np.pad(footprint, 1), sampling=in_plane_spacing)
    radius = float(inscribed[1:-1, 1:-1][footprint].max())
    area = float(np.count_nonzero(footprint)) * float(in_plane_spacing[0]) * float(
        in_plane_spacing[1]
    )
    return (area / (np.pi * radius * radius) if radius > 0 else float("inf")), radius


def _face_contacts(
    labeled: Any, component: int, on_face: np.ndarray, axis: int, face_index: int, scale: np.ndarray
) -> list[np.ndarray]:
    """*component*'s contacts with one face: its footprint *on_face* split
    where the component does not join within :data:`STUMP_CONTACT_DEPTH_UM`
    of the face, each a footprint of the face's shape. Branches of one tree
    that join deeper in leave the face apart, a contact each; a vessel lying
    along the face touches it in patches its own body joins right there,
    one contact, which the footprint test then reads as one."""
    from scipy.ndimage import label

    coords = np.argwhere(on_face)
    if not len(coords):
        return []
    in_plane = [a for a in range(len(labeled.shape)) if a != axis]
    pad = np.ceil(STUMP_CONTACT_DEPTH_UM / scale[in_plane]).astype(int)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, [labeled.shape[a] for a in in_plane])
    depth = max(1, int(np.ceil(STUMP_CONTACT_DEPTH_UM / scale[axis])))
    box = [slice(0, 0)] * len(labeled.shape)
    for a, l, h in zip(in_plane, lo, hi):
        box[a] = slice(int(l), int(h))
    near = (slice(face_index, min(face_index + depth, labeled.shape[axis])) if face_index == 0
            else slice(max(face_index - depth + 1, 0), face_index + 1))
    box[axis] = near
    groups, _count = label(labeled.read(tuple(box)) == component, structure=_STRUCTURE)
    face_plane = np.take(groups, 0 if face_index == 0 else groups.shape[axis] - 1, axis=axis)
    in_box = on_face[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))]
    contacts = []
    for group in np.unique(face_plane[in_box]):
        if group == 0:
            continue
        footprint = np.zeros_like(on_face)
        footprint[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))] = in_box & (face_plane == group)
        contacts.append(footprint)
    return contacts


def find_large_vessel_mask_stump_points(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_component_fraction: float = 0.0,
    max_footprint_cross_sections: float = STUMP_MAX_FOOTPRINT_CROSS_SECTIONS,
) -> list[np.ndarray]:
    """Physical-coordinate points where *mask* is cut off by the volume's edge.

    One point per place a connected component touches a face -- each
    in-plane connected piece of its footprint on that face, so branches of
    one tree that join inside the stack and leave the same face are a stump
    each -- at the centroid of *that piece's own voxels only* -- not the
    whole component, whose
    bulk centroid can sit far from the true stump cross-section for a long
    trunk and would then snap to the wrong graph terminal. Kept per-face
    rather than merged across every face one component touches: a vessel
    that runs corner-to-corner through the whole volume touches two
    different faces at two genuinely different ends, and averaging those
    into one point midway between them would not describe either stump. A
    component that never reaches a face (a vessel that genuinely ends
    inside the tissue, not truncated by the field of view) contributes no
    point.

    Three contacts are not stumps. A component smaller than
    *min_component_fraction* of the mask's largest one is a speck the
    classifier left at the face, not a large vessel (E14.5's arteriole mask
    had 468 components; the vessel was one, the next 0.5% of it), and a
    contact under that share of its component's largest on the same face is
    a graze. A contact
    whose footprint spans more than *max_footprint_cross_sections* of its own
    cross-section is a vessel lying along the face, not cut by it (see
    :data:`STUMP_MAX_FOOTPRINT_CROSS_SECTIONS`).
    """
    _labeled, stumps = _mask_stumps(
        mask,
        voxel_size_zyx=voxel_size_zyx,
        min_component_fraction=min_component_fraction,
        max_footprint_cross_sections=max_footprint_cross_sections,
    )
    return [stump.point for stump in stumps]


def _mask_stumps(
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    min_component_fraction: float = 0.0,
    max_footprint_cross_sections: float = STUMP_MAX_FOOTPRINT_CROSS_SECTIONS,
) -> tuple[Any, list[_Stump]]:
    """:func:`find_large_vessel_mask_stump_points`'s stumps, each with its
    face and component, and the mask's components (``None`` for an empty
    mask)."""
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return None, []

    # Components labelled a slab of planes at a time and held at their voxels
    # (``preprocessing.slab_labels.ForegroundLabels``), not in a label volume
    # (8 bytes a voxel once a stack has 2**31 voxels).
    labeled = ForegroundLabels(mask_bool, _STRUCTURE)
    scale = np.asarray(voxel_size_zyx, dtype=float)
    sizes = labeled.sizes
    min_size = float(min_component_fraction) * float(sizes[1:].max())

    stumps: list[_Stump] = []
    skipped_small = 0
    skipped_flank = 0
    skipped_graze = 0
    for axis in range(mask_bool.ndim):
        in_plane = tuple(float(scale[a]) for a in range(mask_bool.ndim) if a != axis)
        for face in (0, -1):
            face_index = [slice(None)] * mask_bool.ndim
            face_index[axis] = face
            face_axis_value = face_index[axis] % labeled.shape[axis]
            face_box = [slice(0, n) for n in labeled.shape]
            face_box[axis] = slice(face_axis_value, face_axis_value + 1)
            face_labels = np.squeeze(labeled.read(tuple(face_box)), axis=axis)
            for component_id in np.unique(face_labels):
                if component_id == 0:
                    continue
                if sizes[component_id] < min_size:
                    skipped_small += 1
                    continue
                # Each place the component leaves this face is a stump of its
                # own (Harvey, 2026-10-25): pooled, two branches of one tree
                # leaving the face became one stump between them, and four
                # failed the footprint test as a vessel lying along the face.
                # A contact under *min_component_fraction* of the component's
                # largest on that face is a graze, as a speck is to components
                # (E14.5's arteriole touched its x face again in 5 um2, 2% of
                # its 230 um2 stump, 84 um from any vessel end there).
                contacts = []
                for footprint in _face_contacts(
                    labeled, component_id, face_labels == component_id, axis, face_axis_value, scale
                ):
                    cross_sections, radius = _footprint_shape(footprint, in_plane)
                    if cross_sections > float(max_footprint_cross_sections):
                        skipped_flank += 1
                        continue
                    contacts.append((footprint, radius, int(np.count_nonzero(footprint))))
                largest = max((size for *_f, size in contacts), default=0)
                for footprint, radius, size in contacts:
                    if size < float(min_component_fraction) * largest:
                        skipped_graze += 1
                        continue
                    face_coords = np.argwhere(footprint)
                    # Restore the dropped axis so every point is a full (z, y, x)
                    # voxel index, not the 2D coordinate within the face slice.
                    full_coords = np.insert(face_coords, axis, face_axis_value, axis=1)
                    stumps.append(
                        _Stump(
                            point=full_coords.mean(axis=0) * scale,
                            axis=axis,
                            face_index=int(face_axis_value),
                            component=int(component_id),
                            radius_um=radius,
                            footprint_um=full_coords * scale,
                        )
                    )
    if skipped_small or skipped_flank or skipped_graze:
        logger.info(
            "Large-vessel mask stumps: %d kept; %d face contact(s) of components "
            "below %.0f%% of the largest skipped, %d skipped as a vessel lying "
            "along the face (footprint over %.1f cross-sections), %d as a graze "
            "under that share of its component's largest contact on the face.",
            len(stumps),
            skipped_small,
            100.0 * float(min_component_fraction),
            skipped_flank,
            float(max_footprint_cross_sections),
            skipped_graze,
        )
    return labeled, stumps


_FACE_NAMES = ("z", "y", "x")


def _vessel_from_tip(G: nx.Graph, node: Any, pos: dict[Any, np.ndarray]) -> np.ndarray:
    """The first :data:`STUMP_SNAP_VESSEL_UM` of terminal *node*'s vessel,
    from its tip: the edge's ``voxels`` (um), or its two node positions
    sampled every micron when it has none."""
    edges = list(G.edges(node, data=True))
    if not edges:
        return np.empty((0, 3))
    _u, other, data = edges[0]
    path = np.asarray(data.get("voxels") or [], dtype=float).reshape(-1, 3)
    tip = pos[node]
    if len(path) < 2:
        far = np.asarray(G.nodes[other]["pos"], dtype=float)
        steps = max(2, int(np.ceil(np.linalg.norm(far - tip))) + 1)
        path = tip + np.linspace(0.0, 1.0, steps)[:, None] * (far - tip)
    elif np.linalg.norm(path[-1] - tip) < np.linalg.norm(path[0] - tip):
        path = path[::-1]
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return path[along <= STUMP_SNAP_VESSEL_UM]


def _distances_to_component(
    labeled: Any, component: int, points_um: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Each point's distance (um) to the nearest voxel of *component*,
    exact up to :data:`STUMP_SNAP_OFFSET_UM` (``inf`` beyond what the box
    round the points reaches)."""
    from scipy.ndimage import distance_transform_edt

    shape = np.asarray(labeled.shape)
    index = np.clip(np.rint(points_um / scale).astype(int), 0, shape - 1)
    pad = np.ceil(STUMP_SNAP_OFFSET_UM / scale).astype(int) + 1
    lo = np.maximum(index.min(axis=0) - pad, 0)
    hi = np.minimum(index.max(axis=0) + pad + 1, shape)
    inside = labeled.read(tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))) == component
    if not inside.any():
        return np.full(len(points_um), np.inf)
    distance = distance_transform_edt(~inside, sampling=scale)
    return distance[tuple((index - lo).T)]


def _own_face_end(
    G: nx.Graph,
    stump: _Stump,
    labeled: Any,
    candidates: list[Any],
    pos: dict[Any, np.ndarray],
    scale: np.ndarray,
    segmentation: Any = None,
) -> Any | None:
    """The terminal that ends *stump*'s own large vessel at its face, or None.

    A candidate lies within :data:`STUMP_SNAP_OFFSET_UM` of the stump's own
    mask component, near its contact (:func:`_near_contact`), and the first
    :data:`STUMP_SNAP_VESSEL_UM` of its vessel runs along the
    component (:data:`STUMP_SNAP_MIN_VESSEL_FRACTION`) -- the large vessel's
    centreline, not a capillary ending beside it. One inside the component
    comes first, then the one whose vessel runs most along the component,
    then the one nearest the stump's centroid: on E14.5 a vessel beside the
    arteriole, running half along its mask and ending nearer the centroid,
    took the inlet from the arteriole's own end, which runs all along it
    (Harvey's call, 2026-10-25). With the run's *segmentation*, a terminal
    whose vessel does not fill the component it runs in is a capillary ending
    there, not the large vessel's end (:func:`_fills_its_mask`; Harvey,
    2026-10-26).
    """
    box = labeled.box(stump.component)
    lo = np.asarray([s.start for s in box], dtype=float) * scale - STUMP_SNAP_OFFSET_UM
    hi = (np.asarray([s.stop for s in box], dtype=float) - 1.0) * scale + STUMP_SNAP_OFFSET_UM
    best: tuple[tuple[bool, float], Any] | None = None
    for node in candidates:
        tip = pos[node]
        if not _near_contact(tip, stump, scale):
            continue
        if np.any(tip < lo) or np.any(tip > hi):
            continue
        vessel = _vessel_from_tip(G, node, pos)
        distances = _distances_to_component(
            labeled, stump.component, np.vstack([tip[None, :], vessel]), scale
        )
        if distances[0] > STUMP_SNAP_OFFSET_UM:
            continue
        along = float(np.mean(distances[1:] <= STUMP_SNAP_OFFSET_UM)) if len(vessel) else 0.0
        if len(vessel) and along < STUMP_SNAP_MIN_VESSEL_FRACTION:
            continue
        if segmentation is not None and not _fills_its_mask(labeled, stump.component, segmentation, vessel, scale):
            continue
        rank = (bool(distances[0] > 0.0), -along, float(np.linalg.norm(tip - stump.point)))
        if best is None or rank < best[0]:
            best = (rank, node)
    return None if best is None else best[1]


def _depth_um(volume: Any, points_um: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """How far (um) each point lies inside *volume* (``> 0``): 0 outside it,
    from a distance transform of the box round the points; the box's own
    edge is not taken for background."""
    from scipy.ndimage import distance_transform_edt

    points = np.asarray(points_um, dtype=float).reshape(-1, 3)
    shape = np.asarray(np.shape(volume))
    index = np.clip(np.rint(points / scale).astype(int), 0, shape - 1)
    pad = np.ceil(_DEPTH_READ_UM / scale).astype(int)
    lo = np.maximum(index.min(axis=0) - pad, 0)
    hi = np.minimum(index.max(axis=0) + pad + 1, shape)
    box = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]) > 0
    distance = distance_transform_edt(np.pad(box, 1, constant_values=True), sampling=scale)[1:-1, 1:-1, 1:-1]
    return distance[tuple((index - lo).T)]


class _ComponentVolume:
    """One component of a labelled mask as a volume :func:`_depth_um` reads."""

    def __init__(self, labeled: Any, component: int):
        self.labeled = labeled
        self.component = component
        self.shape = labeled.shape

    def __getitem__(self, box):
        return self.labeled.read(box) == self.component

    def __array__(self, dtype=None):  # pragma: no cover - never read whole
        raise TypeError("a component is read a box at a time")


def _fills_its_mask(labeled: Any, component: int, segmentation: Any, vessel: np.ndarray, scale: np.ndarray) -> bool:
    """Whether a terminal's *vessel* (its first :data:`STUMP_SNAP_VESSEL_UM`)
    is the large vessel's own centreline: mostly inside the component, and
    there wider in the *segmentation* than the component is deep
    (:data:`STUMP_SNAP_MIN_INSIDE_FRACTION`, :data:`STUMP_SNAP_MIN_FILL`).
    A vessel the segmentation does not show at all cannot be judged, and is
    taken as before."""
    if not len(vessel):
        return True
    radius = _depth_um(segmentation, vessel, scale)
    if not np.any(radius > 0):
        return True
    depth = _depth_um(_ComponentVolume(labeled, component), vessel, scale)
    inside = depth > 0
    if inside.mean() < STUMP_SNAP_MIN_INSIDE_FRACTION:
        return False
    return float(np.median(radius[inside] / depth[inside])) >= STUMP_SNAP_MIN_FILL


def _in_plane_distance_um(tip: np.ndarray, stump: _Stump) -> float:
    """How far *tip* lies from *stump*'s footprint across its face (um), the
    distance along the face's normal left out."""
    offsets = np.delete(stump.footprint_um - tip, stump.axis, axis=1)
    return float(np.sqrt((offsets * offsets).sum(axis=1)).min())


def _near_contact(tip: np.ndarray, stump: _Stump, scale: np.ndarray) -> bool:
    """Whether *tip* lies within the stump's radius and
    :data:`STUMP_SNAP_OFFSET_UM` of its contact, both off the face and across
    it (from its footprint). Across the face only the component's bounding
    box held a stump with one contact, so on a graph where the arteriole's
    own end just failed the tests the inlet went to an end 239 um away along
    the face (E14.5, 2026-10-26; Harvey's call)."""
    reach = STUMP_SNAP_OFFSET_UM + stump.radius_um
    face_um = stump.face_index * float(scale[stump.axis])
    if abs(float(tip[stump.axis]) - face_um) > reach:
        return False
    return _in_plane_distance_um(tip, stump) <= reach


def _own_candidates(stumps: list[_Stump], pos: dict[Any, np.ndarray], candidates: list[Any]) -> list[list[Any]]:
    """Each stump's candidates: those nearer its footprint, across the face,
    than any other footprint its component leaves the same face by. One
    component's contacts on one face share no end, so each keeps its own
    rather than all taking the end whose vessel runs most along the component
    (:func:`_own_face_end`'s ranking)."""
    siblings: dict[tuple[int, int, int], list[int]] = {}
    for i, stump in enumerate(stumps):
        siblings.setdefault((stump.component, stump.axis, stump.face_index), []).append(i)
    own: list[list[Any]] = []
    for i, stump in enumerate(stumps):
        group = siblings[(stump.component, stump.axis, stump.face_index)]
        if len(group) == 1:
            own.append(list(candidates))
            continue
        own.append([
            node for node in candidates
            if min(group, key=lambda j: (_in_plane_distance_um(pos[node], stumps[j]), j)) == i
        ])
    return own


def _snap_stumps_to_terminals(
    G: nx.Graph,
    stumps: list[_Stump],
    labeled: Any,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    coordinates_setting_name: str,
    exclude_nodes: Iterable[Any] | None,
    segmentation: Any = None,
) -> list[Any]:
    """One terminal per stump: the end of the stump's own large vessel at
    its face (:func:`_own_face_end`, which reads *segmentation* when given),
    so two stumps give two nodes.

    Of one component's contacts on one face each looks only at the ends
    nearer it than the others (:func:`_own_candidates`), and a stump with no
    end of its own takes no node while another stump of the mask has one. A
    stump whose vessel has no such end otherwise -- the mask would have no
    node at all -- takes the nearest of its ends near its contact
    (:func:`_near_contact`), and with none there the nearest terminal inside
    *mask* (every terminal, when none is), with a
    :class:`~haemolynx.graph.boundaries.BoundaryCoordinateWarning` naming the
    stump and how far the node is from it: an inlet chosen there may be
    nowhere near where the vessel enters the image.
    """
    terminals, pos = _terminal_nodes_and_position_map(G)
    excluded = set(exclude_nodes or [])
    candidates = [node for node in terminals if node not in excluded]
    if not candidates:
        return []
    scale = np.asarray(voxel_size_zyx, dtype=float)
    fallback: list[Any] | None = None
    selected: list[Any] = []
    own = _own_candidates(stumps, pos, candidates)
    ends = [
        _own_face_end(G, stump, labeled, mine, pos, scale, segmentation=segmentation)
        for stump, mine in zip(stumps, own)
    ]
    # A stump with no vessel end of its own takes no node while another of
    # the mask's stumps has one, rather than the nearest terminal in the
    # mask, which can lie anywhere: a further branch's contact (2026-10-25),
    # or a vein whose own centreline runs past the face with only capillaries
    # ending there (E14.5's second vein contact; Harvey, 2026-10-26).
    ended = any(node is not None for node in ends)
    for stump, node, mine in zip(stumps, ends, own):
        if node is not None:
            selected.append(node)
            continue
        if ended:
            logger.info(
                "%s: the contact at %s um has no vessel end of its own; another of the "
                "mask's contacts has one, so it takes no node.",
                coordinates_setting_name, tuple(round(float(v), 1) for v in stump.point),
            )
            continue
        # The nearest end near the contact, though none passed the tests,
        # before the nearest terminal anywhere inside the mask: on E14.5 the
        # arteriole's own end, 5 um from its contact, just failed them and
        # the mask's nearest terminal lay 365 um inside it (2026-10-26).
        near = [n for n in mine if _near_contact(pos[n], stump, scale)]
        if near:
            pool, which = near, (
                f"the nearest end within {STUMP_SNAP_OFFSET_UM + stump.radius_um:.1f} um "
                "of the contact (its radius and "
                f"{STUMP_SNAP_OFFSET_UM:g} um)"
            )
        else:
            if fallback is None:
                inside = points_inside_mask(
                    np.asarray([pos[n] for n in candidates], dtype=float),
                    np.asarray(mask, dtype=bool),
                    voxel_size_zyx=voxel_size_zyx,
                )
                fallback = [n for n, is_inside in zip(candidates, inside) if is_inside] or candidates
            pool, which = fallback, "the nearest terminal inside the mask"
        node = min(pool, key=lambda n: float(np.linalg.norm(pos[n] - stump.point)))
        selected.append(node)
        face = f"{_FACE_NAMES[stump.axis]}={'0' if stump.face_index == 0 else stump.face_index}"
        warnings.warn(
            f"{coordinates_setting_name}: no centreline of the large vessel cut off at the "
            f"{face} face at {tuple(round(float(v), 1) for v in stump.point)} um ends there "
            f"(within {STUMP_SNAP_OFFSET_UM:g} um of its mask, along it); it takes node "
            f"{node}, {float(np.linalg.norm(pos[node] - stump.point)):.1f} um away, "
            f"{which}. Check the inlets/outlets, or pick them on the Boundaries tab.",
            BoundaryCoordinateWarning,
            stacklevel=4,
        )
    return sort_nodes(set(selected))


def select_large_vessel_mask_stump_terminal_nodes_for_role(
    G: nx.Graph,
    mask: np.ndarray,
    *,
    node_role: str,
    voxel_size_zyx: tuple[float, float, float],
    image_shape: tuple[int, ...],
    coordinates_setting_name: str,
    exclude_nodes: Iterable[Any] | None = None,
    min_component_fraction: float = 0.0,
    segmentation: np.ndarray | None = None,
) -> list[Any]:
    """One side's terminal nodes at *mask*'s own image-edge stump.

    Each of *mask*'s face-touching components (see
    :func:`find_large_vessel_mask_stump_points`, which also takes
    *min_component_fraction*) takes the terminal ending its own vessel at
    that face (:func:`_own_face_end`): near the component, near its contact,
    its vessel running along the component -- not a capillary tip nearer the
    face centroid, and not a terminal inside the mask somewhere else. When no
    stump of the mask has one, each takes the nearest end near its contact,
    else the nearest terminal inside the mask, with a
    :class:`~haemolynx.graph.boundaries.BoundaryCoordinateWarning`. A mask with no
    face-touching component (the vessel ends inside the tissue, not at the
    field of view's edge) yields no node. *exclude_nodes* keeps the other
    side's already-chosen node(s) out of reach, so an arteriole and venule
    mask that both happen to touch the same corner of the volume cannot snap
    to the same terminal for both the inlet and the outlet. With the run's
    *segmentation* (the binary image the graph was built from, on its grid),
    a terminal whose vessel does not fill the mask it runs in is no stump's
    end (:func:`_fills_its_mask`).
    """
    if node_role not in {"inlet", "outlet"}:
        raise ValueError("node_role must be 'inlet' or 'outlet'.")
    del image_shape  # the same signature as the other boundary selectors
    labeled, stumps = _mask_stumps(
        mask, voxel_size_zyx=voxel_size_zyx, min_component_fraction=min_component_fraction
    )
    if not stumps:
        return []
    return _snap_stumps_to_terminals(
        G,
        stumps,
        labeled,
        mask,
        voxel_size_zyx=voxel_size_zyx,
        coordinates_setting_name=coordinates_setting_name,
        exclude_nodes=exclude_nodes,
        segmentation=segmentation,
    )


def select_large_vessel_stump_terminal_nodes(
    G: nx.Graph,
    *,
    large_arteriole_mask: np.ndarray,
    large_venule_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[list[Any], list[Any]]:
    """Inlet/outlet nodes at the large-vessel masks' own image-edge stumps.

    Both sides of :func:`select_large_vessel_mask_stump_terminal_nodes_for_role`.
    """
    inlet_nodes = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G,
        large_arteriole_mask,
        node_role="inlet",
        voxel_size_zyx=voxel_size_zyx,
        image_shape=image_shape,
        coordinates_setting_name="large_arteriole_mask stump",
    )
    outlet_nodes = select_large_vessel_mask_stump_terminal_nodes_for_role(
        G,
        large_venule_mask,
        node_role="outlet",
        voxel_size_zyx=voxel_size_zyx,
        image_shape=image_shape,
        coordinates_setting_name="large_venule_mask stump",
    )
    return inlet_nodes, outlet_nodes
