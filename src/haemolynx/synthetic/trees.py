"""The arterial and venous trees of a synthetic network.

Each side grows from its roots (the template's inlets or outlets, on the
faces the template's were) one generation at a time, the large tier's orders
first and then the small tier's, so that **the number of vessels in each
order is the template's by construction**: generation ``g`` has the edge
count of the order table's ``g``-th order, split over the tips of generation
``g - 1`` as bifurcations (a trifurcation only when an odd count forces one,
more only when the table asks for more than two children a tip). The tips
that grow are chosen to fill space -- far from the faces and from vessels
already drawn.

The large tier (Large_Art/Large_Ven) is a **trunk with side branches**, as a
run with large-vessel masks finds it: its order rises at every junction, and
most of its junctions are where a capillary leaves it. So within the large
tier a tip may grow one child -- the trunk going on -- and hand a side branch
to the capillary bed (:attr:`SideResult.handoffs`, the large-vessel boundary
nodes), as well as bifurcate when the next order has more vessels. Its widths
are the template's, each order's drawn from that order's distribution
(:func:`draw_root_diameters`): a trunk's measured width hardly falls from one
order to the next, which Murray's law at any junction losing a capillary would
not allow (at the network's exponent it shrinks to nothing within a few
junctions).

In the small tier (Art/Ven) diameters follow Murray's law exactly at every
junction, with the template's exponent: the children of a parent of width
``d`` have widths ``d (1 + a^n)^(-1/n)`` and ``a`` times that, the asymmetry
``a`` drawn round the value that reproduces the small tier's fall in mean
diameter from one order to the next (:func:`fit_asymmetry`). The two
daughters open at the template's daughter-daughter angle, the narrower
deflected more, in the plane round the parent that keeps them inside the box
and away from other vessels. A child that would leave the box is cut at the
face and marked as leaving the image -- in a closed network (*closed*) it is
turned back off the face at its full length (its direction mirrored in it),
or, where that leaves the box too, shortened to end inside it, the plane
chosen that keeps most of it. Given a hole in the box (``synthetic.hole``: an
image's own box, in a surround), a vessel that would come near it is shortened
to end before it, never cut there.

**The trees grow where the template's do** (``PLAN_synthetic_physiology.md``,
Harvey's D5a) when the template carries its layout (``synthetic.organisation``):
a side's vessels are scored by how much of that side's tree length the
template has in the cells they end in (:class:`TreeLayer`) instead of by
their distance from the faces, so a tissue whose large vessels lie in a
plexus on one face gets its trees there. Without a layout they keep away from
the faces, as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.spatial import cKDTree

from haemolynx.haemodynamics.viscosity import PRIES_SINGULARITY_UM

from .draft import TREE, NetworkDraft
from .hole import Hole
from .organisation import Organisation
from .template import (
    DEFAULT_DAUGHTER_ANGLE_DEG,
    DEFAULT_DAUGHTER_ANGLE_SD_DEG,
    DEFAULT_DIAMETER_CV,
    DEFAULT_MURRAY_EXPONENT,
    FACES,
    INTERIOR,
    ClassTarget,
    NetworkTemplate,
    OrderTarget,
    lognormal_from_mean_cv,
    lognormal_from_mean_median,
)

#: Azimuths tried round the parent for the plane a bifurcation lies in.
AZIMUTHS = 8
#: A daughter angle is kept within these.
MIN_DAUGHTER_ANGLE_DEG = 20.0
MAX_DAUGHTER_ANGLE_DEG = 150.0
#: A root's first vessel tilts at most this far from the face's inward normal.
ROOT_TILT_DEG = 30.0
#: Directions tried for a root's first vessel before it is shortened to fit
#: the box, and how far inside the face a shortened one stops (a share).
FIRST_VESSEL_AIMS = 12
FIRST_VESSEL_MARGIN = 0.05
#: A drawn length is kept within these multiples of its order's median.
LENGTH_CLIP = (0.2, 6.0)
#: A width drawn from a lognormal is kept within these multiples of its
#: median too, and never under MIN_DIAMETER_UM: twice the endothelial
#: surface layer the in vivo viscosity law takes off a vessel's width, at
#: which the law is singular. The tail otherwise drew the odd capillary
#: narrower than any the template measured (1.05 um, the template's
#: narrowest 2.3 um), and an endothelial run refused to solve it.
DIAMETER_CLIP = LENGTH_CLIP
MIN_DIAMETER_UM = 2.0 * PRIES_SINGULARITY_UM
#: The asymmetry is drawn uniformly this far either side of the fitted one.
ASYMMETRY_SPREAD = 0.15
#: Below this the fitted asymmetry is not let go (a daughter of no width).
MIN_ASYMMETRY = 0.05
#: The spread (sigma of the log) of the excess tortuosity round its mean.
TORTUOSITY_LOG_SIGMA = 0.6
#: A length drawn when the template gives none at all.
FALLBACK_LENGTH_UM = 30.0
FALLBACK_DIAMETER_UM = 8.0
#: A trunk's side branch's width, for its bend, when the template has no capillary width.
FALLBACK_SIDE_BRANCH_UM = 5.0
#: How much ending in one of the template's tree cells counts in a vessel's
#: score, in spacings (a cell with its median tree length or more: all of it).
LAYER_WEIGHT = 2.0
#: What a cell next to the template's tree cells is worth, as a share of
#: its best neighbour: a slope back towards the layer, not a wall.
LAYER_NEIGHBOUR_SHARE = 0.3
#: A tip reading less than this of its side's TreeLayer is out of the
#: template's tree cells (a neighbour cell reads LAYER_NEIGHBOUR_SHARE).
LAYER_INSIDE = 0.5
#: How much keeping a trunk on its tree's course from the root counts in a
#: junction's score, in spacings (a trunk turning straight back loses twice
#: this). Only with a template that records its trunks' straightness. On
#: E14.5 (five seeds) 2 keeps the trees 0.77-0.81 in the template's tree cells
#: and lets the venous trees reach 115-121 um from their outlets (median;
#: the template's 134); 4 reached further but strayed from the cells.
TRUNK_PERSISTENCE = 2.0
#: How far ahead of a vessel's end, in spacings, the layer and the room to
#: go on are read: a trunk steered only by where it ends runs on out of the
#: template's corridor, or into a face, before it can turn.
LAYER_LOOKAHEAD_SPACINGS = 2.0

_SIDE_CATEGORIES = {
    "arterial": ("large_arteriole", "arteriole"),
    "venous": ("large_venule", "venule"),
}
_SIDE_ROLES = {
    "arterial": ("inlet", "large_arteriole_boundary", "arteriole_boundary"),
    "venous": ("outlet", "large_venule_boundary", "venule_boundary"),
}


@dataclass
class Tip:
    node: int
    position: np.ndarray
    direction: np.ndarray
    diameter_um: float
    generation: int
    #: The root this tip's tree grew from.
    root: int = -1
    at_face: bool = False


@dataclass
class SideResult:
    """What growing one side left: its leaves (``at_face`` ones leave the
    image), the nodes where its large tier hands over to its small tier, and
    the tree nodes added (for the bed to keep clear of)."""

    side: str
    leaves: list = field(default_factory=list)
    large_boundary: list = field(default_factory=list)
    #: Large-tier nodes where the trunk goes on as one child and a side
    #: branch leaves for the bed (each takes one bed vessel).
    handoffs: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    #: ``{root: [its tree's nodes]}``: what each root connects on its own.
    trees: dict = field(default_factory=dict)
    edges: int = 0


# ---------------------------------------------------------------------------
# Distributions per order
# ---------------------------------------------------------------------------


def _length_sigma(cls: Optional[ClassTarget]) -> float:
    if cls is not None and cls.mean_length_um and cls.median_length_um:
        return lognormal_from_mean_median(cls.mean_length_um, cls.median_length_um)[1]
    return lognormal_from_mean_cv(1.0, DEFAULT_DIAMETER_CV)[1]


def draw_length(order: OrderTarget, cls: Optional[ClassTarget], rng: np.random.Generator) -> float:
    """A vessel length for *order*: one of its own vessels' when the template
    carries them, else lognormal round its mean with its class's spread."""
    if order.samples is not None and order.samples.lengths_um:
        return float(rng.choice(order.samples.lengths_um))
    mean = order.mean_length_um or (cls.mean_length_um if cls else None) or FALLBACK_LENGTH_UM
    sigma = _length_sigma(cls)
    mu = math.log(mean) - sigma**2 / 2.0
    value = float(rng.lognormal(mu, sigma))
    median = math.exp(mu)
    return float(np.clip(value, LENGTH_CLIP[0] * median, LENGTH_CLIP[1] * median))


def draw_lengths(order: OrderTarget, cls: Optional[ClassTarget], count: int, rng: np.random.Generator) -> list[float]:
    """*count* lengths for *order* (:func:`draw_length`), scaled so their
    mean is the order's: a generation of two vessels would otherwise miss
    the table's mean by the luck of the draw."""
    lengths = [draw_length(order, cls, rng) for _ in range(count)]
    mean = order.mean_length_um
    if lengths and mean:
        drawn = float(np.mean(lengths))
        if drawn > 0:
            lengths = [length * float(mean) / drawn for length in lengths]
    return lengths


def draw_tortuosity(order: OrderTarget, cls: Optional[ClassTarget], rng: np.random.Generator) -> float:
    """A tortuosity index (length over chord, at least 1) round the order's mean."""
    if order.samples is not None and order.samples.tortuosities:
        return max(1.0, float(rng.choice(order.samples.tortuosities)))
    mean = order.mean_tortuosity or (cls.mean_tortuosity if cls else None) or 1.1
    excess = max(float(mean) - 1.0, 0.0)
    if excess <= 0.0:
        return 1.0
    sigma = TORTUOSITY_LOG_SIGMA
    return 1.0 + excess * float(rng.lognormal(-(sigma**2) / 2.0, sigma))


def draw_root_diameter(order: OrderTarget, cls: Optional[ClassTarget], rng: np.random.Generator) -> float:
    """The width of a tree's first vessel: the first order's distribution."""
    if order.samples is not None and order.samples.diameters_um:
        return float(rng.choice(order.samples.diameters_um))
    mean = order.mean_diameter_um or (cls.mean_diameter_um if cls else None) or FALLBACK_DIAMETER_UM
    cv = order.diameter_cv if order.diameter_cv is not None else DEFAULT_DIAMETER_CV
    mu, sigma = lognormal_from_mean_cv(mean, cv)
    return float(clip_drawn_diameters(rng.lognormal(mu, sigma), math.exp(mu)))


def clip_drawn_diameters(values, median: float):
    """Widths drawn from a lognormal of *median* kept within
    :data:`DIAMETER_CLIP` times it and at least :data:`MIN_DIAMETER_UM`
    (which wins where the two disagree)."""
    low = max(DIAMETER_CLIP[0] * float(median), MIN_DIAMETER_UM)
    high = max(DIAMETER_CLIP[1] * float(median), low)
    return np.clip(values, low, high)


# ---------------------------------------------------------------------------
# Murray's law
# ---------------------------------------------------------------------------


def draw_root_diameters(order: OrderTarget, cls: Optional[ClassTarget], count: int, rng: np.random.Generator) -> list[float]:
    """*count* root widths (:func:`draw_root_diameter`), scaled so their
    mean is the order's when it has one."""
    diameters = [draw_root_diameter(order, cls, rng) for _ in range(count)]
    mean = order.mean_diameter_um or (cls.mean_diameter_um if cls else None)
    if diameters and mean:
        drawn = float(np.mean(diameters))
        if drawn > 0:
            diameters = [d * float(mean) / drawn for d in diameters]
    return diameters


def murray_ratios(asymmetry: float, exponent: float, children: int = 2) -> np.ndarray:
    """Child/parent width ratios ``(r, a r, a r, ...)`` with
    ``sum r_i^n = 1``: the widest child first."""
    a = float(np.clip(asymmetry, MIN_ASYMMETRY, 1.0))
    n = float(exponent)
    widest = (1.0 + (children - 1) * a**n) ** (-1.0 / n)
    return np.array([widest] + [a * widest] * (children - 1))


def mean_child_ratio(asymmetry: float, exponent: float) -> float:
    """The mean child/parent width ratio of a bifurcation of asymmetry *a*:
    ``(1 + a) / (2 (1 + a^n)^(1/n))``, rising from 1/2 to ``2^(-1/n)``."""
    return float(murray_ratios(asymmetry, exponent).mean())


def fit_asymmetry(target_ratio: Optional[float], exponent: float, notes: Optional[list] = None) -> float:
    """The asymmetry whose mean child/parent ratio is *target_ratio* (the
    geometric mean fall in mean diameter per order in the template), by
    bisection; 1 (symmetric) when the table asks for less fall than Murray's
    law allows, the floor when it asks for more."""
    if target_ratio is None or not math.isfinite(target_ratio):
        return 0.8
    top = mean_child_ratio(1.0, exponent)
    bottom = mean_child_ratio(MIN_ASYMMETRY, exponent)
    if target_ratio >= top:
        if notes is not None and target_ratio > top * 1.02:
            notes.append(
                f"the template's diameters fall by {target_ratio:.2f} per order, less than "
                f"Murray's law allows ({top:.2f} at exponent {exponent:g}); symmetric bifurcations are drawn"
            )
        return 1.0
    if target_ratio <= bottom:
        if notes is not None:
            notes.append(
                f"the template's diameters fall by {target_ratio:.2f} per order, more than a "
                f"bifurcation of asymmetry {MIN_ASYMMETRY} gives; that asymmetry is drawn"
            )
        return MIN_ASYMMETRY
    low, high = MIN_ASYMMETRY, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if mean_child_ratio(mid, exponent) < target_ratio:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def diameter_fall_per_order(orders: Sequence[OrderTarget]) -> Optional[float]:
    """The geometric mean of consecutive orders' mean-diameter ratios."""
    ratios = []
    for before, after in zip(orders[:-1], orders[1:]):
        if before.mean_diameter_um and after.mean_diameter_um and before.mean_diameter_um > 0:
            ratios.append(after.mean_diameter_um / before.mean_diameter_um)
    if not ratios:
        return None
    return float(np.exp(np.mean(np.log(ratios))))


def murray_deflections(parent_d: float, daughters: Sequence[float]) -> np.ndarray:
    """The share of a daughter-daughter angle each of two daughters deflects
    by: Murray's optimum, ``cos t_i = (d0^4 + d_i^4 - d_j^4) / (2 d0^2 d_i^2)``,
    normalised (equal shares when it degenerates)."""
    d0, d1, d2 = float(parent_d), float(daughters[0]), float(daughters[1])
    shares = []
    for di, dj in ((d1, d2), (d2, d1)):
        cos = (d0**4 + di**4 - dj**4) / (2.0 * d0**2 * di**2) if d0 > 0 and di > 0 else 1.0
        shares.append(math.acos(float(np.clip(cos, -1.0, 1.0))))
    total = sum(shares)
    if total <= 1e-9:
        return np.array([0.5, 0.5])
    return np.array(shares) / total


# ---------------------------------------------------------------------------
# Generation planning
# ---------------------------------------------------------------------------


def children_per_tip(tips: int, wanted: int, notes: Optional[list] = None, order: str = "") -> list[int]:
    """How many children each growing tip gets so that their sum is
    *wanted*: ``min(tips, wanted // 2)`` tips grow, every one at least two
    children (one would be a degree-2 node), the extra spread one at a time.
    A single wanted vessel becomes two."""
    if wanted <= 0 or tips <= 0:
        return []
    if wanted == 1:
        if notes is not None:
            notes.append(f"{order}: one vessel wanted but a tip cannot grow one child; two drawn")
        wanted = 2
    growing = min(tips, wanted // 2)
    counts = [2] * growing
    extra = wanted - 2 * growing
    i = 0
    while extra > 0:
        counts[i % growing] += 1
        extra -= 1
        i += 1
    if max(counts) > 3 and notes is not None:
        notes.append(f"{order}: {wanted} vessels from {tips} tips need junctions of {max(counts)} children")
    return counts


def trunk_children_per_tip(tips: int, wanted: int, notes: Optional[list] = None, order: str = "") -> list[int]:
    """How many children each growing tip of the large tier gets so that
    their sum is *wanted*: a trunk may go on as one child (its side branch
    goes to the bed), so ``min(tips, wanted)`` tips grow one each and the
    rest of *wanted* is spread one at a time, bifurcations first."""
    if wanted <= 0 or tips <= 0:
        return []
    growing = min(tips, wanted)
    counts = [1] * growing
    extra = wanted - growing
    i = 0
    while extra > 0:
        counts[i % growing] += 1
        extra -= 1
        i += 1
    if max(counts) > 3 and notes is not None:
        notes.append(f"{order}: {wanted} vessels from {tips} tips need junctions of {max(counts)} children")
    return counts


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.normal(size=3))


def _frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = _unit(np.cross(direction, helper))
    n2 = _unit(np.cross(direction, n1))
    return n1, n2


def _tilted(direction: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    n1, n2 = _frame(direction)
    tilt = math.radians(rng.uniform(0.0, max_deg))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return _unit(math.cos(tilt) * direction + math.sin(tilt) * (math.cos(phi) * n1 + math.sin(phi) * n2))


def inward_normal(face: str) -> Optional[np.ndarray]:
    if face == INTERIOR or face not in FACES:
        return None
    axis = "zyx".index(face[0])
    normal = np.zeros(3)
    normal[axis] = 1.0 if face.endswith("0") else -1.0
    return normal


def face_distance_um(point: np.ndarray, extent: np.ndarray) -> float:
    return float(min(np.min(point), np.min(extent - point)))


def exit_fraction(start: np.ndarray, end: np.ndarray, extent: np.ndarray) -> float:
    """Where the segment *start* -> *end* leaves the box, as a fraction of
    it (1 when it stays inside)."""
    fraction = 1.0
    for axis in range(3):
        delta = end[axis] - start[axis]
        if delta > 0 and end[axis] > extent[axis]:
            fraction = min(fraction, (extent[axis] - start[axis]) / delta)
        elif delta < 0 and end[axis] < 0.0:
            fraction = min(fraction, (0.0 - start[axis]) / delta)
    return float(max(fraction, 0.0))


def fit_fraction(
    start: np.ndarray, end: np.ndarray, extent: np.ndarray, hole: Optional[Hole] = None, hole_margin: float = 0.0
) -> float:
    """How much of the segment *start* -> *end* fits: before it leaves the
    box (:func:`exit_fraction`) and, given a *hole*, before it comes within
    *hole_margin* of that (``Hole.entry_fraction``)."""
    fraction = exit_fraction(start, end, extent)
    if hole is None:
        return fraction
    return min(fraction, hole.entry_fraction(start, end, hole_margin))


def room_ahead_um(point: np.ndarray, direction: np.ndarray, extent: np.ndarray) -> float:
    """How far *point* can go along *direction* before it leaves the box."""
    room = math.inf
    for axis in range(3):
        if direction[axis] > 1e-12:
            room = min(room, (extent[axis] - point[axis]) / direction[axis])
        elif direction[axis] < -1e-12:
            room = min(room, -point[axis] / direction[axis])
    return float(max(room, 0.0))


def snap_to_face(point: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """*point* with the coordinate nearest a face set exactly on it."""
    both = np.concatenate([point, extent - point])
    nearest = int(np.argmin(both))
    out = point.copy()
    out[nearest % 3] = 0.0 if nearest < 3 else extent[nearest % 3]
    return out


# ---------------------------------------------------------------------------
# Growing one side
# ---------------------------------------------------------------------------


def _aim_into_layer(
    base: np.ndarray,
    tilt_deg: float,
    start: np.ndarray,
    chord: float,
    extent: np.ndarray,
    rng: np.random.Generator,
    first: np.ndarray,
    layer: TreeLayer,
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> tuple[np.ndarray, float]:
    """A direction for a root's first vessel within *tilt_deg* of *base*:
    of *first* and :data:`FIRST_VESSEL_AIMS` more, the one whose chord fits
    the box (and misses the *hole*) best and, among those, lies most in the
    template's tree cells."""
    best: Optional[tuple[tuple[float, float], np.ndarray, float]] = None
    for k in range(FIRST_VESSEL_AIMS + 1):
        direction = first if k == 0 else _tilted(base, tilt_deg, rng)
        fraction = fit_fraction(start, start + chord * direction, extent, hole, hole_margin)
        end = start + fraction * chord * direction
        key = (min(fraction, 1.0), layer.at(0.5 * (start + end), end) + min(room_ahead_um(end, direction, extent), chord) / max(chord, 1e-9))
        if best is None or key > best[0]:
            best = (key, direction, fraction)
    assert best is not None
    return best[1], best[2]


def _aim_first_vessel(
    base: np.ndarray,
    tilt_deg: float,
    start: np.ndarray,
    chord: float,
    extent: np.ndarray,
    rng: np.random.Generator,
    first: tuple[np.ndarray, float],
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> tuple[np.ndarray, float]:
    """A direction for a root's first vessel within *tilt_deg* of *base*,
    and how much of *chord* fits in the box (and misses the *hole*) along it
    (1 when all of it): the best of *first* (the direction drawn, which did
    not fit) and :data:`FIRST_VESSEL_AIMS` more tries, stopping at one that fits."""
    best_direction, best_fraction = first
    for _ in range(FIRST_VESSEL_AIMS):
        direction = _tilted(base, tilt_deg, rng)
        fraction = fit_fraction(start, start + chord * direction, extent, hole, hole_margin)
        if fraction > best_fraction:
            best_direction, best_fraction = direction, fraction
        if fraction >= 1.0:
            break
    return best_direction, best_fraction


class TreeLayer:
    """Where the template's trees of one side lie: per cell of its layout,
    that side's tree length over the median of the cells holding any,
    capped at 1, and :data:`LAYER_NEIGHBOUR_SHARE` of the best neighbour's
    in the cells round them, read between cell centres. A side the template
    has no tree length on reads both sides'."""

    def __init__(self, organisation: Organisation, side: str, extent_um: Sequence[float]):
        self.extent = np.asarray(extent_um, dtype=float)
        self.shape = tuple(organisation.shape)
        own = organisation.field("arterial_length_um" if side == "arterial" else "venous_length_um")
        if own.sum() <= 0:
            own = organisation.field("arterial_length_um") + organisation.field("venous_length_um")
        held = own[own > 0]
        value = np.clip(own / float(np.median(held)), 0.0, 1.0) if len(held) else np.zeros_like(own)
        around = LAYER_NEIGHBOUR_SHARE * maximum_filter(value, size=3, mode="constant", cval=0.0)
        self.value = np.maximum(value, around)
        self.active = bool(len(held))

    @classmethod
    def of(cls, template: NetworkTemplate, side: str) -> Optional["TreeLayer"]:
        if template.organisation is None:
            return None
        layer = cls(template.organisation, side, template.extent_um)
        return layer if layer.active else None

    def at(self, *points: np.ndarray) -> float:
        """The mean value over *points*, read between the cells' centres
        (trilinear), so it peaks where the template's trees are thickest
        rather than standing flat across a cell."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        coordinates = (points / self.extent * np.asarray(self.shape, dtype=float) - 0.5).T
        return float(map_coordinates(self.value, coordinates, order=1, mode="nearest").mean())


class _Occupancy:
    """Where vessels already are (the draft's nodes, index = node id), for
    keeping new ones apart."""

    def __init__(self, positions: Sequence[np.ndarray]):
        self._tree = cKDTree(np.stack([np.asarray(p, float) for p in positions])) if len(positions) else None

    def nearest(self, point: np.ndarray, exclude: Optional[int] = None) -> float:
        """Distance to the nearest node other than *exclude*."""
        if self._tree is None:
            return math.inf
        distances, indices = self._tree.query(point, k=2)
        for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if math.isfinite(distance) and int(index) != exclude:
                return float(distance)
        return math.inf


def _score(
    end: np.ndarray,
    extent: np.ndarray,
    occupancy: _Occupancy,
    spacing: float,
    parent: int,
    min_sep: float,
    layer: Optional[TreeLayer] = None,
    start: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
) -> float:
    nearest = occupancy.nearest(end, exclude=parent)
    if layer is None:
        placed = min(face_distance_um(end, extent), spacing)
    else:
        # In the layer, heading on along it, and with room to go on: a
        # plexus on a face runs along it, never into it.
        points = [end if start is None else 0.5 * (start + end), end]
        if direction is not None:
            for step in np.linspace(0.0, LAYER_LOOKAHEAD_SPACINGS, 3)[1:]:
                points.append(np.clip(end + step * spacing * direction, 0.0, extent))
        placed = LAYER_WEIGHT * spacing * layer.at(*points)
        if direction is not None:
            reach = LAYER_LOOKAHEAD_SPACINGS * spacing
            placed += spacing * min(room_ahead_um(end, direction, extent), reach) / reach
    score = placed + 0.5 * min(nearest, spacing)
    if nearest < min_sep:
        score -= 10.0 * spacing
    return score


def _place_children(
    tip: Tip,
    chords: Sequence[float],
    deflections: Sequence[float],
    extent: np.ndarray,
    occupancy: _Occupancy,
    spacing: float,
    min_sep: float,
    rng: np.random.Generator,
    layer: Optional[TreeLayer] = None,
    closed: bool = False,
    course: Optional[np.ndarray] = None,
    persistence: float = 0.0,
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, float, bool]]:
    """Each child's ``(end, direction, chord, cut at a face)``, the plane
    round the parent chosen by score; children that cannot stay inside the
    box are cut at the face they leave by, or, *closed*, turned back off
    the face or else shortened to end inside it (a plane needing neither
    chosen first, then the one that keeps the most of them). A child that
    would come within *hole_margin* of the *hole* is shortened to end before
    it, never cut there, the plane needing that least chosen first."""
    start = tip.direction
    n1, n2 = _frame(start)
    parent = np.asarray(tip.position, float)
    count = len(chords)
    best: Optional[tuple[tuple, list]] = None
    trunk = int(np.argmin(deflections)) if len(deflections) else -1
    rotation = rng.uniform(0.0, 2.0 * math.pi)
    for k in range(AZIMUTHS):
        phi = rotation + 2.0 * math.pi * k / AZIMUTHS
        normal = math.cos(phi) * n1 + math.sin(phi) * n2
        children = []
        inside = True
        kept = 1.0
        total = 0.0
        for i in range(count):
            theta = deflections[i]
            if count == 2:
                sign = 1.0 if i == 0 else -1.0
                side = sign * normal
            else:
                azimuth = phi + 2.0 * math.pi * i / count
                side = math.cos(azimuth) * n1 + math.sin(azimuth) * n2
            direction = _unit(math.cos(theta) * start + math.sin(theta) * side)
            end = parent + chords[i] * direction
            fraction = exit_fraction(parent, end, extent)
            cut = fraction < 1.0
            if cut:
                inside = False
                if closed:
                    mirrored = direction.copy()
                    outside = (end < 0.0) | (end > extent)
                    mirrored[outside] *= -1.0
                    if exit_fraction(parent, parent + chords[i] * mirrored, extent) >= 1.0:
                        direction, end = mirrored, parent + chords[i] * mirrored
                    else:
                        kept = min(kept, fraction)
                        end = parent + fraction * (1.0 - FIRST_VESSEL_MARGIN) * chords[i] * direction
                    cut = False
                else:
                    end = snap_to_face(parent + fraction * chords[i] * direction, extent)
            if hole is not None:
                into = hole.entry_fraction(parent, end, hole_margin)
                if into < 1.0:
                    inside = False
                    kept = min(kept, into)
                    end = parent + into * (1.0 - FIRST_VESSEL_MARGIN) * (end - parent)
                    cut = False
            total += _score(end, extent, occupancy, spacing, tip.node, min_sep, layer, parent, direction)
            if course is not None and i == trunk:
                # The trunk going on keeps to its tree's course from the root.
                total += persistence * spacing * float(np.dot(direction, course))
            children.append((end, direction, float(np.linalg.norm(end - parent)), cut))
        key = (inside, kept, total) if closed else (inside, total)
        if best is None or key > best[0]:
            best = (key, children)
    assert best is not None
    return best[1]


def grow_side(
    draft: NetworkDraft,
    template: NetworkTemplate,
    side: str,
    roots: Sequence[tuple[int, str]],
    *,
    rng: np.random.Generator,
    spacing_um: float,
    extent_um: Sequence[float],
    min_separation_um: float,
    daughter_angle_sd_deg: Optional[float] = None,
    closed: bool = False,
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> SideResult:
    """Grow the ``"arterial"`` or ``"venous"`` trees from *roots*
    (``(node, face)`` pairs already in *draft*); *closed*, none of them
    leaving the box; given a *hole*, every vessel ending at least
    *hole_margin* off it. See the module."""
    extent = np.asarray(extent_um, dtype=float)
    result = SideResult(side=side)
    large_tier, small_tier = template.orders_on(side)
    large_category, small_category = _SIDE_CATEGORIES[side]
    generations: list[tuple[OrderTarget, str, bool]] = [(o, large_category, True) for o in large_tier] + [
        (o, small_category, False) for o in small_tier
    ]
    if not generations or not roots:
        return result
    exponent = template.murray_exponent if template.murray_exponent else DEFAULT_MURRAY_EXPONENT
    angle_mean = template.daughter_angle_deg if template.daughter_angle_deg else DEFAULT_DAUGHTER_ANGLE_DEG
    angle_sd = (
        daughter_angle_sd_deg
        if daughter_angle_sd_deg is not None
        else (template.daughter_angle_sd_deg if template.daughter_angle_sd_deg else DEFAULT_DAUGHTER_ANGLE_SD_DEG)
    )
    # Murray's law holds in the small tier only: its asymmetry is fitted to
    # the small tier's fall in width from the large tier's last order on.
    asymmetry = fit_asymmetry(
        diameter_fall_per_order(list(large_tier[-1:]) + list(small_tier)) if small_tier else None,
        exponent,
        draft.notes,
    )
    capillaries = template.class_target("capillary")
    side_branch_um = (capillaries.mean_diameter_um if capillaries else None) or FALLBACK_SIDE_BRANCH_UM
    layer = TreeLayer.of(template, side)
    # The template's side branches: how many of each order's vessels end its
    # tree (organisation.tree_leaves_by_order; Harvey, grid D1b).
    organisation = template.organisation
    leaves_by_order = dict(organisation.tree_leaves_by_order or ()) if organisation is not None else {}
    persistence = TRUNK_PERSISTENCE if organisation is not None and organisation.trunk_straightness is not None else 0.0

    # Generation 1: one vessel from each root, extra ones in pairs so a root
    # is a terminal or a junction, never a degree-2 node.
    first_order, first_category, first_large = generations[0]
    first_class = template.class_target(first_category)
    wanted = max(first_order.edge_count, len(roots))
    per_root = [1] * len(roots)
    extra = wanted - len(roots)
    i = 0
    while extra >= 2:
        per_root[i % len(roots)] += 2
        extra -= 2
        i += 1
    if extra == 1:
        draft.note(f"{first_order.label}: one vessel of {first_order.edge_count} dropped (a root cannot grow a lone second vessel)")
    if first_order.edge_count < len(roots):
        draft.note(
            f"{first_order.label}: {len(roots)} roots but {first_order.edge_count} vessels in the template; one per root drawn"
        )
    tips: list[Tip] = []
    occupancy = _Occupancy(draft.positions)
    first_lengths = draw_lengths(first_order, first_class, sum(per_root), rng)
    first_diameters = draw_root_diameters(first_order, first_class, sum(per_root), rng)
    for (root, face), count in zip(roots, per_root):
        normal = inward_normal(face)
        base = _random_unit(rng) if normal is None else normal
        root_position = draft.position(root)
        for j in range(count):
            tilt = ROOT_TILT_DEG if count == 1 else 60.0
            direction = _tilted(base, tilt, rng)
            diameter = first_diameters.pop()
            length = first_lengths.pop()
            tortuosity = draw_tortuosity(first_order, first_class, rng)
            chord = length / tortuosity
            fraction = fit_fraction(root_position, root_position + chord * direction, extent, hole, hole_margin)
            if layer is not None:
                direction, fraction = _aim_into_layer(
                    base, tilt, root_position, chord, extent, rng, direction, layer, hole, hole_margin
                )
            elif fraction < 1.0:
                direction, fraction = _aim_first_vessel(
                    base, tilt, root_position, chord, extent, rng, (direction, fraction), hole, hole_margin
                )
            # A root's first vessel stays in the box: re-aimed above, else
            # shortened to end inside it. Cut at a face, its tree would be
            # that one vessel.
            if fraction < 1.0:
                kept = fraction * (1.0 - FIRST_VESSEL_MARGIN)
                draft.note(
                    f"{first_order.label}: a first vessel {length:.0f} um long has no room in the box; "
                    f"{length * kept:.0f} um drawn"
                )
                chord *= kept
                length *= kept
            end = root_position + chord * direction
            cut = False
            node = draft.add_node(end)
            result.nodes.append(node)
            result.trees.setdefault(root, []).append(node)
            draft.add_edge(root, node, category=first_category, order=first_order.label,
                           length_um=length, diameter_um=diameter, part=TREE)
            result.edges += 1
            tips.append(Tip(node=node, position=end, direction=_unit(end - root_position), diameter_um=diameter, generation=1, root=root, at_face=cut))
    occupancy = _Occupancy(draft.positions)

    previous_large = first_large
    previous_label = first_order.label
    for order, category, is_large in generations[1:]:
        cls = template.class_target(category)
        growable = [t for t in tips if not t.at_face]
        stop = min(leaves_by_order.get(previous_label, 0), max(len(growable) - 1, 0))
        if stop:
            # As many of the last order's tips end the tree here as the
            # template's do: those already out of its tree cells, then the
            # narrowest (a side branch is narrower than the trunk going on).
            # Ranked by where a tip was heading instead, the ones at the
            # tree's frontier ended and the trees grew into bushes.
            def outside(t: Tip) -> bool:
                return layer is not None and layer.at(t.position) < LAYER_INSIDE

            ranked = sorted(growable, key=lambda t: (not outside(t), t.diameter_um, t.node))
            ended = {t.node for t in ranked[:stop]}
            growable = [t for t in growable if t.node not in ended]
        trunk = is_large and previous_large
        if trunk:
            counts = trunk_children_per_tip(len(growable), order.edge_count, draft.notes, order.label)
        else:
            counts = children_per_tip(len(growable), order.edge_count, draft.notes, order.label)
        if len(counts) < len(growable):
            weights = np.array([
                (
                    min(face_distance_um(t.position, extent), spacing_um)
                    if layer is None
                    else _score(t.position, extent, occupancy, spacing_um, t.node, 0.0, layer, None, t.direction)
                    - 0.5 * min(occupancy.nearest(t.position, exclude=t.node), spacing_um)
                )
                + 0.5 * min(occupancy.nearest(t.position, exclude=t.node), spacing_um)
                + 1e-6
                for t in growable
            ])
            chosen_index = rng.choice(len(growable), size=len(counts), replace=False, p=weights / weights.sum())
            growing = [growable[i] for i in sorted(int(c) for c in chosen_index)]
        else:
            growing = growable
        if previous_large and not is_large:
            # The tips the small tier grows from are the large tier's hand-over nodes.
            for t in growing:
                result.large_boundary.append(t.node)
        leaves = [t for t in tips if t not in growing]
        result.leaves.extend(leaves)
        next_tips: list[Tip] = []
        generation_lengths = draw_lengths(order, cls, sum(counts), rng)
        # The large tier's widths are its orders' own (see the module).
        generation_widths = draw_root_diameters(order, cls, sum(counts), rng) if is_large else None
        for tip, count in zip(growing, counts):
            if generation_widths is not None:
                diameters = np.array([generation_widths.pop() for _ in range(count)])
            else:
                a = float(np.clip(rng.uniform(asymmetry - ASYMMETRY_SPREAD, asymmetry + ASYMMETRY_SPREAD), MIN_ASYMMETRY, 1.0))
                ratios = murray_ratios(a, exponent, count)
                rng.shuffle(ratios)
                diameters = tip.diameter_um * ratios
            lengths = [generation_lengths.pop() for _ in range(count)]
            tortuosities = [draw_tortuosity(order, cls, rng) for _ in range(count)]
            chords = [length / tort for length, tort in zip(lengths, tortuosities)]
            angle = math.radians(float(np.clip(rng.normal(angle_mean, angle_sd), MIN_DAUGHTER_ANGLE_DEG, MAX_DAUGHTER_ANGLE_DEG)))
            if count == 2:
                deflections = list(angle * murray_deflections(tip.diameter_um, diameters))
            elif count == 1:
                # The trunk going on bends away from its side branch by
                # Murray's optimum share of the angle between them.
                deflections = [angle * float(murray_deflections(tip.diameter_um, [diameters[0], side_branch_um])[0])]
                result.handoffs.append(tip.node)
            else:
                deflections = [angle / 2.0] * count
            course = None
            if persistence and tip.root >= 0:
                away = np.asarray(tip.position, float) - np.asarray(draft.position(tip.root), float)
                course = _unit(away) if float(np.linalg.norm(away)) > 0 else None
            children = _place_children(
                tip, chords, deflections, extent, occupancy, spacing_um, min_separation_um, rng, layer, closed,
                course, persistence, hole, hole_margin,
            )
            for (end, direction, chord, cut), diameter, tort in zip(children, diameters, tortuosities):
                node = draft.add_node(end)
                result.nodes.append(node)
                result.trees.setdefault(tip.root, []).append(node)
                draft.add_edge(tip.node, node, category=category, order=order.label,
                               length_um=chord * tort, diameter_um=float(diameter), part=TREE)
                result.edges += 1
                next_tips.append(Tip(node=node, position=end, direction=direction, diameter_um=float(diameter), generation=tip.generation + 1, root=tip.root, at_face=cut))
        occupancy = _Occupancy(draft.positions)
        tips = next_tips
        previous_large = is_large
        previous_label = order.label
    result.leaves.extend(tips)
    return result


__all__ = [
    "AZIMUTHS",
    "SideResult",
    "Tip",
    "TreeLayer",
    "children_per_tip",
    "diameter_fall_per_order",
    "draw_length",
    "draw_lengths",
    "draw_root_diameter",
    "draw_root_diameters",
    "draw_tortuosity",
    "exit_fraction",
    "face_distance_um",
    "fit_fraction",
    "fit_asymmetry",
    "grow_side",
    "inward_normal",
    "mean_child_ratio",
    "murray_deflections",
    "murray_ratios",
    "room_ahead_um",
    "snap_to_face",
    "trunk_children_per_tip",
]
