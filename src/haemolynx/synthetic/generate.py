"""Draw a synthetic vascular network to a :class:`~.template.NetworkTemplate`.

:func:`generate_network` places the template's inlets and outlets on the
faces they were on, grows the arterial and venous trees from them
(:mod:`.trees`), hands their leaves to a capillary bed built to the
template's capillary count, loop count and dead-end count (:mod:`.bed`),
gives every vessel a length and a width drawn from the template, bends each
into a polyline of that length (:mod:`.polylines`) and returns the
``nx.MultiGraph`` the pipeline's Boundaries stage takes: node ``pos`` and
edge ``voxels`` in (z, y, x) microns, ``length``, ``diameter_um`` with
``diameter_source="synthetic"`` and the template's ``diameter_basis`` (so each
vessel's viscosity takes the form of the law the template's widths were
measured for, whatever this run's ``segmentation_source``), the node roles in :data:`SYNTHETIC_ROLES`
(:func:`boundary_nodes` reads them), ``dead_end_kind`` on every dead end,
and the image extent and voxel size on the graph. Nothing here is a branch
order: ``graph.assign_vessel_branch_orders`` labels the network from the
roles, as it does an image run's.

**A closed network** (``open_ends="closed"``, the default; Harvey,
2026-10-25: D2a) has no vessel leaving its box: the trees' children are kept
inside it and the bed draws no ends at the faces, so blood enters only at the
inlets and leaves only at the outlets. ``"template"`` cuts as many vessels at
the faces as the template had leaving its image.

**The template's dead ends not drawn are joined in pairs**: a closed network's
vessels the template had leaving its image, and, without interior dead ends
(the default; D3a), its dead ends inside the tissue, are each taken as half a
vessel broken in two, so every pair is one capillary fewer and one loop more,
the capillaries' total length kept (:func:`dead_end_pairs_joined`). Kept at
the template's count instead, a bed without dead ends needs every vertex a
junction: on E14.5 that drew 422 loops against its 231.

**A hole** (``GenerationOptions.hole``, ``synthetic.hole``) is a box the
network leaves empty: an image's own box, inside a surround drawn round it
(``PLAN_synthetic_surround.md``). Its share of the template's counts is not
drawn (``NetworkTemplate.around_hole``); the trees end before it, the bed lays
no vertex in it and joins none across it, every bow is kept out of it (else
drawn straight), and the graph records it (:data:`hole.GRAPH_HOLE`), which
:func:`validate_synthetic_graph` checks. Without one, nothing here asks a hole
anything.

Every draw goes through one ``numpy`` generator, so a seed repeats a network
exactly. :func:`validate_synthetic_graph` checks the contract.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.dead_ends import (
    BLIND,
    DEAD_END_KIND,
    DEAD_END_KIND_SOURCE,
    LEAVES_IMAGE,
    SOURCE_SYNTHETIC,
    graph_image_extent_um,
)
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes
from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_SYNTHETIC, EDGE_DIAMETER_BASIS

from .bed import (
    BED_MODEL_DELAUNAY,
    FACE_MARGIN_SPACINGS,
    build_bed,
    degree_sequence_targets,
    drawn_from_template_vertices,
)
from .draft import BED, TREE, TRUNK, NetworkDraft
from .hole import FACE_TOLERANCE_UM, GRAPH_HOLE, Hole
from .polylines import DEFAULT_POINTS, bend_path, bend_to_lengths, densify_paths
from .template import (
    DEFAULT_DAUGHTER_ANGLE_DEG,
    GRAPH_NETWORK_SOURCE,
    FACES,
    GRAPH_TEMPLATE_RECORD,
    INTERIOR,
    ROLES,
    SCALE_AXES_XYZ,
    NetworkTemplate,
    lognormal_from_mean_median,
    lognormal_from_median_p10_p90,
)
from .trees import (
    LENGTH_CLIP,
    TORTUOSITY_LOG_SIGMA,
    SideResult,
    face_distance_um,
    grow_side,
    inward_normal,
    room_ahead_um,
    snap_to_face,
)

#: The schema's seeds' default (``pipeline/schema.py`` repeats it; it stays
#: importable without numpy).
DEFAULT_SEED = 20240917
GENERATOR_VERSION = 1

#: Node attribute: the roles the generator gave a node, a tuple from :data:`ROLES`.
SYNTHETIC_ROLES = "synthetic_roles"
#: Edge attribute: the vessel class the generator drew a vessel as.
SYNTHETIC_CLASS = "synthetic_class"
#: Edge attribute the branch-order hierarchy reads a Large_Art/Large_Ven tier
#: from -- ``graph.branch_order.LARGE_MASK_VESSEL_TYPE``, the large mask a
#: vessel lies in -- repeated here so a generated graph carries it whatever
#: that module knows. A large-tier vessel is tagged with its side.
LARGE_MASK_VESSEL_TYPE = "large_mask_vessel_type"
_LARGE_MASK_SIDE = {"large_arteriole": "arteriole", "large_venule": "venule"}
#: What ``G.graph[GRAPH_NETWORK_SOURCE]`` holds on a generated graph.
SYNTHETIC = "synthetic"

#: A boundary leaf takes this many bed edges (with its tree edge, a 3-way junction).
LEAF_BED_EDGES = 2
#: A trunk's hand-off node takes one (with its two trunk edges, a 3-way junction).
HANDOFF_BED_EDGES = 1
#: A root attached to the bed directly (no tree on its side) is a terminal.
ROOT_BED_EDGES = 1
#: Two roots closer than this share of the box's shortest side are moved apart.
ROOT_SEPARATION_FRACTION = 0.02
#: Capillary diameter and length when the template gives none.
FALLBACK_CAPILLARY_DIAMETER_UM = 5.0
FALLBACK_CAPILLARY_LENGTH_UM = 40.0
#: Rank matching's noise: chords are ranked with this much log-normal jitter,
#: so equal chords do not all get the same length.
RANK_JITTER_SIGMA = 0.25
#: Trees grown per side for a template that records its trees' reach; the
#: one whose leaves reach nearest the template's is kept (Harvey's grid D1c:
#: the drawn networks taken for the tissue were those whose venous trees
#: reached across the box as the template's do).
TREE_CANDIDATES = 6
#: How far a capillary's excess over its chord may be stretched past the
#: tortuosity drawn for it to make up the template's capillary length.
MAX_TORTUOSITY_SCALE = 3.0
#: The share of the bend's perpendicular wave a tree vessel keeps when the
#: template's trees' curvature is read per vessel (:func:`_cross_wave`).
TREE_CROSS_WAVE = 0.0
#: The tree vessels' fine wobble (:func:`_with_fine_wobble`): the largest
#: amplitude tried (um) and the bisection's steps.
WOBBLE_MAX_UM = 2.0
WOBBLE_BISECTIONS = 12
#: The standard normal's 90th percentile.
_Z90 = 1.2815515655446004
#: Whether vessels leave the box (the ``synthetic_open_ends`` choices).
OPEN_ENDS_CLOSED = "closed"
OPEN_ENDS_TEMPLATE = "template"
OPEN_ENDS = (OPEN_ENDS_CLOSED, OPEN_ENDS_TEMPLATE)


@dataclass(frozen=True)
class GenerationOptions:
    scale: float = 1.0
    #: ``template.SCALE_AXES``: the axes *scale* grows the box along, all
    #: three or x and y only (the ``synthetic_scale_axes`` setting).
    scale_axes: str = SCALE_AXES_XYZ
    #: The template's dead ends inside the tissue drawn too (the
    #: ``synthetic_interior_dead_ends`` setting; off by default, D3a).
    interior_dead_ends: bool = False
    max_bed_degree: int = 4
    daughter_angle_sd_deg: Optional[float] = None
    #: None: half the bed spacing.
    min_separation_um: Optional[float] = None
    voxel_step_um: float = 0.5
    bend_points: int = DEFAULT_POINTS
    #: ``bed.BED_MODELS``: the Delaunay fill of a jittered lattice, or the
    #: edges of a Voronoi tessellation (the ``synthetic_bed_model`` setting).
    bed_model: str = BED_MODEL_DELAUNAY
    #: :data:`OPEN_ENDS`: a closed network, or the template's vessels
    #: leaving the image (the ``synthetic_open_ends`` setting).
    open_ends: str = OPEN_ENDS_CLOSED
    #: A box the network leaves empty (``synthetic.hole``: an image's own
    #: box, in a surround), its share of the template's counts not drawn;
    #: None fills the whole box.
    hole: Optional[Hole] = None
    #: The arterial and venous trees' density in the part of the box beyond
    #: the template's own, as a share of the template's (the
    #: ``synthetic_extension_*_density`` settings; :func:`drawn_template`):
    #: an image framed round its large vessels has more of them than the
    #: tissue round it.
    arteriole_density: float = 1.0
    venule_density: float = 1.0


#: What a :class:`Join` is drawn as: a node the bed joins with one vessel
#: (a capillary's open end), or a trunk to the box's face (an arteriole's or
#: a venule's: ``PLAN_synthetic_surround.md``'s D5).
JOIN_BED = "bed"
JOIN_TRUNK = "trunk"
#: Node attribute: the caller's key of the join a node is (an image's own node).
SURROUND_JOIN = "surround_join"
#: A trunk leaves the hole at least this steeply (the cosine of its angle to
#: the face's outward normal); a heading lying flatter is turned out.
TRUNK_MIN_OUTWARD = 0.5
#: A trunk's step when the template gives its order and class no length (um).
FALLBACK_TRUNK_STEP_UM = 40.0


@dataclass(frozen=True)
class Join:
    """A node of a network outside the one drawn that the drawn one joins:
    an image's open end, on its hole's face. A :data:`JOIN_BED` join is
    joined by the bed with one capillary; a :data:`JOIN_TRUNK` join by a
    trunk of its own vessel's class and width out to the box's face, a new
    inlet (arterial) or outlet (venous) there, handing a vessel to the bed
    at each joint between. *heading*: the vessel's direction out of the
    hole; *key*: the caller's name for it, kept on the node
    (:data:`SURROUND_JOIN`)."""

    position_um: tuple
    kind: str
    key: Any = None
    side: Optional[str] = None
    heading: Optional[tuple] = None
    diameter_um: Optional[float] = None
    category: Optional[str] = None
    order: Optional[str] = None


def _hole_face_normal(hole: Optional[Hole], point: np.ndarray) -> Optional[np.ndarray]:
    """The outward normal of the face of *hole* nearest *point*."""
    if hole is None:
        return None
    gaps = np.concatenate([np.abs(point - hole.lo), np.abs(point - hole.hi)])
    nearest = int(np.argmin(gaps))
    normal = np.zeros(3)
    normal[nearest % 3] = -1.0 if nearest < 3 else 1.0
    return normal


def _draw_trunk(
    draft: NetworkDraft,
    join: Join,
    template: NetworkTemplate,
    extent: np.ndarray,
    hole: Optional[Hole],
    min_step_um: float = 0.0,
) -> list[int]:
    """A trunk from *join* out to the box's face: its nodes, the join's
    first and the one on the face last. Straight out along the join's
    heading (turned out of the hole where it lies flatter than
    :data:`TRUNK_MIN_OUTWARD`), cut into its order's mean chord but never
    shorter than *min_step_um* (two pieces at least), each piece at the
    join's width and its order's tortuosity."""
    start = np.asarray(join.position_um, dtype=float)
    outward = _hole_face_normal(hole, start)
    heading = outward if join.heading is None else np.asarray(join.heading, dtype=float)
    heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
    if outward is not None and float(heading @ outward) < TRUNK_MIN_OUTWARD:
        heading = heading + outward
        heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
        if float(heading @ outward) < TRUNK_MIN_OUTWARD:
            heading = outward
    reach = room_ahead_um(start, heading, extent)
    order = next((o for o in template.orders if o.label == join.order), None)
    cls = template.class_target(join.category) if join.category else None

    def first(*values: Any) -> Optional[float]:
        return next((float(v) for v in values if v), None)

    tortuosity = max(first(order and order.mean_tortuosity, cls and cls.mean_tortuosity) or 1.0, 1.0)
    step = (first(order and order.mean_length_um, cls and cls.mean_length_um) or FALLBACK_TRUNK_STEP_UM) / tortuosity
    # An image's own template can hold an order only as the sliver its face
    # cut (E14.5's harness: one of 0.44 um, a trunk of 400 joints, each
    # wanting a bed vessel of the few vertices near it).
    step = max(step, float(min_step_um))
    diameter = first(join.diameter_um, order and order.mean_diameter_um, cls and cls.mean_diameter_um) or 5.0
    # At least two: a joint between hands a vessel to the bed, so every
    # trunk feeds or drains the bed round the image.
    pieces = max(2, int(round(reach / step)))
    nodes = [draft.add_node(start)]
    draft.marks[nodes[0]] = {SURROUND_JOIN: join.key}
    for k in range(1, pieces + 1):
        point = start + heading * reach * k / pieces
        nodes.append(draft.add_node(snap_to_face(point, extent) if k == pieces else point))
    for a, b in zip(nodes[:-1], nodes[1:]):
        chord = float(np.linalg.norm(draft.position(b) - draft.position(a)))
        draft.add_edge(
            a, b, category=join.category, order=join.order, length_um=chord * tortuosity, diameter_um=diameter, part=TRUNK
        )
    return nodes


@dataclass
class _Side:
    name: str
    roots: list
    result: Optional[SideResult] = None
    boundary_role: str = ""
    large_role: str = ""
    boundary_count: int = 0
    #: The count is exact: none left means no leaf is a boundary node (a
    #: large tier whose hand-offs took the template's whole count).
    counted: bool = False
    blind: list = None  # type: ignore[assignment]
    face: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.blind = []
        self.face = []


def _place_roots(
    draft: NetworkDraft,
    template: NetworkTemplate,
    extent: np.ndarray,
    rng: np.random.Generator,
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> dict:
    """One node per template inlet and outlet at its relative position,
    exactly on its face; duplicates (a scaled template's) spread apart. One
    that would lie in a *hole* or within *hole_margin* of it is moved along
    its face out of it."""
    placed: list[np.ndarray] = []
    minimum = ROOT_SEPARATION_FRACTION * float(np.min(extent))
    roots: dict[str, list] = {"inlet": [], "outlet": []}
    for role in ("inlet", "outlet"):
        for endpoint in getattr(template, f"{role}s"):
            position = np.asarray(endpoint.position_rel, float) * extent
            normal = inward_normal(endpoint.face)
            for _ in range(50):
                if all(np.linalg.norm(position - p) >= minimum for p in placed):
                    break
                shift = rng.uniform(-0.05, 0.05, size=3) * extent
                if normal is not None:
                    shift[np.argmax(np.abs(normal))] = 0.0
                position = np.clip(position + shift, 0.0, extent)
            if normal is not None:
                position = snap_to_face(position, extent)
                axis = int(np.argmax(np.abs(normal)))
                position[axis] = 0.0 if normal[axis] > 0 else extent[axis]
            if hole is not None:
                along = [a for a in range(3) if normal is None or a != int(np.argmax(np.abs(normal)))]
                position = hole.push_out(position, along, hole_margin, extent)
            placed.append(position)
            node = draft.add_node(position, role)
            roots[role].append((node, endpoint.face))
    return roots


def _choose_boundary_leaves(side: _Side, interior_dead_ends: bool, rng: np.random.Generator, draft: NetworkDraft) -> list:
    """Which interior leaves hand over to the bed: the template's count,
    deepest first; the rest end blind (or all hand over)."""
    result = side.result
    if result is None:
        return []
    interior = [t for t in result.leaves if not t.at_face]
    side.face = [t.node for t in result.leaves if t.at_face]
    chosen: list = []
    if interior:
        wanted = side.boundary_count if interior_dead_ends else len(interior)
        if wanted <= 0 and not side.counted:
            draft.note(f"{side.name} side: the template counts no boundary nodes; every tree leaf joins the bed")
            wanted = len(interior)
        if wanted >= len(interior):
            if wanted > len(interior):
                draft.note(
                    f"{side.name} side: {wanted} boundary nodes in the template, {len(interior)} tree leaves to give"
                )
            chosen = list(interior)
        else:
            keys = np.array([t.generation + rng.uniform(0.0, 0.5) for t in interior])
            order = np.argsort(-keys, kind="stable")[:wanted]
            chosen = [interior[int(i)] for i in sorted(int(i) for i in order)]
    chosen_nodes = {t.node for t in chosen}
    # A tree none of whose leaves joins the bed would hang loose: each root's
    # tree keeps its deepest interior leaf, or, with none, one at a face --
    # also when every leaf of the side is at a face (a root whose first
    # vessel left the box).
    for root, nodes in result.trees.items():
        own = set(nodes)
        if any(t.node in own for t in chosen):
            continue
        candidates = [t for t in interior if t.node in own] or [t for t in result.leaves if t.node in own]
        if candidates:
            keep = max(candidates, key=lambda t: t.generation)
            chosen.append(keep)
            chosen_nodes.add(keep.node)
            if keep.at_face:
                side.face = [n for n in side.face if n != keep.node]
    side.blind = [t.node for t in interior if t.node not in chosen_nodes]
    return [t.node for t in chosen]


def _draw_chain_lengths(template: NetworkTemplate, count: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """*count* interior dead-end chain lengths from the template's
    distribution, or None when it has none."""
    ends = template.dead_ends
    if count <= 0:
        return np.zeros(0)
    if ends.chain_length_samples_um:
        return rng.choice(np.asarray(ends.chain_length_samples_um, float), size=count, replace=True)
    if ends.chain_length_median_um:
        mu, sigma = lognormal_from_median_p10_p90(
            ends.chain_length_median_um,
            ends.chain_length_p10_um or ends.chain_length_median_um * 0.6,
            ends.chain_length_p90_um or ends.chain_length_median_um * 1.7,
        )
        return rng.lognormal(mu, sigma, size=count)
    return None


def _place_blind_ends(
    draft: NetworkDraft,
    template: NetworkTemplate,
    bed_edges: Sequence[Any],
    blind_edge: np.ndarray,
    blind_nodes: set,
    extent: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Draw each blind capillary's length from the chain distribution and
    move its dead end along its edge so the chord fits that length at the
    class's tortuosity; the lengths drawn, per bed edge (NaN elsewhere), or
    None when the template has no chain distribution."""
    indices = np.flatnonzero(blind_edge)
    drawn = _draw_chain_lengths(template, len(indices), rng)
    if drawn is None:
        return None
    cls = template.class_target("capillary")
    tortuosity = max(float((cls.mean_tortuosity if cls else None) or 1.1), 1.0)
    out = np.full(len(bed_edges), np.nan)
    for i, length in zip(indices, drawn):
        edge = bed_edges[int(i)]
        if edge.via:
            out[i] = length
            continue
        end, hub = (edge.u, edge.v) if edge.u in blind_nodes else (edge.v, edge.u)
        start, finish = draft.position(hub), draft.position(end)
        chord = float(np.linalg.norm(finish - start))
        wanted = float(length) / tortuosity
        if 0 < wanted < chord:
            moved = start + (finish - start) * (wanted / chord)
            draft.positions[end] = np.clip(moved, 0.0, extent)
        out[i] = length
    return out


def _capillary_lengths(
    template: NetworkTemplate,
    chords: np.ndarray,
    blind_edge: np.ndarray,
    blind_lengths: Optional[np.ndarray],
    rng: np.random.Generator,
    notes: list,
) -> np.ndarray:
    """A length per bed edge: the capillary distribution rank-matched to the
    chords (never below a chord), the chains ending blind at the lengths
    drawn for them, then one factor making the class total exact."""
    count = len(chords)
    if count == 0:
        return np.zeros(0)
    cls = template.class_target("capillary")
    samples = cls.samples.lengths_um if cls is not None and cls.samples is not None else ()
    if samples:
        drawn = rng.choice(np.asarray(samples, float), size=count, replace=True)
    else:
        mean = (cls.mean_length_um if cls else None) or FALLBACK_CAPILLARY_LENGTH_UM
        median = (cls.median_length_um if cls else None) or mean * 0.9
        mu, sigma = lognormal_from_mean_median(max(mean, median * 1.001), median)
        drawn = rng.lognormal(mu, sigma, size=count)
    ranked_chords = np.argsort(chords * rng.lognormal(0.0, RANK_JITTER_SIGMA, size=count), kind="stable")
    lengths = np.empty(count)
    lengths[ranked_chords] = np.sort(drawn)
    fixed = np.zeros(count, dtype=bool)
    if blind_lengths is not None:
        fixed = blind_edge & np.isfinite(blind_lengths)
        lengths[fixed] = blind_lengths[fixed]
    lengths = np.maximum(lengths, chords * (1.0 + 1e-6))
    target_total = cls.total_length_um if cls is not None and cls.total_length_um else None
    if target_total and (~fixed).any():
        free_target = target_total - float(lengths[fixed].sum())
        for _ in range(3):
            total = float(lengths[~fixed].sum())
            if total <= 0 or free_target <= 0:
                break
            scaled = np.maximum(lengths[~fixed] * (free_target / total), chords[~fixed] * (1.0 + 1e-6))
            lengths[~fixed] = scaled
        if abs(float(lengths.sum()) - target_total) > 0.02 * target_total:
            notes.append(
                f"capillary length {lengths.sum():.0f} um against the template's {target_total:.0f} um: "
                "the chords leave no room for shorter vessels"
            )
    return lengths


def _capillary_lengths_by_tortuosity(
    template: NetworkTemplate,
    chords: np.ndarray,
    blind_edge: np.ndarray,
    blind_lengths: Optional[np.ndarray],
    rng: np.random.Generator,
    notes: list,
) -> np.ndarray:
    """A length per bed edge, for a template that records its layout: each
    chord times a tortuosity drawn from the template's capillaries, the
    excess over the chord scaled alike on every vessel so the class total is
    the template's; the chains ending blind at the lengths drawn for them.

    Drawn from the length distribution instead (:func:`_capillary_lengths`),
    a vessel whose drawn length fell short of its chord was clamped to it and
    came out straight: on E14.5, where the chords leave the template's
    capillaries little more length than they span, 70-78% of them at worst
    (grid D1b), 12-17 degrees per 10 um against 26 in some seeds after."""
    count = len(chords)
    if count == 0:
        return np.zeros(0)
    cls = template.class_target("capillary")
    samples = cls.samples.tortuosities if cls is not None and cls.samples is not None else ()
    median = cls.median_tortuosity if cls is not None else None
    p90 = cls.p90_tortuosity if cls is not None else None
    if samples:
        tortuosity = np.maximum(rng.choice(np.asarray(samples, float), size=count, replace=True), 1.0)
    elif median is not None and p90 is not None and p90 > median > 1.0:
        # The excess over straight a lognormal through the template's median
        # and 90th percentile: its winding tail, which the mean with a fixed
        # spread drew short of (E14.5, grid D1d: P90 1.14-1.26 against 1.48).
        mu = math.log(median - 1.0)
        sigma = (math.log(p90 - 1.0) - mu) / _Z90
        tortuosity = 1.0 + rng.lognormal(mu, sigma, size=count)
    else:
        excess = max(float((cls.mean_tortuosity if cls else None) or 1.1) - 1.0, 0.0)
        sigma = TORTUOSITY_LOG_SIGMA
        tortuosity = 1.0 + excess * rng.lognormal(-(sigma**2) / 2.0, sigma, size=count)
    chords = np.maximum(chords, 1e-6)
    lengths = chords * tortuosity
    fixed = np.zeros(count, dtype=bool)
    if blind_lengths is not None:
        fixed = blind_edge & np.isfinite(blind_lengths)
        lengths[fixed] = np.maximum(blind_lengths[fixed], chords[fixed] * (1.0 + 1e-6))
    target_total = cls.total_length_um if cls is not None and cls.total_length_um else None
    free = ~fixed
    if target_total and free.any():
        room = target_total - float(lengths[fixed].sum()) - float(chords[free].sum())
        excess = float((chords[free] * (tortuosity[free] - 1.0)).sum())
        if room <= 0 or excess <= 0:
            notes.append(
                f"capillary length: the chords alone ({chords[free].sum():.0f} um) leave none of the template's "
                f"{target_total:.0f} um to curve in; drawn straight"
            )
            lengths[free] = chords[free] * (1.0 + 1e-6)
        else:
            scale = room / excess
            if scale > MAX_TORTUOSITY_SCALE:
                notes.append(
                    f"capillary length: the chords leave {room:.0f} um to curve in, more than "
                    f"{MAX_TORTUOSITY_SCALE:g} times the template's tortuosity; "
                    f"{target_total - room + MAX_TORTUOSITY_SCALE * excess:.0f} um drawn against its {target_total:.0f} um"
                )
                scale = MAX_TORTUOSITY_SCALE
            lengths[free] = chords[free] * (1.0 + (tortuosity[free] - 1.0) * scale)
    return lengths


def _capillary_diameters(template: NetworkTemplate, count: int, rng: np.random.Generator) -> np.ndarray:
    cls = template.class_target("capillary")
    samples = cls.samples.diameters_um if cls is not None and cls.samples is not None else ()
    if samples:
        return rng.choice(np.asarray(samples, float), size=count, replace=True)
    mean = (cls.mean_diameter_um if cls else None) or FALLBACK_CAPILLARY_DIAMETER_UM
    median = (cls.median_diameter_um if cls else None) or mean * 0.96
    mu, sigma = lognormal_from_mean_median(max(mean, median * 1.001), median)
    return rng.lognormal(mu, sigma, size=count)


def dead_end_pairs_joined(template: NetworkTemplate, options: Any) -> int:
    """How many pairs of *template*'s dead ends a network drawn with
    *options* (:class:`GenerationOptions`, or its record's dict) joins
    instead of drawing: its ends leaving the image other than its inlets and
    outlets when closed, its ends inside the tissue without interior dead
    ends. Each pair is one capillary fewer and one loop more."""
    def read(name: str, default: Any) -> Any:
        return options.get(name, default) if isinstance(options, Mapping) else getattr(options, name, default)

    missing = 0
    if read("open_ends", OPEN_ENDS_TEMPLATE) == OPEN_ENDS_CLOSED:
        roots_on_faces = sum(1 for end in (*template.inlets, *template.outlets) if end.face != INTERIOR)
        missing += max(int(template.dead_ends.leaves_image) - roots_on_faces, 0)
    if read("interior_dead_ends", True) is False:
        missing += int(template.dead_ends.interior)
    return missing // 2


def _leaf_reach(draft: NetworkDraft, result: SideResult) -> list[float]:
    """The straight distance from each of a grown side's leaves to its root."""
    return [
        float(np.linalg.norm(np.asarray(t.position, float) - draft.position(t.root)))
        for t in result.leaves
        if t.root >= 0
    ]


def _grow_side_to_reach(
    draft: NetworkDraft, template: NetworkTemplate, side: str, roots: Sequence, rng: np.random.Generator, **kwargs: Any
) -> SideResult:
    """One side's trees (:func:`trees.grow_side`). With the template's reach
    for the side (``organisation.tree_reach_um``), :data:`TREE_CANDIDATES` are
    grown, each from a seed drawn off *rng*, and the one whose leaves'
    median and furthest reach from their roots come nearest the template's
    is kept; without it, grown once from *rng* as before."""
    reach = dict(
        (s, (median, far)) for s, median, far in (template.organisation.tree_reach_um or ())
    ) if template.organisation is not None else {}
    if side not in reach:
        return grow_side(draft, template, side, roots, rng=rng, **kwargs)
    median, far = reach[side]
    seeds = rng.integers(0, 2**62, size=TREE_CANDIDATES)
    mark = (len(draft.positions), len(draft.edges), len(draft.notes))
    best: Optional[tuple[float, int]] = None
    for seed in seeds:
        result = grow_side(draft, template, side, roots, rng=np.random.default_rng(int(seed)), **kwargs)
        chords = _leaf_reach(draft, result)
        if chords:
            miss = abs(float(np.median(chords)) - median) / max(median, 1e-9) + abs(max(chords) - far) / max(far, 1e-9)
            if best is None or miss < best[0]:
                best = (miss, int(seed))
        del draft.positions[mark[0]:], draft.edges[mark[1]:], draft.notes[mark[2]:]
        for node in [n for n in draft.roles if n >= mark[0]]:
            del draft.roles[node]
    chosen = int(seeds[0]) if best is None else best[1]
    return grow_side(draft, template, side, roots, rng=np.random.default_rng(chosen), **kwargs)


def extension_share(
    template_extent_um: Sequence[float], extent_um: Sequence[float], hole: Optional[Hole] = None
) -> float:
    """The share of what a network in a box of *extent_um* draws that lies
    beyond the template's own box (*template_extent_um*): round a hole the
    template's size or larger (an image's own, which its graph fills), all
    of it; without one, the box's volume past the template's."""
    box = float(np.prod(np.asarray(extent_um, dtype=float)))
    left = hole.volume_in(extent_um) if hole is not None else 0.0
    drawn = box - left
    if drawn <= 0.0:
        return 0.0
    own = max(float(np.prod(np.asarray(template_extent_um, dtype=float))) - left, 0.0)
    return max(drawn - own, 0.0) / drawn


def face_extension_shares(
    template_extent_um: Sequence[float], extent_um: Sequence[float], hole: Optional[Hole] = None
) -> dict[str, float]:
    """Each face's share beyond the template's own (:data:`FACES`), where an
    extension's inlets and outlets lie: the face's area less what the hole
    covers of it, or with no hole the template's face; and
    :func:`extension_share` for an endpoint inside the box."""
    extent = np.asarray(extent_um, dtype=float)
    own = np.asarray(template_extent_um, dtype=float)
    shares = {INTERIOR: extension_share(own, extent, hole)}
    for face in FACES:
        axis = "zyx".index(face[0])
        across = [a for a in range(3) if a != axis]
        area = float(np.prod(extent[across]))
        if hole is None:
            covered = float(np.prod(np.minimum(own[across], extent[across])))
        elif (hole.lo[axis] if face[1] == "0" else extent[axis] - hole.hi[axis]) <= FACE_TOLERANCE_UM:
            sides = np.minimum(hole.hi[across], extent[across]) - np.maximum(hole.lo[across], 0.0)
            covered = float(np.prod(np.clip(sides, 0.0, None)))
        else:
            covered = 0.0
        shares[face] = max(area - covered, 0.0) / area if area > 0 else 0.0
    return shares


@dataclass(frozen=True)
class BedPlan:
    """The capillary bed a network is planned with before anything is drawn
    (:func:`planned_bed`): the template as drawn, its inlets and outlets less
    the trunks that take their place, and the bed's arithmetic."""

    template: NetworkTemplate
    trunks: Mapping[str, int]
    #: The drawn template's capillaries, and as many fewer as dead-end pairs
    #: are joined instead of drawn (each a loop more).
    template_capillaries: int
    capillaries: int
    loops: int
    joined: int
    #: One per inlet and outlet, and one per join (a trunk's is its end on the face).
    roots: int
    #: The bed's vertices: capillaries less loops less roots, plus one. Below
    #: 1 no bed can be drawn (the generator lays one and fails to join it).
    vertices: int


def planned_bed(template: NetworkTemplate, options: Any = None, joins: Sequence[Join] = ()) -> BedPlan:
    """The bed :func:`generate_network` plans for *template* drawn with
    *options* and *joins*: its own arithmetic, so a check can ask it before
    drawing (a surround's shell too thin for the image's open ends)."""
    T = drawn_template(template, options)
    trunks = {side: sum(1 for j in joins if j.kind == JOIN_TRUNK and j.side == side) for side in ("arterial", "venous")}
    if joins:
        T = replace(
            T,
            inlets=T.inlets[: max(len(T.inlets) - trunks["arterial"], 0)],
            outlets=T.outlets[: max(len(T.outlets) - trunks["venous"], 0)],
        )
    capillaries = T.class_target("capillary")
    template_capillaries = int(capillaries.count) if capillaries is not None else 0
    count = template_capillaries
    # One root per inlet and outlet (_place_roots), placed once the spacing
    # says how far off a hole they keep; a join is one more.
    roots = len(T.inlets) + len(T.outlets) + len(joins)
    loops = T.loops.independent_loops
    if loops is None:
        loops = int(round(count / 3.0))
    joined = dead_end_pairs_joined(T, GenerationOptions() if options is None else options)
    if joined:
        count = max(count - joined, 1)
        loops += joined
    return BedPlan(
        template=T,
        trunks=trunks,
        template_capillaries=template_capillaries,
        capillaries=count,
        loops=int(loops),
        joined=joined,
        roots=roots,
        vertices=count - int(loops) - roots + 1,
    )


def drawn_template(template: NetworkTemplate, options: Any = None, *, whole: bool = False) -> NetworkTemplate:
    """The template a network drawn with *options* (:class:`GenerationOptions`,
    or its record's dict) is drawn to: *template* scaled to the box, the
    hole's share left out (``NetworkTemplate.around_hole``), and the arterial
    and venous trees in the part beyond the template's own box
    (:func:`extension_share`) at their densities, the rest at the
    template's (``NetworkTemplate.with_side_factors``), their inlets and
    outlets so on each face (:func:`face_extension_shares`). Round a hole
    its layout is mirrored across the hole's faces
    (``Organisation.mirrored_to``, D6): the tissue beyond a face of an
    image is the tissue at it. The comparison scores a network against the
    same; with *whole*, a surround's network with the image filling its
    hole, whose share is then not left out."""
    def read(name: str, default: Any) -> Any:
        if options is None:
            return default
        value = options.get(name, default) if isinstance(options, Mapping) else getattr(options, name, default)
        return default if value is None else value

    T = template.scaled(float(read("scale", 1.0)), str(read("scale_axes", SCALE_AXES_XYZ)))
    extent = np.asarray(T.extent_um, dtype=float)
    hole = read("hole", None)
    if isinstance(hole, Mapping):
        hole = Hole.from_record(hole)
    counted = None if whole else hole
    if counted is not None:
        # Tissue the network leaves to an image's own graph: its share of
        # the template's counts is not drawn.
        T = T.around_hole(counted.share_of(extent))
    shares = face_extension_shares(template.extent_um, extent, counted)
    arteriole_density = float(read("arteriole_density", 1.0))
    venule_density = float(read("venule_density", 1.0))
    T = T.with_side_factors(
        1.0 + shares[INTERIOR] * (arteriole_density - 1.0),
        1.0 + shares[INTERIOR] * (venule_density - 1.0),
        inlet_factors={face: 1.0 + share * (arteriole_density - 1.0) for face, share in shares.items()},
        outlet_factors={face: 1.0 + share * (venule_density - 1.0) for face, share in shares.items()},
    )
    if hole is not None and T.organisation is not None:
        T = replace(T, organisation=T.organisation.mirrored_to(extent, hole.lo, hole.hi - hole.lo))
    return T


def _template_record(template: NetworkTemplate) -> dict:
    """The template as a plain dict without its per-vessel samples."""
    data = template.to_dict()
    for item in data["classes"].values():
        item["samples"] = None
    for item in data["orders"]:
        item["samples"] = None
    data["loops"]["loop_length_samples_um"] = []
    data["dead_ends"]["chain_length_samples_um"] = []
    if data.get("organisation"):
        data["organisation"]["capillary_node_positions_rel"] = None
    return data


def generate_network(
    template: NetworkTemplate,
    *,
    seed: Optional[int] = DEFAULT_SEED,
    rng: Optional[np.random.Generator] = None,
    options: Optional[GenerationOptions] = None,
    validate: bool = True,
    joins: Sequence[Join] = (),
) -> nx.MultiGraph:
    """A synthetic network drawn to *template*. See the module.

    *rng* wins over *seed*; ``seed=None`` draws from fresh entropy. Raises
    ``ValueError`` for a template with no capillaries or no inlets, and,
    with *validate*, for a network that breaks the contract. *joins*
    (:class:`Join`) are nodes of another network the one drawn joins, its
    trunks taking the place of as many of the template's inlets and outlets.
    """
    options = options or GenerationOptions()
    if options.open_ends not in OPEN_ENDS:
        raise ValueError(f"open_ends must be one of {', '.join(OPEN_ENDS)}, not {options.open_ends!r}")
    closed = options.open_ends == OPEN_ENDS_CLOSED
    rng = rng if rng is not None else np.random.default_rng(seed)
    plan = planned_bed(template, options, joins)
    T, trunks = plan.template, dict(plan.trunks)
    extent = np.asarray(T.extent_um, dtype=float)
    hole = options.hole
    if (not T.inlets and not trunks["arterial"]) or (not T.outlets and not trunks["venous"]):
        raise ValueError("the template has no inlets or no outlets to draw from")
    if plan.template_capillaries < 1:
        raise ValueError("the template has no capillaries to draw a bed from")

    draft = NetworkDraft()
    # The bed's arithmetic first: its spacing is what the trees fill space to.
    capillary_count, loops, joined, root_count = plan.capillaries, plan.loops, plan.joined, plan.roots
    if joined:
        draft.note(
            f"{2 * joined} of the template's dead ends are not drawn: joined in pairs, "
            f"{joined} capillaries fewer and {joined} loops more"
        )
    bed_vertices = max(plan.vertices, 1)
    # The mean spacing; a bed laid out to the template's organisation has
    # its own spacing cell by cell (bed.density_vertices).
    volume = np.prod(extent) if hole is None else np.prod(extent) - hole.volume_in(extent)
    spacing = float((volume / bed_vertices) ** (1.0 / 3.0))
    min_separation = options.min_separation_um if options.min_separation_um is not None else 0.5 * spacing
    daughter_angle = T.daughter_angle_deg if T.daughter_angle_deg else DEFAULT_DAUGHTER_ANGLE_DEG
    # Off a hole's faces as the bed keeps off the box's.
    hole_margin = FACE_MARGIN_SPACINGS * spacing
    roots = _place_roots(draft, T, extent, rng, hole, hole_margin)

    sides = [
        _Side("arterial", roots["inlet"], boundary_role="arteriole_boundary", large_role="large_arteriole_boundary", boundary_count=T.arteriole_boundary_count),
        _Side("venous", roots["outlet"], boundary_role="venule_boundary", large_role="large_venule_boundary", boundary_count=T.venule_boundary_count),
    ]
    attached: dict[int, int] = {}
    tree_groups: list[list[int]] = []
    tree_nodes: list[int] = []
    for side in sides:
        large, small = T.orders_on(side.name)
        if not (large or small):
            # No tree on this side: the roots join the bed as terminals and
            # are their own boundary nodes (the hierarchy then stops at them).
            for node, _face in side.roots:
                attached[node] = ROOT_BED_EDGES
                tree_groups.append([node])
                if T.scheme == "hierarchical":
                    draft.add_role(node, side.boundary_role)
            continue
        side.result = _grow_side_to_reach(
            draft, T, side.name, side.roots, rng,
            spacing_um=spacing, extent_um=extent, min_separation_um=min_separation,
            daughter_angle_sd_deg=options.daughter_angle_sd_deg, closed=closed, hole=hole, hole_margin=hole_margin,
        )
        tree_nodes.extend(side.result.nodes)
        for node in side.result.large_boundary:
            draft.add_role(node, side.large_role)
        for node in side.result.handoffs:
            draft.add_role(node, side.large_role)
            attached[node] = HANDOFF_BED_EDGES
        if large and not small:
            # The large tier is the whole tree: its leaves, like its trunks'
            # side branches, are where it hands over to the bed, the
            # large-vessel boundary nodes, as the large masks alone leave a
            # run; the template's count of them less the hand-offs.
            side.boundary_role = side.large_role
            side.boundary_count = max(getattr(T, f"{side.large_role}_count") - len(side.result.handoffs), 0)
            side.counted = True
        for node in _choose_boundary_leaves(side, options.interior_dead_ends, rng, draft):
            draft.add_role(node, side.boundary_role)
            attached[node] = LEAF_BED_EDGES
        for root, nodes in side.result.trees.items():
            tree_groups.append([root] + list(nodes))

    trunk_nodes: list[int] = []
    for join in joins:
        if join.kind == JOIN_TRUNK:
            # Joints no closer than bed vertices may lie: each hands a vessel to the bed.
            nodes = _draw_trunk(draft, join, T, extent, hole, min_step_um=min_separation)
            side = sides[0] if join.side == "arterial" else sides[1]
            draft.add_role(nodes[-1], "inlet" if join.side == "arterial" else "outlet")
            for node in nodes[1:-1]:
                draft.add_role(node, side.large_role if str(join.category).startswith("large_") else side.boundary_role)
                attached[node] = HANDOFF_BED_EDGES
            tree_groups.append(nodes)
            trunk_nodes.extend(nodes)
        else:
            node = draft.add_node(join.position_um)
            draft.marks[node] = {SURROUND_JOIN: join.key}
            attached[node] = ROOT_BED_EDGES
            tree_groups.append([node])

    face_ends_on_trees = sum(len(side.face) for side in sides)
    blind_on_trees = sum(len(side.blind) for side in sides)
    roots_on_faces = sum(1 for side in sides for _node, face in side.roots if face != INTERIOR)
    bed_face_ends = 0 if closed else max(T.dead_ends.leaves_image - roots_on_faces - face_ends_on_trees, 0)
    interior_target = T.dead_ends.interior if options.interior_dead_ends else 0
    bed_blind_ends = max(interior_target - blind_on_trees, 0)
    targets = degree_sequence_targets(
        capillary_count=capillary_count,
        loops=loops,
        root_count=root_count,
        attached_caps=list(attached.values()),
        dead_ends=bed_face_ends + bed_blind_ends,
        max_degree=options.max_bed_degree,
    )
    for note in targets.notes:
        draft.note(note)
    # The lattice keeps clear of the trees' junctions and roots, not of
    # their leaves: a leaf needs junctions beside it to join the bed.
    parents = {e.u for e in draft.edges if e.part == TREE}
    keep_clear = [n for n in tree_nodes if n in parents] + [node for node, _f in roots["inlet"] + roots["outlet"]]
    keep_clear += trunk_nodes
    bed = build_bed(
        draft,
        extent_um=extent,
        attached=attached,
        tree_groups=tree_groups,
        keep_clear=keep_clear,
        targets=targets,
        face_ends=bed_face_ends,
        spacing_um=spacing,
        daughter_angle_deg=daughter_angle,
        rng=rng,
        max_degree=options.max_bed_degree,
        target_chord_um=_capillary_chord(T),
        organisation=T.organisation,
        model=options.bed_model,
        at_template_size=options.scale == 1.0,
        junction_angle_deg=_junction_angle(T),
        acute_share=None if T.layout is None else T.layout.acute_junction_share,
        hole=hole,
        hole_margin=hole_margin,
    )

    if bed.edges == 0:
        raise ValueError(
            "The template leaves no capillary bed to draw: its capillary count, loops, dead "
            f"ends and inlets/outlets cannot be put together ({'; '.join(draft.notes) or 'no bed vertex found a junction'})."
        )

    # The trees' vessels: their lengths scaled to the class totals.
    _rescale_tree_lengths(draft, T)

    # Lengths and widths of the bed's vessels; a blind end is set as far
    # from its junction as its chain length allows first.
    bed_edges = [e for e in draft.edges if e.part == BED]
    blind_nodes = set(bed.blind_ends)
    blind_edge = np.array([(e.u in blind_nodes) != (e.v in blind_nodes) for e in bed_edges], dtype=bool)
    blind_lengths = _place_blind_ends(draft, T, bed_edges, blind_edge, blind_nodes, extent, rng)
    chords = np.array([_chord(draft, e) for e in bed_edges]) if bed_edges else np.zeros(0)
    if T.organisation is not None:
        lengths = _capillary_lengths_by_tortuosity(T, chords, blind_edge, blind_lengths, rng, draft.notes)
    else:
        lengths = _capillary_lengths(T, chords, blind_edge, blind_lengths, rng, draft.notes)
    diameters = _capillary_diameters(T, len(bed_edges), rng)
    for edge, length, diameter in zip(bed_edges, lengths, diameters):
        edge.length_um = float(length)
        edge.diameter_um = float(diameter)

    G = _assemble(draft, T, extent, options, rng)
    G.graph[GRAPH_NETWORK_SOURCE] = SYNTHETIC
    G.graph[GRAPH_TEMPLATE_RECORD] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed if rng is None else seed,
        "options": asdict(options),
        "template": _template_record(T),
        "generated": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "capillaries": bed.edges,
            "bed_vertices": bed.vertices,
            "tree_edges": sum(1 for e in draft.edges if e.part == TREE),
            "loops": G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G),
            "repairs": bed.repairs,
            "merged": bed.merged,
            "dropped": bed.dropped,
            "short_of_degree": bed.short_of_degree,
            "fill_phases": dict(bed.phases),
            "tree_face_ends": face_ends_on_trees,
            "tree_blind_ends": blind_on_trees,
            "bed_face_ends": len(bed.face_ends),
            "bed_blind_ends": len(bed.blind_ends),
            "bed_face_ends_wanted": bed_face_ends,
            "bed_blind_ends_wanted": bed_blind_ends,
            "dead_end_pairs_joined": joined,
            "spacing_um": spacing,
            "bed_layout": (
                "even" if T.organisation is None
                else "template vertices" if drawn_from_template_vertices(T.organisation, options.scale == 1.0)
                else "template cells"
            ),
            "notes": list(draft.notes),
        },
    }
    if hole is not None:
        G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["hole_share"] = hole.share_of(extent)
    if joins:
        G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["joins"] = {
            "bed": sum(1 for j in joins if j.kind != JOIN_TRUNK),
            "trunks": dict(trunks),
            "trunk_edges": sum(1 for e in draft.edges if e.part == TRUNK),
        }
    if validate:
        validate_synthetic_graph(G, T)
    return G


def _junction_angle(template: NetworkTemplate) -> Optional[float]:
    """The template's own junction angle -- its junctions' three angles
    summed, over three (``statistics.layout``) -- for the bed to aim each
    pair of vessels at a vertex at; None for a template without a layout."""
    layout = template.layout
    if layout is None or layout.junction_angle_sum_median_deg is None:
        return None
    return float(layout.junction_angle_sum_median_deg) / 3.0


def _capillary_chord(template: NetworkTemplate) -> Optional[float]:
    """The chord (length over tortuosity) the template's capillaries have
    on average, for the bed to draw its edges near."""
    cls = template.class_target("capillary")
    if cls is None:
        return None
    samples = cls.samples
    if (
        samples is not None
        and samples.lengths_um
        and samples.tortuosities
        and len(samples.lengths_um) == len(samples.tortuosities)
    ):
        chords = np.asarray(samples.lengths_um, float) / np.maximum(np.asarray(samples.tortuosities, float), 1.0)
        return float(np.mean(chords))
    if cls.mean_length_um:
        return float(cls.mean_length_um) / max(float(cls.mean_tortuosity or 1.0), 1.0)
    return None


def _rescale_tree_lengths(draft: NetworkDraft, template: NetworkTemplate) -> None:
    """One factor per tree class so its vessels add up to the template's
    total length for that class, never below a vessel's chord, nor above
    the longest and most tortuous vessel the template's class has
    (:func:`_tree_length_caps`); what the caps leave short is noted."""
    by_class: dict = {}
    for edge in draft.edges:
        if edge.part == TREE:
            by_class.setdefault(edge.category, []).append(edge)
    for category, edges in by_class.items():
        cls = template.class_target(category)
        if cls is None or not cls.total_length_um:
            continue
        chords = {id(e): _chord(draft, e) for e in edges}
        longest, most_tortuous = _tree_length_caps(cls, template, category)
        floors = {id(e): chords[id(e)] * (1.0 + 1e-6) for e in edges}
        # The rescale never stretches a vessel past the caps; one drawn
        # beyond them is not shortened to them either.
        ceilings = {
            id(e): max(min(longest, chords[id(e)] * most_tortuous), floors[id(e)], float(e.length_um or 0.0))
            for e in edges
        }
        for _ in range(3):
            total = sum(float(e.length_um or 0.0) for e in edges)
            if total <= 0:
                break
            factor = cls.total_length_um / total
            for e in edges:
                e.length_um = min(max(float(e.length_um or 0.0) * factor, floors[id(e)]), ceilings[id(e)])
        total = sum(float(e.length_um or 0.0) for e in edges)
        if total < 0.98 * cls.total_length_um:
            draft.note(
                f"{category}: {total:.0f} um of the template's {cls.total_length_um:.0f} um drawn; "
                "longer vessels would be longer or more tortuous than any of the template's"
            )


def _tree_length_caps(cls: Any, template: NetworkTemplate, category: str) -> tuple[float, float]:
    """``(longest length, highest tortuosity)`` a tree vessel of *category*
    may be rescaled to: the template's own longest and most tortuous
    vessels of the class (the graph source), else the draws' own limits --
    :data:`trees.LENGTH_CLIP` times the class's median length, and the
    tortuosity two sigmas up the draw's spread."""
    samples = cls.samples
    lengths = list(samples.lengths_um) if samples is not None else []
    tortuosities = list(samples.tortuosities) if samples is not None else []
    for order in template.orders:
        if order.category == category and order.samples is not None:
            lengths += list(order.samples.lengths_um)
            tortuosities += list(order.samples.tortuosities)
    if lengths:
        longest = float(max(lengths))
    else:
        median = cls.median_length_um or cls.mean_length_um
        longest = float(LENGTH_CLIP[1] * median) if median else math.inf
    if tortuosities:
        most_tortuous = max(float(max(tortuosities)), 1.0)
    else:
        excess = max(float(cls.mean_tortuosity or 1.1) - 1.0, 0.0)
        sigma = TORTUOSITY_LOG_SIGMA
        most_tortuous = 1.0 + excess * math.exp(-(sigma**2) / 2.0 + 2.0 * sigma)
    return longest, most_tortuous


def _chord(draft: NetworkDraft, edge: Any) -> float:
    points = [draft.position(edge.u), *edge.via, draft.position(edge.v)]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:])))


def _assemble(draft: NetworkDraft, template: NetworkTemplate, extent: np.ndarray, options: GenerationOptions, rng: np.random.Generator) -> nx.MultiGraph:
    """The draft as the graph: polylines bent to their lengths, ids renumbered."""
    live = [n for n in range(len(draft.positions)) if n not in draft.removed]
    new_id = {old: i for i, old in enumerate(live)}
    G = nx.MultiGraph()
    for old in live:
        G.add_node(new_id[old], pos=np.asarray(draft.position(old), dtype=float).copy())
        roles = draft.roles.get(old)
        if roles:
            G.nodes[new_id[old]][SYNTHETIC_ROLES] = tuple(roles)
        G.nodes[new_id[old]].update(draft.marks.get(old, {}))
    edges = [e for e in draft.edges if e.u not in draft.removed and e.v not in draft.removed]
    simple = [e for e in edges if not e.via]
    paths: list = []
    if simple:
        starts = np.stack([draft.position(e.u) for e in simple])
        ends = np.stack([draft.position(e.v) for e in simple])
        targets = np.array([float(e.length_um or 0.0) for e in simple])
        curvature = None
        if template.organisation is not None:
            curvature = np.array([_curvature_target(template, e.category, rng) for e in simple])
        paths = list(bend_to_lengths(
            starts, ends, targets, rng, points=options.bend_points, extent_um=extent, curvature_deg=curvature,
            cross_wave=np.array([_cross_wave(template, e.category) for e in simple]), hole=options.hole,
        ))
    with_via = [e for e in edges if e.via]
    for edge in with_via:
        paths.append(bend_path(
            [draft.position(edge.u), *edge.via, draft.position(edge.v)], float(edge.length_um or 0.0), rng,
            bend_points=options.bend_points, extent_um=extent,
            curvature_deg=None if template.organisation is None else _curvature_target(template, edge.category, rng),
            cross_wave=_cross_wave(template, edge.category), hole=options.hole,
        ))

    def rebend(index: int, target: float) -> np.ndarray:
        """Edge *index* (of ``simple + with_via``) bent again to *target*."""
        if index < len(simple):
            edge = simple[index]
            return bend_to_lengths(
                starts[index:index + 1], ends[index:index + 1], np.array([target]), rng, points=options.bend_points,
                extent_um=extent, curvature_deg=None if curvature is None else curvature[index:index + 1],
                cross_wave=np.array([_cross_wave(template, edge.category)]), hole=options.hole,
            )[0]
        edge = with_via[index - len(simple)]
        return bend_path(
            [draft.position(edge.u), *edge.via, draft.position(edge.v)], target, rng, bend_points=options.bend_points,
            extent_um=extent,
            curvature_deg=None if template.organisation is None else _curvature_target(template, edge.category, rng),
            cross_wave=_cross_wave(template, edge.category), hole=options.hole,
        )

    paths = _with_fine_wobble(template, simple + with_via, paths, rng, rebend, options.hole)
    voxels, lengths = densify_paths(paths, step_um=options.voxel_step_um)
    for edge, edge_voxels, length in zip(simple + with_via, voxels, lengths):
        _add_edge(G, new_id, edge, edge_voxels, float(length), template.diameter_basis)
    for node in G.nodes:
        if G.degree(node) == 1:
            on_face = face_distance_um(G.nodes[node]["pos"], extent) <= 1e-6
            G.nodes[node][DEAD_END_KIND] = LEAVES_IMAGE if on_face else BLIND
            G.nodes[node][DEAD_END_KIND_SOURCE] = SOURCE_SYNTHETIC
    voxel = tuple(float(v) for v in template.voxel_size_zyx)
    G.graph["voxel_size"] = voxel
    G.graph["image_voxel_size_zyx"] = voxel
    G.graph["image_voxel_size_xyz"] = tuple(reversed(voxel))
    # The template's extent itself, not (shape - 1) * voxel: a template read
    # from a run has both agree, a hand-set one may not, and the faces the
    # dead ends were set on are the extent's.
    G.graph["image_extent_um"] = tuple(float(v) for v in extent)
    if options.hole is not None:
        G.graph[GRAPH_HOLE] = options.hole.to_record()
    return G


def _curvature_target(template: NetworkTemplate, category: str, rng: np.random.Generator) -> float:
    """The turn per 10 um a drawn vessel of *category* is bent toward
    (``organisation``): its capillaries' mean; for a tree vessel one drawn
    from the template's own tree vessels (Harvey, grid D1f: aimed at the
    mean, every drawn tree wound more than the template's median one), else
    their mean; NaN for none."""
    organisation = template.organisation
    if category == "capillary":
        value = organisation.capillary_curvature_deg
    elif organisation.tree_curvature_by_vessel_deg:
        value = float(rng.choice(np.asarray(organisation.tree_curvature_by_vessel_deg, dtype=float)))
    else:
        value = organisation.tree_curvature_deg
    return float("nan") if value is None else float(value)


def _cross_wave(template: NetworkTemplate, category: str) -> float:
    """How much of the bend's second, perpendicular wave a vessel gets: a
    tree vessel of a template whose trees' curvature is read per vessel,
    :data:`TREE_CROSS_WAVE`; every other, all of it."""
    organisation = template.organisation
    if category == "capillary" or organisation is None or not organisation.tree_curvature_by_vessel_deg:
        return 1.0
    return TREE_CROSS_WAVE


def _with_fine_wobble(
    template: NetworkTemplate,
    edges: Sequence,
    paths: list,
    rng: np.random.Generator,
    rebend: Any = None,
    hole: Optional[Hole] = None,
) -> list:
    """*paths* with the tree vessels' fine wobble: their median vessel's turn
    read at ``organisation.FINE_CURVATURE_STEP_UM`` steps brought to the
    template's (Harvey, grid D1f: E14.5's trees turn 88 degrees per 10 um
    read every 2 um, the drawn ones' smooth curves 72-75), by one amplitude
    for every tree vessel (:func:`polylines.fine_wobble`), found by
    bisection; none when they already turn as much, nor on a vessel it would
    take into the *hole*. The wobble lengthens a
    vessel, so with *rebend* (``rebend(index, target_um)``: the vessel bent
    again to that length) each tree vessel's bow is drawn again to its
    length less what the wobble adds, and wobbled: the template's trees owe
    part of their length to their wobble, and a bow drawn to all of it
    turned more than theirs at 10 um."""
    from .organisation import FINE_CURVATURE_STEP_UM, path_turning
    from .polylines import fine_wobble, wobble_stations

    organisation = template.organisation
    target = None if organisation is None else organisation.tree_fine_curvature_deg
    trees_at = [i for i, e in enumerate(edges) if e.category != "capillary"]
    if target is None or not trees_at:
        return paths
    step = FINE_CURVATURE_STEP_UM
    noise = {}
    for i in trees_at:
        total = float(np.linalg.norm(np.diff(np.asarray(paths[i], float), axis=0), axis=1).sum())
        noise[i] = rng.normal(0.0, 1.0, size=(len(wobble_stations(total, step)), 2))

    def median_turn(amplitude: float) -> Optional[float]:
        values = [path_turning(fine_wobble(paths[i], amplitude, noise[i], step), step) for i in trees_at]
        known = [v for v in values if v is not None]
        return float(np.median(known)) if known else None

    now = median_turn(0.0)
    if now is None or now >= target:
        return paths
    low, high = 0.0, WOBBLE_MAX_UM
    if (median_turn(high) or 0.0) < target:
        low = high
    else:
        for _ in range(WOBBLE_BISECTIONS):
            mid = 0.5 * (low + high)
            if (median_turn(mid) or 0.0) < target:
                low = mid
            else:
                high = mid
    amplitude = 0.5 * (low + high) if low < high else low
    out = list(paths)
    for i in trees_at:
        wobbled = fine_wobble(paths[i], amplitude, noise[i], step)
        if rebend is not None:
            bent_length = _path_length(paths[i])
            gain = _path_length(wobbled) / bent_length if bent_length > 0 else 1.0
            if gain > 1.0:
                again = rebend(i, bent_length / gain)
                total = _path_length(again)
                fresh = rng.normal(0.0, 1.0, size=(len(wobble_stations(total, step)), 2))
                wobbled = fine_wobble(again, amplitude, fresh, step)
        if hole is not None and hole.entered_by(np.asarray(wobbled, dtype=float)[None])[0]:
            wobbled = paths[i]
        out[i] = wobbled
    return out


def _path_length(path: Any) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path, float), axis=0), axis=1).sum())


def _add_edge(G: nx.MultiGraph, new_id: Mapping[int, int], edge: Any, voxels: list, length: float, basis: Optional[str]) -> None:
    attributes = {SYNTHETIC_CLASS: edge.category}
    if basis is not None:
        attributes[EDGE_DIAMETER_BASIS] = basis
    if edge.category in _LARGE_MASK_SIDE:
        attributes[LARGE_MASK_VESSEL_TYPE] = _LARGE_MASK_SIDE[edge.category]
    G.add_edge(
        new_id[edge.u],
        new_id[edge.v],
        voxels=voxels,
        length=length,
        diameter_um=float(edge.diameter_um),
        diameter_source=DIAMETER_SOURCE_SYNTHETIC,
        **attributes,
    )


def boundary_nodes(G: nx.Graph) -> dict:
    """``{role: [nodes]}`` for every role in :data:`ROLES`, read off the
    generator's node marks, in node order."""
    out: dict = {role: [] for role in ROLES}
    for node in sorted(G.nodes):
        for role in G.nodes[node].get(SYNTHETIC_ROLES, ()):
            if role in out:
                out[role].append(node)
    return out


#: How near a polyline's end must lie to its node: ``np.allclose``'s test at
#: ``atol=1e-6`` and its default ``rtol``, written out (it was most of the
#: validation's time on a 10,000-vessel network).
_END_ATOL = 1e-6
_END_RTOL = 1e-5


def validate_synthetic_graph(G: nx.Graph, template: Optional[NetworkTemplate] = None) -> None:
    """Raise ``ValueError`` unless *G* keeps the contract in the module docstring."""
    problems: list[str] = []
    if not isinstance(G, nx.MultiGraph):
        problems.append("not an nx.MultiGraph")
    extent = graph_image_extent_um(G)
    if extent is None:
        problems.append("no image_extent_um on the graph")
        extent = np.full(3, np.inf)
    for key in ("voxel_size", "image_voxel_size_zyx", "image_voxel_size_xyz"):
        if key not in G.graph:
            problems.append(f"no {key} on the graph")
    if template is not None and extent is not None and not np.allclose(extent, template.extent_um):
        problems.append("the extent is not the template's")
    for node, data in G.nodes(data=True):
        if not isinstance(node, (int, np.integer)):
            problems.append(f"node {node!r} is not an int")
            continue
        pos = data.get("pos")
        if pos is None or np.shape(pos) != (3,) or not np.all(np.isfinite(pos)):
            problems.append(f"node {node} has no 3-D pos")
            continue
        if np.any(np.asarray(pos) < -1e-6) or np.any(np.asarray(pos) > extent + 1e-6):
            problems.append(f"node {node} lies outside the box")
        degree = G.degree(node)
        if degree == 0:
            problems.append(f"node {node} is isolated")
        if degree == 2:
            problems.append(f"node {node} has degree 2")
        roles = data.get(SYNTHETIC_ROLES, ())
        if any(r.endswith("_boundary") for r in roles) and not ({"inlet", "outlet"} & set(roles)) and degree < 3:
            problems.append(f"boundary node {node} has degree {degree}")
        if degree == 1:
            kind = data.get(DEAD_END_KIND)
            on_face = face_distance_um(np.asarray(pos, float), extent) <= 1e-6
            if kind not in (LEAVES_IMAGE, BLIND):
                problems.append(f"dead end {node} has no kind")
            elif (kind == LEAVES_IMAGE) != on_face:
                problems.append(f"dead end {node} is marked {kind} but is {'on' if on_face else 'off'} a face")
    try:
        assert_no_forbidden_edge_attributes(G, context="synthetic network")
    except ValueError as error:
        problems.append(str(error))
    for u, v, data in G.edges(data=True):
        voxels = data.get("voxels")
        if not voxels or len(voxels) < 2:
            problems.append(f"edge {u}-{v} has no polyline")
            continue
        arr = np.asarray(voxels, float)
        pu, pv = np.asarray(G.nodes[u]["pos"], float), np.asarray(G.nodes[v]["pos"], float)
        # NetworkX yields an undirected edge's ends in node order, so the
        # polyline may run either way; it must touch both nodes.
        ends = np.abs(np.stack([arr[0] - pu, arr[-1] - pv, arr[0] - pv, arr[-1] - pu]))
        close = (ends <= _END_ATOL + _END_RTOL * np.abs(np.stack([pu, pv, pv, pu]))).all(axis=1)
        if not ((close[0] and close[1]) or (close[2] and close[3])):
            problems.append(f"edge {u}-{v}'s polyline does not join its nodes")
        # One pass over the steps for both checks: calculate_path_length's
        # sum of step lengths, read once rather than converting the polyline twice.
        steps = np.diff(arr, axis=0)
        step_lengths = np.sqrt(np.einsum("ij,ij->i", steps, steps))
        if step_lengths.max() > 0.5 + 1e-6:
            problems.append(f"edge {u}-{v}'s polyline steps more than 0.5 um")
        if not math.isclose(float(data.get("length", -1.0)), float(step_lengths.sum()), rel_tol=1e-6, abs_tol=1e-6):
            problems.append(f"edge {u}-{v}'s length is not its polyline's")
        if not (data.get("diameter_um") or 0) > 0:
            problems.append(f"edge {u}-{v} has no diameter")
        if data.get("diameter_source") != DIAMETER_SOURCE_SYNTHETIC:
            problems.append(f"edge {u}-{v} is not marked {DIAMETER_SOURCE_SYNTHETIC}")
        if template is not None and data.get(EDGE_DIAMETER_BASIS) != template.diameter_basis:
            problems.append(f"edge {u}-{v}'s diameter basis is not the template's ({template.diameter_basis})")
        if data.get("branch_order") is not None:
            problems.append(f"edge {u}-{v} already has a branch order")
    hole = Hole.from_record(G.graph.get(GRAPH_HOLE))
    if hole is not None:
        for node, data in G.nodes(data=True):
            if data.get("pos") is not None and bool(hole.contains(data["pos"])):
                problems.append(f"node {node} lies in the hole")
        for u, v, data in G.edges(data=True):
            if data.get("voxels") and bool(hole.contains(np.asarray(data["voxels"], float)).any()):
                problems.append(f"edge {u}-{v}'s polyline enters the hole")
    if G.number_of_nodes() and not nx.is_connected(G):
        problems.append("the network is not connected")
    roles = boundary_nodes(G)
    if not roles["inlet"] or not roles["outlet"]:
        problems.append("no inlet or no outlet")
    if problems:
        raise ValueError("synthetic network: " + "; ".join(problems[:12]) + (" ..." if len(problems) > 12 else ""))


__all__ = [
    "DEFAULT_SEED",
    "GENERATOR_VERSION",
    "GenerationOptions",
    "LARGE_MASK_VESSEL_TYPE",
    "OPEN_ENDS",
    "OPEN_ENDS_CLOSED",
    "OPEN_ENDS_TEMPLATE",
    "SYNTHETIC",
    "SYNTHETIC_CLASS",
    "SYNTHETIC_ROLES",
    "BedPlan",
    "boundary_nodes",
    "dead_end_pairs_joined",
    "drawn_template",
    "extension_share",
    "face_extension_shares",
    "generate_network",
    "planned_bed",
    "JOIN_BED",
    "JOIN_TRUNK",
    "Join",
    "SURROUND_JOIN",
    "validate_synthetic_graph",
]
