"""The capillary bed of a synthetic network.

The bed is a random spatial network whose **capillary count, number of
independent loops (cyclomatic number) and number of dead ends are the
template's exactly, together**: its degree sequence is worked out first from
the handshake lemma (:func:`degree_sequence_targets`), then its vertices are
laid on a jittered cubic lattice at the spacing those counts and the box
imply, its candidate edges are the Delaunay edges between them (so no two
cross), and a degree-constrained greedy fill takes, at each vertex, the
candidates that are short and meet the vessels already there near the
template's daughter angle. The tree leaves that hand over to the bed (the
arteriole and venule boundary nodes) are bed vertices too, with the few bed
edges each takes fixed in advance.

What the fill leaves -- a vertex short of its degree, a piece cut off -- is
repaired along the Delaunay edges (one edge per piece joined, which keeps the
loop count) or merged away (a degree-2 vertex's two edges become one), and
reported. Degree-1 vertices nearest the faces are set on them: they are the
vessels the box cuts; the rest end blind inside.

**The bed follows the template's layout** (``PLAN_synthetic_physiology.md``,
Harvey's D4 and D6) when the template carries one (``synthetic.organisation``):
drawn at the template's own size, its vertices are the template's capillary
vertices each moved at random by about their nearest-neighbour spacing
(:func:`template_vertices`: a plexus a sheet thinner than a cell stays one,
and the vertices keep the template's spacing, so its vessels keep their
length and room to curve); otherwise they are laid cell by cell in
proportion to the template's capillary vertices (:func:`density_vertices`),
so the bed is dense where the template's capillaries are dense and absent
where they are. The fill prefers candidates
running the way the template's capillaries run in that cell
(:data:`ORIENTATION_WEIGHT`); and a vessel's wanted chord scales with the
local spacing. A template without one gets an even lattice over the box.

``model="voronoi"`` (the ``synthetic_bed_model`` setting) draws the bed's
vertices and candidates from a **Voronoi tessellation** of random points
instead (:func:`voronoi_vertices`): the vertices are its cells' corners and
the candidates its cells' edges, the classic Voronoi capillary network. The
same fill prunes them to the template's degree sequence, so the counts stay
exact; what the Voronoi edges cannot fill the same repairs draw from nearest
neighbours, and the bed's record counts how many of its vessels are Voronoi
edges.

Given a hole in the box (``synthetic.hole``: an image's own box, in a
surround), the bed fills the rest of it: as many vertices are laid as would
leave the count wanted outside the hole and a margin round it, those inside
are dropped, and no two vertices are joined across it (:meth:`_Fill.crosses_hole`,
asked by every way the fill joins two).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from haemolynx.statistics.layout import ACUTE_JUNCTION_DEG

from .draft import BED, NetworkDraft
from .hole import Hole
from .organisation import Organisation, cell_index, chord_class, tensor_matrix

#: Candidate edges longer than this many lattice spacings are left out; from
#: an attached (tree) node, which sits off the lattice, a little longer.
MAX_CANDIDATE_SPACINGS = 2.2
MAX_ATTACHED_SPACINGS = 4.0
#: The lattice is laid with this many more vertices than needed, then thinned.
OVERSAMPLE = 1.15
#: Lattice jitter, as a fraction of the lattice step along each axis.
JITTER = 0.35
#: No bed vertex this close (in spacings) to a tree node, nor to a face.
CLEARANCE_SPACINGS = 0.35
FACE_MARGIN_SPACINGS = 0.25
#: Round a hole: how many times the vertices are laid again while too few
#: land outside it, each time this much more than the shortfall asks.
HOLE_ATTEMPTS = 3
HOLE_REDRAW = 1.05
#: How much a candidate's length counts against its angle to the vessels
#: already at the vertex, in the fill's score (``weight * |angle - target| / 90 deg``
#: in spacings).
ANGLE_WEIGHT = 0.5
#: Nearest neighbours tried when the Delaunay candidates run out.
FALLBACK_NEIGHBOURS = 16
#: How much a candidate's alignment with the template's capillaries counts in
#: the fill's score, in local spacings: ``weight * (1/3 - u.T.u) * 3/2`` --
#: nothing in an isotropic cell, a spacing's bonus along a cell's only
#: direction, half a spacing's penalty across it. On E14.5 (2026-10-25, the
#: bed on the template's own vertices, each candidate judged against its
#: direction's own chord) 2 gives its columns -- 9-13 runs along z spanning
#: 100 um against the template's 12 (1: 5-9) -- and keeps its curvature (22-25
#: degrees per 10 um against 26). Heavier, the bonus outweighs a candidate's
#: length: at 4 the chords grew to 38-40 um, left the vessels no length to
#: curve in, and 70-78% of capillaries came out straight (grid D1b, which
#: Harvey marked for it); 2.5 tips one seed in three. A term pushing each
#: cell's drawn share of each direction to the template's was tried as well
#: and did worse the heavier it was.
ORIENTATION_WEIGHT = 2.0
#: Junctions of the largest piece tried, nearest first, for a piece every one
#: of whose junctions has a dead end nearest (:func:`join_pieces`' last resort).
JOIN_NEIGHBOURS = 8
#: The bed models (the ``synthetic_bed_model`` choices; ``pipeline/schema.py``
#: repeats them and a test pins the two equal).
BED_MODEL_DELAUNAY = "delaunay"
BED_MODEL_VORONOI = "voronoi"
BED_MODELS: tuple[str, ...] = (BED_MODEL_DELAUNAY, BED_MODEL_VORONOI)
#: Corners per cell of a Poisson-Voronoi tessellation in 3-D (27.07), each
#: shared by four cells: the tessellation's vertices per seed point.
VORONOI_VERTICES_PER_SEED = 27.07 / 4.0
#: A tessellation laid out to the template is drawn with this many times
#: the corners it keeps: corners of a dense layer's cells spill into the
#: sparse one beside it, and the contraction takes them back out.
VORONOI_LAID_OUT_OVERSAMPLE = 3.0
#: At the template's own size the seeds are its capillary vertices moved by
#: their spacing (:func:`_template_seeds`), drawn this many times over: so
#: little contraction keeps the corners as close as the template's vertices,
#: and its capillaries room to curve. Contracted three times over to the cell
#: field, E14.5's Voronoi bed drew chords as long as its vessels and every
#: capillary straight in some seeds (grid D1c); at 1.5 they curve 22-26
#: degrees per 10 um (26), though its plexus layer holds 31-37% of the
#: vertices against the template's 58% (49% contracted): a tessellation's
#: corners spill out of a sheet thinner than its seeds' spacing.
VORONOI_TEMPLATE_OVERSAMPLE = 1.5
#: The fewest vertices a cell is taken to want when the contraction reads
#: its spacing (a cell the template has none in would want an infinite one).
VORONOI_MIN_CELL_VERTICES = 0.25
#: Tessellations drawn, each with more seeds, before a short one is kept.
VORONOI_ATTEMPTS = 6
#: Whether a Voronoi bed's corners are put at the template's junction
#: depths (:func:`depths_like_template`; Harvey, grid D1e). A tessellation of
#: seeds in a sheet thinner than their spacing puts its corners above and
#: below the sheet, not in it: on E14.5 37% of the Voronoi bed's junctions lay
#: in the bottom 48 um against the template's 61%, and Harvey took no Voronoi
#: network for the tissue in five grids.
VORONOI_DEPTHS_FROM_TEMPLATE = True
#: With a template that records its layout (``statistics.layout``; Harvey,
#: 2026-10-25, after grid D1d): a candidate leaving either of its vertices
#: within ``statistics.layout.ACUTE_JUNCTION_DEG`` of a vessel already there
#: (a tree leaf's tree vessel included) costs up to this many local spacings
#: (all of it at 0 degrees), and the fill aims each pair of vessels at a
#: vertex at the template's own junction angle (its junctions' three angles
#: summed, over three) at both ends of the candidate rather than at the trees'
#: daughter angle at one. On E14.5 (Delaunay, seeds 1-2) 40% of the junctions
#: had an acute pair, 16% with this and :meth:`_Fill.relax_junctions` (the
#: template 1%), the smallest angle's median 53 -> 69 degrees (97). Heavier, or
#: aiming harder at the template's angle (:data:`JUNCTION_ANGLE_WEIGHT` 5:
#: 82 degrees, 5% acute), buys wide junctions with long chords: their sum
#: passed the template's capillary length and every capillary was drawn
#: straight (tortuosity P90 1.0, curvature 0.1 degrees per 10 um).
ACUTE_WEIGHT = 3.0
#: The share of junctions with an acute pair the fill draws unaided on
#: E14.5 (Delaunay 0.39-0.41, Voronoi 0.27-0.29): the acute weight is
#: :data:`ACUTE_WEIGHT` scaled by how far below this the template's own share
#: lies (none for a template at or above it), so a template whose junctions
#: are acute is drawn so, not widened past itself.
UNAIDED_ACUTE_SHARE = 0.38
#: ... and each pair's departure from that angle costs this many spacings per
#: 90 degrees (as :data:`ANGLE_WEIGHT` without a layout).
JUNCTION_ANGLE_WEIGHT = 0.5
#: The greedy fill alone leaves one E14.5 junction in eight or more acute
#: whatever the weights: by a vertex's turn the edges that would have given it
#: a wide Y are taken. So the fill is relaxed after (:meth:`_Fill.relax_junctions`):
#: two edges trade ends (every vertex keeps its degree, so the counts and the
#: loops stay the template's) wherever that lowers the junctions' cost and the
#: chords' departure from their wanted length together, for up to this many
#: rounds.
RELAX_ROUNDS = 6
#: A vertex is relaxed when two of its vessels are closer than this: the
#: acute pairs (relaxing every pair under 90 degrees widened the junctions as
#: far as the heavier weights did, and straightened the capillaries as far).
RELAX_BELOW_DEG = 45.0
_ACUTE = math.radians(ACUTE_JUNCTION_DEG)


@dataclass(frozen=True)
class DegreeTargets:
    """What the bed's degree sequence has to be."""

    #: Bed-only vertices (the tree leaves attached to the bed are not counted).
    vertices: int
    #: ``{degree: how many bed-only vertices}``; 1 for the dead ends.
    degree_counts: dict
    #: Loops this sequence cannot give (0 when the template's count is reachable).
    loops_shortfall: int
    notes: tuple = ()

    @property
    def dead_ends(self) -> int:
        return int(self.degree_counts.get(1, 0))


def degree_sequence_targets(
    *,
    capillary_count: int,
    loops: int,
    root_count: int,
    attached_caps: Sequence[int],
    dead_ends: int,
    max_degree: int = 4,
) -> DegreeTargets:
    """The bed-only vertex count and degree sequence that give exactly
    *capillary_count* edges, *loops* independent loops and *dead_ends*
    degree-1 vertices, the attached nodes taking *attached_caps* bed edges each.

    With the trees adding one node per edge, ``loops = E - N + 1`` gives
    ``P = E - loops - roots + 1`` bed-only vertices; the handshake lemma
    ``sum(degrees) = 2 E - sum(caps)`` then fixes how many of them are
    degree 3 and how many degree 4 (or more, up to *max_degree*) once the
    dead ends are set aside. When the template asks for more dead ends than
    its edges can carry with every other vertex a junction, the dead ends
    are the ones that give (noted); when it asks for more loops than
    junctions of *max_degree* can make, the loops are (``loops_shortfall``).
    """
    notes: list[str] = []
    edges = int(capillary_count)
    loops = max(int(loops), 0)
    vertices = edges - loops - int(root_count) + 1
    if vertices < 1:
        notes.append(
            f"{loops} loops with {edges} capillaries and {root_count} roots leave no bed vertices; "
            f"the loops are reduced to {edges - int(root_count)}"
        )
        loops = max(edges - int(root_count), 0)
        vertices = max(edges - loops - int(root_count) + 1, 1)
    handshake = 2 * edges - int(sum(attached_caps))
    if handshake < vertices:
        notes.append("the attached nodes take more bed edges than there are; some stay short")
        handshake = vertices
    ones = min(max(int(dead_ends), 0), max(vertices - 1, 0))
    hubs = vertices - ones
    remaining = handshake - ones
    shortfall = 0
    if hubs <= 0:
        counts = {1: ones}
    elif remaining < 3 * hubs:
        # Too few edge ends for so many junctions of three: fewer junctions,
        # which is more loops (the edge and dead-end counts are kept).
        hubs = max(remaining // 3, 1)
        vertices = ones + hubs
        counts = {1: ones, 3: hubs}
        spare = remaining - 3 * hubs
        i = 0
        while spare > 0:
            counts[3] -= 1
            counts[4] = counts.get(4, 0) + 1
            spare -= 1
            i += 1
            if i >= hubs:
                break
        new_loops = edges - vertices - int(root_count) + 1
        notes.append(
            f"the template's {loops} loops and {dead_ends} dead ends cannot both be met with "
            f"{edges} capillaries; {new_loops} loops are drawn"
        )
        loops = new_loops
    else:
        degrees = [3] * hubs
        extra = remaining - 3 * hubs
        i = 0
        while extra > 0 and any(d < max_degree for d in degrees):
            if degrees[i % hubs] < max_degree:
                degrees[i % hubs] += 1
                extra -= 1
            i += 1
        if extra > 0:
            # Every junction at max_degree and edges still unplaced: those
            # edges (and their loops) cannot be drawn.
            shortfall = (extra + 1) // 2
            notes.append(
                f"{extra} edge ends have no junction to meet at with junctions of at most "
                f"{max_degree} vessels; about {shortfall} loops fewer than the template"
            )
        counts = {1: ones}
        for d in degrees:
            counts[d] = counts.get(d, 0) + 1
    counts = {d: n for d, n in counts.items() if n > 0}
    return DegreeTargets(vertices=vertices, degree_counts=counts, loops_shortfall=shortfall, notes=tuple(notes))


@dataclass
class BedResult:
    """What building the bed left, in the draft's node ids."""

    face_ends: list = field(default_factory=list)
    blind_ends: list = field(default_factory=list)
    vertices: int = 0
    edges: int = 0
    repairs: int = 0
    merged: int = 0
    dropped: int = 0
    short_of_degree: int = 0
    #: How many vertices were short of their degree after each phase of the fill.
    phases: dict = field(default_factory=dict)


def lattice_vertices(
    extent_um: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    keep_clear: Optional[np.ndarray],
    spacing_um: float,
) -> np.ndarray:
    """About *count* jittered lattice points inside the box, none within
    half a spacing of a point of *keep_clear* nor within a quarter of a face,
    thinned at random to *count* when there are more."""
    extent = np.asarray(extent_um, dtype=float)
    volume = float(np.prod(np.maximum(extent, 1e-9)))
    wanted = max(int(math.ceil(count * OVERSAMPLE)), 1)
    step = (volume / wanted) ** (1.0 / 3.0)
    counts = np.maximum(np.round(extent / step).astype(int), 1)
    while int(np.prod(counts)) < wanted:
        axis = int(np.argmax(extent / counts))
        counts[axis] += 1
    steps = extent / counts
    grids = np.meshgrid(*[(np.arange(n) + 0.5) * s for n, s in zip(counts, steps)], indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    points = points + rng.uniform(-JITTER, JITTER, size=points.shape) * steps[None, :]
    margin = np.minimum(FACE_MARGIN_SPACINGS * spacing_um, extent / 4.0)
    points = np.clip(points, margin, extent - margin)
    if keep_clear is not None and len(keep_clear):
        tree = cKDTree(np.asarray(keep_clear, dtype=float))
        near = tree.query_ball_point(points, r=CLEARANCE_SPACINGS * spacing_um)
        keep = np.array([len(n) == 0 for n in near])
        points = points[keep]
    if len(points) > count:
        chosen = rng.choice(len(points), size=count, replace=False)
        points = points[np.sort(chosen)]
    return points


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """*total* shared out in proportion to *weights* by largest remainder."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    short = int(total - counts.sum())
    if short > 0:
        counts[np.argsort(-(exact - counts), kind="stable")[:short]] += 1
    return counts


def _cell_bounds(extent: np.ndarray, shape: Sequence[int], cell: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.array(np.unravel_index(cell, tuple(shape)), dtype=float)
    size = extent / np.asarray(shape, dtype=float)
    return index * size, (index + 1.0) * size


def density_vertices(
    extent_um: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    keep_clear: Optional[np.ndarray],
    spacing_um: float,
    organisation: Organisation,
) -> np.ndarray:
    """About *count* points laid cell by cell in proportion to the template's
    capillary vertices (``organisation.capillary_nodes``), each cell's a
    jittered lattice of its own spacing: dense where the template's
    capillaries are dense, none where it has none. Kept off the faces and
    *keep_clear* as :func:`lattice_vertices` is, thinned at random to
    *count* when there are more."""
    extent = np.asarray(extent_um, dtype=float)
    shape = tuple(organisation.shape)
    wanted = max(int(math.ceil(count * OVERSAMPLE)), 1)
    per_cell = _allocate(wanted, np.asarray(organisation.capillary_nodes, dtype=float))
    chunks: list[np.ndarray] = []
    for cell in np.flatnonzero(per_cell):
        n = int(per_cell[cell])
        low, high = _cell_bounds(extent, shape, int(cell))
        size = high - low
        step = (float(np.prod(size)) / n) ** (1.0 / 3.0)
        counts = np.maximum(np.round(size / step).astype(int), 1)
        while int(np.prod(counts)) < n:
            counts[int(np.argmax(size / counts))] += 1
        steps = size / counts
        grids = np.meshgrid(*[low[i] + (np.arange(counts[i]) + 0.5) * steps[i] for i in range(3)], indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=1)
        points = points + rng.uniform(-JITTER, JITTER, size=points.shape) * steps[None, :]
        if len(points) > n:
            points = points[np.sort(rng.choice(len(points), size=n, replace=False))]
        chunks.append(points)
    points = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    margin = np.minimum(FACE_MARGIN_SPACINGS * spacing_um, extent / 4.0)
    points = np.clip(points, margin, extent - margin)
    if keep_clear is not None and len(keep_clear) and len(points):
        near = cKDTree(np.asarray(keep_clear, dtype=float)).query_ball_point(points, r=CLEARANCE_SPACINGS * spacing_um)
        points = points[np.array([len(n) == 0 for n in near])]
    if len(points) > count:
        points = points[np.sort(rng.choice(len(points), size=count, replace=False))]
    return points


def template_vertices(
    extent_um: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    keep_clear: Optional[np.ndarray],
    spacing_um: float,
    organisation: Organisation,
) -> np.ndarray:
    """About *count* points drawn from the template's capillary vertices
    (``organisation.capillary_node_positions_rel``), each moved by a normal
    step of their median nearest-neighbour spacing (its third on each
    axis): the template's layout smoothed by its own spacing, not its
    vertices. Kept off the faces and *keep_clear* as :func:`lattice_vertices`
    is, thinned at random to *count* when there are more."""
    extent = np.asarray(extent_um, dtype=float)
    source = np.asarray(organisation.capillary_node_positions_rel, dtype=float) * extent
    wanted = max(int(math.ceil(count * OVERSAMPLE)), 1)
    spread = float(organisation.capillary_node_spacing_um or spacing_um) / math.sqrt(3.0)
    points = source[rng.integers(0, len(source), size=wanted)] + rng.normal(0.0, spread, size=(wanted, 3))
    margin = np.minimum(FACE_MARGIN_SPACINGS * spacing_um, extent / 4.0)
    points = np.clip(points, margin, extent - margin)
    if keep_clear is not None and len(keep_clear) and len(points):
        near = cKDTree(np.asarray(keep_clear, dtype=float)).query_ball_point(points, r=CLEARANCE_SPACINGS * spacing_um)
        points = points[np.array([len(n) == 0 for n in near])]
    if len(points) > count:
        points = points[np.sort(rng.choice(len(points), size=count, replace=False))]
    return points


def acute_weight_for(acute_share: Optional[float]) -> float:
    """What an acute pair costs the fill, in spacings, for a template whose
    share of junctions with one is *acute_share* (None: the full weight)."""
    if acute_share is None:
        return ACUTE_WEIGHT
    return ACUTE_WEIGHT * max(0.0, 1.0 - float(acute_share) / UNAIDED_ACUTE_SHARE)


def drawn_from_template_vertices(organisation: Optional[Organisation], at_template_size: bool) -> bool:
    """Whether a bed is drawn from the template's own vertices
    (:func:`template_vertices`): at the template's own size (a scaled box
    has more or fewer vertices than the template's), when it recorded them."""
    return bool(at_template_size and organisation is not None and organisation.capillary_node_positions_rel)


def local_spacings(points: np.ndarray, extent_um: np.ndarray, organisation: Organisation, default_um: float) -> np.ndarray:
    """Each point's local spacing: its cell's volume over the points in it,
    to the third; *default_um* everywhere without an organisation."""
    if organisation is None or not len(points):
        return np.full(len(points), float(default_um))
    extent = np.asarray(extent_um, dtype=float)
    cells = cell_index(points, extent, organisation.shape)
    counts = np.bincount(cells, minlength=int(np.prod(organisation.shape)))
    volume = float(np.prod(extent)) / float(np.prod(organisation.shape))
    return (volume / np.maximum(counts[cells], 1)) ** (1.0 / 3.0)


def _circumcentres(tetrahedra: np.ndarray) -> np.ndarray:
    """The circumcentre of each tetrahedron (``(n, 4, 3)``); NaN for a flat one."""
    a = tetrahedra[:, 0, :]
    rest = tetrahedra[:, 1:, :] - a[:, None, :]
    rhs = 0.5 * np.einsum("nij,nij->ni", rest, rest)
    out = np.full(a.shape, np.nan)
    det = np.linalg.det(rest)
    scale = np.maximum(np.abs(rest).max(axis=(1, 2)), 1e-12) ** 3
    ok = np.abs(det) > 1e-9 * scale
    if ok.any():
        out[ok] = a[ok] + np.linalg.solve(rest[ok], rhs[ok][..., None])[..., 0]
    return out


def voronoi_vertices(
    extent_um: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    keep_clear: Optional[np.ndarray],
    spacing_um: float,
    organisation: Optional[Organisation] = None,
    at_template_size: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """About *count* corners of a Voronoi tessellation of random points in
    the box, and the cell edges between them (index pairs).

    The seeds are uniform random points over the box padded by two
    spacings, as many as give :data:`OVERSAMPLE` corners per *spacing_um*
    cubed; the corners are the circumcentres of the seeds' Delaunay
    tetrahedra, joined where two tetrahedra share a face. Corners outside
    the box or within a quarter spacing of a face, or near *keep_clear*,
    are left out. With *organisation* the seeds are laid cell by cell in
    proportion to the template's capillary vertices (the padding round the
    box at its face cells' densities), so the corners follow the template's
    layout as :func:`density_vertices` does. Then the shortest edges are contracted
    (their two corners become one, at their midpoint) until *count* remain:
    a random tessellation's corners come in close pairs, and its shortest
    edges are a hundredth of its longest. Laid out, the tessellation is
    drawn :data:`VORONOI_LAID_OUT_OVERSAMPLE` times over and "shortest" is
    against the spacing the template's density wants in the corner's cell;
    by length alone the dense cells' edges went first and the layout was lost.
    """
    extent = np.asarray(extent_um, dtype=float)
    spacing = float(spacing_um)
    pad = 2.0 * spacing
    low, high = -pad * np.ones(3), extent + pad
    own = drawn_from_template_vertices(organisation, at_template_size)
    oversample = (
        OVERSAMPLE if organisation is None else VORONOI_TEMPLATE_OVERSAMPLE if own else VORONOI_LAID_OUT_OVERSAMPLE
    )
    wanted = max(int(math.ceil(count * oversample)), 1)
    seed_count = max(int(round(oversample * float(np.prod(high - low)) / (VORONOI_VERTICES_PER_SEED * spacing**3))), 8)
    margin = np.minimum(FACE_MARGIN_SPACINGS * spacing, extent / 4.0)
    clear_tree = cKDTree(np.asarray(keep_clear, dtype=float)) if keep_clear is not None and len(keep_clear) else None
    for _attempt in range(VORONOI_ATTEMPTS):
        if organisation is None:
            seeds = rng.uniform(low, high, size=(seed_count, 3))
        elif own:
            seeds = _template_seeds(seed_count, extent, pad, organisation, rng)
        else:
            seeds = _laid_out_seeds(seed_count, extent, pad, organisation, rng)
        triangulation = Delaunay(seeds, qhull_options="QJ")
        centres = _circumcentres(seeds[triangulation.simplices])
        inside = np.all(np.isfinite(centres), axis=1)
        inside &= np.all(centres >= margin, axis=1) & np.all(centres <= extent - margin, axis=1)
        if clear_tree is not None:
            near = clear_tree.query_ball_point(np.where(inside[:, None], centres, 0.0), r=CLEARANCE_SPACINGS * spacing)
            inside &= np.array([len(n) == 0 for n in near])
        kept = np.flatnonzero(inside)
        if len(kept) >= wanted:
            break
        # A small box loses many corners to its faces and the trees: more
        # seeds, by the shortfall.
        seed_count = int(math.ceil(seed_count * max(1.2, 1.05 * wanted / max(len(kept), 1))))
    index = -np.ones(len(centres), dtype=int)
    index[kept] = np.arange(len(kept))
    pairs: set = set()
    for t in kept:
        for k in triangulation.neighbors[t]:
            if k >= 0 and index[k] >= 0:
                i, j = int(index[t]), int(index[k])
                pairs.add((min(i, j), max(i, j)))
    points = centres[kept]
    # Contract the shortest edges until *count* corners remain.
    parent = list(range(len(points)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    groups = len(points)
    if groups > count:
        scale = np.ones(len(points)) if organisation is None else _wanted_spacing(points, extent, organisation, count)
        ordered = sorted(
            pairs, key=lambda e: float(np.linalg.norm(points[e[0]] - points[e[1]])) / (0.5 * (scale[e[0]] + scale[e[1]]))
        )
        for i, j in ordered:
            if groups <= count:
                break
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                groups -= 1
    roots = sorted({find(i) for i in range(len(points))})
    new_index = {root: n for n, root in enumerate(roots)}
    members: dict[int, list[int]] = {}
    for i in range(len(points)):
        members.setdefault(new_index[find(i)], []).append(i)
    merged = np.array([points[members[n]].mean(axis=0) for n in range(len(roots))]) if roots else np.zeros((0, 3))
    edges = sorted({
        (min(a, b), max(a, b))
        for a, b in ((new_index[find(i)], new_index[find(j)]) for i, j in pairs)
        if a != b
    })
    if len(merged) > count:
        # More corners than edges can contract (separate pieces): a few at random go.
        chosen = np.sort(rng.choice(len(merged), size=count, replace=False))
        remap = {int(old): n for n, old in enumerate(chosen)}
        merged = merged[chosen]
        edges = [(remap[a], remap[b]) for a, b in edges if a in remap and b in remap]
    depths = None if organisation is None else organisation.capillary_junction_depths_rel
    if VORONOI_DEPTHS_FROM_TEMPLATE and depths and len(merged):
        merged = merged.copy()
        merged[:, 0] = depths_like_template(merged[:, 0], depths, float(extent[0]), float(margin[0]))
    return merged, edges


def depths_like_template(z: np.ndarray, depths_rel: Sequence[float], extent_z: float, margin: float) -> np.ndarray:
    """*z* moved, rank for rank, onto the template's junction depths
    (*depths_rel*, shares of the box's depth): the k-th shallowest point goes
    to the depth a share (k + 1/2)/n of the template's junctions lie above.
    The order in depth is kept, so a tessellation's cells keep their
    neighbours; only how they spread over depth becomes the template's. Kept
    *margin* off the faces."""
    z = np.asarray(z, dtype=float)
    order = np.argsort(z, kind="stable")
    quantiles = (np.arange(len(z)) + 0.5) / len(z)
    source = np.sort(np.asarray(depths_rel, dtype=float))
    levels = np.interp(quantiles, (np.arange(len(source)) + 0.5) / len(source), source) * extent_z
    out = np.empty_like(z)
    out[order] = np.clip(levels, margin, extent_z - margin)
    return out


def _wanted_spacing(points: np.ndarray, extent: np.ndarray, organisation: Organisation, count: int) -> np.ndarray:
    """The spacing *count* vertices laid out to the template have in each
    point's cell (at least :data:`VORONOI_MIN_CELL_VERTICES` there)."""
    shape = tuple(organisation.shape)
    expected = _allocate(count, np.asarray(organisation.capillary_nodes, dtype=float))
    volume = float(np.prod(extent)) / float(np.prod(shape))
    return np.cbrt(volume / np.maximum(expected, VORONOI_MIN_CELL_VERTICES))[cell_index(points, extent, shape)]


def _template_seeds(count: int, extent: np.ndarray, pad: float, organisation: Organisation, rng: np.random.Generator) -> np.ndarray:
    """About *count* seeds over the box padded by *pad*: the box's share of
    them drawn from the template's capillary vertices, each moved by a normal
    step of their spacing (as :func:`template_vertices`), and mirrored across
    each face into the padding, so the faces see the same layout beyond."""
    inside = int(round(count * float(np.prod(extent)) / float(np.prod(extent + 2.0 * pad))))
    source = np.asarray(organisation.capillary_node_positions_rel, dtype=float) * extent
    spread = float(organisation.capillary_node_spacing_um or 0.0) / math.sqrt(3.0)
    seeds = source[rng.integers(0, len(source), size=inside)] + rng.normal(0.0, spread, size=(inside, 3))
    seeds = np.clip(seeds, 0.0, extent)
    parts = [seeds]
    for axis in range(3):
        for face in (0.0, float(extent[axis])):
            near = np.abs(seeds[:, axis] - face) < pad
            mirrored = seeds[near].copy()
            mirrored[:, axis] = 2.0 * face - mirrored[:, axis]
            parts.append(mirrored)
    return np.concatenate(parts)


def _laid_out_seeds(count: int, extent: np.ndarray, pad: float, organisation: Organisation, rng: np.random.Generator) -> np.ndarray:
    """About *count* seeds over the box padded by *pad*: the box's share of
    them over the template's cells in proportion to its capillary vertices,
    and a face cell's density carried on into the padding beyond it."""
    shape = tuple(organisation.shape)
    inside = count * float(np.prod(extent)) / float(np.prod(extent + 2.0 * pad))
    per_cell = _allocate(int(round(inside)), np.asarray(organisation.capillary_nodes, dtype=float))
    chunks = []
    for cell in np.flatnonzero(per_cell):
        low, high = _cell_bounds(extent, shape, int(cell))
        volume = float(np.prod(high - low))
        low = np.where(low <= 0.0, -pad, low)
        high = np.where(high >= extent, extent + pad, high)
        n = int(round(per_cell[cell] * float(np.prod(high - low)) / volume))
        chunks.append(rng.uniform(low, high, size=(n, 3)))
    return np.concatenate(chunks) if chunks else rng.uniform(-pad, extent + pad, size=(count, 3))


def class_chord_ratios(organisation: Organisation) -> Optional[np.ndarray]:
    """Each direction class's mean chord over the mean of the classes'
    means (a ratio: the wanted chord keeps its own scale); 1 for a class with
    none."""
    chords = organisation.chord_by_class_um
    if chords is None:
        return None
    known = [c for c in chords if c]
    if not known:
        return None
    mean = float(np.mean(known))
    return np.array([float(c) / mean if c else 1.0 for c in chords])


def candidate_edges(points: np.ndarray) -> list[tuple[int, int, float]]:
    """``(i, j, length)`` for every Delaunay edge of *points* (every pair
    when there are too few for a tetrahedron)."""
    n = len(points)
    if n < 2:
        return []
    pairs: set = set()
    if n < 5:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.add((i, j))
    else:
        try:
            simplices = Delaunay(points, qhull_options="QJ").simplices
        except Exception:  # noqa: BLE001 -- degenerate input (a plane of points)
            simplices = None
        if simplices is None:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.add((i, j))
        else:
            for simplex in simplices:
                for a in range(len(simplex)):
                    for b in range(a + 1, len(simplex)):
                        i, j = int(simplex[a]), int(simplex[b])
                        if i != j:
                            pairs.add((min(i, j), max(i, j)))
    if not pairs:
        return []
    index = np.array(sorted(pairs), dtype=int)
    lengths = np.linalg.norm(points[index[:, 0]] - points[index[:, 1]], axis=1)
    return [(int(i), int(j), float(length)) for (i, j), length in zip(index, lengths)]


class _Fill:
    """The degree-constrained greedy fill over a fixed vertex set."""

    def __init__(
        self,
        points: np.ndarray,
        caps: np.ndarray,
        attached: np.ndarray,
        rng: np.random.Generator,
        angle_deg: float,
        spacing_um: float,
        target_chord_um: Optional[float] = None,
        local_spacing: Optional[np.ndarray] = None,
        tensors: Optional[np.ndarray] = None,
        class_chords: Optional[np.ndarray] = None,
        junction_angle_deg: Optional[float] = None,
        acute_share: Optional[float] = None,
        fixed_directions: Optional[Mapping[int, Sequence[np.ndarray]]] = None,
        hole: Optional[Hole] = None,
    ):
        self.points = points
        #: No two vertices are joined across it (:meth:`crosses_hole`).
        self.hole = hole
        self._crossings: dict[tuple[int, int], bool] = {}
        self._tuples = [tuple(float(c) for c in p) for p in points]
        self.caps = caps.astype(int)
        self.attached = attached.astype(bool)
        self.rng = rng
        self.angle = math.radians(angle_deg)
        self.spacing = spacing_um
        #: The chord the template's capillaries have on average; None takes
        #: the shortest candidate.
        self.target_chord = target_chord_um
        #: Each vertex's own spacing (a bed laid out to the template's
        #: density), the wanted chord scaled by it; None: *spacing_um* everywhere.
        self.local = local_spacing
        #: Each vertex's cell's capillary orientation tensor; None: no preference.
        self.tensors = tensors
        #: The template's mean chord in each direction class
        #: (``organisation.DIRECTION_CLASSES``) as a ratio: a candidate is
        #: judged against its own direction's length (E14.5's vertical vessels
        #: reach past nearer junctions); None: one length for all.
        self.class_chords = class_chords
        #: The template's own junction angle (``statistics.layout``): with it
        #: the fill judges a candidate's angles at both its ends, and acute
        #: pairs cost :data:`ACUTE_WEIGHT`; None: the daughter angle at one.
        self.junction_angle = None if junction_angle_deg is None else math.radians(junction_angle_deg)
        #: What an acute pair costs, in spacings (:data:`UNAIDED_ACUTE_SHARE`).
        self.acute_weight = acute_weight_for(acute_share) if self.junction_angle is not None else 0.0
        #: Unit directions of vessels already at a vertex outside the fill
        #: (a tree leaf's own tree vessel), judged like its bed vessels.
        self.fixed = {int(k): [tuple(float(c) for c in d) for d in v] for k, v in (fixed_directions or {}).items()}
        self.degree = np.zeros(len(points), dtype=int)
        self.neighbours: dict[int, list[int]] = {i: [] for i in range(len(points))}
        self.edges: list[tuple[int, int]] = []

    def forbidden(self, i: int, j: int) -> bool:
        if i == j or j in self.neighbours[i]:
            return True
        if self.attached[i] and self.attached[j]:
            return True
        return bool(self.caps[i] == 1 and self.caps[j] == 1) or self.crosses_hole(i, j)

    def crosses_hole(self, i: int, j: int) -> bool:
        """Whether *i*-*j* would run through the hole (``synthetic.hole``);
        never without one."""
        if self.hole is None:
            return False
        key = (i, j) if i < j else (j, i)
        found = self._crossings.get(key)
        if found is None:
            found = self._crossings[key] = self.hole.crosses(self.points[i], self.points[j])
        return found

    def _spacing(self, v: int) -> float:
        return self.spacing if self.local is None else float(self.local[v])

    def _base(self, v: int, u: int, length: float, jitter: float = 1.0) -> float:
        """The candidate's departure from its wanted chord, and from the way
        the template's capillaries run at *v*."""
        wanted = self.target_chord if self.target_chord else 0.0
        spacing = self._spacing(v)
        wanted *= spacing / self.spacing
        if self.class_chords is not None and length > 0:
            wanted *= float(self.class_chords[chord_class(self.points[u] - self.points[v])])
        score = abs(length - wanted) * jitter
        if self.tensors is not None and length > 0:
            unit = (self.points[u] - self.points[v]) / length
            along = float(unit @ self.tensors[v] @ unit)
            score += ORIENTATION_WEIGHT * spacing * (1.0 / 3.0 - along) * 1.5
        return score

    def score(self, v: int, u: int, length: float) -> float:
        score = self._base(v, u, length, self.rng.uniform(0.8, 1.2))
        spacing = self._spacing(v)
        if self.junction_angle is None:
            score += self._angle_cost(v, u, length, self.angle, spacing, acute=False)
        elif length > 0:
            score += self._angle_cost(v, u, length, self.junction_angle, spacing, acute=True)
            score += self._angle_cost(u, v, length, self.junction_angle, self._spacing(u), acute=True)
        return score

    def _directions(self, v: int) -> list:
        out = list(self.fixed.get(v, ()))
        pv = self._tuples[v]
        for w in self.neighbours[v]:
            pw = self._tuples[w]
            oz, oy, ox = pw[0] - pv[0], pw[1] - pv[1], pw[2] - pv[2]
            norm = math.sqrt(oz * oz + oy * oy + ox * ox)
            if norm > 0:
                out.append((oz / norm, oy / norm, ox / norm))
        return out

    def junction_cost(self, v: int) -> tuple[float, bool]:
        """What *v*'s vessels cost for their angles to one another (as
        :meth:`_angle_cost` judges each pair), and whether a pair is closer
        than :data:`RELAX_BELOW_DEG`."""
        dirs = self._directions(v)
        if len(dirs) < 2 or self.junction_angle is None:
            return 0.0, False
        spacing = self._spacing(v)
        cost, sharp = 0.0, False
        for i in range(len(dirs)):
            a = dirs[i]
            for j in range(i + 1, len(dirs)):
                b = dirs[j]
                angle = math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))
                cost += JUNCTION_ANGLE_WEIGHT * spacing * abs(angle - self.junction_angle) / (math.pi / 2.0)
                if angle < _ACUTE:
                    cost += self.acute_weight * spacing * (1.0 - angle / _ACUTE)
                if angle < math.radians(RELAX_BELOW_DEG):
                    sharp = True
        return cost, sharp

    def edge_cost(self, v: int, u: int) -> float:
        length = float(np.linalg.norm(self.points[u] - self.points[v]))
        return 0.5 * (self._base(v, u, length) + self._base(u, v, length))

    def _joined(self, start: int, goal: int) -> bool:
        """Whether a path of the fill's edges runs from *start* to *goal*."""
        seen, frontier = {start}, [start]
        while frontier:
            here = frontier.pop()
            if here == goal:
                return True
            for there in self.neighbours[here]:
                if there not in seen:
                    seen.add(there)
                    frontier.append(there)
        return False

    def _rewire(self, v: int, a: int, c: int, d: int) -> None:
        """*v*-*a* and *c*-*d* become *v*-*c* and *a*-*d* in the neighbour
        lists alone (undone by calling it with *c* and *a* swapped)."""
        n = self.neighbours
        n[v].remove(a)
        n[a].remove(v)
        n[c].remove(d)
        n[d].remove(c)
        n[v].append(c)
        n[c].append(v)
        n[a].append(d)
        n[d].append(a)

    def relax_junctions(self, candidates: Mapping[int, Sequence[tuple[int, float]]], rounds: int = RELAX_ROUNDS) -> int:
        """Two edges trade ends -- *v*-*a* and *c*-*d* become *v*-*c* and
        *a*-*d*, both candidates -- wherever a vertex has an acute pair and
        the trade lowers its four vertices' junction cost and the two chords'
        departure from their wanted length together, and *v* and *a* stay
        joined. Every vertex keeps its degree, and the bed its pieces (so the
        loops the pieces are joined with stay the template's). Returns how
        many trades were made."""
        if self.junction_angle is None or self.acute_weight <= 0:
            return 0
        allowed = {v: {u for u, _l in items} for v, items in candidates.items()}
        trades = 0
        for _ in range(rounds):
            made = 0
            for v in range(len(self.points)):
                if not self.junction_cost(v)[1]:
                    continue
                options = []
                own = self.junction_cost(v)[0]
                for a in list(self.neighbours[v]):
                    for c in allowed.get(v, ()):
                        if c == a or self.forbidden(v, c):
                            continue
                        # Only a trade that widens *v* itself is worth pricing.
                        n_v = self.neighbours[v]
                        n_v[n_v.index(a)] = c
                        widened = self.junction_cost(v)[0] < own
                        n_v[n_v.index(c)] = a
                        if not widened:
                            continue
                        for d in list(self.neighbours[c]):
                            if d in (v, a) or d not in allowed.get(a, ()) or self.forbidden(a, d):
                                continue
                            nodes = {v, a, c, d}
                            before = sum(self.junction_cost(x)[0] for x in nodes) + self.edge_cost(v, a) + self.edge_cost(c, d)
                            self._rewire(v, a, c, d)
                            after = sum(self.junction_cost(x)[0] for x in nodes) + self.edge_cost(v, c) + self.edge_cost(a, d)
                            self._rewire(v, c, a, d)
                            delta = after - before
                            if delta < -1e-9:
                                options.append((delta, a, c, d))
                # The best trade that keeps *v* and *a* joined: one that cut
                # the bed in two would cost a loop when the pieces are joined.
                for _delta, a, c, d in sorted(options):
                    self.disconnect(v, a)
                    self.disconnect(c, d)
                    self.connect(v, c)
                    self.connect(a, d)
                    if self._joined(v, a):
                        made += 1
                        break
                    self.disconnect(v, c)
                    self.disconnect(a, d)
                    self.connect(v, a)
                    self.connect(c, d)
            trades += made
            if not made:
                break
        return trades

    def _angle_cost(self, v: int, u: int, length: float, target: float, spacing: float, *, acute: bool) -> float:
        """What the candidate *v*-*u* costs for its angles to the vessels
        already at *v*: their mean departure from *target*, and with *acute*,
        :data:`ACUTE_WEIGHT` for each it leaves within the acute angle of."""
        existing = self.neighbours[v]
        fixed = self.fixed.get(v, ()) if acute else ()
        if (not existing and not fixed) or length <= 0:
            return 0.0
        pv, pu = self._tuples[v], self._tuples[u]
        dz, dy, dx = (pu[0] - pv[0]) / length, (pu[1] - pv[1]) / length, (pu[2] - pv[2]) / length
        others = list(fixed)
        for w in existing:
            pw = self._tuples[w]
            oz, oy, ox = pw[0] - pv[0], pw[1] - pv[1], pw[2] - pv[2]
            norm = math.sqrt(oz * oz + oy * oy + ox * ox)
            if norm > 0:
                others.append((oz / norm, oy / norm, ox / norm))
        penalty, sharp = 0.0, 0.0
        for oz, oy, ox in others:
            angle = math.acos(max(-1.0, min(1.0, dz * oz + dy * oy + dx * ox)))
            penalty += abs(angle - target)
            if acute and angle < _ACUTE:
                sharp += 1.0 - angle / _ACUTE
        if not others:
            return 0.0
        weight = JUNCTION_ANGLE_WEIGHT if acute else ANGLE_WEIGHT
        return weight * spacing * (penalty / len(others)) / (math.pi / 2.0) + self.acute_weight * spacing * sharp

    def connect(self, i: int, j: int) -> None:
        self.neighbours[i].append(j)
        self.neighbours[j].append(i)
        self.degree[i] += 1
        self.degree[j] += 1
        self.edges.append((min(i, j), max(i, j)))

    def disconnect(self, i: int, j: int) -> None:
        self.neighbours[i].remove(j)
        self.neighbours[j].remove(i)
        self.degree[i] -= 1
        self.degree[j] -= 1
        self.edges.remove((min(i, j), max(i, j)))

    def augment(self, candidates: Mapping[int, list[tuple[int, float]]]) -> int:
        """Fill short vertices by rerouting: a short *v* whose candidate *u*
        is full takes *u*'s edge to some *w* away, when *w* can take a short
        *x* of its own instead -- two capacities filled, one edge moved.
        Returns how many edges were added."""
        added = 0
        short = [v for v in range(len(self.points)) if self.degree[v] < self.caps[v]]
        for v in short:
            while self.degree[v] < self.caps[v]:
                done = False
                for u, _length in candidates.get(v, ()):
                    if self.forbidden(v, u) or self.degree[u] < self.caps[u]:
                        continue
                    for w in list(self.neighbours[u]):
                        if w == v or self.attached[w] or self.caps[w] == 1:
                            continue
                        for x, _l in candidates.get(w, ()):
                            if x in (u, v) or self.degree[x] >= self.caps[x] or self.forbidden(w, x):
                                continue
                            self.disconnect(u, w)
                            self.connect(v, u)
                            self.connect(w, x)
                            added += 1
                            done = True
                            break
                        if done:
                            break
                    if done:
                        break
                if not done:
                    break
        return added

    def fill(self, candidates: Mapping[int, list[tuple[int, float]]], order: Sequence[int], passes: int = 3) -> None:
        for _ in range(passes):
            changed = False
            for v in order:
                while self.degree[v] < self.caps[v]:
                    best, best_score = None, math.inf
                    for u, length in candidates.get(v, ()):
                        if self.degree[u] >= self.caps[u] or self.forbidden(v, u):
                            continue
                        s = self.score(v, u, length)
                        if s < best_score:
                            best, best_score = u, s
                    if best is None:
                        break
                    self.connect(v, best)
                    changed = True
            if not changed:
                break


def join_pieces(fill: "_Fill", live: Sequence[int], linked: Sequence[tuple[int, int]] = ()) -> tuple[int, bool]:
    """Join every piece of the bed to the largest, one edge per piece (which
    keeps the loop count), and return ``(edges added, whether one piece is
    left)``. *live* are the vertices in play, *linked* pairs already joined
    outside the fill (a tree's nodes).

    Each pass joins the piece with the shortest join, between junctions
    (never through a terminal -- a root, a dead end -- nor two attached
    nodes directly), whatever their count: a member's nearest vertex of the
    largest piece, when that is a junction; else a lone terminal's nearest
    junction with room for it; else -- a piece whose junctions all have a
    dead end nearest, a junction whose every vessel is a dead end among them
    -- its nearest junction to the largest piece's nearest junction, past
    the dead ends.
    """
    points, caps, is_attached = fill.points, fill.caps, fill.attached
    parent = list(range(len(points)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in fill.edges:
        union(i, j)
    for a, b in linked:
        union(a, b)
    live = list(live)
    repairs = 0
    # Every pass joins one piece, so there are at most as many passes as
    # pieces; a pass that finds nothing to join ends the repair.
    passes_left = len(live) + 1
    while live and passes_left > 0:
        passes_left -= 1
        components: dict[int, list[int]] = {}
        for v in live:
            components.setdefault(find(v), []).append(v)
        if len(components) <= 1:
            return repairs, True
        giant_root = max(components, key=lambda r: len(components[r]))
        giant = components[giant_root]
        giant_tree = cKDTree(points[giant])
        hubs_in_giant = [g for g in giant if caps[g] > 1 and not is_attached[g]]
        hub_only = cKDTree(points[hubs_in_giant]) if hubs_in_giant else None
        best = None
        for root, members in components.items():
            if root == giant_root:
                continue
            d, idx = giant_tree.query(points[members], k=1)
            d, idx = np.atleast_1d(d), np.atleast_1d(idx)
            for m, dist, gi in sorted(zip(members, d, idx), key=lambda t: t[1]):
                g = giant[int(gi)]
                if fill.forbidden(m, g) or caps[m] == 1 or caps[g] == 1:
                    continue
                if best is None or dist < best[0]:
                    best = (float(dist), m, g)
                break
        if best is None and hub_only is not None:
            # A piece with no junction of its own (a lone terminal): it joins
            # the nearest junction of the giant piece, which takes one more.
            for root, members in components.items():
                if root == giant_root:
                    continue
                d, idx = hub_only.query(points[members], k=1)
                d, idx = np.atleast_1d(d), np.atleast_1d(idx)
                order_m = np.argsort(d)
                for i in order_m:
                    m = members[int(i)]
                    g = hubs_in_giant[int(idx[int(i)])]
                    if g not in fill.neighbours[m] and m != g and fill.degree[m] < caps[m] and not fill.crosses_hole(m, g):
                        best = (float(d[int(i)]), m, g)
                        break
                if best is not None:
                    break
        if best is None:
            best = _junction_join(fill, components, giant_root)
        if best is None:
            return repairs, False
        _, m, g = best
        fill.connect(m, g)
        union(m, g)
        repairs += 1
    return repairs, len({find(v) for v in live}) <= 1


def _junction_join(fill: "_Fill", components: Mapping[int, Sequence[int]], giant_root: int) -> Optional[tuple[float, int, int]]:
    """The shortest join from a junction of a piece to a junction of the
    largest piece *giant_root*, past any dead end nearer either: ``(length,
    piece's vertex, largest's vertex)``, or None when no piece has one."""
    points, caps = fill.points, fill.caps
    junctions = [g for g in components[giant_root] if caps[g] > 1]
    if not junctions:
        return None
    tree = cKDTree(points[junctions])
    k = min(len(junctions), JOIN_NEIGHBOURS)
    best = None
    for root, members in components.items():
        if root == giant_root:
            continue
        own = [m for m in members if caps[m] > 1]
        if not own:
            continue
        d, idx = tree.query(points[own], k=k)
        d, idx = np.asarray(d).reshape(len(own), -1), np.asarray(idx).reshape(len(own), -1)
        for row, m in enumerate(own):
            for dist, gi in zip(d[row], idx[row]):
                g = junctions[int(gi)]
                if fill.forbidden(m, g):
                    continue
                if best is None or dist < best[0]:
                    best = (float(dist), m, g)
                break
    return best


def _out_of_hole(
    points: np.ndarray,
    pairs: Optional[list],
    hole: Hole,
    margin: float,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Optional[list]]:
    """*points* not in *hole* nor within *margin* of it, thinned at random to
    *count* when there are more, and of *pairs* (a Voronoi bed's edges, index
    pairs; None for a lattice's) those between two kept points not crossing it."""
    keep = np.flatnonzero(~hole.contains(points, margin))
    if len(keep) > count:
        keep = keep[np.sort(rng.choice(len(keep), size=count, replace=False))]
    if pairs is not None:
        index = -np.ones(len(points), dtype=int)
        index[keep] = np.arange(len(keep))
        pairs = [
            (int(index[a]), int(index[b]))
            for a, b in pairs
            if index[a] >= 0 and index[b] >= 0 and not hole.crosses(points[a], points[b])
        ]
    return points[keep], pairs


def build_bed(
    draft: NetworkDraft,
    *,
    extent_um: Sequence[float],
    attached: Mapping[int, int],
    tree_groups: Sequence[Sequence[int]],
    keep_clear: Sequence[int],
    targets: DegreeTargets,
    face_ends: int,
    spacing_um: float,
    daughter_angle_deg: float,
    rng: np.random.Generator,
    max_degree: int = 4,
    category: str = "capillary",
    target_chord_um: Optional[float] = None,
    organisation: Optional[Organisation] = None,
    model: str = BED_MODEL_DELAUNAY,
    at_template_size: bool = True,
    junction_angle_deg: Optional[float] = None,
    acute_share: Optional[float] = None,
    hole: Optional[Hole] = None,
    hole_margin: float = 0.0,
) -> BedResult:
    """Add the bed to *draft*. *attached* maps the draft nodes the bed joins
    (tree leaves, or the roots themselves) to the bed edges each takes;
    *tree_groups* lists the nodes each tree already connects; *keep_clear*
    the nodes the lattice stays away from; *face_ends* how many of the
    targets' dead ends go on the faces; *target_chord_um* the chord the
    template's capillaries have on average (the fill prefers candidates near
    it; None, the shortest); *organisation*, the template's layout the bed
    follows (``synthetic.organisation``; None: even over the box), from its
    own vertices *at_template_size*; *model* one of :data:`BED_MODELS`;
    *junction_angle_deg* the template's own junction angle (its layout's
    angles summed over three; None without one: the fill judges the
    daughter angle at one end and nothing spreads it); *acute_share* the
    template's share of junctions with an acute pair (the acute weight's
    scale, :data:`UNAIDED_ACUTE_SHARE`). Given a *hole*, no vertex lies in
    it or within *hole_margin* of it and no edge crosses it. See the module."""
    if model not in BED_MODELS:
        raise ValueError(f"bed model must be one of {BED_MODELS}, got {model!r}")
    extent = np.asarray(extent_um, dtype=float)
    result = BedResult()
    attached_nodes = list(attached)
    clear = np.stack([draft.position(n) for n in keep_clear]) if keep_clear else None
    voronoi_pairs: Optional[list] = None

    def lay(count: int) -> tuple[np.ndarray, Optional[list]]:
        if model == BED_MODEL_VORONOI:
            return voronoi_vertices(
                extent, count, rng=rng, keep_clear=clear, spacing_um=spacing_um, organisation=organisation,
                at_template_size=at_template_size,
            )
        if drawn_from_template_vertices(organisation, at_template_size):
            return template_vertices(
                extent, count, rng=rng, keep_clear=clear, spacing_um=spacing_um, organisation=organisation
            ), None
        if organisation is not None:
            return density_vertices(
                extent, count, rng=rng, keep_clear=clear, spacing_um=spacing_um, organisation=organisation
            ), None
        return lattice_vertices(extent, count, rng=rng, keep_clear=clear, spacing_um=spacing_um), None

    if hole is None:
        bed_points, voronoi_pairs = lay(targets.vertices)
    else:
        # Laid over the whole box: as many more as the hole and the margin
        # round it take, and more again while too few land outside them.
        laid = int(math.ceil(targets.vertices / max(1.0 - hole.grown(hole_margin).share_of(extent), 0.05)))
        for attempt in range(HOLE_ATTEMPTS):
            bed_points, voronoi_pairs = lay(laid)
            outside = int(np.sum(~hole.contains(bed_points, hole_margin)))
            if outside >= targets.vertices or attempt == HOLE_ATTEMPTS - 1:
                break
            laid = int(math.ceil(laid * HOLE_REDRAW * targets.vertices / max(outside, 1)))
        bed_points, voronoi_pairs = _out_of_hole(bed_points, voronoi_pairs, hole, hole_margin, targets.vertices, rng)
    if len(bed_points) < targets.vertices:
        draft.note(f"the bed has {len(bed_points)} vertices of the {targets.vertices} wanted (no room beside the trees)")
    bed_nodes = [draft.add_node(p) for p in bed_points]
    result.vertices = len(bed_nodes)

    # Degrees: the dead ends nearest the faces go on them, the other dead
    # ends and the junction degrees fall at random.
    counts = dict(targets.degree_counts)
    ones = min(int(counts.get(1, 0)), len(bed_nodes))
    face_count = min(max(int(face_ends), 0), ones)
    distances = (
        np.minimum(bed_points.min(axis=1), (extent[None, :] - bed_points).min(axis=1))
        if len(bed_points)
        else np.array([])
    )
    nearest_faces = list(np.argsort(distances, kind="stable")[:face_count])
    others = [i for i in range(len(bed_nodes)) if i not in set(nearest_faces)]
    rng.shuffle(others)
    interior_ones = others[: ones - face_count]
    hubs = others[ones - face_count:]
    hub_degrees: list[int] = []
    for degree, how_many in sorted(counts.items()):
        if degree == 1:
            continue
        hub_degrees += [degree] * int(how_many)
    while len(hub_degrees) < len(hubs):
        hub_degrees.append(3)
    hub_degrees = hub_degrees[: len(hubs)]
    rng.shuffle(hub_degrees)

    local_caps = np.zeros(len(bed_nodes), dtype=int)
    for i in nearest_faces + interior_ones:
        local_caps[i] = 1
    for i, degree in zip(hubs, hub_degrees):
        local_caps[i] = min(int(degree), max_degree)
    for i in nearest_faces:
        p = bed_points[i]
        both = np.concatenate([p, extent - p])
        axis = int(np.argmin(both))
        p = p.copy()
        p[axis % 3] = 0.0 if axis < 3 else extent[axis % 3]
        bed_points[i] = p
        draft.positions[bed_nodes[i]] = p

    # One vertex set: attached nodes first, then the bed's.
    all_nodes = attached_nodes + bed_nodes
    points = np.stack([draft.position(n) for n in all_nodes]) if all_nodes else np.zeros((0, 3))
    caps = np.concatenate([np.array([attached[n] for n in attached_nodes], dtype=int), local_caps])
    is_attached = np.concatenate([np.ones(len(attached_nodes), bool), np.zeros(len(bed_nodes), bool)])
    local = tensors = None
    if organisation is not None:
        local = np.concatenate([
            np.full(len(attached_nodes), float(spacing_um)),
            local_spacings(points[len(attached_nodes):], extent, organisation, spacing_um),
        ])
        cells = cell_index(points, extent, organisation.shape)
        by_cell = np.stack([tensor_matrix(t) for t in organisation.orientation])
        tensors = by_cell[cells]
    class_chords = None if organisation is None else class_chord_ratios(organisation)
    fixed: dict[int, list] = {}
    if junction_angle_deg is not None:
        # A tree leaf's bed vessels are judged against its tree vessel too.
        for edge in draft.edges:
            for here, there in ((edge.u, edge.v), (edge.v, edge.u)):
                if here in attached:
                    step = draft.position(there) - draft.position(here)
                    norm = float(np.linalg.norm(step))
                    if norm > 0:
                        fixed.setdefault(attached_nodes.index(here), []).append(step / norm)
    fill = _Fill(
        points, caps, is_attached, rng, daughter_angle_deg, spacing_um, target_chord_um, local, tensors, class_chords,
        junction_angle_deg=junction_angle_deg, acute_share=acute_share, fixed_directions=fixed, hole=hole,
    )

    candidates: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    pairs = candidate_edges(points)
    if voronoi_pairs is not None:
        # A Voronoi bed's own vertices are joined along the cells' edges
        # only; the attached nodes, off the tessellation, as before.
        offset = len(attached_nodes)
        pairs = [(i, j, length) for i, j, length in pairs if is_attached[i] or is_attached[j]]
        pairs += [
            (offset + a, offset + b, float(np.linalg.norm(points[offset + a] - points[offset + b])))
            for a, b in voronoi_pairs
        ]
    for i, j, length in pairs:
        limit = MAX_ATTACHED_SPACINGS if (is_attached[i] or is_attached[j]) else MAX_CANDIDATE_SPACINGS
        reach = spacing_um if local is None else max(float(local[i]), float(local[j]))
        if length <= limit * reach:
            candidates[i].append((j, length))
            candidates[j].append((i, length))
    # A tree leaf sits off the lattice (it was kept clear): besides its
    # Delaunay edges it may reach the nearest junctions outright.
    hubs_index = [v for v in range(len(attached_nodes), len(all_nodes)) if caps[v] > 1]
    if attached_nodes and hubs_index:
        hub_tree = cKDTree(points[hubs_index])
        k = min(len(hubs_index), 12)
        for v in range(len(attached_nodes)):
            distances_v, indices_v = hub_tree.query(points[v], k=k)
            known = {u for u, _l in candidates[v]}
            for d, i in zip(np.atleast_1d(distances_v), np.atleast_1d(indices_v)):
                u = hubs_index[int(i)]
                if u not in known and d <= MAX_ATTACHED_SPACINGS * spacing_um:
                    candidates[v].append((u, float(d)))
                    candidates[u].append((v, float(d)))
    for lst in candidates.values():
        lst.sort(key=lambda item: item[1])
    # Attached nodes first (their few edges are fixed), then the dead ends
    # (each takes one junction's spare edge), then the junctions among
    # themselves: the other way round the junctions fill up among themselves
    # and the dead ends find nothing to hang from.
    order = list(range(len(attached_nodes)))
    rng.shuffle(order)
    ends = [v for v in range(len(attached_nodes), len(all_nodes)) if caps[v] == 1]
    rng.shuffle(ends)
    rest = [v for v in range(len(attached_nodes), len(all_nodes)) if caps[v] != 1]
    rng.shuffle(rest)
    fill.fill(candidates, order + ends + rest)
    result.phases["delaunay"] = int(sum(1 for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))
    result.phases["delaunay_attached"] = int(sum(1 for v in range(len(attached_nodes)) if fill.degree[v] < caps[v]))

    # Capacity the Delaunay candidates could not fill: the nearest vertices.
    short = [v for v in range(len(all_nodes)) if fill.degree[v] < caps[v]]
    fallback: dict[int, list[tuple[int, float]]] = {}
    if short and len(all_nodes) > 1:
        tree = cKDTree(points)
        k = min(FALLBACK_NEIGHBOURS + 1, len(all_nodes))
        for v in short:
            distances_v, indices_v = tree.query(points[v], k=k)
            fallback[v] = [
                (int(u), float(d))
                for d, u in zip(np.atleast_1d(distances_v), np.atleast_1d(indices_v))
                if int(u) != v and d <= MAX_ATTACHED_SPACINGS * spacing_um
            ]
        fill.fill(fallback, short, passes=1)
    result.phases["nearest"] = int(sum(1 for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))
    # What is still short is rerouted: a full neighbour gives up one of its
    # edges to the short vertex when the vertex it was joined to can take
    # another short one instead.
    merged_candidates: dict[int, list[tuple[int, float]]] = {}
    for v in range(len(all_nodes)):
        merged_candidates[v] = list(candidates.get(v, ())) + list(fallback.get(v, ())) if short else list(candidates.get(v, ()))
    for _ in range(3):
        if not fill.augment(merged_candidates):
            break
    result.phases["augmented"] = int(sum(1 for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))
    # Capacity still unfilled is paired up, nearest first, within reach:
    # every pair of ends left over is an edge, and a loop, the template asked
    # for; what is further apart than that stays short (and is reported).
    short = [v for v in range(len(all_nodes)) if fill.degree[v] < caps[v]]
    reach = MAX_ATTACHED_SPACINGS * spacing_um
    for round_reach in (reach, 2.0 * reach):
        while len(short) >= 2:
            short_tree = cKDTree(points[short])
            pairs = short_tree.query_pairs(r=round_reach, output_type="ndarray")
            best = None
            for a_i, b_i in pairs:
                a, b = short[int(a_i)], short[int(b_i)]
                if fill.forbidden(a, b):
                    continue
                d = float(np.linalg.norm(points[a] - points[b]))
                if best is None or d < best[0]:
                    best = (d, a, b)
            if best is None:
                break
            _d, a, b = best
            fill.connect(a, b)
            short = [v for v in short if fill.degree[v] < caps[v]]

    result.phases["paired"] = int(sum(1 for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))
    result.phases["paired_attached"] = int(sum(1 for v in range(len(attached_nodes)) if fill.degree[v] < caps[v]))
    # An attached node still short would be a degree-2 node (its tree edge
    # and one bed edge). It takes, in turn: the nearest junction with a
    # vessel to spare (up to *max_degree*), another short attached node
    # within reach (two tree leaves joined by one capillary), and, last, the
    # nearest junction whatever its count -- so every boundary node is a
    # junction.
    short_attached = [v for v in range(len(attached_nodes)) if fill.degree[v] < caps[v]]
    if short_attached and len(bed_nodes) and hubs_index:
        hub_tree = cKDTree(points[hubs_index])
        k = min(len(hubs_index), 32)
        for v in short_attached:
            _d, idx = hub_tree.query(points[v], k=k)
            for i in np.atleast_1d(idx):
                u = hubs_index[int(i)]
                if u in fill.neighbours[v] or fill.degree[u] >= max_degree or fill.crosses_hole(v, u):
                    continue
                fill.connect(v, u)
                if fill.degree[v] >= caps[v]:
                    break
        short_attached = [v for v in short_attached if fill.degree[v] < caps[v]]
        while len(short_attached) >= 2:
            pair_tree = cKDTree(points[short_attached])
            pairs = pair_tree.query_pairs(r=reach, output_type="ndarray")
            best = None
            for a_i, b_i in pairs:
                a, b = short_attached[int(a_i)], short_attached[int(b_i)]
                if b in fill.neighbours[a] or caps[a] == 1 or caps[b] == 1 or fill.crosses_hole(a, b):
                    continue
                d = float(np.linalg.norm(points[a] - points[b]))
                if best is None or d < best[0]:
                    best = (d, a, b)
            if best is None:
                break
            _d, a, b = best
            fill.connect(a, b)
            result.phases["leaf_pairs"] = result.phases.get("leaf_pairs", 0) + 1
            short_attached = [v for v in short_attached if fill.degree[v] < caps[v]]
        for v in short_attached:
            _d, idx = hub_tree.query(points[v], k=k)
            for i in np.atleast_1d(idx):
                u = hubs_index[int(i)]
                if u in fill.neighbours[v] or fill.crosses_hole(v, u):
                    continue
                fill.connect(v, u)
                result.phases["overfull_hubs"] = result.phases.get("overfull_hubs", 0) + 1
                if fill.degree[v] >= caps[v]:
                    break
    # A bed vertex meant to be a junction but left with one vessel would be
    # a dead end nobody asked for (in a closed network without interior dead
    # ends, the only one): it takes one more, to the nearest vertex with room.
    lone = [v for v in range(len(attached_nodes), len(all_nodes)) if fill.degree[v] == 1 and caps[v] >= 2]
    if lone:
        everything = cKDTree(points)
        for v in lone:
            if fill.degree[v] != 1:
                continue
            _d, idx = everything.query(points[v], k=min(len(points), 32))
            for i in np.atleast_1d(idx):
                u = int(i)
                if u == v or u in fill.neighbours[v] or is_attached[u] or caps[u] < 2 or fill.crosses_hole(v, u):
                    continue
                if fill.degree[u] == 0 or fill.degree[u] >= max_degree:
                    continue
                fill.connect(v, u)
                result.phases["lone_ends"] = result.phases.get("lone_ends", 0) + 1
                break
    result.short_of_degree = int(sum(caps[v] - fill.degree[v] for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))
    if voronoi_pairs is not None:
        offset = len(attached_nodes)
        cells = {(offset + a, offset + b) for a, b in voronoi_pairs}
        bed_only = [e for e in fill.edges if not is_attached[e[0]] and not is_attached[e[1]]]
        result.phases["voronoi_edges"] = int(sum(1 for e in bed_only if e in cells))
        result.phases["bed_only_edges"] = len(bed_only)

    # The junctions widened: edges trade ends where a vertex has an acute pair.
    result.phases["relaxed"] = fill.relax_junctions(merged_candidates)
    # Connectivity: every piece joined to the largest along the shortest pair.
    index_of = {node: i for i, node in enumerate(all_nodes)}
    linked: list[tuple[int, int]] = []
    for group in tree_groups:
        members = [index_of[n] for n in group if n in index_of]
        linked += list(zip(members[:-1], members[1:]))
    isolated = [v for v in range(len(attached_nodes), len(all_nodes)) if fill.degree[v] == 0]
    for v in isolated:
        draft.removed.add(all_nodes[v])
    result.dropped = len(isolated)
    live = [v for v in range(len(all_nodes)) if v not in set(isolated)]
    result.repairs, joined = join_pieces(fill, live, linked)
    if not joined:
        draft.note("a piece of the bed could not be joined to the rest")

    # Degree-2 vertices: their two edges become one through them (a parallel
    # edge when the two neighbours are joined already: two vessels between
    # one pair of junctions is a loop the statistics count like any other).
    entries: list = [[i, j, []] for i, j in fill.edges]
    incident: dict[int, list] = {v: [] for v in range(len(all_nodes))}
    for entry in entries:
        incident[entry[0]].append(entry)
        incident[entry[1]].append(entry)
    live_set = set(live)
    for v in range(len(attached_nodes), len(all_nodes)):
        if v not in live_set or is_attached[v] or len(incident[v]) != 2:
            continue
        first, second = incident[v]
        a = first[1] if first[0] == v else first[0]
        b = second[1] if second[0] == v else second[0]
        if a == b or a == v or b == v:
            # Both edges to one neighbour: merging would be a self-loop, so
            # the vertex takes a third edge to the nearest junction instead.
            hub_candidates = [
                u for u in hubs_index
                if u != v and u != a and all(u not in (e[0], e[1]) for e in incident[v]) and not fill.crosses_hole(v, u)
            ]
            if hub_candidates:
                nearest = min(hub_candidates, key=lambda u: float(np.linalg.norm(points[u] - points[v])))
                extra = [v, nearest, []]
                entries.append(extra)
                incident[v].append(extra)
                incident[nearest].append(extra)
            continue
        leg_a = list(reversed(first[2])) if first[0] == v else list(first[2])   # a -> v
        leg_b = list(second[2]) if second[0] == v else list(reversed(second[2]))  # v -> b
        merged = [a, b, leg_a + [points[v]] + leg_b]
        for entry in (first, second):
            entries.remove(entry)
            incident[entry[0]].remove(entry)
            incident[entry[1]].remove(entry)
        entries.append(merged)
        incident[a].append(merged)
        incident[b].append(merged)
        draft.removed.add(all_nodes[v])
        live_set.discard(v)
        result.merged += 1

    for i, j, via in entries:
        draft.add_edge(
            all_nodes[i], all_nodes[j], category=category, order=None, length_um=None, diameter_um=None,
            via=[np.asarray(p, float) for p in via], part=BED,
        )
    result.edges = len(entries)
    face_set = {bed_nodes[i] for i in nearest_faces}
    for v in range(len(attached_nodes), len(all_nodes)):
        if v in live_set and len(incident[v]) == 1:
            node = all_nodes[v]
            (result.face_ends if node in face_set else result.blind_ends).append(node)
    return result


__all__ = [
    "BED_MODELS",
    "BED_MODEL_DELAUNAY",
    "BED_MODEL_VORONOI",
    "BedResult",
    "DegreeTargets",
    "build_bed",
    "candidate_edges",
    "degree_sequence_targets",
    "density_vertices",
    "depths_like_template",
    "drawn_from_template_vertices",
    "template_vertices",
    "join_pieces",
    "local_spacings",
    "lattice_vertices",
    "voronoi_vertices",
]
