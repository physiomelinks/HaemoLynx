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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .draft import BED, NetworkDraft

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
#: How much a candidate's length counts against its angle to the vessels
#: already at the vertex, in the fill's score (``weight * |angle - target| / 90 deg``
#: in spacings).
ANGLE_WEIGHT = 0.5
#: Nearest neighbours tried when the Delaunay candidates run out.
FALLBACK_NEIGHBOURS = 16


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

    def __init__(self, points: np.ndarray, caps: np.ndarray, attached: np.ndarray, rng: np.random.Generator, angle_deg: float, spacing_um: float, target_chord_um: Optional[float] = None):
        self.points = points
        self._tuples = [tuple(float(c) for c in p) for p in points]
        self.caps = caps.astype(int)
        self.attached = attached.astype(bool)
        self.rng = rng
        self.angle = math.radians(angle_deg)
        self.spacing = spacing_um
        #: The chord the template's capillaries have on average; None takes
        #: the shortest candidate.
        self.target_chord = target_chord_um
        self.degree = np.zeros(len(points), dtype=int)
        self.neighbours: dict[int, list[int]] = {i: [] for i in range(len(points))}
        self.edges: list[tuple[int, int]] = []

    def forbidden(self, i: int, j: int) -> bool:
        if i == j or j in self.neighbours[i]:
            return True
        if self.attached[i] and self.attached[j]:
            return True
        return bool(self.caps[i] == 1 and self.caps[j] == 1)

    def score(self, v: int, u: int, length: float) -> float:
        wanted = self.target_chord if self.target_chord else 0.0
        score = abs(length - wanted) * self.rng.uniform(0.8, 1.2)
        existing = self.neighbours[v]
        if existing and length > 0:
            pv, pu = self._tuples[v], self._tuples[u]
            dz, dy, dx = pu[0] - pv[0], pu[1] - pv[1], pu[2] - pv[2]
            penalty, n = 0.0, 0
            for w in existing:
                pw = self._tuples[w]
                oz, oy, ox = pw[0] - pv[0], pw[1] - pv[1], pw[2] - pv[2]
                other = math.sqrt(oz * oz + oy * oy + ox * ox)
                if other > 0:
                    cos = (dz * oz + dy * oy + dx * ox) / (length * other)
                    penalty += abs(math.acos(max(-1.0, min(1.0, cos))) - self.angle)
                    n += 1
            if n:
                score += ANGLE_WEIGHT * self.spacing * (penalty / n) / (math.pi / 2.0)
        return score

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
) -> BedResult:
    """Add the bed to *draft*. *attached* maps the draft nodes the bed joins
    (tree leaves, or the roots themselves) to the bed edges each takes;
    *tree_groups* lists the nodes each tree already connects; *keep_clear*
    the nodes the lattice stays away from; *face_ends* how many of the
    targets' dead ends go on the faces; *target_chord_um* the chord the
    template's capillaries have on average (the fill prefers candidates near
    it; None, the shortest). See the module."""
    extent = np.asarray(extent_um, dtype=float)
    result = BedResult()
    attached_nodes = list(attached)
    clear = np.stack([draft.position(n) for n in keep_clear]) if keep_clear else None
    bed_points = lattice_vertices(extent, targets.vertices, rng=rng, keep_clear=clear, spacing_um=spacing_um)
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
    fill = _Fill(points, caps, is_attached, rng, daughter_angle_deg, spacing_um, target_chord_um)

    candidates: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(all_nodes))}
    for i, j, length in candidate_edges(points):
        limit = MAX_ATTACHED_SPACINGS if (is_attached[i] or is_attached[j]) else MAX_CANDIDATE_SPACINGS
        if length <= limit * spacing_um:
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
                if u in fill.neighbours[v] or fill.degree[u] >= max_degree:
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
                if b in fill.neighbours[a] or caps[a] == 1 or caps[b] == 1:
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
                if u in fill.neighbours[v]:
                    continue
                fill.connect(v, u)
                result.phases["overfull_hubs"] = result.phases.get("overfull_hubs", 0) + 1
                if fill.degree[v] >= caps[v]:
                    break
    result.short_of_degree = int(sum(caps[v] - fill.degree[v] for v in range(len(all_nodes)) if fill.degree[v] < caps[v]))

    # Connectivity: every piece joined to the largest along the shortest pair.
    parent = list(range(len(all_nodes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    index_of = {node: i for i, node in enumerate(all_nodes)}
    for i, j in fill.edges:
        union(i, j)
    for group in tree_groups:
        members = [index_of[n] for n in group if n in index_of]
        for a, b in zip(members[:-1], members[1:]):
            union(a, b)
    isolated = [v for v in range(len(attached_nodes), len(all_nodes)) if fill.degree[v] == 0]
    for v in isolated:
        draft.removed.add(all_nodes[v])
    result.dropped = len(isolated)
    live = [v for v in range(len(all_nodes)) if v not in set(isolated)]
    # Every pass joins one piece, so there are at most as many passes as
    # pieces; a pass that finds nothing to join ends the repair.
    passes_left = len(live) + 1
    while live and passes_left > 0:
        passes_left -= 1
        components: dict[int, list[int]] = {}
        for v in live:
            components.setdefault(find(v), []).append(v)
        if len(components) <= 1:
            break
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
            # Never join through a terminal (a root, a dead end: one edge
            # each) nor two attached nodes directly: the join goes between
            # junctions, whatever their count.
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
                    if g not in fill.neighbours[m] and m != g and fill.degree[m] < caps[m]:
                        best = (float(d[int(i)]), m, g)
                        break
                if best is not None:
                    break
        if best is None:
            draft.note("a piece of the bed could not be joined to the rest")
            break
        _, m, g = best
        fill.connect(m, g)
        union(m, g)
        result.repairs += 1

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
            hub_candidates = [u for u in hubs_index if u != v and u != a and all(u not in (e[0], e[1]) for e in incident[v])]
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
    "BedResult",
    "DegreeTargets",
    "build_bed",
    "candidate_edges",
    "degree_sequence_targets",
    "lattice_vertices",
]
