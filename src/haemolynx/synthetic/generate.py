"""Draw a synthetic vascular network to a :class:`~.template.NetworkTemplate`.

:func:`generate_network` places the template's inlets and outlets on the
faces they were on, grows the arterial and venous trees from them
(:mod:`.trees`), hands their leaves to a capillary bed built to the
template's capillary count, loop count and dead-end count (:mod:`.bed`),
gives every vessel a length and a width drawn from the template, bends each
into a polyline of that length (:mod:`.polylines`) and returns the
``nx.MultiGraph`` the pipeline's Boundaries stage takes: node ``pos`` and
edge ``voxels`` in (z, y, x) microns, ``length``, ``diameter_um`` with
``diameter_source="synthetic"``, the node roles in :data:`SYNTHETIC_ROLES`
(:func:`boundary_nodes` reads them), ``dead_end_kind`` on every dead end,
and the image extent and voxel size on the graph. Nothing here is a branch
order: ``graph.assign_vessel_branch_orders`` labels the network from the
roles, as it does an image run's.

Every draw goes through one ``numpy`` generator, so a seed repeats a network
exactly. :func:`validate_synthetic_graph` checks the contract.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph._helpers import calculate_path_length
from haemolynx.graph.dead_ends import (
    BLIND,
    DEAD_END_KIND,
    DEAD_END_KIND_SOURCE,
    LEAVES_IMAGE,
    SOURCE_SYNTHETIC,
    graph_image_extent_um,
)
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes
from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_SYNTHETIC

from .bed import build_bed, degree_sequence_targets
from .draft import BED, TREE, NetworkDraft
from .polylines import DEFAULT_POINTS, bend_path, bend_to_lengths, densify_paths
from .template import (
    DEFAULT_DAUGHTER_ANGLE_DEG,
    GRAPH_NETWORK_SOURCE,
    GRAPH_TEMPLATE_RECORD,
    INTERIOR,
    ROLES,
    NetworkTemplate,
    lognormal_from_mean_median,
    lognormal_from_median_p10_p90,
)
from .trees import SideResult, face_distance_um, grow_side, inward_normal, snap_to_face

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


@dataclass(frozen=True)
class GenerationOptions:
    scale: float = 1.0
    interior_dead_ends: bool = True
    max_bed_degree: int = 4
    daughter_angle_sd_deg: Optional[float] = None
    #: None: half the bed spacing.
    min_separation_um: Optional[float] = None
    voxel_step_um: float = 0.5
    bend_points: int = DEFAULT_POINTS


@dataclass
class _Side:
    name: str
    roots: list
    result: Optional[SideResult] = None
    boundary_role: str = ""
    large_role: str = ""
    boundary_count: int = 0
    blind: list = None  # type: ignore[assignment]
    face: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.blind = []
        self.face = []


def _place_roots(draft: NetworkDraft, template: NetworkTemplate, extent: np.ndarray, rng: np.random.Generator) -> dict:
    """One node per template inlet and outlet at its relative position,
    exactly on its face; duplicates (a scaled template's) spread apart."""
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
    if not interior:
        return []
    wanted = side.boundary_count if interior_dead_ends else len(interior)
    if wanted <= 0:
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
    # tree keeps its deepest interior leaf, or, with none, one at a face.
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


def _capillary_diameters(template: NetworkTemplate, count: int, rng: np.random.Generator) -> np.ndarray:
    cls = template.class_target("capillary")
    samples = cls.samples.diameters_um if cls is not None and cls.samples is not None else ()
    if samples:
        return rng.choice(np.asarray(samples, float), size=count, replace=True)
    mean = (cls.mean_diameter_um if cls else None) or FALLBACK_CAPILLARY_DIAMETER_UM
    median = (cls.median_diameter_um if cls else None) or mean * 0.96
    mu, sigma = lognormal_from_mean_median(max(mean, median * 1.001), median)
    return rng.lognormal(mu, sigma, size=count)


def _template_record(template: NetworkTemplate) -> dict:
    """The template as a plain dict without its per-vessel samples."""
    data = template.to_dict()
    for item in data["classes"].values():
        item["samples"] = None
    for item in data["orders"]:
        item["samples"] = None
    data["loops"]["loop_length_samples_um"] = []
    data["dead_ends"]["chain_length_samples_um"] = []
    return data


def generate_network(
    template: NetworkTemplate,
    *,
    seed: Optional[int] = DEFAULT_SEED,
    rng: Optional[np.random.Generator] = None,
    options: Optional[GenerationOptions] = None,
    validate: bool = True,
) -> nx.MultiGraph:
    """A synthetic network drawn to *template*. See the module.

    *rng* wins over *seed*; ``seed=None`` draws from fresh entropy. Raises
    ``ValueError`` for a template with no capillaries or no inlets, and,
    with *validate*, for a network that breaks the contract.
    """
    options = options or GenerationOptions()
    rng = rng if rng is not None else np.random.default_rng(seed)
    T = template.scaled(options.scale)
    extent = np.asarray(T.extent_um, dtype=float)
    if not T.inlets or not T.outlets:
        raise ValueError("the template has no inlets or no outlets to draw from")
    capillaries = T.class_target("capillary")
    capillary_count = int(capillaries.count) if capillaries is not None else 0
    if capillary_count < 1:
        raise ValueError("the template has no capillaries to draw a bed from")

    draft = NetworkDraft()
    roots = _place_roots(draft, T, extent, rng)
    root_count = len(roots["inlet"]) + len(roots["outlet"])

    # The bed's arithmetic first: its spacing is what the trees fill space to.
    loops = T.loops.independent_loops
    if loops is None:
        loops = int(round(capillary_count / 3.0))
    bed_vertices = max(capillary_count - loops - root_count + 1, 1)
    spacing = float((np.prod(extent) / bed_vertices) ** (1.0 / 3.0))
    min_separation = options.min_separation_um if options.min_separation_um is not None else 0.5 * spacing
    daughter_angle = T.daughter_angle_deg if T.daughter_angle_deg else DEFAULT_DAUGHTER_ANGLE_DEG

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
        side.result = grow_side(
            draft, T, side.name, side.roots,
            rng=rng, spacing_um=spacing, extent_um=extent, min_separation_um=min_separation,
            daughter_angle_sd_deg=options.daughter_angle_sd_deg,
        )
        tree_nodes.extend(side.result.nodes)
        for node in side.result.large_boundary:
            draft.add_role(node, side.large_role)
        for node in _choose_boundary_leaves(side, options.interior_dead_ends, rng, draft):
            draft.add_role(node, side.boundary_role)
            attached[node] = LEAF_BED_EDGES
        for root, nodes in side.result.trees.items():
            tree_groups.append([root] + list(nodes))

    face_ends_on_trees = sum(len(side.face) for side in sides)
    blind_on_trees = sum(len(side.blind) for side in sides)
    roots_on_faces = sum(1 for side in sides for _node, face in side.roots if face != INTERIOR)
    bed_face_ends = max(T.dead_ends.leaves_image - roots_on_faces - face_ends_on_trees, 0)
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
            "spacing_um": spacing,
            "notes": list(draft.notes),
        },
    }
    if validate:
        validate_synthetic_graph(G)
    return G


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
    total length for that class, never below a vessel's chord."""
    by_class: dict = {}
    for edge in draft.edges:
        if edge.part == TREE:
            by_class.setdefault(edge.category, []).append(edge)
    for category, edges in by_class.items():
        cls = template.class_target(category)
        if cls is None or not cls.total_length_um:
            continue
        chords = {id(e): _chord(draft, e) for e in edges}
        for _ in range(3):
            total = sum(float(e.length_um or 0.0) for e in edges)
            if total <= 0:
                break
            factor = cls.total_length_um / total
            for e in edges:
                e.length_um = max(float(e.length_um or 0.0) * factor, chords[id(e)] * (1.0 + 1e-6))


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
    edges = [e for e in draft.edges if e.u not in draft.removed and e.v not in draft.removed]
    simple = [e for e in edges if not e.via]
    paths: list = []
    if simple:
        starts = np.stack([draft.position(e.u) for e in simple])
        ends = np.stack([draft.position(e.v) for e in simple])
        targets = np.array([float(e.length_um or 0.0) for e in simple])
        paths = list(bend_to_lengths(starts, ends, targets, rng, points=options.bend_points))
    with_via = [e for e in edges if e.via]
    for edge in with_via:
        paths.append(bend_path([draft.position(edge.u), *edge.via, draft.position(edge.v)], float(edge.length_um or 0.0), rng, bend_points=options.bend_points))
    voxels, lengths = densify_paths(paths, step_um=options.voxel_step_um)
    for edge, edge_voxels, length in zip(simple + with_via, voxels, lengths):
        _add_edge(G, new_id, edge, edge_voxels, float(length))
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
    return G


def _add_edge(G: nx.MultiGraph, new_id: Mapping[int, int], edge: Any, voxels: list, length: float) -> None:
    attributes = {SYNTHETIC_CLASS: edge.category}
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
        pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
        # NetworkX yields an undirected edge's ends in node order, so the
        # polyline may run either way; it must touch both nodes.
        forward = np.allclose(arr[0], pu, atol=1e-6) and np.allclose(arr[-1], pv, atol=1e-6)
        backward = np.allclose(arr[0], pv, atol=1e-6) and np.allclose(arr[-1], pu, atol=1e-6)
        if not (forward or backward):
            problems.append(f"edge {u}-{v}'s polyline does not join its nodes")
        if np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)) > 0.5 + 1e-6:
            problems.append(f"edge {u}-{v}'s polyline steps more than 0.5 um")
        if not math.isclose(float(data.get("length", -1.0)), calculate_path_length(voxels), rel_tol=1e-6, abs_tol=1e-6):
            problems.append(f"edge {u}-{v}'s length is not its polyline's")
        if not (data.get("diameter_um") or 0) > 0:
            problems.append(f"edge {u}-{v} has no diameter")
        if data.get("diameter_source") != DIAMETER_SOURCE_SYNTHETIC:
            problems.append(f"edge {u}-{v} is not marked {DIAMETER_SOURCE_SYNTHETIC}")
        if data.get("branch_order") is not None:
            problems.append(f"edge {u}-{v} already has a branch order")
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
    "SYNTHETIC",
    "SYNTHETIC_CLASS",
    "SYNTHETIC_ROLES",
    "boundary_nodes",
    "generate_network",
    "validate_synthetic_graph",
]
