"""A persistence-based alternative to a single, hand-tuned collapse distance.

``collapse.collapse_node_clusters`` (``cluster_collapse_method="distance_only"``)
merges every node within one fixed ``distance_threshold`` of another, however
far that chains -- see that module's own docstring for how this over-merges a
densely braided region into a cartwheel hub. ``direction_aware_collapse``
fixes that by refusing a merge that would spoke out in every direction.
This module attacks the same problem from a different, purely geometric
angle: instead of one global distance deciding every merge everywhere, it
asks each local cluster of nearby nodes what distance *its own* points
actually separate into "clearly the same point" versus "clearly a different
one" -- a scale the data sets for itself, rather than one the whole image is
held to.

The mathematics this rests on is a real, cited result, not an invented
heuristic: for a finite point set, the sequence in which single-linkage
hierarchical clustering merges points, as the linkage distance grows, is
*exactly* the 0-dimensional persistent homology of the point set's
Vietoris-Rips filtration -- every point is a connected-component feature
"born" at filtration value 0, and it "dies" (merges into an older component)
at the distance single linkage would join it at. Carlsson & Mémoli,
"Characterization, Stability and Convergence of Hierarchical Clustering
Methods", JMLR 11:1425-1470 (2010), prove this correspondence formally
(their Theorem 6 and the single-linkage discussion in Sec. 3-4). The
*persistence* of each such feature -- how long it survives before merging --
is the standard signal topological data analysis uses to tell real structure
from noise: a feature that persists a long time relative to the others is
significant, one that dies almost immediately is not. Deciding a cutoff from
a gap in the sorted list of persistence values, rather than fixing one
number in advance, is the same principle Chazal, Guibas, Oudot & Skraba use
in ToMATo ("Persistence-Based Clustering in Riemannian Manifolds", JACM
60(6):41, 2013) to separate genuine modes of a density from sampling noise.

What this module actually computes, per locally-proximate cluster of nodes:

1. The minimum spanning tree of the cluster's own pairwise distances. A
   classic equivalence (e.g. Gower & Ross 1969) makes an MST's edge weights,
   taken in increasing order, exactly the single-linkage merge sequence --
   so this *is* the local 0-dimensional persistence diagram's death times,
   computed the cheap way, without building a full Vietoris-Rips complex.
2. The biggest jump between consecutive sorted merge distances, measured as
   a fraction of the cluster's own spread (see ``MIN_RELATIVE_GAP`` for why
   the spread, not the neighbouring value, is the steadier yardstick with as
   few distances as a typical local cluster has) -- the persistence gap.
   Below it: merges that happen quickly, one after another, at close range
   -- the signature of skeletonisation noise sharing one real point. Above
   it: a separately-persistent join, more likely two real, distinct pieces
   of structure that only happen to be nearby.
3. Only the below-gap merges are carried out. A cluster with no clear gap
   (a smooth run of similar distances, nothing standing out) falls back to
   accepting everything within the ordinary ``cluster_collapse_distance`` --
   the same answer ``distance_only`` would give, so this is never worse than
   that method when the data offers no signal to act on.

This is a graph-level, single-linkage-equivalent adaptation of the
persistence idea, not the heavier machinery discrete Morse theory / DisPerSE-
style pipelines use for continuous fields (a Delaunay or cubical complex
built over the whole image's own intensity, with discrete gradient vector
fields and critical-cell cancellation -- see e.g. Sousbie, "The persistent
cosmic web and its filamentary structure I: Theory and implementation",
MNRAS 414(1):350-383, 2011, for the method, and "ToFiE: a Topology-aware
Fiber Extraction workflow" (arXiv:2604.18230, 2026, github.com/peirlincklab/
ToFiE) for its use on dense biological fibre networks -- built on the same
DisPerSE software Sousbie's paper describes, run via its own persistence-
threshold parameter, not a from-scratch implementation). Building that
directly on this pipeline's binary vessel volumes, rather than on this
already-extracted skeleton graph, remains future work -- there is no
ready vasculature-specific implementation of it to adapt, and it would be a
genuine research project rather than the graph-level adaptation here.
"""
from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from .direction_aware_collapse import _rewire_edges_deduplicating

logger = logging.getLogger(__name__)

#: How much further than cluster_collapse_distance to look for candidate
#: pairs -- the persistence gap can only be found if both sides of it (the
#: close, noise-scale merges and whatever separately-persistent join comes
#: after) are actually in view. Too small a search and everything in range
#: looks equally close, with no gap to find.
DEFAULT_SEARCH_RADIUS_MULTIPLE = 3.0

#: A candidate gap must span at least this fraction of the cluster's whole
#: merge-distance spread (max minus min) to count as a genuine elbow rather
#: than noise -- below this, there is no persistence signal to act on, and
#: every merge within the ordinary cluster_collapse_distance is accepted
#: (matching distance_only).
#:
#: Measured against the *spread*, not against the merge distance just below
#: the gap: an earlier version compared the gap to its own preceding value
#: (a common way to describe an "elbow"), but with as few as 3-7 merge
#: distances in a typical local cluster, that ratio is dominated by which
#: particular pair of order statistics the gap happens to fall between --
#: repeating the same clearly-unstructured 8-node cluster this module's own
#: tests use, at 2000 different random seeds, threw a spurious "gap" ratio
#: above 2.0 (i.e. the step more than tripling) in 3.5% of draws on chance
#: alone. Measuring the gap as a fraction of the cluster's *whole* spread
#: is far steadier: the same 2000-seed sweep never exceeded 0.89, and a
#: real two-tight-pairs split (this module's other main test) reads 1.0 --
#: the single gap *is* the entire spread when there are genuinely only two
#: distance scales present.
MIN_RELATIVE_GAP = 0.85


def _minimum_spanning_tree_edges(coords: np.ndarray) -> list[tuple[int, int, float]]:
    """The cluster's single-linkage merge sequence: MST edges (i, j, weight),
    sorted by weight -- see this module's docstring for why an MST's edge
    weights, in increasing order, are exactly this."""
    n = len(coords)
    if n < 2:
        return []
    distances = squareform(pdist(coords))
    mst = minimum_spanning_tree(csr_matrix(distances)).tocoo()
    edges = [(int(i), int(j), float(w)) for i, j, w in zip(mst.row, mst.col, mst.data)]
    edges.sort(key=lambda edge: edge[2])
    return edges


def _persistence_cutoff(sorted_weights: list[float]) -> float | None:
    """The distance at the biggest gap in *sorted_weights*, or ``None`` when
    no gap spans enough of the cluster's own spread to act on -- see
    ``MIN_RELATIVE_GAP`` for why that is measured against the spread rather
    than against the merge distance just below the gap."""
    if len(sorted_weights) < 2:
        return None
    weights = np.asarray(sorted_weights, dtype=float)
    spread = weights[-1] - weights[0]
    if spread <= 0.0:
        return None
    gaps = np.diff(weights)
    relative_gaps = gaps / spread
    best = int(np.argmax(relative_gaps))
    if relative_gaps[best] < MIN_RELATIVE_GAP:
        return None
    return float(weights[best])


def collapse_node_clusters_persistence(
    G: Union[nx.Graph, nx.MultiGraph],
    distance_threshold: float = 5.0,
    *,
    search_radius_multiple: float = DEFAULT_SEARCH_RADIUS_MULTIPLE,
    debug: bool = False,
    max_iterations: int = 1,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Collapse nearby node clusters, cutting each one at its own natural gap.

    Candidate pairs are gathered out to ``distance_threshold *
    search_radius_multiple`` (wider than the ordinary collapse distance, so a
    genuine gap has room to appear on both sides of it); each resulting
    proximity-connected group is then cut at its own persistence gap (see
    :func:`_persistence_cutoff`) rather than merged wholesale, so a chained
    proximity component that spans more than one real, separately-persistent
    piece of structure can come back as more than one representative node.

    *max_iterations* defaults to 1, unlike ``collapse_node_clusters`` and
    ``collapse_node_clusters_direction_aware``: a second pass recomputes the
    gap from whatever a first pass left behind, and two already-well-formed
    representative nodes are, by then, exactly the "too few points to show a
    gap" case :func:`_persistence_cutoff` falls back to plain
    ``distance_threshold`` for -- so a second pass can quietly remerge a
    split the first pass correctly made, which a synthetic two-tight-pairs
    fixture in this module's own tests demonstrates directly. Raising this is
    a deliberate opt-in a caller can still make, not a default worth
    reaching for.

    Raises
    ------
    ValueError
        If *search_radius_multiple* is negative. 0.0 is legal and degenerate
        -- a zero-width search finds no candidate pairs, so the method
        never merges anything, rather than raising.
    """
    if search_radius_multiple < 0.0:
        raise ValueError("search_radius_multiple must be >= 0")

    G = G.copy()
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    total_merged = 0
    search_radius = distance_threshold * search_radius_multiple

    for iteration in range(max_iterations):
        nodes_with_pos = [
            (n, np.array(G.nodes[n]["pos"], dtype=float))
            for n in G.nodes()
            if "pos" in G.nodes[n]
        ]
        if len(nodes_with_pos) < 2:
            break

        node_ids = [n for n, _ in nodes_with_pos]
        coords = np.array([p for _, p in nodes_with_pos])

        tree = cKDTree(coords)
        pairs = tree.query_pairs(search_radius)
        if not pairs:
            break

        proximity = nx.Graph()
        proximity.add_nodes_from(range(len(node_ids)))
        proximity.add_edges_from(pairs)

        merged_this_iter = 0
        for component in nx.connected_components(proximity):
            if len(component) < 2:
                continue
            indices = sorted(component)
            local_ids = [node_ids[i] for i in indices]
            local_coords = coords[indices]

            mst_edges = _minimum_spanning_tree_edges(local_coords)
            if not mst_edges:
                continue
            weights = [w for _, _, w in mst_edges]
            cutoff = _persistence_cutoff(weights)
            accept_within = distance_threshold if cutoff is None else cutoff

            parent = list(range(len(local_ids)))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i, j, weight in mst_edges:
                if weight > accept_within:
                    continue
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

            groups: dict[int, list] = {}
            for local_index, node_id in enumerate(local_ids):
                groups.setdefault(find(local_index), []).append(node_id)

            for group in groups.values():
                if len(group) < 2:
                    continue
                group = [n for n in group if G.has_node(n)]
                if len(group) < 2:
                    continue

                rep = max(group, key=lambda n: (G.degree(n), -n))
                others = [n for n in group if n != rep]

                if debug:
                    logger.info(
                        "Collapsing cluster of %d nodes %s -> representative "
                        "%s (persistence cutoff=%s, search=%d nodes)",
                        len(group), group, rep, cutoff, len(local_ids),
                    )

                cluster_positions = np.array(
                    [G.nodes[n]["pos"] for n in group if "pos" in G.nodes[n]]
                )
                G.nodes[rep]["pos"] = cluster_positions.mean(axis=0)

                for other in others:
                    if not G.has_node(other):
                        continue
                    _rewire_edges_deduplicating(G, other, rep, is_multi)
                    G.remove_node(other)
                    merged_this_iter += 1

        total_merged += merged_this_iter
        if debug:
            logger.info(
                "Iteration %d: merged %d nodes (%d total merged)",
                iteration + 1, merged_this_iter, total_merged,
            )
        if merged_this_iter == 0:
            break

    self_loops = list(nx.selfloop_edges(G, keys=True)) if is_multi else [
        (u, v) for u, v in nx.selfloop_edges(G)
    ]
    if self_loops:
        if is_multi:
            for u, v, k in self_loops:
                G.remove_edge(u, v, key=k)
        else:
            G.remove_edges_from(self_loops)
        if debug:
            logger.info("Removed %d self-loops after collapsing", len(self_loops))

    if debug or total_merged > 0:
        logger.info(
            "collapse_node_clusters_persistence: merged %d nodes, "
            "graph now has %d nodes / %d edges",
            total_merged, G.number_of_nodes(), G.number_of_edges(),
        )
    return G
