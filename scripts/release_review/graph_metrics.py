"""How a built graph compares with a phantom's truth (``graph.8a``).

Pure: the graph comes in as node positions, degrees and edge lengths, the
truth as :func:`.phantoms.truth` writes it.

- **Junctions.** A built node of degree three or more matches a true
  junction when it lies within that junction's local radius (its widest
  vessel's), or within one voxel's longest side where that is larger -- the
  skeleton cannot place a junction finer than its voxels. Each true junction
  takes at most one built node, nearest first. Precision is the share of
  built junctions that match, recall the share of true ones matched.
- **Length.** The built graph's total centreline length against the true
  centrelines' length inside the image.
- **Loops.** The first Betti number (vessels - nodes + pieces) of each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class JunctionMatch:
    true_count: int
    built_count: int
    matched: int
    #: True junctions with no built one near them, and built ones with no
    #: true one: (z, y, x) in microns.
    missed: tuple[tuple[float, ...], ...]
    extra: tuple[tuple[float, ...], ...]

    @property
    def precision(self) -> Optional[float]:
        return self.matched / self.built_count if self.built_count else None

    @property
    def recall(self) -> Optional[float]:
        return self.matched / self.true_count if self.true_count else None


def match_junctions(
    true_positions: Sequence[Sequence[float]],
    true_radii: Sequence[float],
    built_positions: Sequence[Sequence[float]],
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> JunctionMatch:
    truth = np.asarray(true_positions, dtype=float).reshape(-1, 3)
    built = np.asarray(built_positions, dtype=float).reshape(-1, 3)
    floor = float(max(voxel_size_zyx))
    tolerance = np.maximum(np.asarray(true_radii, dtype=float), floor)
    pairs = []
    for i, point in enumerate(truth):
        distances = np.linalg.norm(built - point, axis=1) if len(built) else np.array([])
        for j, distance in enumerate(distances):
            if distance <= tolerance[i]:
                pairs.append((float(distance), i, j))
    taken_true, taken_built = set(), set()
    for _distance, i, j in sorted(pairs):
        if i in taken_true or j in taken_built:
            continue
        taken_true.add(i)
        taken_built.add(j)
    return JunctionMatch(
        true_count=len(truth),
        built_count=len(built),
        matched=len(taken_true),
        missed=tuple(tuple(round(float(v), 2) for v in truth[i]) for i in range(len(truth)) if i not in taken_true),
        extra=tuple(tuple(round(float(v), 2) for v in built[j]) for j in range(len(built)) if j not in taken_built),
    )


def betti_1(nodes: int, edges: int, pieces: int) -> int:
    return edges - nodes + pieces


def score_graph(graph, truth: Mapping[str, Any], voxel_size_zyx: Sequence[float]) -> dict[str, Any]:
    """Junction precision and recall, length error and loops of a built
    NetworkX graph (node ``pos`` in microns, edge ``length`` in microns)
    against :func:`.phantoms.truth`."""
    import networkx

    junction_nodes = [node for node in graph.nodes if graph.degree(node) >= 3]
    built = [np.asarray(graph.nodes[node]["pos"], dtype=float) for node in junction_nodes]
    match = match_junctions(
        [each["pos_zyx_um"] for each in truth["junctions"]],
        [each["radius_um"] for each in truth["junctions"]],
        built,
        voxel_size_zyx,
    )
    length = float(sum(float(data.get("length") or 0.0) for _u, _v, data in graph.edges(data=True)))
    true_length = float(truth["length_um"])
    pieces = networkx.number_connected_components(graph) if graph.number_of_nodes() else 0
    return {
        "junctions_true": match.true_count,
        "junctions_built": match.built_count,
        "junctions_matched": match.matched,
        "precision": match.precision,
        "recall": match.recall,
        "missed": list(match.missed),
        "extra": list(match.extra),
        "length_um": round(length, 1),
        "true_length_um": round(true_length, 1),
        "length_error": round((length - true_length) / true_length, 4) if true_length else None,
        "loops": betti_1(graph.number_of_nodes(), graph.number_of_edges(), pieces),
        "true_loops": truth["loops"],
    }


def meets_targets(score: Mapping[str, Any], junction_target: float, length_target: float) -> bool:
    precision, recall, error = score.get("precision"), score.get("recall"), score.get("length_error")
    return (
        precision is not None and recall is not None and error is not None
        and precision >= junction_target and recall >= junction_target and abs(error) <= length_target
    )
