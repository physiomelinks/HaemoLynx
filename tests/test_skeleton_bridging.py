"""Skeleton bridging finds its candidates without comparing every pair.

``connect_skeleton_components`` used to query every voxel of every component
against every other component's tree: quadratic in the fragments, and in the
size of a large component every nearby fragment was measured against. It now
skips pairs whose bounding boxes are too far apart and measures each pair from
its smaller side -- which must give exactly the bridges the old loop gave,
down to which voxel pair a tie picks, since that is what gets drawn.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import cKDTree

import haemolynx.preprocessing.skeleton as skeleton_mod
from haemolynx.preprocessing.skeleton import (
    connect_skeleton_components,
    inter_component_gap_distances,
)


def _components(skeleton, z_weight):
    labeled, n = label(skeleton, structure=generate_binary_structure(3, 3))
    weights = np.array([z_weight, 1.0, 1.0])
    coords = np.argwhere(labeled)
    labels = labeled[tuple(coords.T)]
    order = np.argsort(labels, kind="stable")
    coords, labels = coords[order], labels[order]
    starts = np.searchsorted(labels, np.arange(1, n + 2))
    comp = {c: coords[starts[c - 1]:starts[c]] for c in range(1, n + 1)}
    return comp, weights


def _all_pairs_candidates(skeleton, max_distance, z_weight):
    """The loop this replaced: every pair, every voxel of the first queried."""
    comp, weights = _components(skeleton, z_weight)
    trees = {c: cKDTree(v * weights) for c, v in comp.items()}
    ids = sorted(comp)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            dists, idx = trees[b].query(comp[a] * weights)
            k = int(np.argmin(dists))
            if dists[k] <= max_distance:
                out.append((float(dists[k]), tuple(comp[a][k]), tuple(comp[b][int(idx[k])]), a, b))
    return sorted(out, key=lambda c: c[0])


def _pruned_candidates(skeleton, max_distance, z_weight):
    comp, weights = _components(skeleton, z_weight)
    weighted = {c: v * weights for c, v in comp.items()}
    trees = {c: cKDTree(w) for c, w in weighted.items()}
    out = []
    for a, b in skeleton_mod._pairs_within_reach(sorted(comp), weighted, max_distance):
        found = skeleton_mod._nearest_voxel_pair(
            weighted[a], trees[a], weighted[b], trees[b], max_distance=max_distance
        )
        if found is not None:
            d, i, j = found
            out.append((d, tuple(comp[a][i]), tuple(comp[b][j]), a, b))
    return sorted(out, key=lambda c: c[0])


def _fragments(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(6, 14)), int(rng.integers(16, 32)), int(rng.integers(16, 32)))
    skeleton = rng.random(shape) > rng.uniform(0.96, 0.99)
    for _ in range(int(rng.integers(1, 4))):  # a few long components
        skeleton[rng.integers(0, shape[0]), rng.integers(0, shape[1]), :] = True
    return skeleton


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("max_distance, z_weight", [(3.0, 1.0), (10.0, 1.0), (6.0, 2.0), (6.0, 0.5)])
def test_pruned_candidates_are_the_all_pairs_candidates(seed, max_distance, z_weight):
    """Same pairs, same distances, same voxel pair on every tie, same order."""
    skeleton = _fragments(seed)

    assert _pruned_candidates(skeleton, max_distance, z_weight) == _all_pairs_candidates(
        skeleton, max_distance, z_weight
    )


def test_pairs_out_of_reach_are_never_measured(monkeypatch):
    """Two clusters far apart: no pair across them is measured at all."""
    skeleton = np.zeros((5, 20, 200), dtype=bool)
    skeleton[2, 5, 0:3] = skeleton[2, 5, 5:8] = True  # near each other
    skeleton[2, 5, 150:153] = skeleton[2, 5, 156:159] = True  # far from the first two
    measured = []
    real = skeleton_mod._nearest_voxel_pair

    def recording(points_a, tree_a, points_b, tree_b, **kwargs):
        measured.append((len(points_a), len(points_b)))
        return real(points_a, tree_a, points_b, tree_b, **kwargs)

    monkeypatch.setattr(skeleton_mod, "_nearest_voxel_pair", recording)
    result = connect_skeleton_components(skeleton, max_bridge_distance=5)

    assert len(measured) == 2, "only the two pairs within reach"
    assert result[2, 5, 3:5].all() and result[2, 5, 153:156].all(), "both gaps bridged"


@pytest.mark.parametrize("seed", range(4))
def test_gap_distances_are_the_all_pairs_minimum(seed):
    skeleton = _fragments(seed)
    comp, _ = _components(skeleton, 1.0)
    trees = {c: cKDTree(v.astype(float)) for c, v in comp.items()}
    ids = sorted(comp)
    expected = sorted(
        float(trees[b].query(comp[a].astype(float))[0].min())
        for i, a in enumerate(ids)
        for b in ids[i + 1:]
    )

    assert np.array_equal(inter_component_gap_distances(skeleton), np.asarray(expected))
