"""A template averaged over several runs: a batch condition's network
(``PLAN_batch_templates.md``; Harvey, 2026-10-25: B1a, B2a, B3a).

:func:`average_templates` takes the templates of a condition's runs and
returns one :class:`~.template.NetworkTemplate` a synthetic network can be
drawn to:

- **The box** (B1a) is each axis's mean extent. Every count is the members'
  mean density times that box's volume: a member *f* times the box's volume
  counts *f* times its own (:func:`common_volume_factors`); inlets and outlets
  go with the faces' area.
- **Distributions** are pooled where the members carry samples (the graph
  source) and averaged where they carry moments, a class's or an order's
  means weighted by its members' counts. The order table is taken label by
  label, a label a member lacks counting 0 for it; an order whose mean count
  rounds to none is dropped.
- **The layout** (B2a, ``synthetic.organisation``) is pooled: the per-cell
  fields carried onto the common grid by overlap (cells are shares of each
  box, totals kept) and averaged as densities, the members'
  capillary vertices pooled (as shares of their boxes) for the bed. This
  assumes the members were imaged the same way up.
- **Inlets and outlets** (B3a): the mean count of each, rounded and never
  none, on the faces most members use them, at positions pooled from theirs
  (:func:`representative_positions`).

:func:`common_box_members` gives each member scaled to the common volume, for
the comparison's members' range (B4b).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np

from .organisation import Organisation, grid_shape
from .template import (
    FACES,
    ClassTarget,
    DeadEndTarget,
    Endpoint,
    LoopTarget,
    NetworkTemplate,
    OrderTarget,
    Provenance,
    Samples,
)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    known = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(known)) if known else None


def _weighted(values: Sequence[Optional[float]], weights: Sequence[float]) -> Optional[float]:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and w > 0 and math.isfinite(float(v))]
    if not pairs:
        return _mean(values)
    total = sum(w for _v, w in pairs)
    return sum(v * w for v, w in pairs) / total


def _count(values: Sequence[float]) -> int:
    return int(round(float(np.mean(values)))) if len(values) else 0


def _majority(values: Sequence[Any]) -> Any:
    known = [v for v in values if v is not None]
    if not known:
        return None
    counts = Counter(known)
    best = max(counts.values())
    return next(v for v in known if counts[v] == best)


def _pooled(samples: Sequence[Optional[Samples]]) -> Optional[Samples]:
    if not samples or any(s is None for s in samples):
        return None
    return Samples(
        lengths_um=tuple(v for s in samples for v in s.lengths_um),
        diameters_um=tuple(v for s in samples for v in s.diameters_um),
        tortuosities=tuple(v for s in samples for v in s.tortuosities),
    )


def common_extent(members: Sequence[NetworkTemplate]) -> tuple[float, float, float]:
    """Each axis's mean extent (B1a)."""
    return tuple(float(np.mean([m.extent_um[axis] for m in members])) for axis in range(3))


def common_volume_factors(members: Sequence[NetworkTemplate]) -> list[float]:
    """For each member, the common box's volume over its own: what its
    counts are multiplied by to be its density in the common box."""
    volume = float(np.prod(common_extent(members)))
    return [volume / float(np.prod(m.extent_um)) for m in members]


def common_box_members(members: Sequence[NetworkTemplate]) -> list[NetworkTemplate]:
    """Each member scaled to the common box's volume (``NetworkTemplate.scaled``)."""
    return [m.scaled(f ** (1.0 / 3.0)) for m, f in zip(members, common_volume_factors(members))]


def representative_positions(points: np.ndarray, k: int) -> list[tuple[float, float, float]]:
    """*k* positions standing for *points* (an (n, 3) array of shares of the
    box): the centres of *k* groups found by a few rounds of k-means, started
    from the points farthest apart. Deterministic."""
    points = np.asarray(points, dtype=float)
    if k <= 0 or not len(points):
        return []
    if len(points) <= k:
        return [tuple(float(v) for v in p) for p in points[:k]] + [
            tuple(float(v) for v in points[i % len(points)]) for i in range(k - len(points))
        ]
    centres = [points[int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]]
    while len(centres) < k:
        distance = np.min(np.linalg.norm(points[:, None, :] - np.asarray(centres)[None], axis=2), axis=1)
        centres.append(points[int(np.argmax(distance))])
    centres = np.asarray(centres)
    for _ in range(20):
        nearest = np.argmin(np.linalg.norm(points[:, None, :] - centres[None], axis=2), axis=1)
        moved = np.array([points[nearest == i].mean(axis=0) if np.any(nearest == i) else centres[i] for i in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    return [tuple(float(v) for v in c) for c in centres]


def _endpoints(groups: Sequence[Sequence[Endpoint]], factors: Sequence[float]) -> tuple:
    """B3a: the mean count (each member's at the common box's faces),
    rounded and at least one, shared over the faces by how often the members
    use them, at positions pooled from theirs."""
    wanted = [len(ends) * f ** (2.0 / 3.0) for ends, f in zip(groups, factors)]
    total = max(1, int(round(float(np.mean(wanted))))) if any(groups) else 0
    if total == 0:
        return ()
    faces = Counter(end.face for ends in groups for end in ends)
    order = sorted(faces, key=lambda face: (-faces[face], list(FACES).index(face) if face in FACES else len(FACES)))
    weights = np.array([faces[face] for face in order], dtype=float)
    exact = total * weights / weights.sum()
    per_face = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - per_face), kind="stable")[: total - int(per_face.sum())]:
        per_face[i] += 1
    out: list[Endpoint] = []
    for face, n in zip(order, per_face):
        if n <= 0:
            continue
        pooled = np.array([end.position_rel for ends in groups for end in ends if end.face == face], dtype=float)
        out.extend(Endpoint(face, position) for position in representative_positions(pooled, int(n)))
    return tuple(out)


def _classes(members: Sequence[NetworkTemplate], factors: Sequence[float]) -> dict:
    categories = list(dict.fromkeys(c for m in members for c in m.classes))
    out: dict = {}
    for category in categories:
        targets = [m.classes.get(category) for m in members]
        counts = [(t.count if t else 0) * f for t, f in zip(targets, factors)]
        count = _count(counts)
        # A class the members have none of stays, at none, as theirs does.
        present = [(t, c) for t, c in zip(targets, counts) if t is not None]
        weights = [c for _t, c in present]
        out[category] = ClassTarget(
            category=category,
            count=count,
            total_length_um=float(np.mean([(t.total_length_um if t else 0.0) * f for t, f in zip(targets, factors)])),
            mean_length_um=_weighted([t.mean_length_um for t, _c in present], weights),
            median_length_um=_mean([t.median_length_um for t, _c in present]),
            mean_diameter_um=_weighted([t.mean_diameter_um for t, _c in present], weights),
            median_diameter_um=_mean([t.median_diameter_um for t, _c in present]),
            mean_tortuosity=_weighted([t.mean_tortuosity for t, _c in present], weights),
            median_tortuosity=_mean([t.median_tortuosity for t, _c in present]),
            samples=_pooled([t.samples for t, _c in present]),
        )
    return out


def _orders(members: Sequence[NetworkTemplate], factors: Sequence[float]) -> tuple:
    labels: dict[str, OrderTarget] = {}
    for m in members:
        for o in m.orders:
            labels.setdefault(o.label, o)
    out = []
    for label, first in sorted(labels.items(), key=lambda item: (item[1].category, item[1].index)):
        found = [next((o for o in m.orders if o.label == label), None) for m in members]
        counts = [(o.edge_count if o else 0) * f for o, f in zip(found, factors)]
        count = _count(counts)
        if count <= 0:
            continue
        present = [(o, c) for o, c in zip(found, counts) if o is not None]
        weights = [c for _o, c in present]
        out.append(OrderTarget(
            label=label,
            category=first.category,
            index=first.index,
            edge_count=count,
            mean_length_um=_weighted([o.mean_length_um for o, _c in present], weights),
            mean_tortuosity=_weighted([o.mean_tortuosity for o, _c in present], weights),
            mean_emergence_angle_deg=_weighted([o.mean_emergence_angle_deg for o, _c in present], weights),
            mean_diameter_um=_weighted([o.mean_diameter_um for o, _c in present], weights),
            diameter_cv=_mean([o.diameter_cv for o, _c in present]),
            samples=_pooled([o.samples for o, _c in present]),
        ))
    return tuple(out)


def _loops(members: Sequence[NetworkTemplate], factors: Sequence[float]) -> LoopTarget:
    loops = [m.loops for m in members]
    keys = list(dict.fromkeys(k for lp in loops for k in lp.length_percentiles_um))
    counted = [lp.independent_loops * f for lp, f in zip(loops, factors) if lp.independent_loops is not None]
    return LoopTarget(
        independent_loops=_count(counted) if counted else None,
        independent_loops_per_mm3=_mean([lp.independent_loops_per_mm3 for lp in loops]),
        length_percentiles_um={k: _mean([lp.length_percentiles_um.get(k) for lp in loops]) for k in keys},
        vessels_on_no_loop_fraction=_mean([lp.vessels_on_no_loop_fraction for lp in loops]),
        loop_length_samples_um=tuple(v for lp in loops for v in lp.loop_length_samples_um),
    )


def _dead_ends(members: Sequence[NetworkTemplate], factors: Sequence[float]) -> DeadEndTarget:
    ends = [m.dead_ends for m in members]
    areas = [f ** (2.0 / 3.0) for f in factors]
    return DeadEndTarget(
        total=_count([d.total * f for d, f in zip(ends, factors)]),
        leaves_image=_count([d.leaves_image * a for d, a in zip(ends, areas)]),
        interior=_count([d.interior * f for d, f in zip(ends, factors)]),
        artefacts=_count([d.artefacts * f for d, f in zip(ends, factors)]),
        island_ends=_count([d.island_ends * f for d, f in zip(ends, factors)]),
        interior_per_mm=_mean([d.interior_per_mm for d in ends]),
        chain_length_median_um=_mean([d.chain_length_median_um for d in ends]),
        chain_length_p10_um=_mean([d.chain_length_p10_um for d in ends]),
        chain_length_p90_um=_mean([d.chain_length_p90_um for d in ends]),
        chain_length_samples_um=tuple(v for d in ends for v in d.chain_length_samples_um),
    )


def _axis_overlap(n_from: int, n_to: int) -> np.ndarray:
    """``(n_to, n_from)``: the share of each of *n_from* equal cells along an
    axis lying in each of *n_to* (each column sums to 1)."""
    source = np.linspace(0.0, 1.0, n_from + 1)
    target = np.linspace(0.0, 1.0, n_to + 1)
    low = np.maximum(target[:-1, None], source[None, :-1])
    high = np.minimum(target[1:, None], source[None, 1:])
    return np.clip(high - low, 0.0, None) * n_from


def _onto(organisation: Organisation, values: Sequence[Any], shape: tuple) -> np.ndarray:
    """*values* (one per cell of *organisation*'s grid, a number or a tuple)
    carried onto the cells of *shape*, each cell shared out by how much of it
    lies in each: cells are shares of the box, and totals are kept.
    ``(cells, k)``."""
    v = np.asarray(values, dtype=float).reshape(tuple(organisation.shape) + (-1,))
    wz, wy, wx = (_axis_overlap(a, b) for a, b in zip(organisation.shape, shape))
    return np.einsum("az,by,cx,zyxk->abck", wz, wy, wx, v).reshape(int(np.prod(shape)), -1)


def average_organisations(
    organisations: Sequence[Optional[Organisation]],
    extent_um: Sequence[float],
    factors: Optional[Sequence[float]] = None,
) -> Optional[Organisation]:
    """B2a: the members' layouts on the common box's grid. Per-cell counts
    and lengths are carried onto it by overlap and averaged as densities
    (each times its member's *factors*, the common volume over its own, as
    the counts are: B1a); directions weighted by the capillary length there;
    capillary vertices pooled; the rest averaged."""
    factors = [1.0] * len(organisations) if factors is None else list(factors)
    paired = [(o, f) for o, f in zip(organisations, factors) if o is not None]
    if not paired:
        return None
    members = [o for o, _f in paired]
    shape = grid_shape(extent_um)
    cells = int(np.prod(shape))

    def field(name: str) -> list[float]:
        per_member = [_onto(o, getattr(o, name), shape)[:, 0] * f for o, f in paired]
        return [float(v) for v in np.mean(per_member, axis=0)]

    lengths = [_onto(o, o.capillary_length_um, shape)[:, 0] for o in members]

    def cellwise(name: str, fallback: Sequence[float]) -> tuple:
        weighted = sum(
            _onto(o, np.asarray(getattr(o, name), dtype=float).reshape(len(o.capillary_length_um), -1)
                  * np.asarray(o.capillary_length_um, dtype=float)[:, None], shape)
            for o in members
        )
        weights = np.sum(lengths, axis=0)
        out = []
        for c in range(cells):
            row = weighted[c] / weights[c] if weights[c] > 0 else np.asarray(fallback, dtype=float)
            out.append(tuple(float(v) for v in row))
        return tuple(out)

    global_orientation = tuple(float(v) for v in np.mean([o.global_orientation for o in members], axis=0))
    chords = [o for o in members if o.chord_axis_share_by_cell is not None]
    chord_whole = (
        tuple(float(v) for v in np.mean([o.chord_axis_share for o in chords if o.chord_axis_share is not None], axis=0))
        if any(o.chord_axis_share is not None for o in chords) else None
    )
    positions = [p for o in members for p in (o.capillary_node_positions_rel or ())]
    leaves: dict[str, list[float]] = {}
    with_leaves = [o for o in members if o.tree_leaves_by_order is not None]
    for o in with_leaves:
        for label, _count_ in o.tree_leaves_by_order:
            leaves.setdefault(label, [])
    for label in leaves:
        leaves[label] = [float(dict(o.tree_leaves_by_order).get(label, 0)) for o in with_leaves]
    reach: dict[str, list[tuple[float, float]]] = {}
    for o in members:
        for side, median, far in o.tree_reach_um or ():
            reach.setdefault(side, []).append((median, far))
    runs = [o.long_runs for o in members if o.long_runs is not None]
    classes = [o.chord_by_class_um for o in members if o.chord_by_class_um is not None]
    return Organisation(
        shape=shape,
        capillary_nodes=tuple(field("capillary_nodes")),
        capillary_length_um=tuple(field("capillary_length_um")),
        arterial_length_um=tuple(field("arterial_length_um")),
        venous_length_um=tuple(field("venous_length_um")),
        orientation=cellwise("orientation", global_orientation),
        global_orientation=global_orientation,
        axis_share=tuple(float(v) for v in np.mean([o.axis_share for o in members], axis=0)),
        within_near=float(np.mean([o.within_near for o in members])),
        beyond_far=float(np.mean([o.beyond_far for o in members])),
        trunk_turn_deg=_mean([o.trunk_turn_deg for o in members]),
        capillary_curvature_deg=_mean([o.capillary_curvature_deg for o in members]),
        tree_curvature_deg=_mean([o.tree_curvature_deg for o in members]),
        capillary_node_positions_rel=tuple(tuple(float(v) for v in p) for p in positions) or None,
        capillary_node_spacing_um=_mean([o.capillary_node_spacing_um for o in members]),
        chord_axis_share_by_cell=(
            None if not chords or chord_whole is None else cellwise("chord_axis_share_by_cell", chord_whole)
        ),
        chord_axis_share=chord_whole,
        long_runs=tuple(int(round(float(np.mean([r[a] for r in runs])))) for a in range(3)) if runs else None,
        chord_by_class_um=(
            tuple(_mean([c[k] for c in classes]) for k in range(4)) if classes else None
        ),
        tree_leaves_by_order=(
            tuple(sorted((label, int(round(float(np.mean(v))))) for label, v in leaves.items() if round(float(np.mean(v))) > 0))
            if with_leaves else None
        ),
        trunk_straightness=_mean([o.trunk_straightness for o in members]),
        tree_reach_um=(
            tuple((side, float(np.mean([m for m, _f in v])), float(np.mean([f for _m, f in v]))) for side, v in sorted(reach.items()))
            or None
        ),
    )


def average_templates(
    members: Sequence[NetworkTemplate],
    *,
    template_path: Optional[str] = None,
    source: Optional[str] = None,
    condition: Optional[str] = None,
    member_names: Sequence[str] = (),
) -> NetworkTemplate:
    """One template standing for *members* (see the module). A single
    member comes back as it is, its provenance pointing at the batch."""
    if not members:
        raise ValueError("no templates to average")
    factors = common_volume_factors(members)
    extent = common_extent(members)
    voxel = tuple(float(np.mean([m.voxel_size_zyx[a] for m in members])) for a in range(3))
    shape = tuple(int(round(e / v)) + 1 for e, v in zip(extent, voxel))
    names = list(member_names) or [str(m.provenance.template_path) for m in members]
    notes = [
        f"the average of {len(members)} run{'s' if len(members) != 1 else ''}"
        + (f" of condition '{condition}'" if condition else "")
        + f": {', '.join(names)}"
    ]
    for label, values in (
        ("diameter basis", [m.diameter_basis for m in members]),
        ("scheme", [m.scheme for m in members]),
    ):
        if len({v for v in values if v is not None}) > 1:
            notes.append(f"the runs disagree on their {label} ({', '.join(str(v) for v in values)}); the most common taken")
    iqr = [m.murray_iqr for m in members if m.murray_iqr]
    return NetworkTemplate(
        extent_um=extent,
        voxel_size_zyx=voxel,
        image_shape=shape,
        inlets=_endpoints([m.inlets for m in members], factors),
        outlets=_endpoints([m.outlets for m in members], factors),
        arteriole_boundary_count=_count([m.arteriole_boundary_count * f for m, f in zip(members, factors)]),
        venule_boundary_count=_count([m.venule_boundary_count * f for m, f in zip(members, factors)]),
        large_arteriole_boundary_count=_count([m.large_arteriole_boundary_count * f for m, f in zip(members, factors)]),
        large_venule_boundary_count=_count([m.large_venule_boundary_count * f for m, f in zip(members, factors)]),
        scheme=_majority([m.scheme for m in members]),
        classes=_classes(members, factors),
        orders=_orders(members, factors),
        total_length_um=float(np.mean([m.total_length_um * f for m, f in zip(members, factors)])),
        node_count=_none_or_count([m.node_count for m in members], factors),
        edge_count=_none_or_count([m.edge_count for m in members], factors),
        average_degree=_mean([m.average_degree for m in members]),
        branching_points=_none_or_count([m.branching_points for m in members], factors),
        mean_tortuosity=_mean([m.mean_tortuosity for m in members]),
        daughter_angle_deg=_mean([m.daughter_angle_deg for m in members]),
        daughter_angle_sd_deg=_mean([m.daughter_angle_sd_deg for m in members]),
        murray_exponent=_mean([m.murray_exponent for m in members]),
        murray_iqr=(float(np.mean([i[0] for i in iqr])), float(np.mean([i[1] for i in iqr]))) if iqr else None,
        intercapillary_distance_um=_mean([m.intercapillary_distance_um for m in members]),
        intercapillary_distance_median_um=_mean([m.intercapillary_distance_median_um for m in members]),
        loops=_loops(members, factors),
        dead_ends=_dead_ends(members, factors),
        tissue_volume_um3=_mean([
            None if m.tissue_volume_um3 is None else m.tissue_volume_um3 * f for m, f in zip(members, factors)
        ]),
        provenance=Provenance(
            template_path=template_path,
            source=source or members[0].provenance.source,
            haemolynx_version=_majority([m.provenance.haemolynx_version for m in members]),
            statistics_mode=_majority([m.provenance.statistics_mode for m in members]),
            run_version=None,
            run_commit=None,
            notes=tuple(notes),
        ),
        diameter_basis=_majority([m.diameter_basis for m in members]),
        organisation=average_organisations([m.organisation for m in members], extent, factors),
    )


def _none_or_count(values: Sequence[Optional[int]], factors: Sequence[float]) -> Optional[int]:
    known = [v * f for v, f in zip(values, factors) if v is not None]
    return _count(known) if known else None


__all__ = [
    "average_organisations",
    "average_templates",
    "common_box_members",
    "common_extent",
    "common_volume_factors",
    "representative_positions",
]
