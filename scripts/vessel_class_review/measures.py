"""Per-vessel measures for the arteriole/venule morphology grids (Phase 0a of
PLAN_vessel_classes_from_segmentation.md).

Nothing here reads a branch order or a mask: a junction's parent is its widest
vessel (as Smith et al. 2015's algorithms take it), so the measures carry no
trace of the large masks the truth comes from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

#: Smaller than this is a capillary (Pries-Roy's cut-off, rat mesentery,
#: quoted by Smith et al. 2015). Every measure below is of vessels above it.
CAPILLARY_UNDER_UM = 8.4
#: The centreline is resampled this often for curvature and for the pair finder.
STEP_UM = 2.0
#: A pair: another vessel running within this angle of it...
PAIR_MAX_ANGLE_DEG = 30.0
#: ...within this many of its own diameters, centre to centre...
PAIR_REACH_DIAMETERS = 3.0
#: ...along at least this share of its length.
PAIR_MIN_SHARE = 0.5
#: Emergence angles are read this far along each vessel from the junction.
TANGENT_UM = 10.0
BRIDGE = "is_zero_resistance"


def polyline(G: nx.MultiGraph, u, v, data) -> np.ndarray:
    """The edge's centreline from u to v, microns (z, y, x)."""
    pts = np.asarray(data.get("voxels", []), float).reshape(-1, 3)
    pu, pv = np.asarray(G.nodes[u]["pos"], float), np.asarray(G.nodes[v]["pos"], float)
    if len(pts) == 0:
        return np.vstack([pu, pv])
    if np.linalg.norm(pts[0] - pu) > np.linalg.norm(pts[-1] - pu):
        pts = pts[::-1]
    return np.vstack([pu, pts, pv])


def resample(points: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return points[:1]
    n = max(2, int(round(s[-1] / step)) + 1)
    at = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(at, s, points[:, k]) for k in range(3)])


def curvature_sd(points: np.ndarray, step: float = STEP_UM) -> float:
    """Lorthois et al. 2014's Ksd: the standard deviation of curvature along
    the centreline (1/um), the turning angle per unit length at each interior
    sample of an evenly resampled curve."""
    r = resample(points, step)
    if len(r) < 4:
        return float("nan")
    d = np.diff(r, axis=0)
    lengths = np.linalg.norm(d, axis=1)
    t = d / np.maximum(lengths[:, None], 1e-12)
    cos = np.clip(np.sum(t[:-1] * t[1:], axis=1), -1.0, 1.0)
    kappa = np.arccos(cos) / np.maximum(0.5 * (lengths[:-1] + lengths[1:]), 1e-12)
    return float(np.std(kappa))


def tangent_from(points: np.ndarray, reach: float = TANGENT_UM) -> np.ndarray:
    """The unit direction leaving points[0], read *reach* microns along."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = min(reach, s[-1])
    i = int(np.searchsorted(s, target))
    far = points[min(i, len(points) - 1)]
    vec = far - points[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros(3)


def murray_exponent(parent_d: float, daughters: list[float]) -> float:
    from haemolynx.statistics.bifurcation import fitted_murray_exponent

    value = fitted_murray_exponent(parent_d, daughters)
    return float("nan") if value is None else float(value)


@dataclass
class Vessel:
    key: tuple
    u: Any
    v: Any
    diameter_um: float
    diameter_source: str
    length_um: float
    points: np.ndarray
    truth: str | None = None
    tree: int = -1
    measures: dict = field(default_factory=dict)


def vessels_above(G: nx.MultiGraph, threshold: float = CAPILLARY_UNDER_UM) -> list[Vessel]:
    out = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get(BRIDGE) or u == v:
            continue
        d = data.get("diameter_um")
        if d is None or not np.isfinite(d) or float(d) < threshold:
            continue
        pts = polyline(G, u, v, data)
        length = float(data.get("length") or np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        out.append(Vessel((u, v, k), u, v, float(d), str(data.get("diameter_source")), length, pts,
                          truth=data.get("large_mask_vessel_type")))
    return out


def label_trees(G: nx.MultiGraph, vessels: list[Vessel], threshold: float = CAPILLARY_UNDER_UM) -> int:
    """Each vessel's tree: the pieces of the network left joined by vessels at
    or above *threshold* and the zero-resistance bridges between them."""
    H = nx.Graph()
    for u, v, _k, data in G.edges(keys=True, data=True):
        d = data.get("diameter_um")
        if data.get(BRIDGE) or (d is not None and np.isfinite(d) and float(d) >= threshold):
            H.add_edge(u, v)
    comp = {}
    for i, nodes in enumerate(nx.connected_components(H)):
        for n in nodes:
            comp[n] = i
    for vessel in vessels:
        vessel.tree = comp.get(vessel.u, -1)
    return len(set(comp.values()))


def pair_partners(vessels: list[Vessel]) -> None:
    """For each vessel, the vessel running beside it (not sharing a node with
    it) and the ratio of their widths, or none."""
    samples, owner, tangents = [], [], []
    for i, vessel in enumerate(vessels):
        r = resample(vessel.points, STEP_UM)
        t = np.gradient(r, axis=0) if len(r) > 1 else np.zeros_like(r)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        samples.append(r)
        tangents.append(t)
        owner.extend([i] * len(r))
    all_pts = np.vstack(samples)
    all_tan = np.vstack(tangents)
    owner = np.asarray(owner)
    tree = cKDTree(all_pts)
    cos_max = math.cos(math.radians(PAIR_MAX_ANGLE_DEG))
    for i, vessel in enumerate(vessels):
        reach = PAIR_REACH_DIAMETERS * vessel.diameter_um
        ends = {vessel.u, vessel.v}
        counts: dict[int, int] = {}
        r, t = samples[i], tangents[i]
        for p, tp in zip(r, t):
            best_j, best_dist = None, None
            for idx in tree.query_ball_point(p, reach):
                j = owner[idx]
                if j == i or ends & {vessels[j].u, vessels[j].v}:
                    continue
                if abs(float(np.dot(tp, all_tan[idx]))) < cos_max:
                    continue
                dist = float(np.linalg.norm(all_pts[idx] - p))
                if best_dist is None or dist < best_dist:
                    best_j, best_dist = j, dist
            if best_j is not None:
                counts[best_j] = counts.get(best_j, 0) + 1
        vessel.measures["pair_partner"] = None
        vessel.measures["pair_share"] = 0.0
        vessel.measures["pair_width_ratio"] = float("nan")
        if counts:
            j, n = max(counts.items(), key=lambda kv: kv[1])
            share = n / len(r)
            vessel.measures["pair_share"] = share
            if share >= PAIR_MIN_SHARE:
                vessel.measures["pair_partner"] = j
                vessel.measures["pair_width_ratio"] = vessel.diameter_um / vessels[j].diameter_um


def junction_measures(G: nx.MultiGraph, vessels: list[Vessel]) -> None:
    """At each end of each vessel that is a junction: whether it is the
    junction's widest vessel (its parent); its emergence angle where it is a
    daughter; the Murray exponent where it is the parent."""
    index = {v.key: v for v in vessels}
    for vessel in vessels:
        vessel.measures.update(emergence_deg=float("nan"), murray_exponent=float("nan"),
                               parent_ends=0, daughter_ends=0)
    angles: dict[tuple, list[float]] = {}
    exps: dict[tuple, list[float]] = {}
    for node in G.nodes:
        incident = [(u, v, k, d) for u, v, k, d in G.edges(node, keys=True, data=True) if u != v]
        if len(incident) < 3:
            continue
        widths = []
        for u, v, k, d in incident:
            w = d.get("diameter_um")
            widths.append(float(w) if w is not None and np.isfinite(w) else 0.0)
        order = np.argsort(widths)[::-1]
        if len(order) > 1 and widths[order[0]] == widths[order[1]]:
            continue
        pu, pv, pk, pd = incident[order[0]]
        parent_key = (pu, pv, pk) if (pu, pv, pk) in index else (pv, pu, pk)
        parent_pts = polyline(G, pu, pv, pd)
        if pu != node:
            parent_pts = parent_pts[::-1]
        through = -tangent_from(parent_pts)
        daughters = []
        for i in order[1:]:
            u, v, k, d = incident[i]
            if d.get(BRIDGE):
                continue
            daughters.append(widths[i])
            key = (u, v, k) if (u, v, k) in index else (v, u, k)
            if key in index:
                pts = polyline(G, u, v, d)
                if u != node:
                    pts = pts[::-1]
                cos = float(np.clip(np.dot(through, tangent_from(pts)), -1.0, 1.0))
                angles.setdefault(key, []).append(math.degrees(math.acos(cos)))
                index[key].measures["daughter_ends"] += 1
        if parent_key in index and daughters and not pd.get(BRIDGE):
            index[parent_key].measures["parent_ends"] += 1
            exps.setdefault(parent_key, []).append(murray_exponent(widths[order[0]], daughters))
    for key, values in angles.items():
        index[key].measures["emergence_deg"] = float(np.mean(values))
    for key, values in exps.items():
        finite = [x for x in values if np.isfinite(x)]
        if finite:
            index[key].measures["murray_exponent"] = float(np.median(finite))


def shape_measures(vessels: list[Vessel]) -> None:
    for vessel in vessels:
        chord = float(np.linalg.norm(vessel.points[-1] - vessel.points[0]))
        axis = vessel.points[-1] - vessel.points[0]
        vessel.measures.update(
            length_over_diameter=vessel.length_um / vessel.diameter_um,
            ksd_per_um=curvature_sd(vessel.points),
            distance_factor=vessel.length_um / chord if chord > 0 else float("nan"),
            depth_um=float(np.mean(vessel.points[:, 0])),
            angle_to_z_deg=(float(math.degrees(math.acos(min(1.0, abs(axis[0]) / chord))))
                            if chord > 0 else float("nan")),
        )


def measure(G: nx.MultiGraph, threshold: float = CAPILLARY_UNDER_UM) -> list[Vessel]:
    vessels = vessels_above(G, threshold)
    label_trees(G, vessels, threshold)
    shape_measures(vessels)
    junction_measures(G, vessels)
    pair_partners(vessels)
    return vessels
