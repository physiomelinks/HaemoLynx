"""Does FWHM measure texture? A decoy check on the run's own image.

A line across a vessel on a speckled image can fit one bright speck
cleanly and pass every FWHM gate -- a speck in the vessel's own textured
lumen, or in tissue with no vessel at all. Nothing in a fit tells the two
apart. This asks the image directly: a sample of the edges FWHM measured are
copied, each moved sideways into tissue the segmentation says holds no
vessel (a decoy), and measured again with exactly the run's FWHM settings.

What it reports: the share of decoys FWHM gives a width to (its false-
positive rate on this image) and the range of widths it reads there -- the
speck scale. An FWHM width inside that range is one a speck could have
produced; every measured edge is flagged ``fwhm_in_speck_width_range`` when
there are enough decoy widths to set one. Nothing changes a diameter.

On a real embryonic brain stack, accepting one sample, FWHM measured 28% of
decoys, at 2.1-4.3 um (5-95%) with a median fit R^2 of 0.93 -- better than
on the vessels -- and 53% of the vessels it had measured read inside that
range; several read 2-3 um on a speck inside a band several times wider.
Requiring two samples (``fwhm_min_accepted_samples``, the default) it
measured 112 vessels instead of 324, and the check reported 16% of 91
decoys measured, at 2.3-3.7 um, with 20% of the 112 in that range.
"""
from __future__ import annotations

from typing import Any, Callable

import networkx as nx
import numpy as np

__all__ = [
    "DECOY_SAMPLE_SIZE",
    "decoy_centrelines",
    "decoy_probe_graph",
    "fwhm_decoy_check",
    "speck_width_report",
]

#: Edges the check copies at most: enough for a rate to a few percent, at a
#: fraction of the time the run's own FWHM takes.
DECOY_SAMPLE_SIZE = 100

#: How far sideways a decoy is tried (um), nearest first, and in how many
#: directions at each distance, before its edge is left without one.
DECOY_DISTANCES_UM = (8.0, 12.0, 16.0, 20.0)
_DIRECTIONS_PER_DISTANCE = 24

#: The nearest a decoy's centreline may come to a vessel voxel (um), and
#: more for a wide vessel: three quarters of its guide width, so its
#: transverse lines start in tissue.
DECOY_MIN_CLEARANCE_UM = 3.0

#: Decoy widths needed before their 5-95% range is called the speck range.
_MIN_DECOY_WIDTHS = 5


def _clear_of_vessels(points_um, vessel_mask, spacing, clearance_um) -> bool:
    """Whether every point is inside the volume and at least *clearance_um*
    from every vessel voxel. Reads a small box of the mask round each point
    only, so a memory-mapped mask stays on disk."""
    shape = np.asarray(vessel_mask.shape)
    reach = np.ceil(clearance_um / spacing).astype(int)
    for point in points_um:
        centre = np.rint(point / spacing).astype(int)
        if np.any(centre < 0) or np.any(centre >= shape):
            return False
        lo = np.maximum(centre - reach, 0)
        hi = np.minimum(centre + reach + 1, shape)
        box = np.asarray(vessel_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        if not box.any():
            continue
        voxels = (np.argwhere(box) + lo) * spacing
        if float(np.min(np.linalg.norm(voxels - point, axis=1))) < clearance_um:
            return False
    return True


def decoy_centrelines(
    G: Any,
    edges,
    vessel_mask: np.ndarray,
    voxel_size_zyx,
    *,
    rng: np.random.Generator,
    guide_attribute: str | None = "edt_diameter_um",
) -> list[tuple[tuple, np.ndarray, float]]:
    """``(edge, decoy centreline, guide width)`` for each of *edges* that one
    fits beside: its own centreline moved sideways -- perpendicular to its
    overall direction, nearest distance first -- to where it stays clear of
    every vessel voxel (see :data:`DECOY_MIN_CLEARANCE_UM`)."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    found = []
    for edge in edges:
        data = G.edges[edge]
        poly = np.asarray(data["voxels"], dtype=float)
        direction = poly[-1] - poly[0]
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            continue
        direction /= norm
        guide = data.get(guide_attribute) if guide_attribute else None
        guide = float(guide) if guide is not None and np.isfinite(float(guide)) and float(guide) > 0 else 4.0
        clearance = max(DECOY_MIN_CLEARANCE_UM, 0.75 * guide)
        placed = None
        for distance in DECOY_DISTANCES_UM:
            for _attempt in range(_DIRECTIONS_PER_DISTANCE):
                side = rng.normal(size=3)
                side -= (side @ direction) * direction
                if np.linalg.norm(side) < 1e-9:
                    continue
                side /= np.linalg.norm(side)
                moved = poly + distance * side
                if _clear_of_vessels(moved, vessel_mask, spacing, clearance):
                    placed = moved
                    break
            if placed is not None:
                break
        if placed is not None:
            found.append((edge, placed, guide))
    return found


def decoy_probe_graph(
    G: Any,
    edges,
    vessel_mask: np.ndarray,
    voxel_size_zyx,
    *,
    rng: np.random.Generator,
    guide_attribute: str | None = "edt_diameter_um",
    copy_attributes=(),
) -> nx.MultiGraph | None:
    """A graph of one edge per decoy of *edges* (see :func:`decoy_centrelines`),
    ready to measure, or ``None`` when none fits beside its vessel.

    Each decoy edge carries its guide width as *guide_attribute* and a copy of
    its own edge's *copy_attributes*, so a measurement guided by any of those
    reads it as it would the vessel.
    """
    decoys = decoy_centrelines(
        G, edges, vessel_mask, voxel_size_zyx, rng=rng, guide_attribute=guide_attribute
    )
    if not decoys:
        return None
    probe = nx.MultiGraph()
    probe.graph.update(G.graph)
    for index, (edge, line, guide) in enumerate(decoys):
        attrs = {"voxels": [tuple(p) for p in line],
                 "length": float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))}
        source = G.edges[edge]
        attrs.update({name: source[name] for name in copy_attributes if name in source})
        if guide_attribute:
            attrs[guide_attribute] = guide
        probe.add_edge(2 * index, 2 * index + 1, key=0, **attrs)
    return probe


def speck_width_report(G: Any, probe: nx.MultiGraph, measured_edges) -> dict[str, Any]:
    """What FWHM read on a measured *probe* of decoys, and which of *G*'s
    *measured_edges* have a width inside the speck range it sets.

    Writes ``fwhm_in_speck_width_range`` on each of *measured_edges* when there
    are enough decoy widths for a range (the 5-95% of them).
    """
    widths = np.array([
        float(data["fwhm_diameter_um"])
        for _u, _v, data in probe.edges(data=True)
        if _positive(data.get("fwhm_diameter_um"))
    ])
    decoys = probe.number_of_edges()
    report: dict[str, Any] = dict(
        decoys=decoys,
        decoys_measured=int(widths.size),
        false_positive_rate=float(widths.size / decoys) if decoys else 0.0,
    )
    measured_edges = list(measured_edges)
    if widths.size >= _MIN_DECOY_WIDTHS and measured_edges:
        low, high = (float(v) for v in np.percentile(widths, [5, 95]))
        report["speck_width_range_um"] = (low, high)
        report["speck_width_median_um"] = float(np.median(widths))
        inside = 0
        for edge in measured_edges:
            flag = low <= float(G.edges[edge]["fwhm_diameter_um"]) <= high
            G.edges[edge]["fwhm_in_speck_width_range"] = flag
            inside += flag
        report["measured_in_speck_width_range"] = float(inside / len(measured_edges))
    return report


def fwhm_decoy_check(
    G: Any,
    measure: Callable[[nx.MultiGraph], Any],
    *,
    vessel_mask: np.ndarray | None,
    voxel_size_zyx,
    sample_size: int = DECOY_SAMPLE_SIZE,
    seed: int = 0,
    guide_attribute: str | None = "edt_diameter_um",
) -> dict[str, Any]:
    """Measure decoys of up to *sample_size* FWHM-measured edges of *G* with
    *measure* (the run's own FWHM, on a graph) and report how often, and how
    wide, FWHM reads where there is no vessel.

    Writes the report to ``G.graph["fwhm_decoy_check"]`` and, when there
    are enough decoy widths for a speck range, ``fwhm_in_speck_width_range``
    on every FWHM-measured edge. Returns the report; it has ``skipped`` and
    a ``reason`` when there is no mask to find vessel-free tissue with, or
    nothing FWHM measured.
    """
    report: dict[str, Any] = {}
    for _u, _v, data in G.edges(data=True):
        data.pop("fwhm_in_speck_width_range", None)
    G.graph.pop("fwhm_decoy_check", None)
    if vessel_mask is None:
        report.update(skipped=True, reason="no segmentation to find vessel-free tissue in")
        return report
    measured = [
        (u, v, key)
        for u, v, key, data in G.edges(keys=True, data=True)
        if _positive(data.get("fwhm_diameter_um"))
    ]
    if not measured:
        report.update(skipped=True, reason="FWHM measured no edge")
        return report
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(measured))[: max(1, int(sample_size))]
    probe = decoy_probe_graph(
        G, [measured[i] for i in order], vessel_mask, voxel_size_zyx,
        rng=rng, guide_attribute=guide_attribute,
    )
    if probe is None:
        report.update(skipped=True, reason="no vessel-free tissue beside the measured edges")
        return report
    measure(probe)
    report.update(speck_width_report(G, probe, measured))
    G.graph["fwhm_decoy_check"] = dict(report)
    return report


def _positive(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)
