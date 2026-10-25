"""Where a perturbation's mask objects lie: along the vessels, and in depth.

A pericyte mask (the pericyte perturbations' sites, with
``use_pericyte_mask_constriction``) and a block mask (``capillary_block_mask``)
are both masks of objects -- one connected component each -- that the
perturbation places on vessels. This says, for either, how densely its
objects sit on the network and where they lie in the image's depth:

- **density by branch order**: the objects placed on each order's vessels per
  mm of that order's vessel length. Each object counts once, on the vessel the
  perturbation placed it on: a pericyte on the capillary it was projected
  onto (only pericytes near enough to it and of a pericyte's size are placed,
  and only on capillaries, so only the capillary orders are listed); a block
  on the vessel holding most of its voxels inside the segmentation, also when
  it blocks every vessel it lies in.
- **density by capillary length**: the objects placed on capillaries per mm
  of the network's capillary length (``constriction.is_capillary_branch_order``:
  the B orders and ``Unknown_Cap``).
- **depth**: each object's centroid z, in microns from the image's first plane
  (the crop box's, when the image was cut on import), over every object of the
  mask and over those placed on vessels: least, greatest, range, mean and
  mode. The mode is the peak of a Gaussian kernel density of the depths with
  Silverman's bandwidth (:func:`kernel_density_mode`): a plane-by-plane count
  of a few dozen objects has no peak to speak of.

Thick-vessel bridges are no vessel: they hold no object and no length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph.dead_ends import image_extent_um
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

from .capillary_block_mask import MaskBlock
from .constriction import is_capillary_branch_order
from .pericyte_mask import PericytePlacement

__all__ = [
    "DEPTH_STATISTICS",
    "MaskDistribution",
    "block_mask_distribution",
    "kernel_density_mode",
    "mask_distribution",
    "pericyte_mask_distribution",
]

#: The depth statistics, each a ``depth_<statistic>_um_{all|on_vessels}`` row.
DEPTH_STATISTICS: tuple[str, ...] = ("min", "max", "range", "mean", "mode", "mode_bandwidth")

#: Micrometres per millimetre: densities are objects per mm of vessel.
_UM_PER_MM = 1000.0

#: The grid the kernel density's peak is looked for on, in bandwidths.
_MODE_GRID_STEP_BANDWIDTHS = 0.1

#: Objects summed per pass of the kernel density, to bound its memory.
_MODE_CHUNK = 10_000


@dataclass
class MaskDistribution:
    """One mask's objects on one network: ``summary`` the metric/value rows
    (counts, the capillary density, the image's depth and the objects' depth
    statistics), ``by_branch_order`` one row per branch order."""

    summary: dict[str, Any] = field(default_factory=dict)
    by_branch_order: list[dict[str, Any]] = field(default_factory=list)


def kernel_density_mode(values: Sequence[float]) -> tuple[float, float]:
    """The peak of a Gaussian kernel density estimate of *values*, and the
    bandwidth it used: ``(mode, bandwidth)``.

    The bandwidth is Silverman's rule of thumb, ``0.9 min(SD, IQR / 1.34)
    n^(-1/5)`` (the SD alone when the IQR is 0). The peak is looked for on a
    grid a tenth of a bandwidth apart between the least and the greatest
    value, where every peak of such a density lies; a tie goes to the least.
    One value, or values all alike, are their own mode with a bandwidth of 0;
    no finite value gives ``(nan, nan)``.
    """
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return math.nan, math.nan
    low, high = float(data.min()), float(data.max())
    if data.size == 1 or high == low:
        return low, 0.0
    sd = float(np.std(data, ddof=1))
    q75, q25 = np.percentile(data, [75.0, 25.0])
    iqr = float(q75 - q25)
    spread = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    bandwidth = 0.9 * spread * data.size ** (-0.2)
    step = _MODE_GRID_STEP_BANDWIDTHS * bandwidth
    grid = np.linspace(low, high, int(math.ceil((high - low) / step)) + 1)
    density = np.zeros(len(grid))
    for start in range(0, data.size, _MODE_CHUNK):
        chunk = data[start:start + _MODE_CHUNK]
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bandwidth) ** 2).sum(axis=1)
    return float(grid[int(np.argmax(density))]), float(bandwidth)


def _depth_rows(depths_um: np.ndarray, group: str) -> dict[str, float]:
    """The depth statistics of *depths_um*, keyed ``depth_<statistic>_um_<group>``;
    NaN where there is no depth."""
    depths = depths_um[np.isfinite(depths_um)]
    if depths.size:
        low, high = float(depths.min()), float(depths.max())
        mode, bandwidth = kernel_density_mode(depths)
        values = (low, high, high - low, float(depths.mean()), mode, bandwidth)
    else:
        values = (math.nan,) * len(DEPTH_STATISTICS)
    return {f"depth_{name}_um_{group}": value for name, value in zip(DEPTH_STATISTICS, values)}


def _per_mm(objects: int, length_um: float) -> float:
    return objects / length_um * _UM_PER_MM if length_um > 0.0 else math.nan


def mask_distribution(
    G: nx.MultiGraph,
    placed_on: Sequence[Optional[tuple[Any, Any, Any]]],
    depths_um: Sequence[float],
    *,
    image_depth_um: float,
    capillaries_only: bool = False,
    mask: str = "",
) -> MaskDistribution:
    """How densely a mask's objects sit on *G* and where they lie in depth.

    *placed_on* and *depths_um* hold one entry per object of the mask: the
    vessel ``(u, v, key)`` it counts on (None: on no vessel) and its centroid
    z in microns. *image_depth_um* is the image's z extent (the first plane's
    centre to the last's). *capillaries_only* lists only the capillary orders
    by branch order, for objects that can only be placed on capillaries.
    *mask* names the mask in the summary.
    """
    from haemolynx.statistics.bifurcation import _branch_order_sort_key, branch_order_label

    if len(placed_on) != len(depths_um):
        raise ValueError(
            f"{len(placed_on)} placements for {len(depths_um)} depths: one each per object"
        )
    rows: dict[str, dict[str, Any]] = {}
    label_of: dict[tuple[Any, Any, Any], str] = {}
    capillary: set[tuple[Any, Any, Any]] = set()
    capillary_length = 0.0
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get(IS_ZERO_RESISTANCE):
            continue
        order = data.get("branch_order")
        length = float(data.get("length", 0.0) or 0.0)
        if is_capillary_branch_order(order):
            capillary.add((u, v, key))
            capillary_length += length
        elif capillaries_only:
            continue
        label = branch_order_label(order)
        label_of[(u, v, key)] = label
        row = rows.setdefault(label, {"vessels": 0, "length_um": 0.0, "objects": 0})
        row["vessels"] += 1
        row["length_um"] += length

    on_vessels = [edge for edge in placed_on if edge is not None]
    for edge in on_vessels:
        if edge in label_of:
            rows[label_of[edge]]["objects"] += 1
    on_capillaries = sum(1 for edge in on_vessels if edge in capillary)

    depths = np.asarray(depths_um, dtype=float).reshape(-1)
    placed = np.array([edge is not None for edge in placed_on], dtype=bool)
    summary: dict[str, Any] = {
        "mask": mask,
        "objects": len(placed_on),
        "objects_on_vessels": len(on_vessels),
        "objects_on_capillaries": on_capillaries,
        "capillary_vessels": len(capillary),
        "capillary_length_um": capillary_length,
        "objects_per_mm_capillary": _per_mm(on_capillaries, capillary_length),
        "image_depth_um": float(image_depth_um),
        **_depth_rows(depths, "all"),
        **_depth_rows(depths[placed], "on_vessels"),
    }
    by_order = [
        {
            "branch_order": label,
            "vessels": rows[label]["vessels"],
            "length_um": rows[label]["length_um"],
            "objects": rows[label]["objects"],
            "objects_per_mm": _per_mm(rows[label]["objects"], rows[label]["length_um"]),
        }
        for label in sorted(rows, key=_branch_order_sort_key)
    ]
    return MaskDistribution(summary=summary, by_branch_order=by_order)


def pericyte_mask_distribution(
    G: nx.MultiGraph,
    placement: PericytePlacement,
    *,
    image_shape: Sequence[int],
    voxel_size_zyx: Sequence[float],
) -> MaskDistribution:
    """:func:`mask_distribution` of a pericyte mask placed on *G*
    (:func:`~haemolynx.haemodynamics.pericyte_mask.place_pericytes`): each
    placed pericyte on the capillary it was projected onto. *image_shape* and
    *voxel_size_zyx* are the mask's, on *G*'s grid."""
    eligible = set(placement.eligible)
    count = int(placement.centroids_um.shape[0])
    placed_on = [
        placement.projections[index][0] if index in eligible else None for index in range(count)
    ]
    depths = placement.centroids_um[:, 0] if count else np.empty(0)
    return mask_distribution(
        G,
        placed_on,
        depths,
        image_depth_um=float(image_extent_um(image_shape, voxel_size_zyx)[0]),
        capillaries_only=True,
        mask="pericyte_mask",
    )


def block_mask_distribution(
    G: nx.MultiGraph,
    blocks: Sequence[MaskBlock],
    *,
    image_shape: Sequence[int],
    voxel_size_zyx: Sequence[float],
) -> MaskDistribution:
    """:func:`mask_distribution` of a block mask's blobs on *G*
    (:func:`~haemolynx.haemodynamics.capillary_block_mask.blocks_from_mask`):
    each blob that blocks on its :attr:`~MaskBlock.main_branch_id` vessel."""
    edges = list(G.edges(keys=True))
    placed_on = [
        None if block.main_branch_id is None else edges[block.main_branch_id] for block in blocks
    ]
    return mask_distribution(
        G,
        placed_on,
        [block.centroid_um[0] for block in blocks],
        image_depth_um=float(image_extent_um(image_shape, voxel_size_zyx)[0]),
        mask="capillary_block_mask",
    )
