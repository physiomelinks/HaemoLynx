"""Capillary blocks read from a mask: the vessels a mask's blobs sit in.

The ``capillary_block_mask`` perturbation takes the vessels to block from a
mask of the blocks themselves -- blobs on an otherwise empty volume, such as a
segmentation of stalled leukocytes or platelet aggregates -- rather than
drawing them at random or listing them. Each blob (connected component) is
one block. It is on a vessel when at least ``min_overlap_fraction`` of its
voxels lie inside the run's segmented vessels; a blob mostly off them is
tissue, not a plug, and blocks nothing.

Which vessel a blob blocks is read from where its voxels inside the
segmentation lie: each goes to the vessel whose centreline is nearest
(thick-vessel bridges are no vessel), and only within that vessel's lumen
radius plus a voxel's diagonal -- segmentation the graph has no centreline
through belongs to no vessel. ``vessels_per_block`` then says what the blob
blocks: ``largest_overlap`` the one vessel holding most of those voxels (ties
to the first in edge order), ``every_overlapped_vessel`` every vessel holding
any.

The occlusion itself is :mod:`haemolynx.haemodynamics.capillary_block`'s:
the blocked vessels' resistance multiplied by its factor, then the re-solve,
the stopped vessels' flows set to 0, and the comparison with the baseline.

Without the run's segmentation (a synthetic network, or a caller passing
none) the graph's own lumens stand in for it: a voxel is inside a vessel
when it lies within its nearest centreline's radius plus half a voxel's
diagonal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from haemolynx.io import CANONICAL_AXIS_ORDER, voxel_size_zyx_from_xyz
from haemolynx.io.downsample import graph_downsample, mask_volume_like_graph
from haemolynx.io.load import _binarization_rule, load_binary_mask_voxel_size_and_status
from haemolynx.io.voxel_validation import resolve_voxel_size_xyz

__all__ = [
    "BLOCK_MASK_DESCRIPTION",
    "MASK_BLOCK_VESSEL_CHOICES",
    "MaskBlock",
    "blocks_from_mask",
    "load_capillary_block_mask",
]

#: Names the block mask in loader errors.
BLOCK_MASK_DESCRIPTION = "capillary block mask"

#: What a blob on a vessel blocks: the vessel holding most of it, or every
#: vessel it lies in.
MASK_BLOCK_VESSEL_CHOICES: tuple[str, ...] = ("largest_overlap", "every_overlapped_vessel")


@dataclass
class MaskBlock:
    """One blob of the block mask, and what it blocks.

    ``number`` is its component number (1-based, as ``scipy.ndimage.label``
    numbers them), ``centroid_um`` in physical ``(z, y, x)`` microns like node
    ``pos``. ``vessel_voxels`` maps each vessel's branchID (edge-order index)
    to how many of the blob's voxels inside the segmentation lie in it.
    ``reason`` says why a blob blocks nothing: ``below_overlap`` (too little
    of it inside the segmented vessels) or ``no_vessel`` (what is inside lies
    in no vessel of the graph).
    """

    number: int
    voxels: int
    volume_um3: float
    centroid_um: tuple[float, float, float]
    voxels_in_vessels: int
    overlap_fraction: float
    vessel_voxels: dict[int, int] = field(default_factory=dict)
    blocked_branch_ids: list[int] = field(default_factory=list)
    reason: str = ""

    @property
    def blocks(self) -> bool:
        return bool(self.blocked_branch_ids)

    @property
    def main_branch_id(self) -> int | None:
        """The vessel holding most of its voxels inside the segmentation
        (ties to the first in edge order): the one vessel it counts on in the
        mask's density, also when it blocks every vessel it lies in. None
        when it blocks nothing."""
        if not self.blocks:
            return None
        return max(sorted(self.vessel_voxels), key=lambda branch_id: self.vessel_voxels[branch_id])


def load_capillary_block_mask(
    path: str | Path,
    graph: nx.MultiGraph,
    *,
    h5_dataset_name: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The block mask as a boolean volume on *graph*'s grid (cut to its crop
    box, reduced by its downsample on import, block max), and its voxel size
    ``(z, y, x)``.

    The voxel size is resolved as the pericyte mask's is, and must match the
    main image's (``graph.graph["image_voxel_size_zyx"]`` when recorded):
    a mask exported without z spacing would place every block at the wrong
    depth, on the wrong vessel.
    """
    mask, metadata_voxel_size, metadata_status = load_binary_mask_voxel_size_and_status(
        path,
        h5_dataset_name=h5_dataset_name,
        axis_order=axis_order,
        description=BLOCK_MASK_DESCRIPTION,
    )
    mask = mask_volume_like_graph(np.asarray(mask, dtype=bool), graph, description=BLOCK_MASK_DESCRIPTION)
    voxel_size_xyz, _source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size,
        metadata_status=metadata_status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
    )
    downsample = graph_downsample(graph)
    if downsample is not None:
        voxel_size_xyz = downsample.scale_voxel_size_xyz(voxel_size_xyz)
    voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx_from_xyz(voxel_size_xyz))
    image_voxel_size_zyx = graph.graph.get("image_voxel_size_zyx")
    if image_voxel_size_zyx is not None and not np.allclose(
        voxel_size_zyx, np.asarray(image_voxel_size_zyx, dtype=float), rtol=1e-3, atol=0.0
    ):
        raise ValueError(
            f"The capillary block mask's voxel size (z, y, x) {voxel_size_zyx} does not match "
            f"the main image's {tuple(float(v) for v in image_voxel_size_zyx)}: its blocks "
            "would be placed against the wrong vessels. Export the mask with the image's "
            "voxel size, or give the run's voxel_size_override_xyz with voxel_size_policy "
            "'override'."
        )
    return mask, voxel_size_zyx


def _centreline_samples(
    G: nx.MultiGraph, step_um: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every vessel's centreline sampled at least every *step_um*:
    ``(points (n, 3), branchID of each, lumen radius of each)``. Bridges
    (zero-resistance) are left out: they are no vessel to plug."""
    from haemolynx.graph._helpers import densify_polyline
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    from haemolynx.visualization.geometry import edge_polyline

    points, owners, radii = [], [], []
    for branch_id, (u, v, _key, data) in enumerate(G.edges(keys=True, data=True)):
        if data.get(IS_ZERO_RESISTANCE):
            continue
        try:
            line = edge_polyline(G, u, v, data, orient=False, snap=False)
        except ValueError:
            continue
        line = densify_polyline(line, max_step_um=step_um)
        try:
            radius = 0.5 * float(data.get("diameter_um") or 0.0)
        except (TypeError, ValueError):
            radius = 0.0
        points.append(line)
        owners.append(np.full(len(line), branch_id, dtype=np.int64))
        radii.append(np.full(len(line), radius if np.isfinite(radius) else 0.0))
    if not points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0)
    return np.vstack(points), np.concatenate(owners), np.concatenate(radii)


def blocks_from_mask(
    G: nx.MultiGraph,
    block_mask: np.ndarray,
    voxel_size_zyx,
    *,
    vessel_mask: np.ndarray | None = None,
    min_overlap_fraction: float = 0.5,
    vessels_per_block: str = "largest_overlap",
) -> tuple[list[tuple[Any, Any, Any]], dict[str, Any], list[MaskBlock]]:
    """The vessels *block_mask*'s blobs block, as ``(u, v, key)`` in graph
    edge order; a summary of how they were found; and every blob as a
    :class:`MaskBlock`.

    *vessel_mask* is the run's segmentation on the same grid (any labelling
    ``io.load``'s binarisation reads: 0/1, 0/255, ilastik's 1/2), read only
    at the blobs' voxels; None reads the graph's lumens instead. *G*'s node
    and edge positions are physical microns, *block_mask*'s indices times
    *voxel_size_zyx*.
    """
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    if vessels_per_block not in MASK_BLOCK_VESSEL_CHOICES:
        raise ValueError(
            f"capillary_block_mask_vessels_per_block must be one of "
            f"{MASK_BLOCK_VESSEL_CHOICES}, got {vessels_per_block!r}"
        )
    if not 0.0 <= float(min_overlap_fraction) <= 1.0:
        raise ValueError(
            "capillary_block_mask_min_overlap_fraction must be between 0 and 1, "
            f"got {min_overlap_fraction!r}"
        )
    block_mask = np.asarray(block_mask, dtype=bool)
    if block_mask.ndim != 3:
        raise ValueError(f"The capillary block mask is not a 3D volume: shape {block_mask.shape}")
    if vessel_mask is not None and tuple(vessel_mask.shape) != tuple(block_mask.shape):
        raise ValueError(
            f"The capillary block mask's shape {tuple(block_mask.shape)} is not the "
            f"segmented image's {tuple(vessel_mask.shape)}: they must be the same grid"
        )
    spacing = np.asarray(voxel_size_zyx, dtype=float).reshape(3)
    edges = list(G.edges(keys=True))

    labels = ForegroundLabels(block_mask)
    count = int(labels.count)
    coords = labels.coords() if count > 0 else np.empty((0, 3), dtype=np.int64)
    numbers = np.asarray(labels.numbers, dtype=np.int64) if count > 0 else np.empty(0, dtype=np.int64)
    positions = coords * spacing

    samples, owners, radii = _centreline_samples(G, step_um=0.5 * float(spacing.min()))
    diagonal = float(np.linalg.norm(spacing))
    branch = np.full(len(coords), -1, dtype=np.int64)
    distance = np.full(len(coords), np.inf)
    if len(coords) and len(samples):
        distance, nearest = cKDTree(samples).query(positions)
        within = distance <= radii[nearest] + diagonal
        branch = np.where(within, owners[nearest], -1)
    if vessel_mask is None:
        inside = np.zeros(len(coords), dtype=bool)
        if len(coords) and len(samples):
            inside = distance <= radii[nearest] + 0.5 * diagonal
    else:
        rule = _binarization_rule(np.asanyarray(vessel_mask), use_memmap=isinstance(vessel_mask, np.memmap))
        values = np.asarray(vessel_mask[coords[:, 0], coords[:, 1], coords[:, 2]]) if len(coords) else np.empty(0)
        inside = np.asarray(rule(values), dtype=bool).reshape(-1)

    voxel_volume = float(np.prod(spacing))
    blocks: list[MaskBlock] = []
    blocked_ids: set[int] = set()
    order = np.argsort(numbers, kind="stable")
    bounds = np.searchsorted(numbers[order], np.arange(1, count + 2))
    for number in range(1, count + 1):
        members = order[bounds[number - 1]:bounds[number]]
        n = int(len(members))
        in_vessels = members[inside[members]]
        fraction = float(len(in_vessels)) / n if n else 0.0
        held = branch[in_vessels]
        held = held[held >= 0]
        ids, counts = np.unique(held, return_counts=True)
        block = MaskBlock(
            number=number,
            voxels=n,
            volume_um3=n * voxel_volume,
            centroid_um=tuple(float(c) for c in positions[members].mean(axis=0)),
            voxels_in_vessels=int(len(in_vessels)),
            overlap_fraction=fraction,
            vessel_voxels={int(i): int(c) for i, c in zip(ids, counts)},
        )
        if not len(in_vessels) or fraction < float(min_overlap_fraction):
            block.reason = "below_overlap"
        elif not len(ids):
            block.reason = "no_vessel"
        elif vessels_per_block == "largest_overlap":
            # np.unique sorts the ids, and argmax takes the first of a tie.
            block.blocked_branch_ids = [int(ids[int(np.argmax(counts))])]
        else:
            block.blocked_branch_ids = [int(i) for i in ids]
        blocked_ids.update(block.blocked_branch_ids)
        blocks.append(block)

    blocked = [edges[index] for index in sorted(blocked_ids)]
    on_vessels = [block for block in blocks if block.blocks]
    summary = {
        "selection": "mask",
        "mask_blocks": count,
        "mask_blocks_on_vessels": len(on_vessels),
        "mask_blocks_below_overlap": sum(block.reason == "below_overlap" for block in blocks),
        "mask_blocks_off_graph": sum(block.reason == "no_vessel" for block in blocks),
        "min_overlap_fraction": float(min_overlap_fraction),
        "vessels_per_block": vessels_per_block,
        "vessel_mask_source": "graph_lumens" if vessel_mask is None else "segmentation",
        "blocked_vessels": len(blocked),
    }
    return blocked, summary, blocks
