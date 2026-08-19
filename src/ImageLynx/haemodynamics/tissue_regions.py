"""Joining a segmented tissue mask to the perfusion grid.

H2 §2.3 asks for the segmented TH volume to assign distinct metabolic rates, "a higher rate
for TH-positive voxels, and a lower rate for the surrounding stroma", and then for the hypoxic
fraction strictly within the TH-positive volume. The ADR solver takes a scalar ``M_max``, so
the mask has to become a per-cell array before any of that is possible. This module is that
step, and it is the primitive all four H2 methods need: each of them uses the glomus mask as a
spatial landmark against a quantity solved on the graph or the grid.

**Volume fraction, not a centre sample.** The grid is coarse relative to the segmentation. At
the 10 µm resolution currently used against 1.866 µm voxels there are roughly 154 mask voxels
to a cell, so a cell is rarely wholly tissue or wholly stroma. Sampling the mask at the cell
centre would discard almost all of the mask and would make the answer depend on where the cell
centres happened to fall. The measured tissue-to-vessel distance is 5.3 to 7.9 µm, below one
cell width, which is the same reason S19 gives for the grid needing refinement: at this
resolution the mask carries more spatial information than the grid can hold, and throwing away
the sub-cell part of it is the avoidable half of that loss.
"""
from __future__ import annotations

from typing import Sequence

import warnings

import numpy as np


def mask_fraction_per_cell(
    mask: np.ndarray,
    grid,
    voxel_um: Sequence[float],
    *,
    origin_um: Sequence[float] | None = None,
) -> np.ndarray:
    """Fraction of each grid cell occupied by ``mask``, as a flat per-cell array.

    ``mask`` is a boolean volume in (z, y, x) at ``voxel_um`` spacing. ``origin_um`` is the
    physical position of its first voxel corner, defaulting to the origin, which is correct
    when the mask and the graph were both cropped from the same region.

    Mask voxels falling outside the grid are dropped. They must not be clipped or wrapped: a
    wrapped index would deposit distal tissue into cell 0 and a clipped one would pile it onto
    the boundary cells, and in both cases the error is invisible in the output.

    Dropping is the right behaviour and is still worth hearing about, so more than 1% lost
    warns. The grid is built from the graph's node bounding box, so a specimen whose vessels
    stop short of the region edge gets a grid smaller than the mask, and the tissue in the gap
    leaves the analysis without changing anything that looks wrong: the returned fractions are
    all valid, and simply describe less tissue than was passed in.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be a 3D volume, got shape {mask.shape}")
    voxel = np.asarray(voxel_um, dtype=float)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError(f"voxel_um must be three positive values, got {voxel_um}")

    n_cells = int(grid.n_cells)
    if not mask.any():
        return np.zeros(n_cells, dtype=np.float64)

    origin = np.zeros(3) if origin_um is None else np.asarray(origin_um, dtype=float)
    idx = np.argwhere(mask)                       # (N, 3) voxel indices, zyx
    centres = (idx + 0.5) * voxel + origin        # voxel centres in physical zyx

    dims = np.asarray(grid.dims, dtype=int)       # (nz, ny, nx)
    rel = (centres - np.asarray(grid.min_xyz, dtype=float)) / np.asarray(grid.res, dtype=float)
    ijk = np.floor(rel).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
    dropped = int(inside.size - inside.sum())
    if dropped and dropped > 0.01 * inside.size:
        warnings.warn(
            f"{dropped} of {inside.size} mask voxels ({100.0 * dropped / inside.size:.2f}%) "
            f"fall outside the grid and are not represented in the returned fractions; the "
            f"grid spans {np.round(np.asarray(grid.min_xyz), 1)} to "
            f"{np.round(np.asarray(grid.min_xyz) + dims * np.asarray(grid.res), 1)} um",
            RuntimeWarning, stacklevel=2)
    ijk = ijk[inside]
    if ijk.size == 0:
        return np.zeros(n_cells, dtype=np.float64)

    # PerfusionGrid.get_cell_index is z-fastest: index = z + y*nz + x*nz*ny.
    nz, ny = int(dims[0]), int(dims[1])
    linear = ijk[:, 0] + ijk[:, 1] * nz + ijk[:, 2] * nz * ny
    counts = np.bincount(linear, minlength=n_cells).astype(np.float64)

    voxels_per_cell = float(np.prod(np.asarray(grid.res, dtype=float)) / np.prod(voxel))
    if voxels_per_cell <= 0:
        raise ValueError("grid resolution and voxel size give a non-positive cell occupancy")
    # Clipped at 1: a grid cell finer than one voxel can receive the same voxel centre from
    # no more than one cell, but rounding at the boundary can still push a ratio marginally
    # above unity, and a fraction above 1 would blend past the tissue rate in the caller.
    return np.clip(counts / voxels_per_cell, 0.0, 1.0)


def mask_bounds_um(
    mask_shape: Sequence[int],
    voxel_um: Sequence[float],
    *,
    origin_um: Sequence[float] | None = None,
) -> tuple:
    """Physical extent of a mask volume, as ``(min_zyx, max_zyx)`` in micrometres.

    Ready to hand to ``PerfusionGrid(..., bounds_zyx=...)``, which is the whole point: the
    conversion is two lines and getting it wrong is invisible, because a grid built from
    slightly wrong bounds still solves and still looks like a field.

    The extent is the outer corners of the volume, not the centres of the corner voxels, so it
    matches the convention ``mask_fraction_per_cell`` uses when it places voxel centres.
    """
    shape = np.asarray(mask_shape, dtype=float)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"mask_shape must be three positive lengths, got {mask_shape}")
    voxel = np.asarray(voxel_um, dtype=float)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError(f"voxel_um must be three positive values, got {voxel_um}")

    origin = np.zeros(3) if origin_um is None else np.asarray(origin_um, dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"origin_um must be a (z, y, x) triple, got {origin_um}")
    return origin, origin + shape * voxel


def blend_per_cell_rate(
    fraction: np.ndarray,
    *,
    tissue_rate: float,
    stroma_rate: float,
) -> np.ndarray:
    """Per-cell rate, linearly weighted by the tissue fraction in each cell.

    A cell that is 60% TH-positive consumes at 60% of the way from the stromal rate to the
    tissue rate. That is the volume-weighted average of the two, which is what a mixed cell
    physically contains, and it degenerates to the scalar case when the two rates are equal.

    Returned rather than applied, so the caller decides whether to pass it as ``M_max``. The
    ADR solver uses ``M_max`` elementwise against the PO2 vector, so an array of length
    ``n_cells`` broadcasts there unchanged.
    """
    fraction = np.asarray(fraction, dtype=float)
    if fraction.size and (fraction.min() < 0.0 or fraction.max() > 1.0):
        raise ValueError(
            f"fraction must lie in [0, 1], got [{fraction.min():.4g}, {fraction.max():.4g}]"
        )
    return stroma_rate + (tissue_rate - stroma_rate) * fraction


def edge_tissue_fraction(
    G,
    mask: np.ndarray,
    voxel_um: Sequence[float],
    *,
    origin_um: Sequence[float] | None = None,
    step_um: float | None = None,
) -> dict:
    """Fraction of each edge's centreline length that lies inside ``mask``.

    The graph-side counterpart of :func:`mask_fraction_per_cell`, and what H2 §2.1 and §2.2
    need: §2.1 asks where flow goes relative to the glomus clusters, §2.2 asks which edges
    supply them, and both are the same question about edges rather than grid cells.

    Sampled along the whole centreline, not at the endpoints. A capillary penetrating a glomus
    cluster typically begins and ends in stroma, so an endpoint test would classify exactly the
    vessels §2.1 is about as extra-glomus. Weighted by length rather than by point count,
    because the stored polylines are not uniformly spaced and a densely sampled stretch would
    otherwise outvote a long one.

    Returns ``{(u, v, key): fraction}``. Centreline points outside the mask array count as
    outside rather than being clipped to its border.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be a 3D volume, got shape {mask.shape}")
    voxel = np.asarray(voxel_um, dtype=float)
    origin = np.zeros(3) if origin_um is None else np.asarray(origin_um, dtype=float)
    # Half the finest voxel, so a crossing cannot be stepped over.
    step = float(step_um) if step_um else float(voxel.min()) * 0.5

    def inside(points: np.ndarray) -> np.ndarray:
        idx = np.floor((points - origin) / voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
        out = np.zeros(len(points), dtype=bool)
        if ok.any():
            sel = idx[ok]
            out[ok] = mask[sel[:, 0], sel[:, 1], sel[:, 2]]
        return out

    result: dict = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        pts = data.get("voxels")
        if pts is None or len(pts) < 2:
            pos = G.nodes[u].get("pos"), G.nodes[v].get("pos")
            if pos[0] is None or pos[1] is None:
                result[(u, v, key)] = float("nan")
                continue
            pts = [pos[0], pos[1]]
        poly = np.asarray(pts, dtype=float)

        inside_length = 0.0
        total_length = 0.0
        for a, b in zip(poly[:-1], poly[1:]):
            seg = float(np.linalg.norm(b - a))
            if seg <= 0:
                continue
            n = max(1, int(np.ceil(seg / step)))
            # Midpoints of n equal sub-steps: each carries the same length, so the average
            # over them is a length-weighted average along this segment.
            t = (np.arange(n) + 0.5) / n
            samples = a + np.outer(t, b - a)
            inside_length += float(inside(samples).mean()) * seg
            total_length += seg

        result[(u, v, key)] = (inside_length / total_length) if total_length > 0 else float("nan")
    return result
