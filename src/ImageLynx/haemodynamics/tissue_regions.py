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
