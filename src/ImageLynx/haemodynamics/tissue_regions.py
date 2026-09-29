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


def _origin_or_default(origin_um, voxel: np.ndarray) -> np.ndarray:
    """The physical position of a mask's first voxel *corner*, in micrometres.

    The default is half a voxel below zero. The graph puts voxel k's centre at k x voxel
    (node positions are skeleton indices times the voxel size, checked on WKY-A to 1e-14), so
    a mask cropped at the same ROI has its first corner at -voxel/2, not at 0. Until open item
    40 the default was 0, which shifted every mask lookup half a voxel (about 0.93 um) along
    each axis against the graph and the grid built from it.
    """
    if origin_um is None:
        return -0.5 * voxel
    origin = np.asarray(origin_um, dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"origin_um must be a (z, y, x) triple, got {origin_um}")
    return origin


def _axis_overlap_um(n_vox, voxel, origin, n_cells, res, lo):
    """Overlap length of every voxel with every cell along one axis, shape ``(n_cells, n_vox)``.

    Voxel ``i`` spans ``origin + [i, i+1) * voxel`` and cell ``j`` spans ``lo + [j, j+1) * res``.
    """
    v_lo = origin + np.arange(n_vox) * voxel
    c_lo = lo + np.arange(n_cells) * res
    overlap = (np.minimum(c_lo[:, None] + res, v_lo[None, :] + voxel)
               - np.maximum(c_lo[:, None], v_lo[None, :]))
    return np.clip(overlap, 0.0, None)


def mask_fraction_per_cell(
    mask: np.ndarray,
    grid,
    voxel_um: Sequence[float],
    *,
    origin_um: Sequence[float] | None = None,
) -> np.ndarray:
    """Fraction of each grid cell occupied by ``mask``, as a flat per-cell array.

    ``mask`` is a boolean volume in (z, y, x) at ``voxel_um`` spacing. ``origin_um`` is the
    physical position of its first voxel corner. The default, half a voxel below zero, is
    correct when the mask and the graph were both cropped from the same region, because the
    graph puts voxel centres at whole multiples of the voxel size (open item 40).

    **Exact overlap volume** (open item 38). Each mask voxel is a box, and it adds to every
    cell the volume it shares with that cell. Voxels and cells are both axis-aligned, so the
    overlap factorises into three 1D overlap-length matrices applied in turn. The fractions
    therefore sum to the mask volume inside the grid, and no cell can exceed 1 except by
    rounding. Until item 38 this counted voxel centres per cell and divided by the mean number
    of voxels per cell. On the 3 µm grid against 1.866 µm voxels a cell holds 1 to 8 centres
    against a mean of 4.16, so a solid TH cell read 0.24 to 1.92, and a clip at 1 silently
    threw away about 17% of the TH volume in every specimen.

    Mask volume falling outside the grid is dropped. It must not be clipped or wrapped: a
    wrapped index would deposit distal tissue into cell 0 and a clipped one would pile it onto
    the boundary cells, and in both cases the error is invisible in the output.

    Dropping is the right behaviour and is still worth hearing about, so more than 1% of the
    mask volume lost warns. The grid is built from the graph's node bounding box, so a specimen
    whose vessels stop short of the region edge gets a grid smaller than the mask, and the
    tissue in the gap leaves the analysis without changing anything that looks wrong: the
    returned fractions are all valid, and simply describe less tissue than was passed in.

    Raises if the fractions do not conserve the in-grid mask volume or exceed 1 by more than
    rounding; both would mean the overlap itself is wrong.
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

    origin = _origin_or_default(origin_um, voxel)
    dims = np.asarray(grid.dims, dtype=int)       # (nz, ny, nx)
    res = np.asarray(grid.res, dtype=float)
    lo = np.asarray(grid.min_xyz, dtype=float)    # zyx despite the name
    Wz, Wy, Wx = (_axis_overlap_um(mask.shape[a], voxel[a], origin[a], int(dims[a]), res[a], lo[a])
                  for a in range(3))

    # Occupied volume per cell, (nz, ny, nx), one axis at a time.
    m = mask.astype(np.float64)
    vol = np.einsum("iz,zyx->iyx", Wz, m)
    vol = np.einsum("jy,iyx->ijx", Wy, vol)
    vol = np.einsum("kx,ijx->ijk", Wx, vol)

    voxel_volume = float(np.prod(voxel))
    mask_volume = float(mask.sum()) * voxel_volume
    # In-grid share of each voxel is the product of its per-axis in-grid lengths.
    in_grid = np.einsum("z,y,x,zyx->", Wz.sum(axis=0), Wy.sum(axis=0), Wx.sum(axis=0), m)
    placed = float(vol.sum())
    if not np.isclose(placed, in_grid, rtol=1e-9, atol=1e-9 * voxel_volume):
        raise ValueError(
            f"per-cell mask volume {placed:.6g} um^3 does not conserve the in-grid mask volume "
            f"{in_grid:.6g} um^3")

    dropped = mask_volume - in_grid
    if dropped > 0.01 * mask_volume:
        warnings.warn(
            f"mask voxels holding {dropped:.4g} of {mask_volume:.4g} um^3 "
            f"({100.0 * dropped / mask_volume:.2f}%) fall outside the grid and are not "
            f"represented in the returned fractions; the grid spans {np.round(lo, 1)} to "
            f"{np.round(lo + dims * res, 1)} um",
            RuntimeWarning, stacklevel=2)

    fraction = vol / float(np.prod(res))
    if fraction.max() > 1.0 + 1e-9:
        raise ValueError(
            f"a cell is {fraction.max():.6g} occupied; the overlap cannot exceed the cell")
    fraction = np.minimum(fraction, 1.0)          # rounding only, checked above
    # PerfusionGrid.get_cell_index is z-fastest: index = z + y*nz + x*nz*ny.
    return fraction.ravel(order="F")


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
    matches the convention ``mask_fraction_per_cell`` uses when it places voxel centres. With
    no ``origin_um`` the first corner is half a voxel below zero, as there (open item 40).
    """
    shape = np.asarray(mask_shape, dtype=float)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(f"mask_shape must be three positive lengths, got {mask_shape}")
    voxel = np.asarray(voxel_um, dtype=float)
    if voxel.shape != (3,) or np.any(voxel <= 0):
        raise ValueError(f"voxel_um must be three positive values, got {voxel_um}")

    origin = _origin_or_default(origin_um, voxel)
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


def _edge_inside_lengths(G, mask, voxel_um, origin_um, step_um) -> dict:
    """``{(u, v, key): (inside_um, total_um)}`` along each edge's stored centreline.

    An edge with no usable geometry maps to ``None``; the callers decide what that means.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be a 3D volume, got shape {mask.shape}")
    voxel = np.asarray(voxel_um, dtype=float)
    origin = _origin_or_default(origin_um, voxel)
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
                result[(u, v, key)] = None
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
        result[(u, v, key)] = (inside_length, total_length)
    return result


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

    ``origin_um`` is the mask's first voxel corner, by default half a voxel below zero so that
    voxel k is the one centred on k x voxel, as in the graph (open item 40).

    Returns ``{(u, v, key): fraction}``. Centreline points outside the mask array count as
    outside rather than being clipped to its border. An edge with neither a polyline nor node
    positions maps to NaN.
    """
    lengths = _edge_inside_lengths(G, mask, voxel_um, origin_um, step_um)
    return {
        edge: (pair[0] / pair[1]) if pair is not None and pair[1] > 0 else float("nan")
        for edge, pair in lengths.items()
    }


def edge_length_inside_um(
    G,
    mask: np.ndarray,
    voxel_um: Sequence[float],
    *,
    origin_um: Sequence[float] | None = None,
    step_um: float | None = None,
) -> tuple:
    """Total centreline length of the network, and the part of it inside ``mask``, in µm.

    H1 §1.3's length and length within TH (open item 40). Measured on the network's own edge
    polylines, the same geometry ``per_edge_morphometry.csv`` reports, and classified by the
    same sampling as :func:`edge_tissue_fraction`, so §1.3 and H2 §2.1/§2.2 agree on which
    stretch of vessel is glomus.

    Raises on an edge with no geometry: this is a sum, and leaving an edge out would shorten
    the network without any sign in the result.
    """
    lengths = _edge_inside_lengths(G, mask, voxel_um, origin_um, step_um)
    missing = [edge for edge, pair in lengths.items() if pair is None]
    if missing:
        raise ValueError(
            f"{len(missing)} edge(s) have neither a centreline polyline nor node positions, "
            f"so their length is unknown (first: {missing[0]})")
    inside = sum(pair[0] for pair in lengths.values())
    total = sum(pair[1] for pair in lengths.values())
    return float(total), float(inside)
