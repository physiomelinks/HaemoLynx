"""Per-edge vessel diameter from the segmentation mask's own inscribed radius.

A genuinely different technique from :mod:`haemolynx.haemodynamics.automated`'s
FWHM fit: instead of fitting a Gaussian to a transverse intensity profile,
this samples the mask's own Euclidean-distance-transform (EDT) inscribed
radius (:func:`haemolynx.preprocessing.thick_vessels.inscribed_radius_map`,
already used elsewhere to gate thick-vessel skeletonisation) at each point
along an edge's centerline. Never overwrites FWHM's own
``fwhm_diameter_um`` -- this writes the separate ``edt_diameter_um``, for use
as either a mask-based fallback when FWHM measurement fails for a specific
edge (a vessel-specific estimate, unlike the generic branch-order table), or
a cross-check that flags a FWHM measurement as low-confidence when it
disagrees strongly with the mask's own local radius (see
``haemodynamics.poiseuille.stamp_edge_diameters``'s ``use_edt_fallback`` and
the ``fwhm_edt_disagreement_ratio``/``fwhm_low_confidence_vs_edt`` attributes
it stamps independently of that fallback).

Reuses :mod:`haemolynx.haemodynamics.automated`'s own centerline-sampling
helpers (``_arc_length_parameterize``, ``_interpolate_centerline``,
``physical_points_to_continuous_indices``) and
:func:`haemolynx.haemodynamics.automated._aggregate_edge_diameter` (the same
median-by-default aggregation FWHM uses) rather than re-deriving either --
one centerline sampler and one aggregation policy for both techniques.

A raw EDT reading right at a branch point is a real, documented failure
mode: it returns the *junction's own* larger inscribed sphere, not the
branch's true radius (a real ~34% overestimate has been measured on short
capillary segments this way -- a `d^4` error in the resistance it feeds).
``branch_endpoint_exclusion_um`` guards against this the same way FWHM's own
identically-named parameter does: skip sample positions too close (by
centerline arc length) to an edge endpoint that is itself a bifurcation
(graph degree > 1).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from haemolynx.preprocessing.pointwise_distance import FeatureDistance
from haemolynx.preprocessing.thick_vessels import inscribed_radius_map

from .automated import (
    _aggregate_edge_diameter,
    _arc_length_parameterize,
    _interpolate_centerline,
    physical_points_to_continuous_indices,
)

__all__ = ["measure_edge_diameters_from_binary_mask"]


def measure_edge_diameters_from_binary_mask(
    G: Any,
    *,
    binary_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    sample_spacing_along_edge_um: float,
    branch_endpoint_exclusion_um: float = 0.0,
    aggregation: Literal["median", "mean"] = "median",
    use_memmap: bool = False,
) -> dict[str, Any]:
    """Measure per-edge diameters (µm) from *binary_mask*'s own inscribed radius.

    Diameter at each centerline sample is ``2 * radius``, where *radius* is
    the mask's EDT-based inscribed radius (physical microns) at that point,
    trilinearly interpolated the same way
    :func:`haemolynx.haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`
    samples its own intensity volume.

    Parameters
    ----------
    G :
        MultiGraph with a ``voxels`` edge attribute (physical coordinates,
        same axis order as *binary_mask*).
    binary_mask :
        The segmented vessel mask this edge's centerline was built from.
    voxel_size_zyx :
        Spacing per array axis (same tuple passed to graph building).
    sample_spacing_along_edge_um :
        Distance along the edge centerline between samples.
    branch_endpoint_exclusion_um :
        Distance (µm) excluded from sampling near edge endpoints that are
        bifurcation nodes (graph degree > 1) -- guards against the raw EDT
        reading a junction's own larger inscribed sphere as if it were this
        branch's own radius.
    aggregation :
        How to collapse one edge's accepted per-sample diameters into one
        value -- ``median`` (default) or ``mean``. Shares
        :func:`haemolynx.haemodynamics.automated._aggregate_edge_diameter`
        with the FWHM path, so both techniques resolve outliers the same way.
    use_memmap :
        The low-RAM option. Instead of transforming the whole mask, reads the
        inscribed radius only at the voxels each sample's interpolation
        touches (see :mod:`haemolynx.preprocessing.pointwise_distance`), and
        never converts *binary_mask* to a whole-volume boolean copy. Same
        diameters, bar a last-bit difference where two background voxels are
        exactly equidistant under a non-representable voxel spacing.

    Returns
    -------
    A summary shaped like
    :func:`haemolynx.haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`'s
    own: ``edges_measured``, ``edges_skipped`` (list of
    ``(u, v, key, reason)``), ``per_edge``.
    """
    if sample_spacing_along_edge_um <= 0:
        raise ValueError("sample_spacing_along_edge_um must be positive.")

    sample_radius = None
    if use_memmap:
        sample_radius = _pointwise_radius_sampler(binary_mask, voxel_size_zyx)
    if sample_radius is None:
        mask = np.asarray(binary_mask, dtype=bool)
        radius_map = inscribed_radius_map(mask, voxel_size_zyx)

        def sample_radius(idx: np.ndarray) -> np.ndarray:
            return map_coordinates(radius_map, idx.T, order=1, mode="constant", cval=0.0)

    branch_excl = max(0.0, float(branch_endpoint_exclusion_um))

    summary: dict[str, Any] = {
        "edges_measured": 0,
        "edges_skipped": [],
        "per_edge": [],
    }

    for u, v, key, data in G.edges(keys=True, data=True):
        vox = data.get("voxels")
        if not vox or len(vox) < 2:
            summary["edges_skipped"].append((u, v, key, "no_voxels"))
            continue

        poly = np.asarray(vox, dtype=float)
        s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            summary["edges_skipped"].append((u, v, key, "zero_length"))
            continue

        n_samples = max(1, int(np.floor(total_len / sample_spacing_along_edge_um)) + 1)
        targets = np.linspace(0.0, total_len, n_samples)
        pts = _interpolate_centerline(poly, s, targets)

        u_is_branch = int(G.degree(u)) > 1
        v_is_branch = int(G.degree(v)) > 1
        keep = np.ones(n_samples, dtype=bool)
        if branch_excl > 0.0:
            if u_is_branch:
                keep &= targets >= branch_excl
            if v_is_branch:
                keep &= (total_len - targets) >= branch_excl
        if not np.any(keep):
            summary["edges_skipped"].append((u, v, key, "excluded_near_branch"))
            continue

        idx = physical_points_to_continuous_indices(pts[keep], voxel_size_zyx)
        values = sample_radius(idx)
        radii = [float(value) for value in values if np.isfinite(value) and value > 0]
        if not radii:
            summary["edges_skipped"].append((u, v, key, "edt_failed"))
            continue

        diameters = [2.0 * radius for radius in radii]
        d_final = _aggregate_edge_diameter(diameters, aggregation)
        data["edt_diameter_um"] = d_final
        data["edt_diameter_samples_um"] = diameters
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {
                "edge": (u, v, key),
                "edt_diameter_um": d_final,
                "n_samples": len(diameters),
            }
        )

    return summary


def _pointwise_radius_sampler(binary_mask: np.ndarray, voxel_size_zyx):
    """``map_coordinates(inscribed_radius_map(mask), idx.T, order=1,
    mode="constant")`` without the map, or ``None`` when the mask is all
    foreground (where scipy's transform is not a distance, so only the
    whole-volume call reproduces it).

    Each point is interpolated over the 2x2x2 box of voxels it reads, clipped
    to the volume the same way the whole map is, so scipy's own interpolation
    runs on the same values with the same edge handling.
    """
    shape = np.asarray(binary_mask.shape)
    if not np.any(binary_mask):
        return lambda idx: np.zeros(len(idx), dtype=np.float64)
    distances = FeatureDistance(
        binary_mask, feature_value=False, sampling=tuple(float(v) for v in voxel_size_zyx)
    )
    if not distances.has_surface:
        return None

    def sample(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=float).reshape(-1, 3)
        base = np.floor(idx).astype(np.intp)
        lo = np.clip(base, 0, shape - 1)
        hi = np.clip(base + 1, 0, shape - 1)
        corners = np.stack(
            [np.stack([np.where(bits[d], hi[:, d], lo[:, d]) for d in range(3)], axis=1)
             for bits in np.ndindex(2, 2, 2)],
            axis=1,
        ).reshape(-1, 3)
        unique, inverse = np.unique(corners, axis=0, return_inverse=True)
        corner_radius = distances.at(unique)[inverse.reshape(-1)].reshape(len(idx), 2, 2, 2)
        values = np.empty(len(idx), dtype=np.float64)
        for n in range(len(idx)):
            size = hi[n] - lo[n] + 1
            box = corner_radius[n, : size[0], : size[1], : size[2]]
            values[n] = map_coordinates(
                box, (idx[n] - lo[n]).reshape(3, 1), order=1, mode="constant", cval=0.0
            )[0]
        return values

    return sample
