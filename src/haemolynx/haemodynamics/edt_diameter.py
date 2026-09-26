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

Two ways to read a width off the mask (*method*):

``"cross_section"`` (default)
    The area of the mask's own cross-section in the plane normal to the
    centreline, as the diameter of the circle with that area -- the diameter
    Poiseuille's ``d^4`` needs. The plane is read voxel by voxel (nearest
    voxel) on a grid a quarter of a voxel fine, and only the piece of it
    joined to the centreline counts. On synthetic vessels at 0.98 x 0.98 x 1 um
    voxels it is within 3% on average from 2 to 8 um, at any orientation and
    sub-voxel placement, for elliptical cross-sections (area-equivalent), and
    with the centreline half a voxel off the axis.
``"inscribed_radius"``
    Twice the EDT inscribed radius at the centreline. Kept for comparison:
    the EDT at a voxel centre is the distance to the nearest background
    voxel's *centre*, half a voxel past the wall, so an axis-aligned 2-3 um
    vessel reads 30-40% wide; an oblique one, a flattened one or a centreline
    off the axis reads up to ~30% narrow (it is the largest *inscribed*
    circle, not the section) -- a d^4 error of 0.3x to 3.7x in resistance.

Where the plane's piece of mask is several vessels joined at necks -- two
touching vessels, or a branch and the vessel it leaves -- only the lobe the
centreline is in counts. A section that is still not closed within its
plane, or is more than twice as wide as the inscribed radius there allows
(the plane cutting a larger vessel's body near a junction), is not counted;
an edge with no section left falls back to its inscribed radius
(``edt_diameter_method`` records which was used). On a real 3,191-vessel
network about 60% of edges get a section; the rest are mostly short edges
between close junctions.

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

#: How far from its centreline (µm) a sample looks for its own vessel when the
#: centreline itself lies outside the mask -- a smoothed or reconnected
#: centreline can run a voxel or two off its segmentation, where the
#: inscribed radius is 0 and the vessel would otherwise get no width at all.
OFF_CENTRELINE_SEARCH_UM = 3.0

#: The two ways to read a width off the mask (see the module docstring).
EDT_DIAMETER_METHODS = ("cross_section", "inscribed_radius")

#: Arc length (um) either side of a sample the centreline's direction is
#: taken over, so a voxel-scale wiggle does not tilt the cross-section.
TANGENT_HALF_WINDOW_UM = 2.0

#: The finest the cross-section grid goes (a fraction of the smallest voxel
#: side), and the most points it takes across, so a large vessel's plane
#: stays a few tens of thousands of reads.
_SECTION_STEP_FRACTION = 0.25
_SECTION_MAX_POINTS_ACROSS = 161

#: How far the plane reaches (um), as a multiple of the inscribed radius
#: there plus two voxels. The inscribed radius under-reads a flattened or
#: obliquely cut vessel, and a plane only three times it was too small for
#: its own section on a third of a real network's samples; past five, more
#: of what the plane gains is other vessels.
SECTION_EXTENT_INSCRIBED_MULTIPLE = 5.0

#: A merged section is split into lobes at its necks (a watershed on the
#: in-plane distance to its outline), each lobe a peak at least this much
#: (um) above the neck -- two touching vessels, or a branch and the vessel
#: it leaves. Half a micron keeps one round or flattened section whole.
SECTION_LOBE_MIN_PROMINENCE_UM = 0.5

#: The widest a section's area-equivalent radius may be against the
#: inscribed radius at the same point (plus half a voxel) and still be this
#: vessel's own: 2 admits a 4:1 flattened section. Past it the plane is
#: cutting something else too -- near a junction with a much larger vessel,
#: that vessel's body, which the inscribed radius at the centreline does not
#: see.
SECTION_TO_INSCRIBED_MAX_RATIO = 2.0

__all__ = ["EDT_DIAMETER_METHODS", "measure_edge_diameters_from_binary_mask"]


def measure_edge_diameters_from_binary_mask(
    G: Any,
    *,
    binary_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    sample_spacing_along_edge_um: float,
    branch_endpoint_exclusion_um: float = 0.0,
    aggregation: Literal["median", "mean"] = "median",
    use_memmap: bool = False,
    method: Literal["cross_section", "inscribed_radius"] = "cross_section",
) -> dict[str, Any]:
    """Measure per-edge diameters (µm) from *binary_mask*.

    With *method* ``"cross_section"`` (default) the diameter at each
    centerline sample is that of the circle with the area of the mask's own
    cross-section there; with ``"inscribed_radius"`` it is ``2 * radius``,
    *radius* being the mask's EDT inscribed radius at that point, trilinearly
    interpolated the same way
    :func:`haemolynx.haemodynamics.automated.measure_edge_diameters_fwhm_from_raw_tiff`
    samples its own intensity volume. See the module docstring for how
    they compare.

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
    if method not in EDT_DIAMETER_METHODS:
        raise ValueError(f"method must be one of {EDT_DIAMETER_METHODS}; got {method!r}.")

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
        sample_points = pts[keep]
        sample_targets = targets[keep]
        if not np.any(keep):
            # An edge shorter than its junction zones: excluding it whole left
            # every short capillary with no width at all (238 of 3191 edges on
            # a real run), falling back to the branch-order table. Its
            # midpoint is as far from either junction as this edge allows.
            sample_targets = np.array([0.5 * total_len])
            sample_points = _interpolate_centerline(poly, s, sample_targets)
            data["edt_midpoint_only"] = True

        idx = physical_points_to_continuous_indices(sample_points, voxel_size_zyx)
        values = np.asarray(sample_radius(idx), dtype=float)
        off_mask = ~(np.isfinite(values) & (values > 0))
        if np.any(off_mask):
            values[off_mask] = _nearby_radius(
                sample_radius, idx[off_mask], voxel_size_zyx, OFF_CENTRELINE_SEARCH_UM
            )
            data["edt_off_centreline_samples"] = int(np.count_nonzero(off_mask))
        radii = [float(value) for value in values if np.isfinite(value) and value > 0]
        if not radii:
            summary["edges_skipped"].append((u, v, key, "edt_failed"))
            continue

        diameters = [2.0 * radius for radius in radii]
        used = "inscribed_radius"
        if method == "cross_section":
            tangents = _centreline_tangents(poly, s, total_len, sample_targets)
            sections = [
                _cross_section_diameter(
                    binary_mask, point, tangent, voxel_size_zyx,
                    guide_radius_um=float(radius) if np.isfinite(radius) else 0.0,
                )
                for point, tangent, radius in zip(sample_points, tangents, values)
            ]
            closed = [d for d in sections if np.isfinite(d) and d > 0]
            if closed:
                diameters = closed
                used = "cross_section"
        d_final = _aggregate_edge_diameter(diameters, aggregation)
        data["edt_diameter_um"] = d_final
        data["edt_diameter_samples_um"] = diameters
        data["edt_diameter_method"] = used
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {
                "edge": (u, v, key),
                "edt_diameter_um": d_final,
                "n_samples": len(diameters),
                "method": used,
            }
        )

    return summary


def _centreline_tangents(poly: np.ndarray, s: np.ndarray, total: float, at: np.ndarray) -> np.ndarray:
    """Unit direction of the centreline at arc lengths *at*, over a short window."""
    ahead = _interpolate_centerline(poly, s, np.clip(at + TANGENT_HALF_WINDOW_UM, 0.0, total))
    behind = _interpolate_centerline(poly, s, np.clip(at - TANGENT_HALF_WINDOW_UM, 0.0, total))
    tangents = np.asarray(ahead, dtype=float) - np.asarray(behind, dtype=float)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.where(norms > 0, norms, 1.0)


def _nearest_mask_values(mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Whether the voxel nearest each continuous index is foreground; False
    outside the volume. Reads only those voxels, so a memmap stays on disk."""
    nearest = np.rint(np.asarray(idx, dtype=float)).astype(np.intp)
    shape = np.asarray(mask.shape)
    inside = np.all((nearest >= 0) & (nearest < shape), axis=1)
    values = np.zeros(len(nearest), dtype=bool)
    if np.any(inside):
        within = nearest[inside]
        values[inside] = np.asarray(mask[within[:, 0], within[:, 1], within[:, 2]]) != 0
    return values


def _cross_section_diameter(
    mask: np.ndarray,
    point_um: np.ndarray,
    tangent: np.ndarray,
    voxel_size_zyx,
    *,
    guide_radius_um: float,
) -> float:
    """Diameter of the circle with the area of *mask*'s section through
    *point_um* normal to *tangent*; NaN where there is no closed section.

    The plane is read voxel by voxel (nearest voxel) on a grid a quarter of a
    voxel fine -- no interpolation or smoothing, which erodes a two-voxel
    vessel. Only the piece joined to the point counts (or, for a point just
    off its own mask, the piece nearest it within
    :data:`OFF_CENTRELINE_SEARCH_UM`), and of that piece only the lobe the
    point is in, where it has more than one (see
    :data:`SECTION_LOBE_MIN_PROMINENCE_UM`). A lobe touching the edge of the
    plane is not a closed section -- another vessel runs into it, or the
    vessel is wider than the plane (see
    :data:`SECTION_EXTENT_INSCRIBED_MULTIPLE`) -- so it gives NaN rather than
    an area that is partly something else. So does a closed one much wider
    than the inscribed radius there allows (see
    :data:`SECTION_TO_INSCRIBED_MAX_RATIO`).
    """
    from scipy.ndimage import distance_transform_edt, label
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    spacing = np.asarray(voxel_size_zyx, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    reference = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    across = np.cross(tangent, reference)
    across /= np.linalg.norm(across)
    other = np.cross(tangent, across)
    extent = max(
        4.0,
        SECTION_EXTENT_INSCRIBED_MULTIPLE * max(guide_radius_um, float(spacing.min()))
        + 2.0 * float(spacing.max()),
    )
    step = max(
        _SECTION_STEP_FRACTION * float(spacing.min()),
        2.0 * extent / (_SECTION_MAX_POINTS_ACROSS - 1),
    )
    offsets = np.arange(-extent, extent + 0.5 * step, step)
    plane = (
        np.asarray(point_um, dtype=float)
        + offsets[:, None, None] * across
        + offsets[None, :, None] * other
    )
    idx = physical_points_to_continuous_indices(plane.reshape(-1, 3), voxel_size_zyx)
    inside = _nearest_mask_values(mask, idx).reshape(len(offsets), len(offsets))
    pieces, count = label(inside)
    if count == 0:
        return float("nan")
    centre = len(offsets) // 2
    anchor = (centre, centre)
    piece = pieces[anchor]
    if piece == 0:
        rows, cols = np.nonzero(inside)
        gap = np.hypot(rows - centre, cols - centre) * step
        nearest = int(np.argmin(gap))
        if gap[nearest] > OFF_CENTRELINE_SEARCH_UM:
            return float("nan")
        anchor = (rows[nearest], cols[nearest])
        piece = pieces[anchor]
    section = pieces == piece
    depth = distance_transform_edt(section) * step
    peaks, n_peaks = label(h_maxima(depth, SECTION_LOBE_MIN_PROMINENCE_UM))
    if n_peaks > 1:
        lobes = watershed(-depth, peaks, mask=section)
        section = lobes == lobes[anchor]
    if section[0, :].any() or section[-1, :].any() or section[:, 0].any() or section[:, -1].any():
        return float("nan")
    area = float(np.count_nonzero(section)) * step * step
    radius = float(np.sqrt(area / np.pi))
    if guide_radius_um > 0 and radius > SECTION_TO_INSCRIBED_MAX_RATIO * (
        guide_radius_um + 0.5 * float(spacing.max())
    ):
        return float("nan")
    return 2.0 * radius


def _nearby_radius(
    sample_radius, idx: np.ndarray, voxel_size_zyx, search_um: float
) -> np.ndarray:
    """For each point (continuous voxel indices) whose own inscribed radius is
    0, the inscribed radius of the vessel it was meant to run down: the
    nearest vessel voxel within *search_um*, then uphill on the radius to the
    vessel's own middle. Still 0 when no vessel is that close.

    Uphill, not just the largest radius in reach: from a centreline a voxel
    outside the wall, the best point within reach is near the far side of
    the wall, where the radius is small -- a 6um vessel read 4.5um that way.
    The climb stops at the first local maximum (the vessel's medial axis,
    where moving along the vessel no longer increases the radius), and is
    capped at twice the search distance so it cannot wander along a
    connected network into a larger vessel.
    """
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    reach = np.maximum(1, np.ceil(float(search_um) / spacing).astype(int))
    grid = np.stack(
        np.meshgrid(*(np.arange(-r, r + 1) for r in reach), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    distance = np.linalg.norm(grid * spacing, axis=1)
    inside = distance <= float(search_um) + 1e-9
    grid, distance = grid[inside], distance[inside]
    points = np.asarray(idx, dtype=float).reshape(-1, 3)
    candidates = (points[:, None, :] + grid[None, :, :]).reshape(-1, 3)
    radii = np.asarray(sample_radius(candidates), dtype=float).reshape(len(points), len(grid))
    radii[~np.isfinite(radii)] = 0.0
    # The nearest vessel voxel in reach: smallest distance among those inside.
    nearest = np.where(radii > 0, distance[None, :], np.inf).argmin(axis=1)
    found = radii[np.arange(len(points)), nearest] > 0
    position = points + grid[nearest]
    radius = np.where(found, radii[np.arange(len(points)), nearest], 0.0)

    step_grid = np.stack(
        np.meshgrid(*(np.arange(-1, 2),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    max_steps = int(np.ceil(2.0 * float(search_um) / float(np.min(spacing))))
    climbing = found.copy()
    for _step in range(max_steps):
        if not np.any(climbing):
            break
        rows = np.flatnonzero(climbing)
        neighbours = (position[rows, None, :] + step_grid[None, :, :]).reshape(-1, 3)
        around = np.asarray(sample_radius(neighbours), dtype=float).reshape(len(rows), len(step_grid))
        around[~np.isfinite(around)] = 0.0
        best = around.argmax(axis=1)
        better = around[np.arange(len(rows)), best] > radius[rows] + 1e-9
        moved = rows[better]
        position[moved] = position[moved] + step_grid[best[better]]
        radius[moved] = around[np.arange(len(rows)), best][better]
        climbing[rows[~better]] = False
    return radius


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
        sizes = hi - lo + 1
        local = (idx - lo).T
        # One interpolation per clipped box shape (at most eight), not one per
        # point: the boxes are stacked along a leading axis that every point
        # samples exactly on its own index, so each keeps the box -- and the
        # edge handling -- a call of its own would have given it.
        for size in np.unique(sizes, axis=0):
            rows = np.flatnonzero(np.all(sizes == size, axis=1))
            boxes = np.ascontiguousarray(
                corner_radius[rows, : size[0], : size[1], : size[2]]
            )
            coordinates = np.vstack([np.arange(len(rows), dtype=float), local[:, rows]])
            values[rows] = map_coordinates(
                boxes, coordinates, order=1, mode="constant", cval=0.0
            )
        return values

    return sample
