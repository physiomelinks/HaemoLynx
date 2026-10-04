"""Per-edge internal diameter from an endothelial stain: the lumen inside the wall.

A plasma label shows the column blood fills; an endothelial label (claudin-5,
CD31, isolectin...) shows the wall round it. Across a vessel wide enough to
resolve it, the endothelial channel is a bright ring round a dark lumen, and
the ring's inner edge is the vessel's internal diameter -- wall to wall, what
``diameter_basis="anatomical"`` means, so each edge measured this way carries
that basis for its own viscosity (see
``haemodynamics.poiseuille.edge_diameter_basis``).

At each reading the endothelial channel is sampled in the plane normal to the
centreline and averaged over a few microns of the vessel along its curve
(:mod:`haemolynx.haemodynamics.sections`), then fitted as a background, a
dark elliptical lumen and a bright wall ring round it, both seen through the
microscope's blur projected onto the plane. The internal diameter is the
lumen's area-equivalent one.

On synthetic rings with this repository's embryonic stack's blur (sigma 1.0
um across, 3.5 um along z) and wall (1.3 um), the internal diameter is within
1% from 6 um up at any orientation and 3-6% at 4 um; a 3 um vessel lying in
the image plane is not resolved.

On that stack's claudin-5 channel (3,191 vessels) it calibrated itself at
sigma 1.1 um across, 3.0 um along z and a 1.1 um wall, and measured 674
vessels -- a third of the network's length, mostly the wider vessels; the rest
fell back. There it agrees with the plasma segmentation's cross-section on
8-12 um vessels (1.05x) and reads narrower vessels wider than the
segmentation (1.2x at 5-8 um, 1.8x at 3-5 um), where the images show the
plasma filling out to the ring. Readings along one vessel agree within 3-7%.
Each edge is independent, so edges are measured in worker processes
(``workers``): 500 vessels took 170 s in one and 41 s in eight, with the
same widths.

The blur and the wall's thickness are estimated from the rings of the
image's own wide vessels (:func:`calibrate_from_rings`), unless given, and
every other section is fitted with both held fixed: across the image plane a
thin wall's thickness and the blur look alike, and a free wall trades its
thickness against the lumen's. Where the blur along z is far longer than
across -- a vessel lying in the image plane -- the lumen is fitted round, its
width set by the sharp axis. A lumen narrower than the blur hiding its edge
is not resolved, and falls back.

This is the run's alternative to FWHM (``use_endothelial_diameters``): the
diameter chain becomes endothelium -> segmentation mask -> branch-order table.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares

from .automated import _aggregate_edge_diameter, _arc_length_parameterize
from .edt_diameter import _centreline_tangents, edge_sample_targets
from .sections import (
    averaged_section,
    bootstrap_interval,
    calibration_sites,
    projected_sigma,
    psf_from_section_blurs,
    relative_half_width,
    ring_images,
    run_edge_tasks,
    section_axes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ENDOTHELIAL_MIN_RING_CONTRAST",
    "RingFit",
    "calibrate_from_rings",
    "fit_ring_section",
    "measure_edge_diameters_from_endothelium",
]

#: The weakest wall a reading is accepted for: the fitted wall's brightness
#: over the lumen's, against the scatter of what the fit leaves unexplained.
ENDOTHELIAL_MIN_RING_CONTRAST = 4.0

#: The smallest lumen a reading resolves: its fitted radius over the blur
#: along the axis that sets it. Smaller, the ring's inner edge is inside the
#: blur and the lumen's size is the fit's guess, not the image's.
MIN_LUMEN_RESOLUTION = 1.0

#: The share of the way round the fitted wall it must be seen -- at least half
#: as bright as the fit predicts there -- for the ring to be this vessel's:
#: a junction, a neighbour or a merged structure fits a ring too, with its
#: wall on one side.
MIN_WALL_COVERAGE = 0.75
_COVERAGE_ANGLES = 36

#: Blur this many times longer along one axis of the section than the other
#: makes the lumen round, its width set by the sharp axis.
ROUND_LUMEN_BLUR_RATIO = 2.0

#: The fewest accepted readings an edge is measured from.
MIN_ACCEPTED_READINGS = 2
MAX_READINGS_PER_EDGE = 5

#: How far a section is averaged along the vessel (um).
DEFAULT_AVERAGE_ALONG_VESSEL_UM = 4.0

#: Rings the PSF is estimated from: vessels at least this wide by their guide,
#: read at the same sites as the raw image's blur estimate (see
#: :func:`~haemolynx.haemodynamics.sections.calibration_sites`), up to this
#: many sections.
PSF_CALIBRATION_MIN_GUIDE_UM = 6.0
PSF_CALIBRATION_MAX_SECTIONS = 5000
#: The fewest vessels (edges) with a ring that passes the calibration is made from.
PSF_CALIBRATION_MIN_FITS = 12
PSF_CALIBRATION_MIN_AXIAL_FITS = 5
#: The widest 95% interval (half its width, as a fraction of the estimate)
#: the blur may have and still be used, as for the raw image's
#: (:data:`~haemolynx.haemodynamics.raw_section.PSF_CALIBRATION_MAX_RELATIVE_INTERVAL`).
PSF_CALIBRATION_MAX_RELATIVE_INTERVAL = 0.15
_AXIAL_TILT = 0.5

_PLANE_STEP_UM = 0.25
_PLANE_MAX_POINTS_ACROSS = 97
_MIN_WINDOW_PIXELS = 30
_MAX_ASPECT = 4.0
_SIGMA_BOUNDS_UM = (0.05, 4.0)
_WALL_BOUNDS_UM = (0.2, 5.0)
_INITIAL_WALL_UM = 1.0


@dataclass(frozen=True)
class RingFit:
    """One cross-section's fitted lumen and wall."""

    diameter_um: float
    wall_um: float
    contrast: float
    resolution: float
    wall_coverage: float
    centre_offset_um: float
    aspect: float
    sigma_across_um: float
    sigma_other_um: float
    other_axis_z: float
    #: The lumen's centre in the section's own axes (a, b), um from the centreline.
    centre_ab_um: tuple[float, float] = (0.0, 0.0)


def _ring_radius(smooth, valid, offsets, rr, step, guide_um):
    """Where the angle-averaged section is brightest -- the wall's middle --
    within reach of the centreline, or ``None`` when it is brightest at the
    centreline itself (no dark lumen to speak of)."""
    reach = max(3.0, 0.9 * guide_um + 3.0)
    bins = np.arange(0.0, reach + step, max(step, 0.25))
    index = np.digitize(rr, bins)
    profile = np.array([
        float(np.mean(smooth[(index == k) & valid])) if np.any((index == k) & valid) else np.nan
        for k in range(1, len(bins))
    ])
    if not np.any(np.isfinite(profile)):
        return None
    peak = int(np.nanargmax(profile))
    if peak == 0:
        return None
    return float(0.5 * (bins[peak] + bins[peak + 1]))


def fit_ring_section(
    volume: np.ndarray,
    poly: np.ndarray,
    s: np.ndarray,
    total_len: float,
    at: float,
    voxel_size_zyx,
    *,
    guide_um: float,
    psf_sigma_zyx=None,
    average_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
    wall_um: float | None = None,
) -> tuple[RingFit | None, str]:
    """Fit the endothelial ring of the section at arc length *at*:
    ``(fit, "ok")``, or ``(None, reason)``.

    The section is fitted as a background, a dark lumen and a bright wall
    ring round it, both seen through the blur, starting from where its
    angle-averaged brightness peaks. ``RingFit.resolution`` says whether the
    lumen is wider than the blur that hides its edge (see
    :data:`MIN_LUMEN_RESOLUTION`); a section brightest at the centreline has
    no dark lumen at all.

    *wall_um* holds the wall's thickness fixed: across the y-x plane a thin
    wall's thickness and the blur look alike, and only their sum is in the
    image, so a free wall trades its thickness against the lumen's.

    *psf_sigma_zyx* ``None`` fits the blur too (what
    :func:`calibrate_from_rings` does); given, it is fixed. *guide_um*
    is a rough width (the segmentation's, say) setting how much of the plane
    to read -- the ring lies outside a plasma segmentation, up to about twice
    its radius.
    """
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    tangent = _centreline_tangents(poly, s, total_len, np.array([at]))[0]
    _t, a, b = section_axes(tangent)
    free_sigma = psf_sigma_zyx is None
    if free_sigma:
        sigma_a, sigma_b = 0.5, 1.0
    else:
        sigma_a, sigma_b = projected_sigma(a, psf_sigma_zyx), projected_sigma(b, psf_sigma_zyx)
    blur = max(sigma_a, sigma_b)
    extent = max(6.0, 1.2 * float(guide_um) + 3.0 + 3.0 * blur)
    # Half the finer blur is as fine as the section needs sampling: the
    # blurred model and image have nothing narrower in them.
    step = max(
        _PLANE_STEP_UM,
        0.5 * min(sigma_a, sigma_b),
        2.0 * extent / (_PLANE_MAX_POINTS_ACROSS - 1),
    )
    offsets = np.arange(-extent, extent + 0.5 * step, step)
    plane = averaged_section(volume, poly, s, total_len, at, spacing, offsets, average_um)
    valid = np.isfinite(plane)
    if valid.mean() < 0.5:
        return None, "off_volume"
    plane = np.where(valid, plane, 0.0)
    rr = np.hypot(offsets[:, None], offsets[None, :])
    smooth = gaussian_filter(plane, 0.5 / step)
    ring_mid = _ring_radius(smooth, valid, offsets, rr, step, float(guide_um))
    if ring_mid is None:
        return None, "no_dark_lumen"
    r0 = max(ring_mid - 0.5 * _INITIAL_WALL_UM, step)
    # Out to three blur widths past the wall along each axis: an ellipse,
    # not a circle sized for the blurrier axis.
    reach_a = ring_mid + _INITIAL_WALL_UM + 3.0 * sigma_a + 1.0
    reach_b = ring_mid + _INITIAL_WALL_UM + 3.0 * sigma_b + 1.0
    window = valid & (
        (offsets[:, None] / reach_a) ** 2 + (offsets[None, :] / reach_b) ** 2 <= 1.0
    )
    if int(window.sum()) < _MIN_WINDOW_PIXELS:
        return None, "no_dark_lumen"
    if window[0].any() or window[-1].any() or window[:, 0].any() or window[:, -1].any():
        return None, "ring_not_closed"
    margin = int(np.ceil(3.0 * blur / step)) + 2
    rw, cw = np.nonzero(window)
    crop = (
        slice(max(0, rw.min() - margin), min(len(offsets), rw.max() + margin + 1)),
        slice(max(0, cw.min() - margin), min(len(offsets), cw.max() + margin + 1)),
    )
    xs, ys = offsets[crop[0]], offsets[crop[1]]
    in_crop = window[crop]
    observed = plane[crop][in_crop]
    rr_observed = rr[crop][in_crop]
    far = rr_observed >= ring_mid + 1.0
    background0 = float(np.percentile(observed[far], 20)) if far.any() else float(np.min(observed))
    near = rr_observed <= 0.5 * r0
    lumen0 = float(np.median(observed[near])) - background0 if near.any() else 0.0
    wall0 = max(float(np.percentile(observed, 95)) - background0, 1e-3)

    fixed_wall = None if wall_um is None else float(wall_um)
    # Under a blur several times longer along one axis than the other (a
    # vessel lying in the image plane, its section spanning z), that axis
    # cannot fix the lumen's extent: the lumen is taken as round, its width
    # set by the sharp axis -- as one judges it by eye.
    round_lumen = not free_sigma and max(sigma_a, sigma_b) >= ROUND_LUMEN_BLUR_RATIO * min(sigma_a, sigma_b)

    def unpack(p):
        values = list(p)
        background, level_lumen, level_wall, cx, cy, ra = values[:6]
        rest = values[6:]
        rb = ra if round_lumen else rest.pop(0)
        wall = fixed_wall if fixed_wall is not None else rest.pop(0)
        sa, sb = (rest[0], rest[1]) if free_sigma else (sigma_a, sigma_b)
        return background, level_lumen, level_wall, cx, cy, ra, rb, wall, sa, sb

    def ring_model(p):
        background, level_lumen, level_wall, cx, cy, ra, rb, wall, sa, sb = unpack(p)
        inner, ring = ring_images(xs, ys, cx, cy, ra, rb, wall, sa, sb, step)
        return (background + level_lumen * inner + level_wall * ring)[in_crop]

    start_wall = _INITIAL_WALL_UM if fixed_wall is None else fixed_wall
    r0 = max(ring_mid - 0.5 * start_wall, step)
    lo = [-np.inf, -np.inf, 0.0, -3.0, -3.0, 0.25 * step]
    hi = [np.inf, np.inf, np.inf, 3.0, 3.0, extent]
    p0 = [background0, lumen0, wall0, 0.0, 0.0, r0]
    if not round_lumen:
        lo.append(0.25 * step)
        hi.append(extent)
        p0.append(r0)
    if fixed_wall is None:
        lo.append(_WALL_BOUNDS_UM[0])
        hi.append(_WALL_BOUNDS_UM[1])
        p0.append(_INITIAL_WALL_UM)
    if free_sigma:
        lo += [_SIGMA_BOUNDS_UM[0]] * 2
        hi += [_SIGMA_BOUNDS_UM[1]] * 2
        p0 += [sigma_a, sigma_b]
    p0 = np.clip(p0, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
    try:
        result = least_squares(
            lambda p: ring_model(p) - observed, p0, bounds=(lo, hi), x_scale="jac",
            ftol=1e-5, xtol=1e-5, max_nfev=400 if free_sigma else 150,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None, "fit_failed"
    p = result.x
    _bg, _lumen, _wall_level, _cx, _cy, ra, rb, wall, sigma_a, sigma_b = (float(v) for v in unpack(p))

    noise = float(np.sum(result.fun ** 2)) / max(len(observed) - len(p), 1)
    scatter = float(np.sqrt(noise))
    contrast = (float(p[2]) - float(p[1])) / scatter if scatter > 0 else float("inf")
    if max(ra, rb) + wall + max(abs(float(p[3])), abs(float(p[4]))) > extent - 2.0 * max(sigma_a, sigma_b):
        return None, "ring_not_closed"
    # Is the wall there all the way round, as bright as the fit says it
    # should be (dimmed by the blur where it is)? A junction, a vessel
    # beside this one, or a merged structure fits a ring too, with its
    # "wall" on one side.
    cx, cy = float(p[3]), float(p[4])
    inner_model, ring_model_image = ring_images(offsets, offsets, cx, cy, ra, rb, wall, sigma_a, sigma_b, step)
    angles = np.linspace(0.0, 2.0 * np.pi, _COVERAGE_ANGLES, endpoint=False)
    along_a = cx + (ra + 0.5 * wall) * np.cos(angles)
    along_b = cy + (rb + 0.5 * wall) * np.sin(angles)
    grid = np.vstack([(along_a - offsets[0]) / step, (along_b - offsets[0]) / step])
    seen = map_coordinates(smooth, grid, order=1, mode="nearest") - float(p[0])
    expected = float(p[2]) * map_coordinates(ring_model_image, grid, order=1, mode="nearest") + float(
        p[1]
    ) * map_coordinates(inner_model, grid, order=1, mode="nearest")
    wall_coverage = float(np.mean(seen >= 0.5 * expected))
    return RingFit(
        diameter_um=2.0 * float(np.sqrt(ra * rb)),
        wall_um=wall,
        contrast=contrast,
        resolution=min(ra / max(sigma_a, 1e-9), rb / max(sigma_b, 1e-9))
        if not round_lumen else ra / max(min(sigma_a, sigma_b), 1e-9),
        wall_coverage=wall_coverage,
        centre_offset_um=float(np.hypot(p[3], p[4])),
        aspect=max(ra, rb) / max(min(ra, rb), 1e-9),
        sigma_across_um=sigma_a,
        sigma_other_um=sigma_b,
        other_axis_z=abs(float(b[0])),
        centre_ab_um=(float(p[3]), float(p[4])),
    ), "ok"


def _gate(fit: RingFit, *, min_contrast: float, min_diameter_um: float) -> str:
    """Why *fit* is not this vessel's internal diameter, or ``"ok"``."""
    if fit.contrast < min_contrast:
        return "low_contrast"
    if fit.resolution < MIN_LUMEN_RESOLUTION:
        return "unresolved"
    if fit.wall_coverage < MIN_WALL_COVERAGE:
        return "partial_wall"
    if fit.centre_offset_um > max(1.5, 0.3 * fit.diameter_um):
        return "off_centre"
    if fit.aspect > _MAX_ASPECT:
        return "elongated"
    if fit.diameter_um < min_diameter_um:
        return "unresolved"
    return "ok"


def _guide_diameter(data: dict, guide_attribute: str | None, fallback_um: float) -> float:
    if guide_attribute:
        try:
            value = float(data.get(guide_attribute))
        except (TypeError, ValueError):
            value = None
        if value is not None and np.isfinite(value) and value > 0:
            return value
    return float(fallback_um)


def calibrate_from_rings(
    G: Any,
    volume: np.ndarray,
    voxel_size_zyx,
    *,
    psf_sigma_zyx=None,
    guide_attribute: str | None = "edt_diameter_um",
    average_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
    min_contrast: float = ENDOTHELIAL_MIN_RING_CONTRAST,
    seed: int = 0,
    edges: Iterable[tuple[Any, Any, Any]] | None = None,
    workers: int = 1,
    memmap_directory=None,
) -> tuple[tuple[float, float, float] | None, float | None, dict[str, Any]]:
    """``(psf_sigma_zyx, wall_um, details)`` from the image's wide vessels'
    rings; ``(None, None, details)`` when too few show them, or they do not
    agree on the blur.

    Every edge at least :data:`PSF_CALIBRATION_MIN_GUIDE_UM` wide -- among
    *edges* (``(u, v, key)``) when given -- is read at its middle and every
    few microns out from it, as the raw image's blur estimate reads them
    (:func:`~haemolynx.haemodynamics.raw_section.estimate_psf_sigma`; up to
    :data:`PSF_CALIBRATION_MAX_SECTIONS` sections), each fitted with its wall
    free, and its blur free too unless *psf_sigma_zyx* is given. A robust fit
    over their blurs gives sigma_xy and sigma_z (see
    :func:`~haemolynx.haemodynamics.sections.psf_from_section_blurs`); their
    median wall is the wall every other section is fitted with. Across the
    y-x plane a thin wall's thickness and the blur trade against each other,
    so this pair is the image's, consistent with each other rather than each
    exact: the ring's middle -- what fixes a lumen -- is what they agree on.

    ``details`` reports each value's 95% interval (``sigma_xy_interval_um``,
    ``sigma_z_interval_um``, ``wall_interval_um``; a vessel's sections
    resampled together, *seed* seeding the resampling); a blur interval wider
    than :data:`PSF_CALIBRATION_MAX_RELATIVE_INTERVAL` either way gives
    ``(None, None, details)``. *seed* otherwise only picks which vessels when
    there are more sections than the cap. *workers* processes fit the
    sections; the answer is the same however many.
    """
    sites = calibration_sites(
        G, guide_um=lambda data: _guide_diameter(data, guide_attribute, 0.0),
        min_guide_um=PSF_CALIBRATION_MIN_GUIDE_UM, average_um=average_um,
        max_sections=PSF_CALIBRATION_MAX_SECTIONS, seed=seed, edges=edges,
    )
    payloads = [
        {
            "poly": np.asarray(G.edges[edge]["voxels"], dtype=float),
            "targets": targets,
            "guide": _guide_diameter(G.edges[edge], guide_attribute, 0.0),
            "spacing": tuple(float(v) for v in voxel_size_zyx),
            "psf": None if psf_sigma_zyx is None else tuple(float(v) for v in psf_sigma_zyx),
            "average": float(average_um),
        }
        for edge, targets in sites
    ]
    readings = run_edge_tasks(
        _calibration_rings, payloads, volume=volume, workers=workers,
        memmap_directory=memmap_directory,
    )
    fits: list[RingFit] = []
    groups: list[int] = []
    for group, edge_readings in enumerate(readings):
        for fit, _why in edge_readings:
            if fit is None or _gate(fit, min_contrast=min_contrast, min_diameter_um=0.0) != "ok":
                continue
            if psf_sigma_zyx is None and not all(
                _SIGMA_BOUNDS_UM[0] * 1.01 < v < _SIGMA_BOUNDS_UM[1] * 0.99
                for v in (fit.sigma_across_um, fit.sigma_other_um)
            ):
                continue
            if not _WALL_BOUNDS_UM[0] * 1.01 < fit.wall_um < _WALL_BOUNDS_UM[1] * 0.99:
                continue
            fits.append(fit)
            groups.append(group)
    axial = sum(1 for f in fits if f.other_axis_z >= _AXIAL_TILT)
    details: dict[str, Any] = {
        "sections_tried": int(sum(len(targets) for _edge, targets in sites)),
        "edges_tried": len(sites),
        "sections_fitted": len(fits),
        "edges_fitted": len(set(groups)),
        "axial_sections": axial,
    }
    needs_axial = psf_sigma_zyx is None
    if details["edges_fitted"] < PSF_CALIBRATION_MIN_FITS or (
        needs_axial and axial < PSF_CALIBRATION_MIN_AXIAL_FITS
    ):
        details["reason"] = (
            f"{len(fits)} rings of {details['edges_fitted']} wide vessels showed their wall "
            f"({axial} tilted towards z); at least {PSF_CALIBRATION_MIN_FITS} vessels"
            + (f" ({PSF_CALIBRATION_MIN_AXIAL_FITS} tilted rings)" if needs_axial else "")
            + " are needed"
        )
        logger.warning("Endothelial rings not calibrated: %s.", details["reason"])
        return None, None, details
    walls = np.array([f.wall_um for f in fits])
    wall = float(np.median(walls))
    details["wall_um"] = wall
    if psf_sigma_zyx is None:
        across = np.array([f.sigma_across_um for f in fits])
        other = np.array([f.sigma_other_um for f in fits])
        tilt = np.array([f.other_axis_z for f in fits])
        sigma_xy, sigma_z = psf_from_section_blurs(across, other, tilt)
        xy_interval, z_interval, wall_interval = bootstrap_interval(
            lambda pick: (*psf_from_section_blurs(across[pick], other[pick], tilt[pick]),
                          np.median(walls[pick])),
            groups, seed=seed,
        )
        details.update(
            sigma_xy_um=sigma_xy, sigma_z_um=sigma_z,
            sigma_xy_interval_um=xy_interval, sigma_z_interval_um=z_interval,
        )
        psf_sigma_zyx = (sigma_z, sigma_xy, sigma_xy)
        blur = (
            f"sigma_xy {sigma_xy:.2f} um (95% {xy_interval[0]:.2f}-{xy_interval[1]:.2f}), "
            f"sigma_z {sigma_z:.2f} um (95% {z_interval[0]:.2f}-{z_interval[1]:.2f}), "
        )
        widest = max(relative_half_width(xy_interval, sigma_xy), relative_half_width(z_interval, sigma_z))
    else:
        (wall_interval,) = bootstrap_interval(lambda pick: np.median(walls[pick]), groups, seed=seed)
        blur, widest = "", 0.0
    details["wall_interval_um"] = wall_interval
    described = (
        f"{blur}wall {wall:.2f} um (95% {wall_interval[0]:.2f}-{wall_interval[1]:.2f}) from "
        f"{len(fits)} rings of {details['edges_fitted']} wide vessels ({details['sections_tried']} tried)"
    )
    if widest > PSF_CALIBRATION_MAX_RELATIVE_INTERVAL:
        details["reason"] = (
            f"the wide vessels' rings do not agree on the blur: {described}, an interval of "
            f"+-{widest:.0%}, past +-{PSF_CALIBRATION_MAX_RELATIVE_INTERVAL:.0%}"
        )
        logger.warning("Endothelial rings not calibrated: %s.", details["reason"])
        return None, None, details
    logger.info("Endothelial rings: %s.", described)
    return tuple(float(v) for v in psf_sigma_zyx), wall, details


def _calibration_rings(volume: np.ndarray, payload: dict) -> list[tuple[RingFit | None, str]]:
    """One edge's calibration sections, each fitted with its wall free (and
    its blur, unless the payload fixes it). Self-contained, so it runs the
    same in a worker process (see
    :func:`~haemolynx.haemodynamics.sections.run_edge_tasks`)."""
    poly = payload["poly"]
    s, total_len = _arc_length_parameterize(poly)
    return [
        fit_ring_section(
            volume, poly, s, total_len, float(at), payload["spacing"], guide_um=payload["guide"],
            psf_sigma_zyx=payload["psf"], average_um=payload["average"],
        )
        for at in payload["targets"]
    ]


_EDGE_ATTRIBUTES = (
    "endothelial_diameter_um",
    "endothelial_diameter_samples_um",
    "endothelial_contrast_samples",
    "endothelial_status",
)


def _read_edge(volume: np.ndarray, payload: dict) -> tuple[list[float], list[float], dict[str, int]]:
    """One edge's accepted readings ``(diameters, contrasts, rejected)``.

    Self-contained -- the edge's centreline, where to read it and every
    setting travel in *payload* -- so it runs the same in a worker process
    (see :func:`~haemolynx.haemodynamics.sections.run_edge_tasks`). Stops
    reading once the rest could no longer make the edge's
    :data:`MIN_ACCEPTED_READINGS`: it has failed whatever they say.
    """
    poly = payload["poly"]
    s, total_len = _arc_length_parameterize(poly)
    diameters: list[float] = []
    contrasts: list[float] = []
    reasons: Counter = Counter()
    targets = payload["targets"]
    for index, at in enumerate(targets):
        if len(diameters) + (len(targets) - index) < MIN_ACCEPTED_READINGS:
            break
        fit, why = fit_ring_section(
            volume, poly, s, total_len, float(at), payload["spacing"], guide_um=payload["guide"],
            psf_sigma_zyx=payload["psf"], average_um=payload["average"], wall_um=payload["wall"],
        )
        if fit is not None:
            why = _gate(fit, min_contrast=payload["min_contrast"], min_diameter_um=payload["min_diameter"])
        if why != "ok":
            reasons[why] += 1
            continue
        diameters.append(fit.diameter_um)
        contrasts.append(fit.contrast)
    return diameters, contrasts, dict(reasons)


def measure_edge_diameters_from_endothelium(
    G: Any,
    *,
    endothelial_volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    psf_sigma_zyx: tuple[float, float, float] | None = None,
    edges: Iterable[tuple[Any, Any, Any]] | None = None,
    sample_spacing_along_edge_um: float = 2.0,
    branch_endpoint_exclusion_um: float = 10.0,
    average_along_vessel_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
    min_ring_contrast: float = ENDOTHELIAL_MIN_RING_CONTRAST,
    min_diameter_pixels: float = 2.0,
    aggregation: Literal["median", "mean"] = "median",
    guide_attribute: str | None = "edt_diameter_um",
    fallback_guide_um: float = 4.0,
    wall_um: float | None = None,
    workers: int = 1,
    memmap_directory=None,
    calibration_edges: Iterable[tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Measure *edges* (default every edge) from their endothelial rings.

    *workers* processes share the edges, and the calibration's sections (see
    :func:`~haemolynx.haemodynamics.sections.run_edge_tasks`); the results
    are the same however many there are.

    Writes ``endothelial_diameter_um`` (the *aggregation* of the accepted
    readings' internal diameters), ``endothelial_diameter_samples_um``,
    ``endothelial_contrast_samples`` and ``endothelial_status``
    (``"measured"`` or ``"failed:<commonest reason>"``) on each edge it
    reads, and the PSF and wall in ``G.graph["endothelial_psf_sigma_zyx"]``
    and ``G.graph["endothelial_wall_um"]``. Either left ``None`` is estimated
    from the wide vessels' rings (:func:`calibrate_from_rings`, among
    *calibration_edges* when given); when that cannot, nothing is measured
    and the summary says why.

    A reading is kept when its ring's contrast is at least
    *min_ring_contrast*, its centre lies within FWHM's centring limit of the
    centreline, it is no more than 4:1 elongated and at least
    *min_diameter_pixels* of the finest voxel side across; an edge needs
    :data:`MIN_ACCEPTED_READINGS` of them.
    """
    if sample_spacing_along_edge_um <= 0:
        raise ValueError("sample_spacing_along_edge_um must be positive.")
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    summary: dict[str, Any] = {"edges_measured": 0, "edges_skipped": [], "per_edge": [],
                               "readings_rejected": {}}
    psf_source = "set" if psf_sigma_zyx is not None else "estimated"
    wall_source = "set" if wall_um is not None else "estimated"
    if psf_sigma_zyx is None or wall_um is None:
        psf_found, wall_found, calibration = calibrate_from_rings(
            G, endothelial_volume, spacing, psf_sigma_zyx=psf_sigma_zyx,
            guide_attribute=guide_attribute, average_um=average_along_vessel_um,
            min_contrast=min_ring_contrast, edges=calibration_edges, workers=int(workers),
            memmap_directory=memmap_directory,
        )
        summary["calibration"] = calibration
        if psf_found is None:
            summary["skipped"] = True
            summary["reason"] = "could not calibrate on the endothelial rings: " + calibration["reason"]
            return summary
        psf_sigma_zyx = psf_found
        wall_um = wall_found if wall_um is None else wall_um
    G.graph["endothelial_psf_sigma_zyx"] = tuple(float(v) for v in psf_sigma_zyx)
    G.graph["endothelial_psf_source"] = psf_source
    G.graph["endothelial_wall_um"] = float(wall_um)
    G.graph["endothelial_wall_source"] = wall_source
    summary["psf_sigma_zyx"] = G.graph["endothelial_psf_sigma_zyx"]
    summary["wall_um"] = float(wall_um)
    min_diameter_um = max(0.0, float(min_diameter_pixels)) * float(np.min(spacing))
    rejected: Counter = Counter()

    chosen = list(G.edges(keys=True)) if edges is None else list(edges)
    payloads = []
    for u, v, key in chosen:
        data = G.edges[u, v, key]
        for stale in _EDGE_ATTRIBUTES:
            data.pop(stale, None)
        vox = data.get("voxels")
        if not vox or len(vox) < 2:
            summary["edges_skipped"].append((u, v, key, "no_voxels"))
            data["endothelial_status"] = "failed:no_voxels"
            continue
        poly = np.asarray(vox, dtype=float)
        _s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            summary["edges_skipped"].append((u, v, key, "zero_length"))
            data["endothelial_status"] = "failed:zero_length"
            continue
        targets, _short = edge_sample_targets(
            total_len, sample_spacing_along_edge_um, branch_endpoint_exclusion_um,
            start_is_branch=int(G.degree(u)) > 1, end_is_branch=int(G.degree(v)) > 1,
        )
        if len(targets) > MAX_READINGS_PER_EDGE:
            targets = targets[np.linspace(0, len(targets) - 1, MAX_READINGS_PER_EDGE).round().astype(int)]
        payloads.append({
            "edge": (u, v, key), "poly": poly, "targets": [float(a) for a in targets],
            "guide": _guide_diameter(data, guide_attribute, fallback_guide_um),
            "spacing": tuple(float(x) for x in spacing), "psf": tuple(psf_sigma_zyx),
            "wall": float(wall_um), "average": float(average_along_vessel_um),
            "min_contrast": float(min_ring_contrast), "min_diameter": float(min_diameter_um),
        })
    results = run_edge_tasks(
        _read_edge, payloads, volume=endothelial_volume, workers=int(workers),
        memmap_directory=memmap_directory,
    )
    for payload, (diameters, contrasts, reasons) in zip(payloads, results):
        u, v, key = payload["edge"]
        data = G.edges[u, v, key]
        rejected.update(reasons)
        if len(diameters) < MIN_ACCEPTED_READINGS:
            reason = "too_few_readings" if diameters else (
                Counter(reasons).most_common(1)[0][0] if reasons else "no_readings"
            )
            summary["edges_skipped"].append((u, v, key, reason))
            data["endothelial_status"] = f"failed:{reason}"
            continue
        d_final = _aggregate_edge_diameter(diameters, aggregation)
        data["endothelial_diameter_um"] = d_final
        data["endothelial_diameter_samples_um"] = diameters
        data["endothelial_contrast_samples"] = contrasts
        data["endothelial_status"] = "measured"
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {"edge": (u, v, key), "endothelial_diameter_um": d_final, "n_samples": len(diameters)}
        )
    summary["readings_rejected"] = dict(rejected)
    return summary
