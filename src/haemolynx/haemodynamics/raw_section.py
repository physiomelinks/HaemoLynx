"""Per-edge vessel diameter from the raw image's own cross-section: the step after FWHM.

FWHM (:mod:`haemolynx.haemodynamics.automated`) fits one transverse line at a
time, gated on the fit's R^2, its centring and the line's length. On a dim,
photon-limited channel that fails where a single line cannot carry the
vessel: wide vessels (on a real 3,191-vessel network the R^2 gate alone
rejected 1,162 edges, median 9.9 um, against 3.9 um for those it kept), fits
off the centreline, and every vessel running along z, where a line in the
y-x plane is not across it.

This reads the whole cross-section instead. At each sample point the raw
image is sampled in the plane normal to the centreline, averaged over a few
microns of the vessel along its curve, and fitted as a uniformly filled
elliptical lumen seen through the microscope's blur -- FWHM's own blurred-
lumen model (:func:`automated._blurred_lumen_1d`) in two dimensions, pooling
every photon in the section. The blur is anisotropic (a two-photon PSF is
several times longer in z than across), and the fit knows it: the PSF is
projected onto the plane, so a small vessel whose section contains z is not
read narrow the way a line profile is. The diameter is the ellipse's area-
equivalent one, what Poiseuille's d^4 needs.

On synthetic vessels built like a real dim channel (a few photons per voxel,
shot noise, 0.35 x 1.2 um blur) it is within 6% on average from 3 um up at
any orientation (a 2 um vessel reads 7-38% wide), where FWHM reads 0.83-0.90
of the true width and cannot measure a z-running vessel at all. The blur it
estimates from wide vessels there is within 10% of the true one. It depends
on the PSF: assumed 1.5x too
wide, 2-4 um vessels read ~0.65 of their width. So the PSF is estimated from
the image's own wide vessels (:func:`estimate_psf_sigma`), whose edges show
the blur plainly, unless it is given.

**It measures only a vessel that shows a lumen.** Each reading is gated on
its lumen's contrast against the section's texture outside it, on the lumen
closing within its plane and on its centre lying near the centreline, and a
vessel needs two readings that pass: a blob of speckle can pass once, a
lumen passes all along the vessel. On an
image whose sections are speckle with no clean lumen edge -- a real embryonic
brain stack, where the plasma dye appears to have leaked into the tissue --
fits latch onto the bright halo round a vessel and read about twice its
width, and agree with FWHM within 20% on only a quarter of vessels at any
contrast gate. Against the same stack's endothelial internal diameters (see
:mod:`haemolynx.haemodynamics.endothelial`), on 674 vessels, it read 1.37x
(median) where the plasma segmentation's cross-section read 0.95x and FWHM
0.88x. That is why it is off by default -- and why, though it beats FWHM on
clean synthetic vessels, it did not replace FWHM as the default: check it
against FWHM on your own image before trusting it (``raw_section_*`` edge
attributes).

Vessels are read where the segmentation-mask estimate reads them
(:func:`haemolynx.haemodynamics.edt_diameter.edge_sample_targets`): every
FWHM sample spacing, clear of junction zones, and a short edge several times
across its middle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label
from scipy.optimize import least_squares

from .automated import _aggregate_edge_diameter, _arc_length_parameterize, _interpolate_centerline
from .edt_diameter import _centreline_tangents, edge_sample_targets
from .poiseuille import edge_selection
from .sections import (
    averaged_section,
    lumen_image,
    nearest_section,
    projected_sigma,
    psf_from_section_blurs,
    section_axes,
)

__all__ = [
    "MIN_ACCEPTED_READINGS",
    "RAW_SECTION_MIN_LUMEN_CONTRAST",
    "SectionFit",
    "estimate_psf_sigma",
    "fit_section",
    "measure_edge_diameters_from_raw_sections",
]

#: The weakest lumen a reading is accepted for: its fitted brightness above
#: the background, over the standard deviation of the section outside it --
#: the texture and noise a lumen has to stand out from. On synthetic
#: sections of speckled tissue with no vessel, 90% of fits stay under 3.1;
#: on a vessel embedded in the same speckle the median is 4.1, and on a
#: clean one 33.
RAW_SECTION_MIN_LUMEN_CONTRAST = 3.0

#: The fewest accepted readings an edge is measured from. A blob of speckle
#: can pass the contrast gate at one place; a lumen passes it all along the
#: vessel. With one reading enough, half of synthetic speckle-only edges got
#: a width; with two, none did, and every vessel embedded in speckle kept its.
MIN_ACCEPTED_READINGS = 2

#: How far the section is averaged along the vessel (um) when no length is
#: given: a vessel's width barely changes over a few microns while photon
#: noise is independent voxel to voxel. FWHM's own longitudinal average.
DEFAULT_AVERAGE_ALONG_VESSEL_UM = 4.0

#: Readings per edge at most: an edge's width is its median, and past five
#: more readings of one vessel mostly cost time.
MAX_READINGS_PER_EDGE = 5

#: The PSF is estimated from vessels at least this wide by their guide
#: diameter, whose edges show the blur separately from the width.
PSF_CALIBRATION_MIN_GUIDE_UM = 6.0
PSF_CALIBRATION_MAX_SECTIONS = 150
PSF_CALIBRATION_MIN_FITS = 12
#: Sections tilted this much towards z (the across axis ``b``'s z component)
#: are the ones that see the axial blur; at least this many are needed to
#: estimate it.
PSF_CALIBRATION_MIN_AXIAL_FITS = 5
_AXIAL_TILT = 0.5

_PLANE_STEP_UM = 0.25
_PLANE_MAX_POINTS_ACROSS = 81
_MIN_WINDOW_PIXELS = 30
_MAX_ASPECT = 4.0
_SIGMA_BOUNDS_UM = (0.05, 4.0)


@dataclass(frozen=True)
class SectionFit:
    """One cross-section's fitted lumen."""

    diameter_um: float
    contrast: float
    centre_offset_um: float
    aspect: float
    sigma_across_um: float
    sigma_other_um: float
    other_axis_z: float


def _own_region(plane, valid, offsets, rr, step, guide_um, mask_plane):
    """This vessel's region of the section, and every other structure in it.

    From the segmentation when there is one -- the piece at the centreline,
    or nearest it within 3 um -- else from the raw section's half maximum,
    both of which leave a neighbouring vessel out of the fit."""
    centre = len(offsets) // 2
    if mask_plane is not None:
        inside = np.nan_to_num(mask_plane) >= 0.5
    else:
        smooth = gaussian_filter(np.where(valid, plane, 0.0), 0.5 / step)
        ring = valid & (rr > 0.8 * float(offsets[-1]))
        if not ring.any():
            return None, None
        background = float(np.percentile(smooth[ring], 25))
        core = valid & (rr <= max(1.0, 0.35 * guide_um))
        if not core.any():
            return None, None
        peak = float(np.max(smooth[core]))
        if peak <= background:
            return None, None
        inside = binary_fill_holes(valid & (smooth >= background + 0.5 * (peak - background)))
    pieces, _count = label(inside)
    own_id = int(pieces[centre, centre])
    if own_id == 0:
        near = np.argwhere(inside & (rr <= 3.0))
        if not len(near):
            return None, None
        own_id = int(pieces[tuple(near[np.argmin(rr[tuple(near.T)])])])
    own = pieces == own_id
    return own, inside & ~own


def fit_section(
    raw: np.ndarray,
    poly: np.ndarray,
    s: np.ndarray,
    total_len: float,
    at: float,
    voxel_size_zyx,
    *,
    guide_um: float,
    psf_sigma_zyx=None,
    vessel_mask: np.ndarray | None = None,
    average_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
) -> tuple[SectionFit | None, str]:
    """Fit the lumen of the cross-section at arc length *at* of centreline
    *poly*: ``(fit, "ok")``, or ``(None, reason)`` when there is nothing to fit.

    *psf_sigma_zyx* ``None`` fits the blur too (two widths, across in the y-x
    plane and along the section's other axis) -- what
    :func:`estimate_psf_sigma` does on wide vessels; given, it is fixed.
    *guide_um* is a rough width (the mask's, say) setting how much of the
    plane to read. No gate is applied here but the fit's own; see
    :func:`measure_edge_diameters_from_raw_sections` for those.
    """
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    tangent = _centreline_tangents(poly, s, total_len, np.array([at]))[0]
    _t, a, b = section_axes(tangent)
    free_sigma = psf_sigma_zyx is None
    if free_sigma:
        sigma_a = sigma_b = 0.5
    else:
        sigma_a = projected_sigma(a, psf_sigma_zyx)
        sigma_b = projected_sigma(b, psf_sigma_zyx)
    blur = max(sigma_a, sigma_b, 1.0 if free_sigma else 0.0)
    extent = max(5.0, 1.6 * float(guide_um) + 3.0 * blur)
    step = max(_PLANE_STEP_UM, 2.0 * extent / (_PLANE_MAX_POINTS_ACROSS - 1))
    offsets = np.arange(-extent, extent + 0.5 * step, step)
    plane = averaged_section(raw, poly, s, total_len, at, spacing, offsets, average_um)
    valid = np.isfinite(plane)
    if valid.mean() < 0.5:
        return None, "off_volume"
    rr = np.hypot(offsets[:, None], offsets[None, :])
    mask_plane = None
    if vessel_mask is not None:
        point = _interpolate_centerline(poly, s, np.array([at]))[0]
        mask_plane = nearest_section(vessel_mask, point, a, b, offsets, spacing)
    own, others = _own_region(plane, valid, offsets, rr, step, float(guide_um), mask_plane)
    if own is None or not own.any():
        return None, "no_lumen_region"
    if own[0].any() or own[-1].any() or own[:, 0].any() or own[:, -1].any():
        # Its region runs off the plane -- another vessel joins it, or it is
        # far wider than its guide -- so no lumen fitted to it closes; the
        # mask cross-section rejects the same. Deciding it here spares the
        # fit, which is what most of a textured image's sections cost.
        return None, "lumen_not_closed"
    grow = int(np.ceil(3.0 * blur / step)) + 2
    window = binary_dilation(own, iterations=grow) & valid
    if others is not None and others.any():
        window &= ~binary_dilation(others, iterations=int(np.ceil(blur / step)) + 1)
    if int(window.sum()) < _MIN_WINDOW_PIXELS:
        return None, "no_lumen_region"
    # The model is rendered on the window's own crop (plus a blur margin),
    # not the whole plane: the fit evaluates it a hundred-odd times.
    margin = int(np.ceil(3.0 * blur / step)) + 2
    rows_w, cols_w = np.nonzero(window)
    crop = (
        slice(max(0, rows_w.min() - margin), min(len(offsets), rows_w.max() + margin + 1)),
        slice(max(0, cols_w.min() - margin), min(len(offsets), cols_w.max() + margin + 1)),
    )
    xs, ys = offsets[crop[0]], offsets[crop[1]]
    in_crop = window[crop]
    observed = plane[crop][in_crop]
    rows, cols = np.nonzero(own)
    cx0, cy0 = float(offsets[rows].mean()), float(offsets[cols].mean())
    r0 = max(float(np.sqrt(own.sum() * step * step / np.pi)), step)
    background0 = float(np.percentile(observed, 20))
    amplitude0 = max(float(np.percentile(plane[own & valid], 75)) - background0, 1e-3) if (own & valid).any() else 1.0

    def model(p):
        if free_sigma:
            background, amplitude, cx, cy, ra, rb, sa, sb = p
        else:
            background, amplitude, cx, cy, ra, rb = p
            sa, sb = sigma_a, sigma_b
        return background + amplitude * lumen_image(xs, ys, cx, cy, ra, rb, sa, sb, step)[in_crop]

    lo = [-np.inf, 0.0, cx0 - 3.0, cy0 - 3.0, 0.25 * step, 0.25 * step]
    hi = [np.inf, np.inf, cx0 + 3.0, cy0 + 3.0, extent, extent]
    p0 = [background0, amplitude0, cx0, cy0, r0, r0]
    if free_sigma:
        lo += [_SIGMA_BOUNDS_UM[0]] * 2
        hi += [_SIGMA_BOUNDS_UM[1]] * 2
        p0 += [sigma_a, sigma_b]
    p0 = np.clip(p0, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
    try:
        result = least_squares(
            lambda p: model(p) - observed, p0, bounds=(lo, hi), x_scale="jac",
            # A width to a part in 10^5 is far past what the image carries;
            # tighter only spends iterations.
            ftol=1e-5, xtol=1e-5, max_nfev=400 if free_sigma else 100,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None, "fit_failed"
    p = result.x
    ra, rb = float(p[4]), float(p[5])
    if free_sigma:
        sigma_a, sigma_b = float(p[6]), float(p[7])
    # Contrast against the section's own texture outside the fitted lumen,
    # not against what the fit leaves unexplained: a bright blob of speckle
    # fits a lumen well and leaves little, but stands only a fluctuation or
    # two above the texture it is part of.
    fitted = lumen_image(offsets, offsets, p[2], p[3], ra, rb, sigma_a, sigma_b, step)
    outside = valid & (fitted < 0.02)
    if int(outside.sum()) < _MIN_WINDOW_PIXELS:
        return None, "lumen_not_closed"
    texture = float(np.std(plane[outside]))
    contrast = float(p[1]) / texture if texture > 0 else float("inf")
    if max(ra, rb) + max(abs(float(p[2])), abs(float(p[3]))) > extent - 2.0 * max(sigma_a, sigma_b):
        return None, "lumen_not_closed"
    return SectionFit(
        diameter_um=2.0 * float(np.sqrt(ra * rb)),
        contrast=contrast,
        centre_offset_um=float(np.hypot(p[2], p[3])),
        aspect=max(ra, rb) / max(min(ra, rb), 1e-9),
        sigma_across_um=sigma_a,
        sigma_other_um=sigma_b,
        other_axis_z=abs(float(b[0])),
    ), "ok"


def _gate(fit: SectionFit, *, min_contrast: float, min_diameter_um: float) -> str:
    """Why *fit* is not a width of this vessel, or ``"ok"``."""
    if fit.contrast < min_contrast:
        return "low_contrast"
    # FWHM's own centring gate: 1.5 um, or 30% of the width for a wide vessel.
    if fit.centre_offset_um > max(1.5, 0.3 * fit.diameter_um):
        return "off_centre"
    if fit.aspect > _MAX_ASPECT:
        return "elongated"
    if fit.diameter_um < min_diameter_um:
        return "unresolved"
    return "ok"


def _guide_diameter(data: dict, guide_attribute: str | None, fallback_um: float) -> float:
    if guide_attribute:
        value = data.get(guide_attribute)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and np.isfinite(value) and value > 0:
            return value
    return float(fallback_um)


def estimate_psf_sigma(
    G: Any,
    raw: np.ndarray,
    voxel_size_zyx,
    *,
    vessel_mask: np.ndarray | None = None,
    guide_attribute: str | None = "edt_diameter_um",
    average_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
    min_contrast: float = RAW_SECTION_MIN_LUMEN_CONTRAST,
    seed: int = 0,
    edges: Iterable[tuple[Any, Any, Any]] | None = None,
) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
    """The image's own blur ``(sigma_z, sigma_y, sigma_x)`` in um, from its
    wide vessels -- among *edges* (``(u, v, key)``) when given -- or ``None``
    when too few of them show it.

    A wide lumen's width and its edges' blur are separately measurable, so
    each of up to :data:`PSF_CALIBRATION_MAX_SECTIONS` sections, at the
    middle of an edge at least :data:`PSF_CALIBRATION_MIN_GUIDE_UM` wide, is
    fitted with the blur free. The across axis lies in the y-x plane, so its
    blur is sigma_xy; the other axis's is ``sigma_xy^2 (1 - b_z^2) +
    sigma_z^2 b_z^2``, and a robust least-squares fit over all of them gives
    both. This is the blur the image shows -- the optics, the voxels and the
    interpolation together -- which is what the fit needs.
    """
    rng = np.random.default_rng(seed)
    chosen = edge_selection(edges)
    candidates = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if chosen is not None and (frozenset((u, v)), key) not in chosen:
            continue
        vox = data.get("voxels")
        if not vox or len(vox) < 2:
            continue
        if _guide_diameter(data, guide_attribute, 0.0) >= PSF_CALIBRATION_MIN_GUIDE_UM:
            candidates.append((u, v, key))
    order = rng.permutation(len(candidates))[:PSF_CALIBRATION_MAX_SECTIONS]
    fits: list[SectionFit] = []
    for index in order:
        data = G.edges[candidates[index]]
        poly = np.asarray(data["voxels"], dtype=float)
        s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            continue
        fit, _why = fit_section(
            raw, poly, s, total_len, 0.5 * total_len, voxel_size_zyx,
            guide_um=_guide_diameter(data, guide_attribute, 0.0),
            psf_sigma_zyx=None, vessel_mask=vessel_mask, average_um=average_um,
        )
        if fit is None or _gate(fit, min_contrast=min_contrast, min_diameter_um=0.0) != "ok":
            continue
        if not all(_SIGMA_BOUNDS_UM[0] * 1.01 < v < _SIGMA_BOUNDS_UM[1] * 0.99
                   for v in (fit.sigma_across_um, fit.sigma_other_um)):
            continue  # pinned at a bound: the fit could not see the blur
        fits.append(fit)
    details: dict[str, Any] = {"sections_tried": int(len(order)), "sections_fitted": len(fits)}
    axial = sum(1 for f in fits if f.other_axis_z >= _AXIAL_TILT)
    details["axial_sections"] = axial
    if len(fits) < PSF_CALIBRATION_MIN_FITS or axial < PSF_CALIBRATION_MIN_AXIAL_FITS:
        details["reason"] = (
            f"{len(fits)} wide-vessel sections showed their blur ({axial} tilted towards z); "
            f"at least {PSF_CALIBRATION_MIN_FITS} ({PSF_CALIBRATION_MIN_AXIAL_FITS}) are needed"
        )
        return None, details
    sigma_xy, sigma_z = psf_from_section_blurs(
        [f.sigma_across_um for f in fits],
        [f.sigma_other_um for f in fits],
        [f.other_axis_z for f in fits],
    )
    details.update(sigma_xy_um=sigma_xy, sigma_z_um=sigma_z)
    return (sigma_z, sigma_xy, sigma_xy), details


def measure_edge_diameters_from_raw_sections(
    G: Any,
    *,
    raw_volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    vessel_mask: np.ndarray | None = None,
    psf_sigma_zyx: tuple[float, float, float] | None = None,
    edges: Iterable[tuple[Any, Any, Any]] | None = None,
    sample_spacing_along_edge_um: float = 2.0,
    branch_endpoint_exclusion_um: float = 10.0,
    average_along_vessel_um: float = DEFAULT_AVERAGE_ALONG_VESSEL_UM,
    min_lumen_contrast: float = RAW_SECTION_MIN_LUMEN_CONTRAST,
    min_diameter_pixels: float = 2.0,
    aggregation: Literal["median", "mean"] = "median",
    guide_attribute: str | None = "edt_diameter_um",
    fallback_guide_um: float = 4.0,
    calibration_edges: Iterable[tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Measure *edges* (default every edge) from their raw cross-sections.

    Writes ``raw_section_diameter_um`` (the *aggregation* of the accepted
    readings), ``raw_section_diameter_samples_um``,
    ``raw_section_contrast_samples`` and ``raw_section_status``
    (``"measured"`` or ``"failed:<commonest reason>"``) on each edge it
    reads, and the PSF used in ``G.graph["raw_section_psf_sigma_zyx"]``.
    *psf_sigma_zyx* ``None`` estimates it (:func:`estimate_psf_sigma`, from
    *calibration_edges* when given); when that cannot, nothing is measured
    and the summary says why.

    A reading is kept when its lumen's contrast against the section's
    texture is at least *min_lumen_contrast*, its centre lies within FWHM's
    centring limit of the centreline, it is no more than 4:1 elongated, and
    it is at least *min_diameter_pixels* of the finest voxel side across; an
    edge needs :data:`MIN_ACCEPTED_READINGS` of them.
    """
    if sample_spacing_along_edge_um <= 0:
        raise ValueError("sample_spacing_along_edge_um must be positive.")
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    summary: dict[str, Any] = {"edges_measured": 0, "edges_skipped": [], "per_edge": [],
                               "readings_rejected": {}}
    if vessel_mask is not None and tuple(vessel_mask.shape) != tuple(raw_volume.shape):
        vessel_mask = None
    if psf_sigma_zyx is None:
        psf_sigma_zyx, calibration = estimate_psf_sigma(
            G, raw_volume, spacing, vessel_mask=vessel_mask, guide_attribute=guide_attribute,
            average_um=average_along_vessel_um, min_contrast=min_lumen_contrast,
            edges=calibration_edges,
        )
        summary["psf_calibration"] = calibration
        if psf_sigma_zyx is None:
            summary["skipped"] = True
            summary["reason"] = "could not estimate the image's blur: " + calibration["reason"]
            return summary
        G.graph["raw_section_psf_source"] = "estimated"
    else:
        psf_sigma_zyx = tuple(float(v) for v in psf_sigma_zyx)
        G.graph["raw_section_psf_source"] = "set"
    G.graph["raw_section_psf_sigma_zyx"] = tuple(float(v) for v in psf_sigma_zyx)
    summary["psf_sigma_zyx"] = G.graph["raw_section_psf_sigma_zyx"]
    min_diameter_um = max(0.0, float(min_diameter_pixels)) * float(np.min(spacing))
    rejected: Counter = Counter()

    chosen = list(G.edges(keys=True)) if edges is None else list(edges)
    for u, v, key in chosen:
        data = G.edges[u, v, key]
        for stale in ("raw_section_diameter_um", "raw_section_diameter_samples_um",
                      "raw_section_contrast_samples", "raw_section_status"):
            data.pop(stale, None)
        vox = data.get("voxels")
        if not vox or len(vox) < 2:
            summary["edges_skipped"].append((u, v, key, "no_voxels"))
            data["raw_section_status"] = "failed:no_voxels"
            continue
        poly = np.asarray(vox, dtype=float)
        s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            summary["edges_skipped"].append((u, v, key, "zero_length"))
            data["raw_section_status"] = "failed:zero_length"
            continue
        targets, _short = edge_sample_targets(
            total_len, sample_spacing_along_edge_um, branch_endpoint_exclusion_um,
            start_is_branch=int(G.degree(u)) > 1, end_is_branch=int(G.degree(v)) > 1,
        )
        if len(targets) > MAX_READINGS_PER_EDGE:
            targets = targets[np.linspace(0, len(targets) - 1, MAX_READINGS_PER_EDGE).round().astype(int)]
        guide = _guide_diameter(data, guide_attribute, fallback_guide_um)
        diameters: list[float] = []
        contrasts: list[float] = []
        reasons: Counter = Counter()
        for at in targets:
            fit, why = fit_section(
                raw_volume, poly, s, total_len, float(at), spacing, guide_um=guide,
                psf_sigma_zyx=psf_sigma_zyx, vessel_mask=vessel_mask,
                average_um=average_along_vessel_um,
            )
            if fit is not None:
                why = _gate(fit, min_contrast=min_lumen_contrast, min_diameter_um=min_diameter_um)
            if why != "ok":
                reasons[why] += 1
                continue
            diameters.append(fit.diameter_um)
            contrasts.append(fit.contrast)
        rejected.update(reasons)
        if len(diameters) < MIN_ACCEPTED_READINGS:
            reason = reasons.most_common(1)[0][0] if reasons else "too_few_readings"
            if diameters:
                reason = "too_few_readings"
            summary["edges_skipped"].append((u, v, key, reason))
            data["raw_section_status"] = f"failed:{reason}"
            continue
        d_final = _aggregate_edge_diameter(diameters, aggregation)
        data["raw_section_diameter_um"] = d_final
        data["raw_section_diameter_samples_um"] = diameters
        data["raw_section_contrast_samples"] = contrasts
        data["raw_section_status"] = "measured"
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {"edge": (u, v, key), "raw_section_diameter_um": d_final, "n_samples": len(diameters)}
        )
    summary["readings_rejected"] = dict(rejected)
    return summary
