"""Automated vessel diameter estimation from raw TIFF intensity (transverse profiles).

At each sample along an edge, intensity along the perpendicular line is fit with a Gaussian
plus baseline; vessel size is reported as the Gaussian **FWHM**,
``2 * sqrt(2 ln 2) * σ`` (micrometers). By default the baseline seed uses **outer wings**
of the profile (see ``robust_baseline_from_profile_wings``) to reduce bias from a
neighbour-induced shoulder on one side.

Branch identity for clipping comes from an in-memory label volume rasterized from the graph
(see ``build_graph_branch_label_volume``). Transverse extent follows the configured minimum
relative to the current FWHM estimate unless truncated at another edge or volume bounds.

Transverse profiles are sampled **in the physical x–y plane only** (no displacement along
physical ``z``), so diameter rays follow in-plane directions when voxel spacing
is anisotropic with coarser ``z``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import networkx as nx
import tifffile
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

from ImageLynx.coords import (
    physical_xyz_to_continuous_index_zyx,
    physical_xyz_delta_to_index_zyx_delta,
)

# FWHM of a Gaussian with standard deviation sigma (not 2*sigma^2 in the exponent).
_GAUSSIAN_FWHM_FROM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_DEFAULT_DIAMETER_BOUNDS_BY_CLASS_UM: dict[str, tuple[float, float]] = {
    "capillary": (2.0, 15.0),
    "arteriole": (5.0, 80.0),
    "venule": (5.0, 120.0),
    "default": (1.0, 150.0),
}


def _classify_branch_order(branch_order: Any) -> str:
    """Map branch-order labels to vessel classes used by diameter bounds."""
    if branch_order is None:
        return "default"
    label = str(branch_order).strip().lower()
    if label.startswith("b"):
        return "capillary"
    if label.startswith("art"):
        return "arteriole"
    if label.startswith("ven"):
        return "venule"
    return "default"


def _resolve_diameter_bounds_for_branch_order(
    branch_order: Any,
    bounds_by_class_um: dict[str, tuple[float, float]] | None,
) -> tuple[float, float] | None:
    """Return (lo, hi) diameter bounds for the given branch-order label."""
    if bounds_by_class_um is None:
        return None
    vessel_class = _classify_branch_order(branch_order)
    candidate = bounds_by_class_um.get(vessel_class, bounds_by_class_um.get("default"))
    if candidate is None:
        return None
    lo, hi = float(candidate[0]), float(candidate[1])
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(
            f"Invalid diameter bounds for class '{vessel_class}': ({lo}, {hi})."
        )
    return lo, hi


def _apply_diameter_bounds(
    diameter_um: float | None,
    bounds_um: tuple[float, float] | None,
    mode: Literal["off", "reject", "clamp"],
) -> tuple[float | None, bool]:
    """Apply optional bounds to a diameter, returning (value, touched)."""
    if diameter_um is None or bounds_um is None or mode == "off":
        return diameter_um, False
    lo, hi = float(bounds_um[0]), float(bounds_um[1])
    d = float(diameter_um)
    if lo <= d <= hi:
        return d, False
    if mode == "reject":
        return None, True
    return float(np.clip(d, lo, hi)), True


def load_single_channel_tiff_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF as float32. Single channel / single signal expected."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")
    vol = tifffile.imread(str(path))
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(
            f"Expected 2D slice stack or 3D volume, got shape {vol.shape} for {path}"
        )
    return np.asarray(vol, dtype=np.float32)


def _spacing_vec(voxel_size_xyz: tuple[float, float, float]) -> np.ndarray:
    s = np.asarray(voxel_size_xyz, dtype=float).ravel()
    if s.size != 3:
        raise ValueError("voxel_size_xyz must have length 3")
    return s


def physical_points_to_continuous_indices(
    points_phys: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Convert physical (x,y,z) coordinates to continuous voxel indices (z,y,x)."""
    spacing = _spacing_vec(voxel_size_xyz)
    if np.any(spacing <= 0):
        raise ValueError("voxel_size_xyz components must be positive.")
    return physical_xyz_to_continuous_index_zyx(points_phys, voxel_size_xyz)


def _gram_schmidt_perpendicular(tangent: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``tangent`` in full 3D (unused for FWHM profiles).

    Transverse intensity profiles use ``_transverse_unit_in_physical_yx_plane`` so rays stay
    in the slice (y–x) plane when ``z`` spacing is coarse.
    """
    t = np.asarray(tangent, dtype=float).ravel()
    nrm = np.linalg.norm(t)
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    t = t / nrm
    # Pick a reference axis least parallel to t
    refs = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    best = max(refs, key=lambda r: abs(float(np.dot(r, t))))
    n = best - np.dot(best, t) * t
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        n = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), t) * t
        nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return (n / nn).astype(float)


def _transverse_unit_in_physical_yx_plane(tangent: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``tangent`` in the physical x-y plane.

    Coordinates are physical ``(x, y, z)``. The returned direction keeps ``z`` fixed
    (i.e., varies only ``x`` and ``y``).
    """
    t = np.asarray(tangent, dtype=float).ravel()
    if t.size != 3:
        raise ValueError("tangent must have length 3 (x, y, z).")
    tx, ty = float(t[0]), float(t[1])
    n2 = float(np.hypot(tx, ty))
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return np.array([-ty / n2, tx / n2, 0.0], dtype=float)


def _arc_length_parameterize(poly_phys: np.ndarray) -> tuple[np.ndarray, float]:
    """Cumulative arc length along polyline (micrometers)."""
    p = np.asarray(poly_phys, dtype=float)
    if len(p) < 2:
        return np.array([0.0], dtype=float), 0.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, float(s[-1])


def _interpolate_centerline(poly_phys: np.ndarray, s: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate 3D positions at arc-length values ``targets``."""
    p = np.asarray(poly_phys, dtype=float)
    out = np.empty((len(targets), 3), dtype=float)
    for i, t in enumerate(targets):
        out[i] = np.array(
            [float(np.interp(t, s, p[:, j])) for j in range(3)],
            dtype=float,
        )
    return out


def _tangent_at(poly_phys: np.ndarray, s: np.ndarray, s0: float) -> np.ndarray:
    """Central-difference tangent on the polyline at arc length ``s0``."""
    p = np.asarray(poly_phys, dtype=float)
    if len(p) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if s0 <= s[0] + 1e-9:
        return p[1] - p[0]
    if s0 >= s[-1] - 1e-9:
        return p[-1] - p[-2]
    idx = int(np.searchsorted(s, s0, side="right") - 1)
    idx = max(0, min(idx, len(p) - 2))
    ds = s[idx + 1] - s[idx]
    if ds < 1e-12:
        return p[idx + 1] - p[idx]
    alpha = (s0 - s[idx]) / ds
    t0 = p[idx + 1] - p[idx]
    if idx > 0:
        t0 = (1 - alpha) * (p[idx + 1] - p[idx]) + alpha * (p[idx] - p[idx - 1])
    elif idx + 2 < len(p):
        t0 = p[idx + 2] - p[idx]
    return t0.astype(float)


def _nearest_integer_index(pt_idx: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int]:
    iz, iy, ix = (int(round(float(pt_idx[0]))), int(round(float(pt_idx[1]))), int(round(float(pt_idx[2]))))
    iz = int(np.clip(iz, 0, shape[0] - 1))
    iy = int(np.clip(iy, 0, shape[1] - 1))
    ix = int(np.clip(ix, 0, shape[2] - 1))
    return iz, iy, ix


def _label_at(
    labels: np.ndarray,
    pt_idx: np.ndarray,
    shape: tuple[int, ...],
) -> int:
    iz, iy, ix = _nearest_integer_index(pt_idx, shape)
    return int(labels[iz, iy, ix])


def _max_extent_along_ray(
    center_idx: np.ndarray,
    direction_unit: np.ndarray,
    assigned_label: int,
    labels: np.ndarray,
    max_physical_extent: float,
    voxel_size_xyz: tuple[float, float, float],
    step_um: float,
    *,
    background_label: int,
    junction_label: int | None,
    allow_junction_crossing: bool,
    same_edge_s_lookup: dict[tuple[int, int, int], float] | None = None,
    same_edge_s0_um: float | None = None,
    same_edge_arc_window_um: float | None = None,
) -> float:
    """Positive distance along +direction until hitting another edge or the volume edge.

    Voxels labeled ``assigned_label`` or ``background_label`` (unpainted lumen) allow
    continuation up to ``max_physical_extent``. By default, ``junction_label`` is
    treated as a hard stop to avoid crossing into neighbouring branches at bifurcations.
    If ``same_edge_s_lookup`` + ``same_edge_s0_um`` + ``same_edge_arc_window_um`` are
    provided, same-edge traversal is additionally restricted to local arc-length
    neighbourhood around the current sample to avoid zig-zag self-intersections.
    Any other positive label is treated as a different graph edge and truncates the line.
    """
    if step_um <= 0:
        raise ValueError("step_um must be positive.")
    nrm = float(np.linalg.norm(direction_unit))
    if nrm <= 1e-12:
        return 0.0
    d = np.asarray(direction_unit, dtype=float) / nrm
    n_steps = int(np.ceil(max_physical_extent / step_um))
    shape = labels.shape
    delta_idx_step = physical_xyz_delta_to_index_zyx_delta(d * float(step_um), voxel_size_xyz)
    for k in range(1, n_steps + 1):
        idx = center_idx + float(k) * delta_idx_step
        if (
            idx[0] < 0
            or idx[1] < 0
            or idx[2] < 0
            or idx[0] > shape[0] - 1
            or idx[1] > shape[1] - 1
            or idx[2] > shape[2] - 1
        ):
            return max(0.0, (k - 1) * step_um)
        lab = _label_at(labels, idx, shape)
        if lab == assigned_label:
            if (
                same_edge_s_lookup is not None
                and same_edge_s0_um is not None
                and same_edge_arc_window_um is not None
                and same_edge_arc_window_um > 0
            ):
                key = _nearest_integer_index(idx, shape)
                s_here = same_edge_s_lookup.get(key)
                if s_here is not None and abs(float(s_here) - float(same_edge_s0_um)) > float(
                    same_edge_arc_window_um
                ):
                    return max(0.0, (k - 1) * step_um)
            continue
        if lab == background_label:
            continue
        if junction_label is not None and lab == junction_label:
            if allow_junction_crossing:
                continue
            return max(0.0, (k - 1) * step_um)
        return max(0.0, (k - 1) * step_um)
    return max_physical_extent


def _sample_transverse_profile(
    raw: np.ndarray,
    labels: np.ndarray,
    center_phys: np.ndarray,
    tangent: np.ndarray,
    assigned_label: int,
    half_extent_um: float,
    transverse_step_um: float,
    voxel_size_xyz: tuple[float, float, float],
    *,
    background_label: int,
    junction_label: int | None,
    center_idx: np.ndarray | None = None,
    allow_junction_crossing: bool = False,
    same_edge_s_lookup: dict[tuple[int, int, int], float] | None = None,
    same_edge_s0_um: float | None = None,
    same_edge_arc_window_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample intensity along a line through ``center_phys``, perpendicular to ``tangent``.

    The line lies in the physical x–y plane (fixed ``z``); see
    ``_transverse_unit_in_physical_yx_plane``.

    Returns (positions_along_line_um, intensities).
    """
    n_hat = _transverse_unit_in_physical_yx_plane(tangent)
    c_idx = (
        physical_points_to_continuous_indices(center_phys, voxel_size_xyz)
        if center_idx is None
        else np.asarray(center_idx, dtype=float).ravel()
    )

    pos_plus = _max_extent_along_ray(
        c_idx,
        n_hat,
        assigned_label,
        labels,
        half_extent_um,
        voxel_size_xyz,
        transverse_step_um,
        background_label=background_label,
        junction_label=junction_label,
        allow_junction_crossing=allow_junction_crossing,
        same_edge_s_lookup=same_edge_s_lookup,
        same_edge_s0_um=same_edge_s0_um,
        same_edge_arc_window_um=same_edge_arc_window_um,
    )
    pos_minus = _max_extent_along_ray(
        c_idx,
        -n_hat,
        assigned_label,
        labels,
        half_extent_um,
        voxel_size_xyz,
        transverse_step_um,
        background_label=background_label,
        junction_label=junction_label,
        allow_junction_crossing=allow_junction_crossing,
        same_edge_s_lookup=same_edge_s_lookup,
        same_edge_s0_um=same_edge_s0_um,
        same_edge_arc_window_um=same_edge_arc_window_um,
    )

    n_neg = int(np.floor(pos_minus / transverse_step_um))
    n_pos = int(np.floor(pos_plus / transverse_step_um))
    offsets = np.arange(-n_neg, n_pos + 1, dtype=float) * transverse_step_um
    if offsets.size == 0:
        offsets = np.array([0.0], dtype=float)

    delta_idx_per_um = physical_xyz_delta_to_index_zyx_delta(n_hat, voxel_size_xyz)
    coord_arr = c_idx[:, None] + delta_idx_per_um[:, None] * offsets[None, :]
    zc, yc, xc = coord_arr[0], coord_arr[1], coord_arr[2]
    vals = map_coordinates(
        raw,
        np.vstack([zc, yc, xc]),
        order=1,
        mode="constant",
        cval=0.0,
    )
    return offsets, np.asarray(vals, dtype=float)


def _gaussian_fluorescence_1d(
    x: np.ndarray,
    baseline: float,
    amplitude: float,
    x0: float,
    sigma: float,
) -> np.ndarray:
    """baseline + amplitude * exp(-(x - x0)^2 / (2 sigma^2))."""
    sig = max(float(sigma), 1e-15)
    return baseline + amplitude * np.exp(-0.5 * ((x - float(x0)) / sig) ** 2)


def robust_baseline_from_profile_wings(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    wing_fraction: float = 0.2,
) -> float:
    """Baseline guess from outer ``wing_fraction`` of the line (by position).

    Uses the **minimum** of the left-wing and right-wing intensity medians so a
    neighbour-induced shoulder on **one** side does not raise the whole baseline.
    """
    if wing_fraction <= 0.0 or wing_fraction >= 0.5:
        raise ValueError("wing_fraction must be in (0, 0.5).")
    x = np.asarray(x_sorted, dtype=float).ravel()
    y = np.asarray(y_sorted, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty profile")
    x_min = float(x[0])
    x_max = float(x[-1])
    span = x_max - x_min
    if span <= 0:
        return float(np.median(y))
    lo_cut = x_min + wing_fraction * span
    hi_cut = x_max - wing_fraction * span
    left = y[x <= lo_cut]
    right = y[x >= hi_cut]
    if left.size == 0 and right.size == 0:
        return float(np.median(y))
    if left.size == 0:
        return float(np.median(right))
    if right.size == 0:
        return float(np.median(left))
    return float(min(np.median(left), np.median(right)))


def fwhm_from_profile(
    positions_um: np.ndarray,
    intensities: np.ndarray,
    *,
    min_points: int = 5,
    profile_baseline_mode: Literal["wings", "percentile"] = "wings",
    profile_baseline_wing_fraction: float = 0.2,
    constrain_fitted_baseline: bool = False,
    baseline_constraint_half_width_ptp: float = 0.35,
) -> float | None:
    """FWHM (µm) from a least-squares Gaussian + baseline fit to the 1D intensity profile.

    The model is ``baseline + amplitude * exp(-(x - x0)^2 / (2 sigma^2))`` with
    ``amplitude > 0``. Returns ``FWHM = 2 * sqrt(2 ln 2) * sigma``.

    Parameters
    ----------
    profile_baseline_mode :
        ``wings`` (default): initial baseline from ``robust_baseline_from_profile_wings``,
        which reduces bias when a neighbour adds a shoulder on one side of the profile.
        ``percentile``: legacy initial guess via the 10th percentile of all samples.
    profile_baseline_wing_fraction :
        Fraction of line length (each end) used as wings when ``mode="wings"``.
    constrain_fitted_baseline :
        If True, restrict the fitted baseline to a band around the wing (or percentile)
        guess so the optimiser cannot absorb a shoulder mostly into a higher baseline.
    baseline_constraint_half_width_ptp :
        Half-width of that band as a fraction of peak-to-peak intensity (only if
        ``constrain_fitted_baseline`` is True).
    """
    x = np.asarray(positions_um, dtype=float).ravel()
    y = np.asarray(intensities, dtype=float).ravel()
    if x.size < min_points or y.size != x.size:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    span = float(np.ptp(x))
    if span <= 0 or not np.isfinite(span):
        return None

    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else span
    sigma_min = max(0.25 * dx, span * 1e-6, 1e-9)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_ptp = max(y_max - y_min, 1e-12)

    if profile_baseline_mode == "wings":
        try:
            b_anchor = robust_baseline_from_profile_wings(
                x, y, wing_fraction=profile_baseline_wing_fraction
            )
        except ValueError:
            b_anchor = float(np.percentile(y, 10))
    elif profile_baseline_mode == "percentile":
        b_anchor = float(np.percentile(y, 10))
    else:
        raise ValueError(
            f"Unknown profile_baseline_mode={profile_baseline_mode!r}; "
            "use 'wings' or 'percentile'."
        )

    b0 = min(max(b_anchor, y_min - y_ptp), y_max - 0.01 * y_ptp)
    amp0 = max(y_max - b0, y_ptp * 0.5, 1e-9)
    x0_guess = float(x[int(np.argmax(y))])
    sig0 = max(span / 5.0, sigma_min)

    p0 = np.array([b0, amp0, x0_guess, sig0], dtype=float)
    half_w = float(baseline_constraint_half_width_ptp) * y_ptp
    if constrain_fitted_baseline and half_w > 0:
        b_lo = max(y_min - 0.5 * y_ptp, b_anchor - half_w)
        b_hi = min(y_max + 0.5 * y_ptp, b_anchor + half_w)
        if b_lo >= b_hi:
            b_lo, b_hi = y_min - 2.0 * y_ptp, y_max + 2.0 * y_ptp
    else:
        b_lo = y_min - 5.0 * y_ptp
        b_hi = y_max + 5.0 * y_ptp
    lo = np.array([b_lo, 1e-12, np.min(x) - span, sigma_min], dtype=float)
    hi = np.array([b_hi, max(y_max * 20.0, amp0 * 1e3), np.max(x) + span, span], dtype=float)

    try:
        popt, _ = curve_fit(
            _gaussian_fluorescence_1d,
            x,
            y,
            p0=p0,
            bounds=(lo, hi),
            maxfev=50000,
        )
    except (RuntimeError, ValueError):
        return None

    baseline_fit, amplitude_fit, _, sigma_fit = (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
    if not np.isfinite(sigma_fit) or sigma_fit <= 0:
        return None
    if amplitude_fit <= 0:
        return None

    fwhm = float(_GAUSSIAN_FWHM_FROM_SIGMA * sigma_fit)
    return fwhm if fwhm > 0 else None


def _fwhm_gaussian_fit_with_diagnostics(
    positions_um: np.ndarray,
    intensities: np.ndarray,
    *,
    min_points: int = 5,
    profile_baseline_mode: Literal["wings", "percentile"] = "wings",
    profile_baseline_wing_fraction: float = 0.2,
    constrain_fitted_baseline: bool = False,
    baseline_constraint_half_width_ptp: float = 0.35,
) -> tuple[float | None, float | None, float | None]:
    """Return (fwhm_um, fitted_center_um, fit_r2) using same model as ``fwhm_from_profile``."""
    x = np.asarray(positions_um, dtype=float).ravel()
    y = np.asarray(intensities, dtype=float).ravel()
    if x.size != y.size or x.size < max(2, int(min_points)):
        return None, None, None
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < max(2, int(min_points)):
        return None, None, None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(np.ptp(x))
    if span <= 0 or not np.isfinite(span):
        return None, None, None
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else span
    sigma_min = max(0.25 * dx, span * 1e-6, 1e-9)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_ptp = max(y_max - y_min, 1e-12)
    if profile_baseline_mode == "wings":
        try:
            b_anchor = robust_baseline_from_profile_wings(
                x, y, wing_fraction=profile_baseline_wing_fraction
            )
        except ValueError:
            b_anchor = float(np.percentile(y, 10))
    elif profile_baseline_mode == "percentile":
        b_anchor = float(np.percentile(y, 10))
    else:
        return None, None, None
    b0 = min(max(b_anchor, y_min - y_ptp), y_max - 0.01 * y_ptp)
    amp0 = max(y_max - b0, y_ptp * 0.5, 1e-9)
    x0_guess = float(x[int(np.argmax(y))])
    sig0 = max(span / 5.0, sigma_min)
    p0 = np.array([b0, amp0, x0_guess, sig0], dtype=float)
    half_w = float(baseline_constraint_half_width_ptp) * y_ptp
    if constrain_fitted_baseline and half_w > 0:
        b_lo = max(y_min - 0.5 * y_ptp, b_anchor - half_w)
        b_hi = min(y_max + 0.5 * y_ptp, b_anchor + half_w)
        if b_lo >= b_hi:
            b_lo, b_hi = y_min - 2.0 * y_ptp, y_max + 2.0 * y_ptp
    else:
        b_lo = y_min - 5.0 * y_ptp
        b_hi = y_max + 5.0 * y_ptp
    lo = np.array([b_lo, 1e-12, np.min(x) - span, sigma_min], dtype=float)
    hi = np.array([b_hi, max(y_max * 20.0, amp0 * 1e3), np.max(x) + span, span], dtype=float)
    try:
        popt, _ = curve_fit(
            _gaussian_fluorescence_1d,
            x,
            y,
            p0=p0,
            bounds=(lo, hi),
            maxfev=50000,
        )
    except (RuntimeError, ValueError):
        return None, None, None
    baseline_fit, amplitude_fit, x0_fit, sigma_fit = (
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        float(popt[3]),
    )
    if not np.isfinite(sigma_fit) or sigma_fit <= 0 or amplitude_fit <= 0:
        return None, None, None
    y_hat = _gaussian_fluorescence_1d(x, baseline_fit, amplitude_fit, x0_fit, sigma_fit)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    fwhm = float(_GAUSSIAN_FWHM_FROM_SIGMA * sigma_fit)
    if fwhm <= 0:
        return None, None, None
    return fwhm, x0_fit, r2


def _clip_profile_to_central_lobe(
    positions_um: np.ndarray,
    intensities: np.ndarray,
    *,
    min_drop_fraction_of_center: float = 0.35,
    re_rise_fraction_of_center: float = 0.08,
    min_points_to_clip: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a transverse profile so it stays on the central vessel lobe.

    Profiles can contain a second peak when rays reach a neighbouring branch.
    This helper keeps the segment around offset 0 and truncates each side at the
    first clear valley->rise pattern.
    """
    x = np.asarray(positions_um, dtype=float).ravel()
    y = np.asarray(intensities, dtype=float).ravel()
    if x.size != y.size or x.size < int(min_points_to_clip):
        return x, y
    # Positions from _sample_transverse_profile are monotone and include offset 0.
    i0 = int(np.argmin(np.abs(x)))
    if i0 <= 0 or i0 >= x.size - 1:
        return x, y
    center = float(y[i0])
    if not np.isfinite(center):
        return x, y
    min_drop = float(min_drop_fraction_of_center) * max(center, 1e-12)
    rise_thr = float(re_rise_fraction_of_center) * max(center, 1e-12)

    def _bound_right(start: int) -> int:
        y_min = float(y[start])
        i_min = start
        saw_drop = False
        for i in range(start + 1, y.size):
            yi = float(y[i])
            if yi < y_min:
                y_min = yi
                i_min = i
            if yi <= (center - min_drop):
                saw_drop = True
            if saw_drop and yi >= (y_min + rise_thr):
                return i_min
        return y.size - 1

    def _bound_left(start: int) -> int:
        y_min = float(y[start])
        i_min = start
        saw_drop = False
        for i in range(start - 1, -1, -1):
            yi = float(y[i])
            if yi < y_min:
                y_min = yi
                i_min = i
            if yi <= (center - min_drop):
                saw_drop = True
            if saw_drop and yi >= (y_min + rise_thr):
                return i_min
        return 0

    left = _bound_left(i0)
    right = _bound_right(i0)

    # Additional fallback for tortuous/zig-zag vessels: use the nearest local minima
    # around offset 0 as hard lobe boundaries if available.
    left_min = None
    for i in range(i0 - 1, 0, -1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            left_min = i
            break
    right_min = None
    for i in range(i0 + 1, y.size - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            right_min = i
            break
    if left_min is not None and right_min is not None and (right_min - left_min + 1) >= 5:
        left = max(left, int(left_min))
        right = min(right, int(right_min))

    if right - left + 1 < 5:
        return x, y
    return x[left : right + 1], y[left : right + 1]


def build_graph_branch_label_volume(
    G: nx.MultiGraph,
    volume_shape: tuple[int, int, int],
    voxel_size_xyz: tuple[float, float, float],
    *,
    background_label: int = 0,
    junction_label: int = -1,
) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    """Paint each edge's ``voxels`` path into a 3D label array (same shape as the raw stack).

    Edges are processed in deterministic ``(u, v, key)`` order and receive ids ``1 .. E``.
    Voxels claimed by more than one edge are set to ``junction_label`` so transverse FWHM
    profiles can cross node regions without immediately hitting another edge's id.

    Each edge dict is updated with ``graph_edge_label_id`` (and ``image_branch_label``
    as an alias for the same value).

    Returns
    -------
    labels :
        ``int32`` array of shape ``volume_shape``.
    edge_key_to_label :
        Map ``(u, v, key) -> int`` label id.
    """
    if junction_label == background_label:
        raise ValueError("junction_label must differ from background_label.")
    shape = (int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2]))
    labels = np.full(shape, int(background_label), dtype=np.int32)
    sorted_edges = sorted(G.edges(keys=True), key=lambda t: (t[0], t[1], t[2]))
    edge_key_to_label: dict[tuple[int, int, int], int] = {}
    for i, (u, v, key) in enumerate(sorted_edges, start=1):
        edge_key_to_label[(u, v, key)] = i
        G[u][v][key]["graph_edge_label_id"] = i
        G[u][v][key]["image_branch_label"] = i

    jlab = int(junction_label)
    bg = int(background_label)

    for u, v, key in sorted_edges:
        label = edge_key_to_label[(u, v, key)]
        vox = G[u][v][key].get("voxels")
        if not vox:
            continue
        idx_all = physical_points_to_continuous_indices(
            np.asarray(vox, dtype=float),
            voxel_size_xyz,
        )
        for row in idx_all:
            iz, iy, ix = _nearest_integer_index(row, shape)
            cur = int(labels[iz, iy, ix])
            if cur == bg:
                labels[iz, iy, ix] = int(label)
            elif cur == int(label):
                continue
            elif cur == jlab:
                continue
            else:
                labels[iz, iy, ix] = jlab

    return labels, edge_key_to_label


def measure_edge_diameters_fwhm_from_raw_tiff(
    G: nx.MultiGraph,
    *,
    raw_tiff_path: str | Path,
    voxel_size_xyz: tuple[float, float, float],
    sample_spacing_along_edge_um: float,
    transverse_profile_step_um: float,
    transverse_half_extent_um: float,
    diameter_guess_um: float | None = None,
    background_label: int = 0,
    junction_label: int = -1,
    min_total_extent_multiplier: float = 3.0,
    profile_baseline_mode: Literal["wings", "percentile"] = "wings",
    profile_baseline_wing_fraction: float = 0.2,
    constrain_fitted_baseline: bool = False,
    baseline_constraint_half_width_ptp: float = 0.35,
    allow_junction_crossing: bool = False,
    clip_profile_to_single_vessel: bool = True,
    clip_min_drop_fraction_of_center: float = 0.35,
    clip_re_rise_fraction_of_center: float = 0.08,
    branch_endpoint_exclusion_um: float = 0.0,
    junction_proximity_exclusion_um: float = 0.0,
    store_profile_debug: bool = False,
    enforce_same_edge_locality: bool = True,
    same_edge_arc_window_um: float | None = None,
    same_edge_arc_window_multiplier: float = 1.0,
    same_edge_arc_window_min_um: float = 1.0,
    cap_half_extent_by_nonlocal_same_edge_distance: bool = True,
    nonlocal_same_edge_arc_separation_um: float = 6.0,
    nonlocal_same_edge_half_extent_factor: float = 0.45,
    reject_samples_with_center_offset: bool = True,
    max_fit_center_offset_um: float = 1.5,
    reject_samples_with_low_fit_r2: bool = True,
    min_fit_r2: float = 0.85,
    edge_parallel_workers: int | None = None,
    edge_parallel_batch_size: int = 16,
    diameter_bounds_by_vessel_class_um: dict[str, tuple[float, float]] | None = None,
    diameter_bounds_mode: Literal["off", "reject", "clamp"] = "reject",
) -> dict[str, Any]:
    """Measure per-edge diameters (µm) from a raw TIFF using graph-derived branch labels.

    Diameter at each transverse sample is the FWHM of a Gaussian least-squares fit to
    intensity along the line (see ``fwhm_from_profile``).

    A label volume is built from edge ``voxels`` (see ``build_graph_branch_label_volume``);
    no separate label TIFF is required.

    Parameters
    ----------
    G :
        MultiGraph with ``voxels`` edge attribute (physical coordinates, same axis order
        as the raw TIFF).
    raw_tiff_path :
        Single-channel 3D TIFF (intensity); shape defines the rasterized label grid.
    voxel_size_xyz :
        Spacing per image axis (same tuple passed to graph building).
    sample_spacing_along_edge_um :
        Distance along the edge centerline between FWHM samples.
    transverse_profile_step_um :
        Step size along the transverse line (sampling resolution along the profile).
    transverse_half_extent_um :
        Initial half-length of the transverse line (µm). After a first FWHM estimate,
        the half-extent is enlarged to at least ``min_total_extent_multiplier / 2``
        times that diameter unless truncated earlier.
    diameter_guess_um :
        Optional initial width guess (µm). If None, initial half-extent uses
        ``transverse_half_extent_um`` only.
    junction_label :
        Reserved value marking voxels shared by multiple edges in the rasterized volume.
        Transverse rays stop at this label by default to avoid crossing onto neighbouring
        vessel branches near bifurcations. Set ``allow_junction_crossing=True`` to permit
        traversal through junction-labeled voxels.
    min_total_extent_multiplier :
        Ensures total transverse extent >= this factor × measured FWHM when not truncated.
    profile_baseline_mode :
        ``wings`` (default) or ``percentile`` — see ``fwhm_from_profile``.
    profile_baseline_wing_fraction :
        Outer line fraction used per end for wing baseline (if mode is ``wings``).
    constrain_fitted_baseline :
        If True, narrow the fitted-baseline bounds around the anchor guess.
    baseline_constraint_half_width_ptp :
        Half-width of that band as a fraction of profile peak-to-peak intensity.
    clip_profile_to_single_vessel :
        If True, trim each sampled transverse profile to the central lobe before
        Gaussian fitting so a second peak from a neighbouring vessel does not
        inflate diameter.
    branch_endpoint_exclusion_um :
        Distance (µm) excluded from sampling near edge endpoints that are
        bifurcation nodes (graph degree > 1). Helps avoid unstable diameters
        right where a branch emerges from a junction.
    junction_proximity_exclusion_um :
        Distance (µm) excluded from sampling around automatically detected
        vessel meeting points along each edge. Detection uses the in-memory
        rasterized label volume: centerline points whose nearest voxel is
        ``junction_label`` are treated as meeting points.
    store_profile_debug :
        If True, store accepted transverse profile polylines per edge in
        ``edge['fwhm_profile_lines_phys']`` for visualization/debugging.
    enforce_same_edge_locality :
        If True, stop transverse rays when they re-enter the same edge label at
        centerline arc-length far from the current sample (helps zig-zag vessels).
    same_edge_arc_window_um :
        Optional fixed arc-length window for same-edge locality guard.
    same_edge_arc_window_multiplier :
        If ``same_edge_arc_window_um`` is None, window = max(min, multiplier × current
        diameter estimate).
    same_edge_arc_window_min_um :
        Lower bound for that adaptive window.
    cap_half_extent_by_nonlocal_same_edge_distance :
        If True, cap each sample's transverse half-extent using the nearest
        non-local point on the same edge centerline (geometry-only guard).
    nonlocal_same_edge_arc_separation_um :
        Arc-length separation defining "non-local" centerline points for the
        above cap.
    nonlocal_same_edge_half_extent_factor :
        Half-extent cap = factor × nearest non-local centerline distance.
    reject_samples_with_center_offset :
        If True, discard samples whose fitted Gaussian center is far from the
        intended profile origin (offset 0), indicating oblique/nonlocal transect.
    max_fit_center_offset_um :
        Maximum allowed absolute fitted center offset from 0 for accepted samples.
    reject_samples_with_low_fit_r2 :
        If True, discard samples with poor Gaussian fit quality.
    min_fit_r2 :
        Minimum accepted R² for Gaussian fit.
    edge_parallel_workers :
        Number of worker threads for edge-level FWHM fitting. ``None``/``<=1``
        runs sequentially.
    edge_parallel_batch_size :
        Number of edges processed per submitted worker task.
    diameter_bounds_by_vessel_class_um :
        Optional bounds by vessel class (capillary/arteriole/venule/default).
    diameter_bounds_mode :
        ``off``, ``reject``, or ``clamp`` for applying branch-order class bounds.
    """
    if profile_baseline_mode not in ("wings", "percentile"):
        raise ValueError(
            f"profile_baseline_mode must be 'wings' or 'percentile', got {profile_baseline_mode!r}."
        )

    raw = load_single_channel_tiff_volume(raw_tiff_path)
    labels, _ = build_graph_branch_label_volume(
        G,
        raw.shape,
        voxel_size_xyz,
        background_label=background_label,
        junction_label=junction_label,
    )

    summary: dict[str, Any] = {
        "edges_measured": 0,
        "edges_skipped": [],
        "per_edge": [],
        "bounds_rejected_samples": 0,
        "bounds_clamped_samples": 0,
        "bounds_rejected_edges": 0,
        "bounds_clamped_edges": 0,
    }

    mult = float(min_total_extent_multiplier)
    if mult < 1.0:
        raise ValueError("min_total_extent_multiplier must be >= 1.")

    jn = int(junction_label)
    branch_excl = max(0.0, float(branch_endpoint_exclusion_um))
    junction_excl = max(0.0, float(junction_proximity_exclusion_um))
    d_guess0 = 0.0 if diameter_guess_um is None else max(0.0, float(diameter_guess_um))

    if sample_spacing_along_edge_um <= 0:
        raise ValueError("sample_spacing_along_edge_um must be positive.")
    if edge_parallel_batch_size <= 0:
        raise ValueError("edge_parallel_batch_size must be >= 1.")
    if diameter_bounds_mode not in ("off", "reject", "clamp"):
        raise ValueError(
            f"diameter_bounds_mode must be 'off', 'reject', or 'clamp', got {diameter_bounds_mode!r}."
        )
    bounds_map = (
        dict(diameter_bounds_by_vessel_class_um)
        if diameter_bounds_by_vessel_class_um is not None
        else dict(_DEFAULT_DIAMETER_BOUNDS_BY_CLASS_UM)
    )

    degree_map = {node_id: int(G.degree(node_id)) for node_id in G.nodes()}

    def _measure_single_edge(
        u: Any,
        v: Any,
        key: Any,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        vox = data.get("voxels")
        assigned = data.get("graph_edge_label_id")
        if not vox or len(vox) < 2 or assigned is None:
            return {"edge": (u, v, key), "skip_reason": "no_voxels_or_label"}

        poly = np.asarray(vox, dtype=float)
        s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            return {"edge": (u, v, key), "skip_reason": "zero_length"}

        n_samples = max(1, int(np.floor(total_len / sample_spacing_along_edge_um)) + 1)
        targets = np.linspace(0.0, total_len, n_samples)
        pts = _interpolate_centerline(poly, s, targets)
        u_is_branch = degree_map.get(u, 0) > 1
        v_is_branch = degree_map.get(v, 0) > 1

        junction_s: list[float] = []
        if junction_excl > 0.0 and jn != int(background_label):
            idx_all = physical_points_to_continuous_indices(poly, voxel_size_xyz)
            for i, row in enumerate(idx_all):
                iz, iy, ix = _nearest_integer_index(row, labels.shape)
                if int(labels[iz, iy, ix]) == jn:
                    junction_s.append(float(s[i]))

        same_edge_s_lookup: dict[tuple[int, int, int], float] | None = None
        dense_poly: np.ndarray | None = None
        dense_s: np.ndarray | None = None
        if enforce_same_edge_locality:
            same_edge_s_lookup = {}
            ds_lookup = max(0.1, 0.5 * float(np.min(np.asarray(voxel_size_xyz, dtype=float))))
            dense_pts_list: list[np.ndarray] = []
            dense_s_list: list[float] = []
            for i in range(len(poly) - 1):
                p0 = poly[i]
                p1 = poly[i + 1]
                seg_len = float(np.linalg.norm(p1 - p0))
                if seg_len <= 1e-12:
                    continue
                n_sub = max(1, int(np.ceil(seg_len / ds_lookup)))
                for j in range(n_sub + 1):
                    t = float(j) / float(n_sub)
                    dense_pts_list.append((1.0 - t) * p0 + t * p1)
                    dense_s_list.append((1.0 - t) * float(s[i]) + t * float(s[i + 1]))
            if dense_pts_list:
                dense_poly = np.asarray(dense_pts_list, dtype=float)
                dense_s = np.asarray(dense_s_list, dtype=float)
                dense_idx = physical_points_to_continuous_indices(dense_poly, voxel_size_xyz)
                for idx_row, s_here in zip(dense_idx, dense_s):
                    key_idx = _nearest_integer_index(idx_row, labels.shape)
                    prev = same_edge_s_lookup.get(key_idx)
                    if prev is None or abs(prev - float(s_here)) > 0.5 * ds_lookup:
                        same_edge_s_lookup[key_idx] = float(s_here)

        branch_bounds = _resolve_diameter_bounds_for_branch_order(
            data.get("branch_order"),
            bounds_map,
        )

        diameters: list[float] = []
        profile_lines_phys: list[np.ndarray] = []
        profile_anchors_phys: list[np.ndarray] = []
        bounds_rejected_samples = 0
        bounds_clamped_samples = 0

        for s0, center in zip(targets, pts):
            if u_is_branch and float(s0) < branch_excl:
                continue
            if v_is_branch and float(total_len - s0) < branch_excl:
                continue
            if junction_s and min(abs(float(s0) - sj) for sj in junction_s) < junction_excl:
                continue

            tangent = _tangent_at(poly, s, float(s0))
            n_hat = _transverse_unit_in_physical_yx_plane(tangent)
            center_idx = physical_points_to_continuous_indices(center, voxel_size_xyz)
            if same_edge_arc_window_um is None:
                local_arc_window = max(
                    float(same_edge_arc_window_min_um),
                    float(same_edge_arc_window_multiplier)
                    * max(float(sample_spacing_along_edge_um), float(transverse_profile_step_um)),
                )
            else:
                local_arc_window = float(same_edge_arc_window_um)

            half_extent = max(float(transverse_half_extent_um), 0.5 * mult * d_guess0)
            if cap_half_extent_by_nonlocal_same_edge_distance and len(poly) > 2:
                arc_sep = max(0.0, float(nonlocal_same_edge_arc_separation_um))
                ref_pts = dense_poly if dense_poly is not None else poly
                ref_s = dense_s if dense_s is not None else s
                nonlocal_mask = np.abs(ref_s - float(s0)) >= arc_sep
                if np.any(nonlocal_mask):
                    center_arr = np.asarray(center, dtype=float)
                    d_nonlocal_3d = float(
                        np.min(np.linalg.norm(ref_pts[nonlocal_mask] - center_arr, axis=1))
                    )
                    d_nonlocal_yx = float(
                        np.min(
                            np.linalg.norm(
                                ref_pts[nonlocal_mask][:, 1:3] - center_arr[1:3],
                                axis=1,
                            )
                        )
                    )
                    d_nonlocal = min(d_nonlocal_3d, d_nonlocal_yx)
                    if np.isfinite(d_nonlocal) and d_nonlocal > 0:
                        half_extent = min(
                            half_extent,
                            float(nonlocal_same_edge_half_extent_factor) * d_nonlocal,
                        )

            def _fit_profile_at_extent(current_half_extent: float) -> tuple[np.ndarray, np.ndarray, float | None]:
                pos, prof = _sample_transverse_profile(
                    raw,
                    labels,
                    center,
                    tangent,
                    int(assigned),
                    current_half_extent,
                    float(transverse_profile_step_um),
                    voxel_size_xyz,
                    background_label=int(background_label),
                    junction_label=jn,
                    center_idx=center_idx,
                    allow_junction_crossing=bool(allow_junction_crossing),
                    same_edge_s_lookup=same_edge_s_lookup,
                    same_edge_s0_um=float(s0),
                    same_edge_arc_window_um=local_arc_window,
                )
                pos_fit, prof_fit = (
                    _clip_profile_to_central_lobe(
                        pos,
                        prof,
                        min_drop_fraction_of_center=clip_min_drop_fraction_of_center,
                        re_rise_fraction_of_center=clip_re_rise_fraction_of_center,
                    )
                    if clip_profile_to_single_vessel
                    else (pos, prof)
                )
                d_fit, x0_fit, r2_fit = _fwhm_gaussian_fit_with_diagnostics(
                    pos_fit,
                    prof_fit,
                    profile_baseline_mode=profile_baseline_mode,
                    profile_baseline_wing_fraction=profile_baseline_wing_fraction,
                    constrain_fitted_baseline=constrain_fitted_baseline,
                    baseline_constraint_half_width_ptp=baseline_constraint_half_width_ptp,
                )
                if d_fit is not None:
                    if (
                        reject_samples_with_center_offset
                        and x0_fit is not None
                        and abs(float(x0_fit)) > float(max_fit_center_offset_um)
                    ):
                        d_fit = None
                    if (
                        reject_samples_with_low_fit_r2
                        and r2_fit is not None
                        and float(r2_fit) < float(min_fit_r2)
                    ):
                        d_fit = None
                return pos, prof, d_fit

            accepted_offsets: np.ndarray | None = None
            pos0, _, d0 = _fit_profile_at_extent(half_extent)
            if d0 is not None and d0 > 0:
                half_extent = max(half_extent, 0.5 * mult * float(d0))
                if same_edge_arc_window_um is None:
                    local_arc_window = max(
                        float(same_edge_arc_window_min_um),
                        float(same_edge_arc_window_multiplier) * float(d0),
                    )
                pos1, _, d1 = _fit_profile_at_extent(half_extent)
                accepted_offsets = pos1
                candidate = d1 if (d1 is not None and d1 > 0) else d0
                if d1 is not None and d1 > 0:
                    desired_half = 0.5 * mult * float(d1)
                    if desired_half > (half_extent + float(transverse_profile_step_um)):
                        half_extent = desired_half
                        if same_edge_arc_window_um is None:
                            local_arc_window = max(
                                float(same_edge_arc_window_min_um),
                                float(same_edge_arc_window_multiplier) * float(d1),
                            )
                        pos2, _, d2 = _fit_profile_at_extent(half_extent)
                        accepted_offsets = pos2
                        if d2 is not None:
                            candidate = d2
            else:
                accepted_offsets = pos0
                candidate = d0

            candidate, touched = _apply_diameter_bounds(
                candidate,
                branch_bounds,
                diameter_bounds_mode,
            )
            if touched:
                if candidate is None:
                    bounds_rejected_samples += 1
                else:
                    bounds_clamped_samples += 1
            if candidate is not None and candidate > 0:
                diameters.append(float(candidate))
                if (
                    store_profile_debug
                    and accepted_offsets is not None
                    and accepted_offsets.size > 0
                ):
                    c = np.asarray(center, dtype=float)
                    profile_lines_phys.append(
                        np.stack([c + float(o) * n_hat for o in accepted_offsets], axis=0)
                    )
                    profile_anchors_phys.append(c.copy())

        if not diameters:
            reason = "fwhm_failed"
            if branch_excl > 0 and (u_is_branch or v_is_branch):
                reason = "fwhm_failed_or_excluded_near_branch"
            return {
                "edge": (u, v, key),
                "skip_reason": reason,
                "bounds_rejected_samples": int(bounds_rejected_samples),
                "bounds_clamped_samples": int(bounds_clamped_samples),
            }

        d_med = float(np.median(diameters))
        d_med_bounded, touched_edge = _apply_diameter_bounds(
            d_med,
            branch_bounds,
            diameter_bounds_mode,
        )
        if touched_edge and d_med_bounded is None:
            return {
                "edge": (u, v, key),
                "skip_reason": "edge_diameter_out_of_bounds",
                "bounds_rejected_samples": int(bounds_rejected_samples),
                "bounds_clamped_samples": int(bounds_clamped_samples),
                "bounds_rejected_edge": 1,
                "bounds_clamped_edge": 0,
            }
        return {
            "edge": (u, v, key),
            "skip_reason": None,
            "graph_edge_label_id": int(assigned),
            "fwhm_diameter_um": float(d_med_bounded if d_med_bounded is not None else d_med),
            "fwhm_diameter_samples_um": diameters,
            "fwhm_profile_lines_phys": profile_lines_phys,
            "fwhm_profile_anchors_phys": profile_anchors_phys,
            "bounds_rejected_samples": int(bounds_rejected_samples),
            "bounds_clamped_samples": int(bounds_clamped_samples),
            "bounds_rejected_edge": int(touched_edge and d_med_bounded is None),
            "bounds_clamped_edge": int(
                touched_edge and d_med_bounded is not None and d_med_bounded != d_med
            ),
        }

    def _process_batch(
        edge_batch: list[tuple[Any, Any, Any, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        return [_measure_single_edge(u, v, key, data) for (u, v, key, data) in edge_batch]

    sorted_edges = sorted(G.edges(keys=True, data=True), key=lambda t: (t[0], t[1], t[2]))
    batch_size = int(edge_parallel_batch_size)
    batches: list[list[tuple[Any, Any, Any, dict[str, Any]]]] = [
        sorted_edges[i : i + batch_size]
        for i in range(0, len(sorted_edges), batch_size)
    ]
    worker_count = 1 if edge_parallel_workers is None else int(edge_parallel_workers)

    edge_results_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    if worker_count <= 1 or len(batches) <= 1:
        for batch in batches:
            for item in _process_batch(batch):
                edge_results_by_key[tuple(item["edge"])] = item
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process_batch, batch) for batch in batches]
            for fut in as_completed(futures):
                for item in fut.result():
                    edge_results_by_key[tuple(item["edge"])] = item

    for u, v, key, data in sorted_edges:
        result = edge_results_by_key.get((u, v, key))
        if result is None:
            summary["edges_skipped"].append((u, v, key, "edge_result_missing"))
            continue
        summary["bounds_rejected_samples"] += int(result.get("bounds_rejected_samples", 0))
        summary["bounds_clamped_samples"] += int(result.get("bounds_clamped_samples", 0))
        summary["bounds_rejected_edges"] += int(result.get("bounds_rejected_edge", 0))
        summary["bounds_clamped_edges"] += int(result.get("bounds_clamped_edge", 0))
        skip_reason = result.get("skip_reason")
        if skip_reason:
            summary["edges_skipped"].append((u, v, key, str(skip_reason)))
            continue

        d_med = float(result["fwhm_diameter_um"])
        samples = [float(x) for x in result["fwhm_diameter_samples_um"]]
        data["fwhm_diameter_um"] = d_med
        data["fwhm_diameter_samples_um"] = samples
        if store_profile_debug:
            data["fwhm_profile_lines_phys"] = result.get("fwhm_profile_lines_phys", [])
            data["fwhm_profile_anchors_phys"] = result.get("fwhm_profile_anchors_phys", [])
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {
                "edge": (u, v, key),
                "graph_edge_label_id": int(result["graph_edge_label_id"]),
                "fwhm_diameter_um": d_med,
                "n_samples": len(samples),
                "aggregation": "median",
            }
        )

    return summary
