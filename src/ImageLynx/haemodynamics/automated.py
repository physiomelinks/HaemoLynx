"""Automated vessel diameter estimation from raw TIFF intensity (transverse profiles).

At each sample along an edge, intensity along the perpendicular line is fit with a Gaussian
plus baseline; vessel size is reported as the Gaussian **FWHM**,
``2 * sqrt(2 ln 2) * σ`` (micrometers). By default the baseline seed uses **outer wings**
of the profile (see ``robust_baseline_from_profile_wings``) to reduce bias from a
neighbour-induced shoulder on one side.

Branch identity for clipping comes from an in-memory label volume rasterized from the graph
(see ``build_graph_branch_label_volume``). Transverse extent follows the configured minimum
relative to the current FWHM estimate unless truncated at another edge or volume bounds.

Transverse profiles are sampled **in the physical y–x plane only** (no displacement along
stack axis ``z`` / index 0), so diameter rays follow in-plane directions when voxel spacing
is anisotropic with coarser ``z``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import networkx as nx
import tifffile
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

# FWHM of a Gaussian with standard deviation sigma (not 2*sigma^2 in the exponent).
_GAUSSIAN_FWHM_FROM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


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
    """Convert physical (axis0, axis1, axis2) coordinates to continuous voxel indices.

    Uses the same element-wise scaling as graph construction:
    ``phys[i] = index[i] * voxel_size_xyz[i]``.
    """
    pts = np.asarray(points_phys, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    spacing = _spacing_vec(voxel_size_xyz)
    if np.any(spacing <= 0):
        raise ValueError("voxel_size_xyz components must be positive.")
    return pts / spacing


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
    """Unit vector perpendicular to ``tangent`` with zero component along physical ``z`` (axis 0).

    Coordinates are ``(z, y, x)`` as elsewhere in this module. The returned direction lies in
    the slice plane (varies only ``y`` and ``x``), matching typical microscopy where ``z`` is
    the lower-resolution stack axis and diameters should be measured without stepping along ``z``.
    """
    t = np.asarray(tangent, dtype=float).ravel()
    if t.size != 3:
        raise ValueError("tangent must have length 3 (z, y, x).")
    ty, tx = float(t[1]), float(t[2])
    n2 = float(np.hypot(ty, tx))
    if n2 < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, -tx / n2, ty / n2], dtype=float)


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
) -> float:
    """Positive distance along +direction until hitting another edge or the volume edge.

    Voxels labeled ``assigned_label`` or ``background_label`` (unpainted lumen) allow
    continuation up to ``max_physical_extent``. By default, ``junction_label`` is
    treated as a hard stop to avoid crossing into neighbouring branches at bifurcations.
    Any other positive label is treated as a different graph edge and truncates the line.
    """
    spacing = _spacing_vec(voxel_size_xyz)
    if step_um <= 0:
        raise ValueError("step_um must be positive.")
    d = direction_unit / np.linalg.norm(direction_unit)
    n_steps = int(np.ceil(max_physical_extent / step_um))
    shape = labels.shape
    for k in range(1, n_steps + 1):
        delta_phys = d * (k * step_um)
        idx = center_idx + delta_phys / spacing
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
    allow_junction_crossing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample intensity along a line through ``center_phys``, perpendicular to ``tangent``.

    The line lies in the physical y–x plane (fixed ``z``); see
    ``_transverse_unit_in_physical_yx_plane``.

    Returns (positions_along_line_um, intensities).
    """
    spacing = _spacing_vec(voxel_size_xyz)
    n_hat = _transverse_unit_in_physical_yx_plane(tangent)
    center_idx = center_phys / spacing

    pos_plus = _max_extent_along_ray(
        center_idx,
        n_hat,
        assigned_label,
        labels,
        half_extent_um,
        voxel_size_xyz,
        transverse_step_um,
        background_label=background_label,
        junction_label=junction_label,
        allow_junction_crossing=allow_junction_crossing,
    )
    pos_minus = _max_extent_along_ray(
        center_idx,
        -n_hat,
        assigned_label,
        labels,
        half_extent_um,
        voxel_size_xyz,
        transverse_step_um,
        background_label=background_label,
        junction_label=junction_label,
        allow_junction_crossing=allow_junction_crossing,
    )

    n_neg = int(np.floor(pos_minus / transverse_step_um))
    n_pos = int(np.floor(pos_plus / transverse_step_um))
    offsets = np.arange(-n_neg, n_pos + 1, dtype=float) * transverse_step_um
    if offsets.size == 0:
        offsets = np.array([0.0], dtype=float)

    coords = []
    for off in offsets:
        p_phys = center_phys + off * n_hat
        idx = p_phys / spacing
        coords.append(idx)
    coord_arr = np.stack(coords, axis=1)  # (3, N)
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
    diameter_guess_um: float = 4.0,
    background_label: int = 0,
    junction_label: int = -1,
    min_total_extent_multiplier: float = 3.0,
    profile_baseline_mode: Literal["wings", "percentile"] = "wings",
    profile_baseline_wing_fraction: float = 0.2,
    constrain_fitted_baseline: bool = False,
    baseline_constraint_half_width_ptp: float = 0.35,
    allow_junction_crossing: bool = False,
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
    }

    mult = float(min_total_extent_multiplier)
    if mult < 1.0:
        raise ValueError("min_total_extent_multiplier must be >= 1.")

    jn = int(junction_label)

    for u, v, key, data in G.edges(keys=True, data=True):
        vox = data.get("voxels")
        assigned = data.get("graph_edge_label_id")
        if not vox or len(vox) < 2 or assigned is None:
            summary["edges_skipped"].append((u, v, key, "no_voxels_or_label"))
            continue

        poly = np.asarray(vox, dtype=float)
        s, total_len = _arc_length_parameterize(poly)
        if total_len <= 0:
            summary["edges_skipped"].append((u, v, key, "zero_length"))
            continue

        if sample_spacing_along_edge_um <= 0:
            raise ValueError("sample_spacing_along_edge_um must be positive.")

        n_samples = max(1, int(np.floor(total_len / sample_spacing_along_edge_um)) + 1)
        targets = np.linspace(0.0, total_len, n_samples)
        pts = _interpolate_centerline(poly, s, targets)

        diameters: list[float] = []
        for s0, center in zip(targets, pts):
            tangent = _tangent_at(poly, s, float(s0))

            half_extent = max(
                float(transverse_half_extent_um),
                0.5 * mult * float(diameter_guess_um),
            )
            pos, prof = _sample_transverse_profile(
                raw,
                labels,
                center,
                tangent,
                int(assigned),
                half_extent,
                float(transverse_profile_step_um),
                voxel_size_xyz,
                background_label=int(background_label),
                junction_label=jn,
                allow_junction_crossing=bool(allow_junction_crossing),
            )
            d0 = fwhm_from_profile(
                pos,
                prof,
                profile_baseline_mode=profile_baseline_mode,
                profile_baseline_wing_fraction=profile_baseline_wing_fraction,
                constrain_fitted_baseline=constrain_fitted_baseline,
                baseline_constraint_half_width_ptp=baseline_constraint_half_width_ptp,
            )
            if d0 is not None and d0 > 0:
                half_extent = max(
                    half_extent,
                    0.5 * mult * d0,
                )
                pos, prof = _sample_transverse_profile(
                    raw,
                    labels,
                    center,
                    tangent,
                    int(assigned),
                    half_extent,
                    float(transverse_profile_step_um),
                    voxel_size_xyz,
                    background_label=int(background_label),
                    junction_label=jn,
                    allow_junction_crossing=bool(allow_junction_crossing),
                )
                d1 = fwhm_from_profile(
                    pos,
                    prof,
                    profile_baseline_mode=profile_baseline_mode,
                    profile_baseline_wing_fraction=profile_baseline_wing_fraction,
                    constrain_fitted_baseline=constrain_fitted_baseline,
                    baseline_constraint_half_width_ptp=baseline_constraint_half_width_ptp,
                )
                if d1 is not None:
                    diameters.append(d1)
            elif d0 is not None:
                diameters.append(d0)

        if not diameters:
            summary["edges_skipped"].append((u, v, key, "fwhm_failed"))
            continue

        d_mean = float(np.mean(diameters))
        data["fwhm_diameter_um"] = d_mean
        data["fwhm_diameter_samples_um"] = diameters
        summary["edges_measured"] += 1
        summary["per_edge"].append(
            {
                "edge": (u, v, key),
                "graph_edge_label_id": int(assigned),
                "fwhm_diameter_um": d_mean,
                "n_samples": len(diameters),
            }
        )

    return summary
