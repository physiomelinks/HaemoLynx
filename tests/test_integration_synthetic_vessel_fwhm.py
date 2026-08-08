"""Integration: synthetic 3D vessel TIFF + matching graph, real FWHM diameters, Plotly HTML.

The pytest case writes ``synthetic_vessel_fwhm_viz.html`` under pytest’s ``tmp_path`` (intensity
as a marching-cubes mesh, vessel centerlines, and the transverse polylines used for FWHM sampling — same
logic as ``measure_edge_diameters_fwhm_from_raw_tiff``). Skip plotting I/O with ``-m "not plotting"``.

Run as a script to write the same style figure under ``examples/plots/``::

    python tests/test_integration_synthetic_vessel_fwhm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ``pytest`` adds ``src`` via ``pythonpath``; running this file directly does not.
_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"
if _src.is_dir():
    _src_str = str(_src)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)

import tempfile

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pytest
import tifffile
from skimage.measure import label as sk_label_cc
from skimage.measure import marching_cubes

from haemolynx.haemodynamics import automated

_GAUSSIAN_FWHM_FROM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_I0 = 100.0

# Shared with ``measure_edge_diameters_fwhm_from_raw_tiff`` calls and the visualization helper.
_DEFAULT_FWHM_MEASURE_KWARGS: dict = {
    "sample_spacing_along_edge_um": 2.0,
    "transverse_profile_step_um": 0.2,
    "transverse_half_extent_um": 4.0,
    "diameter_guess_um": None,
    "background_label": 0,
    "junction_label": -1,
    "min_total_extent_multiplier": 3.0,
    "profile_baseline_mode": "wings",
    "profile_baseline_wing_fraction": 0.2,
    "constrain_fitted_baseline": False,
    "baseline_constraint_half_width_ptp": 0.35,
    "clip_profile_to_single_vessel": True,
    "clip_min_drop_fraction_of_center": 0.35,
    "clip_re_rise_fraction_of_center": 0.08,
    "branch_endpoint_exclusion_um": 10.0,
    "junction_proximity_exclusion_um": 10.0,
    "store_profile_debug": True,
    "enforce_same_edge_locality": True,
    "same_edge_arc_window_um": 3.0,
    "same_edge_arc_window_multiplier": 1.0,
    "same_edge_arc_window_min_um": 1.0,
    "cap_half_extent_by_nonlocal_same_edge_distance": True,
    "nonlocal_same_edge_arc_separation_um": 6.0,
    "nonlocal_same_edge_half_extent_factor": 0.45,
    "reject_samples_with_center_offset": True,
    "max_fit_center_offset_um": 1.5,
    "reject_samples_with_low_fit_r2": True,
    "min_fit_r2": 0.85,
}

_EDGE_LINE_COLORS = ("#00ffff", "#ffaa00", "#cc66ff")
_PROFILE_LINE_COLORS = ("#00cc88", "#ff6600", "#9933ff")
_VOLUME_MESH_COLORS = ("#00ffff", "#ffaa00", "#cc66ff")  # match centerlines: low z → high z


def _measure_kwargs_with_overrides(overrides: dict | None = None) -> dict:
    """Copy default measurement kwargs and apply optional scenario-specific overrides."""
    out = dict(_DEFAULT_FWHM_MEASURE_KWARGS)
    if overrides:
        out.update(overrides)
    return out


def _dist_point_to_segment_batch(
    pz: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Per-voxel distance to segment ab; p* have shape (nz, ny, nx)."""
    ab = b - a
    ab2 = float(np.dot(ab, ab)) + 1e-20
    pa_z = pz - a[0]
    pa_y = py - a[1]
    pa_x = px - a[2]
    t = (pa_z * ab[0] + pa_y * ab[1] + pa_x * ab[2]) / ab2
    t = np.clip(t, 0.0, 1.0)
    cz = a[0] + t * ab[0]
    cy = a[1] + t * ab[1]
    cx = a[2] + t * ab[2]
    return np.sqrt((pz - cz) ** 2 + (py - cy) ** 2 + (px - cx) ** 2)


def build_synthetic_vessel_volume_and_targets(
    voxel_size_zyx: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Float32 volume + list of (endpoint_a, endpoint_b, target_fwhm_um) in (z,y,x) µm.

    Three straight vessels parallel to +x with geometric FWHM 3, 5, and 8 µm (Gaussian cross-section).

    Vessels share the same physical *y* and differ in *z*.     For tangents along +x, ``_transverse_unit_in_physical_yx_plane`` yields a transverse
    direction in the y–x plane (here, ±y). If vessels were stacked in *y*, one ray direction
    would hit a neighbour's raster label after a few µm while the other would cross only
    background to the volume edge, producing strongly asymmetric profile lines (not a rendering
    bug). Vessels are therefore separated in *z* at a shared *y*.
    """
    vz, vy, vx = voxel_size_zyx
    y_center = 15.0
    z_centers = (6.0, 14.0, 22.0)
    x0, x1 = 8.0, 42.0
    sigma_max = 8.0 / _GAUSSIAN_FWHM_FROM_SIGMA
    pad_z = 4.0 * sigma_max + 2.0
    pad_y = 16.0  # room for ``transverse_half_extent_um``-scale rays along ±y
    nz = int(np.ceil((max(z_centers) + pad_z) / vz)) + 1
    ny = int(np.ceil((y_center + pad_y) / vy)) + 1
    nx = int(np.ceil(48.0 / vx)) + 1

    iz = np.arange(nz, dtype=float)[:, None, None]
    iy = np.arange(ny, dtype=float)[None, :, None]
    ix = np.arange(nx, dtype=float)[None, None, :]
    pz = iz * vz
    py = iy * vy
    px = ix * vx

    specs: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = [
        ((z_centers[0], y_center, x0), (z_centers[0], y_center, x1), 3.0),
        ((z_centers[1], y_center, x0), (z_centers[1], y_center, x1), 5.0),
        ((z_centers[2], y_center, x0), (z_centers[2], y_center, x1), 8.0),
    ]

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    targets: list[tuple[np.ndarray, np.ndarray, float]] = []
    for a_tup, b_tup, fwhm in specs:
        a = np.array(a_tup, dtype=float)
        b = np.array(b_tup, dtype=float)
        sigma = fwhm / _GAUSSIAN_FWHM_FROM_SIGMA
        d = _dist_point_to_segment_batch(pz, py, px, a, b)
        tube = (_I0 * np.exp(-(d**2) / (2.0 * sigma**2))).astype(np.float32)
        vol = np.maximum(vol, tube)
        targets.append((a, b, fwhm))

    return vol, targets


def build_matching_multigraph(
    targets: list[tuple[np.ndarray, np.ndarray, float]],
    voxel_size_zyx: tuple[float, float, float],
    step_um: float = 0.25,
) -> nx.MultiGraph:
    """Graph edges replicate centerlines; physical coords (z,y,x) match ``automated`` convention."""
    vz, vy, vx = voxel_size_zyx
    step = min(step_um, vz, vy, vx)
    G = nx.MultiGraph()
    branch_orders = ("B01", "B02", "B03")
    node_id = 0
    for (a, b, _), bo in zip(targets, branch_orders):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        length = float(np.linalg.norm(ab))
        assert length > 0
        direc = ab / length
        n = max(2, int(np.floor(length / step)) + 1)
        tvals = np.linspace(0.0, length, n)
        voxels = [tuple((a + t * direc).tolist()) for t in tvals]
        G.add_node(node_id, pos=a.copy())
        G.add_node(node_id + 1, pos=b.copy())
        G.add_edge(
            node_id,
            node_id + 1,
            weight=1.0,
            length=length,
            branch_order=bo,
            voxels=voxels,
        )
        node_id += 2
    return G


def build_synthetic_y_shaped_volume_and_targets(
    voxel_size_zyx: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Float32 Y-shaped vessel volume + segment targets as (a, b, fwhm_um) in (z,y,x) µm.

    Geometry:
    - one stem along +x
    - two daughter branches diverging in y at fixed z
    """
    vz, vy, vx = voxel_size_zyx
    z0 = 14.0
    y0 = 15.0
    x_stem0 = 8.0
    x_bif = 24.0
    x_tip = 42.0
    y_delta = 8.0

    # stem, upper daughter, lower daughter
    specs: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = [
        ((z0, y0, x_stem0), (z0, y0, x_bif), 6.0),
        ((z0, y0, x_bif), (z0, y0 + y_delta, x_tip), 4.0),
        ((z0, y0, x_bif), (z0, y0 - y_delta, x_tip), 4.0),
    ]

    sigma_max = max(f for _, _, f in specs) / _GAUSSIAN_FWHM_FROM_SIGMA
    pad = 4.0 * sigma_max + 2.0
    z_max = z0 + pad
    y_max = y0 + y_delta + pad
    x_max = x_tip + pad

    nz = int(np.ceil(z_max / vz)) + 1
    ny = int(np.ceil(y_max / vy)) + 1
    nx = int(np.ceil(x_max / vx)) + 1

    iz = np.arange(nz, dtype=float)[:, None, None]
    iy = np.arange(ny, dtype=float)[None, :, None]
    ix = np.arange(nx, dtype=float)[None, None, :]
    pz = iz * vz
    py = iy * vy
    px = ix * vx

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    targets: list[tuple[np.ndarray, np.ndarray, float]] = []
    for a_tup, b_tup, fwhm in specs:
        a = np.array(a_tup, dtype=float)
        b = np.array(b_tup, dtype=float)
        sigma = float(fwhm) / _GAUSSIAN_FWHM_FROM_SIGMA
        d = _dist_point_to_segment_batch(pz, py, px, a, b)
        tube = (_I0 * np.exp(-(d**2) / (2.0 * sigma**2))).astype(np.float32)
        vol = np.maximum(vol, tube)
        targets.append((a, b, float(fwhm)))
    return vol, targets


def build_y_shaped_matching_multigraph(
    targets: list[tuple[np.ndarray, np.ndarray, float]],
    voxel_size_zyx: tuple[float, float, float],
    step_um: float = 0.25,
) -> nx.MultiGraph:
    """Graph for Y-shaped targets with a shared bifurcation node at the common point."""
    vz, vy, vx = voxel_size_zyx
    step = min(step_um, vz, vy, vx)
    G = nx.MultiGraph()

    # Node ids: 0=stem start, 1=bifurcation, 2=upper tip, 3=lower tip
    stem_a, stem_b, _ = targets[0]
    up_a, up_b, _ = targets[1]
    lo_a, lo_b, _ = targets[2]
    G.add_node(0, pos=np.asarray(stem_a, dtype=float))
    G.add_node(1, pos=np.asarray(stem_b, dtype=float))
    G.add_node(2, pos=np.asarray(up_b, dtype=float))
    G.add_node(3, pos=np.asarray(lo_b, dtype=float))

    edges = [
        (0, 1, stem_a, stem_b, "B01"),
        (1, 2, up_a, up_b, "B02"),
        (1, 3, lo_a, lo_b, "B03"),
    ]
    for u, v, a, b, bo in edges:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length <= 0:
            continue
        direc = ab / length
        n = max(2, int(np.floor(length / step)) + 1)
        tvals = np.linspace(0.0, length, n)
        voxels = [tuple((a + t * direc).tolist()) for t in tvals]
        G.add_edge(
            u,
            v,
            weight=1.0,
            length=length,
            branch_order=bo,
            voxels=voxels,
        )
    return G


def build_offcenter_matching_multigraph(
    targets: list[tuple[np.ndarray, np.ndarray, float]],
    voxel_size_zyx: tuple[float, float, float],
    step_um: float = 0.25,
    *,
    offcenter_fraction: float = 0.3,
) -> nx.MultiGraph:
    """Straight-vessel graph with centerlines offset in x and z by a diameter fraction.

    The synthetic raw volume stays unchanged; this function intentionally perturbs graph geometry.
    """
    if offcenter_fraction < 0:
        raise ValueError("offcenter_fraction must be non-negative.")
    shifted_targets: list[tuple[np.ndarray, np.ndarray, float]] = []
    for a, b, fwhm in targets:
        a = np.asarray(a, dtype=float).copy()
        b = np.asarray(b, dtype=float).copy()
        dz = float(offcenter_fraction) * float(fwhm)
        dx = float(offcenter_fraction) * float(fwhm)
        a[0] += dz
        b[0] += dz
        a[2] += dx
        b[2] += dx
        shifted_targets.append((a, b, float(fwhm)))
    return build_matching_multigraph(shifted_targets, voxel_size_zyx, step_um=step_um)


def add_background_noise_to_synthetic_volume(
    volume: np.ndarray,
    *,
    noise_sigma: float = 6.0,
    background_offset: float = 4.0,
    seed: int = 123,
) -> np.ndarray:
    """Add reproducible background noise to a synthetic raw volume."""
    vol = np.asarray(volume, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    noisy = vol + float(background_offset) + rng.normal(
        loc=0.0, scale=float(noise_sigma), size=vol.shape
    ).astype(np.float32)
    return np.clip(noisy, 0.0, None).astype(np.float32)


def build_synthetic_x_junction_volume_and_targets(
    voxel_size_zyx: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Float32 X-junction vessel volume + segment targets in (z,y,x) µm.

    Four branches meet at one central node, forming an X in the y-x plane.
    """
    vz, vy, vx = voxel_size_zyx
    z0 = 14.0
    y0 = 15.0
    x0 = 24.0
    arm_x = 14.0
    arm_y = 8.0
    fwhm = 2.5
    center = np.array([z0, y0, x0], dtype=float)
    tips = [
        np.array([z0, y0 - arm_y, x0 - arm_x], dtype=float),
        np.array([z0, y0 + arm_y, x0 + arm_x], dtype=float),
        np.array([z0, y0 + arm_y, x0 - arm_x], dtype=float),
        np.array([z0, y0 - arm_y, x0 + arm_x], dtype=float),
    ]
    specs: list[tuple[np.ndarray, np.ndarray, float]] = [(center, t, fwhm) for t in tips]

    sigma = fwhm / _GAUSSIAN_FWHM_FROM_SIGMA
    pad = 4.0 * sigma + 2.0
    z_max = z0 + pad
    y_max = y0 + arm_y + pad
    x_max = x0 + arm_x + pad
    nz = int(np.ceil(z_max / vz)) + 1
    ny = int(np.ceil(y_max / vy)) + 1
    nx = int(np.ceil(x_max / vx)) + 1

    iz = np.arange(nz, dtype=float)[:, None, None]
    iy = np.arange(ny, dtype=float)[None, :, None]
    ix = np.arange(nx, dtype=float)[None, None, :]
    pz = iz * vz
    py = iy * vy
    px = ix * vx

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    targets: list[tuple[np.ndarray, np.ndarray, float]] = []
    for a, b, fw in specs:
        d = _dist_point_to_segment_batch(pz, py, px, a, b)
        tube = (_I0 * np.exp(-(d**2) / (2.0 * sigma**2))).astype(np.float32)
        vol = np.maximum(vol, tube)
        targets.append((a.copy(), b.copy(), float(fw)))
    return vol, targets


def build_x_junction_matching_multigraph(
    targets: list[tuple[np.ndarray, np.ndarray, float]],
    voxel_size_zyx: tuple[float, float, float],
    step_um: float = 0.25,
    *,
    offcenter_fraction: float = 0.0,
) -> nx.MultiGraph:
    """Graph for X-junction targets (shared center node + four arms), optional x/z off-center."""
    vz, vy, vx = voxel_size_zyx
    step = min(step_um, vz, vy, vx)
    if offcenter_fraction < 0:
        raise ValueError("offcenter_fraction must be non-negative.")

    center = np.asarray(targets[0][0], dtype=float).copy()
    G = nx.MultiGraph()
    G.add_node(0, pos=center.copy())
    branch_orders = ("B01", "B02", "B03", "B04")
    for i, (a, b, fwhm) in enumerate(targets, start=1):
        a = np.asarray(a, dtype=float).copy()
        b = np.asarray(b, dtype=float).copy()
        if offcenter_fraction > 0:
            dz = float(offcenter_fraction) * float(fwhm)
            dx = float(offcenter_fraction) * float(fwhm)
            a[0] += dz
            b[0] += dz
            a[2] += dx
            b[2] += dx
        # Ensure node 0 is used as the branch root.
        root = a
        tip = b
        G.add_node(i, pos=tip.copy())
        ab = tip - root
        length = float(np.linalg.norm(ab))
        if length <= 0:
            continue
        direc = ab / length
        n = max(2, int(np.floor(length / step)) + 1)
        tvals = np.linspace(0.0, length, n)
        voxels = [tuple((root + t * direc).tolist()) for t in tvals]
        G.add_edge(
            0,
            i,
            weight=1.0,
            length=length,
            branch_order=branch_orders[(i - 1) % len(branch_orders)],
            voxels=voxels,
        )
    return G


def build_synthetic_tight_zigzag_volume_and_target(
    voxel_size_zyx: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], np.ndarray]:
    """Float32 tight zig-zag vessel volume + single target + centerline polyline (z,y,x)."""
    vz, vy, vx = voxel_size_zyx
    # Tight alternating lateral oscillation while progressing in +x.
    points = np.array(
        [
            [14.0, 15.0, 8.0],
            [14.0, 11.5, 12.0],
            [14.0, 18.5, 16.0],
            [14.0, 11.5, 20.0],
            [14.0, 18.5, 24.0],
            [14.0, 11.5, 28.0],
            [14.0, 18.5, 32.0],
            [14.0, 11.5, 36.0],
            [14.0, 18.5, 40.0],
        ],
        dtype=float,
    )
    fwhm = 2.5
    sigma = fwhm / _GAUSSIAN_FWHM_FROM_SIGMA
    pad = 4.0 * sigma + 2.0
    z_max = float(np.max(points[:, 0]) + pad)
    y_max = float(np.max(points[:, 1]) + pad)
    x_max = float(np.max(points[:, 2]) + pad)
    nz = int(np.ceil(z_max / vz)) + 1
    ny = int(np.ceil(y_max / vy)) + 1
    nx = int(np.ceil(x_max / vx)) + 1

    iz = np.arange(nz, dtype=float)[:, None, None]
    iy = np.arange(ny, dtype=float)[None, :, None]
    ix = np.arange(nx, dtype=float)[None, None, :]
    pz = iz * vz
    py = iy * vy
    px = ix * vx

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    for a, b in zip(points[:-1], points[1:]):
        d = _dist_point_to_segment_batch(pz, py, px, a, b)
        tube = (_I0 * np.exp(-(d**2) / (2.0 * sigma**2))).astype(np.float32)
        vol = np.maximum(vol, tube)

    targets = [(points[0].copy(), points[-1].copy(), float(fwhm))]
    return vol, targets, points


def build_tight_zigzag_matching_multigraph(
    centerline_points: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] = (0.25, 0.25, 0.25),
    step_um: float = 0.25,
) -> nx.MultiGraph:
    """Single-edge graph whose voxels densely follow the tight zig-zag polyline."""
    pts = np.asarray(centerline_points, dtype=float)
    vz, vy, vx = voxel_size_zyx
    step = min(float(step_um), float(vz), float(vy), float(vx))
    if step <= 0:
        raise ValueError("step_um and voxel_size_zyx must be positive.")

    dense_voxels: list[tuple[float, float, float]] = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-12:
            continue
        n = max(2, int(np.floor(seg_len / step)) + 1)
        tvals = np.linspace(0.0, 1.0, n)
        for t in tvals:
            p = (1.0 - float(t)) * a + float(t) * b
            p_tup = (float(p[0]), float(p[1]), float(p[2]))
            if not dense_voxels or dense_voxels[-1] != p_tup:
                dense_voxels.append(p_tup)
    if not dense_voxels:
        dense_voxels = [tuple(map(float, pts[0])), tuple(map(float, pts[-1]))]

    G = nx.MultiGraph()
    G.add_node(0, pos=pts[0].copy())
    G.add_node(1, pos=pts[-1].copy())
    dense_arr = np.asarray(dense_voxels, dtype=float)
    seg = np.diff(dense_arr, axis=0)
    length = float(np.sum(np.linalg.norm(seg, axis=1)))
    G.add_edge(
        0,
        1,
        weight=1.0,
        length=length,
        branch_order="B01",
        voxels=dense_voxels,
    )
    return G


def _iter_profile_polylines_from_graph(
    G: nx.MultiGraph,
) -> list[tuple[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]]]:
    """Read profile polylines stored by the main pipeline function on edge attributes."""
    out: list[tuple[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]]] = []
    for u, v, key, data in sorted(G.edges(keys=True, data=True), key=lambda t: (t[0], t[1], t[2])):
        lines = data.get("fwhm_profile_lines_phys") or []
        anchors = data.get("fwhm_profile_anchors_phys") or []
        if not lines:
            continue
        segs: list[tuple[np.ndarray, np.ndarray]] = []
        for line, anchor in zip(lines, anchors):
            segs.append((np.asarray(line, dtype=float), np.asarray(anchor, dtype=float)))
        if segs:
            out.append(((u, v, key), segs))
    return out


def _volume_stride_for_display(nz: int, ny: int, nx: int, target_voxels: int = 220_000) -> int:
    """Coarser grid for volume meshing — keeps marching cubes and Mesh3d responsive in the browser."""
    ntot = nz * ny * nx
    if ntot <= target_voxels:
        return 1
    s = int(np.ceil((ntot / float(target_voxels)) ** (1.0 / 3.0)))
    return int(np.clip(s, 2, 8))


def _add_synthetic_volume_trace(
    fig: go.Figure,
    raw_iso: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> None:
    """One ``Mesh3d`` per connected supra-threshold blob (three tubes), not one merged isosurface.

    A single low isolevel joins Gaussian tails between vessels into one giant region; we raise
    the threshold until components separate (or take the largest three components by voxel count).
    """
    dz, dy, dx = spacing_zyx
    iso_max = float(np.nanmax(raw_iso)) if raw_iso.size else 0.0
    if iso_max <= 0:
        return

    min_voxels = max(400, int(raw_iso.size * 2e-5))
    level = 0.17 * iso_max
    best_lab: np.ndarray | None = None
    best_n = 0
    for _ in range(28):
        fg = raw_iso >= level
        if not np.any(fg):
            level *= 0.92
            continue
        lab, n_comp = sk_label_cc(fg, connectivity=3, return_num=True)
        large = [i for i in range(1, n_comp + 1) if int(np.sum(lab == i)) >= min_voxels]
        if len(large) > best_n:
            best_lab, best_n = lab, len(large)
        if len(large) >= 3:
            best_lab, best_n = lab, len(large)
            break
        level = min(level * 1.065, 0.62 * iso_max)

    if best_lab is None or best_n == 0:
        _scatter_volume_fallback(fig, raw_iso, spacing_zyx, max(0.12 * iso_max, 1.0))
        return

    lab = best_lab
    n_comp = int(np.max(lab))
    ids = [i for i in range(1, n_comp + 1) if int(np.sum(lab == i)) >= min_voxels]
    ids.sort(key=lambda i: int(np.sum(lab == i)), reverse=True)
    ids = ids[:3]

    z_mean = []
    for i in ids:
        coords = np.argwhere(lab == i)
        z_mean.append((float(np.mean(coords[:, 0])), i))
    z_mean.sort(key=lambda t: t[0])
    ordered = [t[1] for t in z_mean]

    for rank, comp_id in enumerate(ordered):
        vol = (lab == comp_id).astype(np.float64)
        try:
            verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(dz, dy, dx))
        except (ValueError, RuntimeError):
            continue
        if faces is None or len(faces) == 0:
            continue
        col = _VOLUME_MESH_COLORS[rank % len(_VOLUME_MESH_COLORS)]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 2],
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=col,
                opacity=0.48,
                name=f"Vessel volume (z-stack {rank + 1})",
                lighting=dict(ambient=0.55, diffuse=0.65, specular=0.25),
                flatshading=False,
            )
        )

    if not fig.data:
        _scatter_volume_fallback(fig, raw_iso, spacing_zyx, max(0.12 * iso_max, 1.0))


def _scatter_volume_fallback(
    fig: go.Figure,
    raw_iso: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    level: float,
) -> None:
    dz, dy, dx = spacing_zyx
    mask = raw_iso >= float(level) * 0.85
    if not np.any(mask):
        mask = raw_iso > 0
    iz, iy, ix = np.nonzero(mask)
    npts = iz.size
    if npts == 0:
        return
    step = max(1, npts // 60_000)
    iz, iy, ix = iz[::step], iy[::step], ix[::step]
    z = iz.astype(np.float64) * dz
    y = iy.astype(np.float64) * dy
    x = ix.astype(np.float64) * dx
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2, color="steelblue", opacity=0.35),
            name="Synthetic raw (voxel cloud)",
        )
    )


def build_synthetic_fwhm_integration_figure(
    G: nx.MultiGraph,
    raw: np.ndarray,
    labels: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    measure_kwargs: dict,
    *,
    title: str,
) -> go.Figure:
    """Plotly scene: raw intensity as a triangle mesh (marching cubes), centerlines, profile lines."""
    nz, ny, nx = raw.shape
    vz, vy, vx = voxel_size_zyx

    st = _volume_stride_for_display(nz, ny, nx)
    raw_iso = np.asarray(raw[::st, ::st, ::st], dtype=np.float64)
    spacing_zyx = (vz * float(st), vy * float(st), vx * float(st))

    fig = go.Figure()
    _add_synthetic_volume_trace(fig, raw_iso, spacing_zyx)

    sorted_edges = sorted(G.edges(keys=True), key=lambda t: (t[0], t[1], t[2]))
    for ei, (u, v, key) in enumerate(sorted_edges):
        col = _EDGE_LINE_COLORS[ei % len(_EDGE_LINE_COLORS)]
        vox = G[u][v][key].get("voxels") or []
        if len(vox) < 2:
            continue
        xs = [float(pt[2]) for pt in vox]
        ys = [float(pt[1]) for pt in vox]
        zs = [float(pt[0]) for pt in vox]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=col, width=6),
                name=f"Centerline {u}–{v}",
            )
        )

    polylines_by_edge = _iter_profile_polylines_from_graph(G)
    for ei, ((u, v, key), segs) in enumerate(polylines_by_edge):
        col = _PROFILE_LINE_COLORS[ei % len(_PROFILE_LINE_COLORS)]
        px: list[float | None] = []
        py: list[float | None] = []
        pz: list[float | None] = []
        ax, ay, az = [], [], []
        for seg, anchor in segs:
            for p in seg:
                px.append(float(p[2]))
                py.append(float(p[1]))
                pz.append(float(p[0]))
            px.append(None)
            py.append(None)
            pz.append(None)
            ax.append(float(anchor[2]))
            ay.append(float(anchor[1]))
            az.append(float(anchor[0]))
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="lines",
                line=dict(color=col, width=2),
                name=f"Profile lines {u}–{v}",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="markers",
                marker=dict(size=4, color=col, symbol="diamond", line=dict(width=0.5, color="white")),
                name=f"Profile anchor (offset 0) {u}–{v}",
            )
        )

    fig.update_layout(
        title=title
        + "<br><sup>Profiles lie in the physical y–x plane (no z step); extent stops at "
        "other edges' labels or bounds. Markers: profile anchors.</sup>",
        showlegend=True,
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
            aspectmode="data",
        ),
    )
    return fig


@pytest.mark.integration
@pytest.mark.plotting
def test_synthetic_three_vessels_fwhm_pipeline(tmp_path: Path) -> None:
    """Synthetic Gaussian tubes (3 / 5 / 8 µm FWHM) measured via ``measure_edge_diameters_fwhm_from_raw_tiff``."""
    voxel_size_zyx = (0.25, 0.25, 0.25)
    raw, targets = build_synthetic_vessel_volume_and_targets(voxel_size_zyx)
    raw_path = tmp_path / "synthetic_vessels.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = build_matching_multigraph(targets, voxel_size_zyx, step_um=0.25)
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel_size_zyx,
        **_DEFAULT_FWHM_MEASURE_KWARGS,
    )
    assert summary["edges_measured"] == 3
    assert not summary["edges_skipped"]

    expected = [t[2] for t in targets]
    # Deterministic edge order matches sorted (u,v,key): (0,1,0), (2,3,0), (4,5,0)
    pairs = [(0, 1, 0), (2, 3, 0), (4, 5, 0)]
    for (u, v, k), exp in zip(pairs, expected):
        got = float(G[u][v][k]["fwhm_diameter_um"])
        assert abs(got - exp) < max(0.9, 0.18 * exp), (
            f"edge ({u},{v},{k}): measured {got:.3f} µm vs target {exp:.3f} µm"
        )

    labels, _ = automated.build_graph_branch_label_volume(
        G,
        raw.shape,
        voxel_size_zyx,
        background_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["background_label"]),
        junction_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["junction_label"]),
    )
    title_parts = []
    for (u, v, k), t in zip(pairs, targets):
        d = float(G[u][v][k]["fwhm_diameter_um"])
        title_parts.append(f"FWHM {t[2]:.1f}→{d:.2f} µm")
    title = "Synthetic vessels: volume mesh + centerlines + transverse profile lines | " + " | ".join(
        title_parts
    )
    fig = build_synthetic_fwhm_integration_figure(
        G,
        raw,
        labels,
        voxel_size_zyx,
        _DEFAULT_FWHM_MEASURE_KWARGS,
        title=title,
    )
    fig.write_html(str(tmp_path / "synthetic_vessel_fwhm_viz.html"), include_plotlyjs="cdn")


@pytest.mark.integration
@pytest.mark.plotting
def test_synthetic_x_junction_offcenter_noisy_fwhm_pipeline(tmp_path: Path) -> None:
    """Noisy X-junction raw volume with 30% off-center graph still yields sane diameters."""
    voxel_size_zyx = (0.25, 0.25, 0.25)
    raw_clean, targets = build_synthetic_x_junction_volume_and_targets(voxel_size_zyx)
    raw_noisy = add_background_noise_to_synthetic_volume(
        raw_clean, noise_sigma=6.0, background_offset=4.0, seed=321
    )
    raw_path = tmp_path / "synthetic_x_junction_noisy.tif"
    tifffile.imwrite(str(raw_path), raw_noisy)

    G = build_x_junction_matching_multigraph(
        targets, voxel_size_zyx, step_um=0.25, offcenter_fraction=0.3
    )
    # Keep this scenario close to the original 3x-width behavior; tortuous-vessel
    # guards are useful for zig-zag, but too restrictive for this validation plot.
    xj_measure_kwargs = _measure_kwargs_with_overrides(
        {
            "branch_endpoint_exclusion_um": 10.0,
            "junction_proximity_exclusion_um": 10.0,
            "enforce_same_edge_locality": False,
            "cap_half_extent_by_nonlocal_same_edge_distance": False,
            "reject_samples_with_center_offset": False,
            "reject_samples_with_low_fit_r2": False,
        }
    )
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel_size_zyx,
        **xj_measure_kwargs,
    )
    # Four arms should measure unless noise/truncation causes occasional fitting loss.
    assert summary["edges_measured"] >= 3

    pairs = sorted(G.edges(keys=True), key=lambda t: (t[0], t[1], t[2]))
    expected = [t[2] for t in targets]
    measured = []
    for (u, v, k), exp in zip(pairs, expected):
        d = G[u][v][k].get("fwhm_diameter_um")
        if d is None:
            continue
        d = float(d)
        measured.append(d)
        assert np.isfinite(d)
        assert abs(d - exp) < max(1.3, 0.3 * exp), (
            f"edge ({u},{v},{k}): measured {d:.3f} µm vs target {exp:.3f} µm"
        )
    assert len(measured) >= 3

    labels, _ = automated.build_graph_branch_label_volume(
        G,
        raw_noisy.shape,
        voxel_size_zyx,
        background_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["background_label"]),
        junction_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["junction_label"]),
    )
    title_parts = []
    for (u, v, k), t in zip(pairs, targets):
        d = G[u][v][k].get("fwhm_diameter_um", float("nan"))
        d_txt = f"{float(d):.2f}" if np.isfinite(float(d)) else "nan"
        title_parts.append(f"FWHM {t[2]:.1f}→{d_txt} µm")
    title = "Synthetic noisy X-junction + 30% off-center graph: volume mesh + centerlines + profiles | " + " | ".join(
        title_parts
    )
    fig = build_synthetic_fwhm_integration_figure(
        G,
        raw_noisy,
        labels,
        voxel_size_zyx,
        xj_measure_kwargs,
        title=title,
    )
    fig.write_html(
        str(tmp_path / "synthetic_x_junction_offcenter_noisy_fwhm_viz.html"),
        include_plotlyjs="cdn",
    )


@pytest.mark.integration
@pytest.mark.plotting
def test_synthetic_tight_zigzag_fwhm_pipeline(tmp_path: Path) -> None:
    """Tightly zig-zagging single vessel, measured via real pipeline + HTML output."""
    voxel_size_zyx = (0.25, 0.25, 0.25)
    raw, targets, centerline = build_synthetic_tight_zigzag_volume_and_target(voxel_size_zyx)
    raw_path = tmp_path / "synthetic_tight_zigzag.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = build_tight_zigzag_matching_multigraph(centerline)
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel_size_zyx,
        **_DEFAULT_FWHM_MEASURE_KWARGS,
    )
    assert summary["edges_measured"] == 1
    assert not summary["edges_skipped"]

    d = float(G[0][1][0]["fwhm_diameter_um"])
    exp = float(targets[0][2])
    assert abs(d - exp) < max(1.0, 0.25 * exp), (
        f"zig-zag edge measured {d:.3f} µm vs target {exp:.3f} µm"
    )

    labels, _ = automated.build_graph_branch_label_volume(
        G,
        raw.shape,
        voxel_size_zyx,
        background_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["background_label"]),
        junction_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["junction_label"]),
    )
    title = (
        "Synthetic tight zig-zag vessel: volume mesh + centerline + transverse profile lines"
        f" | FWHM {exp:.1f}→{d:.2f} µm"
    )
    fig = build_synthetic_fwhm_integration_figure(
        G,
        raw,
        labels,
        voxel_size_zyx,
        _DEFAULT_FWHM_MEASURE_KWARGS,
        title=title,
    )
    fig.write_html(
        str(tmp_path / "synthetic_tight_zigzag_fwhm_viz.html"),
        include_plotlyjs="cdn",
    )


def _write_single_demo_html(
    out_path: Path,
    raw: np.ndarray,
    G: nx.MultiGraph,
    voxel_size_zyx: tuple[float, float, float],
    title_prefix: str,
    targets: list[tuple[np.ndarray, np.ndarray, float]],
    measure_kwargs_overrides: dict | None = None,
) -> None:
    """Run FWHM on (raw, G) and write one Plotly HTML file."""
    measure_kwargs = _measure_kwargs_with_overrides(measure_kwargs_overrides)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
        raw_path = Path(tf.name)
    try:
        tifffile.imwrite(str(raw_path), raw)
        automated.measure_edge_diameters_fwhm_from_raw_tiff(
            G,
            raw_tiff_path=raw_path,
            voxel_size_zyx=voxel_size_zyx,
            **measure_kwargs,
        )
        pairs = sorted(G.edges(keys=True), key=lambda t: (t[0], t[1], t[2]))
        title_parts = []
        for (u, v, k), t in zip(pairs, targets):
            d = float(G[u][v][k]["fwhm_diameter_um"])
            title_parts.append(f"FWHM {t[2]:.1f}→{d:.2f} µm")
        title = title_prefix + " | " + " | ".join(title_parts)
        labels, _ = automated.build_graph_branch_label_volume(
            G,
            raw.shape,
            voxel_size_zyx,
            background_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["background_label"]),
            junction_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["junction_label"]),
        )
        fig = build_synthetic_fwhm_integration_figure(
            G,
            raw,
            labels,
            voxel_size_zyx,
            measure_kwargs,
            title=title,
        )
        fig.write_html(str(out_path), include_plotlyjs="cdn")
    finally:
        raw_path.unlink(missing_ok=True)


def _write_demo_html() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "examples" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_size_zyx = (0.25, 0.25, 0.25)

    out_paths: list[Path] = []

    # 1) Baseline straight vessels.
    raw0, targets0 = build_synthetic_vessel_volume_and_targets(voxel_size_zyx)
    G0 = build_matching_multigraph(targets0, voxel_size_zyx, step_um=0.25)
    out0 = out_dir / "synthetic_vessel_fwhm_integration_3d.html"
    _write_single_demo_html(
        out0,
        raw0,
        G0,
        voxel_size_zyx,
        "Synthetic vessels: volume mesh + centerlines + transverse profile lines",
        targets0,
    )
    out_paths.append(out0)

    # 2) Y-shaped vessels.
    raw_y, targets_y = build_synthetic_y_shaped_volume_and_targets(voxel_size_zyx)
    Gy = build_y_shaped_matching_multigraph(targets_y, voxel_size_zyx, step_um=0.25)
    out_y = out_dir / "synthetic_vessel_y_shape_fwhm_integration_3d.html"
    _write_single_demo_html(
        out_y,
        raw_y,
        Gy,
        voxel_size_zyx,
        "Synthetic Y vessels: volume mesh + centerlines + transverse profile lines",
        targets_y,
    )
    out_paths.append(out_y)

    # 3) Straight vessels with graph centerlines off-center in x and z (30% of target diameter).
    raw_off, targets_off = build_synthetic_vessel_volume_and_targets(voxel_size_zyx)
    Goff = build_offcenter_matching_multigraph(
        targets_off, voxel_size_zyx, step_um=0.25, offcenter_fraction=0.3
    )
    out_off = out_dir / "synthetic_vessel_offcenter_graph_fwhm_integration_3d.html"
    _write_single_demo_html(
        out_off,
        raw_off,
        Goff,
        voxel_size_zyx,
        "Synthetic off-center graph: volume mesh + centerlines + transverse profile lines",
        targets_off,
    )
    out_paths.append(out_off)

    # 4) Off-center graph with noisy synthetic background.
    raw_off_noisy = add_background_noise_to_synthetic_volume(
        raw_off,
        noise_sigma=6.0,
        background_offset=4.0,
        seed=123,
    )
    out_off_noisy = out_dir / "synthetic_vessel_offcenter_graph_noisy_fwhm_integration_3d.html"
    _write_single_demo_html(
        out_off_noisy,
        raw_off_noisy,
        Goff,
        voxel_size_zyx,
        "Synthetic off-center graph (noisy raw): volume mesh + centerlines + transverse profile lines",
        targets_off,
    )
    out_paths.append(out_off_noisy)

    # 5) X-junction, noisy synthetic volume, and 30% off-center graph.
    raw_x, targets_x = build_synthetic_x_junction_volume_and_targets(voxel_size_zyx)
    raw_x_noisy = add_background_noise_to_synthetic_volume(
        raw_x, noise_sigma=6.0, background_offset=4.0, seed=321
    )
    Gx_off = build_x_junction_matching_multigraph(
        targets_x, voxel_size_zyx, step_um=0.25, offcenter_fraction=0.3
    )
    out_x_noisy = out_dir / "synthetic_vessel_x_junction_offcenter_noisy_fwhm_integration_3d.html"
    _write_single_demo_html(
        out_x_noisy,
        raw_x_noisy,
        Gx_off,
        voxel_size_zyx,
        "Synthetic noisy X-junction + 30% off-center graph: volume mesh + centerlines + transverse profile lines",
        targets_x,
        measure_kwargs_overrides={
            "branch_endpoint_exclusion_um": 10.0,
            "junction_proximity_exclusion_um": 10.0,
            "enforce_same_edge_locality": False,
            "cap_half_extent_by_nonlocal_same_edge_distance": False,
            "reject_samples_with_center_offset": False,
            "reject_samples_with_low_fit_r2": False,
        },
    )
    out_paths.append(out_x_noisy)

    # 6) Tight zig-zag vessel.
    raw_zig, targets_zig, centerline_zig = build_synthetic_tight_zigzag_volume_and_target(
        voxel_size_zyx
    )
    Gzig = build_tight_zigzag_matching_multigraph(centerline_zig)
    out_zig = out_dir / "synthetic_vessel_tight_zigzag_fwhm_integration_3d.html"
    _write_single_demo_html(
        out_zig,
        raw_zig,
        Gzig,
        voxel_size_zyx,
        "Synthetic tight zig-zag vessel: volume mesh + centerline + transverse profile lines",
        targets_zig,
    )
    out_paths.append(out_zig)

    return out_paths


if __name__ == "__main__":
    paths = _write_demo_html()
    for p in paths:
        print(f"Wrote {p}")
