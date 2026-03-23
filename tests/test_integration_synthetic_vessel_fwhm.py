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

from ImageLynx.haemodynamics import automated

_GAUSSIAN_FWHM_FROM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_I0 = 100.0

# Shared with ``measure_edge_diameters_fwhm_from_raw_tiff`` calls and the visualization helper.
_DEFAULT_FWHM_MEASURE_KWARGS: dict = {
    "sample_spacing_along_edge_um": 2.0,
    "transverse_profile_step_um": 0.2,
    "transverse_half_extent_um": 14.0,
    "diameter_guess_um": 5.0,
    "background_label": 0,
    "junction_label": -1,
    "min_total_extent_multiplier": 3.0,
    "profile_baseline_mode": "wings",
    "profile_baseline_wing_fraction": 0.2,
    "constrain_fitted_baseline": False,
    "baseline_constraint_half_width_ptp": 0.35,
}

_EDGE_LINE_COLORS = ("#00ffff", "#ffaa00", "#cc66ff")
_PROFILE_LINE_COLORS = ("#00cc88", "#ff6600", "#9933ff")
_VOLUME_MESH_COLORS = ("#00ffff", "#ffaa00", "#cc66ff")  # match centerlines: low z → high z


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
    voxel_size_xyz: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    """Float32 volume + list of (endpoint_a, endpoint_b, target_fwhm_um) in (z,y,x) µm.

    Three straight vessels parallel to +x with geometric FWHM 3, 5, and 8 µm (Gaussian cross-section).

    Vessels share the same physical *y* and differ in *z*.     For tangents along +x, ``_transverse_unit_in_physical_yx_plane`` yields a transverse
    direction in the y–x plane (here, ±y). If vessels were stacked in *y*, one ray direction
    would hit a neighbour's raster label after a few µm while the other would cross only
    background to the volume edge, producing strongly asymmetric profile lines (not a rendering
    bug). Vessels are therefore separated in *z* at a shared *y*.
    """
    vz, vy, vx = voxel_size_xyz
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
    voxel_size_xyz: tuple[float, float, float],
    step_um: float = 0.25,
) -> nx.MultiGraph:
    """Graph edges replicate centerlines; physical coords (z,y,x) match ``automated`` convention."""
    vz, vy, vx = voxel_size_xyz
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


def _iter_profile_polylines_phys(
    G: nx.MultiGraph,
    raw: np.ndarray,
    labels: np.ndarray,
    voxel_size_xyz: tuple[float, float, float],
    measure_kwargs: dict,
) -> list[tuple[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]]]:
    """Transverse profile polylines in physical (z,y,x) µm — same sampling as the FWHM measurer.

    Each sample is ``(polyline, anchor)`` where ``anchor`` is the centerline point (offset 0 on the profile).
    """
    mult = float(measure_kwargs["min_total_extent_multiplier"])
    jn = int(measure_kwargs["junction_label"])
    bg = int(measure_kwargs["background_label"])
    t_half = float(measure_kwargs["transverse_half_extent_um"])
    t_step = float(measure_kwargs["transverse_profile_step_um"])
    d_guess = float(measure_kwargs["diameter_guess_um"])
    s_space = float(measure_kwargs["sample_spacing_along_edge_um"])
    p_mode = measure_kwargs["profile_baseline_mode"]
    p_wing = float(measure_kwargs["profile_baseline_wing_fraction"])
    c_fb = bool(measure_kwargs["constrain_fitted_baseline"])
    c_hw = float(measure_kwargs["baseline_constraint_half_width_ptp"])

    out: list[tuple[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]]] = []

    for u, v, key, data in sorted(G.edges(keys=True, data=True), key=lambda t: (t[0], t[1], t[2])):
        vox = data.get("voxels")
        assigned = data.get("graph_edge_label_id")
        if not vox or len(vox) < 2 or assigned is None:
            continue
        centerline = np.asarray(vox, dtype=float)
        s, total_len = automated._arc_length_parameterize(centerline)
        if total_len <= 0:
            continue
        n_samples = max(1, int(np.floor(total_len / s_space)) + 1)
        targets = np.linspace(0.0, total_len, n_samples)
        pts = automated._interpolate_centerline(centerline, s, targets)

        segs: list[tuple[np.ndarray, np.ndarray]] = []
        for s0, center in zip(targets, pts):
            tangent = automated._tangent_at(centerline, s, float(s0))
            n_hat = automated._transverse_unit_in_physical_yx_plane(tangent)
            c = np.asarray(center, dtype=float)

            half_extent = max(t_half, 0.5 * mult * d_guess)
            pos, prof = automated._sample_transverse_profile(
                raw,
                labels,
                c,
                tangent,
                int(assigned),
                half_extent,
                t_step,
                voxel_size_xyz,
                background_label=bg,
                junction_label=jn,
            )
            d0 = automated.fwhm_from_profile(
                pos,
                prof,
                profile_baseline_mode=p_mode,
                profile_baseline_wing_fraction=p_wing,
                constrain_fitted_baseline=c_fb,
                baseline_constraint_half_width_ptp=c_hw,
            )
            pos_used = pos
            if d0 is not None and d0 > 0:
                half_extent = max(half_extent, 0.5 * mult * float(d0))
                pos_used, _ = automated._sample_transverse_profile(
                    raw,
                    labels,
                    c,
                    tangent,
                    int(assigned),
                    half_extent,
                    t_step,
                    voxel_size_xyz,
                    background_label=bg,
                    junction_label=jn,
                )

            if pos_used.size == 0:
                continue
            transect_pts = np.stack([c + float(o) * n_hat for o in pos_used], axis=0)
            segs.append((transect_pts, c.copy()))

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
    voxel_size_xyz: tuple[float, float, float],
    measure_kwargs: dict,
    *,
    title: str,
) -> go.Figure:
    """Plotly scene: raw intensity as a triangle mesh (marching cubes), centerlines, profile lines."""
    nz, ny, nx = raw.shape
    vz, vy, vx = voxel_size_xyz

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

    polylines_by_edge = _iter_profile_polylines_phys(
        G, raw, labels, voxel_size_xyz, measure_kwargs
    )
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
    voxel_size_xyz = (0.25, 0.25, 0.25)
    raw, targets = build_synthetic_vessel_volume_and_targets(voxel_size_xyz)
    raw_path = tmp_path / "synthetic_vessels.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = build_matching_multigraph(targets, voxel_size_xyz, step_um=0.25)
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_xyz=voxel_size_xyz,
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
        voxel_size_xyz,
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
        voxel_size_xyz,
        _DEFAULT_FWHM_MEASURE_KWARGS,
        title=title,
    )
    fig.write_html(str(tmp_path / "synthetic_vessel_fwhm_viz.html"), include_plotlyjs="cdn")


def _write_demo_html() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "examples" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_vessel_fwhm_integration_3d.html"

    voxel_size_xyz = (0.25, 0.25, 0.25)
    raw, targets = build_synthetic_vessel_volume_and_targets(voxel_size_xyz)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
        raw_path = Path(tf.name)
    try:
        tifffile.imwrite(str(raw_path), raw)

        G = build_matching_multigraph(targets, voxel_size_xyz, step_um=0.25)
        automated.measure_edge_diameters_fwhm_from_raw_tiff(
            G,
            raw_tiff_path=raw_path,
            voxel_size_xyz=voxel_size_xyz,
            **_DEFAULT_FWHM_MEASURE_KWARGS,
        )

        pairs = [(0, 1, 0), (2, 3, 0), (4, 5, 0)]
        title_parts = []
        for (u, v, k), t in zip(pairs, targets):
            d = float(G[u][v][k]["fwhm_diameter_um"])
            title_parts.append(f"FWHM {t[2]:.1f}→{d:.2f} µm")
        title = "Synthetic vessels: volume mesh + centerlines + transverse profile lines | " + " | ".join(
            title_parts
        )
        labels, _ = automated.build_graph_branch_label_volume(
            G,
            raw.shape,
            voxel_size_xyz,
            background_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["background_label"]),
            junction_label=int(_DEFAULT_FWHM_MEASURE_KWARGS["junction_label"]),
        )
        fig = build_synthetic_fwhm_integration_figure(
            G,
            raw,
            labels,
            voxel_size_xyz,
            _DEFAULT_FWHM_MEASURE_KWARGS,
            title=title,
        )
        fig.write_html(str(out_path), include_plotlyjs="cdn")
    finally:
        raw_path.unlink(missing_ok=True)
    return out_path


if __name__ == "__main__":
    path = _write_demo_html()
    print(f"Wrote {path}")
