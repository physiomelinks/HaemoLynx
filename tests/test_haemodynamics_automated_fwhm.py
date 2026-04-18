"""Tests for FWHM-based diameter estimation (haemodynamics.automated)."""
from pathlib import Path

import numpy as np
import networkx as nx
import tifffile

from ImageLynx.haemodynamics import automated
from ImageLynx.haemodynamics.poiseuille import PoiseuilleModel


def _cylinder_gaussian_volume(nz: int, ny: int, nx: int, sigma: float) -> np.ndarray:
    zc, yc, xc = 5.0, 5.0, 10.0
    z = np.arange(nz, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    x = np.arange(nx, dtype=float)[None, None, :]
    # Keep intensity cylindrical (independent of x) while still materializing
    # the full requested x extent.
    r2 = (y - yc) ** 2 + (z - zc) ** 2 + 0.0 * (x - xc)
    return (100.0 * np.exp(-r2 / (2.0 * sigma**2))).astype(np.float32)


def test_fwhm_from_profile_gaussian_fit():
    """Ideal 1D Gaussian: fitted sigma -> FWHM matches analytic width."""
    sigma = 1.5
    x = np.linspace(-8.0, 8.0, 161, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    w = automated.fwhm_from_profile(x, y)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert w is not None
    assert abs(w - expected) < 0.2


def test_robust_baseline_wings_one_sided_shoulder():
    x = np.linspace(-8.0, 8.0, 161, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * 1.5**2))
    y = np.asarray(y, dtype=float).copy()
    y[x > 4.0] += 40.0
    b = automated.robust_baseline_from_profile_wings(x, y, wing_fraction=0.2)
    assert b < 12.0


def test_fwhm_percentile_and_wings_modes_symmetric_gaussian():
    sigma = 1.5
    x = np.linspace(-8.0, 8.0, 161, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    w_w = automated.fwhm_from_profile(x, y, profile_baseline_mode="wings")
    w_p = automated.fwhm_from_profile(x, y, profile_baseline_mode="percentile")
    assert w_w is not None and w_p is not None
    assert abs(w_w - w_p) < 0.3


def test_transverse_unit_in_physical_yx_plane():
    """In-plane transverse: orthogonal to tangent, zero z (axis-0) component, unit length."""
    t = np.array([0.0, 0.0, 1.0], dtype=float)
    n = automated._transverse_unit_in_physical_yx_plane(t)
    assert abs(n[0]) < 1e-9
    assert abs(float(np.dot(t, n))) < 1e-9
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-9

    t2 = np.array([1.0, 2.0, 3.0], dtype=float)
    n2 = automated._transverse_unit_in_physical_yx_plane(t2)
    assert abs(n2[0]) < 1e-9
    assert abs(float(np.dot(t2, n2))) < 1e-9
    assert abs(float(np.linalg.norm(n2)) - 1.0) < 1e-9


def test_build_graph_branch_label_volume():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([2.0, 0.0, 0.0]))
    edge_voxels = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    G.add_edge(0, 1, voxels=edge_voxels)
    vol, mapping = automated.build_graph_branch_label_volume(
        G,
        (3, 3, 3),
        (1.0, 1.0, 1.0),
        background_label=0,
        junction_label=-1,
    )
    assert mapping[(0, 1, 0)] == 1
    labeled_coords = {tuple(idx) for idx in np.argwhere(vol == 1)}
    expected_coords = {
        tuple(np.rint(np.asarray(v, dtype=float)).astype(int)) for v in edge_voxels
    }
    assert labeled_coords == expected_coords
    assert G[0][1][0]["graph_edge_label_id"] == 1


def test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder(tmp_path: Path):
    nz, ny, nx_dim = 11, 11, 21
    sigma = 1.5
    raw = _cylinder_gaussian_volume(nz, ny, nx_dim, sigma)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
    G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
    voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
    G.add_edge(
        0,
        1,
        resistance=1.0,
        length=16.0,
        branch_order="B01",
        voxels=voxels,
    )

    voxel_size_xyz = (1.0, 1.0, 1.0)
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_xyz=voxel_size_xyz,
        sample_spacing_along_edge_um=5.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=8.0,
        diameter_guess_um=2.0,
        background_label=0,
        junction_label=-1,
        min_total_extent_multiplier=3.0,
    )
    assert summary["edges_measured"] == 1
    d = G[0][1][0]["fwhm_diameter_um"]
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert abs(d - expected) < 0.35

    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    G2, res = model.set_poiseuille_resistances(
        G,
        {"B01": 99.0},
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    r = G2[0][1][0]["resistance"]
    visc = 1.0 / (d**1.647)
    expect_r = (128.0 * visc * 16.0) / (np.pi * d**4)
    assert abs(r - expect_r) < expect_r * 0.05


def test_set_poiseuille_resistances_prefers_fwhm_optional(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G[0][1][0]["fwhm_diameter_um"] = 2.0
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    _, res = model.set_poiseuille_resistances(
        G,
        {"BO1": 20.0},
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    d_used = 2.0
    visc = 1.0 / (d_used**1.647)
    expect = (128.0 * visc * 5.0) / (np.pi * d_used**4)
    assert abs(G[0][1][0]["resistance"] - expect) < 1e-6
