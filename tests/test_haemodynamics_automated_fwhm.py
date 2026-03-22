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
    r2 = (y - yc) ** 2 + (z - zc) ** 2
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


def test_build_graph_branch_label_volume():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([2.0, 0.0, 0.0]))
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    vol, mapping = automated.build_graph_branch_label_volume(
        G,
        (3, 3, 3),
        (1.0, 1.0, 1.0),
        background_label=0,
        junction_label=-1,
    )
    assert mapping[(0, 1, 0)] == 1
    assert vol[0, 0, 0] == 1 and vol[0, 0, 1] == 1 and vol[0, 0, 2] == 1
    assert G[0][1][0]["graph_edge_label_id"] == 1


def test_measure_edge_diameters_fwhm_from_raw_tiff_cylinder(tmp_path: Path):
    nz, ny, nx = 11, 11, 21
    sigma = 1.5
    raw = _cylinder_gaussian_volume(nz, ny, nx, sigma)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
    G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
    voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
    G.add_edge(
        0,
        1,
        weight=1.0,
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
    G2, res = model.set_poiseuille_weights(
        G,
        {"B01": 99.0},
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    w = G2[0][1][0]["weight"]
    visc = 1.0 / (d**1.647)
    expect_w = (np.pi * d**4) / (128.0 * visc * 16.0)
    assert abs(w - expect_w) < expect_w * 0.05


def test_set_poiseuille_weights_prefers_fwhm_optional(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G[0][1][0]["fwhm_diameter_um"] = 2.0
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    _, res = model.set_poiseuille_weights(
        G,
        {"BO1": 20.0},
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    d_used = 2.0
    visc = 1.0 / (d_used**1.647)
    expect = (np.pi * d_used**4) / (128.0 * visc * 5.0)
    assert abs(G[0][1][0]["weight"] - expect) < 1e-6
