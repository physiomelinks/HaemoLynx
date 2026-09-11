"""Tests for FWHM-based diameter estimation (haemodynamics.automated)."""
import pytest
from pathlib import Path

import numpy as np
import networkx as nx
import tifffile

from haemolynx.haemodynamics import automated
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"profile_baseline_mode": "percentile"},
        {"constrain_fitted_baseline": True, "baseline_constraint_half_width_ptp": 0.2},
    ],
)
def test_fwhm_from_profile_matches_diagnostics_fwhm(kwargs):
    """``fwhm_from_profile`` is a thin wrapper around
    ``_fwhm_gaussian_fit_with_diagnostics`` -- this is the regression test
    that would have caught the two implementations silently diverging when
    they were still separate copies of the same fit."""
    sigma = 1.5
    x = np.linspace(-8.0, 8.0, 161, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    simple = automated.fwhm_from_profile(x, y, **kwargs)
    diagnostic_fwhm, _center, _r2 = automated._fwhm_gaussian_fit_with_diagnostics(
        x, y, **kwargs
    )
    assert simple is not None and diagnostic_fwhm is not None
    assert simple == pytest.approx(diagnostic_fwhm)


def test_fwhm_from_profile_tolerates_non_finite_samples():
    """Folding onto ``_fwhm_gaussian_fit_with_diagnostics`` makes the simple
    wrapper strictly more robust than its old standalone implementation: a
    stray NaN/Inf sample is masked out rather than reaching ``curve_fit``."""
    sigma = 1.5
    x = np.linspace(-8.0, 8.0, 161, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    y_with_gap = y.copy()
    y_with_gap[10] = np.nan
    y_with_gap[50] = np.inf
    w = automated.fwhm_from_profile(x, y_with_gap)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert w is not None
    assert abs(w - expected) < 0.3


def test_full_width_at_fraction_ideal_gaussian():
    """For an ideal Gaussian, width(80%)/width(50%) = sqrt(ln 0.8/ln 0.5),
    independent of sigma -- ~0.567."""
    sigma = 1.5
    x = np.linspace(-10.0, 10.0, 2001, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    w80 = automated._full_width_at_fraction(x, y, 0.0, 100.0, 0.8)
    w50 = automated._full_width_at_fraction(x, y, 0.0, 100.0, 0.5)
    assert w80 is not None and w50 is not None
    expected_ratio = np.sqrt(np.log(0.8) / np.log(0.5))
    assert (w80 / w50) == pytest.approx(expected_ratio, abs=0.01)


def test_full_width_at_fraction_none_when_never_crosses():
    x = np.linspace(-2.0, 2.0, 21, dtype=float)
    y = np.full_like(x, 100.0)  # flat at the peak everywhere -- never drops
    assert automated._full_width_at_fraction(x, y, 0.0, 100.0, 0.5) is None


def test_profile_plateau_shape_ratio_flags_flat_top_profile():
    """A saturated/plateau-shaped profile's ratio approaches 1; an ideal
    Gaussian on the same grid stays comfortably below any reasonable
    plateau threshold."""
    x = np.linspace(-10.0, 10.0, 2001, dtype=float)
    gaussian = 100.0 * np.exp(-(x**2) / (2.0 * 1.5**2))
    gaussian_ratio = automated._profile_plateau_shape_ratio(x, gaussian, 0.0)

    saturated = 100.0 / (1.0 + np.exp((np.abs(x) - 3.0) / 0.05))
    saturated_ratio = automated._profile_plateau_shape_ratio(x, saturated, 0.0)

    assert gaussian_ratio is not None and saturated_ratio is not None
    assert gaussian_ratio < 0.6
    assert saturated_ratio > 0.95
    assert saturated_ratio > gaussian_ratio


def _cylinder_saturated_volume(
    nz: int, ny: int, nx: int, radius: float, saturation: float = 100.0
) -> np.ndarray:
    """A flat-topped (saturated) cylindrical cross-section: ~``saturation``
    inside ``radius`` with a short logistic falloff at the edge, ~0 outside
    -- mimics a detector-clipped signal (or a probability/segmentation
    field mistakenly used as raw intensity), not a real Gaussian bump."""
    zc, yc = 5.0, 5.0
    z = np.arange(nz, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    x = np.arange(nx, dtype=float)[None, None, :]
    r = np.sqrt((y - yc) ** 2 + (z - zc) ** 2) + 0.0 * (x - 10.0)
    tau = 0.3
    return (saturation / (1.0 + np.exp((r - radius) / tau))).astype(np.float32)


def test_measure_edge_diameters_rejects_saturated_synthetic_vessel(tmp_path: Path):
    """A flat-topped/saturated profile is a real risk for the parametric
    Gaussian fit's own R² (a heavily-overparameterized fit can still score
    deceptively well against a rounded plateau) -- the plateau-shape gate
    is independent of that fit and should measurably reduce how much of
    this saturated fixture gets accepted, whether that means fewer per-edge
    samples or the edge failing outright."""
    nz, ny, nx_dim = 11, 11, 21
    raw = _cylinder_saturated_volume(nz, ny, nx_dim, radius=3.0)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    def _build_graph() -> nx.MultiGraph:
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
        graph.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
        voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
        graph.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels)
        return graph

    common = dict(
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=5.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=8.0,
        diameter_guess_um=2.0,
    )
    summary_gated = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        _build_graph(), reject_samples_with_plateau_shape=True, **common
    )
    summary_ungated = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        _build_graph(), reject_samples_with_plateau_shape=False, **common
    )

    gated_n = (
        summary_gated["per_edge"][0]["n_samples"] if summary_gated["edges_measured"] else 0
    )
    ungated_n = (
        summary_ungated["per_edge"][0]["n_samples"]
        if summary_ungated["edges_measured"]
        else 0
    )
    assert gated_n <= ungated_n
    assert ungated_n > 0  # sanity: the fixture is fittable at all without the gate


def test_aggregate_edge_diameter_median_resists_one_outlier_sample():
    """The one skewed fit corrupting an otherwise-good edge average is
    exactly the failure mode `median` exists to resist."""
    diameters = [2.0, 2.1, 1.9, 2.05, 9.0]
    median = automated._aggregate_edge_diameter(diameters, "median")
    mean = automated._aggregate_edge_diameter(diameters, "mean")
    assert median == pytest.approx(2.05)
    assert mean > 3.0  # pulled well away from the true ~2.0 cluster
    assert abs(median - 2.0) < abs(mean - 2.0)


def test_aggregate_edge_diameter_agrees_with_mean_on_tightly_clustered_samples():
    diameters = [3.0, 3.01, 2.99, 3.02]
    median = automated._aggregate_edge_diameter(diameters, "median")
    mean = automated._aggregate_edge_diameter(diameters, "mean")
    assert median == pytest.approx(mean, abs=0.05)


def test_aggregate_edge_diameter_handles_a_single_sample():
    assert automated._aggregate_edge_diameter([4.2], "median") == pytest.approx(4.2)
    assert automated._aggregate_edge_diameter([4.2], "mean") == pytest.approx(4.2)


def test_aggregate_edge_diameter_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        automated._aggregate_edge_diameter([1.0, 2.0], "bogus")


def test_fwhm_from_profile_rejects_an_unknown_baseline_mode():
    """Both functions raise loudly on a typo'd mode string rather than
    silently returning None -- a config typo should be visible, not
    indistinguishable from "no signal found"."""
    x = np.linspace(-8.0, 8.0, 21, dtype=float)
    y = 100.0 * np.exp(-(x**2) / (2.0 * 1.5**2))
    with pytest.raises(ValueError):
        automated.fwhm_from_profile(x, y, profile_baseline_mode="bogus")
    with pytest.raises(ValueError):
        automated._fwhm_gaussian_fit_with_diagnostics(x, y, profile_baseline_mode="bogus")


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


def test_gram_schmidt_perpendicular_true_3d():
    """Unit length, genuinely perpendicular to the real 3D tangent -- unlike
    the in-plane direction, this holds for a tangent with a real z (axis 0)
    component."""
    t = np.array([1.0, 2.0, 3.0], dtype=float)
    n = automated._gram_schmidt_perpendicular(t)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-9
    t_hat = t / np.linalg.norm(t)
    assert abs(float(np.dot(n, t_hat))) < 1e-9


def test_transverse_unit_for_mode_dispatches_correctly():
    t = np.array([1.0, 2.0, 3.0], dtype=float)
    in_plane = automated._transverse_unit_for_mode(t, "in_plane_yx")
    true_3d = automated._transverse_unit_for_mode(t, "true_3d_perpendicular")
    assert in_plane[0] == pytest.approx(0.0)  # in-plane: no z component
    assert true_3d[0] != pytest.approx(0.0)  # true 3D: has a z component here
    with pytest.raises(ValueError):
        automated._transverse_unit_for_mode(t, "bogus")


def test_out_of_plane_fraction_zero_for_in_plane_tangent():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    assert automated._out_of_plane_fraction(t) == pytest.approx(0.0)


def test_out_of_plane_fraction_one_for_pure_z_tangent():
    t = np.array([5.0, 0.0, 0.0], dtype=float)
    assert automated._out_of_plane_fraction(t) == pytest.approx(1.0)


def test_out_of_plane_fraction_matches_the_sine_of_the_tilt_angle():
    t = np.array([1.0, 2.0, 3.0], dtype=float)
    expected = abs(t[0]) / np.linalg.norm(t)
    assert automated._out_of_plane_fraction(t) == pytest.approx(expected)


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


def test_measure_edge_diameters_fwhm_stamps_status_on_skip_paths(tmp_path: Path):
    """Every skip path stamps a matching ``fwhm_status`` on the edge, so a
    caller can tell "FWHM was attempted and failed" from "FWHM never even
    reached this edge" (no key at all). Uses only edges that hit a skip path
    before any Gaussian fit is attempted, so this does not depend on
    ``scipy.optimize.curve_fit`` succeeding."""
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, 2.0]))
    G.add_node(2, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(3, pos=np.array([0.0, 0.0, 0.0]))
    # No `voxels` at all -> "no_voxels_or_label".
    G.add_edge(0, 1, key=0, length=2.0, branch_order="B01")
    # `voxels` present but every point identical -> zero arc length.
    G.add_edge(
        2, 3, key=0, length=0.0, branch_order="B01",
        voxels=[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
    )

    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=1.0,
        transverse_profile_step_um=0.5,
        transverse_half_extent_um=2.0,
    )

    assert summary["edges_measured"] == 0
    assert G[0][1][0]["fwhm_status"] == "failed:no_voxels_or_label"
    assert G[2][3][0]["fwhm_status"] == "failed:zero_length"
    assert "fwhm_diameter_um" not in G[0][1][0]
    assert "fwhm_diameter_um" not in G[2][3][0]


def test_measure_edge_diameters_fwhm_forwards_edge_diameter_aggregation(
    tmp_path: Path, monkeypatch
):
    """The ``edge_diameter_aggregation`` keyword must reach the actual
    aggregation call -- not just exist on the function signature."""
    nz, ny, nx_dim = 11, 11, 21
    sigma = 1.5
    raw = _cylinder_gaussian_volume(nz, ny, nx_dim, sigma)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
    G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
    voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
    G.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels)

    captured_methods = []
    real_aggregate = automated._aggregate_edge_diameter

    def spy(diameters, method):
        captured_methods.append(method)
        return real_aggregate(diameters, method)

    monkeypatch.setattr(automated, "_aggregate_edge_diameter", spy)

    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=5.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=8.0,
        diameter_guess_um=2.0,
        edge_diameter_aggregation="mean",
    )

    assert captured_methods == ["mean"]


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
        weight=1.0,
        length=16.0,
        branch_order="B01",
        voxels=voxels,
    )

    voxel_size_zyx = (1.0, 1.0, 1.0)
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel_size_zyx,
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
        {"B01": 6.9},  # sentinel: must be ignored in favour of fwhm_diameter_um
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    r = G2[0][1][0]["resistance"]
    expect_r = model.resistance_of_uniform_segment(16.0, d)
    assert abs(r - expect_r) < expect_r * 0.05
    assert G2[0][1][0]["conductance"] == pytest.approx(1.0 / r)


def test_set_poiseuille_resistances_prefers_fwhm_optional(multigraph_with_branch_order):
    G = multigraph_with_branch_order.copy()
    G[0][1][0]["fwhm_diameter_um"] = 2.0
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    _, res = model.set_poiseuille_resistances(
        G,
        {"BO1": 6.9},  # sentinel: must be ignored in favour of fwhm_diameter_um
        prefer_edge_fwhm_diameter=True,
    )
    assert res["used_fwhm_edge_diameter"] == 1
    d_used = 2.0
    expect = model.resistance_of_uniform_segment(5.0, d_used)
    assert G[0][1][0]["resistance"] == pytest.approx(expect)
